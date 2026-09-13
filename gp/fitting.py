"""Building, fitting and predicting with the multi-fidelity GP.

Fitting is maximum a posteriori: the exact marginal log likelihood plus the log
priors registered on the kernel, plus the noise model's own prior term. With as
few as three human subjects the likelihood alone is close to flat in the
lengthscale, so those priors are what make the fit identifiable.

The quantity predicted
----------------------
The experiments score the **maximum mean relative APD change** over a
concentration range, which is what :mod:`simulation.calculate_ground_truth`
tabulates. A prediction draws posterior sample paths over a grid of
concentrations and takes each path's maximum, giving samples from the posterior
over that maximum. Every metric in :mod:`gp.metrics` is computed from them.

"""

import gpytorch
import numpy as np
import torch

from drug import ALL_CHANNELS, Drug
from gp.data import apply_standardisation
from gp.metrics import compute_crps
from gp.models import MultitaskGPModel, ObservationNoise

#: Adam learning rate. Fixed across every published experiment.
LEARNING_RATE = 0.1

#: Above this many training points the Cholesky needs a larger jitter. Only the
#: oracle fits reach it.
LARGE_TRAINING_SET = 1000
JITTER_SMALL = 1e-6
JITTER_LARGE = 1e-3


def build_noise_model(num_species, config, conc_idxs=None):
    """Construct the observation-noise model from a config.

    With fewer than three concentrations per species the heteroskedastic
    length-scale ``rho`` is not identifiable, so it is frozen at its initial
    value and only the noise magnitude is learned.

    Args:
        conc_idxs: ``(human, animal)`` concentration index lists to apply that
            test to. Required for the iterative configs, which name their lists
            ``*_prior`` / ``*_target`` rather than the plain keys read here.
    """
    noise_model = ObservationNoise(
        num_species=num_species,
        noise_type=config["noise_type"],
        init_sigma_min=config.get("init_sigma_min", 1e-2),
        shared_noise=config.get("shared_noise", False),
        rho_prior=config.get("rho_prior"),
    )

    if conc_idxs is not None:
        human_concs, animal_concs = conc_idxs
    else:
        human_concs = config.get("human_concentration_idxs", [])
        animal_concs = config.get("animal_concentration_idxs", [])
    too_few = len(human_concs) < 3 or len(animal_concs) < 3
    if too_few and config["noise_type"] == "heteroskedastic":
        noise_model.raw_rho.requires_grad = False

    return noise_model


def build_model(X, y, config, num_species, channel_mask, X_noise=None,
                seed_init=True, conc_idxs=None):
    """Construct model, likelihood and noise model on the given data.

    The RNG is seeded *before* construction because ``IndexKernel`` draws its
    coregionalisation factor from the global RNG at that point;
    :func:`train_model`'s own seeding happens too late to make that
    reproducible.

    ``seed_init=False`` skips that seeding, for a model whose parameters are
    about to be overwritten from a checkpoint. Seeding here would reset a
    sampling stream the caller is part-way through, as in the iterative
    experiment.
    """
    noise_input = X_noise if X_noise is not None else X

    if seed_init:
        torch.manual_seed(config.get("training_seed", 42))
    noise_model = build_noise_model(num_species, config, conc_idxs=conc_idxs)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise_model(noise_input), learn_additional_noise=False
    )
    model = MultitaskGPModel(
        X, y, likelihood,
        num_species=num_species,
        channel_mask=channel_mask,
        ard_num_dims=config["ard_num_dims"],
        mean_function=config["mean_function"],
        boundary_constraint=config.get("boundary_constraint"),
        covar_rank=config.get("covar_rank", 1),
        num_latents=config.get("num_latents", 1),
        shared_mean=config.get("shared_mean", False),
        ls_prior=config.get("ls_prior"),
        index_kernel_prior=config.get("index_kernel_prior"),
        use_signed_mean=config.get("use_signed_mean", False),
    )
    return model, likelihood, noise_model


def train_model(model, likelihood, noise_model, X, y, config, X_noise=None,
                refresh_noise=True):
    """Fit by maximising the marginal log likelihood plus log priors.

    The noise is recomputed inside the loop so its gradient reaches the noise
    model's parameters.

    ``refresh_noise`` controls whether the likelihood's noise is recomputed
    *after* the final optimiser step, i.e. whether the posterior conditions on
    the training noise implied by the final parameters or the ones one step
    earlier — worth about 1e-5 in the predictions. The drug-block experiments
    refresh it; the heart-rate ones do not.
    """
    noise_input = X_noise if X_noise is not None else X

    model.train()
    likelihood.train()
    likelihood.noise = noise_model(noise_input)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": noise_model.parameters()}],
        lr=LEARNING_RATE,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    torch.manual_seed(config.get("training_seed", 42))
    for _ in range(config.get("num_iters", 100)):
        optimizer.zero_grad()
        likelihood.noise = noise_model(noise_input)
        output = model(X)
        loss = -mll(output, y) + noise_model.regularization_loss()
        loss.backward()
        optimizer.step()

    if refresh_noise:
        likelihood.noise = noise_model(noise_input)
    return model, likelihood, noise_model


def make_test_inputs(drug_name, species_idx, num_concs, conc_multiplier,
                     conc_as_input=False):
    """Build a concentration grid and its model inputs for one species.

    Returns:
        ``(concentrations, X)`` with the grid spanning 0 to
        ``conc_multiplier * c_therapeutic`` inclusive.
    """
    drug = Drug(drug_name)
    max_conc = conc_multiplier * drug.c_therapeutic
    concentrations = np.linspace(0, max_conc, num_concs)
    x_species = torch.full((num_concs, 1), float(species_idx), dtype=torch.float32)

    if conc_as_input:
        scaled = torch.tensor(
            concentrations / max_conc, dtype=torch.float32
        ).unsqueeze(1)
        return concentrations, torch.cat((scaled, x_species), dim=1)

    block = {channel: np.zeros(num_concs) for channel in ALL_CHANNELS}
    nonzero = concentrations != 0.0
    if nonzero.any():
        inhibitions = drug.calculate_all_inhibitions(
            concentrations=concentrations[nonzero], return_as_list=False
        )
        for channel, values in inhibitions.items():
            block[channel][nonzero] = values
    x_block = torch.tensor(
        np.column_stack([block[channel] for channel in ALL_CHANNELS]),
        dtype=torch.float32,
    )
    return concentrations, torch.cat((x_block, x_species), dim=1)


def sparse_indices(num_concs, num_sparse):
    """Indices of the grid used for posterior sampling, spanning both endpoints.

    Stride slicing would drop the tail — with 50 dense points and 10 sparse it
    stops at 91.8% of the concentration ceiling — which biases a maximum over
    concentration downward.
    """
    n = min(num_sparse, num_concs)
    return torch.linspace(0, num_concs - 1, n).round().long().unique()


def predict(model, likelihood, noise_model, drug, species_idx, config,
            ground_truth_max=None, standardisation=None):
    """Posterior over the maximum relative APD change for one drug and species.

    Draws from the global RNG **without seeding it**: the seed is set once per
    repeat and successive species run off the same stream, so seeding is the
    caller's job. See :func:`predict_all_species`.

    Returns a dict with the dense posterior curve (for plotting), the sampled
    maxima (what the metrics are computed from), and CRPS against
    ``ground_truth_max`` when given.
    """
    num_concs = int(config["num_post_concs"] * config["conc_multiplier"])
    num_samples = config["num_post_samples"]
    conc_multiplier = config["conc_multiplier"]

    concentrations, X_test = make_test_inputs(
        drug, species_idx, num_concs, conc_multiplier,
        conc_as_input=config.get("conc_as_input", False),
    )

    # The noise model reads raw block magnitudes, so it must see unstandardised
    # inputs even when the GP does not.
    X_test_raw = X_test.clone()
    if standardisation is not None:
        idx = int(species_idx)
        X_test = X_test.clone()
        X_test[:, :-1] = (
            X_test[:, :-1] - standardisation["x_means"][idx]
        ) / standardisation["x_stds"][idx]

    num_train = model.train_inputs[0].shape[0] if model.train_inputs else 0
    jitter = JITTER_LARGE if num_train > LARGE_TRAINING_SET else JITTER_SMALL
    sparse = sparse_indices(num_concs, config["num_post_sparse_concs"])

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
            gpytorch.settings.cholesky_jitter(jitter):
        latent = model(X_test)
        observed = likelihood(latent, noise=noise_model(X_test_raw))
        mean = observed.mean.numpy()
        lower, upper = (t.numpy() for t in observed.confidence_region())
        lower_latent, upper_latent = (t.numpy() for t in latent.confidence_region())

        # Paths are drawn from the latent posterior, not the observed one: the
        # quantity scored is the maximum of the underlying APD response, not of
        # a noisy measurement of it.
        sparse_pred = model(X_test[sparse])
        sparse_mean = sparse_pred.mean.detach().cpu().numpy()
        # Each path's maximum is one draw from the posterior over the maximum.
        samples = sparse_pred.sample(sample_shape=torch.Size([num_samples]))
        sampled_maxima = samples.max(dim=1).values.detach().cpu()

    if standardisation is not None:
        idx = int(species_idx)
        y_mean = standardisation["y_means"][idx].item()
        y_std = standardisation["y_stds"][idx].item()
        mean = mean * y_std + y_mean
        sparse_mean = sparse_mean * y_std + y_mean
        lower = lower * y_std + y_mean
        upper = upper * y_std + y_mean
        lower_latent = lower_latent * y_std + y_mean
        upper_latent = upper_latent * y_std + y_mean
        sampled_maxima = sampled_maxima * y_std + y_mean

    return {
        "concentrations": concentrations,
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "lower_latent": lower_latent,
        "upper_latent": upper_latent,
        "sampled_maxima": sampled_maxima,
        # Max of the sparse mean, not the dense one, so the point prediction
        # uses the grid the samples were drawn on.
        "predicted_max": float(np.max(sparse_mean)),
        "crps": (
            compute_crps(sampled_maxima, ground_truth_max)
            if ground_truth_max is not None
            else float("nan")
        ),
    }


def load_oracle(path):
    """Load an oracle checkpoint: hyperparameters and their standardisation.

    Oracle hyperparameters are fitted once to the maximal training set and
    reused unchanged by every repeat, so repeats differ only in which subjects
    they were given.

    The standardisation is part of the checkpoint and must be used as saved;
    recomputing it from the repeat's own subjects would rescale the inputs away
    from what the hyperparameters were fitted against.
    """
    checkpoint = torch.load(path, weights_only=False)
    return (
        checkpoint["model_state_dict"],
        checkpoint["noise_state_dict"],
        checkpoint.get("standardisation_params"),
    )


def predict_all_species(model, likelihood, noise_model, drug, species_order,
                        config, ground_truth=None, standardisation=None,
                        seed_posterior=True):
    """Predict for each species in turn, off one seeded RNG stream.

    The seed is set **once**, before the first species, and each species draws
    from wherever the previous one left the stream, so a species' samples depend
    on how many were predicted before it. Load-bearing: seeding per species
    instead gives the joint arm different samples.

    ``species_order`` is therefore fixed: the full species list (low fidelity
    first, high fidelity last) for a joint fit, and the high-fidelity species
    alone for a high-fidelity-only fit.

    ``seed_posterior=False`` suppresses the seeding entirely, for callers that
    seed once across *several* calls — the iterative experiment predicts under
    three conditioning sets off one stream.

    Returns:
        ``{species_index: prediction}``.
    """
    if seed_posterior:
        torch.manual_seed(config.get("post_samples_seed", 42))

    predictions = {}
    for species_index in species_order:
        truth = None
        if ground_truth is not None:
            truth = ground_truth.get(species_index)
        predictions[species_index] = predict(
            model, likelihood, noise_model, drug, species_index, config,
            ground_truth_max=truth, standardisation=standardisation,
        )
    return predictions


def fit_hyperparameters(X, y, config, num_species, channel_mask,
                        standardise=True, conc_idxs=None):
    """Fit hyperparameters on one dataset so they can be reused on others.

    The iterative experiment fits once per repeat on the *prior* drugs and holds
    those hyperparameters fixed while the conditioning set grows, so the three
    scenarios differ only in the data conditioned on.

    Returns:
        ``(model_state, noise_state, standardisation)``, ready to hand to
        :func:`fit_and_predict` as ``oracle_state``.
    """
    from gp.data import compute_standardisation

    X_raw = X.clone()
    standardisation = None
    if standardise:
        standardisation = compute_standardisation(
            X, y, config.get("boundary_constraint")
        )
        apply_standardisation(X, y, standardisation)

    model, likelihood, noise_model = build_model(
        X, y, config, num_species, channel_mask, X_noise=X_raw,
        conc_idxs=conc_idxs,
    )
    train_model(model, likelihood, noise_model, X, y, config, X_noise=X_raw)
    return model.state_dict(), noise_model.state_dict(), standardisation


def fit_and_predict(X, y, config, num_species, channel_mask, drug, species_idx,
                    ground_truth_max=None, standardise=True, oracle_path=None,
                    species_order=None, ground_truth=None, oracle_state=None,
                    seed_posterior=True, seed_init=True, conc_idxs=None):
    """Standardise, fit (or load oracle hyperparameters), and predict.

    Args:
        oracle_path: Checkpoint to load instead of fitting. See
            :func:`load_oracle`. When given, the checkpoint's standardisation is
            used in place of one computed from this repeat's data.
        oracle_state: The same three objects already in memory, from
            :func:`fit_hyperparameters`. Takes precedence over ``oracle_path``.
        seed_posterior: Whether to reseed the sampling stream. False when the
            caller seeds once across several calls — see
            :func:`predict_all_species`.
        seed_init: Whether to seed at model construction. False alongside
            ``seed_posterior=False``. See :func:`build_model`.
    """
    from gp.data import compute_standardisation

    X_raw = X.clone()
    if oracle_state is not None:
        oracle = oracle_state
    elif oracle_path is not None:
        oracle = load_oracle(oracle_path)
    else:
        oracle = None

    standardisation = None
    if standardise:
        if oracle is not None and oracle[2] is not None:
            standardisation = oracle[2]
        else:
            standardisation = compute_standardisation(
                X, y, config.get("boundary_constraint")
            )
        apply_standardisation(X, y, standardisation)

    model, likelihood, noise_model = build_model(
        X, y, config, num_species, channel_mask, X_noise=X_raw,
        seed_init=seed_init, conc_idxs=conc_idxs,
    )

    if oracle is not None:
        model_state, noise_state, _ = oracle
        model.load_state_dict(model_state, strict=True)
        noise_model.load_state_dict(noise_state, strict=True)
        # The noise model reads raw block magnitudes, so it is fed the
        # unstandardised inputs whatever the GP sees.
        likelihood.noise = noise_model(X_raw)
    else:
        train_model(model, likelihood, noise_model, X, y, config, X_noise=X_raw)

    model.eval()
    likelihood.eval()

    # Every species this fit covers, in order, off one seeded stream. See
    # predict_all_species.
    order = species_order if species_order is not None else [species_idx]
    truths = ground_truth
    if truths is None and ground_truth_max is not None:
        truths = {species_idx: ground_truth_max}

    predictions = predict_all_species(
        model, likelihood, noise_model, drug, order, config,
        ground_truth=truths, standardisation=standardisation,
        seed_posterior=seed_posterior,
    )
    return (
        predictions[species_idx],
        (model, likelihood, noise_model, standardisation, predictions),
    )
