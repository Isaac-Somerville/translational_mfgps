"""Heart-rate transfer experiment: the same question on measured data.

Each subject contributes a resting heart rate and an intrinsic rate. The model
predicts the intrinsic rate of **held-out subjects** of the high-fidelity
species, either with low-fidelity subjects of another species in the training
set or without them.

Two modes:

``k-sweep`` (default)
    Sweep the number of high-fidelity training subjects from 1 upwards, holding
    out the rest. 

``leave-out-athletes`` (``leave_out_athletes`` in the config)
    Train on sedentary subjects, test on trained ones. The two groups sit in
    different parts of the resting-rate range, so this is extrapolation.

Outputs, under ``outputs/heart_rate/<config>/``:
    ``scores.csv``       one row per repeat, subject and arm
    ``k_sweep.pdf``      posterior fits at each k, and the CRPS improvement
    ``leave_out_athletes.pdf``  the three posterior fits

Examples:
    python -m heart_rate.run_heart_rate --config hr_human_from_dog_autonomic
    python -m heart_rate.run_heart_rate --config hr_dog_from_mouse_sa_node_athletes
"""

import argparse
import json

import numpy as np
import pandas as pd

from gp.fitting import build_model, train_model
from gp.metrics import compute_crps, quantile_interval
from heart_rate.data import (apply_standardisation, athlete_split,
                             k_fold_split,
                             athlete_standardisation, build_tensors,
                             load_heart_rate,
                             sample_split, standardise, QOI_COLUMNS)
from heart_rate.plotting import plot_athlete_leave_out, plot_k_sweep
from paths import CONFIG_DIR, HEART_RATE_OUT, HEART_RATE_TABLE, ensure

import gpytorch
import torch


def load_config(name):
    path = CONFIG_DIR / (name if name.endswith(".json") else f"{name}.json")
    with open(path) as f:
        return json.load(f)


def config_for_repeat(config, repeat):
    """Config with the training seed advanced to this repeat.

    The heart-rate sweeps seed training on ``training_seed + repeat``, so each
    repeat starts from a different initialisation. Only the leave-out-athletes
    fit, a single deterministic split, uses the training seed unmodified.
    """
    return {**config, "training_seed": config.get("training_seed", 42) + repeat}


def fit_once(X, y, config, num_species, standardisation=None):
    """Fit on one training set. No channel mask: the input is one covariate.

    ``standardisation``, when given, replaces one computed from this training
    set; the leave-out-athletes split needs statistics from the full dataset.
    See :func:`heart_rate.data.athlete_standardisation`.
    """
    if standardisation is None:
        X_std, y_std, standardisation = standardise(X, y, num_species)
    else:
        X_std, y_std = apply_standardisation(X, y, standardisation)
    channel_mask = torch.ones(max(num_species - 1, 0), 1, dtype=torch.float32)

    model, likelihood, noise_model = build_model(
        X_std, y_std, config, num_species, channel_mask
    )
    # The heart-rate scripts leave the likelihood's noise as the final loop
    # iteration set it. See gp.fitting.train_model.
    train_model(model, likelihood, noise_model, X_std, y_std, config,
                refresh_noise=False)
    model.eval()
    likelihood.eval()
    noise_model.eval()
    return model, likelihood, noise_model, standardisation


def score_subjects(model, likelihood, noise_model, test_df, hf_index,
                   standardisation, config, qoi):
    """Score every held-out subject. One record each."""
    qoi_column = QOI_COLUMNS[qoi]
    num_samples = config.get("num_post_samples", 500)
    records = []

    # Seeded once for the whole test set, so successive subjects draw from one
    # stream.
    torch.manual_seed(config.get("post_samples_seed", 42))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for _, row in test_df.iterrows():
            baseline = float(row["Baseline"])
            truth = float(row[qoi_column])

            x_test = torch.tensor([[baseline, float(hf_index)]], dtype=torch.float32)
            if standardisation is not None:
                x_test[0, 0] = (
                    (baseline - standardisation["x_means"][hf_index])
                    / standardisation["x_stds"][hf_index]
                )

            latent = model(x_test)
            predictive = likelihood(latent, noise=noise_model(x_test))
            samples = predictive.sample(
                sample_shape=torch.Size([num_samples])
            ).squeeze()
            mean = predictive.mean.squeeze()
            # Standardised quantities, so variances rescale by the square.
            variance_scale = 1.0
            if standardisation is not None:
                scale = standardisation["y_stds"][hf_index]
                offset = standardisation["y_means"][hf_index]
                samples = samples * scale + offset
                mean = mean * scale + offset
                variance_scale = float(scale) ** 2

            records.append({
                "subject_idx": int(row["Subject_Idx"]),
                "fitness": row["Fitness"],
                "baseline_hr": baseline,
                "true_qoi": truth,
                "pred_mean": float(mean.item()),
                "crps": compute_crps(samples, truth),
                "sq_error": (float(mean.item()) - truth) ** 2,
                "f_var": latent.variance.item() * variance_scale,
                "y_var": predictive.variance.item() * variance_scale,
            })
    return records


def posterior_curves(model, likelihood, noise_model, species_index,
                     baselines, standardisation, num_points):
    """The fitted model's posterior across one species' baseline-rate range.

    The grid is padded 10% beyond the observed range at each end, which is
    where the held-out athletes sit.

    Args:
        model, likelihood, noise_model: The fitted arm.
        species_index: Which species to predict for.
        baselines: That species' observed baseline rates, all subjects.
        standardisation: Per-species standardisation, or None.
        num_points: Points on the grid.

    Returns:
        Dict of ``hr_grid``, ``mean``, ``f_lo``/``f_hi``, ``y_lo``/``y_hi``.
    """
    grid = np.linspace(float(np.min(baselines)) * 0.9,
                       float(np.max(baselines)) * 1.1, num_points)
    x_grid = torch.tensor([[hr, float(species_index)] for hr in grid],
                          dtype=torch.float32)
    x_model = x_grid.clone()
    if standardisation is not None:
        x_model[:, 0] = (
            (x_grid[:, 0] - standardisation["x_means"][species_index])
            / standardisation["x_stds"][species_index]
        )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent = model(x_model)
        predictive = likelihood(latent, noise=noise_model(x_model))
        f_lo, f_hi = latent.confidence_region()
        y_lo, y_hi = predictive.confidence_region()
        mean = predictive.mean.clone()

    if standardisation is not None:
        scale = standardisation["y_stds"][species_index]
        offset = standardisation["y_means"][species_index]
        mean, f_lo, f_hi, y_lo, y_hi = (
            value * scale + offset for value in (mean, f_lo, f_hi, y_lo, y_hi)
        )

    return {
        "hr_grid": grid,
        "mean": mean.numpy(),
        "f_lo": f_lo.numpy(), "f_hi": f_hi.numpy(),
        "y_lo": y_lo.numpy(), "y_hi": y_hi.numpy(),
    }


def run_arm(train_frames, test_df, species_idx, hf_species, config, qoi,
            standardisation=None):
    """Fit one arm and score the held-out subjects."""
    X, y = build_tensors(train_frames, species_idx, qoi)
    if X.numel() == 0:
        return [], None
    num_species = len(species_idx)
    model, likelihood, noise_model, standardisation = fit_once(
        X, y, config, num_species, standardisation=standardisation
    )
    records = score_subjects(
        model, likelihood, noise_model, test_df, species_idx[hf_species],
        standardisation, config, qoi,
    )
    return records, (model, likelihood, noise_model, standardisation)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = args.config.replace(".json", "")
    if args.repeats is not None:
        config["num_repeats"] = args.repeats

    hf_species = config["high_fidelity_species"]
    lf_species = config["low_fidelity_species"][0]
    qoi = config["qoi"]
    # High fidelity last: MultitaskGPModel requires it.
    species_list = [lf_species, hf_species]
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    data = load_heart_rate(HEART_RATE_TABLE, species_list, qoi)
    hf_df, lf_df = data[hf_species], data[lf_species]
    print(f"{qoi}: {hf_species} n={len(hf_df)} (high fidelity), "
          f"{lf_species} n={len(lf_df)} (low fidelity)")

    out_dir = ensure(HEART_RATE_OUT / config_name)
    records = []

    if config.get("leave_out_athletes"):
        num_post_concs = config.get("num_post_concs", 100)
        posteriors, training_frames, test_frame = {}, {}, None

        # No repeat loop: the split is deterministic, so there is nothing to
        # resample.
        standardisation_override = athlete_standardisation(data, species_idx, qoi)
        for arm, num_lf in {"hf_only": 0, "hf_lf": len(lf_df)}.items():
            frames, test_df = athlete_split(
                hf_df, lf_df, num_lf, 0, config["subject_selection_seed"]
            )
            scored, fitted = run_arm(
                frames, test_df, species_idx, hf_species, config, qoi,
                standardisation=standardisation_override,
            )
            for record in scored:
                records.append({**record, "arm": arm, "k": None})

            if fitted is None:
                continue
            model, likelihood, noise_model, standardisation = fitted
            drawn = [hf_species] if arm == "hf_only" else species_list
            posteriors[arm] = {
                species: posterior_curves(
                    model, likelihood, noise_model, species_idx[species],
                    data[species]["Baseline"].to_numpy(), standardisation,
                    num_post_concs,
                )
                for species in drawn
            }
            if arm == "hf_lf":
                training_frames = {
                    frame["Species"].iloc[0]: frame
                    for frame in frames if not frame.empty
                }
                test_frame = test_df
        mode = "leave_out_athletes"
    else:
        # k is the number of high-fidelity subjects **held out**, not the number
        # trained on — the published table's num_train_humans column is N - k.
        # Ranges over 1 .. n-1 so at least one subject remains on each side.
        k_values = list(range(1, len(hf_df)))
        for k in k_values:
            for repeat in range(config["num_repeats"]):
                for arm, num_lf in {"hf_only": 0, "hf_lf": len(lf_df)}.items():
                    frames, test_df = k_fold_split(
                        hf_df, lf_df, k, num_lf, repeat,
                        config["subject_selection_seed"],
                    )
                    # Single-task, not two-task with an empty task: the number
                    # of tasks sets the coregionalisation factor's shape.
                    cell_species = species_list if num_lf > 0 else [hf_species]
                    cell_idx = {sp: i for i, sp in enumerate(cell_species)}
                    scored, _ = run_arm(
                        frames, test_df, cell_idx, hf_species,
                        config_for_repeat(config, repeat), qoi,
                    )
                    for record in scored:
                        records.append({**record, "repeat": repeat, "arm": arm, "k": k})
            median = np.nanmedian([
                r["crps"] for r in records if r["k"] == k and r["arm"] == "hf_lf"
            ])
            print(f"  k={k:2d}  median CRPS (HF+LF) = {median:.4f}")
        mode = "k_sweep"

    if not records:
        print("No results produced.")
        return

    scores = pd.DataFrame(records)
    scores.to_csv(out_dir / "scores.csv", index=False)

    if not args.no_plots:
        if mode == "k_sweep":
            plot_k_sweep(scores, hf_species, lf_species, qoi,
                         save_path=out_dir / "k_sweep.pdf",
                         num_subjects=len(hf_df))
        else:
            plot_athlete_leave_out(
                posteriors, training_frames, test_frame,
                hf_species, lf_species, qoi,
                save_path=out_dir / "leave_out_athletes.pdf",
            )

    print(f"\nWrote {len(scores)} rows to {out_dir}")


if __name__ == "__main__":
    main()
