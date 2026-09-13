"""Loading and splitting the measured heart-rate data.

The same transfer question on clinical data. Each subject contributes one
resting heart rate (``Baseline``) and one intrinsic rate measured after
intervention, so the model has a single covariate rather than a
seven-dimensional block vector, and the target is a *held-out subject* rather
than a held-out drug.

Two endpoints, each with its own high-fidelity species:

======================  ==================  ==================================
``qoi``                 High fidelity       Meaning
======================  ==================  ==================================
``autonomic_block``     Human               Rate under pharmacological
                                            blockade of autonomic input
``sa_node``             Dog                 Rate of the isolated sino-atrial
                                            node preparation
======================  ==================  ==================================

Subjects are split into ``Sedentary`` and ``Trained``. 

Columns ``Autonomic_Block`` and ``SA_Node`` are sparsely populated by design; a
config's ``qoi`` selects one and the rows missing it are dropped, which is what
sets each species' usable subject count.
"""

import random

import pandas as pd
import torch

#: Config ``qoi`` value to the column holding it.
QOI_COLUMNS = {
    "autonomic_block": "Autonomic_Block",
    "sa_node": "SA_Node",
}

#: Fitness group whose subjects are held out by the leave-out-athletes configs.
ATHLETE_GROUP = "Trained"


def load_heart_rate(path, species_list, qoi):
    """Usable subjects per species for one endpoint.

    Returns:
        ``{species: DataFrame}``, each with the endpoint column non-null.
    """
    qoi_column = QOI_COLUMNS[qoi]
    df = pd.read_csv(path, float_precision="round_trip")
    return {
        species: (
            df[df["Species"] == species]
            .dropna(subset=["Baseline", qoi_column])
            .reset_index(drop=True)
        )
        for species in species_list
    }


def sample_split(high_fidelity_df, low_fidelity_df, num_hf, num_lf, repeat,
                 selection_seed):
    """Draw a training set; the test set is every remaining high-fidelity subject.

    Only the high-fidelity subjects are split; the low-fidelity species is
    training data throughout.

    Returns:
        ``(training_frames, test_frame)``. The training frames are ordered with
        the high-fidelity species **last**, which
        :class:`~gp.models.MultitaskGPModel` requires.
    """
    n = len(high_fidelity_df)
    random.seed(selection_seed * 1000 + repeat)
    train_idx = random.sample(range(n), num_hf)
    test_idx = [i for i in range(n) if i not in train_idx]

    frames = []
    if num_lf > 0 and low_fidelity_df is not None and len(low_fidelity_df):
        frames.append(
            low_fidelity_df.sample(
                n=min(num_lf, len(low_fidelity_df)),
                random_state=selection_seed + repeat,
            )
        )
    frames.append(high_fidelity_df.iloc[train_idx])

    return frames, high_fidelity_df.iloc[test_idx]


def k_fold_split(high_fidelity_df, low_fidelity_df, k, num_lf, repeat,
                 selection_seed):
    """Hold out ``k`` high-fidelity subjects; train on all the rest.

    Distinct from :func:`sample_split`, which draws the *training* set. Here the
    draw is of the **test** set: ``k`` is the number held out, and the published
    table's ``num_train_humans`` column is ``N - k``. Not interchangeable even
    at matched sizes, since the two consume the seeded stream differently.

    The training subjects are taken in ascending index order, the complement
    being built by a scan.
    """
    n = len(high_fidelity_df)
    random.seed(selection_seed * 1000 + repeat)
    test_idx = random.sample(range(n), k)
    train_idx = [i for i in range(n) if i not in test_idx]

    frames = []
    if num_lf > 0 and low_fidelity_df is not None and len(low_fidelity_df):
        count = min(num_lf, len(low_fidelity_df))
        frames.append(
            low_fidelity_df.sample(n=count, random_state=selection_seed + repeat)
        )
    frames.append(high_fidelity_df.iloc[train_idx])

    return frames, high_fidelity_df.iloc[test_idx]


def athlete_split(high_fidelity_df, low_fidelity_df, num_lf, repeat,
                  selection_seed):
    """Train on sedentary high-fidelity subjects, test on the trained ones.

    The two fitness groups occupy different parts of the baseline-rate range, so
    this is extrapolation rather than interpolation.
    """
    sedentary = high_fidelity_df[high_fidelity_df["Fitness"] != ATHLETE_GROUP]
    trained = high_fidelity_df[high_fidelity_df["Fitness"] == ATHLETE_GROUP]

    frames = []
    if num_lf > 0 and low_fidelity_df is not None and len(low_fidelity_df):
        if num_lf >= len(low_fidelity_df):
            # Every low-fidelity subject, in dataset order. Not
            # `.sample(n=len)`, which shuffles: row order sets the summation
            # order in the marginal likelihood.
            frames.append(low_fidelity_df)
        else:
            frames.append(
                low_fidelity_df.sample(
                    n=num_lf, random_state=selection_seed + repeat,
                )
            )
    frames.append(sedentary)

    return frames, trained


def build_tensors(training_frames, species_idx, qoi):
    """Stack per-species frames into ``(X, y)``.

    ``X`` is ``(N, 2)``: the baseline rate, then the species index.
    """
    qoi_column = QOI_COLUMNS[qoi]
    rows_x, rows_y = [], []
    for frame in training_frames:
        if frame.empty:
            continue
        index = float(species_idx[frame["Species"].iloc[0]])
        for _, row in frame.iterrows():
            rows_x.append([row["Baseline"], index])
            rows_y.append(row[qoi_column])

    X = torch.tensor(rows_x, dtype=torch.float32)
    y = torch.tensor(rows_y, dtype=torch.float32)
    return X, y


def standardise(X, y, num_species):
    """Per-species standardisation of the covariate and the response.

    Species differ several-fold in resting heart rate, so without this the
    shared kernel is dominated by the between-species offset.
    """
    x_means = torch.zeros(num_species)
    x_stds = torch.ones(num_species)
    y_means = torch.zeros(num_species)
    y_stds = torch.ones(num_species)

    for index in range(num_species):
        mask = X[:, -1] == float(index)
        if mask.sum() == 0:
            continue
        x_species, y_species = X[mask, 0], y[mask]
        x_means[index] = x_species.mean()
        x_stds[index] = x_species.std() if x_species.std() > 0 else torch.tensor(1.0)
        y_means[index] = y_species.mean()
        y_stds[index] = y_species.std() if y_species.std() > 0 else torch.tensor(1.0)

    X_std, y_std = X.clone(), y.clone()
    for index in range(num_species):
        mask = X[:, -1] == float(index)
        if mask.sum() == 0:
            continue
        X_std[mask, 0] = (X[mask, 0] - x_means[index]) / x_stds[index]
        y_std[mask] = (y[mask] - y_means[index]) / y_stds[index]

    return X_std, y_std, {
        "x_means": x_means, "x_stds": x_stds,
        "y_means": y_means, "y_stds": y_stds,
    }


def athlete_standardisation(data, species_idx, qoi):
    """Standardisation for the leave-out-athletes split, from the full dataset.

    Asymmetric, and not derivable from the training tensor:

    * **x** statistics use *every* subject of a species, athletes included; a
      scale fitted without them puts the held-out subjects far outside the
      range the kernel has support over.
    * **y** statistics use **sedentary subjects only**, since the athletes'
      responses are the quantity being predicted.

    Returns:
        The same four-tensor dict :func:`standardise` returns, ready to apply.
    """
    qoi_column = QOI_COLUMNS[qoi]
    num_species = len(species_idx)
    x_means = torch.zeros(num_species)
    x_stds = torch.ones(num_species)
    y_means = torch.zeros(num_species)
    y_stds = torch.ones(num_species)

    for species, index in species_idx.items():
        frame = data.get(species)
        if frame is None or frame.empty:
            continue
        sedentary = frame[frame["Fitness"] != ATHLETE_GROUP]
        if sedentary.empty:
            continue
        baselines = frame["Baseline"].dropna().to_numpy(dtype=float)
        responses = sedentary[qoi_column].dropna().to_numpy(dtype=float)
        if baselines.size == 0 or responses.size == 0:
            continue
        x_means[index] = float(baselines.mean())
        x_stds[index] = max(float(baselines.std()), 1e-8)
        y_means[index] = float(responses.mean())
        y_stds[index] = max(float(responses.std()), 1e-8)

    return {"x_means": x_means, "x_stds": x_stds,
            "y_means": y_means, "y_stds": y_stds}


def apply_standardisation(X, y, standardisation):
    """Apply a precomputed standardisation to ``(X, y)``, returning copies."""
    X_std, y_std = X.clone(), y.clone()
    for index in range(standardisation["x_means"].numel()):
        mask = X[:, -1] == float(index)
        if mask.sum() == 0:
            continue
        X_std[mask, 0] = (
            X[mask, 0] - standardisation["x_means"][index]
        ) / standardisation["x_stds"][index]
        y_std[mask] = (
            y[mask] - standardisation["y_means"][index]
        ) / standardisation["y_stds"][index]
    return X_std, y_std
