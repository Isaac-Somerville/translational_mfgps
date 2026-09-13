"""Turning the APD dataset into the tensors the GP consumes.

The dataset is one row per (species, drug, subject, concentration). The GP sees
each row as a point whose input is the drug's per-channel block at that
concentration, tagged with a species index, and whose output is the observed
relative APD change.

Species ordering
----------------
The high-fidelity species is always **last**.
:class:`~gp.models.MultitaskGPModel` treats index ``num_species - 1`` as the
species whose kernel sees every block dimension, and several callers index it
that way.

Standardisation
---------------
Inputs and outputs are standardised per species. Under a boundary constraint the
*centring* is dropped and only the scaling kept, since the constraint forces
``f(0) = 0``.
"""

import random

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

from drug import ALL_CHANNELS, Drug

#: LOWESS span used wherever the population's mean response is drawn.
LOWESS_FRACTION = 0.5

#: How many times the origin is repeated before smoothing, so the curve stays
#: anchored at (0, 0).
ORIGIN_WEIGHT = 5

#: Species in this release.
ALL_SPECIES = ["Guinea Pig", "Mouse", "Dog", "Pig", "Rabbit", "Human"]

#: Species index used to seed subject sampling.
#:
#: Not positions in :data:`ALL_SPECIES`; the gaps are deliberate.
#: :func:`sample_subjects` seeds on
#: ``100 * repeat + selection_seed * SPECIES_IDX[species]``, so an index is an
#: input to the RNG rather than a label: renumbering these, or closing the gaps,
#: changes which subjects every repeat draws and reproduces none of the
#: published results.
SPECIES_IDX = {
    "Guinea Pig": 0,
    "Mouse": 1,
    "Dog": 2,
    # 3 and 4 belonged to models not included in this release
    "Pig": 5,
    "Rabbit": 6,
    "Human": 7,
}

#: The same for the iterative experiments, which order dog, guinea pig and mouse
#: differently. Since the index seeds the sampler, reusing :data:`SPECIES_IDX`
#: here draws different subjects, so the two maps are kept separate.
ITERATIVE_SPECIES_IDX = {
    "Dog": 0,
    "Guinea Pig": 1,
    "Mouse": 2,
    # 3 and 4 belonged to models not included in this release
    "Pig": 5,
    "Rabbit": 6,
    "Human": 7,
}

#: Column holding the observation the GP is fitted to.
OBSERVATION_COLUMN = "APD_90_relative_noisy"

#: Column holding the noise-free truth the model is scored against.
TRUTH_COLUMN = "APD_90_relative"


def smoothed_population_mean(frame, x_dense, column=TRUTH_COLUMN,
                             frac=LOWESS_FRACTION):
    """The population's mean response, smoothed and evaluated on ``x_dense``.

    Smooths the per-concentration means rather than the raw subject points. The
    origin is repeated :data:`ORIGIN_WEIGHT` times so every curve is anchored at
    (0, 0).

    Args:
        frame: Rows for one drug and one species.
        x_dense: Concentrations to evaluate the smoothed curve at.
        column: Column to average.
        frac: LOWESS span.

    Returns:
        Array the shape of ``x_dense``, or None if there is too little data.
    """
    means = frame.groupby("Drug_Concentration")[column].mean().sort_index()
    x = means.index.values.astype(float)
    y = means.values.astype(float)
    if len(x) == 0:
        return None
    if len(x) == 1:
        return np.full_like(np.asarray(x_dense, dtype=float), y[0])

    if np.any(x == 0.0):
        x = np.concatenate([x, np.zeros(ORIGIN_WEIGHT)])
        y = np.concatenate([y, np.zeros(ORIGIN_WEIGHT)])

    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    # The repeated origin comes back as duplicate x, which interp1d would
    # divide by; average the duplicates away first.
    unique_x, inverse = np.unique(smoothed[:, 0], return_inverse=True)
    unique_y = (np.bincount(inverse, weights=smoothed[:, 1])
                / np.bincount(inverse))
    interpolate = interp1d(unique_x, unique_y, kind="linear",
                           fill_value="extrapolate")
    return interpolate(x_dense)


def load_apd_dataset(path, drug=None, species=None, drop_missing=True):
    """Load the APD dataset, optionally filtered.

    ``float_precision="round_trip"`` because pandas' default CSV float parser is
    accurate only to within one ulp.

    ``drop_missing`` removes rows with no ground truth **at load**, before any
    subject sampling. Dropping afterwards would leave all-missing subjects in the
    sampling pool, shifting every later draw. Keyed on the truth column rather
    than the observation column: a row with no truth cannot be scored.
    """
    df = pd.read_csv(path, float_precision="round_trip")
    if drop_missing:
        df = df.dropna(subset=[TRUTH_COLUMN])
    if drug is not None:
        df = df[df["Drug"].isin([drug] if isinstance(drug, str) else drug)]
    if species is not None:
        df = df[df["Species"].isin([species] if isinstance(species, str) else species)]
    return df


def filter_training_concentrations(df, species_conc_idxs, conc_multiplier=None):
    """Restrict training rows to the concentrations each species is allowed.

    The model predicts the *maximum* relative APD change over a concentration
    range while training only on the lowest few concentrations in that range, so
    the task is extrapolation to unobserved exposures.

    Indices are positional into that (species, drug) pair's sorted
    concentrations, so ``[0, 1, 2, 3]`` means the drug-free case plus the three
    lowest doses, whatever those are in nM for that drug.

    Args:
        df: Rows to filter.
        species_conc_idxs: Species name to its list of allowed indices. A
            species mapped to an empty list or absent keeps every concentration.
        conc_multiplier: If given, also drop concentrations above
            ``conc_multiplier * c_therapeutic``, so conditioning data never
            extends past the range being predicted over.

            The single-drug and iterative experiments pass None and filter by
            index alone; the heatmap sweep applies the ceiling. Inert for every
            published config, whose selected indices already sit below its
            ceiling, but the two are kept distinct so a future config means what
            it says.
    """
    parts = []
    for (drug_name, species), group in df.groupby(["Drug", "Species"], sort=False):
        idxs = species_conc_idxs.get(species)
        if not idxs:
            parts.append(group)
            continue

        available = sorted(group["Drug_Concentration"].unique())
        allowed = [available[i] for i in idxs if i < len(available)]
        if conc_multiplier is not None:
            ceiling = conc_multiplier * Drug(drug_name).c_therapeutic
            allowed = [c for c in allowed if c <= ceiling]
        parts.append(group[group["Drug_Concentration"].isin(allowed)])

    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def max_subjects_available(df, species, drug_list=None):
    """Largest subject count every drug in ``drug_list`` can supply for a species.

    Counting distinct subjects across all drugs overstates availability: the
    dataset holds 29 human subjects overall but only 18-25 for any single drug.
    Takes the minimum over the drugs being fitted.
    """
    species_df = df[df["Species"] == species]
    if not drug_list:
        return species_df["Subject_ID"].nunique()
    return min(
        species_df[species_df["Drug"] == d]["Subject_ID"].nunique() for d in drug_list
    )


def sample_subjects(df, drug_list, species_list, num_human, num_animal,
                    repeat, subject_selection_seed,
                    human_conc_idxs=None, animal_conc_idxs=None,
                    species_idx_map=None):
    """Draw a per-repeat training set of subjects.

    Subjects are drawn per (species, drug), so every drug is represented by the
    requested number wherever the data allows.

    The seed depends on the repeat index and the species index, so repeat *r*
    draws the same subjects however many species are in the experiment: the
    human subjects are identical in the HF-only and HF+LF arms.
    ``species_idx_map`` selects which index table supplies that seed; the
    iterative experiments need :data:`ITERATIVE_SPECIES_IDX`.

    Two details of the draw are load-bearing:

    **The candidate pool is per species, not per (species, drug).** With several
    drugs in play a subject missing for one of them is still drawn as a candidate
    and then rejected, consuming a draw from the random stream.

    **A subject with fewer rows than the requested number of concentrations is
    rejected, not truncated** — but only in the single-drug and iterative
    experiments. The heatmap sweep rejects only subjects with no rows at all, so
    it passes ``None`` where the others pass their concentration index lists.

    Returns:
        The sampled DataFrame, or None if any (species, drug) pair has no
        usable subject, in which case the repeat cannot be run.
    """
    frames = []

    for species in species_list:
        num_to_select = num_human if species == "Human" else num_animal
        if num_to_select == 0:
            continue
        conc_idxs = human_conc_idxs if species == "Human" else animal_conc_idxs
        # None: reject only a subject with no rows. A list: reject one that
        # cannot supply every requested concentration.
        min_rows = 1 if conc_idxs is None else len(conc_idxs)

        idx_map = SPECIES_IDX if species_idx_map is None else species_idx_map
        random.seed(int(100 * repeat + subject_selection_seed * idx_map[species]))

        available = list(df[df["Species"] == species]["Subject_ID"].unique())
        if not available:
            print(f"  [warn] no subjects for {species}")
            return None

        for drug in drug_list:
            chosen = set()
            rejected = set()
            while len(chosen) < num_to_select and len(chosen) + len(rejected) < len(
                available
            ):
                subject_id = random.sample(available, 1)[0]
                if subject_id in chosen or subject_id in rejected:
                    continue
                rows = df[
                    (df["Species"] == species)
                    & (df["Drug"] == drug)
                    & (df["Subject_ID"] == subject_id)
                ]
                if len(rows) < min_rows:
                    rejected.add(subject_id)
                    continue
                frames.append(rows.copy())
                chosen.add(subject_id)

            if not chosen:
                print(f"  [warn] no usable subjects for {species}/{drug}")
                return None

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def sample_subjects_iterative(df, drug_list, species_list, num_human, num_animal,
                              repeat, subject_selection_seed,
                              human_conc_idxs=None, animal_conc_idxs=None,
                              seed_offset=0):
    """The iterative experiments' subject draw.

    Separate from :func:`sample_subjects` rather than a flag on it, because the
    two experiments differ in four ways at once:

    ===========================  ======================  =======================
    Behaviour                    single-drug             iterative
    ===========================  ======================  =======================
    Candidate pool               per species             per (species, drug)
    ``random.seed`` called       once per species        once per (species, drug)
    Species index table          :data:`SPECIES_IDX`     :data:`ITERATIVE_SPECIES_IDX`
    A drug with no subjects      abandons the repeat     skipped, others continue
    ===========================  ======================  =======================

    ``seed_offset`` separates the three draws a single repeat makes — the prior
    drugs (0), the target drug's animal subjects (1000) and its human subjects
    (2000) — so they do not replay the same stream. The offsets are load-bearing:
    changing them changes which subjects are drawn.

    The under-length rejection is shared with :func:`sample_subjects`.

    Returns:
        ``(sampled DataFrame or None, skip flag)``. The flag is True when some
        (species, drug) pair that *had* candidates could supply none of them, in
        which case the caller abandons the target drug.
    """
    frames = []
    skip = False

    for species in species_list:
        num_to_select = num_human if species == "Human" else num_animal
        conc_idxs = human_conc_idxs if species == "Human" else animal_conc_idxs
        if num_to_select == 0 or not conc_idxs:
            continue
        if df[df["Species"] == species].empty:
            continue

        for drug in drug_list:
            drug_rows = df[(df["Species"] == species) & (df["Drug"] == drug)]
            available = list(drug_rows["Subject_ID"].unique())
            if not available:
                continue

            random.seed(int(
                100 * repeat
                + seed_offset
                + subject_selection_seed * ITERATIVE_SPECIES_IDX[species]
            ))

            chosen, rejected = set(), set()
            while len(chosen) < num_to_select and len(chosen) + len(rejected) < len(
                available
            ):
                subject_id = random.sample(available, 1)[0]
                if subject_id in chosen or subject_id in rejected:
                    continue
                rows = drug_rows[drug_rows["Subject_ID"] == subject_id]
                if len(rows) < len(conc_idxs):
                    rejected.add(subject_id)
                    continue
                frames.append(rows.copy())
                chosen.add(subject_id)

            if not chosen:
                print(f"  [warn] no suitable subjects for {species}/{drug}")
                skip = True
                break
        if skip:
            break

    if not frames:
        return None, skip
    return pd.concat(frames, ignore_index=True), skip


def add_block_columns(df):
    """Add one ``{channel}_block`` column per channel, via the Hill equation.

    The block fraction depends only on (drug, concentration), so it is computed
    once per distinct pair rather than per row.

    A channel the drug has no parameters for, or that the species model does not
    represent, stays 0.
    """
    df = df.copy()
    for channel in ALL_CHANNELS:
        column = f"{channel}_block"
        if column not in df.columns:
            df[column] = 0.0

    for drug_name, drug_rows in df.groupby("Drug"):
        drug = Drug(drug_name)
        concentrations = np.sort(drug_rows["Drug_Concentration"].unique())
        blocks = drug.calculate_all_inhibitions(
            concentrations=concentrations, return_as_list=False
        )
        conc_position = {c: i for i, c in enumerate(concentrations)}
        for channel, values in blocks.items():
            column = f"{channel}_block"
            df.loc[drug_rows.index, column] = [
                values[conc_position[c]] for c in drug_rows["Drug_Concentration"]
            ]

    # Zero concentration means zero block, whatever the Hill parameters say.
    df.loc[df["Drug_Concentration"] == 0.0, [f"{c}_block" for c in ALL_CHANNELS]] = 0.0
    return df


def build_tensors(df, species_idx_dict, conc_as_input=False, conc_multiplier=1):
    """Build ``(X, y)`` from a training frame.

    Args:
        df: Rows with block columns already added.
        species_idx_dict: Species name to the index used in this experiment.
            The high-fidelity species must map to the largest index.
        conc_as_input: Use scaled concentration as a single input dimension
            instead of the seven-dimensional block vector. A one-dimensional
            input for a single drug, which is the same information — block is a
            monotone function of concentration for a fixed drug — in a form the
            GP can fit from far fewer subjects.
        conc_multiplier: Concentration ceiling as a multiple of therapeutic,
            used to scale concentration onto roughly [0, 1].

    Returns:
        ``(X, y)`` where X is ``(N, D+1)`` with the species index last.
    """
    df = df.copy()
    df["Species_idx"] = [species_idx_dict[s] for s in df["Species"].values]

    if conc_as_input:
        drug = Drug(df["Drug"].iloc[0])
        max_conc = conc_multiplier * drug.c_therapeutic
        x_block = (
            torch.tensor(df["Drug_Concentration"].values, dtype=torch.float32)
            / max_conc
        ).unsqueeze(1)
    else:
        block_columns = [f"{channel}_block" for channel in ALL_CHANNELS]
        x_block = torch.tensor(df[block_columns].values, dtype=torch.float32)

    y = torch.tensor(df[OBSERVATION_COLUMN].values, dtype=torch.float32)
    x_species = torch.tensor(
        df["Species_idx"].values, dtype=torch.float32
    ).unsqueeze(1)
    return torch.cat((x_block, x_species), dim=1), y


def drop_zero_block_rows(X, y, boundary_constraint):
    """Drop training points at zero drug block, under a boundary constraint.

    :class:`~gp.models.BoundaryConstrainedKernel` forces ``f(0) = 0`` exactly, so
    a zero-block observation adds nothing and its kernel row is identically zero,
    leaving the covariance singular. Without the constraint those rows pin the
    intercept, so this applies only when the constraint is active.

    Returns ``(X, y)`` filtered, or unchanged when no constraint is set.
    """
    if boundary_constraint is None:
        return X, y
    keep = X[:, :-1].abs().sum(dim=1) > 0
    return X[keep], y[keep]


def compute_standardisation(X, y, boundary_constraint=None):
    """Per-species means and standard deviations for inputs and outputs.

    Under a boundary constraint the means are left at zero — centring would
    shift the origin away from the pinned ``f(0) = 0`` — so only the scale is
    standardised.
    """
    num_species = int(X[:, -1].max().item()) + 1
    num_features = X.shape[1] - 1

    x_means = torch.zeros(num_species, num_features)
    x_stds = torch.ones(num_species, num_features)
    y_means = torch.zeros(num_species)
    y_stds = torch.ones(num_species)

    for s in X[:, -1].unique().long():
        mask = X[:, -1] == s.float()
        x_species, y_species = X[mask, :-1], y[mask]

        if boundary_constraint is None:
            x_means[s] = x_species.mean(0)
        stds = x_species.std(0)
        stds[stds == 0] = 1.0  # a channel with no variation must not divide by zero
        x_stds[s] = stds

        y_mean, y_std = y_species.mean(), y_species.std()
        if boundary_constraint is None:
            y_means[s] = y_mean
        y_stds[s] = y_std if y_std > 0 else torch.tensor(1.0)

    return {"x_means": x_means, "x_stds": x_stds, "y_means": y_means, "y_stds": y_stds}


def apply_standardisation(X, y, params):
    """Standardise X's block columns and y in place, per species."""
    for s in X[:, -1].unique().long():
        mask = X[:, -1] == s.float()
        X[mask, :-1] = (X[mask, :-1] - params["x_means"][s]) / params["x_stds"][s]
        y[mask] = (y[mask] - params["y_means"][s]) / params["y_stds"][s]


def invert_standardisation(values, species_idx, params):
    """Map standardised outputs back to relative-APD units for one species."""
    return values * params["y_stds"][species_idx] + params["y_means"][species_idx]


def channel_mask_for(species_list, channels_table):
    """Which block dimensions each low-fidelity species actually has.

    ``(len(species_list) - 1, num_channels)``: one row per species except the
    high-fidelity one, which always sees every dimension.
    """
    availability = pd.read_csv(channels_table, index_col=0)
    return torch.tensor(
        [
            [float(availability.loc[channel, species]) for channel in ALL_CHANNELS]
            for species in species_list[:-1]
        ],
        dtype=torch.float32,
    )


#: Row cap on the maximal prior dataset; whole subjects are dropped until the
#: set fits.
MAX_ORACLE_ROWS = 2000


def select_all_prior_subjects(prior_rows, species_list, max_rows=MAX_ORACLE_ROWS):
    """The maximal prior dataset: every subject, capped by dropping whole ones.

    The oracle iterative configs condition on this rather than on a per-repeat
    draw, since the shipped hyperparameters were fitted to exactly this set.

    The cap drops **whole subjects**, never individual rows, in an order fixed by
    ``default_rng(42)``, so the result is identical across repeats and target
    drugs. Changing the seed or the cap invalidates the shipped checkpoints.
    """
    subject_df = prior_rows[prior_rows["Species"].isin(species_list)].copy()
    if len(subject_df) < max_rows:
        return subject_df

    rng = np.random.default_rng(seed=42)
    unique_ids = subject_df["Subject_ID"].unique().copy()
    rng.shuffle(unique_ids)

    kept, running = [], 0
    for subject_id in unique_ids:
        rows = int((subject_df["Subject_ID"] == subject_id).sum())
        if running + rows >= max_rows:
            break
        kept.append(subject_id)
        running += rows
    return subject_df[subject_df["Subject_ID"].isin(kept)]
