"""How does the benefit of animal data change with how much data you have?

The single-drug experiment fixes the training set at 3 human and 9 animal
subjects. This sweeps both axes: for each combination of ``num_humans`` and
``num_animals``, the model is refitted and scored, giving a surface rather than
a point.

Two figures per (species, drug):

1. **CRPS of the joint model.** The absolute performance surface.
2. **CRPS(animal-free) − CRPS(joint), paired by repeat.** What the animal data
   bought at that grid point. Positive means it helped.

The difference is paired within a repeat against the ``num_animals = 0`` column
of the *same* row. :func:`gp.data.sample_subjects` draws humans in a stream that
does not depend on ``num_animals``, so repeat *r* at ``(nh, na)`` and at
``(nh, 0)`` share an identical human draw, and differencing within the repeat
cancels the draw-to-draw variation.

Outputs, under ``outputs/experiments/<config>_sweep/<species>/<drug>/``:
    ``crps_per_repeat.csv``  every grid point and repeat
    ``paired_delta.csv``     the paired improvement per grid point
    ``heatmaps.pdf``         the two figures above

Example:
    python -m experiments.run_single_drug_sweep --config single_drug_1x \\
        --species Dog --drug quinidine
"""

import argparse

import numpy as np
import pandas as pd
import torch

from drug import load_drug_table
from gp.data import (OBSERVATION_COLUMN, add_block_columns, build_tensors,
                     channel_mask_for, drop_zero_block_rows,
                     filter_training_concentrations, load_apd_dataset,
                     max_subjects_available, sample_subjects)
from gp.fitting import fit_and_predict
from gp.plotting import plot_sweep_heatmaps
from paths import (APD_DATASET, CHANNELS_TABLE, EXPERIMENTS_OUT, GP_PARAMS,
                   ensure, ground_truth_table)
from experiments.fit_oracle import fit_oracle
from experiments.run_single_drug import (ANIMAL_SPECIES, HIGH_FIDELITY,
                                         load_config)

#: Step between grid points on both axes; every point is a full refit for every
#: repeat.
INCREMENT = 3

#: Repeats per grid point. Fewer than the headline experiment's 100, since the
#: sweep multiplies them by the number of grid points.
NUM_REPEATS = 10

#: Cap on both grid axes, regardless of how many subjects exist. None removes it.
MAX_AXIS = 30

#: Cap on the oracle's training set; whole subjects are dropped under a fixed
#: seed so the set is deterministic.
MAX_ORACLE_ROWS = 2000
ORACLE_SUBSAMPLE_SEED = 42

#: Torch threads the shipped sweep checkpoints were fitted with. See
#: :func:`sweep_oracle`.
ORACLE_THREADS = 4


def select_all_subjects(df, max_rows=MAX_ORACLE_ROWS, seed=ORACLE_SUBSAMPLE_SEED):
    """Every subject, capped at ``max_rows`` by dropping whole subjects.

    Deterministic, so the oracle fit does not depend on which grid point
    requested it first.
    """
    if len(df) < max_rows:
        return df
    rng = np.random.default_rng(seed=seed)
    subject_ids = df["Subject_ID"].unique().copy()
    rng.shuffle(subject_ids)

    kept, running = [], 0
    for subject_id in subject_ids:
        rows = int((df["Subject_ID"] == subject_id).sum())
        if running + rows >= max_rows:
            break
        kept.append(subject_id)
        running += rows
    return df[df["Subject_ID"].isin(kept)]


def sweep_oracle(unfiltered, species_list, config, channel_mask, cache_path):
    """Fit (or load) the sweep's own oracle hyperparameters.

    Fitted on the concentration-*unfiltered* data. Fitting it on the filtered
    concentrations instead collapses the prior response scale and shrinks the
    noise length-scale until the noise saturates immediately.

    Those are the same rows as the single-drug oracle in
    ``data/gp_params/<config>/``, but it is fitted with
    :data:`ORACLE_THREADS` torch threads rather than the single thread
    :mod:`experiments.fit_oracle` pins, so the two checkpoints differ
    numerically. The shipped sweep checkpoints were fitted with four, so 
    the count is pinned for the fit and restored afterwards.

    Returns the checkpoint path, fitting and caching it if absent.
    """
    if cache_path.exists():
        return cache_path
    previous = torch.get_num_threads()
    torch.set_num_threads(ORACLE_THREADS)
    try:
        return fit_oracle(select_all_subjects(unfiltered), species_list,
                          config, channel_mask, cache_path)
    finally:
        torch.set_num_threads(previous)


def sweep_grid(df, drug, species_list, max_humans_cap=MAX_AXIS,
               max_animals_cap=MAX_AXIS):
    """Grid axes, bounded by the data and by an explicit cap.

    Both axes are capped as well as bounded by availability: past roughly thirty
    subjects the surface has flattened, and every extra grid point costs a full
    refit for every repeat.
    """
    max_humans = max_subjects_available(df, HIGH_FIDELITY, [drug])
    max_animals = max_subjects_available(df, species_list[0], [drug])
    if max_humans_cap is not None:
        max_humans = min(max_humans, max_humans_cap)
    if max_animals_cap is not None:
        max_animals = min(max_animals, max_animals_cap)

    # Humans start at INCREMENT; animals start at 0, the baseline the paired
    # improvement is measured against.
    return (
        list(range(INCREMENT, max_humans + 1, INCREMENT)),
        list(range(0, max_animals + 1, INCREMENT)),
    )


def run_cell(subset, drug, species_list, species_idx, config, num_humans,
             num_animals, ground_truth, channel_mask, oracle, num_repeats):
    """Every repeat at one grid point."""
    records = []
    for repeat in range(num_repeats):
        arm_species = [HIGH_FIDELITY] if num_animals == 0 else species_list
        sampled = sample_subjects(
            subset, [drug], arm_species, num_humans, num_animals,
            repeat, config["subject_selection_seed"],
            # No concentration index lists: the sweep rejects only subjects
            # with no rows. See gp.data.sample_subjects.
        )
        if sampled is None:
            continue
        sampled = sampled.dropna(subset=[OBSERVATION_COLUMN])
        if sampled.empty:
            continue

        X, y = build_tensors(
            add_block_columns(sampled), species_idx,
            conc_as_input=config.get("conc_as_input", False),
            conc_multiplier=config["conc_multiplier"],
        )
        X, y = drop_zero_block_rows(X, y, config.get("boundary_constraint"))
        # The high-fidelity species **only**, in both arms. The single-drug
        # runner's joint arm predicts the animal first, leaving the RNG in a
        # different state before the human draw; since the seed is reset once
        # per grid point, an extra species here shifts every human sample.
        order = [species_idx[HIGH_FIDELITY]]
        result, _ = fit_and_predict(
            X, y, config, len(species_list), channel_mask, drug,
            species_idx[HIGH_FIDELITY],
            ground_truth_max=ground_truth,
            standardise=config.get("standardise_data", True),
            oracle_path=oracle, species_order=order,
        )
        records.append({
            "num_humans": num_humans, "num_animals": num_animals,
            "repeat": repeat, "crps": result["crps"],
        })
    return records


def paired_delta(per_repeat):
    """Paired improvement over the animal-free column, per grid point.

    Returns a frame with one row per grid point: the median paired difference,
    its 5th and 95th percentiles, and the fraction of repeats improved.
    """
    if 0 not in set(per_repeat["num_animals"]):
        return pd.DataFrame()

    rows = []
    for num_humans, block in per_repeat.groupby("num_humans"):
        baseline = block[block["num_animals"] == 0].set_index("repeat")["crps"]
        for num_animals, cell in block.groupby("num_animals"):
            values = cell.set_index("repeat")["crps"]
            shared = baseline.index.intersection(values.index)
            diffs = (baseline[shared] - values[shared]).to_numpy()
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size == 0:
                continue
            rows.append({
                "num_humans": num_humans, "num_animals": num_animals,
                "n_paired": diffs.size,
                "median_delta": float(np.median(diffs)),
                "p05": float(np.percentile(diffs, 5)),
                "p95": float(np.percentile(diffs, 95)),
                "frac_improved": float((diffs > 0).mean()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Config slug.")
    parser.add_argument("--species", nargs="+", default=["Dog"],
                        choices=ANIMAL_SPECIES)
    parser.add_argument("--drug", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = args.config.replace(".json", "")
    drug_list = args.drug or load_drug_table().index.tolist()

    df = load_apd_dataset(args.dataset or APD_DATASET)
    ground_truth = pd.read_csv(
        ground_truth_table(config["conc_multiplier"]), index_col=0
    )

    for animal_species in args.species:
        species_list = [animal_species, HIGH_FIDELITY]
        species_idx = {sp: i for i, sp in enumerate(species_list)}
        channel_mask = channel_mask_for(species_list, CHANNELS_TABLE)

        for drug in drug_list:
            subset = df[df["Species"].isin(species_list) & (df["Drug"] == drug)]
            if subset.empty:
                continue
            subset = filter_training_concentrations(
                subset,
                {
                    HIGH_FIDELITY: config.get("human_concentration_idxs"),
                    animal_species: config.get("animal_concentration_idxs"),
                },
                # The sweep applies the prediction ceiling as well; see
                # gp.data.filter_training_concentrations.
                conc_multiplier=config["conc_multiplier"],
            )
            if subset.empty:
                continue

            oracle = None
            if config.get("use_oracle"):
                # The sweep's oracle is its own, fitted on unfiltered
                # concentrations — see sweep_oracle.
                oracle = sweep_oracle(
                    df[df["Species"].isin(species_list) & (df["Drug"] == drug)],
                    species_list, config, channel_mask,
                    GP_PARAMS / f"{config_name}_sweep"
                    / animal_species.replace(" ", "_") / f"{drug}.pth",
                )
            truth = float(ground_truth.loc[drug, HIGH_FIDELITY])
            human_axis, animal_axis = sweep_grid(subset, drug, species_list)

            records = []
            for num_humans in human_axis:
                for num_animals in animal_axis:
                    records.extend(run_cell(
                        subset, drug, species_list, species_idx, config,
                        num_humans, num_animals, truth, channel_mask, oracle,
                        args.repeats,
                    ))
            if not records:
                print(f"  [skip] {animal_species}/{drug}")
                continue

            per_repeat = pd.DataFrame(records)
            deltas = paired_delta(per_repeat)

            out_dir = ensure(
                EXPERIMENTS_OUT / f"{config_name}_sweep" / animal_species / drug
            )
            per_repeat.to_csv(out_dir / "crps_per_repeat.csv", index=False)
            deltas.to_csv(out_dir / "paired_delta.csv", index=False)

            best = deltas.loc[deltas["median_delta"].idxmax()] if len(deltas) else None
            print(
                f"  {animal_species:11s} {drug:15s} grid "
                f"{len(human_axis)}x{len(animal_axis)}  "
                + (f"best gain {best['median_delta']:+.4f} at "
                   f"nh={int(best['num_humans'])}, na={int(best['num_animals'])}"
                   if best is not None else "")
            )

            if not args.no_plots:
                # The target drug is one the model trains on here, since this
                # config predicts only the drug it was given.
                plot_sweep_heatmaps(
                    per_repeat, deltas, human_axis, animal_axis,
                    heading=lambda quantity, d=drug: (
                        f"{d.capitalize()} (training drug): {quantity}"
                    ),
                    quantity=(f"CRPS of HF+LF model on "
                              f"{HIGH_FIDELITY} predictions"),
                    save_path=out_dir / "heatmaps.pdf",
                )


if __name__ == "__main__":
    main()
