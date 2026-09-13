"""How the heart-rate transfer benefit varies with both sample sizes.

The same sweep as the drug-block experiment, on the measured heart-rate data:
for each combination of high- and low-fidelity training subjects, refit and
score the held-out high-fidelity subjects.

Two figures per config, matching the drug-block sweep:

1. **CRPS of the joint model on held-out high-fidelity subjects.**
2. **CRPS(high-fidelity only) − CRPS(joint), paired by repeat.** Positive means
   the low-fidelity data helped.

The difference is paired against the ``num_lf = 0`` column of the same row:
:func:`heart_rate.data.sample_split` seeds the high-fidelity draw on the repeat
alone, so both cells train on identical high-fidelity subjects.

Outputs, under ``outputs/heart_rate/<config>_sweep/``:
    ``crps_per_subject.csv`` every grid point, repeat and held-out subject
    ``crps_per_repeat.csv``  the same, averaged over subjects within a repeat
    ``paired_delta.csv``     the paired improvement per grid point
    ``heatmaps.pdf``         the two figures above

Example:
    python -m heart_rate.run_heart_rate_sweep --config hr_human_from_dog_autonomic
"""

import argparse

import numpy as np
import pandas as pd

from gp.plotting import plot_sweep_heatmaps
from heart_rate.data import load_heart_rate, sample_split
from heart_rate.run_heart_rate import (config_for_repeat, load_config,
                                       run_arm)
from paths import HEART_RATE_OUT, HEART_RATE_TABLE, ensure

#: Repeats per grid point.
NUM_REPEATS = 10


def sweep_grid(high_fidelity_df, low_fidelity_df):
    """Grid axes: 1..n-1 high-fidelity subjects, 0..m low-fidelity.

    The high-fidelity axis stops one short of the total because at least one
    subject has to remain to be predicted.
    """
    return (
        list(range(1, len(high_fidelity_df))),
        list(range(0, len(low_fidelity_df) + 1)),
    )


def paired_delta(per_subject):
    """Paired improvement over the low-fidelity-free column, per grid point.

    Paired on ``(repeat, subject)``: both arms score the same held-out subjects
    from the same high-fidelity training draw. The per-subject differences are
    then averaged **within a repeat**, so a cell's summary is over repeats.
    
    """
    if 0 not in set(per_subject["num_animals"]):
        return pd.DataFrame()

    rows = []
    for num_hf, block in per_subject.groupby("num_humans"):
        baseline = block[block["num_animals"] == 0].set_index(
            ["repeat", "subject_idx"])["crps"]
        for num_lf, cell in block.groupby("num_animals"):
            values = cell.set_index(["repeat", "subject_idx"])["crps"]
            shared = baseline.index.intersection(values.index)
            if not len(shared):
                continue
            paired = (baseline[shared] - values[shared]).dropna()
            diffs = paired.groupby(level="repeat").mean().to_numpy()
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size == 0:
                continue
            rows.append({
                "num_humans": num_hf, "num_animals": num_lf,
                "n_paired": diffs.size,
                "median_delta": float(np.median(diffs)),
                "p05": float(np.percentile(diffs, 5)),
                "p95": float(np.percentile(diffs, 95)),
                "frac_improved": float((diffs > 0).mean()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = args.config.replace(".json", "")

    hf_species = config["high_fidelity_species"]
    lf_species = config["low_fidelity_species"][0]
    qoi = config["qoi"]
    species_list = [lf_species, hf_species]          # high fidelity last
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    data = load_heart_rate(HEART_RATE_TABLE, species_list, qoi)
    hf_df, lf_df = data[hf_species], data[lf_species]
    hf_axis, lf_axis = sweep_grid(hf_df, lf_df)
    print(f"{qoi}: grid {len(hf_axis)} x {len(lf_axis)} "
          f"({hf_species} n={len(hf_df)}, {lf_species} n={len(lf_df)})")

    records = []
    for num_hf in hf_axis:
        for num_lf in lf_axis:
            for repeat in range(args.repeats):
                frames, test_df = sample_split(
                    hf_df, lf_df, num_hf, num_lf, repeat,
                    config["subject_selection_seed"],
                )
                # With no low-fidelity subjects the model is single-task, not
                # two-task with one task empty: the number of tasks sets the
                # coregionalisation factor's shape.
                cell_species = species_list if num_lf > 0 else [hf_species]
                cell_idx = {sp: i for i, sp in enumerate(cell_species)}
                scored, _ = run_arm(
                    frames, test_df, cell_idx, hf_species,
                    config_for_repeat(config, repeat), qoi,
                )
                for record in scored:
                    records.append({
                        # Column names match the drug-block sweep so the same
                        # plotting code serves both.
                        "num_humans": num_hf, "num_animals": num_lf,
                        "repeat": repeat, "subject_idx": record["subject_idx"],
                        "crps": record["crps"],
                    })
        median = np.nanmedian([
            r["crps"] for r in records
            if r["num_humans"] == num_hf and r["num_animals"] == lf_axis[-1]
        ])
        print(f"  {hf_species} n={num_hf:2d}  median CRPS (all {lf_species}) = {median:.3f}")

    if not records:
        print("No results produced.")
        return

    per_subject = pd.DataFrame(records)
    deltas = paired_delta(per_subject)
    # One row per cell and repeat, a repeat's score being the mean over the
    # subjects it held out. This is what the heatmaps summarise, and it matches
    # the drug-block sweep's schema so the same plotting code serves both.
    per_repeat = (
        per_subject.groupby(["num_humans", "num_animals", "repeat"])["crps"]
        .mean().reset_index()
    )

    out_dir = ensure(HEART_RATE_OUT / f"{config_name}_sweep")
    per_subject.to_csv(out_dir / "crps_per_subject.csv", index=False)
    per_repeat.to_csv(out_dir / "crps_per_repeat.csv", index=False)
    deltas.to_csv(out_dir / "paired_delta.csv", index=False)

    if not args.no_plots:
        plot_sweep_heatmaps(
            per_repeat, deltas, hf_axis, lf_axis,
            heading=lambda quantity: (
                f"HF = {hf_species}, LF = {lf_species}\n{quantity}"
            ),
            quantity="CRPS of HF+LF model on held-out HF subjects",
            save_path=out_dir / "heatmaps.pdf",
        )

    print(f"\nWrote {len(per_subject)} rows to {out_dir}")


if __name__ == "__main__":
    main()
