"""Plot the paired CRPS improvement from adding animal data.

Reads the per-repeat scores written by :mod:`experiments.run_single_drug` and
plots, per drug, the paired difference

    CRPS(high fidelity only) - CRPS(high fidelity + low fidelity)

so that positive means the animal data helped.

The difference is taken **within** a repeat. Both arms of a repeat train on the
same human subjects — :func:`gp.data.sample_subjects` seeds on the repeat index,
not on the arm — so the between-repeat variation in which humans were drawn
cancels.

One figure per low-fidelity species, plus a combined summary.

Example:
    python -m experiments.compare_single_drug --config single_drug_1x
"""

import argparse

import numpy as np
import pandas as pd

from experiments.run_single_drug import load_config
from gp.metrics import quantile_interval
from gp.plotting import plot_paired_crps_difference
from paths import EXPERIMENTS_OUT, ensure


def paired_differences(per_repeat, metric="CRPS"):
    """Per (species, drug) paired difference between the two arms.

    Returns:
        ``{species: {drug: (median, lower, upper)}}``.
    """
    wide = per_repeat.pivot_table(
        index=["Species", "Drug", "Repeat"], columns="Arm", values=metric
    ).dropna(subset=["hf_only", "hf_lf"])

    out = {}
    for (species, drug), group in wide.groupby(["Species", "Drug"]):
        difference = (group["hf_only"] - group["hf_lf"]).to_numpy()
        out.setdefault(species, {})[drug] = quantile_interval(difference)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Config slug.")
    parser.add_argument("--metric", default="CRPS",
                        choices=["CRPS", "MAE", "Sharpness"])
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = EXPERIMENTS_OUT / args.config.replace(".json", "")
    scores_path = run_dir / "crps_per_repeat.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run experiments.run_single_drug "
            f"--config {args.config} first."
        )

    per_repeat = pd.read_csv(scores_path, float_precision="round_trip")
    by_species = paired_differences(per_repeat, metric=args.metric)

    out_dir = ensure(run_dir / "comparisons")
    for species, differences in by_species.items():
        improved = sum(1 for m, lo, _ in differences.values() if lo > 0)
        path = out_dir / f"paired_{args.metric.lower()}_{species.replace(' ', '_')}.pdf"
        plot_paired_crps_difference(
            differences, save_path=path, metric_label=args.metric,
            conc_multiplier=config["conc_multiplier"],
        )
        print(
            f"  {species:11s} {improved}/{len(differences)} drugs improved with "
            f"the interval clear of zero  ->  {path.name}"
        )

    # One row per (species, drug), so the figures' numbers are available as
    # data.
    rows = [
        {"Species": species, "Drug": drug, "Metric": args.metric,
         "Median_Difference": median, "Lower": lower, "Upper": upper}
        for species, differences in by_species.items()
        for drug, (median, lower, upper) in differences.items()
    ]
    table = pd.DataFrame(rows).sort_values(["Species", "Drug"])
    table.to_csv(out_dir / f"paired_{args.metric.lower()}.csv", index=False)
    print(f"\nWrote {len(table)} rows to {out_dir}")


if __name__ == "__main__":
    main()
