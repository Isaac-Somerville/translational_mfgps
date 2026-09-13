"""Per-athlete CRPS improvement from adding low-fidelity data.

Reads the leave-out-athletes scores and plots, for each held-out trained
subject, the improvement from including the low-fidelity species:

    CRPS(high fidelity only) - CRPS(high fidelity + low fidelity)

Positive means the low-fidelity data helped for that subject.

One row per experiment, labelled by that experiment's high-fidelity species, so
the two endpoints can be compared against the same zero line.

Example:
    python -m heart_rate.plot_athlete_crps_improvement
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from gp.plotting import HIGH_FIDELITY_COLOUR
from heart_rate.run_heart_rate import load_config
from paths import HEART_RATE_OUT, ensure

#: The two leave-out-athletes experiments the figure compares, top row first.
DEFAULT_CONFIGS = [
    "hr_dog_from_mouse_sa_node_athletes",
    "hr_human_from_dog_autonomic_athletes",
]

#: Vertical spread within a row, so near-identical improvements do not overlap.
#: Seeded, so the figure is reproducible.
JITTER = 0.18
JITTER_SEED = 0


def load_improvements(config_name):
    """Per-subject paired improvement for one leave-out-athletes run.

    Returns:
        ``(high_fidelity, deltas)``. One delta per held-out subject; the split
        is deterministic, so there is no repeat dimension to average over.
    """
    scores_path = HEART_RATE_OUT / config_name / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run heart_rate.run_heart_rate "
            f"--config {config_name} first."
        )
    config = load_config(config_name)
    scores = pd.read_csv(scores_path, float_precision="round_trip")

    # Paired per subject: both arms score the same held-out athletes from the
    # same sedentary training set.
    wide = scores.pivot_table(
        index="subject_idx", columns="arm", values="crps"
    ).dropna(subset=["hf_only", "hf_lf"])
    per_subject = (wide["hf_only"] - wide["hf_lf"]).sort_index()
    return config["high_fidelity_species"], per_subject.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="Leave-out-athletes config slugs, one row each.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    panels = [load_improvements(name.replace(".json", "")) for name in args.configs]
    species_names = [high_fidelity for high_fidelity, _ in panels]
    y_positions = list(range(len(panels)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(0, color="grey", ls="--", lw=1, zorder=1)

    rng = np.random.default_rng(JITTER_SEED)
    for row, (_, deltas) in enumerate(panels):
        y = row + rng.uniform(-JITTER, JITTER, size=len(deltas))
        ax.scatter(
            deltas,
            y,
            marker="D",
            color=HIGH_FIDELITY_COLOUR,
            s=55,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label="Test Athletes" if row == 0 else None,
        )

    ax.set_xlabel(r"CRPS$_{HF\ only}$ - CRPS$_{HF+LF}$", fontsize=15)
    ax.set_ylabel("HF Species", fontsize=15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(species_names, fontsize=12)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_ylim(len(panels) - 0.5, -0.5)  # row 0 on top, padded frame
    ax.set_title("Improvement in CRPS from Inclusion of LF Data", fontsize=18)
    ax.legend(fontsize=13)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = args.out or (
        ensure(HEART_RATE_OUT) / "athlete_crps_improvement.pdf"
    )
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
