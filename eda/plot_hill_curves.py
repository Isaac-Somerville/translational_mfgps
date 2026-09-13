"""Channel inhibition against concentration, one figure per drug.

The GP's input is the vector of per-channel block fractions a drug produces at a
given concentration, so these curves are the input space.

The two vertical markers are the therapeutic concentration and three times it —
the range the experiments predict over.

Example:
    python -m eda.plot_hill_curves
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from drug import ALL_CHANNELS, Drug, load_drug_table
from paths import EDA_OUT, ensure

#: One colour per channel, stable across drugs.
CHANNEL_COLOURS = dict(
    zip(ALL_CHANNELS, sns.color_palette("tab10", n_colors=len(ALL_CHANNELS)))
)

#: hERG is conventionally written I_Kr in the electrophysiology literature.
CHANNEL_LABELS = {"hERG": "IKr"}

#: Concentration grid, in multiples of therapeutic.
MAX_THERAPEUTIC_MULTIPLE = 4
C_START = 1e-8
NUM_POINTS = 1000


def plot_hill_curves(drug_name, save_path):
    """Percentage block against concentration for one drug."""
    drug = Drug(drug_name)
    c_end = MAX_THERAPEUTIC_MULTIPLE * drug.c_therapeutic
    concentrations = np.linspace(C_START, c_end, NUM_POINTS)

    for channel in drug.channel_list:
        label = CHANNEL_LABELS.get(channel, channel)
        block = 100 * drug.calculate_inhibition(concentrations, channel)
        plt.plot(concentrations, block, label=label,
                 color=CHANNEL_COLOURS[channel])

    plt.axhline(50, color="black", linestyle="--", label="50% Inhibition")
    plt.axvline(drug.c_therapeutic, color="black", linestyle=":",
                label="Therapeutic Concentration")
    plt.axvline(3 * drug.c_therapeutic, color="red", linestyle=":",
                label="3x Therapeutic Concentration")
    plt.xlabel("Drug Concentration (nM)")
    plt.ylabel("Percentage Inhibited")
    plt.title(f"Inhibition Curves for {drug.name}, Risk: {drug.cipa_risk}")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drug", nargs="+", default=None)
    args = parser.parse_args()

    drugs = args.drug or load_drug_table().index.tolist()
    out = ensure(EDA_OUT / "hill_curves")
    for drug_name in drugs:
        path = plot_hill_curves(
            drug_name, out / f"{drug_name}_inhibition_curves.pdf"
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
