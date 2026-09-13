"""Concentration-response curves for every drug and species.

The data the models are asked to learn from, before any model touches it. One
panel per drug; within a panel, one LOWESS curve per species through that
species' population mean.

Concentration is in multiples of each drug's own therapeutic concentration, so
the panels are on a common clinical scale despite spanning four orders of
magnitude in absolute terms.

Example:
    python -m eda.plot_ground_truth_curves
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from drug import Drug, load_drug_table
from gp.data import TRUTH_COLUMN, load_apd_dataset, smoothed_population_mean
from gp.style import apply_publication_font
from paths import APD_DATASET, EDA_OUT, ensure

SPECIES_COLOURS = {
    "Human": "#000000",
    "Dog": "#E69F00",
    "Guinea Pig": "#56B4E9",
    "Mouse": "#009E73",
    "Pig": "#CC79A7",
    "Rabbit": "#0072B2",
}

ZERO_COLOUR = "grey"
THRESHOLD_COLOUR = "#4D4D4D"

#: Relative APD change, in percent, above which a drug is treated as showing a
#: proarrhythmic signal.
THRESHOLD_PERCENT = 10.0

#: Points per drawn curve.
CURVE_POINTS = 200


def apply_curves_style():
    """Poster styling for this figure."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.frameon": False,
    })


def ground_truth_curve(species_df, column, n_dense=CURVE_POINTS):
    """The species' smoothed population mean, over its own concentration range.

    Returns:
        ``(x_dense, y_dense)``, or ``(None, None)`` if there is too little data.
    """
    concentrations = species_df["Drug_Concentration"].unique()
    if len(concentrations) < 2:
        return None, None
    x_dense = np.linspace(0.0, float(np.max(concentrations)), n_dense)
    return x_dense, smoothed_population_mean(species_df, x_dense, column=column)


def plot_curves(df, drugs, species_list, save_path, conc_multiplier=3):
    """One panel per drug, one LOWESS curve per species."""
    apply_curves_style()
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 22), sharey=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    axes_flat = axes.flatten()

    for index, drug_name in enumerate(drugs):
        ax = axes_flat[index]
        drug_df = df[df["Drug"] == drug_name]
        c_therapeutic = Drug(drug_name).c_therapeutic

        for species in species_list:
            species_df = drug_df[drug_df["Species"] == species].dropna(
                subset=[TRUTH_COLUMN]
            )
            if species_df.empty:
                continue

            x_dense, y_smooth = ground_truth_curve(species_df, TRUTH_COLUMN)
            if x_dense is None:
                continue

            ax.plot(x_dense / c_therapeutic, 100 * y_smooth,
                    color=SPECIES_COLOURS[species], lw=3.0)

        ax.axhline(0, color=ZERO_COLOUR, lw=1.4, linestyle="--", zorder=0)
        ax.axhline(THRESHOLD_PERCENT, color=THRESHOLD_COLOUR, lw=1.8,
                   linestyle=":", zorder=0)
        ax.set_title(drug_name.capitalize(), pad=10)
        ax.set_xlim(0, conc_multiplier)
        ax.set_xticks(list(range(conc_multiplier + 1)))
        if index % ncols == 0:
            ax.set_ylabel("APD Change (%)", labelpad=8)
        if index >= ncols * (nrows - 1):
            ax.set_xlabel(r"Concentration ($\times\ C_{therapeutic}$)", labelpad=8)
        sns.despine(ax=ax)

    for index in range(len(drugs), nrows * ncols):
        axes_flat[index].axis("off")

    legend_handles = [
        Line2D([0], [0], color=SPECIES_COLOURS[s], lw=3.0) for s in species_list
    ] + [
        Line2D([0], [0], color=ZERO_COLOUR, lw=1.4, linestyle="--"),
        Line2D([0], [0], color=THRESHOLD_COLOUR, lw=1.8, linestyle=":"),
    ]
    legend_labels = list(species_list) + [
        "No Change (0%)", f"Risk Threshold ({THRESHOLD_PERCENT:.0f}%)"
    ]

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=18,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--conc-multiplier", type=int, default=3,
                        help="Concentration ceiling, as a multiple of therapeutic.")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    df = load_apd_dataset(args.dataset or APD_DATASET)
    drugs = load_drug_table().index.tolist()
    species_list = [s for s in SPECIES_COLOURS if s in set(df["Species"])]

    path = plot_curves(
        df, drugs, species_list,
        ensure(EDA_OUT) / f"ground_truth_curves_{args.conc_multiplier}x.pdf",
        conc_multiplier=args.conc_multiplier,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
