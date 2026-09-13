"""How similar are two drugs, as the model sees them?

As concentration rises from zero, a drug traces a path through the space of
per-channel block fractions, starting at the origin. Two summaries of that path:

**Angle between endpoint vectors.** The direction of a drug's block vector is
*which* channels it hits, in what proportion, with overall potency divided out.
Block fractions are non-negative, so angles lie in [0, 90] and 90 means disjoint
channel sets.

**Endpoint magnitude.** How hard the drug blocks overall, direction discarded.

Both are computed at a concentration ceiling expressed as a multiple of each
drug's own therapeutic concentration, so drugs are compared at matched clinical
exposure rather than matched absolute concentration.

Example:
    python -m eda.plot_trajectory_geometry --conc-multiplier 3
"""

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from drug import ALL_CHANNELS, Drug, load_drug_table
from gp.style import apply_publication_font
from paths import EDA_OUT, ensure

#: The concentration ceiling is a multiple of each drug's therapeutic
#: concentration; this names it in the subtitles.
CONC_ENDPOINT = "c_therapeutic"

#: Heading, subtitle, axis-label, tick and colorbar-tick sizes. The subtitle is
#: drawn as its own text, so it needs its own size.
TITLE_FS, SUB_FS, LABEL_FS, TICK_FS, CBAR_FS = 28, 19, 26, 22, 20


def apply_trajectory_style():
    """Poster styling for the two trajectory-geometry figures."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 26,
        "axes.titlesize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
        "legend.frameon": False,
    })


def _conc_note(conc_multiplier):
    """``c_therapeutic, 3.0x`` for the subtitle."""
    return f"{CONC_ENDPOINT}, {float(conc_multiplier)}x"


def _style_axes(ax, title, subtitle, xlabel, ylabel, rotate_x=False,
                fig_width=None):
    """Heading, labels and ticks, for any axes in this module."""
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    plt.setp(ax.get_yticklabels(), rotation=0)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Wrap the subtitle to the figure width and pad the title for its lines.
    width = fig_width if fig_width is not None else ax.figure.get_size_inches()[0]
    sub_lines = textwrap.wrap(subtitle, width=int(width * 5.2)) if subtitle else []
    ax.set_title(title, fontsize=TITLE_FS,
                 pad=SUB_FS * 1.5 * max(len(sub_lines), 1) + 12)
    if sub_lines:
        ax.text(0.5, 1.015, "\n".join(sub_lines), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=SUB_FS, color="0.30")


def _style_heatmap(ax, title, subtitle, cbar_label, xlabel, ylabel,
                   rotate_x=False, fig_width=None):
    """:func:`_style_axes` plus the colorbar label and tick sizing."""
    _style_axes(ax, title, subtitle, xlabel, ylabel,
                rotate_x=rotate_x, fig_width=fig_width)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=CBAR_FS)
        cbar.set_label(cbar_label, fontsize=LABEL_FS)


def _annot_size(fig_width, fig_height, n_cols, n_rows):
    """Annotation font sized by cell geometry."""
    cell_size = min(fig_width / max(n_cols, 1), fig_height / max(n_rows, 1))
    return int(np.clip(cell_size * 9.0, 9, 22))


def endpoint_vectors(drugs, conc_multiplier):
    """Per-drug block vector at ``conc_multiplier`` times therapeutic.

    Returns:
        ``(drug_names, array of shape (n_drugs, n_channels))``.
    """
    names, vectors = [], []
    for drug_name in drugs:
        drug = Drug(drug_name, channel_list=ALL_CHANNELS)
        ceiling = conc_multiplier * drug.c_therapeutic
        vectors.append(
            drug.calculate_all_inhibitions(np.array([ceiling])).flatten()
        )
        names.append(drug_name)
    return names, np.array(vectors)


def angle_matrix(vectors):
    """Pairwise angle in degrees between endpoint vectors.

    0 means the same channels in the same proportions; 90 means no channel
    blocked appreciably by both.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def plot_angle_heatmap(names, angles, conc_multiplier, save_path):
    """Pairwise angles as a heatmap, drugs in alphabetical order."""
    apply_trajectory_style()
    num_drugs = len(names)
    side = max(14, num_drugs * 1.0 + 4)
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        pd.DataFrame(angles, index=names, columns=names),
        ax=ax,
        annot=True,
        fmt=".1f",
        # RdBu_r: low end blue, so 0 deg (identical direction) blue, 90 deg red.
        cmap="RdBu_r",
        center=45,
        vmin=0,
        vmax=90,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        square=True,
        annot_kws={"size": _annot_size(side, side, num_drugs, num_drugs)},
    )
    _style_heatmap(
        ax,
        "Angle Between Drug Endpoint Inhibition Vectors",
        f"{_conc_note(conc_multiplier)} — 0° = identical channel-blocking direction",
        "Angle (degrees)", "Drug", "Drug", rotate_x=True, fig_width=side,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_endpoint_magnitudes(names, vectors, conc_multiplier, save_path):
    """Overall block strength per drug, direction discarded."""
    apply_trajectory_style()
    # Sorted explicitly, so the bars follow the heatmap's drug order.
    order = np.argsort(np.asarray(names))
    drug_names = [names[i] for i in order]
    magnitudes = np.linalg.norm(vectors, axis=1)[order]

    w = max(14, len(drug_names) * 1.1 + 4)
    fig, ax = plt.subplots(figsize=(w, 10))
    bars = ax.bar(drug_names, magnitudes, color="#3e75e2",
                  edgecolor="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=TICK_FS)
    ax.set_ylim(0, magnitudes.max() * 1.12)  # room for the tallest bar's label
    # House theme is style="white", which leaves no gridlines.
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    _style_axes(
        ax,
        "Endpoint Inhibition Magnitude by Drug",
        f"{_conc_note(conc_multiplier)} — L2 norm of the channel inhibition vector",
        "Drug", "Endpoint magnitude (L2 norm)", rotate_x=True, fig_width=w,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--conc-multiplier", type=int, default=1)
    args = parser.parse_args()

    drugs = sorted(load_drug_table().index.tolist())
    names, vectors = endpoint_vectors(drugs, args.conc_multiplier)
    out = ensure(EDA_OUT)

    print(f"Wrote {plot_angle_heatmap(names, angle_matrix(vectors), args.conc_multiplier, out / f'drug_angle_heatmap_{args.conc_multiplier}x.pdf')}")
    print(f"Wrote {plot_endpoint_magnitudes(names, vectors, args.conc_multiplier, out / f'endpoint_magnitude_{args.conc_multiplier}x.pdf')}")


if __name__ == "__main__":
    main()
