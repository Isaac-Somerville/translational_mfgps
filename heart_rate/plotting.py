"""Figures for the heart-rate experiments.

Both figures answer the same question the drug-block work asks, on measured
data: does adding another species' subjects improve prediction of the species
of interest?

The k-sweep figure sets no style of its own — the published version was drawn
against matplotlib's defaults, and picking up the poster styling the rest of
this release uses would change every font size in it.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gp.plotting import (HIGH_FIDELITY_COLOUR, LOW_FIDELITY_COLOUR,
                         panel_divider)
from heart_rate.data import ATHLETE_GROUP, QOI_COLUMNS

#: Central interval reported across repeats, as quantiles.
INTERVAL = (0.025, 0.975)

#: Matplotlib's own defaults, minus the backend, which cannot be switched from
#: inside an rc_context.
DEFAULT_RC = {
    key: value for key, value in matplotlib.rcParamsDefault.items()
    if key != "backend"
}

QOI_LABELS = {
    "autonomic_block": "Intrinsic heart rate under autonomic block (bpm)",
    "sa_node": "Sino-atrial node rate (bpm)",
}

#: Y-axis label on the posterior panels. Shorter than QOI_LABELS, which has to
#: name the quantity unambiguously in a legend; here the title already has.
QOI_AXIS_LABELS = {
    "autonomic_block": "Heart Rate Post-\n Autonomic Block (bpm)",
    "sa_node": "Sino-atrial\n Node Rate (bpm)",
}

#: Larger type for the posterior row, which is printed very wide.
POSTERIOR_RC = {
    "axes.linewidth": 1.8,
    "axes.labelsize": 28,
    "axes.titlesize": 32,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "legend.frameon": False,
}


def plot_k_sweep(scores, high_fidelity, low_fidelity, qoi, save_path,
                 num_subjects=None):
    """The paired CRPS improvement against high-fidelity training-set size.

    Every point is the median across repeats of that repeat's mean
    improvement over the subjects it held out; the band is the 2.5-97.5%
    interval across repeats. The improvement is paired within a repeat — both
    arms train on the same high-fidelity subjects and score the same held-out
    ones — so the difference isolates the low-fidelity contribution rather than
    the luck of the draw.

    Plotted against the number of subjects *trained on*, ``num_subjects - k``:
    the sweep's ``k`` counts the subjects held out.

    Args:
        num_subjects: Total high-fidelity subjects. Inferred from the largest
            ``k`` when omitted, since the sweep runs ``k = 1 .. n - 1``.
    """
    k_values = sorted(scores["k"].dropna().unique())
    if num_subjects is None:
        num_subjects = int(max(k_values)) + 1

    # Paired within (repeat, subject), then averaged within a repeat: a repeat
    # is one draw of the training set, and that is the unit the interval is
    # over. Quantiles of the pooled per-subject differences would instead
    # describe how much subjects differ from one another.
    wide = scores.pivot_table(
        index=["k", "repeat", "subject_idx"], columns="arm", values="crps"
    ).dropna(subset=["hf_only", "hf_lf"])
    wide["delta"] = wide["hf_only"] - wide["hf_lf"]
    by_repeat = wide.groupby(["k", "repeat"])["delta"].mean()

    train_sizes, medians, lowers, uppers = [], [], [], []
    for k in k_values:
        values = by_repeat.xs(k, level="k").to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        train_sizes.append(num_subjects - k)
        medians.append(np.quantile(values, 0.5))
        lowers.append(np.quantile(values, INTERVAL[0]))
        uppers.append(np.quantile(values, INTERVAL[1]))

    order = np.argsort(train_sizes)
    train_sizes = np.asarray(train_sizes)[order]
    medians = np.asarray(medians)[order]
    lowers = np.asarray(lowers)[order]
    uppers = np.asarray(uppers)[order]

    # Pinned to matplotlib's defaults: the published figure was drawn without
    # any theme, and another figure drawn earlier in the same process would
    # otherwise leave its poster styling and font in rcParams.
    with plt.rc_context(DEFAULT_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        ax.plot(train_sizes, medians, color=HIGH_FIDELITY_COLOUR, lw=2)
        ax.fill_between(train_sizes, lowers, uppers, alpha=0.25,
                        color=HIGH_FIDELITY_COLOUR)
        ax.set_xlabel("Number of HF Training Points", fontsize=15)
        ax.set_ylabel(r"CRPS$_{HF\ only}$ - CRPS$_{HF+LF}$", fontsize=15)
        ax.tick_params(labelsize=13)
        sns.despine(ax=ax)
        fig.suptitle("Improvement in CRPS from Including LF Data", fontsize=18)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return save_path



def _draw_athlete_panel(ax, posterior, colour, qoi, training_frame=None,
                        test_frame=None):
    """One species' posterior across the baseline range, with the subjects on it."""
    qoi_column = QOI_COLUMNS[qoi]

    ax.fill_between(posterior["hr_grid"], posterior["y_lo"], posterior["y_hi"],
                    alpha=0.18, color=colour)
    ax.fill_between(posterior["hr_grid"], posterior["f_lo"], posterior["f_hi"],
                    alpha=0.38, color=colour)
    ax.plot(posterior["hr_grid"], posterior["mean"], color=colour, lw=3)

    if training_frame is not None and len(training_frame) > 0:
        # Sedentary and trained subjects are drawn apart because the split is
        # the experiment: the model sees one group and is asked about the other.
        for group, marker, size in (
            (training_frame[training_frame["Fitness"] != ATHLETE_GROUP], "o", 120),
            (training_frame[training_frame["Fitness"] == ATHLETE_GROUP], "^", 140),
        ):
            if len(group) > 0:
                ax.scatter(group["Baseline"], group[qoi_column], color=colour,
                           marker=marker, s=size, edgecolors="black",
                           linewidths=0.6, zorder=3)

    if test_frame is not None and len(test_frame) > 0:
        ax.scatter(test_frame["Baseline"], test_frame[qoi_column],
                   color="black", marker="*", s=200, zorder=4)

    ax.set_xlabel("Baseline Heart Rate (bpm)", labelpad=8)
    ax.set_ylabel(QOI_AXIS_LABELS.get(qoi, QOI_LABELS.get(qoi, qoi)), labelpad=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    sns.despine(ax=ax)


def _athlete_posterior_page(pdf, posteriors, training_frames, test_frame,
                            high_fidelity, low_fidelity, qoi):
    """Page 1: the high-fidelity fit with and without the animal, and the animal."""
    with plt.rc_context(POSTERIOR_RC):
        fig = plt.figure(figsize=(42, 9))
        grid = GridSpec(2, 3, figure=fig, height_ratios=[8, 0.7], hspace=0.55)
        ax_hf_alone = fig.add_subplot(grid[0, 0])
        ax_hf_joint = fig.add_subplot(grid[0, 1], sharey=ax_hf_alone)
        ax_lf_joint = fig.add_subplot(grid[0, 2])
        legend_ax = fig.add_subplot(grid[1, :])
        legend_ax.axis("off")

        panels = [
            (ax_hf_alone, posteriors.get("hf_only", {}).get(high_fidelity),
             HIGH_FIDELITY_COLOUR, high_fidelity, True,
             "HF Posterior (HF Data Only)"),
            (ax_hf_joint, posteriors.get("hf_lf", {}).get(high_fidelity),
             HIGH_FIDELITY_COLOUR, high_fidelity, True,
             "HF Posterior (HF + LF Data)"),
            (ax_lf_joint, posteriors.get("hf_lf", {}).get(low_fidelity),
             LOW_FIDELITY_COLOUR, low_fidelity, False,
             "LF Posterior (HF + LF Data)"),
        ]

        for ax, posterior, colour, species, is_hf, title in panels:
            if posterior is None:
                ax.axis("off")
                continue
            _draw_athlete_panel(
                ax, posterior, colour, qoi,
                training_frame=training_frames.get(species),
                test_frame=test_frame if is_hf else None,
            )
            ax.set_title(title, pad=14)

        ax_hf_joint.set_ylabel("")
        ax_lf_joint.set_ylabel("")

        legend_ax.legend(
            handles=[
                Patch(alpha=0.18, facecolor="grey", edgecolor="none",
                      label="95% Credible Interval for $y$"),
                Patch(alpha=0.38, facecolor="grey", edgecolor="none",
                      label="95% Credible Interval for $f$"),
                Line2D([0], [0], color="grey", lw=3, label="Mean Prediction"),
                Line2D([0], [0], marker="o", color="none", markersize=10,
                       markerfacecolor="grey", markeredgecolor="black",
                       markeredgewidth=0.6, label="Training (Sedentary)"),
                Line2D([0], [0], marker="^", color="none", markersize=10,
                       markerfacecolor="grey", markeredgecolor="black",
                       markeredgewidth=0.6, label="Training (Athletes)"),
                Line2D([0], [0], marker="*", color="black", markersize=14,
                       linestyle="none", label="Test Athletes"),
            ],
            loc="center", ncol=6, frameon=True, handlelength=2.5,
            markerscale=1.5, fontsize=22,
        )

        panel_divider(fig, ax_hf_alone, ax_hf_joint)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_athlete_leave_out(posteriors, training_frames, test_frame,
                           high_fidelity, low_fidelity, qoi, save_path=None):
    """The leave-out-athletes figure: the three fits.

    The high-fidelity species fitted on its sedentary subjects alone, the same
    species in the joint fit, and the animal from that joint fit — with the
    held-out athletes marked, so it is visible how far outside the training
    range they sit.

    What the animal data was worth per athlete is not shown here:
    :mod:`heart_rate.plot_athlete_crps_improvement` covers it across both
    endpoints at once.
    """
    with PdfPages(save_path) as pdf:
        _athlete_posterior_page(pdf, posteriors, training_frames, test_frame,
                                high_fidelity, low_fidelity, qoi)
    return save_path
