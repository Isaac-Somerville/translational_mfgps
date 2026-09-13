"""Figures for the single-drug and iterative experiments.

The **posterior panel** shows one drug's concentration-response curve three
ways: the high-fidelity species fitted on its own data, the same species fitted
jointly with the animal, and the animal's own posterior from that joint fit.

The **paired CRPS difference** plot summarises the same comparison across drugs.
Both arms of a repeat see identical high-fidelity subjects (see
:func:`gp.data.sample_subjects`), so the difference is taken *within* a repeat.

All axes are in percent APD change, matching the exceedance threshold.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator, ScalarFormatter

from gp.metrics import quantile_interval
from gp.style import apply_publication_font

#: Colours are fixed by fidelity role, not by species.
HIGH_FIDELITY_COLOUR = "#5E3C99"
LOW_FIDELITY_COLOUR = "#E66101"
GROUND_TRUTH_COLOUR = "black"
THRESHOLD_COLOUR = "#4D4D4D"

#: Relative APD change, in percent, above which a drug is treated as showing a
#: proarrhythmic signal.
THRESHOLD_PERCENT = 10.0

#: The posterior panel is printed wide, so it spells the quantity out rather
#: than subscripting it.
APD_PANEL_YLABEL = "APD Change (%)"


def apply_style():
    """House plot style, applied once per figure-producing script."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 24,
        "axes.titlesize": 28,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.frameon": False,
    })


def apply_comparison_style():
    """Larger type again for the across-drug comparison figures."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 32,
        "axes.titlesize": 36,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.frameon": False,
    })


def apply_panel_style():
    """Larger type for the posterior panel, which is printed very wide."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 28,
        "axes.titlesize": 32,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.frameon": False,
    })


def _draw_posterior(ax, prediction, colour, drug, observations=None,
                    ground_truth_curve=None, observation_column=None):
    """One posterior panel: credible bands, mean, data, threshold."""
    concentrations = prediction["concentrations"]

    if ground_truth_curve is not None:
        ax.plot(concentrations, 100 * ground_truth_curve, linestyle="--",
                linewidth=3.5, color=GROUND_TRUTH_COLOUR, zorder=3)

    ax.hlines(THRESHOLD_PERCENT, concentrations[0], concentrations[-1],
              colors=THRESHOLD_COLOUR, linestyles=":", linewidth=3.5, zorder=3)

    # Outer band: predictive for an observation. Inner: the latent function.
    ax.fill_between(concentrations, 100 * prediction["lower"],
                    100 * prediction["upper"], color=colour, alpha=0.18, zorder=1)
    ax.fill_between(concentrations, 100 * prediction["lower_latent"],
                    100 * prediction["upper_latent"], color=colour, alpha=0.38, zorder=2)
    ax.plot(concentrations, 100 * prediction["mean"], linewidth=4.0,
            color=colour, zorder=4)

    if observations is not None and len(observations) > 0:
        ax.scatter(observations["Drug_Concentration"].values,
                   100 * observations[observation_column].values,
                   s=120, marker="o", color=colour, edgecolors="black",
                   linewidths=0.6, alpha=0.9, zorder=5)

    ax.set_xlabel(f"{drug.capitalize()} Concentration (nM)", labelpad=10)
    ax.set_ylabel(APD_PANEL_YLABEL, labelpad=10)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")


def panel_divider(fig, left_ax, right_ax):
    """Grey dashed rule between two axes, spanning title to x-label.

    Placed from the rendered tight bounding boxes so it tracks the real extent
    of the titles and labels rather than the axes rectangles.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    left_box = left_ax.get_tightbbox(renderer).transformed(inverse)
    right_box = right_ax.get_tightbbox(renderer).transformed(inverse)
    x = (left_box.x1 + right_box.x0) / 2
    fig.add_artist(Line2D(
        [x, x], [left_box.y0, left_box.y1], transform=fig.transFigure,
        color="grey", linestyle="--", linewidth=1.8, zorder=10,
    ))


def plot_posterior_panel(predictions, drug, high_fidelity, low_fidelity,
                         observations=None, observation_column=None,
                         ground_truth_curves=None, save_path=None):
    """One row of three posteriors, comparing fitted-alone against fitted-jointly.

    Left and middle are the high-fidelity species on its own data and in the
    joint fit, sharing a y axis; a dashed rule separates them from the
    low-fidelity posterior on the right. There is no low-fidelity-alone fit.

    Args:
        predictions: Dict with keys ``hf_alone``, ``hf_joint`` and ``lf_joint``,
            each a result from :func:`gp.fitting.predict`.
        drug: Drug name, for axis labels.
        high_fidelity, low_fidelity: Species names, unused in the titles but
            kept for callers that label their output by species.
        observations: Optional dict with the same keys, each a DataFrame of the
            training points to overlay.
        observation_column: Column of ``observations`` to plot.
        ground_truth_curves: Optional dict with the same keys, each the
            population's true mean response on that panel's concentration grid.
        save_path: Where to write the PDF. Nothing is written if None.
    """
    apply_panel_style()
    fig = plt.figure(figsize=(42, 9))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[8, 0.7], hspace=0.55)

    ax_hf_alone = fig.add_subplot(grid[0, 0])
    ax_hf_joint = fig.add_subplot(grid[0, 1], sharey=ax_hf_alone)
    ax_lf_joint = fig.add_subplot(grid[0, 2])

    panels = [
        ("hf_alone", ax_hf_alone, HIGH_FIDELITY_COLOUR,
         "HF Posterior (HF Data Only)"),
        ("hf_joint", ax_hf_joint, HIGH_FIDELITY_COLOUR,
         "HF Posterior (HF + LF Data)"),
        ("lf_joint", ax_lf_joint, LOW_FIDELITY_COLOUR,
         "LF Posterior (HF + LF Data)"),
    ]

    for key, ax, colour, title in panels:
        if predictions.get(key) is None:
            ax.axis("off")
            continue
        _draw_posterior(
            ax, predictions[key], colour, drug,
            observations=(observations or {}).get(key),
            ground_truth_curve=(ground_truth_curves or {}).get(key),
            observation_column=observation_column,
        )
        sns.despine(ax=ax, top=True, right=True)
        ax.set_title(title, pad=14)

    # Only the leftmost panel is labelled; the middle shares its y axis.
    ax_hf_joint.set_ylabel("")
    ax_lf_joint.set_ylabel("")

    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=GROUND_TRUTH_COLOUR, linestyle="--",
                   linewidth=3.5, label="Ground Truth"),
            Line2D([0], [0], color=THRESHOLD_COLOUR, linestyle=":",
                   linewidth=3.5, label=f"Threshold ({THRESHOLD_PERCENT:.0f}%)"),
            Patch(facecolor="black", alpha=0.55,
                  label="95% Credible Interval for $y$"),
            Patch(facecolor="black", alpha=0.80,
                  label="95% Credible Interval for $f$"),
            Line2D([0], [0], color="black", linewidth=4.0, label="Mean Prediction"),
            Line2D([0], [0], color="#555", marker="o", markersize=8,
                   linestyle="None", label="Observed Data"),
        ],
        loc="center", ncol=6, frameon=True, handlelength=2.5, markerscale=1.5,
        fontsize=22,
    )

    panel_divider(fig, ax_hf_alone, ax_hf_joint)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


#: Gridline spacing on the paired-difference metric axis. Fixed rather than
#: auto-located, so the 1x and 3x runs can be read against each other. Labelled
#: ticks are sparser at 3x.
PAIRWISE_GRID_STEP = 0.1
PAIRWISE_LABEL_STEP = {1: 0.1, 3: 0.2}


def _pairwise_grid(ax, conc_multiplier):
    """Fixed-step gridlines on the metric axis, labelled every step or two."""
    label_step = PAIRWISE_LABEL_STEP.get(conc_multiplier, PAIRWISE_GRID_STEP)
    ax.yaxis.set_major_locator(MultipleLocator(label_step))
    which = "major"
    if label_step != PAIRWISE_GRID_STEP:
        ax.yaxis.set_minor_locator(MultipleLocator(PAIRWISE_GRID_STEP))
        which = "both"
        # Gridlines only — no minor tick marks on the spine.
        ax.tick_params(axis="y", which="minor", length=0)
    # Explicit colour: the white seaborn theme leaves grid.color white.
    ax.grid(True, which=which, axis="y", color="0.7", linestyle="-",
            linewidth=1.0, alpha=0.6)
    ax.set_axisbelow(True)


def _shade_sign(ax):
    """Green above zero, grey below: which side helped, without reading the axis."""
    low, high = ax.get_ylim()
    if high > 0:
        ax.axhspan(0, high, alpha=0.07, color="#2ca02c", zorder=0)
    if low < 0:
        ax.axhspan(low, 0, alpha=0.07, color="grey", zorder=0)


def plot_paired_crps_difference(differences, save_path=None,
                                metric_label="CRPS", conc_multiplier=1):
    """Per-drug paired difference, high-fidelity-only minus joint.

    Positive means the joint model scored better. The interval is the central
    95% of the *paired* differences across repeats, matching the published
    tables.

    Args:
        differences: Drug name to ``(median, lower, upper)``.
        save_path: Where to write the PDF.
        metric_label: Axis label stem.
        conc_multiplier: Sets the gridline labelling step.
    """
    apply_comparison_style()
    drugs = sorted(differences)
    medians = np.array([differences[d][0] for d in drugs])
    lowers = np.array([differences[d][1] for d in drugs])
    uppers = np.array([differences[d][2] for d in drugs])
    x = np.arange(len(drugs))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(
        x, medians, yerr=[medians - lowers, uppers - medians],
        linestyle="", marker=".", color="purple", capsize=5,
        label=f"Paired Difference (95% CI)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in drugs], rotation=45, ha="right")
    ax.set_ylabel(
        f"{metric_label}$_{{HF\\ only}}$ - {metric_label}$_{{HF+LF}}$",
        fontsize=26,
    )
    ax.set_title(f"Improvement in {metric_label} from Including LF Data",
                 fontsize=30)
    ax.legend(loc="best", fontsize=20, frameon=True)
    _pairwise_grid(ax, conc_multiplier)
    sns.despine(ax=ax)
    _shade_sign(ax)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


#: Colours per conditioning scenario, keyed to how much information each has.
SCENARIO_COLOURS = {
    "prior": "#7f7f7f",    # grey: nothing about the target drug
    "animal": LOW_FIDELITY_COLOUR,
    "human": HIGH_FIDELITY_COLOUR,
}

#: The scenarios name the *roles* rather than the species.
SCENARIO_LABELS = {
    "prior": "Training Drug Data Only",
    "animal": "Training Drug + LF Data from Target Drug",
    "human": "Training Drug + LF + HF Data from Target Drug",
}

#: The same three, phrased from the held-out drugs' point of view for the
#: exceedance transition panels.
STAGE_LABELS = {
    "prior": "Test Drugs (using training drug data only)",
    "animal": "Test Drugs (+ LF test drug data)",
    "human": "Test Drugs (+ LF + HF test drug data)",
}

STAGE_MARKERS = {"prior": "o", "animal": "^", "human": "D"}
STAGE_SIZES = {"prior": 10, "animal": 10, "human": 9}

#: Colour for the drugs the model was trained on, on every exceedance panel.
TRAINING_COLOUR = "#3e75e2"

#: Colour of the in-sample training-drug bars.
TRAINING_BAR_COLOUR = "#e25f3e"

#: Quinidine's true APD change is several times any other drug's, which
#: compresses the exceedance panels. Excluded from those panels only.
EXCEEDANCE_OUTLIER = "quinidine"

#: Which stages each exceedance-transition page connects.
TRANSITIONS = [
    ("prior", "animal"),
    ("prior", "human"),
    ("animal", "human"),
    ("prior", "animal", "human"),
]


def apply_iterative_style():
    """Poster styling for the iterative experiment's figures."""
    apply_publication_font()
    sns.set_theme(style="white", context="poster")
    plt.rcParams.update({
        "axes.linewidth": 1.8,
        "axes.labelsize": 32,
        "axes.titlesize": 36,
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.frameon": False,
    })


def _scenarios_present(per_repeat):
    """The scenarios that actually have rows, in increasing-information order."""
    present = set(per_repeat["Scenario"])
    return [s for s in SCENARIO_COLOURS if s in present]


def _metric_bars(ax, per_repeat, metric, ylabel):
    """Grouped bars: one group per drug, one bar per scenario."""
    drugs = sorted(per_repeat["Drug"].unique())
    scenarios = _scenarios_present(per_repeat)
    width = 0.25
    x = np.arange(len(drugs))

    for i, scenario in enumerate(scenarios):
        medians, lower_error, upper_error = [], [], []
        for drug in drugs:
            values = per_repeat[
                (per_repeat["Drug"] == drug) & (per_repeat["Scenario"] == scenario)
            ][metric].to_numpy()
            median, lower, upper = quantile_interval(values)
            medians.append(median)
            lower_error.append(0 if np.isnan(median - lower) else median - lower)
            upper_error.append(0 if np.isnan(upper - median) else upper - median)

        ax.bar(
            x + (i - 1) * width, medians, width,
            label=SCENARIO_LABELS[scenario], color=SCENARIO_COLOURS[scenario],
            alpha=0.82, yerr=[lower_error, upper_error], capsize=3,
            error_kw={"linewidth": 1.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in drugs], rotation=45, ha="right")
    ax.set_xlabel("Drug", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(f"{ylabel} in HF Species with Sequential Data Addition", pad=14)
    ax.legend(frameon=True, loc="best", fontsize=22)
    sns.despine(ax=ax, top=True, right=True)


def _exceedance_table(ax, drugs, scenarios, exceedance):
    """Exceedance probability per drug and scenario, with its interval."""
    rows = []
    for drug in drugs:
        row = [drug.capitalize()]
        for scenario in scenarios:
            value = exceedance.get(drug, {}).get(scenario)
            row.append(
                "—" if value is None or np.isnan(value[0])
                else f"{value[0]:.3f}\n({value[1]:.3f}, {value[2]:.3f})"
            )
        rows.append(row)

    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Drug"] + [SCENARIO_LABELS[s] for s in scenarios],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    ax.set_title(
        "Exceedance Probability per Drug and Scenario (Median with 95% CI)",
        fontsize=16, pad=20,
    )


def _signed_distance(per_repeat, drug, threshold):
    """How far the drug's true response sits above the threshold.

    Signed: the sign says which side of the decision the drug is on, the
    distance how close a call it was.
    """
    rows = per_repeat[per_repeat["Drug"] == drug]
    if rows.empty:
        return np.nan
    return float(rows["Ground_Truth"].iloc[0]) - threshold


def _errorbar_points(points, ax, fmt, colour, markersize, label):
    """Draw ``(x, mean, lower, upper)`` tuples as an error-bar series."""
    if not points:
        return []
    x, mean, lower, upper = zip(*points)
    ax.errorbar(
        list(x), list(mean),
        yerr=[[m - lo for m, lo in zip(mean, lower)],
              [hi - m for m, hi in zip(mean, upper)]],
        fmt=fmt, color=colour, capsize=5, capthick=1.8, markersize=markersize,
        linewidth=1.8, zorder=3, label=label,
    )
    return list(x)


def _training_points(training_results, threshold):
    """Training drugs as ``(x, mean, lower, upper)``, outlier excluded."""
    points = []
    for drug, values in (training_results or {}).items():
        if drug == EXCEEDANCE_OUTLIER or np.isnan(values.get("Exceedance", np.nan)):
            continue
        points.append((
            values["Ground_Truth"] - threshold, values["Exceedance"],
            values["Exceedance_Lower"], values["Exceedance_Upper"],
        ))
    return points


def _zoom_x(ax, all_x):
    """Pad the x range by 8% so no marker sits on the frame."""
    if not all_x:
        return
    span = max(all_x) - min(all_x)
    pad = span * 0.08 if span > 0 else 0.05
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)


def _exceedance_scatter(ax, per_repeat, drugs, scenario, exceedance,
                        training_points, threshold):
    """One scenario's exceedance probabilities against the truth."""
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.5, zorder=1)

    test_points = []
    for drug in drugs:
        if drug == EXCEEDANCE_OUTLIER:
            continue
        value = exceedance.get(drug, {}).get(scenario)
        if value is None or np.isnan(value[0]):
            continue
        test_points.append(
            (_signed_distance(per_repeat, drug, threshold), *value)
        )

    all_x = _errorbar_points(training_points, ax, "o", TRAINING_COLOUR, 10,
                             "Training drugs")
    all_x += _errorbar_points(test_points, ax, "D", HIGH_FIDELITY_COLOUR, 9,
                              "Test drugs")
    _zoom_x(ax, all_x)

    ax.set_xlabel("Signed distance from threshold (ground truth)", labelpad=10)
    ax.set_ylabel("Predicted exceedance probability", labelpad=10)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(SCENARIO_LABELS[scenario], pad=12)
    ax.legend(frameon=True, loc="upper left")
    sns.despine(ax=ax, top=True, right=True)


def _exceedance_transition(ax, per_repeat, drugs, stages, exceedance,
                           training_points, threshold):
    """Several stages at once, joined per drug so the movement is visible."""
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.5, zorder=1)

    all_x = [point[0] for point in training_points]
    labelled = set()
    for drug in drugs:
        if drug == EXCEEDANCE_OUTLIER:
            continue
        values = [exceedance.get(drug, {}).get(stage) for stage in stages]
        if any(v is None or np.isnan(v[0]) for v in values):
            continue

        x = _signed_distance(per_repeat, drug, threshold)
        all_x.append(x)
        # The vertical line spans the drug's whole trajectory.
        ax.plot([x] * len(values), [v[0] for v in values],
                color="black", linestyle="--", linewidth=1.2, zorder=2)
        for stage, (mean, lower, upper) in zip(stages, values):
            ax.errorbar(
                [x], [mean], yerr=[[mean - lower], [upper - mean]],
                fmt=STAGE_MARKERS[stage], color=SCENARIO_COLOURS[stage],
                capsize=5, capthick=1.8, markersize=STAGE_SIZES[stage],
                linewidth=1.8, zorder=3,
                label=STAGE_LABELS[stage] if stage not in labelled else None,
            )
            labelled.add(stage)

    _errorbar_points(training_points, ax, "s", TRAINING_COLOUR, 10,
                     "Training Drugs")
    _zoom_x(ax, all_x)

    ax.set_xlabel("Signed Distance from Threshold\n(Ground Truth)", labelpad=10)
    ax.set_ylabel("Probability", labelpad=10)
    ax.set_ylim([-0.05, 1.05])
    ax.set_title("Predicted Exceedance Probabilities", pad=12)
    ax.legend(frameon=True, loc="lower right", fontsize=16)
    sns.despine(ax=ax, top=True, right=True)


def _training_metric_bars(fig_axes, training_results, prior_drugs):
    """In-sample scores for the drugs the model was trained on."""
    drugs = [d for d in prior_drugs if d in training_results]
    x = np.arange(len(drugs))
    for ax, (key, ylabel) in zip(fig_axes, [
        ("CRPS", "CRPS (max APD change)"),
        ("MAE", "MAE (max APD change)"),
        ("Sharpness",
         "Sharpness ($\\frac{1}{2}\\,\\mathbb{E}|X-X'|$, max APD change)"),
    ]):
        ax.bar(x, [training_results[d][key] for d in drugs],
               color=TRAINING_BAR_COLOUR, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in drugs], rotation=45,
                           ha="right")
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_title(f"Training drugs (in-sample) – {ylabel}", pad=12)
        sns.despine(ax=ax, top=True, right=True)


def plot_iterative_metrics(per_repeat, species, threshold, exceedance=None,
                           training_results=None, prior_drugs=None,
                           save_path=None):
    """Write the iterative experiment's metrics PDF.

    Twelve pages, in three groups: the scores (CRPS, MAE and sharpness per
    held-out drug, one bar per conditioning set); the exceedance probability as
    a table, then per scenario against the truth, then with the scenarios joined
    per drug; and the training drugs scored in-sample.

    Args:
        per_repeat: One row per repeat, drug and scenario.
        species: Low-fidelity species, for the caller's own labelling.
        threshold: Relative APD change defining exceedance.
        exceedance: Drug to scenario to ``(mean, lower, upper)``, estimated
            from the pooled draws. The exceedance pages are skipped without it.
        training_results: Prior drug to its in-sample scores.
        prior_drugs: The prior drugs, in the order to plot them.
        save_path: Where to write the PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    apply_iterative_style()
    drugs = sorted(per_repeat["Drug"].unique())
    scenarios = _scenarios_present(per_repeat)
    exceedance = exceedance or {}
    width = max(20, 1.4 * len(drugs))

    with PdfPages(save_path) as pdf:
        for metric, ylabel in [
            ("CRPS", "CRPS"),
            ("MAE", "MAE"),
            ("Sharpness", "Sharpness ($\\frac{1}{2}\\,\\mathbb{E}|X-X'|$)"),
        ]:
            fig, ax = plt.subplots(figsize=(width, 7))
            _metric_bars(ax, per_repeat, metric, ylabel)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if exceedance:
            fig, ax = plt.subplots(figsize=(
                max(14, 4.5 * (len(scenarios) + 1)), max(4, 0.8 * len(drugs) + 2)
            ))
            _exceedance_table(ax, drugs, scenarios, exceedance)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            training_points = _training_points(training_results, threshold)

            for scenario in scenarios:
                fig, ax = plt.subplots(1, 1, figsize=(9, 9))
                _exceedance_scatter(ax, per_repeat, drugs, scenario, exceedance,
                                    training_points, threshold)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            for stages in TRANSITIONS:
                if not all(stage in scenarios for stage in stages):
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(9, 9))
                _exceedance_transition(ax, per_repeat, drugs, stages, exceedance,
                                       training_points, threshold)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        if training_results:
            order = prior_drugs or sorted(training_results)
            fig, axes = plt.subplots(
                1, 3, figsize=(max(18, 2.5 * len(order)), 7)
            )
            _training_metric_bars(axes, training_results, order)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return save_path


#: Heading, subtitle, axis-label, tick and colorbar-tick sizes on the sweep
#: heatmaps. Set explicitly because the subtitle is drawn as its own text.
HEATMAP_TITLE_FS, HEATMAP_SUB_FS = 28, 19
HEATMAP_LABEL_FS, HEATMAP_TICK_FS, HEATMAP_CBAR_FS = 26, 22, 20

#: The paired improvement, as the colorbar and title spell it.
CRPS_DIFF_LABEL = r"$\mathrm{CRPS_{HF\ only}} - \mathrm{CRPS_{HF+LF}}$"


def _annot_strings(grid):
    """Cell labels in the grid's own units, plus a scale note when one is needed.

    Fixed decimals chosen from the grid's magnitude, so a cell reads the same
    number as the colorbar. Values below 0.01 fall back to a shared exponent.

    Returns:
        ``(label array, colorbar note, exponent)``.
    """
    finite = grid[np.isfinite(grid) & (grid != 0)]
    if len(finite) == 0:
        return np.full(grid.shape, "", dtype=object), "", 0

    largest = float(np.max(np.abs(finite)))
    labels = np.full(grid.shape, "", dtype=object)

    if largest >= 0.01:
        places = 0 if largest >= 100 else (1 if largest >= 10 else 2)
        for index in np.ndindex(grid.shape):
            value = grid[index]
            if np.isfinite(value):
                text = f"{value:.{places}f}"
                # A tiny negative rounds to "-0.00"; the cell colour already
                # carries the sign.
                labels[index] = text.lstrip("-") if float(text) == 0 else text
        return labels, "", 0

    exponent = int(np.floor(np.log10(largest)))
    scale = 10.0 ** exponent
    for index in np.ndindex(grid.shape):
        value = grid[index]
        if np.isfinite(value):
            labels[index] = f"{value / scale:.2f}"
    return labels, f" ($\\times 10^{{{exponent}}}$)", exponent


def _draw_heatmap(grid, hf_values, lf_values, title, subtitle, cbar_label,
                  diverging=False, vmin=None, vmax=None):
    """One sweep heatmap: low fidelity on x, high fidelity increasing upward.

    ``grid`` is indexed ``[high fidelity, low fidelity]``, matching the order of
    ``hf_values`` and ``lf_values``.

    A diverging map centred on zero with RdBu, so positive is blue. Callers
    sign their differences so positive means the extra data helped.
    """
    apply_style()
    height = max(9, len(hf_values) * 0.75 + 4)
    width = max(13, len(lf_values) * 0.85 + 6)
    fig, ax = plt.subplots(figsize=(width, height))

    # Annotations are always shown; they shrink to fit rather than being dropped.
    cell_size = min(width / max(len(lf_values), 1),
                    height / max(len(hf_values), 1))
    annot_fontsize = int(np.clip(cell_size * 7.0, 8, 22))

    labels, scale_note, scale_exponent = _annot_strings(grid)

    kwargs = {}
    if diverging:
        finite = np.abs(grid[np.isfinite(grid)])
        limit = max(float(np.max(finite)) if len(finite) else 1.0, 1e-10)
        kwargs["norm"] = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    else:
        kwargs["vmin"], kwargs["vmax"] = vmin, vmax

    sns.heatmap(
        grid[::-1],
        ax=ax,
        cmap="RdBu" if diverging else "viridis_r",
        annot=labels[::-1],
        fmt="",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=[str(v) for v in lf_values],
        yticklabels=[str(v) for v in reversed(hf_values)],
        annot_kws={"size": annot_fontsize},
        **kwargs,
    )

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        # Ticks carry the same scale as the cell labels.
        if scale_exponent != 0:
            formatter = FuncFormatter(
                lambda v, _p: f"{v / 10.0 ** scale_exponent:.2f}"
            )
        else:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
        # After update_ticks: it re-derives the formatter from the locator and
        # would discard one set earlier.
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.tick_params(labelsize=HEATMAP_CBAR_FS)
        cbar.ax.yaxis.get_offset_text().set_fontsize(HEATMAP_CBAR_FS)
        cbar.set_label(f"{cbar_label}{scale_note}", fontsize=HEATMAP_LABEL_FS)

    ax.tick_params(labelsize=HEATMAP_TICK_FS)
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Number of LF subjects", fontsize=HEATMAP_LABEL_FS)
    ax.set_ylabel("Number of HF subjects", fontsize=HEATMAP_LABEL_FS)

    sub_lines = textwrap.wrap(subtitle, width=int(width * 5.2)) or [subtitle]
    ax.set_title(title, fontsize=HEATMAP_TITLE_FS,
                 pad=HEATMAP_SUB_FS * 1.5 * len(sub_lines) + 12)
    ax.text(0.5, 1.015, "\n".join(sub_lines), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=HEATMAP_SUB_FS, color="0.30")

    fig.tight_layout()
    return fig


def _pivot(frame, value, y_values, x_values, y_col, x_col):
    """Frame column to a (y, x) grid, with missing cells as NaN."""
    table = frame.pivot_table(index=y_col, columns=x_col, values=value)
    return table.reindex(index=y_values, columns=x_values).to_numpy(dtype=float)


def plot_sweep_heatmaps(per_repeat, deltas, hf_axis, lf_axis, heading,
                        quantity, save_path):
    """The two sweep figures: absolute CRPS, and the paired improvement.

    Args:
        per_repeat: One row per grid cell and repeat, with a ``crps`` column.
        deltas: Per-cell paired differences, with a ``median_delta`` column.
        hf_axis, lf_axis: The swept high- and low-fidelity subject counts.
        heading: Callable turning a quantity into the figure's full title. The
            two research sweeps head their figures differently, so the caller
            supplies it.
        quantity: What the first page's title calls the absolute CRPS surface.
        save_path: Where to write the PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    medians = (
        per_repeat.groupby(["num_humans", "num_animals"])["crps"]
        .median().reset_index()
    )
    # Repeats, not rows: the heart-rate sweep holds one row per
    # (repeat, subject).
    num_repeats = int(per_repeat["repeat"].nunique())
    varies = "as number of HF and LF subjects varies"

    with PdfPages(save_path) as pdf:
        pdf.savefig(
            _draw_heatmap(
                _pivot(medians, "crps", hf_axis, lf_axis,
                       "num_humans", "num_animals"),
                hf_axis, lf_axis,
                heading(quantity),
                f"median across {num_repeats} repeats {varies}",
                "CRPS",
            ),
            bbox_inches="tight",
        )
        plt.close("all")

        if len(deltas):
            pdf.savefig(
                _draw_heatmap(
                    _pivot(deltas, "median_delta", hf_axis, lf_axis,
                           "num_humans", "num_animals"),
                    hf_axis, lf_axis,
                    heading(CRPS_DIFF_LABEL),
                    f"median of paired differences across {num_repeats} "
                    f"repeats {varies}",
                    r"$\Delta$CRPS",
                    diverging=True,
                ),
                bbox_inches="tight",
            )
            plt.close("all")

    return save_path
