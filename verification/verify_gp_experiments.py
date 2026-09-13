"""Check that this release reproduces the published experiment results.

This checks the whole pipeline end to end: run the released experiments and
compare their scores against the published ones.

What is compared
----------------
Every repeat is deterministic — the dataset, model, subject sampling, training
and posterior draws are all seeded — so agreement should be essentially exact.
The only legitimate difference is floating-point summation order, worth about
1e-15 relative.

``elementwise``
    Used wherever the reference stores per-repeat scores. Compares repeat *r*
    against repeat *r* directly, under :data:`EXACT_TOLERANCE`.

``quantile``
    The fallback where the reference stored only summaries. Compares the median
    and the 2.5/97.5 percentiles under a relative tolerance.

Example:
    python -m verification.verify_gp_experiments --config single_drug_1x
    python -m verification.verify_gp_experiments --all --tolerance 0.05
"""

import argparse

import numpy as np
import pandas as pd

from paths import (EXPERIMENTS_OUT, HEART_RATE_OUT, REFERENCE_RESULTS,
                   VERIFICATION_OUT, ensure)

#: Relative tolerance for the elementwise check. The only expected difference
#: is float summation order in restructured loops.
EXACT_TOLERANCE = 1e-9

#: Relative tolerance for the quantile fallback.
TOLERANCE = 0.05

#: Config slug to the reference file that holds its published quantiles.
SINGLE_DRUG_CONFIGS = {
    "single_drug_1x": "single_drug_1x.csv",
    "single_drug_3x": "single_drug_3x.csv",
}

#: Single-drug configs whose reference stores every repeat rather than only
#: summary quantiles. Where present this supersedes the quantile fallback.
SINGLE_DRUG_PER_REPEAT = {
    "single_drug_1x": "single_drug_1x_per_repeat.csv",
    "single_drug_3x": "single_drug_3x_per_repeat.csv",
}

#: Sweep configs whose reference stores every repeat, so the elementwise check
#: applies. Only the 1x sweep was published.
SWEEP_CONFIGS = {"single_drug_1x": "single_drug_1x_sweep_per_repeat.csv"}

ITERATIVE_CONFIGS = {
    "iterative_prior_bep_dil_qui_ver": "iterative_prior_bep_dil_qui_ver.csv",
    "iterative_prior_bep_dil_qui_ver_oracle": "iterative_prior_bep_dil_qui_ver_oracle.csv",
    "iterative_prior_qui_ver": "iterative_prior_qui_ver.csv",
    "iterative_prior_dil_qui": "iterative_prior_dil_qui.csv",
    "iterative_prior_bep_dil_dof_ver_oracle": "iterative_prior_bep_dil_dof_ver_oracle.csv",
}

#: Heart-rate configs whose sweep reference stores every repeat and subject.
HR_SWEEP_CONFIGS = [
    "hr_dog_from_mouse_sa_node",
    "hr_human_from_dog_autonomic",
]

#: Heart-rate leave-out-athletes configs, whose reference stores every held-out
#: subject rather than a distribution over repeats — there is only one split.
HR_ATHLETE_CONFIGS = [
    "hr_dog_from_mouse_sa_node_athletes",
    "hr_human_from_dog_autonomic_athletes",
]


def quantiles(values):
    """Median and central 95% of a repeat distribution, ignoring non-finite."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.quantile(values, 0.5)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def compare_single_drug(config, reference_path, tolerance):
    """Compare a single-drug run's quantiles against the published ones."""
    produced_path = EXPERIMENTS_OUT / config / "crps_per_repeat.csv"
    if not produced_path.exists():
        return None, f"not run: {produced_path} missing"
    if not reference_path.exists():
        return None, f"no reference: {reference_path} missing"

    produced = pd.read_csv(produced_path, float_precision="round_trip")
    reference = pd.read_csv(reference_path, float_precision="round_trip")

    rows = []
    for _, ref in reference.iterrows():
        subset = produced[
            (produced["Drug"] == ref["Drug"])
            & (produced["Species"] == ref["Species"])
            & (produced["Arm"] == ref["Arm"])
        ]["CRPS"]
        if subset.empty:
            continue
        median, lower, upper = quantiles(subset)
        for name, got, expected in [
            ("median", median, ref["Median"]),
            ("lower", lower, ref["Lower"]),
            ("upper", upper, ref["Upper"]),
        ]:
            rows.append({
                "Drug": ref["Drug"], "Species": ref["Species"], "Arm": ref["Arm"],
                "Quantile": name, "Produced": got, "Reference": expected,
                "Relative_Error": abs(got - expected) / max(abs(expected), 1e-12),
            })

    return pd.DataFrame(rows), None


def compare_iterative(config, reference_path, tolerance):
    """Compare an iterative run's quantiles, per drug, species and scenario."""
    reference = None
    produced_frames = []
    for species_dir in sorted((EXPERIMENTS_OUT / config).glob("*/")):
        path = species_dir / "metrics_per_repeat.csv"
        if path.exists():
            produced_frames.append(pd.read_csv(path, float_precision="round_trip"))
    if not produced_frames:
        return None, f"not run: no metrics_per_repeat.csv under {EXPERIMENTS_OUT / config}"
    if not reference_path.exists():
        return None, f"no reference: {reference_path} missing"

    produced = pd.concat(produced_frames, ignore_index=True)
    reference = pd.read_csv(reference_path, float_precision="round_trip")

    rows = []
    for _, ref in reference.iterrows():
        subset = produced[
            (produced["Drug"] == ref["Drug"])
            & (produced["Species"] == ref["Species"])
            & (produced["Scenario"] == ref["Scenario"])
        ][ref["Metric"]]
        if subset.empty:
            continue
        median, lower, upper = quantiles(subset)
        for name, got, expected in [
            ("median", median, ref["Median"]),
            ("lower", lower, ref["Lower"]),
            ("upper", upper, ref["Upper"]),
        ]:
            rows.append({
                "Drug": ref["Drug"], "Species": ref["Species"],
                "Scenario": ref["Scenario"], "Metric": ref["Metric"],
                "Quantile": name, "Produced": got, "Reference": expected,
                "Relative_Error": abs(got - expected) / max(abs(expected), 1e-12),
            })

    return pd.DataFrame(rows), None


#: Metrics an iterative per-repeat reference carries, in the order reported.
ITERATIVE_METRICS = ["CRPS", "MAE", "Sharpness"]

#: Arm naming differs between the heart-rate reference and this release.
HR_ARM_NAMES = {"lf_hf": "hf_lf", "hf_only": "hf_only"}

#: Tolerance for the leave-out-athletes comparison, which — alone among the
#: checks here — cannot be held to the floating-point level.
#:
#: These numbers were produced with the RNG seeded *after* the model was
#: constructed, and ``IndexKernel`` draws its coregionalisation factor at
#: construction, so they depend on an unseeded initialisation and are not
#: reproducible bit-for-bit. Measured: varying only that initial state moves the
#: fitted predictions by ~8e-4 relative and CRPS by ~2e-3, while this release
#: agrees with the published values to ~1e-5 and ~3e-5 — inside that band.
#:
#: Set an order of magnitude above that intrinsic spread, still far below
#: anything a logic difference would produce.
HR_ATHLETE_TOLERANCE = 1e-2


def compare_hr_sweep(config, reference_path, tolerance=EXACT_TOLERANCE):
    """Compare a heart-rate sweep against its per-repeat, per-subject reference.

    The reference scores every held-out subject separately, so the join is on
    grid point, repeat *and* subject — against ``crps_per_subject.csv`` rather
    than the per-repeat means beside it.
    """
    produced_path = HEART_RATE_OUT / f"{config}_sweep" / "crps_per_subject.csv"
    if not produced_path.exists():
        return None, f"not run: {produced_path} missing"

    produced = pd.read_csv(produced_path, float_precision="round_trip")
    reference = pd.read_csv(reference_path, float_precision="round_trip").rename(
        columns={"num_hf": "num_humans", "num_lf": "num_animals"}
    )
    keys = ["num_humans", "num_animals", "subject_idx"]
    results, unmatched = compare_elementwise(
        produced[keys + ["repeat", "crps"]],
        reference[keys + ["repeat", "crps"]],
        keys, value="crps", tolerance=tolerance,
    )
    if unmatched:
        print(f"       [note] {unmatched} produced rows had no reference counterpart")
    return results, None


def compare_hr_athletes(config, reference_path, tolerance=HR_ATHLETE_TOLERANCE):
    """Compare a leave-out-athletes run, one held-out subject at a time.

    A single split rather than a distribution over repeats, so every reference
    row is one subject under one arm.

    Held to :data:`HR_ATHLETE_TOLERANCE` rather than the floating-point level;
    see there.
    """
    produced_path = HEART_RATE_OUT / config / "scores.csv"
    if not produced_path.exists():
        return None, f"not run: {produced_path} missing"

    produced = pd.read_csv(produced_path, float_precision="round_trip")
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    reference["arm"] = reference["variant"].map(HR_ARM_NAMES)

    # Neither side has a repeat dimension here; give both a constant one so the
    # shared elementwise join applies unchanged.
    produced = produced.assign(repeat=0)
    reference = reference.assign(repeat=0)

    keys = ["arm", "subject_idx"]
    results, unmatched = compare_elementwise(
        produced[keys + ["repeat", "crps"]],
        reference[keys + ["repeat", "crps"]],
        keys, value="crps", tolerance=tolerance,
    )
    if unmatched:
        print(f"       [note] {unmatched} produced rows had no reference counterpart")
    return results, None


def compare_iterative_elementwise(config, reference_path,
                                  tolerance=EXACT_TOLERANCE):
    """Compare an iterative run against its per-repeat reference, one for one.

    Checks all three metrics rather than CRPS alone; they come from the same
    posterior samples, so a disagreement in one localises where.
    """
    produced_frames = []
    for species_dir in sorted((EXPERIMENTS_OUT / config).glob("*/")):
        path = species_dir / "metrics_per_repeat.csv"
        if path.exists():
            produced_frames.append(pd.read_csv(path, float_precision="round_trip"))
    if not produced_frames:
        return None, f"not run: no metrics_per_repeat.csv under {EXPERIMENTS_OUT / config}"

    produced = pd.concat(produced_frames, ignore_index=True).rename(
        columns={"Repeat": "repeat"}
    )
    reference = pd.read_csv(reference_path, float_precision="round_trip")

    keys = ["Drug", "Species", "Scenario"]
    frames, unmatched_total = [], 0
    for metric in ITERATIVE_METRICS:
        merged, unmatched = compare_elementwise(
            produced[keys + ["repeat", metric]],
            reference[keys + ["repeat", metric]],
            keys, value=metric, tolerance=tolerance,
        )
        merged["Metric"] = metric
        frames.append(merged)
        unmatched_total = max(unmatched_total, unmatched)

    if unmatched_total:
        print(f"       [note] {unmatched_total} produced rows had no reference counterpart")
    return pd.concat(frames, ignore_index=True), None


def compare_single_drug_elementwise(config, reference_path, tolerance=EXACT_TOLERANCE):
    """Compare a single-drug run against its per-repeat reference, one for one.

    Supersedes :func:`compare_single_drug` wherever the reference exists.
    """
    produced_path = EXPERIMENTS_OUT / config / "crps_per_repeat.csv"
    if not produced_path.exists():
        return None, f"not run: {produced_path} missing"
    if not reference_path.exists():
        return None, f"no per-repeat reference: {reference_path} missing"

    produced = pd.read_csv(produced_path, float_precision="round_trip").rename(
        columns={"Repeat": "repeat", "CRPS": "crps"}
    )
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    results, unmatched = compare_elementwise(
        produced, reference, ["Drug", "Species", "Arm"], tolerance=tolerance
    )
    # compare_elementwise reports how many produced rows found no counterpart.
    # That is a coverage note, not a reason to skip: rows that did match are
    # still worth checking, and an unmatched count is surfaced by the caller.
    if unmatched:
        print(f"       [note] {unmatched} produced rows had no reference counterpart")
    return results, None


def compare_sweep(config, reference_path, species, tolerance=EXACT_TOLERANCE):
    """Elementwise comparison of a sweep against its published per-repeat scores.

    Every repeat is deterministic, so repeat *r* at grid point (nh, na) should
    reproduce the published value exactly.
    """
    produced_root = EXPERIMENTS_OUT / f"{config}_sweep" / species
    if not produced_root.is_dir():
        return None, f"not run: {produced_root} missing"
    if not reference_path.exists():
        return None, f"no reference: {reference_path} missing"

    reference = pd.read_csv(reference_path, float_precision="round_trip")
    reference = reference[reference["Species"] == species]

    frames = []
    for drug_dir in sorted(produced_root.iterdir()):
        produced_path = drug_dir / "crps_per_repeat.csv"
        drug_reference = reference[reference["Drug"] == drug_dir.name]
        if not produced_path.exists() or drug_reference.empty:
            continue
        merged, _ = compare_elementwise(
            pd.read_csv(produced_path, float_precision="round_trip"),
            drug_reference,
            ["num_humans", "num_animals"], tolerance=tolerance,
        )
        merged["Drug"] = drug_dir.name
        merged["Species"] = species
        frames.append(merged)

    if not frames:
        return None, "no matching drugs"
    return pd.concat(frames, ignore_index=True), None


def compare_elementwise(produced, reference, keys, value="crps",
                        tolerance=EXACT_TOLERANCE):
    """Compare per-repeat scores one for one.

    Joins on ``keys`` plus the repeat index, so repeat *r* is compared against
    repeat *r* rather than against the reference distribution as a whole.

    Returns:
        ``(DataFrame of per-row relative errors, number of unmatched rows)``.
    """
    join_keys = list(keys) + ["repeat"]
    merged = produced.merge(
        reference, on=join_keys, how="inner", suffixes=("_produced", "_reference")
    )
    unmatched = len(produced) - len(merged)

    a = merged[f"{value}_produced"].to_numpy(dtype=float)
    b = merged[f"{value}_reference"].to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)

    merged["Relative_Error"] = np.nan
    merged.loc[finite, "Relative_Error"] = (
        np.abs(a[finite] - b[finite]) / np.maximum(np.abs(b[finite]), 1e-12)
    )
    merged["Exact"] = merged["Relative_Error"] <= tolerance
    return merged, unmatched


def compare_hr_k_sweep(config, reference_path, tolerance=TOLERANCE):
    """Compare a heart-rate k-sweep against its published quantiles.

    The reference stores summaries rather than every repeat, so this is the
    quantile check. The aggregation must match how the published summaries
    were formed: within each k,
    **each repeat's mean CRPS across held-out subjects** is one observation, and
    the quantiles are taken over repeats. Pooling subjects and repeats together
    gives the same median but a far narrower interval.

    The paired difference is formed per (repeat, subject) and then averaged
    within a repeat.
    """
    produced_path = HEART_RATE_OUT / config / "scores.csv"
    if not produced_path.exists():
        return None, f"not run: {produced_path} missing"
    if not reference_path.exists():
        return None, f"no reference: {reference_path} missing"

    produced = pd.read_csv(produced_path, float_precision="round_trip")
    reference = pd.read_csv(reference_path, float_precision="round_trip")

    # "with animals" is the joint arm; "hf only model" is the separately fitted
    # high-fidelity-only arm.
    arm_prefix = {"hf_lf": "crps", "hf_only": "hfm_crps"}

    rows = []
    for _, ref in reference.iterrows():
        k_rows = produced[produced["k"] == ref["k"]]
        if k_rows.empty:
            continue

        per_arm = {}
        for arm, prefix in arm_prefix.items():
            arm_rows = k_rows[k_rows["arm"] == arm]
            if arm_rows.empty:
                continue
            per_repeat = arm_rows.groupby("repeat")["crps"].mean().dropna()
            per_arm[arm] = per_repeat
            median, lower, upper = quantiles(per_repeat.to_numpy())
            for name, got in [("median", median), ("p025", lower), ("p975", upper)]:
                column = f"{prefix}_{name}"
                if column not in ref:
                    continue
                rows.append({
                    "k": ref["k"], "Quantity": column,
                    "Produced": got, "Reference": ref[column],
                    "Relative_Error": abs(got - ref[column]) / max(abs(ref[column]), 1e-12),
                })

        if len(per_arm) == 2 and "diff_crps_median" in ref:
            joined = k_rows.pivot_table(
                index=["repeat", "subject_idx"], columns="arm", values="crps"
            ).dropna()
            per_repeat = (joined["hf_only"] - joined["hf_lf"]).groupby("repeat").mean()
            median, lower, upper = quantiles(per_repeat.to_numpy())
            for name, got in [("median", median), ("p025", lower), ("p975", upper)]:
                column = f"diff_crps_{name}"
                rows.append({
                    "k": ref["k"], "Quantity": column,
                    "Produced": got, "Reference": ref[column],
                    "Relative_Error": abs(got - ref[column]) / max(abs(ref[column]), 1e-12),
                })

    return pd.DataFrame(rows), None


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic: the largest ECDF gap."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def plot_comparison(results, config, save_path):
    """Relative error per compared quantity, worst first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from gp.plotting import apply_style

    apply_style()
    ordered = results.sort_values("Relative_Error", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(14, max(7, 0.32 * len(ordered) + 3)))

    labels = [
        " / ".join(str(row[c]) for c in
                   ("Drug", "Species", "Arm", "Scenario", "Metric", "Quantile")
                   if c in ordered.columns)
        for _, row in ordered.iterrows()
    ]
    ax.barh(range(len(ordered)), 100 * ordered["Relative_Error"],
            color="#3E75E2", edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Relative difference from the published value (%)", labelpad=10)
    ax.set_title(f"{config}: largest discrepancies", pad=14)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def report(config, results, tolerance, out_dir):
    """Print a verdict and write the per-quantity detail."""
    if results is None or results.empty:
        print(f"  {config}: nothing to compare")
        return True

    results = results.dropna(subset=["Relative_Error"])
    failures = results[results["Relative_Error"] > tolerance]
    worst = results["Relative_Error"].max()

    results.to_csv(out_dir / f"{config}_comparison.csv", index=False)
    plot_comparison(results, config, out_dir / f"{config}_comparison.pdf")

    if failures.empty:
        print(f"  PASS {config}: {len(results)} quantities, "
              f"worst {100 * worst:.2f}% (tolerance {100 * tolerance:.0f}%)")
        return True

    print(f"  FAIL {config}: {len(failures)} of {len(results)} quantities over "
          f"tolerance, worst {100 * worst:.2f}%")
    for _, row in failures.nlargest(5, "Relative_Error").iterrows():
        keys = " / ".join(
            str(row[c]) for c in
            ("Drug", "Species", "Arm", "Scenario", "Metric", "Quantile")
            if c in row.index
        )
        print(f"       {keys}: {row['Produced']:.6g} vs {row['Reference']:.6g} "
              f"({100 * row['Relative_Error']:.1f}%)")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", nargs="+", default=None,
                        help="Config slugs to check. Default: every one with a "
                             "reference file.")
    parser.add_argument("--all", action="store_true",
                        help="Check every config, including those not yet run.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="Relative tolerance per quantile.")
    parser.add_argument("--sweep", action="store_true",
                        help="Check the sweeps elementwise against their "
                             "per-repeat references. The strongest check.")
    parser.add_argument("--sweep-species", default="Dog",
                        help="Species whose sweep to check.")
    parser.add_argument("--reference-dir", default=None,
                        help="Directory of published reference quantiles. "
                             "Defaults to data/reference_results/.")
    args = parser.parse_args()

    reference_dir = (
        REFERENCE_RESULTS if args.reference_dir is None
        else __import__("pathlib").Path(args.reference_dir)
    )
    out_dir = ensure(VERIFICATION_OUT)

    selected = args.config or (
        list(SINGLE_DRUG_CONFIGS) + list(ITERATIVE_CONFIGS)
        + HR_SWEEP_CONFIGS + HR_ATHLETE_CONFIGS
    )
    passed = True

    if args.sweep:
        for config in selected:
            if config not in SWEEP_CONFIGS:
                continue
            results, skip = compare_sweep(
                config, reference_dir / SWEEP_CONFIGS[config], args.sweep_species,
            )
            if skip:
                print(f"  SKIP {config} sweep: {skip}")
                continue
            exact = int(results["Exact"].sum())
            worst = results["Relative_Error"].max()
            ok = exact == len(results)
            passed &= ok
            print(f"  {'PASS' if ok else 'FAIL'} {config} sweep (elementwise): "
                  f"{exact}/{len(results)} exact, worst "
                  f"{100 * worst:.3g}% relative")
            results.to_csv(out_dir / f"{config}_sweep_elementwise.csv", index=False)
        print()

    print(f"Reference quantiles from {reference_dir}\n")
    for config in selected:
        if config in SINGLE_DRUG_CONFIGS:
            # Prefer the elementwise check; fall back to quantiles only if the
            # per-repeat reference was never written.
            per_repeat_ref = reference_dir / SINGLE_DRUG_PER_REPEAT.get(config, "")
            if per_repeat_ref.exists():
                results, skip = compare_single_drug_elementwise(config, per_repeat_ref)
                if not skip:
                    exact = int(results["Exact"].sum())
                    worst = results["Relative_Error"].max()
                    ok = exact == len(results)
                    passed &= ok
                    print(f"  {'PASS' if ok else 'FAIL'} {config} (elementwise): "
                          f"{exact}/{len(results)} exact, worst "
                          f"{100 * worst:.3g}% relative")
                    results.to_csv(
                        out_dir / f"{config}_elementwise.csv", index=False)
                    continue
            results, skip = compare_single_drug(
                config, reference_dir / SINGLE_DRUG_CONFIGS[config], args.tolerance
            )
        elif config in HR_SWEEP_CONFIGS:
            # These slugs carry two published results: the training-set-size
            # grid written by run_heart_rate_sweep, and the k-sweep written by
            # run_heart_rate. Both are checked.
            results, skip = compare_hr_sweep(
                config,
                reference_dir / f"{config}_sweep_per_repeat.csv",
            )
            if not skip:
                exact = int(results["Exact"].sum())
                worst = results["Relative_Error"].max()
                ok = exact == len(results)
                passed &= ok
                print(f"  {'PASS' if ok else 'FAIL'} {config} sweep (elementwise): "
                      f"{exact}/{len(results)} exact, worst "
                      f"{100 * worst:.3g}% relative")
                results.to_csv(
                    out_dir / f"{config}_sweep_elementwise.csv", index=False)

            k_results, k_skip = compare_hr_k_sweep(
                config,
                reference_dir / f"{config}_k_sweep.csv",
                args.tolerance,
            )
            if k_skip:
                print(f"  SKIP {config} k-sweep: {k_skip}")
            else:
                worst = k_results["Relative_Error"].max()
                ok = bool(worst <= args.tolerance)
                passed &= ok
                print(f"  {'PASS' if ok else 'FAIL'} {config} k-sweep (quantile): "
                      f"{len(k_results)} quantities, worst {100 * worst:.3g}% "
                      f"(tolerance {100 * args.tolerance:.3g}%)")
                k_results.to_csv(
                    out_dir / f"{config}_k_sweep_quantiles.csv", index=False)
            continue
        elif config in HR_ATHLETE_CONFIGS:
            results, skip = compare_hr_athletes(
                config,
                reference_dir / f"{config}_per_subject.csv",
            )
            if not skip:
                exact = int(results["Exact"].sum())
                worst = results["Relative_Error"].max()
                ok = exact == len(results)
                passed &= ok
                print(f"  {'PASS' if ok else 'FAIL'} {config} (elementwise): "
                      f"{exact}/{len(results)} exact, worst "
                      f"{100 * worst:.3g}% relative")
                results.to_csv(
                    out_dir / f"{config}_elementwise.csv", index=False)
                continue
        elif config in ITERATIVE_CONFIGS:
            # Same preference as above: elementwise where the per-repeat
            # reference exists, quantiles only as a fallback.
            per_repeat_ref = reference_dir / f"{config}_per_repeat.csv"
            if per_repeat_ref.exists():
                results, skip = compare_iterative_elementwise(config, per_repeat_ref)
                if not skip:
                    exact = int(results["Exact"].sum())
                    worst = results["Relative_Error"].max()
                    ok = exact == len(results)
                    passed &= ok
                    print(f"  {'PASS' if ok else 'FAIL'} {config} (elementwise): "
                          f"{exact}/{len(results)} exact, worst "
                          f"{100 * worst:.3g}% relative")
                    results.to_csv(
                        out_dir / f"{config}_elementwise.csv", index=False)
                    continue
            results, skip = compare_iterative(
                config, reference_dir / ITERATIVE_CONFIGS[config], args.tolerance
            )
        else:
            print(f"  {config}: no reference mapping")
            continue

        if skip:
            print(f"  SKIP {config}: {skip}")
            continue
        passed &= report(config, results, args.tolerance, out_dir)

    print(f"\nWrote comparisons to {out_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
