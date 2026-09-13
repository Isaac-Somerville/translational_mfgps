"""Single-drug experiment: does animal data sharpen the human prediction?

For one target drug and one animal species, two models are fitted per repeat:

* **HF only** — trained on human subjects alone.
* **HF + LF** — trained on the same human subjects *plus* animal subjects.

Both are scored by CRPS against the ground-truth maximum relative APD change
(``data/simulations/ground_truth_relative_{1,3}x.csv``). The comparison is
paired: :func:`gp.data.sample_subjects` seeds on the repeat index and species
index, so repeat *r* draws the *same* human subjects in both arms. The
difference between the arms within a repeat is therefore attributable to the
animal data rather than to which humans happened to be drawn.

Every animal species is run against every drug, so a single invocation covers
the full grid.

Outputs, under ``outputs/experiments/<config>/``:
    ``crps_per_repeat.csv``   every repeat's score, both arms — the file the
                              verification compares against
    ``summary.csv``           median and 2.5/97.5 percentiles across repeats
    ``posteriors/<species>/<drug>.pdf``   the three-panel posterior figure

Examples:
    python -m experiments.run_single_drug --config single_drug_1x
    python -m experiments.run_single_drug --config single_drug_3x \\
        --drug quinidine --species Dog
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from drug import load_drug_table
from gp.data import (OBSERVATION_COLUMN, TRUTH_COLUMN, add_block_columns,
                     build_tensors, channel_mask_for, drop_zero_block_rows,
                     filter_training_concentrations,
                     load_apd_dataset, sample_subjects,
                     smoothed_population_mean)
from gp.fitting import fit_and_predict
from gp.metrics import quantile_interval, compute_mae, compute_sharpness
from gp.plotting import plot_posterior_panel
from paths import (APD_DATASET, CHANNELS_TABLE, CONFIG_DIR, EXPERIMENTS_OUT,
                   GP_PARAMS, ensure, ground_truth_table)

#: Animal species available as low-fidelity donors.
ANIMAL_SPECIES = ["Dog", "Guinea Pig", "Mouse", "Pig", "Rabbit"]

#: The species being predicted.
HIGH_FIDELITY = "Human"

#: Index each animal takes when it is the sole low-fidelity species: 0, with
#: the high-fidelity species at 1. Distinct from gp.data.SPECIES_IDX.
ANIMAL_SPECIES_INDEX = {species: 0 for species in ANIMAL_SPECIES}


def load_config(name):
    """Load a config by slug, with or without the ``.json`` suffix."""
    path = CONFIG_DIR / (name if name.endswith(".json") else f"{name}.json")
    with open(path) as f:
        return json.load(f)


def oracle_path_for(config_name, animal_species, drug):
    """Path to the oracle checkpoint for one (species, drug), or None."""
    path = (GP_PARAMS / config_name /
            f"{animal_species.replace(' ', '_')}__{drug}.pth")
    return path if path.exists() else None


def run_one(df, drug, animal_species, config, ground_truth, channel_mask,
            config_name=None, collect_plot_data=False):
    """Both arms, all repeats, for one (drug, animal species) pair.

    Returns:
        ``(records, plot_data)``. ``records`` has one row per repeat per arm;
        ``plot_data`` holds the first repeat's predictions for the panel figure,
        or None.
    """
    # Species order is load-bearing: the high-fidelity species must be last.
    species_list = [animal_species, HIGH_FIDELITY]
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    subset = df[df["Species"].isin(species_list) & (df["Drug"] == drug)]
    if subset.empty:
        return [], None

    # Training sees only the lowest few concentrations while the prediction
    # spans the full range. conc_multiplier is not passed: the single-drug
    # experiments filter by index alone. See
    # gp.data.filter_training_concentrations.
    subset = filter_training_concentrations(
        subset,
        {
            HIGH_FIDELITY: config.get("human_concentration_idxs"),
            animal_species: config.get("animal_concentration_idxs"),
        },
    )
    if subset.empty:
        return [], None

    oracle = (
        oracle_path_for(config_name, animal_species, drug)
        if config.get("use_oracle") and config_name
        else None
    )
    if config.get("use_oracle") and oracle is None:
        print(f"  [warn] {animal_species}/{drug}: use_oracle set but no "
              f"checkpoint found; fitting instead")

    num_humans = config["num_humans"]
    num_animals = config["num_animals"]
    records = []
    plot_data = None

    for repeat in range(config["num_repeats"]):
        arms = {}
        for arm, (nh, na) in {
            "hf_only": (num_humans, 0),
            "hf_lf": (num_humans, num_animals),
        }.items():
            arm_species = [HIGH_FIDELITY] if na == 0 else species_list
            sampled = sample_subjects(
                subset, [drug], arm_species, nh, na, repeat,
                config["subject_selection_seed"],
                human_conc_idxs=config.get("human_concentration_idxs"),
                animal_conc_idxs=config.get("animal_concentration_idxs"),
            )
            if sampled is not None:
                sampled = sampled.dropna(subset=[OBSERVATION_COLUMN])
            if sampled is None or sampled.empty:
                # Too few usable subjects abandons the whole combination, not
                # just the arm that needs it: reporting the HF-only arm alone
                # would leave an unpaired row in the output.
                if na > 0:
                    return [], None
                continue

            # The HF-only arm keeps the full species indexing, so the human
            # keeps the same index.
            X, y = build_tensors(
                add_block_columns(sampled), species_idx,
                conc_as_input=config.get("conc_as_input", False),
                conc_multiplier=config["conc_multiplier"],
            )
            X, y = drop_zero_block_rows(X, y, config.get("boundary_constraint"))
            # One stream per arm, in this order, so the joint arm's human
            # samples follow the animal's. See gp.fitting.predict_all_species.
            order = (
                [species_idx[HIGH_FIDELITY]] if na == 0
                else [species_idx[s] for s in species_list]
            )
            result, extras = fit_and_predict(
                X, y, config, len(species_list), channel_mask, drug,
                species_idx[HIGH_FIDELITY],
                ground_truth_max=ground_truth,
                standardise=config.get("standardise_data", True),
                oracle_path=oracle,
                species_order=order,
            )
            arms[arm] = (result, sampled, extras[-1])

            records.append({
                "Drug": drug, "Species": animal_species, "Repeat": repeat,
                "Arm": arm, "CRPS": result["crps"],
                "MAE": compute_mae(result["sampled_maxima"], ground_truth),
                "Sharpness": compute_sharpness(result["sampled_maxima"]),
                "Predicted_Max": result["predicted_max"],
                "Ground_Truth": ground_truth,
            })

        if collect_plot_data and plot_data is None and len(arms) == 2:
            plot_data = arms

    return records, plot_data


def panel_ground_truth(df, drug, predictions, species_by_panel):
    """Each panel's true population mean, on that panel's concentration grid.

    The smoothing sees only the plotted range, so the curve is the truth over
    the interval the posterior is judged on.

    Args:
        df: The full APD dataset.
        drug: Drug being plotted.
        predictions: Panel key to prediction, as passed to the panel figure.
        species_by_panel: Panel key to species name.

    Returns:
        Panel key to curve, or to None where the panel has no data.
    """
    curves = {}
    for key, species in species_by_panel.items():
        prediction = predictions.get(key)
        if prediction is None:
            curves[key] = None
            continue
        concentrations = prediction["concentrations"]
        rows = df[
            (df["Drug"] == drug)
            & (df["Species"] == species)
            & (df["Drug_Concentration"] <= float(np.max(concentrations)))
        ].dropna(subset=[TRUTH_COLUMN])
        curves[key] = (
            None if rows.empty
            else smoothed_population_mean(rows, concentrations)
        )
    return curves


def summarise(records):
    """Median and central 95% interval per drug, species and arm, plus the
    paired difference between arms. Quantiles across repeats, matching how the
    published tables were computed."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for (drug, species, arm), group in df.groupby(["Drug", "Species", "Arm"]):
        for metric in ("CRPS", "MAE", "Sharpness"):
            median, lower, upper = quantile_interval(group[metric].to_numpy())
            rows.append({
                "Drug": drug, "Species": species, "Arm": arm, "Metric": metric,
                "Median": median, "Lower": lower, "Upper": upper,
            })

    paired_rows = []
    wide = df.pivot_table(
        index=["Drug", "Species", "Repeat"], columns="Arm", values="CRPS"
    ).dropna()
    for (drug, species), group in wide.groupby(["Drug", "Species"]):
        # Paired within repeat: both arms saw the same human subjects.
        difference = (group["hf_only"] - group["hf_lf"]).to_numpy()
        median, lower, upper = quantile_interval(difference)
        paired_rows.append({
            "Drug": drug, "Species": species, "Metric": "CRPS",
            "Median_Difference": median, "Lower": lower, "Upper": upper,
            "Fraction_Improved": float(np.mean(difference > 0)),
        })

    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="Config slug, e.g. single_drug_1x.")
    parser.add_argument("--drug", nargs="+", default=None,
                        help="Target drugs. Default: all twelve.")
    parser.add_argument("--species", nargs="+", default=ANIMAL_SPECIES,
                        choices=ANIMAL_SPECIES,
                        help="Low-fidelity species. Default: all five.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Override the config's repeat count.")
    parser.add_argument(
        "--dataset", default=None,
        help="APD dataset to read. Defaults to the shipped one. Use this rather "
             "than swapping files in data/, so a run cannot leave the wrong "
             "dataset in place if it is interrupted.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.repeats is not None:
        config["num_repeats"] = args.repeats

    drug_list = args.drug or load_drug_table().index.tolist()
    df = load_apd_dataset(args.dataset or APD_DATASET)
    ground_truth = pd.read_csv(
        ground_truth_table(config["conc_multiplier"]), index_col=0
    )

    out_dir = ensure(EXPERIMENTS_OUT / args.config.replace(".json", ""))
    all_records = []

    for animal_species in args.species:
        channel_mask = channel_mask_for(
            [animal_species, HIGH_FIDELITY], CHANNELS_TABLE
        )
        for drug in drug_list:
            truth = float(ground_truth.loc[drug, HIGH_FIDELITY])
            records, plot_data = run_one(
                df, drug, animal_species, config, truth, channel_mask,
                config_name=args.config.replace(".json", ""),
                collect_plot_data=not args.no_plots,
            )
            if not records:
                print(f"  [skip] {animal_species}/{drug}: no usable subjects")
                continue
            all_records.extend(records)

            crps = [r["CRPS"] for r in records if r["Arm"] == "hf_lf"]
            print(
                f"  {animal_species:11s} {drug:15s} "
                f"median CRPS (HF+LF) = {np.nanmedian(crps):.5f}"
            )

            if plot_data is not None:
                panel_dir = ensure(out_dir / "posteriors" / animal_species)
                # The joint fit predicts the animal too, filling panel three.
                animal_idx = ANIMAL_SPECIES_INDEX[animal_species]
                joint_predictions = plot_data["hf_lf"][2]
                animal_rows = plot_data["hf_lf"][1]
                predictions = {
                    "hf_alone": plot_data["hf_only"][0],
                    "hf_joint": plot_data["hf_lf"][0],
                    "lf_joint": joint_predictions.get(animal_idx),
                }
                plot_posterior_panel(
                    predictions=predictions,
                    drug=drug,
                    high_fidelity=HIGH_FIDELITY,
                    low_fidelity=animal_species,
                    observations={
                        "hf_alone": plot_data["hf_only"][1][
                            plot_data["hf_only"][1]["Species"] == HIGH_FIDELITY
                        ],
                        "hf_joint": animal_rows[
                            animal_rows["Species"] == HIGH_FIDELITY
                        ],
                        "lf_joint": animal_rows[
                            animal_rows["Species"] == animal_species
                        ],
                    },
                    observation_column=OBSERVATION_COLUMN,
                    ground_truth_curves=panel_ground_truth(
                        df, drug, predictions,
                        {"hf_alone": HIGH_FIDELITY, "hf_joint": HIGH_FIDELITY,
                         "lf_joint": animal_species},
                    ),
                    save_path=panel_dir / f"{drug}.pdf",
                )

    if not all_records:
        print("No results produced.")
        return

    per_repeat = pd.DataFrame(all_records)
    per_repeat.to_csv(out_dir / "crps_per_repeat.csv", index=False)

    summary, paired = summarise(all_records)
    summary.to_csv(out_dir / "summary.csv", index=False)
    paired.to_csv(out_dir / "paired_difference.csv", index=False)

    print(f"\nWrote {len(per_repeat)} per-repeat rows to {out_dir}")


if __name__ == "__main__":
    main()
