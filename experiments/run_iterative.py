"""Iterative data addition: leave-one-drug-out over three conditioning sets.

For each held-out target drug, the same model is scored under three
progressively richer conditioning sets:

===========  =====================================================
``prior``    Other drugs only; nothing about the target drug.
``animal``   Prior drugs, plus low-fidelity data for the target drug.
``human``    Prior drugs, plus low-fidelity *and* high-fidelity data
             for the target drug.
===========  =====================================================

The gap between ``prior`` and ``animal`` is what animal data for a novel
compound is worth; the gap between ``animal`` and ``human`` is what is still
missing without human data.

Every drug outside the prior list is held out in turn, so one invocation covers
the whole leave-one-drug-out sweep.

Outputs, under ``outputs/experiments/<config>/<species>/``:
    ``metrics_per_repeat.csv``  every repeat, drug and scenario
    ``summary.csv``             median and 2.5/97.5 percentiles across repeats
    ``training_results.csv``    the prior drugs scored in-sample
    ``metrics.pdf``             CRPS, MAE and sharpness by drug and scenario,
                                the exceedance table, the exceedance
                                probabilities per scenario and the transitions
                                between them, and the in-sample scores

Examples:
    python -m experiments.run_iterative --config iterative_prior_qui_ver
    python -m experiments.run_iterative --config iterative_prior_qui_ver \\
        --species Dog --drug cisapride
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from drug import load_drug_table
from gp.data import (OBSERVATION_COLUMN, add_block_columns, build_tensors,
                     channel_mask_for, drop_zero_block_rows,
                     filter_training_concentrations,
                     load_apd_dataset, sample_subjects_iterative,
                     select_all_prior_subjects)
from gp.fitting import (fit_and_predict, fit_hyperparameters, load_oracle,
                        predict)
from gp.metrics import (compute_crps, compute_mae, compute_sharpness,
                        exceedance_probability, exceedance_probability_ci,
                        quantile_interval)
from gp.plotting import plot_iterative_metrics
from paths import (APD_DATASET, CHANNELS_TABLE, CONFIG_DIR, EXPERIMENTS_OUT,
                   GP_PARAMS, ensure, ground_truth_table)

#: The three conditioning sets, in increasing order of information.
SCENARIOS = ["prior", "animal", "human"]

#: Animal species available as low-fidelity donors.
ANIMAL_SPECIES = ["Dog", "Guinea Pig", "Mouse", "Pig", "Rabbit"]

#: The species being predicted.
HIGH_FIDELITY = "Human"

#: Relative APD change above which a drug is treated as showing a signal.
EXCEEDANCE_THRESHOLD = 0.1

#: How many repeats the prior drugs are predicted for, to build the
#: training-set figure. Those predictions share the repeat's sampling stream, so
#: the number affects the scores as well as the figure. See run_one_drug.
TRAIN_FIGURE_REPEATS = 20


def load_config(name):
    """Load a config by slug, with or without the ``.json`` suffix."""
    path = CONFIG_DIR / (name if name.endswith(".json") else f"{name}.json")
    with open(path) as f:
        return json.load(f)


def oracle_path_for(config_name, animal_species):
    """Oracle checkpoint for one species, or None.

    The iterative configs fit one oracle per species and share it across target
    drugs, since the hyperparameters describe how species relate.
    """
    path = GP_PARAMS / config_name / f"{animal_species.replace(' ', '_')}__shared.pth"
    return path if path.exists() else None


def training_rows(df, drugs, species_list, config, suffix):
    """Rows for a set of drugs, restricted to the allowed concentrations.

    ``suffix`` selects the prior or target concentration lists: the prior drugs
    and the held-out drug may be observed over different ranges.
    """
    subset = df[df["Drug"].isin(drugs) & df["Species"].isin(species_list)]
    return filter_training_concentrations(
        subset,
        {
            HIGH_FIDELITY: config.get(f"human_concentration_idxs_{suffix}"),
            **{sp: config.get(f"animal_concentration_idxs_{suffix}")
               for sp in species_list if sp != HIGH_FIDELITY},
        },
    )


def build_scenario_data(prior_rows, target_animal_rows, target_human_rows,
                        scenario):
    """Conditioning set for one scenario.

    ``prior`` is the prior drugs alone; ``animal`` adds the target drug's
    low-fidelity rows; ``human`` adds its high-fidelity rows as well. The three
    are nested.
    """
    frames = [prior_rows]
    if scenario in ("animal", "human") and target_animal_rows is not None:
        frames.append(target_animal_rows)
    if scenario == "human" and target_human_rows is not None:
        frames.append(target_human_rows)
    return pd.concat(frames, ignore_index=True)


def run_one_drug(df, target_drug, prior_drugs, animal_species, config,
                 ground_truth, channel_mask, oracle=None, hyper_cache=None,
                 is_first_target=False):
    """All scenarios, all repeats, for one held-out drug.

    Args:
        hyper_cache: Optional dict reused across target drugs, mapping repeat
            index to the hyperparameters fitted on that repeat's prior sample.
            The prior draw does not depend on which drug is held out, so the fit
            is identical for every target at a given repeat. An exact reuse.
        is_first_target: Whether this is the first held-out drug of the sweep.
            The extra prior-drug prediction below runs only then, and it
            consumes the sampling stream.

Under ``use_oracle`` the prior scenario conditions on the **whole** prior
    dataset rather than a per-repeat draw, since that is the set the shipped
    hyperparameters were fitted to. Its only remaining variation across repeats
    is then the posterior draw.

    Returns:
        ``(records, pooled_maxima, prior_fit)``. ``records`` has one row per
        repeat per scenario; ``pooled_maxima`` maps each scenario to every
        repeat's sampled maxima concatenated, which is what the exceedance
        probability is estimated from; ``prior_fit`` is the last repeat's
        prior-only fit, kept so the training drugs can be scored in-sample
        without refitting.
    """
    species_list = [animal_species, HIGH_FIDELITY]
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    prior_pool = training_rows(df, prior_drugs, species_list, config, "prior")
    target_pool = training_rows(df, [target_drug], species_list, config, "target")
    if prior_pool.empty or target_pool.empty:
        return [], {}, None

    human_prior_idxs = config.get("human_concentration_idxs_prior")
    animal_prior_idxs = config.get("animal_concentration_idxs_prior")
    human_target_idxs = config.get("human_concentration_idxs_target")
    animal_target_idxs = config.get("animal_concentration_idxs_target")
    seed = config["subject_selection_seed"]

    records = []
    pooled_maxima = {scenario: [] for scenario in SCENARIOS}
    prior_fit = None
    for repeat in range(config["num_repeats"]):
        # Three separate draws, each off its own offset of the stream: prior
        # drugs, then the target drug's animal subjects, then its human ones.
        # Drawing the two species together consumes the stream differently.
        if config.get("use_oracle"):
            prior_sample = select_all_prior_subjects(prior_pool, species_list)
        else:
            prior_sample, skip = sample_subjects_iterative(
                prior_pool, prior_drugs, species_list,
                config["num_humans_prior"], config["num_animals_prior"],
                repeat, seed,
                human_conc_idxs=human_prior_idxs,
                animal_conc_idxs=animal_prior_idxs,
                seed_offset=0,
            )
            if skip or prior_sample is None:
                print(f"  [skip] insufficient prior subjects for target={target_drug}")
                break

        target_animal_sample = None
        animal_rows = target_pool[target_pool["Species"] == animal_species]
        if config["num_animals_target"] > 0 and not animal_rows.empty:
            target_animal_sample, _ = sample_subjects_iterative(
                animal_rows, [target_drug], [animal_species],
                0, config["num_animals_target"],
                repeat, seed,
                human_conc_idxs=human_target_idxs,
                animal_conc_idxs=animal_target_idxs,
                seed_offset=1000,
            )

        target_human_sample = None
        human_rows = target_pool[target_pool["Species"] == HIGH_FIDELITY]
        if config["num_humans_target"] > 0 and not human_rows.empty:
            target_human_sample, _ = sample_subjects_iterative(
                human_rows, [target_drug], [HIGH_FIDELITY],
                config["num_humans_target"], 0,
                repeat, seed,
                human_conc_idxs=human_target_idxs,
                animal_conc_idxs=animal_target_idxs,
                seed_offset=2000,
            )

        prior_sample = prior_sample.dropna(subset=[OBSERVATION_COLUMN])
        if target_animal_sample is not None:
            target_animal_sample = target_animal_sample.dropna(
                subset=[OBSERVATION_COLUMN])
        if target_human_sample is not None:
            target_human_sample = target_human_sample.dropna(
                subset=[OBSERVATION_COLUMN])

        # Fitted on the prior data alone, then held fixed across all three
        # scenarios, so they differ only in what is conditioned on. A config
        # with ``use_oracle`` supplies them from a checkpoint instead.
        if oracle is not None:
            hyper_state = load_oracle(oracle)
        else:
            hyper_state = None if hyper_cache is None else hyper_cache.get(repeat)
            if hyper_state is None:
                prior_X, prior_y = build_tensors(
                    add_block_columns(prior_sample), species_idx,
                    conc_as_input=config.get("conc_as_input", False),
                    conc_multiplier=config["conc_multiplier"],
                )
                prior_X, prior_y = drop_zero_block_rows(
                    prior_X, prior_y, config.get("boundary_constraint")
                )
                hyper_state = fit_hyperparameters(
                    prior_X, prior_y, config, len(species_list), channel_mask,
                    standardise=config.get("standardise_data", True),
                    conc_idxs=(human_prior_idxs, animal_prior_idxs),
                )
                if hyper_cache is not None:
                    hyper_cache[repeat] = hyper_state

        # One stream per repeat, shared by all three scenarios: each continues
        # where the previous stopped. See gp.fitting.predict_all_species.
        torch.manual_seed(config.get("post_samples_seed", 42) + repeat)

        for scenario in SCENARIOS:
            conditioning = build_scenario_data(
                prior_sample, target_animal_sample, target_human_sample, scenario
            )
            if conditioning.empty:
                continue

            X, y = build_tensors(
                add_block_columns(conditioning), species_idx,
                conc_as_input=config.get("conc_as_input", False),
                conc_multiplier=config["conc_multiplier"],
            )
            X, y = drop_zero_block_rows(X, y, config.get("boundary_constraint"))
            result, extras = fit_and_predict(
                X, y, config, len(species_list), channel_mask, target_drug,
                species_idx[HIGH_FIDELITY],
                ground_truth_max=ground_truth,
                standardise=config.get("standardise_data", True),
                oracle_state=hyper_state,
                species_order=[species_idx[s] for s in species_list],
                seed_posterior=False,
                seed_init=False,
                conc_idxs=(human_prior_idxs, animal_prior_idxs),
            )
            samples = result["sampled_maxima"]
            pooled_maxima[scenario].append(samples)
            if scenario == "prior":
                model, likelihood, noise_model, standardisation, _ = extras
                prior_fit = (model, likelihood, noise_model, standardisation,
                             species_idx[HIGH_FIDELITY])

                # The prior drugs are predicted here for the training-set
                # figure, off the same stream as the scenarios, so they shift
                # what `animal` and `human` sample next. Only for the first
                # target drug and the first TRAIN_FIGURE_REPEATS repeats;
                # dropping it changes those repeats' scores.
                if is_first_target and repeat < TRAIN_FIGURE_REPEATS:
                    for prior_drug in prior_drugs:
                        for species in species_list:
                            predict(
                                model, likelihood, noise_model, prior_drug,
                                species_idx[species], config,
                                standardisation=standardisation,
                            )
            records.append({
                "Drug": target_drug, "Species": animal_species,
                "Repeat": repeat, "Scenario": scenario,
                "CRPS": result["crps"],
                "MAE": compute_mae(samples, ground_truth),
                "Sharpness": compute_sharpness(samples),
                "Exceedance": exceedance_probability(samples, EXCEEDANCE_THRESHOLD),
                "Predicted_Max": result["predicted_max"],
                "Ground_Truth": ground_truth,
            })

    pooled = {
        scenario: torch.cat(chunks)
        for scenario, chunks in pooled_maxima.items() if chunks
    }
    return records, pooled, prior_fit


def score_training_drugs(prior_fit, prior_drugs, config, ground_truth,
                         high_fidelity=HIGH_FIDELITY):
    """In-sample scores for the drugs the model was trained on.

    The model saw these drugs, so they are the floor the held-out drugs are read
    against.

    Returns:
        Drug name to a dict of CRPS, MAE, sharpness and exceedance with its
        bootstrap interval.
    """
    if prior_fit is None:
        return {}
    model, likelihood, noise_model, standardisation, species_index = prior_fit

    results = {}
    for drug in prior_drugs:
        truth = float(ground_truth.loc[drug, high_fidelity])
        torch.manual_seed(config.get("post_samples_seed", 42))
        prediction = predict(
            model, likelihood, noise_model, drug, species_index, config,
            ground_truth_max=truth, standardisation=standardisation,
        )
        samples = prediction["sampled_maxima"]
        mean, lower, upper = exceedance_probability_ci(
            samples, EXCEEDANCE_THRESHOLD
        )
        results[drug] = {
            "CRPS": float(compute_crps(samples, truth)),
            "MAE": float(compute_mae(samples, truth)),
            "Sharpness": float(compute_sharpness(samples)),
            "Exceedance": mean,
            "Exceedance_Lower": lower,
            "Exceedance_Upper": upper,
            "Ground_Truth": truth,
        }
    return results


def summarise(records):
    """Median and central 95% interval per drug and scenario."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (drug, species, scenario), group in df.groupby(
        ["Drug", "Species", "Scenario"]
    ):
        for metric in ("CRPS", "MAE", "Sharpness", "Exceedance"):
            median, lower, upper = quantile_interval(group[metric].to_numpy())
            rows.append({
                "Drug": drug, "Species": species, "Scenario": scenario,
                "Metric": metric, "Median": median, "Lower": lower, "Upper": upper,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Config slug.")
    parser.add_argument("--species", nargs="+", default=ANIMAL_SPECIES,
                        choices=ANIMAL_SPECIES)
    parser.add_argument("--drug", nargs="+", default=None,
                        help="Target drugs to hold out. Default: every drug "
                             "not in the config's prior list.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--dataset", default=None,
                        help="APD dataset to read. Defaults to the shipped one.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_name = args.config.replace(".json", "")
    if args.repeats is not None:
        config["num_repeats"] = args.repeats

    prior_drugs = config["prior_drug_list"]
    all_drugs = load_drug_table().index.tolist()
    # A drug in the prior list is given to the model, not held out.
    targets = args.drug or [d for d in all_drugs if d not in prior_drugs]

    df = load_apd_dataset(args.dataset or APD_DATASET)
    ground_truth = pd.read_csv(
        ground_truth_table(config["conc_multiplier"]), index_col=0
    )

    print(f"Prior drugs: {', '.join(prior_drugs)}")
    print(f"Held out:    {', '.join(targets)}\n")

    for animal_species in args.species:
        channel_mask = channel_mask_for([animal_species, HIGH_FIDELITY],
                                        CHANNELS_TABLE)
        oracle = (
            oracle_path_for(config_name, animal_species)
            if config.get("use_oracle") else None
        )
        if config.get("use_oracle") and oracle is None:
            print(f"  [warn] {animal_species}: use_oracle set but no checkpoint; "
                  f"fitting instead")

        records = []
        exceedance = {}
        prior_fit = None
        # Shared across target drugs; see run_one_drug.
        hyper_cache = {}
        for target_drug in targets:
            truth = float(ground_truth.loc[target_drug, HIGH_FIDELITY])
            drug_records, pooled, drug_prior_fit = run_one_drug(
                df, target_drug, prior_drugs, animal_species, config,
                truth, channel_mask, oracle=oracle, hyper_cache=hyper_cache,
                is_first_target=(target_drug == targets[0]),
            )
            if not drug_records:
                print(f"  [skip] {animal_species}/{target_drug}")
                continue
            records.extend(drug_records)
            prior_fit = prior_fit or drug_prior_fit
            # Pooled over every repeat's draws, not averaged over per-repeat
            # probabilities; the interval is a bootstrap of that pool.
            exceedance[target_drug] = {
                scenario: exceedance_probability_ci(
                    samples, EXCEEDANCE_THRESHOLD
                )
                for scenario, samples in pooled.items()
            }
            medians = {
                s: np.nanmedian([r["CRPS"] for r in drug_records
                                 if r["Scenario"] == s])
                for s in SCENARIOS
            }
            print(
                f"  {animal_species:11s} {target_drug:15s} CRPS "
                + "  ".join(f"{s}={medians[s]:.4f}" for s in SCENARIOS)
            )

        if not records:
            continue

        training_results = score_training_drugs(
            prior_fit, prior_drugs, config, ground_truth
        )

        out_dir = ensure(EXPERIMENTS_OUT / config_name / animal_species)
        per_repeat = pd.DataFrame(records)
        per_repeat.to_csv(out_dir / "metrics_per_repeat.csv", index=False)
        summarise(records).to_csv(out_dir / "summary.csv", index=False)
        if training_results:
            pd.DataFrame([
                {"Drug": drug, **values}
                for drug, values in training_results.items()
            ]).to_csv(out_dir / "training_results.csv", index=False)

        if not args.no_plots:
            plot_iterative_metrics(
                per_repeat, animal_species, EXCEEDANCE_THRESHOLD,
                exceedance=exceedance, training_results=training_results,
                prior_drugs=prior_drugs,
                save_path=out_dir / "metrics.pdf",
            )
        print(f"  wrote {len(per_repeat)} rows to {out_dir}\n")


if __name__ == "__main__":
    main()
