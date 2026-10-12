"""Fit the oracle hyperparameters that the ``use_oracle`` configs load.

An oracle checkpoint holds GP hyperparameters fitted once to the *maximal*
training set — every subject available, rather than a per-repeat draw — and the
standardisation they were fitted against. The runners load these hyperparamters instead of
fitting, so repeats differ only in which subjects they condition on.

Two layouts, chosen by the config:

``single_drug_*``
    One checkpoint per (animal species, drug), fitted on that pair's rows with
    the concentration filter **not** applied: the oracle sees every simulated
    concentration, while the experiment conditions only on the lowest few.
    Written to ``<out>/<Species>__<drug>.pth``.

``iterative_*_oracle``
    One checkpoint per animal species, shared across held-out drugs, fitted on
    the prior drugs at the prior concentrations, capped by
    :func:`gp.data.select_all_prior_subjects`. That is the same set
    :mod:`experiments.run_iterative` conditions on under ``use_oracle``.
    Written to ``<out>/<Species>__shared.pth``.

The shipped checkpoints were fitted with a single torch thread, so this script
pins that. The thread count changes the order of BLAS reductions, 
so a multi-threaded refit does not reproduce them.

The sweep's oracle is fitted lazily by
:mod:`experiments.run_single_drug_sweep` through :func:`fit_oracle`.

Output defaults to ``outputs/gp_params/<config>/`` so a refit never overwrites
the checkpoints shipped in ``data/gp_params/``; pass ``--out`` to choose.

Example:
    python -m experiments.fit_oracle --config single_drug_1x \\
        --species Dog --drug bepridil
    python -m experiments.fit_oracle --config iterative_prior_bep_dil_qui_ver_oracle
"""

import argparse
from pathlib import Path

import torch

from drug import load_drug_table
from gp.data import (OBSERVATION_COLUMN, add_block_columns, build_tensors,
                     channel_mask_for, drop_zero_block_rows,
                     filter_training_concentrations, load_apd_dataset,
                     select_all_prior_subjects)
from gp.fitting import fit_hyperparameters
from paths import APD_DATASET, CHANNELS_TABLE, OUTPUT_ROOT
from experiments.run_single_drug import (ANIMAL_SPECIES, HIGH_FIDELITY,
                                         load_config)

#: Torch threads the shipped checkpoints were fitted with. See the module
#: docstring.
NUM_THREADS = 1


def fit_oracle(rows, species_list, config, channel_mask, path, conc_idxs=None):
    """Fit hyperparameters to ``rows`` as given and save them to ``path``.

    ``rows`` is the maximal training set, already selected. Zero-block rows are
    dropped, as the boundary constraint already pins ``f(0) = 0``.
    """
    species_idx = {sp: i for i, sp in enumerate(species_list)}
    X, y = build_tensors(
        add_block_columns(rows.dropna(subset=[OBSERVATION_COLUMN])),
        species_idx,
        conc_as_input=config.get("conc_as_input", False),
        conc_multiplier=config["conc_multiplier"],
    )
    X, y = drop_zero_block_rows(X, y, config.get("boundary_constraint"))
    model_state, noise_state, standardisation = fit_hyperparameters(
        X, y, config, len(species_list), channel_mask,
        standardise=config.get("standardise_data", True),
        conc_idxs=conc_idxs,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model_state,
        "noise_state_dict": noise_state,
        "standardisation_params": standardisation,
    }, path)
    print(f"    [oracle] fitted on {len(rows)} rows -> {path}")
    return path


def single_drug_rows(df, drug, species_list):
    """Every row for one drug and species pair, at every concentration."""
    return select_all_prior_subjects(df[df["Drug"] == drug], species_list)


def iterative_rows(df, config, species_list):
    """The prior drugs at the prior concentrations, capped by subject."""
    prior = df[df["Drug"].isin(config["prior_drug_list"])
               & df["Species"].isin(species_list)]
    prior = filter_training_concentrations(
        prior,
        {
            HIGH_FIDELITY: config.get("human_concentration_idxs_prior"),
            species_list[0]: config.get("animal_concentration_idxs_prior"),
        },
    )
    return select_all_prior_subjects(prior, species_list)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Config slug.")
    parser.add_argument("--species", nargs="+", default=ANIMAL_SPECIES)
    parser.add_argument("--drug", nargs="+", default=None,
                        help="Single-drug configs only.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", default=None,
                        help="Output directory. Default "
                             "outputs/gp_params/<config>/.")
    args = parser.parse_args()

    torch.set_num_threads(NUM_THREADS)
    config = load_config(args.config)
    config_name = args.config.replace(".json", "")
    out_dir = (Path(args.out) if args.out
               else OUTPUT_ROOT / "gp_params" / config_name)
    df = load_apd_dataset(args.dataset or APD_DATASET)
    iterative = "prior_drug_list" in config

    for animal_species in args.species:
        species_list = [animal_species, HIGH_FIDELITY]
        channel_mask = channel_mask_for(species_list, CHANNELS_TABLE)
        stem = animal_species.replace(" ", "_")

        if iterative:
            fit_oracle(
                iterative_rows(df, config, species_list), species_list,
                config, channel_mask, out_dir / f"{stem}__shared.pth",
                conc_idxs=(config.get("human_concentration_idxs_prior"),
                           config.get("animal_concentration_idxs_prior")),
            )
            continue

        for drug in args.drug or load_drug_table().index.tolist():
            rows = single_drug_rows(df, drug, species_list)
            if rows.empty:
                print(f"  [skip] {animal_species}/{drug}: no rows")
                continue
            fit_oracle(rows, species_list, config, channel_mask,
                       out_dir / f"{stem}__{drug}.pth")


if __name__ == "__main__":
    main()
