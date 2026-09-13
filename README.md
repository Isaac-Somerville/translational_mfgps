# Cross-species Gaussian process models of drug-induced cardiac effects

Gaussian process models that transfer information from animal experiments to
human ones, focusing on examples of action-potential duration (APD) measurements
under drug block and intrinsic heart rate. This accompanies the research article 
"Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical Outcomes" 
(https://arxiv.org/abs/2609.05007).

The cardiac simulations that generate the APD datasets live in a companion
repository, https://github.com/Isaac-Somerville/apd_drug_effects.

## Contents

| Path | Contents |
|---|---|
| `gp/` | GP models, data preparation, fitting, metrics, plotting |
| `experiments/` | Experiment runners and their JSON configs |
| `heart_rate/` | Heart-rate transfer experiments |
| `eda/`, `figures/` | Exploratory plots and the illustration figure |
| `verification/` | Checks that this code reproduces the published results |
| `data/` | Simulated APD data, CiPA parameters, fitted GP hyperparameters |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m experiments.run_single_drug --config single_drug_1x
python -m experiments.run_iterative   --config iterative_prior_qui_ver
python -m heart_rate.run_heart_rate   --config hr_human_from_dog_autonomic
```

Results are written under `outputs/`. Configs live in `experiments/configs/`;
`data/gp_params/` holds hyperparameters fitted once to the maximal dataset and
reused by configs with `use_oracle: true`.

## Experimental configurations

| Config | What it is |
|---|---|
| `single_drug_1x` | One target drug, 3 human subjects plus 9 of one animal species, conditioning on the lowest concentrations, predicting the maximum response up to 1x therapeutic. Oracle hyperparameters. |
| `single_drug_3x` | The same at 3x therapeutic. |
| `iterative_prior_qui_ver` | Leave-one-drug-out, with quinidine and verapamil as the prior. Hyperparameters fitted per repeat. |
| `iterative_prior_dil_qui` | As above, prior of diltiazem and quinidine. |
| `iterative_prior_bep_dil_qui_ver` | As above, four-drug prior. |
| `iterative_prior_bep_dil_qui_ver_oracle` | The four-drug prior with oracle hyperparameters, conditioning on the full prior dataset. |
| `iterative_prior_bep_dil_dof_ver_oracle` | As above, with dofetilide in place of quinidine. |
| `hr_dog_from_mouse_sa_node` | Heart rate: predict dog from mouse, SA-node endpoint, sweeping training-set size. |
| `hr_human_from_dog_autonomic` | Heart rate: predict human from dog, autonomic-block endpoint. |
| `hr_*_athletes` | The same two, split by holding out the athletes rather than at random. |

## Reproducing the published results

In dependency order. Every command writes to `outputs/`, mirroring the figure
and table layout of the paper.

```bash
# 1. Single-drug experiments, both concentration ranges  (~1-2 h each)
python -m experiments.run_single_drug --config single_drug_1x
python -m experiments.run_single_drug --config single_drug_3x
python -m experiments.compare_single_drug --config single_drug_1x

# 2. Training-set-size sweep for the 1x case            (~3-4 h)
python -m experiments.run_single_drug_sweep --config single_drug_1x

# 3. Leave-one-drug-out, five configs                   (~1-2 h each)
python -m experiments.run_iterative --config iterative_prior_qui_ver
python -m experiments.plot_iterative_hyperparams

# 4. Heart rate                                         (minutes)
python -m heart_rate.run_heart_rate --config hr_human_from_dog_autonomic
python -m heart_rate.run_heart_rate_sweep --config hr_human_from_dog_autonomic
python -m heart_rate.plot_athlete_crps_improvement

# 5. Exploratory figures and paper tables               (seconds)
python -m eda.plot_ground_truth_curves
python -m eda.plot_hill_curves
python -m eda.plot_trajectory_geometry
python -m figures.transfer_learning_illustration
```

Timings are for a single CPU core; nothing here needs a GPU. Add `--drug` and
`--species` to any experiment runner to run one cell of the grid.

## Verifying a reproduction

```bash
python -m verification.verify_gp_experiments          # the scores match the paper
python -m verification.verify_gp_experiments --sweep  # and the sweep surfaces
```

Both run against the references in `data/reference_results/`.
`data/reference_results/*_per_repeat.csv` hold the published
per-repeat scores, and the check compares repeat-wise
directly. The tolerance is set at the floating-point level (1e-9).

## Relationship to the simulation repository

The two repositories are separate programs that communicate through CSV files on
disk. `data/simulations/*.csv` are the output of the simulations in the partner repository.
This separation was chosen to allow this repository to be MIT-licensed.

## Citation

If you use this code or the datasets it ships, please cite:

> Hayden, I. S., D'Souza, A., Niederer, S. and Filippi, S. (2026).
> *Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical
> Outcomes.* arXiv:2609.05007. <https://arxiv.org/abs/2609.05007>

```bibtex
@misc{hayden2026multifidelity,
  title         = {Multi-Fidelity Gaussian Processes for Translational
                   Modelling of Clinical Outcomes},
  author        = {Hayden, Isaac S. and D'Souza, Alicia and
                   Niederer, Steven and Filippi, Sarah},
  year          = {2026},
  eprint        = {2609.05007},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.05007}
}
```

## Licence

**MIT** — see `LICENSE`. This covers the code written by the author.

Third-party material is documented in **`THIRD_PARTY_NOTICES.md`**. In particular,
the CiPA ion-channel parameters derive from the U.S. FDA's CiPA project; see that
file for the provenance, the citation to request, and the FDA disclaimer.
