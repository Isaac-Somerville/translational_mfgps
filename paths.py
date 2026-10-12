"""Canonical filesystem locations for the package.

Every path is derived from the location of this file, so scripts can be run from
any working directory:

    python -m experiments.run_single_drug --config single_drug_1x

Directory map
-------------
``data/``    inputs that ship with the repository (read-only in normal use)
``outputs/`` everything the scripts write: PDFs, CSVs, saved hyperparameters

The simulated APD dataset under ``data/simulations/`` is an *input* here,
produced by the companion simulation repository. A copy ships here so these
experiments reproduce without cloning it.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Hill-equation parameters (IC50, h), therapeutic and maximum concentrations,
# and CiPA torsade risk category, one row per drug. Read by drug.py.
CIPA_TABLE = DATA_ROOT / "cipa" / "CiPA_optimal_combined.csv"

# Which of the seven ion channels each species model represents; masks the
# input, since a drug cannot block a current the model does not have.
CHANNELS_TABLE = DATA_ROOT / "channels" / "channels_per_model.csv"

# Simulated APD under drug block: one row per
# (species, drug, subject, concentration). The GP observes
# ``APD_90_relative_noisy`` and is scored against ``APD_90_relative``.
APD_DATASET = DATA_ROOT / "simulations" / "apd_drug_block.csv"

# Exceedance probabilities per drug and species, at 1x and 3x the maximum
# simulated concentration. The target the experiments are scored against.
GROUND_TRUTH_DIR = DATA_ROOT / "simulations"


def ground_truth_table(conc_multiplier):
    """Path to the ground-truth table for a concentration multiplier (1 or 3)."""
    return GROUND_TRUTH_DIR / f"ground_truth_relative_{conc_multiplier}x.csv"


# Measured resting and intrinsic heart rate for Human, Dog and Mouse.
HEART_RATE_TABLE = DATA_ROOT / "heart_rate" / "heart_rate.csv"

# Experiment configurations, one JSON per published experiment.
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"

# Per-repeat scores from the published runs, used only by verification/.
REFERENCE_RESULTS = DATA_ROOT / "reference_results"

# ---------------------------------------------------------------------------
# Inputs that are also outputs
# ---------------------------------------------------------------------------

# Oracle hyperparameters: fitted once to the maximal training set and reused by
# the configs with ``use_oracle: true``. Refit with experiments.fit_oracle, except
# the sweep's, which run_single_drug_sweep fits when absent.
GP_PARAMS = DATA_ROOT / "gp_params"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

EDA_OUT = OUTPUT_ROOT / "eda"
EXPERIMENTS_OUT = OUTPUT_ROOT / "experiments"
HEART_RATE_OUT = OUTPUT_ROOT / "heart_rate"
FIGURES_OUT = OUTPUT_ROOT / "figures"
VERIFICATION_OUT = OUTPUT_ROOT / "verification"


def ensure(path: Path) -> Path:
    """Create ``path`` as a directory if absent, then return it.

    Convenience for output paths: ``ensure(EDA_OUT) / "hill_curves"``.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
