"""What the iterative fits learned about how species relate.

The iterative experiments fit one set of hyperparameters per animal species, on
the prior drugs, and then hold them fixed. Three views of them:

**Species correlation.** The 2x2 correlation between the animal task and the
human task, per latent process. A latent with animal-human correlation near 1
is one where animal data speaks directly about humans.

**Species covariance.** The same matrices unnormalised: how much signal there
is to share, rather than how aligned the species are.

**ARD lengthscales per ion channel.** One lengthscale per channel per latent; a
short lengthscale marks a channel the response varies sharply with.

Latent processes are exchangeable, so they are sorted by animal-human
correlation before plotting and labelled by that rank ("High Correlation" /
"Low Correlation") rather than by their position in the checkpoint.

Each species is taken from whichever of the two oracle configs
:data:`DEFAULT_SOURCES` names; page 1 records that provenance.

Example:
    python -m experiments.plot_iterative_hyperparams
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages

from drug import ALL_CHANNELS
from gp.plotting import apply_style
from paths import EXPERIMENTS_OUT, GP_PARAMS, ensure

#: Which config each species' hyperparameters are taken from.
#:
#: Four species from the quinidine-containing prior,
#: guinea pig from the dofetilide-containing one.
#: Page 1 records the mix.
DEFAULT_SOURCES = {
    "Dog": "iterative_prior_bep_dil_qui_ver_oracle",
    "Guinea Pig": "iterative_prior_bep_dil_dof_ver_oracle",
    "Mouse": "iterative_prior_bep_dil_qui_ver_oracle",
    "Pig": "iterative_prior_bep_dil_qui_ver_oracle",
    "Rabbit": "iterative_prior_bep_dil_qui_ver_oracle",
}

#: Prior drugs per config, for the provenance table.
PRIOR_DRUGS = {
    "iterative_prior_bep_dil_qui_ver_oracle":
        "bepridil, diltiazem, quinidine, verapamil",
    "iterative_prior_bep_dil_dof_ver_oracle":
        "bepridil, diltiazem, dofetilide, verapamil",
}

#: Channel labels, in the order the model's block dimensions are laid out.
CHANNEL_LABELS = {
    "ICaL": "$I_{CaL}$", "IK1": "$I_{K1}$", "IKs": "$I_{Ks}$",
    "INa": "$I_{Na}$", "INaL": "$I_{NaL}$", "Ito": "$I_{to}$",
    "IKr": "$I_{Kr}$",
}

#: The species being predicted. Always the last task index.
HIGH_FIDELITY = "Human"


def covariance_from(covar_factor, raw_var):
    """Rebuild an ``IndexKernel``'s covariance from its stored factors.

    GPyTorch keeps a low-rank factor and a positive diagonal separately: the
    task covariance is ``F @ F.T + diag(softplus(raw_var))``.
    """
    factor = np.atleast_2d(covar_factor)
    if factor.shape[0] == 1 and factor.shape[1] != 1:
        factor = factor.T
    var = F.softplus(torch.as_tensor(raw_var)).numpy().flatten()
    matrix = factor @ factor.T
    if var.shape[0] == matrix.shape[0]:
        matrix = matrix + np.diag(var)
    return matrix


def correlation_from(covariance, eps=1e-12):
    """Correlation matrix of a covariance, clipped to [-1, 1]."""
    scale = np.sqrt(np.clip(np.diag(covariance), eps, None))
    return np.clip(covariance / np.outer(scale, scale), -1.0, 1.0)


def read_checkpoint(path, animal_species):
    """Covariances and lengthscales from one oracle checkpoint.

    Latents are sorted by animal-human correlation, highest first, so index 0
    is the most strongly shared latent.

    Returns:
        ``{"covariances": [...], "correlations": [...], "lengthscales": [...],
        "species": [animal, "Human"]}``, each list ordered by latent.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_state = state["model_state_dict"]

    latents = sorted({
        int(key.split(".")[1])
        for key in model_state if key.startswith("index_kernels.")
    })

    covariances, lengthscales = [], []
    for latent in latents:
        covariances.append(covariance_from(
            model_state[f"index_kernels.{latent}.covar_factor"].numpy(),
            model_state[f"index_kernels.{latent}.raw_var"],
        ))
        raw = model_state[f"rbf_kernels.{latent}.base_kernel.raw_lengthscale"]
        lengthscales.append(F.softplus(raw).numpy().flatten())

    # Descending by animal-to-human correlation; ties keep their original order.
    correlations = [correlation_from(c) for c in covariances]
    order = sorted(range(len(latents)),
                   key=lambda q: (-correlations[q][0, 1], q))

    return {
        "covariances": [covariances[q] for q in order],
        "correlations": [correlations[q] for q in order],
        "lengthscales": [lengthscales[q] for q in order],
        "species": [animal_species, HIGH_FIDELITY],
    }


def latent_label(index, total):
    """Name a latent by its correlation rank rather than its stored position."""
    if total == 2:
        return "High Correlation Latent" if index == 0 else "Low Correlation Latent"
    if index == 0:
        return "Highest Correlation Latent"
    if index == total - 1:
        return "Lowest Correlation Latent"
    return f"Correlation Rank {index + 1}"


def plot_provenance(pdf, records, sources):
    """Page 1: which config each species' hyperparameters came from."""
    fig, ax = plt.subplots(figsize=(11, 2 + 0.5 * len(records)))
    ax.axis("off")
    rows = [
        [species, sources[species].replace("iterative_prior_", ""),
         PRIOR_DRUGS.get(sources[species], "")]
        for species in records
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Species", "Config", "Prior drugs"],
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    ax.set_title("Where each species came from", fontsize=14, pad=18)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_page(pdf, records, key, title, fmt):
    """One row per species, one column per latent, of 2x2 task matrices."""
    num_latents = max(len(r[key]) for r in records.values())
    fig, axes = plt.subplots(
        len(records), num_latents,
        figsize=(4.2 * num_latents, 3.6 * len(records)),
        squeeze=False,
    )

    for row, (species, record) in enumerate(records.items()):
        labels = [species, HIGH_FIDELITY]
        for col in range(num_latents):
            ax = axes[row][col]
            matrix = record[key][col]
            limit = np.abs(matrix).max() or 1.0
            ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, format(matrix[i, j], fmt),
                            ha="center", va="center", fontsize=11)
            ax.set_xticks(range(len(labels)), labels, fontsize=9)
            ax.set_yticks(range(len(labels)), labels, fontsize=9)
            if row == 0:
                ax.set_title(latent_label(col, num_latents), fontsize=11, pad=10)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_lengthscale_page(pdf, records):
    """One row per species, one column per latent, of per-channel ARD bars."""
    num_latents = max(len(r["lengthscales"]) for r in records.values())
    labels = [CHANNEL_LABELS.get(c, c) for c in ALL_CHANNELS]
    fig, axes = plt.subplots(
        len(records), num_latents,
        figsize=(5.0 * num_latents, 2.9 * len(records)),
        squeeze=False, sharex=True,
    )

    for row, (species, record) in enumerate(records.items()):
        for col in range(num_latents):
            ax = axes[row][col]
            values = record["lengthscales"][col]
            ax.bar(range(len(values)), values,
                   color="#3E75E2", edgecolor="black", linewidth=0.6)
            ax.set_xticks(range(len(values)), labels[:len(values)], fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(species, fontsize=11)
            if row == 0:
                ax.set_title(latent_label(col, num_latents), fontsize=11, pad=10)

    fig.suptitle(
        "ARD lengthscales per ion channel\n"
        "Lengthscale (standardised block units); shorter = response varies "
        "more sharply with that channel's block",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source", action="append", default=None, metavar="SPECIES=CONFIG",
        help="Override where one species is taken from; repeatable.",
    )
    args = parser.parse_args()

    sources = dict(DEFAULT_SOURCES)
    for entry in args.source or []:
        species, _, config = entry.partition("=")
        sources[species.strip()] = config.strip()

    apply_style()
    records = {}
    for species, config in sources.items():
        path = GP_PARAMS / config / f"{species.replace(' ', '_')}__shared.pth"
        if not path.exists():
            print(f"  [warn] no checkpoint for {species} in {config}, skipping")
            continue
        records[species] = read_checkpoint(path, species)

    if not records:
        print("No checkpoints found; nothing to plot.")
        return

    out_dir = ensure(EXPERIMENTS_OUT / "iterative_hyperparams")
    save_path = out_dir / "species_comparison.pdf"
    with PdfPages(save_path) as pdf:
        plot_provenance(pdf, records, sources)
        plot_matrix_page(
            pdf, records, "correlations",
            "Species correlation by latent process\n"
            "1 = the animal task and the human task move together",
            ".2f",
        )
        plot_matrix_page(
            pdf, records, "covariances",
            "Species covariance by latent process\n"
            "the same matrices unnormalised: how much signal each latent carries",
            ".3f",
        )
        plot_lengthscale_page(pdf, records)

    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
