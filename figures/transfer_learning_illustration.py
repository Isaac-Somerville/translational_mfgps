import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import FIGURES_OUT, ensure

# ── Aesthetics (matching mf_gp_illustration.py) ──────────────────────────────
sns.set_theme(style="white", context="poster")
plt.rcParams.update({
    "axes.linewidth": 1.8,
    "axes.labelsize": 32,
    "axes.titlesize": 36,
    "legend.fontsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.frameon": False,
})
matplotlib.rcParams["font.family"] = "sans-serif"

HF_COLOR = "#5E3C99"   # human  (high fidelity) — purple
LF_COLOR = "#E66101"   # animal (low  fidelity) — orange


# ── GP utilities ─────────────────────────────────────────────────────────────
def rbf(X1, X2, ls=1.8, var=1.0):
    X1 = np.atleast_1d(X1).reshape(-1, 1)
    X2 = np.atleast_1d(X2).reshape(-1, 1)
    return var * np.exp(-0.5 * (X1 - X2.T) ** 2 / ls ** 2)


def gp_posterior(X_tr, y_tr, X_te, noise_var, m_tr, m_te, ls=1.8, var=1.0):
    """Single-fidelity GP posterior with an informed (non-zero) prior mean.

    Models the residual y - m(x); adds the prior mean back to the predictive mean.
    """
    n = len(X_tr)
    K_tt = rbf(X_tr, X_tr, ls, var) + (noise_var + 1e-8) * np.eye(n)
    K_ts = rbf(X_tr, X_te, ls, var)
    L = np.linalg.cholesky(K_tt)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - m_tr))
    mean = m_te + K_ts.T @ alpha
    v = np.linalg.solve(L, K_ts)
    post_var = np.full(len(X_te), var) - np.sum(v ** 2, axis=0)
    return mean, np.maximum(post_var, 0.0)


def mf_hf_posterior(X_lf, y_lf, X_hf, y_hf, X_te,
                    noise_lf, noise_hf, rho,
                    m_lf_tr, m_hf_tr, m_lf_te, m_hf_te,
                    ls_lf=1.8, var_lf=1.0, ls_d=1.2, var_d=0.3):
    """AR(1) Kennedy & O'Hagan multi-fidelity GP — HF (human) predictive only.

    Borrows LF (animal) data through the rho cross-covariance. Handles the
    empty-HF case (n_hf = 0), where the HF prediction is driven purely by the
    rho * K(X_te, X_lf) cross block. Informed prior means m_* are modelled as
    deterministic offsets: the GP fits residuals y - m(x); m is added back to the
    predictive mean (variance is unaffected by the mean shift).
    """
    X_hf = np.atleast_1d(X_hf)
    X_lf = np.atleast_1d(X_lf)
    n_hf, n_lf = len(X_hf), len(X_lf)

    r_hf = (y_hf - m_hf_tr) if n_hf else np.array([])
    r_lf = y_lf - m_lf_tr
    y_train = np.concatenate([r_hf, r_lf])

    if n_hf:
        K_HH = (rho ** 2 * rbf(X_hf, X_hf, ls_lf, var_lf)
                + rbf(X_hf, X_hf, ls_d, var_d) + noise_hf * np.eye(n_hf))
        K_HL = rho * rbf(X_hf, X_lf, ls_lf, var_lf)
        K_LL = rbf(X_lf, X_lf, ls_lf, var_lf) + noise_lf * np.eye(n_lf)
        K_train = np.block([[K_HH, K_HL], [K_HL.T, K_LL]]) + 1e-8 * np.eye(n_hf + n_lf)
        k_hf = np.hstack([
            rho ** 2 * rbf(X_te, X_hf, ls_lf, var_lf) + rbf(X_te, X_hf, ls_d, var_d),
            rho * rbf(X_te, X_lf, ls_lf, var_lf),
        ])
    else:
        K_train = rbf(X_lf, X_lf, ls_lf, var_lf) + (noise_lf + 1e-8) * np.eye(n_lf)
        k_hf = rho * rbf(X_te, X_lf, ls_lf, var_lf)

    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    mean_hf = m_hf_te + k_hf @ alpha
    v_hf = np.linalg.solve(L, k_hf.T)
    var_hf = np.full(len(X_te), rho ** 2 * var_lf + var_d) - np.sum(v_hf ** 2, axis=0)
    return mean_hf, np.maximum(var_hf, 0.0)


# ── Ground-truth & prior-mean functions ──────────────────────────────────────
def curve(x, start, plateau, x_step, w_step, peak_amp, x_peak, w_peak):
    """Rise → max → decrease → plateau, with the ending plateau ABOVE the
    starting value. A start→plateau sigmoid baseline carries the level up; a
    Gaussian bump adds the peak (peak > plateau > start)."""
    x = np.atleast_1d(x)
    baseline = start + (plateau - start) / (1.0 + np.exp(-(x - x_step) / w_step))
    bump = peak_amp * np.exp(-((x - x_peak) / w_peak) ** 2)
    return baseline + bump


# Two ground truths: same qualitative shape, strongly accentuated difference
# (peak height & location, plateau, start all clearly distinct between species).
def f_hf(x):  # human
    return curve(x, start=0.05, plateau=0.78, x_step=2.7, w_step=1.3,
                 peak_amp=0.55, x_peak=2.6, w_peak=1.15)


def f_lf(x):  # animal
    return curve(x, start=0.15, plateau=0.55, x_step=3.7, w_step=1.5,
                 peak_amp=0.36, x_peak=3.7, w_peak=1.30)


# Informed priors: broadly correct shape but visibly inaccurate (flatter, lower
# plateau, smaller & shifted peak) → true GT still sits inside the band.
def m_hf(x):
    return curve(x, start=0.10, plateau=0.55, x_step=3.3, w_step=1.6,
                 peak_amp=0.22, x_peak=3.3, w_peak=1.40)


def m_lf(x):
    return curve(x, start=0.20, plateau=0.38, x_step=4.4, w_step=1.7,
                 peak_amp=0.14, x_peak=4.5, w_peak=1.50)


# ── Data ─────────────────────────────────────────────────────────────────────
np.random.seed(7)
X_te = np.linspace(0, 10, 250)

LS, VAR = 2.4, 0.12       # shared kernel params (VAR = prior signal var → prior band width)
RHO = 0.90
LS_D, VAR_D = 2.4, 0.04   # delta kernel for MF
NOISE_HF, NOISE_LF = 0.05, 0.10

# Animal (LF): 7 points spread across the interval.
X_lf = np.array([0.5, 1.5, 3.0, 3.5, 4.5, 5.0, 6.5, 7.5, 8.5, 9.5 ]) 
y_lf = f_lf(X_lf) + np.random.normal(0, NOISE_LF, len(X_lf))

# Human (HF): 3 points — fewer than animal.
X_hf = np.array([0.5, 2.5, 4.0, 6.5, 9.0])
y_hf = f_hf(X_hf) + np.random.normal(0, NOISE_HF, len(X_hf))


# ── Posteriors per stage ─────────────────────────────────────────────────────
# Stage 1 — informed prior (no data).
prior_hf_mean, prior_hf_var = m_hf(X_te), np.full(len(X_te), RHO ** 2 * VAR + VAR_D)
prior_lf_mean, prior_lf_var = m_lf(X_te), np.full(len(X_te), VAR)

# Animal = single-fidelity GP on its own data (one-directional transfer:
# identical whether or not human data has since been collected).
lf_post_mean, lf_post_var = gp_posterior(
    X_lf, y_lf, X_te, NOISE_LF ** 2, m_lf(X_lf), m_lf(X_te), LS, VAR)

# Human stage 2 — multi-fidelity, borrowing animal data only.
hf_s2_mean, hf_s2_var = mf_hf_posterior(
    X_lf, y_lf, np.array([]), np.array([]), X_te,
    NOISE_LF ** 2, NOISE_HF ** 2, RHO,
    m_lf(X_lf), np.array([]), m_lf(X_te), m_hf(X_te),
    ls_lf=LS, var_lf=VAR, ls_d=LS_D, var_d=VAR_D)

# Human stage 3 — multi-fidelity, animal + human data.
hf_s3_mean, hf_s3_var = mf_hf_posterior(
    X_lf, y_lf, X_hf, y_hf, X_te,
    NOISE_LF ** 2, NOISE_HF ** 2, RHO,
    m_lf(X_lf), m_hf(X_hf), m_lf(X_te), m_hf(X_te),
    ls_lf=LS, var_lf=VAR, ls_d=LS_D, var_d=VAR_D)


# (name, mean, var, gt, colour, obs_x, obs_y)
panels = [
    ("tl_human_prior",       prior_hf_mean, prior_hf_var, f_hf, HF_COLOR, None,  None),
    ("tl_animal_prior",      prior_lf_mean, prior_lf_var, f_lf, LF_COLOR, None,  None),
    ("tl_human_animaldata",  hf_s2_mean,    hf_s2_var,    f_hf, HF_COLOR, None,  None),
    ("tl_animal_animaldata", lf_post_mean,  lf_post_var,  f_lf, LF_COLOR, X_lf,  y_lf),
    ("tl_human_humandata",   hf_s3_mean,    hf_s3_var,    f_hf, HF_COLOR, X_hf,  y_hf),
    ("tl_animal_humandata",  lf_post_mean,  lf_post_var,  f_lf, LF_COLOR, X_lf,  y_lf),
]

# Shared y-limits across all six plots (widest band = stage-1 prior).
lo = min(np.min(m - 2 * np.sqrt(v)) for _, m, v, *_ in panels)
hi = max(np.max(m + 2 * np.sqrt(v)) for _, m, v, *_ in panels)
pad = 0.06 * (hi - lo)
YLIM = (lo - pad, hi + pad)


# ── Draw & save each plot ────────────────────────────────────────────────────
def make_plot(name, mean, var, gt_fn, colour, obs_x, obs_y, ylim=None):
    std = np.sqrt(var)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill_between(X_te, mean - 2 * std, mean + 2 * std, color=colour, alpha=0.25)
    ax.plot(X_te, mean, color=colour, linewidth=4.5)
    ax.plot(X_te, gt_fn(X_te), color="#000000", linewidth=3.5, linestyle="--")
    if obs_x is not None:
        ax.scatter(obs_x, obs_y, s=260, color=colour, edgecolor="black",
                   linewidth=0.6, alpha=0.9, zorder=3, marker="o")
    ax.set_xlim(0, 10)
    ax.set_ylim(*(ylim if ylim is not None else YLIM))
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    sns.despine(ax=ax)
    out_dir = ensure(FIGURES_OUT)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.close(fig)
    print(f"Wrote {out_dir / (name + '.pdf')}")


for spec in panels:
    make_plot(*spec)


# ══════════════════════════════════════════════════════════════════════════════
# GT-variety demo: 3 more scenarios (distinct HF/LF pair each), final-style plots
# showing the method generalises across shapes that vary in wiggliness /
# monotonicity / nonlinearity. HF = multi-fidelity (borrows animal data); LF = SF.
# ══════════════════════════════════════════════════════════════════════════════
def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# Three similar-but-distinct HF/LF ground-truth pairs, each with its own kernel
# params (lengthscale matched to the shape: long for smooth, short for wiggly/sharp).
GT_SCENARIOS = {
    # A — monotone saturating: logistic rise to a plateau (monotonic, nonlinear).
    "monotone": dict(
        hf=lambda x: 0.15 + 0.75 * sig((np.atleast_1d(x) - 4.5) / 1.1),
        lf=lambda x: 0.22 + 0.60 * sig((np.atleast_1d(x) - 5.2) / 1.3),
        ls=2.6, var=0.20, ls_d=2.6, var_d=0.05),
    # B — wiggly oscillatory: sum of two sines (non-monotone, high wiggle).
    "wiggly": dict(
        hf=lambda x: 0.50 + 0.22 * np.sin(1.25 * np.atleast_1d(x))
                          + 0.12 * np.sin(2.60 * np.atleast_1d(x) + 0.6),
        lf=lambda x: 0.46 + 0.18 * np.sin(1.25 * np.atleast_1d(x) + 0.3)
                          + 0.10 * np.sin(2.60 * np.atleast_1d(x) + 0.1),
        ls=0.9, var=0.18, ls_d=1.1, var_d=0.05),
    # C — sharp localized: narrow Gaussian peak + late step (strong nonlinearity).
    "sharp": dict(
        hf=lambda x: 0.18 + 0.85 * np.exp(-((np.atleast_1d(x) - 3.4) / 0.8) ** 2)
                          + 0.16 * sig((np.atleast_1d(x) - 7.0) / 0.5),
        lf=lambda x: 0.22 + 0.68 * np.exp(-((np.atleast_1d(x) - 3.7) / 0.9) ** 2)
                          + 0.13 * sig((np.atleast_1d(x) - 7.3) / 0.6),
        ls=0.85, var=0.20, ls_d=1.0, var_d=0.06),
}

NOISE_HF2, NOISE_LF2 = 0.04, 0.06

# Denser data to resolve the features; human still fewer than animal.
X_lf2 = np.linspace(0.4, 9.6, 16)
X_hf2 = np.linspace(0.5, 9.3, 7)

for tag, sc in GT_SCENARIOS.items():
    g_hf, g_lf = sc["hf"], sc["lf"]
    ls, var, ls_d, var_d = sc["ls"], sc["var"], sc["ls_d"], sc["var_d"]

    y_lf2 = g_lf(X_lf2) + np.random.normal(0, NOISE_LF2, len(X_lf2))
    y_hf2 = g_hf(X_hf2) + np.random.normal(0, NOISE_HF2, len(X_hf2))

    # Flat constant informed prior mean per species (= observation mean).
    c_lf = np.full(len(X_te), y_lf2.mean())
    c_hf = np.full(len(X_te), y_hf2.mean())

    lf_mean, lf_var = gp_posterior(
        X_lf2, y_lf2, X_te, NOISE_LF2 ** 2,
        np.full(len(X_lf2), y_lf2.mean()), c_lf, ls, var)

    hf_mean, hf_var = mf_hf_posterior(
        X_lf2, y_lf2, X_hf2, y_hf2, X_te,
        NOISE_LF2 ** 2, NOISE_HF2 ** 2, RHO,
        np.full(len(X_lf2), y_lf2.mean()), np.full(len(X_hf2), y_hf2.mean()),
        c_lf, c_hf, ls_lf=ls, var_lf=var, ls_d=ls_d, var_d=var_d)

    # Per-scenario shared y-limits (HF & LF of this pair).
    lo2 = min(np.min(hf_mean - 2 * np.sqrt(hf_var)),
              np.min(lf_mean - 2 * np.sqrt(lf_var)))
    hi2 = max(np.max(hf_mean + 2 * np.sqrt(hf_var)),
              np.max(lf_mean + 2 * np.sqrt(lf_var)))
    pad2 = 0.06 * (hi2 - lo2)
    ylim2 = (lo2 - pad2, hi2 + pad2)

    make_plot(f"tl_demo_{tag}_human",  hf_mean, hf_var, g_hf, HF_COLOR,
              X_hf2, y_hf2, ylim=ylim2)
    make_plot(f"tl_demo_{tag}_animal", lf_mean, lf_var, g_lf, LF_COLOR,
              X_lf2, y_lf2, ylim=ylim2)
