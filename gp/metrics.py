"""Scoring rules for sample-based predictive distributions.

The GP produces posterior *samples*, not closed-form predictives, so every
metric here is computed from a sample vector rather than a mean and variance.
The scored quantity is a maximum over concentration, whose posterior is skewed
even when the underlying GP is Gaussian.
"""

import numpy as np
import torch


def compute_crps(samples, observation):
    """Continuous Ranked Probability Score for samples against one observation.

    Uses the energy-score identity for a sample-based forecast,

    .. math::

        \\mathrm{CRPS} = \\mathbb{E}|X - y| - \\tfrac{1}{2}\\mathbb{E}|X - X'|

    where the first term punishes error and the second rewards confidence.
    Lower is better, in the units of the observation.

    The second term is evaluated in closed form from the sorted samples rather
    than by sampling pairs:
    :math:`\\mathbb{E}|X - X'| = \\frac{2}{n^2}\\sum_i (2i - n - 1) x_{(i)}`.

    Args:
        samples: 1-D tensor of posterior draws.
        observation: The realised value. NaN propagates.

    Returns:
        CRPS as a float, or NaN if ``observation`` is NaN.
    """
    if observation is None or (isinstance(observation, float) and np.isnan(observation)):
        return float("nan")
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples, dtype=torch.float32)

    n = samples.numel()
    sorted_samples, _ = torch.sort(samples)
    obs = torch.tensor(observation, dtype=sorted_samples.dtype)

    mean_absolute_error = torch.mean(torch.abs(sorted_samples - obs)).item()
    ranks = torch.arange(1, n + 1, dtype=sorted_samples.dtype)
    expected_spread = (2.0 / (n * n)) * torch.sum(
        (2 * ranks - n - 1) * sorted_samples
    ).item()

    return mean_absolute_error - 0.5 * expected_spread


def compute_mae(samples, observation):
    """Expected absolute error under the posterior: ``E[|X - y|]``.

    Averaged over the posterior samples rather than evaluated at a point
    estimate, so it charges for spread as well as location. The first term of
    the CRPS decomposition; :func:`compute_sharpness` is the second.
    """
    if observation is None or (isinstance(observation, float) and np.isnan(observation)):
        return float("nan")
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples, dtype=torch.float32)
    target = torch.as_tensor(float(observation), dtype=samples.dtype)
    return torch.mean(torch.abs(samples - target)).item()


def compute_sharpness(samples):
    """Expected absolute difference between two draws, halved.

    .. math::  \\tfrac{1}{2}\\,\\mathbb{E}|X - X'|

    The second term of :func:`compute_crps` on its own: how wide the forecast
    is, with no reference to the truth.
    """
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples, dtype=torch.float32)
    n = samples.numel()
    sorted_samples, _ = torch.sort(samples)
    ranks = torch.arange(1, n + 1, dtype=sorted_samples.dtype)
    expected_spread = (2.0 / (n * n)) * torch.sum(
        (2 * ranks - n - 1) * sorted_samples
    ).item()
    return 0.5 * expected_spread


def exceedance_probability(samples, threshold):
    """Posterior probability that the quantity exceeds ``threshold``.

    The fraction of posterior draws above the threshold at which an APD
    prolongation is considered a proarrhythmic signal.
    """
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples, dtype=torch.float32)
    return (samples >= threshold).float().mean().item()


def exceedance_probability_ci(samples, threshold, alpha=0.05,
                              num_bootstrap=2000):
    """Exceedance probability with a bootstrap interval on it.

    Describes uncertainty in the probability itself given finitely many
    posterior draws, unlike :func:`quantile_interval`, which describes spread
    across repeats. Pass the draws pooled across repeats.

    Returns:
        ``(mean, lower_bound, upper_bound)``.
    """
    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples, dtype=torch.float32)
    n = samples.numel()
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    probabilities = np.empty(num_bootstrap)
    for i in range(num_bootstrap):
        resample = samples[torch.randint(0, n, (n,))]
        probabilities[i] = (resample > threshold).float().mean().item()
    return (
        float(probabilities.mean()),
        float(np.quantile(probabilities, alpha / 2)),
        float(np.quantile(probabilities, 1 - alpha / 2)),
    )


def quantile_interval(values, lower=0.025, upper=0.975):
    """Median and central interval of ``values`` across repeats.

    Quantiles of the repeat distribution itself, **not** a confidence interval
    on the median: how much the score varies from one draw of subjects to the
    next. Much wider than a bootstrap interval on the median.

    Returns:
        ``(median, lower_bound, upper_bound)``, all NaN if nothing is finite.
    """
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.quantile(values, 0.5)),
        float(np.quantile(values, lower)),
        float(np.quantile(values, upper)),
    )
