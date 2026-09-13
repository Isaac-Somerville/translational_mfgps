"""The multi-fidelity GP used throughout this work, and its noise model.

The model
---------
:class:`MultitaskGPModel` is a Linear Model of Coregionalisation:

.. math::

    k\\big((x, s), (x', s')\\big) = \\sum_{q=1}^{Q} k_q(x, x')\\, B_q(s, s')

Each latent process ``q`` contributes a continuous RBF kernel over channel
block, ``k_q``, multiplied by a species-covariance matrix ``B_q`` of rank
``covar_rank``. The ``B_q`` are learned, so how strongly a species informs
another is inferred rather than imposed. ``Q = 1`` recovers the separable ICM
model.

One structural choice:

**Boundary constraint.** At zero concentration the relative APD change is zero
by definition. :class:`BoundaryConstrainedKernel` multiplies the kernel by
:math:`\\phi(x)^\\top \\phi(x')` with :math:`\\phi(0) = 0`, forcing the prior
variance to vanish at the origin and hence :math:`f(0) = 0` exactly.

Noise
-----
:class:`ObservationNoise` supplies the observation variance. Homoskedastic noise
is one constant per species. Heteroskedastic noise grows with distance from the
origin,

.. math::

    \\sigma(x, s) = \\sigma_{\\min}
        + (\\sigma_{\\max,s} - \\sigma_{\\min})\\left(1 - e^{-(\\lVert x \\rVert/\\rho_s)^2}\\right)

pinned at :math:`\\sigma_{\\min}` at zero block.
"""

import gpytorch
import torch
import torch.nn.functional as F
from linear_operator import to_linear_operator


def inv_softplus(y):
    """Inverse of ``softplus``, for initialising a positively-constrained raw parameter."""
    y = torch.as_tensor(y)
    return y + torch.log(-torch.expm1(-y))


def make_rbf_kernel(ard_num_dims, active_dims, boundary_constraint=None, ls_prior=None):
    """Build the continuous kernel over channel block for one latent process.

    Args:
        ard_num_dims: None for a single shared lengthscale across block
            dimensions; anything else for full ARD, one lengthscale per active
            dimension. Only whether it is None matters — the number of
            lengthscales is ``len(active_dims)`` either way.
        active_dims: Column indices of ``x_block`` this kernel sees.
        boundary_constraint: ``"poly"`` wraps the kernel in
            :class:`BoundaryConstrainedKernel` so that ``f(0) = 0``; None leaves
            it unconstrained.
        ls_prior: ``{"loc", "scale"}`` for a LogNormal prior on the lengthscale,
            making the fit MAP rather than maximum likelihood.
    """
    if boundary_constraint is not None and boundary_constraint != "poly":
        raise ValueError(
            f"Unsupported boundary_constraint {boundary_constraint!r}; only "
            f"'poly' is used."
        )

    # With the boundary wrapper, active_dims moves onto the wrapper so that phi
    # and the base kernel are handed the same sliced input.
    base = gpytorch.kernels.RBFKernel(
        ard_num_dims=len(active_dims) if ard_num_dims is not None else None,
        active_dims=None if boundary_constraint is not None else active_dims,
    )
    if ls_prior is not None:
        base.register_prior(
            "lengthscale_prior",
            gpytorch.priors.LogNormalPrior(ls_prior["loc"], ls_prior["scale"]),
            "lengthscale",
        )

    if boundary_constraint is not None:
        return BoundaryConstrainedKernel(base, active_dims=active_dims)
    return base


class BoundaryConstrainedKernel(gpytorch.kernels.Kernel):
    """Wrap a kernel so that the GP is exactly zero at the origin.

    .. math::

        k_{\\text{new}}(x, x') = \\phi(x)^\\top \\phi(x')\\; k_{\\text{base}}(x, x')

    The dot product is a scalar PSD scaling, so the result is still a valid
    kernel. With :math:`\\phi(0) = 0` the prior variance at the origin is zero,
    forcing :math:`f(0) = 0`.

    The only form used here is :math:`\\phi(x) = x`, applied per dimension. The
    species index is never part of ``x`` by the time the kernel sees it — the
    model splits it off and ``active_dims`` selects the block columns — so phi
    acts on the whole input.

    Args:
        base_kernel: The kernel to wrap.
    """

    has_lengthscale = False

    def __init__(self, base_kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        # phi is the identity, so x1 and x2 are phi(x1) and phi(x2).
        if diag:
            scaling_diag = (x1 * x2).sum(dim=-1)
            K_diag = self.base_kernel(x1, x2, diag=True, **params)
            if not isinstance(K_diag, torch.Tensor):
                K_diag = K_diag.to_dense()
            return scaling_diag * K_diag
        K = self.base_kernel(x1, x2, **params).to_dense()
        return (x1 @ x2.transpose(-1, -2)) * K


class MultitaskGPModel(gpytorch.models.ExactGP):
    """Linear Model of Coregionalisation over species.

    See the module docstring for the model. Inputs are ``(N, D+1)``: ``D``
    columns of channel block, then a species index in the last column.

    Args:
        train_x, train_y, likelihood: As for any
            :class:`gpytorch.models.ExactGP`. The likelihood is normally a
            :class:`~gpytorch.likelihoods.FixedNoiseGaussianLikelihood` fed by
            :class:`ObservationNoise`.
        num_species: Number of species. **The high-fidelity species must be
            last**; several callers rely on index ``num_species - 1``.
        ard_num_dims: See :func:`make_rbf_kernel`.
        channel_mask: ``(num_species - 1, D)`` tensor, non-zero where that
            species' cell model represents the channel. Recorded as a buffer so
            that a checkpoint says which channels the fit was given; **it does
            not enter the kernel**. Every species' kernel sees every block
            column.
        mean_function: ``"constant"`` or ``"linear"``.
        covar_rank: Rank of each species-covariance matrix ``B_q``. Rank 1 makes
            every species a scaled copy of one shared latent.
        num_latents: Number of latent processes ``Q``.
        shared_mean: One mean function for all species, or one each.
        boundary_constraint: ``"poly"`` or None. See
            :class:`BoundaryConstrainedKernel`.
        ls_prior: LogNormal prior on the RBF lengthscale.
        index_kernel_prior: Priors on the coregionalisation parameters. Accepts
            ``scale`` (Normal prior on ``covar_factor``) and/or
            ``var_loc``/``var_scale`` (LogNormal on ``var``). The latter two
            must be given together.
        use_signed_mean: For a linear mean, use the signed first block
            dimension rather than the norm, which cannot express APD shortening.
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        num_species,
        ard_num_dims=None,
        channel_mask=None,
        mean_function="constant",
        covar_rank=1,
        num_latents=1,
        shared_mean=False,
        boundary_constraint=None,
        ls_prior=None,
        index_kernel_prior=None,
        use_signed_mean=False,
    ):
        super().__init__(train_x, train_y, likelihood)

        self.num_species = num_species
        self.num_latents = num_latents
        self.shared_mean = shared_mean
        self.mean_function = mean_function
        self.use_signed_mean = use_signed_mean

        # Every column but the trailing species index is channel block, and
        # every kernel sees all of them.
        block_active_dims = list(range(train_x.shape[1] - 1))

        # Kept for provenance and so that published checkpoints round-trip; see
        # the class docstring.
        self.register_buffer("channel_mask", channel_mask)
        self._build_means(mean_function, shared_mean, boundary_constraint)
        self._build_kernels(
            num_latents, ard_num_dims, block_active_dims, boundary_constraint,
            ls_prior, covar_rank, num_species, index_kernel_prior,
        )

    def _build_means(self, mean_function, shared_mean, boundary_constraint):
        """Constant or linear mean, shared across species or one per species.

        Under a boundary constraint a constant mean becomes a zero mean; a
        linear mean without bias already vanishes at the origin.
        """
        if mean_function not in ("constant", "linear"):
            raise ValueError(f"Unsupported mean_function {mean_function!r}")

        def one_mean():
            if mean_function == "constant":
                return (
                    gpytorch.means.ZeroMean()
                    if boundary_constraint is not None
                    else gpytorch.means.ConstantMean()
                )
            return gpytorch.means.LinearMean(input_size=1, bias=False)

        if shared_mean:
            self.mean_module = one_mean()
        else:
            self.mean_modules = torch.nn.ModuleList(
                [one_mean() for _ in range(self.num_species)]
            )

    def _build_kernels(self, num_latents, ard_num_dims, block_active_dims,
                       boundary_constraint, ls_prior, covar_rank, num_species,
                       index_kernel_prior):
        """One RBF kernel and one species-covariance matrix per latent process."""
        self.rbf_kernels = torch.nn.ModuleList(
            [
                make_rbf_kernel(
                    ard_num_dims, block_active_dims, boundary_constraint, ls_prior
                )
                for _ in range(num_latents)
            ]
        )
        self.index_kernels = torch.nn.ModuleList(
            [
                gpytorch.kernels.IndexKernel(num_tasks=num_species, rank=covar_rank)
                for _ in range(num_latents)
            ]
        )

        if index_kernel_prior is None:
            return
        has_var_loc = "var_loc" in index_kernel_prior
        has_var_scale = "var_scale" in index_kernel_prior
        if has_var_loc != has_var_scale:
            raise ValueError(
                "index_kernel_prior: 'var_loc' and 'var_scale' must be given "
                "together, or neither."
            )
        factor_prior = (
            gpytorch.priors.NormalPrior(0.0, index_kernel_prior["scale"])
            if "scale" in index_kernel_prior
            else None
        )
        var_prior = (
            gpytorch.priors.LogNormalPrior(
                index_kernel_prior["var_loc"], index_kernel_prior["var_scale"]
            )
            if has_var_loc
            else None
        )
        for kernel in self.index_kernels:
            if factor_prior is not None:
                kernel.register_prior("covar_factor_prior", factor_prior, "covar_factor")
            if var_prior is not None:
                kernel.register_prior("var_prior", var_prior, "var")

    def _mean_input(self, x_block):
        """Reduce the block vector to the scalar a linear mean acts on.

        The signed first dimension can express APD shortening; the norm cannot.
        """
        if self.use_signed_mean:
            return x_block[:, :1]
        return torch.norm(x_block, dim=-1, keepdim=True)

    def forward(self, x):
        x_block = x[:, :-1]
        species = x[:, -1].long()

        if x.shape[0] == 0:
            # No training data (the human-only comparison passes zero human
            # subjects). IndexKernel cannot handle a 0-row input, and the exact
            # GP posterior correctly reduces to the prior for an empty
            # conditioning set, so return a degenerate distribution.
            return gpytorch.distributions.MultivariateNormal(
                x.new_zeros(0), to_linear_operator(x.new_zeros(0, 0))
            )

        if self.shared_mean:
            mean_input = (
                self._mean_input(x_block)
                if self.mean_function == "linear"
                else x_block
            )
            mean_x = self.mean_module(mean_input)
        else:
            mean_input = (
                self._mean_input(x_block)
                if self.mean_function == "linear"
                else x_block
            )
            means = torch.stack([m(mean_input) for m in self.mean_modules], dim=1)
            mean_x = means[torch.arange(x.size(0)), species]

        covar = None
        for q in range(self.num_latents):
            k_q = self.rbf_kernels[q](x_block).mul(self.index_kernels[q](species))
            covar = k_q if covar is None else covar + k_q

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class ObservationNoise(torch.nn.Module):
    """Per-species observation noise, returning a variance for each point.

    ``forward`` returns :math:`\\sigma^2` per observation, ready to hand to
    :class:`~gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.

    ``noise_type="homoskedastic"``
        One constant per species, :math:`\\sigma_s`.

    ``noise_type="heteroskedastic"``
        Noise grows with the norm of the block vector, saturating at
        :math:`\\sigma_{\\max,s}`:

        .. math::

            \\sigma(x, s) = \\sigma_{\\min} + (\\sigma_{\\max,s} - \\sigma_{\\min})
                \\left(1 - e^{-(\\lVert x \\rVert / \\rho_s)^2}\\right)

        The noise floor :math:`\\sigma_{\\min}` applies at zero block.

    Args:
        num_species: Number of species.
        noise_type: ``"homoskedastic"`` or ``"heteroskedastic"``.
        init_sigma_min: Noise floor, fixed, heteroskedastic only.
        init_sigma_max: Initial saturating noise level, heteroskedastic only.
        init_rho: Initial noise length-scale, heteroskedastic only.
        shared_noise: Tie the noise magnitude across species to one parameter.
        rho_prior: ``{"loc", "scale"}`` LogNormal prior on ``rho``, contributing
            to :meth:`regularization_loss`.
    """

    def __init__(
        self,
        num_species,
        noise_type="heteroskedastic",
        init_sigma_min=1e-2,
        init_sigma_max=0.2,
        init_rho=0.1,
        shared_noise=False,
        rho_prior=None,
    ):
        super().__init__()
        if noise_type not in ("homoskedastic", "heteroskedastic"):
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        self.num_species = num_species
        self.noise_type = noise_type
        self.shared_noise = shared_noise
        self.rho_prior = rho_prior

        # A tied magnitude is backed by a single element rather than one per
        # species; forward() indexes at 0 in that case. rho is always per
        # species.
        self._n_sigma = 1 if shared_noise else num_species

        if noise_type == "homoskedastic":
            self.register_parameter(
                "raw_sigma", torch.nn.Parameter(torch.zeros(self._n_sigma))
            )
        else:
            self.register_parameter(
                "raw_sigma_max", torch.nn.Parameter(torch.zeros(self._n_sigma))
            )
            self.register_parameter(
                "raw_rho", torch.nn.Parameter(torch.zeros(num_species))
            )
            self.register_buffer("sigma_min", torch.tensor(init_sigma_min))
            self.sigma_max = init_sigma_max
            self.rho = init_rho

    @property
    def sigma(self):
        if self.noise_type == "homoskedastic":
            return F.softplus(self.raw_sigma)
        raise AttributeError("No `sigma` attribute for heteroskedastic noise.")

    @property
    def sigma_max(self):
        if self.noise_type == "heteroskedastic":
            return F.softplus(self.raw_sigma_max)
        raise AttributeError("No `sigma_max` attribute for homoskedastic noise.")

    @sigma_max.setter
    def sigma_max(self, value):
        if self.noise_type != "heteroskedastic":
            raise AttributeError("No `sigma_max` attribute for homoskedastic noise.")
        raw = inv_softplus(torch.as_tensor(value))
        if raw.ndim == 0:
            raw = torch.tensor([raw.item()] * self._n_sigma)
        self.raw_sigma_max.data = raw

    @property
    def rho(self):
        if self.noise_type == "heteroskedastic":
            return F.softplus(self.raw_rho)
        raise AttributeError("No `rho` attribute for homoskedastic noise.")

    @rho.setter
    def rho(self, value):
        if self.noise_type != "heteroskedastic":
            raise AttributeError("No `rho` attribute for homoskedastic noise.")
        raw = inv_softplus(torch.as_tensor(value))
        if raw.ndim == 0:
            raw = torch.tensor([raw.item()] * self.num_species)
        self.raw_rho.data = raw

    def forward(self, x):
        block_input = x[:, :-1]
        species = x[:, -1].long()
        idx_sigma = torch.zeros_like(species) if self._n_sigma == 1 else species

        if self.noise_type == "homoskedastic":
            sigma = self.sigma[idx_sigma]
        else:
            distance = torch.norm(block_input, dim=-1)
            sigma_max = self.sigma_max[idx_sigma]
            rho = self.rho[species]
            sigma = self.sigma_min + (sigma_max - self.sigma_min) * (
                1 - torch.exp(-((distance / rho) ** 2))
            )
        return sigma ** 2

    def regularization_loss(self):
        """Negative log prior on ``rho``, to be added to the training loss.

        Zero unless a heteroskedastic model was given a prior on ``rho``.
        """
        if self.noise_type == "heteroskedastic" and self.rho_prior is not None:
            log_prob = (
                gpytorch.priors.LogNormalPrior(
                    self.rho_prior["loc"], self.rho_prior["scale"]
                )
                .log_prob(self.rho)
                .sum()
            )
            return -log_prob
        return 0.0
