# Likelihoods

## Median regression with `Laplace`

The StanBlocks backend accepts `Distributions.Laplace` as an ordinary
likelihood. For example, a robust regression for the conditional median can be
written as:

```julia
using BayesianRegressionModels
using Distributions: Laplace

data = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0],
    y = [-1.1, -0.2, 0.1, 0.6, 1.4],
)

median_model = @brm data begin
    median_y ~ 1 + x
    log(laplace_scale) ~ 1
    y ~ Laplace(median_y, laplace_scale)
end

stan = stan_code(SBBRMI(median_model))
```

This lowers to Stan's native
`y ~ double_exponential(median_y, laplace_scale)`. The second argument is a
**Laplace scale**, not a standard deviation or rate:

```math
f(y \mid \mu, \theta)
= \frac{1}{2\theta}\exp\!\left(-\frac{|y-\mu|}{\theta}\right),
\qquad \theta = \mathtt{laplace\_scale}.
```

This symmetric likelihood is exactly the ``q = 0.5`` special case of the
asymmetric-Laplace likelihood used for quantile regression, after accounting
for parameterization:

- In the check-loss convention used by `brms`, with scale ``s`` and density
  ``q(1-q)s^{-1}\exp[-\rho_q((y-\mu)/s)]``, use
  ``\theta = 2s`` at ``q = 0.5``.
- In the Bambi/PyMC convention
  `AsymmetricLaplace(mu, b, kappa)`, where
  ``\kappa = \sqrt{q/(1-q)}``, use ``\kappa = 1`` and
  ``\theta = 1/b`` at ``q = 0.5``.

The `Laplace` spelling covers median regression only. It does **not** express
an asymmetric likelihood for ``q \ne 0.5``. It also does not translate
Bambi's historical `bs(age, knots=...)` term: that basis mapping is a separate
unresolved formula-semantic question. Thus the response-family component of
the catalogue's `quantile_p50` model is available, while the complete
historical model remains unsupported.

## Circular regression with `VonMises`

BRM exposes two deliberately different von-Mises likelihoods. Use
Distributions.jl's `VonMises` when its exact Julia contract is intended:

```julia
using BayesianRegressionModels
using Distributions: VonMises

exact_model = @brm data begin
    mu ~ 1 + x
    log(kappa) ~ 1
    direction ~ VonMises(mu, kappa)
end
```

This preserves the constructor order `(mu, kappa)`, the shorthand
`VonMises(kappa) == VonMises(0, kappa)`, strict `kappa > 0`, and the moving
closed support `[mu - pi, mu + pi]`. The backend adds those support/domain
guards around Stan's native `von_mises_lpdf`, and recenters native predictive
draws onto the same moving interval. Because the support moves with `mu`, this
surface is usually not the right choice for observations encoded once on a
fixed principal interval.

For conventional circular regression on a fixed interval, use the distinct
BRM distribution `CircularVonMises`:

```julia
circular_model = @brm data begin
    mu ~ 1 + x
    log(kappa) ~ 1
    direction ~ CircularVonMises(mu, kappa; interval=(-pi, pi))
end
```

`interval` is a compile-time pair of finite numbers with length `2pi`; it
defaults to `(-pi, pi)`. Observations must lie in the half-open interval
`[lo, hi)`. BRM wraps `mu` and generated draws into that interval, while the
density itself remains Stan's native `von_mises_lpdf`. Both arguments are
ordinary distributional parameters: BRM supplies no implicit link or prior.
Outside a formula, `CircularVonMises(mu, kappa; interval=...)` is an executable
`ContinuousUnivariateDistribution`: `params`, `logpdf`, and `rand` preserve the
same fixed-interval contract used by the Stan lowering.

For comparison, `brms` uses a fixed `(-pi, pi)` response convention and makes
both `mu` and `kappa` distributional, with default `tan_half` and `log` links.
Those are a useful neutral baseline, but BRM requires links and priors to be
written explicitly rather than silently changing Distributions.jl semantics.
In particular, `Distributions.Gamma` takes a **scale**, whereas Stan/brms gamma
syntax takes a **rate**: the brms prior `gamma(2, 0.01)` is spelled
`Gamma(2, 100.0)` in a BRM formula.
