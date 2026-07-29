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
