# Likelihoods

## Truncation and censoring

BRM preserves the standard Distributions.jl RHS composition for mathematical
truncation and threshold censoring:

```julia
model = @brm begin
    mu ~ 1 + x
    log(sigma) ~ 1

    y_truncated ~ truncated(Normal(mu, sigma); lower=0.0, upper=2.0)
    y_clamped ~ censored(LogNormal(mu, sigma); lower=0.25, upper=1.8)

    # Genuine interval evidence: y_lower stores the open lower endpoint and
    # y_upper stores the closed upper endpoint.
    y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
end
```

These are three different likelihood contracts:

- `truncated(d; lower, upper)` conditions `d` on the inclusive bounds and
  predicts from that conditional distribution;
- `censored(d; lower, upper)` is the distribution of
  `clamp(X, lower, upper)` and predicts clamped values;
- `interval_censored(d; upper)` contributes
  `log(CDF(upper) - CDF(response))` for the genuine interval observation
  `(response, upper]`, while prediction remains on the uncoarsened base scale.

Bounds may be numeric literals or observed row-wise columns. The initial
family-gated surface covers `Normal`, `LogNormal`, `Exponential`, `Weibull`,
and `Poisson`; BRM rejects other base families until their aggregate density,
pointwise likelihood, CDF/CCDF, generated prediction, and stanc paths are all
tested. This composition is currently implemented by [`SBBRMI`](@ref), not
[`VBRMI`](@ref). The legacy `TruncatedNormal` Bordet marker remains a separate
censored-Normal compatibility surface.

## Concise categorical regression

`CategoricalLogit` accepts an explicit nested `@brm(...)` predictor formula:

```julia
data = (;
    x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y = ["b", "a", "c", "b", "c", "a"],
)

categorical_model = @brm data begin
    y ~ CategoricalLogit(@brm(1 + x))
end
```

For an outcome with ``K`` levels, BRM expands the marked formula to ``K-1``
ordinary scalar linear predictors with distinct coefficients, then calls the
same reference-class categorical lowering as the fully explicit form. The
first fitted level has logit zero. Plain vectors use `sort(unique(y))` for the
fitted order; a `CategoricalVector` uses its declared level order. The latter is
the way to select a reference level deliberately.

For example, a three-level outcome above is equivalent in model structure to:

```julia
explicit_model = @brm data begin
    y_nested_arg1_class2 ~ 1 + x
    y_nested_arg1_class3 ~ 1 + x
    y ~ CategoricalLogit(y_nested_arg1_class2, y_nested_arg1_class3)
end
```

The generated names are deterministic implementation names; use the explicit
form when those predictor names are part of another formula. Only nested
`@brm(...)` opts into predictor-formula interpretation. Thus
`CategoricalLogit(1 + x)` remains an ordinary expression and is rejected by
the categorical backend, rather than silently acquiring coefficients.

The marker is not categorical-specific. It selects formula interpretation at
one family-argument position while surrounding expressions retain their usual
meaning. For example, a distributional Normal model can make both predictors
concise while keeping the positive scale link explicit:

```julia
distributional_model = @brm data begin
    y ~ Normal(@brm(1 + x), exp(@brm(1)))
end
```

This introduces distinct scalar predictors for location and log-scale, then
passes `exp(log_scale)` to `Normal`; nested `@brm` never inserts a link. A
standalone fragment such as `@brm(1 + x)` is not yet a first-class value and
produces a targeted error outside an enclosing model.

This is the same broad model structure expressed by a top-level categorical
formula in brms (`y ~ 1 + x`, `family = categorical(link = "logit")`) or Bambi
(`"y ~ 1 + x"`, `family="categorical"`). Defaults for priors, contrasts, and
reference-level selection are package-specific; BRM does not import those
defaults implicitly.

## Ordinal structure and link composition

BRM treats the ordinal probability construction and inverse link as separate
typed choices. A cumulative probit model is:

```julia
ordinal_model = @brm data begin
    eta ~ 0 + x
    y ~ Ordinal(Cumulative(), ProbitLink(), eta)
end
```

The accepted structures are `Cumulative()` and `StoppingRatio()`. Each composes
with `LogitLink()`, `ProbitLink()`, or `CloglogLink()`. This is intentionally a
Julia-native typed surface: BRM does not copy R formula helper names or encode
every structure/link pair in a new family type.

For `Cumulative()`, BRM estimates strictly ordered thresholds ``c_1 < \cdots <
c_{K-1}`` and uses

```math
P(Y \le k) = F\!\left(d(c_k - \eta)\right).
```

For `StoppingRatio()`, the estimated stage intercepts need not be ordered and

```math
q_k = P(Y=k \mid Y\ge k)
    = F\!\left(d(c_k - \eta_k)\right),\qquad
P(Y=k)=q_k\prod_{j<k}(1-q_j),
```

with the final category equal to the probability of continuing through every
stage. `F` is logistic, standard normal, or complementary-log-log according to
the link tag. Both threshold vectors currently receive element-wise standard
normal priors.

The thresholds already supply the model location, so the composed surface
requires an intercept-free common predictor (`eta ~ 0 + ...`). A positive
discrimination parameter is explicit:

```julia
ordinal_disc = @brm data begin
    eta ~ 0 + x
    log(disc) ~ 0 + group
    y ~ Ordinal(Cumulative(), ProbitLink(), eta;
                discrimination=disc)
end
```

`discrimination` defaults to one. Literal or data-supplied values are checked
for finiteness and strict positivity; a modeled value should use a
positive-support prior or a link such as `log(disc) ~ ...`.

Stopping-ratio models may add non-proportional effects with a tuple of raw
numeric predictors:

```julia
sequential = @brm data begin
    eta ~ 0 + period + carry
    y ~ Ordinal(StoppingRatio(), CloglogLink(), eta;
                per_threshold=(treat,))
end
```

BRM estimates one coefficient per predictor and non-terminal stage, so here
``\eta_k = \eta + \mathtt{treat}\,\beta_k``. `per_threshold` is deliberately
restricted to stopping-ratio models for now: unrestricted cumulative
category-specific effects can make cumulative probabilities non-monotone.
The predictors must currently be raw numeric data columns.

Outcome categories follow the declared order of a `CategoricalVector`; plain
vectors use sorted unique values. That fitted order is frozen for replay and
prediction. The legacy `OrderedLogistic(eta)` spelling remains supported and
continues to lower directly to Stan's native ordered-logistic distribution.
The composed cumulative-logit kernel also delegates its scalar density to that
native primitive; the other links use Stan's native stable CDF/log-CDF
functions. Stopping ratio has no native Stan distribution, so BRM supplies the
matching stable lpmf, pointwise log-likelihood, and RNG.

Outside `@brm`, `Ordinal(structure, link, eta, thresholds;
discrimination=1)` is an executable `DiscreteUnivariateDistribution` with
`params`, `probs`, `logpdf`, and `rand`. A stopping-ratio `eta` may be scalar or
a vector with one stage-specific value per threshold.

For neutral comparison, brms exposes the same statistical axes through
families such as cumulative-probit and stopping-ratio complementary-log-log,
and calls threshold-varying terms category-specific effects. BRM preserves
that statistical contract while using the typed composition and
`per_threshold=(...)` tuple above rather than importing brms's R formula
helpers.

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

## Quantile regression with `SkewDoubleExponential`

For a non-median quantile, BRM exposes the executable distribution
`SkewDoubleExponential(mu, sigma, tau)`. Its arguments and scale exactly match
Stan's native `skew_double_exponential` family:

```math
f(y \mid \mu, \sigma, \tau)
= \frac{2\tau(1-\tau)}{\sigma}
  \exp\!\left[-\frac{2}{\sigma}
  \left((1-\tau)\mathbf{1}_{y<\mu}(\mu-y)
        +\tau\mathbf{1}_{y>\mu}(y-\mu)\right)\right].
```

Thus `cdf(SkewDoubleExponential(mu, sigma, tau), mu) == tau`, and
`SkewDoubleExponential(mu, sigma, 0.5)` is exactly `Laplace(mu, sigma)`.
There is no hidden brms-scale conversion on this primary Julia surface.

```julia
quantile_model = @brm data begin
    q25_y ~ 1 + x
    log(native_scale) ~ 1
    y ~ SkewDoubleExponential(q25_y, native_scale, 0.25)
end
```

The brms/check-loss scale ``s`` translates explicitly as
``\sigma = 2s``. Distributions.jl's existing exact special case also remains
available in formulas:

```julia
using Distributions: SkewedExponentialPower

y ~ SkewedExponentialPower(mu, sigma_sepd, 1, tau)
```

BRM lowers that spelling with
``\sigma = 4\,\mathtt{sigma\_sepd}\,\tau(1-\tau)``. The shape must be the
literal value `1`; other `SkewedExponentialPower` shapes are rejected because
Stan's asymmetric double-exponential family is not a native implementation of
the general SEPD. Density, pointwise log likelihood, and predictive RNG all
use the same translation.

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
