---
layout: home

hero:
  name: BayesianRegressionModels.jl
  text: brms-style formula DSL on top of Stan
  tagline: Compose linear-predictor formulas, custom priors, and likelihoods; transpile to Stan via StanBlocks; fit via Pathfinder / NUTS.
  actions:
    - theme: brand
      text: Gallery
      link: /gallery
    - theme: alt
      text: API
      link: /api
    - theme: alt
      text: GitHub
      link: https://github.com/JuliaBayes/BayesianRegressionModels.jl
---

# BayesianRegressionModels.jl

A formula DSL for Bayesian regression. The macro [`@brm`](@ref) parses
brms-style syntax (`y ~ 1 + a + (1 | g)`, `log(err) ~ 1 + b`,
`y ~ Normal(loc, err)`) into a [`BRMI`](@ref) intermediate
representation, then forks into one of two backends:

- [`VBRMI`](@ref) — pure-Julia, vectorised, `LogDensityProblems`-compatible.
- [`SBBRMI`](@ref) — StanBlocks → Stan source, fit via Pathfinder or
  full warmup HMC (`WarmupHMC.adaptive_warmup_mcmc`).

```julia
using BayesianRegressionModels, DataFrames

brmi = @brm df begin
    y ~ Normal(loc, err)
    loc ~ 1 + age + sex + (1 + age | subj)
    err ~ Exponential(1)
end

sbbrmi = SBBRMI(brmi)
src    = stan_code(sbbrmi)
```

## Truncation and censoring (Stan backend)

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

See the [Gallery](/gallery) for live, interactive examples — input
formula, the SLIC submodel body, the transpiled Stan source, and the
auto-generated posterior-predictive check, all in one card. The
[API](/api) page lists every public binding.
