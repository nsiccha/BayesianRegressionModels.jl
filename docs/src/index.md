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

Population coefficients use independent standard-normal priors by default.
Override selected coefficients with separate `effect(...)` statements; the
coefficient names are exactly those returned by `popcoefnames`:

```julia
pk = @brm df begin
    log_ka ~ 1 + weight + (1 | pk | subject)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, weight) ~ Normal(0, 0.1)
end
```

The explicit two-argument address is recommended when a model has several
linear predictors. `effect(weight) ~ Normal(0, 0.1)` is a convenience only
when `:weight` names a population coefficient in exactly one predictor;
ambiguous and unknown addresses error. The first shipped lowering supports
`Normal(location, scale)` overrides and retains the existing
`pop_<predictor>_beta_pop` vector parameter, its `popcoefnames` labels, and
descriptor provenance. Inspect the captured formula statements with
`effect_priors(brmi)`. This surface belongs to `SBBRMI`; `VBRMI` does not
implement it.

See [Formula terms](@ref) and [Likelihoods](@ref) for the supported syntax and
backend-specific contracts. The [Gallery](/gallery) provides live, interactive examples — input
formula, the SLIC submodel body, the transpiled Stan source, and the
auto-generated posterior-predictive check, all in one card. The
[API](/api) page lists every public binding.
