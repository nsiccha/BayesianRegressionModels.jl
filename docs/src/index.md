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

Both slots are mandatory, and `:` is the wildcard for either.
`effect(:, weight) ~ Normal(0, 0.1)` is the *default layer* for `:weight` — it
reaches that column in every predictor owning it, and a more specific address
such as `effect(log_ka, weight)` overrides it. Two addresses of equal
specificity reaching one parameter, and unknown addresses, error. The first shipped lowering supports
`Normal(location, scale)` overrides and retains the existing
`pop_<predictor>_beta_pop` vector parameter, its `popcoefnames` labels, and
descriptor provenance. Inspect the captured formula statements with
`effect_priors(brmi)`. This surface belongs to `SBBRMI`; `VBRMI` does not
implement it.

### Categorical contrasts

A categorical predictor — a bare integer/`CategoricalArray` column, or one
wrapped in `factor(...)` — is *not* a `beta_pop` column, so `popcoefnames`
deliberately never lists it: it owns a separate `cat_<column>_beta` vector
holding its K−1 treatment contrasts, with the reference level pinned at 0.
Those contrasts also default to `std_normal()`, and the same `effect(...)`
address changes them — keyed by the **column** name, not the emitted
`cat_<column>` parameter name:

```julia
m = @brm df begin
    mu ~ 1 + factor(g) + x
    effect(mu, g) ~ Normal(0.0, 0.5)   # ⇒ cat_g_beta ~ normal(0.0, 0.5);
    y ~ Normal(mu, sigma)
end
```

One shared `(location, scale)` covers every contrast in the block; per-level
scales are not expressible here. The `:`-predictor form `effect(:, g)` reaches
the same block in every predictor owning it, and the statement composes with population overrides on the same
predictor (`effect(mu, x) ~ Normal(0, 0.25)`) — each addresses its own
parameter. A non-default reference level emits `cat_<column>__ref_<k>_beta`,
which the plain column name still addresses whenever that is unambiguous;
when two `factor(g; ref=…)` blocks of one column would both claim it, the
bare address is refused and each block is addressed by its exact emitted
name. Models with no such statement emit byte-identical Stan to before.

### Term-internal parameters

Some terms own parameters no coefficient address can reach — `s(x)`'s smoothing
scale, `mo(c)`'s Dirichlet increments, `me(x, sd)`'s latent true covariate. They
are addressed by naming the term itself in the target slot, under the same
head-position grammar:

```julia
m = @brm df begin
    y ~ Normal(mu, 1.)
    mu ~ 1 + s(age) + mo(dose) + me(w_obs, 0.3)

    sd(:, s(age))         ~ Exponential(2)   # smoothing SD
    simplex(mu, mo(dose)) ~ Dirichlet(2)     # monotonic increments
    latent(:, me(w_obs))  ~ Normal(0, 5)     # latent true covariate
end
```

The term is spelled the way the formula spells it, minus numeric and keyword
arguments — `me(w_obs, 0.3)` is addressed as `me(w_obs)`. `term_priors(brmi)`
returns the captured statements. See
[Term-internal priors](@ref) for the full table, the `t2` component slot, and
why the standardized raw innovations are deliberately not configurable.

See [Formula terms](@ref) and [Likelihoods](@ref) for the supported syntax and
backend-specific contracts. The [Gallery](/gallery) provides live, interactive examples — input
formula, the SLIC submodel body, the transpiled Stan source, and the
auto-generated posterior-predictive check, all in one card. The
[API](/api) page lists every public binding.
