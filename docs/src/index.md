````@raw html
---
layout: home

hero:
  name: BayesianRegressionModels.jl
  text: A brms-shaped formula DSL for Julia
  tagline: Same formula grammar, but your columns do not all have to come from one equal-length data frame. Transpiles to Stan via StanBlocks; fits via Pathfinder / NUTS.
  actions:
    - theme: brand
      text: Gallery
      link: /gallery
    - theme: alt
      text: API
      link: /api
    - theme: alt
      text: GitHub
      link: https://github.com/nsiccha/BayesianRegressionModels.jl
---
````

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

## Coming from brms

BRM is deliberately brms-shaped, so most of what you know transfers. The
differences worth knowing up front are one structural gain and a genuinely
shorter feature list.

### What carries over

Formula grammar, random-effect syntax, and the two features people usually
reach brms for:

| brms | `@brm` |
|---|---|
| `y ~ 1 + age + sex` | `y ~ 1 + age + sex` |
| `(1 + age \| subj)` | `(1 + age \| subj)` |
| `(1 + age \|p\| subj)` — correlated across formulas | `(1 + age \|p\| subj)` |
| `(1 \| gr(subj, by = diagnosis))` | `(1 \| gr(subj, by = diagnosis))` |
| `bf(y1) + bf(y2)` | two `~` lines in the same block |
| **Distributional regression** — `bf(y ~ x, sigma ~ x)` | every distributional parameter is just another `~` line |
| **Nonlinear terms** — `bf(y ~ a * exp(-b * x), a ~ 1, b ~ 1, nl = TRUE)` | the same, with **no `nl` switch** |

Distributional regression needs no special form: give the parameter its own
formula, and apply the link yourself where you want one.

```julia
@brm df begin
    y        ~ Normal(mu, sigma)
    mu       ~ 1 + age
    log(sigma) ~ 1 + age          # explicit link, addressed as `sigma`
end
```

Nonlinear models need no `nl = TRUE` and no `nlf()`. A declaration is a named
value, so composing declarations into an arbitrary Julia expression is the
whole feature:

```julia
@brm df begin
    a     ~ 1 + (1 | g)           # ordinary linear predictors …
    b     ~ 1
    y     ~ Normal(a * exp(-b * x), sigma)   # … composed nonlinearly
    sigma ~ Exponential(1)
end
```

That lowers to `y ~ normal(a .* exp(-b .* x), sigma)` in the generated Stan,
and — as with any BRM response — you also get the pointwise log-likelihood
`y_likelihood` and predictive draws `y_gen` for free.

### Where BRM goes further

**Your data does not have to be one data frame with equal-length columns.**
This is the main structural difference. brms takes a single `data.frame`, so
every column shares one row axis. `@brm` takes any column collection — a
`NamedTuple` is fine — whose columns may live on *different* row axes, and
different linear predictors in one model may be defined on different axes:

```julia
@brm schedule begin
    sigma  ~ Exponential(1)
    log_F  ~ 1 + factor(vessel) + mo(diet)   # one row per dose EVENT
    log_CL ~ 1 + weight + (1 | pk | subject) # one row per SUBJECT

    pred ~ kernel(t_obs, y,
                  ragged(dose_amount, dose_subject),
                  ragged(log_F,       dose_subject),
                  log_CL) do ts, yy, doses, lF, lCL
        effective_dose = sum(doses .* exp(lF))
        mu = effective_dose .* exp(-exp(lCL) .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end
```

[`ragged`](@ref)`(x, group)` attaches a flat secondary frame to the grouping
axis, and [`kernel`](@ref) broadcasts a do-block *cell* — arbitrary Julia,
including a structural or ODE-like time course — over pre-grouped rows, taking
the per-group values of ordinary linear predictors as arguments. In brms this
class of model is `nlf()` plus manual `data2` bookkeeping, or Stan by hand.

Two smaller gains: [`VBRMI`](@ref) gives you a pure-Julia
`LogDensityProblems` object with no Stan toolchain in the loop, and
[`brm_descriptor`](@ref) exposes one reflectable description of everything the
model emits, so a consumer mounts a fitted model without keeping a parallel
registry of parameter names.

### Where BRM is behind

Not feature-complete against brms. The gaps a brms user is most likely to hit:

- **No residual correlation between responses.** Several responses in one
  block are modelled independently; there is no `set_rescor(TRUE)`.
- **No fixed / known covariance groups.** `by=` is the only [`gr`](@ref)
  option — no `cov=`, so no phylogenetic or pedigree random effects.
- **Autocorrelation is AR(1) only.** [`ar`](@ref)`(time; p=1)` is the entire
  surface: no MA, ARMA, compound symmetry, unstructured, CAR or SAR.
- **Fewer families.** [Likelihoods](@ref) is the complete list, and it is
  considerably shorter than brms'.

[Formula terms](@ref) has the full catalogue of what *is* supported.

## Configuring priors

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
scale, `mo(c)`'s Dirichlet increments, `me(x, sd)`'s latent true covariate, a
Gaussian process's length scale and amplitude. They are addressed by naming the
term itself in the target slot, under the same head-position grammar:

```julia
m = @brm df begin
    y ~ Normal(mu, 1.)
    mu ~ 1 + s(age) + mo(dose) + me(w_obs, 0.3) + hsgp(conc; k=5)

    sd(:, s(age))               ~ Exponential(2)     # smoothing SD
    simplex(mu, mo(dose))       ~ Dirichlet(2)       # monotonic increments
    latent(:, me(w_obs))        ~ Normal(0, 5)       # latent true covariate
    length_scale(:, hsgp(conc)) ~ Uniform(0.84, 2)   # GP length scale
    sd(:, hsgp(conc))           ~ Normal(0, 0.5)     # GP marginal amplitude
end
```

The term is spelled the way the formula spells it, minus numeric and keyword
arguments — `me(w_obs, 0.3)` is addressed as `me(w_obs)`. `term_priors(brmi)`
returns the captured statements. See
[Term-internal priors](@ref) for the full table, the `t2` component slot, why
the standardized raw innovations are deliberately not configurable, and the
approximation-validity floor an `hsgp` term puts on its length scale by
default.

See [Formula terms](@ref) and [Likelihoods](@ref) for the supported syntax and
backend-specific contracts. The [Gallery](/gallery) provides live, interactive examples — input
formula, the SLIC submodel body, the transpiled Stan source, and the
auto-generated posterior-predictive check, all in one card. The
[API](/api) page lists every public binding.
