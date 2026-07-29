# Formula terms

## Fixed-one contribution: `offset(x)`

[`offset`](@ref) adds `x` directly to a population-level linear predictor with
coefficient one. It never allocates a `beta_pop` column. The argument may be a
raw-data expression, as in an exposure offset, or an already-declared sampled
scalar:

```julia
brmi = @brm df begin
    log_ka_pop ~ Normal(-2.08, 1.0)
    sigma ~ Exponential(1.0)
    eta_ka ~ offset(log_ka_pop) + (1 | p | subject)
    concentration ~ Normal(exp(eta_ka), sigma)
end
```

Use [`protect`](@ref) for literal data transformations whose resulting column
should receive an estimated population coefficient. `protect(log_ka_pop)` is
therefore not an alternative spelling: `protect` materializes raw-data
expressions, while `offset` preserves a model-value reference and fixes its
coefficient to one. Offsets are population-level terms and are rejected inside
`(... | group)` random-effects terms.

## Penalized smooth: `s(x)`

[`s`](@ref) adds a penalized one-dimensional thin-plate regression spline to a
linear predictor. It is available only in the `SBBRMI` StanBlocks backend.

```julia
using BayesianRegressionModels, Distributions

x = collect(range(-2, 2; length=50))
df = (; x, y=sin.(x))

brmi = @brm df begin
    y ~ Normal(mu, sigma)
    mu ~ s(x)
    sigma ~ Exponential(1)
end

sb = SBBRMI(brmi)
```

The supported public call has exactly one numeric predictor and no keyword
arguments. Training values must be finite and contain at least 10 unique
values. Internally, `s(x)` uses a fixed rank-10 basis: two unpenalized
null-space columns for `{1, x}` and eight penalty-whitened range columns. The
range coefficients share a smoothing standard deviation with a standard
half-normal prior. The term contributes this complete smooth directly to the
linear predictor, so it does not receive an additional population coefficient.

For prediction or posterior replay on new data, the default
`reprocess(sb, new_df)` and `restan_data(sb, new_df)` calls evaluate `x` against
the frozen training centers and basis. Passing `freeze_constants=false`
re-estimates the basis from the new data and therefore has fresh-fit rather
than prediction semantics.

### Difference from `bs(...)`

Bambi/Formulae formulas such as `bs(x, knots=knots)` create a deterministic
B-spline design matrix whose dimension and knots are controlled by the formula.
BayesianRegressionModels' `s(x)` instead represents a penalized thin-plate
smooth with the fixed rank and smoothing prior described above. They are not a
one-for-one syntax translation: `s(x; k=...)`, `s(x; knots=...)`, `bs(...)`, and
`t2(...)` are not alternative spellings of this term.

## Tensor-product smooth: `t2(x, z)`

[`t2`](@ref) adds a two-margin tensor-product smooth to an `SBBRMI` linear
predictor. Its public defaults follow the brms/mgcv `t2` catalogue surface,
with Julia tuples for per-margin options:

```julia
brmi = @brm df begin
    loc ~ 1 + t2(area, yearc;
                 k=(5, 5), basis=(:cr, :cr), full=false)
    rent ~ Normal(loc, sigma)
    sigma ~ Exponential(1)
end
```

Both predictors must be finite numeric columns with at least the corresponding
number of unique values in `k`. Each `k` entry must be an integer greater than
2. The current implementation accepts only cubic-regression-spline margins
(`basis=(:cr, :cr)`) and `full=false`; unsupported values and unknown keywords
are rejected while the `BRMI` is built.

Each marginal basis is split into a two-dimensional null space and a
penalty-whitened range space. Their tensor product has three unpenalized
null×null columns after the intercept constraint, plus independently scaled
range×range, range×null, and null×range blocks. At the default `k=(5, 5)`,
those penalized blocks have 9, 6, and 6 coefficients. Each block has its own
standard half-normal smoothing scale, and the complete smooth is added directly
without an extra population coefficient.

`reprocess(sb, new_df)` evaluates all four blocks against the frozen training
knots, penalty decomposition, and intercept constraint. Passing
`freeze_constants=false` re-estimates them from `new_df`. Like [`s`](@ref),
`t2` is implemented only by the StanBlocks backend and is not available to
`VBRMI`.

## R²-induced variance decomposition: `effect(:) ~ r2d2(...)`

[`r2d2`](@ref) replaces the independent per-coefficient priors on a linear
predictor's population block with a *joint* prior induced by a prior on that
predictor's coefficient of determination. It is a separate statement in the
formula block — addressed with [`effect`](@ref), never with a per-column
`effect(lp, coef) ~ Normal(...)` — and is implemented only by the `SBBRMI`
StanBlocks backend.

```julia
using BayesianRegressionModels, Distributions

brmi = @brm df begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    log_V  ~ 1 + wt + (1 | p | subject)

    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_V, :)  ~ r2d2(R2=Beta(2, 3), tau_bsv=0.25)

    conc ~ Normal(exp(log_CL - log_V) * time, 1)
end
```

`effect(lp, :)` addresses every population coefficient of `lp` at once; the
bare `effect(:)` is shorthand that resolves only when the model has exactly one
population predictor. The `Colon` address is deliberately invisible to
[`effect_priors`](@ref) — it names no single labelled column — and is read back
with [`r2d2_priors`](@ref) instead.

### What it emits

Writing `tau_bsv` for the predictor's total scale, the decomposition is

```
R2               ~ Beta(a, b)                                 # parameter
phi              ~ Dirichlet(alpha)                           # simplex
beta_scale[k]     = sqrt(phi[k] * R2 * tau_bsv^2 / Var(x_k))  # transformed
tau_resid         = sqrt((1 - R2) * tau_bsv^2)                # transformed
```

`beta_scale` is injected into the shipped `_popefs_normal` seam, so the
population block still samples as `beta_pop ~ normal(beta_loc, beta_scale)`;
`tau_resid` becomes the random effect's standard deviation. The column
variances `Var(x_k)` depend on data alone and hoist to Stan's transformed-data
block.

The intercept is excluded from the simplex. Its design column is constant, so
`Var(x_k) = 0` there, and an intercept is a location rather than explained
variance. A predictor whose only non-intercept column count is one needs no
simplex parameter at all: `phi = [1]` becomes a data constant.

### The random effect plays the residual role

The inciting shape is population PK: `log_CL` is a *latent* per-subject
parameter with no residual term of its own, so its subject random effect **is**
the unexplained half of `tau_bsv^2`. That is why `r2d2` derives the
random-effect SD rather than sampling it, and why an
`effect(sd, ID, ...)` statement on the same block is rejected — the
decomposition already determines those scales.

### Keywords

| keyword | default | meaning |
|---|---|---|
| `R2` | required | `Beta(a, b)` prior on the coefficient of determination |
| `tau_bsv` | sampled half-standard-normal | the predictor's total scale; pass a number to fix it |
| `alpha` | `1` | Dirichlet concentration over the non-intercept columns |

`tau_bsv` has no data-derived default. A latent per-subject parameter has no
observed response to derive a scale from, so an omitted `tau_bsv` becomes a
sampled `real<lower=0>` with a half-standard-normal prior. Fix it whenever you
have a defensible scale — it is the quantity the whole decomposition is
relative to.

### Current limits

All of these fail loudly rather than silently sampling something else:

- The random effect playing the residual role must be a single intercept —
  `(1 | g)` or one margin per predictor inside a `(1 | ID | g)` bucket.
  Splitting `(1 - R2) * tau_bsv^2` over several margins needs a second simplex
  that this decomposition does not build.
- A shared `(… | ID | g)` bucket is all-or-nothing: either every margin's
  predictor carries an `r2d2` statement or none does.
- `effect(cor, ID)` is not yet composable; an `r2d2` block keeps LKJ `eta = 1`.
- Adaptive centering (`centered_groups`), cv-contagious sizing (`cv_groups`),
  stratified `gr(g, by=b)` groups, and `mm(...)` multi-membership terms are all
  unsupported in combination with `r2d2`.
- A column that also carries its own `effect(lp, coef) ~ Normal(loc, scale)`
  statement is dropped from the simplex and keeps that explicit prior.
