# Formula terms

## Catalogue

Every term below lowers on the StanBlocks backend (`SBBRMI`); the
`VBRMI` column marks the ones the pure-Julia backend also implements. The pages
that follow document the terms whose behaviour is not obvious from the
signature — the rest are covered by their docstrings on the [API](@ref) page.

### Predictor terms

| Term | What it contributes | `VBRMI` |
| --- | --- | --- |
| `offset(x)` | `x` added with coefficient one — no `beta_pop` column | ✓ |
| `s(x)` | rank-10 penalized thin-plate regression spline | — |
| `t2(x, z)` | two-margin tensor-product smooth | — |
| `gp(x…; cov=:exp_quad, iso=true, jitter=1e-9)` | exact latent Gaussian process, noncentered Cholesky draw | — |
| `hsgp(x…; k=20, c=1.5, iso=true, by=nothing)` | Hilbert-space GP approximation over `prod(k)` basis functions | — |
| `ar(time; p=1)` | AR(p) noise process ordered by `time`; only `p=1` is emitted | — |
| `mo(c)`, `mo1(c)` | monotonic effect of an ordered factor via Dirichlet increments | — |
| `me(x, sd)` | measurement-error covariate — `x` is observed with known `sd` | — |
| `factor(c; ref=k)` | treatment contrasts for a categorical column, reference level `k` | ✓ |
| `protect(x)` | materialize a raw data expression as one literal column | ✓ |

A plain RHS expression in raw data columns (`log(exposure)`, `x^2`) is treated
as an implicit `protect(...)` and materialized the same way.

`gp` and `hsgp` are distinct terms with no compatibility alias. Both are direct
predictor summands carrying their own latent draws and hyperparameters, so
neither contributes a `beta_pop` coefficient. `jitter` belongs only to `gp`;
`k`, `c` and `by` only to `hsgp`. Both currently support `cov=:exp_quad` only.

### Column transforms

Applied to a raw column before it enters the design matrix. The constants are
fitted on the training frame and frozen, so [`reprocess`](@ref) replays them
rather than re-deriving them — see the replay contract below.

| Term | What it does |
| --- | --- |
| `zscale(x)` | subtract the mean, divide by the SD |
| `center(x)` | subtract the mean |
| `standardize(x)` | mean/SD standardisation |

### Grouping factors

Used on the right of `|` in a random-effect block.

| Term | What it does |
| --- | --- |
| `gr(g)` | ordinary grouping factor / strata |
| `mm(g1, g2, …; weights=(w1, w2, …), normalize=true)` | multi-membership — one coefficient block shared across two or more levels per row |

`mm` needs at least two group columns; omitting `weights` gives exact equal
weights `1/M`. Supplied weights must be present, real, finite, nonnegative and
sum to something positive on every row. It does not combine with `|ID|`,
`cv_groups` or `centered_groups`, which error explicitly.

### Response-level wrappers

| Term | What it does |
| --- | --- |
| `mi(y)` | brms-style observed/imputed split — missing rows become parameters drawn from the same family |
| [`weighted(y, w)`](@ref) | typed observation weights on the likelihood |
| `truncated`, `censored`, [`interval_censored`](@ref) | truncation and censoring — see [Likelihoods](@ref) |

`mm`'s membership weights are part of the random-effect contribution and are a
different thing from `weighted(...)`, which scales the likelihood.

### Group-local kernels

| Term | What it does |
| --- | --- |
| `kernel(args…) do …` | a per-subject model cell — the PMX/PKPD surface |
| `ragged(x, group)` | group a flat secondary row axis into the ragged per-subject view a `kernel` cell slices |

`ragged`'s `x` may be a linear predictor declared in the same `@brm` block or a
raw flat data column; `group` names, for every row of that frame, which subject
the row belongs to.

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
range coefficients share a smoothing standard deviation whose default prior is
a standard half-normal; `sd(<lp|:>, s(x)) ~ Exponential(scale)` replaces it (see
[Term-internal priors](@ref)). The term contributes this complete smooth
directly to the linear predictor, so it does not receive an additional
population coefficient.

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
smoothing scale, defaulting to a standard half-normal and addressable
individually as `sd(<lp|:>, t2(x, z), <block>)` with `<block>` one of `rr`,
`rn`, `nr` (see [Term-internal priors](@ref)). The complete smooth is added
directly without an extra population coefficient.

`reprocess(sb, new_df)` evaluates all four blocks against the frozen training
knots, penalty decomposition, and intercept constraint. Passing
`freeze_constants=false` re-estimates them from `new_df`. Like [`s`](@ref),
`t2` is implemented only by the StanBlocks backend and is not available to
`VBRMI`.

## Term-internal priors

Some terms own parameters that no coefficient address can reach. `s(x)`'s
smoothing scale, `mo(c)`'s Dirichlet increments, `me(x, sd)`'s latent true
covariate and a Gaussian process's length scale and amplitude all live inside
the term's own submodel, and none of them is a `beta_pop` column or a
grouping-factor margin. They are addressed by naming the term itself in the
target slot, in the same head-position grammar the rest of the prior surface
uses:

```
<quantity>(<linear predictor | :>, <term>[, <component>]) ~ <distribution>
```

```julia
@brm df begin
    y ~ Normal(mu, sigma)
    mu ~ 1 + s(age) + t2(x, z) + mo(dose) + me(w_obs, 0.3) + hsgp(conc; k=5)
    sigma ~ Exponential(1)

    sd(:, s(age))               ~ Exponential(2)     # smoothing scale
    sd(mu, t2(x, z), rr)        ~ Exponential(3)     # one tensor penalty
    simplex(:, mo(dose))        ~ Dirichlet(2)       # monotonic increments
    latent(mu, me(w_obs))       ~ Normal(0, 5)       # latent true covariate
    length_scale(:, hsgp(conc)) ~ Uniform(0.84, 2)   # GP length scale
    sd(:, hsgp(conc))           ~ Normal(0, 0.5)     # GP marginal amplitude
end
```

| head | term | parameter | default |
| --- | --- | --- | --- |
| `sd` | `s(x)` | smoothing SD | half-standard-normal |
| `sd` | `t2(x, z)` | one of the `rr` / `rn` / `nr` penalty SDs | half-standard-normal |
| `simplex` | `mo(c)`, `mo1(c)` | Dirichlet concentration | `Dirichlet(1)` |
| `latent` | `me(x, sd)` | latent true covariate | `Normal(0, 1)` |
| `length_scale` | `gp(x…)`, `hsgp(x…)` | GP length scale `rho` | `LogNormal(0, 1)` |
| `sd` | `gp(x…)`, `hsgp(x…)` | GP marginal amplitude `sigma` | `LogNormal(0, 1)` |

**Spell the term the way the formula does, minus numeric and keyword
arguments.** `me(w_obs, 0.3)` is addressed as `me(w_obs)` and `t2(x, z; k=(5,5))`
as `t2(x, z)` — the address names a term, not a call. A `:` in the predictor
slot is THE DEFAULT: it is the base layer that a statement naming a concrete
predictor overrides, exactly as on the coefficient and grouping-factor
surfaces. Specificity counts concrete slots, so an exact tie is an error rather
than a silent winner.

An address that cannot be honoured is refused by name while the model is
lowered — an unknown term, a term with no such parameter, a `t2` component slot
that is missing or names no penalty block, or two terms in one predictor
spelled identically. `term_priors(brmi)` returns the captured statements
(`class`, `term`, `predictor`, `component`, `family`, `arguments`, `keywords`,
`expression`) for formula-level provenance.

### Only model-scale quantities are exposed

The standardized raw innovations these submodels sample — `b_pen_raw`, `z`,
`beta_raw` — stay iid standard normal and are deliberately NOT configurable.
For a smooth, `Cov(f_pen | sd) = sd² Zpen Zpen'`, so a scale on the raw
coefficients would simply duplicate the smoothing SD and confound the two. The
configurable parameters are the ones that mean something on the model's own
scale.

### Current limits

- `sd` on `s(x)` / `t2(x, z)` accepts `Exponential(scale)` only, matching the
  grouping-factor SD surface. `Distributions.Exponential` is
  scale-parameterized while Stan's `exponential_lpdf` takes a rate; BRM
  performs the conversion.
- `simplex` accepts `Dirichlet(a)` (one concentration, broadcast over every
  increment) or `Dirichlet(a₁, …, a_{K-1})` (one per increment of a `K`-level
  factor). The dimension comes from the data, so the
  `Dirichlet(dimension, concentration)` spelling used for a standalone simplex
  parameter is not accepted here.
- `latent` accepts `Normal(location, scale)`. The observation likelihood
  `x_obs ~ Normal(x_true, sd)` is never configurable.
- `length_scale` and `sd` on a `gp` / `hsgp` term accept `LogNormal`,
  `InverseGamma`, `Gamma`, `Exponential`, `Normal` and `Uniform`, with numeric
  hyperparameters. Both parameters are strictly positive, so every family
  except `Uniform` is emitted truncated at zero (`Normal(0, s)` is therefore a
  half-normal); `Uniform(a, b)` additionally declares `<lower=a, upper=b>` so
  the declaration and the density's support agree. An **anisotropic** term
  (`iso=false`) has one length scale per axis and the statement sets all of
  them; the isotropic form has a single shared one. With `by=g` the length
  scale and amplitude are shared across groups, so one statement configures the
  whole term.
- `ar`'s autocorrelation has no address yet.

### `hsgp` bounds its length scale by default

An HSGP with `k` basis functions over a domain of half-width
`L = c·max|x − mean(x)|` stops approximating the kernel it was asked for once
the length scale falls below

```
(4L/π)·√(log(100)/(k² − 1))
```

and the failure is **silent**: the model transpiles, passes `stanc`, samples,
and returns finite draws that simply are not that Gaussian process. On the
default `LogNormal(0, 1)` at `L = 1.5` that region holds 42.9 % of the prior
mass at `k = 5`, 18.8 % at `k = 10` and 5.7 % at `k = 20`.

Every `hsgp` term therefore declares `rho` with that floor as its lower bound.
The density is unchanged — only the support moves. Exact `gp` has no basis
truncation and is untouched.

The floor is emitted as **data** (`rho_lower_hsgp_<axes>`), not as a literal,
because `L` comes from the covariate: `reprocess(sb, df2; freeze_constants=false)`
re-derives it alongside `PHI` / `omega2` while the Stan source stays
byte-identical. Isotropic spellings share one `rho` across axes and take the
strictest per-axis floor; `iso=false` bounds each axis separately. `k = 1` puts
the formula at infinity, so that degenerate basis stays unbounded.

**An explicit `length_scale` statement replaces the whole declaration, floor
included:**

```julia
mu ~ hsgp(x; k=5, c=1.5)                     # real<lower=rho_lower_hsgp_x>
length_scale(:, hsgp(x)) ~ LogNormal(0, 1)   # real<lower=0.0> — floor gone
```

That is deliberate. It is how a pre-floor posterior is reproduced, and it is
what keeps `Uniform(a, b)` self-consistent — an unconditional floor would
overwrite `lower=a` while the density stayed `uniform(a, b)`, leaving the
declaration and the support disagreeing. The consequence is that the guarantee
is **default-on, not absolute**: an override with mass below the floor restores
the silent approximation error, with no warning. Either keep the floor in the
overriding bounds (`Uniform(0.84, 2)`) or accept the error knowingly.

!!! warning "This changes the posterior of existing unedited `hsgp` models"
    Models that never named `length_scale` sampled an unbounded `rho` before
    this default landed. If you version models by their emitted Stan plus
    data, treat the change as a new model version rather than a refresh of the
    old one.

## R²-induced variance decomposition: `effect(lp, :) ~ r2d2(...)`

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

`effect(lp, :)` addresses every population coefficient of `lp` at once, and
`effect(:, :)` addresses them across every predictor — which for an `r2d2`
decomposition resolves only when the model has exactly one population
predictor, since the decomposition is per-predictor. An `r2d2` statement is
deliberately invisible to [`effect_priors`](@ref) — it names no single labelled
column — and is read back with [`r2d2_priors`](@ref) instead.

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
`sd(...)` statement on the same block is rejected — the
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
- `cor(:, ID)` is not yet composable; an `r2d2` block keeps LKJ `eta = 1`.
- Adaptive centering (`centered_groups`), cv-contagious sizing (`cv_groups`),
  stratified `gr(g, by=b)` groups, and `mm(...)` multi-membership terms are all
  unsupported in combination with `r2d2`.
- A column that also carries its own `effect(lp, coef) ~ Normal(loc, scale)`
  statement is dropped from the simplex and keeps that explicit prior.

## Scalar horseshoe prior: `coef ~ Horseshoe(...)`

[`Horseshoe`](@ref) attaches a scalar Carvalho–Polson–Scott shrinkage
hierarchy to an explicitly named coefficient:

```julia
brmi = @brm df begin
    beta_sparse ~ Horseshoe(local_scale=0.5, global_scale=0.1)
    sigma ~ Exponential(1)
    mu ~ 0 + beta_sparse * x
    y ~ Normal(mu, sigma)
end
```

The emitted non-centered hierarchy is

```text
raw    ~ Normal(0, 1)
lambda ~ half-Cauchy(0, local_scale)
tau    ~ half-Cauchy(0, global_scale)
beta_sparse = raw * lambda * tau
```

`local_scale` and `global_scale` are optional positive finite formula
constants. Both default to `1.0`; the no-keyword spelling
`Horseshoe()` retains the historical emission byte for byte. Literal
arithmetic such as `local_scale=1/2` is evaluated with Julia semantics.
Positional arguments, unknown keywords, nonnumeric values, zero, negative,
and non-finite scales are rejected during `SBBRMI` construction.

The current surface is scalar and each `coef ~ Horseshoe(...)` call owns its
own `(raw, lambda, tau)` triple. Consequently `tau` is “global” only within
that scalar hierarchy; it is not shared across several coefficients. A
genuinely shared global scale requires a vector/group horseshoe declaration,
which this marker does not imply. The standardized `raw` draw is intentionally
not configurable because its scale would duplicate `lambda`/`tau`.

Like other sbimpl prior surfaces, this is implemented by `SBBRMI`; do not
assume `VBRMI` has the same marker-specific lowering.

## Simplex-valued parameter: `s ~ Dirichlet(...)`

A non-data left-hand side with a `Dirichlet` right-hand side declares a
**simplex-valued parameter** — Stan's `simplex[K]` — rather than a linear
predictor. Like every other scalar-parameter prior declaration it is addressable
by name anywhere later in the formula block, `kernel(...)` cells included. It is
implemented only by the `SBBRMI` StanBlocks backend.

```julia
using BayesianRegressionModels, Distributions

brmi = @brm df begin
    sigma ~ Exponential(1)
    diet_share ~ Dirichlet(3, 1.0)
    log_CL ~ 1 + (1 | p | subject)

    pred ~ kernel(t, dose, diet, dv, log_CL) do ts, d, di, yy, lCL
        mu = d * diet_share[di] * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end
```

The inciting shape is a per-**event** multiplier: `diet` scales dose
bioavailability inside the cell, so what the model needs is a `K`-simplex
parameter indexed by an ordinal level — not a per-observation design column.

### What it emits

For `diet_share ~ Dirichlet(3, 1.0)`:

```stan
data        { int diet_share_alpha_n;
              vector[diet_share_alpha_n] diet_share_alpha; }   // [1.0, 1.0, 1.0]
parameters  { simplex[diet_share_alpha_n] diet_share; }
model       { diet_share ~ dirichlet(diet_share_alpha); }
```

The concentration is registered as **data** under `<name>_alpha`, which is what
sizes the simplex; the name is reserved, so a collision is rejected rather than
overwritten. A `simplex[K]` costs `K - 1` unconstrained coordinates and reports
`K` constrained ones.

Because the concentration is data, Stan drops the Dirichlet log-normalizer — a
function of `alpha` alone — from `target`, exactly as it does for any other `~`
statement with data-only hyperparameters. The sampled distribution is unchanged.

### Accepted spellings

Only the two genuine `Distributions.Dirichlet` constructors, with Julia's
parameterization preserved:

| spelling | meaning |
|---|---|
| `Dirichlet(alpha)` | concentration vector literal, e.g. `Dirichlet([2.0, 1.5, 3.0])` |
| `Dirichlet(K, a)` | symmetric: `K` components, each concentration `a` |

`@brm` is a macro over the formula block, so a bare Julia symbol on the
right-hand side is parsed as a formula **local**, not interpolated —
`Dirichlet(3, alpha)` with a captured `alpha` reaches the backend as a column
carrier and is rejected by name. Spell the concentration as a literal; there is
no `$` escape (`$` outside a quote is a Julia syntax error, so the macro never
sees it). This matches the rest of the formula surface — `r2d2`'s `alpha=` and
`effect(:, x) ~ Normal(0, 0.25)` are literals for the same reason.

Concentrations must be finite and strictly positive, and `K >= 2`: a
one-element simplex is deterministically `[1.0]`, so there is no parameter to
sample.

### Difference from `mo(c)` / `mo1(c)`

[`mo`](@ref) and [`mo1`](@ref) are population linear-predictor terms. Their
Dirichlet increments live inside a submodel and are not addressable from the
formula; what a bare `share ~ mo1(c)` hands back is a per-**row** monotonic
contrast vector of length `nrow(df)`. Indexing that by an ordinal level is a
silent double indirection — it transpiles and `stanc`-checks, but computes
something else. Use `Dirichlet` when you want the simplex itself.

### Current limits

- The left-hand side must be a non-data name. A data-backed response with a
  `Dirichlet` right-hand side is a simplex-valued *likelihood*, which has no
  density/pointwise/predictive support and is rejected by the family path.
- The concentration is a constant, not a hyperprior: `Dirichlet(alpha)` with
  `alpha` a sampled parameter is not supported.
- `Dirichlet(K)` — no concentration — is not a `Distributions.jl` constructor
  and is rejected; write `Dirichlet(K, 1.0)` for the flat case.
- StanBlocks-only, like [`s`](@ref), [`t2`](@ref) and [`r2d2`](@ref); not
  available to `VBRMI`.
