````@raw html
---
layout: home

hero:
  name: BayesianRegressionModels.jl
  text: A brms-shaped formula DSL for Julia
  tagline: Same formula grammar, but your columns do not all have to come from one equal-length data frame. Runs through ReactiveKernelsPPL, and transpiles to Stan through StanBlocks.
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

[`@brm`](@ref) parses a brms-style formula block (`y ~ 1 + a + (1 | g)`,
`log(err) ~ 1 + b`, `y ~ Normal(loc, err)`) into a [`BRMI`](@ref), the
model's intermediate representation, and a backend turns that into something
that computes a log density:

- [`RKBRMI`](@ref) emits a ReactiveKernelsPPL program: a Julia log density
  with its gradient, which WarmupHMC samples and which also drives
  NLMEEstimation's FOCE(I), SAEM and empirical-Bayes estimators, minibatch
  variational inference, sequential Monte Carlo, and the model's export as a
  Python/JAX module. The
  [PK method comparison](https://nsiccha.github.io/NLMEEstimation.jl/dev/pk-comparison)
  runs those estimators and full MCMC on one model.
- [`SBBRMI`](@ref) emits a StanBlocks model and its Stan program, which
  BridgeStan compiles and WarmupHMC samples.

Every example on this site is built from its source when the site is built.
The comparison below shows the declaration, its intermediate representation,
the StanBlocks model, the Stan program, and the JAX module of the same model;
the "Compare" button puts them side by side.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
intro_model = (@brm begin
    y ~ Normal(loc, err)
    loc ~ 1 + age + sex + (1 + age | subj)
    err ~ Exponential(1)
end)((;
    age=[21.0, 38.0, 55.0, 29.0, 47.0, 61.0],
    sex=[1, 2, 1, 2, 1, 2],
    subj=[1, 1, 2, 2, 3, 3],
    y=[0.2, 1.1, -0.4, 0.7, 1.4, 1.0],
))
""", :intro_model; title="BRM regression model", jax=true)
```

## What is on this site

[Formula terms](formula-terms.md) is the catalogue of what a formula may
contain, [Likelihoods](likelihoods.md) the list of response families, and
[Reusable components](components.md) shows functions with their own
parameters that a model calls. [Space-time fields](space-time-fields.md)
covers sparse-precision fields on meshes and neighbour graphs.

The worked examples fit real or published models: [Warfarin PK/PD](warfarin.md),
with Weber's two-stage workflow and a joint model on the public data;
[wastewater](wastewater.md) and [CDC ww-inference](wastewater-cdc.md)
renewal models; [epidemic renewal models](renewal.md) and
[epidemiological delays](epidist.md); a [grey-seal population model](seal-brm.md);
[continuous-time state-space models](state-space-models.md) with latent
states sampled or filtered; and the [Monster PBPK model](monster.md). The
[feature atlas](feature-atlas.md) shows one small example per feature.

The adaptive-centering case studies compare fixed coordinates, post-hoc
selection and online adaptation on a [motorcycle HSGP](adaptive-centering.md),
[eight schools](eight-schools-centering.md), [radon](radon-centering.md), two
[pupil](pupil-centering.md) [models](pupil-scale-centering.md),
[air pollution](air-centering.md) and a [RBesT MAP prior](rbest-centering.md),
with the gradient cost of each.

The [intermediate representation](intermediate-representation.md) page
explains the IR pane, [scientific model descriptions](model-description.md)
the prose and equations BRM derives from a model, and the [API](api.md) page
lists every public binding.

## Coming from brms

The formula grammar carries over, and so does the random-effect syntax:

| brms | `@brm` |
|---|---|
| `y ~ 1 + age + sex` | `y ~ 1 + age + sex` |
| `(1 + age \| subj)` | `(1 + age \| subj)` |
| `(1 + age \|p\| subj)`, correlated across formulas | `(1 + age \|p\| subj)` |
| `(1 \| gr(subj, by = diagnosis))` | `(1 \| gr(subj, by = diagnosis))` |
| `bf(y1) + bf(y2)` | two `~` lines in the same block |
| distributional regression, `bf(y ~ x, sigma ~ x)` | every distributional parameter is another `~` line |
| nonlinear terms, `bf(y ~ a * exp(-b * x), a ~ 1, b ~ 1, nl = TRUE)` | the same expression, without an `nl` switch |

A distributional parameter gets its own formula, with the link written where
it applies:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
distributional = (@brm begin
    y        ~ Normal(mu, sigma)
    mu       ~ 1 + age
    log(sigma) ~ 1 + age          # explicit link, addressed as `sigma`
end)((; age=[21.0, 38.0, 55.0, 29.0], y=[0.2, 1.1, -0.4, 0.7]))
""", :distributional; title="Distributional Gaussian regression")
```

A nonlinear model composes declared values in an ordinary Julia expression.
There is no `nl = TRUE` and no `nlf()`:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
nonlinear = (@brm begin
    a     ~ 1 + (1 | g)           # linear predictors
    b     ~ 1
    y     ~ Normal(a * exp(-b * x), sigma)   # composed nonlinearly
    sigma ~ Exponential(1)
end)((;
    g=[1, 1, 2, 2], x=[0.0, 0.5, 1.0, 1.5],
    y=[1.0, 0.8, 0.5, 0.3],
))
""", :nonlinear; title="Nonlinear predictor composition")
```

The Stan program observes `y ~ normal(a .* exp(-b .* x), sigma)`, and as for
every response it also writes the pointwise log likelihood `y_likelihood` and
the predictive draws `y_gen`.

### Where BRM differs

The data need not be one data frame. brms takes a single `data.frame`, so
every column shares one row axis; `@brm` takes any collection of columns, a
`NamedTuple` for instance, whose columns may live on different row axes, and
different linear predictors in one model may be defined on different axes.
The [multi-axis population PK kernel](@ref) is a runnable example: its
subject columns have one row per person, its observation columns one row per
sample.

An `@plate for` cell (see [`kernel`](@ref)) evaluates Julia code once per
group, a structural or ODE time course for instance, with that group's values
of the linear predictors as inputs and another vector's entries read through a
[`groups`](@ref) binding. In brms this class of model is `nlf()` with manual
`data2` bookkeeping, or Stan by hand.

[`RKBRMI`](@ref) gives a pure-Julia `LogDensityProblems` object
(`rk_logdensity_problem`) with no Stan toolchain in the loop, and
[`brm_descriptor`](@ref) describes everything a model emits, so an
application mounts a fitted model without keeping its own registry of
parameter names.

### What brms has and BRM does not

Residual correlation is Gaussian and complete-row only: `[y1, y2] ~
MvNormalCholesky([mu1, mu2], L_res)` with a declared `LKJCovarianceFactor`;
partially missing response vectors and non-Gaussian residual copulas are not
implemented.

There are no fixed or known covariance groups: `by=` is the only [`gr`](@ref)
option, so no phylogenetic or pedigree random effects.

Autocorrelation shorthands are first-order. [`AR1Normal`](@ref) in a
`@plate for p in groups(g)` cell gives Gaussian residuals an AR(1) process
within each group (brms' `ar(time, gr)`, with the exact likelihood and gaps
allowed); [`ar`](@ref)`(time; p=1)` is an AR(1) noise column and
[`dar`](@ref)`(time; p=1)` a differenced AR(1) trajectory. Higher orders, MA,
ARMA and non-Gaussian residual AR are written as a `@scan` recurrence in a
`@plate` cell; there is no compound-symmetry, unstructured, CAR or SAR
residual surface (areal fields exist as the `car` term).

The list of [Likelihoods](@ref) is shorter than brms'.

## Priors

Population coefficients have independent standard-normal priors unless a
formula says otherwise. `effect(predictor, coefficient) ~ prior` sets one
coefficient's prior; the coefficient names are those `popcoefnames` returns:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
pk = (@brm begin
    log_ka ~ 1 + weight + (1 | pk | subject)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, weight) ~ Normal(0, 0.1)
    y ~ Normal(log_ka, 1.0)
end)((;
    weight=[55.0, 65.0, 75.0, 85.0],
    subject=[1, 1, 2, 2], y=[-2.1, -1.9, -1.8, -1.7],
))
""", :pk; title="Population effect priors")
```

Both slots are required, and `:` is the wildcard in either.
`effect(:, weight) ~ Normal(0, 0.1)` is the default for `weight` in every
predictor that has it, and a more specific address such as
`effect(log_ka, weight)` overrides it. Two addresses of equal specificity
reaching one parameter, and unknown addresses, are errors. The override
supports `Normal(location, scale)` and keeps the `pop_<predictor>_beta_pop`
parameter, its `popcoefnames` labels and the descriptor's provenance.
`effect_priors(brmi)` returns the captured statements.

### Categorical contrasts

A categorical predictor, a bare integer or `CategoricalArray` column or one
wrapped in `factor(...)`, is not a `beta_pop` column and `popcoefnames` does
not list it. It owns a separate `cat_<predictor>_<column>_beta` vector of its
K−1 treatment contrasts, the reference level pinned at 0. Those contrasts
also default to `std_normal()`, and the same `effect(...)` address changes
them, keyed by the column name:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
categorical_prior = (@brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + factor(g) + x
    effect(mu, g) ~ Normal(0.0, 0.5)   # ⇒ cat_mu_g_beta ~ normal(0.0, 0.5);
    y ~ Normal(mu, sigma)
end)((;
    g=[1, 2, 3, 1, 2, 3], x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
))
""", :categorical_prior; title="Categorical effect prior")
```

One `(location, scale)` covers every contrast in the block; a treatment
contrast has no per-level address (a cell mean does, below). `effect(:, g)`
reaches the block in every predictor owning it, and the statement composes
with population overrides on the same predictor
(`effect(mu, x) ~ Normal(0, 0.25)`): each addresses its own parameter. A
non-default reference level emits `cat_<predictor>_<column>__ref_<k>_beta`,
which the plain column name still addresses when that is unambiguous; when
two `factor(g; ref=…)` blocks of one column would both claim it, the bare
address is refused and each block is addressed by its reference-qualified
column name. A model without such a statement keeps the `std_normal()`
contrast prior under the predictor-qualified name.

### Cell means: a categorical predictor without an intercept

Treatment contrasts measure each level against a reference, so they need an
intercept to measure from. A predictor without one codes its first
categorical term by cell means instead, one coefficient per level and no
reference level, which is what `0 + factor(g)` means in brms and in R's
`model.matrix`. Each cell mean is addressable on its own as
`<column>_lvl_<k>`, `k` the level's position in the fitted level order, so a
per-group location can carry a per-group prior:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
cell_means = (@brm begin
    sigma ~ Exponential(1)
    mu ~ 0 + factor(site)
    effect(mu, site) ~ Normal(0.0, 2.0)          # every site
    effect(mu, site_lvl_3) ~ Normal(4.0, 0.5)    # ... except the third
    y ~ Normal(mu, sigma)
end)((;
    site=[1, 2, 3, 1, 2, 3],
    y=[-0.4, 0.2, 4.1, -0.1, 0.5, 3.8],
))
""", :cell_means; title="Cell means with a per-level prior")
```

The emitted carrier is the same `cat_<predictor>_<column>_beta` vector, of
length K rather than K−1 and with no pinned zero. The column address sets one
prior over all K cell means; a level address is more specific and overrides
it for that level, under the same most-specific-wins rule as every other
`effect(...)` statement.

The rule in full:

- BRM has no implicit intercept and `0` is only a marker, so "without an
  intercept" means the predictor has no `1` term: `mu ~ site`, `mu ~ 0 + site`
  and `mu ~ 0 + factor(site)` are the same formula.
- Only the first categorical term is cell-mean coded. A second one's full
  indicator set would be collinear with the first's, so later categorical
  terms, and every `&` interaction, stay treatment-coded.
- `factor(site; cmc=false)`, brms' switch, keeps K−1 treatment contrasts in a
  predictor without an intercept, and the cell means pass to the next
  categorical term. A `ref=` alone does not opt out: as in R, a releveled
  factor without an intercept is still cell-mean coded, in its releveled
  order.
- An ordinal model's estimated thresholds are its location predictor's
  intercept, so the `eta` of `Ordinal(...)` / `OrderedLogistic(...)` keeps
  treatment contrasts even though it is written `eta ~ 0 + ...`.
- A random intercept `(1 | g)` is not a population intercept.
- The same rule holds inside a random-effect block, decided on that block's
  own left-hand side: `(0 + c | g)` gives every level of `c` its own
  group-level effect (margins `c_dummy_1 … c_dummy_K`, addressable by
  `sd(lp, ID, c_dummy_k)` on a shared `|ID|` block), `(1 + c | g)` keeps the
  random intercept plus K−1 dummies, and `(0 + factor(c; cmc=false) | g)`
  opts out. BRM merges `(1 | g) + (0 + c | g)` into one block, so the sibling
  intercept keeps `c` treatment-coded there; `(0 + c || g)` is decided on
  its original left-hand side before it is split into uncorrelated terms.

`brm_population_effect_coordinates` reports which coding a block has
(`coding === :cellmeans` or `:treatment`); a cell-mean block has no
`reference_level`, and `cells` pairs each level with its posterior coordinate.
An `r2d2(...)` decomposition allocates its shares over treatment contrasts and
refuses a cell-mean block.

### Term-internal parameters

Some terms own parameters no coefficient address reaches: `s(x)`'s smoothing
scale, `mo(c)`'s Dirichlet increments, `me(x, sd)`'s latent true covariate, a
Gaussian process's length scale and amplitude. They are addressed by naming
the term in the target slot:

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
term_priors_example = (@brm begin
    y ~ Normal(mu, 1.)
    mu ~ 1 + s(age) + mo(dose) + me(w_obs, 0.3) + hsgp(conc; k=5)

    sd(:, s(age))               ~ Exponential(2)     # smoothing SD
    simplex(mu, mo(dose))       ~ Dirichlet(2)       # monotonic increments
    latent(:, me(w_obs))        ~ Normal(0, 5)       # latent true covariate
    length_scale(:, hsgp(conc)) ~ Uniform(0.84, 2)   # GP length scale
    sd(:, hsgp(conc))           ~ Normal(0, 0.5)     # GP marginal amplitude
end)((;
    age=collect(20.0:29.0), dose=repeat(1:5; inner=2),
    w_obs=collect(1.0:10.0), conc=collect(range(-2, 2; length=10)),
    y=collect(range(-1, 1; length=10)),
))
""", :term_priors_example; title="Term-internal priors")
```

The term is spelled as the formula spells it, minus numeric and keyword
arguments: `me(w_obs, 0.3)` is addressed as `me(w_obs)`. `term_priors(brmi)`
returns the captured statements. [Term-internal priors](@ref) has the full
table, the `t2` component slot, why the standardized raw innovations are not
configurable, and the approximation-validity floor an `hsgp` term puts on its
length scale by default.

[Formula terms](@ref) and [Likelihoods](@ref) list the supported syntax and
the backend-specific contracts. The [Gallery](/gallery) shows live examples:
the formula, the emitted SLIC body, the Stan program and the generated
posterior-predictive check in one card.
