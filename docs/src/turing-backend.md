# Turing backend

`TuringBRMI` is the direct-BRMI Turing backend. Loading Turing activates the
package extension:

For a visual three-way comparison of the direct Turing route and the
StanBlocks/Stan route, open the [backend lowering explorer](backend-lowering.md).

```julia
using BayesianRegressionModels, Turing
using Distributions: Binomial
using LogExpFunctions: logit

gaussian_brmi = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    effect(:, :) ~ Normal(0, 2)
    effect(mu, x) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)((;
    x = [-1.0, 0.5, 2.0],
    y = [0.2, 1.1, -0.4],
))

gaussian = TuringBRMI(gaussian_brmi)
gaussian.plan.beta_location, gaussian.plan.beta_scale
```

The backend owns an ordinary `DynamicPPL.Model`, so standard Turing workflows
apply directly:

```julia
chain = sample(gaussian.model, NUTS(), 1_000)
```

The same direct-BRMI path covers the currently supported binary and count
GLMs—there is no intermediate Stan or SLIC model:

```julia
binary = TuringBRMI((@brm begin
    logit(p) ~ 1 + x
    y ~ Bernoulli(p)
end)((; x=[-1.0, 0.5, 2.0, 0.25], y=[0, 1, 1, 0])))

grouped_binary = TuringBRMI((@brm begin
    logit(p) ~ 1 + x
    y ~ Binomial(trials, p)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    trials=[2, 4, 6, 3],
    y=[0, 2, 5, 1],
)))

count = TuringBRMI((@brm begin
    log(lambda) ~ 1 + x
    y ~ Poisson(lambda)
end)((; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])))

overdispersed_count = TuringBRMI((@brm begin
    log(mu) ~ 1 + x
    log(phi) ~ 1
    y ~ BayesianRegressionModels.NegativeBinomial2(mu, phi)
end)((; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])))

(binary=summary(binary.model),
 grouped_binary=summary(grouped_binary.model),
 count=summary(count.model),
 overdispersed_count=summary(overdispersed_count.model))
```

Fitted numeric transforms use the same design and coefficient labels as the
StanBlocks backend. The fitted mean and sample standard deviation belong to the
model plan rather than to Turing:

```julia
transformed = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + zscale(x) + center(w) + zscale(x) & w
    effect(mu, zscale_x) ~ Normal(0, 0.25)
    effect(mu, int_zscale_x_x_w) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    x=[1.0, 2.0, 4.0],
    w=[-2.0, 1.0, 5.0],
    y=[0.2, 1.1, -0.4],
)))

transformed.plan.design.matrix
```

Integer-coded and `CategoricalVector` population predictors use ordered
treatment contrasts with the first level as reference. A single
`effect(mu, group)` prior applies to every non-reference contrast, matching the
StanBlocks categorical block:

```julia
categorical = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + group + x + x & group
    effect(mu, group) ~ Normal(0, 0.5)
    effect(mu, int_x_x_group_lvl_2) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
)))
```

`factor(group; ref=k)` uses the same reference-level swap and address aliases
as StanBlocks. The user-facing column name still addresses the whole contrast
block:

```julia
reference_level = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + factor(group; ref=3)
    effect(mu, group) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
)))
```

Pure numeric expressions remain coefficient-bearing population terms, while
`offset(...)` contributes with coefficient one. This makes the usual
exposure-offset count model direct:

```julia
exposure_model = TuringBRMI((@brm begin
    log_rate ~ 1 + log(x) + offset(log(exposure))
    y ~ Poisson(exp(log_rate))
end)((;
    x=[1.0, 2.0, 4.0],
    exposure=[2.0, 4.0, 8.0],
    y=[0, 2, 5],
)))
```

Fitted numeric transforms use the same design and coefficient labels as the
StanBlocks backend. The fitted mean and sample standard deviation belong to the
model plan rather than to Turing:

```julia
transformed = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + zscale(x) + center(w) + zscale(x) & w
    effect(mu, zscale_x) ~ Normal(0, 0.25)
    effect(mu, int_zscale_x_x_w) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    x=[1.0, 2.0, 4.0],
    w=[-2.0, 1.0, 5.0],
    y=[0.2, 1.1, -0.4],
)))

transformed.plan.design.matrix
```

Integer-coded and `CategoricalVector` population predictors use ordered
treatment contrasts with the first level as reference. A single
`effect(mu, group)` prior applies to every non-reference contrast, matching the
StanBlocks categorical block:

```julia
categorical = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + group + x + x & group
    effect(mu, group) ~ Normal(0, 0.5)
    effect(mu, int_x_x_group_lvl_2) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
)))
```

`factor(group; ref=k)` uses the same reference-level swap and address aliases
as StanBlocks. The user-facing column name still addresses the whole contrast
block:

```julia
reference_level = TuringBRMI((@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + factor(group; ref=3)
    effect(mu, group) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
)))
```

Pure numeric expressions remain coefficient-bearing population terms, while
`offset(...)` contributes with coefficient one. This makes the usual
exposure-offset count model direct:

```julia
exposure_model = TuringBRMI((@brm begin
    log_rate ~ 1 + log(x) + offset(log(exposure))
    y ~ Poisson(exp(log_rate))
end)((;
    x=[1.0, 2.0, 4.0],
    exposure=[2.0, 4.0, 8.0],
    y=[0, 2, 5],
)))
```

Turing is a weak dependency. The core package owns a backend-neutral BRMI
context and population-design plan; the extension turns that plan into a
`DynamicPPL.Model`. It does not construct or inspect `SBBRMI`,
`GenerativePlan`, `StanBlocks.SlicModel`, emitted SLIC, or Stan code.

## Outputs and replay

`turing_pointwise_loglikelihoods` returns response-named, row-aligned
log-likelihood vectors; latent rows of a partly missing response remain
`missing`. `turing_generated_quantities` evaluates the model's deterministic
return value at one constrained draw. `turing_posterior_predictive` regenerates
every response row at one constrained draw, including rows that were latent in
the fitted model. For a fitted chain, `Turing.predict(backend, chain)` performs
the same response-latent exclusion before running DynamicPPL's chain-level
prediction.

`reprocess(backend, new_data)` rebuilds the direct BRMI plan on new rows while
reusing fitted centers, scales, categorical coordinates, interactions, offsets,
and existing group coordinates. `freeze_constants=false` explicitly refits
those preprocessing constants. Reusing existing groups is supported; unseen
categorical levels fail closed. For a new population,
`resample_groups=:subject` takes the selected group's levels from `new_data` and
posterior prediction redraws only its standardized effects while retaining the
fitted scales and correlation factors. The same exclusion applies to
`Turing.predict(backend, chain)`, so fitted group latents are not silently
reused for a resampled block.

```julia
using Random: Xoshiro

training_grouped = TuringBRMI((@brm begin
    log(lambda) ~ 1 + x + (1 | subject)
    y ~ Poisson(lambda)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=["a", "b", "a", "c"],
    y=[0, 2, 5, 1],
)))

new_population = (;
    x=[-0.75, 0.25, 1.25, 2.25],
    subject=["new_b", "new_a", "new_b", "new_c"],
    y=zeros(Int, 4),
)
replayed = reprocess(
    training_grouped, new_population; resample_groups=:subject)

# A constrained draw from the fitted model. Its old `z_group` values are
# ignored for the named resampled block; beta and the fitted scale are retained.
posterior_draw = (;
    beta_pop=[0.1, -0.2],
    log_group_scale=log(0.6),
    z_group=[-0.2, 0.4, 1.1],
)
predicted = turing_posterior_predictive(
    Xoshiro(42), replayed, posterior_draw)

(levels=only(replayed.plan.random_effects).levels, y=predicted.y)
```

## Parity contract

“Parity” means the same admitted BRMI has the same constrained prior and
likelihood semantics, coefficient addressing, preprocessing/replay behavior,
and post-fit outputs in both backends. Merely producing a model that samples is
not parity. Unsupported rows remain loud errors until all of those contracts
are covered.

The matrix below is the durable worklist. **Supported** means exact executable
tests exist. **Partial** names the precise admitted subset. **Pending** means
the Turing backend refuses the surface rather than approximating it.

| BRMI surface | Turing status | Current contract / next shared seam |
| --- | --- | --- |
| Backend boundary | **Supported** | Direct `BRMI` → backend-neutral plan → Turing extension; core loads without Turing |
| Population design | **Partial** | Intercepts, raw and fitted numeric transforms, pure data expressions, offsets, treatment contrasts, and continuous/categorical interactions |
| Population priors | **Partial** | Independent Normal defaults plus shared `effect(...)` specificity and addressing semantics |
| Gaussian identity | **Supported** | Population and admitted grouped predictors with explicit Exponential scale prior |
| Bernoulli/Binomial logit | **Supported** | Canonical linked declarations and explicit stable-logit families |
| Poisson log | **Supported** | Canonical linked declarations, data offsets, and admitted grouped predictors |
| NegativeBinomial2 | **Supported subset** | Shared mean/precision population plans and one admitted group block per predictor |
| BetaBinomial2 | **Supported subset** | Shared mean/precision population plans and one admitted group block per predictor |
| Group effects | **Partial** | Plain intercepts, one correlated raw-continuous slope block, and admitted exact-zero-correlation blocks |
| Response evidence | **Partial** | Truncated, censored, and interval-censored Normal/Poisson observations; wider modifiers remain pending |
| Missing responses | **Supported subset** | `mi(y) ~ Normal(mu, sigma)` imputes missing rows from the same conditional family and keeps observed rows in the likelihood |
| Multiple responses | **Supported subset** | Independent blocks are namespaced; exactly compatible shared predictors/group blocks are sampled once and reused, while partial or incompatible overlaps fail closed |
| Observation weights | **Supported subset** | Analytic Normal weights rescale sigma; frequency and power weights scale density while predictive draws retain the base distribution |
| Advanced terms | **Pending** | Splines, `t2`, `mo`, `me`, GP/HSGP, shared-ID covariance, multiple/crossed blocks, and kernel/ragged models fail loudly |
| Outputs | **Supported subset** | Row-aligned pointwise likelihoods, deterministic returned quantities, one-draw posterior prediction, and chain-level Turing prediction; fitted response latents are excluded before regeneration |
| Replay | **Supported subset** | Frozen preprocessing and existing-group coordinates replay on new rows; refitting constants is explicit; `resample_groups` rebuilds selected level axes and redraws their standardized effects while retaining fitted covariance parameters |

The executable checks live in `test/backend_plan.jl` and
`test/turing_backend.jl`. Each expansion should first add or extend a shared
BRMI-side representation, then add a thin Turing executor and exact density or
postprocessing oracle.

## Current fail-closed boundary

Within the current slice, all of the following are rejected explicitly:

- string/object population columns that are not explicit `CategoricalVector`s;
- multiple/crossed group blocks, shared `|ID|` covariance, transformed or
  categorical random slopes, and advanced group structures;
- non-Normal or nonconstant population priors, random-effect prior overrides,
  R2D2, and term-prior overrides;
- advanced terms such as splines, `t2`, `mo`, `me`, GP/HSGP, and kernel/ragged
  models; and
- unsupported links, likelihood families, evidence/weight compositions, or
  mismatched row axes outside the subsets listed above.

This boundary is intentional and temporary: it prevents the Turing backend
from silently looking compatible while assigning a different model.
