# Turing backend

`TuringBRMI` is the direct-BRMI Turing backend. Loading Turing activates the
package extension:

```julia
using BayesianRegressionModels, Turing

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
    eta ~ 1 + x
    y ~ BernoulliLogit(eta)
end)((; x=[-1.0, 0.5, 2.0, 0.25], y=[0, 1, 1, 0])))

grouped_binary = TuringBRMI((@brm begin
    eta ~ 1 + x
    y ~ BayesianRegressionModels.BinomialLogit(trials, eta)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    trials=[2, 4, 6, 3],
    y=[0, 2, 5, 1],
)))

count = TuringBRMI((@brm begin
    log(lambda) ~ 1 + x
    y ~ Poisson(lambda)
end)((; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])))

(binary=summary(binary.model),
 grouped_binary=summary(grouped_binary.model),
 count=summary(count.model))
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
| Observation topology | **Partial** | Exactly one direct response named `y`; arbitrary names, multiple responses, distributional predictors, and hierarchical/ragged axes are pending |
| Population design | **Partial** | Additive intercept and continuous raw/fitted-transform columns share `_BRMPopulationDesign` with SBBRMI; ordered treatment contrasts reuse SBBRMI's level-coding primitive and effect-address semantics |
| Population transforms and terms | **Partial** | Numeric data expressions, fixed data-derived `offset`, fitted transforms, continuous/categorical interactions, treatment contrasts, and integer `factor(...; ref=k)` are supported; sampled-parameter offsets, `mo`/`mo1`, `me`, `s`, `t2`, `gp`, and `hsgp` are pending |
| Population coefficient priors | **Partial** | Independent `Normal(0, 1)` defaults plus `effect(lp, coef)`, `effect(:, coef)`, and `:` coefficient defaults with the same specificity/tie rules as SBBRMI; current Turing hyperparameters must be finite numeric constants |
| Scalar and structured priors | **Partial** | Gaussian scale accepts explicit `Exponential(scale)`; general scalar, horseshoe, simplex, R2D2, term, and latent priors are pending |
| Gaussian identity likelihood | **Supported** | `sigma ~ Exponential(scale)`, `mu ~ 1 + continuous...`, `y ~ Normal(mu, sigma)` |
| Bernoulli-logit likelihood | **Supported** | `eta ~ 1 + continuous...`, `y ~ BernoulliLogit(eta)` |
| Binomial-logit likelihood | **Supported** | `eta ~ 1 + continuous...`, `y ~ BinomialLogit(trials, eta)` with constant or row-wise nonnegative integer trials |
| Poisson-log likelihood | **Supported** | Canonical `log(lambda) ~ formula; y ~ Poisson(lambda)` and explicit linked-scale `log_rate ~ formula; y ~ Poisson(exp(log_rate))` share one predictor/link plan |
| Other scalar likelihoods | **Pending** | The built-in catalogue in [Likelihoods](likelihoods.md), including negative-binomial, beta-binomial, hurdle/mixture, circular, quantile, and ordinal families |
| Group/random effects | **Pending** | Plain and correlated groups, `|ID|` blocks, centering/CV, stratification, multi-membership, and their SD/correlation/effect priors |
| Response compositions | **Pending** | Truncation, censoring, interval evidence, observation weights, missing-response inference, measurement error, and concise categorical formulas |
| Density decomposition | **Partial** | Turing `logjoint`, `logprior`, and `loglikelihood` are exact for the four supported GLMs; pointwise named log-likelihood outputs are pending |
| Generated quantities | **Partial** | Returns `mu`, `eta`, or `log_rate`/`rate`; BRM-standard predictive draws and output naming are pending |
| Replay and prediction | **Pending** | Frozen preprocessing, new-data replay, population-only prediction, transported group effects, and new-level policy |
| Descriptor/introspection parity | **Pending** | `brm_descriptor`, output coordinates, highlights, and backend capability reporting |

The executable checks live in `test/backend_plan.jl` and
`test/turing_backend.jl`. Each expansion should first add or extend a shared
BRMI-side representation, then add a thin Turing executor and exact density or
postprocessing oracle.

## Current fail-closed boundary

Within the current slice, all of the following are rejected explicitly:

- string/object population columns that are not explicit `CategoricalVector`s;
- random effects and group blocks;
- non-Normal or nonconstant population priors, random-effect priors, R2D2, or term-prior overrides;
- response decorators, multiple likelihoods, or extra model statements;
- unsupported links or likelihood families beyond Normal-identity, Bernoulli-logit, Binomial-logit, and Poisson-log; and
- missing values or mismatched row axes.

This boundary is intentional and temporary: it prevents the Turing backend
from silently looking compatible while assigning a different model.
