# Turing backend

`TuringBRMI` is the direct-BRMI Turing backend. Loading Turing activates the
package extension:

```julia
using BayesianRegressionModels, Turing

brmi = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end)((; x, y))

backend = TuringBRMI(brmi)
chain = sample(backend.model, NUTS(), 1_000)
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
| Population design | **Partial** | Additive intercept plus continuous, non-integer raw columns share `_BRMPopulationDesign` with SBBRMI |
| Population transforms and terms | **Pending** | Categorical contrasts, interactions, `standardize`, `offset`, `mo`/`mo1`, `me`, `s`, `t2`, `gp`, and `hsgp` |
| Population coefficient priors | **Partial** | Independent `Normal(0, 1)` defaults plus `effect(lp, coef)`, `effect(:, coef)`, and `:` coefficient defaults with the same specificity/tie rules as SBBRMI; current Turing hyperparameters must be finite numeric constants |
| Scalar and structured priors | **Partial** | Gaussian scale accepts explicit `Exponential(scale)`; general scalar, horseshoe, simplex, R2D2, term, and latent priors are pending |
| Gaussian identity likelihood | **Supported** | `sigma ~ Exponential(scale)`, `mu ~ 1 + continuous...`, `y ~ Normal(mu, sigma)` |
| Bernoulli-logit likelihood | **Supported** | `eta ~ 1 + continuous...`, `y ~ BernoulliLogit(eta)` |
| Poisson-log likelihood | **Supported** | `log_rate ~ 1 + continuous...`, `y ~ Poisson(exp(log_rate))` |
| Other scalar likelihoods | **Pending** | The built-in catalogue in [Likelihoods](likelihoods.md), including Binomial-logit, negative-binomial, beta-binomial, hurdle/mixture, circular, quantile, and ordinal families |
| Group/random effects | **Pending** | Plain and correlated groups, `|ID|` blocks, centering/CV, stratification, multi-membership, and their SD/correlation/effect priors |
| Response compositions | **Pending** | Truncation, censoring, interval evidence, observation weights, missing-response inference, measurement error, and concise categorical formulas |
| Density decomposition | **Partial** | Turing `logjoint`, `logprior`, and `loglikelihood` are exact for the three supported GLMs; pointwise named log-likelihood outputs are pending |
| Generated quantities | **Partial** | Returns `mu`, `eta`, or `log_rate`/`rate`; BRM-standard predictive draws and output naming are pending |
| Replay and prediction | **Pending** | Frozen preprocessing, new-data replay, population-only prediction, transported group effects, and new-level policy |
| Descriptor/introspection parity | **Pending** | `brm_descriptor`, output coordinates, highlights, and backend capability reporting |

The executable checks live in `test/backend_plan.jl` and
`test/turing_backend.jl`. Each expansion should first add or extend a shared
BRMI-side representation, then add a thin Turing executor and exact density or
postprocessing oracle.

## Current fail-closed boundary

Within the current slice, all of the following are rejected explicitly:

- categorical or integer-coded population columns;
- random effects and group blocks;
- non-Normal or nonconstant population priors, random-effect priors, R2D2, or term-prior overrides;
- response decorators, multiple likelihoods, or extra model statements;
- unsupported links or likelihood families; and
- missing values or mismatched row axes.

This boundary is intentional and temporary: it prevents the Turing backend
from silently looking compatible while assigning a different model.
