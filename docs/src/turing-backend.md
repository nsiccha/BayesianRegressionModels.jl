# Turing backend

`TuringBRMI` is the direct-BRMI Turing backend. Loading Turing activates the
package extension and turns the backend-neutral plan owned by BRM into a
`DynamicPPL.Model`. Standard Turing inference APIs operate on its `model`
field.

Executable models do not live on this backend-specific page. They are presented
in the backend-neutral [BRM feature atlas](feature-atlas.md), where every example
shows BRM authoring, the emitted StanBlocks model, generated Stan, and the
selected Turing model through the same four-pane comparison.

## Architecture

The extension consumes `BRMI` analysis, materialized data, population designs,
priors, observation semantics, and group-effect plans directly. It does not
construct or inspect `SBBRMI`, `GenerativePlan`, a StanBlocks `SlicModel`, SLIC
IR, or generated Stan. This boundary lets either backend become a weak
dependency without changing BRM semantics.

`TuringBRMI.plan` is the strict, backend-neutral semantic receipt;
`TuringBRMI.model` is the executable Turing model. Unsupported surfaces fail
during construction rather than being approximated silently. The feature atlas
keeps that real construction error visible in the Turing pane.

## Parameterization

Population coefficients use the shared design matrix, labels, and independent
Normal priors. Gaussian scales use an explicit Exponential prior. Group effects
are noncentered: plain random intercepts use a log scale and standard-normal
latent values, correlated slopes use half-normal marginal scales plus an LKJ
Cholesky factor, and `||` uses independent scales with no correlation variable.
Multiple and crossed grouping factors remain separate blocks. Matching
`|ID|` terms in a distributional mean and precision predictor instead share one
joint scale vector, LKJ factor, and standardized group draw; each predictor
consumes its own coefficient slice from that joint covariance block.

Canonical link declarations are lowered once in BRM and reused by the Turing
executor. Response modifiers likewise carry materialized bounds, interval
endpoints, and validation into the extension instead of being rediscovered from
backend code.

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
those preprocessing constants. Reusing existing groups is the fail-closed
default. `resample_groups=:group` explicitly derives that grouping coordinate
from `new_data`, keeps fitted population, scale, and correlation parameters,
and regenerates only the named groups' standardized effects. For a shared
`|ID|` block this redraw remains joint across its predictors. Unseen categorical
levels still fail closed.

## Parity contract

“Parity” means the same admitted BRMI has the same constrained prior and
likelihood semantics, coefficient addressing, preprocessing behavior, and
observable outputs in both backends. Producing a model that merely samples is
not sufficient.

| BRMI surface | Turing status | Current contract |
| --- | --- | --- |
| Backend boundary | **Supported** | Direct `BRMI` → backend-neutral plan → Turing extension; core loads without Turing |
| Population design | **Partial** | Intercepts, raw and fitted numeric transforms, pure data expressions, offsets, treatment contrasts, and continuous/categorical interactions |
| Population priors | **Partial** | Independent Normal defaults plus shared `effect(...)` specificity and addressing semantics |
| Gaussian identity | **Supported** | Population and admitted grouped predictors with explicit Exponential scale prior |
| Bernoulli/Binomial logit | **Supported** | Canonical linked declarations and explicit stable-logit families |
| Poisson log | **Supported** | Canonical linked declarations, data offsets, and admitted grouped predictors |
| NegativeBinomial2 | **Supported subset** | Shared mean/precision population plans, independent multiple grouping blocks, and a joint cross-predictor `|ID|` covariance block |
| BetaBinomial2 | **Supported subset** | Shared mean/precision population plans, independent multiple grouping blocks, and a joint cross-predictor `|ID|` covariance block |
| Group effects | **Partial** | Plain intercepts, correlated and exact-zero-correlation slopes, multiple/crossed factors, distributional cross-predictor `|ID|` covariance, fitted transformed/categorical slopes, and `sd`/`cor` prior overrides; `gr(by=)`, multi-membership, and centered/adaptive geometry remain pending |
| Response evidence | **Partial** | Truncated, censored, and interval-censored Normal/Poisson observations; wider modifiers remain pending |
| Missing responses | **Supported subset** | `mi(y) ~ Normal(mu, sigma)` imputes missing rows from the same conditional family and keeps observed rows in the likelihood |
| Multiple responses | **Supported subset** | Independent blocks are namespaced; exactly compatible shared predictors/group blocks are sampled once and reused, while partial or incompatible overlaps fail closed |
| Observation weights | **Supported subset** | Analytic Normal weights rescale sigma; frequency and power weights scale density while predictive draws retain the base distribution |
| Advanced terms | **Pending** | Splines, `t2`, `mo`, `me`, GP/HSGP, and kernel/ragged models fail loudly |
| Outputs | **Supported subset** | Row-aligned pointwise likelihoods, deterministic returned quantities, one-draw posterior prediction, and chain-level Turing prediction; fitted response latents are excluded before regeneration |
| Replay | **Supported subset** | Frozen preprocessing and existing-group coordinates replay on new rows; refitting constants and selective new-group resampling—including joint `|ID|` redraws—are explicit |

The matrix is intentionally an overview. The build-generated examples and
their current unsupported reasons are the executable source of truth.
