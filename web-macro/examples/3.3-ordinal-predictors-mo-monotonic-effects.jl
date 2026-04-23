# label: 3.3 ordinal predictors mo() (monotonic effects)
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_instantiate,stan_shapes,transform,wrap
#=
**What it is.** brms's `mo(x)` for an ordinal predictor with K levels: instead of `K-1` independent treatment-coded coefficients, fit a single "total effect" β plus a `K-1`-dim simplex of inter-level shape. Forces the effect to be monotonic in the ordering of `x`'s levels.

**Why it matters.** Likert-scale predictors and ordered categorical inputs (e.g. age groups) have a natural ordering that treatment coding ignores. `mo()` enforces the monotonicity prior, dramatically reducing the parameter count and tightening posterior inference.

**Implementation.** New block layout: one β coefficient + one Dirichlet-distributed simplex of length `K-1`. `_cat_lookup`-style materialization but the per-level contribution is `β * cumulative_sum(simplex)[level]` instead of `coefficients[level]`.

Needs a new prior block type (Dirichlet) in `lprior!`, plus new parser support for `mo(x)` and a new `vmeta_sampling_rhs` overload.

**Verification.** Preset against synthetic data where the true effect is monotonic but the levels are unordered in the data. Compare fitted shape parameters against the synthetic monotonic curve.

=#

loc ~ 1 + mo(c3)
y1 ~ Normal(loc, 1)
