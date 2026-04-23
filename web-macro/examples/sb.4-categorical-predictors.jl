# label: sb.4 categorical predictors (c1, c2, ...) — needs impl
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**Status: not implemented.** `_sb_predictor_col(t::NamedColumn, data)` errors when the backing vector isn't `AbstractVector{<:Real}`.

**What to implement.** For a categorical column `c`:
1. At sbimpl time, one-hot encode the levels (drop first for reference coding, matching brms default).
2. Stash the one-hot matrix under `data[:<name>_dummies]` (or just the integer level vector + lookup in Stan).
3. Emit additional columns into the `hcat(...)` design matrix.

**Alternative.** Pass integer levels and let Stan construct the dummy matrix via a `@deffun` helper — keeps the Julia side thin, makes recompilation-on-data-size avoidable.

**Verification.** `loc ~ 1 + c1` should produce `K-1` extra columns (where `K = length(levels(c1))`). Posterior dim should match the `cimpl` result.

=#

loc ~ 1 + a + c1 + c2
log(err) ~ 1
y1 ~ Normal(loc, err)
