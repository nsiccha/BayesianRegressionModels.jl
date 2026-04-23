# label: Bruno SB bug: popefs TP matrix size scope
# tier: 2
# status: open
#=
**SB bug repro (pending SB agent fix).** Pinned to the `pop_loc_loc_n_covariates`
scope issue in StanBlocks: when a `pop_*` reduction lifts the covariate matrix
`X_loc` into `transformed parameters`, the size parameter name it emits
(`pop_loc_loc_n_covariates`) is referenced outside its declaring scope and
Stan compilation fails.

**How to reproduce.** GET
`/pipeline/sb_repro_example?name=sb-bug-popefs-tp-size` — the rendered bug
report's "BridgeStan compile output" section shows the error.

**Minimal formula.** `ftime` is supplied by `bruno-ext.jl` as a parameter-
derived column; referencing it forces the intercept + slope linear predictor
into `transformed parameters`, which triggers the scope bug. `y_scale` on the
assay-indexed log scale matches bruno's production residual structure.

**What's expected once fixed.** The Stan source should compile; the generated
`transformed parameters` block should declare `int pop_loc_loc_n_covariates`
in the same scope where it's consumed (or inline the size expression).
=#

loc ~ 1 + ftime
log(y_scale) ~ 0 + assay_idx
y ~ Normal(loc, y_scale)
