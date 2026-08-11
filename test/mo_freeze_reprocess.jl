# test/mo_freeze_reprocess.jl — REGRESSION for snag reprocess-freeze-80ddd7e4.
#
# `reprocess(fit, df; freeze_constants=true, resample_groups=[g])` of a model
# with a `mo(c)` monotonic predictor used to re-emit the increment simplex from
# the NEW frame's level count instead of the FITTED level set. A prediction
# frame carrying a legitimate SUBSET of the training levels (train {1,2,3},
# predict {1,3}) then shrank `simplex[2]` to `simplex[1]` while the frozen
# `<c>_idx` codes still contained `3` — `cumulative_sum(append_row(0, s))[3]`
# indexed a length-2 vector, so `include_tp`/`log_density_gradient` threw
# `vector[multi] indexing: ... index 3 out of range`.
#
# The fix: the `mo` emitter derives its level set (hence the simplex dimension)
# from the recorded `PreprocEntry(:mo)` during frozen re-emission, exactly as
# `_sb_reprocess_entry!` freezes the `<c>_idx` codes — mirroring
# `_sb_hsgp_fit_for_emission`.
#
# Run: julia --project=test test/mo_freeze_reprocess.jl
# (BridgeStan half via StanBlocks; set BRM_PRED_RUNTIME=0 for the
# declaration-level half only.)
using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Normal

const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

builder = @brm begin
    mu ~ 1 + mo(op_diet) + (1 | subject)
    y  ~ Normal(mu, 1.0)
end
# Training: op_diet levels {1,2,3} all present.
train  = (; subject = repeat([1, 2, 3], inner = 2),
            op_diet = repeat([1, 2, 3], outer = 2), y = zeros(6))
# Future: a legitimate SUBSET {1,3} of the training levels; new subjects.
future = (; subject = repeat([101, 203, 307, 409], inner = 2),
            op_diet = repeat([1, 3], outer = 4), y = zeros(8))

fit = SBBRMI(builder(train); mod = @__MODULE__)
to  = reprocess(fit, future; freeze_constants = true, resample_groups = [:subject])

@testset "mo freeze subset — monotonic dimension is the FITTED one" begin
    fit_code = StanBlocks.stan_code(fit.model)
    to_code  = StanBlocks.stan_code(to.model)
    # Fitted 3-level mo => increment simplex[2]; the frozen target keeps it.
    @test occursin("simplex[2]", fit_code)
    @test occursin("simplex[2]", to_code)
    @test !occursin("simplex[1]", to_code)
    # Frozen idx maps the subset through the fitted code, so it still holds a 3.
    @test to.data[:op_diet_idx] == [1, 3, 1, 3, 1, 3, 1, 3]
    # Target stays stanc-valid.
    @test StanBlocks.stanc_check(to_code; warn_pedantic = false).ok
    # freeze_constants=false legitimately RE-DERIVES the level set from the new
    # frame (fresh-fit semantics), so the two-level subset shrinks to simplex[1].
    to_fresh = reprocess(fit, future; freeze_constants = false,
                         resample_groups = [:subject])
    @test occursin("simplex[1]", StanBlocks.stan_code(to_fresh.model))
end

if PRED_RUNTIME
@testset "mo freeze subset — compiled target density + gradient are finite" begin
    to_p = StanBlocks.stan_instantiate(to.model)
    x = zeros(BS.param_unc_num(to_p.model))
    # include_tp + gradient used to throw on the length-2 index.
    @test all(isfinite, BS.param_constrain(to_p.model, x;
                                           include_tp = true, include_gq = false))
    ld, g = BS.log_density_gradient(to_p.model, x)
    @test isfinite(ld)
    @test all(isfinite, g)

    # Named transport onto the resample target stays valid.
    from_p   = StanBlocks.stan_instantiate(fit.model)
    unc_from = BS.param_unc_names(from_p.model)
    unc_to   = BS.param_unc_names(to_p.model)
    @test any(n -> occursin("simplex_incr", n), unc_to)
    rng   = MersenneTwister(41)
    draws = randn(rng, 4, BS.param_unc_num(from_p.model))
    moved = transport_draws(fit, to, draws, unc_from, unc_to;
                            rng = MersenneTwister(7))
    @test size(moved) == (4, length(unc_to))
    @test all(r -> isfinite(BS.log_density(to_p.model, moved[r, :])),
              1:size(moved, 1))
end
end
