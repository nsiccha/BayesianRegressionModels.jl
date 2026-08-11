# test/transport_resample_target.jl — REGRESSION for snag transport-draws-99e66a24.
#
# `transport_draws(fitted, reprocess(fitted, df; resample_groups=[g]), …)` used
# to die with `MethodError: iterate(::StanExpr)`. `resample_groups` moves g's
# standardised draws to GENERATED QUANTITIES, so g's group index is a Stan-side
# `maybecv(...)` expression and its z coords leave `param_unc_names`.
# `ranef_blocks` used to `maximum(...)` that expression; even past that,
# `transport_draws` tried to resolve the now-absent coordinates.
#
# The fix: `ranef_blocks` reads the resample-marked block's count/levels off the
# `:group_index` preprocessing record and flags it `generated`; `ranef_blocks`
# refuses `ranef_coordinates` on it; `transport_draws` SKIPS it and copies the
# fitted covariance (`L`/`tau`) + population coefficients BY NAME, so the target
# re-draws the new subjects from the FITTED covariance.
#
# Run: julia --project=<worktree> test/transport_resample_target.jl
# (needs BridgeStan through StanBlocks; set BRM_PRED_RUNTIME=0 for the
# declaration-level half only.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Normal

const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

rsp_builder = @brm begin
    mu ~ 1 + zscale(zage) + (1 + zage | p | subject)
    y  ~ Normal(mu, 1.0)
end
rsp_train  = (; subject = repeat([1, 2, 3], inner = 2),
                zage = collect(range(-1.0, 1.0; length = 6)), y = zeros(6))
rsp_future = (; subject = repeat([101, 203, 307, 409], inner = 2),
                zage = collect(range(10.0, 20.0; length = 8)), y = zeros(8))

rsp_fit = SBBRMI(rsp_builder(rsp_train); mod = @__MODULE__)
rsp_to  = reprocess(rsp_fit, rsp_future; freeze_constants = true,
                    resample_groups = [:subject])

@testset "ranef_blocks — resample target is DESCRIBED, off the preproc record" begin
    # Used to raise `MethodError: iterate(::StanExpr)`.
    rb = only(ranef_blocks(rsp_to))
    @test rb.binding === :b_p_subject
    @test rb.generated
    @test (rb.n_terms, rb.n_groups) == (2, 4)     # count from the preproc record
    @test rb.levels == [101, 203, 307, 409]       # levels from the preproc record
    # An ordinary fit's block stays a sampled parameter.
    @test only(ranef_blocks(rsp_fit)).generated == false
    # A generated block refuses coordinate resolution, with a clear message.
    @test_throws "generated quantities" ranef_coordinates(rb, String[])
    # The target is stanc-valid (the reporter's precondition).
    @test StanBlocks.stanc_check(StanBlocks.stan_code(rsp_to.model);
                                 warn_pedantic = false).ok
end

@testset "ranef_blocks — a plain cv_groups build is UNCHANGED (still a parameter)" begin
    # The distinguishing case: `cv_groups` at construction sizes `n_g` from
    # `max(idx)` but does NOT mark the index, so the block stays a parameter.
    cv_sb = SBBRMI(rsp_builder(rsp_train); mod = @__MODULE__, cv_groups = [:subject])
    cv = only(ranef_blocks(cv_sb))
    @test cv.generated == false
    @test (cv.n_terms, cv.n_groups) == (2, 3)     # sized from maximum(subject_idx)
    @test occursin("b_p_subject_n_g = max(subject_idx)",
                   StanBlocks.stan_code(cv_sb.model))
end

if !PRED_RUNTIME
    @info "BRM_PRED_RUNTIME=0 — skipping the compiled-model half"
else

@testset "transport_draws — named transport onto the compiled resample target" begin
    from_p = StanBlocks.stan_instantiate(rsp_fit.model)
    to_p   = StanBlocks.stan_instantiate(rsp_to.model)
    unc_from = BS.param_unc_names(from_p.model)
    unc_to   = BS.param_unc_names(to_p.model)

    # The resampled z coords leave the target parameter vector; the covariance
    # hyperparameters remain.
    @test any(n -> occursin("b_p_subject_z_flat", n), unc_from)   # in the source
    @test !any(n -> occursin("b_p_subject_z_flat", n), unc_to)    # GONE from target
    @test any(n -> occursin("b_p_subject_L", n), unc_to)
    @test any(n -> occursin("b_p_subject_tau", n), unc_to)

    rng = MersenneTwister(41)
    draws = randn(rng, 6, BS.param_unc_num(from_p.model))
    moved = transport_draws(rsp_fit, rsp_to, draws, unc_from, unc_to;
                            rng = MersenneTwister(7))
    @test size(moved) == (6, length(unc_to))

    # Every target coordinate is a by-NAME copy of the matching source coordinate
    # (population coefficients + the resampled block's L/tau). Nothing invented,
    # nothing positionally spliced.
    pos_from = Dict(String(n) => i for (i, n) in enumerate(unc_from))
    for (j, nm) in enumerate(unc_to)
        @test moved[:, j] == draws[:, pos_from[String(nm)]]
    end

    # The transported draws are a valid input to the compiled target, which
    # re-draws the new subjects in its generated quantities.
    @test all(r -> isfinite(BS.log_density(to_p.model, moved[r, :])), 1:size(moved, 1))
    @test any(n -> occursin("b_p_subject_z_flat", n),
              BS.param_names(to_p.model; include_tp = false, include_gq = true))
end

end # PRED_RUNTIME
