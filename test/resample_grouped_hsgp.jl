# test/resample_grouped_hsgp.jl — new levels of a grouped HSGP on replay.
#
# `reprocess(sb, rows; resample_groups=[:g])` draws a new level of
# `hsgp(x...; by=g)` as it draws a new level of a `(… | g)` block: its
# standardized basis weights are drawn `Normal(0, 1)` in generated quantities
# and scaled by the spectral density at the draw's fitted length scale and
# amplitude, the law of a fitted level. A per-level hyper-predictor
# (`log(sd(hsgp(x))) ~ 1 + (1 | g)`) draws the new level's hyper effect from
# the fitted SD. `cor=` and `orthogonal_to=` have no value for a new level and
# are refused, as is resampling the grouping on RKBRMI, which has no input for
# new levels' weights (snag `resample-groups-6a6b2d58`).
#
# Set `BRM_RESAMPLE_HSGP_RUNTIME=0` to skip the compiled-model sets.

using Test
using Random
using Statistics
using LinearAlgebra
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal, LKJCholesky
const BRM = BayesianRegressionModels
const RUNTIME = get(ENV, "BRM_RESAMPLE_HSGP_RUNTIME", "1") != "0"

shared_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5)
    y ~ Normal(mu, sigma)
end

# A per-level amplitude beside an ordinary random intercept on the same
# grouping: both draw their new levels.
hyper_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5) + (1 | subject)
    log(sd(hsgp(t))) ~ 1 + (1 | subject)
    y ~ Normal(mu, sigma)
end

cor_builder = @brm begin
    sigma ~ Exponential(1)
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5, cor = Omega)
    y ~ Normal(mu, sigma)
end

projected_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5, orthogonal_to = :intercept)
    y ~ Normal(mu, sigma)
end

const T_GRID = collect(range(0.0, 10.0; length = 6))
train = (; subject = repeat(["s1", "s2", "s3"], inner = 6), t = repeat(T_GRID, 3),
         y = sin.(repeat(T_GRID, 3)) .+ 0.1 .* (1:18))
future = (; subject = repeat(["n1", "n2", "n3", "n4"], inner = 2),
          t = repeat([2.0, 7.0], 4))

section(code, from, to) =
    code[first(findfirst(from, code)):first(findfirst(to, code))]
params_block(code) = section(code, "parameters {", "model {")
gq_block(code) = code[first(findfirst("generated quantities {", code)):end]

@testset "SBBRMI: new levels of a grouped HSGP are drawn in generated quantities" begin
    fitted = SBBRMI(shared_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    @test replay.data[:n_subject] == 4
    @test StanBlocks.stan_data(replay.model)[:subject_idx] == repeat(1:4, inner = 2)
    @test replay.preproc[:subject_idx].const_.levels == ["n1", "n2", "n3", "n4"]
    # The fitted basis interval is kept and evaluated at the new rows.
    @test replay.preproc[:PHI_hsgp_t_by_subject].const_.fits ==
          fitted.preproc[:PHI_hsgp_t_by_subject].const_.fits
    @test size(replay.data[:PHI_hsgp_t_by_subject]) == (8, 6)

    code = StanBlocks.stan_code(replay.model)
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    @test occursin("b_hsgpw_t_subject_n_g = max(subject_idx)", code)
    @test !occursin("zflat_hsgpw_t_subject", params_block(code))
    @test occursin("zflat_hsgpw_t_subject = std_normal_vector_rng", gq_block(code))

    outputs = brm_descriptor(replay).outputs
    kinds = Dict(o.name => o.kind for o in outputs)
    fitted_parameters = Set(o.name for o in brm_descriptor(fitted).outputs
                            if o.kind === :parameter)
    replay_parameters = Set(o.name for o in outputs if o.kind === :parameter)
    @test issubset(replay_parameters, fitted_parameters)
    @test setdiff(fitted_parameters, replay_parameters) == Set([:zflat_hsgpw_t_subject])
    for name in (:zflat_hsgpw_t_subject, :b_hsgpw_t_subject, :mu, :y_gen)
        @test kinds[name] === :generated_quantity
    end

    # The ordinary build is unchanged: the data-sized block stays sampled.
    @test !occursin("_n_g", StanBlocks.stan_code(fitted.model))

    # `cv_groups` alone sizes the block by `max(subject_idx)` but does not
    # mark the index, so the weights stay parameters.
    cv = SBBRMI(shared_builder(train); mod = @__MODULE__, cv_groups = [:subject])
    cv_code = StanBlocks.stan_code(cv.model)
    @test occursin("b_hsgpw_t_subject_n_g = max(subject_idx)", cv_code)
    @test occursin("zflat_hsgpw_t_subject", params_block(cv_code))

    # The other public spellings of the same replay.
    @test restan_data(fitted, future; resample_groups = [:subject]) ==
          StanBlocks.stan_data(replay.model)
    descriptor = brm_descriptor(shared_builder, train; mod = @__MODULE__)
    executed = brm_execute(descriptor, :reprocess, future; resample_groups = [:subject])
    @test StanBlocks.stan_code(executed.plan.model) == code
    # A fresh rebuild is fitted on the rows, so it needs their response.
    fresh = reprocess(fitted, merge(future, (; t = repeat([1.0, 9.0], 4), y = zeros(8)));
                      freeze_constants = false, resample_groups = [:subject])
    @test fresh.preproc[:PHI_hsgp_t_by_subject].const_.fits !=
          fitted.preproc[:PHI_hsgp_t_by_subject].const_.fits
    @test !occursin("zflat_hsgpw_t_subject", params_block(StanBlocks.stan_code(fresh.model)))
end

@testset "SBBRMI: a per-level hyper-predictor draws the new level's hyper effect" begin
    fitted = SBBRMI(hyper_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    code = StanBlocks.stan_code(replay.model)
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    params, gq = params_block(code), gq_block(code)
    # New levels: the basis weights, the hyper effect's standardized draw and
    # the ordinary intercept's.
    for name in ("zflat_hsgpw_t_subject", "hsgp_t_by_subject_z_sigma", "r_mu_subject_xi")
        @test !occursin(name, params)
        @test occursin(name, gq)
    end
    # Fitted: the hyper intercept and SD, the shared length scale, the
    # intercept's SD.
    for name in ("hsgp_t_by_subject_beta0_sigma", "hsgp_t_by_subject_sd_sigma",
                 "hsgp_t_by_subject_rho_iso")
        @test occursin(name, params)
    end
end

@testset "new levels of a construction fitted per level" begin
    # Not built: new levels of a level-correlated grouping need the
    # correlation over a larger level set (todo 0lopk07); refused by name.
    with_cor = SBBRMI(cor_builder(train); mod = @__MODULE__)
    @test_throws "whose levels" reprocess(with_cor, future; resample_groups = [:subject])
    @test_broken (reprocess(with_cor, future; resample_groups = [:subject]); true)
    # refused: `orthogonal_to=` is fitted on each level's training rows and
    # frozen, so a row's value never depends on the other replay rows
    # (decision 15hn91t); a new level has no training rows to fit it on
    projected = SBBRMI(projected_builder(train); mod = @__MODULE__)
    @test_throws "projection is undefined" reprocess(
        projected, future; resample_groups = [:subject])
    # Their frozen replays of the fitted levels are unchanged.
    @test StanBlocks.stan_code(reprocess(projected, train).model) ==
          StanBlocks.stan_code(projected.model)
end

using ReactiveKernels, ReactiveKernelsPPL

@testset "RKBRMI: new levels of a grouped HSGP" begin
    # Not built on RKBRMI (todo 10h1myt): the term draws its weights inside
    # its block, with no input for new levels'. Refused by name meanwhile,
    # because a design reusing the fitted labels kept the fit's program and
    # read the fitted levels' weights as the new levels' (snag
    # resample-groups-6a6b2d58).
    reused = (; subject = repeat(["s1", "s2", "s3"], inner = 2), t = repeat([2.0, 7.0], 3))
    for builder in (shared_builder, hyper_builder)
        rk = RKBRMI(builder(train))
        for design in (reused, future)
            @test_throws "grouped `hsgp(...; by=...)`" reprocess(
                rk, design; resample_groups = [:subject])
            @test_broken (reprocess(rk, design; resample_groups = [:subject]); true)
        end
        state = brm_fitted_state(rk)
        @test_throws "grouped `hsgp(...; by=...)`" RKBRMI(
            builder(future), state; resample_groups = [:subject])
        @test_throws "grouped `hsgp(...; by=...)`" rk_design_binder(
            rk; resample_groups = [:subject])(future)
        # Frozen replays of the fitted levels are unchanged.
        @test reprocess(rk, (; subject = ["s2", "s3"], t = [2.0, 7.0])) isa RKBRMI
    end
end

if !RUNTIME
    @info "BRM_RESAMPLE_HSGP_RUNTIME=0 — skipping the compiled-model sets"
else

import BridgeStan as BS

function constrained_by_name(sm, theta; seed = nothing)
    include_gq = !isnothing(seed)
    names = BS.param_names(sm; include_tp = true, include_gq)
    values = BS.param_constrain(sm, theta; include_tp = true, include_gq,
        rng = include_gq ? BS.StanRNG(sm, seed) : nothing)
    Dict(n => v for (n, v) in zip(names, values))
end
entries(d, prefix) = [d[k] for k in sort!([k for k in keys(d) if startswith(k, prefix * ".")];
                                          by = k -> parse(Int, split(k, '.')[end]))]

# The prior variance of a level's field at each row, given the draw's length
# scale and amplitude: sum_j PHI[i, j]^2 S(omega_j).
field_variance(PHI, omega2, sigma, rho) =
    vec(sum(PHI .^ 2 .* (brm_hsgp_sqrt_spd(omega2, sigma, [rho]) .^ 2)'; dims = 2))

@testset "SBBRMI: a new level's field has the fitted law" begin
    fitted = SBBRMI(shared_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    fit_model = StanBlocks.stan_instantiate(fitted.model).model
    rp_model = StanBlocks.stan_instantiate(replay.model).model
    unc_fit, unc_rp = BS.param_unc_names(fit_model), BS.param_unc_names(rp_model)
    draws = 0.3 .* randn(Xoshiro(3), 2, length(unc_fit))
    moved = transport_draws(fitted, replay, draws, unc_fit, unc_rp)
    @test size(moved) == (2, length(unc_rp))
    position = Dict(n => i for (i, n) in enumerate(unc_fit))
    @test moved == draws[:, [position[n] for n in unc_rp]]

    PHI = replay.data[:PHI_hsgp_t_by_subject]
    omega2 = replay.data[:omega2_hsgp_t_by_subject]
    theta = moved[1, :]
    fixed = constrained_by_name(rp_model, theta)
    rho = fixed["hsgp_t_by_subject_rho_iso"]
    sigma = fixed["hsgp_t_by_subject_sigma"]
    expected = field_variance(PHI, omega2, sigma, rho)

    n = 4000
    field = Matrix{Float64}(undef, n, 8)
    z = Matrix{Float64}(undef, n, 4 * 6)
    for s in 1:n
        values = constrained_by_name(rp_model, theta; seed = s)
        field[s, :] = entries(values, "hsgp_t_by_subject")
        z[s, :] = entries(values, "zflat_hsgpw_t_subject")
        # The fitted hypers are the draw's, whatever the generated draw.
        @test values["hsgp_t_by_subject_rho_iso"] == rho
    end
    # Fresh standardized weights for every new level.
    @test all(abs.(mean(z; dims = 1)) .< 0.1)
    @test all(abs.(std(z; dims = 1) .- 1) .< 0.06)
    # Each new level's field: mean zero, the fitted variance at each row,
    # independent across levels.
    @test all(abs.(vec(mean(field; dims = 1))) .< 4 .* sqrt.(repeat(expected ./ n)))
    @test all(abs.(vec(var(field; dims = 1)) ./ expected .- 1) .< 0.1)
    @test abs(cor(field[:, 1], field[:, 3])) < 0.06
end

@testset "SBBRMI: a new level's amplitude is drawn from the fitted hyper SD" begin
    fitted = SBBRMI(hyper_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    rp_model = StanBlocks.stan_instantiate(replay.model).model
    unc = BS.param_unc_names(rp_model)
    theta = 0.2 .* randn(Xoshiro(5), length(unc))
    fixed = constrained_by_name(rp_model, theta)
    beta0 = fixed["hsgp_t_by_subject_beta0_sigma"]
    sd = fixed["hsgp_t_by_subject_sd_sigma"]
    rho = fixed["hsgp_t_by_subject_rho_iso"]
    PHI = replay.data[:PHI_hsgp_t_by_subject]
    omega2 = replay.data[:omega2_hsgp_t_by_subject]
    level = StanBlocks.stan_data(replay.model)[:subject_idx]
    # Each new level's amplitude is sigma_g = exp(beta0 + sd * z_g) with a
    # fresh z_g ~ Normal(0, 1) at the draw's fitted beta0 and sd, and its
    # field divided by sigma_g is a unit-amplitude field at the fitted rho.
    expected = field_variance(PHI, omega2, 1.0, rho)
    n = 4000
    unit = Matrix{Float64}(undef, n, 8)
    hyper_z = Matrix{Float64}(undef, n, 4)
    amplitudes_follow = true
    for s in 1:n
        values = constrained_by_name(rp_model, theta; seed = s)
        hyper_z[s, :] = entries(values, "hsgp_t_by_subject_z_sigma")
        sigma_g = entries(values, "hsgp_t_by_subject_sigma_vec")
        amplitudes_follow &= sigma_g ≈ exp.(beta0 .+ sd .* hyper_z[s, :])
        unit[s, :] = entries(values, "hsgp_t_by_subject") ./ sigma_g[level]
    end
    @test amplitudes_follow
    @test all(abs.(vec(mean(hyper_z; dims = 1))) .< 0.1)
    @test all(abs.(vec(std(hyper_z; dims = 1)) .- 1) .< 0.06)
    @test all(abs.(vec(var(unit; dims = 1)) ./ expected .- 1) .< 0.1)
end

end
