# test/walk_forecast.jl — a replay whose rows reach past the fitted grid of an
# `rw(time)` or `cdar(step; by=group, cor=C)` term forecasts the walk: the new
# steps' innovations are drawn in generated quantities from their N(0, 1)
# prior, the fit's unconstrained draws feed the forecast program unchanged, and
# every new step continues the fitted path from its last fitted state.
# A replay step inside the fitted grid that the fit did not see has no
# innovation and is refused. Snag per-group-statio-059ae6f6.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra
using Random
using Statistics

const WF_RUNTIME = get(ENV, "BRM_WALK_FORECAST_RUNTIME", "1") != "0"
const WF_CACHE = joinpath(tempdir(), "brm-walk-forecast")
const BS = StanBlocks.BridgeStan

# Group-years: three groups, two units per group, years as steps; group "c" has
# no rows in `missing_year` (its state there is latent).
function cdar_rows(years; missing_year=nothing, with_y=true, C=Matrix(1.0I, 3, 3))
    year = Int[]; kreis = String[]; gem = String[]
    for t in years, k in ("a", "b", "c"), u in 1:2
        (k == "c" && t == missing_year) && continue
        push!(year, t); push!(kreis, k); push!(gem, string(k, u))
    end
    base = (; year, kreis, gem, C)
    with_y ? merge(base, (; y=0.1 .* sin.(1:length(year)))) : base
end

const CDAR_BUILDER = @brm begin
    mu ~ 1 + cdar(year; by = kreis, cor = C)
    sd(:, cdar(year)) ~ LogNormal(-1, 0.5)
    y ~ Normal(mu, 0.3)
end

rw_rows(times; with_y=true) = with_y ? (; t=collect(Float64, times),
    y=0.1 .* cos.(1:length(times))) : (; t=collect(Float64, times))

const RW_BUILDER = @brm begin
    mu ~ 1 + rw(t)
    y ~ Normal(mu, 0.3)
end

stan(sb) = StanBlocks.stan_code(sb.model)
# a forecast binds its appended-step count `maybecv`-marked; this is its number
n_new(sb, key) = StanBlocks.getvalue(sb.data[key])

function instantiate(sb)
    isdir(WF_CACHE) || mkpath(WF_CACHE)
    code = stan(sb)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(WF_CACHE, string(hash(code)) * ".stan"))
end

# Constrained values (parameters, transformed parameters, generated quantities)
# of one unconstrained draw, by name.
function constrained(problem, q; seed=1)
    sm = problem.model
    names = String.(BS.param_names(sm; include_tp=true, include_gq=true))
    values = BS.param_constrain(sm, q; include_tp=true, include_gq=true,
                                rng=BS.StanRNG(sm, seed))
    names, values
end
pick(names, values, prefix) =
    [v for (n, v) in zip(names, values) if n == prefix || startswith(n, prefix * ".")]

@testset "cdar fit: the new-step innovations are empty, the fit is unchanged" begin
    sb = SBBRMI(CDAR_BUILDER(cdar_rows(2018:2021; missing_year=2019)); mod=@__MODULE__)
    @test sb.data[:cdar_mu_year_n_steps] == 4
    @test sb.data[:cdar_mu_year_n_new_steps] == 0
    @test StanBlocks.stanc_check(stan(sb); warn_pedantic=false).ok
    # group "c" has no row in 2019, but its state there is on the grid
    @test sb.data[:cdar_mu_year_step_idx] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                                               3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    # a replay that stays on the fitted grid is the fit's own program
    same = reprocess(sb, cdar_rows(2020:2021))
    @test same.data[:cdar_mu_year_n_new_steps] == 0
    @test same.data[:cdar_mu_year_n_steps] == 4
    @test same.data[:cdar_mu_year_step_idx] == [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    @test stan(same) == stan(sb)
end

@testset "cdar forecast: past the last fitted step only" begin
    sb = SBBRMI(CDAR_BUILDER(cdar_rows([2018, 2020, 2021])); mod=@__MODULE__)
    @test_throws "inside the fitted grid" reprocess(sb, cdar_rows([2019]; with_y=false))
    @test_throws "inside the fitted grid" reprocess(sb, cdar_rows([2017]; with_y=false))
    @test_throws "not a fitted group level" reprocess(sb,
        (; year=[2022], kreis=["d"], gem=["d1"], C=Matrix(1.0I, 3, 3)))
    # fresh-fit semantics sample every step of the replay's grid, so a new step
    # joins it as a parameter wherever it falls
    fresh = reprocess(sb, cdar_rows([2019, 2022]); freeze_constants=false)
    @test fresh.data[:cdar_mu_year_n_steps] == 5
    @test fresh.data[:cdar_mu_year_n_new_steps] == 0
    @test fresh.data[:cdar_mu_year_step_idx] == [2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5]
end

@testset "cdar forecast continues each group's fitted state" begin
    if WF_RUNTIME
        for C in (Matrix(1.0I, 3, 3), [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0])
            fit_rows = cdar_rows(2018:2021; missing_year=2019, C)
            sb = SBBRMI(CDAR_BUILDER(fit_rows); mod=@__MODULE__)
            # forecast rows: the last fitted year (a check on the fitted path)
            # and two new years
            fc_rows = cdar_rows(2021:2023; with_y=false, C)
            fc = reprocess(sb, fc_rows)
            @test fc.data[:cdar_mu_year_n_steps] == 4
            @test n_new(fc, :cdar_mu_year_n_new_steps) == 2
            @test fc.data[:cdar_mu_year_step_idx] == repeat(4:6; inner=6)
            fit_problem, fc_problem = instantiate(sb), instantiate(fc)
            D = LogDensityProblems.dimension(fit_problem)
            @test LogDensityProblems.dimension(fc_problem) == D
            @test String.(BS.param_unc_names(fc_problem.model)) ==
                  String.(BS.param_unc_names(fit_problem.model))

            L = Matrix(cholesky(Symmetric(C)).L)
            q = 0.4 .* randn(Xoshiro(11), D)
            fn, fv = constrained(fit_problem, q)
            sigma = only(pick(fn, fv, "cdar_mu_year_sigma"))
            rho = only(pick(fn, fv, "cdar_mu_year_rho"))
            fit_term = pick(fn, fv, "cdar_mu_year")
            # the fitted state of each group in 2021 (rows 17:22, two per group)
            last_state = fit_term[17:2:22]
            @test fit_term[18:2:22] == last_state

            draws = Matrix{Float64}[]
            eta_draws = Vector{Float64}[]
            for seed in 1:2000
                cn, cv = constrained(fc_problem, q; seed)
                term = pick(cn, cv, "cdar_mu_year")
                eta_new = pick(cn, cv, "cdar_mu_year_eta_new")
                length(eta_new) == 6 || error("expected 3 groups x 2 new steps")
                E = reshape(eta_new, 3, 2)
                state = reshape(term, 2, 3, 3)[1, :, :]   # group x year (2021:2023)
                # exactly the recurrence, given the GQ innovations
                @test state[:, 1] ≈ last_state atol=1e-12
                s = sigma * sqrt(1 - rho^2)
                @test state[:, 2] ≈ rho .* last_state .+ s .* (L * E[:, 1]) atol=1e-10
                @test state[:, 3] ≈ rho .* state[:, 2] .+ s .* (L * E[:, 2]) atol=1e-10
                push!(draws, state)
                push!(eta_draws, eta_new)
            end
            # the innovations are N(0, I) draws, fresh per seed
            H = reduce(hcat, eta_draws)
            @test maximum(abs.(mean(H; dims=2))) < 5 / sqrt(2000)
            @test maximum(abs.(var(H; dims=2) .- 1)) < 5 * sqrt(2 / 2000)
            # one step ahead: mean rho * state_2021, covariance sigma^2 (1 - rho^2) C
            S1 = reduce(hcat, [d[:, 2] for d in draws])
            @test maximum(abs.(mean(S1; dims=2) .- rho .* last_state)) <
                  5 * sigma * sqrt(1 - rho^2) / sqrt(2000)
            @test cov(S1; dims=2) ≈ sigma^2 * (1 - rho^2) .* C rtol=0.15 atol=0.02 * sigma^2

            # the descriptor's :predict runs on the fit's own unconstrained draws
            d = brm_descriptor(fc)
            out = brm_execute(d, :predict; problem=fc_problem,
                              draws=0.4 .* randn(Xoshiro(3), D, 4), seed=1234)
            @test haskey(out, :y_gen) || haskey(out, :y)
        end
    else
        @info "Skipping the cdar forecast runtime gate (BRM_WALK_FORECAST_RUNTIME=0)"
        @test true
    end
end

@testset "cdar forecast through resample_groups keeps the walk's fitted state" begin
    builder = @brm begin
        mu ~ 1 + (1 | gem) + cdar(year; by = kreis, cor = C)
        y ~ Normal(mu, 0.3)
    end
    sb = SBBRMI(builder(cdar_rows(2018:2021)); mod=@__MODULE__)
    rows = (; year=[2022, 2022], kreis=["a", "b"], gem=["a9", "b9"], C=Matrix(1.0I, 3, 3))
    fc = reprocess(sb, rows; resample_groups=[:gem])
    @test n_new(fc, :cdar_mu_year_n_new_steps) == 1
    @test StanBlocks.stanc_check(stan(fc); warn_pedantic=false).ok
    if WF_RUNTIME
        # the resampled groups' effects are generated quantities (draw
        # transport), but the walk keeps the fit's coordinates
        walk(problem) = filter(startswith("cdar_mu_year_"),
                               String.(BS.param_unc_names(problem.model)))
        @test walk(instantiate(fc)) == walk(instantiate(sb))
        @test length(walk(instantiate(sb))) == 2 + 3 * 4
    end
end

@testset "rw forecast continues the fitted walk" begin
    sb = SBBRMI(RW_BUILDER(rw_rows(1:4)); mod=@__MODULE__)
    @test sb.data[:rw_mu_t_n_new_steps] == 0
    @test_throws "inside the fitted grid" reprocess(sb, rw_rows([2.5]; with_y=false))
    fc = reprocess(sb, rw_rows(4:6; with_y=false))
    @test fc.data[:rw_mu_t_n_steps] == 4
    @test n_new(fc, :rw_mu_t_n_new_steps) == 2
    @test fc.data[:rw_mu_t_time_idx] == [4, 5, 6]
    if WF_RUNTIME
        fit_problem, fc_problem = instantiate(sb), instantiate(fc)
        D = LogDensityProblems.dimension(fit_problem)
        @test LogDensityProblems.dimension(fc_problem) == D
        q = 0.4 .* randn(Xoshiro(5), D)
        fn, fv = constrained(fit_problem, q)
        sigma = only(pick(fn, fv, "rw_mu_t_sigma"))
        last_point = pick(fn, fv, "rw_mu_t")[4]
        for seed in 1:50
            cn, cv = constrained(fc_problem, q; seed)
            path = pick(cn, cv, "rw_mu_t")
            z_new = pick(cn, cv, "rw_mu_t_z_new")
            @test length(z_new) == 2
            @test path[1] ≈ last_point atol=1e-12
            @test path[2] ≈ last_point + sigma * z_new[1] atol=1e-10
            @test path[3] ≈ last_point + sigma * (z_new[1] + z_new[2]) atol=1e-10
        end
    end
end

println("walk_forecast.jl: all testsets ran")
