# test/rk_mi_forecast.jl — forecasting new rows of an RKBRMI fit with an `mi()`
# covariate: `reprocess(rk, design; resample_covariates=[:rain], keep_observed=true)`
# and `transport_draws(rk, replay, draws; resample_covariates=[:rain])` (BRM:JAX
# todo `19sdrds`, user decision `10www1m`). The design's observed covariate
# values are used as given, its missing entries are drawn from their law given
# each posterior draw, and every other coordinate is the fit's. Synthetic data.
#
# Run: julia --project=test test/rk_mi_forecast.jl
using Test, Random, Statistics
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

module ForecastModels
using BayesianRegressionModels, Distributions
model = @brm begin
    sigma ~ Exponential(1)
    rain_loc ~ Normal(0, 1)
    rain_scale ~ Exponential(1)
    mi(rain) ~ Normal(rain_loc, rain_scale)
    mu ~ 1 + rain + (1 | station)
    y ~ Normal(mu, sigma)
end
end
const FM = ForecastModels

function station_rows(n; rng, gaps)
    station = string.("s", mod1.(1:n, 3))
    rain = Union{Missing,Float64}[i in gaps ? missing : 0.5 + randn(rng) for i in 1:n]
    y = [0.2 + 0.4 * coalesce(r, 0.5) + 0.1 * randn(rng) for r in rain]
    (; y, rain, station)
end

const TRAIN = station_rows(24; rng = Xoshiro(1), gaps = (3, 8, 13, 21))
const DESIGN = station_rows(9; rng = Xoshiro(2), gaps = (2, 7))

# A value query over a backend's own binding (its own build: a forecast replay
# has coordinates of its own).
query(rk, name) = let q = prepare_query(rk.model, rk_bound_plan(rk), (name,))
    function (u)
        value = Base.invokelatest(q, u)
        vec(collect(value isa Tuple ? only(value) : value))
    end
end
is_gap(name) = startswith(string(name), "rain.")

@testset "forecast new rows of an mi() covariate" begin
    rk = RKBRMI(FM.model(TRAIN); completions = [:rain])
    replay = reprocess(rk, DESIGN; resample_covariates = [:rain], keep_observed = true)
    fitted = BRM._rk_layout_coordinate_names(rk)
    names = BRM._rk_bound_coordinate_names(replay)
    # The replay builds its own kernel, whose layout is its binding's.
    @test replay.model !== rk.model
    @test BRM._rk_layout_coordinate_names(replay) == names
    # The fit's four training gaps are replaced by the design's two.
    @test count(is_gap, fitted) == 4
    @test count(is_gap, names) == 2
    @test filter(!is_gap, fitted) == filter(!is_gap, names)

    U = 0.3 .* randn(Xoshiro(3), 5, length(fitted))
    V = transport_draws(rk, replay, U; resample_covariates = [:rain], rng = Xoshiro(4))
    W = transport_draws(rk, replay, U; resample_covariates = [:rain], rng = Xoshiro(5))
    @test size(V) == (5, length(names))
    for (k, name) in enumerate(names)
        if is_gap(name)
            @test V[:, k] != W[:, k]   # fresh for each stream
        else
            @test V[:, k] == U[:, findfirst(==(name), fitted)]
        end
    end
    # Observed rows read the design's values; the missing rows read the draws.
    rain, mu = query(replay, :rain), query(replay, :mu)
    observed = findall(!ismissing, DESIGN.rain)
    for d in 1:5
        @test rain(V[d, :])[observed] == DESIGN.rain[observed]
        @test mu(V[d, :])[observed] ≈ mu(W[d, :])[observed]
        @test all(isfinite, mu(V[d, :]))
    end
    @test rain(V[1, :])[[2, 7]] != rain(W[1, :])[[2, 7]]

    # The drawn entries follow the law given the posterior draw: many streams
    # at one draw give the law's mean and spread at a missing row.
    many = repeat(U[1:1, :], 4000)
    T = transport_draws(rk, replay, many; resample_covariates = [:rain], rng = Xoshiro(6))
    gap_values = [rain(T[d, :])[2] for d in axes(T, 1)]
    params = ReactiveKernelsPPL.constrain(rk.model.layout, U[1, :])
    loc, scale = params.rain_loc, params.rain_scale
    @test mean(gap_values) ≈ loc atol = 5 * scale / sqrt(4000)
    @test std(gap_values) ≈ scale rtol = 0.05

    # Refusals, each naming what to do.
    @test_throws "changes the fitted coordinates" reprocess(rk, DESIGN)
    @test_throws "SBBRMI's population prediction" reprocess(rk, DESIGN;
        resample_covariates = [:rain])
    @test_throws "needs `resample_covariates`" reprocess(rk, DESIGN; keep_observed = true)
    @test_throws "not an `mi()` covariate" reprocess(rk, DESIGN;
        resample_covariates = [:station], keep_observed = true)
    centered = RKBRMI(FM.model(TRAIN))
    @test_throws "non-centered `Normal` or `LogNormal`" reprocess(centered, DESIGN;
        resample_covariates = [:rain], keep_observed = true)
    @test_throws DimensionMismatch transport_draws(rk, replay, U[:, 1:3];
        resample_covariates = [:rain])
end
