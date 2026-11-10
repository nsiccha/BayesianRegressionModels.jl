# test/rk_jax_models.jl — model-level parity of RKBRMI backends and their
# Python/JAX modules, `jax_source(rk.model, rk_bound_plan(rk))` run in a real
# Python process (`test/jax_module_driver.jl`; brm-use
# references/reactivekernels.md, "A Python/JAX module").
#
# For each model, at several points of the unconstrained space:
# - `log_density`, `log_prior`, `log_likelihood` and `log_jacobian`, eagerly and
#   under `jax.jit` with the prepared data closed over, against RKPPL's native
#   `:sampler`, `:prior`, `:likelihood` and `:log_jacobian` queries;
# - `jax.grad` of `log_density` under `jax.jit` against the native Enzyme
#   gradient of `rk_logdensity_problem`;
# - `constrain` against the native constrained values, and `values=` outputs
#   against the native value query;
# - the same module on a second dataset with other sizes and other prior values
#   given as data: the RKPPL program is identical, the bound values differ, and
#   one module serves both (user principle `0ikeidi`: sizes and numeric values
#   are runtime inputs).
#
# Tolerances. Both sides compute in float64 (the module enables x64 itself,
# which the driver checks) with the same operations; they differ only in
# summation order and fused operations, so values agree to a few ulp times the
# number of summands (measured ≤ 4e-16 relative) and gradients, both exact AD,
# to the same order (measured ≤ 3e-14). `rtol = 1e-12` for values and `1e-10`
# for gradients leave two orders of margin without hiding a modelling
# difference, which shows at 1e-3 or larger.
#
# Each construct that does not translate yet is a `@test_broken` naming its
# owner's gap, so the fix flips it.
#
# Run: RK_JAX_PYTHON=<python with jax> [BRM_JAX_REQUIRE=1] julia --project=test test/rk_jax_models.jl
using Test, Random, LinearAlgebra
using BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, LogDensityProblems

include("jax_module_driver.jl")

# Fixes of RK's JAX lowering this file requires once a pin carries them
# (ReactiveKernels:JAX's review candidates; the test pin `29711427` = RK
# 0.1.120 / RKPPL 0.1.128 and RK main `ba637e43` = RK 0.1.129 / RKPPL 0.1.133
# carry neither):
# - RK 0.1.130 translates the CDF-based laws (`censored`, `interval_censored`,
#   `truncated` responses and truncated priors such as `dar`'s);
# - RKPPL 0.1.134 jits with the prepared data traced as arguments (sizes are
#   static) and states a data-dependent dimension as `m.dimension(prepared)`
#   instead of a module constant, so the module text is size-generic.
const RK_JAX_CDF = pkgversion(ReactiveKernels) >= v"0.1.130"
# RK 0.1.130 also translates a derivative rule's primal, such as the bounded
# transform (`rk_affine_logistic`) of `cdar`'s persistence.
const RK_JAX_RULE_PRIMALS = pkgversion(ReactiveKernels) >= v"0.1.130"
const RK_JAX_STATIC_SIZES = pkgversion(ReactiveKernelsPPL) >= v"0.1.134"
# RKPPL 0.1.136 runs an observed `@scan` in a ragged plate cell (`AR1Normal`
# per group) and returns `constrain` nested by scope, as native `constrain`
# does (a `dar` scan's innovations included).
const RK_JAX_GROUP_SCANS = pkgversion(ReactiveKernelsPPL) >= v"0.1.136"

module JaxModelSet
using BayesianRegressionModels, Distributions
# Prior settings are data fields, so datasets that differ only in them share
# one program (brm-use references/priors.md).
regression = @brm begin
    "Residual scale of the response."
    sigma ~ Exponential(sigma_rate)
    "Expected response: intercept, standardized covariates and a group effect."
    mu ~ 1 + zscale(x) + center(w) + (1 | g)
    effect(mu, Intercept) ~ Normal(b0_loc, b0_scale)
    sd(mu, g) ~ Exponential(sd_rate)
    "Measured response."
    y ~ Normal(mu, sigma)
end
counts = @brm begin
    log(lambda) ~ 1 + x + (1 | g)
    effect(lambda, Intercept) ~ Normal(b0_loc, b0_scale)
    c ~ Poisson(lambda)
end
positive = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    yp ~ LogNormal(mu, sigma)
end
missing_covariate = @brm begin
    rain_loc ~ Normal(0, 1)
    rain_scale ~ Exponential(1)
    mi(rain) ~ Normal(rain_loc, rain_scale)
    sigma ~ Exponential(1)
    mu ~ 1 + rain + (1 | g)
    y ~ Normal(mu, sigma)
end
bounded = @brm begin
    a ~ Uniform(-1, 1)
    b ~ Normal(0, 1; lower = 0.0)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end
# Ragged per-subject series read in a `@plate` cell.
series = @brm begin
    sigma ~ Exponential(sigma_rate)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
# Gaps (see the testset "constructs that do not translate yet").
detection_limit = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ censored(Normal(mu, sigma); lower = lod)
end
rounded = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y_lo ~ interval_censored(Normal(mu, sigma); upper = y_hi)
end
truncated_response = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ truncated(Normal(mu, sigma); lower = 0.0)
end
differenced_walk = @brm begin
    sigma ~ Exponential(1)
    level ~ 1 + dar(day_grid; p = 1)
    mu = level[day_idx]
    y ~ Normal(mu, sigma)
end
ar1_residuals = @brm begin
    sigma ~ Exponential(1)
    phi ~ Normal(0, 0.5)
    mu ~ 1 + x + (1 | g)
    @plate for p in groups(g; order = day)
        y[p] ~ AR1Normal(mu[p], sigma, tanh(phi); time = day[p])
    end
end
# ODLProject's `odl_station_daily` fixture (synthetic): a network level, an
# annual cycle from a period held as data (data-only `sinpi`/`cospi`
# assignments), correlated station levels and trends, latent per-station
# AR(1) deviations (`cdar`) and Normal noise; a variant left-censored at a
# detection limit; one with AR(1) residuals per station instead.
station_daily = @brm begin
    "Residual scale of the log daily mean."
    sigma ~ Exponential(sigma_scale)
    "Annual cycle over a period held as data."
    cycle_sin = sinpi(2 * day / period)
    cycle_cos = cospi(2 * day / period)
    "Day on the scale of the series."
    trend = (day - day_center) / day_span
    "Log daily mean: network level, annual cycle, station level and trend, and each station's AR(1) deviations."
    mu ~ 1 + cycle_sin + cycle_cos + (1 + trend | p | station) + cdar(day; by = station, cor = I_station)
    sd(:, p) ~ Exponential(station_scale)
    y ~ Normal(mu, sigma)
end
station_daily_censored = @brm begin
    sigma ~ Exponential(sigma_scale)
    cycle_sin = sinpi(2 * day / period)
    cycle_cos = cospi(2 * day / period)
    trend = (day - day_center) / day_span
    mu ~ 1 + cycle_sin + cycle_cos + (1 + trend | p | station)
    sd(:, p) ~ Exponential(station_scale)
    y ~ censored(Normal(mu, sigma); lower = lod)
end
end
const JMS = JaxModelSet

regression_data(n, k; rng = Xoshiro(n), sigma_rate = 1.0, b0_loc = 0.0, b0_scale = 1.0, sd_rate = 1.0) =
    (; x = randn(rng, n), w = randn(rng, n), g = string.("g", mod1.(1:n, k)), y = randn(rng, n),
       sigma_rate, b0_loc, b0_scale, sd_rate)
counts_data(n, k; rng = Xoshiro(n), b0_loc = 0.0, b0_scale = 1.0) =
    (; x = randn(rng, n), g = string.("g", mod1.(1:n, k)), c = rand(rng, 0:4, n), b0_loc, b0_scale)
function series_data(lengths; rng = Xoshiro(length(lengths)), sigma_rate = 1.0)
    S = length(lengths)
    (; subject = string.("s", 1:S), weight = 60 .+ 30 .* rand(rng, S), dose = 100 .+ 50 .* rand(rng, S),
       t = [sort(4 .* rand(rng, L)) for L in lengths], y = [rand(rng, L) for L in lengths], sigma_rate)
end
function station_data(; n = 36, k = 4, rng = Xoshiro(42))
    x = randn(rng, n)
    y = 0.5 .+ 0.3 .* x .+ 0.2 .* randn(rng, n)
    rain = Union{Missing,Float64}[i % 5 == 0 ? missing : r for (i, r) in enumerate(randn(rng, n))]
    lo = floor.(y .* 10) ./ 10
    (; x, y, g = string.("g", mod1.(1:n, k)), day = Float64.(div.(0:n-1, k) .+ 1), rain,
       lod = fill(0.3, n), ycens = max.(y, 0.3), y_lo = lo, y_hi = lo .+ 0.1)
end

# `S` stations, each observed on 20–60 days of 90 (absent rows are missing
# days); sizes and every numeric setting are data.
function station_daily_data(S; rng = Xoshiro(S), lod = -Inf, period = 365.0, sigma_scale = 0.1,
        station_scale = 0.3)
    station = String[]; day = Float64[]; y = Float64[]
    for s in 1:S, t in sort(unique(rand(rng, 1:90, rand(rng, 20:60))))
        push!(station, "station_$s"); push!(day, t)
        push!(y, 0.3 * s / S + 0.1 * sinpi(2t / period) + 0.05 * randn(rng))
    end
    (; y = max.(y, lod), day, station, period, day_center = 45.0, day_span = 45.0, sigma_scale,
       station_scale, lod = fill(lod, length(y)), I_station = Matrix{Float64}(I, S, S))
end

points(rk) = [0.3 .* randn(Xoshiro(100 + k), rk.model.layout.total) for k in 1:3]

# Native references of one binding at `us`.
function native_reference(rk, us; values = ())
    built, plan = rk.model, rk_bound_plan(rk)
    queries = Dict(p => prepare_query(built, plan, p) for p in (:sampler, :prior, :likelihood, :log_jacobian))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value_queries = Dict(v => prepare_query(built, plan, (v,)) for v in values)
    map(us) do u
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        (; densities = Dict(p => Base.invokelatest(q, u) for (p, q) in queries), lp, g,
           constrained = ReactiveKernelsPPL.constrain(built.layout, u),
           values = Dict(v => Base.invokelatest(q, u) for (v, q) in value_queries))
    end
end

# The module's `constrain` names a submodel's parameter `<submodel>.<name>`.
flat_names(nt::NamedTuple, prefix = "") = reduce(vcat,
    [v isa NamedTuple ? flat_names(v, prefix * string(k) * ".") : [(prefix * string(k), v)] for (k, v) in pairs(nt)];
    init = Tuple{String,Any}[])

const FUNCTIONS = (("log_density", :sampler), ("log_prior", :prior),
    ("log_likelihood", :likelihood), ("log_jacobian", :log_jacobian))

# Compare dataset `k` of a Python run with the native binding `rk`.
# The module's `constrain`, keyed `<scope>.<name>` as `flat_names` keys the
# native values: nested by scope from RKPPL 0.1.136, flat before.
flat_constrain(d::AbstractDict, prefix = "") = reduce(merge,
    (v isa AbstractDict ? flat_constrain(v, prefix * k * ".") : Dict(prefix * k => v) for (k, v) in d);
    init = Dict{String,Any}())

# `constrain_gaps`: native constrained names the module's `constrain` does not
# return yet (an RK lowering gap each, reported to ReactiveKernels:JAX).
function check_parity(rk, run, k, us; values = (), jit_args = true, constrain_gaps = ())
    reference = native_reference(rk, us; values)
    for (j, u) in enumerate(us)
        p, n = run.datasets[k]["points"][j], reference[j]
        @test n.lp ≈ n.densities[:sampler] rtol = 1e-12
        for (f, preset) in FUNCTIONS
            @test p[f]["eager"] ≈ n.densities[preset] rtol = 1e-12 atol = 1e-12
            @test p[f]["jit"] ≈ n.densities[preset] rtol = 1e-12 atol = 1e-12
            if p[f]["jit_args"]["ok"]
                @test p[f]["jit_args"]["value"] ≈ n.densities[preset] rtol = 1e-12 atol = 1e-12
            end
        end
        # The prepared data traced as arguments: a layout whose offsets follow
        # a level count slices `unconstrained` at a traced stop (RK lowering
        # gap, reported to ReactiveKernels:JAX).
        @test all(p[f]["jit_args"]["ok"] for (f, _) in FUNCTIONS) && p["grad_args"]["ok"] broken = !(jit_args || RK_JAX_STATIC_SIZES)
        if p["grad_args"]["ok"]
            @test vec(collect(p["grad_args"]["value"])) ≈ n.g rtol = 1e-10 atol = 1e-10
        end
        @test vec(collect(p["grad"])) ≈ n.g rtol = 1e-10 atol = 1e-10
        cjit, ceager = flat_constrain(p["constrain_jit"]), flat_constrain(p["constrain_eager"])
        for (name, value) in flat_names(n.constrained)
            @test haskey(cjit, name) broken = name in constrain_gaps
            haskey(cjit, name) || continue
            @test vec(collect(cjit[name])) ≈ vec(collect(Float64.(value))) rtol = 1e-12 atol = 1e-12
            @test vec(collect(ceager[name])) ≈ vec(collect(Float64.(value))) rtol = 1e-12 atol = 1e-12
        end
        for (v, value) in n.values
            @test vec(collect(cjit[String(v)])) ≈ vec(collect(Float64.(value))) rtol = 1e-12 atol = 1e-12
        end
    end
end

function check_model(rk; values = (), jit_args = true, constrain_gaps = ())
    src = jax_source(rk.model, rk_bound_plan(rk); values)
    @test src isa JAXSource
    brm_jax_require_python() || return src
    us = points(rk)
    run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
    @test run.x64_after_import   # the module enables float64 itself
    check_parity(rk, run, 1, us; values, jit_args, constrain_gaps)
    src
end

# The module text without its binding's layout constants (`DIMENSION`,
# `PARAMETERS`), which state the emitting binding's sizes.
function without_layout_constants(src)
    join(filter(l -> !startswith(l, "DIMENSION = ") && !startswith(l, "PARAMETERS = "),
        split(src.text, '\n')), '\n')
end

@testset "one module serves datasets of other sizes and prior values" begin
    A = regression_data(9, 3)
    B = regression_data(23, 5; sigma_rate = 2.5, b0_loc = 0.7, b0_scale = 3.0, sd_rate = 0.4)
    rkA, rkB = RKBRMI(JMS.regression(A)), RKBRMI(JMS.regression(B))
    @test string(rkA.model.program) == string(rkB.model.program)
    @test rk_bound_data(rkA).sigma_rate != rk_bound_data(rkB).sigma_rate
    srcA = jax_source(rkA.model, rk_bound_plan(rkA))
    srcB = jax_source(rkB.model, rk_bound_plan(rkB))
    @test without_layout_constants(srcA) == without_layout_constants(srcB)
    # Before RKPPL 0.1.134 `DIMENSION` and `PARAMETERS` hold the emitting
    # binding's sizes, so the module text differs between the two; its
    # functions do not.
    @test srcA.text == srcB.text broken = !RK_JAX_STATIC_SIZES
    if brm_jax_require_python()
        usA, usB = points(rkA), points(rkB)
        run = brm_jax_run(srcA, [brm_jax_inputs(rkA, srcA), brm_jax_inputs(rkB, srcA)], [usA, usB])
        @test run.x64_after_import
        check_parity(rkA, run, 1, usA; jit_args = false)
        check_parity(rkB, run, 2, usB; jit_args = false)
    end
end

@testset "families, groupings, missing covariates and bounded priors" begin
    S = station_data()
    check_model(RKBRMI(JMS.regression(regression_data(9, 3))); values = (:mu,), jit_args = false)
    check_model(RKBRMI(JMS.counts(counts_data(12, 4))); jit_args = false)
    check_model(RKBRMI(JMS.positive((; yp = exp.(S.y), S.x, S.g))); jit_args = false)
    check_model(RKBRMI(JMS.missing_covariate((; S.y, S.rain, S.g))); jit_args = false)
    check_model(RKBRMI(JMS.bounded((; S.y, S.x))))
end

@testset "ragged per-subject series, an empty subject included" begin
    check_model(RKBRMI(JMS.series(series_data([3, 2, 4, 1, 2]))); jit_args = false)
    check_model(RKBRMI(JMS.series(series_data([3, 0, 2]))); jit_args = false)
end

@testset "replays: reprocess and a data-free replay of the fitted state" begin
    rk = RKBRMI(JMS.regression(regression_data(9, 3)))
    design = regression_data(5, 2; rng = Xoshiro(7))
    check_model(reprocess(rk, design); jit_args = false)
    check_model(RKBRMI(JMS.regression(design), brm_fitted_state(rk)); jit_args = false)
end

# CDF-based laws: required from RK 0.1.130, `JAXEmissionError` ("`##captures#1`
# is not defined" / "`prepared standardized` is not defined in module
# `ReactiveKernelsDistributionKernels.DistributionKernelSources`") before it.
@testset "censored, interval-censored and truncated laws" begin
    S = station_data()
    # `dar`'s scan innovations are a native constrained value the module's
    # `constrain` does not return yet.
    for (build, gaps) in ((() -> RKBRMI(JMS.detection_limit((; y = S.ycens, S.x, S.lod))), ()),
            (() -> RKBRMI(JMS.rounded((; S.y_lo, S.y_hi, S.x))), ()),
            (() -> RKBRMI(JMS.truncated_response((; y = abs.(S.y), S.x))), ()),
            (() -> RKBRMI(JMS.differenced_walk((; S.y, day_grid = collect(1.0:9.0), day_idx = Int.(S.day)))),
                RK_JAX_GROUP_SCANS ? () : ("dar_level_day_grid_level._ppl_scan_z_level",)))
        rk = build()
        if RK_JAX_CDF
            check_model(rk; constrain_gaps = gaps)
        else
            @test_broken try
                jax_source(rk.model, rk_bound_plan(rk)) isa JAXSource
            catch e
                e isa JAXEmissionError || rethrow()
                false
            end
        end
    end
end

# A per-group scan (`AR1Normal` residuals over `groups(g; order = day)`):
# required from RKPPL 0.1.136; before it `jax_source` refuses it ("the
# function `_observed_scan_cell_bound` has loops or branches").
@testset "AR(1) residuals per group" begin
    S = station_data()
    rk = RKBRMI(JMS.ar1_residuals((; S.y, S.x, S.g, S.day)))
    if RK_JAX_GROUP_SCANS
        check_model(rk; jit_args = false)
    else
        @test_broken try
            jax_source(rk.model, rk_bound_plan(rk)) isa JAXSource
        catch e
            e isa JAXEmissionError || rethrow()
            false
        end
    end
end

# Networks of 2, 6 and 12 stations with other prior scales and period. Their
# programs are not identical yet: RKBRMI's `cdar` emission writes the
# stations' and days' levels and the counts P and W into the program, and its
# lag, mask and decay are W × W (O(P W^2) per evaluation); BRM todo for a
# scan with fit-role level inputs. So each network has its own module here.
# Not built yet either: a station with no days, which needs the stations as
# a level table read through positions (brm-use formulas, "Labels from a
# level table"); the per-station AR(1) residual variant waits on the scan gap
# above.
@testset "odl_station_daily: networks of 2, 6 and 12 stations" begin
    datasets = (station_daily_data(6), station_daily_data(2; sigma_scale = 0.2, period = 300.0),
        station_daily_data(12; station_scale = 0.5))
    rks = [RKBRMI(JMS.station_daily(d)) for d in datasets]
    @test allequal(string(rk.model.program) for rk in rks) broken = true
    for rk in rks
        if RK_JAX_RULE_PRIMALS
            check_model(rk; jit_args = false)
        else
            @test_broken try
                jax_source(rk.model, rk_bound_plan(rk)) isa JAXSource
            catch e
                e isa JAXEmissionError || rethrow()
                false
            end
        end
    end
    censored = RKBRMI(JMS.station_daily_censored(station_daily_data(6; lod = 0.1)))
    if RK_JAX_CDF
        check_model(censored; jit_args = false)
    else
        @test_broken try
            jax_source(censored.model, rk_bound_plan(censored)) isa JAXSource
        catch e
            e isa JAXEmissionError || rethrow()
            false
        end
    end
end

# Whether a module emits and runs in Python (prepare_data and log_density at
# one point), for a construct tested as broken until RK translates it.
function runs_in_python(rk)
    try
        src = jax_source(rk.model, rk_bound_plan(rk))
        brm_jax_available() || return false
        brm_jax_run(src, [brm_jax_inputs(rk, src)], [points(rk)[1:1]])
        true
    catch e
        e isa Union{JAXEmissionError,ArgumentError,ErrorException} || rethrow()
        false
    end
end

# The shared fields of ODL-shaped models (synthetic data). `hsgp(t)`
# translates. These do not translate yet; each is an RK lowering gap reported
# to ReactiveKernels:JAX on RK 0.1.130 / RKPPL 0.1.136 (`9cbe68ee`):
# - levels correlated by an LKJ factor of data-valued size: "`keep` names
#   `_ppl_last_Omega`, which is no data-only value of the program";
# - `orthogonal_to = :linear`: `prepare_data` indexes a Python list as a
#   matrix (`raw_basis[:, b - 1]`);
# - multi-axis bases: "`checked_sdiv_int` has no JAX translation".
@testset "ODL-shaped shared fields: HSGP terms" begin
    rng = Xoshiro(6)
    n = 30
    e, nn, t = randn(rng, n), randn(rng, n), collect(range(0, 3; length = n))
    y = sin.(e) .+ 0.1 .* randn(rng, n)
    region = string.("r", mod1.(1:n, 3))
    check_model(RKBRMI(@brm (; y, t) begin
        s ~ Exponential(1)
        mu ~ 1 + hsgp(t)
        y ~ Normal(mu, s)
    end); jit_args = false)
    brm_jax_require_python() || return
    gaps = (
        RKBRMI(@brm (; y, t, region, K = 3, eta = 2.0) begin
            s ~ Exponential(1)
            Omega ~ LKJCholesky(K, eta)
            mu ~ 1 + hsgp(t; by = region, cor = Omega)
            y ~ Normal(mu, s)
        end),
        RKBRMI(@brm (; y, t, region, cb = 5, cc = 1.5) begin
            s ~ Exponential(1)
            mu ~ 1 + t + hsgp(t; k = cb, c = cc, by = region, orthogonal_to = :linear)
            y ~ Normal(mu, s)
        end),
        RKBRMI(@brm (; y, e, nn) begin
            s ~ Exponential(1)
            mu ~ 1 + hsgp(e, nn)
            y ~ Normal(mu, s)
        end),
        RKBRMI(@brm (; y, e, nn, t) begin
            s ~ Exponential(1)
            mu ~ 1 + hsgp(e, nn, t; iso = false)
            y ~ Normal(mu, s)
        end),
    )
    for rk in gaps
        @test_broken runs_in_python(rk)
    end
end
