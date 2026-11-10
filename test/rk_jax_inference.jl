# test/rk_jax_inference.jl — a representative station-network model through
# its Python/JAX module: model-level parity, then the same posterior sampled
# natively (WarmupHMC's quorum sampler on `rk_logdensity_problem`) and in
# Python (NumPyro NUTS on the module's `log_density`), compared
# distributionally, plus predictions at posterior draws.
#
# The data are synthetic and independently authored: stations at positions in
# km observed over windows of unequal start and length (ragged per station),
# a rain covariate with gaps (`mi()`), a network-wide seasonal course (`hsgp`),
# correlated station levels and rain sensitivities, and log-normal readings.
# Prior settings and basis sizes are data fields (user principle `0ikeidi`).
#
# Distributional check: for every unconstrained coordinate (population
# effects, log scales, correlation factors, station effects, basis weights and
# each inferred rain gap, 128 here), the two posterior means differ by less
# than 4.5 combined Monte Carlo standard errors (from each side's bulk ESS):
# a correct pair exceeds it with probability 6.8e-6, so all 128 pass with
# probability above 0.999. The same comparison of every scalar constrained
# quantity is reported, not asserted. Sampler health is reported with the
# warmuphmc-use §0 set (R-hat, bulk/tail ESS, divergences).
#
# Run (about 10 minutes on one CPU):
#   RK_JAX_PYTHON=<python with jax and numpyro> BRM_JAX_REQUIRE=1 julia --project=test test/rk_jax_inference.jl
# `BRM_JAX_RUN_DIR=<dir>` writes the fixture, versions and results as
# ODLProject's run directory (`manifest.json`, `parity.json`, `fit.json`,
# `predictive.json`).
using Test, Random, Statistics, LinearAlgebra
using BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, LogDensityProblems
using WarmupHMC, MCMCDiagnosticTools
using JSON, SHA

include("jax_module_driver.jl")

module StationNetwork
using BayesianRegressionModels, Distributions
model = @brm begin
    "Residual scale of the log dose rate."
    sigma ~ Exponential(sigma_scale)
    "Mean of the rain covariate; its gaps are inferred."
    rain_loc ~ Normal(0, 1)
    "Spread of the rain covariate."
    rain_scale ~ Exponential(1)
    mi(rain) ~ Normal(rain_loc, rain_scale)
    "Log dose rate: a network-wide seasonal course, rain washout, and station levels and rain sensitivities that vary together."
    mu ~ 1 + rain + hsgp(day; k = day_basis, c = day_boundary) + (1 + rain | p | station)
    effect(mu, Intercept) ~ Normal(level_loc, level_scale)
    "Measured dose rate."
    y ~ LogNormal(mu, sigma)
end
end

# Synthetic network: `S` stations, each observed on its own window of days.
function station_network(; S = 12, days = 90, rng = Xoshiro(2026), missing_share = 0.15,
        sigma_scale = 0.1, level_loc = 4.0, level_scale = 1.0, day_basis = 8, day_boundary = 1.5)
    start = rand(rng, 1:30, S)
    len = [rand(rng, 12:(days - start[s] + 1)) for s in 1:S]
    level = 4.3 .+ 0.15 .* randn(rng, S)
    washout = 0.08 .+ 0.03 .* randn(rng, S)
    station = String[]; day = Float64[]; rain = Union{Missing,Float64}[]; y = Float64[]
    for s in 1:S, t in start[s]:(start[s] + len[s] - 1)
        r = max(0.0, 0.6 + randn(rng))
        mu = level[s] + washout[s] * r + 0.05 * sinpi(2t / days)
        push!(station, "station_$(lpad(s, 2, '0'))"); push!(day, t)
        push!(rain, rand(rng) < missing_share ? missing : r)
        push!(y, exp(mu + 0.04 * randn(rng)))
    end
    (; y, rain, day, station, sigma_scale, level_loc, level_scale, day_basis, day_boundary)
end

flat_names(nt::NamedTuple, prefix = "") = reduce(vcat,
    [v isa NamedTuple ? flat_names(v, prefix * string(k) * ".") : [(prefix * string(k), v)] for (k, v) in pairs(nt)];
    init = Tuple{String,Any}[])

# Scalar constrained quantities per draw: `names`, `values[name][draw, chain]`.
function constrained_scalars(layout, draws)
    d, N, C = size(draws)
    first_draw = flat_names(ReactiveKernelsPPL.constrain(layout, draws[:, 1, 1]))
    names = String[]
    for (name, v) in first_draw
        for i in eachindex(v)
            push!(names, length(v) == 1 && !(v isa AbstractArray) ? name : "$name[$i]")
        end
    end
    values = Dict(n => zeros(N, C) for n in names)
    for c in 1:C, j in 1:N
        k = 0
        for (_, v) in flat_names(ReactiveKernelsPPL.constrain(layout, draws[:, j, c]))
            for x in (v isa AbstractArray ? vec(v) : (v,))
                k += 1
                values[names[k]][j, c] = x
            end
        end
    end
    names, values
end

function summarize(x::AbstractMatrix)   # draws × chains
    chains = reshape(x, size(x, 1), size(x, 2), 1)
    bulk = only(MCMCDiagnosticTools.ess(chains; kind = :bulk))
    tail = only(MCMCDiagnosticTools.ess(chains; kind = :tail))
    (; mean = mean(x), sd = std(x), bulk_ess = bulk, tail_ess = tail,
       rhat = only(MCMCDiagnosticTools.rhat(chains)), mcse = std(x) / sqrt(bulk))
end

const DATA = station_network()
const RESULTS = Dict{String,Any}()

# ODLProject's result contract v0: one directory per run with manifest,
# parity, fit and predictive JSON files.
function write_run_dir(dir)
    mkpath(dir)
    lines = readlines(@__FILE__)
    first_line = findfirst(l -> occursin("model = @brm begin", l), lines)
    last_line = first_line + findfirst(==("end"), lines[first_line+1:end])
    declaration = join(lines[first_line:last_line], "\n")
    datatext = sprint(JSON.print, Dict(string(k) => v for (k, v) in pairs(DATA)))
    root = dirname(@__DIR__)
    git(args...) = try readchomp(`git -C $root $args`) catch; "unknown" end
    # Non-finite numbers are JSON `null`, as WarmupHMC writes its run records.
    jsonsafe(x::AbstractFloat) = isfinite(x) ? x : nothing
    jsonsafe(x::NamedTuple) = Dict(string(k) => jsonsafe(v) for (k, v) in pairs(x))
    jsonsafe(x::AbstractDict) = Dict(string(k) => jsonsafe(v) for (k, v) in x)
    jsonsafe(x::AbstractArray) = map(jsonsafe, x)
    jsonsafe(x::Tuple) = map(jsonsafe, collect(x))
    jsonsafe(x) = x
    write_json(name, x) = open(io -> JSON.print(io, jsonsafe(x), 1), joinpath(dir, name), "w")
    write_json("manifest.json", Dict("model" => "station_network",
        "fixture" => "BayesianRegressionModels.jl test/rk_jax_inference.jl (synthetic)",
        "declaration" => declaration, "declaration_sha256" => bytes2hex(sha256(declaration)),
        "module_sha256" => get(RESULTS, "module_sha256", nothing),
        "data_sha256" => bytes2hex(sha256(datatext)), "sizes" => Dict("rows" => length(DATA.y),
            "stations" => length(unique(DATA.station)), "missing_rain" => count(ismissing, DATA.rain),
            "dimension" => get(RESULTS, "dimension", nothing)),
        "pins" => Dict("BayesianRegressionModels" => Dict("version" => string(pkgversion(BayesianRegressionModels)),
                "commit" => git("rev-parse", "HEAD"), "dirty" => !isempty(git("status", "--porcelain", "--", "src", "ext", "test"))),
            "ReactiveKernels" => Dict("version" => string(pkgversion(ReactiveKernels)),
                "commit" => try readchomp(`git -C $(pkgdir(ReactiveKernels)) rev-parse HEAD`) catch; "unknown" end),
            "ReactiveKernelsPPL" => string(pkgversion(ReactiveKernelsPPL)),
            "julia" => string(VERSION), "python" => get(RESULTS, "python", nothing),
            "numpyro" => get(RESULTS, "numpyro", nothing)),
        "host" => get(ENV, "KB_HOST", gethostname()), "stage" => haskey(RESULTS, "inference") ? "fitted" : "constructed"))
    write_json("parity.json", Dict("points" => get(RESULTS, "parity_points", []),
        "worst_relative_error" => get(RESULTS, "parity_worst_relative_error", nothing),
        "tolerances" => Dict("value_rtol" => 1e-12, "gradient_rtol" => 1e-10)))
    haskey(RESULTS, "inference") && write_json("fit.json", Dict(
        "methods" => Dict("native" => "WarmupHMC completion_warmup_mcmc on rk_logdensity_problem (Enzyme Reverse), 4 chains, quorum 3, 1000 draws",
            "python" => "NumPyro NUTS (potential_fn = -log_density of the jax_source module), 4 chains, 1000 warmup, 1000 draws"),
        "results" => RESULTS["inference"]))
    haskey(RESULTS, "predictive") && write_json("predictive.json", RESULTS["predictive"])
end

try
@testset "station network: parity of the emitted module" begin
    rk = RKBRMI(StationNetwork.model(DATA))
    src = jax_source(rk.model, rk_bound_plan(rk); values = (:mu,))
    RESULTS["module_sha256"] = bytes2hex(sha256(src.text))
    RESULTS["dimension"] = rk.model.layout.total
    RESULTS["rows"] = length(DATA.y)
    RESULTS["missing_rain"] = count(ismissing, DATA.rain)
    if brm_jax_require_python()
        us = [0.2 .* randn(Xoshiro(10 + k), rk.model.layout.total) for k in 1:3]
        run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
        RESULTS["python"] = run.versions
        plan = rk_bound_plan(rk)
        queries = Dict(p => prepare_query(rk.model, plan, p) for p in (:sampler, :prior, :likelihood, :log_jacobian))
        mu = prepare_query(rk.model, plan, (:mu,))
        problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
        worst = Dict{String,Float64}()
        RESULTS["parity_points"] = Any[]
        for (j, u) in enumerate(us)
            p = run.datasets[1]["points"][j]
            lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
            push!(RESULTS["parity_points"], Dict("point" => u, "native" => Dict(
                    "log_density" => Base.invokelatest(queries[:sampler], u),
                    "log_prior" => Base.invokelatest(queries[:prior], u),
                    "log_likelihood" => Base.invokelatest(queries[:likelihood], u),
                    "log_jacobian" => Base.invokelatest(queries[:log_jacobian], u), "gradient" => g),
                "jax" => Dict(f => Dict("eager" => p[f]["eager"], "jit_closure" => p[f]["jit"],
                        "jit_arguments" => p[f]["jit_args"]["ok"] ? p[f]["jit_args"]["value"] : p[f]["jit_args"]["error"])
                    for f in ("log_density", "log_prior", "log_likelihood", "log_jacobian")),
                "jax_gradient" => vec(collect(p["grad"]))))
            for (f, preset) in (("log_density", :sampler), ("log_prior", :prior),
                                ("log_likelihood", :likelihood), ("log_jacobian", :log_jacobian))
                native = Base.invokelatest(queries[preset], u)
                @test p[f]["jit"] ≈ native rtol = 1e-12 atol = 1e-12
                @test p[f]["eager"] ≈ native rtol = 1e-12 atol = 1e-12
                worst[f] = max(get(worst, f, 0.0), abs(p[f]["jit"] - native) / max(1, abs(native)))
            end
            @test vec(collect(p["grad"])) ≈ g rtol = 1e-10 atol = 1e-10
            worst["grad"] = max(get(worst, "grad", 0.0), maximum(abs.(vec(collect(p["grad"])) .- g) ./ max.(1, abs.(g))))
            @test vec(collect(p["constrain_jit"]["mu"])) ≈ Base.invokelatest(mu, u) rtol = 1e-12
        end
        RESULTS["parity_worst_relative_error"] = worst
    end
end

@testset "station network: the same posterior natively and in Python" begin
    brm_jax_require_python() || return
    rk = RKBRMI(StationNetwork.model(DATA))
    src = jax_source(rk.model, rk_bound_plan(rk); values = (:mu,))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    chains, n_draws = 4, 1000
    native_time = @elapsed fit = completion_warmup_mcmc([Xoshiro(100 + c) for c in 1:chains], problem;
        min_completed = div(3chains, 4), n_draws, monitor_ess = true)
    nat = cat((r.posterior_position[:, end-n_draws+1:end] for r in fit.results)...; dims = 3)
    # Python starts from native draws, so both chains start in the typical set.
    init = [nat[:, end, mod1(c, size(nat, 3))] for c in 1:chains]
    py = brm_jax_nuts(src, brm_jax_inputs(rk, src), init; chains, warmup = 1000, samples = n_draws, seed = 7)
    names, nv = constrained_scalars(rk.model.layout, nat)
    _, pv = constrained_scalars(rk.model.layout, py.draws)
    comparison = Dict{String,Any}()
    zmax = 0.0
    for n in names
        # A quantity constant in every draw (a correlation factor's unit
        # entry) has no Monte Carlo error: both sides must hold its value.
        if iszero(std(nv[n])) || iszero(std(pv[n]))
            @test nv[n] ≈ fill(first(nv[n]), size(nv[n])) && pv[n] ≈ fill(first(nv[n]), size(pv[n]))
            continue
        end
        a, b = summarize(nv[n]), summarize(pv[n])
        comparison[n] = (; native = a, python = b,
            z = abs(a.mean - b.mean) / sqrt(a.mcse^2 + b.mcse^2))
    end
    # The assertion is on the unconstrained coordinates, where the posterior is
    # closest to Gaussian and a mean's Monte Carlo error is best estimated; a
    # skewed positive scale's constrained mean (an HSGP amplitude) has the least
    # reliable one. The constrained comparison above is reported.
    for i in 1:size(nat, 1)
        a, b = summarize(nat[i, :, :]), summarize(py.draws[i, :, :])
        z = abs(a.mean - b.mean) / sqrt(a.mcse^2 + b.mcse^2)
        zmax = max(zmax, z)
        @test z < 4.5
    end
    native_divergences = sum(r.n_divergent_samples for r in fit.results)
    RESULTS["inference"] = (; chains, n_draws, native_seconds = native_time, python_seconds = py.elapsed,
        native_completed = fit.completion.n_completed, native_divergences, python_divergences = py.divergences,
        max_rhat = (native = maximum(c.native.rhat for c in values(comparison)), python = maximum(c.python.rhat for c in values(comparison))),
        min_bulk_ess = (native = minimum(c.native.bulk_ess for c in values(comparison)), python = minimum(c.python.bulk_ess for c in values(comparison))),
        min_tail_ess = (native = minimum(c.native.tail_ess for c in values(comparison)), python = minimum(c.python.tail_ess for c in values(comparison))),
        max_z = zmax, quantities = comparison)
    RESULTS["numpyro"] = py.versions
    # Predictions at posterior draws: the module's `mu` (`values = (:mu,)`)
    # at native draws of the fit against the native value query. A forecast
    # on new rows is not built for an `mi()` model yet: `reprocess` keeps the
    # fit's coordinates, which include the training rows' gaps.
    mu = prepare_query(rk.model, rk_bound_plan(rk), (:mu,))
    us = [nat[:, j, 1] for j in 1:100:n_draws]
    run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
    perr = maximum(maximum(abs.(vec(collect(run.datasets[1]["points"][j]["constrain_jit"]["mu"])) .-
        Base.invokelatest(mu, u)) ./ max.(1, abs.(Base.invokelatest(mu, u)))) for (j, u) in enumerate(us))
    @test perr < 1e-12
    RESULTS["predictions_at_draws_worst_relative_error"] = perr
    # Posterior summaries of each row's expected log dose rate from both fits
    # (every 10th draw of every chain), through the native value query.
    rowsummary(draws) = begin
        M = reduce(hcat, (Base.invokelatest(mu, draws[:, j, c]) for c in axes(draws, 3) for j in 1:10:size(draws, 2)))
        [(; median = median(r), q025 = quantile(r, 0.025), q25 = quantile(r, 0.25),
            q75 = quantile(r, 0.75), q975 = quantile(r, 0.975)) for r in eachrow(M)]
    end
    RESULTS["predictive"] = (; quantity = "mu (expected log dose rate) per row", native = rowsummary(nat),
        python = rowsummary(py.draws))
end
finally
    haskey(ENV, "BRM_JAX_RUN_DIR") && write_run_dir(ENV["BRM_JAX_RUN_DIR"])
end
