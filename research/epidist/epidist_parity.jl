# Parity and benchmark: epidist's marginal model on this stack.
#
#   julia --project=test research/epidist/epidist_parity.jl <dir> [fit]
#
# <dir> holds what `epidist_reference.R` wrote: `marginal_rows.json`,
# `epidist_marginal.stan`, `epidist_marginal.data.json` (and `epidist_fit.json`).
# Writes `<dir>/parity.json` (density parity at random points, gradient timings)
# and, with `fit`, `<dir>/fits.json` (WarmupHMC fits of all four targets).
using JSON, Random, Statistics, Printf, LinearAlgebra
using LogDensityProblems, BridgeStan, WarmupHMC, MCMCDiagnosticTools
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, ADTypes
using DifferentiationInterface: AutoEnzyme

include(joinpath(@__DIR__, "epidist_models.jl"))

const DIR = abspath(ARGS[1])
const FIT = length(ARGS) >= 2 && ARGS[2] == "fit"

# ---------------------------------------------------------------- data
num(x::Number) = Float64(x)
num(x::AbstractString) = x == "Inf" ? Inf : x == "-Inf" ? -Inf : parse(Float64, x)
num(::Nothing) = NaN
meta = JSON.parsefile(joinpath(DIR, "marginal_rows.json"))
# The rows epidist hands to Stan, aggregated by unique (delay bounds, observation time,
# primary window) with weights n: Y = delay_lwr, vreal1..5 = relative_obs_time,
# pwindow, swindow, delay_upr, delay_min (`marginal_lognormal_lpmf`'s argument order).
standata = JSON.parsefile(joinpath(DIR, "epidist_marginal.data.json"))
delay_lwr = num.(standata["Y"])
relative_obs_time, pwindow, delay_upr = num.(standata["vreal1"]), num.(standata["vreal2"]), num.(standata["vreal4"])
n = round.(Int, num.(standata["weights"]))
all(iszero, num.(standata["vreal5"])) || error("delay_min ≠ 0 is not covered by this script")
N = length(delay_lwr)

# Priors: epidist's own defaults, read from its generated Stan program.
stan_code = read(joinpath(DIR, "epidist_marginal.stan"), String)
function normal_prior(code, param)
    m = match(Regex("normal_lpdf\\($param \\| ([0-9.eE+-]+), ([0-9.eE+-]+)\\)"), code)
    m === nothing && error("no normal prior on $param in epidist's program")
    parse.(Float64, m.captures)
end
prior_mu_location, prior_mu_scale = normal_prior(stan_code, "Intercept")
prior_sigma_location, prior_sigma_scale = normal_prior(stan_code, "Intercept_sigma")
@info "epidist data" N linelist = meta["n_linelist"] prior_mu_location prior_mu_scale prior_sigma_location prior_sigma_scale

df = (; delay_lwr, delay_upr, pwindow, relative_obs_time, n, count = copy(n),
        prior_mu_location, prior_mu_scale, prior_sigma_location, prior_sigma_scale)

# ---------------------------------------------------------------- targets
"A BridgeStan model as a LogDensityProblem (full density: propto = false)."
struct StanTarget
    model::BridgeStan.StanModel
end
LogDensityProblems.dimension(t::StanTarget) = Int(BridgeStan.param_unc_num(t.model))   # BridgeStan returns Int32
LogDensityProblems.capabilities(::Type{StanTarget}) = LogDensityProblems.LogDensityOrder{1}()
# Stan's own samplers treat an exception thrown by the model (a domain error deep in a
# tail) as a rejected point; BridgeStan raises it to Julia. Count and reject the same way.
const STAN_REJECTIONS = Threads.Atomic{Int}(0)
const STAN_FIRST_MESSAGE = Ref("")
function LogDensityProblems.logdensity(t::StanTarget, x)
    try
        BridgeStan.log_density(t.model, collect(Float64, x); propto = false)   # BridgeStan takes a Vector{Float64}, not a view
    catch err
        err isa ErrorException || rethrow()   # BridgeStan raises the model's message as error(...)
        Threads.atomic_add!(STAN_REJECTIONS, 1) == 0 && (STAN_FIRST_MESSAGE[] = err.msg)
        -Inf
    end
end
function LogDensityProblems.logdensity_and_gradient(t::StanTarget, x)
    try
        BridgeStan.log_density_gradient(t.model, collect(Float64, x); propto = false)
    catch err
        err isa ErrorException || rethrow()   # BridgeStan raises the model's message as error(...)
        Threads.atomic_add!(STAN_REJECTIONS, 1) == 0 && (STAN_FIRST_MESSAGE[] = err.msg)
        (-Inf, zero(x))
    end
end
Base.deepcopy_internal(t::StanTarget, ::IdDict) = t   # BridgeStan models are thread-safe and read-only

"A prepared RKPPL sampler query as a LogDensityProblem."
struct RKPPLTarget{Q}
    q::Q
    dim::Int
end
LogDensityProblems.dimension(t::RKPPLTarget) = t.dim
LogDensityProblems.capabilities(::Type{<:RKPPLTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(t::RKPPLTarget, x) = t.q(x)
function LogDensityProblems.logdensity_and_gradient(t::RKPPLTarget, x)
    g = similar(x)
    v, _ = sampler_value_and_gradient!(t.q, g, x)
    return v, g
end

timings = Dict{String,Any}()
targets = Dict{String,Any}()
names_of = Dict{String,Vector{String}}()

# epidist's own program
timings["epidist_stan_build_s"] = @elapsed epi_model = BridgeStan.StanModel(
    joinpath(DIR, "epidist_marginal.stan"), joinpath(DIR, "epidist_marginal.data.json"))
targets["epidist (brms Stan)"] = StanTarget(epi_model)
names_of["epidist (brms Stan)"] = BridgeStan.param_unc_names(epi_model)

# BRM -> SBBRMI (Stan)
brmi = epidist_brm(df)
timings["brm_sb_build_s"] = @elapsed begin
    sb = SBBRMI(brmi; mod = @__MODULE__)
    sb_problem = stan_instantiate(sb)
end
write(joinpath(DIR, "brm_sbbrmi.stan"), string(StanBlocks.stan_code(sb)))
targets["BRM SBBRMI"] = sb_problem
names_of["BRM SBBRMI"] = BridgeStan.param_unc_names(sb_problem.model)

# BRM -> RKBRMI (RKPPL)
const AD = AutoEnzyme(; mode = Enzyme.Reverse)
timings["brm_rk_build_s"] = @elapsed begin
    rk = RKBRMI(brmi)
    rk_problem = rk_logdensity_problem(rk; ad_backend = AD)
end
targets["BRM RKBRMI"] = rk_problem
names_of["BRM RKBRMI"] = string.(coordinate_names(build_kernel(rk_bound_plan(rk)).layout))

# hand-written RKPPL
timings["rkppl_build_s"] = @elapsed begin
    hand_plan = epidist_rkppl(; delay_upr, pwindow, relative_obs_time, n,
                         prior_mu_location, prior_mu_scale, prior_sigma_location, prior_sigma_scale) | (; delay_lwr)
    hand_built = build_kernel(hand_plan)
    hand_names = coordinate_names(hand_built.layout)
    hand_q = prepare_sampler(hand_built, hand_plan, zeros(length(hand_names)); backend = AD)
end
targets["RKPPL @rkppl"] = RKPPLTarget(hand_q, length(hand_names))
names_of["RKPPL @rkppl"] = string.(hand_names)
@info "coordinates" names_of

# Every target has two coordinates: the mu intercept and the log-sigma intercept.
# Map each target's order onto epidist's (Intercept, Intercept_sigma).
function epidist_order(names)
    s = findfirst(n -> occursin("sigma", lowercase(n)), names)
    s === nothing && error("no sigma coordinate among $names")
    length(names) == 2 || error("expected 2 coordinates, got $names")
    m = only(setdiff(1:2, s))
    return [m, s]          # permutation: target position of (mu, log sigma)
end
perm = Dict(k => epidist_order(v) for (k, v) in names_of)
to_target(k, x) = (y = similar(x); y[perm[k]] = x; y)
from_target(k, y) = y[perm[k]]

# ---------------------------------------------------------------- density parity
# 200 points around the truth (meanlog 1.8, log sdlog log 0.5), wide enough to reach
# the tails where a row's probability underflows; a point counts when every target
# returns a finite value there, and the non-finite counts are reported per target.
rng = Xoshiro(20261105)
points = [[1.8 + 0.6 * randn(rng), log(0.5) + 0.5 * randn(rng)] for _ in 1:200]
ref = "epidist (brms Stan)"
vals = Dict(k => [LogDensityProblems.logdensity_and_gradient(t, to_target(k, x)) for x in points] for (k, t) in targets)
finite_at(k, i) = isfinite(first(vals[k][i])) && all(isfinite, last(vals[k][i]))
common = [i for i in eachindex(points) if all(k -> finite_at(k, i), keys(targets))]
parity = Dict{String,Any}("points" => length(points), "finite_everywhere" => length(common))
for k in keys(targets)
    dlp = [first(vals[k][i]) - first(vals[ref][i]) for i in common]
    dg = maximum(maximum(abs.(from_target(k, last(vals[k][i])) .- last(vals[ref][i])) ./ max.(1.0, abs.(last(vals[ref][i])))) for i in common)
    nonfinite = count(i -> !finite_at(k, i), eachindex(points))
    parity[k] = Dict("max_abs_logdensity_diff" => maximum(abs, dlp), "mean_logdensity_diff" => mean(dlp),
                     "max_rel_gradient_diff" => dg, "nonfinite_points" => nonfinite,
                     "nonfinite_logdensities" => [string(first(vals[k][i])) for i in eachindex(points) if !finite_at(k, i)])
    @printf "%-22s max|Δlp| = %.3e  mean Δlp = %.3e  max rel |Δ∇| = %.3e  non-finite at %d of %d points\n" k maximum(abs, dlp) mean(dlp) dg nonfinite length(points)
    worst = common[sortperm(abs.(dlp); rev = true)[1:min(4, end)]]
    for i in worst
        @printf "    largest |Δlp| at (%.3f, %.3f): Δlp = %.3e, epidist lp = %.6g\n" points[i]... first(vals[k][i]) - first(vals[ref][i]) first(vals[ref][i])
    end
end
# The same comparison inside the posterior's bulk (the epidist fit's ±4 sd box).
bulk = [i for i in common if abs(points[i][1] - 1.8) < 0.25 && abs(points[i][2] - log(0.5)) < 0.3]
parity["bulk_points"] = length(bulk)
for k in keys(targets)
    dlp = [first(vals[k][i]) - first(vals[ref][i]) for i in bulk]
    dg = maximum(maximum(abs.(from_target(k, last(vals[k][i])) .- last(vals[ref][i])) ./ max.(1.0, abs.(last(vals[ref][i])))) for i in bulk)
    # SBBRMI's `~` statements drop normalising constants, so compare up to a constant too.
    parity[k]["bulk_max_abs_logdensity_diff"] = maximum(abs, dlp)
    parity[k]["bulk_constant_offset"] = median(dlp)
    parity[k]["bulk_max_abs_logdensity_diff_after_offset"] = maximum(abs, dlp .- median(dlp))
    parity[k]["bulk_max_rel_gradient_diff"] = dg
    @printf "%-22s bulk (%d points): max|Δlp| = %.3e  offset %.6f  max|Δlp - offset| = %.3e  max rel |Δ∇| = %.3e\n" k length(bulk) maximum(abs, dlp) median(dlp) maximum(abs, dlp .- median(dlp)) dg
end
for i in eachindex(points)
    any(k -> !finite_at(k, i), keys(targets)) || continue
    @printf "  point (%.3f, %.3f):" points[i]...
    for k in sort(collect(keys(targets)))
        @printf "  %s %.4g" k first(vals[k][i])
    end
    println()
end

# ---------------------------------------------------------------- gradient timing
function time_gradient(t, x; batches = 7, reps = 2000)
    LogDensityProblems.logdensity_and_gradient(t, x)   # compile
    best = Inf
    for _ in 1:batches
        s = @elapsed for _ in 1:reps
            LogDensityProblems.logdensity_and_gradient(t, x)
        end
        best = min(best, s / reps)
    end
    return best
end
grad_time = Dict{String,Float64}()
for (k, t) in targets
    grad_time[k] = time_gradient(t, to_target(k, [1.8, log(0.5)]))
    @printf "%-22s %.2f µs per density+gradient\n" k 1e6 * grad_time[k]
end

open(joinpath(DIR, "parity.json"), "w") do io
    JSON.print(io, Dict("N_rows" => N, "N_linelist" => meta["n_linelist"], "parity" => parity,
                        "gradient_seconds" => grad_time, "build_seconds" => timings,
                        "coordinates" => names_of, "threads" => Threads.nthreads()), 2)
end

# ---------------------------------------------------------------- fits
if FIT
    fits = Dict{String,Any}()
    # Every target starts its 4 chains at the same 4 points near the prior means
    # (WarmupHMC's default random start in [-2, 2] reaches regions where epidist's
    # program returns NaN on the large data set). All 4 chains run to the end
    # (a sampler comparison: min_completed = n). One untimed fit per target first,
    # so the timed fit pays no compilation.
    init_points = [[prior_mu_location, prior_sigma_location] .+ 0.1 .* randn(Xoshiro(77 + c), 2) for c in 1:4]
    for k in sort(collect(keys(targets)))
        t = targets[k]
        inits = [to_target(k, x) for x in init_points]
        run_fit(seed) = completion_warmup_mcmc([Xoshiro(seed + c) for c in 1:4], t;
                                               init = inits, min_completed = 4, n_draws = 1000)
        out = nothing
        cold = NaN
        secs = NaN
        STAN_REJECTIONS[] = 0
        STAN_FIRST_MESSAGE[] = ""
        try
            cold = @elapsed run_fit(2000)
            GC.gc()
            secs = @elapsed out = run_fit(1000)
        catch err
            fits[k] = Dict("error" => sprint(showerror, err), "first_fit_wall_s" => isnan(cold) ? nothing : cold)
            @printf "%-22s FAILED: %s\n" k fits[k]["error"]
            continue
        end
        chains = [reduce(hcat, [from_target(k, r.posterior_position[:, j]) for j in 1:size(r.posterior_position, 2)]) for r in out.results]
        # draws × chains × parameters for MCMCDiagnosticTools
        arr = permutedims(cat([permutedims(c) for c in chains]...; dims = 3), (1, 3, 2))
        ess_ = MCMCDiagnosticTools.ess(arr)
        rhat_ = MCMCDiagnosticTools.rhat(arr)
        all_draws = reduce(hcat, chains)
        fits[k] = Dict("wall_s" => secs, "first_fit_wall_s" => cold, "chains" => length(chains),
            "stan_model_exceptions_rejected" => STAN_REJECTIONS[],
            "stan_first_exception" => STAN_FIRST_MESSAGE[],
            "omitted" => collect(out.completion.omitted_chain_indices),
            "mu" => Dict("mean" => mean(all_draws[1, :]), "q5" => quantile(all_draws[1, :], 0.05), "q95" => quantile(all_draws[1, :], 0.95), "ess" => ess_[1], "rhat" => rhat_[1]),
            "log_sigma" => Dict("mean" => mean(all_draws[2, :]), "q5" => quantile(all_draws[2, :], 0.05), "q95" => quantile(all_draws[2, :], 0.95), "ess" => ess_[2], "rhat" => rhat_[2]))
        @printf "%-22s %.1f s  mu %.3f [%.3f, %.3f] ess %.0f  log σ %.3f [%.3f, %.3f] ess %.0f\n" k secs fits[k]["mu"]["mean"] fits[k]["mu"]["q5"] fits[k]["mu"]["q95"] ess_[1] fits[k]["log_sigma"]["mean"] fits[k]["log_sigma"]["q5"] fits[k]["log_sigma"]["q95"] ess_[2]
    end
    open(joinpath(DIR, "fits.json"), "w") do io
        JSON.print(io, Dict("fits" => fits, "truth" => meta["truth"], "threads" => Threads.nthreads()), 2)
    end
end
