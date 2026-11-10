# Minibatch and streaming variational inference on a hierarchical count model.
#
# Executable source of the documentation page "Variational inference"
# (docs/src/variational-inference.md). The page shows the model declaration and the
# fitting calls below verbatim, read from this file at build time, and its figures and
# tables are drawn from the summaries that `main()` writes to results/.
#
#   Run:  julia --project=test research/variational_inference/svi.jl [sections] [cache-dir]
#         sections: comma-separated subset of probe,fit,stream,combine (default: fit,stream,combine)
#         cache-dir: where NUTS draws are kept between sections (default: a temporary directory)
#   Then: julia --project=test research/variational_inference/figures.jl
#
# Everything above `site_counts_model` is also evaluated by the docs build
# (docs/backend_comparisons.jl, `evaluate_source_prelude`), so that part loads only
# packages the docs environment has: BayesianRegressionModels, Distributions and the
# standard library. The fitting packages are loaded after the model.

using BayesianRegressionModels, Distributions
using Random, Statistics, LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# 1. The data: counts at monitoring sites over repeated visits
# ═══════════════════════════════════════════════════════════════════════════════

# The values the data are simulated from.
const GENERATING = (; a=1.0, b=0.4, sd_a=0.5, sd_b=0.25, cor_ab=0.3)

# `n_sites` sites, each visited `visits` times. Each site has its own intercept and
# slope, drawn together from a bivariate normal around the population values; the
# count at a visit is Poisson with log rate `a_i + b_i * x`. The data are a NamedTuple
# of nested vectors: one entry per site, holding that site's visits.
function simulate_sites(; n_sites=400, visits=8, seed=1, g=GENERATING)
    rng = Xoshiro(seed)
    C = [g.sd_a^2 g.cor_ab * g.sd_a * g.sd_b; g.cor_ab * g.sd_a * g.sd_b g.sd_b^2]
    L = cholesky(C).L
    effects = [L * randn(rng, 2) for _ in 1:n_sites]
    x = [randn(rng, visits) for _ in 1:n_sites]
    y = [[rand(rng, Poisson(exp(g.a + e[1] + (g.b + e[2]) * xj))) for xj in xi]
         for (e, xi) in zip(effects, x)]
    data = (; site=[string("s", i) for i in 1:n_sites], x, y)
    (; data, effects)
end

site_counts_data(; kwargs...) = simulate_sites(; kwargs...).data

# ═══════════════════════════════════════════════════════════════════════════════
# 2. The model
# ═══════════════════════════════════════════════════════════════════════════════

function site_counts_model(data = site_counts_data())
    @brm data begin
        a ~ 1 + (1 | s | site)
        b ~ 1 + (1 | s | site)
        effect(a, Intercept) ~ Normal(0, 1)
        effect(b, Intercept) ~ Normal(0, 1)
        sd(:, s) ~ Exponential(1)
        @plate for i in eachindex(a)
            y[i] ~ Poisson(exp.(a[i] .+ b[i] .* x[i]))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fitting (not evaluated by the docs build; needs the test environment)
# ═══════════════════════════════════════════════════════════════════════════════

using ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC, MCMCDiagnosticTools
using Serialization
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

const RESULTS = joinpath(@__DIR__, "results")

# ── The two fits of the full data ─────────────────────────────────────────────

# SVI: one call. The units are the sites; each iteration draws a batch of 32 of them.
fit_svi(rk; seed=1) = brm_svi_fit(rk; batch_size=32, rng=Xoshiro(seed), ndraws=2000)

# NUTS: four adaptive chains of WarmupHMC on the same log density, one after the
# other, 1000 draws each after warm-up. Each chain starts near the origin of the
# unconstrained coordinates (unit random-effect sds, no site effects), jittered per
# chain, with a unit diagonal metric.
function fit_nuts(rk; seed=10, chains=4, n_draws=1000)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    d = LogDensityProblems.dimension(problem)
    map(1:chains) do c
        rng = Xoshiro(seed + c)
        init = (; position=0.1 .* randn(rng, d), squared_scale=ones(d))
        adaptive_warmup_mcmc(rng, problem; init, n_draws)
    end
end

# ── Summaries ─────────────────────────────────────────────────────────────────

# R-hat, bulk and tail ESS over the unconstrained coordinates of `chains`
# (each `dimension × draws`), and the divergence count: WarmupHMC's consumer
# acceptance set.
function nuts_diagnostics(chains)
    n = minimum(size(c.posterior_position, 2) for c in chains)
    samples = cat((permutedims(c.posterior_position[:, 1:n]) for c in chains)...; dims=3)
    samples = permutedims(samples, (1, 3, 2))           # draws × chains × parameters
    divergent = sum(c.n_divergent_samples for c in chains)
    (; max_rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
       min_bulk_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
       min_tail_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:tail)),
       divergent, retained=n * length(chains),
       divergent_percent=100divergent / (n * length(chains)),
       gradient_evaluations=sum(c.sampling_evaluation_counter for c in chains))
end

nuts_draws(chains) = reduce(hcat, (convert(Matrix{Float64}, c.posterior_position) for c in chains))

quantiles(v) = (; median=median(v), q025=quantile(v, 0.025), q25=quantile(v, 0.25),
                  q75=quantile(v, 0.75), q975=quantile(v, 0.975), mean=mean(v), sd=std(v))

# The population parameters of `site_counts_model`, read from restored draws, each as
# a vector over draws on its natural scale. The random-effect block `s` of the two
# predictors restores as `b_s_site`: `tau` (margin sds, 2 × draws), `L` (one lower
# Cholesky factor of the margins' correlation per draw) and `z` (one sites × 2 matrix
# of standardized site draws per draw). Margin 1 is `a`, margin 2 is `b`.
function population(restored)
    block = restored.b_s_site
    (; a=restored.a_Intercept, b=restored.b_Intercept,
       sd_a=block.tau[1, :], sd_b=block.tau[2, :],
       cor_ab=[(L * L')[1, 2] for L in block.L])
end

# Each site's intercept and slope deviation from the population values, sites × draws.
function site_effects(restored)
    block = restored.b_s_site
    E = [block.tau[:, k] .* (block.L[k] * block.z[k]') for k in eachindex(block.L)]
    (; a=reduce(hcat, (e[1, :] for e in E)), b=reduce(hcat, (e[2, :] for e in E)))
end

const POPULATION_LABELS = (; a="intercept a", b="slope b", sd_a="sd of site intercepts",
                             sd_b="sd of site slopes", cor_ab="correlation of intercepts and slopes")

function write_table(name, header, rows)
    mkpath(RESULTS)
    open(joinpath(RESULTS, name), "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join(row, '\t'))
        end
    end
    println("TABLE ", name, " (", length(rows), " rows)")
end

fmt(x::AbstractFloat) = string(round(x; sigdigits=6))
fmt(x) = string(x)

# ── Sections ──────────────────────────────────────────────────────────────────

# Section `fit`: the full data, SVI against NUTS.
function section_fit(; cache)
    (; data, effects) = simulate_sites()
    rk = RKBRMI(site_counts_model(data))
    println("FIT sites=$(length(data.site)) observations=$(sum(length, data.y)) " *
            "coordinates=$(length(coordinate_names(rk.model.layout)))")

    svi_seconds = @elapsed svi = fit_svi(rk)
    dg = svi.diagnostics
    println("SVI seconds=$svi_seconds iterations=$(dg.iterations) converged=$(dg.converged) " *
            "unit_gradient_calls=$(dg.unit_gradient_calls) stages=$(dg.stages)")

    nuts_seconds = @elapsed chains = fit_nuts(rk)
    diag = nuts_diagnostics(chains)
    println("NUTS seconds=$nuts_seconds $diag")
    serialize(joinpath(cache, "nuts_full.jls"), (; draws=nuts_draws(chains), diag, seconds=nuts_seconds))

    svi_r = rk_restore_draws(rk, svi.draws)
    nuts_r = rk_restore_draws(rk, nuts_draws(chains))
    svi_pop, nuts_pop = population(svi_r), population(nuts_r)

    rows = []
    for (key, label) in pairs(POPULATION_LABELS), (method, pop) in (("NUTS", nuts_pop), ("SVI", svi_pop))
        q = quantiles(pop[key])
        push!(rows, (label, method, fmt.(values(q))..., fmt(GENERATING[key])))
    end
    write_table("population.tsv", ("parameter", "method", "median", "q025", "q25", "q75", "q975",
                                   "mean", "sd", "generating"), rows)

    # Thinned draws of the population parameters for the marginal densities.
    thin(v, n=500) = v[round.(Int, range(1, length(v); length=n))]
    rows = []
    for (key, label) in pairs(POPULATION_LABELS), (method, pop) in (("NUTS", nuts_pop), ("SVI", svi_pop))
        for v in thin(pop[key])
            push!(rows, (label, method, fmt(v)))
        end
    end
    write_table("population_draws.tsv", ("parameter", "method", "value"), rows)

    svi_e, nuts_e = site_effects(svi_r), site_effects(nuts_r)
    rows = []
    for (key, label) in (("a", "site intercept deviation"), ("b", "site slope deviation"))
        S, N = getproperty(svi_e, Symbol(key)), getproperty(nuts_e, Symbol(key))
        truth = [e[key == "a" ? 1 : 2] for e in effects]
        for i in eachindex(data.site)
            push!(rows, (data.site[i], label, fmt(mean(N[i, :])), fmt(std(N[i, :])),
                         fmt(mean(S[i, :])), fmt(std(S[i, :])), fmt(truth[i])))
        end
    end
    write_table("site_effects.tsv", ("site", "effect", "nuts_mean", "nuts_sd", "svi_mean", "svi_sd",
                                     "generating"), rows)

    # Where the SVI approximation sits against NUTS, per coordinate group: the mean
    # error in NUTS posterior sds and the sd ratio.
    moments(S, N) = (; mean_error=abs.(vec(mean(S; dims=2)) .- vec(mean(N; dims=2))) ./ vec(std(N; dims=2)),
                       sd_ratio=vec(std(S; dims=2)) ./ vec(std(N; dims=2)))
    rows = []
    for (group, S, N) in (("population parameters", reduce(hcat, values(svi_pop))', reduce(hcat, values(nuts_pop))'),
                          ("site intercepts", svi_e.a, nuts_e.a), ("site slopes", svi_e.b, nuts_e.b))
        m = moments(S, N)
        push!(rows, (group, size(S, 1), fmt(median(m.mean_error)), fmt(maximum(m.mean_error)),
                     fmt(quantile(m.sd_ratio, 0.05)), fmt(median(m.sd_ratio)), fmt(quantile(m.sd_ratio, 0.95))))
    end
    write_table("approximation.tsv", ("group", "coordinates", "mean_error_median", "mean_error_max",
                                      "sd_ratio_q05", "sd_ratio_median", "sd_ratio_q95"), rows)

    write_table("fits.tsv", ("method", "seconds", "iterations", "unit_gradient_calls", "gradient_evaluations",
                             "draws", "max_rhat", "min_bulk_ess", "min_tail_ess", "divergent", "divergent_percent",
                             "sites", "observations", "coordinates"),
        [("SVI", fmt(svi_seconds), dg.iterations, dg.unit_gradient_calls, "", size(svi.draws, 2), "", "", "", "", "",
          length(data.site), sum(length, data.y), length(coordinate_names(rk.model.layout))),
         ("NUTS", fmt(nuts_seconds), "", "", diag.gradient_evaluations, diag.retained, fmt(diag.max_rhat),
          fmt(diag.min_bulk_ess), fmt(diag.min_tail_ess), diag.divergent, fmt(diag.divergent_percent),
          length(data.site), sum(length, data.y), length(coordinate_names(rk.model.layout)))])

    # The ELBO trace by window, with the stage each window belongs to.
    rows = []
    per_window = length(dg.elbo) ÷ length(dg.elbo_windows_trimmed)
    bounds = cumsum((dg.stages.exploring, dg.stages.settling, dg.stages.polishing))
    for (w, value) in enumerate(dg.elbo_windows_trimmed)
        last_iteration = min(w * per_window, length(dg.elbo))
        stage = last_iteration <= bounds[1] ? "exploring" : last_iteration <= bounds[2] ? "settling" : "polishing"
        push!(rows, (w, last_iteration, fmt(value), stage))
    end
    write_table("elbo.tsv", ("window", "iteration", "elbo_trimmed_mean", "stage"), rows)
    (; rk, svi, chains)
end

# ── Streaming: the same sites, two more visits per window ─────────────────────

# Window `w` holds visits `2w - 1` and `2w` of the sites present in it. Sites arrive
# over time: 300 in the first window, 40 more in each of the next windows (400 in all).
function stream_windows(data; n_windows=4, first_sites=300, new_sites_per_window=40)
    visits = length(first(data.x))
    per_window = visits ÷ n_windows
    map(1:n_windows) do w
        n = min(length(data.site), first_sites + (w - 1) * new_sites_per_window)
        range = ((w - 1) * per_window + 1):(w * per_window)
        (; site=data.site[1:n], x=[xi[range] for xi in data.x[1:n]], y=[yi[range] for yi in data.y[1:n]])
    end
end

# Every observation seen up to and including window `w`, in one data set.
function accumulated(windows, w)
    sites = windows[w].site
    x = [reduce(vcat, (window.x[i] for window in windows[1:w] if i <= length(window.site))) for i in eachindex(sites)]
    y = [reduce(vcat, (window.y[i] for window in windows[1:w] if i <= length(window.site))) for i in eachindex(sites)]
    (; site=sites, x, y)
end

# The stream: calibrate on the first window, then continue through the others,
# carrying only the checkpoint from one window to the next.
function stream_fits(windows)
    fit = brm_svi_fit(site_counts_model(windows[1]); batch_size=32, rng=Xoshiro(1))
    fits = [fit]
    for w in 2:length(windows)
        state = brm_svi_checkpoint(fit)
        fit = brm_svi_fit(state, site_counts_model(windows[w]); batch_size=32, rng=Xoshiro(w))
        push!(fits, fit)
    end
    fits
end

checkpoint_bytes(state) = (io = IOBuffer(); serialize(io, state); position(io))

function section_stream(; cache)
    (; data) = simulate_sites()
    windows = stream_windows(data)
    stream_seconds = @elapsed fits = stream_fits(windows)
    println("STREAM total_seconds=$stream_seconds")

    rows = []
    cost_rows = []
    for w in eachindex(windows)
        all_data = accumulated(windows, w)
        rk = RKBRMI(site_counts_model(all_data))
        fit = fits[w]
        dg = fit.diagnostics
        bytes = checkpoint_bytes(brm_svi_checkpoint(fit))
        println("STREAM window=$w sites=$(length(windows[w].site)) new_observations=$(sum(length, windows[w].y)) " *
                "accumulated=$(sum(length, all_data.y)) seconds=$(dg.seconds) iterations=$(dg.iterations) " *
                "unit_gradient_calls=$(dg.unit_gradient_calls) converged=$(dg.converged) bytes=$bytes")
        # The streaming fit's draws are in the window's own coordinates, so they are
        # restored with the window's backend.
        stream_pop = population(rk_restore_draws(fit.diagnostics.model.backend, fit.draws))

        # Batch SVI on everything seen so far.
        batch = fit_svi(rk; seed=100 + w)
        batch_pop = population(rk_restore_draws(rk, batch.draws))

        # NUTS on everything seen so far (the last window is the full data of `fit`).
        path = joinpath(cache, w == length(windows) ? "nuts_full.jls" : "nuts_window_$w.jls")
        if isfile(path)
            saved = deserialize(path)
            nuts, nuts_seconds = saved.draws, saved.seconds
        else
            nuts_seconds = @elapsed chains = fit_nuts(rk; seed=200 + 10w)
            nuts = nuts_draws(chains)
            diag = nuts_diagnostics(chains)
            println("STREAM NUTS window=$w seconds=$nuts_seconds $diag")
            serialize(path, (; draws=nuts, diag, seconds=nuts_seconds))
        end
        nuts_pop = population(rk_restore_draws(rk, nuts))

        for (key, label) in pairs(POPULATION_LABELS), (method, pop) in
                (("NUTS", nuts_pop), ("streaming SVI", stream_pop), ("batch SVI", batch_pop))
            q = quantiles(pop[key])
            push!(rows, (w, label, method, fmt.(values(q))...))
        end
        push!(cost_rows, (w, length(windows[w].site), sum(length, windows[w].y), sum(length, all_data.y),
                          fmt(dg.seconds), dg.iterations, dg.unit_gradient_calls, bytes,
                          fmt(batch.diagnostics.seconds), batch.diagnostics.iterations, fmt(nuts_seconds)))
    end
    write_table("streaming.tsv", ("window", "parameter", "method", "median", "q025", "q25", "q75", "q975",
                                  "mean", "sd"), rows)
    write_table("streaming_cost.tsv", ("window", "sites", "new_observations", "accumulated_observations",
                                       "streaming_seconds", "streaming_iterations", "streaming_unit_gradient_calls",
                                       "checkpoint_bytes", "batch_seconds", "batch_iterations", "nuts_seconds"), cost_rows)
end

# ── A start from fits on slices of the sites ──────────────────────────────────

# Fits of four slices of the sites, their checkpoints combined into a start for the
# full fit. `full` is prepared once, so the combination and the fit share it.
function combined_start_fit(data, slices)
    slice_fits = [brm_svi_fit(site_counts_model(slice); batch_size=32, rng=Xoshiro(s))
                  for (s, slice) in enumerate(slices)]
    full = brm_nlme_model(RKBRMI(site_counts_model(data)); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    start = brm_svi_combine(full, brm_svi_checkpoint.(slice_fits))
    started = brm_svi_fit(full; start, batch_size=32, rng=Xoshiro(1), ndraws=2000)
    (; full, start, started)
end

function section_combine(; cache)
    (; data) = simulate_sites()
    slices = [(; site=data.site[r], x=data.x[r], y=data.y[r])
              for r in Iterators.partition(eachindex(data.site), 100)]
    started_seconds = @elapsed (; full, start, started) = combined_start_fit(data, slices)
    cold_seconds = @elapsed cold = brm_svi_fit(full; batch_size=32, rng=Xoshiro(1), ndraws=2000)

    saved = deserialize(joinpath(cache, "nuts_full.jls"))
    rk = full.backend
    nuts_pop = population(rk_restore_draws(rk, saved.draws))
    rows = []
    for (name, fit, seconds) in (("cold start", cold, cold_seconds), ("start from four slices", started, started_seconds))
        dg = fit.diagnostics
        pop = population(rk_restore_draws(rk, fit.draws))
        errors = [abs(mean(pop[k]) - mean(nuts_pop[k])) / std(nuts_pop[k]) for k in keys(POPULATION_LABELS)]
        push!(rows, (name, fmt(dg.seconds), dg.iterations, dg.stages.exploring, dg.unit_gradient_calls,
                     fmt(last(dg.elbo_windows_trimmed)), fmt(maximum(errors))))
        println("COMBINE $name seconds=$seconds iterations=$(dg.iterations) stages=$(dg.stages) " *
                "max_mean_error=$(maximum(errors))")
    end
    r = start.receipt.globals
    println("COMBINE receipt informed=$(r.informed) clamped=$(r.clamped)")
    println("COMBINE wall seconds: started (slices + combination + fit)=$started_seconds cold=$cold_seconds")
    write_table("combine.tsv", ("start", "seconds", "iterations", "exploring_iterations", "unit_gradient_calls",
                                "final_elbo_trimmed_mean", "max_population_mean_error_sds"), rows)
    write_table("combine_receipt.tsv", ("coordinate", "informed", "clamped", "spread"),
        [(string(r.coordinates[k]), r.informed[k], r.clamped[k], fmt(r.spread[k])) for k in eachindex(r.coordinates)])
end

# ── A small run that prints the restored names and shapes ─────────────────────

function section_probe(; cache)
    (; data) = simulate_sites(; n_sites=12, visits=4)
    brmi = site_counts_model(data)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    println("STAN\n", stan_code(sb))
    rk = RKBRMI(brmi)
    println("COORDINATES ", coordinate_names(rk.model.layout))
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    u = zeros(LogDensityProblems.dimension(problem))
    LogDensityProblems.logdensity_and_gradient(problem, u)
    t = @elapsed for _ in 1:20
        LogDensityProblems.logdensity_and_gradient(problem, u)
    end
    println("GRADIENT seconds_each=", t / 20)
    fit = brm_svi_fit(rk; batch_size=4, rng=Xoshiro(1), ndraws=50, max_iterations=40)
    restored = rk_restore_draws(rk, fit.draws)
    describe(prefix, v::NamedTuple) = foreach(((k, x),) -> describe(string(prefix, ".", k), x), pairs(v))
    describe(prefix, v::AbstractArray) = println("RESTORED ", prefix, " ", typeof(v), " ", size(v),
        v isa Vector{<:AbstractMatrix} ? string(" each ", size(first(v))) : "")
    describe(prefix, v) = println("RESTORED ", prefix, " ", typeof(v))
    foreach(((k, v),) -> describe(string(k), v), pairs(restored))
    block = restored.b_s_site
    println("BLOCK L*L' = ", block.L[1] * block.L[1]', " tau = ", block.tau[:, 1])
    pop = population(restored)
    println("POPULATION ", map(v -> round(mean(v); sigdigits=3), pop))
    eff = site_effects(restored)
    println("EFFECTS sizes ", size(eff.a), " ", size(eff.b))
    println("DIAGNOSTICS ", keys(fit.diagnostics))
    println("STAGES ", fit.diagnostics.stages)
    chain = only(fit_nuts(rk; seed=2, chains=1, n_draws=60))
    println("NUTS ", size(chain.posterior_position), " divergent=", chain.n_divergent_samples,
            " evaluations=", chain.sampling_evaluation_counter, " stepsize=", chain.stepsize)
    state = brm_svi_checkpoint(fit)
    println("CHECKPOINT bytes=", checkpoint_bytes(state), " ", fieldnames(typeof(state)))
    windows = stream_windows(data; n_windows=2, first_sites=8, new_sites_per_window=4)
    next = brm_svi_fit(state, site_counts_model(windows[2]); batch_size=4, rng=Xoshiro(3), ndraws=10, max_iterations=5)
    println("STREAM ", size(next.draws), " ", keys(brm_svi_checkpoint(next).units))
    full = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    slices = [brm_svi_fit(site_counts_model((; site=data.site[r], x=data.x[r], y=data.y[r])); batch_size=4,
                          rng=Xoshiro(s), ndraws=0, max_iterations=20) for (s, r) in enumerate((1:6, 7:12))]
    start = brm_svi_combine(full, brm_svi_checkpoint.(slices))
    println("COMBINE ", keys(start.receipt), " ", keys(start.receipt.globals))
    started = brm_svi_fit(full; start, batch_size=4, rng=Xoshiro(1), ndraws=10, max_iterations=5)
    println("STARTED ", size(started.draws))
end

const SECTIONS = Dict("probe" => section_probe, "fit" => section_fit, "stream" => section_stream,
                      "combine" => section_combine)

function main(args=ARGS)
    sections = isempty(args) || startswith(first(args), "/") ? ["fit", "stream", "combine"] : split(first(args), ',')
    cache = length(args) >= 2 ? args[2] : (length(args) == 1 && startswith(first(args), "/")) ? first(args) : mktempdir()
    mkpath(cache)
    println("CACHE ", cache)
    for name in sections
        println("SECTION ", name)
        SECTIONS[String(name)](; cache)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
