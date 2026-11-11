# Minibatch and streaming variational inference on a hierarchical count model.
#
# Executable source of the documentation page "Variational inference"
# (docs/src/variational-inference.md). The page shows the model declaration and the
# fitting calls below verbatim, read from this file at build time, and its figures and
# tables are drawn from the summaries that `main()` writes to results/.
#
#   Run:  julia --project=test research/variational_inference/svi.jl [sections] [cache-dir]
#         sections: comma-separated subset of probe,fit,stream,combine (default: fit,stream,combine)
#         cache-dir: persistent fit cache (default: research/variational_inference/cache)
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
using Serialization, SHA, Dates
using Treebars: with_progress, snapshot_progress
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

const RESULTS = joinpath(@__DIR__, "results")
const SOURCE_SHA256 = bytes2hex(sha256(read(@__FILE__)))
const RUN_STARTED = string(now(UTC))
const RUN_COMMIT = readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
const RUN_DIRTY = !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`))

# ── The two fits of the full data ─────────────────────────────────────────────

# SVI: one call. The units are the sites; each iteration draws a batch of 32 of them.
fit_svi(rk; seed=1, progress=nothing) =
    brm_svi_fit(rk; batch_size=32, rng=Xoshiro(seed), ndraws=2000, progress)

# NUTS: four adaptive chains of WarmupHMC on the same log density, with a quorum
# of three completed chains and at least 4000 draws each after warm-up.
# Run Julia with one thread for these Enzyme gradients. Each chain starts near the origin of the
# unconstrained coordinates (unit random-effect sds, no site effects), jittered per
# chain, with a unit diagonal metric.
function fit_nuts(rk; seed=10, chains=4, n_draws=4000, progress=nothing)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    d = LogDensityProblems.dimension(problem)
    rngs = [Xoshiro(seed + c) for c in 1:chains]
    init = [(; position=0.1 .* randn(rng, d), squared_scale=ones(d)) for rng in rngs]
    completion_warmup_mcmc(rngs, problem;
        init, n_draws, min_completed=max(1, div(3chains, 4)), progress).results
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
    retained = sum(size(c.posterior_position, 2) for c in chains)
    (; max_rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
       min_bulk_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
       min_tail_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:tail)),
       divergent, retained, chains=length(chains),
       divergent_percent=100divergent / retained,
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
    mkpath(dirname(joinpath(RESULTS, name)))
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

# Cache data, never compiled models or AD workspaces. Hash the scientific
# functions without comments or line numbers, plus data, settings and seeds.
# Changes to figures, prose and summary writers therefore reuse measured fits.
const RECIPE = "site-counts-v2-quorum"
function fit_program(source=read(@__FILE__, String))
    names = (:simulate_sites, :site_counts_model, :fit_svi, :fit_nuts,
             :stream_windows, :accumulated, :stream_fits, :combined_start_fit)
    definitions = filter(Meta.parseall(source).args) do ex
        ex isa Expr && ex.head in (:function, :(=)) || return false
        signature = ex.args[1]
        signature isa Expr && signature.head == :call && first(signature.args) in names
    end
    length(definitions) == length(names) || error("Missing scientific function in cache key")
    # remove_linenums! retains the location argument of a macro invocation.
    # Normalize that too, so adding a comment above @brm cannot invalidate a fit.
    normalize(ex) = ex isa LineNumberNode ? LineNumberNode(0) :
        ex isa Expr ? Expr(ex.head, normalize.(ex.args)...) : ex
    join(string.(Base.remove_linenums!.(normalize.(definitions))), "\n")
end

function cache_key(input, settings, seed; program=fit_program())
    io = IOBuffer()
    serialize(io, (RECIPE, program, input, settings, seed, VERSION,
                   pkgversion(BayesianRegressionModels), pkgversion(WarmupHMC), pkgversion(Enzyme)))
    bytes2hex(sha256(take!(io)))
end

function cached(f, cache, name, input; settings, seed, progress=nothing)
    path = joinpath(cache, name * ".jls")
    digest = cache_key(input, settings, seed)
    if isfile(path)
        saved = deserialize(path)
        saved.digest == digest || error("Cache input changed for $name; choose a new cache directory for an intentional refit.")
        println("REUSE ", name)
        return saved.result
    end
    result = with_progress(progress; description=name) do p
        f(p)
    end
    provenance = (; recipe=RECIPE, created=string(now(UTC)), started=RUN_STARTED,
                    commit=RUN_COMMIT, uncommitted_changes=RUN_DIRTY, host="a Linux fleet host", julia=string(VERSION),
                    brm=string(pkgversion(BayesianRegressionModels)),
                    warmuphmc=string(pkgversion(WarmupHMC)),
                    enzyme=string(pkgversion(Enzyme)), threads=Threads.nthreads(),
                    source_sha256=SOURCE_SHA256, settings, seed,
                    program_sha256=bytes2hex(sha256(fit_program())))
    serialize(path * ".tmp", (; digest, provenance, result))
    mv(path * ".tmp", path)
    println("SAVED ", name)
    result
end

function write_receipt(cache; status)
    rows = Pair{String,String}[
        "recipe" => RECIPE,
        "status" => status,
        "summary generated (UTC)" => string(now(UTC)),
        "summary source SHA256" => SOURCE_SHA256,
        "commit" => RUN_COMMIT,
        "uncommitted_changes" => string(RUN_DIRTY),
        "started" => RUN_STARTED,
        "host" => "a Linux fleet host",
        "harness" => "BRM minibatch SVI and WarmupHMC completion sampler; " * RECIPE,
        "Julia" => string(VERSION),
        "BayesianRegressionModels" => string(pkgversion(BayesianRegressionModels)),
        "WarmupHMC" => string(pkgversion(WarmupHMC)),
        "Enzyme" => string(pkgversion(Enzyme)),
        "threads" => string(Threads.nthreads()),
        "data seed" => "1",
        "full SVI seed / draws / batch size" => "1 / 2000 / 32",
        "NUTS draws per chain / requested chains / quorum" => "4000 / 4 / 3",
        "NUTS seeds" => "full: 11–14; window w: 201+10w through 204+10w",
        "streaming / batch / slice SVI seeds" => "window w / 100+w / slice index",
    ]
    for name in sort(readdir(cache))
        endswith(name, ".jls") || continue
        name == "nuts_full.jls" && continue # superseded 1000-draw reference, kept locally
        saved = deserialize(joinpath(cache, name))
        fit_rows = Pair{String,String}[
            "key" => saved.digest,
            "harness" => "BRM minibatch SVI and WarmupHMC completion sampler; " * RECIPE,
            "host" => "a Linux fleet host",
        ]
        for key in (:commit, :uncommitted_changes, :started)
            hasproperty(saved.provenance, key) || push!(fit_rows,
                string(key) => "not recorded by the initial harness")
        end
        for (key, value) in pairs(saved.provenance)
            key == :host && continue
            push!(fit_rows, string(key) => string(value))
        end
        push!(rows, "$name input SHA256" => saved.digest)
        push!(rows, "$name fitted (UTC)" => saved.provenance.created)
        push!(rows, "$name fit source SHA256" => saved.provenance.source_sha256)
        result = saved.result
        if hasproperty(result, :diag)
            push!(rows, "$name seconds" => fmt(result.seconds))
            push!(rows, "$name divergent" => string(result.diag.divergent))
            push!(rows, "$name omitted chains" => string(4 - result.diag.chains))
            push!(rows, "$name dimension" => string(size(result.draws, 1)))
            append!(fit_rows, ["seconds" => fmt(result.seconds),
                "dimension" => string(size(result.draws, 1)),
                "omitted_chains" => string(4 - result.diag.chains)])
            append!(fit_rows, [string(k) => string(v) for (k, v) in pairs(result.diag)])
        elseif hasproperty(result, :seconds)
            push!(rows, "$name seconds" => fmt(result.seconds))
            push!(fit_rows, "total_seconds" => fmt(result.seconds))
        end
        fits = hasproperty(result, :fit) ? [result.fit] : hasproperty(result, :fits) ? result.fits :
               hasproperty(result, :diagnostics) ? [result] : []
        for (i, fit) in enumerate(fits)
            prefix = length(fits) == 1 ? "" : "window_$i "
            push!(fit_rows, prefix * "dimension" => string(size(fit.draws, 1)))
            for key in (:iterations, :converged, :stages, :unit_gradient_calls, :seconds)
                push!(fit_rows, prefix * string(key) => string(getproperty(fit.diagnostics, key)))
            end
            if length(fits) > 1
                window_rows = ["key" => bytes2hex(sha256(saved.digest * "/window_$i")),
                    "stream_key" => saved.digest, "window" => string(i),
                    "host" => "a Linux fleet host", "harness" => "BRM sequential SVI; " * RECIPE,
                    "dimension" => string(size(fit.draws, 1))]
                append!(window_rows, [string(k) => string(v) for (k, v) in pairs(saved.provenance) if k != :host])
                append!(window_rows, [string(k) => string(getproperty(fit.diagnostics, k))
                    for k in (:iterations, :converged, :stages, :unit_gradient_calls, :seconds)])
                write_table(joinpath(splitext(name)[1], "window_$i", "receipt.tsv"),
                    ("field", "value"), window_rows)
            end
        end
        write_table(joinpath(splitext(name)[1], "receipt.tsv"), ("field", "value"), fit_rows)
    end
    write_table("receipt.tsv", ("field", "value"), [(k, v) for (k, v) in rows])
end

function svi_record(fit)
    dg = fit.diagnostics
    diagnostics = (; dg.iterations, dg.converged, dg.stages, dg.elbo,
                     dg.elbo_windows_trimmed, dg.unit_gradient_calls, dg.seconds)
    (; fit.draws, diagnostics, state=brm_svi_checkpoint(fit))
end

function cached_nuts(rk, data, cache, name; seed=10, n_draws=4000, progress=nothing)
    cached(cache, name * "_draws_$n_draws", data;
           settings=(; n_draws, chains=4, quorum=3, initialization="jittered origin, unit diagonal metric"), seed, progress) do p
        seconds = @elapsed chains = fit_nuts(rk; seed, n_draws, progress=p)
        diag = nuts_diagnostics(chains)
        println("NUTS $name seconds=$seconds $diag")
        (; draws=nuts_draws(chains), diag, seconds)
    end
end

function write_population(name, pop; diag=nothing)
    rows = map(collect(pairs(POPULATION_LABELS))) do (key, description)
        q = quantiles(pop[key])
        if diag !== nothing
            # Equal total proves that every completed chain has the requested length.
            @assert diag.retained == 4000 * diag.chains
            samples = reshape(pop[key], 4000, diag.chains, 1)
            diagnostics = (only(MCMCDiagnosticTools.rhat(samples)),
                only(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
                only(MCMCDiagnosticTools.ess(samples; kind=:tail)))
        else
            diagnostics = ("", "", "") # MCMC diagnostics do not apply to variational draws
        end
        (string(key), "dimensionless", description, fmt(q.median), fmt(q.q025), fmt(q.q975), fmt.(diagnostics)...)
    end
    write_table(joinpath(name, "population.tsv"),
        ("quantity", "unit", "description", "median", "q025", "q975", "rhat", "ess_bulk", "ess_tail"), rows)
end

# ── Sections ──────────────────────────────────────────────────────────────────

# Section `fit`: the full data, SVI against NUTS.
function section_fit(; cache, progress=nothing)
    (; data, effects) = simulate_sites()
    rk = RKBRMI(site_counts_model(data))
    println("FIT sites=$(length(data.site)) observations=$(sum(length, data.y)) " *
            "coordinates=$(length(coordinate_names(rk.model.layout)))")

    saved_svi = cached(cache, "svi_full", data; settings=(; batch_size=32, ndraws=2000), seed=1, progress) do p
        seconds = @elapsed fit = fit_svi(rk; progress=p)
        (; fit=svi_record(fit), seconds)
    end
    svi, svi_seconds = saved_svi.fit, saved_svi.seconds
    dg = svi.diagnostics
    println("SVI seconds=$svi_seconds iterations=$(dg.iterations) converged=$(dg.converged) " *
            "unit_gradient_calls=$(dg.unit_gradient_calls) stages=$(dg.stages)")

    saved_nuts = cached_nuts(rk, data, cache, "nuts_full"; progress)
    nuts_seconds, diag = saved_nuts.seconds, saved_nuts.diag
    println("NUTS seconds=$nuts_seconds $diag")

    svi_r = rk_restore_draws(rk, svi.draws)
    nuts_r = rk_restore_draws(rk, saved_nuts.draws)
    svi_pop, nuts_pop = population(svi_r), population(nuts_r)

    rows = []
    for (key, label) in pairs(POPULATION_LABELS), (method, pop) in (("NUTS", nuts_pop), ("SVI", svi_pop))
        q = quantiles(pop[key])
        push!(rows, (label, method, fmt.(values(q))..., fmt(GENERATING[key])))
    end
    write_table("population.tsv", ("parameter", "method", "median", "q025", "q25", "q75", "q975",
                                   "mean", "sd", "generating"), rows)

    write_population("nuts_full_draws_4000", nuts_pop; diag)
    write_population("svi_full", svi_pop)

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
                             "sites", "observations", "coordinates", "chains"),
        [("SVI", fmt(svi_seconds), dg.iterations, dg.unit_gradient_calls, "", size(svi.draws, 2), "", "", "", "", "",
          length(data.site), sum(length, data.y), length(coordinate_names(rk.model.layout)), ""),
         ("NUTS", fmt(nuts_seconds), "", "", diag.gradient_evaluations, diag.retained, fmt(diag.max_rhat),
          fmt(diag.min_bulk_ess), fmt(diag.min_tail_ess), diag.divergent, fmt(diag.divergent_percent),
          length(data.site), sum(length, data.y), length(coordinate_names(rk.model.layout)), diag.chains)])

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
    nothing
end

# ── Streaming: the same sites, two more visits per window ─────────────────────

# Window `w` holds visits `2w - 1` and `2w` of the sites present in it. Sites arrive
# over time: 300 in the first window, then 340, 380 and 400.
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
function stream_fits(windows; progress=nothing)
    fit = brm_svi_fit(site_counts_model(windows[1]); batch_size=32, rng=Xoshiro(1), progress)
    fits = Any[fit] # a continuation carries a BRMSVIState where the initial fit has nothing
    for w in 2:length(windows)
        state = brm_svi_checkpoint(fit)
        fit = brm_svi_fit(state, site_counts_model(windows[w]); batch_size=32, rng=Xoshiro(w), progress)
        push!(fits, fit)
    end
    fits
end

checkpoint_bytes(state) = (io = IOBuffer(); serialize(io, state); position(io))

function section_stream(; cache, progress=nothing)
    (; data) = simulate_sites()
    windows = stream_windows(data)
    saved_stream = cached(cache, "stream_svi", windows;
        settings=(; batch_size=32, ndraws=1000), seed=collect(eachindex(windows)), progress) do p
        seconds = @elapsed fits = stream_fits(windows; progress=p)
        (; fits=svi_record.(fits), seconds)
    end
    fits = saved_stream.fits
    println("STREAM total_seconds=$(saved_stream.seconds)")

    rows = []
    cost_rows = []
    diagnostic_rows = []
    for w in eachindex(windows)
        all_data = accumulated(windows, w)
        rk = RKBRMI(site_counts_model(all_data))
        fit = fits[w]
        dg = fit.diagnostics
        bytes = checkpoint_bytes(fit.state)
        println("STREAM window=$w sites=$(length(windows[w].site)) new_observations=$(sum(length, windows[w].y)) " *
                "accumulated=$(sum(length, all_data.y)) seconds=$(dg.seconds) iterations=$(dg.iterations) " *
                "unit_gradient_calls=$(dg.unit_gradient_calls) converged=$(dg.converged) bytes=$bytes")
        # The streaming fit's draws are in the window's own coordinates, so they are
        # restored with the window's backend.
        window_rk = RKBRMI(site_counts_model(windows[w]))
        stream_pop = population(rk_restore_draws(window_rk, fit.draws))

        # Batch SVI on everything seen so far.
        batch = cached(cache, "batch_svi_window_$w", all_data;
            settings=(; batch_size=32, ndraws=2000), seed=100+w, progress) do p
            svi_record(fit_svi(rk; seed=100 + w, progress=p))
        end
        batch_pop = population(rk_restore_draws(rk, batch.draws))

        # Each window needs its own reference: late sites have no earlier visits
        # in the stream, so even its final accumulated data differ from `fit`.
        saved = cached_nuts(rk, all_data, cache, "nuts_window_$w"; seed=200 + 10w, progress)
        nuts, nuts_seconds = saved.draws, saved.seconds
        diag = saved.diag
        push!(diagnostic_rows, (w, diag.chains, diag.retained, fmt(diag.max_rhat),
            fmt(diag.min_bulk_ess), fmt(diag.min_tail_ess), diag.divergent,
            fmt(diag.divergent_percent), dg.converged, batch.diagnostics.converged))
        nuts_pop = population(rk_restore_draws(rk, nuts))
        write_population("nuts_window_$(w)_draws_4000", nuts_pop; diag)
        write_population("batch_svi_window_$w", batch_pop)
        write_population(joinpath("stream_svi", "window_$w"), stream_pop)

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
    write_table("streaming_diagnostics.tsv", ("window", "chains", "draws", "max_rhat",
        "min_bulk_ess", "min_tail_ess", "divergent", "divergent_percent",
        "streaming_converged", "batch_converged"), diagnostic_rows)
end

# ── A start from fits on slices of the sites ──────────────────────────────────

# Fits of four slices of the sites, their checkpoints combined into a start for the
# full fit. `full` is prepared once, so the combination and the fit share it.
function combined_start_fit(data, slices; progress=nothing)
    slice_fits = [brm_svi_fit(site_counts_model(slice); batch_size=32, rng=Xoshiro(s), progress)
                  for (s, slice) in enumerate(slices)]
    full = brm_nlme_model(RKBRMI(site_counts_model(data)); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    start = brm_svi_combine(full, brm_svi_checkpoint.(slice_fits))
    started = brm_svi_fit(full; start, batch_size=32, rng=Xoshiro(1), ndraws=2000, progress)
    (; full, start, started)
end

function section_combine(; cache, progress=nothing)
    (; data) = simulate_sites()
    slices = [(; site=data.site[r], x=data.x[r], y=data.y[r])
              for r in Iterators.partition(eachindex(data.site), 100)]
    combined = cached(cache, "combined_start", (data, slices);
        settings=(; batch_size=32, slice_draws=1000, ndraws=2000),
        seed=(; slices=collect(eachindex(slices)), full=1), progress) do p
        seconds = @elapsed result = combined_start_fit(data, slices; progress=p)
        (; fit=svi_record(result.started), receipt=result.start.receipt, seconds)
    end
    started, started_seconds = combined.fit, combined.seconds
    # The cold fit is exactly the full-data fit already measured in `fit`.
    cold_saved = cached(cache, "svi_full", data; settings=(; batch_size=32, ndraws=2000), seed=1, progress) do p
        error("Run the fit section first to create the cold-start reference")
    end
    cold, cold_seconds = cold_saved.fit, cold_saved.seconds
    rk = RKBRMI(site_counts_model(data))
    saved = cached_nuts(rk, data, cache, "nuts_full"; progress)
    nuts_pop = population(rk_restore_draws(rk, saved.draws))
    write_population("combined_start", population(rk_restore_draws(rk, started.draws)))
    rows = []
    for (name, fit, seconds) in (("cold start", cold, cold_seconds), ("start from four slices", started, started_seconds))
        dg = fit.diagnostics
        pop = population(rk_restore_draws(rk, fit.draws))
        errors = [abs(mean(pop[k]) - mean(nuts_pop[k])) / std(nuts_pop[k]) for k in keys(POPULATION_LABELS)]
        push!(rows, (name, fmt(dg.seconds), fmt(seconds), dg.iterations, dg.stages.exploring, dg.unit_gradient_calls,
                     fmt(last(dg.elbo_windows_trimmed)), fmt(maximum(errors))))
        println("COMBINE $name seconds=$seconds iterations=$(dg.iterations) stages=$(dg.stages) " *
                "max_mean_error=$(maximum(errors))")
    end
    r = combined.receipt.globals
    println("COMBINE receipt informed=$(r.informed) clamped=$(r.clamped)")
    println("COMBINE wall seconds: started (slices + combination + fit)=$started_seconds cold=$cold_seconds")
    write_table("combine.tsv", ("start", "seconds", "total_seconds", "iterations", "exploring_iterations", "unit_gradient_calls",
                                "final_elbo_trimmed_mean", "max_population_mean_error_sds"), rows)
    write_table("combine_receipt.tsv", ("coordinate", "informed", "clamped", "spread"),
        [(string(r.coordinates[k]), r.informed[k], r.clamped[k], fmt(r.spread[k])) for k in eachindex(r.coordinates)])
end

# ── A small run that prints the restored names and shapes ─────────────────────

function section_probe(; cache, progress=nothing)
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
    cache = length(args) >= 2 ? args[2] : (length(args) == 1 && startswith(first(args), "/")) ? first(args) : joinpath(@__DIR__, "cache")
    mkpath(cache)
    println("CACHE ", cache)
    write_receipt(cache; status="building summaries")
    with_progress(:state; description="Variational inference documentation") do progress
        snapshot_progress(progress, get(ENV, "KB_PROGRESS_FILE", joinpath(cache, "progress.txt"))) do
            for name in sections
                println("SECTION ", name)
                SECTIONS[String(name)](; cache, progress)
            end
        end
    end
    write_receipt(cache; status="completed sections: " * join(sections, ", "))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
