# Shared harness of the docs' MCMC fits (research/warfarin/fit.jl,
# research/monster/fit.jl): WarmupHMC's quorum sampler on an SBBRMI, the
# constrained draws, the population quantities a page reports with their
# diagnostics, the posterior predictive quantiles per observation, and the
# tab-separated tables the pages render.
#
# Expects `using BayesianRegressionModels, StanBlocks, BridgeStan,
# LogDensityProblems, WarmupHMC, Enzyme, Random, Statistics,
# MCMCDiagnosticTools` and `using DifferentiationInterface: AutoEnzyme` in the
# including script.

using Dates: now, Dates, DateTime, Day, @dateformat_str
import Serialization

# ---- sampling -----------------------------------------------------------------

"""
Run the quorum sampler on `sb`; return the BridgeStan problem and each kept
chain's draws. The draws are serialized to the cache directory as soon as
sampling ends, before anything is summarized, so a defect in the
post-processing never costs the sampling: with `BRM_DOCS_FIT_REUSE=1` a
later run of the script loads them instead of sampling again.
"""
function sample_quorum(sb, tag; n_chains, n_draws, seed, cache="brm-docs-fits",
                       inits=nothing, adaptive_centering=true, sampler_kwargs...)
    code = StanBlocks.stan_code(sb.model)
    dir = joinpath(tempdir(), cache)
    isdir(dir) || mkpath(dir)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(dir, "$(tag)_$(hash(code)).stan"))
    # The file name carries every setting that changes the draws, so a reuse
    # never returns draws sampled under other settings.
    settings = hash((inits, adaptive_centering, (; sampler_kwargs...)))
    draws_file = joinpath(dir, "$(tag)_$(hash(code))_$(n_chains)x$(n_draws)_$(seed)_$(settings).jls")
    if get(ENV, "BRM_DOCS_FIT_REUSE", "0") == "1" && isfile(draws_file)
        saved = Serialization.deserialize(draws_file)
        println("REUSED draws from ", draws_file)
        return (; problem, saved...)
    end
    target = adaptive_centering ? adaptive_centering_problem(sb, problem, AutoEnzyme()) : problem
    rng = Xoshiro(seed)
    starts = isnothing(inits) ? (;) :
        (; init=fill(brm_start(sb, problem.model, inits; base=:prior_median), n_chains))
    seconds = @elapsed out = try
        completion_warmup_mcmc(
            [Xoshiro(rand(rng, UInt64)) for _ in 1:n_chains], target;
            min_completed=max(1, div(3n_chains, 4)), n_draws, monitor_ess=true,
            starts..., sampler_kwargs...)
    catch err
        if err isa WarmupHMC.CompletionQuorumError
            for chain in err.outcome.completion.chains
                println(stderr, "CHAIN_FAILURE ", tag, " ", chain.chain_index, " ", chain.status)
                isnothing(chain.error) || showerror(stderr, chain.error)
                println(stderr)
            end
        end
        rethrow()
    end
    chains = [convert(Matrix{Float64}, r.posterior_position) for r in out.results]  # dim × draws, model frame
    saved = (; chains, seconds, n_chains,
               divergent=sum(r.n_divergent_samples for r in out.results),
               omitted=out.completion.omitted_chain_indices)
    Serialization.serialize(draws_file, saved)
    println("SAVED draws to ", draws_file)
    (; problem, saved...)
end

"Constrained draws of one chain with transformed parameters and generated quantities, draws × coordinates."
function constrain_chain(problem, chain, seed)
    rng = BridgeStan.StanRNG(problem.model, seed)
    rows = [BridgeStan.param_constrain(problem.model, collect(Float64, q);
                                       include_tp=true, include_gq=true, rng)
            for q in eachcol(chain)]
    return permutedims(reduce(hcat, rows))
end

constrained_names(problem) = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)

"At most `n` draws of the pooled chains, evenly spaced, as draws × coordinates."
function thin_draws(chains, n)
    pooled = reduce(hcat, chains)
    keep = round.(Int, range(1, size(pooled, 2); length=min(n, size(pooled, 2))))
    return permutedims(pooled[:, keep])
end

# ---- the reported quantities ----------------------------------------------------

"""
Per chain, the draws of one quantity on its natural scale. A spec names the
`kind` (`:intercept`: a predictor's population intercept, `:sd`: a
random-effect block's between-group SD, `:scalar`: a declared scalar
parameter), the descriptor address `logical` (the predictor, the block id, or
the parameter), and the `transform` to the reported scale.
"""
function quantity_draws(d, names, C_chains, spec)
    map(C_chains) do C
        raw = if spec.kind === :intercept
            q = brm_population_effect_coordinates(d, spec.logical, names; coefficient=:Intercept)
            C[:, only(q.coordinates)]
        elseif spec.kind === :sd
            block = only(brm_ranef_draws(d, C, names; id=spec.logical))
            vec(block.sd[:, 1])
        elseif spec.kind === :scalar
            vec(brm_output_draws(d, C, names; logical=spec.logical))
        else
            error("unknown quantity kind $(spec.kind)")
        end
        spec.transform.(raw)
    end
end

"Split R-hat, bulk and tail ESS of one quantity over its chains (the last `m` draws of each)."
function chain_diagnostics(chains::Vector{Vector{Float64}})
    m = minimum(length, chains)
    S = Array{Float64}(undef, m, length(chains), 1)
    for (k, c) in enumerate(chains)
        S[:, k, 1] = c[end-m+1:end]
    end
    (; rhat = only(MCMCDiagnosticTools.rhat(S)),
       ess_bulk = only(MCMCDiagnosticTools.ess(S; kind=:bulk)),
       ess_tail = only(MCMCDiagnosticTools.ess(S; kind=:tail)))
end

fmt(x) = round(x; sigdigits=4)

"The population table rows of one fit: median, 95 % interval and diagnostics per quantity."
function population_rows(d, names, C_chains, quantities)
    map(quantities) do spec
        chains = quantity_draws(d, names, C_chains, spec)
        pooled = reduce(vcat, chains)
        diag = chain_diagnostics(chains)
        (spec.name, spec.unit, spec.description,
         fmt(median(pooled)), fmt(quantile(pooled, 0.025)), fmt(quantile(pooled, 0.975)),
         round(diag.rhat; digits=3), round(Int, diag.ess_bulk), round(Int, diag.ess_tail))
    end
end
# One directory of tables per fit (DocsFits' summary contract): the quantity
# table has exactly these columns, and no model column; the page names the fit.
const POPULATION_HEADER = ["quantity", "unit", "description", "median", "q025", "q975",
                           "rhat", "ess_bulk", "ess_tail"]

"""
Posterior predictive quantiles of one response at every observation. `subjects`,
`times` and `observed` are flat vectors in the response's observation order.
"""
function ppc_rows(d, problem, U, response; subjects, times, observed, seed)
    predicted = getproperty(brm_predictive_draws(d, U; problem, seed), response)   # draws × observations
    size(predicted, 2) == length(observed) ||
        error("$response: $(size(predicted, 2)) predicted columns for $(length(observed)) observations")
    map(1:length(observed)) do j
        column = predicted[:, j]
        (String(response), subjects[j], times[j], observed[j],
         fmt(quantile(column, 0.025)), fmt(quantile(column, 0.25)), fmt(median(column)),
         fmt(quantile(column, 0.75)), fmt(quantile(column, 0.975)))
    end
end
const PPC_HEADER = ["response", "source_subject", "time", "observed",
                    "q025", "q25", "q50", "q75", "q975"]

"The one row of a fit's sampler report."
function fit_row(f)
    draws = sum(size(c, 2) for c in f.chains)
    (LogDensityProblems.dimension(f.problem), f.n_chains, length(f.chains),
     isempty(f.omitted) ? "none" : join(f.omitted, ","), draws,
     f.divergent, fmt(100 * f.divergent / draws), round(f.seconds; digits=1))
end
const FITS_HEADER = ["dimension", "chains_requested", "chains_kept", "omitted_chains",
                     "draws", "divergent", "divergent_pct", "seconds"]

# ---- tables ---------------------------------------------------------------------

function write_tsv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, '\t'))
        for row in rows
            println(io, join(string.(row), '\t'))
        end
    end
    println("TABLE ", basename(path), " (", length(rows), " rows)")
end

"""
Write `receipt.tsv`: what a results directory's numbers are the numbers of.
`entries` are further `key => value` pairs (models, data, seeds, ...).
"""
function write_receipt(path, entries; n_chains, n_draws,
                       sampler="WarmupHMC completion_warmup_mcmc, adaptive_centering_problem over the BridgeStan problem, AutoEnzyme")
    # A docs fit is done once and its results are committed; it is rerun when
    # its model, data or priors change, at most about once a month (user
    # direction of 2026-11-10, brief cth4h2, feedback 2). The previous receipt's
    # start time guards against a casual rerun; BRM_DOCS_FIT_FORCE=1 overrides.
    if isfile(path) && get(ENV, "BRM_DOCS_FIT_FORCE", "0") != "1"
        for line in readlines(path)
            key, value = split(line, '\t'; limit=2)
            key == "started" || continue
            # "2026-11-10 21:58 CET" or "2026-11-10T21:58:00": the first 16 characters.
            stamp = length(value) >= 16 ? replace(value[1:16], ' ' => 'T') : ""
            started = tryparse(DateTime, stamp, dateformat"yyyy-mm-ddTHH:MM")
            if !isnothing(started) && now() - started < Day(30)
                error("$path records a fit started $(value); a docs fit is rerun when its model, " *
                      "data or priors change, at most about once a month. Set BRM_DOCS_FIT_FORCE=1 " *
                      "to rerun it now.")
            end
        end
    end
    repo = dirname(@__DIR__)
    git(args...) = try
        strip(read(`git -C $repo $args`, String))
    catch
        "unavailable"
    end
    version = try
        string(something(pkgversion(BayesianRegressionModels), "unavailable"))
    catch
        "unavailable"
    end
    rows = vcat(
        [("commit", git("rev-parse", "HEAD")),
         ("uncommitted_changes", isempty(git("status", "--porcelain")) ? "none" : "yes"),
         ("started", Dates.format(now(), "yyyy-mm-dd HH:MM") * " local time"),
         ("host", "a Linux fleet host, $(Threads.nthreads()) Julia threads"),
         ("julia", string(VERSION)),
         ("BayesianRegressionModels", version),
         ("sampler", sampler),
         ("chains", "$n_chains per fit, quorum $(max(1, div(3n_chains, 4))), $n_draws draws each")],
        [(String(k), string(v)) for (k, v) in entries])
    write_tsv(path, ["key", "value"], rows)
end

function report_fits(fits)
    for (label, f) in fits
        println(label, ": ", length(f.chains), " chains kept of ", f.n_chains,
                ", omitted ", f.omitted, ", divergent ", f.divergent, ", ",
                round(f.seconds; digits=0), " s")
    end
end
