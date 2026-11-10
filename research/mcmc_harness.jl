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

# ---- sampling -----------------------------------------------------------------

"Run the quorum sampler on `sb`; return the BridgeStan problem and each kept chain's draws."
function sample_quorum(sb, tag; n_chains, n_draws, seed, cache="brm-docs-fits")
    code = StanBlocks.stan_code(sb.model)
    dir = joinpath(tempdir(), cache)
    isdir(dir) || mkpath(dir)
    problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(dir, "$(tag)_$(hash(code)).stan"))
    target = adaptive_centering_problem(sb, problem, AutoEnzyme())
    rng = Xoshiro(seed)
    seconds = @elapsed out = completion_warmup_mcmc(
        [Xoshiro(rand(rng, UInt64)) for _ in 1:n_chains], target;
        min_completed=max(1, div(3n_chains, 4)), n_draws, monitor_ess=true)
    chains = [convert(Matrix{Float64}, r.posterior_position) for r in out.results]  # dim × draws, model frame
    (; problem, chains, seconds, n_chains,
       divergent=sum(r.n_divergent_samples for r in out.results),
       omitted=out.completion.omitted_chain_indices)
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
function population_rows(model, d, names, C_chains, quantities)
    map(quantities) do spec
        chains = quantity_draws(d, names, C_chains, spec)
        pooled = reduce(vcat, chains)
        diag = chain_diagnostics(chains)
        (model, spec.name, spec.unit, spec.description,
         fmt(median(pooled)), fmt(quantile(pooled, 0.025)), fmt(quantile(pooled, 0.975)),
         round(diag.rhat; digits=3), round(Int, diag.ess_bulk), round(Int, diag.ess_tail))
    end
end
const POPULATION_HEADER = ["model", "quantity", "unit", "description", "median", "q025", "q975",
                           "rhat", "ess_bulk", "ess_tail"]

"""
Posterior predictive quantiles of one response at every observation. `subjects`,
`times` and `observed` are flat vectors in the response's observation order.
"""
function ppc_rows(model, d, problem, U, response; subjects, times, observed, seed)
    predicted = getproperty(brm_predictive_draws(d, U; problem, seed), response)   # draws × observations
    size(predicted, 2) == length(observed) ||
        error("$response: $(size(predicted, 2)) predicted columns for $(length(observed)) observations")
    map(1:length(observed)) do j
        column = predicted[:, j]
        (model, String(response), subjects[j], times[j], observed[j],
         fmt(quantile(column, 0.025)), fmt(quantile(column, 0.25)), fmt(median(column)),
         fmt(quantile(column, 0.75)), fmt(quantile(column, 0.975)))
    end
end
const PPC_HEADER = ["model", "response", "source_subject", "time", "observed",
                    "q025", "q25", "q50", "q75", "q975"]

"One row of the sampler report."
function fit_row(label, f)
    draws = sum(size(c, 2) for c in f.chains)
    (label, LogDensityProblems.dimension(f.problem), f.n_chains, length(f.chains),
     isempty(f.omitted) ? "none" : join(f.omitted, ","), draws,
     f.divergent, fmt(100 * f.divergent / draws), round(f.seconds; digits=1))
end
const FITS_HEADER = ["model", "dimension", "chains_requested", "chains_kept", "omitted_chains",
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

function report_fits(fits)
    for (label, f) in fits
        println(label, ": ", length(f.chains), " chains kept of ", f.n_chains,
                ", omitted ", f.omitted, ", divergent ", f.divergent, ", ",
                round(f.seconds; digits=0), " s")
    end
end
