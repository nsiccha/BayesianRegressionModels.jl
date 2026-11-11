module DocsExampleFit

using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools
using TreeArrays, Tables, Serialization, SHA
using DifferentiationInterface: AutoEnzyme

include("mcmc_harness.jl")

export fit_example, population_table, trajectory_table, predictive_table, write_tsv

"Replace receipt fields without changing the original sampling provenance."
function update_receipt(path, entries)
    replacement = Dict(string(k) => string(v) for (k, v) in entries)
    rows = Pair{String,String}[]
    for line in readlines(path)[2:end]
        key, value = split(line, '\t'; limit=2)
        haskey(replacement, key) || push!(rows, String(key) => String(value))
    end
    append!(rows, [string(k) => string(v) for (k, v) in entries])
    write_tsv(path, ["key", "value"], [(k, v) for (k, v) in rows])
end

"Fit once; keep the unconstrained draws outside git so summaries can be repaired without sampling."
function fit_example(sb, data, tag, results; n_draws=1000, n_chains=4,
                     seed=20261111, sources, inits=nothing, sampler_kwargs...)
    mkpath(results)
    code = stan_code(sb)
    harness = bytes2hex(sha256(read(joinpath(@__DIR__, "mcmc_harness.jl"))))
    identity = bytes2hex(sha256(code * repr(data) * repr((n_draws, n_chains, seed, inits, sampler_kwargs)) * harness))
    cache = joinpath(get(ENV, "BRM_DOCS_CACHE", joinpath(tempdir(), "brm-docs-examples")), tag, identity)
    mkpath(cache)
    rawfile = joinpath(cache, "draws.jls")
    receipt = joinpath(results, "receipt.tsv")
    if isfile(rawfile)
        raw = deserialize(rawfile)
        if hasproperty(raw, :receipt)
            write(receipt, raw.receipt)
        else
            # Older caches predate embedded receipts. Require their matching
            # original receipt instead of inventing a new sampling provenance.
            isfile(receipt) || error("saved draws need their original receipt: $receipt")
            fields = Dict(Pair(split(line, '\t'; limit=2)...) for line in readlines(receipt)[2:end])
            get(fields, "content_sha256", "") == identity ||
                error("receipt does not identify the saved draws: $rawfile")
        end
        problem = stan_instantiate(sb)
        fit = merge(raw, (; problem))
        println("CACHED ", tag, " ", identity)
    else
        write_receipt(receipt,
            ["model" => tag, "sources" => sources, "seed" => seed,
             "content_sha256" => identity, "harness_sha256" => harness,
             "settings" => repr((n_draws, n_chains, inits, sampler_kwargs)),
             "WarmupHMC" => string(pkgversion(WarmupHMC)),
             "StanBlocks" => string(pkgversion(StanBlocks)),
             "TreeArrays" => string(pkgversion(TreeArrays)),
             "BridgeStan" => string(pkgversion(BridgeStan)),
             "status" => "sampling started"];
            n_chains, n_draws,
            sampler=get(sampler_kwargs, :adaptive_centering, true) ?
                "WarmupHMC completion_warmup_mcmc, adaptive_centering_problem over BridgeStan, AutoEnzyme" :
                "WarmupHMC completion_warmup_mcmc over BridgeStan; model already uses explicit innovations")
        fit = sample_quorum(sb, tag; n_chains, n_draws, seed, inits, sampler_kwargs...)
        serialize(rawfile, (; fit.chains, fit.seconds, fit.n_chains, fit.divergent, fit.omitted,
                             receipt=read(receipt, String)))
    end
    write_tsv(joinpath(results, "fits.tsv"), FITS_HEADER, [fit_row(fit)])
    update_receipt(receipt, pairs((; status="summarizing", seconds=fit.seconds,
        divergent=fit.divergent, omitted_chains=join(fit.omitted, ","),
        dimension=LogDensityProblems.dimension(fit.problem))))
    d = brm_descriptor(sb; name=Symbol(tag))
    names = constrained_names(fit.problem)
    C = [constrain_chain(fit.problem, c, seed + k) for (k, c) in enumerate(fit.chains)]
    println("DRAWS_CACHE ", rawfile)
    return (; d, names, C, fit, seed, tag, results, identity)
end

"Natural-scale population summaries, with chain-aware R-hat and ESS."
function population_table(r, quantities)
    rows = map(quantities) do spec
        chains = quantity_draws(r.d, r.names, r.C, spec)
        tree = TreeData([TreeData(c, :draw) for c in chains], :chain)
        q = quantile(tree, (q025=0.025, q50=0.5, q975=0.975); dims=(:draw, :chain))
        row = only(Tables.rows(TreeTable(q)))
        diag = chain_diagnostics(chains)
        (spec.name, spec.unit, spec.description, fmt(row.q50), fmt(row.q025), fmt(row.q975),
         round(diag.rhat; digits=3), round(Int, diag.ess_bulk), round(Int, diag.ess_tail))
    end
    write_tsv(joinpath(r.results, "population.tsv"), POPULATION_HEADER, rows)
end

"A labelled trajectory, pooled over chains and draws in one TreeArrays reduction."
function trajectory_table(r, logical; time, observed=fill(NaN, length(time)),
                          truth=fill(NaN, length(time)), transform=identity, role=nothing,
                          filename="trajectory.tsv")
    # The caller owns the time coordinate of this derived vector. The generic
    # descriptor output has values but cannot infer calendar times from a UDF.
    post = TreeData([TreeData(transform.(brm_output_draws(r.d, C, r.names; logical, role)),
                              :draw, :time => time) for C in r.C], :chain)
    q = quantile(post, (q025=0.025, q25=0.25, q50=0.5, q75=0.75, q975=0.975);
                 dims=(:draw, :chain))
    rows = collect(Tables.rows(TreeTable(q)))
    length(rows) == length(time) || error("$logical: $(length(rows)) outputs for $(length(time)) times")
    out = [(time[j], observed[j], truth[j], fmt(v.q025), fmt(v.q25), fmt(v.q50), fmt(v.q75), fmt(v.q975))
           for (j, v) in enumerate(rows)]
    write_tsv(joinpath(r.results, filename),
              ["time", "observed", "truth", "q025", "q25", "q50", "q75", "q975"], out)
end

"Posterior predictive summaries; the descriptor selects the response carrier."
function predictive_table(r, response; time, observed)
    trajectory_table(r, response; time, observed, role=:posterior_predictive, filename="ppc.tsv")
    update_receipt(joinpath(r.results, "receipt.tsv"), pairs((;
        status="complete", completed=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        summary="TreeArrays quantiles pooled over chain and draw; split R-hat, bulk and tail ESS")))
end

end
