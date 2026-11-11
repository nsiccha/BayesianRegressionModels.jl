# Run deliberately, once per model/data/settings key, never from make.jl.
# julia --startup-file=no --threads=1 --project=test research/start/fit.jl
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools, SHA, JSON
using DifferentiationInterface: AutoEnzyme
include("reproduce.jl")
include("../mcmc_harness.jl")

const RESULTS = joinpath(@__DIR__, "results")

function fit_intro(; n_draws=1000, n_chains=4, seed=20261111)
    mkpath(RESULTS)
    model = intro_model()
    sb = SBBRMI(model; mod=@__MODULE__)
    settings = (; n_draws, n_chains, min_completed=3, seed)
    harness_version = bytes2hex(sha256(read(joinpath(@__DIR__, "../mcmc_harness.jl"))))
    program = stan_code(sb)
    data = stan_data(sb)
    key = bytes2hex(sha256(program * JSON.json(data) * JSON.json(settings) * harness_version))
    receipt = joinpath(RESULTS, "receipt.tsv")
    if isfile(receipt)
        recorded = Dict{String,String}(Tuple(split(line, '\t'; limit=2))
                                      for line in readlines(receipt)[2:end])
        get(recorded, "key", "") == key && haskey(recorded, "completed") &&
            all(name -> isfile(joinpath(RESULTS, name)),
                ("population.tsv", "ppc.tsv", "fits.tsv")) && return (; key, cached=true)
    end
    write_receipt(receipt,
        ["key" => key, "harness_version" => harness_version,
         "settings" => JSON.json(settings), "seed" => seed,
         "StanBlocks" => pkgversion(StanBlocks), "WarmupHMC" => pkgversion(WarmupHMC),
         "Enzyme" => pkgversion(Enzyme)];
        n_chains, n_draws)

    fit = sample_quorum(sb, "start"; n_chains, n_draws, seed)
    descriptor = brm_descriptor(sb)
    names = constrained_names(fit.problem)
    constrained = [constrain_chain(fit.problem, c, seed+k) for (k,c) in enumerate(fit.chains)]
    quantities = [("Intercept", "response", "population intercept", :Intercept),
                  ("x", "response/x", "population slope", :x)]
    rows = []
    for (quantity, unit, description, coefficient) in quantities
        coordinate = only(brm_population_effect_coordinates(descriptor, :loc, names; coefficient).coordinates)
        chains = [c[:, coordinate] for c in constrained]
        pooled = reduce(vcat, chains)
        diag = chain_diagnostics(chains)
        push!(rows, (quantity, unit, description, fmt(median(pooled)), fmt(quantile(pooled, 0.025)),
                     fmt(quantile(pooled, 0.975)), round(diag.rhat; digits=3),
                     round(Int, diag.ess_bulk), round(Int, diag.ess_tail)))
    end
    err = (; name="err", unit="response", description="residual standard deviation",
             kind=:scalar, logical=:err, transform=identity)
    append!(rows, population_rows(descriptor, names, constrained, [err]))
    write_tsv(joinpath(RESULTS, "population.tsv"),
        POPULATION_HEADER, rows)
    predictive = ppc_rows(descriptor, fit.problem, thin_draws(fit.chains, 400), :y;
        subjects=column_data(model, :subj), times=column_data(model, :x),
        observed=column_data(model, :y), seed)
    write_tsv(joinpath(RESULTS, "ppc.tsv"), PPC_HEADER, predictive)
    write_tsv(joinpath(RESULTS, "fits.tsv"), FITS_HEADER, [fit_row(fit)])
    telemetry = (; key, seconds=fit.seconds, divergent=fit.divergent,
                   omitted_chains=fit.omitted, dimension=LogDensityProblems.dimension(fit.problem))
    open(receipt, "a") do io
        for (k,v) in pairs(telemetry)
            k == :key && continue
            println(io, k, '\t', v)
        end
        println(io, "completed\t", now())
        println(io, "status\tcompleted")
    end
    report_fits([("start", fit)])
    return telemetry
end

abspath(PROGRAM_FILE) == (@__FILE__) && fit_intro()
