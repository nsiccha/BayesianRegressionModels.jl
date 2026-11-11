# Fit the Monster PBPK model of reproduce.jl with WarmupHMC and write what
# docs/src/monster.md shows: the population geometric means and geometric
# standard deviations with their sampling diagnostics, the sampler report, and
# the posterior predictive quantiles behind the figure (figures.jl draws it).
#
#   julia -t 4 --project=test research/monster/fit.jl [n_draws] [n_chains]

using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme

include(joinpath(@__DIR__, "..", "mcmc_harness.jl"))
include(joinpath(@__DIR__, "reproduce.jl"))

const RESULTS = joinpath(@__DIR__, "results")

const QUANTITY_NAMES = ["VPR", "Fwp", "Fpp", "Ff", "Fl", "Vwp", "Vpp", "Vl",
                        "Pba", "Pwp", "Ppp", "Pf", "Pl", "VMI", "KMI"]
const QUANTITY_DESCRIPTIONS = Dict(
    "VPR" => "ventilation-perfusion ratio",
    "Fwp" => "blood flow fraction, well-perfused tissue (unnormalized)",
    "Fpp" => "blood flow fraction, poorly perfused tissue (unnormalized)",
    "Ff" => "blood flow fraction, fat (unnormalized)",
    "Fl" => "blood flow fraction, liver (unnormalized)",
    "Vwp" => "lean-mass fraction, well-perfused tissue (unnormalized)",
    "Vpp" => "lean-mass fraction, poorly perfused tissue (unnormalized)",
    "Vl" => "lean-mass fraction, liver",
    "Pba" => "blood/air partition coefficient",
    "Pwp" => "tissue/blood partition coefficient, well-perfused tissue",
    "Ppp" => "tissue/blood partition coefficient, poorly perfused tissue",
    "Pf" => "tissue/blood partition coefficient, fat",
    "Pl" => "tissue/blood partition coefficient, liver",
    "VMI" => "maximal metabolic rate",
    "KMI" => "Michaelis constant",
)
const QUANTITIES = vcat(
    [(; name=q, unit="", description="population geometric mean, " * QUANTITY_DESCRIPTIONS[q],
        kind=:intercept, logical=Symbol("log_" * q), transform=exp) for q in QUANTITY_NAMES],
    [(; name="GSD " * q, unit="", description="population geometric SD, " * QUANTITY_DESCRIPTIONS[q],
        kind=:sd, logical=Symbol(q), transform=exp) for q in QUANTITY_NAMES],
    [(; name="sigma_venous", unit="", description="lognormal noise scale, venous blood",
        kind=:scalar, logical=:log_sigma_venous, transform=exp),
     (; name="sigma_exhaled", unit="", description="lognormal noise scale, exhaled air",
        kind=:scalar, logical=:log_sigma_exhaled, transform=exp)],
)

"""
The flat observation vectors of one measurement channel, times in hours. An
index column addresses the concatenated prediction vector of one person,
venous predictions at every time followed by exhaled-air predictions at every
time (`exhaled_index(t) = length(t) .+ (1:length(t))` in reproduce.jl), so an
exhaled index `j` samples the time `t[j - length(t)]`.
"""
function channel(data, response)
    experiment = endswith(String(response), "72") ? "72" : "144"
    times = getproperty(data, Symbol("times_" * experiment))
    index = getproperty(data, Symbol(String(response) * "_index"))
    observed = getproperty(data, response)
    sampled(t, k) = [t[j > length(t) ? j - length(t) : j] for j in k]
    flat_times = reduce(vcat, [sampled(t, k) for (t, k) in zip(times, index)]) ./ 60
    flat_observed = reduce(vcat, observed)
    length(flat_times) == length(flat_observed) ||
        error("$response: $(length(flat_times)) sampled times for $(length(flat_observed)) observations")
    (; subjects=reduce(vcat, [fill(s, length(o)) for (s, o) in zip(data.subject, observed)]),
       times=flat_times, observed=flat_observed)
end

function fit_all(; n_draws=1000, n_chains=4, seed=20261110, n_ppc=400)
    isdir(RESULTS) || mkpath(RESULTS)
    write_receipt(joinpath(RESULTS, "receipt.tsv"),
                  ["model" => "research/monster/reproduce.jl (the corrected nsiccha/monster program)",
                   "data" => "monster_data() in reproduce.jl: six persons, two exposures, nu = 4, 128 substeps",
                   "seed" => "$seed; predictive draws on $n_ppc thinned draws"];
                  n_chains, n_draws)
    data = monster_data()
    sb = monster_sbbrmi(data)
    fit = sample_quorum(sb, "monster"; n_chains, n_draws, seed)
    d = brm_descriptor(sb; name=:monster)
    names = constrained_names(fit.problem)
    C = [constrain_chain(fit.problem, c, seed + k) for (k, c) in enumerate(fit.chains)]
    population = population_rows(d, names, C, QUANTITIES)
    U = thin_draws(fit.chains, n_ppc)
    ppc = reduce(vcat, [ppc_rows(d, fit.problem, U, response; channel(data, response)..., seed)
                        for response in (:venous_72, :exhaled_72, :venous_144, :exhaled_144)])
    write_tsv(joinpath(RESULTS, "fits.tsv"), FITS_HEADER, [fit_row(fit)])
    write_tsv(joinpath(RESULTS, "population.tsv"), POPULATION_HEADER, population)
    write_tsv(joinpath(RESULTS, "ppc.tsv"), PPC_HEADER, ppc)
    report_fits([("monster", fit)])
    return (; fit, population, ppc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_all(; n_draws = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1000,
              n_chains = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4)
end
