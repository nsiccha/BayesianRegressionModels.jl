# Fit the warfarin models of reproduce.jl and joint_reproduce.jl on the public
# 32-subject data (data/README.md) and write what docs/src/warfarin.md shows:
# the population estimates with their sampling diagnostics, each subject's PK
# posterior medians for the second stage, and the posterior predictive
# quantiles behind the figures (figures.jl draws them).
#
#   julia -t 4 --project=test research/warfarin/fit.jl [n_draws] [n_chains]
#
# Order: the PK stage; its per-subject medians go to
# results/pk_subject_medians.tsv, which warfarin_pd_data() reads; then the PD
# stage conditional on them; then the joint model. Every fit runs n_chains
# adaptive WarmupHMC chains through the quorum sampler (brm-use, "Sampling an
# SBBRMI"), keeps the first 3/4 that finish, and reports the chains left out.

using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme

include(joinpath(@__DIR__, "..", "mcmc_harness.jl"))
include(joinpath(@__DIR__, "joint_reproduce.jl"))   # reproduce.jl, data.jl, the three models

const RESULTS = joinpath(@__DIR__, "results")

const PK_QUANTITIES = [
    (; name="tlag", unit="h", description="population lag time", kind=:intercept, logical=:lag_logit, transform=x -> 1 / (1 + exp(-x))),
    (; name="ka", unit="1/h", description="population absorption rate", kind=:intercept, logical=:log_ka, transform=exp),
    (; name="CL", unit="L/h", description="population clearance at 70 kg", kind=:intercept, logical=:log_cl0, transform=exp),
    (; name="V", unit="L", description="population volume at 70 kg", kind=:intercept, logical=:log_v0, transform=exp),
    (; name="sd(logit tlag)", unit="", description="between-subject SD of the lag logit", kind=:sd, logical=:tlag_bsv, transform=identity),
    (; name="sd(log ka)", unit="", description="between-subject SD of log ka", kind=:sd, logical=:ka_bsv, transform=identity),
    (; name="sd(log CL)", unit="", description="between-subject SD of log CL", kind=:sd, logical=:cl_bsv, transform=identity),
    (; name="sd(log V)", unit="", description="between-subject SD of log V", kind=:sd, logical=:v_bsv, transform=identity),
    (; name="sigma_pk", unit="mg/L", description="additive PK noise scale", kind=:scalar, logical=:sigma_pk, transform=identity),
    (; name="kappa_pk", unit="", description="PK overdispersion (times 25)", kind=:scalar, logical=:kappa_pk, transform=identity),
]
const PD_QUANTITIES = [
    (; name="R0", unit="%", description="population baseline prothrombin complex activity", kind=:intercept, logical=:log_r0, transform=exp),
    (; name="1/kout", unit="h", description="population turnover time", kind=:intercept, logical=:log_inv_kout, transform=exp),
    (; name="EC50", unit="mg/L", description="population concentration of half-maximal inhibition", kind=:intercept, logical=:log_ec50, transform=exp),
    (; name="sd(log R0)", unit="", description="between-subject SD of log R0", kind=:sd, logical=:r0_bsv, transform=identity),
    (; name="sd(log 1/kout)", unit="", description="between-subject SD of log turnover time", kind=:sd, logical=:kout_bsv, transform=identity),
    (; name="sd(log EC50)", unit="", description="between-subject SD of log EC50", kind=:sd, logical=:ec50_bsv, transform=identity),
    (; name="sigma_pd", unit="%", description="additive PD noise scale", kind=:scalar, logical=:sigma_pd, transform=identity),
    (; name="kappa_pd", unit="", description="PD overdispersion (times 625)", kind=:scalar, logical=:kappa_pd, transform=identity),
]

"The flat observation vectors of one channel, in the order the model observes them."
function channel(data, time_column, dv_column)
    times = getproperty(data, time_column)
    (; subjects=reduce(vcat, [fill(s, length(t)) for (s, t) in zip(data.source_subject, times)]),
       times=reduce(vcat, times), observed=reduce(vcat, getproperty(data, dv_column)))
end

function fit_all(; n_draws=1000, n_chains=4, seed=20261110, n_ppc=400)
    isdir(RESULTS) || mkpath(RESULTS)
    # The receipt first: it names the commit and settings these results are the
    # results of, and a defect in it fails before an hour of sampling.
    write_receipt(joinpath(RESULTS, "receipt.tsv"),
                  ["models" => "research/warfarin/reproduce.jl and joint_reproduce.jl",
                   "data" => "research/warfarin/data/warfarin_data.txt (32 subjects) through data.jl",
                   "seed" => "$seed (PK), $(seed + 1) (PD), $(seed + 2) (joint); predictive draws on $n_ppc thinned draws"];
                  n_chains, n_draws)
    data = warfarin_data()
    pk_channel = channel(data, :pk_time, :pk_dv)
    pd_channel = channel(data, :pd_time, :pd_dv)
    fits = []
    population = []
    ppc = []

    # Stage 1: population PK.
    sb_pk = SBBRMI(warfarin_pk_brmi(data); mod=@__MODULE__)
    pk = sample_quorum(sb_pk, "warfarin_pk"; n_chains, n_draws, seed)
    d_pk = brm_descriptor(sb_pk; name=:warfarin_pk)
    names_pk = constrained_names(pk.problem)
    C_pk = [constrain_chain(pk.problem, c, seed + k) for (k, c) in enumerate(pk.chains)]
    push!(fits, ("two-stage PK", pk))
    append!(population, population_rows("two-stage PK", d_pk, names_pk, C_pk, PK_QUANTITIES))
    append!(ppc, ppc_rows("two-stage PK", d_pk, pk.problem, thin_draws(pk.chains, n_ppc), :pk_dv;
                          pk_channel..., seed))

    # Each subject's PK medians, the second stage's fixed inputs. The linear
    # predictors have one value per subject; log_cl0 and log_v0 carry the
    # weight effect, as the PD model expects.
    C_all = reduce(vcat, C_pk)
    per_subject(logical) = brm_output_draws(d_pk, C_all, names_pk; logical, role=:linear_predictor)
    lag_logit = per_subject(:lag_logit)
    medians = (;
        pk_log_tlag = vec(median(-log1pexp.(-lag_logit); dims=1)),
        pk_log_ka = vec(median(per_subject(:log_ka); dims=1)),
        pk_log_cl = vec(median(per_subject(:log_cl0); dims=1)),
        pk_log_v = vec(median(per_subject(:log_v0); dims=1)),
    )
    write_tsv(WARFARIN_PK_MEDIANS_FILE,
              ["subject", "source_subject", "pk_log_tlag", "pk_log_ka", "pk_log_cl", "pk_log_v"],
              [(s, data.source_subject[s], medians.pk_log_tlag[s], medians.pk_log_ka[s],
                medians.pk_log_cl[s], medians.pk_log_v[s]) for s in data.subject])

    # Stage 2: turnover PD conditional on those medians.
    pd_data = warfarin_pd_data(data)
    sb_pd = SBBRMI(warfarin_pd_brmi(pd_data); mod=@__MODULE__)
    pd = sample_quorum(sb_pd, "warfarin_pd"; n_chains, n_draws, seed=seed + 1)
    d_pd = brm_descriptor(sb_pd; name=:warfarin_pd)
    names_pd = constrained_names(pd.problem)
    C_pd = [constrain_chain(pd.problem, c, seed + 10 + k) for (k, c) in enumerate(pd.chains)]
    push!(fits, ("two-stage PD", pd))
    append!(population, population_rows("two-stage PD", d_pd, names_pd, C_pd, PD_QUANTITIES))
    append!(ppc, ppc_rows("two-stage PD", d_pd, pd.problem, thin_draws(pd.chains, n_ppc), :pd_dv;
                          pd_channel..., seed))

    # The joint model: one posterior over both likelihoods.
    sb_joint = SBBRMI(warfarin_joint_brmi(data); mod=@__MODULE__)
    joint = sample_quorum(sb_joint, "warfarin_joint"; n_chains, n_draws, seed=seed + 2)
    d_joint = brm_descriptor(sb_joint; name=:warfarin_joint)
    names_joint = constrained_names(joint.problem)
    C_joint = [constrain_chain(joint.problem, c, seed + 20 + k) for (k, c) in enumerate(joint.chains)]
    push!(fits, ("joint", joint))
    append!(population, population_rows("joint", d_joint, names_joint, C_joint, vcat(PK_QUANTITIES, PD_QUANTITIES)))
    U_joint = thin_draws(joint.chains, n_ppc)
    append!(ppc, ppc_rows("joint", d_joint, joint.problem, U_joint, :pk_dv; pk_channel..., seed))
    append!(ppc, ppc_rows("joint", d_joint, joint.problem, U_joint, :pd_dv; pd_channel..., seed))

    write_tsv(joinpath(RESULTS, "fits.tsv"), FITS_HEADER, [fit_row(label, f) for (label, f) in fits])
    write_tsv(joinpath(RESULTS, "population.tsv"), POPULATION_HEADER, population)
    write_tsv(joinpath(RESULTS, "ppc.tsv"), PPC_HEADER, ppc)
    report_fits(fits)
    return (; fits, population, ppc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_all(; n_draws = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1000,
              n_chains = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4)
end
