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
#
# One directory of tables per fit (results/pk, results/pd, results/joint), each
# with its own receipt: the shape DocsFits' summary will have.

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

"""
Fit one model into `results/<tag>/`: the receipt first (it names the commit and
settings these results are the results of, and a defect in it fails before
sampling), then the sampler report, the population table and the posterior
predictive quantiles of each channel in `channels` (`response => channel`).
Returns the fit, the descriptor and the constrained chains for anything the
caller derives from them.
"""
function fit_into(tag, sb, quantities, channels; name, model_source, seed, n_draws, n_chains, n_ppc)
    dir = joinpath(RESULTS, tag)
    isdir(dir) || mkpath(dir)
    write_receipt(joinpath(dir, "receipt.tsv"),
                  ["model" => model_source,
                   "data" => "research/warfarin/data/warfarin_data.txt (32 subjects) through data.jl",
                   "seed" => "$seed; predictive draws on $n_ppc thinned draws"];
                  n_chains, n_draws)
    f = sample_quorum(sb, "warfarin_$tag"; n_chains, n_draws, seed)
    d = brm_descriptor(sb; name)
    names = constrained_names(f.problem)
    C = [constrain_chain(f.problem, c, seed + k) for (k, c) in enumerate(f.chains)]
    U = thin_draws(f.chains, n_ppc)
    write_tsv(joinpath(dir, "fits.tsv"), FITS_HEADER, [fit_row(f)])
    write_tsv(joinpath(dir, "population.tsv"), POPULATION_HEADER, population_rows(d, names, C, quantities))
    write_tsv(joinpath(dir, "ppc.tsv"), PPC_HEADER,
              reduce(vcat, [ppc_rows(d, f.problem, U, response; ch..., seed) for (response, ch) in channels]))
    report_fits([(tag, f)])
    return (; fit=f, descriptor=d, names, chains=C)
end

function fit_all(; n_draws=1000, n_chains=4, seed=20261110, n_ppc=400)
    data = warfarin_data()
    pk_channel = channel(data, :pk_time, :pk_dv)
    pd_channel = channel(data, :pd_time, :pd_dv)

    # Stage 1: population PK.
    pk = fit_into("pk", SBBRMI(warfarin_pk_brmi(data); mod=@__MODULE__), PK_QUANTITIES,
                  [:pk_dv => pk_channel];
                  name=:warfarin_pk, model_source="research/warfarin/reproduce.jl (warfarin_pk_brmi)",
                  seed, n_draws, n_chains, n_ppc)

    # Each subject's PK medians, the second stage's fixed inputs. The linear
    # predictors have one value per subject; log_cl0 and log_v0 carry the
    # weight effect, as the PD model expects.
    C_all = reduce(vcat, pk.chains)
    per_subject(logical) = brm_output_draws(pk.descriptor, C_all, pk.names; logical, role=:linear_predictor)
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
    pd = fit_into("pd", SBBRMI(warfarin_pd_brmi(pd_data); mod=@__MODULE__), PD_QUANTITIES,
                  [:pd_dv => pd_channel];
                  name=:warfarin_pd, model_source="research/warfarin/reproduce.jl (warfarin_pd_brmi) on results/pk_subject_medians.tsv",
                  seed=seed + 1, n_draws, n_chains, n_ppc)

    # The joint model: one posterior over both likelihoods.
    joint = fit_into("joint", SBBRMI(warfarin_joint_brmi(data); mod=@__MODULE__),
                     vcat(PK_QUANTITIES, PD_QUANTITIES), [:pk_dv => pk_channel, :pd_dv => pd_channel];
                     name=:warfarin_joint, model_source="research/warfarin/joint_reproduce.jl (warfarin_joint_brmi)",
                     seed=seed + 2, n_draws, n_chains, n_ppc)
    return (; pk, pd, joint)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_all(; n_draws = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1000,
              n_chains = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4)
end
