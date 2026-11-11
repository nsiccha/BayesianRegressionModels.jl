# Deliberate long fit for the synthetic docs example. Never run during a docs build.
using Test, BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools, SHA, Serialization
using DifferentiationInterface: AutoEnzyme
const repo = normpath(joinpath(@__DIR__, "..", ".."))
length(ARGS) == 1 || error("usage: julia --project=test research/seal/fit.jl NEW_CHECKPOINT_DIRECTORY")
const checkpoints = abspath(only(ARGS))
ispath(checkpoints) && error("use a new checkpoint directory; existing runs are preserved")
const source = joinpath(@__DIR__, "grey_seal_brm.jl")
mkpath(checkpoints)
write(joinpath(checkpoints,"source-hashes.txt"), bytes2hex(sha256(read(source))) * "  grey_seal_brm.jl\n")
write(joinpath(checkpoints,"source-commit.txt"), readchomp(Cmd(["git","rev-parse","HEAD"]; dir=repo)) * "\n")
include(source)
include(joinpath(repo, "research", "mcmc_harness.jl"))
sb = SBBRMI(grey_seal_brm_model())
write(joinpath(checkpoints,"program-sha256.txt"), bytes2hex(sha256(stan_code(sb))) * "\n")
write(joinpath(checkpoints,"data-sha256.txt"), bytes2hex(sha256(repr(grey_seal_brm_fixture()))) * "\n")
inits = (; phi_a_sc=0.88, phi_sc=0.85, survival_shape=0.5,
    male_pup_survival_offset=0.0, male_adult_survival_offset=0.0,
    carrying_capacity=60000.0, density_dependence_scaled=0.4,
    max_baseline_birth_rate=0.9, min_baseline_birth_rate_sc=0.7,
    population_init_size=3000.0, bbr_reference_logit=0.0,
    bbr_logit=(; herring_index_1=0.0, herring_index_2=0.0),
    aerial_mu=0.3, phi_aerial=50.0, harvest_bag_cv=0.15)
fit = sample_quorum(sb, "seal-support-long-fresh"; n_chains=4, n_draws=1500,
    seed=20261111, inits, target_acceptance_rate=0.99,
    max_divergent_fraction=0.1, checkpoint_dir=checkpoints, resume=false,
    cache="brm-seal-support-validation")
serialize(joinpath(checkpoints,"model-frame-draws.jls"),
    (; fit.chains, fit.seconds, fit.n_chains, fit.divergent, fit.omitted))
@testset "long fresh seal support fit: admitted chains and finite draws" begin
    @test length(fit.chains) >= 3
    @test all(c -> size(c, 2) >= 1500, fit.chains)
    for chain in fit.chains, q in eachcol(chain)
        lp, gradient = BridgeStan.log_density_gradient(fit.problem.model, collect(q))
        @test isfinite(lp) && all(isfinite, gradient)
    end
end
diagnostics = [chain_diagnostics([collect(c[j, :]) for c in fit.chains])
    for j in axes(first(fit.chains), 1)]
finite_draws = sum(c -> size(c, 2), fit.chains)
println("SEAL_LONG_FRESH_FIT finite_draws=", finite_draws,
    " kept=", length(fit.chains), " omitted=", fit.omitted,
    " divergences=", fit.divergent,
    " divergence_percent=", 100 * fit.divergent / finite_draws,
    " max_rhat=", maximum(d -> d.rhat, diagnostics),
    " min_bulk_ess=", minimum(d -> d.ess_bulk, diagnostics),
    " min_tail_ess=", minimum(d -> d.ess_tail, diagnostics))
println("SEAL_PROGRAM_SHA256 ", bytes2hex(sha256(stan_code(sb))))
for i in 1:4
    cp = deserialize(joinpath(checkpoints, "chain_$i", "cp_latest.jls"))
    n = size(cp.posterior_position, 2)
    println("SEAL_LONG_FRESH_CHAIN chain=", i, " saved=", n,
        " divergences=", cp.n_divergent_samples,
        " divergence_percent=", 100 * cp.n_divergent_samples / n)
end
