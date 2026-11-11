# Sampling smoke check for the corrected grey-seal model. This short run checks
# finite densities/gradients at retained draws; it does not establish convergence
# or reproduce the paper's fit. Run separately from seal_density_dependence.jl.
using Test, BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme, Random, Statistics, MCMCDiagnosticTools
using DifferentiationInterface: AutoEnzyme

include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_brm.jl"))
include(joinpath(@__DIR__, "..", "research", "mcmc_harness.jl"))

@testset "grey-seal corrected density dependence: finite-gradient sampling" begin
    sb = SBBRMI(grey_seal_brm_model())
    inits = (; phi_a_sc=0.88, phi_sc=0.85, survival_shape=0.5,
        male_pup_survival_offset=0.0, male_adult_survival_offset=0.0,
        carrying_capacity=60000.0, density_dependence_scaled=0.4,
        max_baseline_birth_rate=0.9, min_baseline_birth_rate_sc=0.7,
        population_init_size=3000.0, bbr_reference_logit=0.0,
        bbr_logit=(; herring_index_1=0.0, herring_index_2=0.0),
        aerial_mu=0.3, phi_aerial=50.0, harvest_bag_cv=0.15)
    fit = sample_quorum(sb, "seal-density-correction"; n_chains=4, n_draws=100,
        seed=20261111, inits, target_acceptance_rate=0.95,
        cache="brm-seal-density-validation")
    @test length(fit.chains) >= 3
    @test all(c -> size(c, 2) >= 100, fit.chains)
    finite_draws = 0
    for chain in fit.chains, q in eachcol(chain)
        lp, gradient = BridgeStan.log_density_gradient(fit.problem.model, collect(q))
        @test isfinite(lp) && all(isfinite, gradient)
        finite_draws += 1
    end
    diagnostics = [chain_diagnostics([collect(c[j, :]) for c in fit.chains])
        for j in axes(first(fit.chains), 1)]
    println("SEAL_FIT finite_draws=", finite_draws,
        " kept=", length(fit.chains), " omitted=", fit.omitted,
        " divergences=", fit.divergent,
        " divergence_percent=", 100 * fit.divergent / finite_draws,
        " max_rhat=", maximum(d -> d.rhat, diagnostics),
        " min_bulk_ess=", minimum(d -> d.ess_bulk, diagnostics),
        " min_tail_ess=", minimum(d -> d.ess_tail, diagnostics))
end
