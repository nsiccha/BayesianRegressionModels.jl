using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random

scaled_tuning_model(d) = @brm d begin
    mu ~ 1 + zscale(x)
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

@testset "SMC update tuning preserves fitted preprocessing" begin
    training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
    cumulative = (; x=[-2.0, 0.0, 2.0, 10.0], y=[-0.3, 0.4, 0.9, 1.1])
    rk = RKBRMI(scaled_tuning_model(training))
    options = SMCOptions(; moves=1, stepsize=0.2, max_tree_depth=4)
    fit = brm_smc_fit(rk; rng=Xoshiro(751), particles=64, options)
    schedule = brm_smc_tune(fit, cumulative; rng=Xoshiro(752), init=zeros(2),
        temperatures=(0.0, 0.01, 0.1, 1.0), pilot_steps=16, options)
    @test smc_reference(schedule) === nothing
    update = brm_smc_update(fit, cumulative;
        options=SMCOptions(; tuning=schedule), stages=0)
    @test update.state.temperature == 0
    expected = reprocess(fit.template, cumulative; freeze_constants=true)
    fresh = RKBRMI(scaled_tuning_model(cumulative))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    density(model, u) = LogDensityProblems.logdensity(
        rk_logdensity_problem(model; ad_backend=backend), u)
    u = [0.2, 0.4]
    @test density(update.backend, u) ≈ density(expected, u)
    @test !isapprox(density(update.backend, u), density(fresh, u); atol=1e-6)
    @test_throws ArgumentError brm_smc_fit(fresh; rng=Xoshiro(753),
        options=SMCOptions(; tuning=schedule), particles=10, stages=0)
end
