# brm_smc_runs / brm_smc_compare: agreement between independent completed bridges.
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Statistics
include("smc_tuning_fixture.jl")
using .SMCTuningFixture

normal_model(d) = @brm d begin
    mu ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

@testset "brm_smc_runs and brm_smc_compare on an analytic Normal posterior" begin
    data = (; y=[0.9, 1.6, 0.2, 1.3, -0.4, 0.7])
    n = length(data.y)
    posterior_mean = sum(data.y) / (n + 1)
    posterior_sd = sqrt(1 / (n + 1))
    exact_logz = -n * log(2pi) / 2 - log(n + 1) / 2 - (sum(abs2, data.y) - sum(data.y)^2 / (n + 1)) / 2
    r = brm_smc_runs(normal_model(data); seeds=(11, 12, 13), particles=512)
    @test length(r.fits) == 3
    c = r.comparison
    @test length(c.names) == 1
    for run in c.runs
        @test run.means[1] ≈ posterior_mean atol=0.12
        @test run.sds[1] ≈ posterior_sd rtol=0.25
        @test run.lineages >= 1
        @test run.stages >= 1
        @test run.history_stages == run.stages
        @test run.evaluations > 0
    end
    @test c.pooled_mean[1] ≈ posterior_mean atol=0.1
    @test c.pooled_sd[1] ≈ posterior_sd rtol=0.2
    @test c.mean_difference[1] < 0.6
    @test 0.75 <= c.sd_ratio_min[1] <= 1 <= c.sd_ratio_max[1] <= 1.3
    @test c.worst.coordinate == c.names[1]
    @test c.worst.mean_difference == c.mean_difference[1]
    @test length(c.log_normalizer.values) == 3
    @test c.log_normalizer.mean ≈ exact_logz atol=0.25
    @test c.log_normalizer.sd < 0.3
    @test brm_smc_compare(r.fits...) == c
    again = brm_smc_runs(normal_model(data); seeds=(11, 12, 13), particles=512)
    @test again.comparison == c
    @test_throws ArgumentError brm_smc_compare(r.fits[1])
    @test_throws ArgumentError brm_smc_runs(normal_model(data); seeds=(11,), particles=64)
    @test_throws ArgumentError brm_smc_runs(normal_model(data); seeds=(Xoshiro(1), Xoshiro(2)), particles=64)
    other = brm_smc_fit(normal_model((; y=data.y .+ 1)); rng=Xoshiro(14), particles=512)
    @test_throws ArgumentError brm_smc_compare(r.fits[1], other)
    partial = brm_smc_fit(normal_model(data); rng=Xoshiro(15), particles=512, stages=0)
    @test_throws ArgumentError brm_smc_compare(r.fits[1], partial)
end

@testset "independent tuned bridges agree with the correlated Gaussian oracle" begin
    data = SMCTuningFixture.data()
    rk = RKBRMI(SMCTuningFixture.model(data))
    p = SMCTuningFixture.posterior(data)
    variances = [p.c, p.a] ./ p.determinant
    schedule = brm_smc_tune(rk; rng=Xoshiro(737), init=zeros(2),
        temperatures=(0.0, 1e-4, 0.001, 0.01, 0.1, 1.0), pilot_steps=128,
        options=SMCOptions(; moves=3, stepsize=0.15, max_tree_depth=7))
    r = brm_smc_runs(rk; seeds=(738, 739), particles=768, options=SMCOptions(; tuning=schedule))
    c = r.comparison
    idx = indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")], c.names)
    @test all(!isnothing, idx)
    for (k, i) in enumerate(Int.(idx))
        @test c.pooled_mean[i] ≈ p.mean[k] atol=0.4 * sqrt(variances[k])
        @test c.pooled_sd[i] ≈ sqrt(variances[k]) rtol=0.35
        @test c.mean_difference[i] < 0.6
        @test c.sd_ratio_min[i] >= 0.7
        @test c.sd_ratio_max[i] <= 1.4
    end
    @test c.log_normalizer.mean ≈ p.log_normalizer atol=0.5
    @test all(run.evaluations > 0 for run in c.runs)
    @test all(run.stages >= 1 for run in c.runs)
end
