# Run in a fresh process: julia --project=test test/laplace_start_imports.jl
using BayesianRegressionModels, BridgeStan, Random, Test, WarmupHMC

const BRM = BayesianRegressionModels
const group_ids = repeat(string.(1:16); inner=12)
const observations = repeat(range(-3, 3; length=16); inner=12) .+
    0.3 .* randn(Xoshiro(42), length(group_ids))
const brmi = @brm (; g=group_ids, y=observations) begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, 10)
    y ~ Normal(mu, sigma)
end
const sb = SBBRMI(brmi; mod=@__MODULE__)
const model = stan_instantiate(sb; path=joinpath(mktempdir(), "groups.stan")).model

@testset "Stan starts before importing the RK backend" begin
    @test isnothing(Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt))
    @test !isnothing(Base.get_extension(BRM, :BayesianRegressionModelsPathfinderExt))
    @test isfinite(BridgeStan.log_density(model, brm_start(sb, model, (;))))
    instructions = ["brm_start(sb, stan_model)", "import ReactiveKernels, ReactiveKernelsPPL",
        "import Enzyme, DifferentiationInterface", "import KLU,", "SparseMatrixColorings"]
    @test_throws instructions brm_start(sb, model; chains=4)
end

import ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
include("fitting_packages.jl")

@testset "Stan starts after importing the packages in the fitting guide" begin
    starts = brm_start(sb, model; chains=4, rng=Xoshiro(1))
    @test length(starts) == 4
    @test allunique(starts)
    @test all(s -> length(s) == BridgeStan.param_unc_num(model), starts)
    @test all(s -> isfinite(BridgeStan.log_density(model, s)), starts)
end
