using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, Normal
using Enzyme
using LogDensityProblems
using Random: Xoshiro
using Statistics: std
using WarmupHMC

include("dependency_floors.jl")

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL

require_git_ancestor(
    "WarmupHMC",
    pkgdir(WarmupHMC),
    WARMUPHMC_NATIVE_PPL_MINIMUM;
    reason="Native PPL targets require WarmupHMC's own-gradient Pathfinder initialization.",
)

@testset "native PPL samples end-to-end with WarmupHMC" begin
    x = collect(range(-1.5, 1.5; length=16))
    residual = [
        -0.10, 0.08, -0.04, 0.12, -0.09, 0.03, 0.06, -0.07,
        0.02, -0.05, 0.11, -0.08, 0.04, 0.09, -0.03, 0.01,
    ]
    data = (; x, y=0.4 .+ 1.2 .* x .+ residual)
    brmi = @brm data begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    plan = NP.compile(brmi)
    prepared = NP.prepare(plan)
    problem = NP.LogDensityProblem(prepared, DI.AutoEnzyme())

    dimension = LogDensityProblems.dimension(problem)
    @test dimension == 3
    @test eltype(problem) === Float64
    @test LogDensityProblems.capabilities(problem) isa
          LogDensityProblems.LogDensityOrder{1}

    position = zeros(dimension)
    density, gradient = LogDensityProblems.logdensity_and_gradient(
        problem, position)
    @test isfinite(density)
    @test length(gradient) == dimension
    @test all(isfinite, gradient)
    direct_density, direct_gradient = NP.logdensity_and_gradient!(
        NP.workspace(prepared, Float64, DI.AutoEnzyme()), prepared, position)
    @test density == direct_density
    @test gradient == direct_gradient
    retained_gradient = copy(gradient)
    LogDensityProblems.logdensity_and_gradient(problem, fill(0.1, dimension))
    @test gradient == retained_gradient

    prediction_only = NP.rebind(prepared, (; x=data.x))
    @test !NP.has_response(prediction_only)
    @test_throws ArgumentError NP.LogDensityProblem(
        prediction_only, DI.AutoEnzyme())

    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260803),
        problem;
        n_draws=100,
        n_evaluations=200,
        stepsize_adaptation_limit=25,
        max_tree_depth=8,
        progress=nothing,
        monitor_ess=false,
        nonlinear_adapt=false,
    )

    draws = result.posterior_position
    @test size(draws, 1) == dimension
    @test size(draws, 2) >= 100 # n_draws is a floor for WarmupHMC.
    @test all(isfinite, draws)
    @test all(>(1e-6), vec(std(draws; dims=2)))
    @test result.n_divergent_samples == 0
end
