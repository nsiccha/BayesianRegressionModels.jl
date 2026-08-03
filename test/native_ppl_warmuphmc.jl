using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, LKJCholesky, Normal
using Enzyme
using LogDensityProblems
using Random: Xoshiro
using Statistics: std
using WarmupHMC

include("dependency_floors.jl")

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL

NP.@model function factor_hierarchy_for_warmup()
    population ~ Normal()
    population_scale ~ Exponential(1.0)
    individual_1 ~ Normal(population, population_scale)
    individual_2 ~ Normal(population, population_scale)
    individual_3 ~ Normal(population, population_scale)
    individual_4 ~ Normal(population, population_scale)
    individual_5 ~ Normal(population, population_scale)
    observation_scale ~ Exponential(2.0)
    @. y ~ Normal(individual_1, observation_scale)
    return y
end

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

@testset "transformed grouped factor DAG samples end-to-end with WarmupHMC" begin
    groups = 6
    replicates = 4
    group = repeat(collect(1:groups), outer=replicates)
    x = collect(range(-1.8, 2.2; length=groups * replicates))
    fitted = BRM._native_ppl_fit_zscale(x, :x)
    scaled = (x .- fitted.mean) ./ fitted.scale
    group_intercept = collect(range(-0.3, 0.3; length=groups))
    group_slope = [0.12 * sin(0.7 * index) for index in 1:groups]
    y = [
        0.45 * scaled[row] + group_intercept[group[row]] +
            group_slope[group[row]] * scaled[row] +
            0.12 * sin(0.53 * row)
        for row in eachindex(x)
    ]
    brmi = @brm (; x, group, y) begin
        sigma ~ Exponential(2)
        mu ~ 0 + zscale(x) + (1 + zscale(x) | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, sigma)
    end
    prepared = NP.prepare(NP.compile(brmi))
    problem = NP.LogDensityProblem(prepared, DI.AutoEnzyme())

    dimension = LogDensityProblems.dimension(problem)
    @test dimension == 17
    @test prepared.plan.fitted_nodes.zscale_x_for_mu.mean == fitted.mean
    @test prepared.plan.fitted_nodes.zscale_x_for_mu.scale == fitted.scale
    density, gradient = LogDensityProblems.logdensity_and_gradient(
        problem, zeros(dimension))
    @test isfinite(density)
    @test all(isfinite, gradient)

    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260805),
        problem;
        n_draws=50,
        n_evaluations=220,
        stepsize_adaptation_limit=25,
        max_tree_depth=8,
        progress=nothing,
        monitor_ess=false,
        nonlinear_adapt=false,
    )

    draws = result.posterior_position
    @test size(draws, 1) == dimension
    @test size(draws, 2) >= 50
    @test all(isfinite, draws)
    @test all(>(1e-6), vec(std(draws; dims=2)))
    @test result.n_divergent_samples == 0
end

@testset "weighted grouped factor DAG samples end-to-end with WarmupHMC" begin
    groups = 6
    replicates_per_group = 5
    group = repeat(collect(1:groups), inner=replicates_per_group)
    x = collect(range(-1.6, 1.8; length=groups * replicates_per_group))
    replicates = [1 + mod(3 * row, 4) for row in eachindex(x)]
    group_intercept = collect(range(-0.28, 0.28; length=groups))
    mu = [
        0.22 - 0.48 * x[row] + group_intercept[group[row]]
        for row in eachindex(x)
    ]
    y = [
        mu[row] + 0.42 / sqrt(replicates[row]) * sin(0.61 * row)
        for row in eachindex(x)
    ]
    brmi = @brm (; x, group, replicates, y) begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | g | group)
        sd(:, g) ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), aweights(replicates))
    end
    prepared = NP.prepare(NP.compile(brmi))
    problem = NP.LogDensityProblem(prepared, DI.AutoEnzyme())

    dimension = LogDensityProblems.dimension(problem)
    @test dimension == groups + 4
    @test NP.observation_weight_kind(
        prepared.plan.graph.sites.y.factor.weight) === :analytic
    density, gradient = LogDensityProblems.logdensity_and_gradient(
        problem, zeros(dimension))
    @test isfinite(density)
    @test all(isfinite, gradient)

    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260806),
        problem;
        n_draws=50,
        n_evaluations=220,
        stepsize_adaptation_limit=25,
        max_tree_depth=8,
        progress=nothing,
        monitor_ess=false,
        nonlinear_adapt=false,
    )

    draws = result.posterior_position
    @test size(draws, 1) == dimension
    @test size(draws, 2) >= 50
    @test all(isfinite, draws)
    @test all(>(1e-6), vec(std(draws; dims=2)))
    @test result.n_divergent_samples == 0
end

@testset "factor DAG samples end-to-end with WarmupHMC" begin
    response = [0.28, 0.35, 0.31, 0.41, 0.26, 0.37]
    prepared = NP.prepare(NP.compile(NP.condition(
        factor_hierarchy_for_warmup();
        individual_1=0.28,
        individual_2=0.35,
        individual_3=0.31,
        individual_4=0.41,
        individual_5=0.26,
        y=response)))
    problem = NP.LogDensityProblem(prepared, DI.AutoEnzyme())

    dimension = LogDensityProblems.dimension(problem)
    @test dimension == 3
    density, gradient = LogDensityProblems.logdensity_and_gradient(
        problem, zeros(dimension))
    @test isfinite(density)
    @test all(isfinite, gradient)

    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260803),
        problem;
        n_draws=50,
        n_evaluations=150,
        stepsize_adaptation_limit=20,
        max_tree_depth=8,
        progress=nothing,
        monitor_ess=false,
        nonlinear_adapt=false,
    )

    draws = result.posterior_position
    @test size(draws, 1) == dimension
    @test size(draws, 2) >= 50
    @test all(isfinite, draws)
    @test all(>(1e-6), vec(std(draws; dims=2)))
    @test result.n_divergent_samples == 0
end

@testset "distributional factor DAG samples end-to-end with WarmupHMC" begin
    x = collect(range(-1.4, 1.4; length=20))
    z = [sin(0.37 * row) for row in eachindex(x)]
    residual = [0.18 * sin(0.71 * row) for row in eachindex(x)]
    mu = @. 0.35 + 0.8 * x
    sigma = @. exp(-0.25 + 0.3 * z)
    data = (; x, z, y=mu .+ sigma .* residual)
    brmi = @brm data begin
        mu ~ 1 + x
        log_sigma ~ 1 + z
        y ~ Normal(mu, exp(log_sigma))
    end
    prepared = NP.prepare(NP.compile(brmi))
    problem = NP.LogDensityProblem(prepared, DI.AutoEnzyme())

    dimension = LogDensityProblems.dimension(problem)
    @test dimension == 4
    density, gradient = LogDensityProblems.logdensity_and_gradient(
        problem, zeros(dimension))
    @test isfinite(density)
    @test all(isfinite, gradient)

    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20260804),
        problem;
        n_draws=50,
        n_evaluations=180,
        stepsize_adaptation_limit=20,
        max_tree_depth=8,
        progress=nothing,
        monitor_ess=false,
        nonlinear_adapt=false,
    )

    draws = result.posterior_position
    @test size(draws, 1) == dimension
    @test size(draws, 2) >= 50
    @test all(isfinite, draws)
    @test all(>(1e-6), vec(std(draws; dims=2)))
    @test result.n_divergent_samples == 0
end
