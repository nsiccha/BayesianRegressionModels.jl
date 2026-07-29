# test/formula_offset.jl — fixed-coefficient formula offsets in sbimpl.
#
# Run: julia --project=. test/formula_offset.jl

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Poisson

stanc_ok(model) = StanBlocks.stanc_check(
    StanBlocks.stan_code(model); warn_pedantic=false).ok

const OFFSET_DF = (;
    subject=[1, 2, 1, 2],
    exposure=[1.0, 2.0, 3.0, 4.0],
    y=[0.1, 0.2, 0.3, 0.4],
    count=[0, 1, 2, 1],
)

const SAMPLED_OFFSET_BUILDER = @brm begin
    log_ka_pop ~ Normal(-2.0, 1.0)
    eta_ka ~ offset(log_ka_pop) + (1 | p | subject)
    y ~ Normal(eta_ka, 1.0)
end

const DATA_OFFSET_BUILDER = @brm begin
    log_rate ~ 1 + offset(log(exposure))
    count ~ Poisson(exp(log_rate))
end

const OFFSET_CACHE = joinpath(tempdir(), "brm-formula-offset")

@testset "sampled scalar offset is a direct fixed-one summand" begin
    brmi = SAMPLED_OFFSET_BUILDER(OFFSET_DF)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)

    @test popcoefnames(brmi, :eta_ka) == Symbol[]
    @test :log_ka_pop in dependencies(brmi, :eta_ka).intermediates
    @test !occursin("X_eta_ka", code)
    @test !occursin("pop_eta_ka", code)
    @test occursin("eta_ka = (log_ka_pop +", code)
    @test stanc_ok(sb.model)
end

@testset "data offset remains direct while ordinary terms keep betas" begin
    brmi = DATA_OFFSET_BUILDER(OFFSET_DF)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)

    @test popcoefnames(brmi, :log_rate) == [:Intercept]
    @test occursin("log(exposure)", code)
    @test stanc_ok(sb.model)

    new_df = merge(OFFSET_DF, (; exposure=[2.0, 4.0, 6.0, 8.0]))
    replay = reprocess(sb, new_df)
    @test replay.data[:exposure] == new_df.exposure
end

@testset "protect is not a sampled-value offset" begin
    brmi = @brm OFFSET_DF begin
        log_ka_pop ~ Normal(-2.0, 1.0)
        eta_ka ~ protect(log_ka_pop) + (1 | p | subject)
        y ~ Normal(eta_ka, 1.0)
    end
    @test_throws "cannot materialize NamedColumn `log_ka_pop`" SBBRMI(brmi)
end

@testset "sampled scalar offset has a finite BridgeStan density and gradient" begin
    sb = SBBRMI(SAMPLED_OFFSET_BUILDER(OFFSET_DF); mod=@__MODULE__)
    isdir(OFFSET_CACHE) || mkpath(OFFSET_CACHE)
    path = joinpath(OFFSET_CACHE, string(hash(BayesianRegressionModels.stan_code(sb))) * ".stan")
    problem = StanBlocks.stan_instantiate(sb.model; path)
    dimension = LogDensityProblems.dimension(problem)
    q = [0.1 * ((i % 5) - 2) for i in 1:dimension]
    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)

    @test dimension > 0
    @test isfinite(lp)
    @test length(gradient) == dimension
    @test all(isfinite, gradient)
end
