# test/stanblocks_reexports.jl — `using StanBlocks` beside `using
# BayesianRegressionModels`: every name both export is one binding, so
# `stan_code`, `stan_model` and `stan_instantiate` stay usable unqualified,
# and BRM's methods for its own types extend StanBlocks' functions.
#
# Run: julia --startup-file=no --project=test test/stanblocks_reexports.jl

using Test
using StanBlocks
using BayesianRegressionModels
const BRM = BayesianRegressionModels

@testset "names both packages export are one binding" begin
    shared = intersect(names(StanBlocks), names(BRM))
    @test issubset((:stan_code, :stan_model, :stan_instantiate), shared)
    for name in shared
        @test getfield(BRM, name) === getfield(StanBlocks, name)
    end
end

@testset "BRM's methods extend StanBlocks' functions" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder((; y=[0.5, 1.0, 1.5])); mod=@__MODULE__)
    code = stan_code(sb)
    @test code == stan_code(sb.model)
    @test stan_model(sb) isa StanBlocks.StanModel
    @test stan_code(generative_plan(sb)) == code
    # A caller compiled before the model is built still traces it.
    build_and_trace(data) = stan_code(SBBRMI(builder(data); mod=@__MODULE__))
    @test build_and_trace((; y=[0.5, 1.0])) == stan_code(SBBRMI(builder((; y=[0.5, 1.0]));
                                                               mod=@__MODULE__))
end
