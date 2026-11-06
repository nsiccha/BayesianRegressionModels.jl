# test/stanblocks_reexports.jl — `using StanBlocks` beside `using
# BayesianRegressionModels`: every name both export is one binding, so
# `stan_code`, `stan_model` and `stan_instantiate` stay usable unqualified,
# and BRM's methods for its own types extend StanBlocks' functions.
#
# Run: julia --startup-file=no --project=test test/stanblocks_reexports.jl

using Test
using StanBlocks
using BayesianRegressionModels
import LogDensityProblems
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

@testset "a GenerativePlan takes the entries its SBBRMI takes" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    data = (; x=[0.0, 1.0, 2.0], y=[0.5, 1.0, 1.5])
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    code, inputs = stan_code(sb), stan_data(sb)

    built = generative_plan(builder, data; mod=@__MODULE__)
    for plan in (generative_plan(sb), built)
        @test stan_code(plan) == code
        @test isequal(stan_data(plan), inputs)
        # The results agree with reading the retained model directly.
        @test isequal(stan_data(plan), stan_data(plan.model))
        traced = stan_model(plan)
        @test traced isa StanBlocks.StanModel
        @test stan_code(traced) == code
        # A consumer's `stan_data(stan_model(plan))` reaches BRM's `stan_data`.
        @test isequal(stan_data(traced), inputs)
        @test isequal(stan_data(stan_model(sb)), inputs)
        # Keywords reach StanBlocks' method, as on an SBBRMI.
        @test stan_code(stan_model(plan; info=StanBlocks.StanModel())) == code
    end

    # A plan rebuilt on other rows reads those rows, not its source plan's.
    more = (; x=[0.0, 1.0, 2.0, 3.0], y=[0.5, 1.0, 1.5, 2.0])
    rebuilt = generative_plan(built, more)
    @test isequal(stan_data(rebuilt), stan_data(SBBRMI(builder(more); mod=@__MODULE__)))
    @test !isequal(stan_data(rebuilt), inputs)

    # Compiled: the same density as the SBBRMI, and `path` reaches StanBlocks.
    dir = mktempdir()
    path = joinpath(dir, "plan.stan")
    problem = stan_instantiate(built; path)
    @test isfile(path)
    reference = stan_instantiate(sb; path=joinpath(dir, "sb.stan"))
    n = LogDensityProblems.dimension(problem)
    @test n == LogDensityProblems.dimension(reference)
    at = fill(0.1, n)
    @test LogDensityProblems.logdensity(problem, at) ==
          LogDensityProblems.logdensity(reference, at)
end

# A caller compiled before the helper its model reads exists: the entries still
# trace the model. The frame's own failed direct call is the positive control.
const LATE_BUILDER = @brm begin
    sigma ~ Exponential(1)
    b ~ Normal(0, 1)
    shifted = late_helper(x, b)
    mu ~ 1 + shifted
    y ~ Normal(mu, sigma)
end

function enter_plan_in_one_frame(data)
    Core.eval(@__MODULE__,
        :(StanBlocks.@deffun late_helper(v::vector[n], b::real)::vector[n] = v .+ b))
    older_world = try
        late_helper([1.0], 1.0)
        false
    catch e
        e isa MethodError
    end
    plan = generative_plan(LATE_BUILDER, data; mod=@__MODULE__)
    (; older_world, code=stan_code(plan), traced=stan_code(stan_model(plan)),
       inputs=stan_data(plan), direct=stan_data(plan.model))
end

@testset "a plan's entries work in a caller older than the model's helper" begin
    r = enter_plan_in_one_frame((; x=[0.0, 1.0, 2.0], y=[0.5, 1.0, 1.5]))
    @test r.older_world
    @test occursin("late_helper", r.code)
    @test r.traced == r.code
    @test isequal(r.inputs, r.direct)
end
