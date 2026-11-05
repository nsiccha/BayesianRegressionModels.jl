using Test, BayesianRegressionModels, HTMXObjects

const BRM = BayesianRegressionModels

module ModelViewEdgeCases
using BayesianRegressionModels

const calls = Ref(0)
"""A location hidden inside a mixture's component list."""
function mixture_location(x)
    calls[] += 1
    x
end
default_scale() = error("inspection evaluated a component default")

joint = @brm begin
    [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
end
mixture = @brm begin
    y ~ MixtureModel([Normal(mixture_location(location), scale), Normal(other, 1)], [0.5, 0.5])
end
transformed = @brm begin
    log(y) ~ Normal(location, 1)
end
parameter_only = @brm begin
    alpha ~ Normal(0, 1)
end
computed_only = @brm begin
    alpha ~ Normal(0, 1)
    doubled = 2 * alpha
end

@brm function with_defaults(x, shift=2; scale=default_scale())
    beta ~ Normal(0, scale)
    return x + beta + shift
end
end

@testset "observation roles follow the authored response columns" begin
    data = (; y1=[1., 2.], y2=[3., 4.], mu1=[0., 0.], mu2=[0., 0.], L=[1. 0.; 0. 1.])
    model = ModelViewEdgeCases.joint(data)
    facts = BRM._brm_view_facts(model)
    roles = Dict(row.name => row.role for row in facts.inputs)
    @test roles[:y1] === :response_data
    @test roles[:y2] === :response_data
    @test all(roles[name] === :data for name in (:mu1, :mu2, :L))
    @test isempty(facts.quantities)
    @test data.y1 == [1., 2.]
    transformed = ModelViewEdgeCases.transformed((; y=[1., 2.], location=0.))
    facts = BRM._brm_view_facts(transformed)
    @test only(row for row in facts.inputs if row.name === :y).role === :response_data
    @test isempty(facts.quantities)
end

@testset "nested formula containers retain inputs and callable documentation" begin
    model = ModelViewEdgeCases.mixture((; y=[1., 2.], location=0., scale=1., other=2.))
    facts = BRM._brm_view_facts(model)
    @test Set(row.name for row in facts.inputs) == Set((:y, :location, :scale, :other))
    @test all(row.rank == 0 for row in facts.inputs if row.name !== :y)
    for mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, brm_model_view(model))
        @test occursin("A location hidden inside a mixture", rendered)
    end
    @test ModelViewEdgeCases.calls[] == 0
end

@testset "component inputs distinguish defaults without evaluating them" begin
    view = brm_model_view(ModelViewEdgeCases.with_defaults)
    for mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, view)
        @test occursin("Component inputs", rendered)
        @test occursin("Required", rendered)
        @test occursin("Optional", rendered)
        @test occursin("default_scale()", rendered)
        @test occursin("Default expressions are shown without evaluation", rendered)
    end
end

@testset "empty input inventories explain the absence" begin
    builder = ModelViewEdgeCases.computed_only
    for mime in ("text/html", "text/markdown", "text/plain")
        @test occursin("No potential data inputs", repr(mime, brm_model_view(builder)))
        @test occursin("No bound data inputs", repr(mime, brm_model_view(builder((;)))))
        # A terminal sampled name may be bound as data or left latent.
        unbound = repr(mime, brm_model_view(ModelViewEdgeCases.parameter_only))
        @test occursin("Potential data inputs", unbound)
        @test !occursin("Required input", unbound)
        @test occursin("No bound data inputs",
            repr(mime, brm_model_view(ModelViewEdgeCases.parameter_only((;)))))
    end
end
