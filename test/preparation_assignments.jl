using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

likelihood_query(backend) = prepare_query(backend.model, BRM.rk_bound_plan(backend), :likelihood)

@testset "assignment axes follow dependency order" begin
    data = (; x=[-0.8, 0.3, 1.2], y=[0.1, -0.2, 0.7])
    model = (@brm begin
        shift ~ Normal(0, 1)
        first_value = x + shift
        second_value = 2 * first_value
        y ~ Normal(second_value, 1)
    end)(data)
    # Fragment composition can retain declarations in a different order from
    # their dependencies. Keep the resolved references, reverse the storage.
    names = reverse(keys(model.operations))
    reordered = BRMI(NamedTuple{names}(
        Tuple(getproperty(model.operations, name) for name in names)))
    prepared = BRM._brm_prepare_model(reordered)
    assignments = Dict(value.name => value for value in prepared.assignments)
    @test findfirst(==(:first_value), prepared.order) <
          findfirst(==(:second_value), prepared.order)
    first_reference = only(filter(value -> value isa BRM._BRMPreparedRef,
        assignments[:second_value].expression.args))
    @test first_reference.axis === :observation

    # The reordered model still evaluates in dependency order.
    rk = RKBRMI(reordered)
    shift = 0.25
    mean = 2 .* (data.x .+ shift)
    @test Base.invokelatest(likelihood_query(rk), [shift]) ≈
          sum(logpdf.(Normal.(mean, 1), data.y))
end

@testset "response assignments keep their own row axes through replay" begin
    data = (; x=[-0.5, 0.8], z=[-1.0, 0.2, 0.7, 1.4],
        y_a=[0.1, -0.2], y_b=[0.4, -0.3, 0.6, 1.1])
    builder = @brm begin
        shift ~ Normal(0, 1)
        mean_a = x + shift
        mean_b = 2 * z - shift
        y_a ~ Normal(mean_a, 1)
        y_b ~ Normal(mean_b, 1)
    end
    backend = RKBRMI(builder(data))
    shift = 0.3
    held_out = (; x=[-0.9, 0.1, 1.0], z=[0.6],
        y_a=[0.2, 0.4, -0.1], y_b=[0.5])
    for (model, rows) in ((backend, data), (reprocess(backend, held_out), held_out))
        expected_a = logpdf.(Normal.(rows.x .+ shift, 1), rows.y_a)
        expected_b = logpdf.(Normal.(2 .* rows.z .- shift, 1), rows.y_b)
        @test model.model.layout.total == 1
        @test Base.invokelatest(likelihood_query(model), [shift]) ≈
              sum(expected_a) + sum(expected_b)
    end
end
