# Whole-value captures must order cells after their model dependencies.
# Run: julia --project=test test/plate_capture_dependencies.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const capture_data = (; weights=[0.5, 1.2], y=[0.1, -0.2])
const capture_field = @brm begin
    field = exp(level)
end
const capture_reader = @brm begin
    @plate for i in eachindex(weights)
        local_sum = sum(field)
        mu[i] = local_sum * weights[i] + shift
    end
    y ~ Normal(mu, 0.7)
    # The cell's local_sum shadows this model value. Treating it as a capture
    # would introduce a false mu -> local_sum -> mu cycle.
    local_sum = sum(mu)
end
const capture_priors = @brm begin
    level[1:3] .~ Normal.(location, 0.8)
    location ~ Normal(0, 1)
    shift ~ Normal(0.2, 0.5)
end

@stestset "whole-value plate captures order composed declarations" begin
    # A late hyperparameter blocks field while the plate's row data are ready.
    # A source-order-only example would hide the missing dependency.
    models = (
        merge(capture_field, capture_reader, capture_priors)(capture_data),
        merge(capture_reader, capture_priors, capture_field)(capture_data),
        merge(capture_field(capture_data), capture_reader(capture_data),
              capture_priors(capture_data)),
    )
    before = deepcopy(capture_data)
    for (index, model) in enumerate(models)
        program = BRM._brm_prepare_program(model)
        dependencies = only(op.dependencies for op in program.operations if op.name === :mu)
        @test :field in dependencies
        @test :shift in dependencies
        @test !(:local_sum in dependencies)
        positions = Dict(name => i for (i, name) in pairs(program.order))
        @test positions[:location] < positions[:level] < positions[:field] < positions[:mu]
        @test positions[:shift] < positions[:mu] < positions[:local_sum]
        # First variant includes complete emitted RK source replay. All variants
        # execute both backends against the same independent normalized density.
        rk = index == 1 ? check_rk_source_roundtrip(RKBRMI(model)) : RKBRMI(model)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        stan = consumer_stan(model, "plate-captures-$index")
        names = coordinate_names(rk.model.layout)
        li = [coordinate(names, Symbol("level.", j)) for j in 1:3]
        hi = coordinate(names, :location)
        si = coordinate(names, :shift)
        oracle(u) = logpdf(Normal(0, 1), u[hi]) +
            sum(logpdf.(Normal(u[hi], 0.8), u[li])) +
            logpdf(Normal(0.2, 0.5), u[si]) +
            sum(logpdf.(Normal.(sum(exp.(u[li])) .* capture_data.weights .+ u[si], 0.7),
                        capture_data.y))
        for u in (zeros(5), collect(range(-0.3, 0.4; length=5)))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [name => string(name) for name in names], rk, u)
        end
    end
    @test isequal(capture_data, before)
end

@stestset "cell formals and scan state remain local dependency names" begin
    lambda = :(x -> begin
        local_value = exp(shared)
        @scan begin
            state[1] = local_value
            for step in 2:length(x)
                state[step] = state[step - 1] + x[step] + drift
            end
        end
        state
    end)
    body = BRM._BRMInlineBody(lambda, @__MODULE__)
    refs = BRM._brm_operation_references!(Set([:existing]), body)
    @test refs == Set((:existing, :shared, :drift))
    @test isequal(body.expression, lambda)
end
