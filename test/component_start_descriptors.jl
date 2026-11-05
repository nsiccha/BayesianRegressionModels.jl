# Scoped initial values through existing public descriptor and transport APIs.
# RUN: julia --project=test test/component_start_descriptors.jl
using Test, BayesianRegressionModels, StanBlocks, ReactiveKernels, ReactiveKernelsPPL
import BridgeStan

@brm function local_field(x)
    a ~ Normal(0, 1)
    z[1:length(x)] .~ Normal.(0, 1)
    scale ~ Exponential(1)
    weights ~ Dirichlet([1.0, 1.0, 1.0])
    value = a + z + scale * x + weights[1]
    return value
end

@brm function outer_field(x)
    inner ~ local_field(x)
    shift ~ Normal(0, 1)
    return inner + shift
end

builder = @brm begin
    left ~ local_field(x_left)
    right ~ local_field(x_right)
    nested ~ outer_field(x_nested)
    sigma ~ Exponential(1)
    y_left ~ Normal(left, sigma)
    y_right ~ Normal(right, sigma)
    y_nested ~ Normal(nested, sigma)
end

data = (; x_left=[0.1, 0.3], x_right=[-0.2, 0.2, 0.5],
    x_nested=[-0.4, 0.4], y_left=[0.3, 0.5], y_right=[0.0, 0.1, 0.4],
    y_nested=[-0.1, 0.2])
brmi = builder(data)
println("BUILD_BACKENDS"); flush(stdout)
rk = RKBRMI(brmi)
sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
@test Base.get_extension(BayesianRegressionModels,
    :BayesianRegressionModelsReactiveKernelsExt) !== nothing
d = brm_descriptor(sb)
println("COMPILE_STAN"); flush(stdout)
problem = stan_instantiate(sb; path=joinpath(mktempdir(), "scoped-start.stan"))
model = problem.model
names = BridgeStan.param_names(model)
t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(model))

@testset "public descriptor initialization" begin
    initial = brm_start(sb, model, (; sigma=0.6);
        base=fill(0.17, BridgeStan.param_unc_num(model)))
    original = copy(initial)
    theta = BridgeStan.param_constrain(model, initial)
    original_values = copy(theta)
    changed = Int[]
    expected = [
        (:left, :a, [0.7]), (:left, :z, [-0.2, 0.4]),
        (:left, :scale, [0.8]), (:left, :weights, [0.2, 0.3, 0.5]),
        (:right, :a, [-0.3]), (:right, :z, [0.1, 0.2, -0.4]),
        (:right, :weights, [0.1, 0.6, 0.3]),
        ((:nested, :inner), :a, [0.9]),
        ((:nested, :inner), :z, [0.3, -0.5]),
        ((:nested, :inner), :scale, [0.4]),
        ((:nested, :inner), :weights, [0.5, 0.2, 0.3]),
        (:nested, :shift, [-0.2]),
    ]
    for (scope, name, value) in expected
        @test brm_output(d, name; submodel=scope).kind === :parameter
        coordinates = brm_output_coordinates(d, name, names; submodel=scope)
        @test length(coordinates) == length(value)
        append!(changed, coordinates)
        theta[coordinates] .= value
    end
    stan_start = BridgeStan.param_unconstrain(model, theta)
    rk_start = brm_stan_point_to_rk(t, stan_start)
    @test initial == original
    @test all(isfinite, stan_start)
    @test all(isfinite, rk_start)
    @test brm_rk_point_to_stan(t, rk_start) ≈ stan_start atol=1e-12 rtol=1e-12
    check = brm_check_coordinate_transport(t, rk, model, rk_start)
    @test check.max_error <= 1e-12
    println("PHYSICAL_CHECK=", check)
    readback = BridgeStan.param_constrain(model, stan_start)
    for (scope, name, value) in expected
        @test readback[brm_output_coordinates(d, name, names; submodel=scope)] ≈ value atol=1e-12 rtol=1e-12
    end
    @test readback[brm_output_coordinates(d, :sigma, names)] ≈ [0.6]
    untouched = setdiff(eachindex(theta), changed)
    @test readback[untouched] ≈ original_values[untouched]
end
