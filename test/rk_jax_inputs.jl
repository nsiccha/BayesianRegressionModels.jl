# Run with RK_JAX_PYTHON naming a Python with JAX.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

module JAXInputModels
using BayesianRegressionModels
const calls = Ref(0)
function opaque_scale(x)
    calls[] += 1
    sum(abs2, x) + 0.5
end
const plate_source = """
    calibration = constant(opaque_scale(anchor))
    a ~ Normal(0, 1)
    @plate for i in eachindex(x)
        m = a .* x[i] ./ calibration
        y[i] ~ normal(m, 1.0)
    end
"""
plate(d) = @brm d begin
    calibration = constant(opaque_scale(anchor))
    a ~ Normal(0, 1)
    @plate for i in eachindex(x)
        m = a .* x[i] ./ calibration
        y[i] ~ normal(m, 1.0)
    end
end
untranslated(d) = @brm d begin
    calibration = constant(opaque_scale(x))
    scaled = x / calibration
    mu ~ 1 + scaled
    y ~ Normal(mu, 1.0)
end
vector_constant(d) = @brm d begin
    calibration = constant(anchor .+ 0.5)
    a ~ Normal(0, 1)
    @plate for i in eachindex(x)
        m = a .* x[i] ./ sum(calibration)
        y[i] ~ normal(m, 1.0)
    end
end
end

const INPUT_TRAIN = (; x=[[0.2, 0.5], [0.8]], y=[[0.1, -0.3], [0.4]],
    anchor=[0.4, 0.9, 1.3])
const INPUT_DESIGN = (; x=[[0.7], [0.9, 1.2]], y=[[0.2], [0.4, -0.1]],
    anchor=[1.4, 1.9, 2.3])

function module_inputs(rk)
    source = jax_source(rk.model, rk_bound_plan(rk))
    native = deepcopy(rk_bound_data(rk))
    identity = BayesianRegressionModels._brm_rk_binding_identity(rk, "jax-input-test")
    calls = JAXInputModels.calls[]
    inputs = rk_bound_data(rk, source)
    names = Tuple(Symbol(first(input)) for input in
        only(f for f in source.functions if f.name == "prepare_data").inputs)
    @test keys(inputs) == names
    @test isequal(rk_bound_data(rk, source), inputs)
    @test JAXInputModels.calls[] == calls
    @test isequal(rk_bound_data(rk), native)
    @test BayesianRegressionModels._brm_rk_binding_identity(rk, "jax-input-test") == identity
    source, inputs
end

function check_plate_inputs(rk, data, calibration)
    saved = deepcopy(data)
    source, inputs = module_inputs(rk)
    @test inputs.calibration == calibration
    @test inputs.x == data.x
    @test inputs.y == data.y
    query = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
    scale = calibration isa Number ? calibration : sum(calibration)
    for a in (-0.4, 0.7)
        u = [a]
        residuals = [y - a * x / scale for (xs, ys) in zip(data.x, data.y)
            for (x, y) in zip(xs, ys)]
        expected = -(1 + length(residuals)) * log(2pi) / 2 -
            (a^2 + sum(abs2, residuals)) / 2
        expected_gradient = -a + sum((y - a * x / scale) * x / scale
            for (xs, ys) in zip(data.x, data.y) for (x, y) in zip(xs, ys))
        @test Base.invokelatest(query, u) ≈ expected rtol=1e-12
        if rk_jax_available()
            result = rk_jax_evaluate(source, "log_density", (u, values(inputs)...);
                grad=true, prepare=true)
            @test result.value ≈ expected rtol=1e-12
            @test only(result.grad) ≈ expected_gradient rtol=1e-12 atol=1e-12
        else
            @test_skip "RK_JAX_PYTHON names no Python with jax"
        end
    end
    @test isequal(data, saved)
end

@testset "JAX inputs use authored scalar names on fits and replays" begin
    # Float likelihood scale: integer scale has a separate RK conditional
    # pytree defect, tracked by snag jax-ragged-integ-35502154.
    for build in (JAXInputModels.plate,
            brm(JAXInputModels.plate_source; mod=JAXInputModels))
        fit = RKBRMI(build(INPUT_TRAIN))
        check_plate_inputs(fit, INPUT_TRAIN, 3.16)
        check_plate_inputs(reprocess(fit, INPUT_DESIGN), INPUT_DESIGN, 3.16)
        check_plate_inputs(reprocess(fit, INPUT_DESIGN; freeze_constants=false),
            INPUT_DESIGN, 11.36)
        free = RKBRMI(build(INPUT_DESIGN), brm_fitted_state(fit))
        check_plate_inputs(free, INPUT_DESIGN, 3.16)
    end
end

@testset "ordinary scalar inputs and array constants" begin
    supplied = brm(replace(JAXInputModels.plate_source,
        "calibration = constant(opaque_scale(anchor))" => ""); mod=JAXInputModels)
    data = merge(INPUT_TRAIN, (; calibration=2.7))
    check_plate_inputs(RKBRMI(supplied(data)), data, 2.7)
    check_plate_inputs(RKBRMI(JAXInputModels.vector_constant(INPUT_TRAIN)), INPUT_TRAIN,
        INPUT_TRAIN.anchor .+ 0.5)
end

@testset "untranslated data-only definitions and exact input filtering" begin
    data = (; x=[0.2, 0.5, 0.8], y=[0.1, -0.3, 0.4])
    saved = deepcopy(data)
    rk = RKBRMI(JAXInputModels.untranslated(data))
    source, inputs = module_inputs(rk)
    @test inputs.calibration ≈ 1.43
    @test !occursin("UNTRANSLATED = {}", source.text)
    @test !haskey(inputs, :X_mu)
    @test haskey(rk_bound_data(rk), :X_mu)
    query = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
    if rk_jax_available()
        u = collect(range(-0.4, 0.7; length=rk.model.layout.total))
        result = rk_jax_evaluate(source, "log_density", (u, values(inputs)...);
            grad=true, prepare=true)
        @test result.value ≈ Base.invokelatest(query, u) rtol=1e-12
        @test result.grad ≈ rk_fd_gradient(v -> Base.invokelatest(query, v), u) rtol=1e-6 atol=1e-6
    else
        @test_skip "RK_JAX_PYTHON names no Python with jax"
    end
    @test isequal(data, saved)
end
