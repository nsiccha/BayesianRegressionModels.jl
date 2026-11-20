# Run with RK_JAX_PYTHON naming a Python with JAX.
include(joinpath(@__DIR__, "fitting_packages.jl"))
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

# A `constant(...)` a data-only assignment reads is computed in the RK graph
# from fit-role copies of its inputs. `prepare_data` computes it too, for both
# builders, so the module does not take it as an input (snag
# `jax-source-canno-e77e6b61`: it listed every such constant in UNTRANSLATED).
module JAXConstantModels
using BayesianRegressionModels, Statistics
const SOURCES = (
    scalar_fields=quote
        loc = constant(x_loc)
        scale = constant(x_scale)
        xs = (x - loc) / scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end,
    column_moments=quote
        loc = constant(mean(x))
        scale = constant(std(x))
        xs = (x - loc) / scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end,
    array_constant=quote
        centre = constant([mean(x), x_scale])
        xs = (x - sum(centre)) / x_scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end,
    plate=quote
        loc = constant(w_loc)
        scale = constant(w_scale)
        ws = (w - loc) / scale
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(t)
            m = a .* t[i] .+ b .* ws[i]
            y[i] ~ normal(m, 1.0)
        end
    end,
)
const MACROS = (
    scalar_fields=d -> @brm(d, begin
        loc = constant(x_loc)
        scale = constant(x_scale)
        xs = (x - loc) / scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end),
    column_moments=d -> @brm(d, begin
        loc = constant(mean(x))
        scale = constant(std(x))
        xs = (x - loc) / scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end),
    array_constant=d -> @brm(d, begin
        centre = constant([mean(x), x_scale])
        xs = (x - sum(centre)) / x_scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end),
    plate=d -> @brm(d, begin
        loc = constant(w_loc)
        scale = constant(w_scale)
        ws = (w - loc) / scale
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(t)
            m = a .* t[i] .+ b .* ws[i]
            y[i] ~ normal(m, 1.0)
        end
    end),
)
# The same models with each constant supplied as a data field.
const TWINS = (
    scalar_fields=quote
        xs = (x - loc) / scale
        mu ~ 1 + xs
        y ~ Normal(mu, 1.0)
    end,
    plate=quote
        ws = (w - loc) / scale
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(t)
            m = a .* t[i] .+ b .* ws[i]
            y[i] ~ normal(m, 1.0)
        end
    end,
)
end

@testset "constants read by data-only assignments are prepare_data host code" begin
    rows = (; x=[1.0, 2.0, 3.0, 4.0], y=[0.1, 0.3, 0.2, 0.5], x_loc=2.5, x_scale=1.2)
    subjects = (; w=[60.0, 75.0, 90.0], w_loc=72.0, w_scale=12.0,
        t=[[0.5, 1.0], [0.5], [1.0, 2.0, 3.0]], y=[[0.2, 0.4], [0.1], [0.3, 0.5, 0.2]])
    M = JAXConstantModels
    twin_data = (
        scalar_fields=merge(rows, (; loc=rows.x_loc, scale=rows.x_scale)),
        column_moments=merge(rows, (; loc=M.mean(rows.x), scale=M.std(rows.x))),
        array_constant=merge(rows, (; loc=M.mean(rows.x) + rows.x_scale, scale=rows.x_scale)),
        plate=merge(subjects, (; loc=subjects.w_loc, scale=subjects.w_scale)),
    )
    constants = (scalar_fields=(:loc, :scale), column_moments=(:loc, :scale),
        array_constant=(:centre,), plate=(:loc, :scale))
    for case in keys(M.SOURCES)
        data = case === :plate ? subjects : rows
        twin_source = case === :plate ? M.TWINS.plate : M.TWINS.scalar_fields
        twin = RKBRMI(brm(twin_source; mod=M)(twin_data[case]))
        twin_query = prepare_query(twin.model, rk_bound_plan(twin), :sampler)
        for build in (M.MACROS[case], brm(M.SOURCES[case]; mod=M))
            saved = deepcopy(data)
            rk = RKBRMI(build(data))
            source = jax_source(rk.model, rk_bound_plan(rk))
            @test occursin("\nUNTRANSLATED = {}\n", source.text)
            inputs = rk_bound_data(rk, source)
            @test all(name -> !haskey(inputs, name), constants[case])
            query = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
            for u in ([0.1, -0.2], [0.7, 0.3])
                native = Base.invokelatest(query, u)
                @test native == Base.invokelatest(twin_query, u)
                if rk_jax_available()
                    result = rk_jax_evaluate(source, "log_density", (u, values(inputs)...);
                        grad=true, prepare=true)
                    @test result.value ≈ native rtol=1e-12
                    @test result.grad ≈ rk_fd_gradient(
                        v -> Base.invokelatest(query, v), u) rtol=1e-6 atol=1e-6
                else
                    @test_skip "RK_JAX_PYTHON names no Python with jax"
                end
            end
            # `prepare_data` admits the constant as the graph does
            # (`runtime_constants.jl` checks the native refusal).
            if case === :column_moments && rk_jax_available()
                bad = merge(inputs, (; x_fit=[1.0, Inf, 3.0, 4.0]))
                @test_throws "loc = constant(...)" rk_jax_evaluate(source, "log_density",
                    ([0.1, -0.2], values(bad)...); prepare=true)
            end
            @test isequal(data, saved)
        end
    end
end
