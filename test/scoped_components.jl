include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module ScopedCalibration
using BayesianRegressionModels, Distributions

@brm function calibration(x, shared_scale)
    value = a + b * x
    a ~ Normal(0, shared_scale)
    b ~ Normal(1, 0.2)
    return value
end

@brm function nested(x, shared_scale)
    inner ~ calibration(x, shared_scale)
    shift ~ Normal(0, 0.3)
    return inner + shift
end

@brm function deviations(x, shared_scale)
    z[1:length(x)] .~ Normal.(0, shared_scale)
    return x + z
end

@brm function observed_calibration(x, y, shared_scale)
    value ~ calibration(x, shared_scale)
    y ~ Normal(value, 0.7)
    return value
end

@brm function cell_field(x, shared_scale)
    @plate for i in eachindex(x)
        z ~ Normal(0, shared_scale)
        value[i] = x[i] + z
    end
    return value
end

@brm function intercept_field(x)
    value ~ 1 + offset(x)
    effect(value, Intercept) ~ Normal(0, 1)
    return value
end

@brm function regression_field(x)
    value ~ 1 + x
    effect(value, :) ~ Normal(0, 1)
    return value
end

@brm function nested_cell_field(x, scale)
    inner ~ cell_field(x, scale)
    return inner + 0.1
end

@brm function shape_field(x, theta)
    @plate for i in eachindex(x)
        value[i] = theta .* x[i]
    end
    return value
end
end

module ScopedConsumer
using BayesianRegressionModels, Distributions
using ..ScopedCalibration: calibration, nested, deviations

const repeated = @brm begin
    left ~ calibration(x_left, shared_scale)
    right ~ calibration(x_right, shared_scale)
    y_left ~ Normal(left, obs_sd)
    y_right ~ Normal(right, obs_sd)
    shared_scale ~ Exponential(1)
    obs_sd ~ Exponential(1)
end


const expanded = @brm begin
    left = left_a + left_b * x_left
    right = right_a + right_b * x_right
    left_a ~ Normal(0, shared_scale)
    left_b ~ Normal(1, 0.2)
    right_a ~ Normal(0, shared_scale)
    right_b ~ Normal(1, 0.2)
    y_left ~ Normal(left, obs_sd)
    y_right ~ Normal(right, obs_sd)
    shared_scale ~ Exponential(1)
    obs_sd ~ Exponential(1)
end
end

@stestset "component plate offsets prove rows before flattening local cells" begin
    data = (; x=[-0.4, 0.2, 0.7], read_row=[3, 1, 3], y=[0.2, -0.1, 0.4])
    for component in (:cell_field, :nested_cell_field)
        builder = brm(quote
            field ~ $component(x, scale)
            quantity ~ 1 + offset(2 * field)
            effect(quantity, Intercept) ~ Normal(0, 1)
            y ~ Normal(quantity[read_row], 0.7)
            scale ~ Exponential(1)
        end; mod=ScopedCalibration)
        model = builder(data)
        rk, problem = consumer_problem(model)
        sb = SBBRMI(model; mod=ScopedCalibration, total_groups=())
        stan = consumer_stan(model, "component-cell-offset-$component"; mod=ScopedCalibration)
        mapping = [pair.rk => pair.stan for pair in
            brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
        names = coordinate_names(rk.model.layout)
        scale = coordinate(names, :scale)
        level = coordinate(names, :quantity_Intercept)
        stem = component === :cell_field ? "field.z." : "field.inner.z."
        draws = [coordinate(names, Symbol(stem, i)) for i in eachindex(data.x)]
        shift = component === :cell_field ? 0.0 : 0.1
        oracle(u) = logpdf(Exponential(1), exp(u[scale])) + u[scale] +
            logpdf(Normal(), u[level]) + sum(logpdf.(Normal(0, exp(u[scale])), u[draws])) +
            sum(logpdf.(Normal.(u[level] .+ 2 .* (data.x[data.read_row] .+
                u[draws][data.read_row] .+ shift), 0.7), data.y))
        for point in (zeros(length(names)), collect(range(-0.2, 0.3; length=length(names))))
            check_consumer_point(problem, point, oracle)
            check_consumer_stan(problem, stan, mapping, rk, point)
        end
    end
    ragged = (; x=[[0.2, 0.4], Float64[], [0.8]], fixed=zeros(3), y=ones(3))
    builder = brm(quote
        field ~ shape_field(x, theta)
        quantity ~ 1 + offset(field) + offset(fixed)
        effect(quantity, Intercept) ~ Normal(0, 1)
        y ~ Normal(quantity, 0.7)
        theta ~ Normal(0, 1)
    end; mod=ScopedCalibration)
    @test_throws r"offset of a plate output requires one real scalar per cell" begin
        rk = RKBRMI(builder(ragged))
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        LogDensityProblems.logdensity(problem, fill(0.2, rk.model.layout.total))
    end
end

@stestset "one BRM component gives each call its own local priors" begin
    data = (; x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
        y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1])
    saved = deepcopy(data)
    model = ScopedConsumer.repeated(data)
    backend, problem = consumer_problem(model)
    names = coordinate_names(backend.model.layout)
    @test Set(names) == Set((Symbol("left.a"), Symbol("left.b"),
        Symbol("right.a"), Symbol("right.b"), :shared_scale, :obs_sd))
    ia, ib = coordinate(names, Symbol("left.a")), coordinate(names, Symbol("left.b"))
    ja, jb = coordinate(names, Symbol("right.a")), coordinate(names, Symbol("right.b"))
    is, io = coordinate(names, :shared_scale), coordinate(names, :obs_sd)
    function oracle(u)
        scale, sd = exp(u[is]), exp(u[io])
        logpdf(Exponential(1), scale) + u[is] +
        logpdf(Exponential(1), sd) + u[io] +
        logpdf(Normal(0, scale), u[ia]) + logpdf(Normal(1, 0.2), u[ib]) +
        logpdf(Normal(0, scale), u[ja]) + logpdf(Normal(1, 0.2), u[jb]) +
        sum(logpdf.(Normal.(u[ia] .+ u[ib] .* data.x_left, sd), data.y_left)) +
        sum(logpdf.(Normal.(u[ja] .+ u[jb] .* data.x_right, sd), data.y_right))
    end
    stan = consumer_stan(model, "scoped-calibration"; mod=ScopedConsumer)
    sb = SBBRMI(model; mod=ScopedConsumer, total_groups=())
    transport = brm_coordinate_transport(backend, sb, BridgeStan.param_unc_names(stan.model))
    mapping = [pair.rk => pair.stan for pair in transport.pairs]
    expanded, expanded_problem = consumer_problem(ScopedConsumer.expanded(data))
    expanded_names = coordinate_names(expanded.model.layout)
    permutation = [coordinate(names, Symbol(replace(string(n), "left_" => "left.",
        "right_" => "right."))) for n in expanded_names]
    for u in (zeros(6), collect(range(-0.3, 0.4; length=6)), fill(0.2, 6))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
        actual, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        control, control_gradient = LogDensityProblems.logdensity_and_gradient(
            expanded_problem, u[permutation])
        @test actual ≈ control atol=2e-11 rtol=2e-11
        @test gradient[permutation] ≈ control_gradient atol=2e-10 rtol=2e-10
    end
    @test isequal(data, saved)
end

module ScopedVariants
using BayesianRegressionModels, Distributions, Statistics
using ..ScopedCalibration: calibration, nested, deviations, observed_calibration,
    cell_field, intercept_field, regression_field

const nesting = @brm begin
    field ~ nested(x, scale)
    y ~ Normal(field, 0.7)
    scale ~ Exponential(1)
end
const vector = @brm begin
    field ~ deviations(x, scale)
    y ~ Normal(field, 0.7)
    scale ~ Exponential(1)
end
const plate_body = @brm begin
    field ~ cell_field(x, scale)
    y ~ Normal(field, 0.7)
    scale ~ Exponential(1)
end
const formula_body = @brm begin
    field ~ intercept_field(x)
    y ~ Normal(field, 0.7)
end
const regression_body = @brm begin
    field ~ regression_field(x)
    y ~ Normal(field, 0.7)
end
const rows = @brm begin
    field ~ calibration(x, scale)
    log(quantity) ~ 1 + offset(field) + offset(fixed)
    effect(quantity, Intercept) ~ Normal(0, 1)
    y ~ Normal(quantity, 0.7)
    scale ~ Exponential(1)
end
const cells = @brm begin
    @plate for i in eachindex(x)
        cell ~ calibration(x[i], scale)
        location[i] = cell
        y[i] ~ Normal(location[i], 0.7)
    end
    scale ~ Exponential(1)
end

@brm function defaults(x; scale=0.5)
    a ~ Normal(0, scale)
    return x + a
end
const default_calls = @brm begin
    left ~ defaults(x)
    right ~ defaults(x; scale=0.8)
    y ~ Normal(left + right, 0.7)
end

@brm function calibrated(x)
    anchor = constant(mean(x))
    value ~ 1 + zscale(x)
    effect(value, :) ~ Normal(0, 1)
    return value + anchor
end
const fitted_calls = @brm begin
    left ~ calibrated(x_left)
    right ~ calibrated(x_right)
    y_left ~ Normal(left, 0.7)
    y_right ~ Normal(right, 0.7)
end

const selected_rows = @brm begin
    field ~ calibration(x, scale)
    selected = field[pick]
    y ~ Normal(selected, 0.7)
    scale ~ Exponential(1)
end
const observations_inside = @brm begin
    left ~ observed_calibration(x_left, y_left, scale)
    right ~ observed_calibration(x_right, y_right, scale)
    scale ~ Exponential(1)
end
end

@stestset "component source, ownership and changed-data replay" begin
    data = (; x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
        y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1])
    sb = SBBRMI(ScopedConsumer.repeated(data); mod=ScopedConsumer, total_groups=())
    d = brm_descriptor(sb)
    @test [s.name for s in d.submodels] == [:calibration, :calibration]
    @test [s.target for s in d.submodels] == [:left, :right]
    for target in (:left, :right)
        submodel = only(filter(s -> s.target === target, d.submodels))
        @test Set(dec.target for dec in submodel.declarations if dec.role === :prior) == Set((:a, :b))
        @test brm_output(d, :a; submodel=target).role === :parameter
        @test brm_output(d, :b; submodel=target).role === :parameter
        @test submodel.model.mod === ScopedCalibration
    end
    @test brm_columns(ScopedCalibration.calibration) == (:x, :shared_scale)
    calls = brm_components(sb)
    @test [call.path for call in calls] == [(:left,), (:right,)]
    @test all(call -> call.definition === ScopedCalibration.calibration, calls)
    @test all(call -> call.scope === ScopedCalibration, calls)
    @test all(call -> call.source == brm_source(call.definition), calls)
    @test calls[1].call == :(calibration(x_left, shared_scale))
    empty!(calls[1].call.args)
    empty!(calls[1].source.args)
    @test brm_components(sb)[1].call == :(calibration(x_left, shared_scale))
    @test brm_components(sb)[1].source == brm_source(ScopedCalibration.calibration)
    cell_calls = brm_components(ScopedVariants.cells((; x=[0.1, 0.4], y=[0.2, 0.3])))
    @test [call.path for call in cell_calls] == [(:location, :cell)]
    @test only(cell_calls).definition === ScopedCalibration.calibration
    nested = brm_components(ScopedCalibration.nested)
    @test [call.path for call in nested] == [(), (:inner,)]
    @test nested[2].definition === ScopedCalibration.calibration
    rebuilt = brm(brm_source(ScopedCalibration.calibration); mod=ScopedCalibration)
    @test rebuilt isa BRMComponent
    @test brm_source(rebuilt) == brm_source(ScopedCalibration.calibration)
    @test brm(Meta.parse(sprint(show, rebuilt)); mod=ScopedCalibration) isa BRMComponent
    more = (; x_left=[-0.7, 0.2], x_right=[0.1, -0.2, 0.8, 0.9],
        y_left=[0.0, 0.4], y_right=[0.3, -0.1, 0.6, 0.7])
    rk = RKBRMI(ScopedConsumer.repeated(data))
    for (replay, fresh) in ((reprocess(sb, more), SBBRMI(ScopedConsumer.repeated(more);
                mod=ScopedConsumer, total_groups=())),
            (reprocess(rk, more), RKBRMI(ScopedConsumer.repeated(more))))
        if replay isa SBBRMI
            @test stan_code(replay) == stan_code(fresh)
            @test isequal(stan_data(replay), stan_data(fresh))
        else
            @test coordinate_names(replay.model.layout) == coordinate_names(fresh.model.layout)
            p = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            q = rk_logdensity_problem(fresh; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in (zeros(6), fill(0.2, 6))
                @test LogDensityProblems.logdensity(p, u) ≈ LogDensityProblems.logdensity(q, u)
            end
        end
    end
end

module ScopedDocumented
using BayesianRegressionModels, Distributions
"A documented calibration component."
@brm function documented(x)
    "Local intercept, unique to this call."
    a ~ Normal(0, 1)
    return x + a
end
@brm function enclosing(x)
    "Call the documented component."
    inner ~ documented(x)
    return inner
end
const model = @brm begin
    left ~ enclosing(x)
    right ~ documented(x)
    y ~ Normal(left + right, 0.7)
end
end

module ScopedRedefinedDocs
using BayesianRegressionModels, Distributions
"Keep this header on the original definition."
@brm function replaceable(x)
    a ~ Normal(0, 1)
    return a + x
end
const original = replaceable
@brm function replaceable(x)
    a ~ Normal(0, 2)
    return a + x
end
const replacement = replaceable
end

@stestset "component docs retain headers, body prose and repeated instance paths" begin
    c = ScopedDocumented.documented
    docs = brm_documentation(c)
    @test [d.path for d in docs] == [(), (:a,)]
    @test [d.markdown for d in docs] == ["A documented calibration component.",
        "Local intercept, unique to this call."]
    @test all(d -> d.line > 0, docs)
    @test all(d -> d.file !== nothing, docs)
    @test only(brm_documentation(ScopedRedefinedDocs.original)).markdown ==
        "Keep this header on the original definition."
    @test isempty(brm_documentation(ScopedRedefinedDocs.replacement))
    rebuilt = brm(brm_source(c); mod=ScopedDocumented)
    @test [(d.path, d.markdown) for d in brm_documentation(rebuilt)] ==
        [(d.path, d.markdown) for d in docs]
    @test brm_columns(rebuilt) == (:x,)
    data = (; x=[-0.2, 0.4], y=[0.1, 0.3])
    model = ScopedDocumented.model(data)
    @test [call.path for call in brm_components(model)] ==
        [(:left,), (:left, :inner), (:right,)]
    @test only(brm_documentation(model; path=(:left, :inner, :a))).markdown ==
        "Local intercept, unique to this call."
    @test only(brm_documentation(model; path=(:right,))).markdown ==
        "A documented calibration component."
    copied = brm_documentation(model)
    empty!(last(copied).statement.args)
    @test !isempty(last(brm_documentation(model)).statement.args)
    @test occursin("Local intercept", brm_documentation_markdown(model))
    sb = SBBRMI(model; mod=ScopedDocumented, total_groups=())
    rk = RKBRMI(model)
    @test [(d.path, d.markdown) for d in brm_documentation(sb)] ==
        [(d.path, d.markdown) for d in brm_documentation(rk)] ==
        [(d.path, d.markdown) for d in brm_documentation(model)]
    native = rkppl_documentation(rk_bound_plan(rk))
    for path in ((:left, :inner), (:right,))
        @test any(d -> d.path == path &&
            d.markdown == "A documented calibration component.", native)
        local_doc = only(d for d in native if d.path == (path..., :a))
        @test local_doc.markdown == "Local intercept, unique to this call."
        @test local_doc.file == docs[2].file
        @test local_doc.line == docs[2].line
    end
    artifact = BRM.emit_rk_artifact(model; case_id="scoped-documentation")
    @test [(d.path, d.markdown) for d in rkppl_documentation(rk_translate_artifact(artifact))] ==
        [(d.path, d.markdown) for d in native]
    @test check_rk_source_roundtrip(rk) === rk
end

module ScopedErrors
using BayesianRegressionModels, Distributions
@brm function unbound(x)
    a ~ Normal(0, caller_scale)
    return a + x
end
@brm function recursive(x)
    inner ~ recursive(x)
    return inner
end
const missing_input = @brm begin
    field ~ unbound(x)
    y ~ Normal(field, 1)
    caller_scale ~ Exponential(1)
end
const cyclic = @brm begin
    field ~ recursive(x)
    y ~ Normal(field, 1)
end
end

@stestset "component interfaces reject missing inputs and recursive definitions" begin
    data = (; x=[0.1, 0.3], y=[-0.2, 0.5])
    for backend in (model -> SBBRMI(model; mod=ScopedErrors, total_groups=()), RKBRMI)
        @test_throws r"reads undeclared input `caller_scale`" backend(ScopedErrors.missing_input(data))
        @test_throws r"recursive BRM component calls" backend(ScopedErrors.cyclic(data))
    end
    @test_throws r"ends with `return value`" brm(:(function no_return(x)
        a ~ Normal(0, 1)
    end); mod=ScopedErrors)
end

@stestset "nested, vector, default and scalar-cell components compile on both backends" begin
    data = (; x=[-0.4, 0.2, 0.7], y=[0.1, 0.4, 0.8])
    for (builder, count) in ((ScopedVariants.nesting, 4), (ScopedVariants.vector, 4),
            (ScopedVariants.default_calls, 2), (ScopedVariants.cells, 7),
            (ScopedVariants.plate_body, 4), (ScopedVariants.formula_body, 1),
            (ScopedVariants.regression_body, 2))
        println("COMPONENT_EXPECTED_COORDINATES=", count); flush(stdout)
        model = builder(data)
        rk = check_rk_source_roundtrip(RKBRMI(model))
        sb = SBBRMI(model; mod=ScopedVariants, total_groups=())
        @test rk.model.layout.total == count
        @test !isempty(stan_code(sb))
        if builder === ScopedVariants.cells
            descriptor = brm_descriptor(sb)
            @test only(descriptor.submodels).name === :calibration
            @test brm_output(descriptor, :a; submodel=(:location, :cell)).kind === :parameter
            @test brm_output(descriptor, :b; submodel=(:location, :cell)).kind === :parameter
        end
    end
end

@stestset "component variants preserve normalized density and native reverse gradients" begin
    data = (; x=[-0.4, 0.2, 0.7], y=[0.1, 0.4, 0.8])
    for (label, builder) in (("nested", ScopedVariants.nesting),
            ("vector", ScopedVariants.vector), ("defaults", ScopedVariants.default_calls),
            ("cells", ScopedVariants.cells), ("plate-body", ScopedVariants.plate_body),
            ("formula-body", ScopedVariants.formula_body),
            ("regression-body", ScopedVariants.regression_body))
        println("COMPONENT_DENSITY=", label); flush(stdout)
        model = builder(data)
        rk, problem = consumer_problem(model)
        sb = SBBRMI(model; mod=ScopedVariants, total_groups=())
        stan = consumer_stan(model, "component-" * label; mod=ScopedVariants)
        transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
        mapping = [pair.rk => pair.stan for pair in transport.pairs]
        for point in (zeros(rk.model.layout.total),
                collect(range(-0.2; step=0.5 / max(rk.model.layout.total - 1, 1),
                    length=rk.model.layout.total)))
            check_consumer_stan(problem, stan, mapping, rk, point)
            saved = copy(point)
            stan_point = brm_rk_point_to_stan(transport, point)
            @test brm_stan_point_to_rk(transport, stan_point) ≈ point atol=1e-12 rtol=1e-12
            @test isequal(point, saved)
            checked = brm_check_coordinate_transport(transport, rk, stan.model, point)
            @test checked.pairs == length(transport)
            @test checked.max_error <= 1e-12
        end
    end
end

@stestset "component vector output retains rows for selective consumers" begin
    data = (; x=[-0.4, 0.2, 0.7, 0.9], pick=[4, 1, 4], y=[0.2, -0.1, 0.3])
    model = ScopedVariants.selected_rows(data)
    rk, problem = consumer_problem(model)
    names = coordinate_names(rk.model.layout)
    ia, ib, is = coordinate(names, Symbol("field.a")),
        coordinate(names, Symbol("field.b")), coordinate(names, :scale)
    oracle(u) = logpdf(Exponential(1), exp(u[is])) + u[is] +
        logpdf(Normal(0, exp(u[is])), u[ia]) + logpdf(Normal(1, 0.2), u[ib]) +
        sum(logpdf.(Normal.(u[ia] .+ u[ib] .* data.x[data.pick], 0.7), data.y))
    stan = consumer_stan(model, "component-selected-rows"; mod=ScopedVariants)
    sb = SBBRMI(model; mod=ScopedVariants, total_groups=())
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
    for point in (zeros(3), [-0.1, 0.3, 0.2])
        check_consumer_point(problem, point, oracle)
        check_consumer_stan(problem, stan, mapping, rk, point)
    end
end

@stestset "each component retains its own fitted preprocessing on replay" begin
    train = (; x_left=[-2.0, 0.0, 3.0], x_right=[0.5, 1.5, 2.0, 4.0],
        y_left=[0.1, -0.2, 0.3], y_right=[0.2, 0.1, -0.1, 0.4])
    future = (; x_left=[4.0, 6.0], x_right=[-1.0, 0.2, 0.8],
        y_left=[0.5, 0.3], y_right=[-0.4, 0.1, 0.3])
    builder = ScopedVariants.fitted_calls
    sb = SBBRMI(builder(train); mod=ScopedVariants, total_groups=())
    rk = RKBRMI(builder(train))
    replay_sb, replay_rk = reprocess(sb, future), reprocess(rk, future)
    problem = rk_logdensity_problem(replay_rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    stan = BRM.stan_instantiate(replay_sb;
        path=joinpath(tempdir(), "brm-rk-consumer", "component-fitted.stan"))
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(replay_rk, replay_sb, BridgeStan.param_unc_names(stan.model)).pairs]
    for point in (zeros(rk.model.layout.total), fill(0.3, rk.model.layout.total))
        check_consumer_stan(problem, stan, mapping, replay_rk, point)
    end
    refit = reprocess(rk, future; freeze_constants=false)
    fresh = RKBRMI(builder(future))
    value(backend, u) = LogDensityProblems.logdensity(
        rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
    point = fill(0.2, rk.model.layout.total)
    @test value(refit, point) ≈ value(fresh, point)
    @test !isapprox(value(replay_rk, point), value(fresh, point))
    state = brm_fitted_state(rk)
    data_free = RKBRMI(builder(future), state)
    @test value(data_free, point) ≈ value(replay_rk, point)
end

@stestset "component output has an aligned row frame for formula offsets" begin
    data = (; x=[-0.4, 0.2, 0.7], fixed=[0.1, -0.2, 0.3], y=[0.8, 1.2, 2.1])
    model = ScopedVariants.rows(data)
    rk, problem = consumer_problem(model)
    names = coordinate_names(rk.model.layout)
    ia, ib, is = coordinate(names, Symbol("field.a")),
        coordinate(names, Symbol("field.b")), coordinate(names, :scale)
    @test length(names) == 4
    intercept = only(setdiff(eachindex(names), (ia, ib, is)))
    oracle(u) = logpdf(Exponential(1), exp(u[is])) + u[is] +
        logpdf(Normal(0, exp(u[is])), u[ia]) + logpdf(Normal(1, 0.2), u[ib]) +
        logpdf(Normal(0, 1), u[intercept]) +
        sum(logpdf.(Normal.(exp.(u[intercept] .+ u[ia] .+ u[ib] .* data.x .+ data.fixed),
            0.7), data.y))
    sb = SBBRMI(model; mod=ScopedVariants, total_groups=())
    stan = consumer_stan(model, "component-offset-rows"; mod=ScopedVariants)
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
    for point in (zeros(4), collect(range(-0.2, 0.3; length=4)))
        check_consumer_point(problem, point, oracle)
        check_consumer_stan(problem, stan, mapping, rk, point)
    end
end

@stestset "component observations contribute to the parent likelihood" begin
    data = (; x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
        y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1])
    model = ScopedVariants.observations_inside(data)
    rk, problem = consumer_problem(model)
    @test rk.model.layout.total == 5
    @test Set(BRM._rk_observed_names(rk.plan)) == Set((:y_left, :y_right))
    names = coordinate_names(rk.model.layout)
    is = coordinate(names, :scale)
    ia, ib = coordinate(names, Symbol("left.value.a")), coordinate(names, Symbol("left.value.b"))
    ja, jb = coordinate(names, Symbol("right.value.a")), coordinate(names, Symbol("right.value.b"))
    oracle(u) = logpdf(Exponential(1), exp(u[is])) + u[is] +
        logpdf(Normal(0, exp(u[is])), u[ia]) + logpdf(Normal(1, 0.2), u[ib]) +
        logpdf(Normal(0, exp(u[is])), u[ja]) + logpdf(Normal(1, 0.2), u[jb]) +
        sum(logpdf.(Normal.(u[ia] .+ u[ib] .* data.x_left, 0.7), data.y_left)) +
        sum(logpdf.(Normal.(u[ja] .+ u[jb] .* data.x_right, 0.7), data.y_right))
    stan = consumer_stan(model, "component-owned-observations"; mod=ScopedVariants)
    sb = SBBRMI(model; mod=ScopedVariants, total_groups=())
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
    for point in (zeros(5), fill(0.2, 5))
        check_consumer_point(problem, point, oracle)
        check_consumer_stan(problem, stan, mapping, rk, point)
    end
end

@stestset "plate component coordinates preserve the scalar cell density" begin
    data = (; x=[-0.4, 0.2, 0.7], y=[0.1, 0.4, 0.8])
    rk, problem = consumer_problem(ScopedVariants.cells(data))
    names = coordinate_names(rk.model.layout)
    is = coordinate(names, :scale)
    intercepts = [coordinate(names, Symbol("cell.a.", i)) for i in eachindex(data.x)]
    slopes = [coordinate(names, Symbol("cell.b.", i)) for i in eachindex(data.x)]
    oracle(u) = logpdf(Exponential(1), exp(u[is])) + u[is] +
        sum(logpdf.(Normal(0, exp(u[is])), u[intercepts])) +
        sum(logpdf.(Normal(1, 0.2), u[slopes])) +
        sum(logpdf.(Normal.(u[intercepts] .+ u[slopes] .* data.x, 0.7), data.y))
    for point in (zeros(7), collect(range(-0.2, 0.3; length=7)))
        check_consumer_point(problem, point, oracle)
    end
end

module ScopedNamed
using BayesianRegressionModels, Distributions

"Two related curves from one shared pair of coefficients."
@brm function bounds(x, scale)
    a ~ Normal(0, scale)
    b ~ Normal(1, 0.2)
    lo = a + b * x
    return (; lo, hi = a - b * x)
end

# Two linear predictors whose unit effects share one correlation block.
@brm function kinetics(x, unit)
    recovery ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    return (; recovery, decay)
end

@brm function shifted(x, scale)
    inner ~ bounds(x, scale)
    shift ~ Normal(0, 0.3)
    return (up = inner.lo + shift, down = inner.hi)
end

const fields = @brm begin
    k ~ bounds(x, scale)
    y1 ~ Normal(k.lo, 0.5)
    y2 ~ Normal(k.lo + k.hi, 0.5)
    scale ~ Exponential(1)
end
const offset_field = @brm begin
    k ~ bounds(x, scale)
    log(q) ~ 1 + offset(k.hi)
    effect(q, Intercept) ~ Normal(0, 1)
    y1 ~ Normal(q, 0.5)
    y2 ~ Normal(k.lo, 0.5)
    scale ~ Exponential(1)
end
# A parent binding may take the field's own name; the plate slices it.
const plate_fields = @brm begin
    k ~ bounds(x, scale)
    lo = k.lo
    hi = k.hi
    @plate for i in eachindex(y1)
        y1[i] ~ Normal(2 * lo[i], 0.5)
        y2[i] ~ Normal(hi[i], 0.5)
    end
    scale ~ Exponential(1)
end
const nested_fields = @brm begin
    o ~ shifted(x, scale)
    y1 ~ Normal(o.up, 0.5)
    y2 ~ Normal(o.down, 0.5)
    scale ~ Exponential(1)
end
const kinetic_fields = @brm begin
    kin ~ kinetics(x, unit)
    recovery = kin.recovery
    y1 ~ Normal(recovery, 0.5)
    y2 ~ Normal(kin.decay, 0.5)
end
const kinetic_expanded = @brm begin
    recovery ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    y1 ~ Normal(recovery, 0.5)
    y2 ~ Normal(decay, 0.5)
end
# A grouped field is row-aligned through its label input: a formula offset.
const kinetic_offset = @brm begin
    kin ~ kinetics(x, unit)
    y1 ~ Normal(kin.recovery, 0.5)
    log(q) ~ 1 + offset(kin.decay)
    effect(q, Intercept) ~ Normal(0, 1)
    y2 ~ Normal(q, 0.5)
end
const kinetic_offset_expanded = @brm begin
    recovery ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    y1 ~ Normal(recovery, 0.5)
    log(q) ~ 1 + offset(decay)
    effect(q, Intercept) ~ Normal(0, 1)
    y2 ~ Normal(q, 0.5)
end
const unknown_field = @brm begin
    k ~ bounds(x, scale)
    y1 ~ Normal(k.mid, 0.5)
    scale ~ Exponential(1)
end
const bare_read = @brm begin
    k ~ bounds(x, scale)
    y1 ~ Normal(k, 0.5)
    scale ~ Exponential(1)
end
const cell_call = @brm begin
    @plate for i in eachindex(x)
        cell ~ bounds(x[i], scale)
        y1[i] ~ Normal(cell.lo, 0.5)
    end
    scale ~ Exponential(1)
end
end

@stestset "a component returns named values read as fields on both backends" begin
    data = (; x=[-0.4, 0.2, 0.7], y1=[0.1, 0.4, 0.8], y2=[0.3, -0.1, 0.2])
    @test ScopedNamed.bounds.result isa NamedTuple
    @test keys(ScopedNamed.bounds.result) == (:lo, :hi)
    @test brm_source(ScopedNamed.bounds) == brm_source(brm(brm_source(ScopedNamed.bounds);
        mod=ScopedNamed))
    lo(u, ia, ib) = u[ia] .+ u[ib] .* data.x
    hi(u, ia, ib) = u[ia] .- u[ib] .* data.x
    prior(u, ia, ib, is) = logpdf(Exponential(1), exp(u[is])) + u[is] +
        logpdf(Normal(0, exp(u[is])), u[ia]) + logpdf(Normal(1, 0.2), u[ib])
    for (label, builder, likelihood) in (
            ("fields", ScopedNamed.fields, (u, ia, ib) ->
                sum(logpdf.(Normal.(lo(u, ia, ib), 0.5), data.y1)) +
                sum(logpdf.(Normal.(lo(u, ia, ib) .+ hi(u, ia, ib), 0.5), data.y2))),
            ("plate", ScopedNamed.plate_fields, (u, ia, ib) ->
                sum(logpdf.(Normal.(2 .* lo(u, ia, ib), 0.5), data.y1)) +
                sum(logpdf.(Normal.(hi(u, ia, ib), 0.5), data.y2))))
        println("NAMED_COMPONENT=", label); flush(stdout)
        model = builder(data)
        rk, problem = consumer_problem(model)
        names = coordinate_names(rk.model.layout)
        ia, ib, is = coordinate(names, Symbol("k.a")), coordinate(names, Symbol("k.b")),
            coordinate(names, :scale)
        @test length(names) == 3
        oracle(u) = prior(u, ia, ib, is) + likelihood(u, ia, ib)
        sb = SBBRMI(model; mod=ScopedNamed, total_groups=())
        @test occursin("tuple(vector[x_n], vector[x_n]) k", stan_code(sb))
        stan = consumer_stan(model, "named-component-" * label; mod=ScopedNamed)
        mapping = [pair.rk => pair.stan for pair in
            brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
        for point in (zeros(3), [-0.2, 0.1, 0.3])
            check_consumer_point(problem, point, oracle)
            check_consumer_stan(problem, stan, mapping, rk, point)
        end
    end
    for (label, builder, count) in (("offset", ScopedNamed.offset_field, 4),
            ("nested", ScopedNamed.nested_fields, 4))
        println("NAMED_COMPONENT=", label); flush(stdout)
        model = builder(data)
        rk, problem = consumer_problem(model)
        @test rk.model.layout.total == count
        sb = SBBRMI(model; mod=ScopedNamed, total_groups=())
        stan = consumer_stan(model, "named-component-" * label; mod=ScopedNamed)
        mapping = [pair.rk => pair.stan for pair in
            brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
        for point in (zeros(count), collect(range(-0.3, 0.4; length=count)))
            check_consumer_stan(problem, stan, mapping, rk, point)
        end
    end
    d = brm_descriptor(SBBRMI(ScopedNamed.nested_fields(data); mod=ScopedNamed,
        total_groups=()))
    @test brm_output(d, :up; submodel=:o).name === :o_component_result_up
    @test brm_output(d, :hi; submodel=(:o, :inner)).name === :o_inner_component_result_hi
    @test brm_output(d, :lo; submodel=(:o, :inner)).name === :o_inner_lo
    @test brm_output(d, :shift; submodel=:o).kind === :parameter
end

@stestset "correlated predictors returned by one component equal their expansion" begin
    data = (; x=[-0.4, 0.2, 0.7, 0.1, -0.3], y1=[0.1, 0.4, 0.8, 0.0, -0.2],
        y2=[0.3, -0.1, 0.2, 0.5, 0.1], unit=["a", "b", "a", "c", "b"])
    for (label, component_builder, expanded_builder, count) in (
            ("fields", ScopedNamed.kinetic_fields, ScopedNamed.kinetic_expanded, 12),
            ("offset", ScopedNamed.kinetic_offset, ScopedNamed.kinetic_offset_expanded, 13))
        println("NAMED_KINETICS=", label); flush(stdout)
        component, expanded = component_builder(data), expanded_builder(data)
        # RK: the call scopes the expansion's coordinates; nothing else changes.
        rk_c, problem_c = consumer_problem(component)
        rk_e, problem_e = consumer_problem(expanded)
        names_c = coordinate_names(rk_c.model.layout)
        names_e = coordinate_names(rk_e.model.layout)
        @test length(names_c) == length(names_e) == count
        # Parent coordinates keep their names; the call scopes the rest.
        order = [something(findfirst(==(Symbol("kin.", n)), names_c),
            findfirst(==(n), names_c)) for n in names_e]
        # Stan: the grouping column is the component's own index data, and the
        # carriers are the expansion's under the call prefix.
        stan_c = consumer_stan(component, "named-kinetics-component-" * label; mod=ScopedNamed)
        stan_e = consumer_stan(expanded, "named-kinetics-expanded-" * label; mod=ScopedNamed)
        stan_names_c = BridgeStan.param_unc_names(stan_c.model)
        stan_names_e = BridgeStan.param_unc_names(stan_e.model)
        stan_order = [something(findfirst(==("kin_" * n), stan_names_c),
            findfirst(==(n), stan_names_c)) for n in stan_names_e]
        sb_e = SBBRMI(expanded; mod=ScopedNamed, total_groups=())
        mapping = [pair.rk => pair.stan for pair in
            brm_coordinate_transport(rk_e, sb_e, stan_names_e).pairs]
        for point in (zeros(count), collect(range(-0.3, 0.4; length=count)))
            value_c, gradient_c = LogDensityProblems.logdensity_and_gradient(problem_c, point)
            value_e, gradient_e = LogDensityProblems.logdensity_and_gradient(problem_e,
                point[order])
            @test value_c ≈ value_e atol=2e-11 rtol=2e-11
            @test gradient_c[order] ≈ gradient_e atol=2e-10 rtol=2e-10
            gc, ge = similar(point), similar(point)
            sc, _ = BridgeStan.log_density_gradient!(stan_c.model, point, gc;
                propto=false, jacobian=true)
            se, _ = BridgeStan.log_density_gradient!(stan_e.model, point[stan_order], ge;
                propto=false, jacobian=true)
            @test sc ≈ se atol=2e-11 rtol=2e-11
            @test gc[stan_order] ≈ ge atol=2e-10 rtol=2e-10
            check_consumer_stan(problem_e, stan_e, mapping, rk_e, point[order])
        end
    end
end

@stestset "named component returns refuse unread or unknown fields" begin
    data = (; x=[-0.4, 0.2, 0.7], y1=[0.1, 0.4, 0.8], y2=[0.3, -0.1, 0.2])
    for backend in (model -> SBBRMI(model; mod=ScopedNamed, total_groups=()), RKBRMI)
        @test_throws r"reads field `mid`, but `bounds` returns only `lo`, `hi`" backend(
            ScopedNamed.unknown_field(data))
        @test_throws r"reads `k` as one value, .* read one field, such as `k.lo`" backend(
            ScopedNamed.bare_read(data))
        @test_throws r"inside a `@plate` cell a component call binds one value" backend(
            ScopedNamed.cell_call(data))
    end
    @test_throws r"a positional tuple has no field names" brm(:(function pairwise(x)
        a ~ Normal(0, 1)
        return (a, x)
    end); mod=ScopedNamed)
    @test_throws r"duplicate returned field" brm(:(function twice(x)
        a ~ Normal(0, 1)
        return (; a, a = x)
    end); mod=ScopedNamed)
    @test_throws r"bind `k.lo` at model scope, `lo = k.lo`, and read `lo\[i\]`" brm(quote
        k ~ bounds(x, scale)
        @plate for i in eachindex(y1)
            y1[i] ~ Normal(k.lo[i], 0.5)
        end
        scale ~ Exponential(1)
    end; mod=ScopedNamed)
end
