using Test
using BayesianRegressionModels
using HTMXObjects
include("model_view_plain_text_checks.jl")

const BRM = BayesianRegressionModels
const ViewExt = Base.get_extension(BRM, :BayesianRegressionModelsHTMXObjectsExt)

module ModelViewFixture
using BayesianRegressionModels
const calls = Ref(0)
"""A documented toy response helper. Viewing it must not evaluate it."""
function toy_mean(x)
    calls[] += 1
    x
end
builder = @brm begin
    sigma ~ Exponential(1)
    beta ~ Normal(0, 1)
    weights ~ Dirichlet([1.0, 2.0, 3.0])
    theta[1:n] .~ Normal.(0, 1)
    mu = beta * x + toy_mean(x)
    matrix_read = sum(matrix_data)
    tensor_read = sum(tensor_data)
    y ~ Normal(mu, sigma)
end
data = (; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3], n=2,
    matrix_data=ones(2, 3), tensor_data=ones(2, 0, 3))
plate_builder = @brm begin
    @plate for i in eachindex(x)
        center = toy_mean.(x[i])
        y[i] ~ Normal(sum(center), 1)
    end
end
documented = @brm begin
    """Prior for the **response scale**."""
    sigma ~ Exponential(1)
    """The response uses the toy predictor without executing it for inspection."""
    y ~ Normal(toy_mean(x), sigma)
end

"""A reusable leaf with a documented local shift."""
@brm function leaf(x)
    """Independent shift in each leaf instance."""
    shift ~ Normal(0, 1)
    return toy_mean(x) + shift
end
"""A component that includes the leaf definition."""
@brm function nested(x)
    inner ~ leaf(x)
    return inner
end
repeated_components = @brm begin
    left ~ nested(x)
    right ~ nested(x)
    y ~ Normal(left + right, 1)
end

# The reporter's shape: a named returned field and one local prior.
@brm function named_calibration(x)
    """Calibration slope."""
    b ~ Normal(0, 1)
    value = b * x
    return (; value)
end
named_parent = @brm begin
    """Named calibration call."""
    left ~ named_calibration(x)
    y ~ Normal(left.value, 1)
end

"""A component that observes its own data input and owns a vector prior."""
@brm function observed(x, y, scale)
    z[1:length(x)] .~ Normal.(0, scale)
    value ~ leaf(x)
    y ~ Normal(value + z, scale)
    return value
end
observed_parent = @brm begin
    fit ~ observed(x, y_obs, scale)
    scale ~ Exponential(1)
end

# A local vector whose length only a prepared backend knows, beside a model-scope
# quantity of the same name and a different length.
@brm function vector_local(x)
    z[1:length(x)] .~ Normal.(0, 1)
    return x + z
end
vector_parent = @brm begin
    z[1:n] .~ Normal.(0, 1)
    left ~ vector_local(x)
    y ~ Normal(left .+ sum(z), 1)
end

# Grouped predictors with a shared block and prior addresses: inside a component
# and, for comparison, the same body at model scope.
@brm function kinetics(x, unit)
    log(recovery) ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    sd(recovery, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2.0)
    rate ~ Exponential(1)
    return (; recovery, decay)
end
kinetics_parent = @brm begin
    kin ~ kinetics(x, unit)
    y ~ Normal(kin.recovery + kin.decay, 0.5)
end
flat_kinetics = @brm begin
    log(recovery) ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    sd(recovery, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2.0)
    rate ~ Exponential(1)
    y ~ Normal(recovery + decay, 0.5)
end

@brm function cell_leaf(x, scale)
    a ~ Normal(0, scale)
    return x + a
end
plate_components = @brm begin
    @plate for i in eachindex(x)
        cell ~ cell_leaf(x[i], scale)
        location[i] = cell
        y[i] ~ Normal(location[i], 0.7)
    end
    scale ~ Exponential(1)
end
end

@testset "model view reads declarations without executing helpers" begin
    @test ViewExt !== nothing
    m = ModelViewFixture.builder(ModelViewFixture.data)
    facts = BRM._brm_view_facts(m)
    inputs = Dict(row.name => row for row in facts.inputs)
    quantities = Dict(row.name => row for row in facts.quantities)
    @test inputs[:y].role === :response_data
    @test inputs[:x].role === :data
    @test inputs[:n].dims == ()
    @test inputs[:matrix_data].rank == 2
    @test inputs[:matrix_data].dims == (2, 3)
    @test inputs[:tensor_data].rank == 3
    @test inputs[:tensor_data].dims == (2, 0, 3)
    @test quantities[:sigma].role === :parameter
    @test quantities[:sigma].rank == 0
    @test quantities[:weights].rank == 1
    @test quantities[:weights].dims === nothing
    @test quantities[:theta].rank == 1
    @test quantities[:mu].role === :assignment
    @test quantities[:mu].rank === nothing
    for mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, brm_model_view(m))
        @test occursin("matrix_data", rendered)
        @test occursin("Parameter", rendered)
        @test occursin("2 × 3", rendered)
        @test occursin("3-dimensional array", rendered)
        @test occursin("documented toy response helper", rendered)
        @test occursin("Not inferred", rendered)
    end
    @test ModelViewFixture.calls[] == 0
    text = test_model_view_plain_text(brm_model_view(m; title="Bound regression"),
        "Bound regression", ("Source", "Roles and ranks", "Documentation"))
    @test "sigma ~ Exponential(1)" in strip.(split(text, '\n'))
    html = repr("text/html", brm_model_view(m))
    @test occursin("<details", html)
    @test occursin("<summary", html)
    @test occursin("role=\"tablist\"", html)
    @test !occursin("Base.~", html)
    @test !occursin("Base.*", html)
    @test !occursin("text-overflow:ellipsis", html)
    @test occursin("Unbound declaration", repr("text/plain", brm_model_view(ModelViewFixture.builder)))
    @test ModelViewFixture.calls[] == 0
    plate_model = ModelViewFixture.plate_builder((;
        x=[[1., 2.], Float64[]], y=[0., 1.]))
    plate_html = repr("text/html", brm_model_view(plate_model))
    plate_facts = BRM._brm_view_facts(plate_model)
    @test only(row for row in plate_facts.inputs if row.name === :y).role === :response_data
    @test any(row -> row.role === :cell, plate_facts.quantities)
    @test occursin("documented toy response helper", plate_html)
    @test !occursin("Base.~", plate_html)
    @test ModelViewFixture.calls[] == 0
end

@testset "model view consumes shared authored documentation" begin
    builder = ModelViewFixture.documented
    model = builder(ModelViewFixture.data)
    records = brm_documentation(model)
    @test length(records) == 2
    @test ModelViewFixture.calls[] == 0
    for input in (builder, model)
        view = brm_model_view(input)
        tabs = input === builder ? ("Source", "Potential data inputs", "Documentation") :
            ("Source", "Roles and ranks", "Documentation")
        test_model_view_plain_text(view, "Model", tabs)
        html = repr("text/html", view)
        @test occursin("Authored documentation", html)
        @test occursin("<strong>response scale</strong>", html)
        @test occursin("Declaration and provenance", html)
        for mime in ("text/markdown", "text/plain")
            text = repr(mime, view)
            @test occursin("Prior for the **response scale**.", text)
            @test occursin("(:sigma,)", text)
            @test occursin(string(first(records).file), text)
        end
    end
    @test [(d.path, d.markdown, d.statement, d.file, d.line) for d in brm_documentation(model)] ==
          [(d.path, d.markdown, d.statement, d.file, d.line) for d in records]
    @test ModelViewFixture.calls[] == 0
end

@testset "model view includes shared definitions and all component instances" begin
    model = ModelViewFixture.repeated_components(ModelViewFixture.data)
    facts = BRM._brm_view_facts(model)
    @test Set(record.path for record in facts.components) ==
        Set(((:left,), (:left, :inner), (:right,), (:right, :inner)))
    # Each call lists its own declarations under its exact path; the generated
    # result locals that carry a call's returned value are not declarations.
    rows = Dict(row.path => row for row in facts.quantities)
    @test Set(keys(rows)) == Set(((:left,), (:left, :inner), (:left, :inner, :shift),
        (:right,), (:right, :inner), (:right, :inner, :shift)))
    @test all(rows[path].role === :component for path in
        ((:left,), (:left, :inner), (:right,), (:right, :inner)))
    for path in ((:left, :inner, :shift), (:right, :inner, :shift))
        @test rows[path].role === :parameter
        @test rows[path].rank == 0
        @test rows[path].dims == ()
        @test !rows[path].plate
    end
    definitions = ViewExt._component_definitions(facts.components)
    @test length(definitions) == 2
    @test Set(record.definition for record in definitions) ==
        Set((ModelViewFixture.leaf, ModelViewFixture.nested))
    @test all(record -> length(record.instances) == 2, definitions)
    for record in definitions
        original = deepcopy(record.source)
        source = ViewExt._source_text(record.source)
        @test !occursin("Core.@doc", source)
        @test !occursin("#=", source)
        @test record.source == original
        @test Meta.parse(source) isa Expr
    end
    @test ModelViewFixture.calls[] == 0
    for input in (model, ModelViewFixture.nested), mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, brm_model_view(input))
        @test occursin("function leaf(x)", rendered)
        @test occursin("function nested(x)", rendered)
        @test occursin("Independent shift in each leaf instance", rendered)
        @test occursin("Call instances", rendered)
        @test !occursin("Core.@doc", rendered)
    end
    markdown = repr("text/markdown", brm_model_view(model))
    text = test_model_view_plain_text(brm_model_view(model), "Model",
        ("Source", "Roles and ranks", "Documentation", "Model",
            string(ModelViewFixture, ".leaf"), string(ModelViewFixture, ".nested")))
    @test "shift ~ Normal(0, 1)" in strip.(split(text, '\n'))
    @test occursin("(:left, :inner)", text)
    @test occursin("(:right, :inner)", text)
    @test occursin("(:left, :inner)", markdown)
    @test occursin("(:right, :inner)", markdown)
    @test occursin("Derived · component result", markdown)
    @test occursin("documented toy response helper", markdown)
    @test occursin("(:left, :inner, :shift)", markdown)
    @test occursin("(:right, :inner, :shift)", markdown)
    @test occursin("| left › inner › shift | Parameter | 0 | Scalar |", markdown)
    @test occursin("| right › inner › shift | Parameter | 0 | Scalar |", markdown)
    @test ModelViewFixture.calls[] == 0
end

@testset "bound model view lists component-local declarations under exact call paths" begin
    data = (; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3])
    model = ModelViewFixture.named_parent(data)
    facts = BRM._brm_view_facts(model)
    rows = Dict(row.path => row for row in facts.quantities)
    @test Set(keys(rows)) == Set(((:left,), (:left, :b), (:left, :value)))
    @test rows[(:left,)].role === :component
    @test rows[(:left, :b)].role === :parameter
    @test rows[(:left, :b)].rank == 0
    @test rows[(:left, :b)].dims == ()
    @test rows[(:left, :value)].role === :assignment
    @test rows[(:left, :value)].rank === nothing
    inputs = Dict(row.name => row.role for row in facts.inputs)
    @test inputs == Dict(:x => :data, :y => :response_data)
    for mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, brm_model_view(model))
        @test occursin("left › b", rendered)
        @test occursin("left › value", rendered)
        @test !occursin("component_result", rendered)
    end
    markdown = repr("text/markdown", brm_model_view(model))
    @test occursin("| left › b | Parameter | 0 | Scalar |", markdown)
    @test occursin("| left › value | Derived · assignment | Unknown | Not inferred |", markdown)
    @test occursin("| left | Derived · component result | Unknown | Not inferred |", markdown)

    # A local observation of a data input marks the caller's column, not the
    # component's input name, as response data; a local vector prior keeps rank 1.
    observed = ModelViewFixture.observed_parent((; x=[1., 2.], y_obs=[0.1, 0.2]))
    facts = BRM._brm_view_facts(observed)
    inputs = Dict(row.name => row.role for row in facts.inputs)
    @test inputs == Dict(:x => :data, :y_obs => :response_data)
    rows = Dict(row.path => row for row in facts.quantities)
    @test Set(keys(rows)) == Set(((:scale,), (:fit,), (:fit, :z), (:fit, :value),
        (:fit, :value, :shift)))
    @test rows[(:fit, :z)].role === :parameter
    @test rows[(:fit, :z)].rank == 1
    @test rows[(:fit, :value)].role === :component
    @test rows[(:fit, :value, :shift)].rank == 0

    # A component's local predictors, block and prior addresses get exactly the
    # rows the same declarations get at model scope, under the call path.
    grouped = (; x=[0., 1., 2., 3.], unit=["a", "a", "b", "b"], y=[0.1, 0.2, 0.3, 0.4])
    scoped = BRM._brm_view_facts(ModelViewFixture.kinetics_parent(grouped))
    flat = BRM._brm_view_facts(ModelViewFixture.flat_kinetics(grouped))
    signature(row, path) = (path, row.role, row.rank, row.dims)
    locals = sort([signature(row, row.path[2:end]) for row in scoped.quantities
        if length(row.path) > 1 && first(row.path) === :kin]; by=string)
    expected = sort([signature(row, row.path) for row in flat.quantities]; by=string)
    @test !isempty(expected)
    @test locals == expected
    @test Set(row.role for row in flat.quantities) ⊇ Set((:parameter, :predictor, :prior_modifier))
    @test Dict(row.name => row.role for row in scoped.inputs) ==
        Dict(:x => :data, :unit => :data, :y => :response_data)

    # A component called inside a plate cell declares one set of locals per
    # iteration; the table says so rather than presenting a whole-model shape.
    plate = ModelViewFixture.plate_components((; x=[1., 2., 3.], y=[0.1, 0.2, 0.3]))
    facts = BRM._brm_view_facts(plate)
    calls = [record.path for record in facts.components]
    @test length(calls) == 1
    call = only(calls)
    rows = Dict(row.path => row for row in facts.quantities)
    @test rows[call].role === :component
    @test rows[call].plate
    local_a = rows[(call..., :a)]
    @test local_a.role === :parameter
    @test local_a.plate
    @test local_a.rank == 0
    @test occursin("once per plate iteration",
        repr("text/markdown", brm_model_view(plate)))
    @test ModelViewFixture.calls[] == 0
end

@testset "descriptor view resolves component-local carriers by exact call" begin
    data = (; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3], n=2)
    model = ModelViewFixture.vector_parent(data)
    unprepared = repr("text/markdown", brm_model_view(model))
    @test occursin("| z | Parameter | 1 | Vector · dimensions not bound |", unprepared)
    @test occursin("| left › z | Parameter | 1 | Vector · dimensions not bound |", unprepared)
    d = brm_descriptor(ModelViewFixture.vector_parent, data)
    @test brm_output(d, :z; submodel=:left).dims == (3,)
    @test brm_output(d, :z).dims == (2,)
    md = repr("text/markdown", brm_model_view(d))
    @test occursin("| z | Parameter | 1 | Vector · 2 |", md)
    @test occursin("| left › z | Parameter | 1 | Vector · 3 |", md)
end

@testset "descriptor view retains carrier roles and concrete shapes" begin
    builder = @brm begin
        p ~ Dirichlet([1., 2., 3.])
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu + sum(p), sigma)
    end
    data = (; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3])
    d = brm_descriptor(builder, data)
    p = brm_output(d, :p)
    @test p.dims == (3,)
    md = repr("text/markdown", brm_model_view(d))
    @test occursin("Vector · 3", md)
    @test occursin("posterior predictive", md)
    @test occursin("pointwise loglik", md)
    @test occursin("Internal carrier", md)
    @test occursin("not the number of free parameters", md)
    @test all(o -> occursin(string(o.name), md), d.outputs)
    text = test_model_view_plain_text(brm_model_view(d; title="Prepared regression"),
        "Prepared regression", ("Source", "Roles and ranks", "Documentation"))
    @test occursin("Vector · 3", text)
    @test occursin("Carrier details", text)
    @test d.formula == string(d.plan.parent)
    @test data.x == [-1., 0., 1.]
    if haskey(ENV, "BRM_MODEL_VIEW_HTML")
        page = htmx(HTMXObjects.h.main(
            brm_model_view(d; title="Regression with simplex weights"); class="container");
            pico_version="2", extra_head=(htmxo_syntax_head()...,))
        write(ENV["BRM_MODEL_VIEW_HTML"], repr("text/html", page))
    end
end
