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
    @test all(row -> row.role === :component, facts.quantities)
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
    @test ModelViewFixture.calls[] == 0
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
