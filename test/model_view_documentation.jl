using Test, Markdown, BayesianRegressionModels, HTMXObjects

module DocumentationViewFixture
using BayesianRegressionModels

"""
Callable reference to [`sum`](@ref) and [a *named* target](@ref missing-target).
The [ordinary destination](https://example.com/manual) stays linked.

```jldoctest documentation-view
julia> 1 + 1
2
```
"""
documented_helper(x) = error("inspection must not execute documented_helper")

builder = @brm begin
    """Authored reference to [`Gamma`](@ref) and [another *target*](@ref missing-heading).
    The [authored destination](https://example.com/authored) stays linked.
    """
    sigma ~ Gamma(2, 1)
    total = sum(x) + documented_helper(x)
    y ~ Normal(total, sigma)
end
end

# Inspect the projected Markdown tree: references inside the retained source
# code are intentionally literal and must not be mistaken for navigable links.
function documentation_links(x)
    urls = String[]
    function visit(node)
        if node isa Markdown.Link
            push!(urls, node.url)
        elseif node isa Markdown.Code
            return
        end
        if node isa AbstractArray || node isa Tuple
            foreach(visit, node)
        elseif parentmodule(typeof(node)) === Markdown
            foreach(name -> visit(getfield(node, name)), fieldnames(typeof(node)))
        end
    end
    visit(x)
    urls
end

@testset "model-view documentation exports real links and readable references" begin
    @test Base.get_extension(BayesianRegressionModels, :BayesianRegressionModelsHTMXObjectsExt) !== nothing
    builder = DocumentationViewFixture.builder
    model = builder((; x=[1., 2.], y=[0., 1.]))
    original_source = string(builder)
    records = brm_documentation(model)
    original_records = deepcopy(records)
    callable_doc = @doc(DocumentationViewFixture.documented_helper)
    original_callable_doc = string(callable_doc)
    @test occursin("@ref", original_callable_doc)
    @test occursin("@ref", only(records).markdown)

    for input in (builder, model)
        view = brm_model_view(input)
        html = repr("text/html", view)
        markdown = repr("text/markdown", view)
        @test !occursin(r"href=\"[^\"]*@ref", html)
        @test occursin("Authored reference to <code>Gamma</code>", html)
        @test occursin("another <em>target</em>", html)
        @test occursin("language-julia", html)
        @test !occursin("language-jldoctest", html)
        @test occursin("Authored reference to `Gamma`", markdown)
        urls = documentation_links(Markdown.parse(markdown))
        @test "https://example.com/authored" in urls
        @test !any(url -> startswith(url, "@ref"), urls)
        # Callable identities are available after data binding. Unbound
        # builders expose their retained authored documentation only.
        if input === model
            @test occursin("Callable reference to <code>sum</code>", html)
            @test occursin("a <em>named</em> target", html)
            @test occursin("Callable reference to `sum`", markdown)
            @test occursin("julia> 1 + 1\n2", markdown)
            @test "https://example.com/manual" in urls
            @test occursin("Base.sum", html)
            @test occursin("Distributions.Gamma", html)
        else
            @test !occursin("Callable documentation", html)
        end
    end
    after = brm_documentation(model)
    @test length(after) == length(original_records)
    for (actual, expected) in zip(after, original_records), field in fieldnames(typeof(expected))
        @test getfield(actual, field) == getfield(expected, field)
    end
    @test string(callable_doc) == original_callable_doc
    @test occursin("@ref", original_source)
    @test string(builder) == original_source
end
