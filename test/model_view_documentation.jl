using Test, Markdown, BayesianRegressionModels, HTMXObjects
include("model_view_plain_text_checks.jl")

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

# A callable whose docstring has its own outline, as `Distributions.censored`'s
# does, and skips a level in it (`#` then `###`).
"""
    outlined_helper(x)

# Usage

Outlined usage text.

### Arguments

Outlined argument text.

# Implementation

Outlined implementation text.
"""
outlined_helper(x) = error("inspection must not execute outlined_helper")

outlined = @brm begin
    sigma ~ Exponential(1)
    y ~ Normal(outlined_helper(x), sigma)
end
end

function heading_levels(markdown::AbstractString)
    levels = Pair{Int,String}[]
    for node in Markdown.parse(markdown).content
        node isa Markdown.Header || continue
        push!(levels, typeof(node).parameters[1] => strip(sprint(Markdown.plaininline, node.text)))
    end
    levels
end
html_heading_levels(html) = [parse(Int, m.captures[1]) => strip(replace(m.captures[2], r"<[^>]+>" => ""))
    for m in eachmatch(r"<h([1-6])[^>]*>(.*?)</h\1>"s, html)]
consecutive(levels) = all(i -> first(levels[i]) <= first(levels[i - 1]) + 1, 2:length(levels))

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
        tabs = input === builder ? ("Source", "Potential data inputs", "Documentation") :
            ("Source", "Roles and ranks", "Documentation")
        plain = test_model_view_plain_text(view, "Model", tabs)
        @test occursin("Authored reference to `Gamma`", plain)
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
            @test occursin("Callable reference to `sum`", plain)
            @test occursin("julia> 1 + 1\n2", plain)
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

# Snag brm-model-view-c-623a1e28: a callable docstring's own `#` headings stayed
# at level 1 under nested sections, so `pdf_report` failed PDF/UA-1 ("skipped
# from heading level 1 to 4"). They open below the disclosure that holds them.
@testset "callable docstring headings follow the enclosing sections" begin
    model = DocumentationViewFixture.outlined((; x=[1., 2.], y=[0., 1.]))
    view = brm_model_view(model)
    one_section = SemanticSection("Model specification", view)
    two_sections = SemanticSection("Model and workflow", one_section)
    for (name, content) in ("one enclosing section" => one_section,
            "two enclosing sections" => two_sections)
        projections = ("markdown" => heading_levels(HTMXObjects.to_markdown_string(content)),
            "html" => html_heading_levels(repr("text/html", content)))
        for (projection, levels) in projections
            @testset "$name, $projection" begin
                @test last(first(levels)) == (content === one_section ?
                    "Model specification" : "Model and workflow")
                @test consecutive(levels)
                @test all(level -> 1 <= first(level) <= 6, levels)
                titles = last.(levels)
                callable = findfirst(==("Callable documentation"), titles)
                @test callable !== nothing
                @test titles[callable+1:callable+3] == ["Usage", "Arguments", "Implementation"]
                level = first(levels[callable])
                @test first.(levels[callable+1:callable+3]) ==
                    [level + 1, min(level + 2, 6), level + 1]
            end
        end
    end
    # The docstring keeps its text; only its heading levels are placed.
    markdown = HTMXObjects.to_markdown_string(two_sections)
    for text in ("Outlined usage text.", "Outlined argument text.", "Outlined implementation text.")
        @test occursin(text, markdown)
    end
    # The callable's own documentation is not rewritten by viewing it.
    @test occursin(r"^# Usage$"m, string(@doc(DocumentationViewFixture.outlined_helper)))
end
