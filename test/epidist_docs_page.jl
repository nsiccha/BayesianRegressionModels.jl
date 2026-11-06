# Run with --project=test; no sampling.
#
# The "Epidemiological delays" docs page (docs/src/epidist.md) is built from
# research/epidist/epidist_page.jl and the checked-in results under
# research/epidist/results/. This file evaluates every build-time `@eval` block
# of the page in order, in a fresh module, as Documenter does, and checks that
# the source markers resolve, the one backend comparison carries a Stan pane
# `stanc` accepts, and the tables generated from the checked-in results render.

using Test, Markdown
using BayesianRegressionModels, StanBlocks

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const EPIDIST_PAGE = joinpath(@__DIR__, "..", "docs", "src", "epidist.md")

@testset "epidist docs page" begin
    @test BRMDocsComparisons.validate_no_bypasses([EPIDIST_PAGE]) === nothing
    @test BRMDocsComparisons.validate_generated_templates([EPIDIST_PAGE]) === nothing
    source = read(EPIDIST_PAGE, String)
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", source)]
    page_module = Module(gensym(:EpidistDocs))
    comparisons = tables = 0
    for block in blocks
        rendered = nothing
        for expression in Meta.parseall(block).args
            expression isa LineNumberNode && continue
            rendered = Core.eval(page_module, expression)
        end
        if occursin("Main.BRMDocsComparisons.comparison(", block)
            comparisons += 1
            @test rendered isa Markdown.MD
            stan_panes = [pane for pane in rendered.content
                          if pane isa Markdown.Code && pane.language == "stan"]
            @test length(stan_panes) == 1
            @test occursin("brm_censored_delay", only(stan_panes).code)
            @test StanBlocks.stanc_check(only(stan_panes).code; warn_pedantic=false).ok
        elseif occursin("Markdown.parse", block)
            tables += 1
            @test rendered isa Markdown.MD
            @test any(part -> part isa Markdown.Table, rendered.content)
        end
    end
    @test comparisons == 1
    @test tables == 2
end
