# Execute the spatial-fields example's build-time blocks without fitting it.
# julia --startup-file=no --project=test test/space_time_fields_docs_page.jl
#
# Every `@eval` block of docs/src/spatial-fields.md runs in order in one fresh
# module, as Documenter runs them. The page must pass the docs validators, its
# one backend comparison must carry a Stan pane that `stanc` accepts, every
# summary table under research/space_time_fields/results/ must render, and each
# figure must emit the `brm-figure` payload the page runtime draws.
using Test, Markdown, BayesianRegressionModels, StanBlocks
include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))
const SPATIAL_PAGE = joinpath(@__DIR__, "..", "docs", "src", "spatial-fields.md")

@testset "spatial example: model, summaries and figures" begin
    @test BRMDocsComparisons.validate_no_bypasses([SPATIAL_PAGE]) === nothing
    @test BRMDocsComparisons.validate_generated_templates([SPATIAL_PAGE]) === nothing
    page_module = Module(gensym(:SpatialDocs))
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", read(SPATIAL_PAGE, String))]
    comparisons = 0
    figures = 0
    for (index, block) in enumerate(blocks)
        println("Checking spatial example block ", index)
        flush(stdout)
        rendered = BRMDocsComparisons.evaluate_source(page_module, block)
        if occursin("Main.BRMDocsComparisons.comparison(", block)
            comparisons += 1
            @test rendered isa Markdown.MD
            stan = only([x.code for x in rendered.content if x isa Markdown.Code && x.language == "stan"])
            @test StanBlocks.stanc_check(stan; warn_pedantic=false).ok
        elseif occursin("Main.BRMDocsComparisons.results_table(", block)
            @test any(x -> x isa Markdown.Table, rendered.content)
        elseif occursin("Main.BRMDocsComparisons.source_code_region(", block)
            @test rendered isa Markdown.MD
            @test only(rendered.content) isa Markdown.Code
        elseif occursin("Main.BRMDocsComparisons.figure(", block)
            figures += 1
            # The page runtime reads a `brm-figure` fence holding the spec, caption and alt.
            marker = only(rendered.content)
            @test marker isa Markdown.Code && marker.language == "brm-figure"
            payload = BRMDocsComparisons.JSON.parse(marker.code)
            @test !isempty(payload["caption"]) && !isempty(payload["alt"]) &&
                  haskey(BRMDocsComparisons.JSON.parse(payload["spec"]), "\$schema")
        end
    end
    @test comparisons == 1
    @test figures == 3
end
