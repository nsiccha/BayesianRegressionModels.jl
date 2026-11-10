# Execute the worked example's build-time blocks without fitting it.
# julia --startup-file=no --project=test test/space_time_fields_docs_page.jl
using Test, Markdown, BayesianRegressionModels, StanBlocks
const DOCS_HELPER = get(ENV, "BRM_DOCS_COMPARISONS", joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))
# A temporary integration helper must still resolve this checkout's sources.
Base.include_string(Main, read(DOCS_HELPER, String), joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))
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
        elseif occursin("Main.BRMDocsComparisons.figure(", block)
            figures += 1
            @test occursin("data-brm-vega", sprint(show, MIME"text/plain"(), rendered))
        end
    end
    @test comparisons == 1
    @test figures == 3
end
