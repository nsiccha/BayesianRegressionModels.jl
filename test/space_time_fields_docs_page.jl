# Execute the spatial page's build-time blocks without fitting the example.
# julia --startup-file=no --project=test test/space_time_fields_docs_page.jl
using Test, Markdown, JSON, BayesianRegressionModels, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, TreeArrays, Statistics
include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const SPATIAL_PAGE = joinpath(@__DIR__, "..", "docs", "src", "space-time-fields.md")

@testset "spatial page: generated models and result tables" begin
    @test BRMDocsComparisons.validate_no_bypasses([SPATIAL_PAGE]) === nothing
    @test BRMDocsComparisons.validate_generated_templates([SPATIAL_PAGE]) === nothing
    page_module = Module(gensym(:SpatialDocs))
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", read(SPATIAL_PAGE, String))]
    comparisons = 0
    for (index, block) in enumerate(blocks)
        println("Checking spatial page block ", index)
        flush(stdout)
        rendered = BRMDocsComparisons.evaluate_source(page_module, block)
        if occursin("Main.BRMDocsComparisons.comparison(", block)
            comparisons += 1
            @test rendered isa Markdown.MD
            stan = only([x.code for x in rendered.content if x isa Markdown.Code && x.language == "stan"])
            @test StanBlocks.stanc_check(stan; warn_pedantic=false).ok
        elseif occursin("Markdown.parse", block)
            @test any(x -> x isa Markdown.Table, rendered.content)
        end
    end
    @test comparisons == 5

    @testset "documented replay forms" begin
        for (name, design, output) in (
            (:area_counts, (; area=["island", "b", "a"], log_expected=log.([10.0, 20.0, 12.0])), :rate),
            (:channel_surface, (; east=[20.0, 70.0, 20.0], north=[30.0, 60.0, 30.0], channel=["a", "b", "a"]), :mu),
        )
            # The actual model displayed by the page, not a duplicate declaration.
            brmi = getproperty(page_module, name)
            rk = Base.invokelatest(RKBRMI, brmi)
            u = Base.invokelatest(brm_start, rk, (;); base=:prior_median)
            u .+= 0.07 .* sin.(eachindex(u)) # exercise nonzero field weights
            replay = Base.invokelatest(reprocess, rk, design)
            values = Base.invokelatest(rk_draw_values, replay, reshape(u, :, 1), (output,))
            actual = only(getproperty(values, output))
            @test length(actual) == 3
            @test all(isfinite, actual)
            name == :area_counts && @test all(>(0), actual)
            name == :channel_surface && @test actual[1] == actual[3]
        end
    end
end
