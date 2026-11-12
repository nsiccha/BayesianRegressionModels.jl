# The state-space and wastewater docs pages read their declarations from the
# research files by text markers and evaluate the files' helper preludes.
# Evaluate, in page order and as Documenter does, every `@eval` block that reads
# its page's research source, so a moved marker or a broken prelude fails here
# and not only in the Docs workflow. No sampling.
# Run: julia --project=test test/docs_model_pages.jl
using Test, Markdown
using BayesianRegressionModels, StanBlocks

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const DOCS_MODEL_PAGES = [
    "state-space.md" => "research/ema_ctsem/docs_model.jl",
    "wastewater.md" => "research/wastewater/cdc_ww_inference.jl",
]

function source_eval_blocks(page, source)
    text = read(joinpath(@__DIR__, "..", "docs", "src", page), String)
    blocks = [m.captures[1] for m in eachmatch(r"^```@eval\n(.*?)^```$"ms, text)]
    filter(block -> occursin(source, block), blocks)
end

for (page, source) in DOCS_MODEL_PAGES
    @testset "docs page $page: blocks reading $source" begin
        blocks = source_eval_blocks(page, source)
        # The declaration region, the shared prelude and the backend comparison.
        @test length(blocks) >= 2
        rendered = map(blocks) do block
            Core.eval(Module(), Meta.parseall(block; filename=page))
        end
        @test last(rendered) isa Markdown.MD
        stan = [pane.code for pane in last(rendered).content
                if pane isa Markdown.Code && pane.language == "stan"]
        @test length(stan) == 1
        @test StanBlocks.stanc_check(only(stan); warn_pedantic=false).ok
        # These models' JAX modules need ReactiveKernels 0.1.180 (test/README.md);
        # on it the page's JAX pane holds the module, not the emission error.
        if pkgversion(BRMDocsComparisons.ReactiveKernels) >= v"0.1.180"
            panes = [pane.code for pane in last(rendered).content if pane isa Markdown.Code]
            @test !any(code -> occursin("no JAX module yet", code), panes)
            @test any(code -> occursin("def log_density(", code), panes)
        end
    end
end
