# The delay comparison now lives in research/epidist/; the renewal page links it.
# Keep checking its executable declaration and generated Stan without sampling.
using Test, Markdown
using BayesianRegressionModels, StanBlocks

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

@testset "epidist research declaration" begin
    mod = BRMDocsComparisons.example_module(:epidist_research)
    path = "research/epidist/epidist_page.jl"
    BRMDocsComparisons.evaluate_source_prelude(mod, path; before=:epidist_delay_model)
    rendered = BRMDocsComparisons.comparison(mod,
        BRMDocsComparisons.source_function(path, :epidist_delay_model),
        :epidist_delay_model; title="Interval-censored reporting delays", require_stan=true)
    @test rendered isa Markdown.MD
    stan = only([pane.code for pane in rendered.content
                 if pane isa Markdown.Code && pane.language == "stan"])
    @test occursin("brm_censored_delay", stan)
    @test StanBlocks.stanc_check(stan; warn_pedantic=false).ok
end
