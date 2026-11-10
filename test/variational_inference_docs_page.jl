# Run with --project=test; no sampling or figure generation.
#
# The "Variational inference" docs page (docs/src/variational-inference.md) is built from
# research/variational_inference/svi.jl. This file evaluates every build-time `@eval`
# block of the page in order, in a fresh module, as Documenter does, and checks that
#   - the page passes the docs validators,
#   - each `source_function` / `source_code_region` marker still resolves,
#   - the one backend comparison carries a Stan pane that `stanc` accepts,
#   - every table and sentence generated from research/variational_inference/results/
#     renders,
# and that the model has a finite RK log density and gradient.

using Test, Markdown, Random
using BayesianRegressionModels, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems

include(joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

const VI_PAGE = joinpath(@__DIR__, "..", "docs", "src", "variational-inference.md")
const VI_SOURCE = joinpath(@__DIR__, "..", "research", "variational_inference", "svi.jl")

@testset "variational inference docs page: validators" begin
    @test BRMDocsComparisons.validate_no_bypasses([VI_PAGE]) === nothing
    @test BRMDocsComparisons.validate_generated_templates([VI_PAGE]) === nothing
end

@testset "variational inference docs page: build-time blocks evaluate in order" begin
    source = read(VI_PAGE, String)
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval[^\n]*\n(.*?)^```\s*$", source)]
    @test count(b -> occursin("Main.BRMDocsComparisons.comparison(", b), blocks) == 1
    page_module = Module(gensym(:VariationalInferenceDocs))
    Core.eval(page_module, :(using Markdown))
    for block in blocks
        rendered = nothing
        for expression in Meta.parseall(block).args
            expression isa LineNumberNode && continue
            rendered = Core.eval(page_module, expression)
        end
        if occursin("Main.BRMDocsComparisons.comparison(", block)
            @test occursin("require_stan=true", block)
            @test rendered isa Markdown.MD
            stan_panes = [pane for pane in rendered.content
                          if pane isa Markdown.Code && pane.language == "stan"]
            @test length(stan_panes) == 1
            @test StanBlocks.stanc_check(only(stan_panes).code; warn_pedantic=false).ok
        elseif occursin("source_code_region", block) || occursin("source_function", block)
            @test rendered isa Markdown.MD
            @test only(rendered.content) isa Markdown.Code
        elseif occursin("vi_markdown_table", block) || occursin("results_table", block)
            @test rendered isa Markdown.MD
            @test any(part -> part isa Markdown.Table, rendered.content)
        elseif occursin("Markdown.parse", block)
            @test rendered isa Markdown.MD
            @test any(part -> part isa Markdown.Paragraph, rendered.content)
        elseif occursin("BRMDocsComparisons.figure", block)
            @test rendered isa Markdown.MD
            @test occursin("data-brm-vega", sprint(show, MIME"text/markdown"(), rendered))
        end
    end
    @test count(b -> occursin("BRMDocsComparisons.figure", b), blocks) == 4
    @test !occursin("assets/variational-inference/", source)
end

@testset "site counts model: finite RK density and gradient" begin
    models = Module(gensym(:VariationalInferenceModels))
    Base.include(models, VI_SOURCE)                       # `main()` is guarded by PROGRAM_FILE
    rk = Base.invokelatest(() -> RKBRMI(models.site_counts_model(models.site_counts_data(; n_sites=20, visits=4))))
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    d = LogDensityProblems.dimension(problem)
    @test d == 2 + 2 + 1 + 2 * 20
    q = 0.1 .* randn(Xoshiro(1), d)
    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    @test isfinite(lp)
    @test all(isfinite, gradient)

    # Late sites must not gain unobserved early visits in the MCMC reference.
    data = Base.invokelatest(models.site_counts_data)
    windows = Base.invokelatest(models.stream_windows, data)
    seen = [Base.invokelatest(models.accumulated, windows, w) for w in 1:4]
    @test [sum(length, d.y) for d in seen] == [600, 1280, 2040, 2840]
    @test seen[end].y[end] == data.y[end][7:8]
    @test seen[end].y[1] == data.y[1]
end

@testset "NUTS references support the documented comparisons" begin
    include(joinpath(@__DIR__, "..", "research", "variational_inference", "tables.jl"))
    references = [vi_row("fits.tsv", r -> r.method == "NUTS"); vi_table("streaming_diagnostics.tsv")]
    for r in references
        @test r.chains >= 3
        @test r.max_rhat < 1.01
        @test r.min_bulk_ess >= 400
        @test r.min_tail_ess >= 400
        @test r.divergent == 0
        @test r.divergent_percent == 0
    end
end
