using Test, Markdown, BayesianRegressionModels, Distributions
import StanBlocks
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

@testset "four-pane comparisons emit one model through StanBlocks" begin
    sources = (
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 0.4, 0.7])
        model = (@brm begin
            sigma ~ LogNormal(-0.2, 0.3)
            mu ~ 1 + x
            effect(mu, x) ~ Cauchy(0, 0.5)
            y ~ Normal(mu, sigma)
        end)(data)
        """,
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[1.2, 0.8, 1.1])
        model = (@brm begin
            shape ~ LogNormal(0, 0.3)
            eta ~ 1 + x
            y ~ Gamma(shape, exp(eta))
        end)(data)
        """,
        """
        data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 0.4, 0.7], y2=[0.1, 0.8, 0.2])
        model = (@brm begin
            sigma ~ LogNormal(-0.2, 0.3)
            mu ~ 1 + x
            y ~ Normal(mu, sigma)
            y2 ~ Normal(mu, sigma)
        end)(data)
        """,
    )
    for (i, source) in pairs(sources)
        mod = BRMDocsComparisons.example_module(Symbol(:refactoring_comparison_, i))
        panes = BRMDocsComparisons.comparison(mod, source, :model; require_stan=true)
        @test length(panes.content) == 5
        @test getfield.(panes.content, :language) ==
              ["brm-comparison", "julia", "julia", "julia", "stan"]
        @test panes.content[2].code == strip(source, '\n')
        ir_source = panes.content[3].code
        # The IR pane prints the parsed statements, not the bound data columns.
        @test !occursin("data (eltype", ir_source)
        @test occursin("~", ir_source)
        # Resolved call heads print as written, never module-qualified: the
        # docs build's `Main` does not import `Distributions`, and Julia
        # qualifies bindings invisible from the display context.
        @test !occursin("Distributions.", ir_source)
        @test StanBlocks.stanc_check(panes.content[5].code).ok
    end
end

@testset "IR call heads print bare names" begin
    show_head(x) = (io = IOBuffer();
        BayesianRegressionModels._show_call_head(io, x);
        String(take!(io)))
    @test show_head(Distributions.Normal) == "Normal"
    @test show_head(Distributions.Exponential) == "Exponential"
    @test show_head(+) == "+"
end
