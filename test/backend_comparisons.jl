using Test, Markdown, BayesianRegressionModels, Distributions
import StanBlocks
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "docs", "backend_comparisons.jl"))

@testset "shared comparisons emit StanBlocks, RKPPL and JAX from one model" begin
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
        @test length(panes.content) == 8
        @test getfield.(panes.content, :language) ==
              ["brm-comparison", "julia", "julia", "julia", "stan", "julia", "python", "python"]
        @test endswith(panes.content[1].code, "panes: brm|stanblocks|rkppl|stan|jax:3")
        @test panes.content[2].code == strip(source, '\n')
        rkppl_source = panes.content[4].code
        @test occursin("~", rkppl_source)
        @test occursin("y", rkppl_source)
        @test !occursin("#=", rkppl_source)
        @test StanBlocks.stanc_check(panes.content[5].code).ok
        @test occursin("def prepare_data", panes.content[8].code)
        @test occursin("def log_density", panes.content[8].code)
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
