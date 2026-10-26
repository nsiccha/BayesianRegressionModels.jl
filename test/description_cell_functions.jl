# Run: julia --startup-file=no --project=test test/description_cell_functions.jl
#
# Elementary functions a `@plate for` cell's statements use are described,
# not coverage gaps (snag brm-description-256b3b35, co-report from a zone
# model whose cell sums lognormal moments): `log1p` and `expm1` by their
# standard names, elementwise when broadcast, and StanBlocks'
# `reshape(v, m, n)` as the column-major fill `to_matrix` also describes.
using Test, BayesianRegressionModels, Distributions

const DATA = (; x=[[0.1, 0.4, 0.2, 0.3], [0.5, 0.2, 0.1, 0.6]], total=[3.0, 2.5])

@testset "a cell's log1p, expm1 and reshape are described" begin
    model = @brm DATA begin
        sigma ~ Exponential(1)
        @plate for i in eachindex(total)
            M = reshape(x[i], 2, 2)
            v = log1p.(x[i]) .+ expm1.(x[i])
            total[i] ~ Normal(sum(v) + sum(M), sigma)
        end
    end
    r = brm_description(SBBRMI(model))
    @test r.complete
    @test isempty(r.diagnostics)
    @test any(e -> occursin("\\operatorname{log1p}\\left(", e), r.equations)
    @test any(e -> occursin("\\operatorname{expm1}\\left(", e), r.equations)
    @test any(e -> occursin("\\operatorname{reshape}_{2\\times 2,\\mathrm{col}}\\left(", e), r.equations)
    @test any(p -> startswith(p, "Reshaping fills a matrix"), r.prose)
    # Broadcasting is elementwise application, not a keyword of the function.
    @test !any(e -> occursin("broadcast=true", e), r.equations)
    german = brm_description(SBBRMI(model); texts=description_texts(:de))
    @test german.complete
    @test any(p -> startswith(p, "Die Umformung füllt eine Matrix"), german.prose)
end
