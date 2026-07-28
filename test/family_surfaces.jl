# test/family_surfaces.jl — executable contracts for custom likelihood surfaces.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/family_surfaces.jl

using Test
using BayesianRegressionModels
using Distributions: LocationScale, TDist
using StanBlocks

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y_zip=[0, 1, 0, 2, 3, 0],
    y_nb=[0, 1, 2, 4, 3, 6],
    y_t=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
)

family_builder = @brm begin
    log(lambda) ~ 1 + x
    y_zip ~ ZeroInflatedPoisson(lambda, 0.25)

    log(mu) ~ 1 + x
    log(phi) ~ 1
    y_nb ~ NegativeBinomial2(mu, phi)

    loc ~ 1 + x
    log(scale) ~ 1
    y_t ~ LocationScale(loc, scale, TDist(4.0))
end

@testset "SBBRMI lowers density, pointwise log-lik and RNG paths" begin
    plan = generative_plan(family_builder, df; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(plan)

    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("zero_inflated_poisson(", code)
    @test occursin("neg_binomial_2(", code)
    @test occursin("student_t(", code)

    for target in (:y_zip, :y_nb, :y_t)
        declaration = only(d for d in plan.declarations if d.target === target)
        @test declaration.role === :observation
        @test !isnothing(declaration.draw)
        @test occursin(string(declaration.draw), code)
        @test occursin(string(target, "_likelihood"), code)
    end

    families = Dict(d.target => d.family for d in plan.declarations
                    if d.role === :observation)
    @test families[:y_zip] === :zero_inflated_poisson
    @test families[:y_nb] === :neg_binomial_2
    @test families[:y_t] === :student_t
end
