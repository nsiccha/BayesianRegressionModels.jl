# test/linear_predictor_entries.jl — a `linear_predictors(brmi)` entry is a
# predictor address.
#
# Run: julia --startup-file=no --project=. test/linear_predictor_entries.jl
#
# `linear_predictors(brmi)` returns `(; name, link_lhs_fn)` records, and it lists
# declared parameters and `kernel(...)` bindings beside the population formulas.
# brm-use documents a scan over it with `popcoefnames`; passing an entry on raised
# `MethodError: no method matching popcoefnames(::BRMI, ::@NamedTuple{name::Symbol,
# link_lhs_fn::…})` while every helper took only a Symbol (snag
# scanned-linear-p-5415909b). Every helper that takes a linear predictor's name
# now also takes the entry and reads it as its `name`. No backend is built: the
# helpers read the data-bound BRMI only.

using Test
using BayesianRegressionModels

const LPE_DATA = (;
    y = [0.3, -1.2, 0.8, 1.9, -0.4, 0.1, 1.1, -0.7, 0.5, 2.0, -1.5, 0.9],
    z = [1.3, 0.2, 0.9, 2.4, 0.6, 1.1, 1.7, 0.4, 1.0, 2.8, 0.3, 1.5],
    x = collect(range(-1.0, 1.0; length = 12)),
    w = collect(range(0.0, 2.0; length = 12)),
    g = repeat(["a", "b", "c"], 4),
)

lpe_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    log(spread) ~ 1 + w
    y ~ Normal(mu, sigma)
    z ~ Normal(mu, spread)
end

@testset "a linear_predictors entry is a predictor address" begin
    brmi = lpe_model(LPE_DATA)
    entries = linear_predictors(brmi)
    entry(n) = only(filter(lp -> lp.name === n, entries))

    @testset "the entries: formulas, a linked formula and a declared parameter" begin
        @test issubset([:sigma, :mu, :spread], [lp.name for lp in entries])
        @test entry(:mu).link_lhs_fn === identity
        @test entry(:spread).link_lhs_fn === log
        @test entry(:sigma).link_lhs_fn === identity
    end

    @testset "popcoefnames: the documented scan needs no filtering" begin
        labels = Dict(lp.name => popcoefnames(brmi, lp) for lp in linear_predictors(brmi))
        @test labels[:mu] == [:Intercept, :x]
        # The bare public name of a linked predictor, as for `popcoefnames(brmi, :spread)`.
        @test labels[:spread] == [:Intercept, :w]
        # A declared parameter is not a population formula.
        @test isnothing(labels[:sigma])
        for lp in entries
            @test isequal(popcoefnames(brmi, lp), popcoefnames(brmi, lp.name))
        end
    end

    @testset "linear_predictor_op and dependencies take every entry" begin
        for lp in entries
            @test linear_predictor_op(brmi, lp) === linear_predictor_op(brmi, lp.name)
            @test linear_predictor_op(brmi, lp) !== nothing
            @test isequal(dependencies(brmi, lp), dependencies(brmi, lp.name))
        end
        @test dependencies(brmi, entry(:mu)).data == [:x, :g]
    end

    @testset "predictors and grouping_factors take a formula entry" begin
        for n in (:mu, :spread)
            lp = entry(n)
            @test isequal(predictors(brmi, lp), predictors(brmi, n))
            @test grouping_factors(brmi, lp) == grouping_factors(brmi, n)
        end
        @test predictors(brmi, entry(:mu)).continuous == [:x]
        @test predictors(brmi, entry(:spread)).continuous == [:w]
        @test grouping_factors(brmi, entry(:mu)) == [:g]
        @test isempty(grouping_factors(brmi, entry(:spread)))
    end

    @testset "a NamedTuple of another shape is not an address" begin
        @test_throws MethodError popcoefnames(brmi, (; name = :mu))
        @test_throws MethodError predictors(brmi, (; name = :mu, link = identity))
    end
end
