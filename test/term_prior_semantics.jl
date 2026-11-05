# test/term_prior_semantics.jl — term-prior addresses are validated by the
# shared preparation layer: a misspelled address, a slot the term does not
# have, or an ambiguous term refuse on both backends, and a valid address
# resolves to its most specific statement.
#
# Run: julia --project=test test/term_prior_semantics.jl
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

const prior_data = (; x=collect(range(-1., 1.; length=12)),
                     grade=repeat(1:4, 3), y=zeros(12))

@testset "both backends account for every term-prior address" begin
    wrong_address = @brm begin
            sd(mu, s(misspelled)) ~ Exponential(3)
            mu ~ 1 + s(x)
            y ~ Normal(mu, 1)
    end
    wrong_slot = @brm begin
            simplex(mu, s(x)) ~ Dirichlet(1)
            mu ~ 1 + s(x)
            y ~ Normal(mu, 1)
    end
    ambiguous = @brm begin
            sd(mu, s(x)) ~ Exponential(3)
            mu ~ 1 + s(x) + s(x)
            y ~ Normal(mu, 1)
    end
    builders = (wrong_address, wrong_slot, ambiguous)
    for builder in builders, backend in (SBBRMI, RKBRMI)
        @test_throws ErrorException backend(builder(prior_data))
    end
    valid = (@brm begin
        sd(:, s(x)) ~ Exponential(3)
        sd(mu, s(x)) ~ Exponential(2)
        mu ~ 1 + s(x)
        y ~ Normal(mu, 1)
    end)(prior_data)
    context = BRM._brm_backend_context(valid)
    slot = context.term_priors[:mu][Symbol("s(x)")][:sd]
    @test getargs(slot.spec.expression) == (2,)
end
