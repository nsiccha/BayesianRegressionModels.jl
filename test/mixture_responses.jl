# test/mixture_responses.jl — `MixtureModel` responses on the StanBlocks path:
# continuous and discrete mixtures lower exactly (density, gradient,
# pointwise log-likelihood and predictive draws against Distributions), and
# the mixture contract refuses incompatible inputs by name.
#
# Run: julia --project=test test/mixture_responses.jl
using Test, BayesianRegressionModels, Distributions, Random, Statistics
using LogDensityProblems, LogExpFunctions
import StanBlocks
const BRM = BayesianRegressionModels

@testset "mixture lowering is exact on every Stan path" begin
    data = (; y=[-2.0, -1.8, 1.9, 2.2])
    builder = @brm begin
        mu1 ~ Normal(-2, 0.1)
        mu2 ~ Normal(2, 0.1)
        log(sigma) ~ 1
        y ~ MixtureModel([
            Normal(mu1, exp(log(sigma))),
            Normal(mu2, exp(log(sigma))),
        ], [0.4, 0.6])
    end
    descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
    @test StanBlocks.stanc_check(
        BRM.stan_code(descriptor.plan); warn_pedantic=false).ok
    cache = joinpath(tempdir(), "brm-mixture-response")
    isdir(cache) || mkpath(cache)
    problem = brm_execute(descriptor, :instantiate;
        path=joinpath(cache, string(descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 3
    # Declaration order is the constrained-coordinate order.
    q = [-2.0, 2.0, log(0.3)]
    _, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    reference(v) = sum(logpdf.(MixtureModel(
        [Normal(v[1], exp(v[3])), Normal(v[2], exp(v[3]))], [0.4, 0.6]), data.y)) +
        logpdf(Normal(-2, 0.1), v[1]) + logpdf(Normal(2, 0.1), v[2]) +
        logpdf(Normal(), v[3])
    h = 1e-6
    expected_gradient = [(reference(q .+ h .* basis) - reference(q .- h .* basis)) / 2h
                         for basis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])]
    @test gradient ≈ expected_gradient atol=1e-5
    mixture = MixtureModel([Normal(-2.0, 0.3), Normal(2.0, 0.3)], [0.4, 0.6])
    pointwise = brm_execute(descriptor, :pointwise_loglik;
        problem, draws=q, seed=20260913)
    @test pointwise.y_likelihood ≈ logpdf.(mixture, data.y) atol=1e-10
    predictions = brm_execute(descriptor, :predict;
        problem, draws=repeat(q, 1, 4096), seed=20260913)
    draws = vec(predictions.y_gen)
    @test length(draws) == 4 * 4096
    @test mean(draws) ≈ mean(mixture) atol=0.1

    plan = generative_plan(builder, data; mod=@__MODULE__)
    declaration = only(d for d in plan.declarations if d.target === :y)
    @test declaration.role === :observation
    @test declaration.draw === :y_gen
    # A generated family is a value since d10effe8 (concurrent model
    # generation), not a module function.
    @test declaration.family isa StanBlocks.ValueFamily
    @test occursin("brm_mixture_", string(declaration.family))

    sb = SBBRMI(builder(data); mod=@__MODULE__)
    replayed = reprocess(sb, (; y=[0.1, 0.2]))
    @test replayed.data[:y_mixture_weights] == [0.4, 0.6]
    @test stan_code(replayed) == stan_code(sb)
end

@testset "discrete mixtures lower exactly on the Stan path" begin
    poisson_data = (; y=[0, 1, 3, 5, 2])
    poisson_builder = @brm begin
        lambda1 ~ Exponential(1)
        lambda2 ~ Exponential(1)
        y ~ MixtureModel([Poisson(lambda1), Poisson(lambda2)], [0.3, 0.7])
    end
    poisson_mixture = MixtureModel([Poisson(1.5), Poisson(4.0)], [0.3, 0.7])
    poisson_expected = logpdf.(poisson_mixture, poisson_data.y)
    poisson_code = stan_code(SBBRMI(poisson_builder(poisson_data)))
    @test occursin("poisson_lpmf", poisson_code)
    @test occursin("poisson_rng", poisson_code)
    @test StanBlocks.stanc_check(poisson_code; warn_pedantic=false).ok
    poisson_descriptor = brm_descriptor(
        poisson_builder, poisson_data; mod=@__MODULE__, highlights=())
    cache = joinpath(tempdir(), "brm-mixture-response")
    isdir(cache) || mkpath(cache)
    poisson_problem = brm_execute(poisson_descriptor, :instantiate;
        path=joinpath(cache, string(poisson_descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(poisson_problem) == 2
    # BridgeStan draws are unconstrained: positive rates live in log space.
    poisson_draws_unc = log.([1.5, 4.0])
    poisson_pointwise = brm_execute(poisson_descriptor, :pointwise_loglik;
        problem=poisson_problem, draws=poisson_draws_unc, seed=20260913)
    @test poisson_pointwise.y_likelihood ≈ poisson_expected atol=1e-10
    poisson_predictions = brm_execute(poisson_descriptor, :predict;
        problem=poisson_problem, draws=repeat(poisson_draws_unc, 1, 4096),
        seed=20260913)
    poisson_draws = vec(poisson_predictions.y_gen)
    @test mean(poisson_draws) ≈ mean(poisson_mixture) atol=0.15
    @test all(>=(0), poisson_draws)

    binomial_data = (; y=[1, 8, 3, 9], n=[10, 10, 10, 10])
    binomial_builder = @brm begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Binomial(n, p1), Binomial(n, p2)], [0.5, 0.5])
    end
    binomial_mixture = MixtureModel(
        [Binomial(10, 0.2), Binomial(10, 0.8)], [0.5, 0.5])
    binomial_expected = logpdf.(binomial_mixture, binomial_data.y)
    binomial_code = stan_code(SBBRMI(binomial_builder(binomial_data)))
    @test occursin("binomial_lpmf", binomial_code)
    @test occursin("brm_mixture_rows_int", binomial_code)
    @test StanBlocks.stanc_check(binomial_code; warn_pedantic=false).ok
    binomial_descriptor = brm_descriptor(
        binomial_builder, binomial_data; mod=@__MODULE__, highlights=())
    binomial_problem = brm_execute(binomial_descriptor, :instantiate;
        path=joinpath(cache, string(binomial_descriptor.id) * ".stan"))
    @test LogDensityProblems.dimension(binomial_problem) == 2
    # BridgeStan draws are unconstrained: probabilities live in logit space.
    binomial_draws_unc = logit.([0.2, 0.8])
    binomial_pointwise = brm_execute(binomial_descriptor, :pointwise_loglik;
        problem=binomial_problem, draws=binomial_draws_unc, seed=20260913)
    @test binomial_pointwise.y_likelihood ≈ binomial_expected atol=1e-10
    binomial_predictions = brm_execute(binomial_descriptor, :predict;
        problem=binomial_problem, draws=repeat(binomial_draws_unc, 1, 4096),
        seed=20260913)
    @test mean(vec(binomial_predictions.y_gen)) ≈ mean(binomial_mixture) atol=0.2

    bernoulli_builder = @brm begin
        p1 ~ Beta(2, 2)
        p2 ~ Beta(2, 2)
        y ~ MixtureModel([Bernoulli(p1), Bernoulli(p2)], [0.5, 0.5])
    end
    bernoulli_data = (; y=[0, 1, 1, 0])
    bernoulli_code = stan_code(SBBRMI(bernoulli_builder(bernoulli_data)))
    @test occursin("bernoulli_lpmf", bernoulli_code)
    @test StanBlocks.stanc_check(bernoulli_code; warn_pedantic=false).ok
    # Float 0/1 outcomes coerce to the `int` response Stan needs.
    bernoulli_float_code = stan_code(
        SBBRMI(bernoulli_builder((; y=[0.0, 1.0, 1.0, 0.0]))))
    @test StanBlocks.stanc_check(bernoulli_float_code; warn_pedantic=false).ok
end

@testset "mixture contract rejects incompatible inputs loudly" begin
    @test_throws "full-real" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Exponential(1)], [0.5, 0.5])
    end)((; y=[0.1])))
    @test_throws "is not scalar" SBBRMI((@brm begin
        y ~ MixtureModel([MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])], [1.0])
    end)((; y=[[0.1, 0.2]])))
    @test_throws "simplex" SBBRMI((@brm begin
        y ~ MixtureModel([Categorical([0.5, 0.5])], [1.0])
    end)((; y=[1, 2])))
    @test_throws "parameter-dependent" SBBRMI((@brm begin
        y ~ MixtureModel([Uniform(0, 1), Uniform(0, 1)], [0.5, 0.5])
    end)((; y=[0.5])))
    @test_throws "has no Stan translation" SBBRMI((@brm begin
        y ~ MixtureModel([ZeroInflatedPoisson(1.0, 0.2)], [1.0])
    end)((; y=[0, 1])))
    @test_throws "has no Stan translation" SBBRMI((@brm begin
        y ~ MixtureModel([LocationScale(0, 1, TDist(3))], [1.0])
    end)((; y=[0.1])))
    @test_throws "must sum to 1" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [0.5, 0.6])
    end)((; y=[0.1])))
    @test_throws "2 components but 3 weights" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [0.3, 0.3, 0.4])
    end)((; y=[0.1])))
    @test_throws "must be nonnegative" SBBRMI((@brm begin
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], [-0.5, 1.5])
    end)((; y=[0.1])))
    @test_throws "at least one component" SBBRMI((@brm begin
        y ~ MixtureModel([], Float64[])
    end)((; y=[0.1])))
    @test_throws "identical trial-count" SBBRMI((@brm begin
        y ~ MixtureModel([Binomial(n1, 0.2), Binomial(n2, 0.8)], [0.5, 0.5])
    end)((; y=[1, 8], n1=[10, 10], n2=[10, 10])))
    @test_throws "only integer values" SBBRMI((@brm begin
        y ~ MixtureModel([Poisson(1.0)], [1.0])
    end)((; y=[1.5])))
    @test_throws "must be nonnegative" SBBRMI((@brm begin
        y ~ MixtureModel([Poisson(1.0)], [1.0])
    end)((; y=[-1])))
    @test_throws "Dirichlet" SBBRMI((@brm begin
        w ~ Normal(0, 1)
        y ~ MixtureModel([Normal(0, 1), Normal(1, 1)], w)
    end)((; y=[0.1])))
    @test_throws "AnalyticWeights" SBBRMI((@brm begin
        y ~ weighted(MixtureModel([Normal(0, 1), Normal(1, 1)], [0.5, 0.5]),
                     aweights(w))
    end)((; y=[0.1, 0.2], w=[1.0, 1.0])))
    # Degenerate single-component and objective-weighted mixtures transpile.
    k1 = @brm begin
        y ~ MixtureModel([Normal(0, 1)], [1.0])
    end
    @test StanBlocks.stanc_check(
        stan_code(SBBRMI(k1((; y=[0.1, -0.2])))); warn_pedantic=false).ok
    weighted_builder = @brm begin
        y ~ weighted(MixtureModel([Normal(0, 1), Normal(3, 1)], [0.5, 0.5]),
                     fweights(w))
    end
    @test StanBlocks.stanc_check(stan_code(
        SBBRMI(weighted_builder((; y=[0.1, 2.9], w=[1.0, 2.0]))));
        warn_pedantic=false).ok
end
