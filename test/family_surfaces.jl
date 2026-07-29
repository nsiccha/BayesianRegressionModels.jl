# test/family_surfaces.jl — executable contracts for custom likelihood surfaces.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/family_surfaces.jl

using Test
using BayesianRegressionModels
using Distributions: LocationScale, TDist, Normal, LogNormal, Exponential,
                     Weibull, Poisson, BernoulliLogit, truncated, censored
using StanBlocks
using LogDensityProblems

const FAMILY_SURFACE_CACHE = joinpath(tempdir(), "brm-family-surfaces")

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y_zip=[0, 1, 0, 2, 3, 0],
    y_nb=[0, 1, 2, 4, 3, 6],
    y_t=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
    y_truncated=[-0.7, -0.2, 0.1, 0.7, 1.0, 1.2],
    y_censored=[-0.5, -0.2, 0.1, 0.7, 1.0, 1.0],
    y_lognormal_censored=[0.25, 0.35, 0.60, 1.10, 1.80, 1.80],
    y_exponential_censored=[0.10, 0.30, 0.70, 1.10, 1.50, 1.50],
    y_weibull_censored=[0.20, 0.45, 0.75, 1.20, 1.60, 1.60],
    count_truncated=[1, 1, 2, 4, 3, 4],
    count_censored=[0, 1, 2, 4, 4, 4],
    # Genuine interval evidence: the response is the lower endpoint, while the
    # wrapper carries the row-wise upper endpoint.
    y_interval=[0.10, 0.25, 0.40, 0.70, 0.95, 1.20],
    y_interval_upper=[0.30, 0.50, 0.75, 1.05, 1.35, 1.70],
    y_interval_bad_upper=[0.05, 0.20, 0.35, 0.60, 0.90, 1.10],
    count_interval=[0, 0, 1, 2, 3, 4],
    count_interval_upper=[1, 2, 2, 4, 5, 6],
    y_bernoulli=[0, 1, 0, 1, 1, 0],
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

    y_truncated ~ truncated(Normal(loc, scale); lower=-0.75, upper=1.25)
    y_censored ~ censored(Normal(loc, scale); lower=-0.5, upper=1.0)
    y_lognormal_censored ~ censored(LogNormal(loc, scale); lower=0.25, upper=1.8)
    y_exponential_censored ~ censored(Exponential(lambda); lower=nothing, upper=1.5)
    y_weibull_censored ~ censored(Weibull(1.4, 1.1); upper=1.6)
    count_truncated ~ truncated(Poisson(lambda), 1, 4)
    count_censored ~ censored(Poisson(lambda); lower=0, upper=4)
    y_interval ~ interval_censored(Normal(loc, scale); upper=y_interval_upper)
    count_interval ~ interval_censored(Poisson(lambda); upper=count_interval_upper)
end

@testset "SBBRMI lowers density, pointwise log-lik and RNG paths" begin
    plan = generative_plan(family_builder, df; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(plan)

    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("zero_inflated_poisson(", code)
    @test occursin("neg_binomial_2(", code)
    @test occursin("student_t(", code)

    for target in (:y_zip, :y_nb, :y_t, :y_truncated, :y_censored,
                   :y_lognormal_censored, :y_exponential_censored,
                   :y_weibull_censored, :count_truncated, :count_censored,
                   :y_interval, :count_interval)
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
    @test families[:y_truncated] === :conditioned
    @test families[:y_censored] === :clamped
    @test families[:y_lognormal_censored] === :clamped
    @test families[:y_exponential_censored] === :clamped
    @test families[:y_weibull_censored] === :clamped
    @test families[:count_truncated] === :conditioned
    @test families[:count_censored] === :clamped
    @test families[:y_interval] === :interval_evidence
    @test families[:count_interval] === :interval_evidence
end

@testset "Julia-native wrapper surface and capability gate" begin
    brmi = family_builder(df)
    rhs_for(target) = begin
        op = parent(getproperty(brmi.operations, target))
        only(a for a in getargs(op) if a isa ExprColumn)
    end

    truncated_rhs = rhs_for(:y_truncated)
    @test getf(truncated_rhs) === truncated
    @test keys(getkwargs(truncated_rhs)) == (:lower, :upper)
    @test getf(only(getargs(truncated_rhs))) === Normal

    censored_rhs = rhs_for(:y_censored)
    @test getf(censored_rhs) === censored
    @test getf(only(getargs(censored_rhs))) === Normal

    exponential_rhs = rhs_for(:y_exponential_censored)
    @test BayesianRegressionModels._sb_normalize_bound(
        getkwargs(exponential_rhs).lower) === nothing

    interval_rhs = rhs_for(:y_interval)
    @test getf(interval_rhs) === interval_censored
    @test keys(getkwargs(interval_rhs)) == (:upper,)

    unsupported_builder = @brm begin
        eta ~ 1 + x
        y_bernoulli ~ censored(BernoulliLogit(eta); lower=0, upper=1)
    end
    @test_throws "no generic CDF/CCDF composition capability" SBBRMI(
        unsupported_builder(df); mod=@__MODULE__)
end

@testset "wrapper bounds fail before Stan" begin
    bad_interval = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_interval ~ interval_censored(Normal(loc, scale); upper=y_interval_bad_upper)
    end
    @test_throws "lower endpoints must be strictly below upper endpoints" SBBRMI(
        bad_interval(df); mod=@__MODULE__)

    bad_keyword = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_truncated ~ truncated(Normal(loc, scale); lower=-1.0, typo=1.0)
    end
    @test_throws "accepts only `lower` and `upper` keywords" SBBRMI(
        bad_keyword(df); mod=@__MODULE__)

    bad_discrete_bounds = @brm begin
        log(lambda) ~ 1 + x
        count_truncated ~ truncated(Poisson(lambda); lower=1.5, upper=4.0)
    end
    @test_throws "discrete lower bounds must be integers" SBBRMI(
        bad_discrete_bounds(df); mod=@__MODULE__)

    missing_bounds = @brm begin
        loc ~ 1 + x
        log(scale) ~ 1
        y_truncated ~ truncated(Normal(loc, scale))
    end
    @test_throws "needs at least one non-`nothing` bound" SBBRMI(
        missing_bounds(df); mod=@__MODULE__)
end

@testset "wrapper families execute through BridgeStan" begin
    sb = SBBRMI(family_builder(df); mod=@__MODULE__)
    mkpath(FAMILY_SURFACE_CACHE)
    path = joinpath(FAMILY_SURFACE_CACHE,
                    string(hash(BayesianRegressionModels.stan_code(sb))) * ".stan")
    problem = StanBlocks.stan_instantiate(sb.model; path)
    n = LogDensityProblems.dimension(problem)
    q = [0.05 * ((i % 5) - 2) for i in 1:n]
    lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    @test isfinite(lp)
    @test length(gradient) == n
    @test all(isfinite, gradient)
end
