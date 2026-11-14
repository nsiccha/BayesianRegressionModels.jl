# test/rk_preparation_point.jl — where BRM prepares an RK density (snag
# `construct-rk-log-1c0faf16`).
#
# Preparing an RK gradient runs the model once at a packed point. A model that
# cannot be evaluated at the zero point (here a rate law defined only from 2
# upward; an ODE unstable at unit rates in the report) used to fail inside
# `rk_logdensity_problem` before any caller point was evaluated. BRM now
# prepares at the prior medians, keeping zero where a prior has no finite
# median. Checks:
# - the zero point is not a point of the model (positive control), and an
#   explicit `u0 = zeros` still fails, so `u0` is honoured;
# - the default point is `brm_start(rk, (;); base = :prior_median)`, and the
#   density there equals an independent oracle, with central-difference
#   gradients;
# - two preparation points give identical densities and gradients;
# - a prior without a median keeps its zero-point value, top level and in a
#   component, where `brm_start(...; base = :prior_median)` refuses;
# - BRM's own preparations: the NLME pointwise sampler and both cut forms.
# Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_preparation_point.jl

using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

module PrepModels
using BayesianRegressionModels, Distributions
# A rate law defined only where the rate is at least 2: the zero point (rate 1)
# is outside it, as an unstable ODE solve is.
function guarded_rate(k)
    k >= 2 || throw(DomainError(k, "guarded_rate: the rate must be at least 2"))
    k
end
guarded = @brm begin
    log_k ~ Normal(log(5.0), 0.2)
    sigma ~ Exponential(1)
    mu = guarded_rate(exp(log_k)) .* x
    y ~ Normal(mu, sigma)
end
# A prior without a median (`Flat()`), at top level and in a component.
@brm function flat_shift(x)
    a ~ Flat()
    return x + a
end
flat = @brm begin
    log_k ~ Normal(log(5.0), 0.2)
    b ~ Flat()
    sigma ~ Exponential(1)
    shifted ~ flat_shift(x)
    mu = guarded_rate(exp(log_k)) .* shifted .+ b
    y ~ Normal(mu, sigma)
end
# Two stages: `m` upstream, the guarded rate downstream.
joint = @brm begin
    m ~ Normal(0, 1)
    y_up ~ Normal(m, 1)
    log_k ~ Normal(log(5.0), 0.2)
    sigma ~ Exponential(1)
    mu = guarded_rate(exp(log_k)) .* x .+ m
    y ~ Normal(mu, sigma)
end
# Stage 2 alone, reading the upstream value as data `c`.
downstream = @brm begin
    log_k ~ Normal(log(5.0), 0.2)
    sigma ~ Exponential(1)
    mu = guarded_rate(exp(log_k)) .* x .+ c
    y ~ Normal(mu, sigma)
end
end

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const X = [0.2, 0.5, 0.9, 1.4]
const Y = [1.1, 2.4, 4.6, 7.1]
const DATA = (; x=X, y=Y)

value_names(rk) = BRM._rk_layout_coordinate_names(rk)

# The guarded model's full log density at `u`, from its constrained values:
# the two priors, the exponential transform's log Jacobian and the likelihood.
function guarded_oracle(rk, u)
    v = BRM._rk_constrained_values(rk, u)
    k = exp(v.log_k)
    logpdf(Normal(log(5.0), 0.2), v.log_k) + logpdf(Exponential(1), v.sigma) +
        log(v.sigma) + sum(logpdf.(Normal.(k .* X, v.sigma), Y))
end

function central_gradient(f, u; step=1e-6)
    map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += step; minus[j] -= step
        (f(plus) - f(minus)) / (2step)
    end
end

const RK = RKBRMI(PrepModels.guarded(DATA))
const N = length(value_names(RK))

@stestset "the zero point is not a point of the guarded model" begin
    @test N == 2
    @test_throws "guarded_rate" rk_logdensity_problem(RK; ad_backend=AD, u0=zeros(N))
end

@stestset "the default preparation point is the prior medians" begin
    median_point = brm_start(RK, (;); base=:prior_median)
    @test BRM._rk_preparation_point(RK) == median_point
    values = BRM._rk_constrained_values(RK, median_point)
    @test values.log_k ≈ log(5.0)
    @test values.sigma ≈ median(Exponential(1))
    problem = rk_logdensity_problem(RK; ad_backend=AD)
    @test LogDensityProblems.dimension(problem) == N
    for u in (median_point, median_point .+ [0.1, -0.3], median_point .+ [-0.2, 0.4])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ guarded_oracle(RK, u) atol=1e-10 rtol=1e-10
        @test gradient ≈ central_gradient(w -> guarded_oracle(RK, w), u) atol=1e-5 rtol=1e-6
    end
    # The density does not depend on where it was prepared.
    other = rk_logdensity_problem(RK; ad_backend=AD, u0=median_point .+ [0.3, 0.2])
    for u in (median_point, median_point .+ [0.05, -0.1])
        @test LogDensityProblems.logdensity_and_gradient(other, u) ==
            LogDensityProblems.logdensity_and_gradient(problem, u)
    end
end

@stestset "a prior without a median keeps its zero-point value" begin
    rk = RKBRMI(PrepModels.flat(DATA))
    @test_throws "explicit initializer" brm_start(rk, (;); base=:prior_median)
    point = BRM._rk_preparation_point(rk)
    values = BRM._rk_constrained_values(rk, point)
    zero_values = BRM._rk_constrained_values(rk, zeros(length(point)))
    @test values.log_k ≈ log(5.0)
    @test values.sigma ≈ median(Exponential(1))
    @test values.b == zero_values.b
    @test values.shifted.a == zero_values.shifted.a
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, point)
    @test isfinite(value) && all(isfinite, gradient)
    @test_throws "guarded_rate" rk_logdensity_problem(rk; ad_backend=AD,
        u0=zeros(length(point)))
end

@stestset "the NLME pointwise sampler prepares at the preparation point" begin
    sampler = BRM._rk_retained_pointwise_sampler(RK; ad_backend=AD)
    u = BRM._rk_preparation_point(RK) .+ [0.1, -0.2]
    value, gradient, pointwise = BRM._rk_value_gradient_and_pointwise!(sampler,
        zeros(N), u)
    reference = LogDensityProblems.logdensity_and_gradient(
        rk_logdensity_problem(RK; ad_backend=AD), u)
    @test value ≈ first(reference) atol=1e-12 rtol=1e-12
    @test gradient ≈ last(reference) atol=1e-12 rtol=1e-12
    @test !isempty(pointwise)
end

@stestset "cut targets prepare at the preparation point" begin
    joint = RKBRMI(PrepModels.joint((; x=X, y=Y, y_up=[0.3, -0.1, 0.4])))
    upstream = RKBRMI(PrepModels.joint((; x=X, y_up=[0.3, -0.1, 0.4])))
    @test value_names(upstream) == [:m]
    U = [0.2 -0.4]
    joint_problem = rk_logdensity_problem(joint; ad_backend=AD)
    joint_names = value_names(joint)
    coordinate = brm_cut_family(joint, upstream, U; ad_backend=AD)
    @test coordinate.names == [:log_k, :sigma]
    free = [findfirst(==(n), joint_names) for n in coordinate.names]
    pinned = findfirst(==(:m), joint_names)
    x = BRM._rk_preparation_point(joint)[free] .+ [0.1, -0.2]
    for k in axes(U, 2)
        value, gradient = LogDensityProblems.logdensity_and_gradient(coordinate.targets[k], x)
        u = zeros(length(joint_names)); u[free] = x; u[pinned] = U[1, k]
        @test gradient ≈ last(LogDensityProblems.logdensity_and_gradient(joint_problem,
            u))[free] atol=1e-10 rtol=1e-10
        @test isfinite(value)
    end
    down = RKBRMI(PrepModels.downstream((; x=X, y=Y, c=zeros(length(X)))))
    explicit = brm_cut_family(u -> (; c=fill(u[1], length(X))), down, (; x=X, y=Y),
        upstream, U; ad_backend=AD)
    @test explicit.names == [:log_k, :sigma]
    for k in axes(U, 2)
        value, gradient = LogDensityProblems.logdensity_and_gradient(explicit.targets[k], x)
        u = zeros(length(joint_names)); u[free] = x; u[pinned] = U[1, k]
        @test gradient ≈ last(LogDensityProblems.logdensity_and_gradient(joint_problem,
            u))[free] atol=1e-10 rtol=1e-10
        @test isfinite(value)
    end
end
