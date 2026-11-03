# The embedded Laplace marginal's conditional mode is the one continuation
# from the prior reaches: the mode followed from the prior's mode while the
# likelihood's weight rises to one. A conditional with two modes, whose
# followed mode ceases to exist on part of a line, checks the selected mode
# against a scalar oracle, the jump where it ceases to exist, the fold
# counter, finite differences on each smooth piece, and that the sparse and
# whole-model targets agree. A log-concave model takes one stage.
# RUN: julia --project=test test/laplace_continuation.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems

# One unit with a standardized effect z, m = φ + z (unit SD), observations
# y = 2 of m² and x = 2 of m, both with SD 1. The stationary equation of the
# effect's density at likelihood weight one is φ + 2 + 2m − 2m³ = 0, with
# curvature 6m² − 2 there; the negative mode exists for φ < φc = −2 + 4/(3√3).
const PHI_C = -2 + 4 / (3sqrt(3))
function continuation_mode(φ)
    # The followed mode: the negative branch while it exists, else the only one.
    m = φ < PHI_C ? -0.8 : 1.2
    for _ in 1:50
        m -= (φ + 2 + 2m - 2m^3) / (2 - 6m^2)
    end
    m
end
function oracle(φ)
    m = continuation_mode(φ)
    z = m - φ
    logpdf(Normal(), z) + logpdf(Normal(m^2, 1), 2.0) + logpdf(Normal(m, 1), 2.0) +
        logpdf(Normal(0, 10), φ) + logpdf(LogNormal(), 1.0) + log(2π / (6m^2 - 2)) / 2
end

@testset "Laplace modes by continuation from the prior" begin
    rk = RKBRMI(@brm (; g=["one"], y=[2.0], x=[2.0]) begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 10)
        sd(mu, g) ~ LogNormal(0, 1)
        @plate for i in eachindex(mu)
            squared = mu[i]^2
            y[i] ~ Normal(squared, 1.0)
            x[i] ~ Normal(mu[i], 1.0)
        end
    end)
    targets = [brm_laplace_problem(rk; ad_backend=NLME_AD, structured=false),
        brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g)]
    mean_index = findfirst(==(:mu_Intercept), targets[1].names)
    point(φ) = (x = zeros(2); x[mean_index] = φ; x)
    for p in targets
        @testset "$(nameof(typeof(p)))" begin
            # Both sides of φc, away from it: the followed mode and its value.
            # The unit's effect coordinate, and so its mode m = φ + z.
            zindex = only(p isa BRMSparseLaplaceProblem ? p.integrated :
                setdiff(eachindex(p.u), p.population))
            grid = vcat(range(-1.32, -1.25; length=8), range(-1.21, -1.15; length=7))
            before = p.folds
            for φ in grid
                @test LogDensityProblems.logdensity(p, point(φ)) ≈ oracle(φ) atol=1e-9
                # The negative mode while it exists, then the positive one: the
                # marginal jumps where the followed mode ceases to exist.
                m = φ + p.u[zindex]
                @test m ≈ continuation_mode(φ) atol=1e-6
                @test sign(m) == (φ < PHI_C ? -1 : 1)
                if φ < PHI_C
                    # The negative mode is followed all the way: no fold.
                    @test p.folds == before
                end
            end

            # φ = −1: the mode followed from the prior ceases to exist at weight
            # about 2/3; the evaluation counts a fold and ends at the only mode.
            folds = p.folds
            value, gradient = LogDensityProblems.logdensity_and_gradient(p, point(-1.0))
            gradient = copy(gradient)
            @test p.folds == folds + 1
            @test value ≈ oracle(-1.0) atol=1e-9
            for φ in (-1.4, -1.0), j in 1:2
                x0 = point(φ)
                _, g = LogDensityProblems.logdensity_and_gradient(p, x0)
                g = copy(g)
                h = 1e-5
                e = zeros(2); e[j] = h
                fd = (LogDensityProblems.logdensity(p, x0 + e) -
                      LogDensityProblems.logdensity(p, x0 - e)) / (2h)
                @test g[j] ≈ fd atol=1e-7 rtol=1e-7
            end
        end
    end
    # The two targets select the same modes.
    for φ in (-1.4, -1.0, 0.5)
        @test LogDensityProblems.logdensity(targets[1], point(φ)) ≈
            LogDensityProblems.logdensity(targets[2], point(φ)) atol=1e-10
    end
end

@testset "A log-concave conditional takes one continuation stage" begin
    rk = RKBRMI(@brm (; g=["a", "a", "b", "c", "c"], y=[3, 4, 5, 2, 1]) begin
        log(mu) ~ 1 + (1 | g)
        y ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g)
    @test isfinite(LogDensityProblems.logdensity(p, zeros(length(p.population))))
    @test p.search.stages == [1.0]
    @test p.folds == 0
end
