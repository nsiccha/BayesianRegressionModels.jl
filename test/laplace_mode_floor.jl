# The Laplace mode search's rounding floor. A decrement that has stopped shrinking
# fourfold is at its floor only if the gradient is rounding. On the steep side of an
# exponential, Newton's decrement falls by a factor e per step, so it stops shrinking
# fourfold while the gradient is still far from zero. The marginal must then come from
# the mode, not from that point, on every problem kind. A Gaussian term beside a large
# intercept has a real floor above `tol`, which must still stop the search.
# RUN: julia --project=test test/laplace_mode_floor.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems

const FLOOR_TARGETS = (
    "whole-model" => (rk, kw) -> brm_laplace_problem(rk; ad_backend=NLME_AD, structured=false, kw...),
    "structured" => (rk, kw) -> brm_laplace_problem(rk; ad_backend=NLME_AD, kw...),
    "sparse" => (rk, kw) -> brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g, kw...),
    "matrix-free" => (rk, kw) -> brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g,
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=4), kw...),
)

# The coordinates of the standardized effects, in the packed point.
floor_effects(p) = p isa BRMSparseLaplaceProblem ? p.integrated : setdiff(eachindex(p.u), p.population)

# Poisson row of count y at log rate w = a + τ z, with z ~ N(0, 1): the root of the
# effect's gradient τ (y − exp(w)) − z, by bisection in BigFloat, and the curvature there.
function floor_wall_oracle(a, τ, y)
    lo, hi = big(-300.0), big(60.0)
    for _ in 1:600
        m = (lo + hi) / 2
        τ * (y - exp(m)) - (m - a) / τ > 0 ? (lo = m) : (hi = m)
    end
    w = (lo + hi) / 2
    (; z=Float64((w - a) / τ), h=Float64(1 + τ^2 * exp(w)))
end

@stestset "A decrement that stops shrinking away from the mode is not a rounding floor" begin
    # Level "wall" starts e^20 up an exponential of rate τ = 1e10: the search needs
    # about 65 steps of −1/τ, and its decrement passes 1e-8 near step 38, with a
    # gradient of about 100. Level "easy" converges at once, so one component stops
    # while the other keeps going. τ is reachable by the scale's coordinate.
    y = [0, 3]
    rk = RKBRMI(@brm (; g=["a_wall", "b_easy"], y) begin
        log(mu) ~ 1 + (1 | g)
        y ~ Poisson(mu)
    end)
    a, τ = 20.0, 1e10
    oracle = [floor_wall_oracle(a, τ, yi) for yi in y]
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    for (name, make) in FLOOR_TARGETS
        @testset "$name" begin
            p = make(rk, (; maxiter=200))
            name == "structured" && @test p isa BRMStructuredLaplaceProblem
            mean_index = findfirst(==(:mu_Intercept), p.names)
            φ = zeros(length(p.population))
            φ[mean_index] = a
            φ[only(setdiff(eachindex(φ), mean_index))] = log(τ)
            value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
            gradient = copy(gradient)
            @test p.unconverged == 0
            effects = floor_effects(p)
            # A decrement of `tol` leaves about 1e-12 of z, far below the 2e-9 that
            # separates the mode from where the decrement passes 1e-8.
            @test p.u[effects] ≈ getfield.(oracle, :z) atol=1e-12
            # The marginal at the mode, from the full model's density there. Its
            # determinant moves at τ/2 per unit of z, so that 1e-12 is up to 5e-3 here.
            x = zeros(length(p.u))
            x[p.population] .= φ
            x[effects] .= getfield.(oracle, :z)
            expected = LogDensityProblems.logdensity(full, x) + length(y) * log(2π) / 2 -
                sum(o -> log(o.h), oracle) / 2
            @test value ≈ expected atol=5e-3
            # The marginal is smooth here, so the gradient matches central differences.
            for j in eachindex(φ)
                h = 1e-4
                e = zeros(length(φ)); e[j] = h
                fd = (LogDensityProblems.logdensity(p, φ + e) -
                      LogDensityProblems.logdensity(p, φ - e)) / (2h)
                @test gradient[j] ≈ fd atol=1e-5 rtol=1e-5
            end
        end
    end
end

@stestset "A real rounding floor above tol stops the search" begin
    # The effect enters beside an intercept of 1e6, whose rounding sets the gradient's:
    # about 1e-4 at curvature 1e6, a decrement near 1e-14 the search cannot get under.
    a, σ = 1e6, 1e-3
    rk = RKBRMI(@brm (; g=["one"], y=[a + 0.01]) begin
        mu ~ 1 + (1 | g)
        y ~ Normal(mu, 1e-3)
    end)
    for (name, make) in FLOOR_TARGETS
        @testset "$name" begin
            p = make(rk, (; maxiter=50))
            mean_index = findfirst(==(:mu_Intercept), p.names)
            φ = zeros(length(p.population))
            φ[mean_index] = a
            value = LogDensityProblems.logdensity(p, φ)
            @test isfinite(value)
            @test p.unconverged == 0
            # Scale one: the mode is 0.01 τ²/(τ² + σ²).
            @test only(p.u[floor_effects(p)]) ≈ 0.01 / (1 + σ^2) rtol=1e-6
            p isa BRMSparseLaplaceProblem && @test only(p.search.stop) === :floor
        end
    end
end
