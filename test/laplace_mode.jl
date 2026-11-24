# brm_laplace_mode: the mode of the embedded-Laplace marginal, by L-BFGS on a
# Laplace problem's own value and gradient (src/laplace_mode.jl, the Optim
# extension). Checked: loading WarmupHMC loads the extension; on the sparse
# problem over crossed groupings and on the row-structured problem the result
# is a complete packed point whose population part is a stationary maximum
# (gradient within the tolerance, no higher value nearby), the same from
# another start; a search that stalls or is cut off is not reported converged;
# and the search itself reaches its tolerance where the value's rounding is
# larger than a step's gain (snag `brm-laplace-mode-fcec521b`).
# RUN: julia --project=test test/laplace_mode.jl
include(joinpath(@__DIR__, "fitting_packages.jl"))
using WarmupHMC
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

function mode_checks(p)
    out = brm_laplace_mode(p)
    φ = out.draws[p.population, 1]
    @test size(out.draws) == (length(p.u), 1)
    @test out.diagnostics.converged
    @test out.diagnostics.stop === :tol
    v, g = LogDensityProblems.logdensity_and_gradient(p, φ)
    @test v ≈ out.diagnostics.value rtol=1e-10
    @test out.diagnostics.gradient_norm == maximum(abs, g) <= 1e-6
    for k in eachindex(φ), h in (-1e-3, 1e-3)
        e = zeros(length(φ))
        e[k] = h
        @test LogDensityProblems.logdensity(p, φ .+ e) <= v + 1e-6
    end
    other = brm_laplace_mode(p; init=φ .+ 0.3 .* randn(Xoshiro(2), length(φ)))
    @test other.draws[p.population, 1] ≈ φ atol=1e-4
    out
end

@stestset "loading WarmupHMC loads the Optim extension" begin
    @test Base.get_extension(BRM, :BayesianRegressionModelsOptimExt) !== nothing
    # the extension adds its method beside the fallback: overwriting it is
    # refused by precompilation, and the extension then loads uncompiled
    @test length(methods(BRM._brm_laplace_mode)) == 2
end

@stestset "the mode of the sparse Laplace over crossed groupings" begin
    rng = Xoshiro(20261030)
    la = [string("a", i) for i in 1:8]
    lb = [string("b", i) for i in 1:5]
    ma = Dict(l => randn(rng) for l in la)
    mb = Dict(l => 0.5 * randn(rng) for l in lb)
    a = repeat(la; inner=15)
    b = repeat(repeat(lb; inner=3), 8)
    x = randn(rng, length(a))
    y = [rand(rng, Poisson(exp(0.3 + 0.4xi + ma[ai] + mb[bi]))) for (ai, bi, xi) in zip(a, b, x)]
    rk = RKBRMI(@brm (; a, b, x, y) begin
        log(mu) ~ 1 + x + (1 | a) + (1 | b)
        y ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:a, :b))
    out = mode_checks(p)
    # the integrated effects are their conditional modes at the population mode
    LogDensityProblems.logdensity(p, out.draws[p.population, 1])
    @test out.draws[p.integrated, 1] ≈ p.u[p.integrated] atol=1e-8
end

@stestset "the mode of the row-structured Laplace" begin
    rng = Xoshiro(20261031)
    g = vcat([fill("g$i", 6) for i in 1:20]...)
    x = randn(rng, length(g))
    e = Dict(l => 0.6 * randn(rng) for l in unique(g))
    y = [rand(rng, Poisson(exp(0.2 + 0.5xi + e[gi]))) for (gi, xi) in zip(g, x)]
    rk = RKBRMI(@brm (; g, x, y) begin
        log(mu) ~ 1 + x + (1 | g)
        y ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD)
    @test p isa BRMStructuredLaplaceProblem
    found = mode_checks(p)
    @test_throws DimensionMismatch brm_laplace_mode(p; init=zeros(1))
    # A tolerance no gradient meets: the search ends where its steps change
    # nothing. That is the mode, and it is named a stall, not convergence.
    stalled = @test_logs (:warn, r"no further progress.*above g_abstol = 0\.0") brm_laplace_mode(p; g_abstol=0.0)
    @test !stalled.diagnostics.converged
    @test stalled.diagnostics.stop === :stalled
    @test 0 < stalled.diagnostics.gradient_norm < 1e-6
    @test stalled.draws[p.population, 1] ≈ found.draws[p.population, 1] atol=1e-6
    # Cut off by the iteration limit.
    cut = @test_logs (:warn, r"limit of 1 iterations.*above g_abstol") brm_laplace_mode(p; iterations=1)
    @test !cut.diagnostics.converged
    @test cut.diagnostics.stop === :iterations
    @test cut.diagnostics.iterations == 1
end

# The search itself, on a stand-in for a large marginal near its mode: a smooth
# bowl whose VALUE carries rounding-like noise (a fixed function of the point
# that varies over 1e-9) and whose gradient is exact. The noise sizes span
# what a mesh field's marginal carries (2e-11 at 1,800 integrated effects,
# 6e-10 at 21,000) and beyond. From a gradient near 1e-4 down, a step's gain
# in value is below that noise, so a test on values accepts or refuses it by
# the noise: backtracking missed the tolerance in 8 of the 20 runs at 6e-10
# and 13 of 20 at 1e-7, and spent up to 66 evaluations where 14 suffice
# without noise. The slope test does not see it: 8 evaluations in every run.
@stestset "the search reaches its tolerance under rounding in the value" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsOptimExt)
    A = [680.0 150.0; 150.0 500.0]
    m = [-0.14, -2.0]
    smooth(x) = 29529.3 + dot(x .- m, A, x .- m) / 2 + 40 * sum(abs2, x .- m)^2
    slope(x) = A * (x .- m) .+ 160 * sum(abs2, x .- m) .* (x .- m)
    wiggle(x, phase) = sin(1e9 * (x[1] + 0.618 * x[2]) + phase)
    # rounding in the gradient (`gnoise`), which a computed gradient carries too
    gwiggle(x, phase) = [cos(3e9 * x[1] + 7e8 * x[2] + phase), sin(2e9 * x[2] - 1e9 * x[1] + phase)]
    function search(noise, phase; g_abstol=1e-6, gnoise=0.0)
        evaluations = Ref(0)
        f(x) = (evaluations[] += 1; smooth(x) + noise * wiggle(x, phase))
        fg!(G, x) = (evaluations[] += 1; G .= slope(x) .+ gnoise .* gwiggle(x, phase);
            smooth(x) + noise * wiggle(x, phase))
        r = ext._brm_lbfgs(f, fg!, [0.0, -1.9]; iterations=1000, g_abstol)
        (; r..., evaluations=evaluations[], gradient=maximum(abs, slope(r.minimizer)))
    end
    exact = search(0.0, 0.0)
    @test exact.stop === :tol && exact.gradient <= 1e-6
    for noise in (2e-11, 6e-10, 1e-7), phase in 0.0:19.0
        r = search(noise, phase)
        @test r.stop === :tol
        @test r.gradient_norm == r.gradient <= 1e-6
        @test r.evaluations <= exact.evaluations + 2
    end
    # With a tolerance no computed gradient meets (zero, with the gradient's own
    # rounding of 1e-13), the search ends promptly where its steps change
    # nothing, and names it a stall. Without that rounding it can land on the
    # exact minimizer, a gradient of exactly zero: `:tol`, rightly.
    for phase in 0.0:9.0
        r = search(6e-10, phase; g_abstol=0.0, gnoise=1e-13)
        @test r.stop === :stalled
        @test r.iterations <= 20
        @test r.gradient < 1e-6
    end
end
