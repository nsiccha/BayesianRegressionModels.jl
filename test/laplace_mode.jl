# brm_laplace_mode: the mode of the embedded-Laplace marginal, by L-BFGS on a
# Laplace problem's own value and gradient (src/laplace_mode.jl, the Optim
# extension). Checked: loading WarmupHMC loads the extension; on the sparse
# problem over crossed groupings and on the row-structured problem the result
# is a complete packed point whose population part is a stationary maximum
# (gradient near zero, no higher value nearby), the same from another start.
# RUN: julia --project=test test/laplace_mode.jl
using WarmupHMC
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

function mode_checks(p)
    out = brm_laplace_mode(p)
    φ = out.draws[p.population, 1]
    @test size(out.draws) == (length(p.u), 1)
    @test out.diagnostics.converged
    v, g = LogDensityProblems.logdensity_and_gradient(p, φ)
    @test v ≈ out.diagnostics.value rtol=1e-10
    @test norm(g) < 1e-4
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
    mode_checks(p)
    @test_throws DimensionMismatch brm_laplace_mode(p; init=zeros(1))
end
