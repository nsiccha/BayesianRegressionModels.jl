# brm_ladder_fit (src/laplace_ladder.jl): the staged fit over the sparse
# embedded Laplace. Checked: every rung climbed is a complete result and
# starts the next; rung 2 is a stationary maximum of the marginal; rung 3 is
# skipped and named without units (test/nlme_integrated.jl runs it); rung 4's
# draws keep the point; rung 5a's curvature is
# negative definite there and labelled a diagnostic; rungs 5b and 6 run with a
# sampler (WarmupHMC here) and are named without one; a later call continues
# from an earlier fit; `:auto` re-measures the plan at rung 2's point.
# RUN: julia --project=test test/laplace_ladder.jl
using WarmupHMC   # loads Optim, which rung 2's optimizer needs
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

const LADDER_DATA = let rng = Xoshiro(11)
    a = repeat(["a$i" for i in 1:6], inner=8)
    b = repeat(["b$i" for i in 1:4], outer=12)
    x = randn(rng, length(a))
    ea = Dict(l => 0.7randn(rng) for l in unique(a))
    eb = Dict(l => 0.5randn(rng) for l in unique(b))
    η = [0.4 + 0.6x[i] + ea[a[i]] + eb[b[i]] for i in eachindex(a)]
    (; a, b, x, y=η .+ 0.3 .* randn(rng, length(a)),
        count=[rand(rng, Poisson(exp(0.2 + 0.4e))) for e in η])
end

@stestset "the rungs of a Normal crossed model, up to the curvature" begin
    d = LADDER_DATA
    rk = RKBRMI(@brm (; d.a, d.b, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    fit = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=5, draws=50, rng=Xoshiro(3))
    p = fit.problem
    @test [r.rung for r in fit.rungs] == [1, 2, 3, 4, 5, 5]
    @test [c.rung for c in fit.ladder] == [1, 2, 3, 4, 5, 5]
    @test fit.plan isa BRMLaplacePlan && p.plan === fit.plan
    # rung 1: the modes at the prior medians, a complete packed point
    r1 = fit.rungs[1]
    @test r1.point == BRM._brm_laplace_prior_median(p)
    @test size(r1.draws) == (length(p.u), 1) && all(isfinite, r1.draws)
    @test fit.ladder[1].finite
    # rung 2: a stationary maximum of the marginal
    @test fit.ladder[2].converged && fit.ladder[2].gradient_norm < 1e-4
    @test fit.point == fit.rungs[2].point
    v, g = LogDensityProblems.logdensity_and_gradient(p, fit.point)
    @test norm(g) < 1e-4
    # rung 3 is named, not run: the call names no units
    @test fit.ladder[3].skipped && occursin("no units", fit.ladder[3].reason)
    # rung 4: draws keep the point and vary the integrated effects
    r4 = only(r for r in fit.rungs if r.rung == 4)
    @test size(r4.draws) == (length(p.u), 50)
    @test all(r4.draws[p.population, k] == fit.point for k in 1:50)
    @test all(>(0), vec(std(r4.draws[p.integrated, :]; dims=2)))
    # rung 5a: negative definite curvature at the mode, a diagnostic
    c = fit.ladder[5]
    @test c.diagnostic && c.positive_definite
    @test all(>(0), c.standard_errors)
    r5 = only(r for r in fit.rungs if r.name === :population_curvature)
    @test r5.covariance ≈ inv(-Symmetric(r5.hessian)) rtol=1e-8
    @test fit.draws === r4.draws
    # 5b needs a sampler, and is named without one
    @test fit.ladder[6].name === :population_hmc && fit.ladder[6].skipped

    # Supplying a sampler later runs both skipped sampling rungs, preserving
    # the old result and passing the marginal draw to the exact sampler.
    missing = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=6, continue_from=fit)
    calls = Vector{Vector{Float64}}()
    sampler(q, init, rng) = (push!(calls, copy(init)); repeat(init, 1, 3))
    more = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=6,
        continue_from=missing, sampler, rng=Xoshiro(7))
    hmc = only(r for r in more.rungs if r.name === :population_hmc)
    exact = only(r for r in more.rungs if r.name === :exact)
    @test length(calls) == 2 && calls[1] == fit.point
    @test calls[2] == hmc.draws[:, end]
    @test more.draws === exact.draws && size(more.draws, 2) == 3
    @test only(r for r in missing.rungs if r.name === :exact).skipped
    @test only(r for r in fit.rungs if r.name === :population_hmc).skipped
    @test length(more.rungs) == length(more.ladder) == 7
end

@stestset "rungs 5b and 6 with a sampler" begin
    d = LADDER_DATA
    rk = RKBRMI(@brm (; d.a, d.b, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    sampler(q, init, rng) = Matrix(adaptive_warmup_mcmc(rng, q; init,
        n_draws=100).posterior_position)
    fit = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=6, draws=20, sampler,
        rng=Xoshiro(6))
    p = fit.problem
    hmc = only(r for r in fit.rungs if r.name === :population_hmc)
    @test size(hmc.draws, 1) == length(p.u) && size(hmc.draws, 2) >= 100
    # the population varies across draws; each draw's effects come through the factor
    @test all(>(0), vec(std(hmc.draws[p.population, :]; dims=2)))
    comparison = only(c for c in fit.ladder if c.name === :population_hmc).comparison
    curvature = only(r for r in fit.rungs if r.name === :population_curvature)
    @test comparison.mean ≈ vec(mean(hmc.draws[p.population, :]; dims=2))
    @test comparison.standard_deviations ≈ vec(std(hmc.draws[p.population, :]; dims=2))
    @test comparison.mean_shift_in_standard_errors ≈
        (comparison.mean .- fit.point) ./ curvature.standard_errors
    @test comparison.standard_deviation_ratio ≈
        comparison.standard_deviations ./ curvature.standard_errors
    exact = only(r for r in fit.rungs if r.rung == 6)
    @test size(exact.draws, 1) == length(p.u) && size(exact.draws, 2) >= 100
    @test fit.ladder[end].started_from == 5
    @test fit.draws === exact.draws
    # the exact chain's population means sit near the Laplace HMC's (a Normal
    # model: the Laplace is exact)
    @test maximum(abs, vec(mean(exact.draws[p.population, :]; dims=2)) .-
        vec(mean(hmc.draws[p.population, :]; dims=2))) < 0.5
end

@stestset "a count model: the plan at rung 2's point, and a later call continues" begin
    d = LADDER_DATA
    rk = RKBRMI(@brm (; d.a, d.b, d.x, y=d.count) begin
        log(mu) ~ 1 + x + (1 | a) + (1 | b)
        y ~ Poisson(mu)
    end)
    first2 = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=2)
    @test [r.rung for r in first2.rungs] == [1, 2]
    @test first2.ladder[2].replanned isa Bool
    @test first2.ladder[2].converged
    # re-planned or not, the problem integrates what the plan admits
    admitted = Set(t.grouping for t in first2.plan.terms if t.admitted)
    @test Set(first2.problem.integrate) == admitted
    more = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=4, draws=20,
        continue_from=first2, rng=Xoshiro(4))
    @test [r.rung for r in more.rungs] == [1, 2, 3, 4]
    @test more.rungs[2] === first2.rungs[2] && more.point == first2.point
    @test size(more.draws) == (length(more.problem.u), 20)

    # An explicitly supplied plan remains fixed when a later call omits the
    # integration keyword; the default :auto must not change that choice.
    frozen = first2.plan
    first1 = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=1, integrate=frozen)
    fixed2 = brm_ladder_fit(rk; ad_backend=NLME_AD, upto=2, continue_from=first1)
    @test fixed2.plan === frozen && !fixed2.automatic
    @test !fixed2.ladder[2].replanned
    @test_throws ArgumentError brm_ladder_fit(rk; ad_backend=NLME_AD, upto=7)
end
