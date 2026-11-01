# brm_laplace_plan and integrate = :auto (src/nlme_laplace_plan.jl): which
# latent terms the sparse Laplace integrates. Checked: Normal-only terms are
# integrated without a diagnostic; count terms are gated by their per-block
# importance ESS, which is exactly `draws` on a Gaussian term; the floor admits
# or keeps them sampled; a term on a dispersion is never integrated;
# `integrate` and `sample` override (a term on a dispersion too); `@plate`
# cells are gated; a plan passed to
# brm_laplace_problem is taken as given.
# RUN: julia --project=test test/laplace_plan.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

# 6 levels of `a` crossed with 4 of `b`, 2 rows per cell (48 rows).
const PLAN_DATA = let rng = Xoshiro(11)
    a = repeat(["a$i" for i in 1:6], inner=8)
    b = repeat(["b$i" for i in 1:4], outer=12)
    x = randn(rng, length(a))
    ea = Dict(l => 0.7randn(rng) for l in unique(a))
    eb = Dict(l => 0.5randn(rng) for l in unique(b))
    η = [0.4 + 0.6x[i] + ea[a[i]] + eb[b[i]] for i in eachindex(a)]
    (; a, b, x, y=η .+ 0.3 .* randn(rng, length(a)),
        count=[rand(rng, Poisson(exp(0.2 + 0.4e))) for e in η])
end

plan_of(rk; kwargs...) = brm_laplace_plan(rk; ad_backend=NLME_AD, kwargs...)
entry(plan, name) = only(t for t in plan.terms if t.grouping === name)

@stestset "stratified terms keep their identity and remain sampled" begin
    d = (; g=repeat(["a", "b", "c", "d"], inner=2),
        stratum=repeat(["small", "large"], inner=4),
        x=[-0.8, 0.3, -0.2, 0.7, 0.4, -0.5, 0.9, -0.1],
        y=[-0.2, 0.6, 0.1, 0.9, -0.4, 0.3, 0.7, -0.6])
    stratified = RKBRMI(@brm d begin
        mu ~ 1 + (1 + x | gr(g; by=stratum))
        y ~ Normal(mu, 0.7)
    end)
    plan = plan_of(stratified)
    t = only(plan.terms)
    @test t.grouping === :g && t.kind === :ranef
    @test t.class === :never && !t.admitted && t.importance_ess === nothing
    @test occursin("stratified", t.reason)
    @test occursin(string(t.declaration), sprint(show, MIME"text/plain"(), plan))
    @test plan_of(stratified; sample=(t.declaration,)).terms == plan.terms
    # These refusals are the documented explicit-integration boundary; the
    # automatic planner must first return the complete term inventory.
    for name in (:g, t.declaration)
        @test_throws "stratified" plan_of(stratified; integrate=(name,))
    end

    # An ordinary block over the SAME grouping must still be integrated.
    # Filtering by grouping alone would incorrectly select the stratified one.
    mixed = RKBRMI(@brm d begin
        mu ~ 1 + (1 | ordinary | g) + (0 + x | gr(g; by=stratum))
        y ~ Normal(mu, 0.7)
    end)
    plan = plan_of(mixed)
    @test length(plan.terms) == 2
    @test length(unique(t.declaration for t in plan.terms)) == 2
    sampled = only(t for t in plan.terms if !t.admitted)
    integrated = only(t for t in plan.terms if t.admitted)
    @test sampled.class === :never && occursin("stratified", sampled.reason)
    @test integrated.class === :integrated && integrated.grouping === :g
    p = brm_laplace_problem(mixed; ad_backend=NLME_AD, integrate=:auto)
    @test p.plan.terms == plan.terms
    @test length(p.integrated) == 4
    # Sampled scales times sampled effects are nonlinear in the population;
    # do not build a row-by-population design to rediscover that fact.
    @test !p.affine && isempty(p.jptr)
    names = BRM._rk_layout_coordinate_names(mixed)
    @test all(i -> !BRM._brm_nlme_owned(names[i], sampled.declaration), p.integrated)
    @test all(i -> BRM._brm_nlme_owned(names[i], integrated.declaration), p.integrated)
    full = rk_logdensity_problem(mixed; ad_backend=NLME_AD)
    # Conditional on the sampled stratum scales and slopes, the four ordinary
    # intercepts have independent Gaussian conditionals. Integrate the full
    # model's quadratic exactly, independently of the sparse planner.
    for scale in (0.2, 0.5)
        phi = scale .* randn(Xoshiro(15), length(p.population))
        u = zeros(length(names))
        u[p.population] = phi
        f0 = LogDensityProblems.logdensity(full, u)
        exact = f0
        for i in p.integrated
            u[i] = 1.0
            plus = LogDensityProblems.logdensity(full, u)
            u[i] = -1.0
            minus = LogDensityProblems.logdensity(full, u)
            u[i] = 0.0
            g = (plus - minus) / 2
            h = 2 * f0 - plus - minus
            @test h > 0
            exact += (g^2 / h + log(2pi) - log(h)) / 2
        end
        @test LogDensityProblems.logdensity(p, phi) ≈ exact atol=1e-9 rtol=1e-9
    end
end

@stestset "Normal terms are integrated, count terms gated by their diagnostic" begin
    d = PLAN_DATA
    normal = RKBRMI(@brm (; d.a, d.b, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    plan = plan_of(normal)
    @test plan isa BRMLaplacePlan
    @test all(t -> t.class === :integrated && t.admitted && t.importance_ess === nothing,
        plan.terms)
    p = brm_laplace_problem(normal; ad_backend=NLME_AD, integrate=:auto)
    @test p.plan isa BRMLaplacePlan
    @test sort(p.integrate) == [:a, :b]
    φ = 0.3 .* randn(Xoshiro(1), length(p.population))
    q = brm_laplace_problem(normal; ad_backend=NLME_AD, integrate=(:a, :b))
    @test LogDensityProblems.logdensity(p, φ) ≈ LogDensityProblems.logdensity(q, φ)
    # on a Gaussian term the proposal is the exact conditional: every weight
    # is equal and the importance ESS is the number of draws
    state = BRM._brm_spl_mode!(q, φ)
    ess = BRM._brm_plan_ess(q, state, only(t.declaration for t in plan.terms
        if t.grouping === :a), 50, 0.5, Xoshiro(2))
    @test ess.min ≈ 50 rtol=1e-8
    @test ess.below == 0 && ess.blocks == 6

    counts = RKBRMI(@brm (; d.a, d.b, d.x, y=d.count) begin
        log(mu) ~ 1 + x + (1 | a) + (1 | b)
        y ~ Poisson(mu)
    end)
    plan = plan_of(counts; ess_floor=0.2, draws=100)
    for g in (:a, :b)
        t = entry(plan, g)
        @test t.class === :gated && t.admitted
        @test t.importance_ess.blocks == (g === :a ? 6 : 4)
        @test 0 < t.importance_ess.min <= t.importance_ess.median <= 100
    end
    # a floor no non-Gaussian block reaches keeps them sampled
    strict = plan_of(counts; ess_floor=0.9999, draws=100)
    @test all(t -> t.class === :sampled && !t.admitted, strict.terms)
    @test_throws BRMNLMEViewError brm_laplace_problem(counts; ad_backend=NLME_AD,
        integrate=strict)
    # a plan is taken as given
    p = brm_laplace_problem(counts; ad_backend=NLME_AD, integrate=plan)
    @test p.plan === plan && sort(p.integrate) == [:a, :b]
    @test occursin("2 of 2 latent terms integrated", sprint(show, MIME"text/plain"(), plan))
end

@stestset "overrides, a term on a dispersion, and cells" begin
    d = PLAN_DATA
    counts = RKBRMI(@brm (; d.a, d.b, d.x, y=d.count) begin
        log(mu) ~ 1 + x + (1 | a) + (1 | b)
        y ~ Poisson(mu)
    end)
    plan = plan_of(counts; sample=(:b,), integrate=(:a,))
    @test entry(plan, :b).class === :sampled && !entry(plan, :b).admitted
    @test entry(plan, :a).admitted && entry(plan, :a).importance_ess === nothing
    p = brm_laplace_problem(counts; ad_backend=NLME_AD, integrate=plan)
    @test p.integrate == [:a]
    @test_throws BRMNLMEViewError plan_of(counts; sample=(:nope,))

    shape = RKBRMI(@brm (; d.a, d.b, d.x, y=d.count) begin
        log(mu) ~ 1 + x + (1 | a)
        log(shape) ~ 1 + (1 | b)
        y ~ NegativeBinomial2(mu, shape)
    end)
    plan = plan_of(shape; draws=50)
    @test entry(plan, :b).class === :never && !entry(plan, :b).admitted
    @test occursin("shape or dispersion", entry(plan, :b).reason)
    @test entry(plan, :a).class in (:gated, :sampled)
    # named in `integrate`, it is integrated all the same
    forced = plan_of(shape; integrate=(:b,), draws=50)
    @test entry(forced, :b).admitted && occursin("named in `integrate`", entry(forced, :b).reason)

    cells = RKBRMI(nested_model(NESTED_DATA))
    plan = plan_of(cells; draws=50)
    @test all(t -> occursin("`@plate` cells", t.reason), plan.terms)
end
