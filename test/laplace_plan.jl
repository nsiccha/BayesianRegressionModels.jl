# brm_laplace_plan and integrate = :auto (src/nlme_laplace_plan.jl): which
# latent terms the sparse Laplace integrates. Checked: Normal-only terms are
# integrated without a diagnostic; count terms are gated by their per-block
# importance ESS, which is exactly `draws` on a Gaussian term; the floor admits
# or keeps them sampled; a term on a dispersion is never integrated;
# `integrate` and `sample` override (a term on a dispersion too); `@plate`
# cells require explicit opt-in; a plan passed to
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

@stestset "direct predictor observations retain their structure inside value plans" begin
    data = (; g=["a","a","b","b"], y=[0,1,1,0], n=fill(3,4), v=[0.1,0.2,-0.1,0.3])
    extra = @brm begin
        nuisance[1:4] .~ Normal.(0,1)
        @plate for i in eachindex(v)
            v[i] ~ Normal(nuisance[i]^2,0.7)
        end
    end
    builders = [@brm(begin
        mu ~ 1 + (1 | g)
        sigma ~ LogNormal(0,0.4)
        y ~ Normal(mu,sigma)
    end), @brm(begin
        logit(p) ~ 1 + (1 | g)
        y ~ Bernoulli(p)
    end), @brm(begin
        logit(p) ~ 1 + (1 | g)
        y ~ Binomial(n,p)
    end), @brm(begin
        shape ~ LogNormal(0,0.5)
        log(mu) ~ 1 + (1 | g)
        y ~ NegativeBinomial2(mu,shape)
    end)]
    for (i,builder) in enumerate(builders)
        rk = RKBRMI(merge(builder,extra)(data))
        plan = plan_of(rk; draws=12,ess_floor=0)
        @test entry(plan,:g).admitted
        @test entry(plan,:g).class === (i == 1 ? :integrated : :gated)
        @test !only(t for t in plan.terms if t.declaration === :nuisance).admitted
    end
    # Normal(exp(eta), sigma) is not affine Normal in the effects, despite a
    # direct predictor reference. The family and link must match together.
    nonlinear = @brm begin
        log(mu) ~ 1 + (1 | g)
        y ~ Normal(mu,0.7)
    end
    plan = brm_laplace_plan(RKBRMI(merge(nonlinear,extra)(data)); ad_backend=nothing)
    @test !entry(plan,:g).admitted
    # A direct predictor reference is not enough when its summands read
    # nonlinear model values, either as offsets or fitted design columns.
    for formula in (@brm(mu ~ 0 + offset(curved)), @brm(mu ~ 0 + curved))
        structure = @brm begin
            eta ~ 1 + (1 | g)
            curved = eta^2
            y ~ Normal(mu,0.7)
        end
        rk = RKBRMI(merge(structure,formula)(data))
        plan = brm_laplace_plan(rk; ad_backend=nothing)
        @test !entry(plan,:g).admitted
        @test plan.prepared === nothing
    end
end

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
    sampled = brm_laplace_problem(counts; ad_backend=NLME_AD, integrate=strict)
    @test isempty(sampled.integrated)
    @test LogDensityProblems.dimension(sampled) == length(sampled.u)
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
    @test all(b -> b.group === :a, plan.prepared.problem.blocks)
    @test !any(plan.prepared.problem.moving)
    # named in `integrate`, it is integrated all the same
    forced = plan_of(shape; integrate=(:b,), draws=50)
    @test entry(forced, :b).admitted && occursin("named in `integrate`", entry(forced, :b).reason)

    cells = RKBRMI(nested_model(NESTED_DATA))
    plan = plan_of(cells; draws=50)
    @test all(t -> occursin("`@plate` cells", t.reason), plan.terms)
    @test all(t -> !t.admitted && t.importance_ess === nothing, plan.terms)
    @test plan.prepared === nothing
end

@stestset "automatic admission is structural across outer points; explicit opt-in remains" begin
    rk = RKBRMI(@brm (; y=[4.0]) begin
        amplitude ~ Normal(0, 1)
        field[1:1] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(amplitude * field[i]^2, 0.3)
        end
    end)
    # At amplitude=0 this conditional is exactly Gaussian, and its local ESS
    # passes. At amplitude=1 it has two separated modes. Neither the point nor
    # an arbitrarily permissive ESS floor can authorize automatic integration.
    names = BRM._rk_layout_coordinate_names(rk)
    for amplitude in (0.0, 1.0)
        at = zeros(length(names)); at[findfirst(==(:amplitude), names)] = amplitude
        plan = brm_laplace_plan(rk; ad_backend=nothing, at, ess_floor=0, draws=20)
        @test only(plan.terms).class === :sampled
        @test !only(plan.terms).admitted && only(plan.terms).importance_ess === nothing
        @test occursin("name it in `integrate`", only(plan.terms).reason)
        @test plan.integrated_dimension == 0 && plan.sampled_dimension == length(names)
        @test plan.prepared === nothing
    end
    auto = plan_of(rk)
    sampled = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=auto)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u = zeros(length(names))
    @test LogDensityProblems.logdensity_and_gradient(sampled, u) ==
        LogDensityProblems.logdensity_and_gradient(full, u)
    forced = plan_of(rk; integrate=:field)
    @test only(forced.terms).admitted && only(forced.terms).importance_ess === nothing
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=forced)
    state = BRM._brm_spl_mode!(p, zeros(length(p.population)))
    @test BRM._brm_plan_ess(p, state, :field, 20, 0.5, Xoshiro(3)).min ≈ 20
    u[findfirst(==(:amplitude), names)] = 1.0
    f0 = LogDensityProblems.logdensity(full, u)
    u[p.integrated] .= 2.0
    plus = LogDensityProblems.logdensity(full, u)
    u[p.integrated] .= -2.0
    @test LogDensityProblems.logdensity(full, u) ≈ plus
    @test plus > f0 + 80
end

@stestset "unknown cells do not discard unrelated certified terms" begin
    d = (; a=["a1","a2","a1","a2"], b=["b1","b1","b2","b2"],
        y=[1,2,1,3], v=[0.2,-0.1,0.3,0.4])
    rk = RKBRMI(@brm d begin
        log(mu) ~ 1 + (1 | a)
        eta ~ 1 + (1 | b)
        y ~ Poisson(mu)
        @plate for i in eachindex(v)
            v[i] ~ Normal(eta[i]^2, 0.5)
        end
    end)
    plan = plan_of(rk; draws=20, ess_floor=0)
    @test entry(plan, :a).admitted && entry(plan, :a).importance_ess !== nothing
    @test !entry(plan, :b).admitted && entry(plan, :b).importance_ess === nothing
    @test all(b -> b.group === :a, plan.prepared.problem.blocks)
    reference = plan_of(rk; sample=:b, draws=20, ess_floor=0)
    @test entry(plan, :a).importance_ess == entry(reference, :a).importance_ess
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
    q = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:a,))
    phi = fill(0.1, length(p.population))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
    reference_value, reference_gradient = LogDensityProblems.logdensity_and_gradient(q, phi)
    @test value ≈ reference_value
    @test gradient ≈ reference_gradient
end

@stestset "recognized logit and log-NB2 rows remain eligible" begin
    d = (; g=["a","a","b","b"], y=[0,1,1,0], n=fill(3,4))
    models = [@brm(d, begin
        logit(p) ~ 1 + (1 | g)
        y ~ Bernoulli(p)
    end), @brm(d, begin
        logit(p) ~ 1 + (1 | g)
        y ~ Binomial(n,p)
    end), @brm(d, begin
        shape ~ LogNormal(0,0.5)
        log(mu) ~ 1 + (1 | g)
        y ~ NegativeBinomial2(mu,shape)
    end)]
    for model in models
        rk = RKBRMI(model)
        for amplitude in (0.0, 0.2)
            at = fill(amplitude, length(BRM._rk_layout_coordinate_names(rk)))
            plan = plan_of(rk; at, draws=16, ess_floor=0)
            @test only(plan.terms).admitted && only(plan.terms).class === :gated
            @test only(plan.terms).importance_ess !== nothing
            @test occursin("recognized log-concave conditional", only(plan.terms).reason)
        end
    end
end
