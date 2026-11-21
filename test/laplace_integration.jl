# brm_laplace_integration (src/laplace_integration.jl): deterministic
# integration over the population coordinates of an embedded-Laplace target,
# rung 5c of brm_ladder_fit (design `0r49iwd`). Checked:
#   - the CCD's fractions are resolution V and its weights reproduce a
#     standard Gaussian's first two moments exactly; the grid reproduces them
#     up to its truncation;
#   - on Gaussian models whose Laplace marginal is exact, the weighted design,
#     the marginal tables, the grid's log evidence and the effect draws agree
#     with brute-force quadrature of the exact marginal (grid at d = 2, CCD
#     at d = 3);
#   - formula coefficients integrate as blocks (`integrate = (..., :coefficients)`):
#     exact on a Normal model, differentiable on a count model, admitted by
#     named plans and never by `:auto`, refused for non-Normal priors, with
#     priors read from data fields, `constant`s and per-level data vectors
#     integrating exactly as their literal twins;
#   - the refusals name their reason;
#   - its speed-ups change no result (snag `brm-laplace-inte-e0991a0f`): fixed
#     coefficient C maps equal probed ones, design values started from the
#     centre's modes equal those of a search from the prior's, the draws reuse
#     the design's searches and equal searching each drawn point again, and
#     brm_laplace_mode's first L-BFGS step stays where the searches converge.
# RUN: julia --project=test test/laplace_integration.jl
include(joinpath(@__DIR__, "fitting_packages.jl"))
using WarmupHMC   # loads Optim, which brm_laplace_mode needs
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

@stestset "CCD fractions are resolution V" begin
    sizes = Dict(5 => 16, 6 => 32, 7 => 64, 8 => 64, 9 => 128, 10 => 128)
    for d in 2:10
        F = BRM._brm_ccd_fraction(d)
        @test size(F, 1) == d
        @test all(x -> x == 1 || x == -1, F)
        d >= 5 && @test size(F, 2) == sizes[d]
        d <= 4 && @test size(F, 2) == 2^d
        # main effects and every two-factor interaction are mutually orthogonal
        X = vcat(F, (F[i:i, :] .* F[j:j, :] for i in 1:d for j in (i + 1):d)...)
        @test X * X' == size(F, 2) * I
        @test allunique(eachcol(F))
    end
end

@stestset "design weights reproduce a standard Gaussian's moments" begin
    gaussian(z) = -sum(abs2, z) / 2
    for d in 1:8
        out = BRM._brm_lint_ccd(gaussian, 0.0, d, 1.1)
        lw = out.log_weight .+ out.log_density
        w = exp.(lw .- maximum(lw)); w ./= sum(w)
        @test sum(exp.(out.log_weight)) ≈ 1
        @test out.z * w ≈ zeros(d) atol=1e-12
        @test out.z .* w' * out.z' ≈ Matrix(I, d, d) atol=1e-12
        @test size(out.z, 2) == 1 + 2d + (d == 1 ? 0 : size(BRM._brm_ccd_fraction(d), 2))
    end
    for d in 1:2
        out = BRM._brm_lint_grid(gaussian, 0.0, d, 0.75, 6.0)
        w = exp.(out.log_density); w ./= sum(w)
        @test isempty(out.truncated)
        @test out.z * w ≈ zeros(d) atol=1e-12
        # the drop-6 truncation removes about e^-6 of the mass
        @test out.z .* w' * out.z' ≈ Matrix(I, d, d) rtol=0.03
        @test all(v -> v >= -6.0, out.log_density)
    end
end

const LINT_DATA = (; y=[0.2, -0.4, 0.8, 0.3, 1.1, 0.6, -0.2], idx=[1, 2, 1, 4, 3, 3, 5], n=5)

# location and scale outer (d = 2), plus an offset for d = 3, or the scale
# alone (d = 1) with the location fixed at 0.2
function lint_model(; offset=false, single=false)
    data = LINT_DATA
    model = if single
        @brm data begin
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(0.2, scale)
            prediction = field[idx]
            y ~ Normal(prediction, 0.6)
        end
    elseif offset
        @brm data begin
            offset ~ Normal(0, 1)
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = offset + field[idx]
            y ~ Normal(prediction, 0.6)
        end
    else
        @brm data begin
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = field[idx]
            y ~ Normal(prediction, 0.6)
        end
    end
    rk = RKBRMI(model)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,))
    names = offset ? (:offset, :location, :scale) : single ? (:scale,) : (:location, :scale)
    read = BRM._rk_value_reader(rk, names)
    location(x) = single ? 0.2 : x.location
    B = [Float64(j == data.idx[i]) for i in eachindex(data.y), j in 1:data.n]
    values(phi) = (u = zeros(length(p.u)); u[p.population] .= phi; read(u))
    function exact(phi)
        x = values(phi)
        o = offset ? x.offset : 0.0
        covariance = 0.6^2 * I + x.scale^2 * B * B'
        r = logpdf(MvNormal(o .+ location(x) .* B * ones(data.n), Matrix(covariance)), data.y)
        r += logpdf(LogNormal(0, 0.4), x.scale) + log(x.scale)
        single || (r += logpdf(Normal(0.2, 0.8), x.location))
        offset && (r += logpdf(Normal(0, 1), x.offset))
        r
    end
    # the field's conditional mean given the outer values
    function field_mean(phi)
        x = values(phi)
        o = offset ? x.offset : 0.0
        Q = I / x.scale^2 + B' * B / 0.36
        Q \ (fill(location(x) / x.scale^2, data.n) .+ B' * (data.y .- o) / 0.36)
    end
    (; rk, p, exact, field_mean)
end

# Brute-force quadrature of the exact marginal on a fine rectangular grid in
# the target's population coordinates, ±7 curvature standard errors.
function lint_reference(m, mode, se; n=81)
    d = length(mode)
    axes1 = [range(mode[i] - 7se[i], mode[i] + 7se[i]; length=n) for i in 1:d]
    pts = vec(collect(Iterators.product(axes1...)))
    lv = [m.exact(collect(t)) for t in pts]
    w = exp.(lv .- maximum(lv)); Z = sum(w); w ./= Z
    X = reduce(hcat, collect.(pts))
    μ = X * w
    sd = sqrt.((X .- μ) .^ 2 * w)
    logZ = maximum(lv) + log(Z) + sum(log(step(a)) for a in axes1)
    field = sum(w[k] * m.field_mean(X[:, k]) for k in eachindex(w))
    (; mean=μ, sd, logZ, field)
end

@stestset "grid at d = 2 agrees with quadrature of the exact marginal" begin
    m = lint_model()
    p = m.p
    @test length(p.population) == 2
    out = brm_laplace_integration(p; draws=4000, rng=Xoshiro(7))
    @test out.design === :grid
    @test LogDensityProblems.logdensity(p, out.point) ≈ m.exact(out.point) atol=1e-8
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))))
    @test sum(out.weights) ≈ 1
    @test out.diagnostics.nonfinite == 0 && out.diagnostics.max_gain <= 1e-8
    @test out.mean ≈ ref.mean atol=0.02 * maximum(ref.sd)
    @test out.sd ≈ ref.sd rtol=0.03
    @test out.log_evidence ≈ ref.logZ atol=0.01
    for (i, mg) in enumerate(out.marginals)
        @test mg.name === p.names[i]
        @test length(mg.x) == 401 && all(>=(0), mg.density)
        # a kernel mixture that keeps the design's moments
        @test mg.mean ≈ out.mean[i] atol=1e-3 * out.sd[i]
        @test mg.sd ≈ out.sd[i] rtol=1e-3
        @test mg.mean ≈ ref.mean[i] atol=0.02 * ref.sd[i]
    end
    # draws: population rows are design points; effects are joint draws
    @test size(out.draws) == (length(p.u), 4000)
    D = out.draws
    @test all(any(col ≈ D[p.population, k] for col in eachcol(out.points)) for k in 1:50)
    @test vec(mean(D[p.integrated, :]; dims=2)) ≈ ref.field atol=0.05
    @test all(>(0), vec(std(D[p.integrated, :]; dims=2)))
end

@stestset "CCD at d = 3 agrees with quadrature of the exact marginal" begin
    m = lint_model(; offset=true)
    p = m.p
    @test length(p.population) == 3
    out = brm_laplace_integration(p; draws=2000, rng=Xoshiro(9))
    @test out.design === :ccd
    @test size(out.z, 2) == 15
    @test out.log_evidence === nothing
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))); n=41)
    @test out.mean ≈ ref.mean atol=0.15 * maximum(ref.sd)
    @test out.sd ≈ ref.sd rtol=0.15
    for (i, mg) in enumerate(out.marginals)
        @test mg.mean ≈ out.mean[i] atol=1e-3 * out.sd[i]
        @test mg.sd ≈ out.sd[i] rtol=1e-3
        @test mg.mean ≈ ref.mean[i] atol=0.15 * ref.sd[i]
        @test mg.sd ≈ ref.sd[i] rtol=0.15
    end
    @test vec(mean(out.draws[p.integrated, :]; dims=2)) ≈ ref.field atol=0.08
    # the grid on the same target, forced, is closer
    grid = brm_laplace_integration(p; point=out.point, design=:grid, draws=0)
    @test grid.design === :grid && size(grid.draws, 2) == 0
    @test grid.mean ≈ ref.mean atol=0.03 * maximum(ref.sd)
    @test grid.sd ≈ ref.sd rtol=0.03
end

@stestset "grid at d = 1: the table interpolates the exact marginal" begin
    m = lint_model(; single=true)
    p = m.p
    @test length(p.population) == 1
    out = brm_laplace_integration(p; draws=500, rng=Xoshiro(5))
    @test out.design === :grid
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))); n=401)
    @test out.mean ≈ ref.mean atol=0.01 * ref.sd[1]
    @test out.sd ≈ ref.sd rtol=0.01
    @test out.log_evidence ≈ ref.logZ atol=0.01
    mg = only(out.marginals)
    @test mg.mean ≈ ref.mean[1] atol=0.02 * ref.sd[1]
    @test mg.sd ≈ ref.sd[1] rtol=0.03
    # the table is the exact marginal, normalized, between the grid's ends
    x = mg.x[200]
    logmass = ref.logZ
    @test mg.density[200] ≈ exp(m.exact([x]) - logmass) rtol=0.02
    @test vec(mean(out.draws[p.integrated, :]; dims=2)) ≈ ref.field atol=0.05
end

const LINT_COEF_DATA = (; a=repeat(["a1", "a2", "a3", "a4"], 5), x=randn(Xoshiro(1), 20),
    y=randn(Xoshiro(2), 20), count=rand(Xoshiro(3), Poisson(2.0), 20))

lint_fd(f, x; h=1e-5) = [(f(x .+ h .* (1:length(x) .== j)) - f(x .- h .* (1:length(x) .== j))) / 2h
    for j in eachindex(x)]

@stestset "formula coefficients as integrated blocks (integrate = :coefficients)" begin
    d = LINT_COEF_DATA
    rk = RKBRMI(@brm (; d.a, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        y ~ Normal(mu, sigma)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:a, :coefficients))
    coefficient = [Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")]
    @test [b.declaration for b in p.blocks if b.group === :coefficients] == coefficient
    @test length(p.population) == 2 && !any(in(p.names), coefficient)
    # Gaussian in the coefficients: the marginal is the :a-only marginal with
    # the coefficients integrated in closed form (one Newton step, exact)
    pa = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:a,))
    ib = [findfirst(==(c), pa.names) for c in coefficient]
    io = [findfirst(==(n), pa.names) for n in p.names]
    function closed(phi)
        full(β) = (x = zeros(length(pa.population)); x[io] .= phi; x[ib] .= β; x)
        gβ(β) = last(LogDensityProblems.logdensity_and_gradient(pa, full(β)))[ib]
        H = hcat([(gβ(1e-4 .* e) .- gβ(-1e-4 .* e)) ./ 2e-4 for e in ([1.0, 0], [0, 1.0])]...)
        H = (H + H') / 2
        β = -(H \ gβ(zeros(2)))
        LogDensityProblems.logdensity(pa, full(β)) + log(2π) - logdet(-H) / 2
    end
    for phi in ([0.1, -0.3], [0.4, 0.2], [-0.5, 0.5])
        v, g = LogDensityProblems.logdensity_and_gradient(p, phi)
        @test v ≈ closed(phi) atol=1e-9
        @test g ≈ lint_fd(x -> LogDensityProblems.logdensity(p, x), phi) atol=1e-6
    end
    # a count model: finite marginal and its gradient
    rc = RKBRMI(@brm (; d.a, d.x, y=d.count) begin
        log(mu) ~ 1 + x + (1 | a)
        y ~ Poisson(mu)
    end)
    pc = brm_laplace_problem(rc; ad_backend=NLME_AD, integrate=(:a, :coefficients))
    @test length(pc.population) == 1
    v, g = LogDensityProblems.logdensity_and_gradient(pc, [-0.2])
    @test isfinite(v)
    @test g ≈ lint_fd(x -> LogDensityProblems.logdensity(pc, x), [-0.2]) atol=1e-6
    # plans: named, the coefficients are terms and admitted; :auto is unchanged
    plan = brm_laplace_plan(rc; ad_backend=NLME_AD, integrate=(:a, :coefficients), draws=8)
    @test [t.declaration for t in plan.terms if t.grouping === :coefficients] ==
        [Symbol("pop_log_mu.beta_pop.1"), Symbol("pop_log_mu.beta_pop.2")]
    @test all(t -> t.admitted, plan.terms)
    pp = brm_laplace_problem(rc; ad_backend=NLME_AD, integrate=plan)
    @test LogDensityProblems.logdensity(pp, [-0.2]) ≈ v
    auto = brm_laplace_plan(rc; ad_backend=NLME_AD, draws=8)
    @test !any(t -> t.grouping === :coefficients, auto.terms)
    # refusals: a coefficient whose prior is not a constant Normal, a model without
    rt = RKBRMI(@brm (; d.a, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, sigma)
    end)
    @test_throws "has the prior `Cauchy(0.0, 1.0)`" brm_laplace_problem(rt;
        ad_backend=NLME_AD, integrate=(:a, :coefficients))
    sampled = brm_laplace_plan(rt; ad_backend=NLME_AD, integrate=(:a,),
        sample=(:coefficients,), draws=8)
    @test only(t for t in sampled.terms if t.declaration === Symbol("pop_mu.beta_pop_2")).class === :never
    m = lint_model()
    @test_throws "no population coefficient" brm_laplace_problem(m.rk; ad_backend=NLME_AD,
        integrate=(:field, :coefficients))
    # a lone intercept is a scalar coefficient (its record has no index)
    rl = RKBRMI(@brm (; d.a, y=d.count) begin
        log(mu) ~ 1 + (1 | a)
        y ~ Poisson(mu)
    end)
    pl = brm_laplace_problem(rl; ad_backend=NLME_AD, integrate=(:a, :coefficients))
    @test length(pl.population) == 1
    @test only(b.declaration for b in pl.blocks if b.group === :coefficients) in
        BRM._rk_layout_coordinate_names(rl)
    vl, gl = LogDensityProblems.logdensity_and_gradient(pl, [0.1])
    @test isfinite(vl)
    @test gl ≈ lint_fd(x -> LogDensityProblems.logdensity(pl, x), [0.1]) atol=1e-6
    # 5c's target integrates them: the integration agrees with the outer-coefficient one
    point = brm_laplace_mode(p).draws[p.population, 1]
    out = brm_laplace_integration(p; point, draws=2000, rng=Xoshiro(4))
    @test out.design === :grid && size(out.draws, 1) == length(p.u)
    pa_point = brm_laplace_mode(pa).draws[pa.population, 1]
    outer = brm_laplace_integration(pa; point=pa_point, design=:ccd, draws=2000, rng=Xoshiro(4))
    ia = [findfirst(==(n), BRM._rk_layout_coordinate_names(rk)) for n in coefficient]
    @test vec(mean(out.draws[ia, :]; dims=2)) ≈ vec(mean(outer.draws[ia, :]; dims=2)) atol=0.1
end

# Prior values are runtime model inputs (user principle `runtime-model-dimensions`),
# so a coefficient prior read from the data integrates as its literal twin does
# (snag brm-coefficients-b99ddbca), and so do a factor block's per-level priors.
@stestset "coefficient priors from data integrate as their literal twins" begin
    d = LINT_COEF_DATA
    twin = RKBRMI(@brm (; d.a, d.x, d.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        effect(mu, Intercept) ~ Normal(0.5, 10.0)
        effect(mu, x) ~ Normal(-0.25, 2.0)
        y ~ Normal(mu, sigma)
    end)
    fields = RKBRMI(@brm (; d.a, d.x, d.y, loc=0.5, s=10.0, locx=-0.25, sx=2.0) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        effect(mu, Intercept) ~ Normal(loc, s)
        effect(mu, x) ~ Normal(locx, sx)
        y ~ Normal(mu, sigma)
    end)
    constants = RKBRMI(@brm (; d.a, d.x, d.y, half=5.0) begin
        sigma ~ Exponential(1)
        s = constant(2 * half)
        sx = constant(half - 3.0)
        mu ~ 1 + x + (1 | a)
        effect(mu, Intercept) ~ Normal(0.5, s)
        effect(mu, x) ~ Normal(-0.25, sx)
        y ~ Normal(mu, sigma)
    end)
    pt = brm_laplace_problem(twin; ad_backend=NLME_AD, integrate=(:a, :coefficients))
    for rk in (fields, constants)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:a, :coefficients))
        @test p.names == pt.names
        for phi in ([0.1, -0.3], [0.4, 0.2])
            v, g = LogDensityProblems.logdensity_and_gradient(p, phi)
            vt, gt = LogDensityProblems.logdensity_and_gradient(pt, phi)
            @test v == vt && g == gt
        end
        plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:a, :coefficients), draws=8)
        @test all(t -> t.admitted, plan.terms)
    end
    # a factor block's per-level priors from a data vector: one block per level,
    # exact on a Normal model (closed form in the coefficients)
    g = repeat(["g1", "g2", "g3"], 7)[1:20]
    rf = RKBRMI(@brm (; d.a, g, d.y, w_scale=[1.0, 2.0]) begin
        sigma ~ Exponential(1)
        mu ~ 1 + g + (1 | a)
        effect(mu, g) .~ Normal.(0.0, w_scale)
        y ~ Normal(mu, sigma)
    end)
    p = brm_laplace_problem(rf; ad_backend=NLME_AD, integrate=(:a, :coefficients))
    coefficient = [b.declaration for b in p.blocks if b.group === :coefficients]
    @test coefficient == [Symbol("mu_g.1"), Symbol("mu_g.2"), :mu_Intercept]
    @test [c.prior for c in BRM._brm_spl_coefficient_records(rf)] ==
        [:(Normal(0.0, 1.0)), :(Normal(0.0, 2.0)), :(Normal(0.0, 1.0))]
    pa = brm_laplace_problem(rf; ad_backend=NLME_AD, integrate=(:a,))
    ib = [findfirst(==(c), pa.names) for c in coefficient]
    io = [findfirst(==(n), pa.names) for n in p.names]
    function closed(phi)
        full(β) = (x = zeros(length(pa.population)); x[io] .= phi; x[ib] .= β; x)
        gβ(β) = last(LogDensityProblems.logdensity_and_gradient(pa, full(β)))[ib]
        E = Matrix(1.0I, 3, 3)
        H = hcat([(gβ(1e-4 .* E[:, j]) .- gβ(-1e-4 .* E[:, j])) ./ 2e-4 for j in 1:3]...)
        H = (H + H') / 2
        β = -(H \ gβ(zeros(3)))
        LogDensityProblems.logdensity(pa, full(β)) + 3 / 2 * log(2π) - logdet(-H) / 2
    end
    for phi in ([0.1, -0.3], [0.4, 0.2])
        v, gr = LogDensityProblems.logdensity_and_gradient(p, phi)
        @test v ≈ closed(phi) atol=1e-9
        @test gr ≈ lint_fd(x -> LogDensityProblems.logdensity(p, x), phi) atol=1e-6
    end
    # another per-level family is refused by name, as a coefficient's is
    rc = RKBRMI(@brm (; d.a, g, d.y, w_scale=[1.0, 2.0]) begin
        sigma ~ Exponential(1)
        mu ~ 1 + g + (1 | a)
        effect(mu, g) .~ Cauchy.(0.0, w_scale)
        y ~ Normal(mu, sigma)
    end)
    @test_throws "the coefficient `g_lvl_2` of `mu` has the prior `Cauchy(0.0, 1.0)`" brm_laplace_problem(rc;
        ad_backend=NLME_AD, integrate=(:a, :coefficients))
end

@stestset "refusals name their reason" begin
    m = lint_model()
    p = m.p
    point = brm_laplace_mode(p).draws[p.population, 1]
    @test_throws "max_dimension = 1" brm_laplace_integration(p; point, max_dimension=1)
    @test_throws "design = :foo" brm_laplace_integration(p; point, design=:foo)
    @test_throws "f0 = 1.0" brm_laplace_integration(p; point, f0=1.0)
    @test_throws "not negative definite" brm_laplace_integration(p; point,
        curvature=(; hessian=[1.0 0.0; 0.0 1.0], positive_definite=false))
    @test_throws "point has length 3" brm_laplace_integration(p; point=zeros(3))
end

# A Poisson model whose marginal's gradient at the prior median is 150 in the
# log scale. On epil such a first L-BFGS step went to log scales where a mode
# search ran to its iteration limit (snag `brm-laplace-inte-e0991a0f`); the
# step rule is checked on its own below, with a line search's state.
mutable struct LintLineSearchState
    s::Vector{Float64}
    alpha::Float64
end
@stestset "integration speed-ups change no result" begin
    first_step = Base.get_extension(BRM, :BayesianRegressionModelsOptimExt)._BRMUnitFirstStep
    rule = first_step(true)
    state = LintLineSearchState([30.0, 40.0], NaN)
    rule(nothing, state, 0.0, 0.0, nothing)
    @test state.alpha ≈ 1 / 50           # the first trial step: unit length
    rule(nothing, state, 0.0, 0.0, nothing)
    @test state.alpha == 1.0             # later ones: L-BFGS's own length
    short = LintLineSearchState([0.3, 0.4], NaN)
    first_step(true)(nothing, short, 0.0, 0.0, nothing)
    @test short.alpha == 1.0             # a first step already shorter is kept
    rng = Xoshiro(11)
    ng, per = 30, 4
    g = repeat(1:ng; inner=per)
    x = randn(rng, ng * per)
    b = 0.6 .* randn(rng, ng)
    count = [rand(rng, Poisson(exp(2.5 + 0.3 * x[i] + b[g[i]]))) for i in eachindex(g)]
    rk = RKBRMI(@brm (; g, x, count, sd_scale=1 / -log(0.01)) begin
        log(mu) ~ 1 + x + (1 | g)
        sd(mu, g) ~ Exponential(sd_scale)
        effect(mu, :) ~ Normal(0.0, 10.0)
        count ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:g, :coefficients))
    # the coefficients' C maps are fixed and copied; the random effects' move
    # with their scale and are probed at every point
    @test p.cfixed == [b.group === :coefficients for b in p.blocks]
    q = deepcopy(p)
    fill!(q.cfixed, false)
    for φ in ([-1.0], [0.5], [-3.0])
        v, gr = LogDensityProblems.logdensity_and_gradient(p, φ)
        vq, gq = LogDensityProblems.logdensity_and_gradient(q, φ)
        @test v ≈ vq rtol=1e-12
        @test gr ≈ gq rtol=1e-10
    end
    md = brm_laplace_mode(p)
    @test md.diagnostics.converged
    point = md.draws[p.population, 1]
    out = brm_laplace_integration(p; point, rng=Xoshiro(5))
    cold = deepcopy(p)
    @test [LogDensityProblems.logdensity(cold, out.points[:, k]) for k in axes(out.points, 2)] ≈
        out.log_density rtol=1e-11
    ref, _ = BRM._brm_lint_draws(cold, out.points, out.weights, 1000, Xoshiro(5), 0)
    @test out.draws ≈ ref rtol=1e-9 atol=1e-9
end
