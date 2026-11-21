# test/fixed_hyperparameters.jl — MCMC over every latent with the
# hyperparameters held at a point (snag `mixed-membership-1a1930a6`, design
# decision `0b9jgl5`): `brm_hyperparameters`, `brm_fixed_hyper_problem`,
# `brm_fixed_hyper_point`, `brm_fixed_hyper_fit`.
#
# Run: julia --startup-file=no --project=test test/fixed_hyperparameters.jl [filter...]
#
# The classification is checked by name on models whose coordinates are known:
# scales, correlation factors, term parameters, LKJ factors and constrained
# scalars are hyperparameters; standardized draws, basis weights, walk
# innovations, coefficients, declared matrices and real-line scalars are not.
# The conditional problem's density and gradient are the full target's at the
# completed point, restricted to the free coordinates.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, Statistics

Random.seed!(20261121)
const S, T, n = 4, 6, 2
const X = repeat(collect(range(0.0, 1.0; length=T)), S)
const SERIES = repeat(1:S, inner=T)
const NOBS = length(X)
const MIX_DATA = (; y=round.(randn(NOBS); digits=3), x=X, s=SERIES, z=round.(randn(NOBS); digits=3),
    xg=repeat(X, n), cg=repeat(1:n, inner=NOBS), p1=collect(1:NOBS),
    p2=collect(NOBS+1:2NOBS), n, S)

# Every coordinate kind the classification distinguishes, in one model: a
# correlated random-effect block, an ungrouped and a level-correlated grouped
# HSGP, an LKJ factor, constrained and real-line scalars, population
# coefficients and a matrix of simplex rows (the mixed-membership shape).
const MIXED = @brm begin
    sigma ~ Exponential(1.0)
    Omega ~ LKJCholesky(n, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    eachrow(theta[1:S, 1:n]) .~ Dirichlet(n, 1.0)
    mu ~ 1 + z + (1 + z | s)
    g ~ 0 + hsgp(xg; by = cg, cor = multigroup(Omega, tau), k = 3)
    u ~ 0 + hsgp(x; k = 3)
    g1 = g[p1]
    g2 = g[p2]
    f = mu .+ u .+ b .* z .+ theta[s, 1] .* g1 .+ theta[s, 2] .* g2
    y ~ Normal(f, sigma)
end

const WALK_DATA = (; y=round.(randn(30); digits=3), x=round.(randn(30); digits=3),
    g=repeat(1:3, inner=10), t=collect(1.0:30), step=repeat(1.0:10.0, 3),
    C=[1 0.3 0.1; 0.3 1 0.2; 0.1 0.2 1])
const CDAR = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + cdar(step; by=g, cor=C)
    ar(:, cdar(step)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end
const RW = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + rw(t)
    sd(:, rw(t)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
names_of(rk) = Symbol.(coordinate_names(rk.model.layout))
starts(prefix) = name -> startswith(String(name), String(prefix))

@stestset "hyperparameters of the mixed-membership model" begin
    rk = RKBRMI(MIXED(MIX_DATA))
    names = names_of(rk)
    hyper = brm_hyperparameters(rk)
    @test issubset(hyper, names)
    @test issorted(map(h -> findfirst(==(h), names), hyper))
    expected = Set(Symbol.(["sigma", "tau", "Omega.1", "hsgp_x.rho_iso", "hsgp_x.sigma",
        "hsgp_xg.rho", "hsgp_xg.sigma", "b_s.tau.1", "b_s.tau.2", "b_s.L.1"]))
    @test Set(hyper) == expected
    # The latents: draws, weights, coefficients, the simplex rows and the
    # real-line scalar `b`.
    latent = setdiff(names, hyper)
    @test :b in latent
    @test all(n -> n in latent, filter(starts("b_s.z."), names))
    @test all(n -> n in latent, filter(starts("hsgp_xg.z."), names))
    @test all(n -> n in latent, filter(starts("hsgp_x.beta_raw."), names))
    @test all(n -> n in latent, filter(starts("theta."), names))
    @test all(n -> n in latent, filter(starts("pop_mu."), names))
end

@stestset "hyperparameters of walks" begin
    cdar = RKBRMI(CDAR(WALK_DATA))
    names = names_of(cdar)
    hyper = brm_hyperparameters(cdar)
    @test :sigma in hyper
    walk = setdiff(hyper, [:sigma])
    # the walk's scale and AR coefficient, not its innovations
    @test length(walk) == 2
    @test !any(h -> count(==('.'), String(h)) > 1, walk)
    @test all(n -> !(n in hyper), filter(n -> count(==('.'), String(n)) > 1, names))
    rw = RKBRMI(RW(WALK_DATA))
    names = names_of(rw)
    hyper = brm_hyperparameters(rw)
    # sigma and the walk's scale; its steps and the coefficients are latent
    @test :sigma in hyper
    @test length(hyper) == 2
    @test all(n -> !(n in hyper), filter(n -> count(==('.'), String(n)) > 1, names))
end

@stestset "the conditional problem is the full target restricted" begin
    rk = RKBRMI(MIXED(MIX_DATA))
    names = names_of(rk)
    full = rk_logdensity_problem(rk; ad_backend=AD)
    d = LogDensityProblems.dimension(full)
    point = collect(range(-0.3, 0.3; length=d))
    p = brm_fixed_hyper_problem(rk, point; ad_backend=AD)
    hyper = brm_hyperparameters(rk)
    @test p.fixed_names == hyper
    @test LogDensityProblems.dimension(p) == d - length(hyper)
    @test names[p.free] == p.names
    @test p.position == point[p.free]
    for shift in (0.0, 0.2)
        x = p.position .+ shift .* collect(range(-1.0, 1.0; length=length(p.free)))
        u = brm_fixed_hyper_point(p, x)
        @test u[p.fixed] == point[p.fixed]
        @test u[p.free] == x
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        full_value, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
        @test value ≈ full_value atol=1e-12
        @test gradient ≈ full_gradient[p.free] atol=1e-10
        @test LogDensityProblems.logdensity(p, x) ≈ full_value atol=1e-12
    end
    # A declaration owning several coordinates fixes all of them; an index works.
    q = brm_fixed_hyper_problem(rk, point; ad_backend=AD, fixed=(:Omega, :sigma, 3))
    @test Set(q.fixed_names) == Set([Symbol("Omega.1"), :sigma, names[3]])
    # Nothing fixed: the full model from the point.
    all_free = brm_fixed_hyper_problem(rk, point; ad_backend=AD, fixed=())
    @test LogDensityProblems.dimension(all_free) == d
    @test LogDensityProblems.logdensity(all_free, point) ≈
        LogDensityProblems.logdensity(full, point) atol=1e-12
    # Refusals name their reason.
    @test_throws "no coordinate of this model" brm_fixed_hyper_problem(rk, point;
        ad_backend=AD, fixed=(:not_a_parameter,))
    @test_throws "nothing to sample" brm_fixed_hyper_problem(rk, point;
        ad_backend=AD, fixed=names)
    @test_throws DimensionMismatch brm_fixed_hyper_problem(rk, point[1:end-1]; ad_backend=AD)
    # A copy owns its target and shares the backend.
    c = deepcopy(p)
    @test c.backend === p.backend && c.target !== p.target
    @test LogDensityProblems.logdensity(c, p.position) ≈ LogDensityProblems.logdensity(p, p.position)
end

# A protocol fixture that records the exact problem passed through the public fit.
struct RecordingDeviceSampler end
function BRM.brm_fixed_hyper_sample(::Val{:gpu}, ::RecordingDeviceSampler, p, rng)
    (; draws=hcat(p.position, p.position .+ 0.25), device=:gpu,
        backend=:test, device_name="protocol fixture")
end

@stestset "an SVI fit as the start, and the fit step" begin
    rk = RKBRMI(MIXED(MIX_DATA))
    names = names_of(rk)
    d = length(names)
    draws = 0.1 .* randn(MersenneTwister(1), d, 50) .+ collect(range(-0.2, 0.2; length=d))
    svi = (; draws, coordinates=copy(names), diagnostics=(;))
    p = brm_fixed_hyper_problem(rk, svi; ad_backend=AD)
    @test p.point ≈ vec(mean(draws; dims=2))
    @test_throws "not this model's packed coordinates" brm_fixed_hyper_problem(rk,
        (; draws, coordinates=reverse(names), diagnostics=(;)); ad_backend=AD)
    # The fit step completes the sampler's free draws with the fixed values.
    calls = Ref(0)
    sampler = (problem, init, rng) -> begin
        calls[] += 1
        @test init == problem.position
        init .+ 0.01 .* randn(rng, length(init), 7)
    end
    fit = brm_fixed_hyper_fit(sampler, rk, svi; ad_backend=AD, rng=MersenneTwister(2))
    @test calls[] == 1
    @test_throws "GPU sampling method" brm_fixed_hyper_fit(sampler, rk, svi; device=:gpu, ad_backend=AD)
    @test calls[] == 1
    @test_throws "device must be" brm_fixed_hyper_fit(sampler, rk, svi; device=:gup, ad_backend=AD)
    for fixed in (brm_hyperparameters(rk), ())
        gpu = brm_fixed_hyper_fit(RecordingDeviceSampler(), rk, svi; device=:gpu, fixed, ad_backend=AD)
        @test gpu.execution == (; device=:gpu, backend=:test, device_name="protocol fixture")
        @test gpu.draws[:, 1] == gpu.point
        @test gpu.draws[gpu.problem.free, 2] == gpu.problem.position .+ 0.25
        @test gpu.draws[gpu.problem.fixed, 2] == gpu.point[gpu.problem.fixed]
    end
    @test size(fit.draws) == (d, 7)
    @test all(fit.draws[fit.problem.fixed, k] == fit.point[fit.problem.fixed] for k in 1:7)
    @test fit.fixed == brm_hyperparameters(rk)
    @test all(isfinite, rk_restore_draws(rk, fit.draws).sigma)
    @test_throws DimensionMismatch brm_fixed_hyper_fit((p, i, r) -> zeros(2, 3), rk, svi;
        ad_backend=AD)
end
