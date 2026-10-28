# Minibatch stochastic variational inference over an RK-lowered population
# model (src/nlme_svi.jl), on synthetic random-effect models.
#
# The minibatch objective is checked against the executable RK target (value
# and gradient at a packed point with non-unit covariance), its per-subject
# term against the shared unit density `brm_nlme_unit_problem`, and the fitted
# approximation against long NUTS runs of the same target: means close, and the
# spread of the population and scale coordinates narrower, as variational
# approximations are.
#
# RUN: julia --project=test test/nlme_svi.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics, WarmupHMC

# Correlated random intercept and slope, `nobs` Normal observations per subject.
function svi_gauss_data(rng, N; nobs=5, a=1.0, b=0.5, sa=0.7, sb=0.3, rho=0.3,
        sigma=0.5)
    C = [sa^2 rho*sa*sb; rho*sa*sb sb^2]
    L = cholesky(C).L
    x = [randn(rng, nobs) for _ in 1:N]
    y = map(1:N) do i
        e = L * randn(rng, 2)
        a + e[1] .+ (b + e[2]) .* x[i] .+ sigma .* randn(rng, nobs)
    end
    (; subject=["s$i" for i in 1:N], x, y)
end

svi_gauss_model(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ b[i] .* x[i], sigma)
    end
end

# Poisson counts with a log link and a random intercept.
function svi_pois_data(rng, N; nobs=4, a=0.5, sa=0.6, b=0.3)
    u = sa .* randn(rng, N)
    x = [randn(rng, nobs) for _ in 1:N]
    y = [[rand(rng, Poisson(exp(a + u[i] + b * xj))) for xj in x[i]] for i in 1:N]
    (; subject=["s$i" for i in 1:N], x, y)
end

svi_pois_model(data) = @brm data begin
    a ~ 1 + (1 | subject)
    b ~ Normal(0, 1)
    @plate for i in eachindex(a)
        y[i] ~ Poisson(exp.(a[i] .+ b .* x[i]))
    end
end

# Four NUTS chains of the full RK target, one packed draw per column.
function svi_nuts_reference(rk, m; seed, n_draws=1000, chains=4)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    reduce(hcat, map(1:chains) do c
        run = adaptive_warmup_mcmc(Xoshiro(seed + c), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws)
        convert(Matrix{Float64}, run.posterior_position)
    end)
end

# Each coordinate's mean error in reference sds, and its sd ratio, between SVI
# draws and the reference (both coordinates × draws).
function svi_moments(draws, reference, idx)
    sd = vec(std(reference[idx, :]; dims=2))
    (; mean_error=abs.(vec(mean(draws[idx, :]; dims=2)) .-
        vec(mean(reference[idx, :]; dims=2))) ./ sd,
       sd_ratio=vec(std(draws[idx, :]; dims=2)) ./ sd)
end

@stestset "the minibatch split equals the RK target and the unit density" begin
    rng = Xoshiro(1)
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(rng, 12)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    t = BRM._brm_svi_target(m)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    # A point with non-unit scales and correlation, so the cells' gradients
    # in those coordinates are exercised.
    u = m.unit .+ 0.4 .* randn(rng, length(m.unit))
    g = u[t.globals]
    @test sort(vcat(t.globals, reduce(vcat, t.units))) == eachindex(u)
    P, gP = BRM._brm_svi_global_term(t, g)
    gg, gz = zeros(length(g)), zeros(t.d)
    terms = map(eachindex(t.units)) do i
        value = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
        (; value, gg=copy(gg), gz=copy(gz))
    end
    value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    @test P + sum(x -> x.value, terms) ≈ value rtol = 1e-12
    @test gP .+ sum(x -> x.gg, terms) ≈ gradient[t.globals] rtol = 1e-10
    for (i, block) in enumerate(t.units)
        @test terms[i].gz ≈ gradient[block] rtol = 1e-10
    end
    # Batches of a partition of the subjects, each scaled by N/B, average to
    # the full density and gradient.
    N, B = length(t.units), 4
    batches = collect(Iterators.partition(randperm(rng, N), B))
    @test mean(P + N / B * sum(terms[i].value for i in batch) for batch in batches) ≈
        value rtol = 1e-12
    @test mean(gP .+ N / B .* sum(terms[i].gg for i in batch) for batch in batches) ≈
        gradient[t.globals] rtol = 1e-10
    # The per-subject term is the shared unit density of the two-stage fit at
    # the same population values, in the same standardized coordinates.
    families = BRM._brm_nlme_families(rk, m.view)
    Ω = BRM._brm_nlme_natural(rk, m.view, families, u).omega
    for i in (1, 7)
        p = brm_nlme_unit_problem(m, i, u[m.theta], u[m.sigma], Ω)
        @test LogDensityProblems.logdensity(p, u[t.units[i]]) ≈ terms[i].value rtol = 1e-12
    end
end

@stestset "Gaussian random effects: SVI means match NUTS" begin
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(2), 40)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    reference = svi_nuts_reference(rk, m; seed=10)
    result = brm_svi_fit(m; rng=Xoshiro(3), batch_size=8, ndraws=2000)
    dg = result.diagnostics
    @test dg.converged
    @test result.coordinates == m.view.coordinates
    @test size(result.draws) == (length(m.view.coordinates), 2000)
    globals = svi_moments(result.draws, reference, dg.globals)
    subjects = svi_moments(result.draws, reference,
        setdiff(axes(reference, 1), dg.globals))
    @test maximum(globals.mean_error) < 1.0
    # Mean-field VI narrows the population and scale coordinates; it must not
    # widen them.
    @test all(<(1.2), globals.sd_ratio)
    @test mean(subjects.mean_error) < 0.25
    @test 0.75 < mean(subjects.sd_ratio) < 1.25
    # They are packed RK draws, so the RK readers restore them.
    restored = rk_restore_draws(rk, result.draws)
    @test all(>(0), restored.sigma)
    # The ELBO rose from its first window to its last.
    @test last(dg.elbo_windows) > first(dg.elbo_windows)
end

@stestset "Poisson random intercepts: SVI means match NUTS" begin
    rk = RKBRMI(svi_pois_model(svi_pois_data(Xoshiro(4), 60)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    reference = svi_nuts_reference(rk, m; seed=20)
    for family in (MeanFieldGlobal(), LowRankGlobal(2))
        result = brm_svi_fit(m; rng=Xoshiro(5), batch_size=10, ndraws=2000, family)
        @test result.diagnostics.converged
        globals = svi_moments(result.draws, reference, result.diagnostics.globals)
        subjects = svi_moments(result.draws, reference,
            setdiff(axes(reference, 1), result.diagnostics.globals))
        @test maximum(globals.mean_error) < 1.0
        @test all(<(1.2), globals.sd_ratio)
        @test mean(subjects.mean_error) < 0.25
    end
end

# Daily rows of `G` probes, interleaved, with per-probe positions: a space-time
# panel (snag brm-svi-fit-refu-be4efadf).
function svi_spacetime_data(rng; G=6, T=12)
    probe = String[]; day = Float64[]; c1 = Float64[]; trend = Float64[]
    for d in 1:T, g in 1:G
        push!(probe, "p$g"); push!(day, d); push!(c1, cos(2pi * d / 7)); push!(trend, d / T)
    end
    (; probe, day, c1, trend, y=3.0 .+ 0.5 .* c1 .+ 0.3 .* randn(rng, length(day)),
       probe_x=[100.0 * cos(g) for g in 1:G], probe_y=[100.0 * sin(g) for g in 1:G])
end

# Per-probe random effects, a grouped HSGP over days and AR(1) noise in a
# per-probe cell; `field` is the HSGP term and `extra` the declarations it reads.
svi_spacetime_model(field, extra...) = BRM.brm(quote
    $(extra...)
    mu ~ 1 + c1 + (1 | lv | probe) + (0 + c1 | cc | probe) + $field
    tr ~ 0 + trend + (0 + trend | tt | probe)
    log(sigma) ~ 1 + (1 | ns | probe)
    phi ~ Uniform(-1.0, 1.0)
    @plate for p in groups(probe; order = day)
        m = mu[p] .+ tr[p]
        y[p] ~ AR1Normal(m, sigma[p], phi; time = day[p])
    end
end; mod=@__MODULE__)

@stestset "a space-time panel with a grouped HSGP fits by minibatch SVI" begin
    # Long-format rows with a plate over each probe's rows: one AR(1) series
    # per probe is that probe's cell (snag brm-svi-fit-refu-be4efadf; before,
    # its density could not be attributed and no subject cell existed).
    rng = Xoshiro(8)
    data = svi_spacetime_data(rng)
    omega = :(Omega ~ LKJCholesky(6, 2.0))
    variants = [
        "multigroup coordinates" => svi_spacetime_model(
            :(hsgp(day; k=4, by=probe, cor=multigroup((probe_x, probe_y); k=3)))),
        "per level" => svi_spacetime_model(:(hsgp(day; k=4, by=probe))),
        "cor = Omega" => svi_spacetime_model(:(hsgp(day; k=4, by=probe, cor=Omega)), omega),
        "multigroup(Omega, tau)" => svi_spacetime_model(
            :(hsgp(day; k=4, by=probe, cor=multigroup(Omega, 1.0))), omega),
    ]
    for (tag, builder) in variants
        @testset "$tag" begin
            rk = RKBRMI(builder(data))
            m = brm_nlme_model(rk; ad_backend=NLME_AD)
            names = String.(m.view.coordinates)
            field = findall(n -> startswith(n, "hsgp_day."), names)
            @test !isempty(field)
            # A `cor=` couples the levels, so the whole term is shared; without
            # one each probe's weights are its own unit draws (BRM 0.2.11).
            shared = filter(i -> m.view.role[i] === :shared_latent, field)
            @test all(r -> r in (:shared_latent, :subject_effect), m.view.role[field])
            @test (length(shared) == length(field)) == (tag != "per level")
            @test all(==(:shared_latent),
                m.view.role[findall(n -> startswith(n, "Omega."), names)])
            @test shared ⊆ m.theta
            @test length(m.view.levels) == 6
            t = BRM._brm_svi_target(m)
            full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
            u = m.unit .+ 0.3 .* randn(rng, length(m.unit))
            g = u[t.globals]
            @test shared ⊆ t.globals
            P, gP = BRM._brm_svi_global_term(t, g)
            gg, gz = zeros(length(g)), zeros(t.d)
            terms = map(eachindex(t.units)) do i
                value = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
                (; value, gg=copy(gg), gz=copy(gz))
            end
            value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
            @test P + sum(x -> x.value, terms) ≈ value rtol = 1e-12
            @test gP .+ sum(x -> x.gg, terms) ≈ gradient[t.globals] rtol = 1e-10
            for (i, block) in enumerate(t.units)
                @test terms[i].gz ≈ gradient[block] rtol = 1e-10
            end
            # Each probe's cell is its share of the lockstep likelihoods: one
            # AR(1) series per probe, attributed to the probe owning its rows.
            @test [only(c) for c in only(m.cell_query).positions] == 1:6
            θ, σ = u[m.theta], u[m.sigma]
            H = reduce(hcat, [u[b] for b in m.view.subject_coordinates])
            @test [brm_nlme_loglikelihood(m, i, θ, σ, H[:, i]) for i in axes(H, 2)] ≈
                brm_nlme_loglikelihoods(m, θ, σ, H) rtol = 1e-12
        end
    end
    rk = RKBRMI(last(first(variants))(data))
    result = brm_svi_fit(rk; rng=Xoshiro(9), batch_size=2, ndraws=10,
        max_iterations=200, min_windows=1)
    @test result.coordinates == brm_nlme_view(rk).coordinates
    @test size(result.draws, 2) == 10
    @test all(isfinite, result.draws)
end

@stestset "starts: prior medians, a packed point, or address-keyed inits" begin
    rk = RKBRMI(svi_pois_model(svi_pois_data(Xoshiro(6), 12)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    t = BRM._brm_svi_target(m)
    median = first(BRM._brm_nlme_prior_median_point(m))
    @test BRM._brm_svi_start(m, nothing, nothing) == median
    u = median .+ 0.1
    @test BRM._brm_svi_start(m, u, nothing) == u
    from_inits = BRM._brm_svi_start(m, nothing, (; b=0.7))
    @test from_inits[only(findall(==(:b), m.view.coordinates))] ≈ 0.7
    # refused: a start and inits name two different starting points.
    @test_throws ArgumentError BRM._brm_svi_start(m, u, (; b=0.7))
    @test_throws DimensionMismatch BRM._brm_svi_start(m, u[2:end], nothing)
    # An RKBRMI is prepared with the default reverse-mode AutoEnzyme.
    result = brm_svi_fit(rk; rng=Xoshiro(7), batch_size=4, ndraws=10,
        max_iterations=300, min_windows=1)
    @test size(result.draws) == (length(m.view.coordinates), 10)
    # refused: a prepared model already fixes its AD backend, as brm_nlme_fit
    # refuses it too.
    @test_throws ArgumentError brm_svi_fit(m; ad_backend=NLME_AD, max_iterations=1)
    # refused: a rank-0 low-rank factor is the mean-field family.
    @test_throws ArgumentError LowRankGlobal(0)
end

@stestset "minibatch SVI needs subject cells and flat subject blocks" begin
    # Flat-row grouped observations have no per-subject cell, and nested
    # subject blocks no compact cell gradient yet: capability gaps, todo
    # BayesianRegressionModels:scaling-svi/todos/2026-10-25T22-52-28-193-1mbshme.
    flat = brm_nlme_model(RKBRMI(nlme_pk(NLME_DATA)); ad_backend=NLME_AD)
    @test_broken brm_svi_fit(flat; rng=Xoshiro(1), max_iterations=10, ndraws=5) isa
        NamedTuple
end
