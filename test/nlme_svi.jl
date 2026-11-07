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
include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics, WarmupHMC

include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))

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

# The minibatch split of `m` at the packed point `u`: the global term plus
# every subject's term equals the full RK target in value, in the globals'
# gradient and in each subject's gradient. Returns the terms.
function check_svi_split(rk, m, u)
    t = BRM._brm_svi_target(m)
    @test sort(vcat(t.globals, reduce(vcat, t.units))) == eachindex(u)
    g = u[t.globals]
    P, gP = BRM._brm_svi_global_term(t, g)
    gg = zeros(length(g))
    terms = map(eachindex(t.units)) do i
        gz = zeros(length(t.units[i]))
        value = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
        (; value, gg=copy(gg), gz)
    end
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    @test P + sum(x -> x.value, terms) ≈ value rtol = 1e-12
    @test gP .+ sum(x -> x.gg, terms) ≈ gradient[t.globals] rtol = 1e-10
    for (i, block) in enumerate(t.units)
        @test terms[i].gz ≈ gradient[block] rtol = 1e-10
    end
    (; t, P, gP, terms, value, gradient)
end

@stestset "the minibatch split equals the RK target and the unit density" begin
    rng = Xoshiro(1)
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(rng, 12)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    # A point with non-unit scales and correlation, so the cells' gradients
    # in those coordinates are exercised.
    u = m.unit .+ 0.4 .* randn(rng, length(m.unit))
    (; t, P, gP, terms, value, gradient) = check_svi_split(rk, m, u)
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

@stestset "work-weighted native target and public fit" begin
    rng = Xoshiro(117)
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(rng, 3)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    u = m.unit .+ 0.2randn(rng, length(m.unit))
    (; t, P, gP, terms, value, gradient) = check_svi_split(rk, m, u)
    p = BRM._brm_svi_unit_sampler(:work, [1.0, 7, 40]).probabilities
    expected_value, expected_gradient = 0.0, zeros(length(u))
    for pair in Iterators.product(1:3, 1:3)
        batch = BRM._brm_svi_weighted_batch(pair, p)
        v, grad = P, zeros(length(u))
        grad[t.globals] .= gP
        for (k, i) in enumerate(batch.indices)
            weight = batch.weights[k]
            v += weight * terms[i].value
            grad[t.globals] .+= weight .* terms[i].gg
            grad[t.units[i]] .+= weight .* terms[i].gz
        end
        probability = prod(p[i] for i in pair)
        expected_value += probability * v
        expected_gradient .+= probability .* grad
    end
    @test expected_value ≈ value rtol=1e-11
    @test expected_gradient ≈ gradient rtol=1e-10
    for method in (:uniform, :work), dense_limit in (64, 0)
        fit = brm_svi_fit(m; rng=Xoshiro(118), unit_sampling=method,
            batch_size=3, max_iterations=6, ndraws=2,
            unit_dense_limit=dense_limit, unit_rank=1)
        sampling = fit.diagnostics.sampling
        @test sampling.method == method
        @test sum(sampling.draws) == 18
        @test fit.diagnostics.unit_gradient_calls == 2sum(sampling.visits)
        @test all(sampling.probabilities .> 0)
        @test all(isfinite, fit.draws)
        state = brm_svi_checkpoint(fit)
        @test state.format == 3
        update = brm_svi_fit(state, svi_gauss_model(svi_gauss_data(Xoshiro(119), 3));
            rng=Xoshiro(120), unit_sampling=method, max_iterations=2, ndraws=0)
        @test update.diagnostics.sampling.method == method
        @test brm_svi_checkpoint(update).updates == 1
        @test update.diagnostics.unit_families == fit.diagnostics.unit_families
    end
    @test_throws ArgumentError brm_svi_fit(m; unit_sampling=:unknown)
end

@stestset "a non-finite unit evaluation is named before the iteration changes q" begin
    # Snag connected-svi-un-45d04d2a: a non-finite unit gradient reached the
    # dense natural-gradient step and surfaced as LAPACK's "matrix contains
    # Infs or NaNs" after the global Adam step had already moved q.
    rng = Xoshiro(3)
    rk = RKBRMI(svi_pois_model(svi_pois_data(rng, 8)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    t = BRM._brm_svi_target(m)
    q = BRM._brm_svi_params(t, m.unit[t.globals], MeanFieldGlobal();
        global_scale=0.1, unit_scale=1.0)
    # exp of the linear predictor overflows at this finite point.
    q.M[3] .= 1e4
    before = deepcopy(q)
    opt = BRM._brm_svi_adam(q, 0.01)
    e = try
        BRM._brm_svi_iteration!(Xoshiro(4), t, q, opt, [5, 3], 0.01, 0.1)
        nothing
    catch err
        err
    end
    @test e isa BRMSVINonfiniteError
    @test length(e.terms) == 1
    p = only(e.terms)
    @test p.term === :unit && p.unit == 3 && p.draw_finite
    @test :value_plus in p.quantities && :value_minus in p.quantities
    # Nothing moved: the global factor, the optimizer and the other unit.
    @test opt.t == 0 && q.μ == before.μ && q.ρ == before.ρ
    @test q.M[5] == before.M[5] && q.Λ[5] == before.Λ[5]
    @test occursin("unit 3", sprint(showerror, e))
    # The run-level loop stamps the failing iteration.
    c = BRM._brm_svi_controls(length(t.units); batch_size=length(t.units),
        max_iterations=10, lr_global=0.01, unit_step=0.1, polish_lr=0.05,
        window=nothing, patience=10, min_windows=40, polish_windows=20)
    run = BRM._brm_svi_run(Xoshiro(5), q, c, t)
    run.iteration = 7
    @test_throws BRMSVINonfiniteError BRM._brm_svi_advance!(run, t, c)
    stamped = try BRM._brm_svi_advance!(run, t, c) catch err err end
    @test stamped.iteration == 8
end

@stestset "a unit step far from its data's scale stays positive definite and converges" begin
    # A unit whose rows pin it 30 to 3e5 times more tightly than its N(0, 1)
    # start, as a probe's year of daily rows does (snag
    # brm-svi-fit-posd-0fcb769c): the unbounded step multiplied the precision
    # by up to 1e8 in one visit and lost positive definiteness within the
    # first visits. On a Gaussian unit target the conditional posterior is
    # exact: precision P, mode P \ H z★.
    rng = Xoshiro(11)
    d = 5
    Q = Matrix(qr(randn(rng, d, d)).Q)
    H = Symmetric(Q * Diagonal(1e4 .* exp10.(range(-1.5, 1.5; length=d))) * Q')
    zstar = 0.5 .* randn(rng, d)
    P = Symmetric(Matrix(H) + I)
    mode = P \ (H * zstar)
    grad(z) = -(H * (z .- zstar)) .- z
    q = (; M=[zeros(d)], Λ=[Matrix(1.0I, d, d)], U=[Matrix(1.0I, d, d)])
    positive = Bool[]
    # At a constant step a single draw pair leaves the precision fluctuating
    # around its fixed point (a few percent of runs leave [0.25, 4] at any one
    # visit), so the precision is checked as the fit returns it: averaged over
    # the last visits, as polishing averages q.
    average = zeros(d, d)
    for k in 1:150
        ε = randn(rng, d)
        δ = BRM._brm_svi_unit_offset(q, 1, ε)
        g₊, g₋ = grad(q.M[1] .+ δ), grad(q.M[1] .- δ)
        BRM._brm_svi_unit_step!(q, 1, (g₊ .+ g₋) ./ 2, (g₊ .- g₋) ./ 2, ε, 0.1)
        push!(positive, isposdef(Symmetric(q.Λ[1])))
        k > 100 && (average .+= (q.Λ[1] .- average) ./ (k - 100))
    end
    @test all(positive)
    @test q.U[1] * q.U[1]' ≈ q.Λ[1]
    λ = eigvals(Symmetric(average), P)
    @test 0.25 < minimum(λ) && maximum(λ) < 4
    e = q.M[1] .- mode
    @test sqrt(dot(e, P * e)) < 0.1
end

@stestset "a unit's curvature estimate is the positive part of the Stein estimate" begin
    rng = Xoshiro(21)
    for d in (1, 2, 5, 27), _ in 1:20
        A = randn(rng, d, 2d)
        U = LowerTriangular(Matrix(cholesky(Symmetric(A * A' + d * I)).L))
        gd, ε = randn(rng, d), randn(rng, d)
        λ, e = BRM._brm_svi_unit_curvature(U, gd, ε)
        w = -(U \ gd)
        F = eigen(Symmetric((w * ε' + ε * w') ./ 2))
        @test λ .* (e * e') ≈ F.vectors * Diagonal(max.(F.values, 0.0)) * F.vectors' atol = 1e-10
        # A matched factor's estimate is εεᵀ, which has no negative part.
        λ, e = BRM._brm_svi_unit_curvature(U, -(U * ε), ε)
        @test λ .* (e * e') ≈ ε * ε'
    end
    # No curvature information is zero, not an error; a NaN still fails loudly.
    U = LowerTriangular(Matrix(1.0I, 3, 3))
    @test BRM._brm_svi_unit_curvature(U, zeros(3), [1.0, 2.0, 3.0]) == (0.0, zeros(3))
    @test BRM._brm_svi_unit_curvature(U, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, zeros(3))
    @test_throws ArgumentError BRM._brm_svi_unit_curvature(U, [NaN, 0.0, 0.0], ones(3))
end

@stestset "a unit factor's weak direction recovers on a Gaussian target" begin
    # The true sd along v is 1; every other direction starts at its target. On
    # this exact Gaussian target the full rank-two estimate left v's sd between
    # 155 and 3170 after 1100 visits from the two wider starts, on all five seeds
    # tried, and at 340 and 1820 from the narrowest on two of the five.
    d = 27
    spectrum = vcat(1.0, exp.(range(log(2.0), log(700.0); length=d - 1)))
    for seed in 1:3, start_sd in (5.0, 36.7, 817.0)
        rng = Xoshiro(seed)
        Q = Matrix(qr(randn(rng, d, d)).Q)
        P = Symmetric(Q * Diagonal(spectrum) * Q')
        v = Q[:, 1]
        start = copy(spectrum)
        start[1] = 1 / start_sd^2
        Λ = Matrix(Symmetric(Q * Diagonal(start) * Q'))
        q = (; M=[start_sd / 8 .* v], Λ=[Λ], U=[Matrix(cholesky(Symmetric(Λ)).L)])
        sds = Float64[]
        for k in 1:400
            ε = randn(rng, d)
            δ = BRM._brm_svi_unit_offset(q, 1, ε)
            g₊, g₋ = -(P * (q.M[1] .+ δ)), -(P * (q.M[1] .- δ))
            BRM._brm_svi_unit_step!(q, 1, (g₊ .+ g₋) ./ 2, (g₊ .- g₋) ./ 2, ε, 0.1)
            k > 200 && push!(sds, norm(LowerTriangular(q.U[1]) \ v))
        end
        @test 0.5 < median(sds) < 2
        @test maximum(sds) < 5
        @test abs(dot(q.M[1], v)) < 1
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
            u = m.unit .+ 0.3 .* randn(rng, length(m.unit))
            @test shared ⊆ check_svi_split(rk, m, u).t.globals
            # Each probe's cell is its share of the lockstep likelihoods: one
            # AR(1) series per probe, attributed to the probe owning its rows.
            @test [only(c) for c in only(m.cell_query).positions] == 1:6
            θ, σ = u[m.theta], u[m.sigma]
            H = reduce(hcat, [u[b] for b in m.view.subject_coordinates])
            @test [brm_nlme_loglikelihood(m, i, θ, σ, H[:, i]) for i in axes(H, 2)] ≈
                brm_nlme_loglikelihoods(m, θ, σ, H) rtol = 1e-12
            # Reuse the same prepared queries across local moves, a shared
            # move, and a return to the original point. Both value and AD
            # paths must refresh shared intermediates without stale values.
            for point in (u, u .+ 0.03, u)
                θ, σ = point[m.theta], point[m.sigma]
                H = reduce(hcat, [point[b] for b in m.view.subject_coordinates])
                values, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
                for i in (1, 4)
                    value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[:, i])
                    @test value ≈ values[i] rtol=1e-10
                    @test gradient ≈ gradients[:, i] atol=1e-8 rtol=1e-9
                    @test brm_nlme_loglikelihood(m, i, θ, σ, H[:, i]) ≈ values[i] rtol=1e-10
                end
            end
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

@stestset "flat-row grouped observations: each row is a cell of its subject" begin
    # No `@plate`: the response is observed elementwise on interleaved rows of
    # unequal count, and every row's density belongs to its row's subject.
    rng = Xoshiro(11)
    rk = RKBRMI(nlme_pk(NLME_DATA))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    @test only(m.cell_query).names == (:y,)
    check_svi_split(rk, m, m.unit .+ 0.3 .* randn(rng, length(m.unit)))
    result = brm_svi_fit(rk; rng=Xoshiro(12), batch_size=2, ndraws=10,
        max_iterations=200, min_windows=1)
    @test size(result.draws) == (length(m.view.coordinates), 10)
    @test all(isfinite, result.draws)
end

# Subjects owning 1 to 3 nested units with 2 or 3 rows each, interleaved:
# subject blocks of different lengths.
function svi_nested_data(rng, N; mu=0.5, sp=0.6, su=0.4, sigma=0.3)
    subject = ["s$i" for i in 1:N]
    rows = [(i, "u$i.$k") for i in 1:N for k in 1:mod1(i, 3) for _ in 1:(2 + i % 2)]
    rows = rows[randperm(rng, length(rows))]
    p = Dict(s => sp * randn(rng) for s in subject)
    units = Dict(unit => su * randn(rng) for unit in unique(last.(rows)))
    obs_subject = ["s$(first(r))" for r in rows]
    unit = last.(rows)
    y = [mu + p[s] + units[v] + sigma * randn(rng) for (s, v) in zip(obs_subject, unit)]
    (; subject, obs_subject, unit, y)
end

const SVI_NESTED = (; subject_group=:subject, nested_groups=(unit=:obs_subject,))

@stestset "nested units: structural footprints and target split" begin
    rng = Xoshiro(13)
    rk = RKBRMI(nested_model(NESTED_DATA))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, SVI_NESTED...)
    @test sort(unique(length.(m.view.subject_coordinates))) == [2, 3, 4]
    # The model prepares no compact sampler until a gradient is needed.
    # Structural cell supports cover every derivative at changed points,
    # including coordinates whose derivative is zero at the base point.
    @test m.cell_gradient.state[] === nothing
    st = BRM._brm_nlme_cell_state(m)
    globals = sort!(vcat(m.theta, m.sigma))
    for (i, block) in enumerate(m.view.subject_coordinates)
        selection = sort!(vcat(globals, block))
        slots = [searchsortedfirst(selection, j) for j in block]
        points = [m.unit[selection] .+ 0.2 .* randn(rng, length(selection))
            for _ in 1:2]
        gradient = zeros(length(selection))
        for (g, cell) in enumerate(m.cell_query)
            sampler = compact_cell_oracle(m.backend, cell.names, m.unit,
                selection)
            for (k, p) in enumerate(cell.positions[i])
                touched = falses(length(block))
                for active in points
                    cell_value_and_gradient!(sampler, gradient,
                        m.unit, active, selection, p)
                    touched .|= .!iszero.(gradient[slots])
                end
                @test issubset(block[touched], st.reads[i][g][k])
                footprint_query = BRM._rk_nlme_cell_footprint_query(cell.query)
                for active in points
                    # A footprint kernel takes the packed Vector it was
                    # prepared from, as `_brm_nlme_structural_reads` passes it.
                    point = copy(m.unit)
                    point[selection] = active
                    support = BRM._rk_nlme_cell_footprint(footprint_query, point, p)
                    @test Set(st.reads[i][g][k]) == Set(intersect(block, support))
                end
            end
        end
    end
    # SVI has its own partition workspace and activates the covariance
    # coordinates too. The complete target split below checks those gradients.
    t = BRM._brm_svi_target(m)
    @test t.plan.sampler !== st.plan.sampler
    check_svi_split(rk, m, m.unit .+ 0.3 .* randn(rng, length(m.unit)))
    # A regional panel: nested districts, municipalities and district-years,
    # completed (non-centred) covariate values, and district-year totals.
    panel_rk = RKBRMI(panel(PANEL); completions=:noncentered)
    panel_m = brm_nlme_model(panel_rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    check_svi_split(panel_rk, panel_m,
        panel_m.unit .+ 0.1 .* randn(rng, length(panel_m.unit)))
end

@stestset "nested units: subject blocks of different lengths fit" begin
    # The fit against NUTS: unit factors of every length settle.
    rk = RKBRMI(nested_model(svi_nested_data(Xoshiro(14), 30)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, SVI_NESTED...)
    reference = svi_nuts_reference(rk, m; seed=30)
    result = brm_svi_fit(rk; SVI_NESTED..., rng=Xoshiro(15), batch_size=6, ndraws=2000)
    @test result.diagnostics.converged
    @test [length(M) for M in result.diagnostics.q.M] == length.(m.view.subject_coordinates)
    globals = svi_moments(result.draws, reference, result.diagnostics.globals)
    subjects = svi_moments(result.draws, reference,
        setdiff(axes(reference, 1), result.diagnostics.globals))
    @test maximum(globals.mean_error) < 1.0
    @test all(<(1.2), globals.sd_ratio)
    @test mean(subjects.mean_error) < 0.25
    @test 0.75 < mean(subjects.sd_ratio) < 1.25
    # refused: a prepared model already fixes its subject groupings.
    @test_throws ArgumentError brm_svi_fit(m; SVI_NESTED..., max_iterations=1)
end
