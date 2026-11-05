# Sparse embedded Laplace over native GMRF fields: model-scope and one field
# per `@plate` cell. With Normal observations the Laplace is exact, so the
# integrated target equals the dense marginal y ~ N(0, σ² A Σ Aᵀ + s² I) with
# Σ the field's FEM Matérn × OU × channel covariance, plus the population
# priors, in value and native gradient.
include("gmrf_laplace_oracles.jl")
using Random

@testset "Sparse Laplace integrates a model-scope gmrf field exactly" begin
    y = [0.4, -0.3, 0.2, 0.1]
    for sampled_correlation in (false, true)
        input = merge(data, (; y))
        model = if sampled_correlation
            input = (; (k => v for (k, v) in pairs(input) if k !== :Omega)...)
            @brm input begin
                Omega ~ LKJCholesky(2, 2.0)
                field ~ 0 + gmrf((east, north), time, channel;
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
                length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(0.5, 0.8)
                length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.7, 0.6)
                y ~ Normal(field, 0.2)
            end
        else
            @brm input begin
                field ~ 0 + gmrf((east, north), time, channel;
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
                length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(0.5, 0.8)
                length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.7, 0.6)
                y ~ Normal(field, 0.2)
            end
        end
        rk, _, _ = gmrf_rk_build(model)
        declaration = :gmrf_field_east_north_time_channel
        target = brm_laplace_problem(rk; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse),
            integrate=(declaration,))
        names = target.names
        @test !any(n -> startswith(string(n), "$declaration.field"), names)
        support = BRM.brm_gmrf_support(data.xy, data.triangles, data.time, data.channel)
        A = gmrf_read_matrix(BRM.brm_gmrf_rows(support, data.east, data.north, data.time,
            data.channel), support.layout.ncoordinates)
        function oracle(φ)
            a, b, c = (φ[coordinate(names, "$declaration.$h")]
                for h in ("sigma", "spatial_length", "temporal_length"))
            sigma, ell_s, ell_t = exp(a), exp(b), exp(c)
            lp = logpdf(Exponential(0.7), sigma) + a + logpdf(LogNormal(0.5, 0.8), ell_s) + b +
                logpdf(LogNormal(0.7, 0.6), ell_t) + c
            if sampled_correlation
                rho = tanh(φ[coordinate(names, "Omega.1")])
                lp += 2log1p(-rho^2) - logbeta(0.5, 2.0)
                L = [1.0 0.0; rho sqrt(1 - rho^2)]
            else
                L = Matrix(cholesky(Symmetric(data.Omega)).L)
            end
            lp + gmrf_marginal(y, A, sigma, gmrf_dense_covariance(support, ell_s, ell_t, L), 0.2)
        end
        φ = 0.1 .* cos.(1:length(names))
        for point in (zeros(length(names)), φ)
            @test LDP.logdensity(target, point) ≈ oracle(point) atol=1e-7 rtol=1e-9
        end
        gmrf_laplace_gradient_check(target, oracle, φ)
    end
end

@testset "Sparse Laplace integrates one gmrf field per plate cell exactly" begin
    y = [0.4, -0.3, 0.2, 0.1]
    input = merge(data, (; unit=[10, 10, 20, 20], y))
    cells = Dict(10 => [1, 2], 20 => [3, 4])
    for sampled_correlation in (false, true)
        rows = sampled_correlation ? (; (k => v for (k, v) in pairs(input) if k !== :Omega)...) : input
        model = gmrf_rk_cell_model(; sampled_correlation)
        model = BRM._brm_rebind_brmi(model, rows)
        rk, _, _ = gmrf_rk_build(model)
        declaration = :gmrf_field_east_north_time_channel
        target = brm_laplace_problem(rk; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse),
            integrate=(declaration,))
        names = target.names
        @test !any(n -> startswith(string(n), "$declaration.field"), names)
        function oracle(φ)
            s = φ[coordinate(names, "hyper_scale")]
            hyper_scale = exp(s)
            lp = logpdf(Exponential(0.7), hyper_scale) + s
            if sampled_correlation
                rho = tanh(φ[coordinate(names, "Omega.1")])
                lp += 2log1p(-rho^2) - logbeta(0.5, 2.0)
                L = [1.0 0.0; rho sqrt(1 - rho^2)]
            else
                L = Matrix(cholesky(Symmetric(data.Omega)).L)
            end
            for (cell, label) in enumerate((10, 20))
                h = φ[coordinate(names, "hyper.$cell")]
                hyper = exp(h)
                lp += logpdf(Exponential(hyper_scale), hyper) + h
                a = φ[coordinate(names, "$declaration.sigma.$cell")]
                c = φ[coordinate(names, "$declaration.temporal_length.$cell")]
                sigma, ell_t = exp(a), exp(c)
                lp += logpdf(Exponential(hyper), sigma) + a + logpdf(LogNormal(0.7, 0.6), ell_t) + c
                r = cells[label]
                support = BRM.brm_gmrf_support(data.xy, data.triangles,
                    sort!(unique(input.time[r])), input.channel[r])
                A = gmrf_read_matrix(BRM.brm_gmrf_rows(support, input.east[r], input.north[r],
                    input.time[r], input.channel[r]), support.layout.ncoordinates)
                lp += gmrf_marginal(y[r], A, sigma,
                    gmrf_dense_covariance(support, 0.4, ell_t, L), 0.2)
            end
            lp
        end
        φ = 0.1 .* cos.(1:length(names))
        for point in (zeros(length(names)), φ)
            @test LDP.logdensity(target, point) ≈ oracle(point) atol=1e-7 rtol=1e-9
        end
        gmrf_laplace_gradient_check(target, oracle, φ)
        # The plan's importance diagnostic weighs each cell block by its exact
        # conditional, the field's prior included. With Normal rows that
        # conditional is the Gaussian the proposal draws from, so all weights
        # are equal and every block keeps all its draws.
        state = BRM._brm_spl_mode!(target, φ)
        ess = BRM._brm_plan_ess(target, state, declaration, 200, 0.5, Xoshiro(18))
        @test ess.below == 0
        @test ess.min ≈ ess.draws rtol=1e-8
        # A local Gaussian diagnostic does not certify arbitrary cell algebra.
        # Automatic planning retains the conservative structural admission rule.
        plan = brm_laplace_plan(rk; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse),
            integrate=:auto)
        entry = only(e for e in plan.terms if e.declaration === declaration)
        @test !entry.admitted && entry.importance_ess === nothing
        @test plan.integrated_dimension == 0
        # Applying a plan deep-copies its prepared problem and shares the read-only
        # backend; a per-chain `deepcopy` copies the backend once. All of them hold
        # the cells' lowered programs, and return what the direct target does.
        ad_backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
        named = brm_laplace_problem(rk; ad_backend,
            integrate=brm_laplace_plan(rk; ad_backend, integrate=(declaration,)))
        automatic = brm_laplace_problem(rk; ad_backend, integrate=:auto)
        chain = deepcopy(target)
        @test named.backend === rk && automatic.backend === rk && chain.backend !== rk
        @test isempty(automatic.integrated)
        @test LDP.dimension(automatic) == length(BRM._rk_layout_coordinate_names(rk))
        full = rk_logdensity_problem(rk; ad_backend)
        u = 0.1 .* cos.(1:LDP.dimension(full))
        @test LDP.logdensity_and_gradient(automatic, u) ==
            LDP.logdensity_and_gradient(full, u)
        reference = LDP.logdensity_and_gradient(target, φ)
        for copied in (named, chain)
            @test copied.names == names
            value, gradient = LDP.logdensity_and_gradient(copied, φ)
            @test value ≈ reference[1] && gradient ≈ reference[2]
        end
    end
end

@testset "Laplace plans, conditional modes and draws of a gmrf field" begin
    y = [0.4, -0.3, 0.2, 0.1]
    input = merge(data, (; y))
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
    rk, _, _ = gmrf_rk_build(model)
    declaration = :gmrf_field_east_north_time_channel
    ad_backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    plan = brm_laplace_plan(rk; ad_backend, integrate=:auto)
    entry = only(e for e in plan.terms if e.declaration === declaration)
    @test entry.admitted && entry.class === :integrated
    target = brm_laplace_problem(rk; ad_backend, integrate=(declaration,))
    names = target.names
    φ = 0.1 .* cos.(1:length(names))
    # Applying the plan deep-copies its prepared problem and shares the read-only
    # backend; a per-chain `deepcopy` copies the backend once. Both copies hold
    # the field's lowered program, whose rebuilt law arguments carry modules.
    planned = brm_laplace_problem(rk; ad_backend, integrate=plan)
    chain = deepcopy(target)
    @test planned.backend === rk && chain.backend !== rk
    reference = LDP.logdensity_and_gradient(target, φ)
    for copied in (planned, chain)
        @test copied.names == names
        value, gradient = LDP.logdensity_and_gradient(copied, φ)
        @test value ≈ reference[1] && gradient ≈ reference[2]
    end
    # Matrix-free algebra over the field: its reference and probes stay fixed,
    # so the gradient is the exact derivative of the returned value, and the
    # value estimates the exact one within its probe error.
    mf = brm_laplace_problem(rk; ad_backend, integrate=(declaration,),
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=16))
    value, gradient = LDP.logdensity_and_gradient(mf, φ)
    diagnostics = mf.algebra.diagnostics[:component_1]
    @test abs(value - reference[1]) <= 4diagnostics.probe_mcse + 1e-8
    # The field's own component takes its time-block reference, at the prior medians.
    prepared = only(mf.algebra.references)
    @test prepared isa BRM._BRMGMRFTimeReference
    field_times = BRM.brm_gmrf_support(data.xy, data.triangles, data.time, data.channel).layout.times
    median_length = exp(0.7)
    if length(field_times) > 1
        gap = (field_times[2] - field_times[1]) / median_length
        @test prepared.decay[2] ≈ exp(-gap) / -expm1(-2gap)
    end
    for j in eachindex(φ)
        e = zeros(length(φ)); e[j] = 1e-5
        @test gradient[j] ≈ (LDP.logdensity(mf, φ .+ e) - LDP.logdensity(mf, φ .- e)) / 2e-5 atol=2e-5 rtol=2e-5
    end
    sigma, ell_s, ell_t = (exp(φ[coordinate(names, "$declaration.$h")])
        for h in ("sigma", "spatial_length", "temporal_length"))
    support = BRM.brm_gmrf_support(data.xy, data.triangles, data.time, data.channel)
    A = gmrf_read_matrix(BRM.brm_gmrf_rows(support, data.east, data.north, data.time,
        data.channel), support.layout.ncoordinates)
    Σ = gmrf_dense_covariance(support, ell_s, ell_t, Matrix(cholesky(Symmetric(data.Omega)).L))
    V = inv(Symmetric(inv(Symmetric(Σ)) + sigma^2 * A' * A / 0.2^2))
    m = V * (sigma * A' * y / 0.2^2)
    all_names = BRM._rk_layout_coordinate_names(rk)
    field = [i for (i, n) in enumerate(all_names) if startswith(string(n), "$declaration.field.")]
    state = BRM._brm_spl_mode!(target, φ)
    @test state.u[field] ≈ m atol=1e-8 rtol=1e-7
    # The mode search's likelihood weight leaves the field's prior whole: at
    # weight zero the density is the prior alone (whose mode is zero), halfway
    # its precision is Q + (h − Q) / 2, and the continuation's stages end at the
    # mode a single stage finds, although Normal rows take one stage here.
    pop = BRM._brm_spl_population(target, φ)
    n = length(target.integrated)
    nc = length(target.components)
    assembled(z, weight) = (fc = zeros(nc); g = zeros(n); h = BRM._brm_spl_precision(target);
        BRM._brm_spl_assemble!(fc, g, h, target, pop.A, pop.C, z; derivatives=true,
            uA=pop.uA, weight); (; fc, g, h))
    zprobe = 0.1 .* sin.(1:n)
    prior = assembled(zprobe, 0.0)
    half = assembled(zprobe, 0.5)
    full = assembled(zprobe, 1.0)
    Q = Matrix(Symmetric(prior.h, :L))
    @test sum(prior.fc) ≈ -BRM._brm_spl_prior_quad(target, zprobe) / 2
    @test prior.g ≈ -Q * zprobe
    @test Matrix(Symmetric(half.h, :L)) ≈ (Q + Matrix(Symmetric(full.h, :L))) / 2
    @test iszero(BRM._brm_spl_prior_mode(target, pop))
    stages = BRM._brm_lap_stages(false)
    search = BRM._brm_spl_search(nc, stages)
    followed = zeros(n)
    for (k, s) in enumerate(stages)
        @test BRM._brm_spl_stage!(target, pop, followed, s, target.tol, search, k).status ===
            :converged
    end
    @test followed ≈ state.z rtol=1e-7
    draws = brm_laplace_draws(target, repeat(φ, 1, 4000); rng=Xoshiro(20261103)).draws
    z = draws[field, :]
    se = sqrt.(diag(V) ./ size(z, 2))
    @test all(abs.(vec(sum(z; dims=2)) ./ size(z, 2) .- m) .<= 5 .* se)
    @test all(abs.(vec(sum(abs2, z .- m; dims=2)) ./ size(z, 2) ./ diag(V) .- 1) .<= 0.15)
end

# Beyond the dense oracle's reach for a whole target: a moderate field against
# the exact algebra, its data on a grid mesh at random positions. The returned
# value must estimate exact within its own probe error, the gradient must
# differentiate the returned value, and it must also estimate exact's gradient.
@testset "A moderate gmrf field under matrix-free algebra agrees with exact algebra" begin
    rng = Xoshiro(20261104)
    k, T, nobs = 8, 10, 160
    xy = permutedims(reduce(hcat, [[(i - 1) / (k - 1), (j - 1) / (k - 1)] for j in 1:k for i in 1:k]))
    id(i, j) = i + (j - 1) * k
    triangles = permutedims(reduce(hcat, vcat([[[id(i, j), id(i + 1, j), id(i + 1, j + 1)],
        [id(i, j), id(i + 1, j + 1), id(i, j + 1)]] for j in 1:(k - 1) for i in 1:(k - 1)]...)))
    time = Float64.(rand(rng, 0:(T - 1), nobs)); time[1:T] .= 0:(T - 1)
    input = (; east=0.02 .+ 0.96 .* rand(rng, nobs), north=0.02 .+ 0.96 .* rand(rng, nobs),
        time, y=randn(rng, nobs), xy, triangles)
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(1.0)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.2), 0.5)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(5.0), 0.5)
        y ~ Normal(field, 0.5)
    end
    rk, _, _ = gmrf_rk_build(model)
    declaration = :gmrf_field_east_north_time
    ad_backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    exact = brm_laplace_problem(rk; ad_backend, integrate=(declaration,), solver=:exact)
    mf = brm_laplace_problem(rk; ad_backend, integrate=(declaration,), solver=:matrixfree,
        matrixfree=MatrixFreeLaplace(; probes=32))
    @test only(mf.algebra.references) isa BRM._BRMGMRFTimeReference
    names = mf.names
    φ = [endswith(string(n), "sigma") ? log(0.7) : endswith(string(n), "spatial_length") ? log(0.25) :
        log(4.0) for n in names]
    value, gradient = LDP.logdensity_and_gradient(mf, φ)
    reference, reference_gradient = LDP.logdensity_and_gradient(exact, φ)
    d = mf.algebra.diagnostics[:component_1]
    @test abs(value - reference) <= 4d.probe_mcse + 1e-8
    @test d.max_solve_iterations < 80
    @test d.ritz[1] > 0
    for j in eachindex(φ)
        e = zeros(length(φ)); e[j] = 1e-5
        @test gradient[j] ≈ (LDP.logdensity(mf, φ .+ e) - LDP.logdensity(mf, φ .- e)) / 2e-5 atol=2e-5 rtol=2e-5
    end
    @test norm(gradient - reference_gradient) <= 0.1 * norm(reference_gradient) + 2.0
end

@testset "Two gmrf fields read by the same rows integrate through a star" begin
    rng = Xoshiro(20261105)
    k, T, nobs = 6, 6, 80
    xy = permutedims(reduce(hcat, [[(i - 1) / (k - 1), (j - 1) / (k - 1)] for j in 1:k for i in 1:k]))
    id(i, j) = i + (j - 1) * k
    triangles = permutedims(reduce(hcat, vcat([[[id(i, j), id(i + 1, j), id(i + 1, j + 1)],
        [id(i, j), id(i + 1, j + 1), id(i, j + 1)]] for j in 1:(k - 1) for i in 1:(k - 1)]...)))
    time = Float64.(rand(rng, 0:(T - 1), nobs)); time[1:T] .= 0:(T - 1)
    input = (; east=0.02 .+ 0.96 .* rand(rng, nobs), north=0.02 .+ 0.96 .* rand(rng, nobs),
        time, y=randn(rng, nobs), xy, triangles)
    model = @brm input begin
        slow ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(slow, gmrf((east, north), time)) ~ Exponential(1.0)
        length_scale(slow, gmrf((east, north), time), space) ~ LogNormal(log(0.5), 0.3)
        length_scale(slow, gmrf((east, north), time), time) ~ LogNormal(log(5.0), 0.3)
        fast ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(fast, gmrf((east, north), time)) ~ Exponential(1.0)
        length_scale(fast, gmrf((east, north), time), space) ~ LogNormal(log(0.2), 0.3)
        length_scale(fast, gmrf((east, north), time), time) ~ LogNormal(log(1.5), 0.3)
        y ~ Normal(slow + fast, 0.5)
    end
    rk, _, _ = gmrf_rk_build(model)
    declarations = (:gmrf_slow_east_north_time, :gmrf_fast_east_north_time)
    ad_backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    exact = brm_laplace_problem(rk; ad_backend, integrate=declarations, solver=:exact)
    mf = brm_laplace_problem(rk; ad_backend, integrate=declarations, solver=:matrixfree,
        matrixfree=MatrixFreeLaplace(; probes=32))
    @test only(mf.algebra.references) isa BRM._BRMGMRFStarReference
    φ = 0.2 .* cos.(1:length(mf.names))
    value, gradient = LDP.logdensity_and_gradient(mf, φ)
    reference, reference_gradient = LDP.logdensity_and_gradient(exact, φ)
    d = mf.algebra.diagnostics[:component_1]
    @test abs(value - reference) <= 4d.probe_mcse + 1e-8
    @test d.max_solve_iterations < 200
    for j in eachindex(φ)
        e = zeros(length(φ)); e[j] = 1e-5
        @test gradient[j] ≈ (LDP.logdensity(mf, φ .+ e) - LDP.logdensity(mf, φ .- e)) / 2e-5 atol=2e-5 rtol=2e-5
    end
    @test norm(gradient - reference_gradient) <= 0.5 * norm(reference_gradient) + 5.0
end
