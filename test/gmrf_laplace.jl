# Sparse embedded Laplace over native GMRF fields: model-scope and one field
# per `@plate` cell. With Normal observations the Laplace is exact, so the
# integrated target equals the dense marginal y ~ N(0, σ² A Σ Aᵀ + s² I) with
# Σ the field's FEM Matérn × OU × channel covariance, plus the population
# priors, in value and native gradient.
include("gmrf_laplace_oracles.jl")

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
    end
end

using Random
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
    draws = brm_laplace_draws(target, repeat(φ, 1, 4000); rng=Xoshiro(20261103)).draws
    z = draws[field, :]
    se = sqrt.(diag(V) ./ size(z, 2))
    @test all(abs.(vec(sum(z; dims=2)) ./ size(z, 2) .- m) .<= 5 .* se)
    @test all(abs.(vec(sum(abs2, z .- m; dims=2)) ./ size(z, 2) ./ diag(V) .- 1) .<= 0.15)
end
