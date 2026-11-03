# GMRF Laplace acceptance that waits on ReactiveKernels deliveries; kept failing
# until they land, never weakened:
# - one field per `@plate` cell, exact against the dense per-cell marginals:
#   RK snag prepare-strict-o-332ad6b9 (strict observation-cell partitions of a
#   per-cell joint field and its per-cell vertex reads);
# - applying a Laplace plan, which deep-copies its prepared problem: RK snag
#   prepared-query-h-b7bd890a (a prepared query holding a kernel source op's
#   module cannot be deep-copied).
include("gmrf_laplace_oracles.jl")

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

@testset "A Laplace plan's target over a gmrf field is the direct target" begin
    y = [0.4, -0.3, 0.2, 0.1]
    input = merge(data, (; y))
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
        y ~ Normal(field, 0.2)
    end
    rk, _, _ = gmrf_rk_build(model)
    ad_backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    plan = brm_laplace_plan(rk; ad_backend, integrate=:auto)
    planned = brm_laplace_problem(rk; ad_backend, integrate=plan)
    direct = brm_laplace_problem(rk; ad_backend,
        integrate=(:gmrf_field_east_north_time_channel,))
    φ = 0.1 .* cos.(1:LDP.dimension(direct))
    @test planned.names == direct.names
    @test LDP.logdensity_and_gradient(planned, φ)[2] ≈ LDP.logdensity_and_gradient(direct, φ)[2]
end
