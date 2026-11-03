# Independent gmrf fields in `@plate for p in groups(g)` cells on RKBRMI: each
# cell's field on its own fitted support, its own term hyperparameters, and
# shared or cell-local inputs to its priors, against an independent dense
# covariance oracle (FEM Matérn ν=1 × OU time × channel correlation).
include("gmrf_rk_cell_models.jl")

@testset "RK cell fields: one-channel learned and known hyperparameters" begin
    input = merge(data, (; unit=[10, 20, 10, 10]))
    cells = Dict(10 => [1, 3, 4], 20 => [2])
    for known in (false, true)
        rk, bound, built = gmrf_rk_build(gmrf_plate_model(; known))
        names = BRM._rk_layout_coordinate_names(rk)
        declaration = "gmrf_field_east_north_time"
        @test length(cell_coordinates(names, declaration, 1)) == 15
        @test length(cell_coordinates(names, declaration, 2)) == 5
        @test built.layout.total == (known ? 20 : 26)
        function oracle(u)
            lp = 0.0
            for (cell, label) in enumerate((10, 20))
                z = u[cell_coordinates(names, declaration, cell)]
                if known
                    sigma, ell_s, ell_t = 1.2, 0.4, 1.7
                else
                    a, b, c = (u[coordinate(names, "$declaration.$h.$cell")]
                        for h in ("sigma", "spatial_length", "temporal_length"))
                    sigma, ell_s, ell_t = exp(a), exp(b), exp(c)
                    lp += logpdf(Exponential(0.7), sigma) + a +
                        logpdf(LogNormal(0.5, 0.8), ell_s) + b +
                        logpdf(LogNormal(0.7, 0.6), ell_t) + c
                end
                obs, times = gmrf_cell_rows_logpdf(z, sigma, input, cells[label], nothing)
                lp += obs + gmrf_cell_field_logpdf(z, ell_s, ell_t, ones(1, 1), times)
            end
            lp
        end
        u = 0.05 .* sin.(1:built.layout.total)
        query = prepare_query(built, bound, :sampler)
        for point in (zeros(length(u)), u)
            @test Base.invokelatest(query, point) ≈ oracle(point) atol=2e-8 rtol=2e-10
        end
        gmrf_rk_gradient_check(built, bound, oracle, u)
    end
end

@testset "RK cell fields: channels, shared and cell-local prior inputs" begin
    input = merge(data, (; unit=[10, 10, 20, 20]))
    cells = Dict(10 => [1, 2], 20 => [3, 4])
    for sampled_correlation in (false, true)
        rk, bound, built = gmrf_rk_build(gmrf_rk_cell_model(; sampled_correlation))
        names = BRM._rk_layout_coordinate_names(rk)
        declaration = "gmrf_field_east_north_time_channel"
        @test length(cell_coordinates(names, declaration, 1)) == 20
        @test length(cell_coordinates(names, declaration, 2)) == 20
        function oracle(u)
            s = u[coordinate(names, "hyper_scale")]
            hyper_scale = exp(s)
            lp = logpdf(Exponential(0.7), hyper_scale) + s
            if sampled_correlation
                rho = tanh(u[coordinate(names, "Omega.1")])
                lp += 2log1p(-rho^2) - logbeta(0.5, 2.0)
                L = [1.0 0.0; rho sqrt(1 - rho^2)]
            else
                L = Matrix(cholesky(Symmetric(data.Omega)).L)
            end
            for (cell, label) in enumerate((10, 20))
                h = u[coordinate(names, "hyper.$cell")]
                hyper = exp(h)
                lp += logpdf(Exponential(hyper_scale), hyper) + h
                a = u[coordinate(names, "$declaration.sigma.$cell")]
                c = u[coordinate(names, "$declaration.temporal_length.$cell")]
                sigma, ell_t = exp(a), exp(c)
                lp += logpdf(Exponential(hyper), sigma) + a + logpdf(LogNormal(0.7, 0.6), ell_t) + c
                z = u[cell_coordinates(names, declaration, cell)]
                obs, times = gmrf_cell_rows_logpdf(z, sigma, input, cells[label], :channels)
                lp += obs + gmrf_cell_field_logpdf(z, 0.4, ell_t, L, times)
            end
            lp
        end
        u = 0.05 .* sin.(1:built.layout.total)
        query = prepare_query(built, bound, :sampler)
        for point in (zeros(length(u)), u)
            @test Base.invokelatest(query, point) ≈ oracle(point) atol=2e-8 rtol=2e-10
        end
        gmrf_rk_gradient_check(built, bound, oracle, u)
        # Replay onto the fit's cells, rows reordered and one row dropped:
        # each cell keeps its fitted support and coordinates.
        order = [4, 2, 1]
        fresh = merge(input, (; (k => getproperty(input, k)[order]
            for k in (:east, :north, :time, :channel, :y, :unit))...))
        sampled_correlation && (fresh = (; (k => v for (k, v) in pairs(fresh) if k !== :Omega)...))
        replay = reprocess(rk, fresh)
        @test BRM._rk_layout_coordinate_names(replay) == names
        replay_lp = rk_logdensity_problem(replay; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
        function replay_oracle(u)
            lp = oracle(u)
            # Swap each cell's observation term for the replay rows'.
            for (cell, label) in enumerate((10, 20))
                a = u[coordinate(names, "$declaration.sigma.$cell")]
                z = u[cell_coordinates(names, declaration, cell)]
                lp -= first(gmrf_cell_rows_logpdf(z, exp(a), input, cells[label], :channels))
                rows = findall(==(label), fresh.unit)
                # The replay's cell keeps the fitted times; its rows read them.
                support = BRM.brm_gmrf_support(data.xy, data.triangles,
                    sort!(unique(input.time[cells[label]])), input.channel[cells[label]])
                reads = BRM.brm_gmrf_rows(support, fresh.east[rows], fresh.north[rows],
                    fresh.time[rows], fresh.channel[rows])
                mu = exp(a) .* (reads.weight1 .* z[reads.index1] .+ reads.weight2 .* z[reads.index2] .+
                    reads.weight3 .* z[reads.index3])
                lp += sum(logpdf.(Normal.(mu, 0.2), fresh.y[rows]); init=0.0)
            end
            lp
        end
        @test BRM.LogDensityProblems.logdensity(replay_lp, u) ≈ replay_oracle(u) atol=2e-8 rtol=2e-10
        # A cell latent belongs to the fit's cells: replay onto other cells is refused.
        other = merge(fresh, (; unit=fill(10, length(order))))
        @test_throws ErrorException reprocess(rk, other)
    end
end

import BridgeStan as BS
@testset "RK/SB cell field transport: values, density and native Reverse" begin
    models = (gmrf_plate_model(; known=false), gmrf_plate_model(; known=true),
        gmrf_rk_cell_model(; sampled_correlation=false),
        gmrf_rk_cell_model(; sampled_correlation=true))
    for model in models
        rk, bound, built = gmrf_rk_build(model)
        sb = SBBRMI(model; mod=@__MODULE__)
        stan = BRM.stan_instantiate(sb)
        transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
        @test length(transport.pairs) == built.layout.total
        @test any(p -> get(p.address, :kind, nothing) === :cell_gmrf &&
            get(p.address, :parameter, nothing) === :field, transport.pairs)
        u = 0.05 .* sin.(1:built.layout.total)
        sampler = prepare_sampler(built, bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
        for point in (zeros(length(u)), u)
            @test brm_check_coordinate_transport(transport, rk, stan.model, point).max_error < 1e-10
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
            sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
                propto=false, jacobian=true)
            @test value ≈ sv atol=2e-8 rtol=2e-10
            @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
        end
    end
end
