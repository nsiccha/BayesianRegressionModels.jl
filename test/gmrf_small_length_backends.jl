include("gmrf_rk_numerical.jl")
include("gmrf_small_length_oracle.jl")

@testset "RK and Stan helpers retain short-range GMRF densities" begin
    times = [0.0, 0.3, 1.7]
    mesh = BRM._brm_gmrf_mesh(GMRF_XY, GMRF_TRIANGLES)
    geometry = BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, times, 2)
    factor_data = BRM._brm_gmrf_factor_data(mesh)
    L = [1.0 0.0; 0.45 sqrt(1 - 0.45^2)]
    query = prepare(EXT._gmrf_density_graph)
    for logell in (-2.0, -40.0, -180.0, -300.0, -365.0)
        ell = exp(logell)
        for amplitude in (0.0, ell, 1e-6, 0.1)
            field = amplitude .* sin.(1:30)
            expected = Float64(gmrf_big_logpdf(field, mesh.mass, mesh.stiffness,
                times, ell, 1.4, L))
            rk = query(field, ell, 1.4, L, geometry)
            sb = BRM.brm_gmrf_logdensity(field, ell, 1.4, L, times, values(factor_data)...)
            @test !isnan(rk) && !isnan(sb)
            @test rk ≈ expected rtol=5e-13 atol=2e-11
            @test sb ≈ expected rtol=5e-13 atol=2e-11
        end
    end
    # At a zero field and a field on its natural scale, even the constrained
    # derivatives are representable. Check every derivative in native Reverse
    # against the independent 256-bit oracle with relative displacements.
    differentiated = prepare_ad(query, DI.AutoEnzyme(; mode=Enzyme.Reverse),
        zeros(30), 1.0, 1.4, L, geometry;
        active=(:field, :spatial_length, :temporal_length, :L))
    for logell in (-40.0, -180.0, -365.0), amplitude in (0.0, exp(logell))
        ell = exp(logell)
        field = amplitude .* sin.(1:30)
        value, derivatives = ad_value_and_gradient(differentiated, field, ell, 1.4, L, geometry)
        @test isfinite(value)
        @test all(isfinite, derivatives[1])
        @test all(isfinite, derivatives[2:3])
        @test all(isfinite, derivatives[4])
        h = 1e-5
        expected = (gmrf_big_logpdf(field, mesh.mass, mesh.stiffness, times, ell*exp(h), 1.4, L) -
            gmrf_big_logpdf(field, mesh.mass, mesh.stiffness, times, ell*exp(-h), 1.4, L)) / (2h)
        @test derivatives[2] * ell ≈ Float64(expected) rtol=2e-8 atol=2e-8
        for j in eachindex(field)
            plus, minus = copy(field), copy(field)
            plus[j] += ell*h
            minus[j] -= ell*h
            expected = (gmrf_big_logpdf(plus, mesh.mass, mesh.stiffness, times, ell, 1.4, L) -
                gmrf_big_logpdf(minus, mesh.mass, mesh.stiffness, times, ell, 1.4, L)) / (2h)
            @test derivatives[1][j] * ell ≈ Float64(expected) rtol=2e-8 atol=2e-8
        end
    end
end
