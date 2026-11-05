include("gmrf_precision.jl")
include("gmrf_small_length_oracle.jl")

@testset "GMRF short spatial ranges preserve the normalized law" begin
    times = [0.0, 0.3, 1.7]
    L = [1.0 0.0; 0.45 sqrt(1 - 0.45^2)]
    for logell in (-2.0, -40.0, -180.0, -300.0, -365.0)
        ell = exp(logell)
        Q = gmrf_big_precision(mesh.mass, mesh.stiffness, times, ell, 1.4, L)
        for amplitude in (0.0, ell, 1e-6, 0.1)
            state = reshape(amplitude .* sin.(1:30), 5, 2, 3)
            expected = Float64(gmrf_big_logpdf(state, mesh.mass, mesh.stiffness,
                times, ell, 1.4, L))
            actual = GM._brm_gmrf_logpdf(state, mesh, times, ell, 1.4, L)
            @test !isnan(actual)
            @test actual ≈ expected rtol=5e-13 atol=2e-11
        end
        state = reshape(ell .* sin.(1:30), 5, 2, 3)
        out = similar(state)
        GM._brm_gmrf_precision_mul!(out, state, mesh, times, ell, 1.4, L)
        @test vec(out) ≈ Float64.(Q * BigFloat.(vec(state))) rtol=2e-12
        GM._brm_gmrf_precision_diagonal!(out, mesh, times, ell, 1.4, L)
        @test all(isapprox.(vec(out), Float64.(diag(Q)); rtol=2e-12))
    end
end
