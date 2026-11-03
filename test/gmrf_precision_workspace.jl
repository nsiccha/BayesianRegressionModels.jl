include("gmrf_precision.jl")

@testset "GMRF repeated precision action and solver diagonal" begin
    rng = Xoshiro(5312026)
    for channels in (1, 2, 3), times in ([0.0], [0.0, 0.003, 0.8, 15.0]),
            spatial in (0.2, 1.3), temporal in (0.1, 2.1)
        omega = [0.4^abs(i-j) for i in 1:channels, j in 1:channels]
        L = Matrix(cholesky(Symmetric(omega)).L)
        x = randn(rng, 5, channels, length(times))
        workspace = GM._brm_gmrf_precision_workspace(x)
        out = similar(x)
        Q = dense_prior(mesh, times, spatial, temporal, omega).Q
        GM._brm_gmrf_precision_mul!(out, x, mesh, times, spatial, temporal, L, workspace)
        @test vec(out) ≈ Q * vec(x) rtol=2e-9
        # Reusing dirty buffers at a new point cannot retain a previous draw.
        x .= randn(rng, size(x))
        GM._brm_gmrf_precision_mul!(out, x, mesh, times, spatial, temporal, L, workspace)
        @test vec(out) ≈ Q * vec(x) rtol=2e-9
        GM._brm_gmrf_precision_diagonal!(out, mesh, times, spatial, temporal, L)
        @test vec(out) ≈ diag(Q) rtol=2e-9
        @test_throws ArgumentError GM._brm_gmrf_precision_mul!(x, x, mesh, times,
            spatial, temporal, L, workspace)
    end
end
