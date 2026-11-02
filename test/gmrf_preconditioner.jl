using Test, BayesianRegressionModels, LinearAlgebra, SparseArrays, Random
const BRM = BayesianRegressionModels

@testset "Separable GMRF reference preserves covariance, transpose and conditional solves" begin
    rng = Xoshiro(2112026)
    mesh = BRM._brm_gmrf_mesh([0.0 0; 1 0; 1 1; 0 1; 0.5 0.5],
        [1 2 5; 2 3 5; 3 4 5; 4 1 5])
    for channels in (1, 2), times in ([0.0], [0.0, 0.003, 1.7])
        omega = [0.4^abs(i-j) for i in 1:channels, j in 1:channels]
        L = Matrix(cholesky(Symmetric(omega)).L)
        spatial, temporal = 0.7, 1.3
        Q = BRM._BRMGMRFPrecisionOperator(mesh, times, spatial, temporal, L)
        reference = BRM._brm_gmrf_fixed_reference(Q)
        saved_factor = deepcopy(reference.factor)
        n = size(Q, 1)
        # Analytic time covariance and an independent dense spatial inverse.
        kappa2 = 2/spatial^2
        K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
        Qs = K * Diagonal(1 ./ mesh.mass) * K / (4pi*kappa2)
        Ct = [exp(-abs(t-s)/temporal) for t in times, s in times]
        covariance = kron(Ct, kron(omega, inv(Symmetric(Qs))))
        work, out = zeros(n), zeros(n)
        basis = Matrix{Float64}(I, n, n)
        F = hcat([copy(BRM._brm_gmrf_reference_forward!(out, reference, x, work))
            for x in eachcol(basis)]...)
        Ft = hcat([copy(BRM._brm_gmrf_reference_transpose!(out, reference, x, work))
            for x in eachcol(basis)]...)
        @test F * F' ≈ covariance atol=2e-12 rtol=2e-10
        @test Ft ≈ F' atol=2e-12 rtol=2e-10
        @test reference.logdet ≈ -logdet(Symmetric(covariance)) atol=2e-9
        @test reference.logdet ≈ -2logabsdet(F)[1] atol=2e-9
        B = sparse(repeat(1:4; inner=3), [1,2,5,2,3,5,3,4,5,n-4,n-3,n],
            [0.2,0.3,0.5,0.1,0.7,0.2,0.4,0.2,0.4,0.3,0.2,0.5], 4, n)
        W = spdiagm(0 => [0.3,2.0,1.1,4.0])
        diagonal = diag(Q) + vec(sum(B .* (W * B); dims=1))
        H = BRM._BRMConditionalOperator(Q, B, W, diagonal, 0.0, zeros(4), zeros(4))
        A = BRM._BRMPreconditionedOperator(H, reference)
        denseH = inv(Symmetric(covariance)) + Matrix(B' * W * B)
        @test F' * denseH * F ≈ I + F' * Matrix(B' * W * B) * F atol=3e-9
        options = MatrixFreeLaplace(; solve_tolerance=1e-9, solve_maxiter=200)
        for _ in 1:3
            x, y = randn(rng,n), randn(rng,n)
            @test A(x) ≈ F' * denseH * F * x atol=3e-9 rtol=2e-10
            @test dot(x, A(y)) ≈ dot(A(x), y) atol=3e-9
            result = BRM._brm_operator_solve(A, x, options)
            @test result ≈ denseH \ x atol=3e-8 rtol=2e-8
            @test norm(denseH * result - x) / norm(x) < options.solve_tolerance
        end
        # A prepared reference remains fixed as the target's outer state moves.
        shifted = BRM._BRMGMRFPrecisionOperator(mesh, times, 1.2, 0.8, L)
        B0, W0 = spzeros(0,n), spzeros(0,0)
        H2 = BRM._BRMConditionalOperator(shifted, B0, W0, diag(shifted), 0.0,
            Float64[], Float64[])
        A2 = BRM._BRMPreconditionedOperator(H2, reference)
        x = randn(rng,n)
        @test A2(x) ≈ F' * shifted(F*x) atol=2e-9 rtol=2e-10
        @test reference.factor == saved_factor
    end
end
