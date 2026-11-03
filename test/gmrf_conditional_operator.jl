using Test, BayesianRegressionModels, LinearAlgebra, SparseArrays, Random
const BRM = BayesianRegressionModels

@testset "Conditional GMRF solves combine precision and local likelihood curvature" begin
    rng = Xoshiro(11022026)
    mesh = BRM._brm_gmrf_mesh([0.0 0; 1 0; 1 1; 0 1; 0.5 0.5],
        [1 2 5; 2 3 5; 3 4 5; 4 1 5])
    for channels in (1, 2), times in ([0.0], [0.0, 0.3, 1.7])
        omega = [0.4^abs(i-j) for i in 1:channels, j in 1:channels]
        L = Matrix(cholesky(Symmetric(omega)).L)
        spatial, temporal = 0.7, 1.3
        Q = BRM._BRMGMRFPrecisionOperator(mesh, times, spatial, temporal, L)
        n = size(Q, 1)
        # Independent covariance construction; no transition recurrence.
        kappa2 = 2/spatial^2
        K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
        Qs = K * Diagonal(1 ./ mesh.mass) * K / (4pi*kappa2)
        Ct = [exp(-abs(t-s)/temporal) for t in times, s in times]
        covariance = kron(Ct, kron(omega, inv(Symmetric(Qs))))
        denseQ = inv(Symmetric(covariance))
        B = sparse(repeat(1:4; inner=3), [1,2,5,2,3,5,3,4,5,n-4,n-3,n],
            [0.2,0.3,0.5,0.1,0.7,0.2,0.4,0.2,0.4,0.3,0.2,0.5], 4, n)
        W = spdiagm(0 => [0.3,2.0,1.1,4.0])
        diagonal = diag(Q) + vec(sum(B .* (W * B); dims=1))
        H = BRM._BRMConditionalOperator(Q, B, W, diagonal, 0.0, zeros(4), zeros(4))
        denseH = denseQ + Matrix(B' * W * B)
        options = MatrixFreeLaplace(; solve_tolerance=1e-10, solve_maxiter=200,
            preconditioner_memory=nothing)
        @test diag(Q) ≈ diag(denseQ) rtol=2e-10
        @test H.diagonal ≈ diag(denseH) rtol=2e-10
        for _ in 1:3
            b = randn(rng, n)
            @test Q(b) ≈ denseQ*b rtol=2e-10
            @test H(b) ≈ denseH*b rtol=2e-10
            diagnostics = Dict{Symbol,Any}()
            solution = BRM._brm_operator_solve(H, b, options; diagnostics)
            @test solution ≈ denseH \ b rtol=2e-8 atol=2e-9
            @test norm(denseH*solution-b)/norm(b) < options.solve_tolerance
            @test diagnostics[:linear_solves].max_relative_residual < options.solve_tolerance
        end
        @test_throws BRM._BRMOperatorSolveFailure BRM._brm_operator_solve(H,
            randn(rng,n), MatrixFreeLaplace(; solve_maxiter=1, preconditioner_memory=nothing))
    end
end
