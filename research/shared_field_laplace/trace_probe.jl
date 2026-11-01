# Matrix-free trace-gradient control, after the native Gaussian operator gate.
# Run under one compute token with the repository test environment.
include("operator_probe.jl")

# For a fixed mode, H = -f_zz and E[r*r'] = I. Polarization gives
#   T(r,H^-1*r) = (T(r+H^-1*r,r+H^-1*r)-T(r-H^-1*r,r-H^-1*r))/4.
# Its expectation is tr(H^-1 f_zzx), the determinant's contribution to the
# collapsed gradient. One solve plus two native thirds per random probe;
# no inverse matrix or covariance square root is formed. This is a stochastic
# gradient of the joint Laplace target, NOT a likelihood minibatch estimator.
# Iterative solves introduce separately reported numerical error.
# Reference for solve-based trace gradients: Kündig & Sigrist (2025), §3.3.1,
# https://doi.org/10.1080/01621459.2024.2410004 . We retain the authored HSGP;
# none of that paper's Vecchia prior approximation is used here.
function shared_trace(op, mode, probes; rtol=1e-11, maxiter=128)
    H(v) = shared_precision(op, mode, v)
    result = zeros(length(mode))
    operations, max_residual, n = 0, 0.0, 0
    for r in probes
        length(r) == length(op.field) || throw(DimensionMismatch("field probe"))
        s, info = KrylovKit.linsolve(H, r; issymmetric=true, isposdef=true,
            rtol, atol=1e-13, krylovdim=maxiter, maxiter=1)
        info.converged == 1 || error("trace CG did not converge: $(info.normres)")
        result .+= (shared_third(op, mode, r .+ s) .-
            shared_third(op, mode, r .- s)) ./ 4
        operations += info.numops
        max_residual = max(max_residual, info.normres)
        n += 1
    end
    n > 0 || throw(ArgumentError("at least one field trace probe is required"))
    (; trace=result ./ n, operations, max_residual)
end

@stestset "solve-based trace equals the small exact contraction" begin
    (; m) = shared_field_fixture()
    op = shared_field_operator(m)
    u = first(BRM._brm_nlme_prior_median_point(m))
    u .+= 0.09 .* randn(Xoshiro(40), length(u))
    mode = shared_gaussian_mode(op, u).mode
    d = length(op.field)
    @test d == 8
    # Dense arrays are bounded oracle objects only. The orthogonal probe set
    # removes Monte Carlo noise from this test: mean(r*r') = I exactly.
    basis = Matrix{Float64}(I, d, d)
    H = hcat((shared_precision(op, mode, basis[:, j]) for j in 1:d)...)
    root = inv(cholesky(Symmetric(H)).U)
    exact = sum(shared_third(op, mode, root[:, j]) for j in 1:d)
    orthogonal = Matrix(qr(randn(Xoshiro(41), d, d)).Q)
    probes = [sqrt(d) .* orthogonal[:, j] for j in 1:d]
    estimate = shared_trace(op, mode, probes)
    @test estimate.trace ≈ exact rtol=1e-8 atol=1e-9
    @test_throws ArgumentError shared_trace(op, mode, Vector{Float64}[])
    println((; trace_max_error=maximum(abs.(estimate.trace .- exact)),
        estimate.operations, estimate.max_residual,
        scope="deterministic trace identity control; no random-probe error bound"))
end
