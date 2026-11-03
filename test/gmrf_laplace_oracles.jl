# Dense marginal oracles for the GMRF Laplace tests: with Normal observations
# the embedded Laplace is exact, y ~ N(0, σ² A Σ Aᵀ + s² I).
include("gmrf_rk_cell_models.jl")
import LogDensityProblems as LDP

# The field's dense covariance on a support (state order vertex, channel, time).
function gmrf_dense_covariance(support, ell_s, ell_t, L)
    mesh = support.mesh
    kappa2 = 2 / ell_s^2
    K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
    Q = K * Diagonal(1 ./ mesh.mass) * K / (4pi * kappa2)
    times = support.layout.times
    Ct = [exp(-abs(t - s) / ell_t) for t in times, s in times]
    kron(Ct, kron(L * L', inv(Symmetric(Q))))
end

# Rows × states read matrix of three-vertex reads.
function gmrf_read_matrix(reads, n)
    A = zeros(length(reads.index1), n)
    for r in axes(A, 1), j in 1:3
        A[r, getproperty(reads, Symbol(:index, j))[r]] += getproperty(reads, Symbol(:weight, j))[r]
    end
    A
end

gmrf_marginal(y, A, sigma, Σ, noise) =
    logpdf(MvNormal(zeros(length(y)), Symmetric(sigma^2 * A * Σ * A' + noise^2 * I)), y)

function gmrf_laplace_gradient_check(target, oracle, φ)
    value, gradient = LDP.logdensity_and_gradient(target, φ)
    @test value ≈ oracle(φ) atol=1e-7 rtol=1e-9
    for j in eachindex(φ)
        plus, minus = copy(φ), copy(φ)
        h = 1e-5
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (oracle(plus) - oracle(minus)) / (2h) atol=1e-5 rtol=1e-5
    end
end
