# Matrix-free log quadratic forms. The caller provides KrylovKit and
# LinearAlgebra, and a PROVEN positive spectral lower bound, not a Ritz value.
# For log, Gauss is an upper bound and left Gauss-Radau is a lower bound:
# its even derivatives are negative and its odd derivatives positive.
# Reference: Golub, Matrix Computation and the Theory of Moments, eqs3.2/3.4:
# https://web.stanford.edu/class/cme324/mmq2.pdf
# These are exact-arithmetic quadrature bounds, not interval-arithmetic
# certificates for the floating-point HVPs, mode, or Lanczos recurrence.
#
# Both rules read one Cholesky factorization. The Radau matrix extends T by a
# row, so its factor extends T's by one pivot, and Gauss >= Radau is then an
# algebraic fact about those floats for any positive pivot: Gauss on k nodes is
# exact to degree 2k-1 for the Radau measure, and log's even derivatives are
# negative. Separate `eigen` calls lose this, and near convergence the true gap is
# smaller than their independent errors: eigenvectors of clustered Ritz values
# lose orthogonality, so the weights stop summing to one, and a small eigenvalue
# moves by about eps*norm(T), hence its log by eps*cond(T).
# The value itself still carries about eps*cond(T) from T's own floating entries.

# Cholesky pivots of the Jacobi matrix T and the last pivot of T-aI. `gap` is the
# difference of those two last pivots, a sum of positive terms and so free of
# the cancellation in subtracting them.
function _brm_svi_laplace_pivots(T, a)
    pivots = similar(T.dv)
    shifted, gap = zero(a), a
    for i in eachindex(pivots)
        if i == 1
            pivots[i] = T.dv[i]
            shifted = T.dv[i] - a
        else
            square = T.ev[i-1]^2
            gap = a + square * gap / (pivots[i-1] * shifted)
            pivots[i] = T.dv[i] - square / pivots[i-1]
            shifted = T.dv[i] - a - square / shifted
        end
        pivots[i] > 0 || throw(ArgumentError("nonpositive GaussianBlockLaplace curvature"))
        shifted > 0 || throw(ArgumentError(
            "GaussianBlockLaplace curvature contradicts its spectral lower bound"))
    end
    (; pivots, shifted, gap)
end

# The Gauss rule of the Jacobi matrix with these pivots. T=BB' for a lower
# bidiagonal B, whose singular values bdsqr finds to high relative accuracy, as
# LAPACK's dpteqr does; only the first row of U is needed (Golub-Welsch).
function _brm_svi_laplace_gauss_rule(pivots, offdiagonal)
    k = length(pivots)
    root = sqrt.(pivots)
    row = zeros(1, k); row[1] = 1.0
    singular, _, row, _ = LAPACK.bdsqr!('L', copy(root),
        [offdiagonal[i] / root[i] for i in 1:k-1], zeros(k, 0), row, zeros(k, 0))
    (; value=2dot(abs2.(vec(row)), log.(singular)),
       largest=first(singular)^2, smallest=last(singular)^2)
end

function _brm_svi_laplace_logquadratic(H, r; spectral_lower, atol=1e-9, rtol=1e-9,
        maxdim=64)
    isfinite(spectral_lower) && spectral_lower > 0 ||
        throw(ArgumentError("a positive spectral lower bound is required"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    atol >= 0 && rtol >= 0 || throw(ArgumentError("negative quadrature tolerance"))
    mass = sum(abs2, r)
    iszero(mass) && return (; value=0.0, lower=0.0, upper=0.0,
        iterations=0, converged=true, roundoff=0.0)
    it = KrylovKit.LanczosIterator(H, r, KrylovKit.ModifiedGramSchmidt2())
    fact = KrylovKit.initialize(it)
    limit = min(length(r), maxdim)
    result = nothing
    # A strictly smaller valid bound keeps T-aI nonsingular even when the true
    # spectrum contains spectral_lower exactly.
    a = spectral_lower / 2
    for k in 1:limit
        T = KrylovKit.rayleighquotient(fact)
        factors = _brm_svi_laplace_pivots(T, a)
        gauss = _brm_svi_laplace_gauss_rule(factors.pivots, T.ev)
        scale = max(gauss.largest, spectral_lower)
        slack = 128eps(Float64)*scale
        gauss.smallest >= spectral_lower - slack ||
            throw(ArgumentError("GaussianBlockLaplace curvature contradicts its spectral lower bound"))
        upper = mass * gauss.value
        residual = KrylovKit.normres(fact)
        # At invariant-subspace termination the Gaussian rule is exact up to
        # floating-point error. Do not invert the singular shifted projection.
        terminated = residual <= 64eps(Float64)*scale
        if terminated
            lower = upper
        else
            # The added diagonal alpha=a+beta^2 e_k'(T-aI)^-1 e_k gives the
            # augmented Jacobi matrix the prescribed eigenvalue a, and its pivot
            # alpha-beta^2/pivots[k] is this sum, positive by construction.
            pivot = a + residual^2 * factors.gap / (factors.pivots[k] * factors.shifted)
            lower = mass * _brm_svi_laplace_gauss_rule(
                vcat(factors.pivots, pivot), vcat(T.ev, residual)).value
        end
        roundoff = 256eps(Float64) * max(mass, abs(lower), abs(upper))
        lower <= upper + roundoff || error("inconsistent quadrature bounds")
        # Report a crossed interval inside the roundoff allowance as a point.
        lower = min(lower, upper)
        value = (lower+upper)/2
        converged = upper-lower <= 2*(atol+rtol*abs(value)) + roundoff
        result = (; value, lower, upper, iterations=k, converged, roundoff)
        converged && return result
        k < limit && KrylovKit.expand!(it, fact)
    end
    result # A caller must explicitly handle an exhausted numerical budget.
end

# Fixed independent Rademacher probes. Quadrature uncertainty and sampling
# variability are separate: mc_se is a diagnostic, NOT a confidence guarantee.
# No probes, inverse, or Hessian are retained; basis memory is O(d*maxdim).
function _brm_svi_laplace_logdet(rng, H, d; probes=16, kwargs...)
    d > 0 || throw(ArgumentError("positive field dimension required"))
    probes >= 2 || throw(ArgumentError("at least two probes required"))
    average, m2, lower, upper, operations = 0.0, 0.0, 0.0, 0.0, 0
    max_iterations, max_roundoff = 0, 0.0
    for i in 1:probes
        r = ifelse.(rand(rng, Bool, d), 1.0, -1.0)
        q = _brm_svi_laplace_logquadratic(H, r; kwargs...)
        q.converged || error("GaussianBlockLaplace: logdet quadrature exhausted its Krylov budget")
        delta = q.value-average
        average += delta/i
        m2 += delta*(q.value-average)
        lower += q.lower/probes
        upper += q.upper/probes
        operations += q.iterations
        max_iterations = max(max_iterations, q.iterations)
        max_roundoff = max(max_roundoff, q.roundoff)
    end
    (; value=average, mc_se=sqrt(max(m2,0)/((probes-1)*probes)),
       quadrature_lower=lower, quadrature_upper=upper, probes, operations,
       max_iterations, max_roundoff)
end
