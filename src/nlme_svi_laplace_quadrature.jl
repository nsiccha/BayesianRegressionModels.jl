# Matrix-free log quadratic forms. The caller provides KrylovKit and
# LinearAlgebra, and a PROVEN positive spectral lower bound, not a Ritz value.
# For log, Gauss is an upper bound and left Gauss-Radau is a lower bound:
# its even derivatives are negative and its odd derivatives positive.
# Reference: Golub, Matrix Computation and the Theory of Moments, eqs3.2/3.4:
# https://web.stanford.edu/class/cme324/mmq2.pdf
# These are exact-arithmetic quadrature bounds, not interval-arithmetic
# certificates for the floating-point HVPs, mode, or Lanczos recurrence.
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
    for k in 1:limit
        T = KrylovKit.rayleighquotient(fact)
        E = eigen(T)
        minimum(E.values) > 0 || throw(ArgumentError("nonpositive GaussianBlockLaplace curvature"))
        scale = max(maximum(abs, E.values), spectral_lower)
        slack = 128eps(Float64)*scale
        minimum(E.values) >= spectral_lower - slack ||
            throw(ArgumentError("GaussianBlockLaplace curvature contradicts its spectral lower bound"))
        upper = mass * dot(abs2.(E.vectors[1, :]), log.(E.values))
        residual = KrylovKit.normres(fact)
        # At invariant-subspace termination the Gaussian rule is exact up to
        # floating-point error. Do not invert the singular shifted projection.
        terminated = residual <= 64eps(Float64)*scale
        if terminated
            lower = upper
        else
            # Set the added diagonal so the augmented Jacobi matrix has the
            # prescribed eigenvalue a: alpha=a+beta^2 e_k'(T-aI)^-1 e_k.
            # Using a strictly smaller valid bound keeps that solve nonsingular
            # even when the true spectrum contains spectral_lower exactly.
            a = spectral_lower / 2
            shifted = SymTridiagonal(T.dv .- a, copy(T.ev))
            rhs = zeros(k); rhs[end] = 1.0
            alpha = a + residual^2 * (shifted \ rhs)[end]
            radau = SymTridiagonal(vcat(T.dv, alpha), vcat(T.ev, residual))
            R = eigen(radau)
            minimum(R.values) > 0 || error("nonpositive Radau node")
            lower = mass * dot(abs2.(R.vectors[1, :]), log.(R.values))
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
