# The Lanczos recurrences of the sparse Laplace's covariance root, the
# matrix-free Laplace's log-determinant probes and GaussianBlockLaplace's
# quadrature and solves, loaded with that package (src/fitting_extensions.jl). A
# KrylovKit state never leaves this module except as an opaque value passed
# back into it; `_brm_lanczos_probe` hands BRM the basis as plain vectors.
module BayesianRegressionModelsKrylovKitExt

import BayesianRegressionModels as BRM
import KrylovKit
using LinearAlgebra: Symmetric, dot, eigen, norm

const _BRMOperatorSolveFailure = BRM._BRMOperatorSolveFailure

# Apply S^(1/2) to a standard-normal vector through solves for S = H^(-1)
# (`_brm_spl_covariance_root`, src/nlme_laplace_factors.jl).
function BRM._brm_krylovkit_covariance_root(solve, b; rtol, maxdim)
    β = norm(b)
    β == 0 && return (; draw=copy(b), iterations=0, error_bound=0.0)
    it = KrylovKit.LanczosIterator(solve, b, KrylovKit.ModifiedGramSchmidt2())
    fact = KrylovKit.initialize(it)
    bound = Inf
    limit = min(length(b), maxdim)
    for k in 1:limit
        T = KrylovKit.rayleighquotient(fact)
        E = eigen(T)
        minimum(E.values) > 0 || BRM._brm_nlme_error(
            "brm_laplace_draws: the covariance projection is not positive definite")
        roots = sqrt.(E.values)
        weights = E.vectors * (roots .* E.vectors[1, :])
        draw = β .* (KrylovKit.basis(fact) * weights)
        # X = sqrt(S)V - V sqrt(T) solves sqrt(S)X + X sqrt(T) = r e_k'.
        # Sign alternation makes the Jacobi T a Stieltjes matrix, so the
        # scalar e_k' exp(-sqrt(T)t)e_1 has constant sign. Bound its integral
        # using ||exp(-sqrt(S)t)|| <= 1: ||X e_1|| <= ||r|| |T^-1/2[k,1]|.
        # No estimated smallest eigenvalue enters the stopping condition.
        corner = dot(E.vectors[end, :], E.vectors[1, :] ./ roots)
        bound = β * KrylovKit.normres(fact) * abs(corner)
        bound <= rtol * norm(draw) && return (; draw, iterations=k, error_bound=bound)
        k < limit && KrylovKit.expand!(it, fact)
    end
    BRM._brm_nlme_error("brm_laplace_draws: the covariance square-root action did not " *
        "reach relative tolerance $rtol in $limit Lanczos steps (error bound $bound)")
end

# One probe's Lanczos recurrence on `operator` (`_brm_lanczos_run`,
# src/nlme_laplace_matrixfree.jl), stopped when the solve of `A x = r` it
# implies reaches `rtol`, at `maxiter` iterations, or at nonpositive curvature
# in its directions. That solve's residual is β_k |e_k' T_k⁻¹ e₁|, read from
# T_k = L D L' as T grows, whose pivots also check the curvature.
function BRM._brm_krylovkit_lanczos_run(operator, r, rtol, maxiter)
    iterator = KrylovKit.LanczosIterator(operator, collect(r), KrylovKit.ModifiedGramSchmidt())
    state = KrylovKit.initialize(iterator; verbosity=0)
    pivot = first(state.αs)
    last_entry = 1.0
    while true
        k = length(state)
        pivot > 0 && isfinite(pivot) || return (; state, residual=Inf, curved=false)
        β = KrylovKit.normres(state)
        residual = β * abs(last_entry) / pivot
        (residual <= rtol || k >= maxiter) && return (; state, residual, curved=true)
        KrylovKit.expand!(iterator, state; verbosity=0)
        last_entry *= -β / pivot
        pivot = state.αs[k + 1] - β^2 / pivot
    end
end

function BRM._brm_krylovkit_lanczos(operator, r, options)
    (; state, residual, curved) = BRM._brm_krylovkit_lanczos_run(operator, r,
        options.solve_tolerance / 10, options.solve_maxiter)
    curved || throw(_BRMOperatorSolveFailure("conditional determinant recurrence found " *
        "nonpositive curvature at iteration $(length(state))"))
    residual <= options.solve_tolerance / 10 || throw(_BRMOperatorSolveFailure("conditional " *
        "determinant recurrence failed within $(options.solve_maxiter) iterations: residual $residual"))
    state
end

# `r' log(A) r` on r's Krylov space, the residual of the solve that recurrence
# implies, and the factorization G = P Γ P' (scaled by ‖r‖²) of the sensitivity
# ‖r‖² V G V', with `V G V'` the sensitivity of `e₁' log(T) e₁` to a change of A
# (Daleckii–Krein: G = Q [log[θᵢ,θⱼ] cᵢ cⱼ] Q', T = Q Θ Q', c = Q'e₁).
function _lanczos_log(state, r)
    T = KrylovKit.rayleighquotient(state)
    k = length(state)
    E = eigen(Symmetric(Matrix(T)))
    θ, Q = E.values, E.vectors
    all(>(0), θ) || throw(_BRMOperatorSolveFailure(
        "conditional determinant recurrence found nonpositive curvature: smallest Ritz value $(first(θ))"))
    c = view(Q, 1, :)
    scale = dot(r, r)
    value = scale * sum(i -> c[i]^2 * log(θ[i]), 1:k)
    G = Q * [BRM._brm_log_divided_difference(θ[i], θ[j]) * c[i] * c[j] for i in 1:k, j in 1:k] * Q'
    Γ = eigen(Symmetric(G))
    keep = BRM._brm_lanczos_rank(Γ.values)
    start = zeros(k); start[1] = sqrt(scale)
    (; value, ritz=(first(θ), last(θ)), weights=scale .* Γ.values[keep],
        directions=Γ.vectors[:, keep], solve=T \ start)
end

function BRM._brm_krylovkit_lanczos_probe(operator, r, options)
    state = BRM._brm_krylovkit_lanczos(operator, r, options)
    (; V=collect(KrylovKit.basis(state)), log_r=_lanczos_log(state, r),
        iterations=length(state))
end

# GaussianBlockLaplace's Gauss–Radau bounds on `r' log(H) r`
# (`_brm_svi_laplace_logquadratic`, src/nlme_svi_laplace_quadrature.jl).
function BRM._brm_krylovkit_logquadratic(H, r; spectral_lower, atol, rtol, maxdim)
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
        factors = BRM._brm_svi_laplace_pivots(T, a)
        gauss = BRM._brm_svi_laplace_gauss_rule(factors.pivots, T.ev)
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
            lower = mass * BRM._brm_svi_laplace_gauss_rule(
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

# GaussianBlockLaplace's symmetric positive definite solves
# (`_brm_svi_laplace_solve`, src/nlme_svi_laplace.jl).
function BRM._brm_krylovkit_svi_laplace_solve(t,H,b,operation,krylovdim;rtol,atol)
    x,info=KrylovKit.linsolve(H,b;issymmetric=true,isposdef=true,
        rtol,atol,krylovdim,maxiter=1)
    info.converged==1 || error("GaussianBlockLaplace: $operation exhausted its Krylov budget " *
        "($krylovdim directions; residual $(info.normres))")
    BRM._brm_svi_laplace_record!(t,Symbol(operation,:_residual),info.normres)
    BRM._brm_svi_laplace_record!(t,Symbol(operation,:_operations),info.numops)
    x
end

end
