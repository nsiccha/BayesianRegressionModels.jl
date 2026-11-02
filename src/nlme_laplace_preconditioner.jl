# A reference precision P=C*C' is fixed for a prepared target. It affects
# numerical conditioning and the retained trace probes, never the model's H.
# LimitedLDLFactorizations documents L, D, P and alpha in its public example.
struct _BRMFixedReference{L,U}
    lower::L
    upper::U
    permutation::Vector{Int}
    inverse_scale::Vector{Float64}
    logdet::Float64
    factor_entries::Int
    diagonal_shift::Float64
end

function _brm_fixed_reference(H, memory; ridge=0.01)
    # Do not expand reduction cliques B' W B. Its diagonal can still improve
    # the reference; the complete reduction remains in every H action.
    d = H.diagonal .+ H.shift
    Q = H.D + spdiagm(0 => d .- diag(H.D) .+ ridge .* abs.(d))
    F = LimitedLDLFactorizations.lldl(Q; memory)
    all(isfinite, F.D) && all(!iszero, F.D) ||
        throw(_BRMOperatorSolveFailure("nonfinite or singular fixed conditional preconditioner"))
    L = UnitLowerTriangular(copy(F.L))
    _BRMFixedReference(L, L', copy(F.P), 1 ./ sqrt.(abs.(F.D)),
        sum(log, abs.(F.D)), SparseArrays.nnz(F.L), Float64(F.α))
end

struct _BRMPreconditionedOperator{H,R}
    h::H
    reference::R
    first::Vector{Float64}
    physical::Vector{Float64}
    product::Vector{Float64}
end
_BRMPreconditionedOperator(H, R) =
    _BRMPreconditionedOperator(H, R, zeros(size(H, 1)), zeros(size(H, 1)), zeros(size(H, 1)))
Base.size(A::_BRMPreconditionedOperator) = size(A.h)
Base.size(A::_BRMPreconditionedOperator, d::Integer) = size(A.h, d)
Base.eltype(::_BRMPreconditionedOperator) = Float64
Base.eltype(::Type{<:_BRMPreconditionedOperator}) = Float64

# C^-T maps transformed vectors back to the authored latent coordinates.
function _brm_to_physical!(out, A::_BRMPreconditionedOperator, x)
    R = A.reference
    A.first .= R.inverse_scale .* x
    ldiv!(R.upper, A.first)
    @inbounds for i in eachindex(R.permutation)
        out[R.permutation[i]] = A.first[i]
    end
    out
end
function _brm_from_physical!(out, A::_BRMPreconditionedOperator, x)
    R = A.reference
    @inbounds for i in eachindex(R.permutation)
        out[i] = x[R.permutation[i]]
    end
    ldiv!(R.lower, out)
    out .*= R.inverse_scale
    out
end
function LinearAlgebra.mul!(out, A::_BRMPreconditionedOperator, x)
    _brm_to_physical!(A.physical, A, x)
    mul!(A.product, A.h, A.physical)
    _brm_from_physical!(out, A, A.product)
end
(A::_BRMPreconditionedOperator)(x) = mul!(similar(x), A, x)

function _brm_operator_solve(A::_BRMPreconditionedOperator, b, options; diagnostics=nothing)
    rhs = similar(b, Float64)
    _brm_from_physical!(rhs, A, b)
    # A small transformed residual need not be a small original residual.
    # Tighten the internal solve, then always verify against the authored H.
    x = _brm_checked_cg(A, rhs, options; rtol=options.solve_tolerance / 1000, diagnostics)
    result = similar(x)
    _brm_to_physical!(result, A, x)
    residual = iszero(norm(b)) ? 0.0 : norm(A.h(result) .- b) / norm(b)
    if diagnostics !== nothing && haskey(diagnostics, :linear_solves)
        old = diagnostics[:linear_solves]
        diagnostics[:linear_solves] = merge(old,
            (; max_relative_residual=max(old.max_relative_residual, residual)))
    end
    residual <= options.solve_tolerance || throw(_BRMOperatorSolveFailure(
        "preconditioned conditional solve has original relative residual $residual, " *
        "exceeding $(options.solve_tolerance)"))
    result
end
