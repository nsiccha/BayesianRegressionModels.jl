# A fixed covariance square root of the separable prior, used only by the
# solver. Authored field coordinates and observation footprints do not change.
# F*F' = Q_reference^-1. Only the spatial K factor is stored; time uses an
# OU recurrence and channels use their small lower-triangular factor.
struct _BRMGMRFFixedReference{F}
    factor::F
    pattern::_BRMGMRFFactorPattern
    mass_root::Vector{Float64}
    scale::Float64
    correlation_factor::Matrix{Float64}
    decay::Vector{Float64}
    innovation_scale::Vector{Float64}
    shape::NTuple{3,Int}
    logdet::Float64
    factor_entries::Int
    diagonal_shift::Float64
end

function _brm_gmrf_fixed_reference(Q::_BRMGMRFPrecisionOperator)
    mesh = Q.mesh
    n, channels, nt = Q.shape
    kappa2 = 2 / Q.spatial_length^2
    factor = _brm_gmrf_ldl(kappa2, mesh.factor_mass,
        mesh.factor_stiffness, mesh.factor_pattern)
    L = Matrix(LowerTriangular(Q.correlation_factor))
    decay = zeros(nt)
    innovation_scale = ones(nt)
    for t in 2:nt
        gap = (Q.times[t] - Q.times[t - 1]) / Q.temporal_length
        decay[t] = exp(-gap)
        innovation_scale[t] = sqrt(-expm1(-2gap))
    end
    spatial_logdet = 2sum(log, factor.diagonal) - sum(log, mesh.mass) -
        n * log(4pi * kappa2)
    precision_logdet = channels * nt * spatial_logdet -
        2n * nt * sum(log, diag(L)) - 2n * channels * sum(log, innovation_scale)
    entries = length(factor.lower) + length(factor.diagonal) +
        count(!iszero, L) + length(decay) + length(innovation_scale)
    _BRMGMRFFixedReference(factor, deepcopy(mesh.factor_pattern), sqrt.(mesh.factor_mass),
        sqrt(4pi * kappa2), L, decay, innovation_scale, Q.shape,
        precision_logdet, entries, 0.0)
end

# Spatial S = sqrt(4*pi*kappa²) K^-1 sqrt(C), followed by channel L
# and the stationary OU covariance square root. `work` belongs to the
# prepared operator; the reference itself stays unchanged across evaluations.
function _brm_gmrf_reference_forward!(out, R::_BRMGMRFFixedReference, x, work)
    n, channels, nt = R.shape
    length(out) == length(x) == length(work) == n * channels * nt ||
        throw(DimensionMismatch("GMRF reference vector size"))
    permutation = R.pattern.permutation
    for t in 1:nt, c in 1:channels
        offset = n * (c - 1 + channels * (t - 1))
        column = view(work, offset+1:offset+n)
        for j in 1:n
            column[j] = R.mass_root[j] * x[offset + permutation[j]]
        end
        _brm_gmrf_ldl_solve!(column, column, R.factor, R.pattern)
        for j in 1:n
            out[offset + permutation[j]] = R.scale * column[j]
        end
    end
    for t in 1:nt, c in 1:channels, i in 1:n
        offset = n * channels * (t - 1)
        value = 0.0
        for k in 1:c
            value += R.correlation_factor[c, k] * out[offset + n * (k - 1) + i]
        end
        work[offset + n * (c - 1) + i] = value
    end
    stride = n * channels
    for t in 1:nt, j in 1:stride
        index = stride * (t - 1) + j
        out[index] = R.innovation_scale[t] * work[index] +
            (t == 1 ? 0.0 : R.decay[t] * out[index - stride])
    end
    out
end

function _brm_gmrf_reference_transpose!(out, R::_BRMGMRFFixedReference, x, work)
    n, channels, nt = R.shape
    length(out) == length(x) == length(work) == n * channels * nt ||
        throw(DimensionMismatch("GMRF reference vector size"))
    stride = n * channels
    # Backward accumulation precedes the innovation scales: it is the exact
    # transpose of the forward OU recurrence, including its stationary first row.
    for t in nt:-1:1, j in 1:stride
        index = stride * (t - 1) + j
        out[index] = x[index] + (t == nt ? 0.0 : R.decay[t + 1] * out[index + stride])
    end
    for t in 1:nt, k in 1:channels, i in 1:n
        offset = stride * (t - 1)
        value = 0.0
        for c in k:channels
            value += R.correlation_factor[c, k] * out[offset + n * (c - 1) + i]
        end
        work[offset + n * (k - 1) + i] = R.innovation_scale[t] * value
    end
    permutation = R.pattern.permutation
    for t in 1:nt, c in 1:channels
        offset = n * (c - 1 + channels * (t - 1))
        column = view(out, offset+1:offset+n)
        for j in 1:n
            column[j] = work[offset + permutation[j]]
        end
        _brm_gmrf_ldl_solve!(column, column, R.factor, R.pattern)
        for j in 1:n
            work[offset + permutation[j]] = R.scale * R.mass_root[j] * column[j]
        end
        copyto!(column, view(work, offset+1:offset+n))
    end
    out
end

function _brm_to_physical!(out, A::_BRMPreconditionedOperator{H,R}, x) where
        {H,R<:_BRMGMRFFixedReference}
    _brm_gmrf_reference_forward!(out, A.reference, x, A.first)
end
function _brm_from_physical!(out, A::_BRMPreconditionedOperator{H,R}, x) where
        {H,R<:_BRMGMRFFixedReference}
    _brm_gmrf_reference_transpose!(out, A.reference, x, A.first)
end
