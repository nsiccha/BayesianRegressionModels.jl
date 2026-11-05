# Sparse finite-element Matérn precision and stationary OU transitions.
# Geometry/symbolic elimination depend only on data. Numeric factorization is
# ordinary scalar Julia over the fixed pattern, including when range is live.
# No space-time covariance or factor is formed by these operations.

struct _BRMGMRFFactorPattern
    permutation::Vector{Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    previous_ptr::Vector{Int}
    previous_col::Vector{Int}
    previous_pos::Vector{Int}
end

function _brm_gmrf_factor_pattern(G, mass)
    # kappa=1 only chooses a symbolic pattern. Every positive kappa has the
    # same structural pattern; numeric entries are recomputed at evaluation.
    F = cholesky(Symmetric(G + spdiagm(0 => mass)))
    L = sparse(F.L)
    n = length(mass)
    previous = [Tuple{Int,Int}[] for _ in 1:n]
    for j in 1:n, p in nzrange(L, j)
        i = rowvals(L)[p]
        i > j && push!(previous[i], (j, p))
    end
    ptr = ones(Int, n + 1)
    cols, positions = Int[], Int[]
    for j in 1:n
        for (k, p) in previous[j]
            push!(cols, k); push!(positions, p)
        end
        ptr[j + 1] = length(cols) + 1
    end
    _BRMGMRFFactorPattern(collect(F.p), copy(L.colptr), copy(rowvals(L)),
        ptr, cols, positions)
end

# Left-looking LDL' on a fixed lower-triangular CSC pattern. The workspace
# clears only the current column's fill, not all n rows for every column.
# The diagonal is stored separately; the factor's explicit diagonal is one.
function _brm_gmrf_ldl(mass_scale, stiffness_scale, mass, gptr, grow, gval, lptr, lrow,
        previous_ptr, previous_col, previous_pos)
    n = length(mass)
    T = promote_type(typeof(mass_scale), typeof(stiffness_scale), eltype(mass), eltype(gval))
    lower = zeros(T, length(lrow))
    diagonal = zeros(T, n)
    work = zeros(T, n)
    for j in 1:n
        for p in lptr[j]:(lptr[j + 1] - 1)
            work[lrow[p]] = zero(T)
        end
        for p in gptr[j]:(gptr[j + 1] - 1)
            i = grow[p]
            i >= j && (work[i] = stiffness_scale * gval[p])
        end
        work[j] += mass_scale * mass[j]
        for q in previous_ptr[j]:(previous_ptr[j + 1] - 1)
            k = previous_col[q]
            start = previous_pos[q]
            v = lower[start] * diagonal[k]
            for p in start:(lptr[k + 1] - 1)
                work[lrow[p]] -= lower[p] * v
            end
        end
        d = work[j]
        isfinite(d) && d > 0 || throw(DomainError(d,
            "gmrf spatial operator must be positive definite"))
        diagonal[j] = d
        lower[lptr[j]] = one(T)
        for p in (lptr[j] + 1):(lptr[j + 1] - 1)
            lower[p] = work[lrow[p]] / d
        end
    end
    (; lower, diagonal)
end

function _brm_gmrf_ldl(kappa2, mass, G, pattern::_BRMGMRFFactorPattern)
    _brm_gmrf_ldl(kappa2, one(kappa2), mass, G, pattern)
end

function _brm_gmrf_ldl(mass_scale, stiffness_scale, mass, G, pattern::_BRMGMRFFactorPattern)
    _brm_gmrf_ldl(mass_scale, stiffness_scale, mass, G.colptr, rowvals(G), nonzeros(G),
        pattern.colptr, pattern.rowval, pattern.previous_ptr,
        pattern.previous_col, pattern.previous_pos)
end

function _brm_gmrf_ldl_solve!(x, b, factor, pattern::_BRMGMRFFactorPattern)
    copyto!(x, b)
    n = length(x)
    for j in 1:n
        for p in (pattern.colptr[j] + 1):(pattern.colptr[j + 1] - 1)
            x[pattern.rowval[p]] -= factor.lower[p] * x[j]
        end
    end
    for j in 1:n
        x[j] /= factor.diagonal[j]
    end
    for j in n:-1:1
        for p in (pattern.colptr[j] + 1):(pattern.colptr[j + 1] - 1)
            x[j] -= factor.lower[p] * x[pattern.rowval[p]]
        end
    end
    x
end

struct _BRMGMRFMesh
    vertices::Matrix{Float64}
    cells::Matrix{Int}
    mass::Vector{Float64}
    stiffness::SparseMatrixCSC{Float64,Int}
    factor_mass::Vector{Float64}
    factor_stiffness::SparseMatrixCSC{Float64,Int}
    factor_pattern::_BRMGMRFFactorPattern
end

function _brm_gmrf_mesh(vertices::AbstractMatrix, cells::AbstractMatrix)
    size(vertices, 2) == 2 || throw(ArgumentError("gmrf mesh vertices must have two columns"))
    size(cells, 2) == 3 || throw(ArgumentError("gmrf mesh cells must have three vertex indices"))
    all(isfinite, vertices) || throw(ArgumentError("gmrf mesh vertices must be finite"))
    n = size(vertices, 1)
    n >= 3 && size(cells, 1) > 0 || throw(ArgumentError("gmrf mesh needs a nonempty triangulation"))
    all(i -> i isa Integer && 1 <= i <= n, cells) || throw(ArgumentError(
        "gmrf mesh cells must index the declared vertices with integers"))
    xy, triangles = Matrix{Float64}(vertices), Matrix{Int}(cells)
    mass = zeros(n)
    rows, cols, values = Int[], Int[], Float64[]
    seen = Set{NTuple{3,Int}}()
    edges = Dict{Tuple{Int,Int},Int}()
    for c in axes(triangles, 1)
        ids = Tuple(triangles[c, :])
        length(unique(ids)) == 3 || throw(ArgumentError("gmrf mesh cell $c repeats a vertex"))
        key = Tuple(sort(collect(ids)))
        key in seen && throw(ArgumentError("gmrf mesh repeats cell $c"))
        push!(seen, key)
        i, j, k = ids
        twice_area = abs((xy[j, 1] - xy[i, 1]) * (xy[k, 2] - xy[i, 2]) -
                         (xy[k, 1] - xy[i, 1]) * (xy[j, 2] - xy[i, 2]))
        isfinite(twice_area) && twice_area > 0 || throw(ArgumentError(
            "gmrf mesh cell $c is degenerate"))
        b = (xy[j, 2] - xy[k, 2], xy[k, 2] - xy[i, 2], xy[i, 2] - xy[j, 2])
        d = (xy[k, 1] - xy[j, 1], xy[i, 1] - xy[k, 1], xy[j, 1] - xy[i, 1])
        for a in 1:3
            mass[ids[a]] += twice_area / 6
            for z in 1:3
                push!(rows, ids[a]); push!(cols, ids[z])
                push!(values, (b[a] * b[z] + d[a] * d[z]) / (2 * twice_area))
            end
        end
        for (a, z) in ((i, j), (j, k), (k, i))
            edge = minmax(a, z)
            count = get(edges, edge, 0) + 1
            count <= 2 || throw(ArgumentError("gmrf mesh edge $edge belongs to more than two cells"))
            edges[edge] = count
        end
    end
    all(>(0), mass) || throw(ArgumentError("gmrf mesh has vertices belonging to no cell"))
    G = sparse(rows, cols, values, n, n)
    SparseArrays.dropzeros!(G)
    pattern = _brm_gmrf_factor_pattern(G, mass)
    p = pattern.permutation
    _BRMGMRFMesh(xy, triangles, mass, G, mass[p], G[p, p], pattern)
end

function _brm_gmrf_positive(value, name)
    isfinite(value) && value > 0 || throw(DomainError(value, "gmrf $name must be finite and positive"))
    value
end

# alpha=2 in two dimensions means nu=1. The mass-lumped finite-element
# approximation has Q = tau² K C^-1 K, K = kappa² C + G. Unit *continuum*
# marginal variance means tau² = 1/(4 pi kappa²). Finite-domain variances
# differ near boundaries; there is no claim of exact unit mesh variance.
function _brm_gmrf_spatial_logdet(mesh::_BRMGMRFMesh, spatial_length)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    # B = C/ell + (ell/2) G, so Q = B C^-1 B/(2pi). Factoring B
    # avoids the overflowing kappa² and its cancelling determinant terms.
    factor = _brm_gmrf_ldl(inv(spatial_length), spatial_length / 2, mesh.factor_mass,
        mesh.factor_stiffness, mesh.factor_pattern)
    2sum(log, factor.diagonal) - sum(log, mesh.mass) - length(mesh.mass) * log(2pi)
end

function _brm_gmrf_spatial_mul!(out, x, mesh::_BRMGMRFMesh, spatial_length, work)
    mul!(work, mesh.stiffness, x)
    for i in eachindex(work, x)
        work[i] = x[i] / spatial_length + (spatial_length / 2) * (work[i] / mesh.mass[i])
    end
    mul!(out, mesh.stiffness, work)
    for i in eachindex(out, work)
        out[i] = (mesh.mass[i] * (work[i] / spatial_length) + (spatial_length / 2) * out[i]) / (2pi)
    end
    out
end

function _brm_gmrf_times(times)
    all(isfinite, times) || throw(ArgumentError("gmrf times must be finite"))
    grid = sort!(unique(Float64.(times)))
    isempty(grid) && throw(ArgumentError("gmrf needs at least one time"))
    grid
end

# Prior state order is vertex, channel, time. The first time is stationary;
# every later transition uses its actual elapsed time, including long gaps.
# L is the channel correlation's lower Cholesky factor, never a spatial factor.
function _brm_gmrf_logpdf(state::AbstractArray{<:Real,3}, mesh::_BRMGMRFMesh,
        times, spatial_length, temporal_length, L::AbstractMatrix)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    n, channels, nt = size(state)
    n == length(mesh.mass) && nt == length(times) && size(L) == (channels, channels) ||
        throw(DimensionMismatch("gmrf state, support, times and channel correlation disagree"))
    all(>(0), diag(L)) || throw(ArgumentError("gmrf channel factor needs a positive diagonal"))
    all(i -> isfinite(times[i]) && (i == 1 || times[i] > times[i - 1]), eachindex(times)) ||
        throw(ArgumentError("gmrf times must be finite and strictly increasing"))
    spatial_logdet = _brm_gmrf_spatial_logdet(mesh, spatial_length)
    value = nt * channels * spatial_logdet / 2 - n * nt * sum(log, diag(L)) -
        (n * channels * nt / 2) * log(2pi)
    residual = Matrix{eltype(state)}(undef, n, channels)
    work = zeros(eltype(state), n)
    for t in 1:nt
        a = t == 1 ? zero(temporal_length) : exp(-(times[t] - times[t - 1]) / temporal_length)
        v = t == 1 ? one(temporal_length) : -expm1(-2 * (times[t] - times[t - 1]) / temporal_length)
        for c in 1:channels, i in 1:n
            residual[i, c] = state[i, c, t] - (t == 1 ? zero(eltype(state)) : a * state[i, c, t - 1])
        end
        # Right solve by L': one small channel solve at every vertex.
        for c in 1:channels, i in 1:n
            z = residual[i, c]
            for k in 1:(c - 1)
                z -= residual[i, k] * L[c, k]
            end
            residual[i, c] = z / L[c, c]
        end
        q = zero(value)
        for c in 1:channels
            mul!(work, mesh.stiffness, view(residual, :, c))
            for i in 1:n
                z = mesh.mass[i] * (residual[i, c] / spatial_length) +
                    (spatial_length / 2) * work[i]
                q += abs2(z / (sqrt(4pi * mesh.mass[i]) * sqrt(v)))
            end
        end
        value -= q + n * channels * log(v) / 2
    end
    value
end

# Apply Q_t ⊗ Omega^-1 ⊗ Q_s using two spatial sparse products and two
# small channel triangular solves per time. Storage is linear in the field;
# no Kronecker matrix, covariance, or space-time Cholesky factor is built.
function _brm_gmrf_precision_workspace(state)
    n, channels, _ = size(state)
    residual = Matrix{eltype(state)}(undef, n, channels)
    (; residual, spatial=similar(residual), work=Vector{eltype(state)}(undef, n))
end

function _brm_gmrf_precision_mul!(out, state, mesh::_BRMGMRFMesh,
        times, spatial_length, temporal_length, L)
    _brm_gmrf_precision_mul!(out, state, mesh, times, spatial_length, temporal_length, L,
        _brm_gmrf_precision_workspace(state))
end

function _brm_gmrf_precision_mul!(out, state, mesh::_BRMGMRFMesh,
        times, spatial_length, temporal_length, L, workspace)
    size(out) == size(state) || throw(DimensionMismatch("gmrf precision output shape"))
    n, channels, nt = size(state)
    n == length(mesh.mass) && nt == length(times) && size(L) == (channels, channels) ||
        throw(DimensionMismatch("gmrf precision support shape"))
    Base.mightalias(out, state) && throw(ArgumentError("gmrf precision input and output must not alias"))
    (; residual, spatial, work) = workspace
    size(residual) == size(spatial) == (n, channels) && length(work) == n ||
        throw(DimensionMismatch("gmrf precision workspace shape"))
    fill!(out, zero(eltype(out)))
    for t in 1:nt
        a = t == 1 ? zero(temporal_length) : exp(-(times[t] - times[t - 1]) / temporal_length)
        v = t == 1 ? one(temporal_length) : -expm1(-2 * (times[t] - times[t - 1]) / temporal_length)
        for c in 1:channels, i in 1:n
            residual[i, c] = state[i, c, t] - (t == 1 ? zero(eltype(state)) : a * state[i, c, t - 1])
        end
        for c in 1:channels
            _brm_gmrf_spatial_mul!(view(spatial, :, c), view(residual, :, c), mesh, spatial_length, work)
        end
        for c in 1:channels, i in 1:n
            z = spatial[i, c]
            for k in 1:(c - 1)
                z -= residual[i, k] * L[c, k]
            end
            residual[i, c] = z / L[c, c]
        end
        for c in channels:-1:1, i in 1:n
            z = residual[i, c]
            for k in (c + 1):channels
                z -= residual[i, k] * L[k, c]
            end
            residual[i, c] = z / L[c, c]
        end
        for c in 1:channels, i in 1:n
            z = residual[i, c] / v
            out[i, c, t] += z
            t > 1 && (out[i, c, t - 1] -= a * z)
        end
    end
    out
end

# A conditional solver's diagonal preconditioner needs only the diagonals
# of the three separable precisions. The spatial diagonal is a sum over K's
# sparse columns; no space-time sparse matrix or factor is constructed.
function _brm_gmrf_precision_diagonal!(out, mesh::_BRMGMRFMesh,
        times, spatial_length, temporal_length, L)
    n, channels, nt = size(out)
    n == length(mesh.mass) && nt == length(times) && size(L) == (channels, channels) ||
        throw(DimensionMismatch("gmrf precision diagonal support shape"))
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    G, mass = mesh.stiffness, mesh.mass
    spatial = zeros(eltype(out), n)
    for j in 1:n
        value = zero(eltype(out))
        for p in nzrange(G, j)
            i = rowvals(G)[p]
            entry = (spatial_length / 2) * nonzeros(G)[p]
            i == j && (entry += mass[j] / spatial_length)
            value += abs2(entry / sqrt(2pi * mass[i]))
        end
        spatial[j] = value
    end
    inverse_factor = inv(LowerTriangular(L))
    channel = vec(sum(abs2, inverse_factor; dims=1))
    temporal = zeros(eltype(out), nt)
    temporal[1] = 1
    for t in 2:nt
        gap = times[t] - times[t - 1]
        a = exp(-gap / temporal_length)
        variance = -expm1(-2gap / temporal_length)
        temporal[t] += inv(variance)
        temporal[t - 1] += a^2 / variance
    end
    for t in 1:nt, c in 1:channels, i in 1:n
        out[i, c, t] = spatial[i] * channel[c] * temporal[t]
    end
    out
end

# A solver-facing view of the separable prior, with reusable spatial buffers.
# Each prepared solver owns this mutable workspace; no field-sized factor or
# precision entries are retained by the operator.
struct _BRMGMRFPrecisionOperator{M,T,C,W}
    mesh::M
    times::T
    spatial_length::Float64
    temporal_length::Float64
    correlation_factor::C
    shape::NTuple{3,Int}
    ncoordinates::Int
    workspace::W
end
function _BRMGMRFPrecisionOperator(mesh::_BRMGMRFMesh, times, spatial_length,
        temporal_length, L)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    size(L, 1) == size(L, 2) && all(>(0), diag(L)) ||
        throw(ArgumentError("gmrf channel factor must be square with a positive diagonal"))
    layout = _BRMGMRFStateLayout(length(mesh.mass), collect(1:size(L, 1)), times)
    shape = (layout.nvertices, length(layout.channels), length(layout.times))
    workspace = _brm_gmrf_precision_workspace(zeros(shape[1], shape[2], 1))
    _BRMGMRFPrecisionOperator(mesh, layout.times, Float64(spatial_length),
        Float64(temporal_length), Matrix{Float64}(L), shape, layout.ncoordinates, workspace)
end
Base.size(Q::_BRMGMRFPrecisionOperator) = (Q.ncoordinates, Q.ncoordinates)
Base.size(Q::_BRMGMRFPrecisionOperator, d::Integer) = size(Q)[d]
Base.eltype(::Type{<:_BRMGMRFPrecisionOperator}) = Float64
Base.eltype(::_BRMGMRFPrecisionOperator) = Float64
function LinearAlgebra.mul!(out::AbstractVector, Q::_BRMGMRFPrecisionOperator,
        x::AbstractVector)
    length(out) == length(x) == Q.ncoordinates || throw(DimensionMismatch(
        "gmrf precision vector size"))
    _brm_gmrf_precision_mul!(reshape(out, Q.shape), reshape(x, Q.shape), Q.mesh,
        Q.times, Q.spatial_length, Q.temporal_length, Q.correlation_factor, Q.workspace)
    out
end
(Q::_BRMGMRFPrecisionOperator)(x) = mul!(similar(x, Float64), Q, x)
function LinearAlgebra.diag(Q::_BRMGMRFPrecisionOperator)
    out = zeros(Q.ncoordinates)
    _brm_gmrf_precision_diagonal!(reshape(out, Q.shape), Q.mesh, Q.times,
        Q.spatial_length, Q.temporal_length, Q.correlation_factor)
    out
end
