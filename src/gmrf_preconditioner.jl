# A fixed, data-aware reference for one `gmrf` field's conditional precision,
#   H = Q_t ⊗ Λ ⊗ Q_s + D,   Λ = (L Lᵀ)⁻¹,
# with D the likelihood's curvature. One chain over the times, for a prior scale
# λ and data E_t on a vertex-by-vertex block,
#   T = Σ_t (y_t − G_t y_{t−1})ᵀ S_t (y_t − G_t y_{t−1}),
#   S_t = λ Q_s / v_t + E_t,   G_t = S_t⁻¹ λ (a_t / v_t) Q_s,
# a_t = exp(−gap / ℓ_t), v_t = 1 − a_t² (v_1 = 1). T's blocks between neighbouring
# times equal those of λ Q_t ⊗ Q_s + E. At one time they lack
# λ (a²/v)(Q_s/v − λ Q_s S⁻¹ Q_s), which only the data make nonzero, so for E ⪰ 0
# the transformed spectrum lies at or above one. A prior-only reference (E = 0)
# has the same shape and is much worse: it ignores the data.
#
# Channels combine the chains in one of two ways, and neither is uniformly good:
#  * `decoupled`: in the eigenbasis Λ = V diag(λ) Vᵀ, one independent chain per
#    eigenchannel with E_j = Σ_c V_cj² D_c. Exact across channels when every
#    channel has the same data, whatever the correlation.
#  * `sequential`: in the raw channels, where D is block diagonal, the chain over
#    the channels M = Σ_c (x_c − G_c x_<c)ᵀ T_c (x_c − G_c x_<c), with T_c the time
#    chain of channel c (λ = U_cc², U = L⁻¹, E = D_c), G_c x_<c = −T_c⁻¹ U_cc
#    Σ_{k<c} U_ck (Q_t ⊗ Q_s) x_k. It holds when channels are read at different
#    places, and each of its applications costs about twice as much.
#    (The reverse nesting, channels inside times, is unstable over many times.)
# A correlated field builds both at preparation and keeps the one a fixed short
# trial shows converging faster, so the choice depends only on the target.
#
# One sparse spatial factor per chain and time on a shared fixed pattern; no
# space-time factor or Kronecker matrix. Whitened coordinates of a chain's time
# are in the factor's permuted order.
struct _BRMGMRFTimeReference
    pattern::_BRMGMRFFactorPattern
    spatial::SparseMatrixCSC{Float64,Int}
    # The spatial pattern's permuted storage, the unpermuted position of each of its
    # entries, and each vertex's diagonal position: what any S on it needs.
    permuted::SparseMatrixCSC{Float64,Int}
    order::Vector{Int}
    position::Vector{Int}
    lower::Matrix{Vector{Float64}}
    root::Matrix{Vector{Float64}}
    scale::Vector{Float64}
    decay::Vector{Float64}
    temporal_diagonal::Vector{Float64}
    temporal_off::Vector{Float64}
    # The eigenvectors V (`decoupled`), or nothing.
    basis::Union{Nothing,Matrix{Float64}}
    # U_cc U_ck (`sequential`), or nothing.
    sequential::Union{Nothing,Matrix{Float64}}
    state::Vector{Int}
    shape::NTuple{3,Int}
    logdet::Float64
    factor_entries::Int
    diagonal_shift::Float64
end
# Read-only once prepared, whatever the size: chains share it.
Base.deepcopy_internal(R::_BRMGMRFTimeReference, ::IdDict) = R

_brm_reference_binding(R::_BRMGMRFTimeReference) = (; R.shape, R.pattern.permutation,
    R.pattern.colptr, R.pattern.rowval, lower=reduce(vcat, vec(R.lower)),
    root=reduce(vcat, vec(R.root)), R.scale, R.decay, R.basis, R.sequential, R.logdet)

# The field that is exactly component `c`.
function _brm_gmrf_component_field(p, c)
    p.gmrf === nothing && return nothing
    I = p.components[c]
    k = findfirst(f -> length(f.z) == length(I) && p.component[first(f.z)] == c, p.gmrf.fields)
    k === nothing ? nothing : p.gmrf.fields[k]
end

# The fields that together are exactly component `c` (several read by shared rows).
function _brm_gmrf_component_fields(p, c)
    p.gmrf === nothing && return nothing
    fields = [f for f in p.gmrf.fields if p.component[first(f.z)] == c]
    sum(f -> length(f.z), fields; init=0) == length(p.components[c]) && length(fields) >= 2 ?
        fields : nothing
end

# One S's factor on the shared pattern, shifting its diagonal only if the
# likelihood's curvature at the reference point is not positive semidefinite.
function _brm_gmrf_chain_factor(values, permuted, order, diagonal, pattern)
    n = size(permuted, 2)
    gval = values[order]
    shift = 0.0
    scale = maximum(abs, view(values, diagonal))
    while true
        try
            return (; factor=_brm_gmrf_ldl(0.0, zeros(n), permuted.colptr, rowvals(permuted), gval,
                pattern.colptr, pattern.rowval, pattern.previous_ptr, pattern.previous_col,
                pattern.previous_pos), shift)
        catch err
            err isa DomainError || rethrow()
            added = shift == 0 ? 1e-8scale : shift
            shift += added
            for j in 1:n, k in nzrange(permuted, j)
                rowvals(permuted)[k] == j && (gval[k] += added)
            end
            shift <= 1e6scale || throw(_BRMOperatorSolveFailure(
                "the gmrf reference's spatial factor is not positive definite at any bounded shift"))
        end
    end
end

# `H` is the component's operator at the reference point, `h` the problem's
# precision there, and the field's factors hold that point's values. `mode` is
# `:auto`, `:decoupled` or `:sequential` (tests name a combination to check it).
# `data=false` leaves the likelihood out: a star's other fields.
function _brm_gmrf_time_reference(field, H, h, localindex; mode=:auto, data=true)
    nv, nc, nt = field.shape
    D = h isa _BRMSumPrecision ? h.D : h
    values = nonzeros(D)
    ns = length(field.qs)
    curvature = zeros(data ? ns : 0, nc, nt)
    sp = field.spatial.pattern
    state = localindex[field.z]
    position = [only(k for k in nzrange(sp, j) if rowvals(sp)[k] == j) for j in 1:nv]
    if data
        _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
            ta == tb && ca == cb && (curvature[ks, ca, ta] = values[field.nz[k]] -
                field.qt[ta] * field.Λ[ca, ca] * field.qs[ks])
        end
        # The diagonal of a reduction's B'WB: it stays out of H.D, as in the LDL reference.
        Dd = diag(H.D)
        for t in 1:nt, c in 1:nc, i in 1:nv
            l = state[i + nv * ((c - 1) + nc * (t - 1))]
            curvature[position[i], c, t] += H.diagonal[l] - Dd[l]
        end
    end
    spatial = SparseMatrixCSC(nv, nv, copy(sp.colptr), copy(rowvals(sp)), copy(field.qs))
    pattern = _brm_gmrf_factor_pattern(spatial)
    p = pattern.permutation
    # Map the fixed pattern to its permuted storage once: every S shares it.
    tracked = SparseMatrixCSC(nv, nv, sp.colptr, rowvals(sp), collect(1.0:ns))[p, p]
    order = Int.(nonzeros(tracked))
    lt = field.θ[2]
    a, v = zeros(nt), ones(nt)
    for t in 2:nt
        gap = (field.times[t] - field.times[t - 1]) / lt
        a[t] = exp(-gap)
        v[t] = -expm1(-2gap)
    end
    shared = (; pattern, spatial, tracked, order, position, curvature, a, v, state)
    U = Matrix(inv(LowerTriangular(Matrix(field.L))))
    diagonal = all(iszero, U .- Diagonal(diag(U)))
    decoupled() = begin
        E = diagonal ? nothing : eigen(Symmetric(Matrix(field.Λ)))
        scale = E === nothing ? diag(U) .^ 2 : E.values
        V = E === nothing ? Matrix{Float64}(I, nc, nc) : E.vectors
        _brm_gmrf_chains(field, shared, scale, V .^ 2, V, nothing)
    end
    sequential() = _brm_gmrf_chains(field, shared, diag(U) .^ 2, Matrix{Float64}(I, nc, nc),
        nothing, [c > k ? U[c, c] * U[c, k] : 0.0 for c in 1:nc, k in 1:nc])
    mode === :decoupled && return decoupled()
    mode === :sequential && return sequential()
    mode === :auto || throw(ArgumentError("unknown gmrf reference mode $mode"))
    (nc == 1 || diagonal) && return decoupled()
    candidates = (decoupled(), sequential())
    trial = 2.0 .* rand(Random.Xoshiro(_BRM_GMRF_REFERENCE_TRIAL_SEED), Bool, size(H, 1)) .- 1
    residuals = map(candidates) do R
        _brm_lanczos_run(_BRMPreconditionedOperator(H, R), trial, 0.0,
            _BRM_GMRF_REFERENCE_TRIAL_ITERATIONS).residual
    end
    candidates[residuals[2] < residuals[1] ? 2 : 1]
end
const _BRM_GMRF_REFERENCE_TRIAL_SEED = 20261104
const _BRM_GMRF_REFERENCE_TRIAL_ITERATIONS = 24

# The factors of chains with these prior `scales` and data `weights[c, j]`.
function _brm_gmrf_chains(field, shared, scales, weights, basis, sequential)
    nv, nc, nt = field.shape
    (; pattern, spatial, tracked, order, position, curvature, a, v, state) = shared
    ns = length(field.qs)
    lower = Matrix{Vector{Float64}}(undef, nc, nt)
    root = Matrix{Vector{Float64}}(undef, nc, nt)
    logdetM = 0.0
    entries = nt + nc^2
    largest_shift = 0.0
    chain = zeros(ns)
    # Without data a chain's S is the same at every equal time gap: share its factor.
    shared_factor = Dict{Float64,Any}()
    for t in 1:nt, j in 1:nc
        scale = scales[j] / v[t]
        if isempty(curvature) && haskey(shared_factor, scale)
            known = shared_factor[scale]
            lower[j, t], root[j, t] = known.lower, known.root
            logdetM += known.logdet
            continue
        end
        chain .= scale .* field.qs
        isempty(curvature) || for c in 1:nc
            weight = weights[c, j]
            iszero(weight) && continue
            for k in 1:ns
                chain[k] += weight * curvature[k, c, t]
            end
        end
        (; factor, shift) = _brm_gmrf_chain_factor(chain, tracked, order, position, pattern)
        largest_shift = max(largest_shift, shift)
        lower[j, t] = factor.lower
        root[j, t] = sqrt.(factor.diagonal)
        logdetM += sum(log, factor.diagonal)
        entries += length(factor.lower)
        isempty(curvature) && (shared_factor[scale] = (; lower=lower[j, t], root=root[j, t],
            logdet=sum(log, factor.diagonal)))
    end
    _BRMGMRFTimeReference(pattern, spatial, tracked, order, position, lower, root,
        Vector{Float64}(scales), a ./ v,
        copy(field.qt), copy(field.qo), basis, sequential, state, field.shape,
        logdetM, entries, largest_shift)
end

_brm_gmrf_work(nv) = (; a=zeros(nv), b=zeros(nv), c=zeros(nv), previous=zeros(nv))

# w ← C⁻¹ y on chain j, consuming y's entries of that chain.
function _brm_gmrf_chain_whiten!(w, y, R, j, work)
    nv, nc, nt = R.shape
    p = R.pattern.permutation
    stride = nv * nc
    a, b, c = work.a, work.b, work.c
    for t in nt:-1:1
        offset = nv * (j - 1) + stride * (t - 1)
        lower, root = R.lower[j, t], R.root[j, t]
        @inbounds for i in 1:nv
            a[i] = y[offset + p[i]]
        end
        _brm_gmrf_ldl_forward!(a, lower, R.pattern)
        @inbounds for i in 1:nv
            a[i] /= root[i]
            w[offset + i] = a[i]
        end
        t == 1 && continue
        @inbounds for i in 1:nv
            a[i] /= root[i]
        end
        _brm_gmrf_ldl_backward!(a, lower, R.pattern)
        @inbounds for i in 1:nv
            b[p[i]] = a[i]
        end
        mul!(c, R.spatial, b)
        coupling = R.scale[j] * R.decay[t]
        @inbounds for i in 1:nv
            y[offset - stride + i] += coupling * c[i]
        end
    end
    w
end

# x ← C⁻ᵀ w on chain j.
function _brm_gmrf_chain_unwhiten!(x, w, R, j, work)
    nv, nc, nt = R.shape
    p = R.pattern.permutation
    stride = nv * nc
    a, b, c, previous = work.a, work.b, work.c, work.previous
    for t in 1:nt
        offset = nv * (j - 1) + stride * (t - 1)
        lower, root = R.lower[j, t], R.root[j, t]
        @inbounds for i in 1:nv
            a[i] = w[offset + i]
        end
        if t > 1
            mul!(c, R.spatial, previous)
            coupling = R.scale[j] * R.decay[t]
            @inbounds for i in 1:nv
                b[i] = coupling * c[p[i]]
            end
            _brm_gmrf_ldl_forward!(b, lower, R.pattern)
            @inbounds for i in 1:nv
                a[i] += b[i] / root[i]
            end
        end
        @inbounds for i in 1:nv
            a[i] /= root[i]
        end
        _brm_gmrf_ldl_backward!(a, lower, R.pattern)
        @inbounds for i in 1:nv
            previous[p[i]] = a[i]
        end
        @inbounds for i in 1:nv
            x[offset + i] = previous[i]
        end
    end
    x
end

# out[all channels k < c] ±= weight[c, k] (Q_t ⊗ Q_s) x[channel c].
function _brm_gmrf_sequential_coupling!(out, x, R, c, sign)
    nv, nc, nt = R.shape
    stride = nv * nc
    product = Matrix{Float64}(undef, nv, nt)
    for t in 1:nt
        offset = nv * (c - 1) + stride * (t - 1)
        mul!(view(product, :, t), R.spatial, view(x, (offset + 1):(offset + nv)))
    end
    weights = R.sequential
    @inbounds for k in 1:(c - 1)
        weight = sign * weights[c, k]
        iszero(weight) && continue
        for t in 1:nt
            offset = nv * (k - 1) + stride * (t - 1)
            d = R.temporal_diagonal[t]
            below = t == 1 ? 0.0 : R.temporal_off[t - 1]
            above = t == nt ? 0.0 : R.temporal_off[t]
            for i in 1:nv
                value = d * product[i, t]
                t > 1 && (value += below * product[i, t - 1])
                t < nt && (value += above * product[i, t + 1])
                out[offset + i] += weight * value
            end
        end
    end
    out
end

# y ← (I ⊗ V ⊗ I) y and y ← (I ⊗ Vᵀ ⊗ I) y, over each time and vertex.
function _brm_gmrf_channel_basis!(y, V, nv, nc, nt, transposed)
    scratch = zeros(nc)
    @inbounds for t in 1:nt, i in 1:nv
        offset = i + nv * nc * (t - 1)
        for c in 1:nc
            scratch[c] = y[offset + nv * (c - 1)]
        end
        for c in 1:nc
            value = 0.0
            for k in 1:nc
                value += (transposed ? V[k, c] : V[c, k]) * scratch[k]
            end
            y[offset + nv * (c - 1)] = value
        end
    end
    y
end

_brm_from_physical!(out, A::_BRMPreconditionedOperator{H,_BRMGMRFTimeReference}, b) where {H} =
    _brm_gmrf_whiten!(out, A.reference, b, A.first, A.physical)
_brm_to_physical!(out, A::_BRMPreconditionedOperator{H,_BRMGMRFTimeReference}, x) where {H} =
    _brm_gmrf_unwhiten!(out, A.reference, x, A.first, A.product)

# out ← C⁻¹ b and out ← C⁻ᵀ x on this reference's coordinates of the component,
# whose other entries they neither read nor write; y and w are scratch.
function _brm_gmrf_whiten!(out, R::_BRMGMRFTimeReference, b, y, w)
    nv, nc, nt = R.shape
    state = R.state
    @inbounds for k in eachindex(state)
        y[k] = b[state[k]]
    end
    work = _brm_gmrf_work(nv)
    if R.sequential === nothing
        nc == 1 || _brm_gmrf_channel_basis!(y, R.basis, nv, nc, nt, true)
        for j in 1:nc
            _brm_gmrf_chain_whiten!(w, y, R, j, work)
        end
    else
        solved = zeros(length(state))
        for c in nc:-1:1
            _brm_gmrf_chain_whiten!(w, y, R, c, work)
            c == 1 && break
            _brm_gmrf_chain_unwhiten!(solved, w, R, c, work)
            _brm_gmrf_sequential_coupling!(y, solved, R, c, -1.0)
        end
    end
    @inbounds for k in eachindex(state)
        out[state[k]] = w[k]
    end
    out
end

function _brm_gmrf_unwhiten!(out, R::_BRMGMRFTimeReference, x, y, w)
    nv, nc, nt = R.shape
    state = R.state
    @inbounds for k in eachindex(state)
        w[k] = x[state[k]]
    end
    work = _brm_gmrf_work(nv)
    if R.sequential === nothing
        for j in 1:nc
            _brm_gmrf_chain_unwhiten!(y, w, R, j, work)
        end
        nc == 1 || _brm_gmrf_channel_basis!(y, R.basis, nv, nc, nt, false)
    else
        fill!(y, 0.0)
        accumulated, whitened = zeros(length(state)), zeros(length(state))
        for c in 1:nc
            if c > 1
                # w_c − C_c⁻¹ Σ_{k<c} weight (Q_t ⊗ Q_s) x_k.
                fill!(accumulated, 0.0)
                _brm_gmrf_sequential_gather!(accumulated, y, R, c)
                _brm_gmrf_chain_whiten!(whitened, accumulated, R, c, work)
                @inbounds for t in 1:nt, i in 1:nv
                    index = nv * (c - 1) + nv * nc * (t - 1) + i
                    w[index] -= whitened[index]
                end
            end
            _brm_gmrf_chain_unwhiten!(y, w, R, c, work)
        end
    end
    @inbounds for k in eachindex(state)
        out[state[k]] = y[k]
    end
    out
end

# out[channel c] = (Q_t ⊗ Q_s) Σ_{k<c} weight[c, k] x[channel k].
function _brm_gmrf_sequential_gather!(out, x, R, c)
    nv, nc, nt = R.shape
    stride = nv * nc
    combined = zeros(nv, nt)
    weights = R.sequential
    @inbounds for k in 1:(c - 1), t in 1:nt
        weight = weights[c, k]
        iszero(weight) && continue
        offset = nv * (k - 1) + stride * (t - 1)
        for i in 1:nv
            combined[i, t] += weight * x[offset + i]
        end
    end
    product = similar(combined)
    for t in 1:nt
        mul!(view(product, :, t), R.spatial, view(combined, :, t))
    end
    @inbounds for t in 1:nt
        offset = nv * (c - 1) + stride * (t - 1)
        d = R.temporal_diagonal[t]
        for i in 1:nv
            value = d * product[i, t]
            t > 1 && (value += R.temporal_off[t - 1] * product[i, t - 1])
            t < nt && (value += R.temporal_off[t] * product[i, t + 1])
            out[offset + i] = value
        end
    end
    out
end

# Several fields that shared rows couple into one component. Curvature between
# them comes from their common data, so a diagonal block of references fails:
# the data pin only sums of fields and leave their differences to the priors. In
#   M = Σ_{g≠h} x_gᵀ P_g x_g + (x_h + T_h⁻¹ Σ_{g≠h} D_hg x_g)ᵀ T_h (x_h + T_h⁻¹ Σ_{g≠h} D_hg x_g)
# one hub h takes the data (T_h its time-block reference), the others only their
# priors (P_g), and M's blocks between the hub and another field equal H's. Which
# field makes the best hub varies by orders of magnitude (matrix level, 20,000
# states: 35 iterations against 800 for the same two fields), so preparation
# measures each on a fixed short trial. There is no control variate here yet.
struct _BRMGMRFStarReference
    members::Vector{_BRMGMRFTimeReference}
    hub::Int
    coupling::SparseMatrixCSC{Float64,Int}
    logdet::Float64
    factor_entries::Int
    diagonal_shift::Float64
end
Base.deepcopy_internal(R::_BRMGMRFStarReference, ::IdDict) = R
_brm_reference_binding(R::_BRMGMRFStarReference) = (; members=map(_brm_reference_binding, R.members),
    R.hub, R.coupling.colptr, R.coupling.rowval, R.coupling.nzval, R.logdet)

function _brm_gmrf_star_reference(fields, H, h, localindex)
    n = size(H, 1)
    priors = [_brm_gmrf_time_reference(f, H, h, localindex; mode=:decoupled, data=false) for f in fields]
    trial = 2.0 .* rand(Random.Xoshiro(_BRM_GMRF_REFERENCE_TRIAL_SEED), Bool, n) .- 1
    best, best_residual = nothing, Inf
    for (k, field) in enumerate(fields)
        members = copy(priors)
        members[k] = _brm_gmrf_time_reference(field, H, h, localindex)
        hub = falses(n)
        hub[members[k].state] .= true
        rows, cols, values = Int[], Int[], Float64[]
        D = H.D
        for j in findall(hub), a in nzrange(D, j)
            i = rowvals(D)[a]
            hub[i] && continue
            push!(rows, i); push!(cols, j); push!(values, nonzeros(D)[a])
        end
        R = _BRMGMRFStarReference(members, k, sparse(rows, cols, values, n, n),
            sum(m -> m.logdet, members), sum(m -> m.factor_entries, members),
            maximum(m -> m.diagonal_shift, members))
        residual = _brm_lanczos_run(_BRMPreconditionedOperator(H, R), trial, 0.0,
            _BRM_GMRF_REFERENCE_TRIAL_ITERATIONS).residual
        residual < best_residual && ((best, best_residual) = (R, residual))
    end
    something(best, first(priors))
end

function _brm_from_physical!(out, A::_BRMPreconditionedOperator{H,_BRMGMRFStarReference}, b) where {H}
    R = A.reference
    hub = R.members[R.hub]
    y, w = A.first, A.physical
    _brm_gmrf_whiten!(out, hub, b, y, w)
    solved = zeros(length(b))
    _brm_gmrf_unwhiten!(solved, hub, out, y, w)
    rest = collect(b)
    mul!(rest, R.coupling, solved, -1.0, 1.0)
    for (k, member) in enumerate(R.members)
        k == R.hub || _brm_gmrf_whiten!(out, member, rest, y, w)
    end
    out
end

function _brm_to_physical!(out, A::_BRMPreconditionedOperator{H,_BRMGMRFStarReference}, x) where {H}
    R = A.reference
    hub = R.members[R.hub]
    y, w = A.first, A.product
    others = zeros(length(x))
    for (k, member) in enumerate(R.members)
        k == R.hub || _brm_gmrf_unwhiten!(others, member, x, y, w)
    end
    carried = transpose(R.coupling) * others
    whitened = zeros(length(x))
    _brm_gmrf_whiten!(whitened, hub, carried, y, w)
    whitened .= x .- whitened
    copyto!(out, others)
    _brm_gmrf_unwhiten!(out, hub, whitened, y, w)
    out
end
