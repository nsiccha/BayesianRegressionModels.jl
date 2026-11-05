# The reference family as a control variate for a field's determinant. A fixed
# probe's quadratic form in a sampled operator carries noise that fixing the
# reference does not remove: its gradient's noise is that of `r' C⁻¹ dH C⁻ᵀ r`,
# and with noise growing as √N and information as N the mode of one fixed-probe
# target stays a fixed number of posterior standard deviations from the exact
# one at every size. So the field's reference is built again at each point,
# M(θ) = M(Q(θ), D(θ, ẑ(θ))), whose determinant and its derivative are exact (a
# sum over one sparse spatial factor per chain and time), and only the difference
# from it is estimated, with the fixed reference C̄'s probes:
#   V(θ) = log det M(θ) + r' [log(C̄⁻¹ H(θ) C̄⁻ᵀ) − log(C̄⁻¹ M(θ) C̄⁻ᵀ)] r.
# E[r' log(C̄⁻¹ X C̄⁻ᵀ) r] = log det X − log det C̄C̄ᵀ, so V estimates log det H for
# any M. It stays one smooth fixed-probe function of θ, and the gradient is its
# exact derivative.
#
# One chain has, with β_t = a_t / v_t, P = λ Q_s and w_t = S_t⁻¹ P u_{t−1},
#   uᵀ M u = Σ_t uₜᵀ S_t uₜ − 2 β_t uₜᵀ P u_{t−1} + β_t² u_{t−1}ᵀ P S_t⁻¹ P u_{t−1},
# whose derivative in S_t is u_t u_tᵀ − β_t² w_t w_tᵀ, in P is the symmetric part of
# −2 β_t u_t u_{t−1}ᵀ + 2 β_t² w_t u_{t−1}ᵀ, and in β_t is −2 uₜᵀ P u_{t−1} + 2 β_t u_{t−1}ᵀ P w_t.
# Channels use the fixed reference's basis V̄, with λ_j(θ) = v̄_jᵀ Λ(θ) v̄_j and data
# E_tj = Σ_c V̄_cj² D_tc, so only a `decoupled` reference has this control variate.

mutable struct _BRMGMRFTracked
    reference::_BRMGMRFTimeReference
    spatial::SparseMatrixCSC{Float64,Int}
    derivative::SparseMatrixCSC{Float64,Int}
    chains::Matrix{SparseMatrixCSC{Float64,Int}}
    lower::Matrix{Vector{Float64}}
    diagonal::Matrix{Vector{Float64}}
    scale::Vector{Float64}
    v::Vector{Float64}
    β::Vector{Float64}
    dinverse_v::Vector{Float64}
    dβ::Vector{Float64}
    logdet::Float64
    shift::Float64
    # Exact and stochastic derivatives of V in M's arguments, accumulated.
    block::Array{Float64,3}
    dspatial::Float64
    dtemporal::Float64
    dscale::Vector{Float64}
end
Base.size(M::_BRMGMRFTracked) = (length(M.reference.state), length(M.reference.state))
Base.size(M::_BRMGMRFTracked, d::Integer) = size(M)[d]
Base.eltype(::_BRMGMRFTracked) = Float64
Base.eltype(::Type{_BRMGMRFTracked}) = Float64

# The likelihood's curvature on each time and channel of a field, on the spatial
# pattern: h's entries less the prior's, at the field's current factors.
function _brm_gmrf_field_data(field, h)
    nv, nc, nt = field.shape
    D = h isa _BRMSumPrecision ? h.D : h
    values = nonzeros(D)
    data = zeros(length(field.qs), nc, nt)
    _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
        ta == tb && ca == cb && (data[ks, ca, ta] = values[field.nz[k]] -
            field.qt[ta] * field.Λ[ca, ca] * field.qs[ks])
    end
    data
end

# Rebuild a reference at the field's current factors, on the data given (the current
# curvature unless a star fixes it). The chain basis stays the fixed reference's.
function _brm_gmrf_tracked(field, reference::_BRMGMRFTimeReference, h;
        data=_brm_gmrf_field_data(field, h))
    reference.sequential === nothing || return nothing
    nv, nc, nt = field.shape
    ns = length(field.qs)
    sp = field.spatial.pattern
    colptr, rowval = sp.colptr, rowvals(sp)
    spatial = SparseMatrixCSC(nv, nv, colptr, rowval, copy(field.qs))
    derivative = SparseMatrixCSC(nv, nv, colptr, rowval, copy(field.dqs))
    V = reference.basis
    scale = [dot(view(V, :, j), field.Λ, view(V, :, j)) for j in 1:nc]
    lt = field.θ[2]
    v, β, dinverse_v, dβ = ones(nt), zeros(nt), zeros(nt), zeros(nt)
    for t in 2:nt
        gap = field.times[t] - field.times[t - 1]
        a = exp(-gap / lt)
        v[t] = -expm1(-2gap / lt)
        da = a * gap / lt^2
        dv = -2a * da
        β[t] = a / v[t]
        dinverse_v[t] = -dv / v[t]^2
        dβ[t] = (da * v[t] - a * dv) / v[t]^2
    end
    (; order, position) = reference
    chains = Matrix{SparseMatrixCSC{Float64,Int}}(undef, nc, nt)
    lower = Matrix{Vector{Float64}}(undef, nc, nt)
    diagonal = Matrix{Vector{Float64}}(undef, nc, nt)
    logdetM = 0.0
    largest_shift = 0.0
    for t in 1:nt, j in 1:nc
        chain = (scale[j] / v[t]) .* field.qs
        for c in 1:nc
            weight = V[c, j]^2
            iszero(weight) && continue
            for k in 1:ns
                chain[k] += weight * data[k, c, t]
            end
        end
        (; factor, shift) = _brm_gmrf_chain_factor(chain, reference.permuted, order, position, reference.pattern)
        largest_shift = max(largest_shift, shift)
        shift > 0 && (view(chain, position) .+= shift)
        chains[j, t] = SparseMatrixCSC(nv, nv, colptr, rowval, chain)
        lower[j, t] = factor.lower
        diagonal[j, t] = factor.diagonal
        logdetM += sum(log, factor.diagonal)
    end
    _BRMGMRFTracked(reference, spatial, derivative, chains, lower, diagonal, scale, v, β,
        dinverse_v, dβ, logdetM, largest_shift, zeros(ns, nc, nt), 0.0, 0.0, zeros(nc))
end

# out ← S_t⁻¹ x for chain j, in the state's vertex order.
function _brm_gmrf_tracked_solve!(out, M::_BRMGMRFTracked, j, t, x, work)
    R = M.reference
    p = R.pattern.permutation
    @inbounds for i in eachindex(work)
        work[i] = x[p[i]]
    end
    _brm_gmrf_ldl_forward!(work, M.lower[j, t], R.pattern)
    work ./= M.diagonal[j, t]
    _brm_gmrf_ldl_backward!(work, M.lower[j, t], R.pattern)
    @inbounds for i in eachindex(work)
        out[p[i]] = work[i]
    end
    out
end

# y ← M(θ) x in the component's coordinates.
function LinearAlgebra.mul!(y, M::_BRMGMRFTracked, x)
    R = M.reference
    nv, nc, nt = R.shape
    state = R.state
    u = Vector{Float64}(undef, length(state))
    @inbounds for k in eachindex(state)
        u[k] = x[state[k]]
    end
    nc == 1 || _brm_gmrf_channel_basis!(u, R.basis, nv, nc, nt, true)
    out = zeros(length(state))
    stride = nv * nc
    product = Matrix{Float64}(undef, nv, nt)
    a, solved, carried, work = zeros(nv), zeros(nv), zeros(nv), zeros(nv)
    for j in 1:nc
        for t in 1:nt
            offset = nv * (j - 1) + stride * (t - 1)
            mul!(view(product, :, t), M.spatial, view(u, (offset + 1):(offset + nv)))
        end
        λ = M.scale[j]
        for t in 1:nt
            offset = nv * (j - 1) + stride * (t - 1)
            mul!(a, M.chains[j, t], view(u, (offset + 1):(offset + nv)))
            if t > 1
                @inbounds for i in 1:nv
                    a[i] -= M.β[t] * λ * product[i, t - 1]
                end
            end
            if t < nt
                @inbounds for i in 1:nv
                    a[i] -= M.β[t + 1] * λ * product[i, t + 1]
                    carried[i] = λ * product[i, t]
                end
                _brm_gmrf_tracked_solve!(solved, M, j, t + 1, carried, work)
                mul!(carried, M.spatial, solved)
                coefficient = (M.β[t + 1])^2 * λ
                @inbounds for i in 1:nv
                    a[i] += coefficient * carried[i]
                end
            end
            @inbounds for i in 1:nv
                out[offset + i] = a[i]
            end
        end
    end
    nc == 1 || _brm_gmrf_channel_basis!(out, R.basis, nv, nc, nt, false)
    @inbounds for k in eachindex(state)
        y[state[k]] = out[k]
    end
    y
end
(M::_BRMGMRFTracked)(x) = mul!(similar(x, Float64), M, x)

# Accumulate `weight` × the derivative of uᵀ M u in M's arguments, u in the
# component's coordinates.
function _brm_gmrf_tracked_contract!(M::_BRMGMRFTracked, x, weight)
    R = M.reference
    nv, nc, nt = R.shape
    state = R.state
    u = Vector{Float64}(undef, length(state))
    @inbounds for k in eachindex(state)
        u[k] = x[state[k]]
    end
    nc == 1 || _brm_gmrf_channel_basis!(u, R.basis, nv, nc, nt, true)
    stride = nv * nc
    sp = M.spatial
    rows = rowvals(sp)
    current, previous, w = zeros(nv), zeros(nv), zeros(nv)
    Qu, dQu, Qw, dQw, previous_Q, previous_dQ, work, right = (zeros(nv) for _ in 1:8)
    for j in 1:nc
        λ = M.scale[j]
        for t in 1:nt
            offset = nv * (j - 1) + stride * (t - 1)
            copyto!(current, 1, u, offset + 1, nv)
            mul!(Qu, M.spatial, current)
            mul!(dQu, M.derivative, current)
            s1 = dot(current, Qu)
            ds1 = dot(current, dQu)
            β = M.β[t]
            block = view(M.block, :, j, t)
            if t > 1
                # w_t = S_t⁻¹ P u_{t−1}, with P u_{t−1} = λ previous_Q.
                @inbounds for i in 1:nv
                    right[i] = λ * previous_Q[i]
                end
                _brm_gmrf_tracked_solve!(w, M, j, t, right, work)
                mul!(Qw, M.spatial, w)
                mul!(dQw, M.derivative, w)
                s1 -= β^2 * dot(w, Qw)
                ds1 -= β^2 * dot(w, dQw)
                p1 = dot(current, previous_Q)
                dp1 = dot(current, previous_dQ)
                p2 = dot(previous, Qw)
                dp2 = dot(previous, dQw)
                pair = -2β * p1 + 2β^2 * p2
                dpair = -2β * dp1 + 2β^2 * dp2
                M.dscale[j] += weight * pair
                M.dspatial += weight * λ * dpair
                M.dtemporal += weight * λ * M.dβ[t] * (-2p1 + 2β * p2)
                coefficient = weight * β^2
                @inbounds for col in 1:nv
                    wc = coefficient * w[col]
                    for k in nzrange(sp, col)
                        block[k] -= wc * w[rows[k]]
                    end
                end
            end
            M.dscale[j] += weight * s1 / M.v[t]
            M.dspatial += weight * λ / M.v[t] * ds1
            M.dtemporal += weight * λ * M.dinverse_v[t] * s1
            @inbounds for col in 1:nv
                uc = weight * current[col]
                for k in nzrange(sp, col)
                    block[k] += uc * current[rows[k]]
                end
            end
            copyto!(previous, current)
            copyto!(previous_Q, Qu)
            copyto!(previous_dQ, dQu)
        end
    end
    M
end

# The exact part: derivative of log det M(θ) = Σ log det S_tj in M's arguments,
# from each S_tj's selected inverse on the spatial pattern.
function _brm_gmrf_tracked_exact!(M::_BRMGMRFTracked)
    nv, nc, nt = M.reference.shape
    ns = length(nonzeros(M.spatial))
    z = zeros(ns)
    # One symbolic analysis for every S: they share a pattern.
    F = cholesky(Symmetric(first(M.chains)))
    destination = copy(M.spatial)
    plan = nothing
    for t in 1:nt, j in 1:nc
        t == 1 && j == 1 || cholesky!(F, Symmetric(M.chains[j, t]))
        selected = SelectedInversion.selinv(F; depermute=true).Z
        if selected isa SelectedInversion.SupernodalMatrix
            plan === nothing && (plan = SelectedInversion.selinv_extract_setup(selected, M.spatial))
            SelectedInversion.selinv_extract!(destination, selected, plan)
            copyto!(z, nonzeros(destination))
        else
            copyto!(z, nonzeros(SelectedInversion.selinv_extract(selected, M.spatial)))
        end
        λ = M.scale[j]
        with_q = dot(z, nonzeros(M.spatial))
        with_dq = dot(z, nonzeros(M.derivative))
        M.dscale[j] += with_q / M.v[t]
        M.dspatial += λ / M.v[t] * with_dq
        M.dtemporal += λ * M.dinverse_v[t] * with_q
        view(M.block, :, j, t) .+= z
    end
    M
end

# D_tc's share of the accumulated derivative, Σ_j V̄_cj² block_tj: what the
# likelihood's chain rule contracts beyond the precision's own sensitivity.
function _brm_gmrf_tracked_data(M::_BRMGMRFTracked)
    V = M.reference.basis
    _, nc, nt = M.reference.shape
    ns = size(M.block, 1)
    data = zeros(ns, nc, nt)
    @inbounds for t in 1:nt, c in 1:nc, j in 1:nc
        weight = V[c, j]^2
        iszero(weight) && continue
        for k in 1:ns
            data[k, c, t] += weight * M.block[k, j, t]
        end
    end
    data
end

# Add that share to the sensitivity of h, entry by entry of its time and channel.
function _brm_gmrf_tracked_sensitivity!(S, M::_BRMGMRFTracked, field)
    data = _brm_gmrf_tracked_data(M)
    _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
        ta == tb && ca == cb && (S[field.nz[k]] += data[ks, ca, ta])
    end
    data
end

# T⁻¹ b for a rebuilt field reference, in the component's coordinates.
function _brm_gmrf_tracked_ldiv!(out, M::_BRMGMRFTracked, b)
    R = M.reference
    nv, nc, nt = R.shape
    state = R.state
    y = Vector{Float64}(undef, length(state))
    @inbounds for k in eachindex(state)
        y[k] = b[state[k]]
    end
    nc == 1 || _brm_gmrf_channel_basis!(y, R.basis, nv, nc, nt, true)
    stride = nv * nc
    a, c, solved, work, previous = zeros(nv), zeros(nv), zeros(nv), zeros(nv), zeros(nv)
    for j in 1:nc
        λ = M.scale[j]
        for t in nt:-1:2
            offset = nv * (j - 1) + stride * (t - 1)
            copyto!(a, 1, y, offset + 1, nv)
            _brm_gmrf_tracked_solve!(solved, M, j, t, a, work)
            mul!(c, M.spatial, solved)
            coefficient = M.β[t] * λ
            @inbounds for i in 1:nv
                y[offset - stride + i] += coefficient * c[i]
            end
        end
        for t in 1:nt
            offset = nv * (j - 1) + stride * (t - 1)
            copyto!(a, 1, y, offset + 1, nv)
            if t > 1
                mul!(c, M.spatial, previous)
                coefficient = M.β[t] * λ
                @inbounds for i in 1:nv
                    a[i] += coefficient * c[i]
                end
            end
            _brm_gmrf_tracked_solve!(previous, M, j, t, a, work)
            copyto!(y, offset + 1, previous, 1, nv)
        end
    end
    nc == 1 || _brm_gmrf_channel_basis!(y, R.basis, nv, nc, nt, false)
    @inbounds for k in eachindex(state)
        out[state[k]] = y[k]
    end
    out
end

# The same control variate over a star, tracking only the fields' priors: the
# data stay at their values in the fixed reference, the hub's in its chains and
# those between fields in the coupling. Tracking the data too raises the noise of a
# field's scale (measured: gradient SD 40 against 9 to 12 for the plain estimate,
# at 20,000 states), because M stays far from H where the data pin only a sum of
# fields, so the two estimates' data terms do not cancel. The priors' lengths gain
# (SD 2.0, 1.1, 2.7 against 6.7, 6.4, 10.9). With s = Σ_{g≠h} D_hg u_g, w = T_h⁻¹ s,
#   uᵀ M u = Σ_{g≠h} u_gᵀ P_g u_g + u_hᵀ T_h u_h + 2 u_hᵀ s + sᵀ T_h⁻¹ s,
# whose derivative in the priors is each chain's at u, less the hub's at w.
mutable struct _BRMGMRFStarTracked
    reference::_BRMGMRFStarReference
    members::Vector{_BRMGMRFTracked}
    fields::Vector{Any}
    logdet::Float64
    shift::Float64
end
Base.size(M::_BRMGMRFStarTracked) = size(M.reference.coupling)
Base.size(M::_BRMGMRFStarTracked, d::Integer) = size(M.reference.coupling, d)
Base.eltype(::_BRMGMRFStarTracked) = Float64
Base.eltype(::Type{_BRMGMRFStarTracked}) = Float64

_brm_gmrf_hub(M::_BRMGMRFStarTracked) = M.members[M.reference.hub]

function _brm_gmrf_tracked(fields, R::_BRMGMRFStarReference, h)
    any(m -> m.sequential !== nothing, R.members) && return nothing
    members = map(enumerate(zip(fields, R.members))) do (k, (f, m))
        _brm_gmrf_tracked(f, m, h; data=k == R.hub ? R.hub_data : zeros(length(f.qs), f.shape[2], f.shape[3]))
    end
    _BRMGMRFStarTracked(R, members, collect(Any, fields), sum(m -> m.logdet, members),
        maximum(m -> m.shift, members))
end

function LinearAlgebra.mul!(y, M::_BRMGMRFStarTracked, x)
    C = M.reference.coupling
    fill!(y, 0.0)
    for member in M.members
        mul!(y, member, x)
    end
    hub = _brm_gmrf_hub(M)
    s = transpose(C) * x
    w = zeros(length(x))
    _brm_gmrf_tracked_ldiv!(w, hub, s)
    at_hub = zeros(length(x))
    @inbounds for k in hub.reference.state
        y[k] += s[k]
        at_hub[k] = x[k] + w[k]
    end
    mul!(y, C, at_hub, 1.0, 1.0)
    y
end
(M::_BRMGMRFStarTracked)(x) = mul!(similar(x, Float64), M, x)

function _brm_gmrf_tracked_contract!(M::_BRMGMRFStarTracked, u, weight)
    hub = _brm_gmrf_hub(M)
    for member in M.members
        _brm_gmrf_tracked_contract!(member, u, weight)
    end
    w = zeros(length(u))
    _brm_gmrf_tracked_ldiv!(w, hub, transpose(M.reference.coupling) * u)
    _brm_gmrf_tracked_contract!(hub, w, -weight)
    M
end

function _brm_gmrf_tracked_exact!(M::_BRMGMRFStarTracked)
    foreach(_brm_gmrf_tracked_exact!, M.members)
    M
end

# A rebuilt reference's derivative in the likelihood's entries of h: a star has none.
_brm_gmrf_tracked_sensitivity!(S, M::_BRMGMRFTracked, p, c) =
    _brm_gmrf_tracked_sensitivity!(S, M, _brm_gmrf_component_field(p, c))
_brm_gmrf_tracked_sensitivity!(S, M::_BRMGMRFStarTracked, p, c) = S

# A field's rebuilt reference, and whether the likelihood's data enter it.
_brm_gmrf_tracked_member(::Nothing, field) = (nothing, false)
_brm_gmrf_tracked_member(M::_BRMGMRFTracked, field) = (M, true)
function _brm_gmrf_tracked_member(M::_BRMGMRFStarTracked, field)
    k = findfirst(f -> f === field, M.fields)
    k === nothing ? (nothing, false) : (M.members[k], false)
end
