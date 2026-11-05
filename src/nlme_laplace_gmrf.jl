# Native GMRF fields (`gmrf(...)`, model scope or one per `@plate` cell)
# integrated by the sparse Laplace. Their draws keep their authored state
# coordinates (vertex, channel, time). Their prior precision
#   Q(θ) = Q_t(ℓ_t) ⊗ Λ ⊗ Q_s(ℓ_s),   Λ = (L Lᵀ)⁻¹,
# moves with the field's spatial and temporal lengths and its channel factor,
# which are population values read through RK value ports. RK's prior at
# z = 0 keeps the normalizer, as for declared Gaussian vectors
# (`nlme_laplace_gaussian.jl`); the engine adds −½ zᵀQz, writes Q on h's
# pattern, and differentiates
#   −½ ẑᵀ∂Qẑ − ½ tr(H⁻¹ ∂Q) − vᵀ∂Q ẑ
# through the separable factors, never forming a Kronecker matrix. The SD
# acts on the observation reads (C), as a random-effect scale does.

# The spatial precision Q_s = (κ²M + 2G + G M⁻¹ G / κ²) / 4π, κ² = 2/ℓ_s²
# (the mass-lumped FEM Matérn ν = 1 of `_brm_gmrf_spatial_mul!`), as values of
# its three parts on one fixed pattern.
struct _BRMGMRFSpatialParts
    pattern::SparseMatrixCSC{Float64,Int}
    mass::Vector{Float64}
    stiffness::Vector{Float64}
    product::Vector{Float64}
end

function _brm_gmrf_spatial_parts(mesh::_BRMGMRFMesh)
    G, n = mesh.stiffness, length(mesh.mass)
    product = G * Diagonal(1 ./ mesh.mass) * G
    union = product + G + sparse(1.0I, n, n)
    pattern = SparseMatrixCSC(n, n, copy(union.colptr), copy(rowvals(union)),
        zeros(length(nonzeros(union))))
    on(A) = [A[rowvals(pattern)[k], j] for j in 1:n for k in nzrange(pattern, j)]
    mass = [rowvals(pattern)[k] == j ? mesh.mass[j] : 0.0 for j in 1:n
        for k in nzrange(pattern, j)]
    _BRMGMRFSpatialParts(pattern, mass, on(G), on(product))
end

function _brm_gmrf_spatial_values!(out, d, parts::_BRMGMRFSpatialParts, spatial_length)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    kappa2 = 2 / spatial_length^2
    dkappa2 = -4 / spatial_length^3
    for k in eachindex(out)
        out[k] = (kappa2 * parts.mass[k] + 2parts.stiffness[k] +
            parts.product[k] / kappa2) / 4pi
        d === nothing || (d[k] = (parts.mass[k] - parts.product[k] / kappa2^2) / 4pi * dkappa2)
    end
    out
end

# The OU precision over the fitted times: tridiagonal `diagonal`, `off`
# (entry t, t+1), and their derivatives in the temporal length.
function _brm_gmrf_temporal_values!(diagonal, off, ddiagonal, doff, times, temporal_length)
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    fill!(diagonal, 0.0); fill!(off, 0.0); fill!(ddiagonal, 0.0); fill!(doff, 0.0)
    diagonal[1] = 1.0
    for t in 2:length(times)
        gap = times[t] - times[t - 1]
        a = exp(-gap / temporal_length)
        v = -expm1(-2gap / temporal_length)
        da = a * gap / temporal_length^2
        dv = -2a * da
        diagonal[t] += 1 / v
        ddiagonal[t] -= dv / v^2
        diagonal[t - 1] += a^2 / v
        ddiagonal[t - 1] += (2a * da * v - a^2 * dv) / v^2
        off[t - 1] = -a / v
        doff[t - 1] = -(da * v - a * dv) / v^2
    end
    nothing
end

# One integrated field: its draws in state order, its fitted support, the
# ports (or constants) of its lengths and channel factor, and, once h's
# pattern exists, the pattern position of every Q nonzero in a fixed order.
mutable struct _BRMSPLGMRFField
    block::Int
    z::Vector{Int}
    shape::NTuple{3,Int}
    spatial::_BRMGMRFSpatialParts
    mesh::_BRMGMRFMesh
    times::Vector{Float64}
    dense_channels::Bool
    spatial_port::Any
    temporal_port::Any
    factor_port::Any
    # A color per state level under which one row's vertex reads differ.
    colors::Vector{Int}
    nz::Vector{Int}
    # values at the current population point
    qs::Vector{Float64}
    dqs::Vector{Float64}
    qt::Vector{Float64}
    qo::Vector{Float64}
    dqt::Vector{Float64}
    dqo::Vector{Float64}
    L::Matrix{Float64}
    Λ::Matrix{Float64}
    θ::NTuple{2,Float64}
end

# The ports' reader and pullback are prepared RK queries: each copy of a
# problem (one per chain) prepares its own on first use, so no copy shares the
# other's AD buffers and no prepared query is deep-copied.
struct _BRMSPLGMRF
    fields::Vector{_BRMSPLGMRFField}
    ports::Tuple{Vararg{Symbol}}
    backend::Any
    read::Base.RefValue{Any}
    pullback::Base.RefValue{Any}
end
_BRMSPLGMRF(fields, ports, backend) =
    _BRMSPLGMRF(fields, ports, backend, Ref{Any}(nothing), Ref{Any}(nothing))

function Base.deepcopy_internal(g::_BRMSPLGMRF, stack::IdDict)
    haskey(stack, g) && return stack[g]
    backend = get(stack, g.backend, g.backend)
    stack[g] = _BRMSPLGMRF(Base.deepcopy_internal(g.fields, stack), g.ports, backend)
end

function _brm_gmrf_read(g::_BRMSPLGMRF, u)
    g.read[] === nothing && (g.read[] = _rk_value_reader(g.backend, g.ports))
    Tuple(g.read[](u))
end

function _brm_gmrf_pullback(g::_BRMSPLGMRF, u, seeds)
    g.pullback[] === nothing && (g.pullback[] = _rk_value_pullback(g.backend, g.ports, u))
    g.pullback[](u, seeds)
end

# Each field's Q nonzeros in a fixed order: time pair, channel pair, then the
# spatial pattern column by column. `f(i, j, ks, ta, tb, ca, cb)` gets the
# field's local positions, the spatial nonzero and the factor indices.
function _brm_gmrf_foreach_entry(f::F, field::_BRMSPLGMRFField) where {F}
    nv, nc, nt = field.shape
    sp = field.spatial.pattern
    rows = rowvals(sp)
    dense = field.dense_channels
    k = 0
    @inbounds for t in 1:nt, tb in max(1, t - 1):min(nt, t + 1), c in 1:nc,
            cb in (dense ? 1 : c):(dense ? nc : c), j in 1:nv, ks in nzrange(sp, j)
        k += 1
        f(k, rows[ks] + nv * ((c - 1) + nc * (t - 1)), j + nv * ((cb - 1) + nc * (tb - 1)),
            ks, t, tb, c, cb)
    end
    nothing
end

_brm_gmrf_temporal_entry(field, ta, tb) = ta == tb ? field.qt[ta] : field.qo[min(ta, tb)]
_brm_gmrf_dtemporal_entry(field, ta, tb) = ta == tb ? field.dqt[ta] : field.dqo[min(ta, tb)]

# The fields' Q pairs, as h's pattern needs them (positions in the integrated
# draws).
function _brm_gmrf_pattern_pairs(gmrf::_BRMSPLGMRF)
    pairs = Tuple{Int,Int}[]
    for field in gmrf.fields
        _brm_gmrf_foreach_entry(field) do _, i, j, _...
            i < j && push!(pairs, (field.z[i], field.z[j]))
        end
    end
    pairs
end

function _brm_gmrf_pattern_positions!(gmrf::_BRMSPLGMRF, pattern)
    for field in gmrf.fields
        field.nz = Int[]
        _brm_gmrf_foreach_entry(field) do _, i, j, _...
            push!(field.nz, _brm_spl_nzindex(pattern, field.z[i], field.z[j]))
        end
    end
    nothing
end

# A port is `(k, e)`: the k-th read value, or its e-th entry for a per-cell
# value (`e = 0` for a model-scope one). Anything else is a constant.
function _brm_gmrf_port_value(values, port::Tuple{Int,Int})
    value = values[first(port)]
    last(port) == 0 || value isa Number ? value : value[last(port)]
end
_brm_gmrf_port_value(values, constant) = constant

function _brm_gmrf_seed!(seeds, port::Tuple{Int,Int}, value)
    k, e = port
    if seeds[k] isa Number
        seeds[k] += value
    elseif e == 0
        seeds[k] .+= value
    else
        seeds[k][e] = seeds[k][e] .+ value
    end
    seeds
end
_brm_gmrf_seed!(seeds, constant, value) = seeds

# The structured priors of the problem's `:gmrf` blocks, from the model's
# field records: each field's fitted support (a graph value), and ports for
# its lengths and, when sampled, its channel factor.
function _brm_spl_gmrf_prior(p, records, u0)
    chosen = [r for r in records if r.kind in (:gmrf, :cell_gmrf)]
    isempty(chosen) && return nothing
    ports = Symbol[]
    port!(name, e) = begin
        k = findfirst(==(name), ports)
        k === nothing && (push!(ports, name); k = length(ports))
        (k, e)
    end
    support_names = Tuple(unique(r.kind === :gmrf ? r.locals.support : r.cells for r in chosen))
    supports = _rk_value_reader(p.backend, support_names)(u0)
    parts = IdDict{Any,_BRMGMRFSpatialParts}()
    fields = _BRMSPLGMRFField[]
    for r in chosen
        blocks = [j for (j, b) in enumerate(p.blocks)
            if b.prior === :gmrf && b.declaration === r.declaration]
        for (cell, j) in enumerate(blocks)
            support = r.kind === :gmrf ? supports[r.locals.support] :
                supports[r.cells][cell]
            layout = support.layout
            nc, nt = length(layout.channels), length(layout.times)
            shape = (layout.nvertices, nc, nt)
            z = vec(p.blocks[j].z)
            length(z) == prod(shape) || _brm_nlme_error("brm_laplace_problem: " *
                "internal: field `$(r.declaration)` has $(length(z)) draws for state shape $shape")
            e = r.kind === :gmrf ? 0 : cell
            correlation = r.correlation
            factor = correlation === nothing ? Matrix{Float64}(I, nc, nc) :
                correlation.kind === :parameter ? port!(correlation.name, 0) :
                Matrix{Float64}(correlation.factor)
            spatial = get!(() -> _brm_gmrf_spatial_parts(support.mesh), parts, support.mesh)
            ns = length(nonzeros(spatial.pattern))
            push!(fields, _BRMSPLGMRFField(j, z, shape, spatial, support.mesh,
                collect(Float64, layout.times), correlation !== nothing,
                port!(Symbol(r.declaration, ".", r.locals.spatial_length), e),
                port!(Symbol(r.declaration, ".", r.locals.temporal_length), e),
                factor, _brm_gmrf_level_colors(support.mesh, shape),
                Int[], zeros(ns), zeros(ns), zeros(nt), zeros(max(nt - 1, 0)),
                zeros(nt), zeros(max(nt - 1, 0)), zeros(nc, nc), zeros(nc, nc), (NaN, NaN)))
        end
    end
    _BRMSPLGMRF(fields, Tuple(ports), p.backend)
end

# A coloring of a field's state levels under which no row's three vertex reads
# share a color: a greedy coloring of the mesh's triangle vertices, repeated
# over channels and times (a row reads one time and channel). One probe per
# color then names every level each likelihood entry reads, in a fixed number
# of passes over the rows (`_brm_spl_design!`, `_brm_spl_cmaps`).
function _brm_gmrf_level_colors(mesh, shape)
    nv = shape[1]
    neighbours = [Int[] for _ in 1:nv]
    for c in axes(mesh.cells, 1), a in 1:3, b in 1:3
        a == b || push!(neighbours[mesh.cells[c, a]], mesh.cells[c, b])
    end
    color = zeros(Int, nv)
    for v in 1:nv
        used = Set(color[u] for u in neighbours[v])
        k = 1
        while k in used
            k += 1
        end
        color[v] = k
    end
    repeat(color, shape[2] * shape[3])
end

# The level colors of block `j`, when it is a native field.
function _brm_spl_gmrf_colors(p, j)
    p.gmrf === nothing && return nothing
    k = findfirst(field -> field.block == j, p.gmrf.fields)
    k === nothing ? nothing : p.gmrf.fields[k].colors
end

# A plan's restricted copy keeps the fields of its retained blocks.
function _brm_spl_gmrf_restrict(gmrf, blockmap, zmap)
    gmrf === nothing && return nothing
    fields = _BRMSPLGMRFField[]
    for field in gmrf.fields
        blockmap[field.block] == 0 && continue
        kept = deepcopy(field)
        kept.block = blockmap[field.block]
        kept.z = zmap[field.z]
        push!(fields, kept)
    end
    isempty(fields) ? nothing : _BRMSPLGMRF(fields, gmrf.ports, gmrf.backend)
end

# Refresh every field's separable factors at the population point `u`.
function _brm_spl_gmrf_parameters!(p, u)
    gmrf = p.gmrf
    gmrf === nothing && return nothing
    values = _brm_gmrf_read(gmrf, u)
    for field in gmrf.fields
        ls = Float64(_brm_gmrf_port_value(values, field.spatial_port))
        lt = Float64(_brm_gmrf_port_value(values, field.temporal_port))
        L = Matrix{Float64}(_brm_gmrf_port_value(values, field.factor_port))
        size(L) == (field.shape[2], field.shape[2]) || throw(DimensionMismatch(
            "gmrf channel factor is $(size(L)) for $(field.shape[2]) channels"))
        field.θ = (ls, lt)
        _brm_gmrf_spatial_values!(field.qs, field.dqs, field.spatial, ls)
        _brm_gmrf_temporal_values!(field.qt, field.qo, field.dqt, field.dqo, field.times, lt)
        field.L = L
        field.Λ = inv(Symmetric(L * L'))
    end
    nothing
end

# −½ zᵀQz per field and its gradient −Qz, through the separable product.
function _brm_spl_gmrf_prior!(fc, g, p, z, fb, correction)
    gmrf = p.gmrf
    gmrf === nothing && return nothing
    for field in gmrf.fields
        x = reshape(z[field.z], field.shape)
        Qx = similar(x)
        _brm_gmrf_precision_mul!(Qx, x, field.mesh, field.times, field.θ..., field.L)
        value = -dot(x, Qx) / 2
        component = p.component[first(field.z)]
        _brm_spl_add_value!(fc, correction, component, value)
        fb === nothing || (fb[p.sblock[first(field.z)]] += value)
        if g !== nothing
            for (k, i) in enumerate(field.z)
                g[i] -= Qx[k]
            end
        end
    end
    nothing
end

function _brm_spl_gmrf_precision!(nz, p)
    gmrf = p.gmrf
    gmrf === nothing && return nothing
    for field in gmrf.fields
        _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
            nz[field.nz[k]] += _brm_gmrf_temporal_entry(field, ta, tb) *
                field.Λ[ca, cb] * field.qs[ks]
        end
    end
    nothing
end

# The fields' prior precision Q over the `n` integrated draws at the current
# population point, both triangles, or `nothing` without fields. For local
# (per-block) prior values, as the plan's importance diagnostic needs.
function _brm_spl_gmrf_sparse(p, n)
    gmrf = p.gmrf
    gmrf === nothing && return nothing
    rows, cols, vals = Int[], Int[], Float64[]
    for field in gmrf.fields
        _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
            push!(rows, field.z[i]); push!(cols, field.z[j])
            push!(vals, _brm_gmrf_temporal_entry(field, ta, tb) * field.Λ[ca, cb] *
                field.qs[ks])
        end
    end
    sparse(rows, cols, vals, n, n)
end

function _brm_spl_gmrf_quad(p, z)
    gmrf = p.gmrf
    gmrf === nothing && return 0.0
    q = 0.0
    for field in gmrf.fields
        x = reshape(z[field.z], field.shape)
        Qx = similar(x)
        _brm_gmrf_precision_mul!(Qx, x, field.mesh, field.times, field.θ..., field.L)
        q += dot(x, Qx)
    end
    q
end

# The lengths' and channel factor's gradient: W = −½(S + ẑẑᵀ + vẑᵀ + ẑvᵀ) on
# Q's pattern, contracted with ∂Q through its three factors.
function _brm_spl_gmrf_gradient!(gradient, p, state, S, v, active)
    gmrf = p.gmrf
    gmrf === nothing && return gradient
    components = active === nothing ? nothing : Set(p.term_component[t]
        for t in eachindex(active) if active[t])
    values = _brm_gmrf_read(gmrf, state.uA)
    seeds = Any[x isa Number ? 0.0 : zeros(size(x)) for x in values]
    for field in gmrf.fields
        components === nothing || p.component[first(field.z)] in components || continue
        nc, nt = field.shape[2], field.shape[3]
        gs = zeros(length(field.qs))
        gΛ = zeros(nc, nc)
        gt = zeros(nt)
        go = zeros(max(nt - 1, 0))
        z = state.z
        # A rebuilt reference's derivative in the data is in S for the likelihood's
        # chain rule. The prior takes its own, in the same factors, below.
        tracked, with_data = state.F isa _BRMOperatorFactor ?
            _brm_gmrf_tracked_member(state.F.tracked[p.component[first(field.z)]], field) :
            (nothing, false)
        share = with_data ? _brm_gmrf_tracked_data(tracked) : nothing
        _brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
            a, b = field.z[i], field.z[j]
            selected = S[field.nz[k]]
            share === nothing || (ta == tb && ca == cb && (selected -= share[ks, ca, ta]))
            w = -(selected + z[a] * z[b] + v[a] * z[b] + z[a] * v[b]) / 2
            qt = _brm_gmrf_temporal_entry(field, ta, tb)
            λ = field.Λ[ca, cb]
            gs[ks] += w * qt * λ
            gΛ[ca, cb] += w * qt * field.qs[ks]
            if ta == tb
                gt[ta] += w * λ * field.qs[ks]
            else
                go[min(ta, tb)] += w * λ * field.qs[ks]
            end
        end
        dls = dot(gs, field.dqs)
        dlt = dot(gt, field.dqt) + dot(go, field.dqo)
        if tracked !== nothing
            dls -= tracked.dspatial / 2
            dlt -= tracked.dtemporal / 2
            basis = tracked.reference.basis
            for jj in axes(basis, 2)
                gΛ .-= (tracked.dscale[jj] / 2) .* (view(basis, :, jj) * view(basis, :, jj)')
            end
        end
        Λ = field.Λ
        dL = -2 .* (Λ * Symmetric((gΛ .+ gΛ') ./ 2) * Λ * field.L)
        _brm_gmrf_seed!(seeds, field.spatial_port, dls)
        _brm_gmrf_seed!(seeds, field.temporal_port, dlt)
        _brm_gmrf_seed!(seeds, field.factor_port, dL)
    end
    gradient .+= _brm_gmrf_pullback(gmrf, state.uA, Tuple(seeds))[p.population]
    gradient
end
