# The unit border separation of a fitted `car` graph: the same field, sampled
# so that the areas of different units are conditionally independent given a
# few shared values (todo `0v882rf`).
#
# Per connected component, the field's density is the Gaussian whose precision
# is `P = Q + λ 11'`: the graph Laplacian `Q` plus the soft sum-to-zero
# `sum(phi_c) ~ Normal(0, 0.001 n_c)`, `λ = 1 / (0.001 n_c)^2`. A separator `S`
# holds an end of every edge between two units, so `Q` couples no interior
# areas of different units. The sum-to-zero still couples them, through the
# total `1'phi`; it is a function of `phi_S` and of each unit's interior sum
# `T_u`, so given `x = (phi_S, T)` the units' interiors are independent, each a
# Gaussian on the plane `sum(phi_I_u) = T_u`. The field is then the linear
# image of standard normals:
#
#     x = F w,                          w ~ N(0, I)  (shared)
#     phi_I_u = mu_u(x[cols_u]) + G_u z_u,   z_u ~ N(0, I)  (unit u's draws)
#
# with `F F'` the exact marginal covariance of `x`, so the density over `phi`
# is the current one up to a constant Jacobian. Neither `F` nor any `G_u` is
# ever a dense matrix (below).
#
# Integrating unit u's interior (`K = Q[I, I]`, `B = Q[I, S]`, `c = 1'K⁻¹1`,
# `h = B'K⁻¹1`) out over its plane leaves `½ phi_S'B'K⁻¹B phi_S - (T_u + h'phi_S)² / 2c`,
# so `x`'s precision is
#
#     P_SS = Q_SS + λ11' - Σ_u B'K⁻¹B + Σ_u h h' / c
#     P_ST_u = λ1 + h / c,   P_T_uT_v = λ + δ_uv / c,
#
# and the interior's conditional mean is `K⁻¹(ν1 - B phi_S)`, `ν = (T_u + h'phi_S) / c`.
#
# Without the sum-to-zero, `x`'s precision `P̃ = P - λ11'` is sparse: the
# separator's edges plus one block per unit over the separator areas it
# borders and its sum. The constant field is its null direction,
# `v0 = (1_S, |I_u|)`, and `1'v0 = n_c`. So `x` is sampled as a whole
# component is (below): grounded at its first entry, `x_g[2:m] = P'L⁻ᵀ w[1:m-1]`
# with `L` the sparse factor of `P̃[2:m, 2:m]` and `x_g[1] = 0`, its total
# `s = 0.001 n_c w_m` its own draw, and `x = x_g + v0 (s - 1'x_g) / n_c`; then
# `x'P̃x = x_g'P̃x_g` and `λ(1'x)² = w_m²`, so `F` is that linear map.
#
# A unit's interior map is never a dense matrix (snag `car-units-border-aa5e8712`):
# `K` is a sparse graph Laplacian block, so it is applied through its sparse
# Cholesky factor `K[p, p] = L L'` (fill-reducing order `p`), linear in the
# factor's entries. With `y = K⁻¹ B phi_S` (so `1'y = h'phi_S`), `k = K⁻¹1` and
# the plane's covariance `K⁻¹ - k k'/c = (I - k1'/c) K⁻¹ (I - 1k'/c)`,
#
#     phi_I = v + k (T_u - 1'v) / c,   v[p] = L⁻ᵀ (H [z; 0] - L⁻¹ (B phi_S)[p]),
#
# where the Householder reflection `H` maps the last axis onto `u0 = L⁻¹1`,
# the one direction `(I - k1'/c) P'L⁻ᵀ` loses: so the `n - 1` draws `z` cover
# the plane exactly once, and `G'KG = I` on it. A component inside one unit has
# no separator. Its sum `s = 0.001 n z_n` is its own draw and its values on the
# plane `sum = 0` are a grounded field: `psi = P'L⁻ᵀ z[1:n-1]` on all areas but
# the last (`L` of the grounded Laplacian `Q[1:n-1, 1:n-1]`), `psi_n = 0`, and
# `phi = psi + (s - 1'psi) / n`. Then `phi'Qphi = psi'Qpsi = z'z` again.

# The separator: every area of no unit (no rows, or rows in several units) and,
# on each border between two units, the border areas of the side with fewer of
# them (ties: the earlier unit). `unit[p]` is the unit of structured area `p`.
function _brm_car_separator(graph, unit::AbstractVector{Int})
    shared = unit .== 0
    sides = Dict{Tuple{Int,Int},NTuple{2,Set{Int}}}()
    for (a, b) in zip(graph.node1, graph.node2)
        u, v = unit[a], unit[b]
        (u == 0 || v == 0 || u == v) && continue
        u > v && ((a, b, u, v) = (b, a, v, u))
        border = get!(() -> (Set{Int}(), Set{Int}()), sides, (u, v))
        push!(border[1], a)
        push!(border[2], b)
    end
    for pair in sort!(collect(keys(sides)))
        su, sv = sides[pair]
        for p in (length(su) <= length(sv) ? su : sv)
            shared[p] = true
        end
    end
    shared
end

# The sparse Laplacian block `Q[rows, cols]` of the structured areas' edges.
function _brm_car_laplacian(graph, rows::AbstractVector{Int}, cols::AbstractVector{Int})
    I, J, V = _brm_car_laplacian_entries!(Int[], Int[], Float64[], graph, rows, cols)
    sparse(I, J, V, length(rows), length(cols))
end

# Appends the entries of `Q[rows, cols]` (repeated positions add up).
function _brm_car_laplacian_entries!(I, J, V, graph, rows::AbstractVector{Int},
        cols::AbstractVector{Int})
    r = Dict(p => i for (i, p) in enumerate(rows))
    c = Dict(p => j for (j, p) in enumerate(cols))
    for (a, b) in zip(graph.node1, graph.node2)
        for (p, q) in ((a, b), (b, a))
            i = get(r, p, 0)
            i == 0 && continue
            j = get(c, p, 0)
            j == 0 || (push!(I, i); push!(J, j); push!(V, 1.0))
            j = get(c, q, 0)
            j == 0 || (push!(I, i); push!(J, j); push!(V, -1.0))
        end
    end
    I, J, V
end

# A sparse lower-triangular factor in compressed columns, each column's
# diagonal entry first, and its fill-reducing order: `K[perm, perm] = L L'`.
struct _BRMCarFactor
    perm::Vector{Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{Float64}
end

function _BRMCarFactor(K::SparseMatrixCSC)
    factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(K))
    L = sparse(factor.L)
    for j in axes(L, 2)
        L.rowval[L.colptr[j]] == j || error(
            "_BRMCarFactor: internal: column $j of the Cholesky factor does not " *
            "start at its diagonal")
    end
    _BRMCarFactor(Vector{Int}(factor.p), L.colptr, L.rowval, L.nzval)
end

# `y = L⁻¹ y` and `y = L⁻ᵀ y`, in place on a fresh buffer.
function _brm_car_lsolve!(y::AbstractVector, f::_BRMCarFactor)
    colptr, rowval, nzval = f.colptr, f.rowval, f.nzval
    @inbounds for j in eachindex(y)
        k0 = colptr[j]
        yj = y[j] / nzval[k0]
        y[j] = yj
        for k in (k0 + 1):(colptr[j + 1] - 1)
            y[rowval[k]] -= nzval[k] * yj
        end
    end
    y
end

function _brm_car_ltsolve!(y::AbstractVector, f::_BRMCarFactor)
    colptr, rowval, nzval = f.colptr, f.rowval, f.nzval
    @inbounds for j in reverse(eachindex(y))
        k0 = colptr[j]
        s = y[j]
        for k in (k0 + 1):(colptr[j + 1] - 1)
            s -= nzval[k] * y[rowval[k]]
        end
        y[j] = s / nzval[k0]
    end
    y
end

# `y = L' y` and `y = L y`, in place on a fresh buffer (the inverse maps).
function _brm_car_ltmul!(y::AbstractVector, f::_BRMCarFactor)
    colptr, rowval, nzval = f.colptr, f.rowval, f.nzval
    @inbounds for j in eachindex(y)
        s = nzval[colptr[j]] * y[j]
        for k in (colptr[j] + 1):(colptr[j + 1] - 1)
            s += nzval[k] * y[rowval[k]]
        end
        y[j] = s
    end
    y
end

# One piece of a unit's interior: its areas in one component. `:interior`
# borders the separator and reads the `x` entries of its border (`edge_col`,
# one per edge to the area at `edge_pos` of the factor's order) and its sum
# (`sum_col`), positions in the unit's `cols`; `:component` is a whole
# component, with no `x` entries.
# `first_area` / `first_draw` place it in the unit's areas and draws.
struct _BRMCarPiece
    kind::Symbol
    n::Int
    first_area::Int
    first_draw::Int
    factor::_BRMCarFactor
    k1::Vector{Float64}
    c::Float64
    reflector::Vector{Float64}
    reflector_norm2::Float64
    edge_pos::Vector{Int}
    edge_col::Vector{Int}
    sum_col::Int
end

_brm_car_piece_draws(piece::_BRMCarPiece) =
    piece.kind === :interior ? piece.n - 1 : piece.n

"""
    _BRMCarUnitFactor

One unit's interior map of a separated `car` field: its `nareas` interior
areas from its `ndraws` standard-normal draws and the `x` entries it reads,
piece by piece (`_brm_car_unit_field`). Every piece is a sparse Cholesky
factor, so memory and each evaluation are linear in the factors' entries.
"""
struct _BRMCarUnitFactor
    nareas::Int
    ndraws::Int
    ncols::Int
    pieces::Vector{_BRMCarPiece}
end

# An interior piece: the factor of `K = Q[I, I]`, its border `edges` (area,
# unit column) and the unit column of its sum.
function _brm_car_interior_piece(factor::_BRMCarFactor, edge_area, edge_col, sum_col,
        first_area, first_draw)
    n = length(factor.perm)
    u0 = _brm_car_lsolve!(ones(n), factor)
    k1 = zeros(n)
    k1[factor.perm] = _brm_car_ltsolve!(copy(u0), factor)
    # `H e_n = -σ u0/|u0|` with `w = e_n + σ u0/|u0|`, `w'w ≥ 2`.
    u = u0 ./ LinearAlgebra.norm(u0)
    w = (u[n] >= 0 ? 1.0 : -1.0) .* u
    w[n] += 1.0
    _BRMCarPiece(:interior, n, first_area, first_draw, factor, k1, sum(k1), w,
        LinearAlgebra.dot(w, w), invperm(factor.perm)[edge_area], Vector{Int}(edge_col),
        sum_col)
end

# A component inside one unit, grounded at its last area.
function _brm_car_component_piece(Q::SparseMatrixCSC, first_area, first_draw)
    n = size(Q, 1)
    factor = n == 1 ? _BRMCarFactor(Int[], [1], Int[], Float64[]) :
        _BRMCarFactor(Q[1:(n - 1), 1:(n - 1)])
    _BRMCarPiece(:component, n, first_area, first_draw, factor, Float64[], 0.0,
        Float64[], 0.0, Int[], Int[], 0)
end

"""
    _brm_car_unit_field(z, xs, f::_BRMCarUnitFactor)

A unit's interior areas of a separated `car` field from its draws `z` and the
`x` entries `xs = x[cols]` it reads. Linear in the unit's factor entries;
reads its inputs and writes only fresh buffers, so Enzyme differentiates it.
"""
function _brm_car_unit_field(z::AbstractVector, xs::AbstractVector, f::_BRMCarUnitFactor)
    length(z) == f.ndraws || throw(DimensionMismatch(
        "_brm_car_unit_field: $(length(z)) draws for a unit of $(f.ndraws)"))
    length(xs) == f.ncols || throw(DimensionMismatch(
        "_brm_car_unit_field: $(length(xs)) shared values for a unit reading $(f.ncols)"))
    T = promote_type(eltype(z), eltype(xs), Float64)
    phi = Vector{T}(undef, f.nareas)
    for piece in f.pieces
        if piece.kind === :interior
            _brm_car_interior_field!(phi, piece, z, xs)
        else
            _brm_car_component_field!(phi, piece, z)
        end
    end
    phi
end

const _BRM_CAR_NO_VALUES = Float64[]
_brm_car_unit_field(z::AbstractVector, f::_BRMCarUnitFactor) =
    _brm_car_unit_field(z, _BRM_CAR_NO_VALUES, f)
_brm_car_unit_field_shared(xs::AbstractVector, f::_BRMCarUnitFactor) =
    _brm_car_unit_field(_BRM_CAR_NO_VALUES, xs, f)

function _brm_car_interior_field!(phi::AbstractVector{T}, piece, z, xs) where {T}
    n, a0, d0 = piece.n, piece.first_area, piece.first_draw
    perm, w = piece.factor.perm, piece.reflector
    # One buffer: `L⁻¹ (B xs)[p]`, then `H [z; 0]` less it, then `v[p]`.
    t = zeros(T, n)
    @inbounds for k in eachindex(piece.edge_pos)
        t[piece.edge_pos[k]] -= xs[piece.edge_col[k]]
    end
    _brm_car_lsolve!(t, piece.factor)
    a = zero(T)
    @inbounds for i in 1:(n - 1)
        a += w[i] * z[d0 + i]
    end
    a *= 2 / piece.reflector_norm2
    @inbounds for i in 1:n
        t[i] = (i < n ? z[d0 + i] : zero(T)) - w[i] * a - t[i]
    end
    _brm_car_ltsolve!(t, piece.factor)
    shift = (xs[piece.sum_col] - sum(t)) / piece.c
    @inbounds for i in 1:n
        phi[a0 + perm[i]] = t[i] + piece.k1[perm[i]] * shift
    end
    phi
end

function _brm_car_component_field!(phi::AbstractVector{T}, piece, z) where {T}
    n, a0, d0 = piece.n, piece.first_area, piece.first_draw
    perm = piece.factor.perm
    e = Vector{T}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        e[i] = z[d0 + i]
    end
    _brm_car_ltsolve!(e, piece.factor)
    # `psi[p] = e` on all areas but the grounded last one.
    shift = (0.001 * n * z[d0 + n] - sum(e)) / n
    @inbounds for i in 1:(n - 1)
        phi[a0 + perm[i]] = e[i] + shift
    end
    phi[a0 + n] = shift
    phi
end

"""
    _brm_car_unit_draws(phi, xs, f::_BRMCarUnitFactor)

The inverse of [`_brm_car_unit_field`](@ref): the draws `z` that give the
unit's interior areas `phi` from its shared values `xs`. `phi` must lie on
each piece's plane (`sum = T_u`), as every field does.
"""
function _brm_car_unit_draws(phi::AbstractVector, xs::AbstractVector, f::_BRMCarUnitFactor)
    T = promote_type(eltype(phi), eltype(xs), Float64)
    z = Vector{T}(undef, f.ndraws)
    zero_draws = zeros(T, f.ndraws)
    mean = _brm_car_unit_field(zero_draws, xs, f)
    for piece in f.pieces
        n, a0, d0 = piece.n, piece.first_area, piece.first_draw
        perm = piece.factor.perm
        if piece.kind === :interior
            # `z = (H L' g[p])[1:n-1]`, `g = phi - mean`.
            g = Vector{T}(undef, n)
            for i in 1:n
                g[i] = phi[a0 + perm[i]] - mean[a0 + perm[i]]
            end
            _brm_car_ltmul!(g, piece.factor)
            w = piece.reflector
            a = 2 * LinearAlgebra.dot(w, g) / piece.reflector_norm2
            for i in 1:(n - 1)
                z[d0 + i] = g[i] - w[i] * a
            end
        else
            # `psi = phi - phi_n`, `z[1:n-1] = L' psi[p]`, `s = sum(phi)`.
            g = Vector{T}(undef, n - 1)
            for i in 1:(n - 1)
                g[i] = phi[a0 + perm[i]] - phi[a0 + n]
            end
            _brm_car_ltmul!(g, piece.factor)
            for i in 1:(n - 1)
                z[d0 + i] = g[i]
            end
            z[d0 + n] = sum(phi[a0 + i] for i in 1:n) / (0.001 * n)
        end
    end
    z
end

"""
    _brm_car_unit_pullback(dphi, f::_BRMCarUnitFactor)

The transpose of [`_brm_car_unit_field`](@ref)'s linear map: the cotangents
`(dz, dxs)` of the draws and shared values from those of the unit's areas.
"""
function _brm_car_unit_pullback(dphi::AbstractVector, f::_BRMCarUnitFactor)
    T = promote_type(eltype(dphi), Float64)
    dz = zeros(T, f.ndraws)
    dxs = zeros(T, f.ncols)
    for piece in f.pieces
        n, a0, d0 = piece.n, piece.first_area, piece.first_draw
        perm = piece.factor.perm
        g = dphi[(a0 + 1):(a0 + n)]
        if piece.kind === :interior
            # phi = v + k (T - 1'v)/c: dv = g - 1 (k'g)/c, dT = k'g/c
            kg = LinearAlgebra.dot(piece.k1, g) / piece.c
            dxs[piece.sum_col] += kg
            e = Vector{T}(undef, n)
            for i in 1:n
                e[i] = g[perm[i]] - kg
            end
            _brm_car_lsolve!(e, piece.factor)
            # e = [z; 0] - w a - t, a = 2 w[1:n-1]'z / w'w
            w = piece.reflector
            we = 2 * LinearAlgebra.dot(w, e) / piece.reflector_norm2
            for i in 1:(n - 1)
                dz[d0 + i] += e[i] - w[i] * we
            end
            # t = L⁻¹ r, r[edge_pos] -= xs[edge_col]: dr = L⁻ᵀ(-e)
            r = _brm_car_ltsolve!(-e, piece.factor)
            for k in eachindex(piece.edge_pos)
                dxs[piece.edge_col[k]] -= r[piece.edge_pos[k]]
            end
        else
            # phi = psi + (s - 1'psi)/n: dpsi = g - mean(g), ds = mean(g)
            mg = sum(g) / n
            e = Vector{T}(undef, n - 1)
            for i in 1:(n - 1)
                e[i] = g[perm[i]] - mg
            end
            _brm_car_lsolve!(e, piece.factor)
            for i in 1:(n - 1)
                dz[d0 + i] += e[i]
            end
            dz[d0 + n] += 0.001 * n * mg
        end
    end
    dz, dxs
end

# One component's shared values `x` (its separator areas, then its owners'
# interior sums) from its `m` shared draws: grounded at the first entry, the
# sparse factor of `P̃[2:m, 2:m]`, the null direction `v0` and the component's
# area count `n` (header).
struct _BRMCarSharedBlock
    offset::Int
    m::Int
    n::Int
    v0::Vector{Float64}
    factor::_BRMCarFactor
end

"""
    _BRMCarSharedFactor

The map `x = F w` of a separated `car` field's shared values, one sparse block
per component with a separator (`_brm_car_shared_values`). Memory and each
evaluation are linear in the blocks' factor entries.
"""
struct _BRMCarSharedFactor
    m::Int
    blocks::Vector{_BRMCarSharedBlock}
end

"""
    _brm_car_shared_values(w, f::_BRMCarSharedFactor)

A separated field's shared values `x = F w` from its shared draws `w`.
"""
function _brm_car_shared_values(w::AbstractVector, f::_BRMCarSharedFactor)
    length(w) == f.m || throw(DimensionMismatch(
        "_brm_car_shared_values: $(length(w)) draws for $(f.m) shared values"))
    T = promote_type(eltype(w), Float64)
    x = Vector{T}(undef, f.m)
    for b in f.blocks
        k, perm = b.m - 1, b.factor.perm
        e = Vector{T}(undef, k)
        @inbounds for i in 1:k
            e[i] = w[b.offset + i]
        end
        _brm_car_ltsolve!(e, b.factor)
        # `x_g[1 + p] = e`, `x_g[1] = 0`.
        shift = (0.001 * b.n * w[b.offset + b.m] - sum(e)) / b.n
        x[b.offset + 1] = b.v0[1] * shift
        @inbounds for i in 1:k
            j = 1 + perm[i]
            x[b.offset + j] = e[i] + b.v0[j] * shift
        end
    end
    x
end

# Its inverse: the shared draws of shared values `x`.
function _brm_car_shared_draws(x::AbstractVector, f::_BRMCarSharedFactor)
    T = promote_type(eltype(x), Float64)
    w = Vector{T}(undef, f.m)
    for b in f.blocks
        k, perm = b.m - 1, b.factor.perm
        # `x = x_g + v0 α` with `x_g[1] = 0` and `v0[1] = 1`.
        alpha = x[b.offset + 1]
        g = Vector{T}(undef, k)
        for i in 1:k
            j = 1 + perm[i]
            g[i] = x[b.offset + j] - b.v0[j] * alpha
        end
        _brm_car_ltmul!(g, b.factor)
        for i in 1:k
            w[b.offset + i] = g[i]
        end
        w[b.offset + b.m] = sum(x[b.offset + i] for i in 1:b.m) / (0.001 * b.n)
    end
    w
end

# Its transpose: the cotangent of `w` from that of `x`.
function _brm_car_shared_pullback(gx::AbstractVector, f::_BRMCarSharedFactor)
    T = promote_type(eltype(gx), Float64)
    gw = Vector{T}(undef, f.m)
    for b in f.blocks
        k, perm = b.m - 1, b.factor.perm
        # x = x_g + v0 (s - 1'x_g)/n: dx_g = g - 1 (v0'g)/n, ds = v0'g/n
        vg = sum(b.v0[i] * gx[b.offset + i] for i in 1:b.m) / b.n
        e = Vector{T}(undef, k)
        for i in 1:k
            e[i] = gx[b.offset + 1 + perm[i]] - vg
        end
        _brm_car_lsolve!(e, b.factor)
        for i in 1:k
            gw[b.offset + i] = e[i]
        end
        gw[b.offset + b.m] = 0.001 * b.n * vg
    end
    gw
end

"""
    _brm_car_separation(graph, unit, nunits)

The separated sampling of a fitted `car` graph (`_brm_car_graph`) whose
structured area `p` lies in unit `unit[p]` (`0`: in no single unit). Returns

- `shared` — the separator, per structured area;
- `m`, `F` — the shared coordinates' count and their sparse map `x = F w`
  (`_brm_car_shared_values`);
- `x_areas` — the structured area of each `x` entry, `0` for an interior sum;
- `x_sums` — the `(unit, component)` of each `x` entry that is an interior sum,
  `nothing` for an area;
- `units` — per unit `(; areas, cols, factor)`: its interior areas (structured
  positions), the `x` entries its interior reads, and its sparse interior map
  `phi_I = _brm_car_unit_field(z, x[cols], factor)`;
- `order` — the structured areas in the order `vcat(x[x_phi], phi_I_1, …)`
  lays them out, with `x_phi = findall(>(0), x_areas)`;
- `comp_size` — each connected component's area count.
"""
function _brm_car_separation(graph, unit::AbstractVector{Int}, nunits::Int)
    length(unit) == graph.n_struct || error(
        "_brm_car_separation: internal: $(length(unit)) unit labels for " *
        "$(graph.n_struct) structured areas")
    shared = _brm_car_separator(graph, unit)
    x_areas = Int[]
    x_sums = Union{Nothing,Tuple{Int,Int}}[]
    shared_blocks = _BRMCarSharedBlock[]
    # per unit: (areas, kind, K or Q, border (area, x entry), x entry of its sum)
    pieces = [NamedTuple[] for _ in 1:nunits]
    for (c, (start, size)) in enumerate(zip(graph.comp_start, graph.comp_size))
        positions = start:(start + size - 1)
        S = [p for p in positions if shared[p]]
        if isempty(S)
            # One unit owns the whole component and its sum-to-zero.
            u = unit[start]
            all(p -> unit[p] == u, positions) || error(
                "_brm_car_separation: internal: component $c spans units without " *
                "a separator")
            areas = collect(positions)
            push!(pieces[u], (; areas, kind=:component,
                K=_brm_car_laplacian(graph, areas, areas), factor=nothing,
                border=Tuple{Int,Int}[], sum=0))
            continue
        end
        owners = sort!(unique(unit[p] for p in positions if !shared[p]))
        base = length(x_areas)
        append!(x_areas, S)
        append!(x_sums, fill(nothing, length(S)))
        nS = length(S)
        S_index = Dict(p => k for (k, p) in enumerate(S))
        # `P̃ = P - λ11'`, sparse, and its null direction `v0` (header).
        Pi, Pj, Pv = _brm_car_laplacian_entries!(Int[], Int[], Float64[], graph, S, S)
        v0 = vcat(ones(nS), zeros(length(owners)))
        for (j, u) in enumerate(owners)
            I = [p for p in positions if !shared[p] && unit[p] == u]
            I_index = Dict(p => k for (k, p) in enumerate(I))
            factor = _BRMCarFactor(_brm_car_laplacian(graph, I, I))
            # The interior's border: each edge to a separator area enters
            # `B = Q[I, S]` as -1.
            border = Tuple{Int,Int}[]
            for (a, b) in zip(graph.node1, graph.node2), (p, q) in ((a, b), (b, a))
                i, s = get(I_index, p, 0), get(S_index, q, 0)
                i > 0 && s > 0 && push!(border, (i, s))
            end
            adjacent = sort!(unique(last.(border)))
            column = Dict(s => k for (k, s) in enumerate(adjacent))
            n = length(I)
            # `L⁻¹ B[p, adjacent]`, one sparse solve per bordering separator area.
            W = zeros(n, length(adjacent))
            inv_perm = invperm(factor.perm)
            for (i, s) in border
                W[inv_perm[i], column[s]] -= 1.0
            end
            for k in axes(W, 2)
                _brm_car_lsolve!(Base.view(W, :, k), factor)
            end
            u0 = _brm_car_lsolve!(ones(n), factor)
            cu = LinearAlgebra.dot(u0, u0)
            h = W' * u0
            schur = h * h' ./ cu .- W' * W
            for (a, sa) in enumerate(adjacent)
                for (b, sb) in enumerate(adjacent)
                    push!(Pi, sa); push!(Pj, sb); push!(Pv, (schur[a, b] + schur[b, a]) / 2)
                end
                push!(Pi, sa, nS + j); push!(Pj, nS + j, sa); push!(Pv, h[a] / cu, h[a] / cu)
            end
            push!(Pi, nS + j); push!(Pj, nS + j); push!(Pv, 1 / cu)
            v0[nS + j] = n
            push!(pieces[u], (; areas=I, kind=:interior, K=nothing, factor,
                border=[(i, base + s) for (i, s) in border], sum=base + nS + j))
            push!(x_areas, 0)
            push!(x_sums, (u, c))
        end
        mc = nS + length(owners)
        P = sparse(Pi, Pj, Pv, mc, mc)
        push!(shared_blocks, _BRMCarSharedBlock(base, mc, size, v0,
            _BRMCarFactor(P[2:mc, 2:mc])))
    end
    m = length(x_areas)
    F = _BRMCarSharedFactor(m, shared_blocks)
    units = map(pieces) do parts
        areas = reduce(vcat, (p.areas for p in parts); init=Int[])
        cols = sort!(unique(vcat((last.(p.border) for p in parts)...,
            (p.sum for p in parts if p.kind === :interior)..., Int[])))
        col_of = Dict(k => j for (j, k) in enumerate(cols))
        built = _BRMCarPiece[]
        r = d = 0
        for p in parts
            piece = p.kind === :interior ?
                _brm_car_interior_piece(p.factor, first.(p.border),
                    [col_of[k] for k in last.(p.border)], col_of[p.sum], r, d) :
                _brm_car_component_piece(p.K, r, d)
            push!(built, piece)
            r += piece.n
            d += _brm_car_piece_draws(piece)
        end
        (; areas, cols, factor=_BRMCarUnitFactor(length(areas), d, length(cols), built))
    end
    order = vcat(filter(>(0), x_areas), (u.areas for u in units)...)
    sort(order) == 1:graph.n_struct || error(
        "_brm_car_separation: internal: the separated layout misses or repeats areas")
    (; shared, m, F, x_areas, x_sums, units, order, comp_size=collect(Int, graph.comp_size))
end
