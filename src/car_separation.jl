# The unit border separation of a fitted `car` graph: the same field, sampled
# so that the areas of different units are conditionally independent given a
# few shared values, and each area's value reads `⌈log₂ n⌉` of its unit's draws
# (BRM 0.2.74; design decisions 1g11z26 and 1y5br8u, snag
# saem-s-nuts-warm-28e5f1d9).
#
# Per connected component, the field's density is the ICAR quadratic form
# `-½ Σ_edges (phi_a - phi_b)²` plus the soft sum-to-zero
# `sum(phi_c) ~ Normal(0, 0.001 n_c)`. A separator `S` holds an end of every
# edge between two units, so no edge joins interior areas of different units.
# The sum-to-zero still couples them, through the total; it is a function of
# `phi_S` and of each unit's interior mean `m_u`, so given `x = (phi_S, m)`
# the units' interiors are independent. The field is then a linear map of
# its coordinates, area by area:
#
#     phi_a = x[a]                              (a separator area)
#     phi_a = m_p + Σ_s h_s[a] z_s              (an area of a unit's piece p)
#
# A piece is a unit's interior areas in one component, in breadth-first order
# over its edges. Its `n_p - 1` draws `z` are coordinates in an orthonormal Haar
# basis of its zero-sum plane: the ordered areas are split into halves `L | R`
# recursively, and each split is one direction
# `h_s = c (1_L / |L| - 1_R / |R|)`, `c = sqrt(|L||R| / (|L| + |R|))`. So the
# draws cover the plane `sum = n_p m_p` exactly once and each area reads the
# `⌈log₂ n_p⌉` splits holding it. Consecutive differences would read two, but
# make the draws a random walk along the order, which a diagonal metric cannot
# fix: a unit's diagonally scaled condition number on the atlas's graph is
# 13–133 in this basis and 9.4e4–2.7e7 in differences (decision 1y5br8u). A
# piece of one area reads only its mean. A component inside one unit has no
# separator: its mean is `0.001 z_n`, its `n`-th draw, which makes its sum
# exactly `Normal(0, 0.001 n)`.
#
# The shared coordinates `w` are `x` with each component's last entry replaced
# by its total: `x_last = (0.001 n w_last - Σ a_j x_j) / a_last`, `a` 1 for an
# area and `n_p` for a piece mean, so `w_last ~ Normal(0, 1)` is the soft
# sum-to-zero. The density is split by what it reads:
#
# - the edges whose two ends read only `x` (separator areas and one-area
#   pieces), and each `w_last`'s standard normal, are the density of `w`
#   (`_BRMCarShared`, a shared prior);
# - every other edge reads its unit's draws, and is a term of that unit
#   (`-½ (phi_a - phi_b)²`), as is `+½ z²` for each plane draw, which cancels
#   the standard normal it is declared under (NLME estimators put
#   `Normal(0, 1)` on unit draws of a family fixed at 1).
#
# So the RK density is the ICAR density with its normalized sum, plus a
# constant, and no map reads more than a few coordinates: the units' terms,
# their η Hessians and their cells are local (`brm_nlme_model`). Before 0.2.74
# each piece's draws were whitened through its sparse Cholesky factor, so
# every area read all of its unit's draws.

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


# Appends the Haar splits of `order[lo:hi]` (header) to `reads`, one `(draw,
# coefficient)` per area and split, from draw `d + 1` on. Returns the next `d`.
function _brm_car_haar!(reads, left, right, order, lo, hi, d)
    lo >= hi && return d
    mid = (lo + hi) ÷ 2
    d += 1
    nl, nr = mid - lo + 1, hi - mid
    c = sqrt(nl * nr / (nl + nr))
    left[d], right[d] = c / nl, -c / nr
    for k in lo:mid
        push!(reads[order[k]], (d, :left))
    end
    for k in (mid + 1):hi
        push!(reads[order[k]], (d, :right))
    end
    d = _brm_car_haar!(reads, left, right, order, lo, mid, d)
    _brm_car_haar!(reads, left, right, order, mid + 1, hi, d)
end

# The areas of one piece in breadth-first order over the edges among them; a
# piece may fall apart where the separator cuts it, and each part is searched
# from its first area in `areas` order.
function _brm_car_piece_order(areas::AbstractVector{Int}, adjacency)
    inside = Set(areas)
    seen = Set{Int}()
    order = Int[]
    for root in areas
        root in seen && continue
        push!(seen, root)
        queue = [root]
        head = 1
        while head <= length(queue)
            a = queue[head]
            head += 1
            push!(order, a)
            for b in adjacency[a]
                b in inside && !(b in seen) && (push!(seen, b); push!(queue, b))
            end
        end
    end
    order
end

"""
    _brm_car_separation(graph, unit, nunits)

The local separated sampling of a fitted `car` graph (`_brm_car_graph`) whose
structured area `p` lies in unit `unit[p]` (`0`: in no single unit). Its field
is `phi[p] = means[msrc[p]] + Σ_l scaled[reads[l][p]]` with
`means = vcat(x, 0.001 .* zext)`, `scaled = vcat(left .* zext, right .* zext)`,
`zext = vcat(z, 0.0)`, `x` the shared values of the shared coordinates `w`
(`_brm_car_shared_values`) and `z` every unit's draws, unit after unit. Returns

- `shared` — the separator, per structured area;
- `m`, `blocks` — the shared coordinates' count, and per component with a
  separator its block `(; offset, mc, n, a)`: entries `offset + 1:mc` of `x`,
  the component's area count and each entry's weight in its sum;
- `x_areas` — the structured area of each `x` entry, `0` for a piece mean;
- `x_means` — the `(unit, component)` of each `x` entry that is a piece mean,
  `nothing` for an area;
- `units` — per unit `(; areas, ndraws)`: its interior areas (structured
  positions, piece by piece in their order) and its draw count;
- `msrc`, `reads`, `left`, `right` — the field's map (above): each `reads[l]`
  holds one position per structured area, and `left` / `right` each draw's
  coefficient on the two halves of its split (0 for a component's mean draw and
  for the zero slot);
- `unit_edges` — `(first, second, units)`: the edges that read draws, as
  structured areas, with the unit whose draws each reads;
- `cancel`, `cancel_units` — the plane draws (positions in `z`) whose standard
  normal the unit terms cancel, and their units;
- `shared_edges` — `(first, second)`: the other edges, as `x` entries;
- `ndraws`, `comp_size`.
"""
function _brm_car_separation(graph, unit::AbstractVector{Int}, nunits::Int)
    n = graph.n_struct
    length(unit) == n || error(
        "_brm_car_separation: internal: $(length(unit)) unit labels for $n " *
        "structured areas")
    shared = _brm_car_separator(graph, unit)
    adjacency = [Int[] for _ in 1:n]
    for (a, b) in zip(graph.node1, graph.node2)
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    x_areas = Int[]
    x_means = Union{Nothing,Tuple{Int,Int}}[]
    blocks = NamedTuple[]
    # Per area: its mean's source (an `x` entry or its unit's mean draw) and
    # its unit's local draws with the side of each split it lies on.
    mean_x = zeros(Int, n)
    mean_draw = zeros(Int, n)
    reads_local = [Tuple{Int,Symbol}[] for _ in 1:n]
    left_local = [Dict{Int,Float64}() for _ in 1:nunits]
    right_local = [Dict{Int,Float64}() for _ in 1:nunits]
    area_unit = zeros(Int, n)
    unit_areas = [Int[] for _ in 1:nunits]
    unit_draws = zeros(Int, nunits)
    cancel_local = Tuple{Int,Int}[]
    # The `n_p - 1` Haar draws of a piece in `order`, from the unit's next
    # draw on.
    function plane!(u, order)
        d0 = unit_draws[u]
        np = length(order)
        foreach(a -> area_unit[a] = u, order)
        _brm_car_haar!(reads_local, left_local[u], right_local[u], order, 1, np, d0)
        for k in 1:(np - 1)
            push!(cancel_local, (u, d0 + k))
        end
        unit_draws[u] += np - 1
        append!(unit_areas[u], order)
    end
    for (c, (start, size)) in enumerate(zip(graph.comp_start, graph.comp_size))
        positions = start:(start + size - 1)
        S = [p for p in positions if shared[p]]
        if isempty(S)
            # One unit owns the whole component and its sum-to-zero: its mean
            # is its last draw, scaled.
            u = unit[start]
            all(p -> unit[p] == u, positions) || error(
                "_brm_car_separation: internal: component $c spans units without " *
                "a separator")
            order = _brm_car_piece_order(collect(positions), adjacency)
            plane!(u, order)
            unit_draws[u] += 1
            for a in order
                mean_draw[a] = unit_draws[u]
            end
            continue
        end
        base = length(x_areas)
        for (k, p) in enumerate(S)
            push!(x_areas, p)
            push!(x_means, nothing)
            mean_x[p] = base + k
        end
        owners = sort!(unique(unit[p] for p in positions if !shared[p]))
        weights = vcat(ones(length(S)), zeros(length(owners)))
        for (j, u) in enumerate(owners)
            I = [p for p in positions if !shared[p] && unit[p] == u]
            push!(x_areas, 0)
            push!(x_means, (u, c))
            xm = base + length(S) + j
            weights[length(S) + j] = length(I)
            for a in I
                mean_x[a] = xm
            end
            if length(I) == 1
                area_unit[only(I)] = u
                push!(unit_areas[u], only(I))
            else
                plane!(u, _brm_car_piece_order(I, adjacency))
            end
        end
        push!(blocks, (; offset=base, mc=length(weights), n=size, a=weights))
    end
    m = length(x_areas)
    ndraws = sum(unit_draws; init=0)
    offsets = cumsum(vcat(0, unit_draws[1:end-1]))
    zero_slot, zero_mean = ndraws + 1, m + ndraws + 1
    msrc = [mean_x[a] > 0 ? mean_x[a] : mean_draw[a] > 0 ?
        m + offsets[area_unit[a]] + mean_draw[a] : zero_mean for a in 1:n]
    any(==(zero_mean), msrc) && error(
        "_brm_car_separation: internal: an area reads no mean")
    left, right = zeros(ndraws + 1), zeros(ndraws + 1)
    for u in 1:nunits, (d, c) in left_local[u]
        left[offsets[u] + d] = c
    end
    for u in 1:nunits, (d, c) in right_local[u]
        right[offsets[u] + d] = c
    end
    depth = maximum(length, reads_local; init=0)
    reads = [fill(zero_slot, n) for _ in 1:depth]
    for a in 1:n, (l, (d, side)) in enumerate(reads_local[a])
        reads[l][a] = offsets[area_unit[a]] + d + (side === :left ? 0 : zero_slot)
    end
    reads_draws(a) = mean_draw[a] > 0 || !isempty(reads_local[a])
    unit_first, unit_second, edge_units = Int[], Int[], Int[]
    shared_first, shared_second = Int[], Int[]
    for (a, b) in zip(graph.node1, graph.node2)
        if reads_draws(a) || reads_draws(b)
            us = unique(area_unit[p] for p in (a, b) if reads_draws(p))
            length(us) == 1 || error("_brm_car_separation: internal: an edge reads " *
                "the draws of $(length(us)) units")
            push!(unit_first, a); push!(unit_second, b); push!(edge_units, only(us))
        else
            push!(shared_first, mean_x[a]); push!(shared_second, mean_x[b])
        end
    end
    cancel = [offsets[u] + d for (u, d) in cancel_local]
    cancel_units = [u for (u, _) in cancel_local]
    units = [(; areas=unit_areas[u], ndraws=unit_draws[u]) for u in 1:nunits]
    sort!(vcat(filter(>(0), x_areas), (u.areas for u in units)...)) == 1:n || error(
        "_brm_car_separation: internal: the separated layout misses or repeats areas")
    (; shared, m, blocks, x_areas, x_means, units, msrc, reads, left, right,
       unit_edges=(unit_first, unit_second, edge_units), cancel, cancel_units,
       shared_edges=(shared_first, shared_second), ndraws,
       comp_size=collect(Int, graph.comp_size))
end

# The data `_brm_car_shared_values` reads, per `x` entry: the entry its
# component's dependent one is (`last`), the scale of that entry's total, and
# each other entry's weight in it relative to the dependent one's.
function _brm_car_shared_data(blocks, m)
    last = zeros(Int, m)
    scale = zeros(m)
    weight = zeros(m)
    for b in blocks
        l = b.offset + b.mc
        for j in 1:b.mc
            last[b.offset + j] = l
            weight[b.offset + j] = b.a[j] / b.a[b.mc]
        end
        scale[l] = 0.001 * b.n / b.a[b.mc]
    end
    (; last, scale, weight)
end

"""
    _brm_car_shared_values(w, last, scale, weight)

A separated field's shared values `x` from its shared coordinates `w`: each
entry its coordinate, except each component's dependent last entry,
`(0.001 n w_last - Σ a_j x_j) / a_last` (`_brm_car_shared_data`). Linear in
`m`; reads its inputs and writes only a fresh buffer, so Enzyme differentiates
it.
"""
function _brm_car_shared_values(w::AbstractVector, last, scale, weight)
    T = promote_type(eltype(w), Float64)
    x = Vector{T}(undef, length(w))
    @inbounds for j in eachindex(w)
        x[j] = last[j] == j ? scale[j] * w[j] : w[j]
    end
    @inbounds for j in eachindex(w)
        l = last[j]
        l == j || (x[l] -= weight[j] * w[j])
    end
    x
end

# Its inverse and transpose (the coordinate transport).
function _brm_car_shared_draws(x::AbstractVector, last, scale, weight)
    # A total is `Σ a_j x_j / (0.001 n)`; the dependent entry's weight is 1.
    w = Vector{promote_type(eltype(x), Float64)}(x)
    totals = zeros(eltype(w), length(x))
    for j in eachindex(x)
        totals[last[j]] += weight[j] * x[j]
    end
    for j in eachindex(x)
        last[j] == j && (w[j] = totals[j] / scale[j])
    end
    w
end
function _brm_car_shared_pullback(gx::AbstractVector, last, scale, weight)
    gw = Vector{promote_type(eltype(gx), Float64)}(gx)
    for j in eachindex(gx)
        l = last[j]
        l == j ? (gw[j] = scale[j] * gx[j]) : (gw[j] -= weight[j] * gx[l])
    end
    gw
end

# The shared coordinates' density (`_BRMCarShared`): each component's total
# a standard normal, and the edges whose ends read only `x`.
function _brm_car_shared_logdensity(w::AbstractVector, last, scale, weight,
        first_entry, second_entry)
    x = _brm_car_shared_values(w, last, scale, weight)
    total = zero(eltype(x))
    @inbounds for j in eachindex(w)
        last[j] == j && (total -= (w[j]^2 + log(2π)) / 2)
    end
    @inbounds for k in eachindex(first_entry)
        total -= (x[first_entry[k]] - x[second_entry[k]])^2 / 2
    end
    total
end

"""
    _brm_car_field(x, z, sep)

A separated field in graph order from its shared values `x` and its draws `z`
(every unit's, unit after unit): `sep.msrc`, `sep.reads`, `sep.left`, `sep.right`
of `_brm_car_separation`.
"""
function _brm_car_field(x::AbstractVector, z::AbstractVector, sep)
    zext = vcat(z, 0.0)
    means = vcat(x, 0.001 .* zext)
    scaled = vcat(sep.left .* zext, sep.right .* zext)
    [means[sep.msrc[p]] + sum(r -> scaled[r[p]], sep.reads; init=zero(eltype(scaled)))
        for p in eachindex(sep.msrc)]
end

# Its inverse: the shared values and draws of a field `phi` (graph order). A
# piece's mean is the mean of its areas, and each plane draw is `h_s'phi`: the
# directions are orthonormal and orthogonal to the mean.
function _brm_car_field_coordinates(phi::AbstractVector, sep)
    T = promote_type(eltype(phi), Float64)
    m, nd = sep.m, sep.ndraws
    sums = zeros(T, m + nd)
    counts = zeros(Int, m + nd)
    for p in eachindex(phi)
        k = sep.msrc[p]
        sums[k] += phi[p]
        counts[k] += 1
    end
    x = [counts[k] == 0 ? zero(T) : sums[k] / counts[k] for k in 1:m]
    z = zeros(T, nd)
    for k in (m + 1):(m + nd)
        counts[k] > 0 && (z[k - m] = sums[k] / counts[k] / 0.001)
    end
    zero_slot = nd + 1
    for r in sep.reads, p in eachindex(phi)
        k = r[p]
        k == zero_slot && continue
        k < zero_slot ? (z[k] += sep.left[k] * phi[p]) :
            (z[k - zero_slot] += sep.right[k - zero_slot] * phi[p])
    end
    x, z
end

# Its transpose: the cotangents of `(x, z)` from those of `phi`.
function _brm_car_field_transpose(g::AbstractVector, sep)
    T = promote_type(eltype(g), Float64)
    m, nd = sep.m, sep.ndraws
    gmeans = zeros(T, m + nd + 1)
    gz = zeros(T, nd)
    zero_slot = nd + 1
    for p in eachindex(g)
        gmeans[sep.msrc[p]] += g[p]
    end
    for r in sep.reads, p in eachindex(g)
        k = r[p]
        k == zero_slot && continue
        k < zero_slot ? (gz[k] += sep.left[k] * g[p]) :
            (gz[k - zero_slot] += sep.right[k - zero_slot] * g[p])
    end
    gmeans[1:m], gz .+ 0.001 .* gmeans[(m + 1):(m + nd)]
end

# The units' terms of a separated field at `phi` and its draws `z`: each edge
# that reads draws, then each plane draw's cancelled standard normal.
_brm_car_unit_terms(phi::AbstractVector, z::AbstractVector, sep) =
    vcat([-(phi[a] - phi[b])^2 / 2 for (a, b) in zip(sep.unit_edges[1], sep.unit_edges[2])],
        [z[d]^2 / 2 for d in sep.cancel])

# ---- RKBRMI: the law of the shared coordinates ---------------------------------
#
# `w[1:m] ~ _BRMCarShared(last, scale, weight, first, second)` (rk_ast.jl): `w`
# is real and packed as itself (the ReactiveKernels extension), and its density
# is `_brm_car_shared_logdensity`. A shared prior, so NLME estimators read it
# in the population prior.
struct _BRMCarShared{L,S,W,F,G}
    last::L
    scale::S
    weight::W
    first_entry::F
    second_entry::G
end
