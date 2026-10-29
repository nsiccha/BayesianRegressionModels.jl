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
#     x = F w,                      w ~ N(0, I)  (shared)
#     phi_I_u = A_u x[cols_u] + G_u z_u,   z_u ~ N(0, I)  (unit u's draws)
#
# with `F F'` the exact marginal covariance of `x`, so the density over `phi`
# is the current one up to a constant Jacobian. A component inside one unit
# has no separator: all its values are that unit's draws (`F` is empty there).
#
# Integrating unit u's interior (`K = Q[I, I]`, `B = Q[I, S]`, `c = 1'K⁻¹1`,
# `h = B'K⁻¹1`) out over its plane leaves `½ phi_S'B'K⁻¹B phi_S - (T_u + h'phi_S)² / 2c`,
# so `x`'s precision is
#
#     P_SS = Q_SS + λ11' - Σ_u B'K⁻¹B + Σ_u h h' / c
#     P_ST_u = λ1 + h / c,   P_T_uT_v = λ + δ_uv / c,
#
# and the interior's conditional mean is `K⁻¹(ν1 - B phi_S)`, `ν = (T_u + h'phi_S) / c`.

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

# The dense Laplacian block `Q[rows, cols]` of the structured areas' edges.
function _brm_car_laplacian(graph, rows::AbstractVector{Int}, cols::AbstractVector{Int})
    r = Dict(p => i for (i, p) in enumerate(rows))
    c = Dict(p => j for (j, p) in enumerate(cols))
    Q = zeros(length(rows), length(cols))
    for (a, b) in zip(graph.node1, graph.node2)
        for (p, q) in ((a, b), (b, a))
            i = get(r, p, 0)
            i == 0 && continue
            j = get(c, p, 0)
            j == 0 || (Q[i, j] += 1)
            j = get(c, q, 0)
            j == 0 || (Q[i, j] -= 1)
        end
    end
    Q
end

# `G` with `G G'` the covariance of a Gaussian with precision `K` on the plane
# `sum = const`: `G = V R⁻ᵀ` for the plane's basis `V = [I; -1']` and
# `V'KV = R R'`.
function _brm_car_plane_factor(K::AbstractMatrix)
    n = size(K, 1)
    n == 1 && return zeros(1, 0)
    V = vcat(Matrix{Float64}(LinearAlgebra.I, n - 1, n - 1), fill(-1.0, 1, n - 1))
    R = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(V' * K * V)).L
    Matrix((R \ V')')
end

# `F` with `F F' = P⁻¹`: `F = L⁻ᵀ` for `P = L L'`.
_brm_car_inverse_factor(P::AbstractMatrix) =
    Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(P)).U \ LinearAlgebra.I)

"""
    _brm_car_separation(graph, unit, nunits)

The separated sampling of a fitted `car` graph (`_brm_car_graph`) whose
structured area `p` lies in unit `unit[p]` (`0`: in no single unit). Returns

- `shared` — the separator, per structured area;
- `m`, `F` — the shared coordinates' count and the `m × m` factor of `x = F w`;
- `x_areas` — the structured area of each `x` entry, `0` for an interior sum;
- `x_sums` — the `(unit, component)` of each `x` entry that is an interior sum,
  `nothing` for an area;
- `units` — per unit `(; areas, cols, A, G)`: its interior areas (structured
  positions), the `x` entries its interior reads, and the `A` (areas × cols)
  and `G` (areas × draws) of `phi_I = A x[cols] + G z`;
- `order` — the structured areas in the order `vcat(x[x_phi], phi_I_1, …)`
  lays them out, with `x_phi = findall(>(0), x_areas)`.
"""
function _brm_car_separation(graph, unit::AbstractVector{Int}, nunits::Int)
    length(unit) == graph.n_struct || error(
        "_brm_car_separation: internal: $(length(unit)) unit labels for " *
        "$(graph.n_struct) structured areas")
    shared = _brm_car_separator(graph, unit)
    x_areas = Int[]
    x_sums = Union{Nothing,Tuple{Int,Int}}[]
    F_blocks = Matrix{Float64}[]
    rows = [Int[] for _ in 1:nunits]
    # per unit and component: (areas, x entries read, A, G)
    pieces = [NamedTuple[] for _ in 1:nunits]
    for (c, (start, size)) in enumerate(zip(graph.comp_start, graph.comp_size))
        positions = start:(start + size - 1)
        λ = 1 / (0.001 * size)^2
        S = [p for p in positions if shared[p]]
        if isempty(S)
            # One unit owns the whole component and its sum-to-zero.
            u = unit[start]
            all(p -> unit[p] == u, positions) || error(
                "_brm_car_separation: internal: component $c spans units without " *
                "a separator")
            P = _brm_car_laplacian(graph, positions, positions) .+ λ
            push!(pieces[u], (; areas=collect(positions), cols=Int[],
                A=zeros(size, 0), G=_brm_car_inverse_factor(P)))
            continue
        end
        owners = sort!(unique(unit[p] for p in positions if !shared[p]))
        base = length(x_areas)
        append!(x_areas, S)
        append!(x_sums, fill(nothing, length(S)))
        nS = length(S)
        P = zeros(nS + length(owners), nS + length(owners))
        P[1:nS, 1:nS] .= _brm_car_laplacian(graph, S, S)
        for (j, u) in enumerate(owners)
            I = [p for p in positions if !shared[p] && unit[p] == u]
            K = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(
                _brm_car_laplacian(graph, I, I)))
            # Only the separator areas the interior borders enter its terms.
            B = _brm_car_laplacian(graph, I, S)
            adjacent = findall(s -> any(!iszero, Base.view(B, :, s)), 1:nS)
            B = B[:, adjacent]
            Kinv1 = K \ ones(length(I))
            cu = sum(Kinv1)
            KinvB = K \ B
            h = B' * Kinv1
            P[adjacent, adjacent] .+= h * h' ./ cu .- B' * KinvB
            P[adjacent, nS + j] .= h ./ cu
            P[nS + j, adjacent] .= h ./ cu
            P[nS + j, nS + j] = 1 / cu
            # The interior reads the separator areas it borders and its sum.
            A = hcat(Kinv1 * h' ./ cu .- KinvB, Kinv1 ./ cu)
            G = _brm_car_plane_factor(Matrix(K))
            push!(pieces[u], (; areas=I, cols=vcat(base .+ adjacent, base + nS + j), A, G))
            push!(x_areas, 0)
            push!(x_sums, (u, c))
        end
        P .+= λ
        push!(F_blocks, _brm_car_inverse_factor(P))
    end
    m = length(x_areas)
    F = zeros(m, m)
    offset = 0
    for block in F_blocks
        k = Base.size(block, 1)
        F[offset .+ (1:k), offset .+ (1:k)] .= block
        offset += k
    end
    units = map(pieces) do parts
        areas = reduce(vcat, (p.areas for p in parts); init=Int[])
        cols = sort!(unique(reduce(vcat, (p.cols for p in parts); init=Int[])))
        col_of = Dict(k => j for (j, k) in enumerate(cols))
        draws = sum((Base.size(p.G, 2) for p in parts); init=0)
        A = zeros(length(areas), length(cols))
        G = zeros(length(areas), draws)
        r = d = 0
        for p in parts
            n, k = Base.size(p.G)
            for (j, col) in enumerate(p.cols)
                A[r .+ (1:n), col_of[col]] .= p.A[:, j]
            end
            G[r .+ (1:n), d .+ (1:k)] .= p.G
            r += n
            d += k
        end
        (; areas, cols, A, G)
    end
    order = vcat(filter(>(0), x_areas), (u.areas for u in units)...)
    sort(order) == 1:graph.n_struct || error(
        "_brm_car_separation: internal: the separated layout misses or repeats areas")
    (; shared, m, F, x_areas, x_sums, units, order)
end
