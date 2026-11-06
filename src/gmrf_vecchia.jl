# A nearest-neighbour (Vecchia) representation of the space-time Matérn × OU
# field on the (position, time) pairs its rows read. The state is bounded by
# that footprint: there is no mesh and no position × time lattice. Each state
# is Gaussian given a bounded set of earlier states, under the term's exact
# covariance
#
#     σ² Ω_cc′ M₁(κ r) exp(-|d| / ℓ_time),    M₁(x) = x K₁(x),  κ = √2 / ℓ_space,
#
# so the joint precision (I - B)' F⁻¹ (I - B) is sparse. Neighbour selection
# and every index are data-only; only the conditional weights and variances
# move with the two lengths. Channels are separable: the conditionals are
# solved once in the scalar correlation and the channel factor whitens each
# state, so a state holds one coordinate per channel.
#
# States are ordered by time, then by a maximin position order fixed for the
# whole footprint. A state conditions on
#   - its `neighbors` nearest lower-order positions read at its own time,
#   - the previous read of its own position and of each of those neighbours,
#   - when its own position was not read at the previous read time, its
#     `neighbors` nearest among the latest earlier reads, reaching back until
#     they hold at least `neighbors` reads.
# On positions read at consecutive times this conditional is exactly
# `a x(s, t⁻) + w′(x(N, t) - a x(N, t⁻))` with variance `(1 - a²) F`, so the
# implied covariance is the spatial Vecchia covariance times the exact OU
# correlation. Every read time conditions on earlier reads, whatever the gap,
# so no temporal block is cut.

"""
The fitted footprint of a Vecchia `gmrf` term: data only, no sampled value.
Conditionals with the same geometry (target position, conditioning positions,
lags) share a pattern, so a term solves each distinct geometry once per
evaluation however many states repeat it.
"""
struct _BRMGMRFFootprint
    east::Vector{Float64}
    north::Vector{Float64}
    # (east, north) → position, in the maximin order.
    position_index::Dict{Tuple{Float64,Float64},Int32}
    times::Vector{Float64}
    # States of time t are `time_ptr[t]:(time_ptr[t + 1] - 1)`, ascending in position.
    time_ptr::Vector{Int}
    position::Vector{Int32}
    # Earlier states each state conditions on, in its pattern's item order.
    neighbor_ptr::Vector{Int}
    neighbor::Vector{Int32}
    # States of pattern p are `pattern_states[pattern_ptr[p]:(pattern_ptr[p + 1] - 1)]`.
    pattern_ptr::Vector{Int}
    pattern_states::Vector{Int32}
    # Pattern geometry: distinct positions (the target first), distinct lags
    # from the target's time (zero first), and for each conditioning item
    # its entries in both tables.
    geometry_ptr::Vector{Int}
    geometry_position::Vector{Int32}
    lag_ptr::Vector{Int}
    lag::Vector{Float64}
    item_ptr::Vector{Int}
    item_position::Vector{UInt8}
    item_lag::Vector{UInt8}
    neighbors::Int
    nchannels::Int
    ncoordinates::Int
    max_items::Int
    max_positions::Int
    max_lags::Int
end

# A footprint is plain data, so it equals a copy of itself. Queries that keep a copy of
# their bound data compare it this way, and a footprint is bound as a model input.
_brm_vecchia_same_fields(same, a::T, b::T) where {T} =
    all(name -> same(getfield(a, name), getfield(b, name)), fieldnames(T))
Base.isequal(a::_BRMGMRFFootprint, b::_BRMGMRFFootprint) = _brm_vecchia_same_fields(isequal, a, b)
Base.:(==)(a::_BRMGMRFFootprint, b::_BRMGMRFFootprint) = _brm_vecchia_same_fields(==, a, b)
Base.hash(f::_BRMGMRFFootprint, h::UInt) =
    hash(_BRMGMRFFootprint, hash(f.ncoordinates, hash(f.neighbors, hash(f.nchannels, h))))

Base.show(io::IO, f::_BRMGMRFFootprint) = print(io, "_BRMGMRFFootprint(", length(f.position),
    " states, ", length(f.east), " positions, ", length(f.times), " times, ",
    length(f.pattern_ptr) - 1, " patterns, ", length(f.neighbor), " conditioning entries)")

# A balanced point tree whose nodes count their inserted points, so nearest
# queries see only what has been inserted and skip empty subtrees. Queries stay
# local however unevenly the points cluster, and points are slots `1:n`.
struct _BRMVecchiaTree
    x::Vector{Float64}
    y::Vector{Float64}
    order::Vector{Int32}
    first_slot::Vector{Int32}
    last_slot::Vector{Int32}
    left::Vector{Int32}
    right::Vector{Int32}
    parent::Vector{Int32}
    box::Vector{NTuple{4,Float64}}
    count::Vector{Int32}
    leaf::Vector{Int32}
    active::Vector{Bool}
end

function _BRMVecchiaTree(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    tree = _BRMVecchiaTree(x, y, collect(Int32(1):Int32(n)), Int32[], Int32[], Int32[], Int32[],
        Int32[], NTuple{4,Float64}[], Int32[], zeros(Int32, n), fill(false, n))
    n > 0 && _brm_vecchia_tree_node!(tree, 1, n, Int32(0))
    tree
end

function _brm_vecchia_tree_node!(tree, first_slot, last_slot, parent)
    x0, x1, y0, y1 = Inf, -Inf, Inf, -Inf
    @inbounds for k in first_slot:last_slot
        i = tree.order[k]
        x0, x1 = min(x0, tree.x[i]), max(x1, tree.x[i])
        y0, y1 = min(y0, tree.y[i]), max(y1, tree.y[i])
    end
    push!(tree.first_slot, first_slot)
    push!(tree.last_slot, last_slot)
    push!(tree.left, 0)
    push!(tree.right, 0)
    push!(tree.parent, parent)
    push!(tree.box, (x0, x1, y0, y1))
    push!(tree.count, 0)
    node = Int32(length(tree.box))
    if last_slot - first_slot < 8
        @inbounds for k in first_slot:last_slot
            tree.leaf[tree.order[k]] = node
        end
        return node
    end
    middle = (first_slot + last_slot) ÷ 2
    coordinate = x1 - x0 >= y1 - y0 ? tree.x : tree.y
    partialsort!(view(tree.order, first_slot:last_slot), middle - first_slot + 1;
        by=i -> (coordinate[i], i))
    tree.left[node] = _brm_vecchia_tree_node!(tree, first_slot, middle, node)
    tree.right[node] = _brm_vecchia_tree_node!(tree, middle + 1, last_slot, node)
    node
end

function _brm_vecchia_insert!(tree::_BRMVecchiaTree, slot)
    tree.active[slot] = true
    node = tree.leaf[slot]
    @inbounds while node != 0
        tree.count[node] += 1
        node = tree.parent[node]
    end
    nothing
end
_brm_vecchia_insert_all!(tree::_BRMVecchiaTree) =
    foreach(slot -> _brm_vecchia_insert!(tree, slot), eachindex(tree.x))

@inline function _brm_vecchia_box_distance(box, x, y)
    dx = max(box[1] - x, 0.0, x - box[2])
    dy = max(box[3] - y, 0.0, y - box[4])
    dx^2 + dy^2
end

# The `k` inserted slots nearest (x, y) in `out`, nearest first, with their
# squared distances in `distance`; returns how many there are.
function _brm_vecchia_nearest!(out::Vector{Int32}, distance::Vector{Float64}, tree::_BRMVecchiaTree,
        x, y, k)
    isempty(tree.box) && return 0
    _brm_vecchia_nearest!(out, distance, tree, Int32(1), x, y, k, 0)
end
function _brm_vecchia_nearest!(out, distance, tree::_BRMVecchiaTree, node, x, y, k, found)
    @inbounds begin
        tree.count[node] == 0 && return found
        left = tree.left[node]
        if left == 0
            for slot in tree.first_slot[node]:tree.last_slot[node]
                candidate = tree.order[slot]
                tree.active[candidate] || continue
                d = (tree.x[candidate] - x)^2 + (tree.y[candidate] - y)^2
                if found < k || d < distance[k]
                    position = found < k ? (found += 1) : k
                    while position > 1 && (distance[position - 1], out[position - 1]) > (d, candidate)
                        distance[position] = distance[position - 1]
                        out[position] = out[position - 1]
                        position -= 1
                    end
                    distance[position] = d
                    out[position] = candidate
                end
            end
            return found
        end
        right = tree.right[node]
        near_left = _brm_vecchia_box_distance(tree.box[left], x, y)
        near_right = _brm_vecchia_box_distance(tree.box[right], x, y)
        first_child, second_child, second_distance = near_left <= near_right ?
            (left, right, near_right) : (right, left, near_left)
        found = _brm_vecchia_nearest!(out, distance, tree, first_child, x, y, k, found)
        if found < k || second_distance < distance[k]
            found = _brm_vecchia_nearest!(out, distance, tree, second_child, x, y, k, found)
        end
    end
    found
end

# Greedy maximin order: each position is the one farthest from all chosen so
# far, starting from the position nearest the centroid. Distances only shrink,
# so a stale heap entry is an upper bound and is re-queried only when it reaches
# the top. A function of the positions as listed, ties resolved by list index,
# which `_brm_vecchia_footprint` fixes by listing them sorted.
function _brm_vecchia_position_order(east::Vector{Float64}, north::Vector{Float64})
    n = length(east)
    n == 0 && return Int32[]
    tree = _BRMVecchiaTree(east, north)
    cx, cy = sum(east) / n, sum(north) / n
    start = argmin(i -> ((east[i] - cx)^2 + (north[i] - cy)^2, i), 1:n)
    order = Vector{Int32}(undef, n)
    order[1] = start
    _brm_vecchia_insert!(tree, start)
    # A max-heap of (bound on squared distance to the chosen set, -index).
    heap = [((east[i] - east[start])^2 + (north[i] - north[start])^2, -Int32(i)) for i in 1:n if i != start]
    _brm_vecchia_heapify!(heap)
    out, distance = Vector{Int32}(undef, 1), Vector{Float64}(undef, 1)
    chosen = 1
    while !isempty(heap)
        negative = heap[1][2]
        i = -negative
        _brm_vecchia_nearest!(out, distance, tree, east[i], north[i], 1)
        current = distance[1]
        if length(heap) == 1 || (current, negative) >= _brm_vecchia_heap_second(heap)
            _brm_vecchia_heap_pop!(heap)
            chosen += 1
            order[chosen] = i
            _brm_vecchia_insert!(tree, i)
        else
            heap[1] = (current, negative)
            _brm_vecchia_sift_down!(heap, 1)
        end
    end
    order
end

function _brm_vecchia_sift_down!(heap, i)
    n = length(heap)
    @inbounds while true
        l, r, largest = 2i, 2i + 1, i
        l <= n && heap[l] > heap[largest] && (largest = l)
        r <= n && heap[r] > heap[largest] && (largest = r)
        largest == i && return heap
        heap[i], heap[largest] = heap[largest], heap[i]
        i = largest
    end
end
function _brm_vecchia_heapify!(heap)
    for i in (length(heap) ÷ 2):-1:1
        _brm_vecchia_sift_down!(heap, i)
    end
    heap
end
_brm_vecchia_heap_second(heap) = length(heap) >= 3 ? max(heap[2], heap[3]) : heap[2]
function _brm_vecchia_heap_pop!(heap)
    heap[1] = heap[end]
    pop!(heap)
    isempty(heap) || _brm_vecchia_sift_down!(heap, 1)
    heap
end

# The lower-order nearest neighbours of every position of one time, as slots
# into that time's ascending members.
struct _BRMVecchiaWithin
    members::Vector{Int32}
    lists::Vector{Int32}
    ptr::Vector{Int}
end

function _brm_vecchia_within(east, north, members, neighbors)
    n = length(members)
    tree = _BRMVecchiaTree(east[members], north[members])
    lists = Int32[]
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    out = Vector{Int32}(undef, neighbors)
    distance = Vector{Float64}(undef, neighbors)
    for i in 1:n
        found = _brm_vecchia_nearest!(out, distance, tree, tree.x[i], tree.y[i], neighbors)
        for a in 1:found
            push!(lists, out[a])
        end
        ptr[i + 1] = length(lists) + 1
        _brm_vecchia_insert!(tree, i)
    end
    _BRMVecchiaWithin(collect(members), lists, ptr)
end

function _brm_vecchia_cached_within!(cache, east, north, members, neighbors)
    entries = get!(Vector{_BRMVecchiaWithin}, cache, hash(members))
    for entry in entries
        entry.members == members && return entry
    end
    entry = _brm_vecchia_within(east, north, members, neighbors)
    push!(entries, entry)
    entry
end

# Interns conditional geometries. A key is verified against the stored
# pattern, so a hash collision cannot merge two geometries.
mutable struct _BRMVecchiaPatterns
    table::Dict{UInt64,Vector{Int32}}
    target::Vector{Int32}
    item_ptr::Vector{Int}
    item_position::Vector{Int32}
    item_lag::Vector{Float64}
end
_BRMVecchiaPatterns() = _BRMVecchiaPatterns(Dict{UInt64,Vector{Int32}}(), Int32[], [1], Int32[], Float64[])

function _brm_vecchia_intern!(p::_BRMVecchiaPatterns, target, positions, lags)
    h = hash(target)
    for a in eachindex(positions)
        h = hash(lags[a], hash(positions[a], h))
    end
    bucket = get!(Vector{Int32}, p.table, h)
    m = length(positions)
    for candidate in bucket
        p.target[candidate] == target || continue
        first_item = p.item_ptr[candidate]
        p.item_ptr[candidate + 1] - first_item == m || continue
        same = true
        for a in 1:m
            if p.item_position[first_item + a - 1] != positions[a] || p.item_lag[first_item + a - 1] != lags[a]
                same = false
                break
            end
        end
        same && return candidate
    end
    push!(p.target, target)
    append!(p.item_position, positions)
    append!(p.item_lag, lags)
    push!(p.item_ptr, length(p.item_position) + 1)
    id = Int32(length(p.target))
    push!(bucket, id)
    id
end

"""
    _brm_vecchia_footprint(east, north, time, nchannels; neighbors)

The footprint of rows reading `(east[i], north[i], time[i])`: one state per
distinct read pair. Data only, and a function of the set of read pairs: the
same reads in another row order give the same footprint. Log-linear in the rows
(one sort and one tree query per state, a neighbour search once per distinct set
of read positions).
"""
function _brm_vecchia_footprint(east::AbstractVector, north::AbstractVector, time::AbstractVector,
        nchannels::Integer=1; neighbors::Integer=10)
    n = length(time)
    length(east) == length(north) == n || throw(DimensionMismatch(
        "gmrf footprint coordinates need one entry per row"))
    n > 0 || throw(ArgumentError("gmrf footprint needs at least one read"))
    all(isfinite, east) && all(isfinite, north) ||
        throw(ArgumentError("gmrf spatial coordinates must be finite"))
    all(isfinite, time) || throw(ArgumentError("gmrf times must be finite"))
    neighbors >= 1 || throw(ArgumentError("gmrf neighbors must be a positive integer"))
    # A pattern's tables index its distinct positions and lags in one byte, and a
    # conditioning set of 3m + 1 reads costs the cube of that in every solve.
    neighbors <= 100 || throw(ArgumentError("gmrf neighbors is limited to 100; got $neighbors"))
    nchannels >= 1 || throw(ArgumentError("gmrf needs at least one channel"))
    # Positions, in the fixed maximin order. Sorted first, so a tied step of that
    # order resolves the same way whatever order the rows read them in.
    seen = Set{Tuple{Float64,Float64}}()
    for i in 1:n
        push!(seen, (Float64(east[i]), Float64(north[i])))
    end
    positions = sort!(collect(seen))
    input_east, input_north = first.(positions), last.(positions)
    order = _brm_vecchia_position_order(input_east, input_north)
    position_east, position_north = input_east[order], input_north[order]
    position_index = Dict{Tuple{Float64,Float64},Int32}(
        (position_east[r], position_north[r]) => Int32(r) for r in eachindex(order))
    # Distinct (time, position) pairs, sorted: the states.
    times = sort!(unique(Float64.(time)))
    time_rows = Vector{Int32}(undef, n)
    let index = Dict(t => Int32(i) for (i, t) in enumerate(times))
        for i in 1:n
            time_rows[i] = index[Float64(time[i])]
        end
    end
    keys = Vector{Int64}(undef, n)
    npos = Int64(length(order))
    for i in 1:n
        keys[i] = (Int64(time_rows[i]) - 1) * npos +
            (Int64(position_index[(Float64(east[i]), Float64(north[i]))]) - 1)
    end
    sort!(keys)
    unique!(keys)
    nstates = length(keys)
    nstates <= typemax(Int32) || throw(ArgumentError("gmrf footprint has too many states"))
    position = Vector{Int32}(undef, nstates)
    time_ptr = ones(Int, length(times) + 1)
    for s in 1:nstates
        t, p = divrem(keys[s], npos)
        position[s] = Int32(p + 1)
        time_ptr[t + 2] = s + 1
    end
    for t in 2:(length(times) + 1)
        time_ptr[t] = max(time_ptr[t], time_ptr[t - 1])
    end
    keys = Int64[]
    # Conditioning sets.
    last_read = zeros(Int32, npos)
    neighbor_ptr = Vector{Int}(undef, nstates + 1)
    neighbor_ptr[1] = 1
    neighbor = Int32[]
    pattern_of = Vector{Int32}(undef, nstates)
    patterns = _BRMVecchiaPatterns()
    # Lower-order neighbours are computed once per distinct set of read positions.
    within_cache = Dict{UInt64,Vector{_BRMVecchiaWithin}}()
    out = Vector{Int32}(undef, neighbors)
    distance = Vector{Float64}(undef, neighbors)
    item_state, positions, lags = Int32[], Int32[], Float64[]
    time_of = Vector{Int32}(undef, nstates)
    for t in eachindex(times)
        range = time_ptr[t]:(time_ptr[t + 1] - 1)
        time_of[range] .= t
        members = view(position, range)
        within = _brm_vecchia_cached_within!(within_cache, position_east, position_north, members, neighbors)
        # The latest earlier reads, back to at least `neighbors` of them, searched
        # only for positions the previous time did not read.
        recent_first = time_ptr[t]
        for earlier in (t - 1):-1:1
            time_ptr[t] - recent_first >= neighbors && break
            recent_first = time_ptr[earlier]
        end
        recent_range = recent_first:(time_ptr[t] - 1)
        recent_tree = nothing
        for (i, s) in enumerate(range)
            target = position[s]
            empty!(item_state)
            for q in within.ptr[i]:(within.ptr[i + 1] - 1)
                slot = within.lists[q]
                push!(item_state, Int32(first(range) + slot - 1))
                last_read[members[slot]] != 0 && push!(item_state, last_read[members[slot]])
            end
            own = last_read[target]
            own != 0 && push!(item_state, own)
            if t > 1 && !(own != 0 && time_of[own] == t - 1)
                if recent_tree === nothing
                    recent = view(position, recent_range)
                    recent_tree = _BRMVecchiaTree(position_east[recent], position_north[recent])
                    _brm_vecchia_insert_all!(recent_tree)
                end
                found = _brm_vecchia_nearest!(out, distance, recent_tree,
                    position_east[target], position_north[target], neighbors)
                for a in 1:found
                    push!(item_state, Int32(first(recent_range) + out[a] - 1))
                end
            end
            # A short list: the default sort's radix scratch would dominate the build.
            sort!(item_state; alg=InsertionSort)
            unique!(item_state)
            resize!(positions, length(item_state))
            resize!(lags, length(item_state))
            for (a, q) in enumerate(item_state)
                positions[a] = position[q]
                lags[a] = times[t] - times[time_of[q]]
            end
            pattern_of[s] = _brm_vecchia_intern!(patterns, target, positions, lags)
            append!(neighbor, item_state)
            neighbor_ptr[s + 1] = length(neighbor) + 1
        end
        for s in range
            last_read[position[s]] = Int32(s)
        end
    end
    empty!(within_cache)
    # Group states by pattern and index each pattern's geometry tables.
    npatterns = length(patterns.target)
    counts = zeros(Int, npatterns + 1)
    for s in 1:nstates
        counts[pattern_of[s] + 1] += 1
    end
    pattern_ptr = cumsum(counts) .+ 1
    next_slot = copy(pattern_ptr)
    pattern_states = Vector{Int32}(undef, nstates)
    for s in 1:nstates
        pattern_states[next_slot[pattern_of[s]]] = Int32(s)
        next_slot[pattern_of[s]] += 1
    end
    geometry_ptr = ones(Int, npatterns + 1)
    lag_ptr = ones(Int, npatterns + 1)
    geometry_position, lag_table = Int32[], Float64[]
    item_position = Vector{UInt8}(undef, length(patterns.item_position))
    item_lag = Vector{UInt8}(undef, length(patterns.item_position))
    max_items = max_positions = max_lags = 0
    for p in 1:npatterns
        first_geometry = length(geometry_position) + 1
        first_lag = length(lag_table) + 1
        push!(geometry_position, patterns.target[p])
        push!(lag_table, 0.0)
        range = patterns.item_ptr[p]:(patterns.item_ptr[p + 1] - 1)
        for q in range
            u = findfirst(==(patterns.item_position[q]), view(geometry_position, first_geometry:length(geometry_position)))
            if u === nothing
                push!(geometry_position, patterns.item_position[q])
                u = length(geometry_position) - first_geometry + 1
            end
            l = findfirst(==(patterns.item_lag[q]), view(lag_table, first_lag:length(lag_table)))
            if l === nothing
                push!(lag_table, patterns.item_lag[q])
                l = length(lag_table) - first_lag + 1
            end
            item_position[q], item_lag[q] = UInt8(u), UInt8(l)
        end
        geometry_ptr[p + 1] = length(geometry_position) + 1
        lag_ptr[p + 1] = length(lag_table) + 1
        max_items = max(max_items, length(range))
        max_positions = max(max_positions, geometry_ptr[p + 1] - first_geometry)
        max_lags = max(max_lags, lag_ptr[p + 1] - first_lag)
    end
    coordinates = Base.checked_mul(nstates, Int(nchannels))
    _BRMGMRFFootprint(position_east, position_north, position_index, times, time_ptr, position,
        neighbor_ptr, neighbor, pattern_ptr, pattern_states,
        geometry_ptr, geometry_position, lag_ptr, lag_table, patterns.item_ptr, item_position, item_lag,
        Int(neighbors), Int(nchannels), coordinates, max_items, max_positions, max_lags)
end

# The state read by one (east, north, time) triple, or an error naming the row.
function _brm_vecchia_state(f::_BRMGMRFFootprint, x, y, t, row)
    position = get(f.position_index, (Float64(x), Float64(y)), Int32(0))
    position == 0 && throw(ArgumentError(
        "gmrf row $row reads position ($x, $y), which is not in the fitted footprint; " *
        "a footprint field has states only where the fit read it"))
    k = searchsortedfirst(f.times, Float64(t))
    k <= length(f.times) && f.times[k] == t || throw(ArgumentError(
        "gmrf row $row has a time outside the fitted footprint"))
    range = f.time_ptr[k]:(f.time_ptr[k + 1] - 1)
    offset = searchsortedfirst(view(f.position, range), position)
    offset <= length(range) && f.position[first(range) + offset - 1] == position || throw(ArgumentError(
        "gmrf row $row reads ($x, $y) at time $t, which the fit did not read; " *
        "a footprint field has states only at the fitted (position, time) pairs"))
    first(range) + offset - 1
end

# The field coordinate each row reads: `channel + nchannels * (state - 1)`.
function _brm_vecchia_field_index(f::_BRMGMRFFootprint, channels, east, north, time, row_channels)
    length(east) == length(north) == length(time) == length(row_channels) || throw(DimensionMismatch(
        "gmrf row columns need one entry per row"))
    C = length(channels)
    C == f.nchannels || throw(DimensionMismatch("gmrf footprint channel axis disagrees"))
    channel_index = Dict(c => i for (i, c) in enumerate(channels))
    index = Vector{Int}(undef, length(time))
    for row in eachindex(index)
        c = get(channel_index, row_channels[row], 0)
        c != 0 || throw(ArgumentError("gmrf row $row has an unknown channel"))
        index[row] = c + C * (_brm_vecchia_state(f, east[row], north[row], time[row], row) - 1)
    end
    index
end

# Which position, time and channel a field coordinate holds.
function _brm_vecchia_coordinate(f::_BRMGMRFFootprint, channels, index::Integer)
    1 <= index <= f.ncoordinates || throw(BoundsError(f, index))
    state, channel = divrem(index - 1, f.nchannels)
    p = f.position[state + 1]
    t = searchsortedlast(f.time_ptr, state + 1)
    (; east=f.east[p], north=f.north[p], time=f.times[t], channel=channels[channel + 1])
end

@inline _brm_vecchia_matern(x) = x == 0 ? 1.0 : x * besselk1(x)
# d/dx [x K₁(x)] = -x K₀(x): the value's only derivative the term needs.
@inline _brm_vecchia_matern_slope(x) = x == 0 ? 0.0 : -x * besselk0(x)

# One evaluation's reusable scratch, sized by the footprint's largest pattern.
struct _BRMVecchiaWork
    A::Matrix{Float64}
    b::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
    space::Matrix{Float64}
    dspace::Matrix{Float64}
    time::Matrix{Float64}
    dtime::Matrix{Float64}
end
function _BRMVecchiaWork(f::_BRMGMRFFootprint)
    m, u, l = f.max_items, f.max_positions, f.max_lags
    _BRMVecchiaWork(zeros(m, m), zeros(m), zeros(m), zeros(m),
        zeros(u, u), zeros(u, u), zeros(l, l), zeros(l, l))
end

# The conditional weights `work.b`, the Cholesky factor of the conditioning
# correlation in `work.A` and the conditional variance of pattern `p`.
function _brm_vecchia_solve!(work, f::_BRMGMRFFootprint, p, kappa, temporal_length, want_slopes)
    geometry = f.geometry_ptr[p]
    nu = f.geometry_ptr[p + 1] - geometry
    space, dspace, timecor, dtime = work.space, work.dspace, work.time, work.dtime
    @inbounds for j in 1:nu
        pj = f.geometry_position[geometry + j - 1]
        space[j, j] = 1.0
        dspace[j, j] = 0.0
        for i in (j + 1):nu
            pi_ = f.geometry_position[geometry + i - 1]
            x = kappa * hypot(f.east[pi_] - f.east[pj], f.north[pi_] - f.north[pj])
            value = _brm_vecchia_matern(x)
            space[i, j] = space[j, i] = value
            if want_slopes
                # d/dκ, converted by the caller: x·slope(x)/κ.
                slope = x * _brm_vecchia_matern_slope(x) / kappa
                dspace[i, j] = dspace[j, i] = slope
            end
        end
    end
    lags = f.lag_ptr[p]
    nl = f.lag_ptr[p + 1] - lags
    @inbounds for j in 1:nl
        timecor[j, j] = 1.0
        dtime[j, j] = 0.0
        for i in (j + 1):nl
            gap = abs(f.lag[lags + i - 1] - f.lag[lags + j - 1])
            value = exp(-gap / temporal_length)
            timecor[i, j] = timecor[j, i] = value
            want_slopes && (dtime[i, j] = dtime[j, i] = value * gap / temporal_length^2)
        end
    end
    first_item = f.item_ptr[p]
    m = f.item_ptr[p + 1] - first_item
    A, b = work.A, work.b
    @inbounds for j in 1:m
        uj, lj = f.item_position[first_item + j - 1], f.item_lag[first_item + j - 1]
        for i in j:m
            ui, li = f.item_position[first_item + i - 1], f.item_lag[first_item + i - 1]
            A[i, j] = space[ui, uj] * timecor[li, lj]
        end
        b[j] = space[1, uj] * timecor[1, lj]
    end
    # Lower Cholesky in place, then the forward and back solves.
    @inbounds for j in 1:m
        d = A[j, j]
        for k in 1:(j - 1)
            d -= A[j, k]^2
        end
        d > 0 && isfinite(d) || throw(DomainError(d,
            "gmrf conditional correlation is not positive definite (spatial length " *
            "$(sqrt(2) / kappa), temporal length $temporal_length)"))
        d = sqrt(d)
        A[j, j] = d
        for i in (j + 1):m
            value = A[i, j]
            for k in 1:(j - 1)
                value -= A[i, k] * A[j, k]
            end
            A[i, j] = value / d
        end
    end
    variance = 1.0
    @inbounds for i in 1:m
        value = b[i]
        for k in 1:(i - 1)
            value -= A[i, k] * b[k]
        end
        b[i] = value / A[i, i]
        variance -= b[i]^2
    end
    variance > 0 && isfinite(variance) || throw(DomainError(variance,
        "gmrf conditional variance is not positive (spatial length " *
        "$(sqrt(2) / kappa), temporal length $temporal_length)"))
    @inbounds for i in m:-1:1
        value = b[i]
        for k in (i + 1):m
            value -= A[k, i] * b[k]
        end
        b[i] = value / A[i, i]
    end
    (; m, variance)
end

# Solves the conditioning system for `x` in place with the factor in `work.A`.
function _brm_vecchia_cholesky_solve!(x, A, m)
    @inbounds for i in 1:m
        value = x[i]
        for k in 1:(i - 1)
            value -= A[i, k] * x[k]
        end
        x[i] = value / A[i, i]
    end
    @inbounds for i in m:-1:1
        value = x[i]
        for k in (i + 1):m
            value -= A[k, i] * x[k]
        end
        x[i] = value / A[i, i]
    end
    x
end

# Channel whitening `z[:, s] = L \ x[:, s]`, one small solve per state.
function _brm_vecchia_whiten(field, L, f::_BRMGMRFFootprint)
    C, n = f.nchannels, length(f.position)
    length(field) == f.ncoordinates || throw(DimensionMismatch(
        "gmrf footprint field has $(length(field)) coordinates, expected $(f.ncoordinates)"))
    size(L) == (C, C) || throw(DimensionMismatch("gmrf channel factor must be $C × $C"))
    z = Matrix{Float64}(undef, C, n)
    @inbounds for s in 1:n, c in 1:C
        value = field[c + C * (s - 1)]
        for k in 1:(c - 1)
            value -= z[k, s] * L[c, k]
        end
        z[c, s] = value / L[c, c]
    end
    z
end

"""
    _brm_vecchia_logdensity(field, spatial_length, temporal_length, L, footprint)

The normalized joint log density of the standardized field. `field` holds
`nchannels` coordinates per state, channel first.
"""
function _brm_vecchia_logdensity(field, spatial_length, temporal_length, L, f::_BRMGMRFFootprint)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    C, n = f.nchannels, length(f.position)
    z = _brm_vecchia_whiten(field, L, f)
    work = _BRMVecchiaWork(f)
    kappa = sqrt(2.0) / spatial_length
    value = 0.0
    for p in 1:(length(f.pattern_ptr) - 1)
        solved = _brm_vecchia_solve!(work, f, p, kappa, temporal_length, false)
        m, F = solved.m, solved.variance
        b = work.b
        q = 0.0
        @inbounds for slot in f.pattern_ptr[p]:(f.pattern_ptr[p + 1] - 1)
            s = f.pattern_states[slot]
            first_neighbor = f.neighbor_ptr[s]
            for c in 1:C
                r = z[c, s]
                for a in 1:m
                    r -= b[a] * z[c, f.neighbor[first_neighbor + a - 1]]
                end
                q += r^2
            end
        end
        count = f.pattern_ptr[p + 1] - f.pattern_ptr[p]
        value -= (count * C * log(F) + q / F) / 2
    end
    logL = 0.0
    @inbounds for c in 1:C
        logL += log(L[c, c])
    end
    value - n * logL - n * C * log(2pi) / 2
end

"""
    _brm_vecchia_gradient(field, spatial_length, temporal_length, L, footprint)

The log density and its reverse derivative with respect to the field, both
lengths and the channel factor, computed per pattern from the rows' sufficient
statistics: one conditional solve and one contraction for each distinct
geometry, one linear pass over the states.
"""
function _brm_vecchia_gradient(field, spatial_length, temporal_length, L, f::_BRMGMRFFootprint)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    C, n = f.nchannels, length(f.position)
    z = _brm_vecchia_whiten(field, L, f)
    gz = zeros(C, n)
    work = _BRMVecchiaWork(f)
    kappa = sqrt(2.0) / spatial_length
    dkappa = -kappa / spatial_length
    value = 0.0
    dkappa_total = 0.0
    dtemporal = 0.0
    v, w = work.v, work.w
    for p in 1:(length(f.pattern_ptr) - 1)
        solved = _brm_vecchia_solve!(work, f, p, kappa, temporal_length, true)
        m, F = solved.m, solved.variance
        b, A = work.b, work.A
        fill!(view(v, 1:m), 0.0)
        q = 0.0
        @inbounds for slot in f.pattern_ptr[p]:(f.pattern_ptr[p + 1] - 1)
            s = f.pattern_states[slot]
            first_neighbor = f.neighbor_ptr[s]
            for c in 1:C
                r = z[c, s]
                for a in 1:m
                    r -= b[a] * z[c, f.neighbor[first_neighbor + a - 1]]
                end
                q += r^2
                g = r / F
                gz[c, s] -= g
                for a in 1:m
                    neighbor = f.neighbor[first_neighbor + a - 1]
                    gz[c, neighbor] += b[a] * g
                    v[a] += r * z[c, neighbor]
                end
            end
        end
        count = (f.pattern_ptr[p + 1] - f.pattern_ptr[p]) * C
        value -= (count * log(F) + q / F) / 2
        m == 0 && continue
        copyto!(w, 1, v, 1, m)
        _brm_vecchia_cholesky_solve!(w, A, m)
        scale = (q / F - count) / (2F)
        first_item = f.item_ptr[p]
        space, dspace, timecor, dtime = work.space, work.dspace, work.time, work.dtime
        @inbounds for a in 1:m
            ua, la = f.item_position[first_item + a - 1], f.item_lag[first_item + a - 1]
            weight = w[a] / F - 2scale * b[a]
            dkappa_total += weight * dspace[1, ua] * timecor[1, la]
            dtemporal += weight * space[1, ua] * dtime[1, la]
            for e in 1:m
                ue, le = f.item_position[first_item + e - 1], f.item_lag[first_item + e - 1]
                weight = scale * b[a] * b[e] - w[a] * b[e] / F
                dkappa_total += weight * dspace[ua, ue] * timecor[la, le]
                dtemporal += weight * space[ua, ue] * dtime[la, le]
            end
        end
    end
    logL = 0.0
    @inbounds for c in 1:C
        logL += log(L[c, c])
    end
    value -= n * logL + n * C * log(2pi) / 2
    # Back from whitened coordinates: x = L z, and z moves with L.
    field_gradient = Vector{Float64}(undef, length(field))
    L_gradient = zeros(C, C)
    gx = zeros(C)
    @inbounds for s in 1:n
        for c in C:-1:1
            g = gz[c, s]
            for k in (c + 1):C
                g -= gx[k] * L[k, c]
            end
            gx[c] = g / L[c, c]
            field_gradient[c + C * (s - 1)] = gx[c]
        end
        for c in 1:C, k in 1:c
            L_gradient[c, k] -= gx[c] * z[k, s]
        end
    end
    @inbounds for c in 1:C
        L_gradient[c, c] -= n / L[c, c]
    end
    (; value, field=field_gradient, spatial_length=dkappa_total * dkappa,
       temporal_length=dtemporal, L=L_gradient)
end

# A column of the implied covariance of the standardized scalar field,
# `(I - B)⁻¹ F (I - B)⁻ᵀ e_state`. One pass over the conditional entries, so
# covariance checks at actual supports never form a dense matrix.
function _brm_vecchia_factors(f::_BRMGMRFFootprint, spatial_length, temporal_length)
    n = length(f.position)
    work = _BRMVecchiaWork(f)
    kappa = sqrt(2.0) / spatial_length
    B = Vector{Float64}(undef, length(f.neighbor))
    F = Vector{Float64}(undef, n)
    for p in 1:(length(f.pattern_ptr) - 1)
        solved = _brm_vecchia_solve!(work, f, p, kappa, temporal_length, false)
        for slot in f.pattern_ptr[p]:(f.pattern_ptr[p + 1] - 1)
            s = f.pattern_states[slot]
            F[s] = solved.variance
            copyto!(B, f.neighbor_ptr[s], work.b, 1, solved.m)
        end
    end
    (; B, F)
end

function _brm_vecchia_covariance_column!(out, adjoint_work, factors, f::_BRMGMRFFootprint, state)
    n = length(f.position)
    fill!(adjoint_work, 0.0)
    adjoint_work[state] = 1.0
    B, F = factors.B, factors.F
    # (I - B)⁻ᵀ e, restricted to the state's ancestors by the zero pattern.
    @inbounds for s in state:-1:1
        value = adjoint_work[s]
        value == 0 && continue
        for q in f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1)
            adjoint_work[f.neighbor[q]] += B[q] * value
        end
    end
    @inbounds for s in 1:n
        value = adjoint_work[s] * F[s]
        for q in f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1)
            value += B[q] * out[f.neighbor[q]]
        end
        out[s] = value
    end
    out
end
