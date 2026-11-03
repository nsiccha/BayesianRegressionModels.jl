# A balanced bounding-box tree keeps row preparation local as the mesh grows.
# Only geometry is indexed. Predictions use the returned three vertex indices
# and barycentric weights, never a dense observation-by-vertex matrix.
struct _BRMGMRFLocator
    bounds::Vector{NTuple{4,Float64}}
    children::Vector{Tuple{Int,Int}}
    ranges::Vector{UnitRange{Int}}
    order::Vector{Int}
    cell_bounds::Vector{NTuple{4,Float64}}
end

function _brm_gmrf_locator(mesh::_BRMGMRFMesh)
    xy, cells = mesh.vertices, mesh.cells
    boxes = map(axes(cells, 1)) do c
        ids = view(cells, c, :)
        (minimum(xy[i, 1] for i in ids), maximum(xy[i, 1] for i in ids),
         minimum(xy[i, 2] for i in ids), maximum(xy[i, 2] for i in ids))
    end
    tree = _BRMGMRFLocator(NTuple{4,Float64}[], Tuple{Int,Int}[],
        UnitRange{Int}[], collect(axes(cells, 1)), boxes)
    _brm_gmrf_locator_node!(tree, 1:length(tree.order))
    tree
end

function _brm_gmrf_locator_node!(tree, range)
    boxes = tree.cell_bounds
    box = (minimum(boxes[tree.order[i]][1] for i in range),
           maximum(boxes[tree.order[i]][2] for i in range),
           minimum(boxes[tree.order[i]][3] for i in range),
           maximum(boxes[tree.order[i]][4] for i in range))
    push!(tree.bounds, box)
    push!(tree.children, (0, 0))
    push!(tree.ranges, range)
    node = length(tree.bounds)
    if length(range) > 8
        axis = box[2] - box[1] >= box[4] - box[3] ? 1 : 3
        sort!(view(tree.order, range); by=c -> (boxes[c][axis] / 2 + boxes[c][axis + 1] / 2, c))
        middle = (first(range) + last(range)) ÷ 2
        left = _brm_gmrf_locator_node!(tree, first(range):middle)
        right = _brm_gmrf_locator_node!(tree, (middle + 1):last(range))
        tree.children[node] = (left, right)
    end
    node
end

function _brm_gmrf_in_box(box, x, y)
    tolerance = 16eps(Float64) * max(1.0, abs(x), abs(y), maximum(abs, box))
    box[1] - tolerance <= x <= box[2] + tolerance &&
        box[3] - tolerance <= y <= box[4] + tolerance
end

function _brm_gmrf_barycentric(mesh, c, x, y)
    i, j, k = mesh.cells[c, :]
    xy = mesh.vertices
    ax, ay = xy[i, 1], xy[i, 2]
    bx, by = xy[j, 1] - ax, xy[j, 2] - ay
    cx, cy = xy[k, 1] - ax, xy[k, 2] - ay
    dx, dy = x - ax, y - ay
    determinant = bx * cy - cx * by
    v = (dx * cy - cx * dy) / determinant
    w = (bx * dy - dx * by) / determinant
    (1 - v - w, v, w)
end

function _brm_gmrf_interpolation(mesh::_BRMGMRFMesh, x, y,
        tree::_BRMGMRFLocator=_brm_gmrf_locator(mesh))
    length(x) == length(y) || throw(DimensionMismatch("gmrf spatial columns must have equal lengths"))
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("gmrf spatial coordinates must be finite"))
    indices = Matrix{Int}(undef, length(x), 3)
    weights = Matrix{Float64}(undef, length(x), 3)
    stack = Int[]
    tolerance = 64eps(Float64)
    for row in eachindex(x, y)
        push!(stack, 1)
        chosen = 0
        interior = 0
        chosen_weights = (0.0, 0.0, 0.0)
        while !isempty(stack)
            node = pop!(stack)
            _brm_gmrf_in_box(tree.bounds[node], x[row], y[row]) || continue
            left, right = tree.children[node]
            if left != 0
                push!(stack, right, left)
                continue
            end
            for p in tree.ranges[node]
                c = tree.order[p]
                _brm_gmrf_in_box(tree.cell_bounds[c], x[row], y[row]) || continue
                b = _brm_gmrf_barycentric(mesh, c, x[row], y[row])
                minimum(b) >= -tolerance || continue
                if minimum(b) > tolerance
                    interior == 0 || throw(ArgumentError(
                        "gmrf spatial row $row lies inside overlapping mesh cells $interior and $c"))
                    interior = c
                end
                # A shared edge belongs to either adjacent triangle. Choosing
                # by cell index makes replay independent of tree traversal.
                if chosen == 0 || c < chosen
                    chosen, chosen_weights = c, b
                end
            end
        end
        chosen != 0 || throw(ArgumentError("gmrf spatial row $row is outside the declared mesh"))
        b = max.(chosen_weights, 0.0)
        total = sum(b)
        for a in 1:3
            indices[row, a] = mesh.cells[chosen, a]
            weights[row, a] = b[a] / total
        end
    end
    (; indices, weights)
end

# The state layout is independent of sparse-factor permutations and row order.
# Keep only the axes: a Cartesian label vector would itself be field-sized.
struct _BRMGMRFStateLayout{C<:AbstractVector,T<:AbstractVector}
    nvertices::Int
    channels::C
    times::T
    ncoordinates::Int
    function _BRMGMRFStateLayout(nvertices::Integer, channels::AbstractVector,
            times::AbstractVector{<:Real})
        nvertices > 0 || throw(ArgumentError("gmrf needs at least one state vertex"))
        !isempty(channels) && !any(ismissing, channels) && allunique(channels) ||
            throw(ArgumentError("gmrf state channels must be nonempty, unique and nonmissing"))
        !isempty(times) && all(isfinite, times) && issorted(times) && allunique(times) ||
            throw(ArgumentError("gmrf state times must be finite and strictly increasing"))
        n = Base.checked_mul(Base.checked_mul(Int(nvertices), length(channels)), length(times))
        cc, tt = copy(channels), copy(times)
        new{typeof(cc),typeof(tt)}(Int(nvertices), cc, tt, n)
    end
end

function _brm_gmrf_state_coordinate(layout::_BRMGMRFStateLayout, index::Integer)
    1 <= index <= layout.ncoordinates || throw(BoundsError(layout, index))
    plane, vertex = divrem(index - 1, layout.nvertices)
    time, channel = divrem(plane, length(layout.channels))
    (; vertex=vertex + 1, channel=layout.channels[channel + 1], time=layout.times[time + 1])
end

# Time and channel identities are frozen by the fitted term. Adding a new
# time requires a conditional state, not interpolating unrelated coefficients.
# This mapping therefore refuses unknown times/channels during draw replay.
function _brm_gmrf_state_indices(vertices, row_times, row_channels, times, channels, nvertices)
    _brm_gmrf_state_indices(vertices, row_times, row_channels,
        _BRMGMRFStateLayout(nvertices, channels, times))
end

function _brm_gmrf_state_indices(vertices, row_times, row_channels, layout::_BRMGMRFStateLayout)
    length(row_times) == length(row_channels) == size(vertices, 1) ||
        throw(DimensionMismatch("gmrf row support lengths disagree"))
    nvertices, times, channels = layout.nvertices, layout.times, layout.channels
    all(v -> 1 <= v <= nvertices, vertices) || throw(ArgumentError(
        "gmrf row reads a vertex outside the fitted mesh"))
    time_index = Dict(t => i for (i, t) in enumerate(times))
    channel_index = Dict(c => i for (i, c) in enumerate(channels))
    out = similar(vertices)
    for row in axes(vertices, 1)
        t = get(time_index, row_times[row], 0)
        c = get(channel_index, row_channels[row], 0)
        t != 0 || throw(ArgumentError("gmrf row $row has a time outside the fitted state grid"))
        c != 0 || throw(ArgumentError("gmrf row $row has an unknown channel"))
        offset = nvertices * ((c - 1) + length(channels) * (t - 1))
        for a in axes(vertices, 2)
            out[row, a] = vertices[row, a] + offset
        end
    end
    out
end
