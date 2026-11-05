# Inference ownership is separate from the scientific NLME coordinate roles.
# Likelihood footprints seed owners; authored atomic blocks and their directed
# dependencies close that assignment. A label is unseen (0), one unit (> 0),
# or shared (-1). Each node changes at most twice, so closure costs the total
# structural support, not units times the packed model size.
using DataStructures: IntDisjointSets, find_root

function _brm_svi_coordinate_owners(footprints, subject_coordinates, owned, support,
        estimated::AbstractVector{Bool}; global_coordinates=Int[])
    n = length(estimated)
    components = IntDisjointSets(n)
    for block in owned
        isempty(block) && continue
        for j in block
            union!(components, first(block), j)
        end
    end
    roots = [find_root(components, j) for j in 1:n]
    labels = zeros(Int, n)
    queue = Int[]
    function mark(j, label)
        r = roots[j]
        old = labels[r]
        new = old == 0 ? label : old == label ? old : -1
        if old != new
            labels[r] = new
            push!(queue, r)
        end
        nothing
    end
    # Scientific subject draws retain their unit even if observations do not
    # read them. Ordinary coordinates acquire a unit only through structure.
    for (i, block) in enumerate(subject_coordinates), j in block
        mark(j, i)
    end
    for (i, footprint) in enumerate(footprints), j in footprint
        mark(j, i)
    end
    for j in eachindex(estimated)
        estimated[j] || mark(j, -1)
    end
    for j in global_coordinates
        mark(j, -1)
    end
    # A child's unit is also a reader of its prior's hyperparents. The reverse
    # direction is deliberately absent: a shared scale does not make every
    # independent child shared.
    dependencies = [Int[] for _ in 1:n]
    for (block, reads) in zip(owned, support)
        isempty(block) && continue
        r = roots[first(block)]
        for j in reads
            roots[j] == r || push!(dependencies[r], roots[j])
        end
    end
    head = 1
    function propagate!()
        while head <= length(queue)
            r = queue[head]
            head += 1
            for p in dependencies[r]
                mark(p, labels[r])
            end
        end
        nothing
    end
    propagate!()
    # Prior-only blocks have no likelihood unit. Keep them global, and keep
    # their hyperparents global too rather than inventing a unit assignment.
    for j in eachindex(estimated)
        labels[roots[j]] == 0 && mark(j, -1)
    end
    propagate!()
    # A derived density owns nothing. If its parents cross units, promote
    # those blocks together; otherwise the one unit can carry the event.
    for (block, reads) in zip(owned, support)
        isempty(block) || continue
        unit = 0
        crossing = false
        for j in reads
            label = labels[roots[j]]
            label <= 0 && continue
            unit == 0 ? (unit = label) : (crossing |= unit != label)
        end
        crossing || continue
        for j in reads
            mark(j, -1)
        end
    end
    propagate!()
    [estimated[j] ? max(0, labels[roots[j]]) : 0 for j in eachindex(estimated)]
end

function _brm_svi_global_coordinates(m, coordinates)
    coordinates === nothing && return Int[]
    names = Dict(name => i for (i, name) in enumerate(m.view.coordinates))
    result = map(collect(coordinates)) do coordinate
        j = coordinate isa Integer ? Int(coordinate) : get(names, Symbol(coordinate), 0)
        1 <= j <= length(m.unit) || throw(ArgumentError(
            "brm_svi_fit: unknown global coordinate $(repr(coordinate))"))
        j
    end
    sort!(unique!(result))
end

function _rk_svi_factors end

_brm_svi_previous_ownership!(estimated, seeds, promoted, m, ::Nothing) = Int[]

function _brm_svi_ownership(m; global_coordinates=nothing, previous=nothing)
    factors = _rk_svi_factors(m.backend, m.unit)
    estimated = falses(length(m.unit))
    estimated[_brm_svi_globals(m)] .= true
    for block in m.view.subject_coordinates
        estimated[block] .= true
    end
    promoted = _brm_svi_global_coordinates(m, global_coordinates)
    all(j -> estimated[j], promoted) || throw(ArgumentError(
        "brm_svi_fit: global_coordinates cannot promote fixed coordinates"))
    seeds = copy.(m.view.subject_coordinates)
    absent = _brm_svi_previous_ownership!(estimated, seeds, promoted, m, previous)
    owner = _brm_svi_coordinate_owners(m.cell_gradient.footprints,
        seeds, factors.owned, factors.support, estimated;
        global_coordinates=promoted)
    absent_set = Set(absent)
    for footprint in m.cell_gradient.footprints
        clash = findfirst(j -> j in absent_set, footprint)
        clash === nothing || throw(ArgumentError("brm_svi_fit: an observation reads " *
            "absent carried unit coordinate $(m.view.coordinates[footprint[clash]])"))
    end
    for (block, reads) in zip(factors.owned, factors.support)
        any(j -> owner[j] > 0, block) || continue
        clash = findfirst(j -> j in absent_set, reads)
        clash === nothing || throw(ArgumentError("brm_svi_fit: a current unit's " *
            "prior reads absent carried coordinate $(m.view.coordinates[reads[clash]])"))
    end
    globals = Int[]
    units = [[j for j in block if owner[j] == i]
        for (i, block) in enumerate(m.view.subject_coordinates)]
    subject = falses(length(owner))
    for block in m.view.subject_coordinates
        subject[block] .= true
    end
    for j in eachindex(owner)
        estimated[j] || continue
        owner[j] == 0 ? push!(globals, j) : subject[j] || push!(units[owner[j]], j)
    end
    (; owner, globals, units, factors, promoted, absent)
end
