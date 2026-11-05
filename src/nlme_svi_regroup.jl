# SVI inference units may split a connected NLME component after separator
# promotion. The scientific view stays intact: this metadata carries its own
# unit labels, subject membership, observation attribution and packed owners.
function _brm_svi_factor_inventory(view, ncoordinates)
    names = sort!(collect(keys(view.factor_supports)))
    supports = Vector{Int}[]
    sources = Tuple{Symbol,Int}[]
    whole = nothing
    for name in names, (i, support) in enumerate(view.factor_supports[name])
        if support === :whole
            whole === nothing && (whole = collect(1:ncoordinates))
            push!(supports, whole)
        else
            push!(supports, support)
        end
        push!(sources, (name, i))
    end
    (; names, supports, sources)
end

# RK's partition declaration assigns an entry's first axis. Coordinates in
# the same axis slice must therefore move together when choosing a separator.
# This is an inspectable approximation constraint, not a density alteration.
function _brm_svi_partition_blocks(backend)
    blocks = Vector{Int}[]
    for entry in backend.model.layout.entries
        n = isempty(entry.dims) ? entry.size : first(entry.dims)
        n > 0 || continue
        for i in 1:n
            push!(blocks, collect((entry.offset+i-1):n:(entry.offset+entry.size-1)))
        end
    end
    blocks
end

function _brm_svi_atomic_blocks(n, owned, axis_blocks, estimated)
    components = IntDisjointSets(n)
    for block in Iterators.flatten((owned, axis_blocks))
        isempty(block) && continue
        for j in block
            union!(components, first(block), j)
        end
    end
    groups = Dict{Int,Vector{Int}}()
    for j in 1:n
        push!(get!(groups, find_root(components, j), Int[]), j)
    end
    blocks = [block for block in values(groups) if all(j -> estimated[j], block)]
    sort!(blocks; by=first)
end

# Each pass closes ownership before its result is usable. At the bounded
# refinement limit, units may be coarser than necessary, but no factor crosses
# local owners: the exact prior/likelihood split remains valid.
function _brm_svi_regroup_layout(view, inventory, factors, estimated, axis_blocks,
        promoted; refinement_passes=4)
    n = length(estimated)
    shared = Set(promoted)
    owned = vcat(factors.owned, axis_blocks)
    support = vcat(factors.support, axis_blocks)
    owner = zeros(Int, n)
    mapping = nothing
    footprints = Vector{Int}[]
    for pass in 1:refinement_passes
        coordinate_subject = [j in shared ? 0 : view.coordinate_subject[j] for j in 1:n]
        mapping = _brm_nlme_connected_unit_map(length(view.subject_levels),
            coordinate_subject, inventory.supports)
        seeds = [Int[] for _ in mapping.members]
        for j in 1:n
            s = coordinate_subject[j]
            s == 0 || push!(seeds[mapping.subject_unit[s]], j)
        end
        reads = [Set{Int}() for _ in mapping.members]
        for (i, footprint) in enumerate(inventory.supports)
            union!(reads[mapping.factor_unit[i]], footprint)
        end
        footprints = [sort!(collect(r)) for r in reads]
        owner = _brm_svi_coordinate_owners(footprints, seeds, owned, support,
            estimated; global_coordinates=collect(shared))
        # Only a scientific effect promotion can split the component graph.
        changed = false
        for j in 1:n
            if estimated[j] && owner[j] == 0 && view.coordinate_subject[j] != 0 && !(j in shared)
                push!(shared, j)
                changed = true
            end
        end
        changed || break
    end
    units = [Int[] for _ in mapping.members]
    globals = Int[]
    for j in 1:n
        estimated[j] || continue
        owner[j] == 0 ? push!(globals, j) : push!(units[owner[j]], j)
    end
    # Structural read work counts every observation cell, not only the union
    # of its unit's parameter reads. Vector-valued cells remain indivisible.
    work = Float64.(length.(units))
    for (i, footprint) in enumerate(inventory.supports)
        work[mapping.factor_unit[i]] += 1 + length(footprint)
    end
    for reads in factors.support
        i = 0
        for j in reads
            owner[j] == 0 && continue
            i = owner[j]
            break
        end
        i == 0 || (work[i] += 1 + length(reads))
    end
    levels = Any[Tuple(view.subject_levels[members]) for members in mapping.members]
    attribution = Pair{Symbol,Vector{Int}}[
        name => zeros(Int, length(view.factor_supports[name])) for name in inventory.names]
    byname = Dict(attribution)
    for ((name, cell), i) in zip(inventory.sources, mapping.factor_unit)
        byname[name][cell] = i
    end
    (; owner, globals, units, levels, work, attribution, footprints,
        subject_unit=mapping.subject_unit, unit_members=mapping.members)
end
