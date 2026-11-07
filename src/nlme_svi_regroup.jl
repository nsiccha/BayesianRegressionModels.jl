# SVI inference units may split a connected NLME component after separator
# promotion. The scientific view stays intact: this metadata carries its own
# unit labels, subject membership, observation attribution and packed owners.
function _brm_svi_factor_inventory(view, ncoordinates; rows=(_, _) -> 1)
    names = sort!(collect(keys(view.factor_supports)))
    supports = Vector{Int}[]
    sources = Tuple{Symbol,Int}[]
    row_counts = Int[]
    whole = nothing
    for name in names, (i, support) in enumerate(view.factor_supports[name])
        if support === :whole
            whole === nothing && (whole = collect(1:ncoordinates))
            push!(supports, whole)
        else
            push!(supports, support)
        end
        push!(sources, (name, i))
        push!(row_counts, rows(name, i))
    end
    (; names, supports, sources, rows=row_counts)
end

# How many rows a cell evaluates: those a plate joins to it, or the length of its
# observed element, so all of a unit's rows count when they form one cell. Its
# footprint cannot say this, because rows reading coordinates other rows also
# read add nothing to it.
_brm_svi_cell_rows(element::AbstractArray) = length(element)
_brm_svi_cell_rows(_) = 1
function _brm_svi_observed_rows(backend, view)
    plan, nrows = backend.plan, sum(length, view.rows; init=0)
    joined = Dict{Symbol,Any}()
    (name, cell) -> begin
        rows = get!(() -> _brm_nlme_cell_rows(plan, name, view.group, nrows), joined, name)
        rows !== nothing && checkbounds(Bool, rows, cell) && return length(rows[cell])
        column = get(plan.columns, name, nothing)
        column isa AbstractVector && checkbounds(Bool, column, cell) ?
            _brm_svi_cell_rows(column[cell]) : 1
    end
end

# A cell's structural work: one for the cell, one per coordinate it reads, and
# `_BRM_SVI_ROW_WORK` for each row beyond its first. An elementwise row counts
# what it always did.
const _BRM_SVI_ROW_WORK = 16
_brm_svi_cell_work(reads, rows) = 1 + reads + _BRM_SVI_ROW_WORK * max(rows - 1, 0)

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
        promoted; refinement_passes=4, cell_units=:first_unit)
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
        cell_units === :components && (mapping = _brm_svi_cell_unit_map(mapping,
            coordinate_subject, inventory.supports, owned, estimated, shared))
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
        # A scientific effect promotion can split the component graph, and so
        # can a global coordinate that joined subjectless cells in this pass.
        joined = hasproperty(mapping, :linked) ? mapping.linked : falses(n)
        changed = false
        for j in 1:n
            if estimated[j] && owner[j] == 0 && !(j in shared) &&
                    (view.coordinate_subject[j] != 0 || joined[j])
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
        # `:first_unit` reproduces earlier versions' work, so their layouts,
        # separator trials and sampling replay unchanged.
        work[mapping.factor_unit[i]] += _brm_svi_cell_work(length(footprint),
            cell_units === :first_unit ? 1 : inventory.rows[i])
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
    scientific = maximum(mapping.subject_unit; init=0)
    levels = Any[i <= scientific ? Tuple(view.subject_levels[members]) :
        _brm_svi_cell_unit_level(view, units[i], i - scientific)
        for (i, members) in enumerate(mapping.members)]
    attribution = Pair{Symbol,Vector{Int}}[
        name => zeros(Int, length(view.factor_supports[name])) for name in inventory.names]
    byname = Dict(attribution)
    for ((name, cell), i) in zip(inventory.sources, mapping.factor_unit)
        byname[name][cell] = i
    end
    (; owner, globals, units, levels, work, attribution, footprints,
        subject_unit=mapping.subject_unit, unit_members=mapping.members)
end

function _brm_svi_connected_ownership(m; global_coordinates=nothing, previous=nothing,
        separator=:auto, max_unit_coordinates=4096, max_unit_work=1_000_000,
        max_promoted_coordinates=128, cell_units=:components, fields=())
    factors = _brm_svi_strip_fields(_rk_svi_factors(m.backend, m.unit), fields)
    estimated = falses(length(m.unit))
    estimated[_brm_svi_globals(m)] .= true
    for block in m.view.subject_coordinates
        estimated[block] .= true
    end
    promoted = _brm_svi_global_coordinates(m, global_coordinates)
    all(j -> estimated[j], promoted) || throw(ArgumentError(
        "brm_svi_fit: global_coordinates cannot promote fixed coordinates"))
    # A carried global cannot become local on another window.
    if previous !== nothing
        append!(promoted, _brm_svi_global_coordinates(m,
            intersect(previous.globals, m.view.coordinates)))
        sort!(unique!(promoted))
    end
    inventory = _brm_svi_factor_inventory(m.view, length(m.unit);
        rows=_brm_svi_observed_rows(m.backend, m.view))
    axis_blocks = _brm_svi_partition_blocks(m.backend)
    blocks = _brm_svi_atomic_blocks(length(m.unit), factors.owned, axis_blocks, estimated)
    incidence = zeros(Int, length(m.unit))
    for support in inventory.supports, j in support
        incidence[j] += 1
    end
    evaluate = requested -> _brm_svi_regroup_layout(m.view, inventory, factors,
        estimated, axis_blocks, requested; cell_units)
    chosen = _brm_svi_separator_plan(evaluate, blocks, incidence;
        global_coordinates=promoted, separator, max_unit_coordinates,
        max_unit_work, max_promoted_coordinates, preserve_globals=previous !== nothing)
    layout = chosen.layout
    # Preserve carried ordinary coordinates that have no current observations.
    # Reclose after applying their saved seeds; membership changes are refused
    # by the streaming initializer before any optimizer update.
    seeds = copy.(layout.units)
    absent = _brm_svi_previous_ownership!(estimated, seeds, promoted, m, previous;
        levels=layout.levels)
    if previous !== nothing
        owner = _brm_svi_coordinate_owners(layout.footprints, seeds,
            vcat(factors.owned, axis_blocks), vcat(factors.support, axis_blocks),
            estimated; global_coordinates=layout.globals)
        units = [Int[] for _ in layout.units]
        globals = Int[]
        for j in eachindex(owner)
            estimated[j] || continue
            owner[j] == 0 ? push!(globals, j) : push!(units[owner[j]], j)
        end
        layout = merge(layout, (; owner, units, globals))
    end
    absent_set = Set(absent)
    for reads in layout.footprints
        any(in(absent_set), reads) && throw(ArgumentError(
            "brm_svi_fit: an observation reads an absent carried coordinate"))
    end
    for (block, reads) in zip(factors.owned, factors.support)
        any(j -> layout.owner[j] > 0, block) || continue
        any(in(absent_set), reads) && throw(ArgumentError(
            "brm_svi_fit: a current prior reads an absent carried coordinate"))
    end
    merge(layout, (; factors, promoted=chosen.global_coordinates, absent,
        separator=merge(chosen.policy, (; cell_units))))
end

# Compile one final partition after all structural trials. The scientific
# BRMNLMEModel remains unchanged; only its SVI target uses these inference IDs.
function _brm_svi_partition(m, ownership)
    view = (; subject=ownership.owner, coordinates=m.view.coordinates,
        subject_coordinates=ownership.units, unit_partition=:connected,
        factor_supports=m.view.factor_supports)
    partitioned = _rk_nlme_partitioned(m.backend, ownership.owner)
    cells = Tuple(map(ownership.attribution) do (name, owners)
        positions = [Int[] for _ in ownership.units]
        for (cell, owner) in enumerate(owners)
            push!(positions[owner], cell)
        end
        (; names=(name,), positions, elementwise=true)
    end)
    partition = _brm_nlme_partition_queries(m.backend, cells, view, m.unit, partitioned)
    (; partition, view)
end

_brm_svi_unit_levels(t) = hasproperty(t.ownership, :levels) ? t.ownership.levels : t.m.view.levels
# The options that rebuild a layout, for resume and streaming alike. `cell_units`
# is recorded only when it departs from the layout every earlier checkpoint has,
# so those checkpoints keep their identity and replay unchanged.
function _brm_svi_policy_options(policy)
    options = (; (key => getproperty(policy, key) for key in
        (:separator, :max_unit_coordinates, :max_unit_work, :max_promoted_coordinates))...)
    cell_units = get(policy, :cell_units, :first_unit)
    cell_units === :first_unit ? options : merge(options, (; cell_units))
end
_brm_svi_separator_options(t) = hasproperty(t.ownership, :separator) ?
    _brm_svi_policy_options(t.ownership.separator) : (; separator=:none)
_brm_svi_requested_globals(t) = hasproperty(t.ownership, :separator) ?
    t.ownership.separator.requested : t.ownership.promoted
