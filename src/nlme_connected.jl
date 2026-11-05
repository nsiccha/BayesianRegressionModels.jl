using DataStructures: IntDisjointSets, find_root

"""
    _brm_nlme_connected_unit_map(nsubjects, coordinate_subject, factor_supports)

Build stable inference units from structural packed-coordinate supports.
`coordinate_subject[c] == 0` identifies shared/population coordinates. Each
support identifies one indivisible likelihood cell (possibly vector-valued),
and is an index collection or RK's `:whole` sentinel. No numerical derivatives
are used. Subject order determines component order; factor order is retained.

This is preparation only: callers must retain and validate the individual
supports with the strict partition API, and keep authored Gaussian priors with
their original coordinates. A factor with no local effects counts once in unit
1, as in BRM's existing attribution. That assignment never links subjects.
"""
function _brm_nlme_connected_unit_map(nsubjects::Int, coordinate_subject, factor_supports)
    nsubjects > 0 || throw(ArgumentError("at least one subject is required"))
    all(s -> 0 <= s <= nsubjects, coordinate_subject) ||
        throw(ArgumentError("coordinate owner outside 0:nsubjects"))
    components = IntDisjointSets(nsubjects)
    factor_subject = Int[]
    whole_subject = nothing
    for support in factor_supports
        # Whole supports share the same coordinate inventory. Reuse their
        # union instead of scanning the entire inventory for every such cell.
        if support === :whole && whole_subject !== nothing
            push!(factor_subject, whole_subject)
            continue
        end
        first_subject = 0
        for c in (support === :whole ? eachindex(coordinate_subject) : support)
            subject = coordinate_subject[c]
            subject == 0 && continue
            if first_subject == 0
                first_subject = subject
            else
                union!(components, first_subject, subject)
            end
        end
        support === :whole && (whole_subject = first_subject)
        push!(factor_subject, first_subject)
    end
    root_unit = zeros(Int, nsubjects)
    subject_unit = zeros(Int, nsubjects)
    members = Vector{Int}[]
    for subject in 1:nsubjects
        root = find_root(components, subject)
        unit = root_unit[root]
        if unit == 0
            push!(members, Int[])
            unit = root_unit[root] = length(members)
        end
        subject_unit[subject] = unit
        push!(members[unit], subject)
    end
    factor_unit = [subject == 0 ? 1 : subject_unit[subject] for subject in factor_subject]
    unit_factors = [Int[] for _ in members]
    for (factor, unit) in enumerate(factor_unit)
        push!(unit_factors[unit], factor)
    end
    (; subject_unit, members, factor_unit, unit_factors)
end

"""
Retain scientific subjects and covariance-family instances while joining their
compact blocks. No family is deduplicated: repeated family IDs represent
independent draws with the same covariance parameters. Ranges address each
original subject inside its new unit, including empty blocks.
"""
function _brm_nlme_joined_layout(mapping, levels, coordinates, families, margins)
    n = length(levels)
    length(mapping.subject_unit) == length(coordinates) == length(families) ==
        length(margins) == n || throw(DimensionMismatch("subject metadata lengths differ"))
    unit_coordinates = [Int[] for _ in mapping.members]
    unit_families = [Int[] for _ in mapping.members]
    unit_margins = [NamedTuple[] for _ in mapping.members]
    subject_ranges = Vector{UnitRange{Int}}(undef, n)
    for (unit, subjects) in enumerate(mapping.members), subject in subjects
        length(coordinates[subject]) == length(margins[subject]) ||
            throw(DimensionMismatch("coordinate/margin lengths differ for subject $subject"))
        first_coordinate = length(unit_coordinates[unit]) + 1
        append!(unit_coordinates[unit], coordinates[subject])
        append!(unit_families[unit], families[subject])
        append!(unit_margins[unit], margins[subject])
        subject_ranges[subject] = first_coordinate:length(unit_coordinates[unit])
    end
    (; subject_levels=copy(levels), subject_unit=copy(mapping.subject_unit),
        unit_members=deepcopy(mapping.members), subject_ranges,
        unit_coordinates, unit_families, unit_margins)
end

# Inventory likelihood cells before positional attribution or strict unit
# validation. The producer's structural footprint includes zero coefficients
# and retained branches. Gaussian subject priors remain independent family
# instances and are never used to merge or deduplicate those instances.
function _brm_nlme_connected_view(backend, view)
    unit = _brm_nlme_unit_point(backend, view)
    fields = _rk_pointwise_values(_rk_pointwise_query(backend), unit)
    partitioned = _rk_nlme_partitioned(backend, view)
    supports = Dict{Symbol,Vector{Union{Symbol,Vector{Int}}}}()
    for (name, values) in pairs(fields)
        query = _rk_nlme_cell_query(backend, (name,), partitioned, unit)
        footprint = _rk_nlme_cell_footprint_query(query)
        supports[name] = Union{Symbol,Vector{Int}}[
            let read = _rk_nlme_raw_cell_footprint(footprint, unit, i)
                read === :whole ? :whole : copy(read)
            end for i in eachindex(values)]
    end
    mapping = _brm_nlme_connected_unit_map(length(view.levels), view.subject,
        Iterators.flatten(supports[name] for name in keys(fields)))
    layout = _brm_nlme_joined_layout(mapping, view.levels, view.subject_coordinates,
        view.subject_families, view.subject_margins)
    owners = [i == 0 ? 0 : mapping.subject_unit[i] for i in view.subject]
    merged = any(members -> length(members) > 1, mapping.members)
    levels = Any[Tuple(view.levels[members]) for members in mapping.members]
    rows = [reduce(vcat, view.rows[members]; init=Int[]) for members in mapping.members]
    BRMNLMEView(view.group, levels, view.coordinates, view.role, owners,
        layout.unit_coordinates, view.block_margins,
        merged ? NamedTuple[] : view.mu_references, rows, view.family_sizes,
        layout.unit_families, layout.unit_margins, view.nested_groups,
        view.shared_groups, view.families, view.ragged || merged, :connected,
        layout.subject_levels, layout.subject_unit, layout.unit_members,
        layout.subject_ranges, copy(view.subject), supports)
end

function _brm_nlme_connected_attribution(view, names)
    attribution = Pair{Symbol,Vector{Int}}[]
    for name in names
        owners = map(view.factor_supports[name]) do support
            # A whole support has already joined all local owners into unit 1.
            support === :whole && return 1
            coordinates = support
            owner = 0
            for c in coordinates
                i = view.subject[c]
                i == 0 && continue
                owner == 0 && (owner = i)
                owner == i || error("connected-unit support spans components")
            end
            # A factor depending only on globals is counted exactly once.
            owner == 0 ? 1 : owner
        end
        push!(attribution, name => owners)
    end
    attribution, Symbol[]
end

# Unit positions are the sampler/estimator API. Named results restore the
# original subject labels and slices, including when only some units ran.
function _brm_nlme_original_effects(view, units, effects)
    positions = Dict(i => j for (j, i) in enumerate(units))
    subjects = [s for s in eachindex(view.subject_levels)
        if haskey(positions, view.subject_unit[s])]
    values = [copy(_brm_nlme_subject_eta(effects, positions[view.subject_unit[s]])[
        view.subject_ranges[s]]) for s in subjects]
    (; subjects, values)
end
