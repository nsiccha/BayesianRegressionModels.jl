# Likelihood cells that read no scientific subject's coordinate. The scientific
# unit map counts each once in unit 1 (`_brm_nlme_connected_unit_map`), which no
# full-data estimator can tell apart from any other placement. A minibatch
# target can: unit 1 would pay for all of them, with its factor holding every
# ordinary coordinate only they read, however small the batch.
#
# So in connected SVI they get inference units of their own, appended after the
# scientific units, whose order and labels stay as they are. Two such cells
# share a unit when they read a common linkable coordinate, transitively.
# Nothing here changes the density: each cell is still counted once, and the
# ownership closure still decides what is local.

# A coordinate can join cells only if estimated, not a subject's, not requested
# global, and read by no subject-attached cell. A coordinate those cells also
# read has readers in two units, so it is global and joins nothing. Further
# coordinate classes that must not join cells clear their entries in this mask.
function _brm_svi_linkable_coordinates(coordinate_subject, supports, subjectless,
        estimated, shared)
    linkable = [estimated[j] && coordinate_subject[j] == 0 && !(j in shared)
        for j in eachindex(coordinate_subject)]
    for (i, support) in enumerate(supports)
        subjectless[i] && continue
        for j in support
            linkable[j] = false
        end
    end
    linkable
end

# `mapping` with every subjectless cell moved to an appended unit. Cost is linear
# in the supports and blocks. Blocks move together: one unlinkable coordinate
# holds back its whole block, because the ownership closure treats it as one
# node. Components are numbered by their first cell in inventory order. Cells
# that read no linkable coordinate have no local factor to share, so any grouping
# leaves the variational family unchanged. They are grouped into about √m units
# of about √m cells each, which bounds both the unit count and the unit's cost.
function _brm_svi_split_subjectless_cells(mapping, supports, subjectless, blocks, linkable)
    n = length(linkable)
    cells = findall(subjectless)
    isempty(cells) && return merge(mapping, (; linked=falses(n)))
    components = IntDisjointSets(n + length(cells))
    for block in blocks
        isempty(block) && continue
        for j in block
            union!(components, first(block), j)
        end
    end
    blocked = falses(n + length(cells))
    for j in 1:n
        linkable[j] || (blocked[find_root(components, j)] = true)
    end
    linked = BitVector(!blocked[find_root(components, j)] for j in 1:n)
    readers = falses(length(cells))
    for (k, cell) in enumerate(cells), j in supports[cell]
        linked[j] || continue
        readers[k] = true
        union!(components, n + k, j)
    end
    factor_unit = copy(mapping.factor_unit)
    unit = length(mapping.members)
    numbered = Dict{Int,Int}()
    for (k, cell) in enumerate(cells)
        readers[k] || continue
        root = find_root(components, n + k)
        factor_unit[cell] = get!(() -> (unit += 1), numbered, root)
    end
    alone = cells[.!readers]
    if !isempty(alone)
        per_unit = ceil(Int, sqrt(length(alone)))
        for (k, cell) in enumerate(alone)
            factor_unit[cell] = unit + 1 + (k - 1) ÷ per_unit
        end
        unit += ceil(Int, length(alone) / per_unit)
    end
    members = vcat(mapping.members, [Int[] for _ in length(mapping.members)+1:unit])
    unit_factors = [Int[] for _ in members]
    for (factor, owner) in enumerate(factor_unit)
        push!(unit_factors[owner], factor)
    end
    (; mapping.subject_unit, members, factor_unit, unit_factors, linked)
end

function _brm_svi_cell_unit_map(mapping, coordinate_subject, supports, blocks, estimated, shared)
    subjectless = [all(j -> coordinate_subject[j] == 0, support) for support in supports]
    linkable = _brm_svi_linkable_coordinates(coordinate_subject, supports, subjectless,
        estimated, shared)
    _brm_svi_split_subjectless_cells(mapping, supports, subjectless, blocks, linkable)
end

# A cell unit is named by the first coordinate it owns, which is what streaming
# matches across windows. A unit without local coordinates carries nothing, so its
# position among the cell units is enough. The label holds one element, so no two
# units share a member under the streaming membership check.
_brm_svi_cell_unit_level(view, block, position) =
    ((:cells, isempty(block) ? position : view.coordinates[first(block)]),)
_brm_svi_is_cell_unit_level(level) = level isa Tuple && length(level) == 1 &&
    first(level) isa Tuple && length(first(level)) == 2 && first(first(level)) === :cells

# The supports that form cell units can show less than a unit's partition then
# reads: a response reading a computed vector whole is partitioned reading every
# row's entries (ReactiveKernels snag whole-vector-res-64373f39), while the same
# read by a data index is not. The refusal speaks of subjects, so name the cell
# units and what avoids them.
function _brm_svi_cell_unit_refusal(error, levels)
    cell = findall(_brm_svi_is_cell_unit_level, levels)
    units = length(cell) == 1 ? "unit $(only(cell))" : "units $(first(cell))-$(last(cell))"
    "brm_svi_fit: cell_units=:components gave likelihood cells that read no subject's " *
    "effect $units of their own. If the partition refused below is one of those, its " *
    "cells read another unit's coordinates beyond their structural supports, as a " *
    "response reading a computed vector whole (`y ~ Normal(mu, sigma)`) does. Read it " *
    "by a data index (`Normal(mu[idx], sigma[idx])`), or pass cell_units=:first_unit " *
    "to keep these cells in the first unit, as earlier versions did. " *
    sprint(showerror, error)
end

_brm_svi_cell_units_option(cell_units, connected) = begin
    cell_units === nothing && return connected ? :components : :first_unit
    cell_units in (:components, :first_unit) || throw(ArgumentError(
        "brm_svi_fit: cell_units must be :components or :first_unit, got $(repr(cell_units))"))
    connected || cell_units === :first_unit || throw(ArgumentError(
        "brm_svi_fit: cell_units=:components requires unit_partition=:connected"))
    cell_units
end
