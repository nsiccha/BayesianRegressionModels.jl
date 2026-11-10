# The row partition of a ragged join, written once (snag
# `jax-source-brm-r-f6503407`, decision `1fg2s0z`).
#
# A grouped read `x[rows[i]]` with `rows = groups(group => cells)` reads, in
# each cell, the rows of `x`'s frame whose `group` label is that cell's label.
# BRM's preparation (SBBRMI, the RK design binder) and the RK graph
# (`brm_ragged_rows` and its two variants, emitted by
# `_rk_emit_kernel_partition!`) compute that partition with the definitions
# below. Each is a `@_brm_traceable` definition: this file defines it for
# Julia, and BRM's ReactiveKernels extension defines the same source once more
# under `ReactiveKernels.@traceable`, with untyped formals. Julia keeps calling
# the typed method defined here, and `jax_source` translates the RK program's
# partition from the recorded source into the module's `prepare_data`
# (reactivekernels-use §4j). So the bodies keep to what RK translates on the
# host: positional formals, at least one of them typed; loops and `if`
# statements; `throw` guards whose messages are single string literals; `Dict`
# lookups; type tests written `typeof(x) <: T` or `eltype(x) <: T` (RK has no
# `isa`), which are decided while emitting; values the untaken arm of such a
# test could not compute kept out of reach; and no closures, keyword
# arguments, `break` or early `return`.

# The `@_brm_traceable` definitions, in definition order, for the RK extension.
const _BRM_TRACEABLE_DEFINITIONS = Expr[]

macro _brm_traceable(def)
    formals = _brm_traceable_formals(def)
    any(f -> Meta.isexpr(f, :(::)), formals) || throw(ArgumentError(
        "@_brm_traceable: `$(_brm_traceable_name(def))` needs a typed formal, so \
        that its Julia method stays more specific than the untyped one BRM's RK \
        extension defines from the same source"))
    esc(Expr(:block, def, :(push!(_BRM_TRACEABLE_DEFINITIONS, $(QuoteNode(def)))), nothing))
end

function _brm_traceable_call(def)
    Meta.isexpr(def, (:function, :(=)), 2) || throw(ArgumentError(
        "@_brm_traceable expects a method definition; got `$def`"))
    signature = def.args[1]
    Meta.isexpr(signature, :where) && throw(ArgumentError(
        "@_brm_traceable: `$signature` has a `where` clause, which the untyped \
        definition cannot keep"))
    Meta.isexpr(signature, :call) && signature.args[1] isa Symbol || throw(ArgumentError(
        "@_brm_traceable expects `f(args...)`; got `$signature`"))
    signature
end
_brm_traceable_name(def) = _brm_traceable_call(def).args[1]
function _brm_traceable_formals(def)
    formals = _brm_traceable_call(def).args[2:end]
    for f in formals
        f isa Symbol || Meta.isexpr(f, :(::), 2) && f.args[1] isa Symbol || throw(ArgumentError(
            "@_brm_traceable: `$f` is not a plain positional formal"))
    end
    formals
end
# The definition with its formals untyped: the method BRM's RK extension
# defines under `ReactiveKernels.@traceable`.
_brm_untyped_definition(def) = Expr(def.head,
    Expr(:call, _brm_traceable_name(def),
        (f isa Symbol ? f : f.args[1] for f in _brm_traceable_formals(def))...),
    def.args[2])

# The BRM functions the recorded bodies call, which the RK extension imports
# so that the bodies read the same names there.
function _brm_traceable_callees()
    names = Set{Symbol}()
    walk(ex) = ex isa Symbol ? (isdefined(@__MODULE__, ex) &&
        (startswith(string(ex), "_brm") || startswith(string(ex), "brm_")) &&
        push!(names, ex)) : ex isa Expr ? foreach(walk, ex.args) : nothing
    foreach(walk, _BRM_TRACEABLE_DEFINITIONS)
    sort!(collect(names))
end

# Refusals whose message names a type. Out of line, so the `@_brm_traceable`
# bodies hold no `typeof`: RK translates a call that never returns as a raise.
@noinline _brm_not_a_label_vector(raw, group, prefix) = throw(ErrorException(
    "$prefix: kernel(...) subject grouping column `$group` must be a vector, got \
    $(typeof(raw))."))
@noinline _brm_order_not_a_column(times, order, group, nrows, prefix) = throw(ErrorException(
    "$prefix: kernel(...) `ragged(…, $group; order=$order)`: `$order` must be a column \
    on `$group`'s $nrows rows; got $(typeof(times))."))

# The labels of a grouping column (a categorical column's values,
# `preparation_kernel.jl`).
@_brm_traceable _brm_group_values(raw::AbstractVector) = raw

# The cells' labels: a vector with one non-missing, unique label per row. No
# rows are no cells.
@_brm_traceable function _brm_cell_labels(raw, group::Symbol, prefix::AbstractString)
    if !(typeof(raw) <: AbstractVector)
        _brm_not_a_label_vector(raw, group, prefix)
    end
    values = _brm_group_values(raw)
    unique_labels = true
    seen = Dict{eltype(values),Int}()
    for k in eachindex(values)
        if ismissing(values[k]) || haskey(seen, values[k])
            unique_labels = false
        end
        seen[values[k]] = k
    end
    if !unique_labels
        throw(ErrorException("$prefix: kernel(...) needs pre-grouped per-subject data — \
            `$group` must list one non-missing unique subject per row. Repeated levels \
            indicate long-format data. Got $(values)."))
    end
    values
end

# A column read through a join lies on the join's rows: `x` has `len` entries
# in its data vector `column`, and `group` has `n`.
@_brm_traceable function _brm_check_ragged_axis(x, group::Symbol, column, len::Integer,
                                                n::Integer, prefix::AbstractString)
    if len != n
        throw(ErrorException("$prefix: `$x` read through `$group`: `$x` has $len entries \
            (data vector `$column`) but `$group` has $n. The grouping column must label \
            EVERY entry of `$x`; an entry whose label names no cell is read by no cell."))
    end
    nothing
end
# The same for every column in `axes`, named `names[offset + k]`; `names[1]` is
# `x` and `names[2]` the grouping column. The tuple is iterated, not indexed:
# RK's host code indexes a tuple at literal positions only.
@_brm_traceable function _brm_check_ragged_axes(names::AbstractVector, offset::Integer,
                                                axes::Tuple, n::Integer,
                                                prefix::AbstractString)
    # No axes, decided while emitting (a tuple's length is a literal there): an
    # empty tuple's elements have no type to translate.
    if length(axes) > 0
        k = offset
        for axis in axes
            k += 1
            _brm_check_ragged_axis(names[1], names[2], names[k], length(axis), n, prefix)
        end
    end
    nothing
end

# A row label is never missing.
@_brm_traceable function _brm_check_missing_labels(x, group::Symbol, values::AbstractVector,
                                                   prefix::AbstractString)
    if Missing <: eltype(values)
        for r in eachindex(values)
            if ismissing(values[r])
                throw(ErrorException("$prefix: kernel(...) `ragged($x, $group)`: `$group` \
                    must be a column with no missing labels."))
            end
        end
    end
    nothing
end

# Where each label sits among the cells, the last position where a label
# repeats. Machine-integer labels over a compact range index a dense table from
# the smallest label; any others a hash table keyed by the labels' own element
# type, which compares by `isequal`.
@_brm_traceable _brm_dense_labels(labels::AbstractVector) = !isempty(labels) &&
    Int128(maximum(labels)) - Int128(minimum(labels)) + 1 <= 4 * length(labels) + 64
@_brm_traceable function _brm_dense_label_table(labels::AbstractVector)
    lo = minimum(labels)
    table = zeros(Int, Int(Int128(maximum(labels)) - Int128(lo) + 1))
    for k in eachindex(labels)
        table[Int(Int128(labels[k]) - Int128(lo)) + 1] = k
    end
    table
end
@_brm_traceable function _brm_hashed_label_table(labels::AbstractVector)
    table = Dict{eltype(labels),Int}()
    for k in eachindex(labels)
        table[labels[k]] = k
    end
    table
end
# The position of `v`'s label, 0 when no cell has it: `lo` and `hi` are a
# dense table's smallest and largest label.
@_brm_traceable _brm_label_position(table::Union{AbstractDict,AbstractVector{Int}}, lo, hi,
                                    v) =
    typeof(table) <: AbstractDict ? get(table, v, 0) :
        lo <= v && v <= hi ? table[Int(Int128(v) - Int128(lo)) + 1] : 0

# Each of `ncells` cells' rows, ascending. One pass counts every cell's rows
# and a second fills each cell in place, so the partition allocates its cells
# and nothing per row. Rows usually arrive grouped by label, so a row
# repeating the previous row's label reuses its position.
@_brm_traceable function _brm_partition_rows(values::AbstractVector, ncells::Integer, table,
                                             lo, hi)
    counts = zeros(Int, ncells)
    if !isempty(values)
        previous = values[1]
        at = _brm_label_position(table, lo, hi, previous)
        for r in eachindex(values)
            if !isequal(values[r], previous)
                previous = values[r]
                at = _brm_label_position(table, lo, hi, previous)
            end
            if at != 0
                counts[at] += 1
            end
        end
    end
    rows = [Vector{Int}(undef, counts[k]) for k in 1:ncells]
    for k in 1:ncells
        counts[k] = 0
    end
    if !isempty(values)
        previous = values[1]
        at = _brm_label_position(table, lo, hi, previous)
        for r in eachindex(values)
            if !isequal(values[r], previous)
                previous = values[r]
                at = _brm_label_position(table, lo, hi, previous)
            end
            if at != 0
                counts[at] += 1
                rows[at][counts[at]] = r
            end
        end
    end
    rows
end

# The rows `values` (the join's row labels) of each cell `labels`, ascending.
# A row whose label names no cell is in no cell (decision `0z97g0b`); a
# missing label is refused.
@_brm_traceable function _brm_ragged_partition(x, group::Symbol, values::AbstractVector,
                                               labels::AbstractVector,
                                               prefix::AbstractString)
    _brm_check_missing_labels(x, group, values, prefix)
    if eltype(values) <: Base.BitInteger && eltype(labels) <: Base.BitInteger
        if _brm_dense_labels(labels)
            rows = _brm_partition_rows(values, length(labels),
                _brm_dense_label_table(labels), minimum(labels), maximum(labels))
        else
            rows = _brm_partition_rows(values, length(labels),
                _brm_hashed_label_table(labels), nothing, nothing)
        end
    else
        rows = _brm_partition_rows(values, length(labels), _brm_hashed_label_table(labels),
            nothing, nothing)
    end
    rows
end

# The order column of an ordered join (`ragged(…, group; order=order)`): a
# column on the join's `nrows` rows with no missing values.
@_brm_traceable function _brm_check_order(times, order::Symbol, group::Symbol,
                                          nrows::Integer, prefix::AbstractString)
    if !(typeof(times) <: AbstractVector)
        _brm_order_not_a_column(times, order, group, nrows, prefix)
    end
    if length(times) != nrows
        throw(ErrorException("$prefix: kernel(...) `ragged(…, $group; order=$order)`: \
            `$order` must be a column on `$group`'s $nrows rows; got $(length(times)) \
            rows."))
    end
    if Missing <: eltype(times)
        for r in eachindex(times)
            if ismissing(times[r])
                throw(ErrorException("$prefix: kernel(...) `ragged(…, $group; \
                    order=$order)`: `$order` has missing values."))
            end
        end
    end
    nothing
end

# The positions of `times` in ascending order, ties in position order: a
# bottom-up merge sort. Comparisons are `<`, with a NaN after every other value
# as `isless` orders them; a -0.0 and a 0.0 tie.
@_brm_traceable function _brm_stable_order(times::AbstractVector)
    n = length(times)
    order = Vector{Int}(undef, n)
    for k in 1:n
        order[k] = k
    end
    merged = Vector{Int}(undef, n)
    width = 1
    while width < n
        lo = 1
        while lo <= n
            mid = min(lo + width - 1, n)
            hi = min(lo + 2 * width - 1, n)
            i = lo
            j = mid + 1
            for k in lo:hi
                take_right = j <= hi
                if take_right && i <= mid
                    left = times[order[i]]
                    right = times[order[j]]
                    # `left != left` holds for a NaN only, of any element type.
                    take_right = left != left ? right == right : right < left
                end
                if take_right
                    merged[k] = order[j]
                    j += 1
                else
                    merged[k] = order[i]
                    i += 1
                end
            end
            lo = hi + 1
        end
        for k in 1:n
            order[k] = merged[k]
        end
        width = 2 * width
    end
    order
end

# Each cell's rows in ascending `times`, ties in row order.
@_brm_traceable _brm_order_cells(rows::AbstractVector, times::AbstractVector) =
    [rows[k][_brm_stable_order(times[rows[k]])] for k in eachindex(rows)]

# The RK graph's row partition of a ragged join, from the bound columns:
# `names` is `[x, group, cell, axes...]`, `axes` the data columns of `x`'s own
# frame read through the join, each checked against `groups`. The plate then
# iterates `cells`. `jax_source` translates it into the module's
# `prepare_data`.
@_brm_traceable function brm_ragged_rows(names::AbstractVector, groups, cells, axes::Tuple)
    labels = _brm_cell_labels(cells, names[3], "RK backend")
    values = _brm_group_values(groups)
    _brm_check_ragged_axes(names, 3, axes, length(values), "RK backend")
    _brm_ragged_partition(names[1], names[2], values, labels, "RK backend")
end
# The same with each cell's rows in ascending `order`, ties in row order:
# `names` is `[x, group, cell, order, axes...]`.
@_brm_traceable function brm_ragged_rows_ordered(names::AbstractVector, groups, cells, order,
                                                 axes::Tuple)
    labels = _brm_cell_labels(cells, names[3], "RK backend")
    values = _brm_group_values(groups)
    _brm_check_ragged_axes(names, 4, axes, length(values), "RK backend")
    rows = _brm_ragged_partition(names[1], names[2], values, labels, "RK backend")
    _brm_check_order(order, names[4], names[2], length(values), "RK backend")
    _brm_order_cells(rows, order)
end
# Another column read through a join whose partition `rows` an earlier graph
# statement computed (`brm_ragged_rows` or `brm_ragged_rows_ordered` over the
# same group, cell and order columns): `names` is `[x, group, axes...]`, and
# `x`'s axes are checked against `groups` as `brm_ragged_rows` checks them.
# The partition is returned as it is, so the graph partitions each join once
# however many columns its cells read through it.
@_brm_traceable function brm_ragged_rows_shared(rows::AbstractVector, names::AbstractVector,
                                                groups, axes::Tuple)
    _brm_check_ragged_axes(names, 2, axes, length(groups), "RK backend")
    rows
end
