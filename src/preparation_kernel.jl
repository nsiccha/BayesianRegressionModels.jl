# Group identity and event-row partitions shared by kernel emitters.
# This layer owns no SLIC statements or backend data cleanup.
# (Any other vector's labels are the vector itself, `ragged_rows.jl`.)
_brm_group_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)

# Computed values own their result axes. Their inputs can be shared arrays,
# or live on entirely different axes, and are not a formula's row columns.
_brm_kernel_value_input(_value) = false
_brm_kernel_value_input(value::NamedColumn) = _brm_kernel_value_input(parent(value))
_brm_kernel_value_input(::ExprColumn{typeof(assign)}) = true
function _brm_kernel_value_input(value::ExprColumn{typeof(~)})
    rhs = last(getargs(value))
    rhs isa ExprColumn && (getf(rhs) === kernel || getf(rhs) isa BRMComponent)
end

_brm_collect_data_columns!(_acc, _x) = nothing
_brm_collect_data_columns!(acc, x::NamedColumn) = begin
    p = parent(x)
    p isa DataColumn ? _brm_push_data_column!(acc, name(x), parent(p)) :
        _brm_collect_data_columns!(acc, p)
    nothing
end
# A scalar data column — a scalar field of the container or a `constant(...)`
# declaration — broadcasts over whatever frame reads it, so it names no row axis.
_brm_push_data_column!(acc, column::Symbol, values) = push!(acc, (column, values))
_brm_push_data_column!(_acc, ::Symbol, ::Number) = nothing
_brm_collect_data_columns!(acc, x::ExprColumn) = begin
    getf(x) === (~) && return nothing
    for a in getargs(x); _brm_collect_data_columns!(acc, a); end
    for (key, v) in pairs(getkwargs(x))
        _brm_row_keyword(getf(x), Val(key)) && _brm_collect_data_columns!(acc, v)
    end
    nothing
end
# A gather `x[idx]` lies on its index's rows; `x` holds one value per entry of
# its own axis (`c_b = c[pos_b]`, part 1's value on part 2's rows), so its data
# columns name no row axis of the reader.
_brm_collect_data_columns!(acc, x::ExprColumn{typeof(getindex)}) =
    length(getargs(x)) == 2 && isempty(getkwargs(x)) ?
        _brm_collect_data_columns!(acc, last(getargs(x))) :
        invoke(_brm_collect_data_columns!, Tuple{Any,ExprColumn}, acc, x)
# A term keyword whose data lives on another axis than the term's rows names
# no row axis: a level correlation (`cor = Omega`, a known `cor = C`, or the
# levels' coordinates in `cor = multigroup((x, y))`) is one entry per level of
# the term's grouping (`cdar`'s `C` one row and column per level of `by`), and
# an areal term's `edges = (from, to)` one per edge and its `nodes = all_areas`
# one per area of the graph.
_brm_row_keyword(_f, ::Val) = true
_brm_row_keyword(::Union{typeof(gr),typeof(hsgp),typeof(cdar)}, ::Val{:cor}) = false
_brm_row_keyword(::typeof(car), ::Union{Val{:edges},Val{:nodes}}) = false
_brm_row_keyword(::typeof(gmrf), ::Union{Val{:mesh},Val{:cor}}) = false
_brm_collect_data_columns!(acc, x::Union{Tuple,AbstractVector}) = begin
    for a in x; _brm_collect_data_columns!(acc, a); end
    nothing
end
# `(name, length)` of each row-wise data column `x` reads.
function _brm_collect_data_lengths!(acc, x)
    columns = Tuple{Symbol,Any}[]
    _brm_collect_data_columns!(columns, x)
    append!(acc, ((column, length(values)) for (column, values) in columns))
    nothing
end

# `(name, values)` of every row-wise raw data column a linear predictor's terms
# read; scalar columns are skipped.
function _brm_kernel_lp_data_columns(lp_col)
    isnothing(_brm_parameter_length(lp_col)) || return Tuple{Symbol,Any}[]
    columns = Tuple{Symbol,Any}[]
    _brm_collect_data_columns!(columns, getargs(parent(lp_col))[2])
    columns
end
function _brm_kernel_lp_data_lengths(lp_col)
    n = _brm_parameter_length(lp_col)
    isnothing(n) || return Tuple{Symbol,Int}[(name(lp_col), n)]
    Tuple{Symbol,Int}[(column, length(values)) for (column, values) in _brm_kernel_lp_data_columns(lp_col)]
end

# A per-subject linear predictor without a random effect, or whose random
# effects group the rows coarser than one label per row (a class over the
# cells), defines no grouping; it rides on the subject frame a ranef LP — or
# the cell labels, or in a no-random-effects panel the pre-grouped columns —
# defines, sliced by the same row index. That is sound only when every data
# column its terms name, grouping columns included, has exactly one row per
# subject. An LP with no data column has no rows at all.
function _brm_kernel_check_subject_frame_lp(lp_col, nsub; prefix="BRM kernel preparation")
    lp = name(lp_col)
    n = _brm_parameter_length(lp_col)
    if !isnothing(n)
        n == nsub || error(
            "$prefix: kernel(...) vector parameter `$lp` has $n entries for $nsub " *
            "plate cells; a direct `$lp[i]` read needs one entry per cell.")
        return nothing
    end
    lens = _brm_kernel_lp_data_lengths(lp_col)
    isempty(lens) && error(
        "$prefix: kernel(...) per-subject linear predictor `$lp` has no random-effect " *
        "term and no data column, so it has no per-subject rows. Declare a shared " *
        "scalar as `$lp ~ <distribution>` and read it in the cell without `[i]`.")
    for (column, n) in lens
        n == nsub || error(
            "$prefix: kernel(...) linear predictor `$lp` has no random effect whose " *
            "grouping lists one label per row, so it is read per subject (one entry " *
            "per subject), but its data vector `$column` has $n rows for $nsub " *
            "subjects. Read it at each subject's indices instead, `$lp[rows[i]]` " *
            "with `rows = groups(group => labels)`.")
    end
    nothing
end

# One non-missing unique label per row: what lets a grouping column label a
# plate's cells.
function _brm_kernel_labels_rows(raw::AbstractVector)
    values = _brm_group_values(raw)
    !isempty(values) && !any(ismissing, values) && allunique(values)
end
_brm_kernel_labels_rows(_raw) = false

# No rows are no cells: the plate observes nothing and the priors stay
# (`_brm_cell_labels`, `ragged_rows.jl`).
_brm_kernel_subject_values(raw, group::Symbol; prefix="BRM kernel preparation") =
    collect(_brm_cell_labels(raw, group, prefix))

# Each cell's entries of a flat column. The result's type comes from the column,
# not from the entries, so a plate with no cells binds the same element type as
# its populated twin and emits the same program. An absent replay response stays
# absent.
_brm_gather_cells(values, rows) = [values[r] for r in rows]
_brm_gather_cells(values::AbstractVector, rows) =
    typeof(values[Int[]])[values[r] for r in rows]
_brm_gather_cells(values::_BRMAbsentResponse, rows) = values

# An empty `ev_vals` is a zero-row axis (a design with no rows on it, such as
# no placebo observations): every cell gets an empty slice. A row whose label
# names no cell is read by no cell, as `groups(g => labels)` leaves it in no
# group (decision `0z97g0b`). An `observed` response is the exception: an
# entry no cell observes would be an observation carried into the fit
# unconditioned (`_brm_kernel_unobserved_entries`). Each cell holds its rows
# ascending. The partition is `_brm_ragged_partition` (`ragged_rows.jl`), the
# one the RK graph computes on every call of a reader whose design is a
# runtime input: it builds the label lookup once, counts every cell's rows and
# then fills each cell, allocating nothing per row beyond the cells.
function _brm_kernel_ragged_partition(a_name::Symbol, grp_name::Symbol,
                                     ev_vals::AbstractVector,
                                     subject_vals::AbstractVector;
                                     prefix="BRM kernel preparation", observed::Bool=false)
    rows = _brm_ragged_partition(a_name, grp_name, ev_vals, subject_vals, prefix)
    observed && sum(length, rows; init=0) != length(ev_vals) &&
        _brm_kernel_unobserved_entries(a_name, grp_name, ev_vals,
            _brm_label_positions(ev_vals, subject_vals); prefix)
    rows
end

# Refuses what `_brm_kernel_ragged_partition` refuses, with the same messages,
# without building the partition. The RK design binder validates each design
# this way, and the graph builds the partition itself.
function _brm_kernel_check_ragged_partition(a_name::Symbol, grp_name::Symbol,
                                           ev_vals::AbstractVector,
                                           subject_vals::AbstractVector;
                                           prefix="BRM kernel preparation", observed::Bool=false)
    _brm_check_missing_labels(a_name, grp_name, ev_vals, prefix)
    observed && !_brm_labels_name_cells(ev_vals, subject_vals) &&
        _brm_kernel_unobserved_entries(a_name, grp_name, ev_vals,
            _brm_label_positions(ev_vals, subject_vals); prefix)
    nothing
end

# refused: a response entry in no cell is an observation the fit would carry
# without conditioning on it — the row mask the user removed (`1bwqnxs`); a
# fit receives only the observations it conditions on (`0q8hcwq`).
function _brm_kernel_unobserved_entries(a_name, grp_name, ev_vals, cell; prefix)
    unknown = Any[v for (v, i) in zip(ev_vals, cell) if i == 0]
    error(
        "$prefix: kernel(...) `ragged($a_name, $grp_name)`: the response `$a_name` ",
        "is observed through `$grp_name`, but its entries labelled $(unique(unknown)) ",
        "belong to no cell, so no cell would observe them. A fit receives only the ",
        "observations it conditions on: pass `$a_name` with only the entries the cells ",
        "observe, and predict the others by replaying the fit (`reprocess`).")
end

# The position of each value's label in `labels` (`0` when none matches, the
# last match when a label repeats), compared by `isequal`: the lookup
# `_brm_ragged_partition` uses (`ragged_rows.jl`).
_brm_label_positions(values::AbstractVector, labels::AbstractVector) =
    _brm_with_label_table(values, labels) do table, lo, hi
        Int[_brm_label_position(table, lo, hi, v) for v in values]
    end
# Whether every value's label is in `labels`, without recording positions.
_brm_labels_name_cells(values::AbstractVector, labels::AbstractVector) =
    _brm_with_label_table(values, labels) do table, lo, hi
        all(v -> _brm_label_position(table, lo, hi, v) != 0, values)
    end
_brm_with_label_table(f, values, labels) =
    eltype(values) <: Base.BitInteger && eltype(labels) <: Base.BitInteger &&
        _brm_dense_labels(labels) ?
        f(_brm_dense_label_table(labels), minimum(labels), maximum(labels)) :
        f(_brm_hashed_label_table(labels), nothing, nothing)

# `ragged(x, group)` or `ragged(x, group, cells)` — the kernel positional a
# model-scope `ragged(x, group => cells)` lowers to.
function _brm_kernel_ragged_args(c; prefix="BRM kernel preparation")
    args = getargs(c)
    length(args) in (2, 3) || error(
        "$prefix: kernel(...) `ragged(...)` takes the thing to group, its grouping ",
        "column and optionally the plate's cell label column — got ",
        "$(length(args)) positional args.")
    args
end

# The cell label column of a plate no random-effect linear predictor labels:
# the `cells` of every `ragged(x, group => cells)` input. Its rows are the
# cells, in row order, and each row of `x` joins its cell by its `group`
# label. Without it the only order left is position — the first appearance of
# `group`'s labels — which nothing can check against a per-cell column such
# as district totals, so a differently ordered frame would bind to the wrong
# cells silently (refused at the user's direction, 2026-10-19).
function _brm_kernel_cell_label_column(ragged_cols; prefix="BRM kernel preparation")
    unlabelled = [name(first(getargs(c))) for c in ragged_cols if length(getargs(c)) == 2]
    isempty(unlabelled) || error(
        "$prefix: kernel(...) `ragged(...)` input(s) ",
        join(("`$n`" for n in unlabelled), ", "), " have no cell labels to join ",
        "against: no per-cell linear predictor carries a random effect that would ",
        "label the plate's cells. Name the cells' label vector — `x[rows[i]]` with ",
        "`rows = groups(group => cells)` in a `@plate for` cell, or ",
        "`ragged(x, group, cells)` as a `kernel(...)` positional, where `cells` lists ",
        "each cell's label once — so every entry joins its cell by label. Matching ",
        "entries to cells by position is refused: a per-cell vector ordered ",
        "differently from `group`'s first appearance would bind to the wrong cell ",
        "without any error.")
    cells = [getargs(c)[3] for c in ragged_cols]
    names = unique(name.(cells))
    length(names) == 1 || error(
        "$prefix: kernel(...) `ragged(...)` inputs name different cell label ",
        "columns (", join(("`$n`" for n in names), ", "), "); one plate has one ",
        "set of cells.")
    column = first(cells)
    (column isa NamedColumn && parent(column) isa DataColumn) || error(
        "$prefix: kernel(...) the cell label column must be a raw data column; ",
        "got $(typeof(column)).")
    column
end

# The cells of a plate no random-effect linear predictor labels, as
# `(; column, values, distinct)`: one cell per row of the cell label column
# every `ragged(x, group => cells)` input names. A `@plate for p in groups(g)`
# header lowers each read to `ragged(x, g, g)`, its grouping column as its own
# cell labels: one cell per distinct label of `g`, sorted (`distinct`). Such a
# cell has no per-cell input and collects nothing (the header refuses it), so
# nothing is matched to a cell by position (decision `1pnob5t`); `@brm`
# refuses `ragged(x, g => g)` written at model scope.
function _brm_kernel_plate_cells(ragged_cols; prefix="BRM kernel preparation")
    column = _brm_kernel_cell_label_column(ragged_cols; prefix)
    distinct = all(c -> name(getargs(c)[2]) === name(column), ragged_cols)
    values = distinct ?
        _brm_kernel_distinct_labels(parent(parent(column)), name(column); prefix) :
        _brm_kernel_subject_values(parent(parent(column)), name(column); prefix)
    (; column, values, distinct)
end

# The distinct labels of a grouping column, sorted: the cells of a plate that
# only joins rows by that column.
function _brm_kernel_distinct_labels(raw, group::Symbol; prefix="BRM kernel preparation")
    raw isa AbstractVector || error(
        "$prefix: kernel(...) grouping column `$group` must be a vector, got $(typeof(raw)).")
    values = collect(_brm_group_values(raw))
    any(ismissing, values) && error(
        "$prefix: kernel(...) grouping column `$group` has missing labels.")
    sort!(unique(values))
end

# Under a random-effect linear predictor the cells are already labelled by its
# grouping; a `ragged(x, group => cells)` must then name that same column.
function _brm_kernel_check_cell_labels(ragged_cols, group::Symbol;
                                       prefix="BRM kernel preparation")
    for c in ragged_cols
        args = getargs(c)
        length(args) == 3 && name(args[3]) !== group || continue
        error(
            "$prefix: kernel(...) `ragged($(name(args[1])), $(name(args[2])) => ",
            "$(name(args[3])))` labels the cells by `$(name(args[3]))`, but the ",
            "plate's random-effect linear predictor already labels them by ",
            "`$group`. Name `$group`, or drop `=> $(name(args[3]))`.")
    end
end

# A raw column's rows, checked against its ragged group column. An absent
# replay response (`_BRMAbsentResponse`) lies on the group column's rows.
_brm_ragged_column_length!(lens, name, values) = push!(lens, (name, length(values)))
_brm_ragged_column_length!(lens, name, ::_BRMAbsentResponse) = lens

function _brm_kernel_ragged_rows(arg_col, grp_arg, g_vals;
                                 prefix="BRM kernel preparation",
                                 allow_unbound_observation=false, order=nothing,
                                 observed::Bool=false, output_reads=())
    a_name = name(arg_col)
    decl = parent(arg_col)
    # `is_lp` selects a runtime gather of a model value. Assignments, like
    # predictors and completions, must keep their dependencies and gradients.
    # Non-last indexed plate outputs have no top-level declaration of their
    # own. The program's validated output-reader map identifies their writer;
    # they need the same runtime gather as the collected output.
    is_output = decl isa MissingColumn && a_name in output_reads
    is_lp = is_output || decl isa ExprColumn && getf(decl) in ((~), assign)
    is_unbound = decl isa MissingColumn && allow_unbound_observation && !is_output
    is_lp || decl isa DataColumn || is_unbound || error(
        "$prefix: kernel(...) positional `ragged($a_name, …)`: `$a_name` must be either a ",
        "model value declared in this @brm block (a plate output, predictor, completion or ",
        "assignment) or a raw data column; got $(typeof(decl)).")
    (grp_arg isa NamedColumn && parent(grp_arg) isa DataColumn) || error(
        "$prefix: kernel(...) `ragged($a_name, …)` needs a raw data column naming the ",
        "subject of every row of `$a_name`'s row axis; got $(typeof(grp_arg)).")
    grp_name = name(grp_arg)
    ev_vals = collect(_brm_group_values(parent(parent(grp_arg))))
    n_ev = length(ev_vals)

    # The frame `$a_name` lives on must be the one `$grp_name` describes. For a
    # model expression that means its row-wise data dependencies (a gather
    # follows its index's axis); for a raw column, itself.
    lens = Tuple{Symbol,Int}[]
    if is_lp && !is_output
        append!(lens, _brm_kernel_lp_data_lengths(arg_col))
    elseif decl isa DataColumn
        v = parent(decl)
        v isa AbstractVector{<:AbstractVector} && error(
            "$prefix: `$a_name` read through `$grp_name`: `$a_name` is ALREADY a ",
            "ragged per-subject column, so there is nothing to group. Read it per cell ",
            "directly, `$a_name[i]`.")
        _brm_ragged_column_length!(lens, a_name, v)
    end
    _brm_kernel_check_ragged_axis(a_name, grp_name, lens, n_ev; prefix)

    # An omitted response is observed too: its draw covers every entry.
    rows = _brm_kernel_ragged_partition(a_name, grp_name, ev_vals, g_vals; prefix,
        observed=observed || is_unbound)
    if !isnothing(order)
        (order isa NamedColumn && parent(order) isa DataColumn) || error(
            "$prefix: kernel(...) `ragged($a_name, $grp_name; order=…)`: `order=` ",
            "names a raw data column on `$grp_name`'s rows; got $(typeof(order)).")
        _brm_kernel_order_rows!(rows, parent(parent(order)), name(order), grp_name, n_ev;
            prefix)
    end

    (; rows, is_lp, is_unbound, group_name=grp_name, group_values=ev_vals)
end

function _brm_kernel_check_ragged_axis(a_name, grp_name, lens, n_ev;
                                       prefix="BRM kernel preparation")
    for (nm, L) in lens
        _brm_check_ragged_axis(a_name, grp_name, nm, L, n_ev, prefix)
    end
    nothing
end

# The RK graph computes the same partition from the bound columns
# (`brm_ragged_rows` and its variants, `ragged_rows.jl`, emitted by
# `_rk_emit_kernel_partition!`), so a design's row geometry is a graph value of
# its raw columns rather than a planned input. A row in no cell is skipped
# there; a response's coverage is refused where its data is bound
# (`_rk_prepare_kernel_value`, `_rk_design_cell_check`), not in the graph.
#
# Refuses what `brm_ragged_rows` refuses (`brm_ragged_rows_ordered` given
# `order` and its `times`), with the same messages, without building the
# partition; with `observed`, also a response entry in no cell.
function _brm_check_ragged_rows(x, group, groups, cell, cells, axis_names, axes;
                               order=nothing, times=nothing, observed::Bool=false)
    prefix = "RK backend"
    labels = _brm_cell_labels(cells, cell, prefix)
    ev_vals = _brm_group_values(groups)
    for (nm, a) in zip(axis_names, axes)
        _brm_check_ragged_axis(x, group, nm, length(a), length(ev_vals), prefix)
    end
    _brm_kernel_check_ragged_partition(x, group, ev_vals, labels; prefix, observed)
    order === nothing || _brm_check_order(times, order, group, length(groups), prefix)
    nothing
end

# `ragged(x, group; order=time)`: each cell's rows in ascending `time`, ties in
# row order (`_brm_order_cells`, `ragged_rows.jl`).
function _brm_kernel_order_rows!(rows, times, order::Symbol, group::Symbol, nrows;
                                 prefix="BRM kernel preparation")
    _brm_check_order(times, order, group, nrows, prefix)
    copyto!(rows, _brm_order_cells(rows, times))
end

# The `into=` target and `order=` column of a `ragged(...)` kernel positional,
# each `nothing` when absent.
function _brm_kernel_ragged_options(c; prefix="BRM kernel preparation")
    kwargs = getkwargs(c)
    unknown = [k for k in keys(kwargs) if !(k in (:into, :order))]
    isempty(unknown) || error(
        "$prefix: kernel(...) `ragged(...)` accepts only the `into=` and `order=` ",
        "keywords; got ", join(("`$k`" for k in unknown), ", "), ".")
    (; into=get(kwargs, :into, nothing), order=get(kwargs, :order, nothing))
end

# `ragged(idx, group; into=target)`: `idx` holds 1-based row numbers on the flat
# axis whose subjects `target` labels, with `0` meaning "no row". Each becomes
# its position among that subject's rows of `target`, in flat order — the order
# a `ragged(x, target)` gather builds — so a cell indexes its own subject's
# slice of the target axis. Flat in, flat out: the result is then gathered by
# `group` exactly like any other data column. An index into another subject's
# rows would read the wrong subject silently, so it is refused.
function _brm_kernel_ragged_index_values(arg_col, grp_arg, into_col;
                                         prefix="BRM kernel preparation")
    idx_name, grp_name = name(arg_col), name(grp_arg)
    parent(arg_col) isa DataColumn || error(
        "$prefix: kernel(...) `ragged($idx_name, $grp_name; into=…)`: `into=` ",
        "rebases a data column of row numbers; `$idx_name` is not a raw data column.")
    (into_col isa NamedColumn && parent(into_col) isa DataColumn) || error(
        "$prefix: kernel(...) `ragged($idx_name, $grp_name; into=…)`: `into=` must ",
        "name a raw data column labelling the subject of every row of the indexed ",
        "axis; got $(typeof(into_col)).")
    _brm_kernel_rebase_index(idx_name, parent(parent(arg_col)),
        grp_name, parent(parent(grp_arg)), name(into_col), parent(parent(into_col));
        prefix)
end

function _brm_kernel_rebase_index(idx_name::Symbol, idx, grp_name::Symbol, raw_groups,
                                  into_name::Symbol, raw_targets;
                                  prefix="BRM kernel preparation")
    decl = "`ragged($idx_name, $grp_name; into=$into_name)`"
    idx isa AbstractVector{<:Integer} && !(eltype(idx) <: Bool) || error(
        "$prefix: kernel(...) $decl: `$idx_name` must be a flat integer column of ",
        "row numbers; got $(typeof(idx)).")
    raw_targets isa AbstractVector || error(
        "$prefix: kernel(...) $decl: `$into_name` must be a vector; got ",
        "$(typeof(raw_targets)).")
    groups = collect(_brm_group_values(raw_groups))
    targets = collect(_brm_group_values(raw_targets))
    length(idx) == length(groups) || error(
        "$prefix: kernel(...) $decl: `$idx_name` has $(length(idx)) rows but ",
        "`$grp_name` has $(length(groups)).")
    any(ismissing, targets) && error(
        "$prefix: kernel(...) $decl: `$into_name` has missing labels.")
    seen = Dict{Any,Int}()
    position = [seen[label] = get(seen, label, 0) + 1 for label in targets]
    map(eachindex(idx)) do k
        r = idx[k]
        r == 0 && return 0
        1 <= r <= length(targets) || error(
            "$prefix: kernel(...) $decl: row $k of `$idx_name` is $r, outside the ",
            "$(length(targets)) rows of `$into_name`.")
        isequal(targets[r], groups[k]) || error(
            "$prefix: kernel(...) $decl: row $k of `$idx_name` belongs to subject ",
            "$(repr(groups[k])) but points at row $r of `$into_name`, which belongs ",
            "to subject $(repr(targets[r])). An index must stay inside its own subject.")
        position[r]
    end
end

# The indexed outputs (`name[i] = …`) of each `@plate for` cell besides its
# collected one, keyed by name, with the kernel targets of the cells writing
# it. Only outputs the cell body still binds count: response pruning may have
# dropped one.
function _brm_kernel_side_outputs(@nospecialize(brmi::BRMI))
    writers = Dict{Symbol,Vector{Symbol}}()
    for (target, raw) in _brm_operation_entries(brmi)
        op = _named_op(raw)
        (op !== nothing && getf(op) === (~)) || continue
        rhs = _as_expr_column(last(getargs(op)))
        (rhs !== nothing && getf(rhs) === kernel && !isempty(getargs(rhs))) || continue
        lam = first(getargs(rhs))
        outputs = _brm_inline_outputs(lam)
        isempty(outputs) && continue
        body = _brm_inline_expr(lam).args[2]
        stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
        bound = Set{Symbol}(stmt.args[1] for stmt in stmts
            if Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol)
        for output in outputs
            output !== target && output in bound || continue
            push!(get!(writers, output, Symbol[]), target)
        end
    end
    writers
end

# Reads of such outputs: each free name the model reads at model scope or
# captures in another cell, mapped to its producer's kernel target. Captures
# need not have a MissingColumn entry of their own. The cell's value is the
# model value of that name, exactly like the collected output, whichever
# indexed assignment of the cell comes last.
function _brm_kernel_output_reads(@nospecialize(brmi::BRMI))
    writers = _brm_kernel_side_outputs(brmi)
    reads = Dict{Symbol,Symbol}()
    isempty(writers) && return reads
    refs = Set{Symbol}()
    for (key, raw) in _brm_operation_entries(brmi)
        if raw isa NamedColumn && parent(raw) isa MissingColumn
            push!(refs, key)
        else
            op = _named_op(raw)
            op isa ExprColumn || continue
            _brm_operation_references!(refs, last(getargs(op)))
        end
    end
    for key in refs
        cells = get(writers, key, nothing)
        cells === nothing && continue
        length(cells) == 1 || error(
            "@brm: the model reads `$key`, an indexed output `$key[i] = …` of " *
            "several `@plate for` cells (those collecting `" * join(cells, "`, `") *
            "`); give each cell's output its own name")
        reads[key] = only(cells)
    end
    reads
end

# A captured plate output still holds whole cells when another cell selects
# it, `out[which[i]][positions[i]]`. These reads remain raw cell expressions,
# unlike model-scope ExprColumns, so both emitters must lower them too. The
# backend supplies the representation of a selected cell; locals and formals
# of an inner cell keep their own indexing semantics.
_brm_plate_cell_refs(select, x, outputs, bound=()) = x
function _brm_plate_cell_refs(select, x::Expr, outputs, bound=())
    recur(value) = _brm_plate_cell_refs(select, value, outputs, bound)
    if Meta.isexpr(x, :->, 2)
        inner = _brm_cell_bound_names!(Set{Symbol}(bound), x)
        return Expr(:->, x.args[1],
            _brm_plate_cell_refs(select, x.args[2], outputs, inner))
    elseif Meta.isexpr(x, :generator)
        inner = Set{Symbol}(bound)
        function ranges(spec)
            if Meta.isexpr(spec, :(=), 2)
                range = _brm_plate_cell_refs(select, spec.args[2], outputs, inner)
                _brm_bound_symbols!(inner, spec.args[1])
                return Expr(:(=), spec.args[1], range)
            elseif Meta.isexpr(spec, :filter)
                bindings = map(ranges, spec.args[2:end])
                return Expr(:filter,
                    _brm_plate_cell_refs(select, spec.args[1], outputs, inner), bindings...)
            end
            spec
        end
        specs = map(ranges, x.args[2:end])
        return Expr(:generator, _brm_plate_cell_refs(select, x.args[1], outputs, inner),
            specs...)
    elseif Meta.isexpr(x, :quote)
        return x
    elseif Meta.isexpr(x, :ref, 2) && x.args[1] isa Symbol &&
            x.args[1] in outputs && !(x.args[1] in bound)
        return select(x.args[1], recur(x.args[2]))
    elseif x.head in (:call, :macrocall, :kw) || _brm_is_broadcast_call(x)
        return Expr(x.head, first(x.args), map(recur, x.args[2:end])...)
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode
        return Expr(:., recur(x.args[1]), x.args[2])
    end
    Expr(x.head, map(recur, x.args)...)
end
