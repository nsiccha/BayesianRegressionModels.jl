# A `@plate for` cell described as written (snag brm-description-256b3b35,
# decision 011ibo0). `@plate for i in …` lowers to a kernel with one formal per
# input, each bound to what the cell reads at entry `i`: an entry `x[i]`, a
# vector at a model-scope binding's rows `x[rows[i]]`, positions
# `indexin(idx[rows[i]], target[i])`, or, in a `groups(g)` plate, the group's
# rows `x[p]`. The kernel keeps the authored cell (`_BRMInlineBody.source`), so
# the plate is read from it and the cell's statements render in its indices
# instead of through the formals' bindings. A kernel without that source, the
# deprecated `kernel(...) do`, keeps its input-binding rendering.

# `x => (; rows, target, zero)` for each `x[rows[i]]` and
# `indexin(x[rows[i]], target[i])` read in the authored cell statements.
function _brmd_plate_authored_reads(body, i, groupings)
    reads = Dict{Symbol,NamedTuple}()
    function visit(ex)
        ex isa Expr || return
        indexed = _brm_plate_indexin_read(ex, i, groupings)
        if indexed !== nothing
            name, rows, target, zero = indexed
            reads[name] = (; rows, target, zero)
            return
        end
        read = _brm_plate_binding_read(ex, i, groupings)
        if read !== nothing
            name, rows = read
            haskey(reads, name) || (reads[name] = (; rows, target=nothing, zero=false))
            return
        end
        foreach(visit, ex.args)
    end
    visit(body)
    reads
end

# What the formal `name`, bound to the kernel argument `value`, reads at the
# plate index, or `nothing` when the authored cell does not say.
_brmd_plate_read(name, _value::BRMDescriptionReference, _header, _authored) =
    (; name, kind=:entry, rows=nothing, target=nothing, zero=false)
function _brmd_plate_read(name, value::BRMDescriptionComponent, header, authored)
    value.callable === ragged || return nothing
    header === nothing || return (; name, kind=:group, rows=nothing, target=nothing, zero=false)
    authored === nothing && return nothing
    (; name, kind=haskey(value.keywords, :into) ? :positions : :rows, authored...)
end
_brmd_plate_read(_name, _value, _header, _authored) = nothing

# The authored plate of a kernel, or `nothing`: its loop `index`, the `anchor`
# of an `eachindex(x)` range or the `header` grouping of a `groups(g)` range,
# the model-scope `bindings` it reads in authored order, its `outputs`
# (`name[i] = …`) and one `reads` record per kernel formal.
function _brmd_plate(body, params, outer)
    source = body isa _BRMInlineBody ? body.source : nothing
    source === nothing && return nothing
    statements = [s for s in Meta.parseall(source).args if !(s isa LineNumberNode)]
    bindings = Tuple((; name=first(s.args), _brm_plate_binding(last(s.args))...)
        for s in statements if _is_brm_plate_binding(s))
    groupings = Dict{Symbol,_BRMPlateGrouping}(b.name => (; b.group, b.cells, b.order) for b in bindings)
    i, range, cell = _brm_plate_loop(only(filter(_is_brm_plate, statements)))
    header = _brm_plate_header_grouping(range)
    authored = _brmd_plate_authored_reads(cell, i, groupings)
    reads = map((name, value) -> _brmd_plate_read(name, value, header, get(authored, name, nothing)),
        Tuple(params), outer)
    any(isnothing, reads) && return nothing
    (; index=i, anchor=header === nothing ? _brm_plate_range_input(range) : nothing,
       header, bindings, outputs=Tuple(last(_brm_plate_outputs(cell, i))), reads)
end

_brmd_cell_plate(c) = get(c.provenance, :cell_plate, nothing)
function _brmd_plate_read_of(c, name::Symbol)
    plate = _brmd_cell_plate(c)
    plate === nothing && return nothing
    found = filter(r -> r.name === name, plate.reads)
    isempty(found) ? nothing : only(found)
end

# A plate output `name[i] = …` is a cell local the cell collects; as written it
# is the model's `name` at entry `i`. Other cell locals keep their scoped names.
function _brmd_plate_output_math(c, name::Symbol)
    plate = _brmd_cell_plate(c)
    plate === nothing || !(name in plate.outputs) ? nothing :
        "{" * _brmd_identifier_math(c, name) * "}_{" * _brmd_plate_index_math(plate.index) * "}"
end
_brmd_cell_name_math(c, name) = something(_brmd_plate_output_math(c, name), brm_description_symbol(c, name))
# A model-scope name inside a cell: its notation symbol, never a cell local's.
_brmd_identifier_math(c, name) = brm_description_symbol(
    _brmd_render_context(c; provenance=merge(c.provenance, (; cell_names=()))), name)

# The plate index and a row index distinct from it, as math.
_brmd_plate_index_math(x) = length(string(x)) == 1 ? _brmd_escape(x) : "\\mathit{" * _brmd_escape(x) * "}"
_brmd_plate_row_math(plate) = plate.index === :j ? "l" : "j"
# The rows entry `i` reads: a binding's `rows_i`, or in a `groups(g)` plate the
# group's rows `J_p`.
_brmd_plate_rows_math(plate, rows::Symbol) =
    _brmd_identifier(rows) * "_{" * _brmd_plate_index_math(plate.index) * "}"
_brmd_plate_rows_math(plate, ::Nothing) = "J_{" * _brmd_plate_index_math(plate.index) * "}"
_brmd_plate_rows_name(read) = read.kind === :group ? :group : read.rows

# A formal's read at the plate index. `outer` renders the kernel argument
# without the formal's alias; inside a sum over rows (`plate_rows`), a read of
# those rows is its row entry `x_j`.
function _brmd_plate_read_math(outer, read, value)
    plate = _brmd_cell_plate(outer)
    index = _brmd_plate_index_math(plate.index)
    read.kind === :entry && return "{" * brm_description_math(outer, value) * "}_{" * index * "}"
    base = brm_description_math(outer, first(value.arguments))
    rows = read.kind === :group ? nothing : read.rows
    bound = get(outer.provenance, :plate_rows, nothing)
    !isnothing(bound) && read.kind !== :positions && bound.rows === _brmd_plate_rows_name(read) &&
        return "{" * base * "}_{" * bound.symbol * "}"
    entries = "{" * base * "}_{" * _brmd_plate_rows_math(plate, rows) * "}"
    read.kind === :positions || return entries
    positions = "\\operatorname{indexin}\\left(" * entries * "," *
        _brmd_plate_rows_math(plate, read.target) * "\\right)"
    read.zero ? "\\operatorname{something}\\left(" * positions * ",0\\right)" : positions
end

# `sum(f.(x[rows[i]], …))` over one row set, with every operation elementwise
# and every operand a read of those rows or a number, is the sum over the rows,
# `Σ_{j ∈ rows_i} f(x_j, …)`. Any other sum keeps its call form.
_brmd_plate_elementwise_rows(_c, _x) = nothing
_brmd_plate_elementwise_rows(_c, _x::Number) = ()
function _brmd_plate_elementwise_rows(c, x::BRMDescriptionReference)
    any(b -> b.name === x.name && b.role === :alias, c.bindings) || return nothing
    read = _brmd_plate_read_of(c, x.name)
    read === nothing || !(read.kind in (:rows, :group)) ? nothing : (_brmd_plate_rows_name(read),)
end
function _brmd_plate_elementwise_rows(c, x::BRMDescriptionComponent)
    x.kind === :call && get(x.keywords, :broadcast, false) === true || return nothing
    sets = map(a -> _brmd_plate_elementwise_rows(c, a), x.arguments)
    any(isnothing, sets) ? nothing : Tuple(r for s in sets for r in s)
end
function _brmd_plate_sum_math(c, x)
    plate = _brmd_cell_plate(c)
    (plate === nothing || haskey(c.provenance, :plate_rows) || length(x.arguments) != 1) && return nothing
    sets = _brmd_plate_elementwise_rows(c, only(x.arguments))
    (sets === nothing || length(unique(sets)) != 1) && return nothing
    rows = only(unique(sets))
    symbol = _brmd_plate_row_math(plate)
    inner = _brmd_render_context(c; provenance=merge(c.provenance, (; plate_rows=(; rows, symbol))))
    "\\sum_{" * symbol * "\\in " * _brmd_plate_rows_math(plate, rows === :group ? nothing : rows) * "}" *
        brm_description_math(inner, only(x.arguments))
end

# The cell as written: for which index its statements hold, each row set it
# reads, and the value it collects. Its statements are their own components,
# described in the plate's indices; no input binding needs a symbol.
function _brmd_plate_fragment(c, plate)
    owner = c.provenance.owner
    index = _brmd_plate_index_math(plate.index)
    row = _brmd_plate_row_math(plate)
    prose = String[]; equations = String[]
    if plate.header === nothing
        push!(prose, _brmd_text(c, :plate_cell; owner, index, anchor=plate.anchor))
    else
        (; group, order) = plate.header
        rows = _brmd_plate_rows_math(plate, nothing)
        push!(prose, _brmd_text(c, order === nothing ? :plate_cell_groups : :plate_cell_groups_ordered;
            owner, index, group, order, rows))
        push!(equations, rows * "=\\{" * row * ":" * _brmd_identifier_math(c, group) * "_{" * row * "}=" * index * "\\}")
    end
    for b in plate.bindings
        rows = _brmd_plate_rows_math(plate, b.name)
        push!(prose, _brmd_text(c, b.order === nothing ? :plate_rows : :plate_rows_ordered;
            name=b.name, group=b.group, cells=b.cells, order=b.order, index, rows))
        push!(equations, rows * "=\\{" * row * ":" * _brmd_identifier_math(c, b.group) * "_{" * row * "}=" *
            _brmd_identifier_math(c, b.cells) * "_{" * index * "}\\}")
    end
    positions = filter(r -> r.kind === :positions, plate.reads)
    isempty(positions) || push!(prose, _brmd_text(c, any(r -> r.zero, positions) ?
        :plate_positions_zero : :plate_positions))
    # The value the cell collects, unless an assignment above already states it.
    final = _brmd_kernel_final(c)
    isnothing(final) || final isa BRMDescriptionReference && final.axis === :cell &&
        final.name in plate.outputs || push!(equations, _brmd_assignment_equation(c,
            "{" * _brmd_identifier_math(c, owner) * "}_{" * index * "}", final))
    BRMDescriptionFragment(; prose=Tuple(prose), equations=Tuple(equations), covers=(c.id,))
end

# A cell latent `x ~ law` among the cell's statements, in the cell's names.
function _brmd_plate_latent_fragment(c)
    lhs, rhs = c.arguments
    BRMDescriptionFragment(prose=(_brmd_text(c, :cell_latent; name=lhs.name, owner=c.provenance.cell_owner),),
        equations=(_brmd_distribution_equation(c, brm_description_math(c, lhs), rhs),), covers=(c.id,))
end
