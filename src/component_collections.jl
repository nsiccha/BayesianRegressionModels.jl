# A model-level component loop allocates over its records, independently of
# any observation plate. Emit one native plate for each selected declaration
# body. Record keys, counts and numeric fields stay in the bound design.
struct _BRMCollectionLoop
    item::Symbol
    source::Symbol
    target::Symbol
    key::Union{Nothing,Symbol}
    component::BRMComponent
    arguments::Vector{Any}
    record::Int
end

struct _BRMCollectionBuilder
    statements::Vector{Any}
    scope::Module
    occupied::Set{Symbol}
end
Base.deepcopy_internal(b::_BRMCollectionBuilder, seen::IdDict) =
    _BRMCollectionBuilder(Base.deepcopy_internal(b.statements, seen), b.scope, copy(b.occupied))

function _brm_collection_source(source::Expr)
    body = source.head === :(=) && Meta.isexpr(first(source.args), :call) ?
        last(source.args) : source
    Meta.isexpr(body, :for) && return Expr(:block, body)
    Meta.isexpr(body, :block) &&
        any(st -> Meta.isexpr(st, :for), body.args) ? body : nothing
end

function _brm_collection_loop(statement, scope)
    header, body = statement.args
    Meta.isexpr(header, :(=), 2) && all(x -> x isa Symbol, header.args) ||
        throw(ArgumentError("@brm: use `for item in records` for a component collection"))
    item, source = header.args
    statements = _brm_source_statements!(Any[], body)
    length(statements) == 1 && isxcall(only(statements), :~) || throw(ArgumentError(
        "@brm: a component loop binds one member with `blocks[item.id] ~ component(item)`"))
    _, lhs, rhs = only(statements).args
    Meta.isexpr(lhs, :ref, 2) && first(lhs.args) isa Symbol || throw(ArgumentError(
        "@brm: a collection member is bound as `blocks[item.id]` or `blocks[item]`"))
    target, index = lhs.args
    key = if index === item
        nothing  # use collection positions
    elseif Meta.isexpr(index, :., 2) && first(index.args) === item && last(index.args) isa QuoteNode
        last(index.args).value
    else
        throw(ArgumentError("@brm: a collection key must be a field of `$item`"))
    end
    component = Meta.isexpr(rhs, :call) ? _sb_cell_binding(scope, first(rhs.args)) : nothing
    component isa BRMComponent || throw(ArgumentError("@brm: a component loop calls an @brm function"))
    arguments = rhs.args[2:end]
    record = findall(isequal(item), arguments)
    length(record) == 1 && length(arguments) == length(component.inputs) &&
        isempty(component.keywords) || throw(ArgumentError(
        "@brm: pass the loop record once and supply the component's positional inputs"))
    _BRMCollectionLoop(item, source, target, key, component, arguments, only(record))
end

function _brm_collection_builder(source, scope)
    statements = Any[Meta.isexpr(st, :for) ? _brm_collection_loop(st, scope) : st
        for st in _brm_source_statements!(Any[], source)]
    loops = filter(st -> st isa _BRMCollectionLoop, statements)
    names = [loop.target for loop in loops]
    allunique(names) || throw(ArgumentError("@brm: duplicate component collection binding"))
    plain = Expr(:block, (st for st in statements if !(st isa _BRMCollectionLoop))...)
    columns = setdiff(collect(Symbol, brm_columns(brm(plain; mod=scope))), names)
    for loop in loops
        push!(columns, loop.source)
        for arg in loop.arguments
            arg isa Symbol && arg !== loop.item && push!(columns, arg)
        end
    end
    occupied = _brm_cell_value_refs!(Set{Symbol}(columns), source)
    BRMBuilder(_BRMCollectionBuilder(statements, scope, occupied), source,
        Tuple(sort!(unique!(columns))), scope)
end

_brm_record_field(record, key) = getproperty(record, key)
_brm_record_field(record::AbstractDict, key) = record[key]

# A numeric cell axis is also the provenance for rebuilding columns from a new
# table of records. It is an ordinary integer vector to the two emitters.
struct _BRMCollectionAxis <: AbstractVector{Int}
    builder::_BRMCollectionBuilder
    collection::Symbol
    keys::Vector{Union{Int,Symbol}}
    positions::Vector{Int}
    path::Tuple{Vararg{Symbol}}
end
Base.size(axis::_BRMCollectionAxis) = size(axis.positions)
Base.getindex(axis::_BRMCollectionAxis, i::Int) = axis.positions[i]
Base.IndexStyle(::Type{_BRMCollectionAxis}) = IndexLinear()
_brm_data_vec(::Symbol, axis::_BRMCollectionAxis) = axis.positions

function _brm_record_keys(loop, records)
    keys = Union{Int,Symbol}[]
    for (i, record) in enumerate(records)
        key = isnothing(loop.key) ? i : _brm_record_field(record, loop.key)
        key isa Union{Symbol,AbstractString,Integer} || throw(ArgumentError(
            "@brm: collection `$(loop.target)` needs a symbol, string or integer key, got $(repr(key))"))
        push!(keys, key isa AbstractString ? Symbol(key) : key)
    end
    allunique(keys) || throw(ArgumentError("@brm: duplicate keys in collection `$(loop.target)`"))
    keys
end

# A field read becomes a data port of this variant. Neither the label nor the
# field value becomes a literal in the emitted program.
function _brm_project_record(expr, record, fields, occupied)
    if Meta.isexpr(expr, :., 2) && expr.args[1] === record && expr.args[2] isa QuoteNode
        key = expr.args[2].value
        return get!(fields, key) do
            _rk_ast_fresh_name(string(record, "_", key), occupied)
        end
    elseif expr === record
        throw(ArgumentError("@brm: use a field of component record `$record` in a numeric declaration"))
    elseif expr isa Expr
        return Expr(expr.head, (_brm_project_record(arg, record, fields, occupied) for arg in expr.args)...)
    end
    expr
end

function _brm_variant_component(loop, selected, value)
    original = loop.component
    formal = original.inputs[loop.record]
    fields = OrderedDict{Symbol,Symbol}()
    occupied = _brm_cell_value_refs!(Set{Symbol}(original.inputs), Expr(:block, selected..., value))
    body = _brm_project_record(Expr(:block, selected..., Expr(:return, value)), formal, fields, occupied)
    inputs = [s for s in original.inputs if s !== formal]
    append!(inputs, values(fields))
    source = Expr(:function, Expr(:call, original.name, inputs...), body)
    _brm_component(source, original.scope), fields
end

function _brm_collection_variant!(out, frame, builder, occupied, loop, records, keys,
        positions, selected, value, variant)
    name = loop.target
    fresh(parts...) = _rk_ast_fresh_name(string(parts...), occupied)
    axis_name = fresh(name, "_axis_", variant)
    fields_prefix = string(name, "_", variant, "_")
    component, fields = _brm_variant_component(loop, selected, value)
    index = fresh("brm_entry")
    args = Any[arg for (i, arg) in enumerate(loop.arguments) if i != loop.record]
    for (key, formal) in fields
        column = fresh(fields_prefix, formal)
        frame[column] = [_brm_record_field(records[i], key) for i in positions]
        push!(args, Expr(:ref, column, index))
    end
    member = fresh(name, "_member_", variant)
    result = fresh(name, "_result_", variant)
    frame[axis_name] = _BRMCollectionAxis(builder, name, keys[positions], positions, (result, member))
    call = Expr(:call, component, args...)
    body = Expr(:block, Expr(:call, :~, member, call),
        Expr(:(=), Expr(:ref, result, index), member))
    header = Expr(:(=), index, Expr(:call, :eachindex, axis_name))
    push!(out, Expr(:macrocall, Symbol("@plate"), nothing, Expr(:for, header, body)))
    result
end

function _brm_lower_collection!(out, frame, builder, occupied, loop, df)
    records = getproperty(df, loop.source)
    records isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("@brm: `$(loop.source)` must be a vector or tuple of records"))
    keys = _brm_record_keys(loop, records)
    (; body, value) = _brm_component_parts(loop.component.source)
    input = loop.component.inputs[loop.record]
    groups = Dict{Tuple,Vector{Int}}()
    selected_bodies = Dict{Tuple,Vector{Any}}()
    for (i, record) in enumerate(records)
        decisions = Bool[]
        selected = _brm_select_declarations!(Any[], Expr(:block, body...), Dict(input => record), decisions)
        group = Tuple(decisions)
        push!(get!(Vector{Int}, groups, group), i)
        get!(selected_bodies, group, selected)
    end
    if isempty(groups)
        frame[loop.target] = _BRMCollectionAxis(builder, loop.target, keys, Int[], ())
        return
    end
    results, order = Symbol[], Int[]
    for group in sort!(collect(Base.keys(groups)))
        variant = isempty(group) ? :all : Symbol(join(Int.(group)))
        # Native coordinates follow the stable keys, while the returned vector
        # below follows the supplied records. Reordering a labelled design
        # therefore never exchanges two fitted parameters.
        positions = groups[group]
        sort!(positions; by=i -> (keys[i] isa Symbol, keys[i]))
        push!(results, _brm_collection_variant!(out, frame, builder, occupied, loop, records, keys,
            positions, selected_bodies[group], value, variant))
        append!(order, positions)
    end
    permutation = _rk_ast_fresh_name(string(loop.target, "_row_order"), occupied)
    frame[permutation] = invperm(order)
    joined = length(results) == 1 ? only(results) : Expr(:call, :vcat, results...)
    push!(out, Expr(:(=), loop.target, Expr(:ref, joined, permutation)))
end

function _brm_collection_design(builder, df)
    statements, columns = Any[], Dict{Symbol,Any}()
    occupied = copy(builder.occupied)
    for st in builder.statements
        if st isa _BRMCollectionLoop
            _brm_lower_collection!(statements, columns, builder, occupied, st, df)
        else
            push!(statements, st)
        end
    end
    Expr(:block, statements...), columns
end

function (builder::_BRMCollectionBuilder)(df)
    source, columns = _brm_collection_design(builder, df)
    brm(source; mod=builder.scope)(_BRMReplayData(df, columns))
end

function _brm_collection_axes(brmi)
    axes = _BRMCollectionAxis[]
    for column in _brm_operation_values(brmi)
        column isa NamedColumn && parent(column) isa DataColumn || continue
        raw = parent(parent(column))
        raw isa _BRMCollectionAxis && push!(axes, raw)
    end
    axes
end

# The backend plate has one native component call per declaration variant,
# with its first dimension indexing the records. Present the authored paths.
function _brm_collection_paths(brmi)
    Dict(axis.path => axis for axis in _brm_collection_axes(brmi) if !isempty(axis.path))
end

function _brm_collection_public_records(brmi, records)
    axes = _brm_collection_paths(brmi)
    isempty(axes) && return map(_brm_public_component_record, records)
    result = NamedTuple[]
    for record in records
        native = _brm_public_component_record(record)
        axis = length(record.path) < 2 ? nothing : get(axes, record.path[1:2], nothing)
        if isnothing(axis)
            push!(result, native)
            continue
        end
        loop = only(st for st in axis.builder.statements
            if st isa _BRMCollectionLoop && st.target === axis.collection)
        for key in axis.keys
            path = (axis.collection, key, record.path[3:end]...)
            definition = length(record.path) == 2 ? loop.component : native.definition
            push!(result, (; native..., path, definition, source=brm_source(definition)))
        end
    end
    Tuple(result)
end

function _brm_collection_replay_data(brmi, df)
    axes = _brm_collection_axes(brmi)
    isempty(axes) && return df
    _, columns = _brm_collection_design(first(axes).builder, df)
    layout(axes) = Dict((axis.collection, axis.path) => axis.keys for axis in axes)
    training = layout(axes)
    rebound = layout(value for value in values(columns) if value isa _BRMCollectionAxis)
    training == rebound || throw(ArgumentError(
        "component collection membership or declarations changed; build a new model for these settings"))
    _BRMReplayData(df, columns)
end
