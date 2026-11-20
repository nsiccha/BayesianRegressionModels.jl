# The declarations of a member run once per record. Its observation clause
# runs inside a caller's subject plate, where the shared trajectory is in scope.
_brm_is_observer(st) = Meta.isexpr(st, :macrocall) && first(st.args) === Symbol("@observe")
_brm_observe_call(st) = isxcall(st, :observe)
_brm_has_observer(st) = st isa Expr && (_brm_is_observer(st) || any(_brm_has_observer, st.args))

struct _BRMObservedMemberRows{T,A<:AbstractVector{T},L} <: AbstractVector{T}
    rows::A
    layout::L
end
Base.size(r::_BRMObservedMemberRows) = size(r.rows)
Base.getindex(r::_BRMObservedMemberRows, i::Int) = r.rows[i]
Base.IndexStyle(::Type{<:_BRMObservedMemberRows}) = IndexLinear()
_brm_data_vec(::Symbol, r::_BRMObservedMemberRows) = r.rows

# One row of each selected member's vector. The subject may read several rows
# of one member and none of another. The indices are ordinary data.
_brm_observed_member_rows(values, members, rows) =
    [values[members[i]][rows[i]] for i in eachindex(members, rows)]

function _brm_strip_observers(st)
    _brm_is_observer(st) && return Expr(:block)
    if Meta.isexpr(st, (:if, :elseif, :block))
        return Expr(st.head, map(_brm_strip_observers, st.args)...)
    end
    st
end

function _brm_observation_inventory(st, loops)
    st isa Expr || return st
    if _brm_observe_call(st)
        length(st.args) == 4 || throw(ArgumentError("@brm: use observe(collection, subject, trajectory) in a subject plate"))
        _, target, subject, trajectory = st.args
        pos = findfirst(loop -> loop.target === target, loops)
        isnothing(pos) && throw(ArgumentError("@brm: unknown observation collection `$target`"))
        _brm_has_observer(loops[pos].component.source) || throw(ArgumentError(
            "@brm: component `$(loops[pos].component.name)` needs an @observe clause for observe($target, ... )"))
        # The binding inventory only needs the parent inputs. This response
        # witnesses the record source even in a plate with no other observation.
        return :($(Expr(:ref, loops[pos].source, subject)) ~ Normal($trajectory, 1))
    end
    Expr(st.head, map(arg -> _brm_observation_inventory(arg, loops), st.args)...)
end

function _brm_observer_field(ex, record)
    Meta.isexpr(ex, :., 2) && first(ex.args) === record && last(ex.args) isa QuoteNode ?
        last(ex.args).value : nothing
end

# Each record supplies its own observation rows per subject. Concatenate the
# observations of the records with this declaration, keeping a member index
# on every row. Empty row sets have zero likelihood contributions.
function _brm_observe_variant!(spec, subject, walk, columns, occupied, before_plate)
    (; loop, records, positions, result, variant, observer) = spec
    length(observer.args) == 6 || throw(ArgumentError(
        "@brm: write `@observe subject trajectory member begin ... end` in a component"))
    s, w, member, body = observer.args[3:end]
    all(x -> x isa Symbol, (s,w,member)) && allunique((s,w,member)) || throw(ArgumentError(
        "@brm: the three observation clause arguments must be distinct names"))
    statements = _brm_source_statements!(Any[], body)
    responses = filter(st -> isxcall(st, :~), statements)
    length(responses) == 1 || throw(ArgumentError("@brm: an @observe clause observes one response"))
    lhs = only(responses).args[2]
    record = loop.component.inputs[loop.record]
    response = Meta.isexpr(lhs, :ref, 2) && last(lhs.args) === s ?
        _brm_observer_field(first(lhs.args), record) : nothing
    isnothing(response) && throw(ArgumentError("@brm: observe an entry's response as `$record.y[$s] ~ law`"))
    rows = [_brm_record_field(records[i], response) for i in positions]
    n = length(first(rows))
    all(y -> length(y) == n, rows) || throw(DimensionMismatch("component observations disagree on the number of subjects"))
    width(cell) = cell isa AbstractVector ? length(cell) : 1
    fresh(part) = _rk_ast_fresh_name(string(loop.target, "_", variant, "_", part), occupied)
    index = fresh(:observation_member)
    columns[index] = [reduce(vcat, (fill(i, width(y[j])) for (i,y) in enumerate(rows))) for j in 1:n]
    ports = Dict{Tuple{Symbol,Bool},Symbol}()
    function sliced(field; by_subject=true)
        name = get!(ports, (field, by_subject)) do
            name = fresh(field)
            inputs = [_brm_record_field(records[i], field) for i in positions]
            !by_subject || all(x -> length(x) == n, inputs) ||
                throw(DimensionMismatch("observation field `$field` has a different subject axis"))
            columns[name] = map(1:n) do j
                reduce(vcat, map(eachindex(inputs)) do i
                    cell = by_subject ? inputs[i][j] : inputs[i]
                    size = width(rows[i][j])
                    cell isa AbstractVector || return fill(cell, size)
                    length(cell) == size || throw(DimensionMismatch("observation field `$field` has a different row count from `$response`"))
                    cell
                end)
            end
            name
        end
        Expr(:ref, name, subject)
    end
    locals = Dict(st.args[1] => fresh(st.args[1]) for st in statements
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol)
    vector_reads = Dict{Tuple{Symbol,Any},Expr}()
    function read_vector(returned, selection)
        get!(vector_reads, (returned, selection)) do
            # Read the collected vectors in model scope, before either
            # backend flattens the producer for a row-aligned likelihood.
            picks = if selection isa Integer
                [fill(selection, length(ids)) for ids in columns[index]]
            elseif Meta.isexpr(selection, :ref, 2) && selection.args[2] === subject &&
                    haskey(columns, first(selection.args))
                columns[first(selection.args)]
            else
                throw(ArgumentError("@brm: index the returned curve with one data row per observation, such as `member[setting.pick[s]]`"))
            end
            all(row -> all(i -> i isa Integer && i > 0, row), picks) ||
                throw(ArgumentError("@brm: member curve row indices must be positive integers"))
            members, indices, segments, values = fresh(:member_ids), fresh(:curve_rows),
                fresh(:subject_rows), fresh(:read_values)
            columns[members] = reduce(vcat, columns[index]; init=Int[])
            columns[indices] = reduce(vcat, picks; init=Int[])
            ends = cumsum(length.(picks))
            columns[segments] = [collect(stop-length(pick)+1:stop) for (stop,pick) in zip(ends,picks)]
            call = Expr(:call, GlobalRef(@__MODULE__, :_brm_observed_member_rows),
                returned, members, indices)
            push!(before_plate, Expr(:(=), values, call))
            Expr(:ref, values, Expr(:ref, segments, subject))
        end
    end
    function rewrite(ex)
        ex === w && return walk
        if ex === member
            result isa Symbol || throw(ArgumentError("@brm: read a returned member field in the @observe clause"))
            return Expr(:ref, result, Expr(:ref, index, subject))
        end
        ex isa Symbol && return get(locals, ex, ex)
        ex isa Expr || return ex
        if Meta.isexpr(ex, :ref, 2)
            object, selection = ex.args
            field = _brm_observer_field(object, member)
            if object === member || (!isnothing(field) && result isa NamedTuple)
                returned = object === member ? result : result[field]
                returned isa Symbol || throw(ArgumentError("@brm: index one returned member field"))
                return read_vector(returned, rewrite(selection))
            end
        end
        field = _brm_observer_field(ex, member)
        if !isnothing(field) && result isa NamedTuple
            haskey(result, field) || throw(ArgumentError("@brm: unknown returned member field `$field`"))
            return Expr(:ref, result[field], Expr(:ref, index, subject))
        end
        if Meta.isexpr(ex, :ref, 2) && last(ex.args) === s
            field = _brm_observer_field(first(ex.args), record)
            isnothing(field) || return sliced(field)
        end
        setting = _brm_observer_field(ex, record)
        isnothing(setting) || return sliced(setting; by_subject=false)
        Expr(ex.head, map(rewrite, ex.args)...)
    end
    result_statements = map(rewrite, statements)
    source = ports[(response, true)]
    layout = (; collection=loop.target, spec.keys, response, source,
        members=reduce(vcat, columns[index]; init=Int[]))
    columns[source] = _BRMObservedMemberRows(columns[source], layout)
    result_statements
end

function _brm_expand_observers(st, observers, columns, occupied, before_plate)
    Meta.isexpr(st, :macrocall) && first(st.args) === Symbol("@plate") || return st
    loop = last(st.args)
    subject, _, body = _brm_plate_loop(st)
    statements = _brm_source_statements!(Any[], body)
    any(_brm_observe_call, statements) || return st
    rewritten = Any[]
    for stmt in statements
        if _brm_observe_call(stmt)
            _, collection, index, walk = stmt.args
            index === subject || throw(ArgumentError("@brm: observe uses the enclosing subject index `$subject`"))
            for spec in observers[collection]
                append!(rewritten, _brm_observe_variant!(spec, subject, walk, columns, occupied, before_plate))
            end
        else
            push!(rewritten, stmt)
        end
    end
    # Removing an empty collection also removes its observation-only plate.
    any(x -> isxcall(x, :~) || (Meta.isexpr(x, :(=)) && _brm_plate_indexed(first(x.args), subject)), rewritten) ||
        return Expr(:block)
    Expr(:macrocall, st.args[1:end-1]..., Expr(:for, first(loop.args), Expr(:block, rewritten...)))
end
