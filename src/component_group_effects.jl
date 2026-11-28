# A parent block spans the parent and whichever members explicitly select it.
# Every member has a margin in the shared draw matrix. Row and margin indices
# are bound data, including when some members have no observations.
struct _BRMMemberCodes <: AbstractVector{Int}
    rows::Vector{Int}
    extent::Symbol
    collection::Symbol
    keys::Vector{Union{Int,Symbol}}
    logical::Symbol
    predictor::Symbol
    id::Symbol
end
Base.size(c::_BRMMemberCodes) = size(c.rows)
Base.getindex(c::_BRMMemberCodes, i::Int) = c.rows[i]
Base.IndexStyle(::Type{_BRMMemberCodes}) = IndexLinear()
_brm_fit_levels(c::_BRMMemberCodes) = collect(eachindex(c.keys))

function _brm_parent_block(term)
    isxcall(term, :|) && length(term.args) == 3 || return nothing
    lhs, group = term.args[2:3]
    isxcall(lhs, :|) && length(lhs.args) == 3 || return nothing
    effect, marker = lhs.args[2:3]
    isxcall(marker, :parentblock) || return nothing
    length(marker.args) == 2 && marker.args[2] isa QuoteNode &&
        marker.args[2].value isa Symbol || throw(ArgumentError("@brm: use parentblock(:id) to name the parent's correlation block"))
    effect == 1 || throw(ArgumentError("@brm: parentblock currently takes an intercept margin, (1 | parentblock(:id) | group)"))
    (; id=marker.args[2].value, group)
end

# The name a member predictor binds: `x` for `x ~ ...` and for a linked
# `f(x) ~ ...`, which binds `x` on the response scale.
_brm_parent_target(lhs::Symbol) = lhs
_brm_parent_target(lhs) = Meta.isexpr(lhs, :call, 2) && lhs.args[2] isa Symbol ? lhs.args[2] : nothing

function _brm_parent_margins!(out, columns, occupied, loop, records, positions, keys, variant,
        selected, index, df, group_sources)
    declarations, ports, arguments = Any[], Symbol[], Any[]
    record = loop.component.inputs[loop.record]
    for stmt in selected
        if !isxcall(stmt, :~) || isnothing(_brm_parent_target(stmt.args[2]))
            push!(declarations, stmt)
            continue
        end
        lhs, formula = stmt.args[2:end]
        target = _brm_parent_target(lhs)
        terms = isxcall(formula, :+) ? formula.args[2:end] : Any[formula]
        shared = map(_brm_parent_block, terms)
        if all(isnothing, shared)
            push!(declarations, stmt)
            continue
        end
        fresh(stem) = _rk_ast_fresh_name(string(loop.target, "_", variant, "_", target, "_", stem), occupied)
        local_terms = terms[isnothing.(shared)]
        # A member with terms of its own keeps its authored predictor. Each
        # margin then enters as a fixed summand gathered at the member's own
        # rows, which also give the predictor its row axis. A linked predictor
        # is kept the same way, with `0` for its own terms when it has none,
        # so it still binds `x` on the response scale.
        own = !isempty(local_terms) && local_terms != [0]
        predictor = own || !(lhs isa Symbol)
        own || (local_terms = Any[0])
        summands = Any[]
        for spec in filter(!isnothing, shared)
            field = _brm_observer_field(spec.group, record)
            source, slices = if !isnothing(field)
                field, [_brm_record_field(records[i], field) for i in positions]
            else
                p = findfirst(isequal(spec.group), loop.component.inputs)
                isnothing(p) && throw(ArgumentError("@brm: parentblock groups come from a record field or a component input"))
                arg = loop.arguments[p]
                arg isa Symbol || throw(ArgumentError("@brm: parentblock needs a named grouping column"))
                arg, fill(getproperty(df, arg), length(positions))
            end
            all(x -> x isa AbstractVector, slices) || throw(ArgumentError("@brm: parentblock needs a grouping vector for every member"))
            group, margin, rows, effect, port = fresh(:group), fresh(:margin), fresh(:rows), fresh(:effect), fresh(:input)
            columns[group] = reduce(vcat, slices)
            isnothing(group_sources) || (group_sources[group] = source)
            columns[margin] = _BRMMemberCodes(reduce(vcat, [fill(i, length(x)) for (i,x) in enumerate(slices)]),
                fresh(:members), loop.target, keys[positions], target, effect, spec.id)
            ends = cumsum(length.(slices))
            columns[rows] = [collect(stop-length(x)+1:stop) for (stop,x) in zip(ends,slices)]
            push!(out, :($effect ~ 0 + (0 + $margin | $(spec.id) | $group)))
            if predictor
                member_rows = fresh(:input_rows)
                append!(ports, (port, member_rows))
                append!(arguments, (effect, :($rows[$index])))
                push!(summands, :(offset($port[$member_rows])))
            else
                push!(ports, port)
                push!(arguments, :($effect[$rows[$index]]))
                push!(summands, port)
            end
        end
        if predictor
            push!(declarations, Expr(:call, :~, lhs, Expr(:call, :+, local_terms..., summands...)))
        else
            value = length(summands) == 1 ? only(summands) : Expr(:call, :+, summands...)
            push!(declarations, Expr(:(=), target, value))
        end
    end
    declarations, ports, arguments
end
