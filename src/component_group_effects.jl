# A parent block spans the parent and whichever members explicitly select it.
# Every member has a margin in the shared draw matrix. Row and margin indices
# are bound data, including when some members have no observations.
struct _BRMMemberCodes <: AbstractVector{Int}
    rows::Vector{Int}
    count::Int
    extent::Symbol
end
Base.size(c::_BRMMemberCodes) = size(c.rows)
Base.getindex(c::_BRMMemberCodes, i::Int) = c.rows[i]
Base.IndexStyle(::Type{_BRMMemberCodes}) = IndexLinear()
_brm_fit_levels(c::_BRMMemberCodes) = collect(1:c.count)

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

function _brm_parent_margins!(out, columns, occupied, loop, records, positions, variant,
        selected, index, df)
    declarations, ports, arguments = Any[], Symbol[], Any[]
    record = loop.component.inputs[loop.record]
    for stmt in selected
        if !isxcall(stmt, :~) || !(stmt.args[2] isa Symbol)
            push!(declarations, stmt)
            continue
        end
        target, formula = stmt.args[2:end]
        terms = isxcall(formula, :+) ? formula.args[2:end] : Any[formula]
        shared = map(_brm_parent_block, terms)
        if all(isnothing, shared)
            push!(declarations, stmt)
            continue
        end
        fresh(stem) = _rk_ast_fresh_name(string(loop.target, "_", variant, "_", target, "_", stem), occupied)
        local_terms = terms[isnothing.(shared)]
        summands = Symbol[]
        if !isempty(local_terms) && local_terms != [0]
            local_target = fresh(:local)
            local_formula = length(local_terms) == 1 ? only(local_terms) : Expr(:call, :+, local_terms...)
            push!(declarations, Expr(:call, :~, local_target, local_formula))
            push!(summands, local_target)
        end
        for spec in filter(!isnothing, shared)
            field = _brm_observer_field(spec.group, record)
            slices = if !isnothing(field)
                [_brm_record_field(records[i], field) for i in positions]
            else
                p = findfirst(isequal(spec.group), loop.component.inputs)
                isnothing(p) && throw(ArgumentError("@brm: parentblock groups come from a record field or a component input"))
                arg = loop.arguments[p]
                arg isa Symbol || throw(ArgumentError("@brm: parentblock needs a named grouping column"))
                fill(getproperty(df, arg), length(positions))
            end
            all(x -> x isa AbstractVector, slices) || throw(ArgumentError("@brm: parentblock needs a grouping vector for every member"))
            group, margin, rows, effect, port = fresh(:group), fresh(:margin), fresh(:rows), fresh(:effect), fresh(:input)
            columns[group] = reduce(vcat, slices)
            columns[margin] = _BRMMemberCodes(reduce(vcat, [fill(i, length(x)) for (i,x) in enumerate(slices)]),
                length(slices), fresh(:members))
            ends = cumsum(length.(slices))
            columns[rows] = [collect(stop-length(x)+1:stop) for (stop,x) in zip(ends,slices)]
            push!(out, :($effect ~ 0 + (0 + $margin | $(spec.id) | $group)))
            push!(ports, port)
            push!(arguments, :($effect[$rows[$index]]))
            push!(summands, port)
        end
        value = length(summands) == 1 ? only(summands) : Expr(:call, :+, summands...)
        push!(declarations, Expr(:(=), target, value))
    end
    declarations, ports, arguments
end
