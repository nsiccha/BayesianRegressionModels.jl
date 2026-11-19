# A declaration guard chooses model structure from the component's explicit
# data inputs. Ordinary Julia expressions in a formula or a plate cell retain
# their existing meaning. Only statement-level if/elseif/else is selected here.
struct _BRMGuardedComponentBody
    statements::Expr
    inputs::Tuple{Vararg{Symbol}}
    scope::Module
end

_brm_declaration_guard(stmt) = Meta.isexpr(_brm_undoc(stmt), (:if, :elseif))

function _brm_guard_input(input, value)
    value isa NamedColumn && (value = parent(value))
    value isa DataColumn && return parent(value)
    value isa AbstractColumn && throw(ArgumentError(
        "@brm: declaration guard reads `$input`, which must be supplied as data"))
    value
end

function _brm_guard_value(expr, inputs)
    if expr isa Symbol
        haskey(inputs, expr) || throw(ArgumentError(
            "@brm: declaration guard reads `$expr`, which is not a component input"))
        return _brm_guard_input(expr, inputs[expr])
    elseif expr isa QuoteNode
        return expr.value
    elseif !(expr isa Expr)
        return expr
    elseif Meta.isexpr(expr, :block)
        # Julia wraps an elseif condition and its source location in a block.
        return _brm_guard_value(only(arg for arg in expr.args if !(arg isa LineNumberNode)), inputs)
    elseif Meta.isexpr(expr, :., 2) && expr.args[2] isa QuoteNode
        record = _brm_guard_value(expr.args[1], inputs)
        field = expr.args[2].value
        return record isa AbstractDict ? record[field] : getproperty(record, field)
    elseif Meta.isexpr(expr, :&&, 2)
        return _brm_guard_value(expr.args[1], inputs) && _brm_guard_value(expr.args[2], inputs)
    elseif Meta.isexpr(expr, :||, 2)
        return _brm_guard_value(expr.args[1], inputs) || _brm_guard_value(expr.args[2], inputs)
    elseif Meta.isexpr(expr, :call) && first(expr.args) in (:!, :(==), :(!=), :(===), :(!==), :<, :<=, :>, :>=)
        op = getfield(Base, first(expr.args))
        return op((_brm_guard_value(arg, inputs) for arg in expr.args[2:end])...)
    end
    throw(ArgumentError("@brm: expected a data flag or comparison in a declaration guard, got `$expr`"))
end

function _brm_guarded_statements(body::_BRMGuardedComponentBody, inputs)
    _brm_select_declarations!(Any[], body.statements, inputs)
end

function _brm_select_declarations!(out, statement, inputs)
    if Meta.isexpr(statement, :block)
        for child in statement.args
            _brm_select_declarations!(out, child, inputs)
        end
    elseif _brm_declaration_guard(statement)
        condition, yes, other... = _brm_undoc(statement).args
        flag = _brm_guard_value(condition, inputs)
        flag isa Bool || throw(ArgumentError("@brm: declaration guard `$condition` must be Bool"))
        if flag
            _brm_select_declarations!(out, yes, inputs)
        elseif !isempty(other)
            _brm_select_declarations!(out, only(other), inputs)
        end
    elseif !(statement isa LineNumberNode)
        push!(out, statement)
    end
    out
end

function (body::_BRMGuardedComponentBody)(df)
    columns = data(df)
    inputs = Dict(key => getproperty(columns, key) for key in body.inputs)
    selected = _brm_guarded_statements(body, inputs)
    brm(Expr(:block, selected...); mod=body.scope)(df)
end
