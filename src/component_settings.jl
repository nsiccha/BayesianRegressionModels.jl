# A declaration guard chooses model structure from the component's explicit
# data inputs. Ordinary Julia expressions in a formula or a plate cell retain
# their existing meaning. Only statement-level if/elseif/else is selected here.
struct _BRMGuardedComponentBody
    statements::Expr
    inputs::Tuple{Vararg{Symbol}}
    scope::Module
end

_brm_declaration_guard(stmt) = Meta.isexpr(_brm_undoc(stmt), (:if, :elseif))

# A collection binds record fields to column names before lowering its plates.
# At component definition time the record is still a formal argument.
function _brm_plate_slices_input_field(stmt, inputs)
    plate = _brm_undoc(stmt)
    _is_brm_plate(plate) || return false
    index, range, body = _brm_plate_loop(plate)
    field(x) = Meta.isexpr(x, :., 2) && x.args[1] in inputs && x.args[2] isa QuoteNode
    slices(x) = x isa Expr && ((_brm_plate_indexed(x, index) && field(first(x.args))) ||
        any(slices, x.args))
    (Meta.isexpr(range, :call) && any(field, range.args[2:end])) || slices(body)
end

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
        return _brm_record_field(record, field)
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

function _brm_select_declarations!(out, statement, inputs, decisions=nothing)
    if Meta.isexpr(statement, :block)
        for child in statement.args
            _brm_select_declarations!(out, child, inputs, decisions)
        end
    elseif _brm_declaration_guard(statement)
        condition, yes, other... = _brm_undoc(statement).args
        flag = _brm_guard_value(condition, inputs)
        flag isa Bool || throw(ArgumentError("@brm: declaration guard `$condition` must be Bool"))
        isnothing(decisions) || push!(decisions, flag)
        if flag
            _brm_select_declarations!(out, yes, inputs, decisions)
        elseif !isempty(other)
            _brm_select_declarations!(out, only(other), inputs, decisions)
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
    any(_brm_has_record_rhs, selected) && throw(ArgumentError(
        "@brm: rhs(record.field) requires a component bound in a record collection"))
    brm(Expr(:block, selected...); mod=body.scope)(df)
end

# A record supplies fixed-effect syntax as data. Only the following term
# grammar is admitted. The record's text and expressions never run as Julia.
_brm_is_record_rhs(x) = isxcall(x, :rhs) && length(x.args) == 2 &&
    Meta.isexpr(x.args[2], :., 2) && x.args[2].args[1] isa Symbol && x.args[2].args[2] isa QuoteNode
_brm_has_record_rhs(x) = _brm_is_record_rhs(x) ||
    (x isa Expr && any(_brm_has_record_rhs, x.args))

function _brm_record_term(x, formal, field)
    x isa Symbol && Base.isidentifier(string(x)) && return Expr(:., formal, QuoteNode(x))
    x isa Integer && !(x isa Bool) && x in (0, 1) && return Int(x)
    if Meta.isexpr(x, :call) && length(x.args) >= 3 && first(x.args) in (:+, :&)
        return Expr(:call, first(x.args), (_brm_record_term(a, formal, field) for a in x.args[2:end])...)
    elseif Meta.isexpr(x, :call, 2) && first(x.args) in (:center, :zscale, :standardize) &&
            last(x.args) isa Symbol
        return Expr(:call, first(x.args), _brm_record_term(last(x.args), formal, field))
    elseif Meta.isexpr(x, :ref, 2) && last(x.args) isa Symbol
        return Expr(:ref, (_brm_record_term(a, formal, field) for a in x.args)...)
    end
    throw(ArgumentError("@brm: rhs($formal.$field) expects a fixed-effect term, got `$x`"))
end

# Splice additive fragments into the predictor's summands so a neighbouring
# parentblock, for example, still sees the ordinary flat formula.
function _brm_record_summands!(out, term, formal, record, signature)
    if isxcall(term, :+)
        foreach(t -> _brm_record_summands!(out, t, formal, record, signature), term.args[2:end])
    elseif _brm_is_record_rhs(term)
        field = _brm_observer_field(last(term.args), formal)
        isnothing(field) && throw(ArgumentError("@brm: use rhs($formal.field) in a member predictor"))
        raw = _brm_record_field(record, field)
        parsed = raw isa AbstractString ? Meta.parse(raw; raise=false) : raw
        Meta.isexpr(parsed, (:error, :incomplete)) && throw(ArgumentError(
            "@brm: invalid formula text in rhs($formal.$field): $(repr(raw))"))
        fragment = _brm_record_term(parsed, formal, field)
        terms = Any[]
        _brm_record_summands!(terms, fragment, formal, record, signature)
        push!(signature, repr(terms))
        append!(out, terms)
    elseif _brm_has_record_rhs(term)
        throw(ArgumentError("@brm: rhs($formal.field) must be an additive predictor term"))
    else
        push!(out, term)
    end
    out
end

function _brm_select_record_rhs(stmt, formal, record, signature)
    _brm_has_record_rhs(stmt) || return stmt
    if _brm_is_doc(stmt)
        return Expr(stmt.head, stmt.args[1:3]...,
            _brm_select_record_rhs(stmt.args[4], formal, record, signature))
    end
    isxcall(stmt, :~) && length(stmt.args) == 3 && !_brm_has_record_rhs(stmt.args[2]) ||
        throw(ArgumentError("@brm: use rhs($formal.field) in a member predictor declaration"))
    terms = _brm_record_summands!(Any[], stmt.args[3], formal, record, signature)
    Expr(:call, :~, stmt.args[2], length(terms) == 1 ? only(terms) : Expr(:call, :+, terms...))
end
