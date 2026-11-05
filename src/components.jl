"""
    @brm function name(inputs...)
        # BRM declarations, local to each call
        return value
    end

A reusable BRM component with explicit inputs and a returned value. Embed it
with `value ~ name(inputs...)`. Each call owns its declarations; its arguments
refer to the parent's values. Both StanBlocks and ReactiveKernels lower the
same component through their native submodel scopes.

A component may return several named values as a named tuple,
`return (; recovery, decay = exp(log_decay))`. The parent then reads each one
as a property of the call, `kin.recovery`, with its own row axis; the call's
priors and correlations remain one scope.

The short definition `@brm model(df) = begin … end` retains its existing
meaning: a data-to-model builder.
"""
struct BRMComponent <: Function
    name::Symbol
    inputs::Tuple{Vararg{Symbol}}
    keywords::Tuple{Vararg{Symbol}}
    defaults::Dict{Symbol,Any}
    source::Expr
    builder::BRMBuilder
    # One local result name, or a named tuple from each returned field to its
    # local result name.
    result::Union{Symbol,NamedTuple}
    scope::Module
end

const _SB_COMPONENT_NAMES_KEY = :__brm_component_names__

"""The local result names of a component, one per returned value."""
_brm_component_results(component::BRMComponent) =
    component.result isa Symbol ? (component.result,) : Tuple(values(component.result))

"""The returned fields of a component, or `nothing` for a single returned value."""
_brm_component_fields(component::BRMComponent) =
    component.result isa Symbol ? nothing : component.result

# A returned named tuple is written `(; a, b = expr)` or `(a = x, b = y)`.
# Its field names are the parent's property names; a positional tuple has none.
function _brm_component_return_fields(value)
    Meta.isexpr(value, :tuple) || return nothing
    fields = Pair{Symbol,Any}[]
    function field!(item)
        if item isa Symbol
            push!(fields, item => item)
        elseif Meta.isexpr(item, (:kw, :(=)), 2) && first(item.args) isa Symbol
            push!(fields, first(item.args) => last(item.args))
        else
            throw(ArgumentError("@brm: a component returns one value or a named tuple " *
                "`(; a, b = expr)`; `$(sprint(Base.show_unquoted, item))` names no field"))
        end
    end
    for arg in value.args
        Meta.isexpr(arg, :parameters) ? foreach(field!, arg.args) :
            Meta.isexpr(arg, :(=), 2) ? field!(arg) :
            throw(ArgumentError("@brm: a component returns one value or a named tuple " *
                "`(; a, b = expr)`; a positional tuple has no field names for the parent to read"))
    end
    isempty(fields) && throw(ArgumentError("@brm: a component returns at least one value"))
    allunique(first.(fields)) || throw(ArgumentError(
        "@brm: duplicate returned field in `$(sprint(Base.show_unquoted, value))`"))
    fields
end

# Native return statements. A parent's `kin.recovery` reads the field's private
# local result name (`_brm_component_field_read`). StanBlocks returns a Stan
# tuple labelled by those names (`kin.1`). RKPPL reads a call's local directly
# (`kin.component_result_recovery`, rkppl-use §12), so its submodel returns one
# value and every field stays a local of the call.
_sb_component_return(component::BRMComponent) = component.result isa Symbol ?
    Expr(:return, component.result) :
    Expr(:return, Expr(:tuple, Expr(:parameters,
        (Expr(:kw, local_name, local_name) for local_name in values(component.result))...)))
_rk_component_return(component::BRMComponent) =
    Expr(:return, first(_brm_component_results(component)))

# A prepared read of one returned field: the RK emitter spells it as a native
# local read `call.label`.
_brm_component_field(value, label::Symbol) = getproperty(value, label)

function _brm_component_parts(source::Expr)
    source = _brm_normalize_docs(_brm_undoc(source))
    signature, raw = source.args
    Meta.isexpr(signature, :call) && first(signature.args) isa Symbol ||
        throw(ArgumentError("@brm: a component needs a named argument list"))
    positional = Symbol[]
    keywords = Symbol[]
    defaults = Dict{Symbol,Any}()
    function input!(names, input)
        if Meta.isexpr(input, :kw, 2) && first(input.args) isa Symbol
            key, value = input.args
            defaults[key] = value
            push!(names, key)
        elseif input isa Symbol
            push!(names, input)
        else
            throw(ArgumentError("@brm: a component input is a name or `name = default`; " *
                "typed dispatch and variadic component signatures are not defined"))
        end
    end
    for input in signature.args[2:end]
        Meta.isexpr(input, :parameters) ? foreach(x -> input!(keywords, x), input.args) :
            input!(positional, input)
    end
    inputs = Tuple((positional..., keywords...))
    allunique(inputs) || throw(ArgumentError("@brm: duplicate component input"))
    body = _brm_source_statements!(Any[], raw)
    !isempty(body) && Meta.isexpr(last(body), :return, 1) ||
        throw(ArgumentError("@brm: a component ends with `return value`"))
    any(statement -> Meta.isexpr(statement, :return), body[1:end-1]) &&
        throw(ArgumentError("@brm: a component has one final return"))
    (; name=first(signature.args), inputs, keywords=Tuple(keywords), defaults,
        body=body[1:end-1], value=only(last(body).args))
end

function _brm_component_source(source)
    source isa Expr || return nothing
    if source.head === :block
        statements = _brm_source_statements!(Any[], _brm_normalize_docs(source))
        length(statements) == 1 || return nothing
        return _brm_component_source(only(statements))
    end
    Meta.isexpr(_brm_undoc(source), :function) ? source : nothing
end

function _brm_component_definition(source::Expr)
    parts = _brm_component_parts(source)
    previous = gensym(:previous_documentation)
    scope = Expr(:macrocall, Symbol("@__MODULE__"), LineNumberNode(0))
    binding = Expr(:(=), parts.name, Expr(:call, _brm_component,
        QuoteNode(deepcopy(source)), scope))
    marked = Expr(:macrocall, GlobalRef(Core, Symbol("@__doc__")), LineNumberNode(0), binding)
    quote
        local $previous = $_brm_binding_docstring($scope, $(QuoteNode(parts.name)))
        $marked
        $(parts.name) = $_brm_component_attach_documentation($(parts.name),
            $_brm_binding_documentation($scope, $(QuoteNode(parts.name)),
                $(QuoteNode(deepcopy(_brm_undoc(source)))); previous=$previous))
    end
end

function _brm_component_attach_documentation(component::BRMComponent, docs)
    (isempty(docs) || _brm_is_doc(component.source)) && return component
    source = deepcopy(component.source)
    for doc in reverse(docs)
        source = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
            LineNumberNode(doc.line, something(doc.file, :none)), doc.markdown, source)
    end
    BRMComponent(component.name, component.inputs, component.keywords,
        component.defaults, source, component.builder, component.result, component.scope)
end

function _brm_component(source::Expr, scope::Module)
    (; name, inputs, keywords, defaults, body, value) = _brm_component_parts(source)
    source = _brm_normalize_docs(source)
    occupied = _brm_cell_value_refs!(Set{Symbol}(inputs), _brm_undoc(source).args[2])
    function fresh(stem)
        candidate = stem
        while candidate in occupied
            candidate = Symbol(candidate, :_)
        end
        push!(occupied, candidate)
        candidate
    end
    fields = _brm_component_return_fields(value)
    if fields === nothing
        result = fresh(:component_result)
        assignments = Any[Expr(:(=), result, value)]
    else
        locals = Tuple(fresh(Symbol(:component_result_, field)) for field in first.(fields))
        result = NamedTuple{Tuple(first.(fields))}(locals)
        assignments = Any[Expr(:(=), local_name, expression)
            for (local_name, expression) in zip(locals, last.(fields))]
    end
    builder = brm(Expr(:block, body..., assignments...); mod=scope)
    BRMComponent(name, inputs, keywords, defaults, source, builder, result, scope)
end

Base.nameof(component::BRMComponent) = component.name
Base.parentmodule(component::BRMComponent) = component.scope
Base.show(io::IO, component::BRMComponent) = Base.show_unquoted(io, brm_source(component))
Base.show(io::IO, ::MIME"text/plain", component::BRMComponent) = show(io, component)
Base.deepcopy_internal(component::BRMComponent, ::IdDict) = component

"""Return a copy of a reusable component's authored function expression."""
brm_source(component::BRMComponent) = deepcopy(component.source)
brm_columns(component::BRMComponent) = component.inputs

function _brm_component_own_documentation(component::BRMComponent)
    header = Tuple(BRMDocumentation((), component.name, doc.markdown,
        doc.statement, doc.file, doc.line) for doc in _brm_source_documentation(brm_source(component)))
    (header..., _brm_documentation(component.builder)...)
end
function _brm_component_documentation(model)
    Tuple(BRMDocumentation((call.path..., doc.path...), first(call.path),
        doc.markdown, doc.statement, doc.file, doc.line)
        for call in brm_components(model) if !isempty(call.path)
        for doc in _brm_component_own_documentation(call.definition))
end
_brm_documentation(component::BRMComponent) =
    (_brm_component_own_documentation(component)..., _brm_component_documentation(component)...)

"""
    brm_components(model::BRMI)
    brm_components(builder::BRMBuilder, data)
    brm_components(component::BRMComponent)

Discover recursively included BRM component calls from the authored model.
Each record contains `path` (the call's logical scope), `definition` (the
shared `BRMComponent` object), `source` (a copy of its authored function
expression), `scope` (its defining module), and `call` (a copied call expression,
without bound data values).
Repeated calls have separate paths and share their definition identity.
For a component definition itself, the first record has path `()` and no call.
No backend compilation or manually maintained source registry is required.
"""
function brm_components(model::BRMI)
    records = NamedTuple[]
    _brm_component_calls!(records, model, ())
    Tuple(records)
end
brm_components(builder::BRMBuilder, data) = brm_components(builder(data))
function brm_components(component::BRMComponent)
    placeholders = Dict(input => NamedColumn(input, _BRMComponentInput(input))
        for input in component.inputs)
    model = component.builder(_BRMCellFormulaInputs((;), placeholders))
    _brm_check_component_recursion(component)
    records = NamedTuple[(; path=(), definition=component,
        source=brm_source(component), scope=component.scope, call=nothing)]
    _brm_component_calls!(records, model, ())
    Tuple(records)
end

function _brm_component_call!(records, target, component, call, path)
    address = (path..., target)
    push!(records, (; path=address, definition=component,
        source=brm_source(component), scope=component.scope,
        call=Meta.parse(sprint(show, call))))
    _brm_component_calls!(records, _brm_component_instance(component, call), address)
end

function _brm_component_calls!(records, model, path)
    for (target, column) in _brm_operation_entries(model)
        operation = column isa NamedColumn ? parent(column) : column
        operation isa ExprColumn{typeof(~)} || continue
        rhs = last(getargs(operation))
        rhs isa ExprColumn || continue
        callee = getf(rhs)
        if callee isa BRMComponent
            _brm_component_call!(records, target, callee, rhs, path)
        elseif callee === kernel && first(getargs(rhs)) isa _BRMInlineBody
            body = first(getargs(rhs))
            _brm_component_cell_calls!(records, _brm_inline_expr(body),
                _brm_inline_scope(body), (path..., target))
        end
    end
    records
end

function _brm_component_cell_calls!(records, expression, scope, path)
    expression isa Expr || return records
    if Meta.isexpr(expression, :call, 3) && first(expression.args) === :~
        lhs, call = expression.args[2:end]
        component = _brm_component_cell_head(call, scope)
        if component !== nothing
            target = lhs isa Symbol ? lhs :
                Meta.isexpr(lhs, :ref) ? first(lhs.args) : nothing
            target isa Symbol || error("component call has no named target: $lhs")
            rhs = _brm_component_cell_ir(component, call, scope)
            _brm_component_call!(records, target, component, rhs, path)
            return records
        end
    end
    foreach(child -> _brm_component_cell_calls!(records, child, scope, path), expression.args)
    records
end

function _brm_component_arguments(component::BRMComponent, call::ExprColumn)
    positional, keywords = getargs(call), getkwargs(call)
    npos = length(component.inputs) - length(component.keywords)
    length(positional) <= npos || throw(ArgumentError(
        "component `$(component.name)` accepts $npos positional inputs, got $(length(positional))"))
    extra = setdiff(keys(keywords), component.keywords)
    isempty(extra) || throw(ArgumentError(
        "component `$(component.name)` has no keyword input(s) $(join(extra, ", "))"))
    scope = _BRMScope(component.scope)
    args = Any[]
    for (i, input) in enumerate(component.inputs)
        value = if i <= length(positional)
            positional[i]
        elseif i > npos && haskey(keywords, input)
            keywords[input]
        elseif haskey(component.defaults, input)
            _brm_interpret(_brm_expand(_x(component.defaults[input]), component.scope), scope)
        else
            throw(ArgumentError("component `$(component.name)` requires input `$input`"))
        end
        scope.vars[input] = value
        push!(args, value)
    end
    Tuple(args)
end

# An external model value is a reference, never a local declaration. Keep its
# original column for shape/row provenance without copying or redeclaring it.
struct _BRMComponentInput <: AbstractColumn
    value::Any
end

function _brm_component_input(input::Symbol, value)
    backing = value isa NamedColumn ? parent(value) : value
    NamedColumn(input, backing isa DataColumn ? backing :
        value isa AbstractColumn ? _BRMComponentInput(value) : DataColumn(value))
end

function _brm_component_instance(component::BRMComponent, call::ExprColumn)
    _brm_check_component_recursion(component)
    args = _brm_component_arguments(component, call)
    bindings = Dict{Symbol,Any}(input => _brm_component_input(input, value)
        for (input, value) in zip(component.inputs, args))
    model = component.builder(_BRMCellFormulaInputs((;), bindings))
    # The component's module resolves functions; all model inputs come from
    # its interface. An undeclared name must never capture a caller parameter.
    for (key, column) in _brm_operation_entries(model)
        (key in component.inputs || key === :Inf) && continue
        column isa NamedColumn && parent(column) isa MissingColumn &&
            throw(ArgumentError("component `$(component.name)` reads undeclared input `$key`"))
    end
    model
end

function _brm_check_component_recursion(component::BRMComponent,
        active=BRMComponent[], finished=Set{BRMComponent}())
    component in finished && return nothing
    any(c -> c === component, active) && throw(ArgumentError(
        "recursive BRM component calls: " * join(string.(nameof.((active..., component))), " → ")))
    push!(active, component)
    function visit(expression)
        expression isa Expr || return
        if Meta.isexpr(expression, :call)
            callee = _sb_cell_binding(component.scope, first(expression.args))
            callee isa BRMComponent && _brm_check_component_recursion(callee, active, finished)
        end
        foreach(visit, expression.args)
    end
    visit(_brm_undoc(component.source).args[2])
    pop!(active)
    push!(finished, component)
    nothing
end
