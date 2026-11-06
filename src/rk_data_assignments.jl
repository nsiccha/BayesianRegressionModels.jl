# A data-only assignment (`_brm_resolve_data_assignments`, user decision
# `1tb1jrg`) is planned as a data column, so every site that reads a raw column
# reads it. The graph then computes it from its own inputs: each one the
# program reads becomes a definition of its right-hand side, which
# `_rk_source_data_columns` leaves unbound, so a design binds only raw columns.
# A fit read off one (`standardize(tt)`) reads a fit-role twin defined over its
# inputs' fit-role copies (`tt_fit = f(t_fit)`), and a `constant(...)` its
# right-hand side reads is computed from fit-role copies of the constant's
# inputs, so a replay freezes both at their fitted values
# (`_rk_fitted_definitions`). The structural emitter writes these definitions
# first, in this order (`:_rk_data_assignment`).
function _rk_attach_data_assignments!(plan, assignments, constants)
    isempty(assignments) && return plan
    _rk_attach_data_assignments!(_rk_plan_derived(plan), plan.columns,
        _rk_plan_taken(plan), assignments, constants)
    plan
end
_rk_attach_data_assignments!(plan::_RKUnconditionedPlan, _assignments, _constants) = plan

_rk_plan_taken(plan::_RKStructuralPlan) = union!(Set{Symbol}(keys(plan.columns)),
    keys(_rk_input_bindings(plan)), keys(_rk_generated_inputs(plan)),
    (d.name for d in plan.derived), (p.name for p in plan.parameters),
    (p.name for p in plan.predictors), (a.name for a in plan.assignments),
    (v.name for v in plan.vector_parameters),
    (first(binding) for binding in _rk_extent_bindings(plan)))
_rk_plan_taken(plan::_RKValuePlan) = union!(_rk_plan_taken(plan.regression),
    keys(plan.columns), (a.name for a in plan.assignments))

function _rk_attach_data_assignments!(derived::Vector{_RKDerivedSpec}, columns,
        taken::Set{Symbol}, assignments, constants)
    by_name = Dict{Symbol,Any}(a.name => a for a in assignments)
    constant_nodes = Dict{Symbol,Any}(c.name => c.node for c in constants)
    constant_values = Dict{Symbol,Any}(c.name => c.value for c in constants)
    reads(name) = (input for input in by_name[name].inputs if haskey(by_name, input))
    closure(roots) = let found = Set{Symbol}()
        visit(name) = name in found || (push!(found, name); foreach(visit, reads(name)))
        foreach(visit, roots)
        found
    end
    # The assignments the program reads, directly or through a fit.
    twinned = closure(t for spec in derived
        if Meta.isexpr(spec.expression, :_rk_fit_input)
        for t in (_rk_fit_twinned(spec.label),) if t !== nothing && haskey(by_name, t))
    needed = union(closure(a.name for a in assignments if haskey(columns, a.name)), twinned)
    isempty(needed) && return
    order = Symbol[]
    place(name) = name in order ||
        (foreach(place, reads(name)); push!(order, name))
    foreach(place, (a.name for a in assignments if a.name in needed))

    raw = Dict{Symbol,Any}()
    used_constants = Symbol[]
    lowered = Dict(name => _rk_lower_data_assignment(by_name[name].node, by_name,
        constant_nodes, raw, used_constants) for name in order)
    # A masked HSGP already defines its numeric axis from a raw source. An
    # authored assignment supplies that source, preserving its missing values
    # for validation and for other assignments that read it. Defining the axis
    # a second time would both lose that distinction and violate single assignment.
    masked = Dict(spec.name => spec.expression.args[2] for spec in derived
        if haskey(by_name, spec.name) && Meta.isexpr(spec.expression, :_rk_data_preparation) &&
            first(spec.expression.args) === :brm_hsgp_masked_axis)
    live(x::Symbol) = get(masked, x, x)
    live(x::Expr) = Expr(x.head, map(live, x.args)...)
    live(x) = x
    # Every constant first: it reads fit-role inputs alone.
    for key in unique(used_constants)
        node = constant_nodes[key]
        spec = first(getargs(node))
        inputs = map(input -> _rk_constant_fit_input!(derived, columns, taken, input),
            Base.tail(getargs(node)))
        evaluate = _RKConstantEvaluator(key, spec.evaluate, constant_values[key])
        push!(derived, _RKDerivedSpec(key, Expr(:_rk_data_assignment,
            Expr(:call, _rk_whole_call(evaluate, length(inputs), ()), inputs...)), key))
    end
    for (key, value) in raw
        value isa AbstractArray && !haskey(columns, key) && (columns[key] = value)
    end
    # A fit-role twin reads the fit-role copy of every column and assignment.
    fit_twins = Dict{Symbol,Symbol}()
    twin(x::Symbol) = haskey(fit_twins, x) ? fit_twins[x] :
        !haskey(raw, x) ? x :
        raw[x] isa AbstractArray ?
            _rk_fit_input!(derived, columns, taken, x, _brm_fit_input_source(x, raw[x])) :
        _rk_fit_scalar!(derived, columns, taken, x, raw[x])
    twin(x::Expr) = Expr(x.head, map(twin, x.args)...)
    twin(x) = x
    for name in order
        push!(derived, _RKDerivedSpec(live(name),
            Expr(:_rk_data_assignment, live(lowered[name])), name))
        name in twinned || continue
        label = _rk_fit_label(name)
        index = findfirst(spec -> spec.label === label, derived)
        twin_name = index === nothing ? _rk_fit_name!(derived, columns, taken, name) :
            derived[index].name
        index === nothing || deleteat!(derived, index)
        raw_twin = haskey(masked, name) ?
            _rk_mint_generated!(taken, columns, string(masked[name], "_fit")) : twin_name
        push!(derived, _RKDerivedSpec(raw_twin,
            Expr(:_rk_data_assignment, twin(lowered[name])),
            haskey(masked, name) ? _rk_fit_label(masked[name]) : label))
        raw_twin === twin_name || push!(derived, _RKDerivedSpec(twin_name,
            Expr(:_rk_data_preparation, :brm_hsgp_masked_axis, raw_twin), label))
        fit_twins[name] = raw_twin
    end
    nothing
end

# A data scalar a statement reads by name is a bound input of the program
# (`_RKStructuralPlan.inputs`, user principle `0ikeidi`). What else a prior reads
# by name has no such input: an assignment holding a literal (`rate = 1.7`) folded
# out of the plan, or a data vector, whose whole value a component's prior reads
# (`b ~ brm_group_effects(g, v[1])`). Define those in the main block: such a value
# stays in the fitted program, no design binds it, and a replay that changes it
# fails the program identity check. The value plan binds its columns instead
# (`_brm_rk_value_plan`).
_rk_define_read_scalars!(plan, _known) = plan
function _rk_define_read_scalars!(plan::_RKStructuralPlan, known)
    (; consts, aliases, parameters, assignments, data) = known
    # Whether a statement reads one by name takes an emission to tell. A model
    # all of whose data are columns or inputs, with no folded assignment, has
    # none. Prior dependencies cannot decide it: they omit an R2D2 `reference_scale`.
    candidate(name) = !haskey(plan.columns, name) && !haskey(plan.inputs, name) &&
        (haskey(consts, name) || haskey(aliases, name) || haskey(data, name))
    any(candidate, Iterators.flatten((keys(consts), keys(aliases), keys(data)))) ||
        return plan
    emitted = _rk_emit_ast(plan)
    main = emitted.main
    defined = Set{Symbol}(keys(plan.columns))
    union!(defined, (first(binding) for binding in _rk_extent_bindings(plan)))
    for statement in main.args
        _rk_source_assignments!(defined, statement)
        _rk_source_outputs!(defined, statement)
    end
    reads = _rk_program_value_reads(emitted)
    for name in sort!(collect(setdiff(reads, defined)); by=string)
        candidate(name) || continue
        expression = if haskey(consts, name) || haskey(aliases, name)
            _, value = _rk_resolve_use_ref(name, consts, aliases, parameters,
                assignments, "assignment `$name`")
            value isa Symbol ? _BRMPreparedRef(value, :scalar) : value
        else
            value = data[name]
            value isa AbstractVector{<:Real} || error(
                "RK backend: a random-effect or term prior reads data field " *
                "`$name`, a $(typeof(value)); its inputs must be numeric scalars " *
                "or numeric vectors")
            collect(value)
        end
        push!(plan.assignments, _RKAssignmentSpec(name, expression, name))
    end
    plan
end

# The names a program reads as values: its main block's, and its submodels' free
# reads.
function _rk_program_value_reads(emitted)
    reads = _rk_value_reads!(Set{Symbol}(), emitted.main)
    foreach(definition -> _rk_free_reads!(reads, definition), emitted.defs)
    reads
end

# What a submodel reads from its caller: a block's law reads a prior's input
# freely (`_RKBlockBody`), so it is neither an argument nor a local of the block.
# A `@kernel` or an ordinary function reads only what it is passed.
function _rk_free_reads!(names, definition)
    _rk_source_definition(definition).kind === :rkppl || return names
    (; parameters, statements) = _rk_definition_parts(definition)
    locals = _rk_source_symbols!(Set{Symbol}(), collect(Any, parameters))
    _rk_source_assignments!(locals, statements)
    _rk_source_outputs!(locals, statements)
    union!(names, setdiff(_rk_value_reads!(Set{Symbol}(), statements), locals))
end

# The names a statement reads as values: not call heads, keyword names or the
# targets of `=` and `~`. A data scalar may share its name with a callee.
_rk_value_reads!(names, _value) = names
_rk_value_reads!(names, value::Symbol) = push!(names, value)
function _rk_value_reads!(names, value::Expr)
    skip = value.head === :call && first(value.args) in (:~, :.~) ? 2 :
        value.head in (:call, :(=), :kw, :., :macrocall) ? 1 : 0
    foreach(arg -> _rk_value_reads!(names, arg), value.args[skip+1:end])
    names
end

# A right-hand side with the model-scope semantics of `_rk_value_expr!`:
# arithmetic, comparisons and built-in math per element, indexing and vector
# literals as written, any other call on whole arrays. A data column, another
# data-only assignment and a numeric scalar data field are read by name.
_rk_lower_data_assignment(x, _assignments, _constants, _raw, _used) = x
function _rk_lower_data_assignment(x::NamedColumn, assignments, constants, raw, used)
    _brm_is_colon(x) && return :(:)
    key = name(x)
    backing = parent(x)
    if backing isa DataColumn
        haskey(constants, key) && (push!(used, key); return key)
        value = parent(backing)
        value isa AbstractArray || _rk_scalar_value(value) || return value
        raw[key] = value
        return key
    end
    backing isa ExprColumn{typeof(assign)} || error(
        "RK backend: internal: data-only assignment reads `$key`, which is not data")
    haskey(assignments, key) && return key
    # A scalar assignment (`m = mean(x)`) or a literal (`n = 3`) is not a data
    # column: it is computed where it is read, so a design recomputes it too.
    _rk_lower_data_assignment(last(getargs(backing, 2)), assignments, constants, raw, used)
end
function _rk_lower_data_assignment(x::ExprColumn, assignments, constants, raw, used)
    lower(arg) = _rk_lower_data_assignment(arg, assignments, constants, raw, used)
    f = _brm_is_colon(getf(x)) ? Colon() : getf(x)
    args = map(lower, getargs(x))
    kwargs = getkwargs(x)
    if isempty(kwargs)
        f === getindex && return Expr(:ref, args...)
        f === Base.vect && return Expr(:vect, args...)
        haskey(_RK_DERIVED_BINOPS, f) && return Expr(:call, _RK_DERIVED_BINOPS[f], args...)
        haskey(_RK_DERIVED_CMP, f) && return Expr(:call, _RK_DERIVED_CMP[f], args...)
        haskey(_RK_DERIVED_MATH, f) && return _rk_ast_dotted(_RK_DERIVED_MATH[f], args...)
    end
    Expr(:call, _rk_whole_call(f, length(args), Tuple(keys(kwargs))), args...,
        map(lower, values(kwargs))...)
end

# A call on whole arrays: no input broadcasts (`_rk_ast_data_expr!`).
_rk_whole_call(f, npos::Int, keywords::Tuple) =
    _RKDataCall(f, npos, keywords, ntuple(_ -> false, npos + length(keywords)))

# One input of a constant computed in the graph: a column's fit-role copy (the
# training column on a frozen replay), a scalar data field's fitted value, or a
# builder-scope value.
_rk_constant_fit_input!(_derived, _columns, _taken, input::_BRMConstantScopeValue) =
    input.value
function _rk_constant_fit_input!(derived, columns, taken, input::NamedColumn)
    key = name(input)
    backing = parent(input)
    # A replay that does not carry the input restores the constant's fitted
    # value without reading it.
    values = backing isa DataColumn ? parent(backing) : Float64[]
    values = _brm_fit_input_source(key, values)
    values isa AbstractVector || return values
    _rk_fit_input!(derived, columns, taken, key, values)
end

# A constant computed in the graph calls its evaluator there, and RK infers
# every value that reads it from that call. A `brm(source; mod)` evaluator is
# an interpreted closure (`_BRMClosure`), whose result Julia cannot infer, so
# those values would dispatch at run time on every evaluation (snag
# `constant-of-a-sc-942369d4`). The graph's value is the constant's value as
# planning admits it (`_brm_constant_value`), of the type of the value the
# plan resolved from the same inputs, and the call states that type.
struct _RKConstantEvaluator{T,F} <: Function
    name::Symbol
    evaluate::F
end
_RKConstantEvaluator(name::Symbol, evaluate, value) =
    _RKConstantEvaluator{typeof(value),typeof(evaluate)}(name, evaluate)
function (f::_RKConstantEvaluator{T})(args...) where {T}
    _rk_constant_result(T, f.name, f.evaluate(args...))
end
function _rk_constant_result(::Type{T}, name, result) where {T}
    value = _brm_constant_value(result)
    value isa T || error("RK backend: `$name = constant(...)` evaluated to " *
        "$(repr(result)) (::$(typeof(result))) in the graph, but to a $T when " *
        "the model was planned from the same inputs")
    value::T
end

# The builder stays eval-free, but a runtime graph must not interpret its
# constant's AST on every call. Emit that expression as ordinary Julia in the
# backend's existing compilation module. A result assertion alone types the
# graph value, not the interpreter's internals under RK's strict audit.
function _rk_callable_source!(definitions, bindings, entry,
        f::_RKConstantEvaluator{T,<:_BRMClosure}) where {T}
    closure = f.evaluate
    names = _brm_body_names!(Set{Symbol}(), closure.body)
    # Julia lowers a dotted operator through its undotted binding.
    for name in collect(names)
        text = string(name)
        if startswith(text, '.') && length(text) > 1 && Base.isoperator(Symbol(text[2:end]))
            push!(names, Symbol(text[2:end]))
        end
    end
    taken = union(copy(names), closure.parameters, [entry])
    module_name = _rk_ast_fresh_name(string(entry, "_scope"), taken)
    type_name = _rk_ast_fresh_name(string(entry, "_type"), taken)
    push!(bindings, module_name => closure.scope.mod, type_name => T)
    aliases = Expr[]
    for name in sort!(collect(names); by=string)
        name in closure.parameters && continue
        owner = _brm_binding_scope(closure.scope, name)
        if owner !== nothing
            capture = _rk_ast_fresh_name(string(entry, "_capture_", name), taken)
            push!(bindings, capture => owner.vars[name])
            push!(aliases, Expr(:(=), name, capture))
        elseif isdefined(closure.scope.mod, name)
            push!(aliases, Expr(:(=), name, Expr(:., module_name, QuoteNode(name))))
        end
    end
    # An outer let preserves inner lambda/comprehension bindings, including
    # those that shadow a global. Only referenced captures leave the scope.
    body = Expr(:let, Expr(:block, aliases...),
        _rk_constant_source_body!(bindings, entry, taken, closure.body))
    checked = Expr(:call, GlobalRef(@__MODULE__, :_rk_constant_result),
        type_name, QuoteNode(f.name), body)
    push!(definitions, Expr(:function, Expr(:call, entry, closure.parameters...),
        Expr(:block, checked)))
    true
end

# Interpolated functions, arrays and modules are values, not printable Julia
# syntax. Keep their identities as leaf bindings, just like captured locals.
_rk_constant_source_body!(_bindings, _entry, _taken,
    x::Union{Symbol,LineNumberNode,Number,AbstractString,Char,Nothing}) = x
function _rk_constant_source_body!(bindings, entry, taken, x::Expr)
    x.head === :quote && return x
    Expr(x.head, map(a -> _rk_constant_source_body!(bindings, entry, taken, a), x.args)...)
end
function _rk_constant_source_body!(bindings, entry, taken, x::GlobalRef)
    mod = _rk_constant_source_body!(bindings, entry, taken, x.mod)
    Expr(:., mod, QuoteNode(x.name))
end
function _rk_constant_source_body!(bindings, entry, taken, x::QuoteNode)
    x.value isa Symbol && return x
    _rk_constant_source_literal!(bindings, entry, taken, x.value)
end
_rk_constant_source_body!(bindings, entry, taken, x) =
    _rk_constant_source_literal!(bindings, entry, taken, x)
function _rk_constant_source_literal!(bindings, entry, taken, x)
    name = _rk_ast_fresh_name(string(entry, "_literal"), taken)
    push!(bindings, name => x)
    name
end
