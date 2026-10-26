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
    (d.name for d in plan.derived), (p.name for p in plan.parameters),
    (p.name for p in plan.predictors), (a.name for a in plan.assignments),
    (v.name for v in plan.vector_parameters))
_rk_plan_taken(plan::_RKValuePlan) = union!(_rk_plan_taken(plan.regression),
    keys(plan.columns), (a.name for a in plan.assignments))

function _rk_attach_data_assignments!(derived::Vector{_RKDerivedSpec}, columns,
        taken::Set{Symbol}, assignments, constants)
    by_name = Dict{Symbol,Any}(a.name => a for a in assignments)
    constant_nodes = Dict{Symbol,Any}(c.name => c.node for c in constants)
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
    # Every constant first: it reads fit-role inputs alone.
    for key in unique(used_constants)
        node = constant_nodes[key]
        spec = first(getargs(node))
        inputs = map(input -> _rk_constant_fit_input!(derived, columns, taken, input),
            Base.tail(getargs(node)))
        push!(derived, _RKDerivedSpec(key, Expr(:_rk_data_assignment,
            Expr(:call, _rk_whole_call(spec.evaluate, length(inputs), ()), inputs...)), key))
    end
    for (key, value) in raw
        value isa AbstractArray && !haskey(columns, key) && (columns[key] = value)
    end
    # A fit-role twin reads the fit-role copy of every column and assignment.
    fit_twins = Dict{Symbol,Symbol}()
    twin(x::Symbol) = haskey(fit_twins, x) ? fit_twins[x] :
        haskey(raw, x) && raw[x] isa AbstractArray ?
            _rk_fit_input!(derived, columns, taken, x, _brm_fit_input_source(x, raw[x])) : x
    twin(x::Expr) = Expr(x.head, map(twin, x.args)...)
    twin(x) = x
    for name in order
        push!(derived, _RKDerivedSpec(name, Expr(:_rk_data_assignment, lowered[name]), name))
        name in twinned || continue
        label = _rk_fit_label(name)
        index = findfirst(spec -> spec.label === label, derived)
        twin_name = index === nothing ? _rk_fit_name!(derived, columns, taken, name) :
            derived[index].name
        index === nothing || deleteat!(derived, index)
        push!(derived, _RKDerivedSpec(twin_name,
            Expr(:_rk_data_assignment, twin(lowered[name])), label))
        fit_twins[name] = twin_name
    end
    nothing
end

# A right-hand side with the model-scope semantics of `_rk_value_expr!`:
# arithmetic, comparisons and built-in math per element, indexing and vector
# literals as written, any other call on whole arrays. A data column or another
# data-only assignment is read by name; a scalar data field is its value.
_rk_lower_data_assignment(x, _assignments, _constants, _raw, _used) = x
function _rk_lower_data_assignment(x::NamedColumn, assignments, constants, raw, used)
    key = name(x)
    backing = parent(x)
    if backing isa DataColumn
        haskey(constants, key) && (push!(used, key); return key)
        value = parent(backing)
        value isa AbstractArray || return value
        raw[key] = value
        return key
    end
    backing isa ExprColumn{typeof(assign)} || error(
        "RK backend: internal: data-only assignment reads `$key`, which is not data")
    haskey(assignments, key) && return key
    # A literal assignment (`n = 3`) reads no data; it is its value.
    _brm_data_assignment_value(last(getargs(backing, 2)),
        _BRMDataAssignments("RK backend"), Set{Symbol}())
end
function _rk_lower_data_assignment(x::ExprColumn, assignments, constants, raw, used)
    lower(arg) = _rk_lower_data_assignment(arg, assignments, constants, raw, used)
    f = getf(x)
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
