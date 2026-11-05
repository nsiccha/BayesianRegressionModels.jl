# A component is prepared by the ordinary BRM compiler, then embedded in the
# backend's native submodel scope. Inputs and prepared data are explicit;
# source module identities never cross from one backend's implementation to
# the other.
brm_components(model::Union{SBBRMI,RKBRMI}) = brm_components(brm_source(model))
function _brm_component_input_axis(input::_BRMComponentInput)
    value = input.value
    if value isa NamedColumn
        backing = parent(value)
        backing isa _BRMComponentInput && return _brm_component_input_axis(backing)
        backing isa ExprColumn{typeof(~)} &&
            _brm_prior_expression(last(getargs(backing))) &&
            return _brm_parameter_reference_axis(last(getargs(backing)))
        backing isa ExprColumn{typeof(assign)} &&
            return _brm_component_input_axis(_BRMComponentInput(last(getargs(backing))))
    end
    prepared = _brm_prepare_expr(value)
    prepared isa _BRMPreparedRef ? prepared.axis : :scalar
end

function _rk_model_value_axis(column::NamedColumn, input::_BRMComponentInput, context)
    axis = _rk_model_value_axis(input.value, context)
    (axis === nothing || isempty(axis)) && return axis
    # A formal input's conditional cell proof belongs to its actual argument.
    # Recover it when substituting the component's output axis below.
    (; source=name(column), nrows=axis.nrows)
end

function _brm_component_output_axis(target, component, call, context;
        result=component.result)
    # A component returning named values has no single row axis; each field
    # read (`target.field`) supplies its own local `result`.
    result isa Symbol || return nothing
    instance = _brm_component_instance(component, call)
    inner = _brm_backend_context(instance; retain_mm_sources=true)
    axis = _rk_model_value_axis(_brm_operation(instance, result), inner)
    (axis === nothing || isempty(axis)) && return axis
    index = findfirst(==(axis.source), component.inputs)
    argument = index === nothing ? nothing : _brm_component_arguments(component, call)[index]
    mapped = index === nothing ? axis : _rk_model_value_axis(argument, context)
    if mapped === nothing && !isempty(get(axis, :scalar_cells, ())) &&
            argument isa NamedColumn && parent(argument) isa DataColumn &&
            parent(parent(argument)) isa AbstractVector
        # The inner authored plate explicitly declares this argument's cell
        # axis, including a vector of vectors. Its guard still has to prove
        # that the returned cells are scalar before treating them as rows.
        mapped = (; source=name(argument), nrows=axis.nrows)
    elseif mapped === nothing && argument isa NamedColumn &&
            parent(argument) isa DataColumn && parent(parent(argument)) isa AbstractVector &&
            length(parent(parent(argument))) == axis.nrows
        # A grouping input's label column (`(1 | p | unit)` inside the
        # component) has one row per entry whatever its element type.
        mapped = (; source=name(argument), nrows=axis.nrows)
    end
    (mapped === nothing || isempty(mapped)) && return mapped
    local_cells = Tuple(Symbol(target, ".", output) for output in get(axis, :scalar_cells, ()))
    external_cells = index === nothing ? () : get(mapped, :scalar_cells, ())
    scalar_cells = Tuple(unique((external_cells..., local_cells...)))
    isempty(scalar_cells) ? mapped : (; mapped..., scalar_cells)
end

# `kin.recovery` reads one returned field of a component call. Its rows are
# those of the component's local result for that field.
function _brm_component_field_read(value::ExprColumn{typeof(getproperty)})
    object, field = getargs(value)
    object isa NamedColumn || return nothing
    backing = parent(object)
    backing isa ExprColumn{typeof(~)} || return nothing
    rhs = last(getargs(backing))
    rhs isa ExprColumn && getf(rhs) isa BRMComponent || return nothing
    component = getf(rhs)
    fields = _brm_component_fields(component)
    fields === nothing && return nothing
    key = field isa QuoteNode ? field.value : field
    haskey(fields, key) || throw(ArgumentError(
        "component call `$(name(object))` reads field `$key`, but `$(component.name)` " *
        "returns only $(join(("`$f`" for f in keys(fields)), ", "))"))
    (; target=name(object), component, call=rhs, result=fields[key])
end
# A returned field read is prepared with its private label; every other
# property read keeps its authored field.
function _brm_prepare_call_expr(callable::typeof(getproperty), expression, axes)
    read = _brm_component_field_read(expression)
    object, field = getargs(expression)
    read === nothing && return _BRMPreparedExpr(callable,
        (_brm_prepare_expr(object, axes), field),
        map(value -> _brm_prepare_expr(value, axes), getkwargs(expression)))
    _BRMPreparedExpr(_brm_component_field,
        (_brm_prepare_expr(object, axes), QuoteNode(read.result)), (;))
end

# A component returning named values is read one field at a time: the call
# itself is not a value. Refuse a bare read and an unknown field before
# either backend lowers the model.
function _brm_check_component_reads(model::BRMI)
    entries = _brm_operation_entries(model)
    named = Dict{Symbol,BRMComponent}()
    for (target, column) in entries
        operation = column isa NamedColumn ? parent(column) : column
        operation isa ExprColumn{typeof(~)} || continue
        rhs = last(getargs(operation))
        rhs isa ExprColumn && getf(rhs) isa BRMComponent || continue
        _brm_component_fields(getf(rhs)) === nothing || (named[target] = getf(rhs))
    end
    isempty(named) && return nothing
    function visit(value, reader)
        if value isa ExprColumn{typeof(getproperty)}
            object = first(getargs(value))
            if object isa NamedColumn && haskey(named, name(object))
                _brm_component_field_read(value)
                return
            end
        end
        if value isa NamedColumn
            haskey(named, name(value)) || return
            component = named[name(value)]
            fields = keys(_brm_component_fields(component))
            throw(ArgumentError("`$reader` reads `$(name(value))` as one value, but " *
                "component `$(component.name)` returns the named values " *
                "$(join(("`$f`" for f in fields), ", ")); read one field, such as " *
                "`$(name(value)).$(first(fields))`"))
        end
        value isa ExprColumn || return
        foreach(arg -> visit(arg, reader), getargs(value))
        foreach(arg -> visit(arg, reader), values(getkwargs(value)))
    end
    for (target, column) in entries
        haskey(named, target) && continue
        visit(column isa NamedColumn ? parent(column) : column, target)
    end
    nothing
end

function _rk_model_value_axis(value::ExprColumn{typeof(getproperty)}, context)
    read = _brm_component_field_read(value)
    read === nothing && return nothing
    _brm_component_output_axis(read.target, read.component, read.call, context;
        result=read.result)
end

_sb_emit!(statements, data, key, ::_BRMComponentInput; kwargs...) = nothing
_sb_prior_arg_named(value::NamedColumn, ::_BRMComponentInput) = name(value)
_sb_records_submodel_hook(::BRMComponent) = false

struct _BRMComponentReplay
    component::BRMComponent
    target::Symbol
    rhs::ExprColumn
    fitted::SBBRMI
    columns::Dict{Symbol,Symbol}
end
Base.deepcopy_internal(record::_BRMComponentReplay, ::IdDict) = record
_brm_component_replay_key(target) = Symbol(:__brm_component__, target)

function _sb_submodel_rhs!(statements, data, target::Symbol,
        component::BRMComponent, call)
    instance = _brm_component_instance(component, call)
    compiled = SBBRMI(instance; mod=component.scope, total_groups=(),
        _component_output=_brm_component_results(component))
    args = _brm_component_arguments(component, call)
    # Pass only the inputs the compiled body reads. A grouping input's labels
    # are prepared into the component's own index data (`unit_idx`), and the
    # raw label column is not Stan data in the parent.
    read = _rk_source_symbols!(Set{Symbol}(), compiled.model.model)
    inputs = Pair{Symbol,Any}[input => _sb_scalar_expr(value, data)
        for (input, value) in zip(component.inputs, args) if input in read]
    columns = Dict{Symbol,Symbol}()
    taken = get(data, _SB_COMPONENT_NAMES_KEY, Set{Symbol}())
    union!(taken, keys(data))
    for (key, value) in sort!(collect(compiled.data); by=first)
        key in component.inputs && continue
        outer = _rk_ast_fresh_name(string(target, "_component_", key), taken)
        data[outer] = value
        columns[key] = outer
        push!(inputs, key => outer)
    end
    body = Expr(:block, compiled.model.model.args..., _sb_component_return(component))
    submodel = StanBlocks.SlicModel(body, Dict{Symbol,Any}(), component.scope,
        compiled.model.observations)
    keywords = Expr(:parameters, (Expr(:kw, key, value) for (key, value) in inputs)...)
    push!(statements, Expr(:call, :~, target, Expr(:call, submodel, keywords)))
    _sb_record_preproc!(data, _brm_component_replay_key(target), PreprocEntry(
        :brm_component, _BRMComponentReplay(component, target, call, compiled, columns),
        target, true))
    :component
end

function _sb_replay_component!(new_data, new_preproc, handled, key, entry, sb, df, freeze)
    record = entry.const_
    scalars = _sb_hook_scalar_columns!(Dict{Symbol,Any}(), record.rhs)
    if !freeze
        for name in collect(keys(scalars))
            fitted = get(sb.preproc, name, nothing)
            if fitted isa PreprocEntry && fitted.kind === :constant
                scalars[name] = new_data[name]
            elseif _sb_df_has_column(df, name)
                delete!(scalars, name)
            end
        end
    end
    rhs = _brm_rebind_value(record.rhs, _BRMReplayData(df, scalars))
    args = _brm_component_arguments(record.component, rhs)
    inputs = Dict{Symbol,Any}()
    for (input, value) in zip(record.component.inputs, args)
        backing = value isa NamedColumn ? parent(value) : value
        backing isa DataColumn && (inputs[input] = parent(backing))
        value isa AbstractColumn || (inputs[input] = value)
    end
    replay = reprocess(record.fitted, (; inputs...); freeze_constants=freeze)
    for (inner, outer) in record.columns
        haskey(replay.data, inner) || error(
            "component `$(record.target)` replay lost prepared input `$inner`")
        new_data[outer] = replay.data[inner]
        push!(handled, outer)
    end
    new_preproc[key] = PreprocEntry(:brm_component,
        _BRMComponentReplay(record.component, record.target, rhs, replay, record.columns),
        entry.raw_ref, entry.dim_coupled)
    nothing
end

function _brm_component_replay_scope(build, target, component, call)
    frozen = _brm_frozen_replay()
    frozen === nothing && return build()
    raw = Dict{Symbol,Any}()
    args = _brm_component_arguments(component, call)
    foreach(value -> _rk_raw_data_columns!(raw, value), args)
    frame = _BRMReplayData((; raw...), frozen.training)
    training = Dict{Symbol,Any}()
    resample = Set{Symbol}()
    for (input, value) in zip(component.inputs, args)
        rebound = _brm_rebind_value(value, frame)
        data = _brm_data_assignment_value(rebound,
            _BRMDataAssignments("BRM component replay"), Set{Symbol}())
        data === nothing || (training[input] = data)
        refs = _brm_operation_references!(Set{Symbol}(), value)
        isempty(intersect(refs, frozen.resample)) || push!(resample, input)
    end
    # Fits use the same local names as the component body, but a fitted-state
    # export owns each call's fits under its full enclosing component path.
    prefix = string(target, ".")
    fits = frozen.fits === nothing ? nothing : Dict{Tuple{Symbol,Symbol},Any}(
        (tag, Symbol(chop(string(name); head=length(prefix), tail=0))) => value
        for ((tag, name), value) in frozen.fits if startswith(string(name), prefix))
    record = frozen.record === nothing ? nothing : Dict{Tuple{Symbol,Symbol},Any}()
    scoped = _BRMFrozenReplay(training, resample, fits, record)
    result = _brm_with_frozen_replay(build, scoped)
    if record !== nothing
        for ((tag, name), value) in record
            frozen.record[(tag, Symbol(prefix, name))] = value
        end
    end
    result
end

function _brm_component_rk_plan(component, call)
    instance = _brm_component_instance(component, call)
    instance = _brm_resolve_prior_addresses(instance)
    frozen = _brm_frozen_replay()
    function fitted_constant(name, _source)
        frozen === nothing && return nothing
        node = last(getargs(parent(_brm_operation(instance, name))))
        _brm_frozen_fit(frozen, (:constant, name)) do
            raw = _rk_training_columns(instance)
            rebound = _brm_rebind_value(node, _BRMReplayData((; raw...), frozen.training))
            _brm_evaluate_constant(name, rebound; prefix="BRM component")
        end
    end
    instance, constants = _brm_resolve_constants(instance;
        fitted=fitted_constant, prefix="BRM component")
    instance, assignments = _brm_resolve_data_assignments(instance; prefix="BRM component")
    program = _brm_prepare_program(instance;
        context=_brm_backend_context(instance; retain_mm_sources=true))
    submodels = _rk_prepare_submodel_values(program)
    kernels = Tuple(_rk_prepare_kernel_value(instance, program, name, rhs)
        for (name, rhs) in _rk_kernel_ops(instance))
    observations = _brm_direct_observation_list(instance;
        prefix="BRM component", require_observation=false)
    plan = _brm_rk_value_plan(instance, program, observations;
        kernels, submodels, value_roots=_brm_component_results(component),
        external_inputs=component.inputs)
    _rk_attach_data_assignments!(plan, assignments, constants)
    plan, instance
end

function _rk_submodel_rhs!(definitions, statements, data, bindings,
        target::Symbol, component::BRMComponent, call; reserved=())
    plan, instance = _brm_component_replay_scope(target, component, call) do
        _brm_component_rk_plan(component, call)
    end
    cells = Dict{Symbol,Symbol}()
    coordinates = Any[]
    emitted = _rk_emit_ast(plan; cell_sources=cells, coordinates,
        documentation=brm_documentation(instance))
    taken = union(Set{Symbol}(component.inputs), Set(_brm_operation_names(instance)),
        Set{Symbol}(reserved))
    # Helpers are module-level definitions. Give each instantiated subprogram
    # its own helper names; sampled and deterministic locals remain native
    # submodel locals, under the call's logical path.
    renames = Dict{Symbol,Symbol}()
    for key in (_rk_source_definition(d).name for d in emitted.defs)
        renames[key] = _rk_ast_fresh_name(string(target, "_component_", key), taken)
    end
    for (key, value) in emitted.bindings
        # Native distribution vocabulary is also used by local priors. A
        # bound observation constructor must not rename those prior heads
        # into opaque external constructors. Shared compiler bindings retain
        # their exact identity; component-specific callables get private names.
        shared = isdefined(@__MODULE__, key) && getfield(@__MODULE__, key) === value
        renamed = shared ? key : _rk_ast_fresh_name(string(target, "_component_", key), taken)
        shared || (renames[key] = renamed)
        push!(bindings, renamed => value)
    end
    append!(definitions, (_rk_rename_symbols(definition, renames) for definition in emitted.defs))
    # Fitted graph summaries belong to this call, but are named in the parent
    # graph so its ordinary frozen-values export can retain them. Their native
    # computations still execute at binding time; no Julia-side fitting or
    # sampled value is substituted for them.
    fit_inputs = _rk_plan_fit_inputs(plan)
    fitted = _rk_fitted_definitions(emitted, plan, Set{Symbol}(keys(fit_inputs)))
    for key in sort!(collect(keys(fitted)))
        renames[key] = _rk_ast_fresh_name(string(target, "_component_", key), taken)
    end
    local_statements = [statement for statement in emitted.main.args
        if _rk_fitted_definition(statement, fitted) === nothing]
    body = _rk_rename_symbols(Expr(:block, local_statements...), renames)
    push!(body.args, _rk_component_return(component))
    params = Symbol[component.inputs...]
    args = Any[_rk_value_expr!(bindings, _brm_prepare_expr(value), taken)
        for value in _brm_component_arguments(component, call)]
    columns = merge(_rk_source_data_columns(plan, emitted), Dict(_rk_extent_bindings(plan)))
    column_names = Dict{Symbol,Any}(input => arg for (input, arg) in zip(params, args))
    local_reads = _rk_source_symbols!(Set{Symbol}(), body)
    for (key, value) in sort!(collect(columns); by=first)
        key in component.inputs && continue
        outer = _rk_ast_fresh_name(string(target, "_component_", key), taken)
        data[outer] = value
        column_names[key] = outer
        if key in local_reads
            push!(params, key)
            push!(args, outer)
        end
    end
    hoisted_names = merge(renames, column_names)
    for statement in emitted.main.args
        _rk_fitted_definition(statement, fitted) === nothing && continue
        push!(statements, _rk_rename_symbols(statement, hoisted_names))
    end
    for key in sort!(collect(keys(fitted)))
        push!(params, renames[key])
        push!(args, renames[key])
    end
    entry = _rk_ast_fresh_name(string(target, "_component"), taken)
    header = Tuple(doc for doc in _brm_component_own_documentation(component) if isempty(doc.path))
    definition = Expr(:(=), Expr(:call, entry, params...), body)
    push!(definitions, _brm_documented_statement(definition, header))
    push!(statements, Expr(:call, :~, target, Expr(:call, entry, args...)))
    outer_fits = Dict{Symbol,Symbol}()
    for (input, source) in fit_inputs
        outer_fits[column_names[input]] = column_names[source]
    end
    observed = Tuple(column_names[name] for name in _rk_observed_names(plan))
    row_cells = Dict{Symbol,Any}()
    for (output, raw) in cells
        row_cells[Symbol(target, ".", output)] = (; definition=entry,
            output=get(renames, output, output), cells=get(renames, raw, raw))
    end
    for assignment in plan.assignments
        assignment isa _RKPreparedSubmodelAssignment || continue
        assignment.component === nothing && continue
        for (path, record) in assignment.component.row_cells
            row_cells[Symbol(target, ".", path)] =
                NamedTuple{keys(record)}(Tuple(get(renames, value, value) for value in values(record)))
        end
    end
    metadata = (; definition=component, source=instance, plan, entry, renames, row_cells,
        coordinates)
    (; kind=:component, fit_inputs=outer_fits, observed, component=metadata)
end

# The caller's offset can require a row proof inside a component. Retained
# emission records identify the exact local output and its unflattened cells;
# no output length or private-name parsing establishes this correspondence.
function _rk_component_offset_cells!(definitions, plan, outputs, callee)
    definitions_by_name = Dict(_rk_source_definition(d).name => d for d in definitions)
    assignments_by_definition = Dict{Symbol,Dict{Symbol,Vector{Expr}}}()
    for assignment in plan.assignments
        assignment isa _RKPreparedSubmodelAssignment || continue
        metadata = assignment.component
        metadata === nothing && continue
        for output in intersect(outputs, keys(metadata.row_cells))
            record = metadata.row_cells[output]
            assignments = get!(assignments_by_definition, record.definition) do
                body = last(_brm_undoc(definitions_by_name[record.definition]).args)
                by_name = Dict{Symbol,Vector{Expr}}()
                for statement in body.args
                    plain = _brm_undoc(statement)
                    Meta.isexpr(plain, :(=), 2) && first(plain.args) isa Symbol || continue
                    push!(get!(Vector{Expr}, by_name, first(plain.args)), plain)
                end
                by_name
            end
            selected = get(assignments, record.output, Expr[])
            length(selected) == 1 || error("component row proof lost local output `$output`")
            only(selected).args[2] = Expr(:call, callee, record.cells)
        end
    end
    nothing
end
