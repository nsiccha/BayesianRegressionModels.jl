# Authored component paths share one address grammar and prior environment on
# both backends. Native layout transforms remain responsible for constraints.
function _brm_rk_start_scopes!(scopes, source, records, path=())
    for record in records
        record.kind === :submodel || continue
        metadata = get(record, :component, nothing)
        metadata === nothing && continue
        address = (path..., record.target)
        call = last(getargs(parent(_brm_operation(source, record.target))))
        scopes[address] = (; source=metadata.source, call,
            definition=metadata.definition, constants=(;), records=metadata.coordinates)
        _brm_rk_start_scopes!(scopes, metadata.source, metadata.coordinates, address)
    end
    scopes
end

function _brm_sb_start_scopes!(scopes, sb, path=())
    for entry in values(sb.preproc)
        entry.kind === :brm_component || continue
        replay = entry.const_
        address = (path..., replay.target)
        scopes[address] = (; source=parent(replay.fitted), call=replay.rhs,
            definition=replay.component, constants=replay.fitted.data)
        _brm_sb_start_scopes!(scopes, replay.fitted, address)
    end
    scopes
end

_brm_component_start_pairs(inits) = inits
_brm_component_start_pairs(inits::AbstractDict) = pairs(inits)
function _brm_component_start_inits(scopes, inits)
    by_name = Dict{Symbol,Vector{Tuple}}()
    for path in keys(scopes)
        push!(get!(Vector{Tuple}, by_name, last(path)), path)
    end
    result = Dict{Tuple,NamedTuple}()
    for pair in _brm_component_start_pairs(inits)
        pair isa Pair || throw(ArgumentError(
            "brm_start: component_inits takes scope => NamedTuple pairs"))
        address, values = pair
        path = if address isa Symbol
            matches = get(by_name, address, Tuple[])
            length(matches) == 1 || throw(ArgumentError(
                "brm_start: component scope $(repr(address)) matches $(length(matches)) calls; " *
                "use a full path from $(collect(keys(scopes)))"))
            only(matches)
        elseif address isa Tuple{Vararg{Symbol}} && haskey(scopes, address)
            address
        else
            throw(ArgumentError("brm_start: unknown component scope $(repr(address)); " *
                "available paths: $(collect(keys(scopes)))"))
        end
        haskey(result, path) && throw(ArgumentError(
            "brm_start: component_inits repeats scope $(repr(path))"))
        values isa NamedTuple || throw(ArgumentError(
            "brm_start: component $(repr(path)) takes a NamedTuple of local parameter values"))
        result[path] = map(value -> _brm_component_start_value(path, value), values)
    end
    result
end

_brm_component_start_value(path, value::Real) = _brm_start_real("component $path value", value)
_brm_component_start_value(path, value::AbstractArray{<:Real}) =
    map(x -> _brm_start_real("component $path value", x), value)
_brm_component_start_value(path, value) = throw(ArgumentError(
    "brm_start: component $path values must be real scalars or arrays, got $(repr(value))"))

# Lazy call contexts allow prior arguments to read an enclosing explicit start,
# another scalar's median, or another component's returned value. Each scope
# has its own cache: repeated definitions never share parameter values.
function _brm_component_start_envs(source, inits, scopes, starts; constants=(;))
    environments = Dict{Tuple,Any}()
    function environment(path)
        get!(environments, path) do
            if isempty(path)
                local_source, local_inits, fitted, inputs = source, inits, constants, Dict()
            else
                scope = scopes[path]
                enclosing = environment(Base.front(path))
                inputs = Dict(input => (() -> _brm_start_law(argument, enclosing.value))
                    for (input, argument) in zip(scope.definition.inputs,
                        _brm_component_arguments(scope.definition, scope.call)))
                local_source, local_inits, fitted = scope.source, get(starts, path, (;)), scope.constants
            end
            function component_value(key)
                child = (path..., key)
                haskey(scopes, child) || return nothing
                environment(child).value(scopes[child].definition.result)
            end
            _brm_start_prior_values(local_source, local_inits;
                constants=fitted, inputs, component_value)
        end
    end
    environment
end

_brm_component_start_name(path, name) = Symbol(join((path..., name), "."))

function _brm_rk_start_components!(writes, rk, records, values, inits, component_inits, base)
    isempty(component_inits) && !_brm_start_prior_requested(base) && return nothing
    scopes = _brm_rk_start_scopes!(Dict{Tuple,Any}(), parent(rk), records)
    starts = _brm_component_start_inits(scopes, component_inits)
    fitted = _rk_fitted_state(rk)
    environment = _brm_component_start_envs(parent(rk), inits, scopes, starts;
        constants=fitted === nothing ? (;) : fitted.constants)
    for (path, scope) in scopes
        parameters = Dict(r.declaration => r for r in scope.records
            if r.kind in (:scalar, :vector, :lkj_cholesky))
        explicit = get(starts, path, (;))
        for (name, value) in pairs(explicit)
            haskey(parameters, name) || throw(ArgumentError(
                "brm_start: $(repr(path)) has no sampled local parameter `$name`"))
            declaration = _brm_component_start_name(path, name)
            current = _brm_rk_declaration_value(values, declaration)
            size(current) == size(value) || throw(DimensionMismatch(
                "brm_start: component $(repr(path)) `$name` has size $(size(current)), got $(size(value))"))
            push!(writes, (declaration, (), value))
        end
        _brm_start_prior_requested(base) || continue
        env = environment(path)
        for (name, record) in parameters
            record.kind === :scalar && !haskey(explicit, name) || continue
            spelling = _brm_component_start_name(path, name)
            law = _brm_start_evaluate_law(record.prior, env.value, spelling)
            push!(writes, (spelling, (), _brm_start_median(law, spelling)))
        end
    end
    _brm_start_prior_requested(base) ? environment(()) : nothing
end

function _brm_sb_component_start_outputs(d, submodel, output_names, model_names)
    outputs = Dict(o.name => o for o in submodel.outputs)
    owned = Set(keys(outputs))
    names = _brm_submodel_bound_names(submodel)
    bound = names === nothing ? nothing : Set(names)
    function output(name)
        (bound === nothing || name in bound) || throw(ArgumentError(
            "brm_start: component $(_brm_submodel_address(submodel)) has no local parameter `$name`"))
        resolved = Symbol(join((_brm_submodel_address(submodel)..., name), "_"))
        carriers = _brm_binding_carriers(d.stan, output_names, model_names, resolved, owned)
        parameters = [outputs[n] for n in carriers if outputs[n].kind === :parameter]
        length(parameters) == 1 || throw(ArgumentError(
            "brm_start: component $(_brm_submodel_address(submodel)) `$name` must name one sampled local parameter"))
        only(parameters)
    end
    output
end

function _brm_sb_start_components!(writes, sb, d, names, inits, component_inits, base)
    isempty(component_inits) && !_brm_start_prior_requested(base) && return nothing
    scopes = _brm_sb_start_scopes!(Dict{Tuple,Any}(), sb)
    starts = _brm_component_start_inits(scopes, component_inits)
    environment = _brm_component_start_envs(parent(sb), inits, scopes, starts; constants=d.plan.data)
    index = _BRMNameIndex(names)
    bounds = _brm_start_prior_requested(base) ? _brm_sb_start_bounds(d, index) : nothing
    submodels = Dict(_brm_submodel_address(s) => s for s in d.submodels)
    output_names = Set(o.name for o in d.stan.outputs)
    model_names = Set(keys(d.stan.model))
    for (path, scope) in scopes
        local_output = _brm_sb_component_start_outputs(d, submodels[path], output_names, model_names)
        explicit = get(starts, path, (;))
        for (name, value) in pairs(explicit)
            _brm_is_prior_declaration(scope.source, name) || throw(ArgumentError(
                "brm_start: component $(repr(path)) `$name` is not a sampled local parameter"))
            output = local_output(name)
            output.kind === :parameter || throw(ArgumentError(
                "brm_start: component $(repr(path)) `$name` is not a sampled parameter"))
            output.dims === nothing || size(value) == output.dims || throw(DimensionMismatch(
                "brm_start: component $(repr(path)) `$name` has size $(output.dims), got $(size(value))"))
            coordinates = _brm_element_coordinates(output, index)
            flat = value isa Real ? [value] : vec(value)
            length(coordinates) == length(flat) || throw(DimensionMismatch(
                "brm_start: component $(repr(path)) `$name` needs $(length(coordinates)) values, got $(length(flat))"))
            append!(writes, zip(coordinates, flat))
        end
        _brm_start_prior_requested(base) || continue
        env = environment(path)
        for op in env.program.operations
            op.role === :parameter && !haskey(explicit, op.name) || continue
            output = local_output(op.name)
            output.kind === :parameter && isempty(output.size) || continue
            coordinates = _brm_element_coordinates(output, index)
            spelling = _brm_component_start_name(path, op.name)
            law = _brm_start_evaluate_law(last(getargs(op.expression)), env.value, spelling)
            for i in coordinates
                lo, hi = get(bounds, i, (-Inf, Inf))
                push!(writes, (i, _brm_start_median(law, spelling; lower=lo, upper=hi)))
            end
        end
    end
    _brm_start_prior_requested(base) ? environment(()) : nothing
end
