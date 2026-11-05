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
            definition=metadata.definition, constants=(;), records=metadata.coordinates,
            plan=metadata.plan)
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
            definition=replay.component, constants=replay.fitted.data,
            fitted=replay.fitted)
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
        source = scopes[path].source
        # A key naming one of the call's own linear predictors or `|ID|`
        # blocks takes the top-level `inits` grammar; `_brm_start_targets`
        # parses it against the call's model.
        result[path] = NamedTuple{keys(values)}(map(keys(values)) do key
            value = getproperty(values, key)
            _brm_component_formula_key(source, key) ? value :
                _brm_component_start_value(path, value)
        end)
    end
    result
end

# A nested component call (`inner ~ field(x)`) is a predictor but no formula.
function _brm_component_call_key(source, key)
    op = linear_predictor_op(source, key)
    op === nothing && return false
    rhs = last(getargs(op))
    rhs isa ExprColumn && getf(rhs) isa BRMComponent
end
_brm_component_formula_key(source, key) = !_brm_component_call_key(source, key) &&
    (!isnothing(popcoefnames(source, key)) || !isnothing(ranefcoefnames(source, key)))

# A component's own predictor is computed, never a sampled value.
function _brm_component_formula_targets(source, formula)
    targets = _brm_start_targets(source, formula)
    for t in targets
        t.kind === :parameter && error("brm_start: `$(t.name)` is a linear predictor " *
            "of this call, a computed value; give its coefficients as a NamedTuple, " *
            "e.g. `$(t.name) = (; Intercept = value)`")
    end
    targets
end

# The call's predictor and `|ID|` entries, and its local parameters.
function _brm_component_split_inits(source, explicit::NamedTuple)
    formula = Tuple(k for k in keys(explicit) if _brm_component_formula_key(source, k))
    local_keys = Tuple(k for k in keys(explicit) if !(k in formula))
    (; formula=NamedTuple{formula}(map(k -> getproperty(explicit, k), formula)),
       locals=NamedTuple{local_keys}(map(k -> getproperty(explicit, k), local_keys)))
end

# Errors of the shared top-level writers name the component call they occur in.
function _brm_component_start_context(f, path)
    try
        f()
    catch err
        err isa Union{ErrorException,ArgumentError,DimensionMismatch} || rethrow()
        message = "brm_start: in component $(repr(path)): " *
            replace(err.msg, r"^brm_start: " => "")
        err isa DimensionMismatch && throw(DimensionMismatch(message))
        err isa ArgumentError && throw(ArgumentError(message))
        error(message)
    end
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

# A call's own coordinate records at the declarations its call scopes them
# under in the parent program: the local `b_g.z` of call `left` is `left.b_g.z`.
const _RK_SCOPED_RECORD_FIELDS = (:declaration, :scale, :scales, :z, :L, :mixing,
    :level_correlation, :beta_declaration)
_brm_rk_scoped_name(path, name::Symbol) = _brm_component_start_name(path, name)
_brm_rk_scoped_name(path, ::Nothing) = nothing
_brm_rk_scoped_name(path, names::Tuple) = map(n -> _brm_rk_scoped_name(path, n), names)
function _brm_rk_scoped_record(path, r)
    fields = Tuple(f for f in _RK_SCOPED_RECORD_FIELDS if haskey(r, f))
    merge(r, NamedTuple{fields}(map(f -> _brm_rk_scoped_name(path, r[f]), fields)))
end

# The call's grouping columns, which a block record's levels come from.
_brm_rk_scope_owner(scope) = (; plan=scope.plan)

function _brm_rk_start_components!(writes, effects, rk, records, values, inits,
        component_inits, base)
    isempty(component_inits) && !_brm_start_prior_requested(base) && return nothing
    scopes = _brm_rk_start_scopes!(Dict{Tuple,Any}(), parent(rk), records)
    starts = _brm_component_start_inits(scopes, component_inits)
    fitted = _rk_fitted_state(rk)
    environment = _brm_component_start_envs(parent(rk), inits, scopes, starts;
        constants=fitted === nothing ? (;) : fitted.constants)
    scoped = Dict(path => [_brm_rk_start_record(_brm_rk_scope_owner(scope),
            _brm_rk_scoped_record(path, r)) for r in scope.records]
        for (path, scope) in scopes)
    for (path, scope) in scopes
        explicit = get(starts, path, (;))
        (; formula) = _brm_component_split_inits(scope.source, explicit)
        _brm_component_start_context(path) do
            for target in _brm_component_formula_targets(scope.source, formula)
                _brm_rk_start_writes!(writes, Val(target.kind), target, scoped[path], values)
                target.kind === :ranef && !isnothing(target.effects) || continue
                entry = (; target, owner=_brm_rk_scope_owner(scope),
                    record=_brm_rk_ranef_record(target, scoped[path]))
                # The given levels must be this call's own, checked here so the
                # refusal names the call.
                _brm_rk_effect_rows(entry.owner, target, entry.record)
                push!(effects, entry)
            end
        end
    end
    for (path, scope) in scopes
        parameters = Dict(r.declaration => r for r in scope.records
            if r.kind in (:scalar, :vector, :lkj_cholesky))
        explicit = _brm_component_split_inits(scope.source, get(starts, path, (;))).locals
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
        # The call's population coefficients and `|ID|` scales, as at top level.
        chosen = Set((d, i) for (d, i, _) in writes)
        function add(d, i, prior, spelling; lower=-Inf, upper=Inf)
            ((d, ()) in chosen || (d, i) in chosen) && return
            local_spelling = "$spelling in component $(repr(path))"
            law = _brm_start_evaluate_law(prior, env.value, local_spelling)
            push!(writes, (d, i, _brm_start_median(law, local_spelling; lower, upper)))
        end
        _brm_rk_start_prior_records!(add, scoped[path], values; scalars=false)
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

# A call's Stan values are its own model's, named under the call's path:
# `left_<name>` for call `left`, `nested_inner_<name>` inside `nested`. The
# parent's names with that prefix removed are the call's own; every other name
# is masked, so a coordinate found among them is a position in the parent's.
_brm_sb_component_prefix(path) = string(join(path, "_"), "_")
_brm_sb_component_local_names(names, prefix) =
    [startswith(n, prefix) ? n[nextind(n, ncodeunits(prefix)):end] : "\0" * n
     for n in String.(names)]

# The call's own descriptor and local names, built once per call and only for
# a call with formulas. The call is traced on its own, where nothing reads its
# result, so StanBlocks would place its blocks and coefficients in generated
# quantities; inside the parent they are sampled. Each output therefore takes
# its placement (`kind`) from the parent's traced program, under the call's
# prefix.
_brm_with_field(x, field, value) = typeof(x)((f === field ? value : getfield(x, f)
    for f in fieldnames(typeof(x)))...)
_brm_component_has_formulas(source) =
    any(lp -> _brm_component_formula_key(source, lp.name), linear_predictors(source))

# A parameter-valued input (`center` in `a ~ Normal(center, .2)`) is a keyword
# the parent passes, absent from the call's own data, so the standalone trace
# cannot type the predictive draws that read it. A placeholder of the
# argument's shape types it; its value is never read, because placement comes
# from the parent's trace. The shape is the parent's value of the same name;
# an argument the parent does not hold by name is a scalar.
function _brm_sb_component_input_placeholders(scope, path, model, parent_d)
    enclosing = length(path) > 1 ? _brm_sb_component_prefix(Base.front(path)) : ""
    dims = Dict(o.name => o.dims for o in parent_d.stan.outputs)
    placeholders = Dict{Symbol,Any}()
    for (input, argument) in zip(scope.definition.inputs,
            _brm_component_arguments(scope.definition, scope.call))
        haskey(model.data, input) && continue
        shape = argument isa NamedColumn ?
            get(dims, Symbol(enclosing, name(argument)), ()) : ()
        placeholders[input] = isempty(shape) ? 0.0 : zeros(shape...)
    end
    placeholders
end

function _brm_sb_component_scope(scope, path, names, parent_d)
    prefix = _brm_sb_component_prefix(path)
    plan = _brm_plan_of(scope.fitted)
    model = plan.model
    typed = StanBlocks.SlicModel(model.model,
        merge(model.data, _brm_sb_component_input_placeholders(scope, path, model, parent_d)),
        model.mod, model.observations)
    own = Base.invokelatest(StanBlocks.stan_descriptor, typed)
    kinds = Dict(o.name => o.kind for o in parent_d.stan.outputs)
    outputs = map(own.outputs) do o
        kind = get(kinds, Symbol(prefix, o.name), o.kind)
        kind === o.kind ? o : _brm_with_field(o, :kind, kind)
    end
    stan = _brm_with_field(own, :outputs, outputs)
    d = _brm_descriptor(plan, stan, Dict{Symbol,Any}(), Dict{Symbol,String}(), ())
    (; d, prefix, names=_brm_sb_component_local_names(names, prefix))
end

function _brm_sb_start_components!(writes, deferred, sb, d, names, inits, component_inits, base)
    isempty(component_inits) && !_brm_start_prior_requested(base) && return nothing
    scopes = _brm_sb_start_scopes!(Dict{Tuple,Any}(), sb)
    starts = _brm_component_start_inits(scopes, component_inits)
    environment = _brm_component_start_envs(parent(sb), inits, scopes, starts; constants=d.plan.data)
    index = _BRMNameIndex(names)
    bounds = _brm_start_prior_requested(base) ? _brm_sb_start_bounds(d, index) : nothing
    submodels = Dict(_brm_submodel_address(s) => s for s in d.submodels)
    output_names = Set(o.name for o in d.stan.outputs)
    model_names = Set(keys(d.stan.model))
    locals = Dict{Tuple,Any}()
    scoped(path) = get!(() -> _brm_sb_component_scope(scopes[path], path, names, d),
        locals, path)
    for (path, scope) in scopes
        (; formula) = _brm_component_split_inits(scope.source, get(starts, path, (;)))
        isempty(formula) && continue
        own = scoped(path)
        _brm_component_start_context(path) do
            for target in _brm_component_formula_targets(scope.source, formula)
                local_deferred = (; absorbed=Dict{Symbol,Vector{Pair{Symbol,Float64}}}(),
                    s2z=Dict{Symbol,Vector{Pair{Symbol,Float64}}}(),
                    claimed=Dict{Tuple{Symbol,Int},NamedTuple}(), blocks=NamedTuple[])
                _brm_sb_start_writes!(writes, local_deferred, Val(target.kind), target,
                    own.d, own.names, total_effect_blocks(own.d.plan))
                isempty(local_deferred.absorbed) && isempty(local_deferred.s2z) &&
                    isempty(local_deferred.claimed) || error(
                        "brm_start: internal: a component call samples no exact-total " *
                        "or sum-to-zero block")
                for entry in local_deferred.blocks
                    # The given levels must be this call's own (named here).
                    _brm_start_level_positions(entry.t.id, entry.t.effects.levels,
                        entry.block.levels)
                    push!(deferred.blocks, merge(entry, (; d=own.d, prefix=own.prefix, path)))
                end
            end
        end
    end
    for (path, scope) in scopes
        local_output = _brm_sb_component_start_outputs(d, submodels[path], output_names, model_names)
        explicit = _brm_component_split_inits(scope.source, get(starts, path, (;))).locals
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
        # The call's population coefficients and `|ID|` scales, as at top level.
        _brm_component_has_formulas(scope.source) || continue
        own = scoped(path)
        chosen = Set(first(w) for w in writes)
        function add(indices, prior, label; lower=-Inf, upper=Inf)
            spelling = "$label in component $(repr(path))"
            remaining = filter(i -> !(i in chosen), indices)
            isempty(remaining) && return
            law = _brm_start_evaluate_law(prior, env.value, spelling)
            law isa MultivariateDistribution && return
            for i in remaining
                lo, hi = get(bounds, i, (-Inf, Inf))
                push!(writes, (i, _brm_start_median(law, spelling;
                    lower=max(lower, lo), upper=min(upper, hi))))
            end
        end
        _brm_sb_start_prior_population!(add, scope.fitted, own.d, own.names)
        _brm_component_start_context(path) do
            _brm_sb_start_prior_ranefs!(writes, own.d, own.names, chosen, env.value)
        end
    end
    _brm_start_prior_requested(base) ? environment(()) : nothing
end
