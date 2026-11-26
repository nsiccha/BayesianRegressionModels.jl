# A member fit describes its own executable body. Qualify its public nodes and
# priors with the authored key before offering them to report hooks. The same
# expression can occur in arguments, bindings and children; qualify it once.
_brmd_member_id(scope, id) = (:member, scope.member..., id...)
_brmd_member_value(x, scope) = x
_brmd_member_value(xs::Union{Tuple,NamedTuple}, scope) = map(x -> _brmd_member_value(x, scope), xs)
function _brmd_member_value(x::BRMDescriptionReference, scope)
    logical = x.logical isa Symbol ? (:parameter,x.logical) : x.logical
    logical in scope.local_priors || return x
    BRMDescriptionReference(x.name,x.axis,_brmd_member_id(scope,logical))
end

function _brmd_member_value(c::BRMDescriptionComponent, scope)
    haskey(scope.nodes, c.id) && return scope.nodes[c.id]
    qualify(x) = _brmd_member_value(x, scope)
    id = _brmd_member_id(scope, c.id)
    bindings = map(c.bindings) do b
        merge(b, (; value=qualify(b.value), prior_ids=map(id -> _brmd_member_id(scope,id), b.prior_ids)))
    end
    # Keep the fitted cell's aliases (an input can have the same name on the
    # two sides of a plate). Resolve other projected scalar inputs by value.
    bindings = (bindings..., (b for b in scope.inputs if !any(a -> a.name === b.name,bindings))...)
    provenance = merge(c.provenance, (; scope.member, scope.variant, local_id=c.id,
        model_id=scope.model_id, path=id))
    factor = get(provenance,:covariance_factor,nothing)
    isnothing(factor) || (provenance=merge(provenance,(;covariance_factor=map(id -> _brmd_member_id(scope,id),factor))))
    sources = get(provenance,:observation_sources,())
    any(s -> s in scope.held_out,sources) && (provenance=merge(provenance,(;observation_role=:held_out)))
    outputs = map(o -> _brmd_member_output(o, scope), c.outputs)
    result = BRMDescriptionComponent(id, c.kind, c.callable, qualify(c.arguments), qualify(c.keywords),
        c.axes, outputs, scope.priors, c.fitted_constants,
        qualify(c.children), provenance, c.notation, bindings)
    scope.nodes[c.id] = result
end

function _brmd_member_output(o, scope)
    response = get(scope.response_outputs, (o.logical,o.name), nothing)
    isnothing(response) || return response
    name = get(scope.output_names, o.name, o.name)
    output = merge(o, (; name, scope.member))
    selection = get(scope.selections, (scope.member,name), nothing)
    isnothing(selection) ? output : merge(output, (; selection))
end

function _brmd_member_prior(p, scope, anchors)
    id = _brmd_member_id(scope,p.id)
    source = merge(p.source,(; (k => _brmd_member_id(scope,p.source[k]) for k in
        (:binding_id,:allocation,:integrated) if haskey(p.source,k))...))
    BRMPriorDescription(id, _brmd_member_value(p.distribution,scope),
        _brmd_member_value(p.support,scope),source,get(anchors,id,nothing),
        merge(p.provenance,(; model_id=scope.model_id, scope.member, local_id=p.id)))
end

# The fit supplies the selected declarations and frozen preprocessing. A
# record port bound directly by the parent call reads the parent's current
# data, including a replay with new raw values and frozen preprocessing.
function _brmd_member_plan(fit, call, axis, data, index)
    isnothing(fit) && return nothing
    plan = generative_plan(fit)
    for port in axis.ports
        get(call.inputs,port.formal,nothing) === port.column || continue
        haskey(data,port.column) && (plan.data[port.formal]=data[port.column][index])
    end
    plan
end

# The scoped program can read a parameter, predictor or trajectory computed
# by its caller. Its fitted data alone cannot trace that program. Reflect the
# outputs and their aliases from the trace in which the call actually ran.
function _brmd_member_trace(d, path)
    prefix = string(join(path, "_"), "_")
    localname(k) = Symbol(chopprefix(String(k), prefix))
    names = Dict(o.name => localname(o.name) for o in d.stan.outputs
                 if startswith(String(o.name), prefix))
    outputs = [(; name=names[o.name], o.kind, o.type, o.size, o.constraints,
                  o.generative, o.segments, source=isnothing(o.source) ? nothing : localname(o.source))
               for o in d.stan.outputs if haskey(names, o.name)]
    # A named value may be a view of several emitted outputs. Keep the
    # links the descriptor follows, expressed in the member's namespace.
    model = Dict{Symbol,Tuple}()
    for key in keys(d.stan.model)
        startswith(String(key), prefix) && !haskey(names, key) || continue
        leaves = Set{Symbol}()
        _brm_output_leaves!(leaves, d.stan.model[key], keys(names))
        model[localname(key)] = Tuple(names[leaf] for leaf in leaves)
    end
    (; outputs, model, names=Dict(local_name => name for (name, local_name) in names))
end

function _brmd_fitted_member(parent, call, plan, trace, responses, held_out)
    semantics = _brm_descriptor_semantics(plan.parent)
    by_name = Dict(_brm_stan_name(site) => site for site in plan.declarations)
    outputs = _brm_declaration_outputs(plan, trace, semantics, by_name)
    for twin in values(responses)
        o = parent.outputs[twin.name]
        formal = twin.logical
        push!(outputs, BRMOutput(o.name, o.kind, o.type, o.size, o.constraints, o.generative,
            formal, twin.role, get(by_name, formal, nothing), formal, o.labels, o.segments))
    end
    columns = Set(semantics.columns)
    inputs = NamedTuple[]
    # Only the caller's data inputs can supply observed or held-out values.
    # A fitted parent parameter stays a reference in the member's formula.
    for (formal, actual) in pairs(call.inputs)
        i = get(parent.inputs, actual, nothing)
        isnothing(i) && continue
        column, transform = _brm_input_schema(plan, formal, columns)
        push!(inputs, (; name=formal, column, transform, i.observed,
                         held_out=i.held_out || formal in held_out))
    end
    (; parent.id, plan, stan=trace, columns=semantics.columns, inputs, outputs)
end

function _brmd_member_inputs(axis, loop, d, index, plan, env)
    # Record ports are exact lowered bindings. Use the fitted member's data on
    # replay. Vector responses remain named references with their local axis.
    fields = map(axis.ports) do port
        raw = if !isnothing(plan) && haskey(plan.data,port.formal)
            plan.data[port.formal]
        elseif haskey(d.plan.data,port.column)
            d.plan.data[port.column][index]
        end
        raw isa Union{Number,AbstractString,Symbol} ? raw : BRMDescriptionReference(port.formal,:observation)
    end
    inputs = NamedTuple[]
    for (key,port) in pairs(axis.ports)
        v = fields[key]
        v isa BRMDescriptionReference && continue
        push!(inputs,(;name=port.formal,role=:constant,value=v,prior_ids=()))
    end
    args = map(eachindex(loop.arguments)) do i
        arg = loop.arguments[i]
        i == loop.record && return fields
        value = _brmd_value(arg,env,(:member,axis.collection,axis.keys[index],:argument,i))
        formal = loop.component.inputs[i]
        value isa BRMDescriptionReference && value.name === formal ||
            push!(inputs,(;name=formal,role=:alias,value,prior_ids=()))
        value
    end
    Tuple(inputs), Tuple(args)
end

# A fitted member sees its local response port; the parent carries predictive
# and log-likelihood draws under the projected input column. Segment ends are
# cumulative, including repeated ends for a member with zero readings.
function _brmd_member_responses(d, axis)
    outputs = [OrderedDict{Tuple{Symbol,Symbol},NamedTuple}() for _ in axis.keys]
    held_out = Set{Symbol}()
    for port in axis.ports
        port.column in d.plan.held_out && push!(held_out,port.formal)
        for o in d.outputs
            o.logical === port.column && o.source !== nothing || continue
            stan_output = only(s for s in d.stan.outputs if s.name === o.name)
            ends = isnothing(stan_output.groups) ? o.segments :
                [g == 0 ? 0 : o.segments[g] for g in stan_output.groups]
            common = (; o.name, logical=port.formal, o.role, o.kind, source=port.column,
                segments=_brmd_snapshot(o.segments))
            for (index,key) in enumerate(axis.keys)
                selection = isnothing(ends) ? index : _brm_member_segment(ends,index)
                output = (; common..., member=(axis.collection,key), selection)
                outputs[index][(port.formal,o.name)] = output
            end
        end
    end
    outputs, held_out
end

# Components without a formula or nested plate need no separate fit. The
# descriptor already holds their emitted SLIC declaration and its input ports.
function _brmd_simple_member(d, call, inputs, axis, index, env, notation, texts)
    values = Dict{Symbol,Any}(pairs(call.inputs))
    for b in inputs
        haskey(values,b.name) && (values[b.name]=b.value)
    end
    # A vector-valued data port of this member is one entry in a parent
    # column. Preserve its own values in the effective law.
    prior_values = copy(values)
    for port in axis.ports
        haskey(values,port.formal) && haskey(d.plan.data,port.column) || continue
        prior_values[port.formal]=d.plan.data[port.column][index]
        values[port.formal] isa Symbol && (values[port.formal]=BRMDescriptionReference(port.formal,:observation))
    end
    ps = BRMPriorDescription[]
    _brmd_submodel_priors!(ps,d,call.model,prior_values,(:parameter,),Dict(),notation,texts)
    priors=Tuple(ps)
    body=_brmd_included_model(merge(env,(;priors)),call.model,values,(:body,),(:parameter,))
    roots=(body,(p.distribution for p in priors if p.distribution isa BRMDescriptionComponent)...)
    (; roots,priors,notation=(),constants=())
end

function _brmd_members(d, axes, labels, anchors, texts, parent_priors, constants, notation)
    result = Dict{Symbol,Vector{BRMDescriptionComponent}}()
    isempty(axes) && return result
    calls = Dict((call.path...,call.target) => call for call in d.submodels)
    parent = (; d.id, inputs=Dict(i.name => i for i in d.inputs),
              outputs=Dict(o.name => o for o in d.outputs))
    selections = Dict((path[1:2],o.name) => (o.member isa Vector ? Tuple(o.member) : o.member)
        for (path,call) in calls if length(path) >= 2 for o in call.outputs if !isnothing(o.member))
    for axis in axes
        isempty(axis.path) && continue
        loop = only(st for st in axis.builder.statements if st isa _BRMCollectionLoop && st.target === axis.collection)
        fits = _brm_member_fits(d,axis)
        outputs_by_member, held_out = _brmd_member_responses(d,axis)
        trace = isnothing(fits) ? nothing : _brmd_member_trace(d,axis.path)
        roots = BRMDescriptionComponent[]
        env = _brmd_environment(d,axis.collection,:submodel,parent_priors,constants,notation,texts)
        for (index,key) in enumerate(axis.keys)
            member = (axis.collection,key)
            call = calls[member]
            fit = isnothing(fits) ? nothing : fits[index]
            plan = _brmd_member_plan(fit,call,axis,d.plan.data,index)
            inputs, arguments = _brmd_member_inputs(axis,loop,d,index,plan,env)
            parts = if isnothing(plan)
                _brmd_simple_member(d,call,inputs,axis,index,env,notation,texts)
            else
                fitted = _brmd_fitted_member(parent,call,plan,trace,outputs_by_member[index],held_out)
                _brmd_model_parts(fitted,labels,Dict(),texts)
            end
            scope = (; member, variant=axis.path, inputs, held_out, model_id=d.id, selections,
                output_names=isnothing(trace) ? (;) : trace.names,
                response_outputs=outputs_by_member[index], local_priors=Set(p.id for p in parts.priors),
                nodes=Dict{Tuple,BRMDescriptionComponent}(), priors=())
            priors = map(p -> _brmd_member_prior(p,scope,anchors),parts.priors)
            external = Tuple(arg for (i,arg) in enumerate(arguments) if i != loop.record)
            uses = _brmd_component(env,(:inputs,),:submodel,loop.component,external)
            inherited = brm_description_prior_references(uses)
            scope = merge(scope,(;priors=(priors..., (p for p in parent_priors if p.id in inherited)...)))
            children = Tuple(_brmd_member_value(c,scope) for c in parts.roots)
            id = _brmd_member_id(scope,())
            outputs = Tuple((;o.name,o.logical,o.role,o.kind,segments=_brmd_snapshot(o.segments),
                selection=o.member isa Vector ? Tuple(o.member) : o.member, member) for o in call.outputs)
            bindings = Tuple((;name,role=:alias,value,prior_ids=()) for
                (name,value) in zip(loop.component.inputs,arguments))
            provenance = merge(env.provenance,(;member, scope.variant,local_id=(),
                member_prior_ids=Tuple(p.id for p in priors)))
            wrapper = merge(env,(;priors=scope.priors, bindings, provenance, outputs, parts.notation, parts.constants))
            push!(roots,_brmd_component(wrapper,id,:submodel,loop.component,arguments;extra_children=children))
        end
        result[first(axis.path)] = roots
    end
    result
end

# Compare complete member explanations after hooks have run. A hook on a
# nested helper can distinguish two otherwise identical observation equations.
function _brmd_member_explanations(inventory, fragments, covered)
    explanations = OrderedDict{Tuple,Vector{Any}}()
    for c in inventory
        haskey(c.provenance,:member) || continue
        (; member,variant,local_id) = c.provenance
        fragment = get(fragments,c.id,nothing)
        rendered = if !isnothing(fragment)
            notes = map(fragment.notation) do note
                name = note.name
                # Built-in intermediate quantities are local to this equation.
                if name isa Tuple && length(name) >= 2 && name[2] == c.id &&
                        first(name) in (:description_expression,:kernel_binding)
                    return merge(note,(;name=(first(name),local_id,name[3:end]...)))
                end
                note
            end
            (fragment.prose, fragment.equations, notes)
        end
        push!(get!(Vector{Any},explanations,member),(variant,local_id,c.id in covered,rendered))
    end
    groups = Dict{Vector{Any},Vector{Tuple}}()
    Dict(member => push!(get!(Vector{Tuple},groups,parts),member) for (member,parts) in explanations)
end

function _brmd_sections(inventory, fragments, covered, texts)
    members = _brmd_member_explanations(inventory,fragments,covered)
    sections = NamedTuple[]
    notation = NamedTuple[]
    previous = nothing
    for c in inventory
        fragment = get(fragments,c.id,nothing)
        isnothing(fragment) && continue
        member = get(c.provenance,:member,nothing)
        group = get(members,member,nothing)
        isnothing(group) || first(group) == member || continue
        append!(notation,fragment.notation)
        isempty(fragment.prose) && isempty(fragment.equations) && continue
        section = (; id=c.id, prose=fragment.prose, equations=fragment.equations)
        if !isnothing(group)
            if group !== previous
                keys = join((_brmd_path_markdown(last(address)) for address in group),", ")
                heading = _brmd_text(texts,:collection_members; collection=first(member), keys)
                section = merge(section,(;prose=(heading,section.prose...)))
            end
            section = (; section..., members=Tuple(group))
        end
        push!(sections,section)
        previous = group
    end
    Tuple(sections), Tuple(notation)
end
