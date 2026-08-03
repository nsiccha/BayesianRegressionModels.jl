"""
Public, unbound declarations for the native PPL graph.

These values describe logical model semantics without training data, fitted
preprocessing state, flat coordinates, or workspaces. Direct Julia authors and
BRM lowerings construct the same `Model`; binding/compilation turns it into the
existing executable `Plan`.
"""

abstract type AbstractInputDeclaration end
abstract type AbstractPriorDeclaration end
abstract type AbstractNodeDeclaration end
abstract type AbstractObservationDeclaration end

"""
A named open value port. The surrounding `NamedTuple` key is its identity.

`Role` remains as typed provenance for the transitional BRM compiler, but it
does not determine whether a connected value is data, a parameter, or a
deterministic graph value. New direct declarations use generic `:value` ports.
"""
struct Input{Role} <: AbstractInputDeclaration end

input() = Input{:value}()

function input(role::Symbol)
    role in (:value, :predictor, :response, :data) || throw(ArgumentError(
        "native PPL input role must be :value, :predictor, :response, or " *
        ":data; got $role"))
    Input{role}()
end

input_role(::Input{Role}) where {Role} = Role

const RealSupport = BRM.NativePPLRealSupport
const PositiveSupport = BRM.NativePPLPositiveSupport
const IdentityTransform = BRM.NativePPLIdentityTransform
const ExpTransform = BRM.NativePPLExpTransform

Identity() = IdentityTransform()
Exp() = ExpTransform()

"""Independent standard-normal prior declaration."""
struct StandardNormal <: AbstractPriorDeclaration end

"""Scalar Normal prior declaration with literal location and scale."""
struct NormalPrior{L<:Real,S<:Real} <: AbstractPriorDeclaration
    location::L
    scale::S

    function NormalPrior(location::L, scale::S) where {L<:Real,S<:Real}
        isfinite(location) || throw(ArgumentError(
            "native PPL Normal prior location must be finite"))
        isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
            "native PPL Normal prior scale must be finite and positive"))
        new{L,S}(location, scale)
    end
end

normal_prior(location::Real, scale::Real) = NormalPrior(location, scale)

"""Exponential prior declaration parameterized by its positive scale."""
struct ExponentialPrior{T<:Real} <: AbstractPriorDeclaration
    scale::T

    function ExponentialPrior(scale::T) where {T<:Real}
        isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
            "native PPL Exponential prior scale must be finite and positive"))
        new{T}(scale)
    end
end

Exponential(scale::Real) = ExponentialPrior(scale)

"""
An unbound logical parameter declaration.

`axis_keys` name the constrained scalar coordinates. Compilation assigns their
contiguous locations in the flat unconstrained vector.
"""
struct Parameter{S,Tr,K,P<:AbstractPriorDeclaration}
    support::S
    transform::Tr
    axis_keys::K
    prior::P
end

function parameter(support::S, axis_keys::Tuple;
                   transform::Tr, prior::P) where {
                       S<:BRM.NativePPLSupport,
                       Tr<:BRM.NativePPLTransform,
                       P<:AbstractPriorDeclaration,
                   }
    isempty(axis_keys) && throw(ArgumentError(
        "native PPL parameter axis must contain at least one key"))
    all(key -> key isa Symbol, axis_keys) || throw(ArgumentError(
        "native PPL parameter axis keys must be Symbols"))
    length(unique(axis_keys)) == length(axis_keys) || throw(ArgumentError(
        "native PPL parameter axis keys must be unique; got $axis_keys"))
    transform isa BRM.NativePPLTransform{S} || throw(ArgumentError(
        "native PPL transform $(typeof(transform)) does not target support $S"))
    if prior isa StandardNormal
        support isa RealSupport || throw(ArgumentError(
            "native PPL StandardNormal prior requires RealSupport"))
    elseif prior isa NormalPrior
        support isa RealSupport || throw(ArgumentError(
            "native PPL Normal prior requires RealSupport"))
        length(axis_keys) == 1 || throw(ArgumentError(
            "native PPL Normal prior currently requires one scalar coordinate"))
    elseif prior isa ExponentialPrior
        support isa PositiveSupport || throw(ArgumentError(
            "native PPL Exponential prior requires PositiveSupport"))
        length(axis_keys) == 1 || throw(ArgumentError(
            "native PPL Exponential prior currently requires one scalar coordinate"))
    end
    Parameter(support, transform, axis_keys, prior)
end

"""Fit and apply mean centering to one raw input."""
struct Center{Input} <: AbstractNodeDeclaration end
center(input::Symbol) = Center{input}()

"""Fit and apply corrected-sample-SD standardization to one raw input."""
struct ZScale{Input} <: AbstractNodeDeclaration end
zscale(input::Symbol) = ZScale{input}()
standardize(input::Symbol) = zscale(input)

"""Intercept plus one slope per named input/node over one parameter block."""
struct Affine{Inputs,Coefficients} <: AbstractNodeDeclaration end
function affine(inputs::Tuple, coefficients::Symbol)
    isempty(inputs) && throw(ArgumentError(
        "native PPL affine declaration requires at least one input"))
    all(input -> input isa Symbol, inputs) || throw(ArgumentError(
        "native PPL affine inputs must be named Symbols; got $inputs"))
    length(unique(inputs)) == length(inputs) || throw(ArgumentError(
        "native PPL affine inputs must be unique; got $inputs"))
    Affine{inputs,coefficients}()
end
affine(input::Symbol, coefficients::Symbol) = affine((input,), coefficients)

"""Elementwise exponential link over one named deterministic node."""
struct ExpLink{Input} <: AbstractNodeDeclaration end
exp_link(input::Symbol) = ExpLink{input}()

node_input(::Center{Input}) where {Input} = Input
node_input(::ZScale{Input}) where {Input} = Input
node_inputs(::Affine{Inputs}) where {Inputs} = Inputs
function node_input(node::Affine)
    inputs = node_inputs(node)
    length(inputs) == 1 || throw(ArgumentError(
        "native PPL affine declaration has $(length(inputs)) inputs; use " *
        "`node_inputs`"))
    only(inputs)
end
node_input(::ExpLink{Input}) where {Input} = Input
affine_parameter(::Affine{Inputs,Coefficients}) where {Inputs,Coefficients} =
    Coefficients

"""One scalar Normal stochastic-site declaration."""
struct NormalObservation{Response,Location,Scale} <:
       AbstractObservationDeclaration end
normal(response::Symbol, location::Symbol, scale::Symbol) =
    NormalObservation{response,location,scale}()

"""One scalar Bernoulli stochastic-site declaration parameterized by logits."""
struct BernoulliLogitObservation{Response,Logit} <:
       AbstractObservationDeclaration end
bernoulli_logit(response::Symbol, logit::Symbol) =
    BernoulliLogitObservation{response,logit}()

"""One scalar Poisson stochastic-site declaration parameterized by a positive rate."""
struct PoissonObservation{Response,Rate} <:
       AbstractObservationDeclaration end
poisson(response::Symbol, rate::Symbol) =
    PoissonObservation{response,rate}()

"""Explicit Julia-broadcast lifting of one scalar stochastic-site declaration."""
struct BroadcastObservation{O<:AbstractObservationDeclaration} <:
       AbstractObservationDeclaration
    scalar::O
end

broadcasted(observation::AbstractObservationDeclaration) =
    BroadcastObservation(observation)

scalar_observation(observation::AbstractObservationDeclaration) = observation
scalar_observation(observation::BroadcastObservation) = observation.scalar
is_broadcast_observation(::AbstractObservationDeclaration) = false
is_broadcast_observation(::BroadcastObservation) = true

observation_response(::NormalObservation{Response}) where {Response} = Response
observation_response(::BernoulliLogitObservation{Response}) where {Response} =
    Response
observation_response(::PoissonObservation{Response}) where {Response} = Response
observation_response(observation::BroadcastObservation) =
    observation_response(observation.scalar)
observation_dependencies(
    ::NormalObservation{Response,Location,Scale},
) where {Response,Location,Scale} = (Location, Scale)
observation_dependencies(
    ::BernoulliLogitObservation{Response,Logit},
) where {Response,Logit} = (Logit,)
observation_dependencies(
    ::PoissonObservation{Response,Rate},
) where {Response,Rate} = (Rate,)
observation_dependencies(observation::BroadcastObservation) =
    observation_dependencies(observation.scalar)

"""
An unbound, typed native-PPL declaration shared by direct authors and BRM.

Names are the keys of the declaration `NamedTuple`s. `outputs` is either
`nothing` for the legacy implicit-export surface or a named map from public
output aliases to graph identities. The model owns logical graph semantics
only; data binding, fitted transform state, axes, coordinates, output
signatures, and workspace layout belong to compilation/preparation.
"""
struct Model{I,P,N,O,R}
    inputs::I
    parameters::P
    nodes::N
    observations::O
    outputs::R
end

Model(inputs, parameters, nodes, observations) =
    Model(inputs, parameters, nodes, observations, nothing)

"""
A composed call of a staged `@model` function.

`bindings` connect open value ports to arbitrary Julia or graph values.
`conditions` attach realized values to stochastic sites while retaining their
generating factors. They remain separate because substitution and conditioning
have different probability semantics.
"""
struct ModelInstance{M<:Model,B,C}
    declaration::M
    bindings::B
    conditions::C
end

"""
A stable reference to one named value exported by a namespaced component.

`Namespace`, `Name`, and `Kind` are part of the type so composition preserves
identity without flattening unrelated component-local names. Kinds currently
distinguish connected input values, parameter blocks, deterministic nodes, and
stochastic sites; execution activity is deliberately derived later.
"""
struct GraphRef{Namespace,Name,Kind}
    function GraphRef{Namespace,Name,Kind}() where {Namespace,Name,Kind}
        Namespace isa Symbol || throw(ArgumentError(
            "native PPL graph reference namespace must be a Symbol; got " *
            "$(repr(Namespace))"))
        Name isa Symbol || throw(ArgumentError(
            "native PPL graph reference name must be a Symbol; got " *
            "$(repr(Name))"))
        Kind in (:binding, :parameter, :node, :site) || throw(ArgumentError(
            "native PPL graph reference kind must be one of `:binding`, " *
            "`:parameter`, `:node`, or `:site`; got $(repr(Kind))"))
        new{Namespace,Name,Kind}()
    end
end

graph_namespace(::GraphRef{Namespace}) where {Namespace} = Namespace
graph_name(::GraphRef{Namespace,Name}) where {Namespace,Name} = Name
graph_kind(::GraphRef{Namespace,Name,Kind}) where {Namespace,Name,Kind} = Kind

"""One model instance under a collision-safe composition namespace."""
struct Component{Namespace,I<:ModelInstance}
    instance::I
    function Component{Namespace,I}(instance::I) where {Namespace,I<:ModelInstance}
        Namespace isa Symbol || throw(ArgumentError(
            "native PPL component namespace must be a Symbol; got " *
            "$(repr(Namespace))"))
        isempty(string(Namespace)) && throw(ArgumentError(
            "native PPL component namespace must not be empty"))
        _validate_instance(instance)
        new{Namespace,I}(instance)
    end
end

component_namespace(::Component{Namespace}) where {Namespace} = Namespace

function component(namespace::Symbol, instance::ModelInstance)
    Component{namespace,typeof(instance)}(instance)
end

component(namespace::Symbol, declaration::Model) =
    component(namespace, ModelInstance(declaration, (;)))

function _component_local_output_name(component::Component, name::Symbol)
    outputs = component.instance.declaration.outputs
    outputs === nothing && return name
    hasproperty(outputs, name) || throw(ArgumentError(
        "native PPL component `$(component_namespace(component))` does not " *
        "export `$name`; declared outputs are $(keys(outputs))"))
    getproperty(outputs, name)
end

function _component_output_kind(component::Component, name::Symbol)
    instance = component.instance
    declaration = instance.declaration
    local_name = _component_local_output_name(component, name)
    if hasproperty(declaration.inputs, local_name)
        hasproperty(instance.bindings, local_name) || throw(ArgumentError(
            "native PPL component `$(component_namespace(component))` cannot " *
            "export open value port `$local_name`; connect it before exporting it"))
        return :binding
    elseif hasproperty(declaration.parameters, local_name)
        return :parameter
    elseif hasproperty(declaration.nodes, local_name)
        return :node
    elseif hasproperty(declaration.observations, local_name)
        return :site
    end
    throw(ArgumentError(
        "native PPL component `$(component_namespace(component))` has no " *
        "exportable value `$name`"))
end

"""Reference a connected port, parameter, node, or stochastic site output."""
function output(component::Component{Namespace}, name::Symbol) where {Namespace}
    kind = _component_output_kind(component, name)
    GraphRef{Namespace,name,kind}()
end

"""
A topologically ordered collection of namespaced model components.

Bindings may contain ordinary Julia values or `GraphRef`s. A reference may
only point to an earlier component, making cycles and forward references fail
at declaration time. Compilation of graph-valued connections is a separate
stage from this collision-safe public composition boundary.
"""
struct Composition{C<:NamedTuple}
    components::C
    function Composition(components::C) where {C<:NamedTuple}
        _validate_composition(components)
        new{C}(components)
    end
end

Composition(components) = throw(ArgumentError(
    "native PPL composition components must be a NamedTuple; got " *
    "$(typeof(components))"))

function _validate_graph_reference(reference::GraphRef, available, components)
    namespace = graph_namespace(reference)
    namespace in available || throw(ArgumentError(
        "native PPL graph reference `$(namespace).$(graph_name(reference))` " *
        "must target an earlier component; available namespaces are " *
        "$(Tuple(available))"))
    source = getproperty(components, namespace)
    expected_kind = _component_output_kind(source, graph_name(reference))
    expected_kind === graph_kind(reference) || throw(ArgumentError(
        "native PPL graph reference `$(namespace).$(graph_name(reference))` " *
        "has kind `$(graph_kind(reference))`, expected `$expected_kind`"))
    nothing
end

function _validate_composition(components::NamedTuple)
    isempty(components) && throw(ArgumentError(
        "native PPL composition requires at least one component"))
    available = Symbol[]
    for (namespace, component_value) in pairs(components)
        component_value isa Component || throw(ArgumentError(
            "native PPL composition `$namespace` must be a Component; got " *
            "$(typeof(component_value))"))
        component_namespace(component_value) === namespace || throw(ArgumentError(
            "native PPL component namespace `$(component_namespace(component_value))` " *
            "does not match composition key `$namespace`"))
        _validate_instance(component_value.instance)
        for value in values(component_value.instance.bindings)
            value isa GraphRef &&
                _validate_graph_reference(value, available, components)
        end
        for value in values(component_value.instance.conditions)
            value isa GraphRef &&
                _validate_graph_reference(value, available, components)
        end
        push!(available, namespace)
    end
    components
end

function compose(components::Component...)
    isempty(components) && throw(ArgumentError(
        "native PPL compose requires at least one component"))
    namespaces = map(component_namespace, components)
    length(unique(namespaces)) == length(namespaces) || throw(ArgumentError(
        "native PPL component namespaces must be unique; got $namespaces"))
    values = NamedTuple{namespaces}(components)
    Composition(values)
end

"""Collision-free flattened identity for a component-local graph name."""
function qualified_name(namespace::Symbol, name::Symbol)
    isempty(string(namespace)) && throw(ArgumentError(
        "native PPL component namespace must not be empty"))
    Symbol(
        "#component#", ncodeunits(string(namespace)), ":", namespace, "#", name)
end

function _qualified_parameter(
    namespace::Symbol, name::Symbol, declaration::Parameter)
    axis_keys = Tuple(qualified_name(namespace, key)
                      for key in declaration.axis_keys)
    parameter(
        declaration.support, axis_keys;
        transform=declaration.transform, prior=declaration.prior)
end

_mapped_name(mapping, name::Symbol, owner::Symbol, kind::AbstractString) =
    get(mapping, name) do
        throw(CapabilityError(
            :composition_identity,
            "component `$owner` $kind references unavailable local value `$name`"))
    end

function _qualified_node(namespace::Symbol, declaration::Center, mapping)
    center(_mapped_name(
        mapping, node_input(declaration), namespace, "center node"))
end

function _qualified_node(namespace::Symbol, declaration::ZScale, mapping)
    zscale(_mapped_name(
        mapping, node_input(declaration), namespace, "zscale node"))
end


function _qualified_node(namespace::Symbol, declaration::Affine, mapping)
    inputs = Tuple(_mapped_name(
        mapping, name, namespace, "affine node")
        for name in node_inputs(declaration))
    coefficients = _mapped_name(
        mapping, affine_parameter(declaration), namespace,
        "affine coefficient")
    affine(inputs, coefficients)
end


function _qualified_node(namespace::Symbol, declaration::ExpLink, mapping)
    exp_link(_mapped_name(
        mapping, node_input(declaration), namespace, "exp node"))
end


function _qualified_observation(namespace::Symbol, observation, mapping)
    scalar = scalar_observation(observation)
    response = qualified_name(namespace, observation_response(scalar))
    declaration = if scalar isa NormalObservation
        dependencies = observation_dependencies(scalar)
        normal(
            response,
            _mapped_name(mapping, dependencies[1], namespace, "Normal factor"),
            _mapped_name(mapping, dependencies[2], namespace, "Normal factor"))
    elseif scalar isa BernoulliLogitObservation
        bernoulli_logit(
            response,
            _mapped_name(
                mapping, only(observation_dependencies(scalar)),
                namespace, "Bernoulli-logit factor"))
    else
        poisson(
            response,
            _mapped_name(
                mapping, only(observation_dependencies(scalar)),
                namespace, "Poisson factor"))
    end
    is_broadcast_observation(observation) ? broadcasted(declaration) : declaration
end

function _resolved_reference(reference::GraphRef, resolved)
    key = (graph_namespace(reference), graph_name(reference))
    haskey(resolved, key) || throw(CapabilityError(
        :composition_identity,
        "component output `$(first(key)).$(last(key))` has no resolved identity"))
    resolved[key]
end

function _push_declaration!(names, values, name::Symbol, value, kind)
    name in names && throw(CapabilityError(
        :composition_identity,
        "composed $kind identity `$name` is not unique"))
    push!(names, name)
    push!(values, value)
    nothing
end

function _component_local_value_active(
    component_value::Component, name::Symbol, components, cache)
    namespace = component_namespace(component_value)
    key = (namespace, name)
    haskey(cache, key) && return cache[key]
    declaration = component_value.instance.declaration
    active = if hasproperty(declaration.inputs, name)
        if hasproperty(component_value.instance.bindings, name)
            value = getproperty(component_value.instance.bindings, name)
            value isa GraphRef ?
                _reference_active(value, components, cache) : false
        else
            false
        end
    elseif hasproperty(declaration.parameters, name)
        true
    elseif hasproperty(declaration.nodes, name)
        node = getproperty(declaration.nodes, name)
        node isa Affine || _component_local_value_active(
            component_value, node_input(node), components, cache)
    elseif hasproperty(declaration.observations, name)
        true
    else
        throw(CapabilityError(
            :composition_identity,
            "component `$namespace` has no value `$name` for activity analysis"))
    end
    cache[key] = active
    active
end

function _component_value_active(
    component_value::Component, name::Symbol, components, cache)
    local_name = _component_local_output_name(component_value, name)
    _component_local_value_active(component_value, local_name, components, cache)
end

function _reference_active(reference::GraphRef, components, cache)
    source = getproperty(components, graph_namespace(reference))
    _component_value_active(source, graph_name(reference), components, cache)
end

function _lower_composition(composition::Composition, public_outputs=nothing)
    components = _validate_composition(composition.components)
    input_names = Symbol[]
    input_values = Any[]
    parameter_names = Symbol[]
    parameter_values = Any[]
    node_names = Symbol[]
    node_values = Any[]
    observation_names = Symbol[]
    observation_values = Any[]
    binding_names = Symbol[]
    binding_values = Any[]
    condition_names = Symbol[]
    condition_values = Any[]
    resolved = Dict{Tuple{Symbol,Symbol},Symbol}()
    activity = Dict{Tuple{Symbol,Symbol},Bool}()

    for (namespace, component_value) in pairs(components)
        instance = component_value.instance
        declaration = instance.declaration
        mapping = Dict{Symbol,Symbol}()

        for (name, input_declaration) in pairs(declaration.inputs)
            if hasproperty(instance.bindings, name)
                value = getproperty(instance.bindings, name)
                if value isa GraphRef
                    if graph_kind(value) === :site
                        source = getproperty(
                            components, graph_namespace(value))
                        local_site = _component_local_output_name(
                            source, graph_name(value))
                        site_declaration = getproperty(
                            source.instance.declaration.observations,
                            local_site)
                        is_broadcast_observation(site_declaration) && throw(
                            CapabilityError(
                                :active_site_connection,
                                "component `$namespace` port `$name` consumes " *
                                "broadcast stochastic-site output " *
                                "`$(graph_namespace(value)).$(graph_name(value))`; " *
                                "the first factor-to-value schedule accepts a " *
                                "scalar site"))
                    end
                    _reference_active(value, components, activity)
                    mapping[name] = _resolved_reference(value, resolved)
                    resolved[(namespace, name)] = mapping[name]
                    continue
                end
            end
            qualified = qualified_name(namespace, name)
            mapping[name] = qualified
            resolved[(namespace, name)] = qualified
            _push_declaration!(
                input_names, input_values, qualified, input_declaration, "input")
            if hasproperty(instance.bindings, name)
                push!(binding_names, qualified)
                push!(binding_values, getproperty(instance.bindings, name))
            end
        end

        for (name, parameter_declaration) in pairs(declaration.parameters)
            qualified = qualified_name(namespace, name)
            mapping[name] = qualified
            resolved[(namespace, name)] = qualified
            _push_declaration!(
                parameter_names, parameter_values, qualified,
                _qualified_parameter(namespace, name, parameter_declaration),
                "parameter")
        end

        for (name, node_declaration) in pairs(declaration.nodes)
            qualified = qualified_name(namespace, name)
            mapping[name] = qualified
            resolved[(namespace, name)] = qualified
            _push_declaration!(
                node_names, node_values, qualified,
                _qualified_node(namespace, node_declaration, mapping), "node")
        end

        for (name, observation_declaration) in pairs(declaration.observations)
            qualified = qualified_name(namespace, name)
            mapping[name] = qualified
            resolved[(namespace, name)] = qualified
            _push_declaration!(
                observation_names, observation_values, qualified,
                _qualified_observation(namespace, observation_declaration, mapping),
                "observation")
            if hasproperty(instance.conditions, name)
                value = getproperty(instance.conditions, name)
                value isa GraphRef && throw(CapabilityError(
                    :active_condition,
                    "component `$namespace` conditions `$name` on active graph " *
                    "output `$(graph_namespace(value)).$(graph_name(value))`; " *
                    "active evidence lowering is not available yet"))
                push!(condition_names, qualified)
                push!(condition_values, value)
            end
        end

        if declaration.outputs !== nothing
            for (alias, local_name) in pairs(declaration.outputs)
                resolved[(namespace, alias)] = resolved[(namespace, local_name)]
            end
        end
    end

    flattened_outputs = if public_outputs === nothing
        nothing
    else
        public_outputs isa NamedTuple || throw(ArgumentError(
            "native PPL staged outputs must be a NamedTuple; got " *
            "$(typeof(public_outputs))"))
        all(value -> value isa GraphRef, values(public_outputs)) || throw(
            ArgumentError(
                "native PPL staged outputs must be graph values"))
        names = keys(public_outputs)
        flattened_values = Tuple(
            _resolved_reference(reference, resolved)
            for reference in Base.values(public_outputs))
        NamedTuple{names}(flattened_values)
    end

    declaration = model(
        inputs=NamedTuple{Tuple(input_names)}(Tuple(input_values)),
        parameters=NamedTuple{Tuple(parameter_names)}(Tuple(parameter_values)),
        nodes=NamedTuple{Tuple(node_names)}(Tuple(node_values)),
        observations=NamedTuple{Tuple(observation_names)}(
            Tuple(observation_values)),
        outputs=flattened_outputs)
    ModelInstance(
        declaration,
        NamedTuple{Tuple(binding_names)}(Tuple(binding_values)),
        NamedTuple{Tuple(condition_names)}(Tuple(condition_values)))
end

lower(composition::Composition) = _lower_composition(composition)

function _staged_component_output(
    namespace::Symbol, value, connection::Symbol)
    value isa ModelInstance || throw(ArgumentError(
        "native PPL staged call `$namespace` must return a ModelInstance; " *
        "got $(typeof(value))"))
    _validate_instance(value)
    outputs = value.declaration.outputs
    outputs isa NamedTuple && length(outputs) == 1 || throw(ArgumentError(
        "native PPL staged call `$namespace` must return exactly one named " *
        "graph value; got $(outputs === nothing ? 0 : length(outputs))"))
    alias = only(keys(outputs))
    staged_component = component(namespace, value)
    reference = output(staged_component, alias)
    kind = graph_kind(reference)
    valid = connection === :stochastic ? kind in (:parameter, :site) :
        connection === :deterministic ? kind in (:binding, :node) : false
    valid || throw(ArgumentError(
        "native PPL `$connection` staged call `$namespace` returned a " *
        "`$kind` graph value"))
    staged_component, reference
end

_staged_stochastic_component(namespace::Symbol, value) =
    _staged_component_output(namespace, value, :stochastic)

_staged_deterministic_component(namespace::Symbol, value) =
    _staged_component_output(namespace, value, :deterministic)

function _staged_composition(components::Tuple, outputs::NamedTuple)
    composition = compose(components...)
    _lower_composition(composition, outputs)
end

function bind(composition::Composition)
    lowered = lower(composition)
    bind(lowered)
end

compile(composition::Composition) = bind(composition)
prepare(composition::Composition; kwargs...) =
    prepare(compile(composition); kwargs...)

function Base.show(io::IO, reference::GraphRef)
    print(io, "NativePPL.GraphRef(", graph_namespace(reference), ".",
          graph_name(reference), ", kind=", graph_kind(reference), ")")
end

function Base.show(io::IO, component::Component)
    print(io, "NativePPL.Component(", component_namespace(component), ", ")
    show(io, component.instance)
    print(io, ")")
end

function Base.show(io::IO, composition::Composition)
    print(io, "NativePPL.Composition(components=",
          keys(composition.components), ")")
end

ModelInstance(declaration::Model, bindings) =
    ModelInstance(declaration, bindings, (;))

function instantiate(declaration::Model, bindings; conditions=(;))
    _validate_model(declaration)
    _validate_binding_names(declaration, bindings)
    canonical_conditions = _canonical_conditions(declaration, conditions)
    ModelInstance(declaration, bindings, canonical_conditions)
end

"""Connect a subset of a model's open value ports."""
function substitute(declaration::Model, bindings)
    _validate_model(declaration)
    _validate_substitution_names(declaration, bindings)
    ModelInstance(declaration, bindings, (;))
end

function substitute(instance::ModelInstance, bindings)
    _validate_instance(instance)
    _validate_substitution_names(instance.declaration, bindings)
    ModelInstance(
        instance.declaration, merge(instance.bindings, bindings),
        instance.conditions)
end

"""Condition stochastic sites while retaining and scoring their factors."""
function condition(declaration::Model, conditions)
    _validate_model(declaration)
    canonical_conditions = _canonical_conditions(declaration, conditions)
    ModelInstance(declaration, (;), canonical_conditions)
end

function condition(instance::ModelInstance, conditions)
    _validate_instance(instance)
    canonical_conditions = _canonical_conditions(
        instance.declaration, conditions)
    ModelInstance(
        instance.declaration, instance.bindings,
        merge(instance.conditions, canonical_conditions))
end

condition(declaration::Model; kwargs...) = condition(declaration, (; kwargs...))
condition(instance::ModelInstance; kwargs...) = condition(instance, (; kwargs...))
substitute(declaration::Model; kwargs...) = substitute(declaration, (; kwargs...))
substitute(instance::ModelInstance; kwargs...) = substitute(instance, (; kwargs...))

function Base.show(io::IO, instance::ModelInstance)
    print(io, "NativePPL.ModelInstance(")
    show(io, instance.declaration)
    print(io, ", bindings=", keys(instance.bindings),
          ", conditions=", keys(instance.conditions), ")")
end

function _check_named_declarations(values, kind::AbstractString, type)
    values isa NamedTuple || throw(ArgumentError(
        "native PPL $kind declarations must be a NamedTuple; got $(typeof(values))"))
    for (name, value) in pairs(values)
        value isa type || throw(ArgumentError(
            "native PPL $kind `$name` must be a $type; got $(typeof(value))"))
    end
    values
end

function _validate_model_components(
    inputs, parameters, nodes, observations, outputs)
    _check_named_declarations(
        inputs, "input", AbstractInputDeclaration)
    _check_named_declarations(
        parameters, "parameter", Parameter)
    _check_named_declarations(
        nodes, "node", AbstractNodeDeclaration)
    _check_named_declarations(
        observations, "observation", AbstractObservationDeclaration)

    input_names = Set(keys(inputs))
    parameter_names = Set(keys(parameters))
    node_names = Set(keys(nodes))
    isempty(intersect(input_names, parameter_names)) || throw(ArgumentError(
        "native PPL input and parameter identities must be distinct"))
    isempty(intersect(input_names, node_names)) || throw(ArgumentError(
        "native PPL input and node identities must be distinct"))
    isempty(intersect(parameter_names, node_names)) || throw(ArgumentError(
        "native PPL parameter and node identities must be distinct"))

    available = union(input_names, parameter_names)
    for (name, declaration) in pairs(nodes)
        dependencies = declaration isa Affine ?
            node_inputs(declaration) : (node_input(declaration),)
        for dependency in dependencies
            dependency in available || throw(ArgumentError(
                "native PPL node `$name` references unavailable input " *
                "`$dependency`; nodes must be topologically ordered"))
        end
        declaration isa Affine &&
            affine_parameter(declaration) ∉ parameter_names &&
            throw(ArgumentError(
                "native PPL affine node `$name` references unknown parameter " *
                "`$(affine_parameter(declaration))`"))
        push!(available, name)
    end

    for (name, declaration) in pairs(observations)
        response = observation_response(declaration)
        name === response || throw(ArgumentError(
            "native PPL observation key `$name` must match response identity `$response`"))
        if response in input_names
            input_role(getproperty(inputs, response)) === :response ||
                throw(ArgumentError(
                    "native PPL legacy observation input `$response` must use " *
                    "role :response; direct stochastic sites must not also be " *
                    "declared as value ports"))
        else
            response in parameter_names && throw(ArgumentError(
                "native PPL stochastic site `$response` collides with a " *
                "parameter identity"))
            response in node_names && throw(ArgumentError(
                "native PPL stochastic site `$response` collides with a node identity"))
        end
        for dependency in observation_dependencies(declaration)
            dependency in available || throw(ArgumentError(
                "native PPL observation `$name` references unavailable dependency " *
                "`$dependency`"))
        end
        push!(available, response)
    end

    if outputs !== nothing
        outputs isa NamedTuple || throw(ArgumentError(
            "native PPL model outputs must be a NamedTuple; got " *
            "$(typeof(outputs))"))
        isempty(outputs) && throw(ArgumentError(
            "native PPL model outputs must declare at least one named graph value"))
        all(name -> name isa Symbol, values(outputs)) || throw(ArgumentError(
            "native PPL model outputs must reference named graph values"))
        length(unique(values(outputs))) == length(outputs) || throw(
            ArgumentError(
                "native PPL model outputs must reference distinct graph values"))
        exportable = union(
            input_names, parameter_names, node_names, Set(keys(observations)))
        for (alias, name) in pairs(outputs)
            name in exportable || throw(ArgumentError(
                "native PPL model output `$alias` references unavailable " *
                "value `$name`"))
        end
    end

    nothing
end

function model(;
    inputs, parameters=(;), nodes=(;), observations, outputs=nothing)
    _validate_model_components(
        inputs, parameters, nodes, observations, outputs)
    Model(inputs, parameters, nodes, observations, outputs)
end

function _validate_model(declaration::Model)
    _validate_model_components(
        declaration.inputs, declaration.parameters,
        declaration.nodes, declaration.observations, declaration.outputs)
    declaration
end

function _validate_binding_names(declaration::Model, bindings)
    bindings isa NamedTuple || throw(ArgumentError(
        "native PPL bindings must be a NamedTuple; got $(typeof(bindings))"))
    missing_bindings = setdiff(Set(keys(declaration.inputs)), Set(keys(bindings)))
    isempty(missing_bindings) || throw(ArgumentError(
        "native PPL bindings are missing declared inputs: " *
        join(sort!(collect(missing_bindings)), ", ")))
    extra_bindings = setdiff(Set(keys(bindings)), Set(keys(declaration.inputs)))
    isempty(extra_bindings) || throw(ArgumentError(
        "native PPL bindings contain undeclared inputs: " *
        join(sort!(collect(extra_bindings)), ", ")))
    bindings
end

function _validate_substitution_names(declaration::Model, bindings)
    bindings isa NamedTuple || throw(ArgumentError(
        "native PPL substitutions must be a NamedTuple; got $(typeof(bindings))"))
    extra_bindings = setdiff(Set(keys(bindings)), Set(keys(declaration.inputs)))
    isempty(extra_bindings) || throw(ArgumentError(
        "native PPL substitutions reference undeclared value ports: " *
        join(sort!(collect(extra_bindings)), ", ")))
    bindings
end

function _canonical_conditions(declaration::Model, conditions)
    conditions isa NamedTuple || throw(ArgumentError(
        "native PPL conditions must be a NamedTuple; got $(typeof(conditions))"))
    names = Symbol[]
    values = Any[]
    for (name, value) in pairs(conditions)
        canonical = if hasproperty(declaration.observations, name)
            name
        elseif declaration.outputs !== nothing &&
               hasproperty(declaration.outputs, name)
            output_name = getproperty(declaration.outputs, name)
            hasproperty(declaration.observations, output_name) || throw(
                ArgumentError(
                    "native PPL condition `$name` refers to output " *
                    "`$output_name`, which is not a stochastic site"))
            output_name
        else
            throw(ArgumentError(
                "native PPL conditions reference undeclared stochastic " *
                "site `$name`"))
        end
        canonical in names && throw(ArgumentError(
            "native PPL conditions name stochastic site `$canonical` more " *
            "than once through public and internal identities"))
        push!(names, canonical)
        push!(values, value)
    end
    NamedTuple{Tuple(names)}(Tuple(values))
end

_validate_condition_names(declaration::Model, conditions) =
    (_canonical_conditions(declaration, conditions); conditions)

function _validate_instance(instance::ModelInstance)
    _validate_model(instance.declaration)
    _validate_substitution_names(instance.declaration, instance.bindings)
    _validate_condition_names(instance.declaration, instance.conditions)
    instance
end

function _validated_plan(plan::Plan)
    BRM._native_ppl_validate_predictor_graph(plan)
    plan
end

function Base.show(io::IO, declaration::Model)
    print(io, "NativePPL.Model(inputs=", keys(declaration.inputs),
          ", parameters=", keys(declaration.parameters),
          ", nodes=", keys(declaration.nodes),
          ", observations=", keys(declaration.observations))
    declaration.outputs === nothing ||
        print(io, ", outputs=", keys(declaration.outputs))
    print(io, ")")
end

function _one_declaration(values::NamedTuple, type, kind::AbstractString)
    matches = [(name, value) for (name, value) in pairs(values)
               if value isa type]
    length(matches) == 1 || throw(CapabilityError(
        :declaration_shape,
        "the current native compiler requires exactly one $kind; found " *
        "$(length(matches))"))
    only(matches)
end

function _binding(bindings, name::Symbol, role::Symbol)
    hasproperty(bindings, name) || throw(ArgumentError(
        "native PPL binding is missing $role input `$name`"))
    value = getproperty(bindings, name)
    value isa AbstractVector || throw(ArgumentError(
        "native PPL $role input `$name` must be an AbstractVector; got " *
        "$(typeof(value))"))
    value
end

function _compile_transform(declaration::Center, name::Symbol,
                            predictor_name::Symbol, axis, predictor)
    node_input(declaration) === predictor_name || throw(CapabilityError(
        :declaration_shape,
        "fitted center node `$name` must consume predictor `$predictor_name`"))
    BRM.NativePPLCenterNode(
        name, predictor_name, axis,
        BRM._native_ppl_fit_center(predictor, predictor_name))
end

function _compile_transform(declaration::ZScale, name::Symbol,
                            predictor_name::Symbol, axis, predictor)
    node_input(declaration) === predictor_name || throw(CapabilityError(
        :declaration_shape,
        "fitted zscale node `$name` must consume predictor `$predictor_name`"))
    fit = BRM._native_ppl_fit_zscale(predictor, predictor_name)
    BRM.NativePPLZScaleNode(
        name, predictor_name, axis, fit.mean, fit.scale)
end

function _validate_coefficient_parameter(
    name::Symbol, declaration::Parameter, predictor_count::Int)
    declaration.support isa RealSupport || throw(CapabilityError(
        :parameter_support,
        "affine coefficient parameter `$name` must use RealSupport"))
    declaration.transform isa IdentityTransform || throw(CapabilityError(
        :parameter_transform,
        "affine coefficient parameter `$name` must use Identity()"))
    valid_prior = declaration.prior isa StandardNormal ||
        (predictor_count == 0 && declaration.prior isa NormalPrior)
    valid_prior || throw(CapabilityError(
        :parameter_prior,
        "affine coefficient parameter `$name` must use StandardNormal(); " *
        "a scalar location may instead use NormalPrior"))
    expected = predictor_count + 1
    length(declaration.axis_keys) == expected || throw(CapabilityError(
        :parameter_axis,
        "affine coefficient parameter `$name` must have one intercept and " *
        "$predictor_count slope coordinates; got $(length(declaration.axis_keys))"))
    nothing
end

function _validate_scale_parameter(name::Symbol, declaration::Parameter)
    declaration.support isa PositiveSupport || throw(CapabilityError(
        :parameter_support,
        "Normal scale parameter `$name` must use PositiveSupport"))
    declaration.transform isa ExpTransform || throw(CapabilityError(
        :parameter_transform,
        "Normal scale parameter `$name` must use Exp()"))
    declaration.prior isa ExponentialPrior || throw(CapabilityError(
        :parameter_prior,
        "Normal scale parameter `$name` must use Exponential(scale)"))
    length(declaration.axis_keys) == 1 || throw(CapabilityError(
        :parameter_axis,
        "Normal scale parameter `$name` must have one scalar coordinate"))
    only(declaration.axis_keys) === name || throw(CapabilityError(
        :parameter_axis,
        "Normal scale parameter `$name` must use its parameter identity as its axis key"))
    nothing
end

function _bind_scalar_location(
    declaration::Model, bindings::NamedTuple, conditions::NamedTuple)
    isempty(bindings) || throw(CapabilityError(
        :value_ports,
        "scalar-only active composition cannot retain ordinary value bindings"))
    isempty(declaration.nodes) || throw(CapabilityError(
        :additional_nodes,
        "the current scalar active compiler accepts no deterministic nodes"))
    length(declaration.observations) == 1 || throw(CapabilityError(
        :outcomes,
        "the scalar active compiler requires exactly one observation declaration"))
    response_name, lifted_observation =
        only(collect(pairs(declaration.observations)))
    is_broadcast_observation(lifted_observation) || throw(CapabilityError(
        :broadcast_lifting,
        "the scalar active compiler requires explicit dotted sampling"))
    observation = scalar_observation(lifted_observation)
    observation isa NormalObservation || throw(CapabilityError(
        :likelihood,
        "the first scalar active compiler slice supports Normal observations"))
    observation_response(observation) === response_name || throw(
        CapabilityError(
            :graph_identity,
            "scalar active stochastic-site identity must match its response"))
    hasproperty(conditions, response_name) || throw(CapabilityError(
        :observation_axis,
        "a scalar-only unconditioned model has no observation-axis source; " *
        "condition `$response_name` before compiling it"))
    response = _binding(conditions, response_name, :conditioned_response)
    eltype(response) <: Real || throw(CapabilityError(
        :response_type,
        "conditioned site `$response_name` must be a real vector"))
    observation_count = length(response)
    observation_count > 0 || throw(CapabilityError(
        :observation_axis, "the observation axis cannot be empty"))

    location_name, scale_name = observation_dependencies(observation)
    location_name === scale_name && throw(CapabilityError(
        :graph_identity,
        "Normal location and scale must be distinct graph values"))
    length(declaration.parameters) == 2 || throw(CapabilityError(
        :additional_parameters,
        "scalar Normal composition requires one location and one scale parameter"))
    hasproperty(declaration.parameters, location_name) || throw(CapabilityError(
        :parameter_binding,
        "scalar Normal location references missing parameter `$location_name`"))
    hasproperty(declaration.parameters, scale_name) || throw(CapabilityError(
        :parameter_binding,
        "scalar Normal scale references missing parameter `$scale_name`"))
    location_declaration = getproperty(declaration.parameters, location_name)
    scale_declaration = getproperty(declaration.parameters, scale_name)
    _validate_coefficient_parameter(location_name, location_declaration, 0)
    only(location_declaration.axis_keys) === location_name || throw(
        CapabilityError(
            :parameter_axis,
            "scalar location parameter `$location_name` must use its graph " *
            "identity as its axis key"))
    _validate_scale_parameter(scale_name, scale_declaration)

    observation_axis = BRM.NativePPLAxis(
        :observation, Base.OneTo(observation_count))
    coefficient_axis = BRM.NativePPLAxis(
        Symbol(location_name, :_scalar), location_declaration.axis_keys)
    scale_axis = BRM.NativePPLAxis(
        Symbol(scale_name, :_scalar), scale_declaration.axis_keys)
    response_input = BRM.NativePPLInput(
        response_name, :response, observation_axis, eltype(response))
    coefficients = BRM.NativePPLParameter(
        location_name, location_declaration.support,
        location_declaration.transform, coefficient_axis, 1:1)
    scale = BRM.NativePPLParameter(
        scale_name, scale_declaration.support, scale_declaration.transform,
        scale_axis, 2:2)
    location = BRM.NativePPLScalarBroadcastNode(
        location_name, location_name, observation_axis, 1)
    coefficient_prior = if location_declaration.prior isa StandardNormal
        BRM.NativePPLStandardNormalFactor(location_name, 1:1)
    else
        prior = location_declaration.prior
        BRM.NativePPLScalarNormalFactor(
            location_name, Symbol(location_name, :_prior_location),
            Symbol(location_name, :_prior_scale), 1,
            prior.location, prior.scale)
    end
    scale_prior = BRM.NativePPLExponentialFactor(
        scale_name, 2, scale_declaration.prior.scale)
    likelihood = BRM.NativePPLNormalFactor(
        response_name, location_name, scale_name, observation_axis)
    _validated_plan(BRM.NativePPLPlan(
        (; observation=observation_axis, coefficient=coefficient_axis,
           scale=scale_axis),
        (; predictors=(;), response=response_input),
        (; coefficients, scale),
        (; transforms=(;), location),
        (; coefficient_prior, scale_prior, likelihood),
        BRM._native_ppl_queries(observation_axis, likelihood),
        NamedTuple{(response_name,)}((response,))))
end

function _scalar_constant_binding(
    bindings::NamedTuple, name::Symbol, role::AbstractString,
)
    hasproperty(bindings, name) || throw(ArgumentError(
        "native PPL binding is missing $role `$name`"))
    value = getproperty(bindings, name)
    value isa Real && isfinite(value) || throw(ArgumentError(
        "native PPL $role `$name` must be one finite real scalar; got $value"))
    value
end

function _bind_scalar_site_schedule(
    declaration::Model, bindings::NamedTuple, conditions::NamedTuple,
)
    isempty(declaration.nodes) || throw(CapabilityError(
        :additional_nodes,
        "the first scalar-site schedule accepts no deterministic nodes"))
    length(declaration.observations) == 2 || throw(CapabilityError(
        :factor_schedule,
        "the first scalar-site schedule requires one generating factor and " *
        "one conditioned likelihood"))
    unconditioned = Tuple(name for name in keys(declaration.observations)
                          if !hasproperty(conditions, name))
    conditioned = Tuple(name for name in keys(declaration.observations)
                        if hasproperty(conditions, name))
    length(unconditioned) == 1 && length(conditioned) == 1 || throw(
        CapabilityError(
            :factor_schedule,
            "the first scalar-site schedule requires exactly one unconditioned " *
            "site followed by one conditioned site"))

    site_name = only(unconditioned)
    response_name = only(conditioned)
    generating = getproperty(declaration.observations, site_name)
    likelihood_declaration = getproperty(
        declaration.observations, response_name)
    is_broadcast_observation(generating) && throw(CapabilityError(
        :factor_schedule,
        "the first exported stochastic-site schedule requires a scalar site"))
    is_broadcast_observation(likelihood_declaration) || throw(CapabilityError(
        :broadcast_lifting,
        "the downstream likelihood must use explicit dotted sampling"))
    generating = scalar_observation(generating)
    likelihood = scalar_observation(likelihood_declaration)
    generating isa NormalObservation || throw(CapabilityError(
        :likelihood,
        "the first exported stochastic-site schedule supports a Normal " *
        "generating factor"))
    likelihood isa NormalObservation || throw(CapabilityError(
        :likelihood,
        "the first exported stochastic-site schedule supports a Normal " *
        "downstream likelihood"))
    observation_response(generating) === site_name || throw(CapabilityError(
        :graph_identity,
        "scalar generating-factor identity must match stochastic site `$site_name`"))
    observation_response(likelihood) === response_name || throw(
        CapabilityError(
            :graph_identity,
            "downstream likelihood identity must match response `$response_name`"))

    prior_location_name, prior_scale_name = observation_dependencies(generating)
    Set(keys(declaration.inputs)) == Set((prior_location_name, prior_scale_name)) ||
        throw(CapabilityError(
            :factor_schedule,
            "the first scalar-site generating factor requires exactly two " *
            "constant value ports"))
    Set(keys(bindings)) == Set((prior_location_name, prior_scale_name)) || throw(
        CapabilityError(
            :factor_schedule,
            "the scalar-site generating-factor ports must both be substituted"))
    prior_location = _scalar_constant_binding(
        bindings, prior_location_name, "Normal location")
    prior_scale = _scalar_constant_binding(
        bindings, prior_scale_name, "Normal scale")
    prior_scale > zero(prior_scale) || throw(ArgumentError(
        "native PPL Normal scale `$prior_scale_name` must be positive"))

    downstream_location, downstream_scale = observation_dependencies(likelihood)
    downstream_location === site_name || throw(CapabilityError(
        :factor_schedule,
        "downstream likelihood must consume exported scalar site `$site_name`"))
    length(declaration.parameters) == 1 || throw(CapabilityError(
        :additional_parameters,
        "the first scalar-site schedule requires one downstream scale parameter"))
    hasproperty(declaration.parameters, downstream_scale) || throw(
        CapabilityError(
            :parameter_binding,
            "downstream Normal scale references missing parameter " *
            "`$downstream_scale`"))
    scale_declaration = getproperty(declaration.parameters, downstream_scale)
    _validate_scale_parameter(downstream_scale, scale_declaration)

    response = _binding(conditions, response_name, :conditioned_response)
    eltype(response) <: Real || throw(CapabilityError(
        :response_type,
        "conditioned site `$response_name` must be a real vector"))
    observation_count = length(response)
    observation_count > 0 || throw(CapabilityError(
        :observation_axis, "the observation axis cannot be empty"))

    observation_axis = BRM.NativePPLAxis(
        :observation, Base.OneTo(observation_count))
    site_axis = BRM.NativePPLAxis(
        Symbol(site_name, :_scalar), (site_name,))
    scale_axis = BRM.NativePPLAxis(
        Symbol(downstream_scale, :_scalar), scale_declaration.axis_keys)
    response_input = BRM.NativePPLInput(
        response_name, :response, observation_axis, eltype(response))
    site_parameter = BRM.NativePPLParameter(
        site_name, BRM.NativePPLRealSupport(),
        BRM.NativePPLIdentityTransform(), site_axis, 1:1)
    scale_parameter = BRM.NativePPLParameter(
        downstream_scale, scale_declaration.support,
        scale_declaration.transform, scale_axis, 2:2)
    location = BRM.NativePPLScalarBroadcastNode(
        site_name, site_name, observation_axis, 1)
    site_prior = BRM.NativePPLScalarNormalFactor(
        site_name, prior_location_name, prior_scale_name, 1,
        prior_location, prior_scale)
    scale_prior = BRM.NativePPLExponentialFactor(
        downstream_scale, 2, scale_declaration.prior.scale)
    likelihood_factor = BRM.NativePPLNormalFactor(
        response_name, site_name, downstream_scale, observation_axis)
    parameters = (; site=site_parameter, scale=scale_parameter)
    _validated_plan(BRM.NativePPLPlan(
        (; observation=observation_axis, coefficient=site_axis,
           scale=scale_axis),
        (; predictors=(;), response=response_input),
        parameters,
        (; transforms=(;), location),
        (; site_prior, scale_prior, likelihood=likelihood_factor),
        BRM._native_ppl_queries(observation_axis, likelihood_factor),
        NamedTuple{(response_name,)}((response,))))
end

"""
    bind(model::Model, bindings; conditions=(;)) -> Plan

Fit data-derived nodes and compile the current direct declaration subset into
the same typed executable `Plan` used by BRM. The compiler accepts either one
or more continuous predictors feeding an affine location, or an actively
connected scalar parameter broadcast over a conditioned observation axis. The
affine path supports optional fitted center/zscale nodes, an optional
exponential rate link, and one Normal/BernoulliLogit/Poisson stochastic site;
the first scalar-active path supports Normal. The executor requires explicit
broadcast lifting. Unsupported graph shapes fail closed.
"""
function _bind(declaration::Model, bindings, conditions)
    _validate_model(declaration)
    _validate_binding_names(declaration, bindings)
    _validate_condition_names(declaration, conditions)
    length(declaration.observations) > 1 && length(conditions) == 1 &&
        return _bind_scalar_site_schedule(declaration, bindings, conditions)
    isempty(declaration.inputs) &&
        return _bind_scalar_location(declaration, bindings, conditions)
    predictor_names = Tuple(keys(declaration.inputs))
    predictors = map(predictor_names) do predictor_name
        predictor_declaration = getproperty(declaration.inputs, predictor_name)
        input_role(predictor_declaration) in (:value, :predictor) ||
            throw(CapabilityError(
                :value_ports,
                "affine input port `$predictor_name` must be generic or carry " *
                "legacy predictor provenance"))
        predictor = _binding(bindings, predictor_name, :predictor)
        eltype(predictor) <: Real && !(eltype(predictor) <: Integer) ||
            throw(CapabilityError(
                :predictor_type,
                "predictor `$predictor_name` must be a continuous real vector"))
        predictor
    end
    observation_count = length(first(predictors))
    observation_count > 0 || throw(CapabilityError(
        :observation_axis, "the observation axis cannot be empty"))
    for (predictor_name, predictor) in zip(predictor_names, predictors)
        length(predictor) == observation_count || throw(CapabilityError(
            :observation_axis,
            "value port `$predictor_name` has $(length(predictor)) rows; " *
            "expected $observation_count"))
    end
    predictor_bindings = NamedTuple{predictor_names}(predictors)

    length(declaration.observations) == 1 || throw(CapabilityError(
        :outcomes,
        "the current native compiler requires exactly one observation declaration"))
    observation_name, observation = only(collect(pairs(declaration.observations)))
    is_broadcast_observation(observation) || throw(CapabilityError(
        :broadcast_lifting,
        "the current vector executor requires explicit dotted sampling, for " *
        "example `@. $observation_name ~ Normal(location, scale)`"))
    observation = scalar_observation(observation)
    response_name = observation_response(observation)
    observation_name === response_name || throw(CapabilityError(
        :graph_identity,
        "stochastic-site identity `$observation_name` must match `$response_name`"))
    response = hasproperty(conditions, response_name) ?
        _binding(conditions, response_name, :conditioned_response) : nothing
    if response !== nothing && !(eltype(response) <: Real)
        throw(CapabilityError(
            :response_type,
            "conditioned site `$response_name` must be a real or Bool vector"))
    end
    response !== nothing && observation_count != length(response) &&
        throw(CapabilityError(
            :observation_axis,
            "predictor ports have $observation_count rows but " *
            "conditioned site `$response_name` has $(length(response))"))

    affine_name, affine_declaration = _one_declaration(
        declaration.nodes, Affine, "affine node")
    coefficient_name = affine_parameter(affine_declaration)
    hasproperty(declaration.parameters, coefficient_name) ||
        throw(CapabilityError(
            :parameter_binding,
            "affine node `$affine_name` references missing coefficient parameter " *
            "`$coefficient_name`"))
    coefficient_declaration = getproperty(
        declaration.parameters, coefficient_name)
    affine_inputs = node_inputs(affine_declaration)
    _validate_coefficient_parameter(
        coefficient_name, coefficient_declaration, length(affine_inputs))

    transform_names = Symbol[]
    transform_values = Any[]
    affine_raw_inputs = Symbol[]
    for affine_input in affine_inputs
        if hasproperty(declaration.inputs, affine_input)
            push!(affine_raw_inputs, affine_input)
            continue
        end
        hasproperty(declaration.nodes, affine_input) || throw(CapabilityError(
            :graph_identity,
            "affine node `$affine_name` references unknown input `$affine_input`"))
        transform_declaration = getproperty(declaration.nodes, affine_input)
        transform_declaration isa Union{Center,ZScale} ||
            throw(CapabilityError(
                :predictor_transform,
                "affine input `$affine_input` must be a fitted center/zscale node"))
        raw_input = node_input(transform_declaration)
        hasproperty(declaration.inputs, raw_input) || throw(CapabilityError(
            :graph_identity,
            "fitted node `$affine_input` references unknown value port `$raw_input`"))
        push!(affine_raw_inputs, raw_input)
        push!(transform_names, affine_input)
        push!(transform_values, transform_declaration)
    end
    length(unique(affine_raw_inputs)) == length(affine_raw_inputs) ||
        throw(CapabilityError(
            :graph_identity,
            "each affine feature must consume a distinct raw value port"))
    Set(affine_raw_inputs) == Set(predictor_names) || throw(CapabilityError(
        :additional_inputs,
        "the current affine compiler requires every value port exactly once; " *
        "ports=$(predictor_names), affine sources=$(Tuple(affine_raw_inputs))"))

    observation_axis = BRM.NativePPLAxis(
        :observation, Base.OneTo(observation_count))
    coefficient_axis = BRM.NativePPLAxis(
        Symbol(affine_name, :_coefficient),
        coefficient_declaration.axis_keys)
    predictor_inputs = NamedTuple{predictor_names}(map(
        (predictor_name, predictor) -> BRM.NativePPLInput(
            predictor_name, :predictor, observation_axis, eltype(predictor)),
        predictor_names, predictors))
    response_eltype = response === nothing ?
        (observation isa NormalObservation ? eltype(first(predictors)) :
         observation isa BernoulliLogitObservation ? Bool : Int) :
        eltype(response)
    response_input = BRM.NativePPLInput(
        response_name, :response, observation_axis, response_eltype)
    coefficients = BRM.NativePPLParameter(
        coefficient_name,
        coefficient_declaration.support,
        coefficient_declaration.transform,
        coefficient_axis,
        1:(length(affine_inputs) + 1))
    compiled_transform_values = map(
        (name, declaration) -> begin
            raw_input = node_input(declaration)
            _compile_transform(
                declaration, name, raw_input, observation_axis,
                getproperty(predictor_bindings, raw_input))
        end,
        transform_names, transform_values)
    compiled_transforms = NamedTuple{Tuple(transform_names)}(
        Tuple(compiled_transform_values))
    slope_indices = Tuple(2:(length(affine_inputs) + 1))
    location = BRM.NativePPLAffineNode(
        affine_name, affine_inputs, observation_axis, 1, slope_indices)
    compiled_nodes = (; transforms=compiled_transforms, location)
    coefficient_prior = BRM.NativePPLStandardNormalFactor(
        coefficient_name, 1:(length(affine_inputs) + 1))
    compiled_bindings = response === nothing ?
        predictor_bindings : merge(
            predictor_bindings,
            NamedTuple{(response_name,)}((response,)))

    if observation isa NormalObservation
        observation_dependencies(observation)[1] === affine_name ||
            throw(CapabilityError(
                :graph_identity,
                "Normal observation must consume affine node `$affine_name` as location"))
        scale_name = observation_dependencies(observation)[2]
        hasproperty(declaration.parameters, scale_name) ||
            throw(CapabilityError(
                :parameter_binding,
                "Normal observation references missing scale parameter `$scale_name`"))
        length(declaration.parameters) == 2 || throw(CapabilityError(
            :additional_parameters,
            "Normal declaration currently requires exactly coefficient and scale parameters"))
        scale_declaration = getproperty(declaration.parameters, scale_name)
        _validate_scale_parameter(scale_name, scale_declaration)
        length(declaration.nodes) == length(compiled_transforms) + 1 ||
            throw(CapabilityError(
                :additional_nodes,
                "Normal declaration contains unsupported extra nodes"))
        scale_axis = BRM.NativePPLAxis(
            Symbol(scale_name, :_scalar), scale_declaration.axis_keys)
        scale = BRM.NativePPLParameter(
            scale_name, scale_declaration.support, scale_declaration.transform,
            scale_axis,
            (length(affine_inputs) + 2):(length(affine_inputs) + 2))
        scale_prior = BRM.NativePPLExponentialFactor(
            scale_name, length(affine_inputs) + 2,
            scale_declaration.prior.scale)
        likelihood = BRM.NativePPLNormalFactor(
            response_name, affine_name, scale_name, observation_axis)
        return _validated_plan(BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis,
               scale=scale_axis),
            (; predictors=predictor_inputs, response=response_input),
            (; coefficients, scale),
            compiled_nodes,
            (; coefficient_prior, scale_prior, likelihood),
            BRM._native_ppl_queries(observation_axis, likelihood),
            compiled_bindings))
    end

    length(declaration.parameters) == 1 || throw(CapabilityError(
        :additional_parameters,
        "BernoulliLogit/Poisson declarations currently accept only coefficients"))
    if observation isa BernoulliLogitObservation
        only(observation_dependencies(observation)) === affine_name ||
            throw(CapabilityError(
                :graph_identity,
                "BernoulliLogit observation must consume affine node `$affine_name`"))
        if response !== nothing
            all(value -> value == 0 || value == 1, response) ||
                throw(CapabilityError(
                    :response_support,
                    "BernoulliLogit condition `$response_name` must contain Bool/0/1"))
        end
        length(declaration.nodes) == length(compiled_transforms) + 1 ||
            throw(CapabilityError(
                :additional_nodes,
                "BernoulliLogit declaration contains unsupported extra nodes"))
        likelihood = BRM.NativePPLBernoulliLogitFactor(
            response_name, affine_name, observation_axis)
        return _validated_plan(BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis),
            (; predictors=predictor_inputs, response=response_input),
            (; coefficients), compiled_nodes,
            (; coefficient_prior, likelihood),
            BRM._native_ppl_queries(observation_axis, likelihood),
            compiled_bindings))
    end

    observation isa PoissonObservation || throw(CapabilityError(
        :likelihood,
        "unsupported observation declaration $(typeof(observation))"))
    if response !== nothing
        all(BRM._native_ppl_is_count, response) || throw(CapabilityError(
            :response_support,
            "Poisson condition `$response_name` must contain nonnegative " *
            "integer-valued counts"))
    end
    rate_name = only(observation_dependencies(observation))
    hasproperty(declaration.nodes, rate_name) || throw(CapabilityError(
        :graph_identity,
        "Poisson observation references missing rate node `$rate_name`"))
    rate_declaration = getproperty(declaration.nodes, rate_name)
    rate_declaration isa ExpLink || throw(CapabilityError(
        :likelihood_link,
        "Poisson rate node `$rate_name` must be exp_link(...)"))
    node_input(rate_declaration) === affine_name || throw(CapabilityError(
        :graph_identity,
        "Poisson rate node `$rate_name` must consume affine node `$affine_name`"))
    length(declaration.nodes) == length(compiled_transforms) + 2 ||
        throw(CapabilityError(
            :additional_nodes,
            "Poisson declaration contains unsupported extra nodes"))
    rate = BRM.NativePPLExpNode(
        rate_name, affine_name, observation_axis)
    compiled_nodes = merge(compiled_nodes, (; rate))
    likelihood = BRM.NativePPLPoissonFactor(
        response_name, rate_name, observation_axis)
    _validated_plan(BRM.NativePPLPlan(
        (; observation=observation_axis, coefficient=coefficient_axis),
        (; predictors=predictor_inputs, response=response_input),
        (; coefficients), compiled_nodes,
        (; coefficient_prior, likelihood),
        BRM._native_ppl_queries(observation_axis, likelihood),
        compiled_bindings))
end

bind(declaration::Model, bindings; conditions=(;)) =
    _bind(declaration, bindings, conditions)
compile(declaration::Model, bindings; conditions=(;)) =
    bind(declaration, bindings; conditions)
prepare(declaration::Model, bindings; conditions=(;), kwargs...) =
    prepare(bind(declaration, bindings; conditions); kwargs...)
bind(instance::ModelInstance) =
    _bind(instance.declaration, instance.bindings, instance.conditions)
compile(instance::ModelInstance) = bind(instance)
prepare(instance::ModelInstance; kwargs...) =
    prepare(bind(instance); kwargs...)

function _lower_brmi(brmi::BRM.BRMI)
    observed = BRM.outcomes(brmi)
    length(observed) == 1 || throw(CapabilityError(
        :outcomes,
        "expected exactly one observed response, got $(length(observed))"))
    outcome = only(observed)
    family = outcome.family
    family === BRM.Normal || family === BRM.BernoulliLogit ||
        family === BRM.Poisson || throw(CapabilityError(
            :likelihood,
            "expected `Normal(location, scale)`, `BernoulliLogit(logit)`, " *
            "or `Poisson(exp(log_rate))`, got `$family`"))

    response = outcome.response
    response_lhs, likelihood = BRM._native_ppl_sampling_rhs(brmi, response)
    response_lhs isa BRM.NamedColumn &&
        parent(response_lhs) isa BRM.DataColumn || throw(CapabilityError(
            :response_decorator,
            "response `$response` must be a bare observed data column"))
    likelihood isa BRM.ExprColumn && BRM.getf(likelihood) === family ||
        throw(CapabilityError(
            :likelihood,
            "response `$response` must use `$family` consistently"))
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "$family likelihood for `$response` has keywords"))
    likelihood_args = BRM.getargs(likelihood)
    expected_arguments = family === BRM.Normal ? 2 : 1
    length(likelihood_args) == expected_arguments || throw(CapabilityError(
        :likelihood,
        "$family likelihood for `$response` needs $expected_arguments argument(s)"))

    rate_name = nothing
    location = if family === BRM.Poisson
        rate_expression = only(likelihood_args)
        rate_expression isa BRM.ExprColumn &&
            BRM.getf(rate_expression) === exp || throw(CapabilityError(
                :likelihood_link,
                "Poisson response `$response` must use `Poisson(exp(log_rate))`"))
        isempty(BRM.getkwargs(rate_expression)) || throw(CapabilityError(
            :likelihood_link,
            "Poisson `exp` link cannot have keywords"))
        rate_arguments = BRM.getargs(rate_expression)
        length(rate_arguments) == 1 || throw(CapabilityError(
            :likelihood_link,
            "Poisson `exp` link needs one named predictor"))
        log_rate = BRM._native_ppl_ref_name(only(rate_arguments))
        log_rate === nothing && throw(CapabilityError(
            :likelihood_location,
            "Poisson `exp` link must consume one named linear predictor"))
        rate_name = Symbol(:exp_, log_rate)
        log_rate
    else
        BRM._native_ppl_ref_name(likelihood_args[1])
    end
    location === nothing && throw(CapabilityError(
        :likelihood_location,
        "$family predictor must be one named linear predictor"))
    scale_name = family === BRM.Normal ?
        BRM._native_ppl_ref_name(likelihood_args[2]) : nothing
    family === BRM.Normal && scale_name === nothing &&
        throw(CapabilityError(
            :likelihood_scale,
            "Normal scale must be one named scalar parameter"))

    predictor_terms = BRM._native_ppl_affine_predictors(brmi, location)
    predictor_columns = map(term -> term.column, predictor_terms)
    predictor_names = map(BRM.name, predictor_columns)
    response in predictor_names && throw(CapabilityError(
        :input_roles,
        "predictor `$response` is also the observed response"))
    predictors = map(column -> parent(parent(column)), predictor_columns)
    response_values = parent(parent(response_lhs))

    scalar_location = isempty(predictor_names)
    intercept_prior = scalar_location ?
        BRM._native_ppl_intercept_normal_prior(brmi, location) :
        (; location=0.0, scale=1.0, operation=nothing)

    prior_scale = family === BRM.Normal ?
        BRM._native_ppl_exponential_prior(brmi, scale_name) : nothing
    expected_values = family === BRM.Normal ?
        Set((location, scale_name, response, predictor_names...)) :
        Set((location, response, predictor_names...))
    intercept_prior.operation === nothing ||
        push!(expected_values, intercept_prior.operation)
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported formula operations: " *
        join(sort!(collect(extras)), ", ")))

    input_declarations = _declaration_namedtuple(
        predictor_names, map(_ -> input(), predictor_names))
    scalar_location && family !== BRM.Normal && throw(CapabilityError(
        :likelihood,
        "the first intercept-only BRM native-PPL slice supports Normal"))
    coefficient_name = scalar_location ? location : Symbol(:beta_, location)
    coefficient_declaration = parameter(
        RealSupport(),
        scalar_location ? (location,) : (:Intercept, predictor_names...);
        transform=Identity(),
        prior=intercept_prior.operation === nothing ? StandardNormal() :
            normal_prior(intercept_prior.location, intercept_prior.scale))
    parameter_declarations = if family === BRM.Normal
        scale_declaration = parameter(
            PositiveSupport(), (scale_name,);
            transform=Exp(), prior=Exponential(prior_scale))
        NamedTuple{(coefficient_name, scale_name)}(
            (coefficient_declaration, scale_declaration))
    else
        NamedTuple{(coefficient_name,)}((coefficient_declaration,))
    end

    transform_names = Symbol[]
    transform_declarations = Any[]
    affine_inputs = map(predictor_terms, predictor_names) do predictor_term,
                                                          predictor_name
        predictor_term.transform === :identity && return predictor_name
        canonical = predictor_term.transform
        transform_name = Symbol(canonical, :_, predictor_name, :_for_, location)
        transform_declaration = canonical === :center ?
            center(predictor_name) : zscale(predictor_name)
        push!(transform_names, transform_name)
        push!(transform_declarations, transform_declaration)
        transform_name
    end
    node_names = scalar_location ? () : (Tuple(transform_names)..., location)
    node_values = scalar_location ? () : (
        Tuple(transform_declarations)...,
        affine(Tuple(affine_inputs), coefficient_name))
    node_declarations = _declaration_namedtuple(node_names, node_values)

    observation_declaration = if family === BRM.Normal
        normal(response, location, scale_name)
    elseif family === BRM.BernoulliLogit
        bernoulli_logit(response, location)
    else
        node_declarations = merge(
            node_declarations,
            NamedTuple{(rate_name,)}((exp_link(location),)))
        poisson(response, rate_name)
    end
    observation_declarations = NamedTuple{(response,)}(
        (broadcasted(observation_declaration),))
    declaration = model(
        inputs=input_declarations,
        parameters=parameter_declarations,
        nodes=node_declarations,
        observations=observation_declarations)
    bindings = _declaration_namedtuple(predictor_names, predictors)
    conditions = NamedTuple{(response,)}((response_values,))
    (; declaration, bindings, conditions)
end

"""Lower a supported BRM formula to the public unbound native-PPL declaration."""
lower(brmi::BRM.BRMI) = _lower_brmi(brmi).declaration

function compile(brmi::BRM.BRMI)
    lowered = _lower_brmi(brmi)
    bind(lowered.declaration, lowered.bindings;
         conditions=lowered.conditions)
end

_declaration_namedtuple(names::Tuple, values::Tuple) =
    NamedTuple{names}(values)

function _syntax_name(expression)
    expression isa Symbol && return expression
    expression isa GlobalRef && return expression.name
    if expression isa Expr && expression.head === :. &&
       expression.args[end] isa QuoteNode
        return expression.args[end].value
    end
    nothing
end

function _syntax_sampling_statement(statement)
    if statement isa Expr && statement.head === :macrocall &&
       _syntax_name(first(statement.args)) in (Symbol("@."), Symbol("@__dot__"))
        expanded = macroexpand(@__MODULE__, statement)
        expanded isa Expr && expanded.head === :. &&
            _syntax_name(first(expanded.args)) === :~ ||
            throw(ArgumentError(
                "native PPL @model `@.` must wrap one sampling statement"))
        arguments = expanded.args[2]
        arguments isa Expr && arguments.head === :tuple &&
            length(arguments.args) == 2 || throw(ArgumentError(
                "native PPL @model cannot decode dotted sampling `$statement`"))
        return (; lhs=arguments.args[1], rhs=arguments.args[2],
                broadcasted=true)
    end
    if statement isa Expr && statement.head === :call &&
       first(statement.args) in (:~, :.~)
        return (; lhs=statement.args[2], rhs=statement.args[3],
                broadcasted=first(statement.args) === :.~)
    end
    nothing
end

function _syntax_call(expression, context::AbstractString)
    expression isa Expr && expression.head === :call || throw(ArgumentError(
        "native PPL @model $context must be a call; got `$expression`"))
    name = _syntax_name(first(expression.args))
    name === nothing && throw(ArgumentError(
        "native PPL @model cannot identify the function in `$expression`"))
    name, expression.args[2:end]
end

function _syntax_distribution_call(expression, context::AbstractString)
    if expression isa Expr && expression.head === :. &&
       length(expression.args) == 2 && expression.args[2] isa Expr &&
       expression.args[2].head === :tuple
        name = _syntax_name(first(expression.args))
        name === nothing && throw(ArgumentError(
            "native PPL @model cannot identify the distribution in `$expression`"))
        return name, expression.args[2].args
    end
    _syntax_call(expression, context)
end

function _syntax_argument_name(argument)
    argument isa Symbol && return argument
    if argument isa Expr && argument.head === :(::) &&
       first(argument.args) isa Symbol
        return first(argument.args)
    end
    throw(ArgumentError(
        "native PPL @model currently requires plain or typed positional " *
        "arguments; got `$argument`"))
end

_syntax_ref(name::Symbol) = GlobalRef(@__MODULE__, name)

function _syntax_namedtuple(names::Vector{Symbol}, values::Vector)
    Expr(:call, _syntax_ref(:_declaration_namedtuple),
         QuoteNode(Tuple(names)), Expr(:tuple, values...))
end

function _syntax_parameter(statement, argument_names::Set{Symbol})
    lhs, rhs = statement.args[2], statement.args[3]
    prior_name, prior_arguments = _syntax_call(rhs, "parameter prior")
    parameter_name = lhs isa Symbol ? lhs :
        (lhs isa Expr && lhs.head === :ref && first(lhs.args) isa Symbol ?
         first(lhs.args) : nothing)
    parameter_name === nothing && throw(ArgumentError(
        "native PPL @model parameter left-hand side must be a name or " *
        "indexed name; got `$lhs`"))
    parameter_name in argument_names && throw(ArgumentError(
        "native PPL @model input `$parameter_name` cannot also be a parameter"))

    if prior_name === :StandardNormal
        isempty(prior_arguments) || throw(ArgumentError(
            "native PPL @model StandardNormal() takes no arguments"))
        lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 ||
            throw(ArgumentError(
                "native PPL @model StandardNormal parameters require axis " *
                "keys, for example `beta[(:Intercept, :x)]`"))
        axis_keys = lhs.args[2]
        value = Expr(
            :call, _syntax_ref(:parameter),
            Expr(:parameters,
                 Expr(:kw, :transform, Expr(:call, _syntax_ref(:Identity))),
                 Expr(:kw, :prior, Expr(:call, _syntax_ref(:StandardNormal)))),
            Expr(:call, _syntax_ref(:RealSupport)), axis_keys)
        return parameter_name, value
    elseif prior_name === :Exponential
        lhs isa Symbol || throw(ArgumentError(
            "native PPL @model Exponential parameter must have a bare name"))
        length(prior_arguments) == 1 || throw(ArgumentError(
            "native PPL @model Exponential(scale) needs one scale"))
        value = Expr(
            :call, _syntax_ref(:parameter),
            Expr(:parameters,
                 Expr(:kw, :transform, Expr(:call, _syntax_ref(:Exp))),
                 Expr(:kw, :prior,
                      Expr(:call, _syntax_ref(:Exponential),
                           only(prior_arguments)))),
            Expr(:call, _syntax_ref(:PositiveSupport)),
            QuoteNode((parameter_name,)))
        return parameter_name, value
    end
    throw(ArgumentError(
        "native PPL @model unsupported parameter prior `$prior_name`"))
end

function _syntax_node(statement)
    lhs, rhs = statement.args
    lhs isa Symbol || throw(ArgumentError(
        "native PPL @model deterministic node needs a bare name; got `$lhs`"))
    function_name, arguments = _syntax_call(rhs, "deterministic assignment")
    all(argument -> argument isa Symbol, arguments) || throw(ArgumentError(
        "native PPL @model deterministic node `$lhs` currently accepts only " *
        "named dependencies"))
    builder = if function_name === :center
        length(arguments) == 1 || throw(ArgumentError(
            "native PPL @model center needs one input"))
        :center
    elseif function_name === :zscale || function_name === :standardize
        length(arguments) == 1 || throw(ArgumentError(
            "native PPL @model $function_name needs one input"))
        function_name
    elseif function_name === :affine
        length(arguments) == 2 || throw(ArgumentError(
            "native PPL @model affine needs input and coefficient parameter"))
        :affine
    elseif function_name === :exp || function_name === :exp_link
        length(arguments) == 1 || throw(ArgumentError(
            "native PPL @model exp needs one input"))
        :exp_link
    else
        throw(ArgumentError(
            "native PPL @model unsupported deterministic function " *
            "`$function_name`"))
    end
    value = Expr(:call, _syntax_ref(builder),
                 (QuoteNode(argument) for argument in arguments)...)
    lhs, value
end

function _syntax_standard_normal(statement)
    lhs, rhs = statement.args[2], statement.args[3]
    lhs isa Symbol || return nothing
    prior_name, prior_arguments = _syntax_call(rhs, "parameter prior")
    prior_name === :Normal || return nothing
    valid = isempty(prior_arguments) ||
        (length(prior_arguments) == 2 && prior_arguments[1] == 0 &&
         prior_arguments[2] == 1)
    valid ? lhs : nothing
end

function _syntax_scalar_standard_normal_parameter(name::Symbol)
    Expr(
        :call, _syntax_ref(:parameter),
        Expr(:parameters,
             Expr(:kw, :transform, Expr(:call, _syntax_ref(:Identity))),
             Expr(:kw, :prior, Expr(:call, _syntax_ref(:StandardNormal)))),
        Expr(:call, _syntax_ref(:RealSupport)), QuoteNode((name,)))
end

function _syntax_affine_terms(expression)
    expression isa Expr && expression.head === :call &&
        first(expression.args) in (:+, :.+) || return nothing
    terms = Any[]
    function append_terms!(term)
        if term isa Expr && term.head === :call &&
           first(term.args) in (:+, :.+)
            for argument in term.args[2:end]
                append_terms!(argument)
            end
        else
            push!(terms, term)
        end
    end
    append_terms!(expression)
    terms
end

function _syntax_affine_assignment(statement,
                                   scalar_priors::Set{Symbol})
    lhs, rhs = statement.args
    lhs isa Symbol || return nothing
    terms = _syntax_affine_terms(rhs)
    terms === nothing && return nothing
    length(terms) >= 2 || return nothing
    intercepts = [term for term in terms
                  if term isa Symbol && term in scalar_priors]
    length(intercepts) == 1 || return nothing
    intercept = only(intercepts)

    slopes = Symbol[]
    predictor_names = Symbol[]
    raw_predictor_names = Symbol[]
    transform_names = Symbol[]
    transform_values = Any[]
    for product in terms
        product === intercept && continue
        product isa Expr && product.head === :call &&
            first(product.args) in (:*, :.*) && length(product.args) == 3 ||
            return nothing
        product_terms = product.args[2:end]
        coefficient_indices = findall(
            term -> term isa Symbol && term in scalar_priors &&
                    term !== intercept,
            product_terms)
        length(coefficient_indices) == 1 || return nothing
        coefficient_index = only(coefficient_indices)
        slope = product_terms[coefficient_index]
        predictor_expression = product_terms[3 - coefficient_index]
        raw_predictor_name = nothing
        predictor_name = if predictor_expression isa Symbol
            raw_predictor_name = predictor_expression
            predictor_expression
        elseif predictor_expression isa Expr &&
               predictor_expression.head === :call
            function_name, arguments = _syntax_call(
                predictor_expression, "affine predictor transform")
            function_name in (:center, :zscale, :standardize) || return nothing
            length(arguments) == 1 && only(arguments) isa Symbol ||
                throw(ArgumentError(
                    "native PPL @model $function_name requires one named input"))
            raw_input = only(arguments)
            raw_predictor_name = raw_input
            canonical = function_name === :center ? :center : :zscale
            transform_name = Symbol(canonical, :_, raw_input, :_for_, lhs)
            push!(transform_names, transform_name)
            push!(transform_values, Expr(
                :call, _syntax_ref(canonical), QuoteNode(raw_input)))
            transform_name
        else
            return nothing
        end
        push!(slopes, slope)
        push!(predictor_names, predictor_name)
        push!(raw_predictor_names, raw_predictor_name)
    end
    isempty(slopes) && return nothing
    length(unique(slopes)) == length(slopes) || throw(ArgumentError(
        "native PPL @model affine coefficients must be used once each"))
    length(unique(predictor_names)) == length(predictor_names) ||
        throw(ArgumentError(
            "native PPL @model affine predictor paths must be unique"))
    length(unique(raw_predictor_names)) == length(raw_predictor_names) ||
        throw(ArgumentError(
            "native PPL @model affine features must use distinct raw inputs"))

    coefficient_name = Symbol(:beta_, lhs)
    parameter_value = Expr(
        :call, _syntax_ref(:parameter),
        Expr(:parameters,
             Expr(:kw, :transform, Expr(:call, _syntax_ref(:Identity))),
             Expr(:kw, :prior, Expr(:call, _syntax_ref(:StandardNormal)))),
        Expr(:call, _syntax_ref(:RealSupport)),
        QuoteNode((intercept, slopes...)))
    affine_value = Expr(
        :call, _syntax_ref(:affine), QuoteNode(Tuple(predictor_names)),
        QuoteNode(coefficient_name))
    (; location=lhs, intercept, slopes=Tuple(slopes), coefficient_name,
       parameter_value, transform_names=Tuple(transform_names),
       transform_values=Tuple(transform_values), affine_value)
end

function _syntax_observation(lhs, rhs; broadcasted::Bool)
    lhs isa Symbol || throw(ArgumentError(
        "native PPL @model stochastic-site left-hand side must be a bare " *
        "name; got `$lhs`"))
    family, arguments = _syntax_distribution_call(rhs, "observation family")
    extra_node_name = nothing
    extra_node_value = nothing
    if family === :Poisson && length(arguments) == 1 &&
       only(arguments) isa Expr && only(arguments).head in (:call, :.)
        link_name, link_arguments = _syntax_distribution_call(
            only(arguments), "Poisson rate link")
        link_name === :exp && length(link_arguments) == 1 &&
            only(link_arguments) isa Symbol || throw(ArgumentError(
                "native PPL @model Poisson currently supports a named rate or " *
                "`exp(named_log_rate)`"))
        log_rate = only(link_arguments)
        extra_node_name = Symbol(:exp_, log_rate)
        extra_node_value = Expr(
            :call, _syntax_ref(:exp_link), QuoteNode(log_rate))
        arguments = Any[extra_node_name]
    end
    all(argument -> argument isa Symbol, arguments) || throw(ArgumentError(
        "native PPL @model observation `$lhs` currently accepts named " *
        "distribution parameters"))
    builder, expected = if family === :Normal
        (:normal, 2)
    elseif family === :BernoulliLogit
        (:bernoulli_logit, 1)
    elseif family === :Poisson
        (:poisson, 1)
    else
        throw(ArgumentError(
            "native PPL @model unsupported observation family `$family`"))
    end
    length(arguments) == expected || throw(ArgumentError(
        "native PPL @model $family observation needs $expected parameter(s)"))
    scalar_value = Expr(:call, _syntax_ref(builder), QuoteNode(lhs),
                        (QuoteNode(argument) for argument in arguments)...)
    value = broadcasted ?
        Expr(:call, _syntax_ref(:broadcasted), scalar_value) : scalar_value
    (; name=lhs, value, extra_node_name, extra_node_value,
       dependencies=Tuple(arguments))
end

function _syntax_outputs(statement)
    statement isa Expr && statement.head === :return || return nothing
    length(statement.args) == 1 || throw(ArgumentError(
        "native PPL @model return requires one named value or named tuple"))
    value = only(statement.args)
    value isa Symbol && return ((value,), (value,))
    value isa Expr && value.head === :tuple || throw(ArgumentError(
        "native PPL @model return must be a named value or named tuple"))

    entries = value.args
    if length(entries) == 1 && entries[1] isa Expr &&
       entries[1].head === :parameters
        entries = entries[1].args
        isempty(entries) && throw(ArgumentError(
            "native PPL @model return requires at least one named graph value"))
        all(entry -> entry isa Symbol, entries) || throw(ArgumentError(
            "native PPL @model named return shorthand requires bare names"))
        names = Tuple(entries)
        return names, names
    end

    aliases = Symbol[]
    names = Symbol[]
    isempty(entries) && throw(ArgumentError(
        "native PPL @model return requires at least one named graph value"))
    for entry in entries
        entry isa Expr && entry.head in (:(=), :kw) &&
            length(entry.args) == 2 && entry.args[1] isa Symbol &&
            entry.args[2] isa Symbol || throw(ArgumentError(
                "native PPL @model tuple returns must use `alias=name` entries"))
        push!(aliases, entry.args[1])
        push!(names, entry.args[2])
    end
    length(unique(aliases)) == length(aliases) || throw(ArgumentError(
        "native PPL @model return aliases must be unique"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "native PPL @model returned graph values must be distinct"))
    Tuple(aliases), Tuple(names)
end

const _SYNTAX_DISTRIBUTION_NAMES =
    (:Normal, :StandardNormal, :Exponential, :BernoulliLogit, :Poisson)
const _SYNTAX_DETERMINISTIC_NAMES =
    (:center, :zscale, :standardize, :affine, :exp, :exp_link)
const _SYNTAX_OPERATOR_NAMES =
    (:+, :.+, :-, :.-, :*, :.*, :/, :./, :^, :.^)

function _syntax_submodel_connection(statement)
    sampling = _syntax_sampling_statement(statement)
    if sampling !== nothing
        family = _syntax_name(
            sampling.rhs isa Expr ? first(sampling.rhs.args) : sampling.rhs)
        family in _SYNTAX_DISTRIBUTION_NAMES && return nothing
        sampling.broadcasted && throw(ArgumentError(
            "native PPL stochastic submodel calls use scalar `lhs ~ child(...)`; " *
            "broadcast belongs inside the child model"))
        sampling.lhs isa Symbol || throw(ArgumentError(
            "native PPL stochastic submodel call needs a bare result name"))
        callee, arguments = _syntax_call(
            sampling.rhs, "stochastic submodel call")
        return (; name=sampling.lhs, callee, arguments,
                call=sampling.rhs, connection=:stochastic)
    end

    statement isa Expr && statement.head === :(=) || return nothing
    lhs, rhs = statement.args
    lhs isa Symbol || return nothing
    rhs isa Expr && rhs.head === :call || return nothing
    callee = _syntax_name(first(rhs.args))
    callee === nothing && return nothing
    callee in _SYNTAX_DETERMINISTIC_NAMES && return nothing
    callee in _SYNTAX_OPERATOR_NAMES && return nothing
    _, arguments = _syntax_call(rhs, "deterministic submodel call")
    (; name=lhs, callee, arguments, call=rhs, connection=:deterministic)
end

function _syntax_composed_model(
    signature, function_name::Symbol, argument_names::Vector{Symbol},
    connections, explicit_outputs)
    explicit_outputs === nothing && throw(ArgumentError(
        "native PPL model composition requires an explicit returned graph value"))
    available = Set(argument_names)
    result_names = Symbol[]
    component_names = Symbol[]
    generated = Any[]
    for connection in connections
        first(connection.call.args) === function_name && throw(ArgumentError(
            "native PPL model `$function_name` cannot recursively stage itself"))
        connection.name in available && throw(ArgumentError(
            "native PPL staged result `$(connection.name)` is already defined"))
        for argument in connection.arguments
            argument isa Expr && throw(ArgumentError(
                "native PPL staged call `$(connection.name)` currently accepts " *
                "named graph values or literal arguments; got `$argument`"))
            argument isa Symbol && argument ∉ available && throw(ArgumentError(
                "native PPL staged call `$(connection.name)` references " *
                "unavailable value `$argument`"))
        end
        component_name = gensym(Symbol(connection.name, :_component))
        helper = connection.connection === :stochastic ?
            :_staged_stochastic_component : :_staged_deterministic_component
        push!(generated, Expr(
            :(=), Expr(:tuple, component_name, connection.name),
            Expr(:call, _syntax_ref(helper), QuoteNode(connection.name),
                 connection.call)))
        push!(component_names, component_name)
        push!(result_names, connection.name)
        push!(available, connection.name)
    end
    output_aliases, output_names = explicit_outputs
    result_name_set = Set(result_names)
    all(name -> name in result_name_set, output_names) || throw(ArgumentError(
        "native PPL composed model may return only staged child outputs; got " *
        "$(output_names)"))
    public_outputs = _syntax_namedtuple(
        collect(output_aliases), Any[output_names...])
    push!(generated, Expr(
        :return,
        Expr(:call, _syntax_ref(:_staged_composition),
             Expr(:tuple, component_names...), public_outputs)))
    Expr(:function, signature, Expr(:block, generated...))
end

function _model_function_syntax(definition)
    definition isa Expr && definition.head === :function ||
        throw(ArgumentError(
            "NativePPL.@model must wrap a function definition"))
    signature, body = definition.args
    signature isa Expr && signature.head === :call || throw(ArgumentError(
        "NativePPL.@model requires a named function definition"))
    function_name = first(signature.args)
    function_name isa Symbol || throw(ArgumentError(
        "NativePPL.@model currently requires an unqualified function name"))
    arguments = signature.args[2:end]
    argument_names = Symbol[_syntax_argument_name(argument)
                            for argument in arguments]
    length(unique(argument_names)) == length(argument_names) ||
        throw(ArgumentError(
            "NativePPL.@model function arguments must have unique names"))
    argument_name_set = Set(argument_names)

    parameter_names = Symbol[]
    parameter_values = Any[]
    scalar_prior_names = Symbol[]
    consumed_scalar_priors = Set{Symbol}()
    node_names = Symbol[]
    node_values = Any[]
    observation_names = Symbol[]
    observation_values = Any[]
    factor_dependency_names = Set{Symbol}()
    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    statements = Any[statement for statement in statements
                     if !(statement isa LineNumberNode)]
    return_indices = findall(
        statement -> statement isa Expr && statement.head === :return,
        statements)
    length(return_indices) <= 1 || throw(ArgumentError(
        "native PPL @model supports one explicit return statement"))
    explicit_outputs = nothing
    if !isempty(return_indices)
        only(return_indices) == length(statements) || throw(ArgumentError(
            "native PPL @model return must be the final statement"))
        explicit_outputs = _syntax_outputs(pop!(statements))
    end
    connections = [_syntax_submodel_connection(statement)
                   for statement in statements]
    if any(connection -> connection !== nothing, connections)
        all(connection -> connection !== nothing, connections) || throw(
            ArgumentError(
                "native PPL cannot yet mix staged submodel connections with " *
                "local parameters, nodes, or factors in one model"))
        return _syntax_composed_model(
            signature, function_name, argument_names, connections,
            explicit_outputs)
    end
    for statement in statements
        sampling = _syntax_sampling_statement(statement)
        if sampling !== nothing
            sampling.lhs isa Symbol && sampling.lhs in argument_name_set &&
                throw(ArgumentError(
                    "native PPL @model stochastic site `$(sampling.lhs)` " *
                    "cannot also be a function argument; condition it after " *
                    "constructing the model"))
            normalized = Expr(:call, :~, sampling.lhs, sampling.rhs)
            scalar_prior = sampling.broadcasted ? nothing :
                _syntax_standard_normal(normalized)
            prior_name = _syntax_name(
                sampling.rhs isa Expr ? first(sampling.rhs.args) : sampling.rhs)
            is_explicit_parameter = !sampling.broadcasted &&
                prior_name in (:StandardNormal, :Exponential)
            if scalar_prior === nothing && !is_explicit_parameter
                observation = _syntax_observation(
                    sampling.lhs, sampling.rhs;
                    broadcasted=sampling.broadcasted)
                if observation.extra_node_name !== nothing
                    push!(node_names, observation.extra_node_name)
                    push!(node_values, observation.extra_node_value)
                end
                push!(observation_names, observation.name)
                push!(observation_values, observation.value)
                union!(factor_dependency_names, observation.dependencies)
            else
                if is_explicit_parameter
                    name, value = _syntax_parameter(
                        normalized, argument_name_set)
                    push!(parameter_names, name)
                    push!(parameter_values, value)
                else
                    scalar_prior in scalar_prior_names && throw(ArgumentError(
                        "native PPL @model parameter `$scalar_prior` is declared twice"))
                    push!(scalar_prior_names, scalar_prior)
                end
            end
        elseif statement isa Expr && statement.head === :(=)
            affine_declaration = _syntax_affine_assignment(
                statement, Set(scalar_prior_names))
            if affine_declaration === nothing
                name, value = _syntax_node(statement)
                push!(node_names, name)
                push!(node_values, value)
            else
                # Coefficient blocks precede support-transformed scalar
                # parameters in the flat coordinate ABI, independent of where
                # the deterministic affine assignment appears in source.
                pushfirst!(
                    parameter_names, affine_declaration.coefficient_name)
                pushfirst!(
                    parameter_values, affine_declaration.parameter_value)
                for (transform_name, transform_value) in zip(
                    affine_declaration.transform_names,
                    affine_declaration.transform_values)
                    push!(node_names, transform_name)
                    push!(node_values, transform_value)
                end
                push!(node_names, affine_declaration.location)
                push!(node_values, affine_declaration.affine_value)
                push!(consumed_scalar_priors, affine_declaration.intercept)
                union!(consumed_scalar_priors, affine_declaration.slopes)
            end
        else
            throw(ArgumentError(
                "native PPL @model unsupported statement `$statement`"))
        end
    end
    returned_names = explicit_outputs === nothing ? Set{Symbol}() :
        Set(last(explicit_outputs))
    union!(returned_names, factor_dependency_names)
    standalone_scalar_priors = setdiff(
        intersect(Set(scalar_prior_names), returned_names),
        consumed_scalar_priors)
    if !isempty(standalone_scalar_priors)
        isempty(consumed_scalar_priors) || throw(ArgumentError(
            "native PPL @model cannot yet mix explicitly returned scalar " *
            "priors with packed affine coefficients in one component"))
        for name in scalar_prior_names
            name in standalone_scalar_priors || continue
            push!(parameter_names, name)
            push!(parameter_values,
                  _syntax_scalar_standard_normal_parameter(name))
            push!(consumed_scalar_priors, name)
        end
    end
    unused_scalar_priors = setdiff(
        Set(scalar_prior_names), consumed_scalar_priors)
    isempty(unused_scalar_priors) || throw(ArgumentError(
        "native PPL @model standard-normal parameters are not used by a " *
        "supported affine expression: " *
        join(sort!(collect(unused_scalar_priors)), ", ")))
    length(unique(parameter_names)) == length(parameter_names) ||
        throw(ArgumentError(
            "native PPL @model generated duplicate parameter identities"))
    length(unique(node_names)) == length(node_names) || throw(ArgumentError(
        "native PPL @model generated duplicate node identities"))
    explicit_outputs === nothing && isempty(observation_names) &&
        throw(ArgumentError(
            "native PPL @model without an explicit return requires at least " *
            "one stochastic site"))
    input_values = Any[
        Expr(:call, _syntax_ref(:input))
        for name in argument_names
    ]
    model_keywords = Any[
        Expr(:kw, :inputs,
             _syntax_namedtuple(argument_names, input_values)),
        Expr(:kw, :parameters,
             _syntax_namedtuple(parameter_names, parameter_values)),
        Expr(:kw, :nodes,
             _syntax_namedtuple(node_names, node_values)),
        Expr(:kw, :observations,
             _syntax_namedtuple(observation_names, observation_values)),
    ]
    if explicit_outputs !== nothing
        output_aliases, output_names = explicit_outputs
        push!(model_keywords, Expr(
            :kw, :outputs,
            _syntax_namedtuple(
                collect(output_aliases),
                Any[QuoteNode(name) for name in output_names])))
    end
    declaration = Expr(
        :call, _syntax_ref(:model),
        Expr(:parameters, model_keywords...))
    bindings = _syntax_namedtuple(argument_names, Any[argument_names...])
    instance = Expr(:call, _syntax_ref(:instantiate), declaration, bindings)
    Expr(:function, signature, Expr(:block, Expr(:return, instance)))
end

"""
    NativePPL.@model function name(inputs...)
        ...
    end

Define a staged probabilistic model function. Calling it returns a
`ModelInstance`; it does not execute density or RNG work. The current language
slice accepts generic composable positional value ports, scalar Normal and
standard-normal
coefficient sites, positive Exponential-prior scalar sites, fitted
center/zscale nodes, affine and exp deterministic nodes, and staged scalar or
explicitly broadcast Normal/BernoulliLogit/Poisson stochastic sites. A final
`return value` or named-tuple return declares component outputs and permits a
deterministic zero-observation model. The current vector executor requires the
explicit broadcast form (`@.` or dotted `~`). Composition-only outer models
use `site ~ stochastic_child(...)` for stochastic child outputs and
`value = deterministic_child(...)` for deterministic child outputs; the macro
generates the explicit namespaced graph machinery.
"""
macro model(definition)
    esc(_model_function_syntax(definition))
end

export Model, ModelInstance, GraphRef, Component, Composition, Input, Parameter
export RealSupport, PositiveSupport, IdentityTransform, ExpTransform
export StandardNormal, NormalPrior, ExponentialPrior
export Center, ZScale, Affine, ExpLink
export NormalObservation, BernoulliLogitObservation, PoissonObservation
export BroadcastObservation
export model, input, parameter, Identity, Exp, normal_prior, Exponential
export center, zscale, standardize, affine, exp_link
export normal, bernoulli_logit, poisson, broadcasted
export instantiate, substitute, condition, component, output, compose, bind, lower
export graph_namespace, graph_name, graph_kind, component_namespace, qualified_name
export @model
