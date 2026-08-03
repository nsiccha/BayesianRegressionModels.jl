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

"""Exponential prior declaration parameterized by its positive scale."""
struct ExponentialPrior{T<:Real} <: AbstractPriorDeclaration
    scale::T
end

function Exponential(scale::Real)
    isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
        "native PPL Exponential prior scale must be finite and positive"))
    ExponentialPrior(scale)
end

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

"""Intercept plus one slope over a named input/node and parameter block."""
struct Affine{Input,Coefficients} <: AbstractNodeDeclaration end
affine(input::Symbol, coefficients::Symbol) = Affine{input,coefficients}()

"""Elementwise exponential link over one named deterministic node."""
struct ExpLink{Input} <: AbstractNodeDeclaration end
exp_link(input::Symbol) = ExpLink{input}()

node_input(::Center{Input}) where {Input} = Input
node_input(::ZScale{Input}) where {Input} = Input
node_input(::Affine{Input}) where {Input} = Input
node_input(::ExpLink{Input}) where {Input} = Input
affine_parameter(::Affine{Input,Coefficients}) where {Input,Coefficients} =
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

Names are the keys of the four `NamedTuple`s. The model owns logical graph
semantics only; data binding, fitted transform state, axes, coordinates, output
signatures, and workspace layout belong to compilation/preparation.
"""
struct Model{I,P,N,O}
    inputs::I
    parameters::P
    nodes::N
    observations::O
end

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

ModelInstance(declaration::Model, bindings) =
    ModelInstance(declaration, bindings, (;))

function instantiate(declaration::Model, bindings; conditions=(;))
    _validate_model(declaration)
    _validate_binding_names(declaration, bindings)
    _validate_condition_names(declaration, conditions)
    ModelInstance(declaration, bindings, conditions)
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
    _validate_condition_names(declaration, conditions)
    ModelInstance(declaration, (;), conditions)
end

function condition(instance::ModelInstance, conditions)
    _validate_instance(instance)
    _validate_condition_names(instance.declaration, conditions)
    ModelInstance(
        instance.declaration, instance.bindings,
        merge(instance.conditions, conditions))
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

function _validate_model_components(inputs, parameters, nodes, observations)
    _check_named_declarations(
        inputs, "input", AbstractInputDeclaration)
    _check_named_declarations(
        parameters, "parameter", Parameter)
    _check_named_declarations(
        nodes, "node", AbstractNodeDeclaration)
    _check_named_declarations(
        observations, "observation", AbstractObservationDeclaration)
    isempty(observations) && throw(ArgumentError(
        "native PPL model requires at least one observation declaration"))

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
        node_input(declaration) in available || throw(ArgumentError(
            "native PPL node `$name` references unavailable input " *
            "`$(node_input(declaration))`; nodes must be topologically ordered"))
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
    end

    nothing
end

function model(; inputs, parameters=(;), nodes=(;), observations)
    _validate_model_components(inputs, parameters, nodes, observations)
    Model(inputs, parameters, nodes, observations)
end

function _validate_model(declaration::Model)
    _validate_model_components(
        declaration.inputs, declaration.parameters,
        declaration.nodes, declaration.observations)
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

function _validate_condition_names(declaration::Model, conditions)
    conditions isa NamedTuple || throw(ArgumentError(
        "native PPL conditions must be a NamedTuple; got $(typeof(conditions))"))
    extra_conditions = setdiff(
        Set(keys(conditions)), Set(keys(declaration.observations)))
    isempty(extra_conditions) || throw(ArgumentError(
        "native PPL conditions reference undeclared stochastic sites: " *
        join(sort!(collect(extra_conditions)), ", ")))
    conditions
end

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
          ", observations=", keys(declaration.observations), ")")
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
    name::Symbol, declaration::Parameter)
    declaration.support isa RealSupport || throw(CapabilityError(
        :parameter_support,
        "affine coefficient parameter `$name` must use RealSupport"))
    declaration.transform isa IdentityTransform || throw(CapabilityError(
        :parameter_transform,
        "affine coefficient parameter `$name` must use Identity()"))
    declaration.prior isa StandardNormal || throw(CapabilityError(
        :parameter_prior,
        "affine coefficient parameter `$name` must use StandardNormal()"))
    length(declaration.axis_keys) == 2 || throw(CapabilityError(
        :parameter_axis,
        "affine coefficient parameter `$name` must have intercept and slope coordinates"))
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

"""
    bind(model::Model, bindings; conditions=(;)) -> Plan

Fit data-derived nodes and compile the current direct declaration subset into
the same typed executable `Plan` used by BRM. This initial compiler accepts one
continuous predictor, one affine node, an optional fitted center/zscale node,
an optional exponential rate link, and one Normal/BernoulliLogit/Poisson
stochastic site. The executor subset requires explicit broadcast lifting;
conditions are optional so the same declaration can compile for generative
prediction. Unsupported graph shapes fail closed.
"""
function _bind(declaration::Model, bindings, conditions)
    _validate_model(declaration)
    _validate_binding_names(declaration, bindings)
    _validate_condition_names(declaration, conditions)
    length(declaration.inputs) == 1 || throw(CapabilityError(
        :value_ports,
        "the current native compiler requires exactly one connected value port"))
    predictor_name, predictor_declaration = only(collect(pairs(declaration.inputs)))
    input_role(predictor_declaration) in (:value, :predictor) ||
        throw(CapabilityError(
            :value_ports,
            "the affine input port `$predictor_name` must be generic or carry " *
            "legacy predictor provenance"))
    predictor = _binding(bindings, predictor_name, :predictor)
    eltype(predictor) <: Real && !(eltype(predictor) <: Integer) ||
        throw(CapabilityError(
            :predictor_type,
            "predictor `$predictor_name` must be a continuous real vector"))
    isempty(predictor) && throw(CapabilityError(
        :observation_axis, "the observation axis cannot be empty"))

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
    response !== nothing && length(predictor) != length(response) &&
        throw(CapabilityError(
            :observation_axis,
            "value port `$predictor_name` has $(length(predictor)) rows but " *
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
    _validate_coefficient_parameter(
        coefficient_name, coefficient_declaration)

    transform_pairs = [(name, value) for (name, value) in pairs(declaration.nodes)
                       if value isa Union{Center,ZScale}]
    length(transform_pairs) <= 1 || throw(CapabilityError(
        :predictor_transform,
        "the current native compiler accepts at most one fitted predictor transform"))
    transform_name, transform_declaration = isempty(transform_pairs) ?
        (nothing, nothing) : only(transform_pairs)
    expected_affine_input = transform_name === nothing ?
        predictor_name : transform_name
    node_input(affine_declaration) === expected_affine_input ||
        throw(CapabilityError(
            :graph_identity,
            "affine node `$affine_name` must consume `$expected_affine_input`"))

    observation_axis = BRM.NativePPLAxis(
        :observation, Base.OneTo(length(predictor)))
    coefficient_axis = BRM.NativePPLAxis(
        Symbol(affine_name, :_coefficient),
        coefficient_declaration.axis_keys)
    predictor_input = BRM.NativePPLInput(
        predictor_name, :predictor,
        observation_axis, eltype(predictor))
    response_eltype = response === nothing ?
        (observation isa NormalObservation ? eltype(predictor) :
         observation isa BernoulliLogitObservation ? Bool : Int) :
        eltype(response)
    response_input = BRM.NativePPLInput(
        response_name, :response, observation_axis, response_eltype)
    coefficients = BRM.NativePPLParameter(
        coefficient_name,
        coefficient_declaration.support,
        coefficient_declaration.transform,
        coefficient_axis,
        1:2)
    transform = transform_name === nothing ? nothing :
        _compile_transform(
            transform_declaration, transform_name,
            predictor_name, observation_axis, predictor)
    location = BRM.NativePPLAffineNode(
        affine_name, expected_affine_input, observation_axis, 1, 2)
    compiled_nodes = transform === nothing ?
        (; location) : (; transform, location)
    coefficient_prior = BRM.NativePPLStandardNormalFactor(
        coefficient_name, 1:2)
    compiled_bindings = response === nothing ?
        NamedTuple{(predictor_name,)}((predictor,)) :
        NamedTuple{(predictor_name, response_name)}((predictor, response))

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
        length(declaration.nodes) == (transform === nothing ? 1 : 2) ||
            throw(CapabilityError(
                :additional_nodes,
                "Normal declaration contains unsupported extra nodes"))
        scale_axis = BRM.NativePPLAxis(
            Symbol(scale_name, :_scalar), scale_declaration.axis_keys)
        scale = BRM.NativePPLParameter(
            scale_name, scale_declaration.support, scale_declaration.transform,
            scale_axis, 3:3)
        scale_prior = BRM.NativePPLExponentialFactor(
            scale_name, 3, scale_declaration.prior.scale)
        likelihood = BRM.NativePPLNormalFactor(
            response_name, affine_name, scale_name, observation_axis)
        return _validated_plan(BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis,
               scale=scale_axis),
            (; predictor=predictor_input, response=response_input),
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
        length(declaration.nodes) == (transform === nothing ? 1 : 2) ||
            throw(CapabilityError(
                :additional_nodes,
                "BernoulliLogit declaration contains unsupported extra nodes"))
        likelihood = BRM.NativePPLBernoulliLogitFactor(
            response_name, affine_name, observation_axis)
        return _validated_plan(BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis),
            (; predictor=predictor_input, response=response_input),
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
    length(declaration.nodes) == (transform === nothing ? 2 : 3) ||
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
        (; predictor=predictor_input, response=response_input),
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

    predictor_term = BRM._native_ppl_affine_predictor(brmi, location)
    predictor_column = predictor_term.column
    predictor_name = BRM.name(predictor_column)
    predictor_name === response && throw(CapabilityError(
        :input_roles,
        "predictor `$predictor_name` is also the observed response"))
    predictor = parent(parent(predictor_column))
    response_values = parent(parent(response_lhs))

    prior_scale = family === BRM.Normal ?
        BRM._native_ppl_exponential_prior(brmi, scale_name) : nothing
    expected = family === BRM.Normal ?
        Set((location, scale_name, response, predictor_name)) :
        Set((location, response, predictor_name))
    extras = setdiff(Set(keys(brmi.operations)), expected)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported formula operations: " *
        join(sort!(collect(extras)), ", ")))

    input_declarations = NamedTuple{(predictor_name,)}((input(),))
    coefficient_name = Symbol(:beta_, location)
    coefficient_declaration = parameter(
        RealSupport(), (:Intercept, predictor_name);
        transform=Identity(), prior=StandardNormal())
    parameter_declarations = if family === BRM.Normal
        scale_declaration = parameter(
            PositiveSupport(), (scale_name,);
            transform=Exp(), prior=Exponential(prior_scale))
        NamedTuple{(coefficient_name, scale_name)}(
            (coefficient_declaration, scale_declaration))
    else
        NamedTuple{(coefficient_name,)}((coefficient_declaration,))
    end

    transform_name = if predictor_term.transform === :center
        Symbol("#native_ppl_center#", location, "#", predictor_name)
    elseif predictor_term.transform === :zscale
        Symbol("#native_ppl_zscale#", location, "#", predictor_name)
    else
        nothing
    end
    transform_declaration = if predictor_term.transform === :center
        center(predictor_name)
    elseif predictor_term.transform === :zscale
        zscale(predictor_name)
    else
        nothing
    end
    affine_input = transform_name === nothing ? predictor_name : transform_name
    affine_declaration = affine(affine_input, coefficient_name)
    node_declarations = if transform_name === nothing
        NamedTuple{(location,)}((affine_declaration,))
    else
        NamedTuple{(transform_name, location)}(
            (transform_declaration, affine_declaration))
    end

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
    bindings = NamedTuple{(predictor_name,)}((predictor,))
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

function _syntax_affine_assignment(statement,
                                   scalar_priors::Set{Symbol})
    lhs, rhs = statement.args
    lhs isa Symbol || return nothing
    rhs isa Expr && rhs.head === :call &&
        first(rhs.args) in (:+, :.+) && length(rhs.args) == 3 || return nothing
    terms = rhs.args[2:end]
    intercept_index = findfirst(
        term -> term isa Symbol && term in scalar_priors, terms)
    intercept_index === nothing && return nothing
    intercept = terms[intercept_index]
    product = terms[3 - intercept_index]
    product isa Expr && product.head === :call &&
        first(product.args) in (:*, :.*) && length(product.args) == 3 ||
        return nothing
    product_terms = product.args[2:end]
    slope_index = findfirst(
        term -> term isa Symbol && term in scalar_priors && term !== intercept,
        product_terms)
    slope_index === nothing && return nothing
    slope = product_terms[slope_index]
    predictor_expression = product_terms[3 - slope_index]

    transform_name = nothing
    transform_value = nothing
    predictor_name = if predictor_expression isa Symbol
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
        canonical = function_name === :center ? :center : :zscale
        transform_name = Symbol(
            "#native_ppl_", canonical, "#", lhs, "#", raw_input)
        transform_value = Expr(
            :call, _syntax_ref(canonical), QuoteNode(raw_input))
        transform_name
    else
        return nothing
    end

    coefficient_name = Symbol(:beta_, lhs)
    parameter_value = Expr(
        :call, _syntax_ref(:parameter),
        Expr(:parameters,
             Expr(:kw, :transform, Expr(:call, _syntax_ref(:Identity))),
             Expr(:kw, :prior, Expr(:call, _syntax_ref(:StandardNormal)))),
        Expr(:call, _syntax_ref(:RealSupport)),
        QuoteNode((intercept, slope)))
    affine_value = Expr(
        :call, _syntax_ref(:affine), QuoteNode(predictor_name),
        QuoteNode(coefficient_name))
    (; location=lhs, intercept, slope, coefficient_name, parameter_value,
       transform_name, transform_value, affine_value)
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
    (; name=lhs, value, extra_node_name, extra_node_value)
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
    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    for statement in statements
        statement isa LineNumberNode && continue
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
                if affine_declaration.transform_name !== nothing
                    push!(node_names, affine_declaration.transform_name)
                    push!(node_values, affine_declaration.transform_value)
                end
                push!(node_names, affine_declaration.location)
                push!(node_values, affine_declaration.affine_value)
                push!(consumed_scalar_priors, affine_declaration.intercept)
                push!(consumed_scalar_priors, affine_declaration.slope)
            end
        else
            throw(ArgumentError(
                "native PPL @model unsupported statement `$statement`"))
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
    length(observation_names) == 1 || throw(ArgumentError(
        "native PPL @model currently requires exactly one observation"))
    input_values = Any[
        Expr(:call, _syntax_ref(:input))
        for name in argument_names
    ]
    declaration = Expr(
        :call, _syntax_ref(:model),
        Expr(:parameters,
             Expr(:kw, :inputs,
                  _syntax_namedtuple(argument_names, input_values)),
             Expr(:kw, :parameters,
                  _syntax_namedtuple(parameter_names, parameter_values)),
             Expr(:kw, :nodes,
                  _syntax_namedtuple(node_names, node_values)),
             Expr(:kw, :observations,
                  _syntax_namedtuple(observation_names, observation_values))))
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
slice accepts generic composable positional value ports, standard-normal
coefficient sites, positive Exponential-prior scalar sites, fitted
center/zscale nodes, affine and exp deterministic nodes, and one scalar or
explicitly broadcast Normal/BernoulliLogit/Poisson stochastic site. The current
vector executor requires the explicit broadcast form (`@.` or dotted `~`).
"""
macro model(definition)
    esc(_model_function_syntax(definition))
end

export Model, ModelInstance, Input, Parameter
export RealSupport, PositiveSupport, IdentityTransform, ExpTransform
export StandardNormal, ExponentialPrior
export Center, ZScale, Affine, ExpLink
export NormalObservation, BernoulliLogitObservation, PoissonObservation
export BroadcastObservation
export model, input, parameter, Identity, Exp, Exponential
export center, zscale, standardize, affine, exp_link
export normal, bernoulli_logit, poisson, broadcasted
export instantiate, substitute, condition, bind, lower
export @model
