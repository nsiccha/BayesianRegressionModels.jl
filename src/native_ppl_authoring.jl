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

"""A named model input role. The surrounding `NamedTuple` key is its identity."""
struct Input{Role} <: AbstractInputDeclaration end

function input(role::Symbol)
    role in (:predictor, :response, :data) || throw(ArgumentError(
        "native PPL input role must be :predictor, :response, or :data; got $role"))
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

"""Row-wise Normal observation declaration."""
struct NormalObservation{Response,Location,Scale} <:
       AbstractObservationDeclaration end
normal(response::Symbol, location::Symbol, scale::Symbol) =
    NormalObservation{response,location,scale}()

"""Row-wise Bernoulli observation declaration parameterized by logits."""
struct BernoulliLogitObservation{Response,Logit} <:
       AbstractObservationDeclaration end
bernoulli_logit(response::Symbol, logit::Symbol) =
    BernoulliLogitObservation{response,logit}()

"""Row-wise Poisson observation declaration parameterized by a positive rate."""
struct PoissonObservation{Response,Rate} <:
       AbstractObservationDeclaration end
poisson(response::Symbol, rate::Symbol) =
    PoissonObservation{response,rate}()

observation_response(::NormalObservation{Response}) where {Response} = Response
observation_response(::BernoulliLogitObservation{Response}) where {Response} =
    Response
observation_response(::PoissonObservation{Response}) where {Response} = Response
observation_dependencies(
    ::NormalObservation{Response,Location,Scale},
) where {Response,Location,Scale} = (Location, Scale)
observation_dependencies(
    ::BernoulliLogitObservation{Response,Logit},
) where {Response,Logit} = (Logit,)
observation_dependencies(
    ::PoissonObservation{Response,Rate},
) where {Response,Rate} = (Rate,)

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

function _check_named_declarations(values, kind::AbstractString, type)
    values isa NamedTuple || throw(ArgumentError(
        "native PPL $kind declarations must be a NamedTuple; got $(typeof(values))"))
    for (name, value) in pairs(values)
        value isa type || throw(ArgumentError(
            "native PPL $kind `$name` must be a $type; got $(typeof(value))"))
    end
    values
end

function model(; inputs, parameters=(;), nodes=(;), observations)
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

    response_names = Set(name for (name, declaration) in pairs(inputs)
                         if input_role(declaration) === :response)
    for (name, declaration) in pairs(observations)
        response = observation_response(declaration)
        response in response_names || throw(ArgumentError(
            "native PPL observation `$name` references response input `$response`, " *
            "which is not declared with role :response"))
        name === response || throw(ArgumentError(
            "native PPL observation key `$name` must match response identity `$response`"))
        for dependency in observation_dependencies(declaration)
            dependency in available || throw(ArgumentError(
                "native PPL observation `$name` references unavailable dependency " *
                "`$dependency`"))
        end
    end

    Model(inputs, parameters, nodes, observations)
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

function _validate_coefficient_parameter(name::Symbol, declaration::Parameter)
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
    first(declaration.axis_keys) === :Intercept || throw(CapabilityError(
        :parameter_axis,
        "affine coefficient parameter `$name` must name its first coordinate :Intercept"))
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
    nothing
end

"""
    bind(model::Model, bindings) -> Plan

Fit data-derived nodes and compile the current direct declaration subset into
the same typed executable `Plan` used by BRM. This initial compiler accepts one
continuous predictor, one affine node, an optional fitted center/zscale node,
an optional exponential rate link, and one Normal/BernoulliLogit/Poisson
observation. Unsupported graph shapes fail closed.
"""
function bind(declaration::Model, bindings)
    length(declaration.inputs) == 2 || throw(CapabilityError(
        :input_roles,
        "the current native compiler requires one predictor and one response input"))
    predictor_name, predictor_declaration = _one_declaration(
        declaration.inputs, Input{:predictor}, "predictor input")
    response_name, response_declaration = _one_declaration(
        declaration.inputs, Input{:response}, "response input")
    predictor = _binding(bindings, predictor_name, :predictor)
    response = _binding(bindings, response_name, :response)
    eltype(predictor) <: Real && !(eltype(predictor) <: Integer) ||
        throw(CapabilityError(
            :predictor_type,
            "predictor `$predictor_name` must be a continuous real vector"))
    eltype(response) <: Real || throw(CapabilityError(
        :response_type,
        "response `$response_name` must be a real or Bool vector"))
    length(predictor) == length(response) || throw(CapabilityError(
        :observation_axis,
        "predictor `$predictor_name` has $(length(predictor)) rows but " *
        "`$response_name` has $(length(response))"))
    isempty(response) && throw(CapabilityError(
        :observation_axis, "the observation axis cannot be empty"))

    length(declaration.observations) == 1 || throw(CapabilityError(
        :outcomes,
        "the current native compiler requires exactly one observation declaration"))
    observation_name, observation = only(collect(pairs(declaration.observations)))
    observation_name === response_name || throw(CapabilityError(
        :graph_identity,
        "observation identity `$observation_name` must match response `$response_name`"))

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
    _validate_coefficient_parameter(coefficient_name, coefficient_declaration)

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
        :observation, Base.OneTo(length(response)))
    coefficient_axis = BRM.NativePPLAxis(
        Symbol(affine_name, :_coefficient),
        coefficient_declaration.axis_keys)
    predictor_input = BRM.NativePPLInput(
        predictor_name, input_role(predictor_declaration),
        observation_axis, eltype(predictor))
    response_input = BRM.NativePPLInput(
        response_name, input_role(response_declaration),
        observation_axis, eltype(response))
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
    compiled_bindings = NamedTuple{(predictor_name, response_name)}(
        (predictor, response))

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
        scale_axis = BRM.NativePPLAxis(
            Symbol(scale_name, :_scalar), scale_declaration.axis_keys)
        scale = BRM.NativePPLParameter(
            scale_name, scale_declaration.support, scale_declaration.transform,
            scale_axis, 3:3)
        scale_prior = BRM.NativePPLExponentialFactor(
            scale_name, 3, scale_declaration.prior.scale)
        likelihood = BRM.NativePPLNormalFactor(
            response_name, affine_name, scale_name, observation_axis)
        return BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis,
               scale=scale_axis),
            (; predictor=predictor_input, response=response_input),
            (; coefficients, scale),
            compiled_nodes,
            (; coefficient_prior, scale_prior, likelihood),
            BRM._native_ppl_queries(observation_axis, likelihood),
            compiled_bindings)
    end

    length(declaration.parameters) == 1 || throw(CapabilityError(
        :additional_parameters,
        "BernoulliLogit/Poisson declarations currently accept only coefficients"))
    if observation isa BernoulliLogitObservation
        only(observation_dependencies(observation)) === affine_name ||
            throw(CapabilityError(
                :graph_identity,
                "BernoulliLogit observation must consume affine node `$affine_name`"))
        all(value -> value == 0 || value == 1, response) ||
            throw(CapabilityError(
                :response_support,
                "BernoulliLogit response `$response_name` must contain Bool/0/1"))
        length(declaration.nodes) == (transform === nothing ? 1 : 2) ||
            throw(CapabilityError(
                :additional_nodes,
                "BernoulliLogit declaration contains unsupported extra nodes"))
        likelihood = BRM.NativePPLBernoulliLogitFactor(
            response_name, affine_name, observation_axis)
        return BRM.NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis),
            (; predictor=predictor_input, response=response_input),
            (; coefficients), compiled_nodes,
            (; coefficient_prior, likelihood),
            BRM._native_ppl_queries(observation_axis, likelihood),
            compiled_bindings)
    end

    observation isa PoissonObservation || throw(CapabilityError(
        :likelihood,
        "unsupported observation declaration $(typeof(observation))"))
    all(BRM._native_ppl_is_count, response) || throw(CapabilityError(
        :response_support,
        "Poisson response `$response_name` must contain nonnegative integer-valued counts"))
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
    BRM.NativePPLPlan(
        (; observation=observation_axis, coefficient=coefficient_axis),
        (; predictor=predictor_input, response=response_input),
        (; coefficients), compiled_nodes,
        (; coefficient_prior, likelihood),
        BRM._native_ppl_queries(observation_axis, likelihood),
        compiled_bindings)
end

compile(declaration::Model, bindings) = bind(declaration, bindings)
prepare(declaration::Model, bindings; kwargs...) =
    prepare(bind(declaration, bindings); kwargs...)

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

    input_declarations = NamedTuple{(predictor_name, response)}(
        (input(:predictor), input(:response)))
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
        (observation_declaration,))
    declaration = model(
        inputs=input_declarations,
        parameters=parameter_declarations,
        nodes=node_declarations,
        observations=observation_declarations)
    bindings = NamedTuple{(predictor_name, response)}(
        (predictor, response_values))
    (; declaration, bindings)
end

"""Lower a supported BRM formula to the public unbound native-PPL declaration."""
lower(brmi::BRM.BRMI) = _lower_brmi(brmi).declaration

function compile(brmi::BRM.BRMI)
    lowered = _lower_brmi(brmi)
    bind(lowered.declaration, lowered.bindings)
end

export Model, Input, Parameter
export RealSupport, PositiveSupport, IdentityTransform, ExpTransform
export StandardNormal, ExponentialPrior
export Center, ZScale, Affine, ExpLink
export NormalObservation, BernoulliLogitObservation, PoissonObservation
export model, input, parameter, Identity, Exp, Exponential
export center, zscale, standardize, affine, exp_link
export normal, bernoulli_logit, poisson
export bind, lower
