"""A semantic axis whose name is part of its type and whose keys are explicit."""
struct NativePPLAxis{Name,K}
    keys::K
end

NativePPLAxis(name::Symbol, keys::K) where {K} = NativePPLAxis{name,K}(keys)
native_axis_name(::NativePPLAxis{Name}) where {Name} = Name
Base.length(axis::NativePPLAxis) = length(axis.keys)

"""An external value required by a native PPL plan."""
struct NativePPLInput{Name,Role,A,T}
    axis::A
end

NativePPLInput(name::Symbol, role::Symbol, axis::A, ::Type{T}) where {A,T} =
    NativePPLInput{name,role,A,T}(axis)
native_input_name(::NativePPLInput{Name}) where {Name} = Name
native_input_role(::NativePPLInput{Name,Role}) where {Name,Role} = Role
Base.eltype(::NativePPLInput{Name,Role,A,T}) where {Name,Role,A,T} = T

abstract type NativePPLSupport end
struct NativePPLRealSupport <: NativePPLSupport end
struct NativePPLPositiveSupport <: NativePPLSupport end

"""A parameter block and its coordinates in the flat unconstrained vector."""
struct NativePPLParameter{Name,S<:NativePPLSupport,A,R}
    support::S
    axis::A
    unconstrained::R
end

NativePPLParameter(name::Symbol, support::S, axis::A, unconstrained::R) where {S,A,R} =
    NativePPLParameter{name,S,A,R}(support, axis, unconstrained)
native_parameter_name(::NativePPLParameter{Name}) where {Name} = Name

abstract type NativePPLNode end
abstract type NativePPLFactor end

"""The first native deterministic node: an intercept plus one continuous slope."""
struct NativePPLAffineNode{Name,Input,A} <: NativePPLNode
    axis::A
    intercept_index::Int
    slope_index::Int
end

NativePPLAffineNode(name::Symbol, input::Symbol, axis::A,
                    intercept_index::Int, slope_index::Int) where {A} =
    NativePPLAffineNode{name,input,A}(axis, intercept_index, slope_index)
native_node_name(::NativePPLAffineNode{Name}) where {Name} = Name
native_affine_input(::NativePPLAffineNode{Name,Input}) where {Name,Input} = Input

"""Independent standard-normal prior over an unconstrained parameter range."""
struct NativePPLStandardNormalFactor{Parameter,R} <: NativePPLFactor
    unconstrained::R
end

NativePPLStandardNormalFactor(parameter::Symbol, unconstrained::R) where {R} =
    NativePPLStandardNormalFactor{parameter,R}(unconstrained)

"""Exponential prior on an exp-transformed scalar, including its Jacobian."""
struct NativePPLExponentialFactor{Parameter,T} <: NativePPLFactor
    unconstrained_index::Int
    scale::T
end

NativePPLExponentialFactor(parameter::Symbol, index::Int, scale::T) where {T} =
    NativePPLExponentialFactor{parameter,T}(index, scale)

"""Row-wise Normal observation factor."""
struct NativePPLNormalFactor{Response,Location,Scale,A} <: NativePPLFactor
    axis::A
end

NativePPLNormalFactor(response::Symbol, location::Symbol, scale::Symbol, axis::A) where {A} =
    NativePPLNormalFactor{response,location,scale,A}(axis)

"""
    NativePPLPlan

Typed, inspectable native-Julia execution plan. `bindings` are the initial
input values captured from the source `BRMI`; execution preparation copies
them into separately owned buffers, so the plan itself is never a workspace.

This is deliberately internal while the public native-PPL naming is unsettled.
The first lowering accepts one structural model shape and fails closed for all
other formula capabilities.
"""
struct NativePPLPlan{A,I,P,N,F,B}
    axes::A
    inputs::I
    parameters::P
    nodes::N
    factors::F
    bindings::B
end

"""Buffers required by one primal native-PPL evaluation."""
mutable struct NativePPLBuffers{T,V<:Vector{T}}
    location::V
    pointwise_loglikelihood::V
end

"""Inspectable allocation contract for a prepared native PPL plan."""
struct NativePPLWorkspaceSpec{A}
    observation_axis::A
    gradient_length::Int
end

"""Validated, numeric input binding for a native PPL plan."""
struct NativePPLPrepared{P,X,Y,S}
    plan::P
    predictor::X
    response::Y
    workspace_spec::S
end

Base.eltype(prepared::NativePPLPrepared) = eltype(prepared.response)
LogDensityProblems.dimension(prepared::NativePPLPrepared) =
    LogDensityProblems.dimension(prepared.plan)

"""
Reusable storage for density and reverse-mode gradient execution.

The returned gradient and pointwise buffers alias this workspace. Callers that
need to retain a result across another evaluation must copy it. A workspace is
single-task state; concurrent callers use separate workspaces over the same
immutable `NativePPLPrepared` value.
"""
struct NativePPLWorkspace{T,B,G}
    primal::B
    adjoint::B
    gradient::G
end

Base.eltype(::NativePPLWorkspace{T}) where {T} = T

function NativePPLWorkspace(prepared::NativePPLPrepared, ::Type{T}=eltype(prepared)) where {T<:AbstractFloat}
    n = length(prepared.workspace_spec.observation_axis)
    buffers() = NativePPLBuffers(zeros(T, n), zeros(T, n))
    NativePPLWorkspace{T,NativePPLBuffers{T,Vector{T}},Vector{T}}(
        buffers(), buffers(), zeros(T, prepared.workspace_spec.gradient_length))
end

LogDensityProblems.dimension(plan::NativePPLPlan) =
    sum(length(parameter.unconstrained) for parameter in plan.parameters)

function Base.show(io::IO, plan::NativePPLPlan)
    print(io, "NativePPLPlan(", LogDensityProblems.dimension(plan),
          " unconstrained parameters, ", length(plan.axes.observation),
          " observations)\n")
    print(io, "  inputs: ",
          join((string(native_input_name(input)) for input in plan.inputs), ", "), "\n")
    print(io, "  parameters: ",
          join((string(native_parameter_name(parameter)) for parameter in plan.parameters),
               ", "), "\n")
    print(io, "  nodes: ",
          join((string(native_node_name(node)) for node in plan.nodes), ", "), "\n")
    print(io, "  factors: ",
          join((string(nameof(typeof(factor))) for factor in plan.factors), ", "))
end

function Base.show(io::IO, prepared::NativePPLPrepared)
    print(io, "NativePPLPrepared(", length(prepared.response),
          " observations, eltype=", eltype(prepared), ")")
end

function Base.show(io::IO, workspace::NativePPLWorkspace)
    print(io, "NativePPLWorkspace(eltype=", eltype(workspace),
          ", location=", length(workspace.primal.location),
          ", pointwise_loglikelihood=",
          length(workspace.primal.pointwise_loglikelihood),
          ", gradient=", length(workspace.gradient), ")")
end

"""Fail-closed diagnostic for a formula capability not yet in the native PPL."""
struct NativePPLCapabilityError <: Exception
    capability::Symbol
    detail::String
end

function Base.showerror(io::IO, err::NativePPLCapabilityError)
    print(io, "native PPL does not support capability `", err.capability,
          "` in the walking skeleton: ", err.detail)
end

_native_ppl_ref_name(x::NamedColumn{<:Any,<:MissingColumn}) = name(x)
_native_ppl_ref_name(_) = nothing

function _native_ppl_sampling_rhs(brmi::BRMI, key::Symbol)
    haskey(brmi.operations, key) || throw(NativePPLCapabilityError(
        :missing_binding, "formula name `$key` has no defining operation"))
    named = brmi.operations[key]
    named isa NamedColumn || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is not a named formula operation"))
    operation = parent(named)
    operation isa ExprColumn || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is a data binding, not a sampling statement"))
    getf(operation) === (~) || throw(NativePPLCapabilityError(
        :operation_shape, "`$key` is not defined by `~`"))
    isempty(getkwargs(operation)) || throw(NativePPLCapabilityError(
        :sampling_keywords, "sampling statement `$key` has keywords"))
    getargs(operation, 2)
end

function _native_ppl_affine_predictor(brmi::BRMI, key::Symbol)
    lhs, predictor = _native_ppl_sampling_rhs(brmi, key)
    _native_ppl_ref_name(lhs) === key || throw(NativePPLCapabilityError(
        :linked_predictor, "`$key` must have a bare, unlinked left-hand side"))
    predictor isa ExprColumn && getf(predictor) === (+) ||
        throw(NativePPLCapabilityError(:predictor_terms,
            "`$key` must be exactly `1 + x` for one continuous data column"))
    isempty(getkwargs(predictor)) || throw(NativePPLCapabilityError(
        :predictor_keywords, "predictor `$key` has keywords"))
    terms = getargs(predictor)
    length(terms) == 2 || throw(NativePPLCapabilityError(:predictor_terms,
        "`$key` must have exactly an intercept and one continuous predictor"))

    intercepts = filter(term -> term isa Number && term == 1, terms)
    data_terms = filter(term -> term isa NamedColumn && parent(term) isa DataColumn, terms)
    length(intercepts) == 1 && length(data_terms) == 1 ||
        throw(NativePPLCapabilityError(:predictor_terms,
            "`$key` must be exactly `1 + x`; offsets, interactions, and groups are not lowered yet"))
    only(data_terms)
end

function _native_ppl_exponential_prior(brmi::BRMI, key::Symbol)
    lhs, prior = _native_ppl_sampling_rhs(brmi, key)
    _native_ppl_ref_name(lhs) === key || throw(NativePPLCapabilityError(
        :parameter_transform, "scale `$key` must have a bare left-hand side"))
    prior isa ExprColumn && getf(prior) === Exponential ||
        throw(NativePPLCapabilityError(:scale_prior,
            "scale `$key` must use `Exponential(scale)`"))
    isempty(getkwargs(prior)) || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` prior for `$key` cannot have keywords"))
    args = getargs(prior)
    length(args) == 1 && only(args) isa Real || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` prior for `$key` needs one numeric scale"))
    scale = only(args)
    isfinite(scale) && scale > 0 || throw(NativePPLCapabilityError(
        :scale_prior, "`Exponential` scale for `$key` must be finite and positive"))
    scale
end

"""
    _native_ppl_plan(brmi::BRMI) -> NativePPLPlan

Lower the initial workflow-complete model family:

```julia
y     ~ Normal(mu, sigma)
mu    ~ 1 + x
sigma ~ Exponential(scale)
```

Names may vary, but the structure may not. Unsupported structure raises a
`NativePPLCapabilityError` naming the missing capability.
"""
function _native_ppl_plan(brmi::BRMI)
    observed = outcomes(brmi)
    length(observed) == 1 || throw(NativePPLCapabilityError(
        :outcomes, "expected exactly one observed response, got $(length(observed))"))
    outcome = only(observed)
    outcome.family === Normal || throw(NativePPLCapabilityError(
        :likelihood, "expected `Normal(location, scale)`, got `$(outcome.family)`"))

    response = outcome.response
    response_lhs, likelihood = _native_ppl_sampling_rhs(brmi, response)
    response_lhs isa NamedColumn && parent(response_lhs) isa DataColumn ||
        throw(NativePPLCapabilityError(:response_decorator,
            "response `$response` must be a bare observed data column"))
    likelihood isa ExprColumn && getf(likelihood) === Normal ||
        throw(NativePPLCapabilityError(:likelihood,
            "response `$response` must use `Normal(location, scale)`"))
    isempty(getkwargs(likelihood)) || throw(NativePPLCapabilityError(
        :likelihood_keywords, "Normal likelihood for `$response` has keywords"))
    likelihood_args = getargs(likelihood)
    length(likelihood_args) == 2 || throw(NativePPLCapabilityError(
        :likelihood, "Normal likelihood for `$response` needs two arguments"))
    location = _native_ppl_ref_name(likelihood_args[1])
    scale_parameter = _native_ppl_ref_name(likelihood_args[2])
    location === nothing && throw(NativePPLCapabilityError(
        :likelihood_location, "Normal location must be one named linear predictor"))
    scale_parameter === nothing && throw(NativePPLCapabilityError(
        :likelihood_scale, "Normal scale must be one named scalar parameter"))

    predictor = _native_ppl_affine_predictor(brmi, location)
    predictor_name = name(predictor)
    x = parent(parent(predictor))
    y = parent(parent(response_lhs))
    x isa AbstractVector{<:Real} && !(eltype(x) <: Integer) ||
        throw(NativePPLCapabilityError(:predictor_type,
            "predictor `$predictor_name` must be a continuous real vector"))
    y isa AbstractVector{<:Real} || throw(NativePPLCapabilityError(
        :response_type, "response `$response` must be a real vector"))
    length(x) == length(y) || throw(NativePPLCapabilityError(
        :observation_axis,
        "predictor `$predictor_name` has $(length(x)) rows but `$response` has $(length(y))"))
    !isempty(y) || throw(NativePPLCapabilityError(
        :observation_axis, "the observation axis cannot be empty"))

    prior_scale = _native_ppl_exponential_prior(brmi, scale_parameter)
    expected = Set((location, scale_parameter, response, predictor_name))
    extras = setdiff(Set(keys(brmi.operations)), expected)
    isempty(extras) || throw(NativePPLCapabilityError(
        :additional_operations,
        "unsupported formula operations: $(join(sort!(collect(extras)), ", "))"))

    observation_axis = NativePPLAxis(:observation, Base.OneTo(length(y)))
    coefficient_axis = NativePPLAxis(Symbol(location, :_coefficient),
                                     (:Intercept, predictor_name))
    scale_axis = NativePPLAxis(Symbol(scale_parameter, :_scalar), (scale_parameter,))

    predictor_input = NativePPLInput(predictor_name, :predictor,
                                     observation_axis, eltype(x))
    response_input = NativePPLInput(response, :response,
                                    observation_axis, eltype(y))
    coefficient_name = Symbol(:beta_, location)
    coefficients = NativePPLParameter(coefficient_name, NativePPLRealSupport(),
                                      coefficient_axis, 1:2)
    scale = NativePPLParameter(scale_parameter, NativePPLPositiveSupport(),
                               scale_axis, 3:3)
    location_node = NativePPLAffineNode(location, predictor_name,
                                        observation_axis, 1, 2)

    coefficient_prior = NativePPLStandardNormalFactor(coefficient_name, 1:2)
    scale_prior = NativePPLExponentialFactor(scale_parameter, 3, prior_scale)
    likelihood_factor = NativePPLNormalFactor(response, location,
                                              scale_parameter, observation_axis)
    bindings = NamedTuple{(predictor_name, response)}((x, y))

    NativePPLPlan(
        (; observation=observation_axis, coefficient=coefficient_axis,
           scale=scale_axis),
        (; predictor=predictor_input, response=response_input),
        (; coefficients, scale),
        (; location=location_node),
        (; coefficient_prior, scale_prior, likelihood=likelihood_factor),
        bindings,
    )
end

function _native_ppl_copy_input(::Type{T}, input, role::Symbol, name::Symbol) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(input))
    for i in eachindex(input)
        value = input[i]
        value isa Real && isfinite(value) || throw(ArgumentError(
            "native PPL $role `$name` contains a non-finite or non-real value at row $i"))
        output[i] = value
    end
    output
end

"""
    _native_ppl_prepare(plan; T=Float64) -> NativePPLPrepared

Validate and copy the plan's initial input binding into numeric, executor-owned
storage. The returned value is shareable; mutable evaluation state is supplied
separately by `NativePPLWorkspace`.
"""
function _native_ppl_prepare(plan::NativePPLPlan; T::Type{<:AbstractFloat}=Float64)
    predictor_name = native_input_name(plan.inputs.predictor)
    response_name = native_input_name(plan.inputs.response)
    predictor = _native_ppl_copy_input(
        T, getproperty(plan.bindings, predictor_name), :predictor, predictor_name)
    response = _native_ppl_copy_input(
        T, getproperty(plan.bindings, response_name), :response, response_name)
    length(predictor) == length(response) || throw(DimensionMismatch(
        "native PPL predictor `$predictor_name` has $(length(predictor)) rows but " *
        "response `$response_name` has $(length(response))"))
    workspace_spec = NativePPLWorkspaceSpec(
        plan.axes.observation, LogDensityProblems.dimension(plan))
    NativePPLPrepared(plan, predictor, response, workspace_spec)
end

function _native_ppl_check_execution(workspace::NativePPLWorkspace,
                                     prepared::NativePPLPrepared,
                                     position::AbstractVector)
    dimension = LogDensityProblems.dimension(prepared)
    length(position) == dimension || throw(DimensionMismatch(
        "native PPL position has length $(length(position)); expected $dimension"))
    length(workspace.gradient) == dimension || throw(DimensionMismatch(
        "native PPL workspace gradient has length $(length(workspace.gradient)); expected $dimension"))
    length(workspace.primal.location) == length(prepared.response) ||
        throw(DimensionMismatch("native PPL workspace observation axis does not match prepared data"))
    eltype(position) === eltype(workspace) || throw(ArgumentError(
        "native PPL position eltype $(eltype(position)) does not match workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL prepared eltype $(eltype(prepared)) does not match workspace eltype $(eltype(workspace))"))
    nothing
end

const _NATIVE_PPL_LOG2PI = log(2π)

@inline function _native_ppl_logdensity_kernel(position::AbstractVector{T},
                                               prepared::NativePPLPrepared,
                                               buffers::NativePPLBuffers{T}) where {T}
    node = prepared.plan.nodes.location
    scale_factor = prepared.plan.factors.scale_prior
    intercept = position[node.intercept_index]
    slope = position[node.slope_index]
    log_scale = position[scale_factor.unconstrained_index]
    scale = exp(log_scale)
    prior_scale = T(scale_factor.scale)
    half = T(0.5)
    normalizer = T(_NATIVE_PPL_LOG2PI) * half

    # Standard-normal population coefficients, then Exponential(scale_prior)
    # over exp(log_scale), including the positive transform's log Jacobian.
    density = -half * intercept * intercept - normalizer
    density += -half * slope * slope - normalizer
    density += -log(prior_scale) - scale / prior_scale + log_scale

    inverse_scale = inv(scale)
    for i in eachindex(prepared.predictor, prepared.response)
        location = intercept + slope * prepared.predictor[i]
        residual = (prepared.response[i] - location) * inverse_scale
        pointwise = -log_scale - normalizer - half * residual * residual
        buffers.location[i] = location
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end

"""
    _native_ppl_logdensity!(workspace, prepared, position)

Evaluate in the caller-owned reusable workspace. Derived locations and
pointwise likelihoods are refreshed as part of the same graph execution.
"""
function _native_ppl_logdensity!(workspace::NativePPLWorkspace,
                                 prepared::NativePPLPrepared,
                                 position::AbstractVector)
    _native_ppl_check_execution(workspace, prepared, position)
    _native_ppl_logdensity_kernel(position, prepared, workspace.primal)
end

# Enzyme is an optional dependency. Its extension installs the concrete typed
# method; this catch-all keeps the core package's failure actionable without
# claiming a second AD implementation.
function _native_ppl_logdensity_and_gradient! end
_native_ppl_logdensity_and_gradient!(args...) = throw(ArgumentError(
    "native PPL gradients require loading Enzyme"))
