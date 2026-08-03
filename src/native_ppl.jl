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

"""A typed map from an unconstrained coordinate to a parameter's support."""
abstract type NativePPLTransform{S<:NativePPLSupport} end
struct NativePPLIdentityTransform <: NativePPLTransform{NativePPLRealSupport} end
struct NativePPLExpTransform <: NativePPLTransform{NativePPLPositiveSupport} end

@inline native_transform_forward(::NativePPLIdentityTransform, value) = value
@inline native_transform_forward(::NativePPLExpTransform, value) = exp(value)
@inline native_transform_logforward(transform::NativePPLTransform, value) =
    log(native_transform_forward(transform, value))
@inline native_transform_logforward(::NativePPLExpTransform, value) = value
@inline native_transform_logabsdetjac(::NativePPLIdentityTransform, value) = zero(value)
@inline native_transform_logabsdetjac(::NativePPLExpTransform, value) = value

"""A parameter block and its coordinates in the flat unconstrained vector."""
struct NativePPLParameter{
    Name,S<:NativePPLSupport,Tr<:NativePPLTransform{S},A,R,
}
    support::S
    transform::Tr
    axis::A
    unconstrained::R
end

NativePPLParameter(name::Symbol, support::S, transform::Tr,
                   axis::A, unconstrained::R) where {S,Tr,A,R} =
    NativePPLParameter{name,S,Tr,A,R}(support, transform, axis, unconstrained)
native_parameter_name(::NativePPLParameter{Name}) where {Name} = Name

@inline function native_parameter_value(parameter::NativePPLParameter,
                                         position::AbstractVector,
                                         coordinate::Int)
    native_transform_forward(parameter.transform, position[coordinate])
end

@inline function native_parameter_logabsdetjac(parameter::NativePPLParameter,
                                               position::AbstractVector,
                                               coordinate::Int)
    native_transform_logabsdetjac(parameter.transform, position[coordinate])
end

@inline function native_parameter_logvalue(parameter::NativePPLParameter,
                                           position::AbstractVector,
                                           coordinate::Int)
    native_transform_logforward(parameter.transform, position[coordinate])
end

@inline native_parameter_value(parameter::NativePPLParameter,
                               position::AbstractVector) =
    native_parameter_value(parameter, position, first(parameter.unconstrained))
@inline native_parameter_logabsdetjac(parameter::NativePPLParameter,
                                      position::AbstractVector) =
    native_parameter_logabsdetjac(
        parameter, position, first(parameter.unconstrained))
@inline native_parameter_logvalue(parameter::NativePPLParameter,
                                  position::AbstractVector) =
    native_parameter_logvalue(parameter, position, first(parameter.unconstrained))

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

"""A staged elementwise exponential link over one named deterministic node."""
struct NativePPLExpNode{Name,Input,A} <: NativePPLNode
    axis::A
end

NativePPLExpNode(name::Symbol, input::Symbol, axis::A) where {A} =
    NativePPLExpNode{name,input,A}(axis)
native_node_name(::NativePPLExpNode{Name}) where {Name} = Name
native_exp_input(::NativePPLExpNode{Name,Input}) where {Name,Input} = Input

"""A fitted mean-centering transform over one raw input column."""
struct NativePPLCenterNode{Name,Input,A,T} <: NativePPLNode
    axis::A
    mean::T
end

NativePPLCenterNode(name::Symbol, input::Symbol, axis::A, mean::T) where {A,T} =
    NativePPLCenterNode{name,input,A,T}(axis, mean)
native_node_name(::NativePPLCenterNode{Name}) where {Name} = Name
native_center_input(::NativePPLCenterNode{Name,Input}) where {Name,Input} = Input
native_fitted_transform_input(node::NativePPLCenterNode) =
    native_center_input(node)

"""A fitted corrected-sample-SD standardization over one raw input column."""
struct NativePPLZScaleNode{Name,Input,A,T} <: NativePPLNode
    axis::A
    mean::T
    scale::T
end

NativePPLZScaleNode(name::Symbol, input::Symbol, axis::A,
                    mean::T, scale::T) where {A,T} =
    NativePPLZScaleNode{name,input,A,T}(axis, mean, scale)
native_node_name(::NativePPLZScaleNode{Name}) where {Name} = Name
native_zscale_input(::NativePPLZScaleNode{Name,Input}) where {Name,Input} = Input
native_fitted_transform_input(node::NativePPLZScaleNode) =
    native_zscale_input(node)

"""Independent standard-normal prior over an unconstrained parameter range."""
struct NativePPLStandardNormalFactor{Parameter,R} <: NativePPLFactor
    unconstrained::R
end

NativePPLStandardNormalFactor(parameter::Symbol, unconstrained::R) where {R} =
    NativePPLStandardNormalFactor{parameter,R}(unconstrained)

"""Exponential prior on a constrained scalar; its parameter owns the transform."""
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

"""Row-wise Bernoulli observation factor parameterized by logits."""
struct NativePPLBernoulliLogitFactor{Response,Logit,A} <: NativePPLFactor
    axis::A
end

NativePPLBernoulliLogitFactor(response::Symbol, logit::Symbol, axis::A) where {A} =
    NativePPLBernoulliLogitFactor{response,logit,A}(axis)

"""Row-wise Poisson observation factor linked to a staged positive rate."""
struct NativePPLPoissonFactor{Response,Rate,A} <: NativePPLFactor
    axis::A
end

NativePPLPoissonFactor(response::Symbol, rate::Symbol, axis::A) where {A} =
    NativePPLPoissonFactor{response,rate,A}(axis)

abstract type NativePPLQuery end
struct NativePPLLinearPredictor <: NativePPLQuery end
struct NativePPLPointwiseLogLikelihood <: NativePPLQuery end
struct NativePPLPosteriorPredictive <: NativePPLQuery end

"""Rule selecting the prepared executor's numeric scalar as an output eltype."""
struct NativePPLPreparedElementType end

"""Rule selecting a fixed semantic output element type."""
struct NativePPLFixedElementType{T} end

"""Dense, one-dimensional caller-owned output layout."""
struct NativePPLDenseVectorLayout end

"""Dense draw-by-observation output owned by the caller."""
struct NativePPLDenseMatrixLayout end

"""Pre-execution semantic axis, element-type rule, and layout for one output."""
struct NativePPLOutputSignature{A,E,L}
    axis::A
    element_type::E
    layout::L
end

"""Pre-execution signature for a draw-by-observation query block."""
struct NativePPLBatchOutputSignature{D,A,E,L}
    draw_axis::D
    observation_axis::A
    element_type::E
    layout::L
end

native_output_axis(signature::NativePPLOutputSignature) = signature.axis
native_output_axis(signature::NativePPLBatchOutputSignature) =
    signature.observation_axis
native_output_axes(signature::NativePPLOutputSignature) = (signature.axis,)
native_output_axes(signature::NativePPLBatchOutputSignature) =
    (signature.draw_axis, signature.observation_axis)
native_output_draw_axis(signature::NativePPLBatchOutputSignature) =
    signature.draw_axis
native_output_eltype(signature::NativePPLOutputSignature, prepared) =
    native_output_eltype(signature.element_type, prepared)
native_output_eltype(signature::NativePPLBatchOutputSignature, prepared) =
    native_output_eltype(signature.element_type, prepared)
native_output_eltype(::NativePPLPreparedElementType, prepared) = eltype(prepared)
native_output_eltype(::NativePPLPreparedElementType, ::Type{T}) where {T<:AbstractFloat} = T
native_output_eltype(::NativePPLFixedElementType{T}, _) where {T} = T
native_output_layout(signature::NativePPLOutputSignature) = signature.layout
native_output_layout(signature::NativePPLBatchOutputSignature) = signature.layout
native_output_layout_accepts(::NativePPLDenseVectorLayout, output) =
    output isa DenseVector
native_output_layout_accepts(::NativePPLDenseMatrixLayout, output) =
    output isa DenseMatrix

"""
Output, frequency, effect, and lifetime contract for one graph query.

The query kind and execution properties are type parameters so query planning
does not depend on a post-hoc name registry.
"""
struct NativePPLQuerySpec{Kind,Stage,Effect,Lifetime,S}
    output::S
end

NativePPLQuerySpec(kind::Symbol, stage::Symbol, effect::Symbol,
                   lifetime::Symbol, output::S) where {S} =
    NativePPLQuerySpec{kind,stage,effect,lifetime,S}(output)
native_query_name(::NativePPLQuerySpec{Kind}) where {Kind} = Kind
native_query_effect(::NativePPLQuerySpec{Kind,Stage,Effect}) where {Kind,Stage,Effect} =
    Effect
native_query_output(spec::NativePPLQuerySpec) = spec.output

function _native_ppl_queries(
    observation_axis,
    predictive_element=NativePPLPreparedElementType(),
)
    numeric_output = NativePPLOutputSignature(
        observation_axis, NativePPLPreparedElementType(),
        NativePPLDenseVectorLayout())
    predictive_output = predictive_element isa NativePPLPreparedElementType ?
        numeric_output : NativePPLOutputSignature(
            observation_axis, predictive_element, NativePPLDenseVectorLayout())
    linear_predictor = NativePPLQuerySpec(
        :linear_predictor, :per_draw, :workspace, :until_next_evaluation,
        numeric_output)
    pointwise_loglikelihood = NativePPLQuerySpec(
        :pointwise_loglikelihood, :per_draw, :workspace,
        :until_next_evaluation, numeric_output)
    posterior_predictive = NativePPLQuerySpec(
        :posterior_predictive, :per_draw, :rng, :caller_owned,
        predictive_output)
    (; linear_predictor, pointwise_loglikelihood, posterior_predictive)
end

_native_ppl_queries(observation_axis, ::NativePPLNormalFactor) =
    _native_ppl_queries(observation_axis)
_native_ppl_queries(observation_axis, ::NativePPLBernoulliLogitFactor) =
    _native_ppl_queries(observation_axis, NativePPLFixedElementType{Bool}())
_native_ppl_queries(observation_axis, ::NativePPLPoissonFactor) =
    _native_ppl_queries(observation_axis, NativePPLFixedElementType{Int}())

"""
    NativePPLPlan

Typed, inspectable native-Julia execution plan. `bindings` are the initial
input values captured from the source `BRMI`; execution preparation copies
them into separately owned buffers, so the plan itself is never a workspace.

This is deliberately internal while the public native-PPL naming is unsettled.
The initial lowerings accept two structural model shapes and fail closed for
all other formula capabilities.
"""
struct NativePPLPlan{A,I,P,N,F,Q,B}
    axes::A
    inputs::I
    parameters::P
    nodes::N
    factors::F
    queries::Q
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

"""Explicit absence of an observation binding in a prediction-only replay."""
struct NativePPLNoResponse end

Base.eltype(prepared::NativePPLPrepared) = eltype(prepared.predictor)
native_ppl_has_response(prepared::NativePPLPrepared) =
    !(prepared.response isa NativePPLNoResponse)
LogDensityProblems.dimension(prepared::NativePPLPrepared) =
    LogDensityProblems.dimension(prepared.plan)

"""
Reusable storage for density and reverse-mode gradient execution.

The returned gradient and pointwise buffers alias this workspace. Callers that
need to retain a result across another evaluation must copy it. A workspace is
single-task state; concurrent callers use separate workspaces over the same
immutable `NativePPLPrepared` value.
"""
struct NativePPLWorkspace{T,B,G,D}
    primal::B
    gradient::G
    derivative::D
end

Base.eltype(::NativePPLWorkspace{T}) where {T} = T

function NativePPLWorkspace(prepared::NativePPLPrepared, ::Type{T}=eltype(prepared)) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError(
        "native PPL workspace element type must be concrete; got $T"))
    n = length(prepared.workspace_spec.observation_axis)
    buffers() = NativePPLBuffers(zeros(T, n), zeros(T, n))
    NativePPLWorkspace{T,NativePPLBuffers{T,Vector{T}},Vector{T},Nothing}(
        buffers(), zeros(T, prepared.workspace_spec.gradient_length), nothing)
end

function _native_ppl_workspace end
_native_ppl_workspace(prepared::NativePPLPrepared, ::Type{T}) where {T<:AbstractFloat} =
    NativePPLWorkspace(prepared, T)
_native_ppl_workspace(::NativePPLPrepared, ::Type{<:AbstractFloat}, backend) =
    throw(ArgumentError(
        "native PPL derivative workspaces require loading DifferentiationInterface; " *
        "only AutoEnzyme is tested, recommended, and guaranteed"))

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
          join((string(nameof(typeof(factor))) for factor in plan.factors), ", "), "\n")
    print(io, "  queries: ",
          join((string(native_query_name(query)) for query in plan.queries), ", "))
end

function Base.show(io::IO, prepared::NativePPLPrepared)
    print(io, "NativePPLPrepared(", length(prepared.plan.axes.observation),
          " observations, eltype=", eltype(prepared))
    native_ppl_has_response(prepared) || print(io, ", prediction-only")
    print(io, ")")
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

_native_ppl_ref_name(x::NamedColumn) = _native_ppl_ref_name(x, parent(x))
_native_ppl_ref_name(x::NamedColumn, ::MissingColumn) = name(x)
_native_ppl_ref_name(x::NamedColumn, operation::ExprColumn{typeof(~)}) = name(x)
_native_ppl_ref_name(_, _) = nothing
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

function _native_ppl_predictor_term(term, key::Symbol)
    if term isa NamedColumn && parent(term) isa DataColumn
        return (; column=term, transform=:identity)
    end
    if term isa ExprColumn &&
       (getf(term) === center || getf(term) === zscale ||
        getf(term) === standardize)
        transform = getf(term) === center ? :center : :zscale
        spelling = getf(term) === standardize ? :standardize : transform
        isempty(getkwargs(term)) || throw(NativePPLCapabilityError(
            :predictor_transform,
            "`$spelling` in predictor `$key` cannot have keywords"))
        args = getargs(term)
        length(args) == 1 || throw(NativePPLCapabilityError(
            :predictor_transform,
            "`$spelling` in predictor `$key` needs one raw data column"))
        column = only(args)
        column isa NamedColumn && parent(column) isa DataColumn ||
            throw(NativePPLCapabilityError(
                :predictor_transform,
                "`$spelling` in predictor `$key` must wrap one raw data column"))
        return (; column, transform)
    end
    throw(NativePPLCapabilityError(
        :predictor_terms,
        "`$key` must be exactly `1 + x`, `1 + center(x)`, or " *
        "`1 + zscale(x)` (`standardize(x)` is an alias); " *
        "offsets, interactions, groups, and other transforms are not lowered yet"))
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
    length(intercepts) == 1 ||
        throw(NativePPLCapabilityError(:predictor_terms,
            "`$key` must contain exactly one intercept"))
    predictor_terms = filter(
        term -> !(term isa Number && term == 1), terms)
    length(predictor_terms) == 1 || throw(NativePPLCapabilityError(
        :predictor_terms, "`$key` must contain exactly one predictor term"))
    _native_ppl_predictor_term(only(predictor_terms), key)
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

function _native_ppl_fit_mean(values::AbstractVector, name::Symbol,
                              transform::Symbol)
    all(value -> value isa Real && isfinite(value), values) ||
        throw(NativePPLCapabilityError(
            :predictor_transform,
            "`$transform($name)` requires finite real training values"))
    isempty(values) && throw(NativePPLCapabilityError(
        :predictor_transform,
        "`$transform($name)` requires at least one training value"))
    fitted_mean = float(first(values))
    for (offset, value) in enumerate(Iterators.drop(values, 1))
        count = offset + 1
        # Dividing before subtracting avoids overflow for finite values near
        # `floatmax`, including samples spanning both signs.
        fitted_mean += float(value) / count - fitted_mean / count
    end
    isfinite(fitted_mean) || throw(NativePPLCapabilityError(
        :predictor_transform,
        "`$transform($name)` produced a non-finite fitted mean"))
    fitted_mean
end

_native_ppl_fit_center(values::AbstractVector, name::Symbol) =
    _native_ppl_fit_mean(values, name, :center)

function _native_ppl_fit_zscale(values::AbstractVector, name::Symbol)
    length(values) >= 2 || throw(NativePPLCapabilityError(
        :predictor_transform,
        "`zscale($name)` requires at least two training values for sample SD"))
    fitted_mean = _native_ppl_fit_mean(values, name, :zscale)

    # Scaled sum-of-squares avoids the overflow and underflow of directly
    # accumulating `(x - mean)^2`, while preserving corrected `n - 1`
    # sample-standard-deviation semantics.
    magnitude_scale = zero(fitted_mean)
    scaled_squares = zero(fitted_mean)
    for value in values
        deviation = float(value) - fitted_mean
        isfinite(deviation) || throw(NativePPLCapabilityError(
            :predictor_transform,
            "`zscale($name)` produced a non-finite centered training value"))
        magnitude = abs(deviation)
        iszero(magnitude) && continue
        if magnitude_scale < magnitude
            ratio = magnitude_scale / magnitude
            scaled_squares = one(magnitude) + scaled_squares * ratio * ratio
            magnitude_scale = magnitude
        else
            ratio = magnitude / magnitude_scale
            scaled_squares += ratio * ratio
        end
    end
    iszero(magnitude_scale) && throw(NativePPLCapabilityError(
        :predictor_transform,
        "`zscale($name)` requires nonzero sample variance"))
    fitted_scale = magnitude_scale *
        sqrt(scaled_squares / (length(values) - 1))
    isfinite(fitted_scale) && fitted_scale > zero(fitted_scale) ||
        throw(NativePPLCapabilityError(
            :predictor_transform,
            "`zscale($name)` produced a non-finite or zero sample SD"))
    (; mean=fitted_mean, scale=fitted_scale)
end

"""
    _native_ppl_plan(brmi::BRMI) -> NativePPLPlan

Lower either initial workflow-complete model family:

```julia
y     ~ Normal(mu, sigma)
mu    ~ 1 + x
sigma ~ Exponential(scale)

# or

y     ~ BernoulliLogit(eta)
eta   ~ 1 + x

# or

y     ~ Poisson(exp(eta))
eta   ~ 1 + x
```

Names may vary, but the structure may not. Unsupported structure raises a
`NativePPLCapabilityError` naming the missing capability.
"""
function _native_ppl_plan(brmi::BRMI)
    observed = outcomes(brmi)
    length(observed) == 1 || throw(NativePPLCapabilityError(
        :outcomes, "expected exactly one observed response, got $(length(observed))"))
    outcome = only(observed)
    family = outcome.family
    family === Normal || family === BernoulliLogit || family === Poisson ||
        throw(NativePPLCapabilityError(
            :likelihood,
            "expected `Normal(location, scale)`, `BernoulliLogit(logit)`, " *
            "or `Poisson(exp(log_rate))`, got `$family`"))

    response = outcome.response
    response_lhs, likelihood = _native_ppl_sampling_rhs(brmi, response)
    response_lhs isa NamedColumn && parent(response_lhs) isa DataColumn ||
        throw(NativePPLCapabilityError(:response_decorator,
            "response `$response` must be a bare observed data column"))
    likelihood isa ExprColumn && getf(likelihood) === family ||
        throw(NativePPLCapabilityError(:likelihood,
            "response `$response` must use `$family` consistently"))
    isempty(getkwargs(likelihood)) || throw(NativePPLCapabilityError(
        :likelihood_keywords, "$family likelihood for `$response` has keywords"))
    likelihood_args = getargs(likelihood)
    expected_arguments = family === Normal ? 2 : 1
    length(likelihood_args) == expected_arguments ||
        throw(NativePPLCapabilityError(
            :likelihood,
            "$family likelihood for `$response` needs $expected_arguments argument(s)"))
    rate = nothing
    location = if family === Poisson
        rate_expression = only(likelihood_args)
        rate_expression isa ExprColumn && getf(rate_expression) === exp ||
            throw(NativePPLCapabilityError(
                :likelihood_link,
                "Poisson response `$response` must use `Poisson(exp(log_rate))`"))
        isempty(getkwargs(rate_expression)) || throw(NativePPLCapabilityError(
            :likelihood_link, "Poisson `exp` link cannot have keywords"))
        rate_arguments = getargs(rate_expression)
        length(rate_arguments) == 1 || throw(NativePPLCapabilityError(
            :likelihood_link, "Poisson `exp` link needs one named predictor"))
        log_rate = _native_ppl_ref_name(only(rate_arguments))
        log_rate === nothing && throw(NativePPLCapabilityError(
            :likelihood_location,
            "Poisson `exp` link must consume one named linear predictor"))
        rate = Symbol(:exp_, log_rate)
        log_rate
    else
        _native_ppl_ref_name(likelihood_args[1])
    end
    location === nothing && throw(NativePPLCapabilityError(
        :likelihood_location, "$family predictor must be one named linear predictor"))
    scale_parameter = family === Normal ? _native_ppl_ref_name(likelihood_args[2]) : nothing
    family === Normal && scale_parameter === nothing &&
        throw(NativePPLCapabilityError(
            :likelihood_scale, "Normal scale must be one named scalar parameter"))

    predictor_term = _native_ppl_affine_predictor(brmi, location)
    predictor = predictor_term.column
    predictor_name = name(predictor)
    predictor_name === response && throw(NativePPLCapabilityError(
        :input_roles,
        "predictor `$predictor_name` is also the observed response; the first native plan requires distinct input roles"))
    x = parent(parent(predictor))
    y = parent(parent(response_lhs))
    x isa AbstractVector{<:Real} && !(eltype(x) <: Integer) ||
        throw(NativePPLCapabilityError(:predictor_type,
            "predictor `$predictor_name` must be a continuous real vector"))
    y isa AbstractVector{<:Real} || throw(NativePPLCapabilityError(
        :response_type, "response `$response` must be a real or Bool vector"))
    if family === BernoulliLogit
        all(value -> value == 0 || value == 1, y) ||
            throw(NativePPLCapabilityError(
                :response_support,
                "BernoulliLogit response `$response` must contain only Bool/0/1 values"))
    elseif family === Poisson
        all(_native_ppl_is_count, y) ||
            throw(NativePPLCapabilityError(
                :response_support,
                "Poisson response `$response` must contain nonnegative integer-valued counts representable as Int"))
    end
    length(x) == length(y) || throw(NativePPLCapabilityError(
        :observation_axis,
        "predictor `$predictor_name` has $(length(x)) rows but `$response` has $(length(y))"))
    !isempty(y) || throw(NativePPLCapabilityError(
        :observation_axis, "the observation axis cannot be empty"))
    transform_fit = if predictor_term.transform === :center
        _native_ppl_fit_center(x, predictor_name)
    elseif predictor_term.transform === :zscale
        _native_ppl_fit_zscale(x, predictor_name)
    else
        nothing
    end

    prior_scale = family === Normal ?
        _native_ppl_exponential_prior(brmi, scale_parameter) : nothing
    expected = family === Normal ?
        Set((location, scale_parameter, response, predictor_name)) :
        Set((location, response, predictor_name))
    extras = setdiff(Set(keys(brmi.operations)), expected)
    isempty(extras) || throw(NativePPLCapabilityError(
        :additional_operations,
        "unsupported formula operations: $(join(sort!(collect(extras)), ", "))"))

    observation_axis = NativePPLAxis(:observation, Base.OneTo(length(y)))
    coefficient_axis = NativePPLAxis(Symbol(location, :_coefficient),
                                     (:Intercept, predictor_name))

    predictor_input = NativePPLInput(predictor_name, :predictor,
                                     observation_axis, eltype(x))
    response_input = NativePPLInput(response, :response,
                                    observation_axis, eltype(y))
    coefficient_name = Symbol(:beta_, location)
    coefficients = NativePPLParameter(
        coefficient_name, NativePPLRealSupport(), NativePPLIdentityTransform(),
        coefficient_axis, 1:2)
    transform_node = if predictor_term.transform === :center
        NativePPLCenterNode(
            Symbol("#native_ppl_center#", location, "#", predictor_name),
            predictor_name,
            observation_axis, transform_fit)
    elseif predictor_term.transform === :zscale
        NativePPLZScaleNode(
            Symbol("#native_ppl_zscale#", location, "#", predictor_name),
            predictor_name, observation_axis,
            transform_fit.mean, transform_fit.scale)
    else
        nothing
    end
    location_input = transform_node === nothing ?
        predictor_name : native_node_name(transform_node)
    location_node = NativePPLAffineNode(location, location_input,
                                        observation_axis, 1, 2)
    nodes = transform_node === nothing ?
        (; location=location_node) :
        (; transform=transform_node, location=location_node)

    coefficient_prior = NativePPLStandardNormalFactor(coefficient_name, 1:2)
    bindings = NamedTuple{(predictor_name, response)}((x, y))

    if family === Normal
        scale_axis = NativePPLAxis(
            Symbol(scale_parameter, :_scalar), (scale_parameter,))
        scale = NativePPLParameter(
            scale_parameter, NativePPLPositiveSupport(), NativePPLExpTransform(),
            scale_axis, 3:3)
        scale_prior = NativePPLExponentialFactor(scale_parameter, 3, prior_scale)
        likelihood_factor = NativePPLNormalFactor(
            response, location, scale_parameter, observation_axis)
        NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis,
               scale=scale_axis),
            (; predictor=predictor_input, response=response_input),
            (; coefficients, scale),
            nodes,
            (; coefficient_prior, scale_prior, likelihood=likelihood_factor),
            _native_ppl_queries(observation_axis, likelihood_factor),
            bindings,
        )
    elseif family === BernoulliLogit
        likelihood_factor = NativePPLBernoulliLogitFactor(
            response, location, observation_axis)
        NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis),
            (; predictor=predictor_input, response=response_input),
            (; coefficients),
            nodes,
            (; coefficient_prior, likelihood=likelihood_factor),
            _native_ppl_queries(observation_axis, likelihood_factor),
            bindings,
        )
    else
        rate_node = NativePPLExpNode(rate, location, observation_axis)
        likelihood_factor = NativePPLPoissonFactor(
            response, rate, observation_axis)
        NativePPLPlan(
            (; observation=observation_axis, coefficient=coefficient_axis),
            (; predictor=predictor_input, response=response_input),
            (; coefficients),
            merge(nodes, (; rate=rate_node)),
            (; coefficient_prior, likelihood=likelihood_factor),
            _native_ppl_queries(observation_axis, likelihood_factor),
            bindings,
        )
    end
end

function _native_ppl_copy_input(::Type{T}, input, role::Symbol, name::Symbol) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(input))
    for (i, value) in enumerate(input)
        value isa Real && isfinite(value) || throw(ArgumentError(
            "native PPL $role `$name` contains a non-finite or non-real value at row $i"))
        output[i] = value
        isfinite(output[i]) || throw(ArgumentError(
            "native PPL $role `$name` cannot be represented as $T at row $i"))
    end
    output
end

function _native_ppl_prepare_response(::Type{T}, response::AbstractVector,
                                      name::Symbol, observations::Int) where {T<:AbstractFloat}
    output = _native_ppl_copy_input(T, response, :response, name)
    length(output) == observations || throw(DimensionMismatch(
        "native PPL response `$name` has $(length(output)) rows but predictor has " *
        "$observations"))
    output
end

_native_ppl_prepare_response(::Type{<:AbstractFloat},
                             response::NativePPLNoResponse,
                             ::Symbol, ::Int) = response

_native_ppl_validate_response(::NativePPLNormalFactor, ::AbstractVector, ::Symbol) =
    nothing

function _native_ppl_is_count(value)
    value isa Real && isfinite(value) && value >= zero(value) &&
        isinteger(value) || return false
    try
        Int(value)
        true
    catch
        false
    end
end

function _native_ppl_validate_response(::NativePPLBernoulliLogitFactor,
                                       response::AbstractVector, name::Symbol)
    for (i, value) in enumerate(response)
        value == 0 || value == 1 || throw(ArgumentError(
            "native PPL BernoulliLogit response `$name` must be Bool/0/1; " *
            "got $value at row $i"))
    end
    nothing
end
function _native_ppl_validate_response(::NativePPLPoissonFactor,
                                       response::AbstractVector, name::Symbol)
    for (i, value) in enumerate(response)
        _native_ppl_is_count(value) || throw(ArgumentError(
            "native PPL Poisson response `$name` must be a nonnegative " *
            "integer-valued count representable as Int; got $value at row $i"))
    end
    nothing
end
_native_ppl_validate_response(::NativePPLFactor, ::NativePPLNoResponse, ::Symbol) =
    nothing

function _native_ppl_apply_predictor!(
    node::NativePPLCenterNode{Name,Input},
    ::NativePPLInput{Input},
    predictor::Vector{T},
) where {Name,Input,T}
    mean = T(node.mean)
    isfinite(mean) || throw(ArgumentError(
        "native PPL fitted center mean for `$Input` cannot be represented as $T"))
    for i in eachindex(predictor)
        predictor[i] -= mean
        isfinite(predictor[i]) || throw(ArgumentError(
            "native PPL centered predictor `$Input` is non-finite at row $i"))
    end
    predictor
end

function _native_ppl_apply_predictor!(
    node::NativePPLZScaleNode{Name,Input},
    ::NativePPLInput{Input},
    predictor::Vector{T},
) where {Name,Input,T}
    mean = T(node.mean)
    scale = T(node.scale)
    isfinite(mean) || throw(ArgumentError(
        "native PPL fitted zscale mean for `$Input` cannot be represented as $T"))
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "native PPL fitted zscale sample SD for `$Input` cannot be represented " *
        "as a finite positive $T"))
    for i in eachindex(predictor)
        raw = predictor[i]
        standardized = (raw - mean) / scale
        if !isfinite(standardized)
            # The separated form can represent a wide-span rebound value when
            # `raw - mean` overflows even though the standardized result does not.
            standardized = raw / scale - mean / scale
        end
        isfinite(standardized) || throw(ArgumentError(
            "native PPL standardized predictor `$Input` is non-finite at row $i"))
        predictor[i] = standardized
    end
    predictor
end

function _native_ppl_apply_predictor!(plan::NativePPLPlan,
                                      predictor::Vector)
    hasproperty(plan.nodes, :transform) || return predictor
    _native_ppl_apply_predictor!(
        plan.nodes.transform, plan.inputs.predictor, predictor)
end

_native_ppl_validate_response_conversion(
    ::NativePPLFactor, ::Any, ::Any, ::Symbol,
) = nothing
function _native_ppl_validate_response_conversion(
    ::NativePPLPoissonFactor,
    response::AbstractVector,
    prepared_response::AbstractVector,
    name::Symbol,
)
    for i in eachindex(response, prepared_response)
        prepared_response[i] == response[i] || throw(ArgumentError(
            "native PPL Poisson response `$name` count $(response[i]) at row $i " *
            "cannot be represented exactly as $(eltype(prepared_response))"))
    end
    nothing
end

function _native_ppl_prepare_bindings(plan::NativePPLPlan, predictor,
                                      response, ::Type{T}) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError(
        "native PPL prepared element type must be concrete; got $T"))
    predictor_name = native_input_name(plan.inputs.predictor)
    response_name = native_input_name(plan.inputs.response)
    prepared_predictor = _native_ppl_copy_input(
        T, predictor, :predictor, predictor_name)
    length(prepared_predictor) == length(plan.axes.observation) ||
        throw(DimensionMismatch(
            "native PPL predictor `$predictor_name` has " *
            "$(length(prepared_predictor)) rows but the observation axis has " *
            "$(length(plan.axes.observation))"))
    _native_ppl_apply_predictor!(plan, prepared_predictor)
    _native_ppl_validate_response(
        plan.factors.likelihood, response, response_name)
    prepared_response = _native_ppl_prepare_response(
        T, response, response_name, length(prepared_predictor))
    _native_ppl_validate_response_conversion(
        plan.factors.likelihood, response, prepared_response, response_name)
    workspace_spec = NativePPLWorkspaceSpec(
        plan.axes.observation, LogDensityProblems.dimension(plan))
    NativePPLPrepared(plan, prepared_predictor, prepared_response, workspace_spec)
end

"""
    _native_ppl_prepare(plan; T=Float64) -> NativePPLPrepared

Validate and copy the plan's initial input binding into numeric, executor-owned
storage. The returned value is shareable; mutable evaluation state is supplied
separately by `NativePPLWorkspace`.
"""
function _native_ppl_prepare(plan::NativePPLPlan; T::Type{<:AbstractFloat}=Float64)
    isconcretetype(T) || throw(ArgumentError(
        "native PPL prepared element type must be concrete; got $T"))
    predictor_name = native_input_name(plan.inputs.predictor)
    response_name = native_input_name(plan.inputs.response)
    predictor = getproperty(plan.bindings, predictor_name)
    response = getproperty(plan.bindings, response_name)
    _native_ppl_prepare_bindings(plan, predictor, response, T)
end


function _native_ppl_require_response(prepared::NativePPLPrepared,
                                      operation::AbstractString)
    native_ppl_has_response(prepared) || throw(ArgumentError(
        "native PPL $operation requires an observed response binding; " *
        "this prepared value is prediction-only, so rebind it with both " *
        "predictor and response inputs"))
    prepared.response
end

function _native_ppl_check_location_execution(workspace::NativePPLWorkspace,
                                              prepared::NativePPLPrepared,
                                              position::AbstractVector)
    dimension = LogDensityProblems.dimension(prepared)
    observations = length(prepared.plan.axes.observation)
    length(position) == dimension || throw(DimensionMismatch(
        "native PPL position has length $(length(position)); expected $dimension"))
    length(prepared.predictor) == observations || throw(DimensionMismatch(
        "native PPL prepared predictor has $(length(prepared.predictor)) rows; expected $observations"))
    length(workspace.primal.location) == observations || throw(DimensionMismatch(
        "native PPL workspace location has $(length(workspace.primal.location)) rows; expected $observations"))
    eltype(position) === eltype(workspace) || throw(ArgumentError(
        "native PPL position eltype $(eltype(position)) does not match workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL prepared eltype $(eltype(prepared)) does not match workspace eltype $(eltype(workspace))"))
    Base.mightalias(position, workspace.primal.location) && throw(ArgumentError(
        "native PPL position must not alias the workspace location buffer"))
    nothing
end

function _native_ppl_check_execution(workspace::NativePPLWorkspace,
                                     prepared::NativePPLPrepared,
                                     position::AbstractVector)
    response = _native_ppl_require_response(prepared, "log density")
    _native_ppl_check_location_execution(workspace, prepared, position)
    dimension = LogDensityProblems.dimension(prepared)
    observations = length(prepared.plan.axes.observation)
    length(workspace.gradient) == dimension || throw(DimensionMismatch(
        "native PPL workspace gradient has length $(length(workspace.gradient)); expected $dimension"))
    length(response) == observations || throw(DimensionMismatch(
        "native PPL prepared response has $(length(response)) rows; expected $observations"))
    length(workspace.primal.pointwise_loglikelihood) == observations ||
        throw(DimensionMismatch(
            "native PPL workspace pointwise likelihood has " *
            "$(length(workspace.primal.pointwise_loglikelihood)) rows; expected $observations"))
    Base.mightalias(position, workspace.gradient) && throw(ArgumentError(
        "native PPL position must not alias the workspace gradient"))
    Base.mightalias(position, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL position must not alias the workspace pointwise likelihood buffer"))
    nothing
end

const _NATIVE_PPL_LOG2PI = log(2π)
const _NATIVE_PPL_HALF_LOG2PI = _NATIVE_PPL_LOG2PI / 2

# Distributions may supply formula-boundary vocabulary, but numerical execution
# is PPL-owned. Keep these vectorized factor/transform kernels independent of
# Distributions and Bijectors so later compiler passes (including a possible
# Reactant lowering) see the complete executable semantics here.

@inline function _native_ppl_factor_logdensity(
    factor::NativePPLStandardNormalFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
    position::AbstractVector{T},
) where {Parameter,T}
    half = T(0.5)
    normalizer = T(_NATIVE_PPL_HALF_LOG2PI)
    density = zero(T)
    for coordinate in factor.unconstrained
        value = native_parameter_value(parameter, position, coordinate)
        logjac = native_parameter_logabsdetjac(parameter, position, coordinate)
        density += -half * value * value - normalizer + logjac
    end
    density
end

@inline function _native_ppl_factor_logdensity(
    factor::NativePPLExponentialFactor{Parameter},
    parameter::NativePPLParameter{Parameter},
    position::AbstractVector{T},
) where {Parameter,T}
    coordinate = factor.unconstrained_index
    value = native_parameter_value(parameter, position, coordinate)
    logjac = native_parameter_logabsdetjac(parameter, position, coordinate)
    prior_scale = T(factor.scale)
    -log(prior_scale) - value / prior_scale + logjac
end

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLNormalFactor{Response,Location,Scale},
    ::NativePPLInput{Response},
    ::NativePPLAffineNode{Location},
    scale_parameter::NativePPLParameter{Scale},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Location,Scale,T}
    scale = native_parameter_value(scale_parameter, position)
    log_scale = native_parameter_logvalue(scale_parameter, position)
    inverse_scale = inv(scale)
    half = T(0.5)
    normalizer = T(_NATIVE_PPL_HALF_LOG2PI)
    density = zero(T)
    for i in eachindex(prepared.response)
        difference = prepared.response[i] - buffers.location[i]
        residual = iszero(difference) ? zero(T) : difference * inverse_scale
        pointwise = -log_scale - normalizer - half * residual * residual
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end

@inline _native_ppl_softplus(value) = value > zero(value) ?
    value + log1p(exp(-value)) : log1p(exp(value))

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLBernoulliLogitFactor{Response,Logit},
    ::NativePPLInput{Response},
    ::NativePPLAffineNode{Logit},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Logit,T}
    density = zero(T)
    for i in eachindex(prepared.response)
        logit = buffers.location[i]
        pointwise = isone(prepared.response[i]) ?
            -_native_ppl_softplus(-logit) : -_native_ppl_softplus(logit)
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end

@inline function _native_ppl_stirling_correction(value::T) where {T}
    inverse = inv(value)
    inverse2 = inverse * inverse
    inverse * (
        T(1 / 12) + inverse2 * (
            T(-1 / 360) + inverse2 * (
                T(1 / 1260) + inverse2 * (
                    T(-1 / 1680) + inverse2 * T(1 / 1188)))))
end

@inline function _native_ppl_logfactorial(::Type{T}, count::Int) where {T}
    count < 2 && return zero(T)
    if count < 32
        value = zero(T)
        for k in 2:count
            value += log(T(k))
        end
        return value
    end

    value = T(count)
    correction = _native_ppl_stirling_correction(value)
    (value + T(0.5)) * log(value) - value +
        T(_NATIVE_PPL_HALF_LOG2PI) + correction
end

@inline _native_ppl_logfactorial(value::T) where {T<:AbstractFloat} =
    _native_ppl_logfactorial(T, Int(value))

@inline function _native_ppl_delta_minus_expm1(value::T) where {T}
    abs(value) > T(0.01) && return value - expm1(value)
    series = T(1 / 479001600)
    series = T(1 / 39916800) + value * series
    series = T(1 / 3628800) + value * series
    series = T(1 / 362880) + value * series
    series = T(1 / 40320) + value * series
    series = T(1 / 5040) + value * series
    series = T(1 / 720) + value * series
    series = T(1 / 120) + value * series
    series = T(1 / 24) + value * series
    series = T(1 / 6) + value * series
    series = T(1 / 2) + value * series
    -(value * value) * series
end

@inline function _native_ppl_poisson_logdensity(count::T, log_rate::T) where {T}
    iszero(count) && return -exp(log_rate)
    count_int = Int(count)
    count_int < 32 && return count * log_rate - exp(log_rate) -
        _native_ppl_logfactorial(T, count_int)

    delta = log_rate - log(count)
    count * _native_ppl_delta_minus_expm1(delta) -
        T(0.5) * log(count) - T(_NATIVE_PPL_HALF_LOG2PI) -
        _native_ppl_stirling_correction(count)
end

@inline function _native_ppl_factor_logdensity!(
    ::NativePPLPoissonFactor{Response,Rate},
    ::NativePPLInput{Response},
    ::NativePPLExpNode{Rate,LogRate},
    ::NativePPLAffineNode{LogRate},
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {Response,Rate,LogRate,T}
    density = zero(T)
    for i in eachindex(prepared.response)
        count = prepared.response[i]
        log_rate = buffers.location[i]
        pointwise = _native_ppl_poisson_logdensity(count, log_rate)
        buffers.pointwise_loglikelihood[i] = pointwise
        density += pointwise
    end
    density
end


@inline function _native_ppl_logistic(value)
    if value >= zero(value)
        inverse = exp(-value)
        inv(one(value) + inverse)
    else
        forward = exp(value)
        forward / (one(value) + forward)
    end
end

@inline function _native_ppl_rand_poisson(
    rng::AbstractRNG, ::Type{T}, log_rate::T,
) where {T<:AbstractFloat}
    isnan(log_rate) && throw(DomainError(
        log_rate, "native PPL Poisson log-rate must not be NaN"))
    log_rate == -T(Inf) && return 0
    rate = exp(log_rate)
    max_rate = min(T(typemax(Int)), maxintfloat(T)) / T(4)
    isfinite(rate) && rate <= max_rate || throw(DomainError(
        log_rate,
        "native PPL Poisson rate is too large for an Int predictive output"))
    iszero(rate) && return 0

    if rate < T(10)
        threshold = exp(-rate)
        count = 0
        product = rand(rng, T)
        while product > threshold
            count += 1
            product *= rand(rng, T)
        end
        return count
    end

    # PTRS transformed rejection. This branch avoids the underflow and linear
    # work of product inversion for moderate and large rates.
    root_rate = sqrt(rate)
    b = T(0.931) + T(2.53) * root_rate
    a = T(-0.059) + T(0.02483) * b
    inverse_alpha = T(1.1239) + T(1.1328) / (b - T(3.4))
    squeeze = T(0.9277) - T(3.6224) / (b - T(2))
    while true
        u = rand(rng, T) - T(0.5)
        v = rand(rng, T)
        us = T(0.5) - abs(u)
        iszero(us) && continue
        candidate_value = floor((T(2) * a / us + b) * u + rate + T(0.43))
        candidate_value < zero(T) && continue
        candidate_value <= T(typemax(Int)) || throw(DomainError(
            candidate_value,
            "native PPL Poisson draw is too large for an Int predictive output"))
        count = Int(candidate_value)
        us >= T(0.07) && v <= squeeze && return count
        (us < T(0.013) && v > us) && continue
        acceptance = log(
            v * inverse_alpha / (a / (us * us) + b))
        target = _native_ppl_poisson_logdensity(T(count), log_rate)
        acceptance <= target && return count
    end
end


@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLNormalFactor{Response,Location,Scale},
    ::NativePPLInput{Response},
    ::NativePPLAffineNode{Location},
    output::AbstractVector,
    scale_parameter::NativePPLParameter{Scale},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Location,Scale,T}
    scale = native_parameter_value(scale_parameter, position)
    for i in eachindex(output)
        output[i] = buffers.location[i] + scale * randn(rng, T)
    end
    output
end

@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLPoissonFactor{Response,Rate},
    ::NativePPLInput{Response},
    ::NativePPLExpNode{Rate,LogRate},
    ::NativePPLAffineNode{LogRate},
    output::AbstractVector{Int},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Rate,LogRate,T}
    for i in eachindex(output)
        output[i] = _native_ppl_rand_poisson(rng, T, buffers.location[i])
    end
    output
end

@inline function _native_ppl_factor_simulate!(
    rng::AbstractRNG,
    ::NativePPLBernoulliLogitFactor{Response,Logit},
    ::NativePPLInput{Response},
    ::NativePPLAffineNode{Logit},
    output::AbstractVector{Bool},
    position::AbstractVector,
    buffers::NativePPLBuffers{T},
) where {Response,Logit,T}
    for i in eachindex(output)
        output[i] = rand(rng, T) < _native_ppl_logistic(buffers.location[i])
    end
    output
end


@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLNormalFactor,
    output::AbstractVector,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, output, prepared.plan.parameters.scale,
        position, buffers)
end

@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLBernoulliLogitFactor,
    output::AbstractVector{Bool},
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, output, position, buffers)
end

@inline function _native_ppl_model_simulate!(
    rng::AbstractRNG,
    factor::NativePPLPoissonFactor,
    output::AbstractVector{Int},
    prepared::NativePPLPrepared,
    position::AbstractVector,
    buffers::NativePPLBuffers,
)
    _native_ppl_factor_simulate!(
        rng, factor, prepared.plan.inputs.response,
        prepared.plan.nodes.rate, prepared.plan.nodes.location,
        output, position, buffers)
end

@inline function _native_ppl_location_kernel!(position::AbstractVector{T},
                                              prepared::NativePPLPrepared,
                                              buffers::NativePPLBuffers{T}) where {T}
    node = prepared.plan.nodes.location
    coefficient_parameter = prepared.plan.parameters.coefficients
    intercept = native_parameter_value(
        coefficient_parameter, position, node.intercept_index)
    slope = native_parameter_value(
        coefficient_parameter, position, node.slope_index)
    for i in eachindex(prepared.predictor)
        buffers.location[i] = intercept + slope * prepared.predictor[i]
    end
    nothing
end

@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLNormalFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    scale_factor = prepared.plan.factors.scale_prior
    coefficient_factor = prepared.plan.factors.coefficient_prior
    coefficient_parameter = prepared.plan.parameters.coefficients
    scale_parameter = prepared.plan.parameters.scale
    density = _native_ppl_factor_logdensity(
        coefficient_factor, coefficient_parameter, position)
    density += _native_ppl_factor_logdensity(
        scale_factor, scale_parameter, position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, scale_parameter, position, prepared,
        buffers)
    density
end


@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLBernoulliLogitFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    density = _native_ppl_factor_logdensity(
        prepared.plan.factors.coefficient_prior,
        prepared.plan.parameters.coefficients,
        position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.location, position, prepared, buffers)
    density
end

@inline function _native_ppl_model_logdensity!(
    likelihood_factor::NativePPLPoissonFactor,
    position::AbstractVector{T},
    prepared::NativePPLPrepared,
    buffers::NativePPLBuffers{T},
) where {T}
    density = _native_ppl_factor_logdensity(
        prepared.plan.factors.coefficient_prior,
        prepared.plan.parameters.coefficients,
        position)
    density += _native_ppl_factor_logdensity!(
        likelihood_factor, prepared.plan.inputs.response,
        prepared.plan.nodes.rate, prepared.plan.nodes.location,
        position, prepared, buffers)
    density
end


@inline function _native_ppl_logdensity_kernel(position::AbstractVector{T},
                                               prepared::NativePPLPrepared,
                                               buffers::NativePPLBuffers{T}) where {T}
    _native_ppl_location_kernel!(position, prepared, buffers)
    _native_ppl_model_logdensity!(
        prepared.plan.factors.likelihood, position, prepared, buffers)
end

function _native_ppl_location!(workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector)
    _native_ppl_check_location_execution(workspace, prepared, position)
    _native_ppl_location_kernel!(position, prepared, workspace.primal)
    workspace.primal.location
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

# DifferentiationInterface is optional. Its extension installs the concrete
# typed method and owns derivative preparation. AutoEnzyme is the sole backend
# this package tests, recommends, or guarantees; other DI backends remain
# outside the compatibility contract.
function _native_ppl_logdensity_and_gradient! end
function _native_ppl_logdensity_and_gradient!(workspace::NativePPLWorkspace,
                                              prepared::NativePPLPrepared,
                                              position::AbstractVector)
    _native_ppl_require_response(prepared, "gradient")
    throw(ArgumentError(
        "native PPL gradients require a DifferentiationInterface-prepared workspace; " *
        "construct one with AutoEnzyme() for the supported path"))
end
_native_ppl_logdensity_and_gradient!(args...) = throw(ArgumentError(
    "native PPL gradients require a DifferentiationInterface-prepared workspace; " *
    "construct one with AutoEnzyme() for the supported path"))

function _native_ppl_required_binding(bindings, name::Symbol, role::Symbol)
    hasproperty(bindings, name) || throw(ArgumentError(
        "native PPL rebind is missing required $role input `$name`"))
    value = getproperty(bindings, name)
    value isa AbstractVector || throw(ArgumentError(
        "native PPL $role input `$name` must be an AbstractVector; got $(typeof(value))"))
    value
end


function _native_ppl_response_binding(bindings, name::Symbol)
    hasproperty(bindings, name) || return NativePPLNoResponse()
    _native_ppl_required_binding(bindings, name, :response)
end

_native_ppl_rebind_likelihood(
    ::NativePPLNormalFactor{Response,Location,Scale}, axis,
) where {Response,Location,Scale} =
    NativePPLNormalFactor(Response, Location, Scale, axis)
_native_ppl_rebind_likelihood(
    ::NativePPLBernoulliLogitFactor{Response,Logit}, axis,
) where {Response,Logit} =
    NativePPLBernoulliLogitFactor(Response, Logit, axis)
_native_ppl_rebind_likelihood(
    ::NativePPLPoissonFactor{Response,Rate}, axis,
) where {Response,Rate} =
    NativePPLPoissonFactor(Response, Rate, axis)

function _native_ppl_rebind_transform(
    old_transform::NativePPLCenterNode,
    observation_axis,
    predictor_name::Symbol,
    predictor,
    freeze_constants::Bool,
)
    mean = freeze_constants ? old_transform.mean :
        _native_ppl_fit_center(predictor, predictor_name)
    NativePPLCenterNode(
        native_node_name(old_transform), predictor_name,
        observation_axis, mean)
end

function _native_ppl_rebind_transform(
    old_transform::NativePPLZScaleNode,
    observation_axis,
    predictor_name::Symbol,
    predictor,
    freeze_constants::Bool,
)
    fit = freeze_constants ?
        (; mean=old_transform.mean, scale=old_transform.scale) :
        _native_ppl_fit_zscale(predictor, predictor_name)
    NativePPLZScaleNode(
        native_node_name(old_transform), predictor_name,
        observation_axis, fit.mean, fit.scale)
end

function _native_ppl_rebind_nodes(plan::NativePPLPlan, observation_axis,
                                  predictor_name::Symbol, predictor,
                                  freeze_constants::Bool)
    old_location = plan.nodes.location
    location_name = native_node_name(old_location)
    transform = if hasproperty(plan.nodes, :transform)
        old_transform = plan.nodes.transform
        native_fitted_transform_input(old_transform) === predictor_name ||
            throw(NativePPLCapabilityError(
                :graph_identity,
                "rebound fitted transform must consume the compiled raw predictor"))
        _native_ppl_rebind_transform(
            old_transform, observation_axis, predictor_name,
            predictor, freeze_constants)
    else
        nothing
    end
    location_input = transform === nothing ?
        predictor_name : native_node_name(transform)
    native_affine_input(old_location) === location_input ||
        throw(NativePPLCapabilityError(
            :graph_identity,
            "rebound affine node must consume the compiled predictor transform"))
    location = NativePPLAffineNode(
        location_name, location_input, observation_axis,
        old_location.intercept_index, old_location.slope_index)
    nodes = transform === nothing ? (; location) : (; transform, location)
    hasproperty(plan.nodes, :rate) || return nodes

    old_rate = plan.nodes.rate
    native_exp_input(old_rate) === location_name || throw(NativePPLCapabilityError(
        :graph_identity,
        "rebound exponential node must consume the compiled affine predictor"))
    rate = NativePPLExpNode(
        native_node_name(old_rate), location_name, observation_axis)
    merge(nodes, (; rate))
end

"""
    _native_ppl_rebind(prepared, bindings;
                       T=eltype(prepared), freeze_constants=true)

Rebind the same graph semantics to compatible predictor and optional response
vectors. Omitting the response creates an explicit prediction-only prepared
value. The observation axis and every node/factor/query carrying it are rebuilt
from the new row count; parameter coordinates and semantic identities are reused.
Fitted preprocessing constants are reused by default. Pass
`freeze_constants=false` to refit them from the rebound predictor.
"""
function _native_ppl_rebind(prepared::NativePPLPrepared, bindings;
                            T::Type{<:AbstractFloat}=eltype(prepared),
                            freeze_constants::Bool=true)
    plan = prepared.plan
    predictor_name = native_input_name(plan.inputs.predictor)
    response_name = native_input_name(plan.inputs.response)
    predictor = _native_ppl_required_binding(bindings, predictor_name, :predictor)
    response = _native_ppl_response_binding(bindings, response_name)
    eltype(predictor) <: Real && !(eltype(predictor) <: Integer) ||
        throw(ArgumentError(
            "native PPL predictor `$predictor_name` must preserve the compiled " *
            "continuous real input role; got eltype $(eltype(predictor))"))
    if response isa AbstractVector
        eltype(response) <: Real || throw(ArgumentError(
            "native PPL response `$response_name` must preserve its compiled real input role; " *
            "got eltype $(eltype(response))"))
        length(predictor) == length(response) || throw(DimensionMismatch(
            "native PPL predictor `$predictor_name` has $(length(predictor)) rows but " *
            "response `$response_name` has $(length(response))"))
    end
    !isempty(predictor) || throw(DimensionMismatch(
        "native PPL rebound observation axis cannot be empty"))

    observation_axis = NativePPLAxis(:observation, Base.OneTo(length(predictor)))
    predictor_input = NativePPLInput(
        predictor_name, :predictor, observation_axis, eltype(predictor))
    response_eltype = response isa AbstractVector ?
        eltype(response) : eltype(plan.inputs.response)
    response_input = NativePPLInput(
        response_name, :response, observation_axis, response_eltype)

    nodes = _native_ppl_rebind_nodes(
        plan, observation_axis, predictor_name, predictor, freeze_constants)
    likelihood = _native_ppl_rebind_likelihood(
        plan.factors.likelihood, observation_axis)
    new_bindings = response isa AbstractVector ?
        NamedTuple{(predictor_name, response_name)}((predictor, response)) :
        NamedTuple{(predictor_name,)}((predictor,))

    rebound_plan = NativePPLPlan(
        merge(plan.axes, (; observation=observation_axis)),
        (; predictor=predictor_input, response=response_input),
        plan.parameters,
        nodes,
        merge(plan.factors, (; likelihood)),
        _native_ppl_queries(observation_axis, likelihood),
        new_bindings,
    )
    _native_ppl_prepare_bindings(rebound_plan, predictor, response, T)
end

_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLLinearPredictor) =
    plan.queries.linear_predictor
_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLPointwiseLogLikelihood) =
    plan.queries.pointwise_loglikelihood
_native_ppl_query_spec(plan::NativePPLPlan, ::NativePPLPosteriorPredictive) =
    plan.queries.posterior_predictive
_native_ppl_query_spec(prepared::NativePPLPrepared, query::NativePPLQuery) =
    _native_ppl_query_spec(prepared.plan, query)

function _native_ppl_check_batch_positions(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
)
    dimension = LogDensityProblems.dimension(prepared)
    size(positions, 2) == dimension || throw(DimensionMismatch(
        "native PPL draw matrix has $(size(positions, 2)) coordinate columns; " *
        "expected $dimension"))
    axes(positions, 1) == Base.OneTo(size(positions, 1)) ||
        throw(ArgumentError(
            "native PPL batch queries require a one-based draw axis; got " *
            "$(axes(positions, 1))"))
    axes(positions, 2) == Base.OneTo(dimension) || throw(ArgumentError(
        "native PPL batch queries require one-based coordinate columns; got " *
        "$(axes(positions, 2))"))
    nothing
end

function _native_ppl_batch_output_signature_unchecked(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    element_signature = native_query_output(
        _native_ppl_query_spec(prepared, query))
    NativePPLBatchOutputSignature(
        NativePPLAxis(:draw, Base.OneTo(size(positions, 1))),
        native_output_axis(element_signature),
        element_signature.element_type,
        NativePPLDenseMatrixLayout(),
    )
end

function _native_ppl_batch_output_signature(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_batch_output_signature_unchecked(prepared, positions, query)
end

function _native_ppl_named_map(f, values::NamedTuple{Names}) where {Names}
    NamedTuple{Names}(map(f, Tuple(values)))
end

function _native_ppl_check_bundle_queries(queries::NamedTuple)
    for (name, query) in pairs(queries)
        query isa NativePPLQuery || throw(ArgumentError(
            "native PPL bundle query `$name` must be a typed graph query; " *
            "got $(typeof(query))"))
    end
    nothing
end

function _native_ppl_batch_output_signature(
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple,
)
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_check_bundle_queries(queries)
    _native_ppl_named_map(
        query -> _native_ppl_batch_output_signature_unchecked(
            prepared, positions, query),
        queries)
end

function _native_ppl_allocate_output(signature::NativePPLOutputSignature,
                                     prepared::NativePPLPrepared)
    _native_ppl_allocate_output(
        native_output_layout(signature), signature, prepared)
end

function _native_ppl_allocate_output(::NativePPLDenseVectorLayout,
                                     signature::NativePPLOutputSignature,
                                     prepared::NativePPLPrepared)
    axis = native_output_axis(signature)
    axis.keys == Base.OneTo(length(axis)) || throw(NativePPLCapabilityError(
        :output_layout,
        "dense-vector allocation requires a one-based semantic axis; " *
        "got $(axis.keys)"))
    T = native_output_eltype(signature, prepared)
    Vector{T}(undef, length(axis))
end

function _native_ppl_allocate_output(signature::NativePPLBatchOutputSignature,
                                     prepared::NativePPLPrepared)
    _native_ppl_allocate_output(
        native_output_layout(signature), signature, prepared)
end

function _native_ppl_allocate_output(::NativePPLDenseMatrixLayout,
                                     signature::NativePPLBatchOutputSignature,
                                     prepared::NativePPLPrepared)
    draw_axis, observation_axis = native_output_axes(signature)
    draw_axis.keys == Base.OneTo(length(draw_axis)) ||
        throw(NativePPLCapabilityError(
            :output_layout,
            "dense-matrix allocation requires a one-based draw axis; got " *
            "$(draw_axis.keys)"))
    observation_axis.keys == Base.OneTo(length(observation_axis)) ||
        throw(NativePPLCapabilityError(
            :output_layout,
            "dense-matrix allocation requires a one-based observation axis; got " *
            "$(observation_axis.keys)"))
    T = native_output_eltype(signature, prepared)
    Matrix{T}(undef, length(draw_axis), length(observation_axis))
end


_native_ppl_allocate_output(signatures::NamedTuple,
                            prepared::NativePPLPrepared) =
    _native_ppl_named_map(
        signature -> _native_ppl_allocate_output(signature, prepared),
        signatures)

function _native_ppl_check_query_output(output::AbstractVector,
                                        prepared::NativePPLPrepared,
                                        query::NativePPLQuery)
    query_spec = _native_ppl_query_spec(prepared, query)
    query_name = native_query_name(query_spec)
    signature = native_query_output(query_spec)
    expected_axis = native_output_axis(signature).keys
    axes(output, 1) == expected_axis || throw(DimensionMismatch(
        "native PPL `$query_name` output axis $(axes(output, 1)) does not match " *
        "the declared observation axis $expected_axis"))
    expected_eltype = native_output_eltype(signature, prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL `$query_name` output eltype $(eltype(output)) does not match " *
        "declared output eltype $expected_eltype"))
    native_output_layout_accepts(native_output_layout(signature), output) ||
        throw(ArgumentError(
            "native PPL `$query_name` output $(typeof(output)) does not satisfy " *
            "declared layout $(typeof(native_output_layout(signature)))"))
    length(output) == length(expected_axis) || throw(DimensionMismatch(
        "native PPL `$query_name` output has length $(length(output)); expected " *
        "$(length(expected_axis)) for the observation axis"))
    output
end

function _native_ppl_check_batch_execution(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLQuery,
)
    signature = _native_ppl_batch_output_signature(prepared, positions, query)
    draw_axis, observation_axis = native_output_axes(signature)
    axes(output) == (draw_axis.keys, observation_axis.keys) ||
        throw(DimensionMismatch(
            "native PPL batch output axes $(axes(output)) do not match " *
            "declared draw-by-observation axes " *
            "$((draw_axis.keys, observation_axis.keys))"))
    expected_eltype = native_output_eltype(signature, prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL batch output eltype $(eltype(output)) does not match " *
        "declared output eltype $expected_eltype"))
    native_output_layout_accepts(native_output_layout(signature), output) ||
        throw(ArgumentError(
            "native PPL batch output $(typeof(output)) does not satisfy " *
            "declared layout $(typeof(native_output_layout(signature)))"))
    eltype(positions) === eltype(workspace) || throw(ArgumentError(
        "native PPL draw matrix eltype $(eltype(positions)) does not match " *
        "workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL prepared eltype $(eltype(prepared)) does not match " *
        "workspace eltype $(eltype(workspace))"))
    length(workspace.primal.location) == length(observation_axis) ||
        throw(DimensionMismatch(
            "native PPL workspace location has " *
            "$(length(workspace.primal.location)) rows; expected " *
            "$(length(observation_axis))"))
    Base.mightalias(output, positions) && throw(ArgumentError(
        "native PPL batch output must not alias the draw matrix"))
    Base.mightalias(positions, workspace.primal.location) && throw(ArgumentError(
        "native PPL draw matrix must not alias the workspace location buffer"))
    Base.mightalias(positions, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL draw matrix must not alias the workspace pointwise buffer"))
    Base.mightalias(positions, workspace.gradient) && throw(ArgumentError(
        "native PPL draw matrix must not alias the workspace gradient buffer"))
    Base.mightalias(output, workspace.primal.location) && throw(ArgumentError(
        "native PPL batch output must not alias the workspace location buffer"))
    Base.mightalias(output, workspace.primal.pointwise_loglikelihood) &&
        throw(ArgumentError(
            "native PPL batch output must not alias the workspace pointwise buffer"))
    Base.mightalias(output, workspace.gradient) && throw(ArgumentError(
        "native PPL batch output must not alias the workspace gradient buffer"))
    signature
end

_native_ppl_check_batch_query_state(
    ::NativePPLWorkspace, ::NativePPLPrepared, ::NativePPLLinearPredictor,
) = nothing
_native_ppl_check_batch_query_state(
    ::NativePPLWorkspace, ::NativePPLPrepared, ::NativePPLPosteriorPredictive,
) = nothing
function _native_ppl_check_batch_query_state(
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    ::NativePPLPointwiseLogLikelihood,
)
    response = _native_ppl_require_response(
        prepared, "batched pointwise log likelihood")
    observations = length(prepared.plan.axes.observation)
    length(response) == observations || throw(DimensionMismatch(
        "native PPL prepared response has $(length(response)) rows; expected " *
        "$observations"))
    length(workspace.primal.pointwise_loglikelihood) == observations ||
        throw(DimensionMismatch(
            "native PPL workspace pointwise likelihood has " *
            "$(length(workspace.primal.pointwise_loglikelihood)) rows; expected " *
            "$observations"))
    length(workspace.gradient) == LogDensityProblems.dimension(prepared) ||
        throw(DimensionMismatch(
            "native PPL workspace gradient has $(length(workspace.gradient)) " *
            "coordinates; expected $(LogDensityProblems.dimension(prepared))"))
    nothing
end

function _native_ppl_evaluate!(output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLLinearPredictor)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_location!(workspace, prepared, position)
    copyto!(output, workspace.primal.location)
end

function _native_ppl_evaluate!(output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLPointwiseLogLikelihood)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_require_response(prepared, "pointwise log likelihood")
    _native_ppl_logdensity!(workspace, prepared, position)
    copyto!(output, workspace.primal.pointwise_loglikelihood)
end

function _native_ppl_simulate!(rng::AbstractRNG,
                               output::AbstractVector,
                               workspace::NativePPLWorkspace,
                               prepared::NativePPLPrepared,
                               position::AbstractVector,
                               query::NativePPLPosteriorPredictive)
    _native_ppl_check_query_output(output, prepared, query)
    _native_ppl_location!(workspace, prepared, position)
    _native_ppl_model_simulate!(
        rng, prepared.plan.factors.likelihood, output,
        prepared, position, workspace.primal)
end

function _native_ppl_evaluate_draws!(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLLinearPredictor,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        _native_ppl_location!(
            workspace, prepared, @view(positions[draw, :]))
        for observation in axes(output, 2)
            @inbounds output[draw, observation] =
                workspace.primal.location[observation]
        end
    end
    output
end

function _native_ppl_evaluate_draws!(
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLPointwiseLogLikelihood,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        _native_ppl_logdensity!(
            workspace, prepared, @view(positions[draw, :]))
        for observation in axes(output, 2)
            @inbounds output[draw, observation] =
                workspace.primal.pointwise_loglikelihood[observation]
        end
    end
    output
end

function _native_ppl_simulate_draws!(
    rng::AbstractRNG,
    output::AbstractMatrix,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    query::NativePPLPosteriorPredictive,
)
    _native_ppl_check_batch_execution(
        output, workspace, prepared, positions, query)
    _native_ppl_check_batch_query_state(workspace, prepared, query)
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _native_ppl_location!(workspace, prepared, position)
        _native_ppl_model_simulate!(
            rng, prepared.plan.factors.likelihood,
            @view(output[draw, :]), prepared, position, workspace.primal)
    end
    output
end

function _native_ppl_check_bundle_outputs(
    outputs::NamedTuple{Names},
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple{Names},
) where {Names}
    _native_ppl_check_batch_positions(prepared, positions)
    _native_ppl_check_bundle_queries(queries)
    output_values = Tuple(outputs)
    query_values = Tuple(queries)
    for i in eachindex(output_values)
        output_values[i] isa AbstractMatrix || throw(ArgumentError(
            "native PPL bundle output `$((Names[i]))` must be an " *
            "AbstractMatrix; got $(typeof(output_values[i]))"))
        _native_ppl_check_batch_execution(
            output_values[i], workspace, prepared, positions, query_values[i])
        _native_ppl_check_batch_query_state(
            workspace, prepared, query_values[i])
    end
    for i in eachindex(output_values), j in (i + 1):length(output_values)
        Base.mightalias(output_values[i], output_values[j]) &&
            throw(ArgumentError(
                "native PPL bundle outputs `$((Names[i]))` and `$((Names[j]))` " *
                "must not alias"))
    end
    nothing
end

function _native_ppl_check_bundle_outputs(
    outputs::NamedTuple,
    ::NativePPLWorkspace,
    ::NativePPLPrepared,
    ::AbstractMatrix,
    queries::NamedTuple,
)
    throw(ArgumentError(
        "native PPL bundle output keys $(keys(outputs)) do not match " *
        "query keys $(keys(queries))"))
end

function _native_ppl_bundle_requires_rng(prepared::NativePPLPrepared,
                                         queries::NamedTuple)
    _native_ppl_check_bundle_queries(queries)
    for query in Tuple(queries)
        native_query_effect(_native_ppl_query_spec(prepared, query)) === :rng &&
            return true
    end
    false
end

function _native_ppl_bundle_has_pointwise(queries::NamedTuple)
    for query in Tuple(queries)
        query isa NativePPLPointwiseLogLikelihood && return true
    end
    false
end

@inline function _native_ppl_write_bundle_query!(
    rng,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLLinearPredictor,
)
    for observation in axes(output, 2)
        @inbounds output[draw, observation] =
            workspace.primal.location[observation]
    end
    nothing
end

@inline function _native_ppl_write_bundle_query!(
    rng,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLPointwiseLogLikelihood,
)
    for observation in axes(output, 2)
        @inbounds output[draw, observation] =
            workspace.primal.pointwise_loglikelihood[observation]
    end
    nothing
end

@inline function _native_ppl_write_bundle_query!(
    rng::AbstractRNG,
    output::AbstractMatrix,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
    ::NativePPLPosteriorPredictive,
)
    _native_ppl_model_simulate!(
        rng, prepared.plan.factors.likelihood, @view(output[draw, :]),
        prepared, position, workspace.primal)
    nothing
end

@inline _native_ppl_write_bundle_queries!(
    rng, ::Tuple{}, ::Tuple{}, draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
) = nothing

@inline function _native_ppl_write_bundle_queries!(
    rng,
    outputs::Tuple,
    queries::Tuple,
    draw::Int,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    position::AbstractVector,
)
    _native_ppl_write_bundle_query!(
        rng, first(outputs), draw, workspace, prepared, position,
        first(queries))
    _native_ppl_write_bundle_queries!(
        rng, Base.tail(outputs), Base.tail(queries), draw,
        workspace, prepared, position)
end

function _native_ppl_execute_draws!(
    rng,
    outputs::NamedTuple,
    workspace::NativePPLWorkspace,
    prepared::NativePPLPrepared,
    positions::AbstractMatrix,
    queries::NamedTuple,
)
    rng === nothing && _native_ppl_bundle_requires_rng(prepared, queries) &&
        throw(ArgumentError(
            "native PPL query bundle contains an RNG-effect query; " *
            "call `execute_draws!` with an explicit RNG"))
    _native_ppl_check_bundle_outputs(
        outputs, workspace, prepared, positions, queries)
    isempty(queries) && return outputs

    pointwise = _native_ppl_bundle_has_pointwise(queries)
    output_values = Tuple(outputs)
    query_values = Tuple(queries)
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _native_ppl_location!(workspace, prepared, position)
        pointwise && _native_ppl_model_logdensity!(
            prepared.plan.factors.likelihood, position, prepared,
            workspace.primal)
        _native_ppl_write_bundle_queries!(
            rng, output_values, query_values, draw,
            workspace, prepared, position)
    end
    outputs
end

"""
Namespaced walking-skeleton API for the native PPL.

`DifferentiationInterface.AutoEnzyme()` is the sole tested, recommended, and
guaranteed derivative backend. Other DI backends are outside this prototype's
compatibility contract even when they happen to execute.
"""
module NativePPL

const BRM = parentmodule(@__MODULE__)

const Plan = BRM.NativePPLPlan
const Prepared = BRM.NativePPLPrepared
const Workspace = BRM.NativePPLWorkspace
const CapabilityError = BRM.NativePPLCapabilityError
const OutputSignature = BRM.NativePPLOutputSignature
const BatchOutputSignature = BRM.NativePPLBatchOutputSignature
const DenseVectorLayout = BRM.NativePPLDenseVectorLayout
const DenseMatrixLayout = BRM.NativePPLDenseMatrixLayout
const LinearPredictor = BRM.NativePPLLinearPredictor
const PointwiseLogLikelihood = BRM.NativePPLPointwiseLogLikelihood
const PosteriorPredictive = BRM.NativePPLPosteriorPredictive

compile(brmi::BRM.BRMI) = BRM._native_ppl_plan(brmi)
prepare(plan::Plan; kwargs...) = BRM._native_ppl_prepare(plan; kwargs...)
workspace(prepared::Prepared, ::Type{T}=eltype(prepared)) where {T<:AbstractFloat} =
    BRM._native_ppl_workspace(prepared, T)
workspace(prepared::Prepared, ::Type{T}, backend) where {T<:AbstractFloat} =
    BRM._native_ppl_workspace(prepared, T, backend)
rebind(prepared::Prepared, bindings; kwargs...) =
    BRM._native_ppl_rebind(prepared, bindings; kwargs...)
has_response(prepared::Prepared) = BRM.native_ppl_has_response(prepared)
output_signature(plan::Plan, query::BRM.NativePPLQuery) =
    BRM.native_query_output(BRM._native_ppl_query_spec(plan, query))
output_signature(prepared::Prepared, query::BRM.NativePPLQuery) =
    output_signature(prepared.plan, query)
batch_output_signature(prepared::Prepared, positions::AbstractMatrix,
                       query::BRM.NativePPLQuery) =
    BRM._native_ppl_batch_output_signature(prepared, positions, query)
batch_output_signature(prepared::Prepared, positions::AbstractMatrix,
                       queries::NamedTuple) =
    BRM._native_ppl_batch_output_signature(prepared, positions, queries)
output_axis(signature::OutputSignature) = BRM.native_output_axis(signature)
output_axis(signature::BatchOutputSignature) = BRM.native_output_axis(signature)
output_axes(signature::Union{OutputSignature,BatchOutputSignature}) =
    BRM.native_output_axes(signature)
output_draw_axis(signature::BatchOutputSignature) =
    BRM.native_output_draw_axis(signature)
output_eltype(signature::OutputSignature, prepared::Prepared) =
    BRM.native_output_eltype(signature, prepared)
output_eltype(signature::BatchOutputSignature, prepared::Prepared) =
    BRM.native_output_eltype(signature, prepared)
output_eltype(signature::OutputSignature, ::Type{T}) where {T<:AbstractFloat} =
    BRM.native_output_eltype(signature, T)
output_eltype(signature::BatchOutputSignature, ::Type{T}) where {T<:AbstractFloat} =
    BRM.native_output_eltype(signature, T)
output_layout(signature::OutputSignature) = BRM.native_output_layout(signature)
output_layout(signature::BatchOutputSignature) = BRM.native_output_layout(signature)
allocate_output(signature::OutputSignature, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signature, prepared)
allocate_output(signature::BatchOutputSignature, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signature, prepared)
allocate_output(signatures::NamedTuple, prepared::Prepared) =
    BRM._native_ppl_allocate_output(signatures, prepared)
allocate_output(prepared::Prepared, query::BRM.NativePPLQuery) =
    allocate_output(output_signature(prepared, query), prepared)
logdensity!(work::Workspace, prepared::Prepared, position::AbstractVector) =
    BRM._native_ppl_logdensity!(work, prepared, position)
logdensity_and_gradient!(work::Workspace, prepared::Prepared,
                         position::AbstractVector) =
    BRM._native_ppl_logdensity_and_gradient!(work, prepared, position)
evaluate!(output::AbstractVector, work::Workspace, prepared::Prepared,
          position::AbstractVector, query::BRM.NativePPLQuery) =
    BRM._native_ppl_evaluate!(output, work, prepared, position, query)
function evaluate(work::Workspace, prepared::Prepared,
                  position::AbstractVector, query::BRM.NativePPLQuery)
    output = allocate_output(prepared, query)
    evaluate!(output, work, prepared, position, query)
end
simulate!(rng::BRM.AbstractRNG, output::AbstractVector, work::Workspace,
          prepared::Prepared, position::AbstractVector,
          query::BRM.NativePPLQuery=PosteriorPredictive()) =
    BRM._native_ppl_simulate!(rng, output, work, prepared, position, query)
function simulate(rng::BRM.AbstractRNG, work::Workspace, prepared::Prepared,
                  position::AbstractVector,
                  query::BRM.NativePPLQuery=PosteriorPredictive())
    output = allocate_output(prepared, query)
    simulate!(rng, output, work, prepared, position, query)
end
evaluate_draws!(output::AbstractMatrix, work::Workspace, prepared::Prepared,
                positions::AbstractMatrix, query::BRM.NativePPLQuery) =
    BRM._native_ppl_evaluate_draws!(
        output, work, prepared, positions, query)
function evaluate_draws(work::Workspace, prepared::Prepared,
                        positions::AbstractMatrix,
                        query::BRM.NativePPLQuery)
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    evaluate_draws!(output, work, prepared, positions, query)
end
simulate_draws!(rng::BRM.AbstractRNG, output::AbstractMatrix,
                work::Workspace, prepared::Prepared,
                positions::AbstractMatrix,
                query::BRM.NativePPLQuery=PosteriorPredictive()) =
    BRM._native_ppl_simulate_draws!(
        rng, output, work, prepared, positions, query)
function simulate_draws(rng::BRM.AbstractRNG, work::Workspace,
                        prepared::Prepared, positions::AbstractMatrix,
                        query::BRM.NativePPLQuery=PosteriorPredictive())
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    simulate_draws!(rng, output, work, prepared, positions, query)
end
execute_draws!(outputs::NamedTuple, work::Workspace, prepared::Prepared,
               positions::AbstractMatrix, queries::NamedTuple) =
    BRM._native_ppl_execute_draws!(
        nothing, outputs, work, prepared, positions, queries)
execute_draws!(rng::BRM.AbstractRNG, outputs::NamedTuple, work::Workspace,
               prepared::Prepared, positions::AbstractMatrix,
               queries::NamedTuple) =
    BRM._native_ppl_execute_draws!(
        rng, outputs, work, prepared, positions, queries)
function execute_draws(work::Workspace, prepared::Prepared,
                       positions::AbstractMatrix, queries::NamedTuple)
    BRM._native_ppl_bundle_requires_rng(prepared, queries) &&
        throw(ArgumentError(
            "native PPL query bundle contains an RNG-effect query; " *
            "call `execute_draws` with an explicit RNG"))
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(outputs, work, prepared, positions, queries)
end
function execute_draws(rng::BRM.AbstractRNG, work::Workspace,
                       prepared::Prepared, positions::AbstractMatrix,
                       queries::NamedTuple)
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(rng, outputs, work, prepared, positions, queries)
end

include("native_ppl_authoring.jl")

export Plan, Prepared, Workspace, CapabilityError
export OutputSignature, BatchOutputSignature
export DenseVectorLayout, DenseMatrixLayout
export LinearPredictor, PointwiseLogLikelihood, PosteriorPredictive
export compile, prepare, workspace, rebind, has_response
export output_signature, batch_output_signature
export output_axis, output_axes, output_draw_axis, output_eltype, output_layout
export allocate_output, evaluate, simulate
export evaluate_draws, simulate_draws, execute_draws
export logdensity!, logdensity_and_gradient!, evaluate!, simulate!
export evaluate_draws!, simulate_draws!, execute_draws!

end
