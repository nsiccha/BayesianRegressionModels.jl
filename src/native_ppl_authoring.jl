"""
Public, unbound declarations for the native PPL graph.

These values describe logical model semantics without training data, fitted
preprocessing state, flat coordinates, or workspaces. Direct Julia authors and
BRM lowerings construct the same `Model`; binding/compilation turns it into the
existing executable `Plan`.
"""

abstract type AbstractInputDeclaration end
abstract type AbstractPriorDeclaration end
abstract type AbstractParameterDeclaration end
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
const CholeskyCorrelationSupport =
    BRM.NativePPLCholeskyCorrelationSupport
const IdentityTransform = BRM.NativePPLIdentityTransform
const ExpTransform = BRM.NativePPLExpTransform
const CholeskyCorrelationTransform =
    BRM.NativePPLCholeskyCorrelationTransform

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

function normal_prior(location::Real, scale::Real)
    isfinite(location) || throw(ArgumentError(
        "native PPL Normal prior location must be finite"))
    isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
        "native PPL Normal prior scale must be finite and positive"))
    NormalPrior(location, scale)
end

"""Exponential prior declaration parameterized by its positive scale."""
struct ExponentialPrior{T<:Real} <: AbstractPriorDeclaration
    scale::T

    function ExponentialPrior(scale::T) where {T<:Real}
        isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
            "native PPL Exponential prior scale must be finite and positive"))
        new{T}(scale)
    end
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
struct Parameter{S,Tr,K,P<:AbstractPriorDeclaration} <:
       AbstractParameterDeclaration
    support::S
    transform::Tr
    axis_keys::K
    prior::P
end

"""A group-indexed Normal latent block whose axis is fitted from one input."""
struct GroupedNormalParameter{Group,L,S} <: AbstractParameterDeclaration
    location::L
    scale::S
end

"""Group-major independent standard-normal coordinates for named coefficients."""
struct GroupedStandardNormalParameter{Group,Coefficients} <:
       AbstractParameterDeclaration end

function grouped_standard_normal(group::Symbol, coefficients::Tuple)
    isempty(coefficients) && throw(ArgumentError(
        "native PPL grouped standard-normal coefficients cannot be empty"))
    all(coefficient -> coefficient isa Symbol, coefficients) || throw(
        ArgumentError(
            "native PPL grouped standard-normal coefficient keys must be Symbols"))
    length(unique(coefficients)) == length(coefficients) || throw(ArgumentError(
        "native PPL grouped standard-normal coefficient keys must be unique"))
    GroupedStandardNormalParameter{group,coefficients}()
end

group_input(::GroupedStandardNormalParameter{Group}) where {Group} = Group
group_coefficients(
    ::GroupedStandardNormalParameter{Group,Coefficients},
) where {Group,Coefficients} = Coefficients

"""One LKJ prior over a named Cholesky correlation factor."""
struct CholeskyCorrelationParameter{Coefficients,Eta} <:
       AbstractParameterDeclaration
    eta::Eta
end

function cholesky_correlation(coefficients::Tuple, eta::Real)
    length(coefficients) >= 2 || throw(ArgumentError(
        "native PPL Cholesky correlation needs at least two coefficients"))
    all(coefficient -> coefficient isa Symbol, coefficients) || throw(
        ArgumentError(
            "native PPL Cholesky correlation coefficient keys must be Symbols"))
    length(unique(coefficients)) == length(coefficients) || throw(ArgumentError(
        "native PPL Cholesky correlation coefficient keys must be unique"))
    isfinite(eta) && eta > zero(eta) || throw(ArgumentError(
        "native PPL LKJ shape must be finite and positive"))
    CholeskyCorrelationParameter{coefficients,typeof(eta)}(eta)
end

correlation_coefficients(
    ::CholeskyCorrelationParameter{Coefficients},
) where {Coefficients} = Coefficients

_factor_declaration_value(value::Symbol) = SiteValue{value}()
_factor_declaration_value(value::Real) = LiteralValue(value)

function grouped_normal(group::Symbol, location::Union{Symbol,Real},
                        scale::Union{Symbol,Real})
    location isa Real && !isfinite(location) && throw(ArgumentError(
        "native PPL grouped-Normal location must be finite"))
    scale isa Real && !(isfinite(scale) && scale > zero(scale)) && throw(
        ArgumentError(
            "native PPL grouped-Normal scale must be finite and positive"))
    GroupedNormalParameter{group,
        typeof(_factor_declaration_value(location)),
        typeof(_factor_declaration_value(scale))}(
            _factor_declaration_value(location),
            _factor_declaration_value(scale))
end

group_input(::GroupedNormalParameter{Group}) where {Group} = Group

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

"""Intercept plus one slope per named input/node and optional additive offsets."""
struct Affine{Inputs,Coefficients,Offsets,Intercept} <: AbstractNodeDeclaration end
function affine(inputs::Tuple, coefficients::Symbol; offsets::Tuple=(),
                intercept::Bool=true)
    isempty(inputs) && !intercept && isempty(offsets) && throw(ArgumentError(
        "native PPL affine declaration requires an input, intercept, or offset"))
    all(input -> input isa Symbol, inputs) || throw(ArgumentError(
        "native PPL affine inputs must be named Symbols; got $inputs"))
    length(unique(inputs)) == length(inputs) || throw(ArgumentError(
        "native PPL affine inputs must be unique; got $inputs"))
    all(offset -> offset isa Symbol, offsets) || throw(ArgumentError(
        "native PPL affine offsets must be named Symbols; got $offsets"))
    length(unique(offsets)) == length(offsets) || throw(ArgumentError(
        "native PPL affine offsets must be unique; got $offsets"))
    isempty(intersect(Set(inputs), Set(offsets))) || throw(ArgumentError(
        "native PPL affine inputs and offsets must have distinct names"))
    Affine{inputs,coefficients,offsets,intercept}()
end
affine(input::Symbol, coefficients::Symbol; offsets::Tuple=(),
       intercept::Bool=true) =
    affine((input,), coefficients; offsets, intercept)

"""Elementwise exponential link over one named deterministic node."""
struct ExpLink{Input} <: AbstractNodeDeclaration end
exp_link(input::Symbol) = ExpLink{input}()

"""Elementwise natural logarithm over one positive named row value."""
struct LogLink{Input} <: AbstractNodeDeclaration end
log_link(input::Symbol) = LogLink{input}()

"""Elementwise logistic link over one named deterministic node."""
struct LogisticLink{Input} <: AbstractNodeDeclaration end
logistic_link(input::Symbol) = LogisticLink{input}()

"""Gather a group-indexed latent block onto the observation-row axis."""
struct GroupGather{Values,Group} <: AbstractNodeDeclaration end
group_gather(values::Symbol, group::Symbol) = GroupGather{values,group}()

"""Multiply two named scalar-or-row values on the observation-row axis."""
struct RowProduct{Left,Right} <: AbstractNodeDeclaration end
row_product(left::Symbol, right::Symbol) = RowProduct{left,right}()

"""Apply selected columns of one correlated group-coefficient block on each row."""
struct GroupedAffine{
        Standardized,Scales,Correlation,Group,Predictors,
        CoefficientIndices,CoefficientCount} <:
       AbstractNodeDeclaration end

function grouped_affine(standardized::Symbol, scales::Symbol,
                        correlation::Symbol, group::Symbol,
                        predictors::Tuple;
                        coefficient_indices::Union{Nothing,Tuple}=nothing,
                        coefficient_count::Union{Nothing,Integer}=nothing)
    isempty(predictors) && throw(ArgumentError(
        "native PPL grouped affine predictors cannot be empty"))
    all(predictor -> predictor === nothing || predictor isa Symbol,
        predictors) || throw(ArgumentError(
        "native PPL grouped affine predictors must be Symbols or `nothing` " *
        "for an intercept"))
    count(isnothing, predictors) <= 1 || throw(ArgumentError(
        "native PPL grouped affine predictors may contain at most one intercept"))
    named = Tuple(predictor for predictor in predictors
                  if predictor !== nothing)
    length(unique(named)) == length(named) || throw(ArgumentError(
        "native PPL grouped affine predictors must be unique"))
    indices = coefficient_indices === nothing ?
        ntuple(identity, length(predictors)) : coefficient_indices
    length(indices) == length(predictors) || throw(ArgumentError(
        "native PPL grouped affine coefficient selection must have one " *
        "index per predictor"))
    all(index -> index isa Integer && index > 0, indices) || throw(
        ArgumentError(
            "native PPL grouped affine coefficient indices must be positive " *
            "integers"))
    length(unique(indices)) == length(indices) || throw(ArgumentError(
        "native PPL grouped affine coefficient indices must be unique"))
    total = coefficient_count === nothing ? length(predictors) :
        Int(coefficient_count)
    total >= 2 || throw(ArgumentError(
        "native PPL grouped affine coefficient count must be at least two"))
    maximum(indices) <= total || throw(ArgumentError(
        "native PPL grouped affine coefficient selection exceeds the full " *
        "coefficient count"))
    GroupedAffine{
        standardized,scales,correlation,group,predictors,indices,total}()
end

node_input(::Center{Input}) where {Input} = Input
node_input(::ZScale{Input}) where {Input} = Input
node_inputs(::Affine{Inputs}) where {Inputs} = Inputs
affine_offsets(::Affine{Inputs,Coefficients,Offsets}) where {
    Inputs,Coefficients,Offsets} = Offsets
affine_has_intercept(::Affine{Inputs,Coefficients,Offsets,Intercept}) where {
    Inputs,Coefficients,Offsets,Intercept} = Intercept
function node_input(node::Affine)
    inputs = node_inputs(node)
    length(inputs) == 1 || throw(ArgumentError(
        "native PPL affine declaration has $(length(inputs)) inputs; use " *
        "`node_inputs`"))
    only(inputs)
end
node_input(::ExpLink{Input}) where {Input} = Input
node_input(::LogLink{Input}) where {Input} = Input
node_input(::LogisticLink{Input}) where {Input} = Input
group_values(::GroupGather{Values}) where {Values} = Values
group_input(::GroupGather{Values,Group}) where {Values,Group} = Group
row_product_inputs(::RowProduct{Left,Right}) where {Left,Right} = (Left, Right)
grouped_standardized(
    ::GroupedAffine{Standardized},
) where {Standardized} = Standardized
grouped_scales(
    ::GroupedAffine{Standardized,Scales},
) where {Standardized,Scales} = Scales
grouped_correlation(
    ::GroupedAffine{Standardized,Scales,Correlation},
) where {Standardized,Scales,Correlation} = Correlation
group_input(
    ::GroupedAffine{Standardized,Scales,Correlation,Group},
) where {Standardized,Scales,Correlation,Group} = Group
grouped_predictors(
    ::GroupedAffine{Standardized,Scales,Correlation,Group,Predictors},
) where {Standardized,Scales,Correlation,Group,Predictors} = Predictors
grouped_coefficient_indices(
    ::GroupedAffine{Standardized,Scales,Correlation,Group,Predictors,Indices},
) where {Standardized,Scales,Correlation,Group,Predictors,Indices} = Indices
grouped_coefficient_count(
    ::GroupedAffine{
        Standardized,Scales,Correlation,Group,Predictors,Indices,Count},
) where {Standardized,Scales,Correlation,Group,Predictors,Indices,Count} = Count
affine_parameter(::Affine{Inputs,Coefficients}) where {Inputs,Coefficients} =
    Coefficients

"""One scalar Normal stochastic-site declaration."""
struct NormalObservation{Response,Location,Scale} <:
       AbstractObservationDeclaration end
normal(response::Symbol, location::Symbol, scale::Symbol) =
    NormalObservation{response,location,scale}()

"""One location-scale Student-t site with positive degrees of freedom."""
struct StudentTObservation{Response,Df,Location,Scale} <:
       AbstractObservationDeclaration end
student_t(response::Symbol, degrees_freedom::Symbol,
          location::Symbol, scale::Symbol) =
    StudentTObservation{response,degrees_freedom,location,scale}()

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

"""One zero-inflated Poisson site with positive rate and inflation probability."""
struct ZeroInflatedPoissonObservation{Response,Rate,ZeroInflation} <:
       AbstractObservationDeclaration end
zero_inflated_poisson(response::Symbol, rate::Symbol,
                      zero_inflation::Symbol) =
    ZeroInflatedPoissonObservation{response,rate,zero_inflation}()

"""One NegativeBinomial2 site parameterized by positive mean and precision."""
struct NegativeBinomial2Observation{Response,Mean,Precision} <:
       AbstractObservationDeclaration end
negative_binomial2(response::Symbol, mean::Symbol, precision::Symbol) =
    NegativeBinomial2Observation{response,mean,precision}()

"""One BetaBinomial2 site with trials, mean probability, and precision."""
struct BetaBinomial2Observation{Response,Trials,Mean,Precision} <:
       AbstractObservationDeclaration end
beta_binomial2(response::Symbol, trials::Symbol, mean::Symbol,
               precision::Symbol) =
    BetaBinomial2Observation{response,trials,mean,precision}()

"""One BinomialLogit site with row-varying trials and success logit."""
struct BinomialLogitObservation{Response,Trials,Logit} <:
       AbstractObservationDeclaration end
binomial_logit(response::Symbol, trials::Symbol, logit::Symbol) =
    BinomialLogitObservation{response,trials,logit}()

"""One categorical-logit site with class 1 fixed at zero."""
struct CategoricalLogitObservation{Response,Logits} <:
       AbstractObservationDeclaration end
function categorical_logit(response::Symbol, logits::Vararg{Symbol})
    isempty(logits) && throw(ArgumentError(
        "native PPL CategoricalLogit needs at least one non-reference logit"))
    CategoricalLogitObservation{response,logits}()
end

"""Typed observation-weight semantics retained in the public PPL graph."""
abstract type AbstractObservationWeight end

struct ObservationWeight{Kind,Source} <: AbstractObservationWeight end

function observation_weight(kind::Symbol, source::Symbol)
    kind in (:analytic, :frequency, :power, :unit) || throw(ArgumentError(
        "native PPL observation-weight kind must be :analytic, :frequency, " *
        ":power, or :unit; got $kind"))
    ObservationWeight{kind,source}()
end

observation_weight_kind(::ObservationWeight{Kind}) where {Kind} = Kind
observation_weight_source(
    ::ObservationWeight{Kind,Source}) where {Kind,Source} = Source

"""One base observation decorated with typed observation-weight semantics."""
struct WeightedObservation{
        O<:AbstractObservationDeclaration,
        W<:AbstractObservationWeight,
    } <: AbstractObservationDeclaration
    observation::O
    weight::W
end

weighted_observation(observation::AbstractObservationDeclaration,
                     weight::AbstractObservationWeight) =
    WeightedObservation(observation, weight)

"""Typed evidence semantics retained around one observation declaration."""
abstract type AbstractObservationEvidence end

struct TruncatedEvidence{L,U} <: AbstractObservationEvidence
    lower::L
    upper::U
end

struct CensoredEvidence{L,U} <: AbstractObservationEvidence
    lower::L
    upper::U
end

struct IntervalEvidence{U} <: AbstractObservationEvidence
    upper::U
end

function _observation_bound(bound, label::AbstractString)
    bound === nothing && return nothing
    bound isa Symbol && return bound
    bound isa Real || throw(ArgumentError(
        "native PPL $label must be a finite real literal, a bound input " *
        "name, or `nothing`; got $(typeof(bound))"))
    isfinite(bound) || throw(ArgumentError(
        "native PPL $label literal must be finite; got $bound"))
    bound
end

function _observation_interval(lower, upper, label::AbstractString)
    lower = _observation_bound(lower, "$label lower bound")
    upper = _observation_bound(upper, "$label upper bound")
    lower === nothing && upper === nothing && throw(ArgumentError(
        "native PPL $label needs at least one bound"))
    lower isa Real && upper isa Real && lower >= upper &&
        throw(ArgumentError(
            "native PPL $label lower bound must be below its upper bound; " *
            "got ($lower, $upper)"))
    lower, upper
end

function truncated_evidence(; lower=nothing, upper=nothing)
    lower, upper = _observation_interval(lower, upper, "truncation")
    TruncatedEvidence(lower, upper)
end

function censored_evidence(; lower=nothing, upper=nothing)
    lower, upper = _observation_interval(lower, upper, "censoring")
    CensoredEvidence(lower, upper)
end

function interval_evidence(upper)
    upper = _observation_bound(upper, "interval-evidence upper endpoint")
    upper === nothing && throw(ArgumentError(
        "native PPL interval evidence needs an upper endpoint"))
    IntervalEvidence(upper)
end

evidence_kind(::TruncatedEvidence) = :truncated
evidence_kind(::CensoredEvidence) = :censored
evidence_kind(::IntervalEvidence) = :interval_censored
evidence_lower(evidence::Union{TruncatedEvidence,CensoredEvidence}) =
    evidence.lower
evidence_lower(::IntervalEvidence) = nothing
evidence_upper(evidence::AbstractObservationEvidence) = evidence.upper

"""One base observation decorated with truncation/censoring evidence."""
struct EvidenceObservation{
        O<:AbstractObservationDeclaration,
        E<:AbstractObservationEvidence,
    } <: AbstractObservationDeclaration
    observation::O
    evidence::E
end

evidence_observation(observation::AbstractObservationDeclaration,
                     evidence::AbstractObservationEvidence) =
    EvidenceObservation(observation, evidence)

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
observation_response(::StudentTObservation{Response}) where {Response} = Response
observation_response(::BernoulliLogitObservation{Response}) where {Response} =
    Response
observation_response(::PoissonObservation{Response}) where {Response} = Response
observation_response(
    ::ZeroInflatedPoissonObservation{Response},
) where {Response} = Response
observation_response(
    ::NegativeBinomial2Observation{Response},
) where {Response} = Response
observation_response(
    ::BetaBinomial2Observation{Response},
) where {Response} = Response
observation_response(
    ::BinomialLogitObservation{Response},
) where {Response} = Response
observation_response(
    ::CategoricalLogitObservation{Response},
) where {Response} = Response
observation_response(observation::WeightedObservation) =
    observation_response(observation.observation)
observation_response(observation::EvidenceObservation) =
    observation_response(observation.observation)
observation_response(observation::BroadcastObservation) =
    observation_response(observation.scalar)
observation_dependencies(
    ::NormalObservation{Response,Location,Scale},
) where {Response,Location,Scale} = (Location, Scale)
observation_dependencies(
    ::StudentTObservation{Response,Df,Location,Scale},
) where {Response,Df,Location,Scale} = (Df, Location, Scale)
observation_dependencies(
    ::BernoulliLogitObservation{Response,Logit},
) where {Response,Logit} = (Logit,)
observation_dependencies(
    ::PoissonObservation{Response,Rate},
) where {Response,Rate} = (Rate,)
observation_dependencies(
    ::ZeroInflatedPoissonObservation{Response,Rate,ZeroInflation},
) where {Response,Rate,ZeroInflation} = (Rate, ZeroInflation)
observation_dependencies(
    ::NegativeBinomial2Observation{Response,Mean,Precision},
) where {Response,Mean,Precision} = (Mean, Precision)
observation_dependencies(
    ::BetaBinomial2Observation{Response,Trials,Mean,Precision},
) where {Response,Trials,Mean,Precision} = (Trials, Mean, Precision)
observation_dependencies(
    ::BinomialLogitObservation{Response,Trials,Logit},
) where {Response,Trials,Logit} = (Trials, Logit)
observation_dependencies(
    ::CategoricalLogitObservation{Response,Logits},
) where {Response,Logits} = Logits
observation_dependencies(observation::WeightedObservation) =
    (observation_dependencies(observation.observation)...,
     observation_weight_source(observation.weight))
_evidence_bound_dependencies(bound) = bound isa Symbol ? (bound,) : ()
observation_dependencies(observation::EvidenceObservation) =
    (observation_dependencies(observation.observation)...,
     _evidence_bound_dependencies(evidence_lower(observation.evidence))...,
     _evidence_bound_dependencies(evidence_upper(observation.evidence))...)
observation_dependencies(observation::BroadcastObservation) =
    observation_dependencies(observation.scalar)

base_observation(observation::AbstractObservationDeclaration) = observation
base_observation(observation::WeightedObservation) =
    base_observation(observation.observation)
base_observation(observation::EvidenceObservation) =
    base_observation(observation.observation)

"""
An unbound, typed native-PPL declaration shared by direct authors and BRM.

Names are the keys of the declaration `NamedTuple`s. `outputs` is either
`nothing` for the legacy implicit-export surface or a named map from public
output aliases to graph identities. The model owns logical graph semantics
only; data binding, fitted transform state, axes, coordinates, output
signatures, and workspace layout belong to compilation/preparation.
"""
struct Model{I,P,N,O,R,S}
    inputs::I
    parameters::P
    nodes::N
    observations::O
    outputs::R
    site_order::S
end

_default_site_order(parameters, observations) =
    (keys(parameters)..., keys(observations)...)

Model(inputs, parameters, nodes, observations) =
    Model(inputs, parameters, nodes, observations, nothing,
          _default_site_order(parameters, observations))

Model(inputs, parameters, nodes, observations, outputs) =
    Model(inputs, parameters, nodes, observations, outputs,
          _default_site_order(parameters, observations))

"""Reference one graph value as a stochastic-factor argument."""
struct SiteValue{Name} end
site_value_name(::SiteValue{Name}) where {Name} = Name

"""Reference one bound deterministic input as a stochastic-factor argument."""
struct InputValue{Name} end
input_value_name(::InputValue{Name}) where {Name} = Name

"""Reference one staged deterministic-node output as a factor argument."""
struct NodeValue{Name} end
node_value_name(::NodeValue{Name}) where {Name} = Name

"""Literal stochastic-factor argument retained in the typed graph."""
struct LiteralValue{T}
    value::T
end

abstract type AbstractSiteFactor end
struct StandardNormalSiteFactor <: AbstractSiteFactor end
struct NormalSiteFactor{L,S} <: AbstractSiteFactor
    location::L
    scale::S
end
struct StudentTSiteFactor{D,L,S} <: AbstractSiteFactor
    degrees_freedom::D
    location::L
    scale::S
end
struct ExponentialSiteFactor{S} <: AbstractSiteFactor
    scale::S
end
struct LKJCholeskySiteFactor{E,N} <: AbstractSiteFactor
    eta::E
    log_normalizers::N
end
function LKJCholeskySiteFactor(eta, ::Val{K}) where {K}
    log_normalizers = ntuple(K - 1) do column
        alpha = eta + (K - column - 1) / 2
        BRM.loggamma(alpha + 0.5) - BRM.loggamma(alpha) -
            0.5 * log(BRM.pi)
    end
    LKJCholeskySiteFactor(eta, log_normalizers)
end
struct BernoulliLogitSiteFactor{L} <: AbstractSiteFactor
    logit::L
end
struct PoissonSiteFactor{R} <: AbstractSiteFactor
    rate::R
end
struct ZeroInflatedPoissonSiteFactor{R,Z} <: AbstractSiteFactor
    rate::R
    zero_inflation::Z
end
struct NegativeBinomial2SiteFactor{M,P} <: AbstractSiteFactor
    mean::M
    precision::P
end
struct BetaBinomial2SiteFactor{N,M,P} <: AbstractSiteFactor
    trials::N
    mean::M
    precision::P
end
struct BinomialLogitSiteFactor{N,L} <: AbstractSiteFactor
    trials::N
    logit::L
end
struct CategoricalLogitSiteFactor{L} <: AbstractSiteFactor
    logits::L
end
struct WeightedSiteFactor{
        F<:AbstractSiteFactor,
        W<:AbstractObservationWeight,
        V,
    } <: AbstractSiteFactor
    factor::F
    weight::W
    values::V
end

struct EvidenceSiteFactor{
        F<:AbstractSiteFactor,
        E<:AbstractObservationEvidence,
        L,U,
    } <: AbstractSiteFactor
    factor::F
    evidence::E
    lower::L
    upper::U
end

base_site_factor(factor::AbstractSiteFactor) = factor
base_site_factor(factor::Union{WeightedSiteFactor,EvidenceSiteFactor}) =
    base_site_factor(factor.factor)

site_factor_dependencies(::StandardNormalSiteFactor) = ()
site_factor_dependencies(factor::NormalSiteFactor) =
    _factor_value_dependencies((factor.location, factor.scale))
site_factor_dependencies(factor::StudentTSiteFactor) =
    _factor_value_dependencies((
        factor.degrees_freedom, factor.location, factor.scale))
site_factor_dependencies(factor::ExponentialSiteFactor) =
    _factor_value_dependencies((factor.scale,))
site_factor_dependencies(::LKJCholeskySiteFactor) = ()
site_factor_dependencies(factor::BernoulliLogitSiteFactor) =
    _factor_value_dependencies((factor.logit,))
site_factor_dependencies(factor::PoissonSiteFactor) =
    _factor_value_dependencies((factor.rate,))
site_factor_dependencies(factor::ZeroInflatedPoissonSiteFactor) =
    _factor_value_dependencies((factor.rate, factor.zero_inflation))
site_factor_dependencies(factor::NegativeBinomial2SiteFactor) =
    _factor_value_dependencies((factor.mean, factor.precision))
site_factor_dependencies(factor::BetaBinomial2SiteFactor) =
    _factor_value_dependencies((factor.trials, factor.mean, factor.precision))
site_factor_dependencies(factor::BinomialLogitSiteFactor) =
    _factor_value_dependencies((factor.trials, factor.logit))
site_factor_dependencies(factor::CategoricalLogitSiteFactor) =
    _factor_value_dependencies(factor.logits)
site_factor_dependencies(factor::WeightedSiteFactor) =
    site_factor_dependencies(factor.factor)
site_factor_dependencies(factor::EvidenceSiteFactor) =
    (site_factor_dependencies(factor.factor)...,
     _factor_value_dependencies((factor.lower, factor.upper))...)

_factor_value_name(value::SiteValue) = site_value_name(value)
_factor_value_name(value::NodeValue) = node_value_name(value)

function _factor_value_dependencies(values)
    Tuple(_factor_value_name(value) for value in values
          if value isa Union{SiteValue,NodeValue})
end

abstract type AbstractFactorNode end

struct CenterFactorNode{I} <: AbstractFactorNode
    input::I
end

struct ZScaleFactorNode{I} <: AbstractFactorNode
    input::I
end

"""Training-set mean owned by a bound fitted-center factor node."""
struct FittedCenter{T}
    mean::T
end

"""Training-set mean and corrected sample SD owned by a bound zscale node."""
struct FittedZScale{T}
    mean::T
    scale::T
end

struct AffineFactorNode{I,C,O,H} <: AbstractFactorNode
    inputs::I
    coefficients::C
    offsets::O
end

AffineFactorNode(inputs::I, coefficients::C, offsets::O,
                 intercept::Bool) where {I,C,O} =
    AffineFactorNode{I,C,O,intercept}(inputs, coefficients, offsets)

affine_has_intercept(::AffineFactorNode{I,C,O,H}) where {I,C,O,H} = H

struct ExpFactorNode{I} <: AbstractFactorNode
    input::I
end

struct LogFactorNode{I} <: AbstractFactorNode
    input::I
end

struct LogisticFactorNode{I} <: AbstractFactorNode
    input::I
end

struct GroupGatherFactorNode{V,G} <: AbstractFactorNode
    values::V
    group::G
end

struct RowProductFactorNode{L,R} <: AbstractFactorNode
    left::L
    right::R
end

struct GroupedAffineFactorNode{Z,S,C,G,P,I,K} <: AbstractFactorNode
    standardized::Z
    scales::S
    correlation::C
    group::G
    predictors::P
end

grouped_coefficient_indices(
    ::GroupedAffineFactorNode{Z,S,C,G,P,I},
) where {Z,S,C,G,P,I} = I
grouped_coefficient_count(
    ::GroupedAffineFactorNode{Z,S,C,G,P,I,K},
) where {Z,S,C,G,P,I,K} = K

factor_node_dependencies(node::Union{
        CenterFactorNode,ZScaleFactorNode,ExpFactorNode,LogFactorNode,
        LogisticFactorNode}) =
    _factor_value_dependencies((node.input,))
factor_node_dependencies(node::AffineFactorNode) =
    _factor_value_dependencies((
        node.inputs..., node.coefficients, node.offsets...))
factor_node_dependencies(node::GroupGatherFactorNode) =
    _factor_value_dependencies((node.values, node.group))
factor_node_dependencies(node::RowProductFactorNode) =
    _factor_value_dependencies((node.left, node.right))
factor_node_dependencies(node::GroupedAffineFactorNode) =
    _factor_value_dependencies((
        node.standardized, node.scales, node.correlation,
        node.group, node.predictors...))

abstract type AbstractSiteShape end
struct ScalarSiteShape <: AbstractSiteShape end
struct BlockSiteShape <: AbstractSiteShape end
struct BroadcastSiteShape <: AbstractSiteShape end

abstract type AbstractSiteActivity end
struct FreeSite <: AbstractSiteActivity end
struct ConditionedSite <: AbstractSiteActivity end
struct GeneratedSite <: AbstractSiteActivity end

"""
One canonical stochastic site after declaration normalization.

The site owns its support/transform, typed factor arguments, scalar/block/
broadcast shape, activity after conditioning, and semantic coordinate keys.
This is the shared representation that replaces the transitional semantic
split between `Model.parameters` and `Model.observations`.
"""
struct StochasticSite{S,T,F<:AbstractSiteFactor,H<:AbstractSiteShape,
                      A<:AbstractSiteActivity,K}
    support::S
    transform::T
    factor::F
    shape::H
    activity::A
    coordinate_keys::K
end

"""Contiguous unconstrained coordinates assigned to one free site."""
struct SiteCoordinates{K,R}
    keys::K
    indices::R
end


"""One semantic coordinate of a group-indexed latent site."""
struct GroupCoordinateKey{L}
    site::Symbol
    level::L
end

"""One group-major coefficient coordinate of a structured latent block."""
struct GroupCoefficientKey{L}
    site::Symbol
    level::L
    coefficient::Symbol
end

"""One unconstrained strict-lower-triangular correlation coordinate."""
struct CorrelationCoordinateKey
    site::Symbol
    row::Int
    column::Int
end

"""Typed, ordered stochastic-site graph and its free-coordinate allocation."""
struct FactorGraph{Order,S,N,C}
    sites::S
    nodes::N
    coordinates::C
    dimension::Int
end

FactorGraph(sites::S, nodes::N, schedule::O, coordinates::C,
            dimension::Int) where {S,N,O,C} =
    FactorGraph{schedule,S,N,C}(sites, nodes, coordinates, dimension)

factor_schedule(::FactorGraph{Order}) where {Order} = Order

function Base.getproperty(graph::FactorGraph, name::Symbol)
    name === :schedule && return factor_schedule(graph)
    getfield(graph, name)
end

Base.propertynames(::FactorGraph, private::Bool=false) =
    (fieldnames(FactorGraph)..., :schedule)

function _parameter_stochastic_site(
    name::Symbol, declaration::Parameter, conditions::NamedTuple)
    prior = declaration.prior
    factor = if prior isa StandardNormal
        StandardNormalSiteFactor()
    elseif prior isa NormalPrior
        NormalSiteFactor(
            LiteralValue(prior.location), LiteralValue(prior.scale))
    else
        ExponentialSiteFactor(LiteralValue(prior.scale))
    end
    shape = length(declaration.axis_keys) == 1 ?
        ScalarSiteShape() : BlockSiteShape()
    activity = hasproperty(conditions, name) ? ConditionedSite() : FreeSite()
    coordinate_keys = activity isa FreeSite ? declaration.axis_keys : ()
    StochasticSite(
        declaration.support, declaration.transform, factor, shape,
        activity, coordinate_keys)
end


function _group_levels(values, name::Symbol)
    values isa AbstractVector || throw(ArgumentError(
        "native PPL group input `$name` must be a vector"))
    isempty(values) && throw(ArgumentError(
        "native PPL group input `$name` cannot be empty"))
    any(ismissing, values) && throw(ArgumentError(
        "native PPL group input `$name` cannot contain missing values"))
    Tuple(unique(values))
end

function _parameter_stochastic_site(
    name::Symbol, declaration::GroupedNormalParameter,
    conditions::NamedTuple, bindings::NamedTuple,
    fitted_levels=nothing, new_groups::Symbol=:error)
    group_name = group_input(declaration)
    hasproperty(bindings, group_name) || throw(ArgumentError(
        "native PPL grouped site `$name` requires binding `$group_name`"))
    observed_levels = _group_levels(
        getproperty(bindings, group_name), group_name)
    levels = fitted_levels === nothing ? observed_levels : fitted_levels
    unknown = Tuple(level for level in observed_levels if level ∉ levels)
    if !isempty(unknown) && new_groups === :error
        throw(CapabilityError(
            :new_group,
            "native PPL group input `$group_name` contains new levels " *
            "$(unknown); replay reuses fitted group effects by default; " *
            "pass `new_groups=:resample` for prediction-only conditional " *
            "simulation"))
    end
    activity = hasproperty(conditions, name) ? ConditionedSite() : FreeSite()
    coordinate_keys = Tuple(
        GroupCoordinateKey(name, level) for level in levels)
    StochasticSite(
        RealSupport(), Identity(),
        NormalSiteFactor(declaration.location, declaration.scale),
        BlockSiteShape(), activity, coordinate_keys)
end

function _parameter_stochastic_site(
    name::Symbol, declaration::GroupedStandardNormalParameter,
    conditions::NamedTuple, bindings::NamedTuple,
    fitted_levels=nothing, new_groups::Symbol=:error)
    group_name = group_input(declaration)
    hasproperty(bindings, group_name) || throw(ArgumentError(
        "native PPL grouped site `$name` requires binding `$group_name`"))
    observed_levels = _group_levels(
        getproperty(bindings, group_name), group_name)
    levels = fitted_levels === nothing ? observed_levels : fitted_levels
    unknown = Tuple(level for level in observed_levels if level ∉ levels)
    if !isempty(unknown) && new_groups === :error
        throw(CapabilityError(
            :new_group,
            "native PPL group input `$group_name` contains new levels " *
            "$(unknown); replay reuses fitted group effects by default; " *
            "pass `new_groups=:resample` for prediction-only conditional " *
            "simulation"))
    end
    coefficients = group_coefficients(declaration)
    activity = hasproperty(conditions, name) ? ConditionedSite() : FreeSite()
    coordinate_keys = Tuple(
        GroupCoefficientKey(name, level, coefficient)
        for level in levels for coefficient in coefficients)
    StochasticSite(
        RealSupport(), Identity(), StandardNormalSiteFactor(),
        BlockSiteShape(), activity, coordinate_keys)
end

function _parameter_stochastic_site(
    name::Symbol, declaration::CholeskyCorrelationParameter,
    conditions::NamedTuple)
    coefficients = correlation_coefficients(declaration)
    dimension = length(coefficients)
    activity = hasproperty(conditions, name) ? ConditionedSite() : FreeSite()
    coordinate_keys = activity isa FreeSite ? Tuple(
        CorrelationCoordinateKey(name, row, column)
        for column in 1:(dimension - 1)
        for row in (column + 1):dimension) : ()
    StochasticSite(
        CholeskyCorrelationSupport{dimension}(),
        CholeskyCorrelationTransform{dimension}(),
        LKJCholeskySiteFactor(declaration.eta, Val(dimension)),
        BlockSiteShape(), activity,
        coordinate_keys)
end

function _factor_value(name::Symbol, inputs::NamedTuple, nodes::NamedTuple)
    hasproperty(inputs, name) && return InputValue{name}()
    hasproperty(nodes, name) && return NodeValue{name}()
    SiteValue{name}()
end

_factor_bound_value(bound::Symbol, inputs::NamedTuple, nodes::NamedTuple) =
    _factor_value(bound, inputs, nodes)
_factor_bound_value(bound, ::NamedTuple, ::NamedTuple) = LiteralValue(bound)

function _decorate_site_factor(observation::AbstractObservationDeclaration,
                               factor::AbstractSiteFactor,
                               inputs::NamedTuple, nodes::NamedTuple,
                               name::Symbol)
    observation === base_observation(observation) && return factor
    if observation isa WeightedObservation
        factor = _decorate_site_factor(
            observation.observation, factor, inputs, nodes, name)
        source = observation_weight_source(observation.weight)
        hasproperty(inputs, source) || throw(CapabilityError(
            :graph_identity,
            "weighted observation `$name` references unknown weight input " *
            "`$source`"))
        return WeightedSiteFactor(
            factor, observation.weight, _factor_value(source, inputs, nodes))
    end
    factor = _decorate_site_factor(
        observation.observation, factor, inputs, nodes, name)
    evidence = observation.evidence
    EvidenceSiteFactor(
        factor, evidence,
        _factor_bound_value(evidence_lower(evidence), inputs, nodes),
        _factor_bound_value(evidence_upper(evidence), inputs, nodes))
end

function _factor_node(declaration::AbstractNodeDeclaration,
                      inputs::NamedTuple, nodes::NamedTuple)
    reference(name) = _factor_value(name, inputs, nodes)
    if declaration isa Center
        CenterFactorNode(reference(node_input(declaration)))
    elseif declaration isa ZScale
        ZScaleFactorNode(reference(node_input(declaration)))
    elseif declaration isa Affine
        AffineFactorNode(
            map(reference, node_inputs(declaration)),
            SiteValue{affine_parameter(declaration)}(),
            map(reference, affine_offsets(declaration)),
            affine_has_intercept(declaration))
    elseif declaration isa ExpLink
        ExpFactorNode(reference(node_input(declaration)))
    elseif declaration isa LogLink
        LogFactorNode(reference(node_input(declaration)))
    elseif declaration isa LogisticLink
        LogisticFactorNode(reference(node_input(declaration)))
    elseif declaration isa GroupGather
        GroupGatherFactorNode(
            SiteValue{group_values(declaration)}(),
            InputValue{group_input(declaration)}())
    elseif declaration isa RowProduct
        left, right = row_product_inputs(declaration)
        RowProductFactorNode(reference(left), reference(right))
    else
        GroupedAffineFactorNode{
            typeof(SiteValue{grouped_standardized(declaration)}()),
            typeof(SiteValue{grouped_scales(declaration)}()),
            typeof(SiteValue{grouped_correlation(declaration)}()),
            typeof(InputValue{group_input(declaration)}()),
            typeof(map(predictor -> predictor === nothing ? nothing :
                reference(predictor), grouped_predictors(declaration))),
            grouped_coefficient_indices(declaration),
            grouped_coefficient_count(declaration)}(
            SiteValue{grouped_standardized(declaration)}(),
            SiteValue{grouped_scales(declaration)}(),
            SiteValue{grouped_correlation(declaration)}(),
            InputValue{group_input(declaration)}(),
            map(predictor -> predictor === nothing ? nothing :
                reference(predictor), grouped_predictors(declaration)))
    end
end

function _factor_nodes(declaration::Model)
    names = Tuple(keys(declaration.nodes))
    values = map(names) do name
        _factor_node(
            getproperty(declaration.nodes, name), declaration.inputs,
            declaration.nodes)
    end
    NamedTuple{names}(values)
end

function _observation_stochastic_site(
    name::Symbol, declaration::AbstractObservationDeclaration,
    inputs::NamedTuple, nodes::NamedTuple, conditions::NamedTuple)
    scalar = scalar_observation(declaration)
    base = base_observation(scalar)
    dependencies = observation_dependencies(base)
    factor = if base isa NormalObservation
        NormalSiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes))
    elseif base isa StudentTObservation
        StudentTSiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes),
            _factor_value(dependencies[3], inputs, nodes))
    elseif base isa BernoulliLogitObservation
        BernoulliLogitSiteFactor(_factor_value(
            only(dependencies), inputs, nodes))
    elseif base isa ZeroInflatedPoissonObservation
        ZeroInflatedPoissonSiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes))
    elseif base isa NegativeBinomial2Observation
        NegativeBinomial2SiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes))
    elseif base isa BetaBinomial2Observation
        BetaBinomial2SiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes),
            _factor_value(dependencies[3], inputs, nodes))
    elseif base isa BinomialLogitObservation
        BinomialLogitSiteFactor(
            _factor_value(dependencies[1], inputs, nodes),
            _factor_value(dependencies[2], inputs, nodes))
    elseif base isa CategoricalLogitObservation
        CategoricalLogitSiteFactor(Tuple(
            _factor_value(dependency, inputs, nodes)
            for dependency in dependencies))
    else
        PoissonSiteFactor(_factor_value(
            only(dependencies), inputs, nodes))
    end
    factor = _decorate_site_factor(scalar, factor, inputs, nodes, name)
    shape = is_broadcast_observation(declaration) ?
        BroadcastSiteShape() : ScalarSiteShape()
    base_factor = base_site_factor(factor)
    activity = if hasproperty(conditions, name)
        ConditionedSite()
    elseif shape isa BroadcastSiteShape
        GeneratedSite()
    elseif base_factor isa NormalSiteFactor
        FreeSite()
    else
        throw(CapabilityError(
            :discrete_latent,
            "unconditioned scalar site `$name` has unsupported discrete " *
            "factor $(typeof(factor))"))
    end
    coordinate_keys = activity isa FreeSite ? (name,) : ()
    support = base_factor isa NormalSiteFactor ? RealSupport() : nothing
    transform = base_factor isa NormalSiteFactor ? Identity() : nothing
    StochasticSite(
        support, transform, factor, shape, activity, coordinate_keys)
end

function _factor_schedule(declaration::Model, sites::NamedTuple,
                          nodes::NamedTuple)
    pending = Symbol[declaration.site_order..., keys(nodes)...]
    available = Set{Symbol}(keys(declaration.inputs))
    schedule = Symbol[]
    while !isempty(pending)
        selected = nothing
        for index in eachindex(pending)
            name = pending[index]
            dependencies = hasproperty(sites, name) ?
                site_factor_dependencies(getproperty(sites, name).factor) :
                factor_node_dependencies(getproperty(nodes, name))
            all(dependency -> dependency in available, dependencies) ||
                continue
            selected = index
            break
        end
        selected === nothing && throw(CapabilityError(
            :factor_schedule,
            "stochastic/deterministic factor graph contains a cycle or " *
            "unavailable dependency among $(Tuple(pending))"))
        name = pending[selected]
        push!(schedule, name)
        push!(available, name)
        deleteat!(pending, selected)
    end
    Tuple(schedule)
end

function _site_coordinates(sites::NamedTuple)
    names = Symbol[]
    allocations = Any[]
    next_index = 1
    for (name, site) in pairs(sites)
        site.activity isa FreeSite || continue
        count = length(site.coordinate_keys)
        indices = next_index:(next_index + count - 1)
        push!(names, name)
        push!(allocations, SiteCoordinates(site.coordinate_keys, indices))
        next_index += count
    end
    NamedTuple{Tuple(names)}(Tuple(allocations)), next_index - 1
end

"""
    factor_graph(model; conditions=(;)) -> FactorGraph

Normalize parameters and stochastic observations into one ordered typed site
graph, normalize deterministic value nodes, derive one mixed topological
schedule, then allocate semantic unconstrained coordinates for free sites.
`FactorPlan` executes the supported subset while preserving this graph as the
public mid-level semantic representation.
"""
function factor_graph(declaration::Model; conditions=(;), bindings=(;),
                      group_levels=(;), new_groups::Symbol=:error)
    new_groups in (:error, :resample) || throw(ArgumentError(
        "native PPL new-groups policy must be :error or :resample; got " *
        "`$new_groups`"))
    _validate_model(declaration)
    canonical_conditions = _canonical_factor_conditions(
        declaration, conditions)
    names = Symbol[]
    sites = Any[]
    available_sites = Set{Symbol}()
    all_sites = Set(declaration.site_order)
    for name in declaration.site_order
        site = if hasproperty(declaration.parameters, name)
            parameter = getproperty(declaration.parameters, name)
            parameter isa Union{
                GroupedNormalParameter,GroupedStandardNormalParameter,
            } ?
                _parameter_stochastic_site(
                    name, parameter, canonical_conditions, bindings,
                    hasproperty(group_levels, name) ?
                        getproperty(group_levels, name) : nothing,
                    new_groups) :
                _parameter_stochastic_site(
                    name, parameter, canonical_conditions)
        else
            _observation_stochastic_site(
                name, getproperty(declaration.observations, name),
                declaration.inputs, declaration.nodes,
                canonical_conditions)
        end
        unavailable = setdiff(
            Set(site_factor_dependencies(site.factor)), available_sites)
        unresolved_sites = intersect(unavailable, all_sites)
        isempty(unresolved_sites) || throw(CapabilityError(
            :site_order,
            "stochastic site `$name` depends on unscheduled sites " *
            "$(sort!(collect(unresolved_sites)))"))
        push!(names, name)
        push!(sites, site)
        push!(available_sites, name)
    end
    named_sites = NamedTuple{Tuple(names)}(Tuple(sites))
    named_nodes = _factor_nodes(declaration)
    schedule = _factor_schedule(declaration, named_sites, named_nodes)
    coordinates, dimension = _site_coordinates(named_sites)
    FactorGraph(named_sites, named_nodes, schedule, coordinates, dimension)
end

function _canonical_factor_conditions(declaration::Model, conditions)
    conditions isa NamedTuple || throw(ArgumentError(
        "native PPL factor-graph conditions must be a NamedTuple; got " *
        "$(typeof(conditions))"))
    stochastic_names = Set((keys(declaration.parameters)...,
                            keys(declaration.observations)...))
    names = Symbol[]
    values = Any[]
    for (name, value) in pairs(conditions)
        canonical = if name in stochastic_names
            name
        elseif declaration.outputs !== nothing &&
               hasproperty(declaration.outputs, name)
            getproperty(declaration.outputs, name)
        else
            throw(ArgumentError(
                "native PPL factor-graph condition `$name` does not name a " *
                "stochastic site"))
        end
        canonical in stochastic_names || throw(ArgumentError(
            "native PPL factor-graph condition `$name` resolves to " *
            "non-stochastic output `$canonical`"))
        canonical in names && throw(ArgumentError(
            "native PPL factor-graph conditions name site `$canonical` more " *
            "than once"))
        push!(names, canonical)
        push!(values, value)
    end
    NamedTuple{Tuple(names)}(Tuple(values))
end

"""Executable plan for an ordered continuous stochastic-site DAG."""
struct FactorPlan{Output,M,G,F,B,C,O,I,N,J,K,L,A}
    declaration::M
    graph::G
    fitted_nodes::F
    bindings::B
    conditions::C
    outcome_levels::O
    site_indices::I
    node_indices::N
    group_indices::J
    generated_group_levels::K
    generated_group_indices::L
    observation_axis::A
end

FactorPlan(declaration::M, graph::G, fitted_nodes::F,
           bindings::B, conditions::C, outcome_levels::O,
           site_indices::I, node_indices::N, group_indices::J,
           generated_group_levels::K,
           generated_group_indices::L,
           observation_axis::A,
           output_sites::Tuple) where {M,G,F,B,C,O,I,N,J,K,L,A} =
    FactorPlan{output_sites,M,G,F,B,C,O,I,N,J,K,L,A}(
        declaration, graph, fitted_nodes, bindings, conditions,
        outcome_levels, site_indices,
        node_indices, group_indices, generated_group_levels,
        generated_group_indices, observation_axis)

FactorPlan(declaration, graph, fitted_nodes, bindings, conditions,
           outcome_levels, site_indices, node_indices, group_indices,
           generated_group_levels, generated_group_indices,
           observation_axis, output_site::Symbol) =
    FactorPlan(
        declaration, graph, fitted_nodes, bindings, conditions,
        outcome_levels, site_indices, node_indices, group_indices,
        generated_group_levels, generated_group_indices,
        observation_axis, (output_site,))

factor_output_sites(::FactorPlan{Outputs}) where {Outputs} = Outputs
function factor_output_site(plan::FactorPlan)
    outputs = factor_output_sites(plan)
    length(outputs) == 1 || throw(CapabilityError(
        :factor_outputs,
        "factor plan has multiple terminal stochastic outputs " *
        "$(collect(outputs)); select an outcome explicitly"))
    only(outputs)
end

function Base.getproperty(plan::FactorPlan, name::Symbol)
    name === :output_site && return factor_output_site(plan)
    name === :output_sites && return factor_output_sites(plan)
    getfield(plan, name)
end

Base.propertynames(::FactorPlan, private::Bool=false) =
    (fieldnames(FactorPlan)..., :output_site, :output_sites)

BRM.LogDensityProblems.dimension(plan::FactorPlan) = plan.graph.dimension

function _uses_factor_executor(declaration::Model, conditions, bindings=(;))
    graph = factor_graph(declaration; conditions, bindings)
    any(site -> site.shape isa BroadcastSiteShape, values(graph.sites)) ||
        return false
    stochastic_names = Set(declaration.site_order)
    dependent_site = any(declaration.site_order) do name
        hasproperty(declaration.observations, name) || return false
        site = getproperty(graph.sites, name)
        site.shape isa ScalarSiteShape &&
            any(dependency -> dependency in stochastic_names,
                site_factor_dependencies(site.factor))
    end
    sampled_offset_affine = any(values(graph.nodes)) do node
        node isa AffineFactorNode && !isempty(node.offsets)
    end
    grouped_gather = any(node -> node isa GroupGatherFactorNode,
                         values(graph.nodes))
    grouped_affine_node = any(node -> node isa GroupedAffineFactorNode,
                              values(graph.nodes))
    row_scale = any(values(graph.sites)) do site
        site.shape isa BroadcastSiteShape &&
            base_site_factor(site.factor) isa NormalSiteFactor &&
            base_site_factor(site.factor).scale isa NodeValue
    end
    weighted_observation = any(values(graph.sites)) do site
        _weighted_site_factor(site.factor) !== nothing
    end
    evidence_observation = any(values(graph.sites)) do site
        _evidence_site_factor(site.factor) !== nothing
    end
    zero_inflated_observation = any(values(graph.sites)) do site
        base_site_factor(site.factor) isa ZeroInflatedPoissonSiteFactor
    end
    overdispersed_count_observation = any(values(graph.sites)) do site
        base_site_factor(site.factor) isa NegativeBinomial2SiteFactor
    end
    bounded_count_observation = any(values(graph.sites)) do site
        base_site_factor(site.factor) isa
            Union{BetaBinomial2SiteFactor,BinomialLogitSiteFactor}
    end
    categorical_observation = any(values(graph.sites)) do site
        base_site_factor(site.factor) isa CategoricalLogitSiteFactor
    end
    student_t_observation = any(values(graph.sites)) do site
        base_site_factor(site.factor) isa StudentTSiteFactor
    end
    broadcast_output_count = count(values(graph.sites)) do site
        site.shape isa BroadcastSiteShape
    end
    dependent_site || sampled_offset_affine || grouped_gather ||
        grouped_affine_node || row_scale || weighted_observation ||
        evidence_observation || zero_inflated_observation ||
        overdispersed_count_observation || bounded_count_observation ||
        categorical_observation || student_t_observation ||
        broadcast_output_count > 1
end

_weighted_site_factor(::AbstractSiteFactor) = nothing
_weighted_site_factor(factor::WeightedSiteFactor) = factor
_weighted_site_factor(factor::EvidenceSiteFactor) =
    _weighted_site_factor(factor.factor)

_evidence_site_factor(::AbstractSiteFactor) = nothing
_evidence_site_factor(factor::EvidenceSiteFactor) = factor
_evidence_site_factor(factor::WeightedSiteFactor) =
    _evidence_site_factor(factor.factor)

function _factor_validate_weight(factor::WeightedSiteFactor,
                                 bindings::NamedTuple,
                                 response::Symbol, observation_count::Int)
    factor.values isa InputValue || throw(CapabilityError(
        :observation_weights,
        "weighted response `$response` requires one bound weight input"))
    source = input_value_name(factor.values)
    raw = getproperty(bindings, source)
    raw isa AbstractVector{<:Real} || throw(ArgumentError(
        "native PPL weight input `$source` for response `$response` must be " *
        "a real vector"))
    length(raw) == observation_count || throw(DimensionMismatch(
        "native PPL weight input `$source` has length $(length(raw)) but " *
        "response `$response` has length $observation_count"))
    all(isfinite, raw) || throw(ArgumentError(
        "native PPL weight input `$source` for response `$response` must be finite"))
    kind = observation_weight_kind(factor.weight)
    base = base_site_factor(factor)
    if kind === :analytic
        base isa NormalSiteFactor || throw(CapabilityError(
            :observation_weights,
            "analytic weights currently require a Normal response; got " *
            "$(typeof(base)) for `$response`"))
        _evidence_site_factor(factor) === nothing || throw(CapabilityError(
            :observation_weights,
            "analytic weights combined with response evidence are not yet " *
            "supported for `$response`"))
        all(>(0), raw) || throw(ArgumentError(
            "native PPL analytic weights for `$response` must be strictly positive"))
    elseif kind === :frequency
        all(value -> value >= zero(value) && isinteger(value), raw) || throw(
            ArgumentError(
                "native PPL frequency weights for `$response` must be " *
                "nonnegative integer-valued counts"))
    elseif kind === :power
        all(>=(0), raw) || throw(ArgumentError(
            "native PPL power weights for `$response` must be nonnegative"))
    else
        kind === :unit || throw(CapabilityError(
            :observation_weights,
            "unsupported observation-weight kind `$kind`"))
        all(isone, raw) || throw(ArgumentError(
            "native PPL unit weights for `$response` must all equal one"))
    end
    nothing
end

function _factor_validate_evidence_bound(bound, bindings::NamedTuple,
                                         response::Symbol,
                                         label::AbstractString,
                                         discrete::Bool)
    raw = if bound isa LiteralValue
        bound.value
    elseif bound isa InputValue
        getproperty(bindings, input_value_name(bound))
    else
        return nothing
    end
    raw === nothing && return nothing
    values = raw isa AbstractVector ? raw : (raw,)
    all(value -> value isa Real && isfinite(value), values) ||
        throw(ArgumentError(
            "native PPL $label for response `$response` must contain " *
            "finite real values"))
    discrete && all(value -> isinteger(value), values) || !discrete ||
        throw(ArgumentError(
            "native PPL $label for discrete response `$response` must be " *
            "integer-valued"))
    nothing
end

function _factor_validate_evidence(factor::EvidenceSiteFactor,
                                   bindings::NamedTuple,
                                   response::Symbol)
    base = base_site_factor(factor)
    base isa Union{NormalSiteFactor,PoissonSiteFactor} || throw(
        CapabilityError(
            :response_evidence,
            "response evidence currently supports Normal and Poisson " *
            "factors; got $(typeof(base)) for `$response`"))
    _weighted_site_factor(factor) === nothing || throw(CapabilityError(
        :response_evidence,
        "response evidence cannot wrap a weighted factor for `$response`; " *
        "apply weighting outside the evidence wrapper"))
    discrete = base isa PoissonSiteFactor
    _factor_validate_evidence_bound(
        factor.lower, bindings, response, "evidence lower bound", discrete)
    _factor_validate_evidence_bound(
        factor.upper, bindings, response, "evidence upper bound", discrete)
    nothing
end

function _group_input_names(declaration::Model)
    Tuple(unique(group_input(parameter)
        for parameter in values(declaration.parameters)
        if parameter isa Union{
            GroupedNormalParameter,GroupedStandardNormalParameter}))
end

_factor_arguments(::StandardNormalSiteFactor) = ()
_factor_arguments(::LKJCholeskySiteFactor) = ()
_factor_arguments(factor::NormalSiteFactor) =
    (factor.location, factor.scale)
_factor_arguments(factor::StudentTSiteFactor) =
    (factor.degrees_freedom, factor.location, factor.scale)
_factor_arguments(factor::ExponentialSiteFactor) = (factor.scale,)
_factor_arguments(factor::BernoulliLogitSiteFactor) = (factor.logit,)
_factor_arguments(factor::PoissonSiteFactor) = (factor.rate,)
_factor_arguments(factor::ZeroInflatedPoissonSiteFactor) =
    (factor.rate, factor.zero_inflation)
_factor_arguments(factor::NegativeBinomial2SiteFactor) =
    (factor.mean, factor.precision)
_factor_arguments(factor::BetaBinomial2SiteFactor) =
    (factor.trials, factor.mean, factor.precision)
_factor_arguments(factor::BinomialLogitSiteFactor) =
    (factor.trials, factor.logit)
_factor_arguments(factor::CategoricalLogitSiteFactor) = factor.logits
_factor_arguments(factor::WeightedSiteFactor) =
    (_factor_arguments(factor.factor)..., factor.values)
_factor_arguments(factor::EvidenceSiteFactor) =
    (_factor_arguments(factor.factor)..., factor.lower, factor.upper)

_factor_value_is_row(::LiteralValue, graph, bindings) = false
_factor_value_is_row(::InputValue{Name}, graph, bindings) where {Name} =
    getproperty(bindings, Name) isa AbstractVector
function _factor_value_is_row(::NodeValue{Name}, graph, bindings) where {Name}
    node = getproperty(graph.nodes, Name)
    node isa Union{ExpFactorNode,LogFactorNode,LogisticFactorNode} &&
        return _factor_value_is_row(node.input, graph, bindings)
    node isa Union{
        CenterFactorNode,ZScaleFactorNode,AffineFactorNode,
        GroupGatherFactorNode,RowProductFactorNode,
        GroupedAffineFactorNode}
end
_factor_value_is_row(::SiteValue{Name}, graph, bindings) where {Name} =
    getproperty(graph.sites, Name).shape isa
        Union{BlockSiteShape,BroadcastSiteShape}

function _fit_factor_nodes(graph::FactorGraph, bindings::NamedTuple)
    names = Tuple(name for (name, node) in pairs(graph.nodes)
                  if node isa Union{CenterFactorNode,ZScaleFactorNode})
    values = map(names) do name
        node = getproperty(graph.nodes, name)
        node.input isa InputValue || throw(CapabilityError(
            :factor_nodes,
            "fitted transform node `$name` requires one bound vector input"))
        input_name = input_value_name(node.input)
        predictor = getproperty(bindings, input_name)
        predictor isa AbstractVector || throw(ArgumentError(
            "fitted transform input `$input_name` must be a vector"))
        if node isa CenterFactorNode
            FittedCenter(BRM._native_ppl_fit_center(predictor, input_name))
        else
            fit = BRM._native_ppl_fit_zscale(predictor, input_name)
            FittedZScale(fit.mean, fit.scale)
        end
    end
    NamedTuple{names}(values)
end

function _factor_fitted_nodes(graph::FactorGraph, bindings::NamedTuple,
                              fitted_nodes)
    expected = Tuple(name for (name, node) in pairs(graph.nodes)
                     if node isa Union{CenterFactorNode,ZScaleFactorNode})
    fitted_nodes === nothing && return _fit_factor_nodes(graph, bindings)
    fitted_nodes isa NamedTuple || throw(ArgumentError(
        "native PPL fitted factor constants must be a NamedTuple"))
    keys(fitted_nodes) == expected || throw(CapabilityError(
        :fitted_nodes,
        "frozen fitted-node identities $(keys(fitted_nodes)) do not match " *
        "the rebound graph identities $expected"))
    for name in expected
        node = getproperty(graph.nodes, name)
        fitted = getproperty(fitted_nodes, name)
        node isa CenterFactorNode && fitted isa FittedCenter ||
        node isa ZScaleFactorNode && fitted isa FittedZScale || throw(
            CapabilityError(
                :fitted_nodes,
                "frozen constants for `$name` do not match its transform"))
    end
    fitted_nodes
end

_factor_raw_category_values(raw) = raw
_factor_raw_category_values(raw::BRM.CA.CategoricalVector) =
    BRM.CA.unwrap.(raw)

function _factor_fit_category_levels(raw)
    raw isa BRM.CA.CategoricalVector && return Tuple(BRM.CA.levels(raw))
    Tuple(sort(unique(raw)))
end

function _factor_initial_category_levels(raw, class_count::Int)
    values = _factor_raw_category_values(raw)
    if any(value -> value isa Real, values)
        all(value -> value isa Real && BRM._native_ppl_is_count(value) &&
                     1 <= value <= class_count, values) ||
            throw(ArgumentError(
                "native PPL CategoricalLogit numeric responses must be " *
                "integer class codes in 1:$class_count"))
        return Tuple(1:class_count)
    end
    _factor_fit_category_levels(raw)
end

function _factor_apply_category_levels(levels, raw, response::Symbol)
    mapping = Dict(level => index for (index, level) in enumerate(levels))
    values = _factor_raw_category_values(raw)
    map(values) do value
        haskey(mapping, value) || throw(ArgumentError(
            "native PPL CategoricalLogit response `$response` contains " *
            "unseen level `$value`; fitted levels are $(collect(levels))"))
        mapping[value]
    end
end

function _factor_categorical_conditions(
        declaration::Model, conditions::NamedTuple, fitted_levels)
    categorical_names = Tuple(
        name for (name, observation) in pairs(declaration.observations)
        if base_observation(scalar_observation(observation)) isa
            CategoricalLogitObservation)
    isempty(categorical_names) && return conditions, (;)
    length(categorical_names) == 1 || throw(CapabilityError(
        :categorical_outputs,
        "native PPL currently supports one CategoricalLogit response"))
    response = only(categorical_names)
    observation = base_observation(scalar_observation(
        getproperty(declaration.observations, response)))
    class_count = length(observation_dependencies(observation)) + 1
    levels = if fitted_levels !== nothing &&
                hasproperty(fitted_levels, response)
        getproperty(fitted_levels, response)
    elseif hasproperty(conditions, response)
        _factor_initial_category_levels(
            getproperty(conditions, response), class_count)
    else
        Tuple(1:class_count)
    end
    length(levels) == class_count || throw(ArgumentError(
        "native PPL CategoricalLogit response `$response` needs " *
        "$class_count fitted levels for $(class_count - 1) non-reference " *
        "predictors; got $(collect(levels))"))
    outcome_levels = NamedTuple{(response,)}((Tuple(levels),))
    hasproperty(conditions, response) || return conditions, outcome_levels
    encoded = _factor_apply_category_levels(
        levels, getproperty(conditions, response), response)
    merge(conditions, NamedTuple{(response,)}((encoded,))), outcome_levels
end

function _bind_factor_plan(declaration::Model, bindings, conditions;
                           group_levels=(;), new_groups::Symbol=:error,
                           fitted_nodes=nothing, outcome_levels=nothing)
    input_axis_length = nothing
    group_inputs = Set(_group_input_names(declaration))
    for (name, declaration_input) in pairs(declaration.inputs)
        input_role(declaration_input) in (:value, :predictor) || throw(CapabilityError(
            :factor_inputs,
            "multi-latent factor input `$name` must be a generic value " *
            "port or predictor"))
        value = getproperty(bindings, name)
        if value isa AbstractVector
            isempty(value) && throw(ArgumentError(
                "multi-latent factor input `$name` cannot be empty"))
            if name in group_inputs
                any(ismissing, value) && throw(ArgumentError(
                    "multi-latent factor group input `$name` cannot contain " *
                    "missing values"))
            else
                all(element -> element isa Real, value) || throw(ArgumentError(
                    "multi-latent factor input `$name` must contain real values"))
            end
            input_axis_length === nothing && (input_axis_length = length(value))
            length(value) == input_axis_length || throw(DimensionMismatch(
                "multi-latent factor vector inputs must have equal lengths"))
        elseif !(value isa Real)
            throw(ArgumentError(
                "multi-latent factor input `$name` must bind a real scalar " *
                "or vector; got $(typeof(value))"))
        end
    end
    canonical_conditions = _canonical_factor_conditions(
        declaration, conditions)
    canonical_conditions, outcome_levels = _factor_categorical_conditions(
        declaration, canonical_conditions, outcome_levels)
    graph = factor_graph(
        declaration; conditions=canonical_conditions, bindings,
        group_levels, new_groups)
    fitted_nodes = _factor_fitted_nodes(graph, bindings, fitted_nodes)
    for (name, node) in pairs(graph.nodes)
        node isa Union{
            CenterFactorNode,ZScaleFactorNode,
            ExpFactorNode,LogFactorNode,LogisticFactorNode,
            AffineFactorNode,GroupGatherFactorNode,
            RowProductFactorNode,GroupedAffineFactorNode,
        } ||
            throw(CapabilityError(
            :factor_nodes,
            "multi-latent factor node `$name` has unsupported operation " *
            "$(typeof(node)); the current executable slice accepts scalar " *
            "fitted transforms and row-valued exp, log, logistic, affine, " *
            "group-gather, product, and grouped-affine nodes"))
        if node isa Union{CenterFactorNode,ZScaleFactorNode}
            node.input isa InputValue || throw(CapabilityError(
                :factor_nodes,
                "fitted transform node `$name` requires one bound input"))
        end
        if node isa LogFactorNode
            node.input isa InputValue || throw(CapabilityError(
                :factor_nodes,
                "log node `$name` currently requires one bound input"))
            input_name = input_value_name(node.input)
            input = getproperty(bindings, input_name)
            all(value -> value isa Real && isfinite(value) &&
                         value > zero(value),
                input isa AbstractVector ? input : (input,)) || throw(
                ArgumentError(
                    "native PPL log input `$input_name` must contain finite " *
                    "positive values"))
        end
        if node isa ExpFactorNode && node.input isa SiteValue
            source_name = site_value_name(node.input)
            source_site = getproperty(graph.sites, source_name)
            source_site.shape isa BlockSiteShape && throw(CapabilityError(
                :factor_shape,
                "exp node `$name` cannot consume block site `$source_name` " *
                "directly; lower it through a typed materializing node"))
        end
        if node isa AffineFactorNode
            coefficient_name = site_value_name(node.coefficients)
            coefficient_site = getproperty(graph.sites, coefficient_name)
            coefficient_site.shape isa Union{ScalarSiteShape,BlockSiteShape} ||
                throw(CapabilityError(
                :factor_nodes,
                "affine node `$name` requires a scalar or block coefficient site"))
            coefficient_site.factor isa StandardNormalSiteFactor || throw(
                CapabilityError(
                    :factor_nodes,
                    "affine node `$name` currently requires standard-normal " *
                    "coefficients"))
            coefficient_count = length(node.inputs) +
                (affine_has_intercept(node) ? 1 : 0)
            length(coefficient_site.coordinate_keys) == coefficient_count ||
                throw(CapabilityError(
                    :factor_nodes,
                    "affine node `$name` has the wrong coefficient count"))
            for argument in (node.inputs..., node.offsets...)
                argument isa SiteValue || continue
                dependency = getproperty(
                    graph.sites, site_value_name(argument))
                dependency.shape isa BroadcastSiteShape && continue
                dependency.shape isa ScalarSiteShape || throw(CapabilityError(
                    :factor_shape,
                    "affine node `$name` cannot consume non-scalar site " *
                    "`$(site_value_name(argument))` directly"))
            end
        elseif node isa RowProductFactorNode
            for argument in (node.left, node.right)
                argument isa SiteValue || continue
                dependency_name = site_value_name(argument)
                dependency = getproperty(graph.sites, dependency_name)
                dependency.shape isa BlockSiteShape || continue
                throw(CapabilityError(
                    :factor_shape,
                    "row-product node `$name` cannot consume block site " *
                    "`$dependency_name` directly; lower it through a typed " *
                    "materializing node such as group_gather"))
            end
        end
        if node isa GroupGatherFactorNode
            values_name = site_value_name(node.values)
            values_site = getproperty(graph.sites, values_name)
            values_site.shape isa BlockSiteShape || throw(CapabilityError(
                :factor_nodes,
                "group gather node `$name` requires a block latent site"))
            group_name = input_value_name(node.group)
            group_name in group_inputs || throw(CapabilityError(
                :factor_nodes,
                "group gather node `$name` requires a fitted group input"))
        end
        if node isa GroupedAffineFactorNode
            standardized_name = site_value_name(node.standardized)
            scales_name = site_value_name(node.scales)
            correlation_name = site_value_name(node.correlation)
            standardized_declaration = getproperty(
                declaration.parameters, standardized_name)
            scales_declaration = getproperty(
                declaration.parameters, scales_name)
            correlation_declaration = getproperty(
                declaration.parameters, correlation_name)
            standardized_declaration isa GroupedStandardNormalParameter ||
                throw(CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` requires grouped " *
                    "standard-normal coordinates"))
            scales_declaration isa Parameter || throw(CapabilityError(
                :factor_nodes,
                "grouped affine node `$name` requires a positive scale block"))
            correlation_declaration isa CholeskyCorrelationParameter ||
                throw(CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` requires a Cholesky " *
                    "correlation parameter"))
            coefficients = group_coefficients(standardized_declaration)
            length(coefficients) >= 2 || throw(CapabilityError(
                :factor_nodes,
                "grouped affine node `$name` requires at least two " *
                "correlated coefficients"))
            scales_declaration.axis_keys == coefficients || throw(
                CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` scale keys must match its " *
                    "group coefficient keys"))
            correlation_coefficients(correlation_declaration) == coefficients ||
                throw(CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` correlation keys must match " *
                    "its group coefficient keys"))
            grouped_coefficient_count(node) == length(coefficients) || throw(
                CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` full coefficient count must " *
                    "match its group coefficient axis"))
            indices = grouped_coefficient_indices(node)
            length(node.predictors) == length(indices) || throw(
                CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` needs one predictor per " *
                    "selected group coefficient"))
            all(index -> index in eachindex(coefficients), indices) || throw(
                CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` selects a coefficient " *
                    "outside its group coefficient axis"))
            for predictor in node.predictors
                predictor isa SiteValue || continue
                predictor_name = site_value_name(predictor)
                predictor_site = getproperty(graph.sites, predictor_name)
                predictor_site.shape isa BlockSiteShape || continue
                throw(CapabilityError(
                    :factor_shape,
                    "grouped affine node `$name` cannot consume block site " *
                    "`$predictor_name` as a row predictor; lower it through " *
                    "a typed materializing node"))
            end
            group_input(standardized_declaration) ==
                input_value_name(node.group) || throw(CapabilityError(
                    :factor_nodes,
                    "grouped affine node `$name` must use its grouped " *
                    "coordinate site's fitted group input"))
            scales_site = getproperty(graph.sites, scales_name)
            scales_site.support isa PositiveSupport &&
                scales_site.factor isa ExponentialSiteFactor &&
                scales_site.shape isa BlockSiteShape || throw(
                    CapabilityError(
                        :factor_nodes,
                        "grouped affine node `$name` scales must be a positive " *
                        "Exponential-prior block"))
            correlation_site = getproperty(graph.sites, correlation_name)
            dimension = length(coefficients)
            correlation_site.support isa
                CholeskyCorrelationSupport{dimension} &&
                correlation_site.factor isa LKJCholeskySiteFactor &&
                correlation_site.activity isa FreeSite || throw(
                    CapabilityError(
                        :factor_nodes,
                        "grouped affine node `$name` requires one free " *
                        "$dimension-dimensional LKJ Cholesky factor"))
        end
    end
    for (name, site) in pairs(graph.sites)
        base_factor = base_site_factor(site.factor)
        if site.shape isa ScalarSiteShape
            for argument in _factor_arguments(site.factor)
                _factor_value_is_row(argument, graph, bindings) || continue
                throw(CapabilityError(
                    :factor_shape,
                    "scalar factor site `$name` cannot consume a row-valued " *
                    "argument"))
            end
        end
        for argument in _factor_arguments(site.factor)
            argument isa SiteValue || continue
            dependency_name = site_value_name(argument)
            dependency = getproperty(graph.sites, dependency_name)
            dependency.shape isa BlockSiteShape || continue
            throw(CapabilityError(
                :factor_shape,
                "factor site `$name` cannot consume block site " *
                "`$dependency_name` directly; lower it through a typed " *
                "materializing node"))
        end
        if base_factor isa NormalSiteFactor
            scale = base_factor.scale
            if scale isa NodeValue
                scale_name = node_value_name(scale)
                getproperty(graph.nodes, scale_name) isa ExpFactorNode ||
                    throw(CapabilityError(
                        :factor_scale,
                        "Normal site `$name` node scale `$scale_name` must " *
                        "be produced by `exp`"))
            elseif scale isa SiteValue
                scale_name = site_value_name(scale)
                getproperty(graph.sites, scale_name).support isa
                    PositiveSupport || throw(CapabilityError(
                        :factor_scale,
                        "Normal site `$name` scale site `$scale_name` must " *
                        "have positive support"))
            elseif scale isa InputValue
                scale_name = input_value_name(scale)
                values = getproperty(bindings, scale_name)
                all(value -> value isa Real && isfinite(value) &&
                             value > zero(value),
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "Normal site `$name` input scale `$scale_name` must " *
                        "contain finite positive values"))
            elseif scale isa LiteralValue
                isfinite(scale.value) && scale.value > zero(scale.value) ||
                    throw(ArgumentError(
                        "Normal site `$name` literal scale must be finite " *
                        "and positive"))
            end
        elseif base_factor isa StudentTSiteFactor
            for (label, argument) in
                    (("degrees of freedom", base_factor.degrees_freedom),
                     ("scale", base_factor.scale))
                if argument isa NodeValue
                    argument_name = node_value_name(argument)
                    getproperty(graph.nodes, argument_name) isa
                        ExpFactorNode || throw(CapabilityError(
                            :factor_support,
                            "Student-t site `$name` $label node " *
                            "`$argument_name` must be produced by `exp`"))
                elseif argument isa SiteValue
                    argument_name = site_value_name(argument)
                    getproperty(graph.sites, argument_name).support isa
                        PositiveSupport || throw(CapabilityError(
                            :factor_support,
                            "Student-t site `$name` $label site " *
                            "`$argument_name` must have positive support"))
                elseif argument isa InputValue
                    argument_name = input_value_name(argument)
                    values = getproperty(bindings, argument_name)
                    all(value -> value isa Real && isfinite(value) &&
                                 value > zero(value),
                        values isa AbstractVector ? values : (values,)) ||
                        throw(ArgumentError(
                            "Student-t $label input `$argument_name` must " *
                            "contain finite positive values"))
                elseif argument isa LiteralValue
                    value = argument.value
                    isfinite(value) && value > zero(value) || throw(
                        ArgumentError(
                            "Student-t literal $label must be finite and " *
                            "positive"))
                else
                    throw(CapabilityError(
                        :factor_support,
                        "Student-t site `$name` has unsupported $label " *
                        "source $(typeof(argument))"))
                end
            end
        elseif base_factor isa ZeroInflatedPoissonSiteFactor
            rate = base_factor.rate
            zero_inflation = base_factor.zero_inflation
            if rate isa NodeValue
                rate_name = node_value_name(rate)
                getproperty(graph.nodes, rate_name) isa ExpFactorNode || throw(
                    CapabilityError(
                        :factor_support,
                        "zero-inflated Poisson site `$name` rate node " *
                        "`$rate_name` must be produced by `exp`"))
            elseif rate isa InputValue
                rate_name = input_value_name(rate)
                values = getproperty(bindings, rate_name)
                all(value -> value isa Real && isfinite(value) &&
                             value >= zero(value),
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "zero-inflated Poisson rate input `$rate_name` must " *
                        "contain finite nonnegative values"))
            elseif rate isa LiteralValue
                isfinite(rate.value) && rate.value >= zero(rate.value) ||
                    throw(ArgumentError(
                        "zero-inflated Poisson literal rate must be finite " *
                        "and nonnegative"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "zero-inflated Poisson site `$name` has unsupported " *
                    "rate source $(typeof(rate))"))
            end
            if zero_inflation isa NodeValue
                probability_name = node_value_name(zero_inflation)
                getproperty(graph.nodes, probability_name) isa
                    LogisticFactorNode || throw(CapabilityError(
                        :factor_support,
                        "zero-inflated Poisson site `$name` probability node " *
                        "`$probability_name` must be produced by `logistic`"))
            elseif zero_inflation isa InputValue
                probability_name = input_value_name(zero_inflation)
                values = getproperty(bindings, probability_name)
                all(value -> value isa Real && isfinite(value) &&
                             zero(value) <= value <= one(value),
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "zero-inflated Poisson probability input " *
                        "`$probability_name` must lie in [0, 1]"))
            elseif zero_inflation isa LiteralValue
                value = zero_inflation.value
                isfinite(value) && zero(value) <= value <= one(value) ||
                    throw(ArgumentError(
                        "zero-inflated Poisson literal probability must lie " *
                        "in [0, 1]"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "zero-inflated Poisson site `$name` has unsupported " *
                    "probability source $(typeof(zero_inflation))"))
            end
        elseif base_factor isa NegativeBinomial2SiteFactor
            for (label, argument) in
                    (("mean", base_factor.mean),
                     ("precision", base_factor.precision))
                if argument isa NodeValue
                    argument_name = node_value_name(argument)
                    getproperty(graph.nodes, argument_name) isa
                        ExpFactorNode || throw(CapabilityError(
                            :factor_support,
                            "NegativeBinomial2 site `$name` $label node " *
                            "`$argument_name` must be produced by `exp`"))
                elseif argument isa InputValue
                    argument_name = input_value_name(argument)
                    values = getproperty(bindings, argument_name)
                    all(value -> value isa Real && isfinite(value) &&
                                 value > zero(value),
                        values isa AbstractVector ? values : (values,)) ||
                        throw(ArgumentError(
                            "NegativeBinomial2 $label input `$argument_name` " *
                            "must contain finite positive values"))
                elseif argument isa LiteralValue
                    value = argument.value
                    isfinite(value) && value > zero(value) || throw(
                        ArgumentError(
                            "NegativeBinomial2 literal $label must be " *
                            "finite and positive"))
                else
                    throw(CapabilityError(
                        :factor_support,
                        "NegativeBinomial2 site `$name` has unsupported " *
                        "$label source $(typeof(argument))"))
                end
            end
        elseif base_factor isa BetaBinomial2SiteFactor
            trials = base_factor.trials
            if trials isa InputValue
                trials_name = input_value_name(trials)
                values = getproperty(bindings, trials_name)
                all(BRM._native_ppl_is_count,
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "BetaBinomial2 trials input `$trials_name` must " *
                        "contain nonnegative integer-valued counts"))
            elseif trials isa LiteralValue
                BRM._native_ppl_is_count(trials.value) || throw(ArgumentError(
                    "BetaBinomial2 literal trials must be a nonnegative " *
                    "integer-valued count"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "BetaBinomial2 site `$name` has unsupported trials " *
                    "source $(typeof(trials))"))
            end
            mean = base_factor.mean
            if mean isa NodeValue
                mean_name = node_value_name(mean)
                getproperty(graph.nodes, mean_name) isa LogisticFactorNode ||
                    throw(CapabilityError(
                        :factor_support,
                        "BetaBinomial2 site `$name` mean node `$mean_name` " *
                        "must be produced by `logistic`"))
            elseif mean isa InputValue
                mean_name = input_value_name(mean)
                values = getproperty(bindings, mean_name)
                all(value -> value isa Real && isfinite(value) &&
                             zero(value) < value < one(value),
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "BetaBinomial2 mean input `$mean_name` must lie " *
                        "strictly in (0, 1)"))
            elseif mean isa LiteralValue
                value = mean.value
                isfinite(value) && zero(value) < value < one(value) || throw(
                    ArgumentError(
                        "BetaBinomial2 literal mean must lie strictly in " *
                        "(0, 1)"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "BetaBinomial2 site `$name` has unsupported mean " *
                    "source $(typeof(mean))"))
            end
            precision = base_factor.precision
            if precision isa NodeValue
                precision_name = node_value_name(precision)
                getproperty(graph.nodes, precision_name) isa ExpFactorNode ||
                    throw(CapabilityError(
                        :factor_support,
                        "BetaBinomial2 site `$name` precision node " *
                        "`$precision_name` must be produced by `exp`"))
            elseif precision isa InputValue
                precision_name = input_value_name(precision)
                values = getproperty(bindings, precision_name)
                all(value -> value isa Real && isfinite(value) &&
                             value > zero(value),
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "BetaBinomial2 precision input `$precision_name` " *
                        "must contain finite positive values"))
            elseif precision isa LiteralValue
                value = precision.value
                isfinite(value) && value > zero(value) || throw(ArgumentError(
                    "BetaBinomial2 literal precision must be finite and " *
                    "positive"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "BetaBinomial2 site `$name` has unsupported precision " *
                    "source $(typeof(precision))"))
            end
        elseif base_factor isa BinomialLogitSiteFactor
            trials = base_factor.trials
            if trials isa InputValue
                trials_name = input_value_name(trials)
                values = getproperty(bindings, trials_name)
                all(BRM._native_ppl_is_count,
                    values isa AbstractVector ? values : (values,)) || throw(
                    ArgumentError(
                        "BinomialLogit trials input `$trials_name` must " *
                        "contain nonnegative integer-valued counts"))
            elseif trials isa LiteralValue
                BRM._native_ppl_is_count(trials.value) || throw(ArgumentError(
                    "BinomialLogit literal trials must be a nonnegative " *
                    "integer-valued count"))
            else
                throw(CapabilityError(
                    :factor_support,
                    "BinomialLogit site `$name` has unsupported trials " *
                    "source $(typeof(trials))"))
            end
        end
        site.shape isa BlockSiteShape &&
            !(base_factor isa Union{
                StandardNormalSiteFactor,NormalSiteFactor,
                ExponentialSiteFactor,LKJCholeskySiteFactor}) &&
            throw(CapabilityError(
                :factor_shape,
                "multi-latent block site `$name` has an unsupported factor"))
        site.factor isa Union{
            StandardNormalSiteFactor,NormalSiteFactor,ExponentialSiteFactor,
            LKJCholeskySiteFactor,StudentTSiteFactor,BernoulliLogitSiteFactor,
            PoissonSiteFactor,ZeroInflatedPoissonSiteFactor,
            NegativeBinomial2SiteFactor,BetaBinomial2SiteFactor,
            BinomialLogitSiteFactor,CategoricalLogitSiteFactor,
            WeightedSiteFactor,EvidenceSiteFactor,
        } || throw(CapabilityError(
            :factor_family,
            "multi-latent factor site `$name` has unsupported factor " *
            "$(typeof(site.factor))"))
        base_factor isa Union{
            BernoulliLogitSiteFactor,PoissonSiteFactor,
            ZeroInflatedPoissonSiteFactor,NegativeBinomial2SiteFactor,
            BetaBinomial2SiteFactor,BinomialLogitSiteFactor,
            CategoricalLogitSiteFactor,
        } &&
            !(site.shape isa BroadcastSiteShape) && throw(CapabilityError(
                :factor_shape,
                "discrete factor site `$name` must be a broadcast output"))
    end
    broadcast_sites = Tuple(
        name for (name, site) in pairs(graph.sites)
        if site.shape isa BroadcastSiteShape)
    isempty(broadcast_sites) && throw(CapabilityError(
        :factor_outputs,
        "multi-latent factor executor requires at least one broadcast " *
        "stochastic output"))
    observation_count = input_axis_length
    for output_site in broadcast_sites
        hasproperty(canonical_conditions, output_site) || continue
        response = getproperty(canonical_conditions, output_site)
        response isa AbstractVector || throw(ArgumentError(
            "native PPL broadcast condition `$output_site` must be a vector"))
        isempty(response) && throw(ArgumentError(
            "native PPL broadcast condition `$output_site` cannot be empty"))
        observation_count === nothing &&
            (observation_count = length(response))
        length(response) == observation_count || throw(DimensionMismatch(
            "native PPL broadcast outputs and vector inputs must share the " *
            "same observation axis; `$output_site` has $(length(response)) " *
            "rows, expected $observation_count"))
    end
    observation_keys = observation_count === nothing ?
        Base.OneTo(0) : Base.OneTo(observation_count)
    for output_site in broadcast_sites
        output_factor = getproperty(graph.sites, output_site).factor
        weight_factor = _weighted_site_factor(output_factor)
        weight_factor === nothing || _factor_validate_weight(
            weight_factor, bindings, output_site, length(observation_keys))
        evidence_factor = _evidence_site_factor(output_factor)
        evidence_factor === nothing || _factor_validate_evidence(
            evidence_factor, bindings, output_site)
    end
    site_names = Tuple(keys(graph.sites))
    site_indices = NamedTuple{site_names}(
        ntuple(identity, length(site_names)))
    node_names = Tuple(keys(graph.nodes))
    node_indices = NamedTuple{node_names}(
        ntuple(identity, length(node_names)))
    grouped_node_names = Tuple(name for (name, node) in pairs(graph.nodes)
        if node isa Union{GroupGatherFactorNode,GroupedAffineFactorNode})
    grouped_site_names = Tuple(name for (name, parameter) in pairs(
        declaration.parameters) if parameter isa Union{
            GroupedNormalParameter,GroupedStandardNormalParameter})
    generated_level_values = map(grouped_site_names) do site_name
        parameter = getproperty(declaration.parameters, site_name)
        group_name = group_input(parameter)
        observed_levels = _group_levels(
            getproperty(bindings, group_name), group_name)
        fitted_site = getproperty(graph.sites, site_name)
        fitted_levels = Tuple(unique(
            key.level for key in fitted_site.coordinate_keys))
        Tuple(level for level in observed_levels if level ∉ fitted_levels)
    end
    generated_group_levels = NamedTuple{grouped_site_names}(
        generated_level_values)
    next_generated_index = 1
    generated_index_values = map(
        grouped_site_names, generated_level_values) do site_name, levels
        parameter = getproperty(declaration.parameters, site_name)
        coefficient_count = parameter isa GroupedStandardNormalParameter ?
            length(group_coefficients(parameter)) : 1
        count = length(levels) * coefficient_count
        indices = next_generated_index:(next_generated_index + count - 1)
        next_generated_index += count
        indices
    end
    generated_group_indices = NamedTuple{grouped_site_names}(
        generated_index_values)
    grouped_indices = map(grouped_node_names) do name
        node = getproperty(graph.nodes, name)
        site_name = node isa GroupGatherFactorNode ?
            site_value_name(node.values) :
            site_value_name(node.standardized)
        site = getproperty(graph.sites, site_name)
        levels = Tuple(unique(key.level for key in site.coordinate_keys))
        generated_levels = getproperty(generated_group_levels, site_name)
        groups = getproperty(bindings, input_value_name(node.group))
        Tuple(map(groups) do group
            index = findfirst(isequal(group), levels)
            index === nothing || return index
            generated_index = findfirst(isequal(group), generated_levels)
            generated_index === nothing && throw(ArgumentError(
                "native PPL group value `$group` is absent from fitted and " *
                "generated levels"))
            -generated_index
        end)
    end
    group_indices = NamedTuple{grouped_node_names}(grouped_indices)
    has_generated_groups = any(!isempty, values(generated_group_levels))
    has_generated_groups &&
        any(output_site -> hasproperty(canonical_conditions, output_site),
            broadcast_sites) && throw(
            CapabilityError(
                :new_group_activity,
                "native PPL new-group resampling is prediction-only; " *
                "conditioned likelihood evaluation requires explicit " *
                "group-effect coordinates"))
    FactorPlan(
        declaration, graph, fitted_nodes, bindings, canonical_conditions,
        outcome_levels,
        site_indices,
        node_indices, group_indices, generated_group_levels,
        generated_group_indices,
        BRM.NativePPLAxis(:observation, observation_keys), broadcast_sites)
end

function Base.show(io::IO, plan::FactorPlan)
    generated = sum(length, values(plan.generated_group_levels); init=0)
    print(io, "NativePPL.FactorPlan(", plan.graph.dimension,
          " unconstrained parameters, ", length(plan.graph.sites),
          " stochastic sites, ", generated, " generated groups, ",
          length(plan.observation_axis), " observations")
    length(factor_output_sites(plan)) > 1 && print(
        io, ", ", length(factor_output_sites(plan)), " terminal outcomes")
    print(io, ")")
end

"""Prepared, independently owned bindings for a `FactorPlan`."""
struct FactorPrepared{T,P,C}
    plan::P
    conditions::C
end

Base.eltype(::FactorPrepared{T}) where {T} = T
BRM.LogDensityProblems.dimension(prepared::FactorPrepared) =
    BRM.LogDensityProblems.dimension(prepared.plan)

"""Reusable primal storage for an ordered factor-graph evaluation."""
mutable struct FactorBuffers{T,V<:Vector{T},M<:Matrix{T}}
    values::V
    generated_group_values::V
    node_values::V
    node_rows::M
    pointwise_loglikelihood::V
end

"""Caller-owned workspace for factor-graph density and gradient execution."""
struct FactorWorkspace{T,B,G,D}
    primal::B
    gradient::G
    derivative::D
end

Base.eltype(::FactorWorkspace{T}) where {T} = T

function FactorWorkspace(prepared::FactorPrepared,
                         ::Type{T}=eltype(prepared)) where {T<:AbstractFloat}
    isconcretetype(T) || throw(ArgumentError(
        "native PPL factor workspace element type must be concrete; got $T"))
    site_values = zeros(T, length(prepared.plan.graph.sites))
    generated_group_values = zeros(
        T, sum(length, values(prepared.plan.generated_group_indices); init=0))
    node_values = zeros(T, length(prepared.plan.graph.nodes))
    node_rows = zeros(T, length(prepared.plan.graph.nodes),
                      length(prepared.plan.observation_axis))
    pointwise = zeros(T, length(prepared.plan.observation_axis))
    FactorWorkspace{T,typeof(FactorBuffers(
        site_values, generated_group_values, node_values, node_rows, pointwise)),
        Vector{T},Nothing}(
        FactorBuffers(
            site_values, generated_group_values, node_values, node_rows,
            pointwise),
        zeros(T, BRM.LogDensityProblems.dimension(prepared)), nothing)
end

function _factor_workspace end
_factor_workspace(prepared::FactorPrepared,
                  ::Type{T}) where {T<:AbstractFloat} =
    FactorWorkspace(prepared, T)
_factor_workspace(::FactorPrepared, ::Type{<:AbstractFloat}, backend) =
    throw(ArgumentError(
        "native PPL factor gradients require loading " *
        "DifferentiationInterface; only AutoEnzyme is tested, recommended, " *
        "and guaranteed"))

function _factor_prepare_condition(value::Real, name::Symbol,
                                   ::Type{T}) where {T<:AbstractFloat}
    converted = T(value)
    isfinite(converted) || throw(ArgumentError(
        "native PPL condition `$name` must be finite in $T"))
    converted
end

function _factor_prepare_condition(value::AbstractVector, name::Symbol,
                                   ::Type{T}) where {T<:AbstractFloat}
    isempty(value) && throw(ArgumentError(
        "native PPL condition `$name` cannot be empty"))
    all(element -> element isa Real, value) || throw(ArgumentError(
        "native PPL condition `$name` must contain real values"))
    converted = T.(value)
    all(isfinite, converted) || throw(ArgumentError(
        "native PPL condition `$name` must be finite in $T"))
    converted
end

_factor_prepare_condition(value, name::Symbol, ::Type{T}) where {T} =
    throw(ArgumentError(
        "native PPL condition `$name` must be a real scalar or vector; got " *
        "$(typeof(value))"))

function _factor_validate_condition_support(site::StochasticSite,
                                            value, name::Symbol)
    scalar_values = value isa AbstractVector ? value : (value,)
    if site.support isa PositiveSupport
        all(element -> element > zero(element), scalar_values) ||
            throw(ArgumentError(
                "native PPL condition `$name` must lie in positive support"))
    end
    nothing
end

function _factor_validate_observation(::AbstractSiteFactor, value,
                                      name::Symbol)
    nothing
end

_factor_validate_observation(factor::WeightedSiteFactor, value,
                             name::Symbol) =
    _factor_validate_observation(factor.factor, value, name)
_factor_validate_observation(factor::EvidenceSiteFactor, value,
                             name::Symbol) =
    _factor_validate_observation(factor.factor, value, name)

function _factor_validate_observation(::BernoulliLogitSiteFactor, value,
                                      name::Symbol)
    for (index, element) in enumerate(value)
        element == 0 || element == 1 || throw(ArgumentError(
            "native PPL BernoulliLogit response `$name` must be Bool/0/1; " *
            "got $element at row $index"))
    end
    nothing
end

function _factor_validate_observation(
        factor::CategoricalLogitSiteFactor, value, name::Symbol)
    classes = length(factor.logits) + 1
    for (index, element) in enumerate(value)
        BRM._native_ppl_is_count(element) && one(element) <= element <= classes ||
            throw(ArgumentError(
                "native PPL CategoricalLogit response `$name` must be an " *
                "integer class in 1:$classes; got $element at row $index"))
    end
    nothing
end

function _factor_validate_observation(
        ::Union{PoissonSiteFactor,ZeroInflatedPoissonSiteFactor,
                NegativeBinomial2SiteFactor,BetaBinomial2SiteFactor,
                BinomialLogitSiteFactor}, value,
        name::Symbol)
    for (index, element) in enumerate(value)
        BRM._native_ppl_is_count(element) || throw(ArgumentError(
            "native PPL Poisson response `$name` must be a nonnegative " *
            "integer-valued count representable as Int; got $element at " *
            "row $index"))
    end
    nothing
end

function _factor_validate_observation_conversion(::AbstractSiteFactor,
                                                 original, converted,
                                                 name::Symbol)
    nothing
end

_factor_validate_observation_conversion(factor::WeightedSiteFactor,
                                        original, converted,
                                        name::Symbol) =
    _factor_validate_observation_conversion(
        factor.factor, original, converted, name)
_factor_validate_observation_conversion(factor::EvidenceSiteFactor,
                                        original, converted,
                                        name::Symbol) =
    _factor_validate_observation_conversion(
        factor.factor, original, converted, name)

function _factor_validate_observation_conversion(
        ::Union{PoissonSiteFactor,ZeroInflatedPoissonSiteFactor,
                NegativeBinomial2SiteFactor,BetaBinomial2SiteFactor,
                BinomialLogitSiteFactor},
        original, converted, name::Symbol)
    for index in eachindex(original, converted)
        converted[index] == original[index] || throw(ArgumentError(
            "native PPL Poisson response `$name` count $(original[index]) " *
            "at row $index cannot be represented exactly as " *
            "$(eltype(converted))"))
    end
    nothing
end

function _factor_validate_observation_conversion(
        ::CategoricalLogitSiteFactor, original, converted, name::Symbol)
    for index in eachindex(original, converted)
        converted[index] == original[index] || throw(ArgumentError(
            "native PPL CategoricalLogit response `$name` class " *
            "$(original[index]) at row $index cannot be represented exactly " *
            "as $(eltype(converted))"))
    end
    nothing
end

function _factor_validate_beta_binomial_response(
        factor::BetaBinomial2SiteFactor, response, plan,
        name::Symbol, ::Type{T}) where {T}
    raw_trials = factor.trials isa InputValue ?
        getproperty(plan.bindings, input_value_name(factor.trials)) :
        factor.trials.value
    for index in eachindex(response)
        raw = raw_trials isa AbstractVector ? raw_trials[index] : raw_trials
        converted = T(raw)
        converted == raw || throw(ArgumentError(
            "native PPL BetaBinomial2 trials $raw at row $index cannot be " *
            "represented exactly as $T"))
        trials = Int(converted)
        response[index] <= trials || throw(ArgumentError(
            "native PPL BetaBinomial2 response `$name` at row $index " *
            "exceeds its trial count $trials"))
    end
    nothing
end

function _factor_validate_binomial_logit_response(
        factor::BinomialLogitSiteFactor, response, plan,
        name::Symbol, ::Type{T}) where {T}
    raw_trials = factor.trials isa InputValue ?
        getproperty(plan.bindings, input_value_name(factor.trials)) :
        factor.trials.value
    for index in eachindex(response)
        raw = raw_trials isa AbstractVector ? raw_trials[index] : raw_trials
        converted = T(raw)
        converted == raw || throw(ArgumentError(
            "native PPL BinomialLogit trials $raw at row $index cannot be " *
            "represented exactly as $T"))
        trials = Int(converted)
        response[index] <= trials || throw(ArgumentError(
            "native PPL BinomialLogit response `$name` at row $index " *
            "exceeds its $trials trials"))
    end
    nothing
end

_factor_prepared_bound_at(::LiteralValue{Nothing}, index, plan,
                          ::Type{T}) where {T} = nothing
_factor_prepared_bound_at(bound::LiteralValue, index, plan,
                          ::Type{T}) where {T} = T(bound.value)
function _factor_prepared_bound_at(::InputValue{Name}, index, plan,
                                   ::Type{T}) where {Name,T}
    raw = getproperty(plan.bindings, Name)
    T(raw isa AbstractVector ? raw[index] : raw)
end
_factor_prepared_bound_at(::Union{NodeValue,SiteValue}, index, plan,
                          ::Type{T}) where {T} = nothing

function _factor_validate_evidence_response(factor::EvidenceSiteFactor,
                                            response::AbstractVector,
                                            plan, name::Symbol,
                                            ::Type{T}) where {T}
    kind = evidence_kind(factor.evidence)
    for index in eachindex(response)
        lower = _factor_prepared_bound_at(factor.lower, index, plan, T)
        upper = _factor_prepared_bound_at(factor.upper, index, plan, T)
        lower !== nothing && upper !== nothing && lower >= upper && throw(
            ArgumentError(
                "native PPL response evidence for `$name` has lower bound " *
                "$lower not below upper bound $upper at row $index"))
        value = response[index]
        if kind === :interval_censored
            upper === nothing || value < upper || throw(ArgumentError(
                "native PPL interval evidence for `$name` requires an open " *
                "lower endpoint below its upper endpoint at row $index"))
        else
            lower === nothing || value >= lower || throw(ArgumentError(
                "native PPL $kind response `$name` lies below its lower " *
                "bound at row $index"))
            upper === nothing || value <= upper || throw(ArgumentError(
                "native PPL $kind response `$name` lies above its upper " *
                "bound at row $index"))
        end
    end
    nothing
end


function _factor_validate_evidence_representation(
        factor::EvidenceSiteFactor, plan, name::Symbol,
        ::Type{T}) where {T}
    discrete = base_site_factor(factor) isa PoissonSiteFactor
    for (label, bound) in (("lower", factor.lower), ("upper", factor.upper))
        raw = if bound isa LiteralValue
            bound.value
        elseif bound isa InputValue
            getproperty(plan.bindings, input_value_name(bound))
        else
            continue
        end
        raw === nothing && continue
        values = raw isa AbstractVector ? raw : (raw,)
        for value in values
            converted = T(value)
            isfinite(converted) || throw(ArgumentError(
                "native PPL evidence $label bound for `$name` cannot be " *
                "represented as finite $T"))
            discrete && converted != value && throw(ArgumentError(
                "native PPL discrete evidence $label bound $value for " *
                "`$name` cannot be represented exactly as $T"))
        end
    end
    nothing
end

function prepare(plan::FactorPlan; T::Type{<:AbstractFloat}=Float64)
    isconcretetype(T) || throw(ArgumentError(
        "native PPL factor prepared element type must be concrete; got $T"))
    for (name, fitted) in pairs(plan.fitted_nodes)
        mean = T(fitted.mean)
        isfinite(mean) || throw(ArgumentError(
            "native PPL fitted mean for `$name` cannot be represented as $T"))
        if fitted isa FittedZScale
            scale = T(fitted.scale)
            isfinite(scale) && scale > zero(T) || throw(ArgumentError(
                "native PPL fitted sample SD for `$name` cannot be " *
                "represented as a finite positive $T"))
        end
    end
    group_inputs = Set(_group_input_names(plan.declaration))
    binding_names = Tuple(keys(plan.bindings))
    binding_values = map(binding_names) do name
        value = getproperty(plan.bindings, name)
        name in group_inputs && return copy(value)
        _factor_prepare_condition(value, name, T)
    end
    bindings = NamedTuple{binding_names}(binding_values)
    for output_site in factor_output_sites(plan)
        output_factor = getproperty(plan.graph.sites, output_site).factor
        weight_factor = _weighted_site_factor(output_factor)
        if weight_factor !== nothing
            source = input_value_name(weight_factor.values)
            original = getproperty(plan.bindings, source)
            converted = getproperty(bindings, source)
            kind = observation_weight_kind(weight_factor.weight)
            if kind === :frequency
                all(index -> converted[index] == original[index],
                    eachindex(original)) || throw(ArgumentError(
                        "native PPL frequency weights from `$source` cannot " *
                        "be represented exactly as $T"))
            elseif kind === :analytic
                all(>(zero(T)), converted) || throw(ArgumentError(
                    "native PPL analytic weights from `$source` cannot be " *
                    "represented as positive $T values"))
            end
        end
        evidence_factor = _evidence_site_factor(output_factor)
        evidence_factor === nothing ||
            _factor_validate_evidence_representation(
                evidence_factor, plan, output_site, T)
    end
    names = Tuple(keys(plan.conditions))
    values = map(names) do name
        original = getproperty(plan.conditions, name)
        site = getproperty(plan.graph.sites, name)
        site.shape isa BroadcastSiteShape &&
            _factor_validate_observation(site.factor, original, name)
        value = _factor_prepare_condition(
            original, name, T)
        site.shape isa BroadcastSiteShape &&
            _factor_validate_observation_conversion(
                site.factor, original, value, name)
        base_factor = base_site_factor(site.factor)
        site.shape isa BroadcastSiteShape &&
            base_factor isa BetaBinomial2SiteFactor &&
            _factor_validate_beta_binomial_response(
                base_factor, value, plan, name, T)
        site.shape isa BroadcastSiteShape &&
            base_factor isa BinomialLogitSiteFactor &&
            _factor_validate_binomial_logit_response(
                base_factor, value, plan, name, T)
        evidence_factor = _evidence_site_factor(site.factor)
        site.shape isa BroadcastSiteShape &&
            evidence_factor !== nothing &&
            _factor_validate_evidence_response(
                evidence_factor, value, plan, name, T)
        if site.shape isa BlockSiteShape
            value isa AbstractVector || throw(ArgumentError(
                "native PPL block condition `$name` must be a vector"))
            length(value) == length(site.coordinate_keys) || throw(
                DimensionMismatch(
                    "native PPL block condition `$name` has the wrong length"))
        elseif site.shape isa ScalarSiteShape
            value isa Real || throw(ArgumentError(
                "native PPL scalar condition `$name` must be a scalar"))
        end
        _factor_validate_condition_support(site, value, name)
        value
    end
    conditions = NamedTuple{names}(values)
    owned_plan = FactorPlan(
        plan.declaration, plan.graph, plan.fitted_nodes,
        bindings, conditions, plan.outcome_levels,
        plan.site_indices, plan.node_indices, plan.group_indices,
        plan.generated_group_levels, plan.generated_group_indices,
        plan.observation_axis, factor_output_sites(plan))
    FactorPrepared{T,typeof(owned_plan),typeof(conditions)}(
        owned_plan, conditions)
end

workspace(prepared::FactorPrepared,
          ::Type{T}=eltype(prepared)) where {T<:AbstractFloat} =
    _factor_workspace(prepared, T)
workspace(prepared::FactorPrepared, ::Type{T}, backend) where {T<:AbstractFloat} =
    _factor_workspace(prepared, T, backend)

@inline _factor_argument(value::LiteralValue, plan, buffers, ::Type{T}) where {T} =
    T(value.value)
@inline _factor_argument(::InputValue{Name}, plan, buffers,
                         ::Type{T}) where {Name,T} =
    T(getproperty(plan.bindings, Name))
@inline _factor_argument(::NodeValue{Name}, plan, buffers,
                         ::Type{T}) where {Name,T} =
    buffers.node_values[getproperty(plan.node_indices, Name)]
@inline _factor_argument(::SiteValue{Name}, plan, buffers,
                         ::Type{T}) where {Name,T} =
    buffers.values[getproperty(plan.site_indices, Name)]

@inline _factor_argument_at(value::LiteralValue, index, plan, buffers,
                            ::Type{T}) where {T} = T(value.value)
@inline function _factor_argument_at(::InputValue{Name}, index, plan, buffers,
                                     ::Type{T}) where {Name,T}
    value = getproperty(plan.bindings, Name)
    T(value isa AbstractVector ? value[index] : value)
end
@inline function _factor_argument_at(::NodeValue{Name}, index, plan, buffers,
                                     ::Type{T}) where {Name,T}
    node_index = getproperty(plan.node_indices, Name)
    node = getproperty(plan.graph.nodes, Name)
    node isa Union{
        CenterFactorNode,ZScaleFactorNode,
        ExpFactorNode,LogFactorNode,LogisticFactorNode,AffineFactorNode,
        GroupGatherFactorNode,RowProductFactorNode,
        GroupedAffineFactorNode,
    } ?
        buffers.node_rows[node_index, index] :
        buffers.node_values[node_index]
end
@inline _factor_argument_at(::SiteValue{Name}, index, plan, buffers,
                            ::Type{T}) where {Name,T} =
    buffers.values[getproperty(plan.site_indices, Name)]

@inline _factor_transform(::IdentityTransform, unconstrained) =
    (unconstrained, zero(unconstrained))
@inline _factor_transform(::ExpTransform, unconstrained) =
    (exp(unconstrained), unconstrained)

@inline function _factor_logsech2(value::T) where {T}
    magnitude = abs(value)
    T(2) * (log(T(2)) - magnitude - log1p(exp(-T(2) * magnitude)))
end

@inline function _factor_sech(value::T) where {T}
    magnitude = abs(value)
    twice = T(2) * exp(-magnitude)
    twice / (one(T) + exp(-T(2) * magnitude))
end

@inline function _factor_logdensity(::StandardNormalSiteFactor, value::T,
                                    plan, buffers) where {T}
    -T(0.5) * value * value - T(BRM._NATIVE_PPL_HALF_LOG2PI)
end

@inline function _factor_logdensity(factor::NormalSiteFactor, value::T,
                                    plan, buffers) where {T}
    location = _factor_argument(factor.location, plan, buffers, T)
    scale = _factor_argument(factor.scale, plan, buffers, T)
    residual = (value - location) / scale
    -T(0.5) * residual * residual - log(scale) -
        T(BRM._NATIVE_PPL_HALF_LOG2PI)
end

@inline function _factor_logdensity(factor::ExponentialSiteFactor, value::T,
                                    plan, buffers) where {T}
    scale = _factor_argument(factor.scale, plan, buffers, T)
    -log(scale) - value / scale
end

@inline function _factor_logdensity_at(factor::NormalSiteFactor, value::T,
                                       index, plan, buffers) where {T}
    location = _factor_argument_at(factor.location, index, plan, buffers, T)
    scale = _factor_argument_at(factor.scale, index, plan, buffers, T)
    scale > zero(T) || return -T(Inf)
    residual = (value - location) / scale
    -T(0.5) * residual * residual - log(scale) -
        T(BRM._NATIVE_PPL_HALF_LOG2PI)
end

@inline function _factor_student_t_log_normalizer(degrees_freedom::T) where {T}
    !isfinite(degrees_freedom) && return -T(BRM._NATIVE_PPL_HALF_LOG2PI)
    if degrees_freedom > T(1_000_000)
        inverse_df = inv(degrees_freedom)
        return -T(BRM._NATIVE_PPL_HALF_LOG2PI) -
            T(0.25) * inverse_df +
            T(1 / 24) * inverse_df * inverse_df * inverse_df
    end
    half_df = degrees_freedom / T(2)
    BRM.loggamma(half_df + T(0.5)) - BRM.loggamma(half_df) -
        T(0.5) * (log(degrees_freedom) + log(T(BRM.pi)))
end

@inline function _factor_logdensity_at(factor::StudentTSiteFactor, value::T,
                                       index, plan, buffers) where {T}
    degrees_freedom = _factor_argument_at(
        factor.degrees_freedom, index, plan, buffers, T)
    location = _factor_argument_at(
        factor.location, index, plan, buffers, T)
    scale = _factor_argument_at(
        factor.scale, index, plan, buffers, T)
    degrees_freedom > zero(T) && scale > zero(T) || return -T(Inf)
    residual = (value - location) / scale
    if !isfinite(degrees_freedom)
        return -T(0.5) * residual * residual - log(scale) -
            T(BRM._NATIVE_PPL_HALF_LOG2PI)
    end
    _factor_student_t_log_normalizer(degrees_freedom) - log(scale) -
        (degrees_freedom + one(T)) / T(2) *
        log1p(residual * residual / degrees_freedom)
end

@inline function _factor_logdensity_at(
        factor::BernoulliLogitSiteFactor, value::T,
        index, plan, buffers) where {T}
    logit = _factor_argument_at(factor.logit, index, plan, buffers, T)
    isone(value) ? -BRM._native_ppl_softplus(-logit) :
        -BRM._native_ppl_softplus(logit)
end

@inline function _factor_poisson_log_rate(factor::PoissonSiteFactor,
                                          index, plan, buffers,
                                          ::Type{T}) where {T}
    rate = factor.rate
    if rate isa NodeValue
        name = node_value_name(rate)
        node = getproperty(plan.graph.nodes, name)
        node isa ExpFactorNode && return _factor_argument_at(
            node.input, index, plan, buffers, T)
    end
    log(_factor_argument_at(rate, index, plan, buffers, T))
end

@inline function _factor_logdensity_at(factor::PoissonSiteFactor, value::T,
                                       index, plan, buffers) where {T}
    log_rate = _factor_poisson_log_rate(factor, index, plan, buffers, T)
    BRM._native_ppl_poisson_logdensity(value, log_rate)
end

@inline function _factor_zip_log_rate(
        factor::ZeroInflatedPoissonSiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    rate = factor.rate
    if rate isa NodeValue
        name = node_value_name(rate)
        node = getproperty(plan.graph.nodes, name)
        node isa ExpFactorNode && return _factor_argument_at(
            node.input, index, plan, buffers, T)
    end
    log(_factor_argument_at(rate, index, plan, buffers, T))
end

@inline function _factor_zip_log_probabilities(
        factor::ZeroInflatedPoissonSiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    probability = factor.zero_inflation
    if probability isa NodeValue
        name = node_value_name(probability)
        node = getproperty(plan.graph.nodes, name)
        if node isa LogisticFactorNode
            logit = _factor_argument_at(node.input, index, plan, buffers, T)
            return (-BRM._native_ppl_softplus(-logit),
                    -BRM._native_ppl_softplus(logit))
        end
    end
    value = _factor_argument_at(probability, index, plan, buffers, T)
    (log(value), log1p(-value))
end

@inline function _factor_logdensity_at(
        factor::ZeroInflatedPoissonSiteFactor, value::T,
        index, plan, buffers) where {T}
    log_rate = _factor_zip_log_rate(factor, index, plan, buffers, T)
    log_zero, log_count = _factor_zip_log_probabilities(
        factor, index, plan, buffers, T)
    iszero(value) && log_rate > log(floatmax(T)) && return log_zero
    poisson = BRM._native_ppl_poisson_logdensity(value, log_rate)
    iszero(value) ? _factor_logaddexp(log_zero, log_count + poisson) :
        log_count + poisson
end

@inline function _factor_nb2_log_parameter(
        argument, index, plan, buffers, ::Type{T}) where {T}
    if argument isa NodeValue
        name = node_value_name(argument)
        node = getproperty(plan.graph.nodes, name)
        node isa ExpFactorNode && return _factor_argument_at(
            node.input, index, plan, buffers, T)
    end
    log(_factor_argument_at(argument, index, plan, buffers, T))
end

@inline function _factor_nb2_logdensity(
        count::T, log_mean::T, log_precision::T) where {T}
    log_precision > log(floatmax(T)) &&
        return BRM._native_ppl_poisson_logdensity(count, log_mean)
    count_int = Int(count)
    precision = exp(log_precision)

    log_total = _factor_logaddexp(log_mean, log_precision)
    log_rising = if iszero(count)
        zero(T)
    elseif count_int < 32
        total = log_precision
        for outcome in 1:(count_int - 1)
            total += log(precision + T(outcome))
        end
        total
    else
        T(BRM.loggamma(precision + count)) - T(BRM.loggamma(precision))
    end
    log_rising - BRM._native_ppl_logfactorial(T, count_int) +
        precision * (log_precision - log_total) +
        count * (log_mean - log_total)
end

@inline function _factor_logdensity_at(
        factor::NegativeBinomial2SiteFactor, value::T,
        index, plan, buffers) where {T}
    log_mean = _factor_nb2_log_parameter(
        factor.mean, index, plan, buffers, T)
    log_precision = _factor_nb2_log_parameter(
        factor.precision, index, plan, buffers, T)
    _factor_nb2_logdensity(value, log_mean, log_precision)
end

@inline function _factor_bb2_log_probabilities(
        factor::BetaBinomial2SiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    mean = factor.mean
    if mean isa NodeValue
        name = node_value_name(mean)
        node = getproperty(plan.graph.nodes, name)
        if node isa LogisticFactorNode
            logit = _factor_argument_at(node.input, index, plan, buffers, T)
            return (-BRM._native_ppl_softplus(-logit),
                    -BRM._native_ppl_softplus(logit))
        end
    end
    value = _factor_argument_at(mean, index, plan, buffers, T)
    (log(value), log1p(-value))
end

@inline function _factor_bb2_log_precision(
        factor::BetaBinomial2SiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    _factor_nb2_log_parameter(
        factor.precision, index, plan, buffers, T)
end

@inline function _factor_log_rising(
        log_base::T, base::T, count::Int) where {T}
    count == 0 && return zero(T)
    total = log_base
    for index in 1:(count - 1)
        total += log(base + T(index))
    end
    total
end

@inline function _factor_bb2_logdensity(
        outcome::T, trials::Int, log_mean::T, log_complement::T,
        log_precision::T) where {T}
    successes = Int(outcome)
    failures = trials - successes
    log_choose = BRM._native_ppl_logfactorial(T, trials) -
        BRM._native_ppl_logfactorial(T, successes) -
        BRM._native_ppl_logfactorial(T, failures)
    log_precision > log(floatmax(T)) && return log_choose +
        outcome * log_mean + T(failures) * log_complement

    precision = exp(log_precision)
    log_alpha = log_precision + log_mean
    log_beta = log_precision + log_complement
    alpha = exp(log_alpha)
    beta = exp(log_beta)
    log_choose +
        _factor_log_rising(log_alpha, alpha, successes) +
        _factor_log_rising(log_beta, beta, failures) -
        _factor_log_rising(log_precision, precision, trials)
end

@inline function _factor_logdensity_at(
        factor::BetaBinomial2SiteFactor, value::T,
        index, plan, buffers) where {T}
    trials = Int(_factor_argument_at(
        factor.trials, index, plan, buffers, T))
    log_mean, log_complement = _factor_bb2_log_probabilities(
        factor, index, plan, buffers, T)
    log_precision = _factor_bb2_log_precision(
        factor, index, plan, buffers, T)
    _factor_bb2_logdensity(
        value, trials, log_mean, log_complement, log_precision)
end

@inline function _factor_logdensity_at(
        factor::BinomialLogitSiteFactor, value::T,
        index, plan, buffers) where {T}
    trials = Int(_factor_argument_at(
        factor.trials, index, plan, buffers, T))
    successes = Int(value)
    failures = trials - successes
    logit = _factor_argument_at(
        factor.logit, index, plan, buffers, T)
    BRM._native_ppl_logfactorial(T, trials) -
        BRM._native_ppl_logfactorial(T, successes) -
        BRM._native_ppl_logfactorial(T, failures) -
        value * BRM._native_ppl_softplus(-logit) -
        T(failures) * BRM._native_ppl_softplus(logit)
end

@inline function _factor_logdensity_at(
        factor::CategoricalLogitSiteFactor, value::T,
        index, plan, buffers) where {T}
    largest = _factor_categorical_largest(
        factor.logits, index, plan, buffers, T, zero(T))
    total = _factor_categorical_total(
        factor.logits, index, plan, buffers, T,
        largest, exp(-largest))
    selected = _factor_categorical_selected(
        factor.logits, Int(value) - 1, index, plan, buffers, T,
        zero(T), 1)
    selected - largest - log(total)
end

@inline _factor_categorical_largest(
    ::Tuple{}, index, plan, buffers, ::Type{T}, largest::T) where {T} = largest

@inline function _factor_categorical_largest(
        arguments::Tuple, index, plan, buffers, ::Type{T},
        largest::T) where {T}
    logit = _factor_argument_at(
        first(arguments), index, plan, buffers, T)
    _factor_categorical_largest(
        Base.tail(arguments), index, plan, buffers, T,
        max(largest, logit))
end

@inline _factor_categorical_total(
    ::Tuple{}, index, plan, buffers, ::Type{T}, largest::T,
    total::T) where {T} = total

@inline function _factor_categorical_total(
        arguments::Tuple, index, plan, buffers, ::Type{T},
        largest::T, total::T) where {T}
    logit = _factor_argument_at(
        first(arguments), index, plan, buffers, T)
    _factor_categorical_total(
        Base.tail(arguments), index, plan, buffers, T, largest,
        total + exp(logit - largest))
end


@inline _factor_categorical_selected(
    ::Tuple{}, target, index, plan, buffers, ::Type{T},
    selected::T, offset::Int) where {T} = selected

@inline function _factor_categorical_selected(
        arguments::Tuple, target, index, plan, buffers, ::Type{T},
        selected::T, offset::Int) where {T}
    logit = _factor_argument_at(
        first(arguments), index, plan, buffers, T)
    _factor_categorical_selected(
        Base.tail(arguments), target, index, plan, buffers, T,
        offset == target ? logit : selected, offset + 1)
end

@inline _factor_categorical_draw(
    ::Tuple{}, threshold, index, plan, buffers, ::Type{T}, largest::T,
    cumulative::T, offset::Int) where {T} = offset

@inline function _factor_categorical_draw(
        arguments::Tuple, threshold, index, plan, buffers, ::Type{T},
        largest::T, cumulative::T, offset::Int) where {T}
    logit = _factor_argument_at(
        first(arguments), index, plan, buffers, T)
    next_cumulative = cumulative + exp(logit - largest)
    threshold < next_cumulative && return offset + 1
    _factor_categorical_draw(
        Base.tail(arguments), threshold, index, plan, buffers, T,
        largest, next_cumulative, offset + 1)
end

@inline _factor_evidence_bound_at(::LiteralValue{Nothing}, index, plan,
                                  buffers, ::Type{T}) where {T} = nothing
@inline _factor_evidence_bound_at(value, index, plan, buffers,
                                  ::Type{T}) where {T} =
    _factor_argument_at(value, index, plan, buffers, T)

@inline function _factor_logaddexp(left::T, right::T) where {T}
    left == -T(Inf) && return right
    right == -T(Inf) && return left
    largest = max(left, right)
    largest + log1p(exp(min(left, right) - largest))
end

@inline function _factor_logdiffexp(upper::T, lower::T) where {T}
    lower < upper || return -T(Inf)
    upper + BRM.log1mexp(lower - upper)
end

@inline function _factor_standard_normal_logcdf(value::T) where {T}
    if value < -T(10)
        inverse_square = inv(value * value)
        series = one(T) - inverse_square + T(3) * inverse_square^2 -
            T(15) * inverse_square^3 + T(105) * inverse_square^4
        return -T(0.5) * value * value -
            T(BRM._NATIVE_PPL_HALF_LOG2PI) - log(-value) + log(series)
    end
    log(BRM.erfc(-value / sqrt(T(2)))) - log(T(2))
end

@inline _factor_standard_normal_logccdf(value::T) where {T} =
    _factor_standard_normal_logcdf(-value)

@inline function _factor_logcdf_at(
        factor::NormalSiteFactor, value::T, index, plan, buffers) where {T}
    location = _factor_argument_at(
        factor.location, index, plan, buffers, T)
    scale = _factor_argument_at(factor.scale, index, plan, buffers, T)
    scale > zero(T) || return -T(Inf)
    _factor_standard_normal_logcdf((value - location) / scale)
end

@inline function _factor_logccdf_at(
        factor::NormalSiteFactor, value::T, index, plan, buffers) where {T}
    location = _factor_argument_at(
        factor.location, index, plan, buffers, T)
    scale = _factor_argument_at(factor.scale, index, plan, buffers, T)
    scale > zero(T) || return -T(Inf)
    _factor_standard_normal_logccdf((value - location) / scale)
end

@inline function _factor_poisson_logcdf(value::T, log_rate::T) where {T}
    count = floor(Int, value)
    count < 0 && return -T(Inf)
    term = -exp(log_rate)
    total = term
    for outcome in 1:count
        term += log_rate - log(T(outcome))
        total = _factor_logaddexp(total, term)
    end
    min(total, zero(T))
end

@inline function _factor_poisson_logccdf_direct(
        count::Int, log_rate::T) where {T}
    outcome = count + 1
    term = -exp(log_rate) + T(outcome) * log_rate -
        T(BRM.loggamma(outcome + 1))
    total = term
    rate = exp(log_rate)
    for _ in 1:100_000
        outcome += 1
        term += log_rate - log(T(outcome))
        updated = _factor_logaddexp(total, term)
        if outcome > rate && term - updated < log(eps(T)) - T(2)
            return min(updated, zero(T))
        end
        total = updated
    end
    total
end

@inline function _factor_poisson_logccdf(value::T, log_rate::T) where {T}
    count = floor(Int, value)
    count < 0 && return zero(T)
    lower = _factor_poisson_logcdf(T(count), log_rate)
    lower < zero(T) && return BRM.log1mexp(lower)
    _factor_poisson_logccdf_direct(count, log_rate)
end

@inline function _factor_logcdf_at(
        factor::PoissonSiteFactor, value::T, index, plan, buffers) where {T}
    _factor_poisson_logcdf(
        value, _factor_poisson_log_rate(factor, index, plan, buffers, T))
end

@inline function _factor_logccdf_at(
        factor::PoissonSiteFactor, value::T, index, plan, buffers) where {T}
    _factor_poisson_logccdf(
        value, _factor_poisson_log_rate(factor, index, plan, buffers, T))
end

@inline function _factor_truncation_logmass(
        base::NormalSiteFactor, lower, upper, index, plan, buffers,
        ::Type{T}) where {T}
    lower === nothing && upper === nothing && return zero(T)
    lower === nothing && return _factor_logcdf_at(
        base, upper, index, plan, buffers)
    upper === nothing && return _factor_logccdf_at(
        base, lower, index, plan, buffers)
    _factor_logdiffexp(
        _factor_logcdf_at(base, upper, index, plan, buffers),
        _factor_logcdf_at(base, lower, index, plan, buffers))
end

@inline function _factor_truncation_logmass(
        base::PoissonSiteFactor, lower, upper, index, plan, buffers,
        ::Type{T}) where {T}
    lower === nothing && upper === nothing && return zero(T)
    lower_cdf = lower === nothing ? -T(Inf) :
        _factor_logcdf_at(base, lower - one(T), index, plan, buffers)
    upper === nothing && return BRM.log1mexp(lower_cdf)
    _factor_logdiffexp(
        _factor_logcdf_at(base, upper, index, plan, buffers), lower_cdf)
end

@inline function _factor_interval_logmass(
        base, lower::T, upper::T, index, plan, buffers) where {T}
    _factor_logdiffexp(
        _factor_logcdf_at(base, upper, index, plan, buffers),
        _factor_logcdf_at(base, lower, index, plan, buffers))
end

@inline function _factor_logdensity_at(
        factor::EvidenceSiteFactor, value::T,
        index, plan, buffers) where {T}
    base = base_site_factor(factor)
    lower = _factor_evidence_bound_at(
        factor.lower, index, plan, buffers, T)
    upper = _factor_evidence_bound_at(
        factor.upper, index, plan, buffers, T)
    lower !== nothing && upper !== nothing && lower >= upper &&
        return -T(Inf)
    kind = evidence_kind(factor.evidence)
    if kind === :interval_censored
        upper === nothing && return -T(Inf)
        value < upper || return -T(Inf)
        return _factor_interval_logmass(
            base, value, upper, index, plan, buffers)
    elseif kind === :censored
        lower !== nothing && value == lower && return _factor_logcdf_at(
            base, lower, index, plan, buffers)
        if upper !== nothing && value == upper
            threshold = base isa PoissonSiteFactor ? upper - one(T) : upper
            return _factor_logccdf_at(
                base, threshold, index, plan, buffers)
        end
        lower !== nothing && value < lower && return -T(Inf)
        upper !== nothing && value > upper && return -T(Inf)
        return _factor_logdensity_at(
            factor.factor, value, index, plan, buffers)
    end
    lower !== nothing && value < lower && return -T(Inf)
    upper !== nothing && value > upper && return -T(Inf)
    logmass = _factor_truncation_logmass(
        base, lower, upper, index, plan, buffers, T)
    isfinite(logmass) || return -T(Inf)
    _factor_logdensity_at(factor.factor, value, index, plan, buffers) - logmass
end

@inline function _factor_logdensity_at(
        factor::WeightedSiteFactor{F,ObservationWeight{:analytic,Source}},
        value::T, index, plan, buffers) where {F,Source,T}
    base = factor.factor
    weight = _factor_argument_at(factor.values, index, plan, buffers, T)
    location = _factor_argument_at(base.location, index, plan, buffers, T)
    scale = _factor_argument_at(base.scale, index, plan, buffers, T)
    adjusted_scale = scale / sqrt(weight)
    residual = (value - location) / adjusted_scale
    -T(0.5) * residual * residual - log(adjusted_scale) -
        T(BRM._NATIVE_PPL_HALF_LOG2PI)
end

@inline function _factor_logdensity_at(
        factor::WeightedSiteFactor{F,ObservationWeight{Kind,Source}},
        value::T, index, plan, buffers) where {
            F,Kind,Source,T}
    Kind === :unit && return _factor_logdensity_at(
        factor.factor, value, index, plan, buffers)
    weight = _factor_argument_at(factor.values, index, plan, buffers, T)
    iszero(weight) && return zero(T)
    weight * _factor_logdensity_at(
        factor.factor, value, index, plan, buffers)
end

@inline @generated function _factor_lkj_logdensity(
        ::Val{K}, eta_value, log_normalizers, indices,
        position::AbstractVector{T}) where {K,T}
    terms = Any[]
    coordinate_offset = 0
    for column in 1:(K - 1)
        alpha_offset = (K - column - 1) / 2
        for row in (column + 1):K
            push!(terms, quote
                let alpha_value = eta_value + $alpha_offset,
                    alpha = T(alpha_value),
                    log_constant = T(getfield(log_normalizers, $column)),
                    raw = position[first(indices) + $coordinate_offset]
                    log_constant + alpha * _factor_logsech2(raw)
                end
            end)
            coordinate_offset += 1
        end
    end
    isempty(terms) && return :(zero(T))
    foldl((left, right) -> :($left + $right), terms)
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{S,Tr,F,ScalarSiteShape,FreeSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    coordinates = getproperty(prepared.plan.graph.coordinates, Name)
    length(coordinates.indices) == 1 || throw(CapabilityError(
        :factor_coordinates,
        "free scalar site `$Name` must own exactly one coordinate"))
    unconstrained = position[first(coordinates.indices)]
    value, logjac = _factor_transform(site.transform, unconstrained)
    buffers.values[getproperty(prepared.plan.site_indices, Name)] = value
    _factor_logdensity(site.factor, value, prepared.plan, buffers) + logjac
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{S,Tr,F,ScalarSiteShape,ConditionedSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    value = getproperty(prepared.conditions, Name)
    buffers.values[getproperty(prepared.plan.site_indices, Name)] = value
    _factor_logdensity(site.factor, value, prepared.plan, buffers)
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{S,Tr,F,BlockSiteShape,FreeSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    coordinates = getproperty(prepared.plan.graph.coordinates, Name)
    density = zero(T)
    for coordinate in coordinates.indices
        value, logjac = _factor_transform(
            site.transform, position[coordinate])
        density += _factor_logdensity(
            site.factor, value, prepared.plan, buffers) + logjac
    end
    density
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{
            CholeskyCorrelationSupport{K},
            CholeskyCorrelationTransform{K},
            F,BlockSiteShape,FreeSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {
            Name,K,T,F<:LKJCholeskySiteFactor}
    coordinates = getproperty(prepared.plan.graph.coordinates, Name)
    expected_coordinates = K * (K - 1) ÷ 2
    length(coordinates.indices) == expected_coordinates || throw(CapabilityError(
        :factor_coordinates,
        "$K-dimensional LKJ Cholesky site `$Name` must own " *
        "$expected_coordinates coordinates"))
    _factor_lkj_logdensity(
        Val(K), site.factor.eta, site.factor.log_normalizers,
        coordinates.indices, position)
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{S,Tr,F,BlockSiteShape,ConditionedSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    values = getproperty(prepared.conditions, Name)
    density = zero(T)
    for value in values
        density += _factor_logdensity(
            site.factor, value, prepared.plan, buffers)
    end
    density
end

@inline function _factor_site_logdensity!(::Val{Name},
        site::StochasticSite{S,Tr,F,BroadcastSiteShape,ConditionedSite},
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    response = getproperty(prepared.conditions, Name)
    density = zero(T)
    for index in eachindex(response)
        pointwise = _factor_logdensity_at(
            site.factor, response[index], index, prepared.plan, buffers)
        buffers.pointwise_loglikelihood[index] = pointwise
        density += pointwise
    end
    density
end


@inline function _factor_coefficient(::SiteValue{Name}, coefficient_index,
                                     position::AbstractVector{T},
                                     prepared::FactorPrepared,
                                     buffers::FactorBuffers{T}) where {Name,T}
    site = getproperty(prepared.plan.graph.sites, Name)
    if site.activity isa FreeSite
        coordinates = getproperty(
            prepared.plan.graph.coordinates, Name).indices
        value, _ = _factor_transform(
            site.transform, position[coordinates[coefficient_index]])
        value
    else
        conditioned = getproperty(prepared.conditions, Name)
        T(conditioned isa AbstractVector ?
            conditioned[coefficient_index] : conditioned)
    end
end

@inline _factor_site_logdensity!(::Val,
        ::StochasticSite{S,Tr,F,BroadcastSiteShape,GeneratedSite},
    position::AbstractVector{T}, prepared::FactorPrepared,
    buffers::FactorBuffers{T}) where {S,Tr,F,T} = zero(T)

@inline function _factor_node_logdensity!(::Val{Name},
        node::CenterFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    fitted = getproperty(prepared.plan.fitted_nodes, Name)
    mean = T(fitted.mean)
    first_value = zero(T)
    for row in prepared.plan.observation_axis.keys
        value = _factor_argument_at(
            node.input, row, prepared.plan, buffers, T) - mean
        isfinite(value) || throw(ArgumentError(
            "native PPL centered predictor is non-finite at row $row"))
        buffers.node_rows[node_index, row] = value
        row == first(prepared.plan.observation_axis.keys) &&
            (first_value = value)
    end
    buffers.node_values[node_index] = first_value
    zero(T)
end

@inline function _factor_node_logdensity!(::Val{Name},
        node::ZScaleFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    fitted = getproperty(prepared.plan.fitted_nodes, Name)
    mean = T(fitted.mean)
    scale = T(fitted.scale)
    first_value = zero(T)
    for row in prepared.plan.observation_axis.keys
        raw = _factor_argument_at(
            node.input, row, prepared.plan, buffers, T)
        value = (raw - mean) / scale
        if !isfinite(value)
            value = raw / scale - mean / scale
        end
        isfinite(value) || throw(ArgumentError(
            "native PPL standardized predictor is non-finite at row $row"))
        buffers.node_rows[node_index, row] = value
        row == first(prepared.plan.observation_axis.keys) &&
            (first_value = value)
    end
    buffers.node_values[node_index] = first_value
    zero(T)
end

@inline function _factor_node_logdensity!(::Val{Name}, node::ExpFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    first_value = zero(T)
    for row in prepared.plan.observation_axis.keys
        value = exp(_factor_argument_at(
            node.input, row, prepared.plan, buffers, T))
        buffers.node_rows[node_index, row] = value
        row == first(prepared.plan.observation_axis.keys) &&
            (first_value = value)
    end
    buffers.node_values[node_index] = first_value
    zero(T)
end

@inline _factor_schedule_node_logdensity!(name, node,
        position, prepared, buffers) =
    _factor_node_logdensity!(name, node, position, prepared, buffers)

@inline function _factor_site_has_log_rate_node(
        ::Val{Name}, site::StochasticSite) where {Name}
    factor = base_site_factor(site.factor)
    rate = if factor isa PoissonSiteFactor
        factor.rate
    elseif factor isa ZeroInflatedPoissonSiteFactor
        factor.rate
    elseif factor isa NegativeBinomial2SiteFactor
        return (factor.mean isa NodeValue{Name}) ||
               (factor.precision isa NodeValue{Name})
    elseif factor isa BetaBinomial2SiteFactor
        return factor.precision isa NodeValue{Name}
    else
        return false
    end
    rate isa NodeValue{Name}
end

@inline @generated function _factor_terminal_log_rate_node(
        name::Val{Name}, plan::FactorPlan{Outputs}) where {Name,Outputs}
    checks = map(Outputs) do output
        :(_factor_site_has_log_rate_node(
            name, getproperty(plan.graph.sites, $(QuoteNode(output)))))
    end
    isempty(checks) && return :(false)
    foldl((left, right) -> :($left || $right), checks)
end

@inline function _factor_schedule_node_logdensity!(
        name::Val, node::ExpFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {T}
    _factor_terminal_log_rate_node(name, prepared.plan) && return zero(T)
    _factor_node_logdensity!(name, node, position, prepared, buffers)
end

@inline function _factor_node_logdensity!(::Val{Name}, node::LogFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    for row in prepared.plan.observation_axis.keys
        buffers.node_rows[node_index, row] = log(_factor_argument_at(
            node.input, row, prepared.plan, buffers, T))
    end
    zero(T)
end

@inline function _factor_node_logdensity!(
        ::Val{Name}, node::LogisticFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    first_value = zero(T)
    for row in prepared.plan.observation_axis.keys
        value = BRM._native_ppl_logistic(_factor_argument_at(
            node.input, row, prepared.plan, buffers, T))
        buffers.node_rows[node_index, row] = value
        row == first(prepared.plan.observation_axis.keys) &&
            (first_value = value)
    end
    buffers.node_values[node_index] = first_value
    zero(T)
end

@inline function _factor_node_logdensity!(::Val{Name}, node::AffineFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    for row in prepared.plan.observation_axis.keys
        has_intercept = affine_has_intercept(node)
        value = has_intercept ? _factor_coefficient(
            node.coefficients, 1, position, prepared, buffers) : zero(T)
        for input_index in eachindex(node.inputs)
            value += _factor_coefficient(
                node.coefficients, input_index + (has_intercept ? 1 : 0),
                position, prepared, buffers) *
                _factor_argument_at(
                    node.inputs[input_index], row,
                    prepared.plan, buffers, T)
        end
        for offset in node.offsets
            value += _factor_argument_at(
                offset, row, prepared.plan, buffers, T)
        end
        buffers.node_rows[node_index, row] = value
    end
    zero(T)
end

@inline function _factor_node_logdensity!(::Val{Name},
        node::GroupGatherFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    site_name = site_value_name(node.values)
    site = getproperty(prepared.plan.graph.sites, site_name)
    group_indices = getproperty(prepared.plan.group_indices, Name)
    for row in prepared.plan.observation_axis.keys
        group_index = group_indices[row]
        value = if group_index < 0
            generated_indices = getproperty(
                prepared.plan.generated_group_indices, site_name)
            buffers.generated_group_values[generated_indices[-group_index]]
        elseif site.activity isa FreeSite
            coordinates = getproperty(
                prepared.plan.graph.coordinates, site_name).indices
            transformed, _ = _factor_transform(
                site.transform, position[coordinates[group_index]])
            transformed
        else
            getproperty(prepared.conditions, site_name)[group_index]
        end
        buffers.node_rows[node_index, row] = value
    end
    zero(T)
end

@inline function _factor_node_logdensity!(::Val{Name},
        node::RowProductFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    for row in prepared.plan.observation_axis.keys
        buffers.node_rows[node_index, row] =
            _factor_argument_at(
                node.left, row, prepared.plan, buffers, T) *
            _factor_argument_at(
                node.right, row, prepared.plan, buffers, T)
    end
    zero(T)
end

@inline function _factor_grouped_standardized(
        node::GroupedAffineFactorNode, group_index::Int,
        coefficient_index::Int, coefficient_count::Int,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {T}
    standardized_offset = (abs(group_index) - 1) * coefficient_count
    standardized_name = site_value_name(node.standardized)
    if group_index < 0
        generated_indices = getproperty(
            prepared.plan.generated_group_indices, standardized_name)
        buffers.generated_group_values[
            generated_indices[standardized_offset + coefficient_index]]
    else
        _factor_coefficient(
            node.standardized, standardized_offset + coefficient_index,
            position, prepared, buffers)
    end
end

@inline function _factor_correlation_raw(
        position::AbstractVector, correlation_coordinates,
        coefficient_index::Int, source_index::Int,
        coefficient_count::Int)
    coordinate_offset =
        (source_index - 1) * coefficient_count -
        (source_index - 1) * source_index ÷ 2 +
        coefficient_index - source_index
    position[first(correlation_coordinates) + coordinate_offset - 1]
end

@inline @generated function _factor_grouped_affine_value(
        node::GroupedAffineFactorNode{Z,S,C,G,P,I,K},
        group_index::Int, row::Int, correlation_coordinates,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Z,S,C,G,P,I,K,T}
    coefficient_count = K
    contributions = Any[]
    for (predictor_index, coefficient_index) in enumerate(I)
        correlated_terms = Any[]
        residual = :(one(T))
        for source_index in 1:(coefficient_index - 1)
            raw = :(_factor_correlation_raw(
                position, correlation_coordinates,
                $coefficient_index, $source_index, $coefficient_count))
            standardized = :(_factor_grouped_standardized(
                node, group_index, $source_index, $coefficient_count,
                position, prepared, buffers))
            push!(correlated_terms,
                  :($residual * tanh($raw) * $standardized))
            residual = :($residual * _factor_sech($raw))
        end
        diagonal = :(_factor_grouped_standardized(
            node, group_index, $coefficient_index, $coefficient_count,
            position, prepared, buffers))
        push!(correlated_terms, :($residual * $diagonal))
        correlated = foldl(
            (left, right) -> :($left + $right), correlated_terms)
        effect = :(_factor_coefficient(
            node.scales, $coefficient_index,
            position, prepared, buffers) * $correlated)
        predictor = :(getfield(node.predictors, $predictor_index))
        push!(contributions, :(
            $predictor === nothing ? $effect :
                $effect * _factor_argument_at(
                    $predictor, row, prepared.plan, buffers, T)))
    end
    isempty(contributions) && return :(zero(T))
    foldl((left, right) -> :($left + $right), contributions)
end

@inline function _factor_node_logdensity!(::Val{Name},
        node::GroupedAffineFactorNode,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    node_index = getproperty(prepared.plan.node_indices, Name)
    group_indices = getproperty(prepared.plan.group_indices, Name)
    correlation_name = site_value_name(node.correlation)
    correlation_coordinates = getproperty(
        prepared.plan.graph.coordinates, correlation_name).indices
    for row in prepared.plan.observation_axis.keys
        group_index = group_indices[row]
        buffers.node_rows[node_index, row] = _factor_grouped_affine_value(
            node, group_index, row, correlation_coordinates,
            position, prepared, buffers)
    end
    zero(T)
end

@inline _has_generated_groups(plan::FactorPlan) =
    any(!isempty, values(plan.generated_group_levels))

@inline function _factor_require_fixed_coordinates(plan::FactorPlan)
    _has_generated_groups(plan) && throw(CapabilityError(
        :new_group_activity,
        "native PPL density and deterministic queries require fixed group " *
        "coordinates; this replay contains generated groups and must use " *
        "an RNG posterior-predictive query"))
    nothing
end

@generated function _factor_execute_schedule!(::Val{Names},
        ::Val{SiteNames}, sites::Sites,
        ::Val{NodeNames}, nodes::Nodes,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {
            Names,SiteNames,Sites<:Tuple,NodeNames,Nodes<:Tuple,T}
    calls = Any[]
    for name in Names
        site_index = findfirst(==(name), SiteNames)
        entry, callee = if site_index === nothing
            node_index = findfirst(==(name), NodeNames)
            node_index === nothing && error("unknown factor schedule entry $name")
            (:(getfield(nodes, $node_index)),
             :_factor_schedule_node_logdensity!)
        else
            (:(getfield(sites, $site_index)), :_factor_site_logdensity!)
        end
        push!(calls, :($callee(
            Val($(QuoteNode(name))), $entry,
            position, prepared, buffers)))
    end
    isempty(calls) && return :(zero(T))
    foldl((left, right) -> :($left + $right), calls)
end

@inline function _factor_logdensity_kernel(position::AbstractVector{T},
                                           prepared::FactorPrepared,
                                           buffers::FactorBuffers{T}) where {T}
    _factor_require_fixed_coordinates(prepared.plan)
    graph = prepared.plan.graph
    _factor_execute_schedule!(
        Val(graph.schedule), Val(Tuple(keys(graph.sites))),
        Tuple(graph.sites), Val(Tuple(keys(graph.nodes))), Tuple(graph.nodes),
        position, prepared, buffers)
end

@inline function _factor_node_query_kernel(
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T},
        ::BRM.NativePPLNodeOutput{Name}) where {T,Name}
    if !_has_generated_groups(prepared.plan)
        density = _factor_logdensity_kernel(position, prepared, buffers)
        node = getproperty(prepared.plan.graph.nodes, Name)
        node isa ExpFactorNode &&
            _factor_terminal_log_rate_node(Val(Name), prepared.plan) &&
            _factor_node_logdensity!(
                Val(Name), node, position, prepared, buffers)
        return density
    end
    node = getproperty(prepared.plan.graph.nodes, Name)
    input_only = node isa Union{
        CenterFactorNode,ZScaleFactorNode,LogFactorNode,
        LogisticFactorNode} &&
        node.input isa InputValue
    input_only || _factor_require_fixed_coordinates(prepared.plan)
    _factor_node_logdensity!(Val(Name), node, position, prepared, buffers)
end

function _factor_check_execution(workspace::FactorWorkspace,
                                 prepared::FactorPrepared,
                                 position::AbstractVector)
    dimension = BRM.LogDensityProblems.dimension(prepared)
    length(position) == dimension || throw(DimensionMismatch(
        "native PPL factor position has length $(length(position)); " *
        "expected $dimension"))
    eltype(position) === eltype(workspace) || throw(ArgumentError(
        "native PPL factor position eltype $(eltype(position)) does not " *
        "match workspace eltype $(eltype(workspace))"))
    eltype(prepared) === eltype(workspace) || throw(ArgumentError(
        "native PPL factor prepared eltype $(eltype(prepared)) does not " *
        "match workspace eltype $(eltype(workspace))"))
    length(workspace.gradient) == dimension || throw(DimensionMismatch(
        "native PPL factor gradient has length $(length(workspace.gradient)); " *
        "expected $dimension"))
    length(workspace.primal.values) == length(prepared.plan.graph.sites) ||
        throw(DimensionMismatch(
            "native PPL factor workspace has the wrong site-value layout"))
    length(workspace.primal.generated_group_values) ==
        sum(length, values(prepared.plan.generated_group_indices); init=0) ||
        throw(DimensionMismatch(
            "native PPL factor workspace has the wrong generated-group layout"))
    length(workspace.primal.node_values) ==
        length(prepared.plan.graph.nodes) || throw(DimensionMismatch(
            "native PPL factor workspace has the wrong node-value layout"))
    axes(workspace.primal.node_rows) == (
        Base.OneTo(length(prepared.plan.graph.nodes)),
        prepared.plan.observation_axis.keys) || throw(DimensionMismatch(
            "native PPL factor workspace has the wrong row-node layout"))
    length(workspace.primal.pointwise_loglikelihood) ==
        length(prepared.plan.observation_axis) || throw(DimensionMismatch(
            "native PPL factor workspace has the wrong observation layout"))
    for buffer in (workspace.gradient, workspace.primal.values,
                   workspace.primal.generated_group_values,
                   workspace.primal.node_values, workspace.primal.node_rows,
                   workspace.primal.pointwise_loglikelihood)
        Base.mightalias(position, buffer) && throw(ArgumentError(
            "native PPL factor position must not alias workspace storage"))
    end
    nothing
end

function logdensity!(work::FactorWorkspace, prepared::FactorPrepared,
                     position::AbstractVector)
    _factor_check_execution(work, prepared, position)
    _factor_logdensity_kernel(position, prepared, work.primal)
end

function _factor_logdensity_and_gradient! end
function _factor_logdensity_and_gradient!(work::FactorWorkspace,
                                          prepared::FactorPrepared,
                                          position::AbstractVector)
    throw(ArgumentError(
        "native PPL factor gradients require a " *
        "DifferentiationInterface-prepared workspace; construct one with " *
        "AutoEnzyme() for the supported path"))
end

logdensity_and_gradient!(work::FactorWorkspace, prepared::FactorPrepared,
                         position::AbstractVector) =
    _factor_logdensity_and_gradient!(work, prepared, position)

has_response(prepared::FactorPrepared) = any(
    output -> hasproperty(prepared.conditions, output),
    factor_output_sites(prepared.plan))

function _factor_group_levels(plan::FactorPlan)
    names = Tuple(name for (name, parameter) in pairs(
        plan.declaration.parameters)
        if parameter isa Union{
            GroupedNormalParameter,GroupedStandardNormalParameter})
    values = map(names) do name
        site = getproperty(plan.graph.sites, name)
        Tuple(unique(key.level for key in site.coordinate_keys))
    end
    NamedTuple{names}(values)
end

function rebind(prepared::FactorPrepared, conditions;
                bindings=prepared.plan.bindings,
                T::Type{<:AbstractFloat}=eltype(prepared),
                new_groups::Symbol=:error,
                freeze_constants::Bool=true, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "native PPL factor replay does not accept keyword options yet; got " *
        "$(keys(kwargs))"))
    conditions isa NamedTuple || throw(ArgumentError(
        "native PPL factor replay bindings must be a NamedTuple; got " *
        "$(typeof(conditions))"))
    bindings isa NamedTuple || throw(ArgumentError(
        "native PPL factor replay value bindings must be a NamedTuple; got " *
        "$(typeof(bindings))"))
    _validate_binding_names(prepared.plan.declaration, bindings)
    rebound = _bind_factor_plan(
        prepared.plan.declaration, bindings, conditions;
        group_levels=_factor_group_levels(prepared.plan), new_groups,
        fitted_nodes=freeze_constants ? prepared.plan.fitted_nodes : nothing,
        outcome_levels=(freeze_constants || isempty(conditions)) ?
            prepared.plan.outcome_levels : nothing)
    if !any(output -> hasproperty(rebound.conditions, output),
            factor_output_sites(rebound)) &&
       isempty(rebound.observation_axis.keys)
        rebound = FactorPlan(
            rebound.declaration, rebound.graph, rebound.fitted_nodes,
            rebound.bindings,
            rebound.conditions,
            rebound.outcome_levels,
            rebound.site_indices, rebound.node_indices,
            rebound.group_indices, rebound.generated_group_levels,
            rebound.generated_group_indices,
            prepared.plan.observation_axis,
            factor_output_sites(rebound))
    end
    prepare(rebound; T)
end

function _factor_output_signature(plan::FactorPlan)
    BRM.NativePPLOutputSignature(
        plan.observation_axis, BRM.NativePPLPreparedElementType(),
        BRM.NativePPLDenseVectorLayout())
end

function _factor_predictive_output_signature(
        plan::FactorPlan, output_site::Symbol)
    output_site in factor_output_sites(plan) || throw(CapabilityError(
        :query,
        "`$output_site` is not a terminal stochastic output"))
    _factor_predictive_output_signature(plan, Val(output_site))
end

function _factor_predictive_output_signature(
        plan::FactorPlan, ::Val{Output}) where {Output}
    site = getproperty(plan.graph.sites, Output)
    factor = base_site_factor(site.factor)
    element_type = if factor isa BernoulliLogitSiteFactor
        BRM.NativePPLFixedElementType{Bool}()
    elseif factor isa Union{
            PoissonSiteFactor,ZeroInflatedPoissonSiteFactor,
            NegativeBinomial2SiteFactor,BetaBinomial2SiteFactor,
            BinomialLogitSiteFactor,CategoricalLogitSiteFactor}
        BRM.NativePPLFixedElementType{Int}()
    else
        BRM.NativePPLPreparedElementType()
    end
    BRM.NativePPLOutputSignature(
        plan.observation_axis, element_type,
        BRM.NativePPLDenseVectorLayout())
end

@generated function _factor_predictive_output_signature(
        plan::FactorPlan{Outputs}) where {Outputs}
    length(Outputs) == 1 || return :(throw(CapabilityError(
        :factor_outputs,
        "predictive signature requires one selected terminal outcome")))
    :(_factor_predictive_output_signature(
        plan, Val($(QuoteNode(only(Outputs))))))
end

function _factor_outcome_signatures(plan::FactorPlan, signature)
    outputs = factor_output_sites(plan)
    NamedTuple{outputs}(map(signature, outputs))
end

const _FactorRowNode = Union{
    CenterFactorNode,ZScaleFactorNode,ExpFactorNode,LogFactorNode,
    LogisticFactorNode,
    AffineFactorNode,GroupGatherFactorNode,
    RowProductFactorNode,GroupedAffineFactorNode,
}

function _factor_node_output(plan::FactorPlan,
                             ::BRM.NativePPLNodeOutput{Name}) where {Name}
    hasproperty(plan.graph.nodes, Name) || throw(CapabilityError(
        :query, "factor graph has no deterministic node `$Name`"))
    node = getproperty(plan.graph.nodes, Name)
    node isa _FactorRowNode || throw(CapabilityError(
        :query,
        "deterministic node `$Name` is not materialized on the observation axis"))
    node
end

function output_signature(plan::FactorPlan,
                          ::BRM.NativePPLLinearPredictor)
    length(factor_output_sites(plan)) == 1 || throw(CapabilityError(
        :query,
        "multiple-outcome factor plans have no single linear predictor; " *
        "query each named predictor with NodeOutput or use a fused bundle"))
    factor = base_site_factor(getproperty(
        plan.graph.sites, factor_output_site(plan)).factor)
    factor isa CategoricalLogitSiteFactor && throw(CapabilityError(
        :query,
        "CategoricalLogit has multiple non-reference predictors; query " *
        "their named NodeOutput values individually or in one fused bundle"))
    _factor_output_signature(plan)
end
function output_signature(plan::FactorPlan,
                          ::BRM.NativePPLPointwiseLogLikelihood)
    length(factor_output_sites(plan)) == 1 &&
        return _factor_output_signature(plan)
    _factor_outcome_signatures(plan, _ -> _factor_output_signature(plan))
end
function output_signature(plan::FactorPlan,
                          ::BRM.NativePPLPosteriorPredictive)
    length(factor_output_sites(plan)) == 1 &&
        return _factor_predictive_output_signature(plan)
    _factor_outcome_signatures(
        plan, output -> _factor_predictive_output_signature(plan, output))
end
function output_signature(plan::FactorPlan,
                          query::BRM.NativePPLNodeOutput)
    _factor_node_output(plan, query)
    _factor_output_signature(plan)
end
output_signature(prepared::FactorPrepared, query::BRM.NativePPLQuery) =
    output_signature(prepared.plan, query)
output_eltype(signature::BRM.NativePPLOutputSignature,
              prepared::FactorPrepared) =
    BRM.native_output_eltype(signature, prepared)
output_eltype(signatures::NamedTuple, prepared::FactorPrepared) =
    map(signature -> output_eltype(signature, prepared), signatures)

function allocate_output(signature::BRM.NativePPLOutputSignature,
                         prepared::FactorPrepared)
    axis = BRM.native_output_axis(signature)
    axis.keys == Base.OneTo(length(axis)) || throw(CapabilityError(
        :output_layout,
        "dense factor output requires a one-based semantic axis"))
    Vector{BRM.native_output_eltype(signature, prepared)}(
        undef, length(axis))
end

allocate_output(prepared::FactorPrepared, query::BRM.NativePPLQuery) =
    allocate_output(output_signature(prepared, query), prepared)

function _factor_check_query_output(output::AbstractVector,
                                    work::FactorWorkspace,
                                    prepared::FactorPrepared,
                                    position::AbstractVector,
                                    signature::BRM.NativePPLOutputSignature)
    _factor_check_execution(work, prepared, position)
    expected_axis = BRM.native_output_axis(signature).keys
    axes(output, 1) == expected_axis || throw(DimensionMismatch(
        "native PPL factor output axis $(axes(output, 1)) does not match " *
        "declared axis $expected_axis"))
    eltype(output) === BRM.native_output_eltype(signature, prepared) ||
        throw(ArgumentError(
            "native PPL factor output eltype $(eltype(output)) does not " *
            "match prepared eltype $(eltype(prepared))"))
    for buffer in (work.gradient, work.primal.values,
                   work.primal.generated_group_values,
                   work.primal.node_values, work.primal.node_rows,
                   work.primal.pointwise_loglikelihood)
        Base.mightalias(output, buffer) && throw(ArgumentError(
            "native PPL factor output must not alias workspace storage"))
    end
    Base.mightalias(output, position) && throw(ArgumentError(
        "native PPL factor output must not alias the position"))
    output
end

function _factor_check_query_output(output::AbstractVector,
                                    work::FactorWorkspace,
                                    prepared::FactorPrepared,
                                    position::AbstractVector,
                                    query::BRM.NativePPLQuery)
    signature = output_signature(prepared, query)
    signature isa BRM.NativePPLOutputSignature || throw(CapabilityError(
        :query,
        "multiple-outcome query output must be a NamedTuple keyed by " *
        "$(collect(factor_output_sites(prepared.plan)))"))
    _factor_check_query_output(
        output, work, prepared, position, signature)
end

@inline @generated function _factor_check_outcome_query_outputs(
        outputs::NamedTuple{Names}, work::FactorWorkspace,
        prepared::FactorPrepared, position::AbstractVector,
        query::Q) where {Names,Q<:Union{
            BRM.NativePPLPointwiseLogLikelihood,
            BRM.NativePPLPosteriorPredictive}}
    checks = Any[]
    for index in eachindex(Names)
        name = Names[index]
        signature = Q <: BRM.NativePPLPointwiseLogLikelihood ?
            :(_factor_output_signature(prepared.plan)) :
            :(_factor_predictive_output_signature(
                prepared.plan, Val($(QuoteNode(name)))))
        push!(checks, :(_factor_check_query_output(
            getfield(outputs, $index), work, prepared, position,
            $signature)))
    end
    for left in eachindex(Names), right in (left + 1):length(Names)
        push!(checks, quote
            Base.mightalias(getfield(outputs, $left),
                            getfield(outputs, $right)) && throw(
                ArgumentError(
                    "native PPL outcome query outputs must not alias " *
                    "each other"))
        end)
    end
    quote
        expected = factor_output_sites(prepared.plan)
        Names == expected || throw(ArgumentError(
            "native PPL outcome query fields $Names do not match " *
            "terminal outcomes $expected"))
        $(checks...)
        outputs
    end
end

@inline function _factor_pointwise_at(
        ::Val{Name}, prepared::FactorPrepared,
        buffers::FactorBuffers, index::Int) where {Name}
    hasproperty(prepared.conditions, Name) || throw(ArgumentError(
        "native PPL pointwise log likelihood for `$Name` requires that " *
        "outcome to be conditioned"))
    site = getproperty(prepared.plan.graph.sites, Name)
    response = getproperty(prepared.conditions, Name)
    _factor_logdensity_at(
        site.factor, response[index], index, prepared.plan, buffers)
end

@inline @generated function _factor_write_pointwise_outputs!(
        outputs::NamedTuple{Names}, prepared::FactorPrepared,
        buffers::FactorBuffers) where {Names}
    calls = map(eachindex(Names)) do output_index
        name = Names[output_index]
        quote
            output = getfield(outputs, $output_index)
            for row in eachindex(output)
                output[row] = _factor_pointwise_at(
                    Val($(QuoteNode(name))), prepared, buffers, row)
            end
        end
    end
    Expr(:block, calls..., :(outputs))
end

@inline @generated function _factor_write_predictive_outputs!(
        rng::BRM.AbstractRNG, outputs::NamedTuple{Names},
        prepared::FactorPrepared, buffers::FactorBuffers) where {Names}
    calls = map(eachindex(Names)) do output_index
        name = Names[output_index]
        quote
            output = getfield(outputs, $output_index)
            for row in eachindex(output)
                output[row] = _factor_terminal_sample(
                    rng, Val($(QuoteNode(name))), prepared, buffers, row)
            end
        end
    end
    Expr(:block, calls..., :(outputs))
end

@inline function _factor_terminal_linear(::Val{Name},
                                         prepared::FactorPrepared,
                                         buffers::FactorBuffers,
                                         index::Int) where {Name}
    site = getproperty(prepared.plan.graph.sites, Name)
    factor = base_site_factor(site.factor)
    T = eltype(buffers.values)
    if factor isa NormalSiteFactor
        _factor_argument_at(
            factor.location, index, prepared.plan, buffers, T)
    elseif factor isa StudentTSiteFactor
        _factor_argument_at(
            factor.location, index, prepared.plan, buffers, T)
    elseif factor isa BernoulliLogitSiteFactor
        _factor_argument_at(
            factor.logit, index, prepared.plan, buffers, T)
    elseif factor isa ZeroInflatedPoissonSiteFactor
        _factor_zip_log_rate(
            factor, index, prepared.plan, buffers, T)
    elseif factor isa NegativeBinomial2SiteFactor
        _factor_nb2_log_parameter(
            factor.mean, index, prepared.plan, buffers, T)
    elseif factor isa BetaBinomial2SiteFactor
        mean = factor.mean
        if mean isa NodeValue
            node = getproperty(
                prepared.plan.graph.nodes, node_value_name(mean))
            node isa LogisticFactorNode && return _factor_argument_at(
                node.input, index, prepared.plan, buffers, T)
        end
        probability = _factor_argument_at(
            mean, index, prepared.plan, buffers, T)
        log(probability) - log1p(-probability)
    elseif factor isa BinomialLogitSiteFactor
        _factor_argument_at(
            factor.logit, index, prepared.plan, buffers, T)
    else
        _factor_poisson_log_rate(
            factor, index, prepared.plan, buffers, T)
    end
end

@inline _factor_terminal_linear(prepared::FactorPrepared,
                                buffers::FactorBuffers, index::Int) =
    _factor_terminal_linear(
        Val(factor_output_site(prepared.plan)), prepared, buffers, index)

@inline function _factor_node_output_at(
    ::BRM.NativePPLNodeOutput{Name}, prepared::FactorPrepared,
    buffers::FactorBuffers, index::Int) where {Name}
    node = getproperty(prepared.plan.graph.nodes, Name)
    if node isa ExpFactorNode &&
       _factor_terminal_log_rate_node(Val(Name), prepared.plan)
        T = eltype(buffers.values)
        return exp(_factor_argument_at(
            node.input, index, prepared.plan, buffers, T))
    end
    _factor_argument_at(
        NodeValue{Name}(), index, prepared.plan, buffers,
        eltype(buffers.values))
end

@inline function _factor_rand_unit_gamma(
        rng::BRM.AbstractRNG, ::Type{T}, shape::T) where {T}
    isfinite(shape) && shape > zero(T) || throw(DomainError(
        shape, "native PPL Gamma shape must be finite and positive"))
    if shape < one(T)
        draw = _factor_rand_unit_gamma(rng, T, shape + one(T))
        uniform = BRM.rand(rng, T)
        iszero(uniform) && return zero(T)
        return draw * exp(log(uniform) / shape)
    end

    shifted = shape - T(1 / 3)
    inverse_scale = inv(sqrt(T(9) * shifted))
    while true
        normal = BRM.randn(rng, T)
        root = one(T) + inverse_scale * normal
        root > zero(T) || continue
        candidate = root * root * root
        uniform = BRM.rand(rng, T)
        normal2 = normal * normal
        uniform < one(T) - T(0.0331) * normal2 * normal2 &&
            return shifted * candidate
        log(uniform) < T(0.5) * normal2 +
            shifted * (one(T) - candidate + log(candidate)) &&
            return shifted * candidate
    end
end

@inline function _factor_rand_negative_binomial2(
        rng::BRM.AbstractRNG, factor::NegativeBinomial2SiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    log_mean = _factor_nb2_log_parameter(
        factor.mean, index, plan, buffers, T)
    log_precision = _factor_nb2_log_parameter(
        factor.precision, index, plan, buffers, T)
    precision = exp(log_precision)
    isfinite(precision) || return BRM._native_ppl_rand_poisson(
        rng, T, log_mean)
    gamma = _factor_rand_unit_gamma(rng, T, precision)
    iszero(gamma) && return 0
    BRM._native_ppl_rand_poisson(
        rng, T, log_mean + log(gamma) - log_precision)
end

@inline function _factor_rand_binomial(
        rng::BRM.AbstractRNG, ::Type{T}, trials::Int,
        probability::T) where {T}
    successes = 0
    for _ in 1:trials
        successes += BRM.rand(rng, T) < probability
    end
    successes
end

@inline function _factor_rand_beta_binomial2(
        rng::BRM.AbstractRNG, factor::BetaBinomial2SiteFactor,
        index, plan, buffers, ::Type{T}) where {T}
    trials = Int(_factor_argument_at(
        factor.trials, index, plan, buffers, T))
    log_mean, log_complement = _factor_bb2_log_probabilities(
        factor, index, plan, buffers, T)
    mean = exp(log_mean)
    log_precision = _factor_bb2_log_precision(
        factor, index, plan, buffers, T)
    log_precision > log(floatmax(T)) && return _factor_rand_binomial(
        rng, T, trials, mean)
    alpha = exp(log_precision + log_mean)
    beta = exp(log_precision + log_complement)
    probability = if iszero(alpha) && iszero(beta)
        BRM.rand(rng, T) < mean ? one(T) : zero(T)
    elseif iszero(alpha)
        zero(T)
    elseif iszero(beta)
        one(T)
    else
        left = _factor_rand_unit_gamma(rng, T, alpha)
        right = _factor_rand_unit_gamma(rng, T, beta)
        total = left + right
        iszero(total) ?
            (BRM.rand(rng, T) < mean ? one(T) : zero(T)) : left / total
    end
    _factor_rand_binomial(rng, T, trials, probability)
end

@inline function _factor_terminal_base_sample(rng::BRM.AbstractRNG,
                                              ::Val{Name},
                                              prepared::FactorPrepared,
                                              buffers::FactorBuffers,
                                              index::Int) where {Name}
    site = getproperty(prepared.plan.graph.sites, Name)
    site_factor = site.factor
    factor = base_site_factor(site_factor)
    T = eltype(buffers.values)
    if factor isa NormalSiteFactor
        location = _factor_argument_at(
            factor.location, index, prepared.plan, buffers, T)
        scale = _factor_argument_at(
            factor.scale, index, prepared.plan, buffers, T)
        weight_factor = _weighted_site_factor(site_factor)
        if weight_factor !== nothing &&
           observation_weight_kind(weight_factor.weight) === :analytic
            weight = _factor_argument_at(
                weight_factor.values, index, prepared.plan, buffers, T)
            scale /= sqrt(weight)
        end
        location + scale * BRM.randn(rng, T)
    elseif factor isa StudentTSiteFactor
        degrees_freedom = _factor_argument_at(
            factor.degrees_freedom, index, prepared.plan, buffers, T)
        location = _factor_argument_at(
            factor.location, index, prepared.plan, buffers, T)
        scale = _factor_argument_at(
            factor.scale, index, prepared.plan, buffers, T)
        degrees_freedom > zero(T) && scale > zero(T) || throw(DomainError(
            (degrees_freedom, scale),
            "native PPL Student-t degrees of freedom and scale must be " *
            "positive"))
        if !isfinite(degrees_freedom)
            return location + scale * BRM.randn(rng, T)
        end
        chi_square = T(2) * _factor_rand_unit_gamma(
            rng, T, degrees_freedom / T(2))
        location + scale * BRM.randn(rng, T) /
            sqrt(chi_square / degrees_freedom)
    elseif factor isa BernoulliLogitSiteFactor
        logit = _factor_argument_at(
            factor.logit, index, prepared.plan, buffers, T)
        BRM.rand(rng, T) < BRM._native_ppl_logistic(logit)
    elseif factor isa ZeroInflatedPoissonSiteFactor
        log_zero, _ = _factor_zip_log_probabilities(
            factor, index, prepared.plan, buffers, T)
        BRM.rand(rng, T) < exp(log_zero) ? 0 :
            BRM._native_ppl_rand_poisson(
                rng, T, _factor_zip_log_rate(
                    factor, index, prepared.plan, buffers, T))
    elseif factor isa NegativeBinomial2SiteFactor
        _factor_rand_negative_binomial2(
            rng, factor, index, prepared.plan, buffers, T)
    elseif factor isa BetaBinomial2SiteFactor
        _factor_rand_beta_binomial2(
            rng, factor, index, prepared.plan, buffers, T)
    elseif factor isa BinomialLogitSiteFactor
        trials = Int(_factor_argument_at(
            factor.trials, index, prepared.plan, buffers, T))
        logit = _factor_argument_at(
            factor.logit, index, prepared.plan, buffers, T)
        _factor_rand_binomial(
            rng, T, trials, BRM._native_ppl_logistic(logit))
    elseif factor isa CategoricalLogitSiteFactor
        largest = _factor_categorical_largest(
            factor.logits, index, prepared.plan, buffers, T, zero(T))
        reference_mass = exp(-largest)
        total = _factor_categorical_total(
            factor.logits, index, prepared.plan, buffers, T,
            largest, reference_mass)
        threshold = BRM.rand(rng, T) * total
        threshold < reference_mass && return 1
        _factor_categorical_draw(
            factor.logits, threshold, index, prepared.plan, buffers, T,
            largest, reference_mass, 1)
    else
        log_rate = _factor_poisson_log_rate(
            factor, index, prepared.plan, buffers, T)
        BRM._native_ppl_rand_poisson(rng, T, log_rate)
    end
end


@inline _factor_terminal_base_sample(
        rng::BRM.AbstractRNG, prepared::FactorPrepared,
        buffers::FactorBuffers, index::Int) =
    _factor_terminal_base_sample(
        rng, Val(factor_output_site(prepared.plan)),
        prepared, buffers, index)

@inline function _factor_terminal_sample(rng::BRM.AbstractRNG,
                                         ::Val{Name},
                                         prepared::FactorPrepared,
                                         buffers::FactorBuffers,
                                         index::Int) where {Name}
    site = getproperty(prepared.plan.graph.sites, Name)
    evidence_factor = _evidence_site_factor(site.factor)
    evidence_factor === nothing && return _factor_terminal_base_sample(
        rng, Val(Name), prepared, buffers, index)
    T = eltype(buffers.values)
    lower = _factor_evidence_bound_at(
        evidence_factor.lower, index, prepared.plan, buffers, T)
    upper = _factor_evidence_bound_at(
        evidence_factor.upper, index, prepared.plan, buffers, T)
    kind = evidence_kind(evidence_factor.evidence)
    sample = _factor_terminal_base_sample(
        rng, Val(Name), prepared, buffers, index)
    kind === :interval_censored && return sample
    if kind === :censored
        lower !== nothing && sample < lower &&
            (sample = convert(typeof(sample), lower))
        upper !== nothing && sample > upper &&
            (sample = convert(typeof(sample), upper))
        return sample
    end
    attempts = 1
    while (lower !== nothing && sample < lower) ||
          (upper !== nothing && sample > upper)
        attempts == 1_000_000 && throw(CapabilityError(
            :predictive_support,
            "native PPL truncated predictive sampler did not reach its " *
            "support after $attempts proposals"))
        sample = _factor_terminal_base_sample(
            rng, Val(Name), prepared, buffers, index)
        attempts += 1
    end
    sample
end

@inline _factor_terminal_sample(
        rng::BRM.AbstractRNG, prepared::FactorPrepared,
        buffers::FactorBuffers, index::Int) =
    _factor_terminal_sample(
        rng, Val(factor_output_site(prepared.plan)),
        prepared, buffers, index)

function evaluate!(output::AbstractVector, work::FactorWorkspace,
                   prepared::FactorPrepared, position::AbstractVector,
                   query::BRM.NativePPLLinearPredictor)
    _factor_check_query_output(output, work, prepared, position, query)
    _factor_logdensity_kernel(position, prepared, work.primal)
    for index in eachindex(output)
        output[index] = _factor_terminal_linear(
            prepared, work.primal, index)
    end
    output
end

function evaluate!(output::AbstractVector, work::FactorWorkspace,
                   prepared::FactorPrepared, position::AbstractVector,
                   query::BRM.NativePPLNodeOutput)
    _factor_check_query_output(output, work, prepared, position, query)
    _factor_node_query_kernel(position, prepared, work.primal, query)
    for index in eachindex(output)
        output[index] = _factor_node_output_at(
            query, prepared, work.primal, index)
    end
    output
end

function evaluate!(output::AbstractVector, work::FactorWorkspace,
                   prepared::FactorPrepared, position::AbstractVector,
                   query::BRM.NativePPLPointwiseLogLikelihood)
    _factor_check_query_output(output, work, prepared, position, query)
    has_response(prepared) || throw(ArgumentError(
        "native PPL pointwise log likelihood requires a conditioned " *
        "broadcast response"))
    _factor_logdensity_kernel(position, prepared, work.primal)
    copyto!(output, work.primal.pointwise_loglikelihood)
end

function evaluate!(outputs::NamedTuple, work::FactorWorkspace,
                   prepared::FactorPrepared, position::AbstractVector,
                   query::BRM.NativePPLPointwiseLogLikelihood)
    _factor_check_outcome_query_outputs(
        outputs, work, prepared, position, query)
    for output in factor_output_sites(prepared.plan)
        hasproperty(prepared.conditions, output) || throw(ArgumentError(
            "native PPL pointwise log likelihood for `$output` requires " *
            "that outcome to be conditioned"))
    end
    _factor_logdensity_kernel(position, prepared, work.primal)
    _factor_write_pointwise_outputs!(outputs, prepared, work.primal)
end

function evaluate(work::FactorWorkspace, prepared::FactorPrepared,
                  position::AbstractVector, query::BRM.NativePPLQuery)
    output = allocate_output(prepared, query)
    evaluate!(output, work, prepared, position, query)
end

function evaluate!(rng::BRM.AbstractRNG, output::AbstractVector,
                   work::FactorWorkspace, prepared::FactorPrepared,
                   position::AbstractVector,
                   query::BRM.NativePPLLinearPredictor)
    _factor_check_query_output(output, work, prepared, position, query)
    if _has_generated_groups(prepared.plan)
        _factor_predictive_kernel!(
            rng, position, prepared, work.primal)
    else
        _factor_logdensity_kernel(position, prepared, work.primal)
    end
    for index in eachindex(output)
        output[index] = _factor_terminal_linear(
            prepared, work.primal, index)
    end
    output
end

function evaluate(rng::BRM.AbstractRNG, work::FactorWorkspace,
                  prepared::FactorPrepared, position::AbstractVector,
                  query::BRM.NativePPLLinearPredictor)
    output = allocate_output(prepared, query)
    evaluate!(rng, output, work, prepared, position, query)
end

function evaluate!(rng::BRM.AbstractRNG, output::AbstractVector,
                   work::FactorWorkspace, prepared::FactorPrepared,
                   position::AbstractVector,
                   query::BRM.NativePPLNodeOutput)
    _factor_check_query_output(output, work, prepared, position, query)
    if _has_generated_groups(prepared.plan)
        _factor_predictive_kernel!(
            rng, position, prepared, work.primal)
    else
        _factor_logdensity_kernel(position, prepared, work.primal)
    end
    for index in eachindex(output)
        output[index] = _factor_node_output_at(
            query, prepared, work.primal, index)
    end
    output
end

function evaluate(rng::BRM.AbstractRNG, work::FactorWorkspace,
                  prepared::FactorPrepared, position::AbstractVector,
                  query::BRM.NativePPLNodeOutput)
    output = allocate_output(prepared, query)
    evaluate!(rng, output, work, prepared, position, query)
end

function simulate!(rng::BRM.AbstractRNG, output::AbstractVector,
                   work::FactorWorkspace, prepared::FactorPrepared,
                   position::AbstractVector,
    query::BRM.NativePPLPosteriorPredictive=
                       BRM.NativePPLPosteriorPredictive())
    _factor_check_query_output(output, work, prepared, position, query)
    if _has_generated_groups(prepared.plan)
        _factor_predictive_kernel!(
            rng, position, prepared, work.primal)
    else
        _factor_logdensity_kernel(position, prepared, work.primal)
    end
    for index in eachindex(output)
        output[index] = _factor_terminal_sample(
            rng, prepared, work.primal, index)
    end
    output
end

function simulate!(rng::BRM.AbstractRNG, outputs::NamedTuple,
                   work::FactorWorkspace, prepared::FactorPrepared,
                   position::AbstractVector,
                   query::BRM.NativePPLPosteriorPredictive=
                       BRM.NativePPLPosteriorPredictive())
    _factor_check_outcome_query_outputs(
        outputs, work, prepared, position, query)
    if _has_generated_groups(prepared.plan)
        _factor_predictive_kernel!(
            rng, position, prepared, work.primal)
    else
        _factor_logdensity_kernel(position, prepared, work.primal)
    end
    _factor_write_predictive_outputs!(
        rng, outputs, prepared, work.primal)
end

function simulate(rng::BRM.AbstractRNG, work::FactorWorkspace,
                  prepared::FactorPrepared, position::AbstractVector,
                  query::BRM.NativePPLPosteriorPredictive=
                      BRM.NativePPLPosteriorPredictive())
    output = allocate_output(prepared, query)
    simulate!(rng, output, work, prepared, position, query)
end

@inline _factor_inverse(::IdentityTransform, value) = value
@inline _factor_inverse(::ExpTransform, value) = log(value)

@inline _factor_rand(rng::BRM.AbstractRNG, ::StandardNormalSiteFactor,
                     plan, buffers, ::Type{T}) where {T} =
    BRM.randn(rng, T)

@inline function _factor_rand(rng::BRM.AbstractRNG,
        factor::NormalSiteFactor, plan, buffers, ::Type{T}) where {T}
    location = _factor_argument(factor.location, plan, buffers, T)
    scale = _factor_argument(factor.scale, plan, buffers, T)
    location + scale * BRM.randn(rng, T)
end

@inline function _factor_rand(rng::BRM.AbstractRNG,
        factor::ExponentialSiteFactor, plan, buffers, ::Type{T}) where {T}
    scale = _factor_argument(factor.scale, plan, buffers, T)
    scale * BRM.randexp(rng, T)
end

@inline _factor_require_prior_simulation(
    name::Symbol,
    factor::Union{StandardNormalSiteFactor,NormalSiteFactor,
                  ExponentialSiteFactor}) = nothing

@noinline function _factor_require_prior_simulation(name::Symbol, factor)
    throw(CapabilityError(
        :prior_simulation,
        "native PPL prior simulation has no sampler for free site `$name` " *
        "with factor `$(typeof(factor))`"))
end

@inline _factor_require_prior_simulation(
    name::Symbol,
    site::StochasticSite{S,Tr,F,Sh,A,K}) where {S,Tr,F,Sh,A,K} = nothing

@inline function _factor_require_prior_simulation(
        name::Symbol,
        site::StochasticSite{S,Tr,F,Sh,FreeSite,K}) where {S,Tr,F,Sh,K}
    _factor_require_prior_simulation(name, site.factor)
end

@generated function _factor_require_prior_simulation(
        ::Val{Names}, sites::Sites) where {Names,Sites<:Tuple}
    calls = Any[]
    for (index, name) in enumerate(Names)
        push!(calls, :(_factor_require_prior_simulation(
            $(QuoteNode(name)), getfield(sites, $index))))
    end
    Expr(:block, calls..., :(nothing))
end

@inline function _factor_require_prior_simulation(plan::FactorPlan)
    sites = plan.graph.sites
    _factor_require_prior_simulation(
        Val(Tuple(keys(sites))), Tuple(sites))
end

@inline function _factor_generated_group_sample!(rng::BRM.AbstractRNG,
        ::Val{Name},
        site::StochasticSite,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,T}
    hasproperty(prepared.plan.generated_group_indices, Name) || return nothing
    generated_indices = getproperty(
        prepared.plan.generated_group_indices, Name)
    for index in generated_indices
        buffers.generated_group_values[index] = _factor_rand(
            rng, site.factor, prepared.plan, buffers, T)
    end
    nothing
end

@generated function _factor_predictive_schedule!(rng::BRM.AbstractRNG,
        ::Val{Names}, ::Val{SiteNames}, sites::Sites,
        ::Val{NodeNames}, nodes::Nodes,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {
            Names,SiteNames,Sites<:Tuple,NodeNames,Nodes<:Tuple,T}
    calls = Any[]
    for name in Names
        site_index = findfirst(==(name), SiteNames)
        call = if site_index === nothing
            node_index = findfirst(==(name), NodeNames)
            node_index === nothing && error("unknown factor schedule entry $name")
            :(_factor_node_logdensity!(
                Val($(QuoteNode(name))), getfield(nodes, $node_index),
                position, prepared, buffers))
        else
            quote
                _factor_site_logdensity!(
                    Val($(QuoteNode(name))), getfield(sites, $site_index),
                    position, prepared, buffers)
                _factor_generated_group_sample!(
                    rng, Val($(QuoteNode(name))),
                    getfield(sites, $site_index), position, prepared, buffers)
            end
        end
        push!(calls, call)
    end
    Expr(:block, calls..., :(nothing))
end

@inline function _factor_predictive_kernel!(rng::BRM.AbstractRNG,
        position::AbstractVector{T}, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {T}
    graph = prepared.plan.graph
    _factor_predictive_schedule!(
        rng, Val(graph.schedule), Val(Tuple(keys(graph.sites))),
        Tuple(graph.sites), Val(Tuple(keys(graph.nodes))), Tuple(graph.nodes),
        position, prepared, buffers)
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,ScalarSiteShape,FreeSite},
        position::AbstractVector{T}, output,
        prepared::FactorPrepared, buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    value = _factor_rand(rng, site.factor, prepared.plan, buffers, T)
    coordinates = getproperty(prepared.plan.graph.coordinates, Name)
    position[first(coordinates.indices)] = _factor_inverse(
        site.transform, value)
    buffers.values[getproperty(prepared.plan.site_indices, Name)] = value
    nothing
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,ScalarSiteShape,ConditionedSite},
        position::AbstractVector{T}, output,
        prepared::FactorPrepared, buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    buffers.values[getproperty(prepared.plan.site_indices, Name)] =
        getproperty(prepared.conditions, Name)
    nothing
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,BlockSiteShape,FreeSite},
        position::AbstractVector{T}, output,
        prepared::FactorPrepared, buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    coordinates = getproperty(prepared.plan.graph.coordinates, Name)
    for coordinate in coordinates.indices
        value = _factor_rand(
            rng, site.factor, prepared.plan, buffers, T)
        position[coordinate] = _factor_inverse(site.transform, value)
    end
    nothing
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,BlockSiteShape,ConditionedSite},
        position::AbstractVector{T}, output,
        prepared::FactorPrepared, buffers::FactorBuffers{T}) where {Name,S,Tr,F,T}
    nothing
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,BroadcastSiteShape,A},
        position::AbstractVector{T}, output::AbstractVector,
        prepared::FactorPrepared, buffers::FactorBuffers{T}) where {Name,S,Tr,F,A,T}
    for index in eachindex(output)
        output[index] = _factor_terminal_sample(
            rng, Val(Name), prepared, buffers, index)
    end
    nothing
end

@inline function _factor_site_sample!(rng::BRM.AbstractRNG, ::Val{Name},
        site::StochasticSite{S,Tr,F,BroadcastSiteShape,A},
        position::AbstractVector{T}, outputs::NamedTuple,
        prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {Name,S,Tr,F,A,T}
    output = getproperty(outputs, Name)
    for index in eachindex(output)
        output[index] = _factor_terminal_sample(
            rng, Val(Name), prepared, buffers, index)
    end
    nothing
end

@generated function _factor_sample_schedule!(rng::BRM.AbstractRNG,
        ::Val{Names}, ::Val{SiteNames}, sites::Sites,
        ::Val{NodeNames}, nodes::Nodes,
        position::AbstractVector{T},
        output, prepared::FactorPrepared,
        buffers::FactorBuffers{T}) where {
            Names,SiteNames,Sites<:Tuple,NodeNames,Nodes<:Tuple,T}
    calls = Any[]
    for name in Names
        site_index = findfirst(==(name), SiteNames)
        call = if site_index === nothing
            node_index = findfirst(==(name), NodeNames)
            node_index === nothing && error("unknown factor schedule entry $name")
            :(_factor_node_logdensity!(
                Val($(QuoteNode(name))), getfield(nodes, $node_index),
                position, prepared, buffers))
        else
            :(_factor_site_sample!(
                rng, Val($(QuoteNode(name))),
                getfield(sites, $site_index), position, output,
                prepared, buffers))
        end
        push!(calls, call)
    end
    Expr(:block, calls..., :(output))
end

function simulate_prior!(rng::BRM.AbstractRNG, position::AbstractVector,
                         output::AbstractVector, work::FactorWorkspace,
                         prepared::FactorPrepared)
    _factor_require_fixed_coordinates(prepared.plan)
    _factor_require_prior_simulation(prepared.plan)
    _factor_check_query_output(
        output, work, prepared, position,
        BRM.NativePPLPosteriorPredictive())
    graph = prepared.plan.graph
    _factor_sample_schedule!(
        rng, Val(graph.schedule), Val(Tuple(keys(graph.sites))),
        Tuple(graph.sites), Val(Tuple(keys(graph.nodes))), Tuple(graph.nodes),
        position, output, prepared, work.primal)
end

function simulate_prior!(rng::BRM.AbstractRNG, position::AbstractVector,
                         outputs::NamedTuple, work::FactorWorkspace,
                         prepared::FactorPrepared)
    _factor_require_fixed_coordinates(prepared.plan)
    _factor_require_prior_simulation(prepared.plan)
    _factor_check_outcome_query_outputs(
        outputs, work, prepared, position,
        BRM.NativePPLPosteriorPredictive())
    graph = prepared.plan.graph
    _factor_sample_schedule!(
        rng, Val(graph.schedule), Val(Tuple(keys(graph.sites))),
        Tuple(graph.sites), Val(Tuple(keys(graph.nodes))), Tuple(graph.nodes),
        position, outputs, prepared, work.primal)
end

function simulate_prior(rng::BRM.AbstractRNG, work::FactorWorkspace,
                        prepared::FactorPrepared)
    position = Vector{eltype(work)}(
        undef, BRM.LogDensityProblems.dimension(prepared))
    output = allocate_output(
        prepared, BRM.NativePPLPosteriorPredictive())
    simulate_prior!(rng, position, output, work, prepared)
    (; position, response=output)
end

function _factor_check_positions(prepared::FactorPrepared,
                                 positions::AbstractMatrix)
    dimension = BRM.LogDensityProblems.dimension(prepared)
    size(positions, 2) == dimension || throw(DimensionMismatch(
        "native PPL factor draw matrix has $(size(positions, 2)) columns; " *
        "expected $dimension"))
    axes(positions) == (Base.OneTo(size(positions, 1)),
                        Base.OneTo(dimension)) || throw(ArgumentError(
        "native PPL factor draw matrices require one-based axes"))
    nothing
end

function batch_output_signature(prepared::FactorPrepared,
                                positions::AbstractMatrix,
                                query::BRM.NativePPLQuery)
    _factor_check_positions(prepared, positions)
    element = output_signature(prepared, query)
    _factor_batch_output_signature(prepared, positions, element)
end

function _factor_batch_output_signature(
        prepared::FactorPrepared, positions::AbstractMatrix,
        element::BRM.NativePPLOutputSignature)
    BRM.NativePPLBatchOutputSignature(
        BRM.NativePPLAxis(:draw, Base.OneTo(size(positions, 1))),
        BRM.native_output_axis(element), element.element_type,
        BRM.NativePPLDenseMatrixLayout())
end


function _factor_batch_output_signature(
        prepared::FactorPrepared, positions::AbstractMatrix,
        elements::NamedTuple)
    map(element -> _factor_batch_output_signature(
            prepared, positions, element), elements)
end

function batch_output_signature(prepared::FactorPrepared,
                                positions::AbstractMatrix,
                                queries::NamedTuple{Names}) where {Names}
    _factor_check_positions(prepared, positions)
    all(query -> query isa BRM.NativePPLQuery, Tuple(queries)) ||
        throw(ArgumentError(
            "native PPL factor query bundles require typed graph queries"))
    NamedTuple{Names}(map(
        query -> batch_output_signature(prepared, positions, query),
        Tuple(queries)))
end

function allocate_output(signature::BRM.NativePPLBatchOutputSignature,
                         prepared::FactorPrepared)
    draw_axis, observation_axis = BRM.native_output_axes(signature)
    draw_axis.keys == Base.OneTo(length(draw_axis)) || throw(CapabilityError(
        :output_layout, "factor draw output requires a one-based draw axis"))
    observation_axis.keys == Base.OneTo(length(observation_axis)) ||
        throw(CapabilityError(
            :output_layout,
            "factor draw output requires a one-based observation axis"))
    Matrix{BRM.native_output_eltype(signature, prepared)}(
        undef, length(draw_axis), length(observation_axis))
end

function allocate_output(signatures::NamedTuple{Names},
                         prepared::FactorPrepared) where {Names}
    NamedTuple{Names}(map(
        signature -> allocate_output(signature, prepared),
        Tuple(signatures)))
end

function _factor_check_batch_output(output::AbstractMatrix,
                                    work::FactorWorkspace,
                                    prepared::FactorPrepared,
                                    positions::AbstractMatrix,
                                    signature::BRM.NativePPLBatchOutputSignature)
    draw_axis, observation_axis = BRM.native_output_axes(signature)
    axes(output) == (draw_axis.keys, observation_axis.keys) ||
        throw(DimensionMismatch(
            "native PPL factor batch output axes $(axes(output)) do not " *
            "match $((draw_axis.keys, observation_axis.keys))"))
    expected_eltype = BRM.native_output_eltype(signature, prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL factor batch output eltype $(eltype(output)) does not " *
        "match declared eltype $expected_eltype"))
    eltype(positions) === eltype(work) || throw(ArgumentError(
        "native PPL factor draw eltype $(eltype(positions)) does not match " *
        "workspace eltype $(eltype(work))"))
    Base.mightalias(output, positions) && throw(ArgumentError(
        "native PPL factor batch output must not alias draw positions"))
    for buffer in (work.gradient, work.primal.values,
                   work.primal.generated_group_values,
                   work.primal.node_values, work.primal.node_rows,
                   work.primal.pointwise_loglikelihood)
        Base.mightalias(output, buffer) && throw(ArgumentError(
            "native PPL factor batch output must not alias workspace storage"))
        Base.mightalias(positions, buffer) && throw(ArgumentError(
            "native PPL factor draw positions must not alias workspace storage"))
    end
    nothing
end


function _factor_check_batch_output(output::AbstractMatrix,
                                    work::FactorWorkspace,
                                    prepared::FactorPrepared,
                                    positions::AbstractMatrix,
                                    query::BRM.NativePPLQuery)
    signature = batch_output_signature(prepared, positions, query)
    signature isa BRM.NativePPLBatchOutputSignature || throw(
        CapabilityError(
            :query,
            "multiple-outcome batch output must be a NamedTuple keyed by " *
            "$(collect(factor_output_sites(prepared.plan)))"))
    _factor_check_batch_output(
        output, work, prepared, positions, signature)
end

@inline @generated function _factor_check_outcome_batch_outputs(
        outputs::NamedTuple{Names}, work::FactorWorkspace,
        prepared::FactorPrepared, positions::AbstractMatrix,
        query::Q) where {Names,Q<:Union{
            BRM.NativePPLPointwiseLogLikelihood,
            BRM.NativePPLPosteriorPredictive}}
    checks = Any[]
    for index in eachindex(Names)
        name = Names[index]
        element = Q <: BRM.NativePPLPointwiseLogLikelihood ?
            :(_factor_output_signature(prepared.plan)) :
            :(_factor_predictive_output_signature(
                prepared.plan, Val($(QuoteNode(name)))))
        push!(checks, :(_factor_check_batch_output(
            getfield(outputs, $index), work, prepared, positions,
            _factor_batch_output_signature(
                prepared, positions, $element))))
    end
    for left in eachindex(Names), right in (left + 1):length(Names)
        push!(checks, quote
            Base.mightalias(getfield(outputs, $left),
                            getfield(outputs, $right)) && throw(
                ArgumentError(
                    "native PPL outcome batch outputs must not alias " *
                    "each other"))
        end)
    end
    quote
        expected = factor_output_sites(prepared.plan)
        Names == expected || throw(ArgumentError(
            "native PPL outcome batch fields $Names do not match terminal " *
            "outcomes $expected"))
        $(checks...)
        outputs
    end
end

@inline @generated function _factor_write_pointwise_output_matrices!(
        outputs::NamedTuple{Names}, draw,
        prepared::FactorPrepared, buffers::FactorBuffers) where {Names}
    calls = map(eachindex(Names)) do output_index
        name = Names[output_index]
        quote
            output = getfield(outputs, $output_index)
            for row in axes(output, 2)
                output[draw, row] = _factor_pointwise_at(
                    Val($(QuoteNode(name))), prepared, buffers, row)
            end
        end
    end
    Expr(:block, calls..., :(outputs))
end

@inline @generated function _factor_write_predictive_output_matrices!(
        rng::BRM.AbstractRNG, outputs::NamedTuple{Names}, draw,
        prepared::FactorPrepared, buffers::FactorBuffers) where {Names}
    calls = map(eachindex(Names)) do output_index
        name = Names[output_index]
        quote
            output = getfield(outputs, $output_index)
            for row in axes(output, 2)
                output[draw, row] = _factor_terminal_sample(
                    rng, Val($(QuoteNode(name))), prepared, buffers, row)
            end
        end
    end
    Expr(:block, calls..., :(outputs))
end

function evaluate_draws!(output::AbstractMatrix, work::FactorWorkspace,
                         prepared::FactorPrepared,
                         positions::AbstractMatrix,
                         query::BRM.NativePPLQuery)
    _factor_check_batch_output(
        output, work, prepared, positions, query)
    for draw in axes(positions, 1)
        evaluate!(
            @view(output[draw, :]), work, prepared,
            @view(positions[draw, :]), query)
    end
    output
end

function evaluate_draws!(outputs::NamedTuple, work::FactorWorkspace,
                         prepared::FactorPrepared,
                         positions::AbstractMatrix,
                         query::BRM.NativePPLPointwiseLogLikelihood)
    _factor_check_outcome_batch_outputs(
        outputs, work, prepared, positions, query)
    for output in factor_output_sites(prepared.plan)
        hasproperty(prepared.conditions, output) || throw(ArgumentError(
            "native PPL pointwise log likelihood for `$output` requires " *
            "that outcome to be conditioned"))
    end
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _factor_check_execution(work, prepared, position)
        _factor_logdensity_kernel(position, prepared, work.primal)
        _factor_write_pointwise_output_matrices!(
            outputs, draw, prepared, work.primal)
    end
    outputs
end

function evaluate_draws(work::FactorWorkspace, prepared::FactorPrepared,
                        positions::AbstractMatrix,
                        query::BRM.NativePPLQuery)
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    evaluate_draws!(output, work, prepared, positions, query)
end

function evaluate_draws!(rng::BRM.AbstractRNG, output::AbstractMatrix,
                         work::FactorWorkspace,
                         prepared::FactorPrepared,
                         positions::AbstractMatrix,
                         query::Union{BRM.NativePPLLinearPredictor,
                                      BRM.NativePPLNodeOutput})
    _factor_check_batch_output(
        output, work, prepared, positions, query)
    for draw in axes(positions, 1)
        evaluate!(
            rng, @view(output[draw, :]), work, prepared,
            @view(positions[draw, :]), query)
    end
    output
end

function evaluate_draws(rng::BRM.AbstractRNG, work::FactorWorkspace,
                        prepared::FactorPrepared,
                        positions::AbstractMatrix,
                        query::Union{BRM.NativePPLLinearPredictor,
                                     BRM.NativePPLNodeOutput})
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    evaluate_draws!(rng, output, work, prepared, positions, query)
end

function simulate_draws!(rng::BRM.AbstractRNG, output::AbstractMatrix,
                         work::FactorWorkspace, prepared::FactorPrepared,
                         positions::AbstractMatrix,
                         query::BRM.NativePPLPosteriorPredictive=
                             BRM.NativePPLPosteriorPredictive())
    _factor_check_batch_output(
        output, work, prepared, positions, query)
    for draw in axes(positions, 1)
        simulate!(
            rng, @view(output[draw, :]), work, prepared,
            @view(positions[draw, :]), query)
    end
    output
end

function simulate_draws!(rng::BRM.AbstractRNG, outputs::NamedTuple,
                         work::FactorWorkspace,
                         prepared::FactorPrepared,
                         positions::AbstractMatrix,
                         query::BRM.NativePPLPosteriorPredictive=
                             BRM.NativePPLPosteriorPredictive())
    _factor_check_outcome_batch_outputs(
        outputs, work, prepared, positions, query)
    for draw in axes(positions, 1)
        position = @view positions[draw, :]
        _factor_check_execution(work, prepared, position)
        if _has_generated_groups(prepared.plan)
            _factor_predictive_kernel!(
                rng, position, prepared, work.primal)
        else
            _factor_logdensity_kernel(position, prepared, work.primal)
        end
        _factor_write_predictive_output_matrices!(
            rng, outputs, draw, prepared, work.primal)
    end
    outputs
end

function simulate_draws(rng::BRM.AbstractRNG, work::FactorWorkspace,
                        prepared::FactorPrepared,
                        positions::AbstractMatrix,
                        query::BRM.NativePPLPosteriorPredictive=
                            BRM.NativePPLPosteriorPredictive())
    signature = batch_output_signature(prepared, positions, query)
    output = allocate_output(signature, prepared)
    simulate_draws!(rng, output, work, prepared, positions, query)
end

@inline function _factor_write_bundle_query!(rng, output::AbstractVector,
        ::BRM.NativePPLLinearPredictor, work::FactorWorkspace,
        prepared::FactorPrepared)
    for index in eachindex(output)
        output[index] = _factor_terminal_linear(
            prepared, work.primal, index)
    end
    output
end

@inline function _factor_write_bundle_query!(rng, output::AbstractVector,
        query::BRM.NativePPLNodeOutput, work::FactorWorkspace,
        prepared::FactorPrepared)
    for index in eachindex(output)
        output[index] = _factor_node_output_at(
            query, prepared, work.primal, index)
    end
    output
end

@inline function _factor_write_bundle_query!(rng, output::AbstractVector,
        ::BRM.NativePPLPointwiseLogLikelihood, work::FactorWorkspace,
        prepared::FactorPrepared)
    copyto!(output, work.primal.pointwise_loglikelihood)
end

@inline function _factor_write_bundle_query!(rng::BRM.AbstractRNG,
        output::AbstractVector, ::BRM.NativePPLPosteriorPredictive,
        work::FactorWorkspace, prepared::FactorPrepared)
    for index in eachindex(output)
        output[index] = _factor_terminal_sample(
            rng, prepared, work.primal, index)
    end
    output
end

@inline function _factor_write_bundle_matrix_query!(rng,
        output::AbstractMatrix, draw, ::BRM.NativePPLLinearPredictor,
        work::FactorWorkspace, prepared::FactorPrepared)
    for index in axes(output, 2)
        output[draw, index] = _factor_terminal_linear(
            prepared, work.primal, index)
    end
    output
end

@inline function _factor_write_bundle_matrix_query!(rng,
        output::AbstractMatrix, draw, query::BRM.NativePPLNodeOutput,
        work::FactorWorkspace, prepared::FactorPrepared)
    for index in axes(output, 2)
        output[draw, index] = _factor_node_output_at(
            query, prepared, work.primal, index)
    end
    output
end

@inline function _factor_write_bundle_matrix_query!(rng,
        output::AbstractMatrix, draw,
        ::BRM.NativePPLPointwiseLogLikelihood,
        work::FactorWorkspace, prepared::FactorPrepared)
    for index in axes(output, 2)
        output[draw, index] = work.primal.pointwise_loglikelihood[index]
    end
    output
end

@inline function _factor_write_bundle_matrix_query!(rng::BRM.AbstractRNG,
        output::AbstractMatrix, draw, ::BRM.NativePPLPosteriorPredictive,
        work::FactorWorkspace, prepared::FactorPrepared)
    for index in axes(output, 2)
        output[draw, index] = _factor_terminal_sample(
            rng, prepared, work.primal, index)
    end
    output
end

@inline _factor_write_bundle_matrix_query!(rng,
        outputs::NamedTuple, draw,
        ::BRM.NativePPLPointwiseLogLikelihood,
        work::FactorWorkspace, prepared::FactorPrepared) =
    _factor_write_pointwise_output_matrices!(
        outputs, draw, prepared, work.primal)

@inline _factor_write_bundle_matrix_query!(rng::BRM.AbstractRNG,
        outputs::NamedTuple, draw,
        ::BRM.NativePPLPosteriorPredictive,
        work::FactorWorkspace, prepared::FactorPrepared) =
    _factor_write_predictive_output_matrices!(
        rng, outputs, draw, prepared, work.primal)

@inline _factor_bundle_requires_rng(::Tuple{}) = false
@inline _factor_bundle_requires_rng(queries::Tuple) =
    first(queries) isa BRM.NativePPLPosteriorPredictive ||
    _factor_bundle_requires_rng(Base.tail(queries))

@inline _factor_bundle_requires_response(::Tuple{}) = false
@inline _factor_bundle_requires_response(queries::Tuple) =
    first(queries) isa BRM.NativePPLPointwiseLogLikelihood ||
    _factor_bundle_requires_response(Base.tail(queries))

@inline function _factor_write_bundle_draw!(rng, ::Tuple{}, ::Tuple{},
        draw, work::FactorWorkspace, prepared::FactorPrepared)
    nothing
end

@inline function _factor_write_bundle_draw!(rng, outputs::Tuple,
        queries::Tuple, draw, work::FactorWorkspace,
        prepared::FactorPrepared)
    _factor_write_bundle_matrix_query!(
        rng, first(outputs), draw, first(queries), work, prepared)
    _factor_write_bundle_draw!(
        rng, Base.tail(outputs), Base.tail(queries), draw, work, prepared)
end

@inline @generated function _factor_write_bundle_draw!(rng,
        outputs::NamedTuple{Names}, queries::NamedTuple{Names}, draw,
        work::FactorWorkspace, prepared::FactorPrepared) where {Names}
    calls = map(eachindex(Names)) do index
        :(_factor_write_bundle_matrix_query!(
            rng, getfield(outputs, $index), draw,
            getfield(queries, $index), work, prepared))
    end
    Expr(:block, calls..., :(nothing))
end

@inline function _factor_check_bundle_matrix(
        output::AbstractMatrix, work::FactorWorkspace,
        prepared::FactorPrepared, positions::AbstractMatrix,
        query::BRM.NativePPLQuery)
    expected_axes = (
        Base.OneTo(size(positions, 1)), prepared.plan.observation_axis.keys)
    axes(output) == expected_axes || throw(DimensionMismatch(
        "native PPL factor bundle output axes $(axes(output)) do not match " *
        "$expected_axes"))
    expected_eltype = BRM.native_output_eltype(
        output_signature(prepared, query), prepared)
    eltype(output) === expected_eltype || throw(ArgumentError(
        "native PPL factor bundle output eltype $(eltype(output)) does not " *
        "match declared eltype $expected_eltype"))
    eltype(positions) === eltype(work) || throw(ArgumentError(
        "native PPL factor draw eltype $(eltype(positions)) does not match " *
        "workspace eltype $(eltype(work))"))
    Base.mightalias(output, positions) && throw(ArgumentError(
        "native PPL factor bundle output must not alias draw positions"))
    for buffer in (work.gradient, work.primal.values,
                   work.primal.generated_group_values,
                   work.primal.node_values, work.primal.node_rows,
                   work.primal.pointwise_loglikelihood)
        Base.mightalias(output, buffer) && throw(ArgumentError(
            "native PPL factor bundle output must not alias workspace storage"))
        Base.mightalias(positions, buffer) && throw(ArgumentError(
            "native PPL factor draw positions must not alias workspace storage"))
    end
    nothing
end

@inline function _factor_check_bundle_value(
        output::AbstractMatrix, work::FactorWorkspace,
        prepared::FactorPrepared, positions::AbstractMatrix,
        query::BRM.NativePPLQuery)
    _factor_check_bundle_matrix(
        output, work, prepared, positions, query)
end

@inline function _factor_check_bundle_value(
        outputs::NamedTuple, work::FactorWorkspace,
        prepared::FactorPrepared, positions::AbstractMatrix,
        query::BRM.NativePPLPointwiseLogLikelihood)
    _factor_check_outcome_batch_outputs(
        outputs, work, prepared, positions, query)
    for output in factor_output_sites(prepared.plan)
        hasproperty(prepared.conditions, output) || throw(ArgumentError(
            "native PPL pointwise log likelihood for `$output` requires " *
            "that outcome to be conditioned"))
    end
    outputs
end

@inline function _factor_check_bundle_value(
        outputs::NamedTuple, work::FactorWorkspace,
        prepared::FactorPrepared, positions::AbstractMatrix,
        query::BRM.NativePPLPosteriorPredictive)
    _factor_check_outcome_batch_outputs(
        outputs, work, prepared, positions, query)
end

@inline function _factor_check_bundle_fields(outputs, queries,
        work::FactorWorkspace, prepared::FactorPrepared,
        positions::AbstractMatrix, ::Val{0})
    nothing
end

@inline function _factor_check_bundle_fields(outputs, queries,
        work::FactorWorkspace, prepared::FactorPrepared,
        positions::AbstractMatrix, ::Val{Index}) where {Index}
    query = getfield(queries, Index)
    query isa BRM.NativePPLQuery || throw(ArgumentError(
        "native PPL factor query bundles require typed graph queries"))
    output = getfield(outputs, Index)
    output isa Union{AbstractMatrix,NamedTuple} || throw(ArgumentError(
        "native PPL factor bundle outputs must be matrices or named " *
        "outcome matrices"))
    _factor_check_bundle_value(
        output, work, prepared, positions, query)
    _factor_check_bundle_fields(
        outputs, queries, work, prepared, positions, Val(Index - 1))
end

function _factor_output_leaf_expressions(type, expression)
    type <: NamedTuple || return type <: AbstractArray ? [expression] : Any[]
    leaves = Any[]
    for (index, field_type) in enumerate(fieldtypes(type))
        append!(leaves, _factor_output_leaf_expressions(
            field_type, :(getfield($expression, $index))))
    end
    leaves
end

@inline @generated function _factor_check_bundle_output_aliases(
        outputs::NamedTuple{Names,Types}) where {Names,Types}
    leaves = Any[]
    for (index, field_type) in enumerate(fieldtypes(Types))
        append!(leaves, _factor_output_leaf_expressions(
            field_type, :(getfield(outputs, $index))))
    end
    calls = Any[]
    for left in eachindex(leaves), right in (left + 1):length(leaves)
        push!(calls, quote
            Base.mightalias(
                $(leaves[left]), $(leaves[right])) &&
                throw(ArgumentError(
                    "native PPL factor bundle outputs must not alias each other"))
        end)
    end
    Expr(:block, calls..., :(nothing))
end


@inline function _factor_execute_bundle_draw!(rng,
        work::FactorWorkspace, prepared::FactorPrepared,
        position::AbstractVector)
    _factor_check_execution(work, prepared, position)
    if _has_generated_groups(prepared.plan)
        _factor_predictive_kernel!(
            rng, position, prepared, work.primal)
    else
        _factor_logdensity_kernel(position, prepared, work.primal)
    end
    nothing
end

@inline function _factor_execute_draws!(rng, outputs::NamedTuple{Names},
                                work::FactorWorkspace,
                                prepared::FactorPrepared,
                                positions::AbstractMatrix,
                                queries::NamedTuple{Names}) where {Names}
    query_values = Tuple(queries)
    rng === nothing && _factor_bundle_requires_rng(query_values) &&
        throw(ArgumentError(
            "native PPL factor query bundle contains an RNG query; pass an RNG"))
    rng === nothing && _has_generated_groups(prepared.plan) && throw(
        ArgumentError(
            "native PPL generated-group queries require an RNG; pass one " *
            "even when requesting only a linear predictor"))
    _factor_bundle_requires_response(query_values) && !has_response(prepared) &&
        throw(ArgumentError(
            "native PPL factor pointwise query requires a conditioned response"))
    _factor_check_positions(prepared, positions)
    _factor_check_bundle_fields(
        outputs, queries, work, prepared, positions, Val{length(Names)}())
    _factor_check_bundle_output_aliases(outputs)
    for draw in axes(positions, 1)
        _factor_execute_bundle_draw!(
            rng, work, prepared, @view(positions[draw, :]))
        _factor_write_bundle_draw!(
            rng, outputs, queries, draw, work, prepared)
    end
    outputs
end

execute_draws!(outputs::NamedTuple, work::FactorWorkspace,
               prepared::FactorPrepared, positions::AbstractMatrix,
               queries::NamedTuple) =
    _factor_execute_draws!(
        nothing, outputs, work, prepared, positions, queries)
execute_draws!(rng::BRM.AbstractRNG, outputs::NamedTuple,
               work::FactorWorkspace, prepared::FactorPrepared,
               positions::AbstractMatrix, queries::NamedTuple) =
    _factor_execute_draws!(rng, outputs, work, prepared, positions, queries)

function execute_draws(work::FactorWorkspace, prepared::FactorPrepared,
                       positions::AbstractMatrix, queries::NamedTuple)
    _factor_bundle_requires_rng(Tuple(queries)) && throw(ArgumentError(
        "native PPL factor query bundle contains an RNG query; pass an RNG"))
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(outputs, work, prepared, positions, queries)
end

function execute_draws(rng::BRM.AbstractRNG, work::FactorWorkspace,
                       prepared::FactorPrepared, positions::AbstractMatrix,
                       queries::NamedTuple)
    signatures = batch_output_signature(prepared, positions, queries)
    outputs = allocate_output(signatures, prepared)
    execute_draws!(rng, outputs, work, prepared, positions, queries)
end

"""LogDensityProblems adapter for a prepared factor-graph executor."""
struct FactorLogDensityProblem{P,W}
    prepared::P
    workspace::W
end

function FactorLogDensityProblem(prepared::FactorPrepared, backend)
    work = workspace(prepared, eltype(prepared), backend)
    FactorLogDensityProblem{typeof(prepared),typeof(work)}(prepared, work)
end

Base.eltype(problem::FactorLogDensityProblem) = eltype(problem.prepared)
BRM.LogDensityProblems.capabilities(::Type{<:FactorLogDensityProblem}) =
    BRM.LogDensityProblems.LogDensityOrder{1}()
BRM.LogDensityProblems.dimension(problem::FactorLogDensityProblem) =
    BRM.LogDensityProblems.dimension(problem.prepared)
BRM.LogDensityProblems.logdensity(problem::FactorLogDensityProblem,
                                  position::AbstractVector) =
    logdensity!(problem.workspace, problem.prepared, position)
function BRM.LogDensityProblems.logdensity_and_gradient(
        problem::FactorLogDensityProblem, position::AbstractVector)
    density, gradient = logdensity_and_gradient!(
        problem.workspace, problem.prepared, position)
    density, copy(gradient)
end

BRM.NativePPLLogDensityProblem(prepared::FactorPrepared, backend) =
    FactorLogDensityProblem(prepared, backend)

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

factor_graph(instance::ModelInstance) = factor_graph(
    instance.declaration; conditions=instance.conditions)

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

function _qualified_parameter(
    namespace::Symbol, name::Symbol,
    declaration::GroupedNormalParameter)
    qualify(value::LiteralValue) = value.value
    qualify(value::SiteValue) = qualified_name(
        namespace, site_value_name(value))
    grouped_normal(
        qualified_name(namespace, group_input(declaration)),
        qualify(declaration.location), qualify(declaration.scale))
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
    offsets = Tuple(_mapped_name(
        mapping, name, namespace, "affine offset")
        for name in affine_offsets(declaration))
    affine(inputs, coefficients;
           offsets, intercept=affine_has_intercept(declaration))
end


function _qualified_node(namespace::Symbol, declaration::ExpLink, mapping)
    exp_link(_mapped_name(
        mapping, node_input(declaration), namespace, "exp node"))
end

function _qualified_node(namespace::Symbol, declaration::LogLink, mapping)
    log_link(_mapped_name(
        mapping, node_input(declaration), namespace, "log node"))
end

function _qualified_node(namespace::Symbol, declaration::LogisticLink, mapping)
    logistic_link(_mapped_name(
        mapping, node_input(declaration), namespace, "logistic node"))
end

function _qualified_node(namespace::Symbol, declaration::GroupGather, mapping)
    group_gather(
        _mapped_name(
            mapping, group_values(declaration), namespace, "group gather"),
        _mapped_name(
            mapping, group_input(declaration), namespace, "group gather"))
end


function _qualified_node(namespace::Symbol, declaration::RowProduct, mapping)
    left, right = row_product_inputs(declaration)
    row_product(
        _mapped_name(mapping, left, namespace, "row product"),
        _mapped_name(mapping, right, namespace, "row product"))
end


function _qualified_node(namespace::Symbol, declaration::GroupedAffine, mapping)
    grouped_affine(
        _mapped_name(mapping, grouped_standardized(declaration), namespace,
                     "grouped affine standardized block"),
        _mapped_name(mapping, grouped_scales(declaration), namespace,
                     "grouped affine scales"),
        _mapped_name(mapping, grouped_correlation(declaration), namespace,
                     "grouped affine correlation"),
        _mapped_name(mapping, group_input(declaration), namespace,
                     "grouped affine group input"),
        map(predictor -> predictor === nothing ? nothing :
            _mapped_name(mapping, predictor, namespace,
                         "grouped affine predictor"),
            grouped_predictors(declaration));
        coefficient_indices=grouped_coefficient_indices(declaration),
        coefficient_count=grouped_coefficient_count(declaration))
end


function _qualified_observation(namespace::Symbol, observation, mapping)
    scalar = scalar_observation(observation)
    base = base_observation(scalar)
    response = qualified_name(namespace, observation_response(base))
    declaration = if base isa NormalObservation
        dependencies = observation_dependencies(base)
        normal(
            response,
            _mapped_name(mapping, dependencies[1], namespace, "Normal factor"),
            _mapped_name(mapping, dependencies[2], namespace, "Normal factor"))
    elseif base isa BernoulliLogitObservation
        bernoulli_logit(
            response,
            _mapped_name(
                mapping, only(observation_dependencies(base)),
                namespace, "Bernoulli-logit factor"))
    elseif base isa ZeroInflatedPoissonObservation
        dependencies = observation_dependencies(base)
        zero_inflated_poisson(
            response,
            _mapped_name(mapping, dependencies[1], namespace,
                         "zero-inflated Poisson factor"),
            _mapped_name(mapping, dependencies[2], namespace,
                         "zero-inflated Poisson factor"))
    elseif base isa NegativeBinomial2Observation
        dependencies = observation_dependencies(base)
        negative_binomial2(
            response,
            _mapped_name(mapping, dependencies[1], namespace,
                         "NegativeBinomial2 factor"),
            _mapped_name(mapping, dependencies[2], namespace,
                         "NegativeBinomial2 factor"))
    elseif base isa BetaBinomial2Observation
        dependencies = observation_dependencies(base)
        beta_binomial2(
            response,
            _mapped_name(mapping, dependencies[1], namespace,
                         "BetaBinomial2 factor"),
            _mapped_name(mapping, dependencies[2], namespace,
                         "BetaBinomial2 factor"),
            _mapped_name(mapping, dependencies[3], namespace,
                         "BetaBinomial2 factor"))
    elseif base isa BinomialLogitObservation
        dependencies = observation_dependencies(base)
        binomial_logit(
            response,
            _mapped_name(mapping, dependencies[1], namespace,
                         "BinomialLogit factor"),
            _mapped_name(mapping, dependencies[2], namespace,
                         "BinomialLogit factor"))
    elseif base isa CategoricalLogitObservation
        dependencies = observation_dependencies(base)
        categorical_logit(
            response,
            (_mapped_name(mapping, dependency, namespace,
                          "CategoricalLogit factor")
             for dependency in dependencies)...)
    else
        poisson(
            response,
            _mapped_name(
                mapping, only(observation_dependencies(base)),
                namespace, "Poisson factor"))
    end
    declaration = _qualified_observation_decorators(
        namespace, scalar, declaration, mapping)
    is_broadcast_observation(observation) ? broadcasted(declaration) : declaration
end

_qualified_evidence_bound(bound::Symbol, mapping, namespace) =
    _mapped_name(mapping, bound, namespace, "observation-evidence bound")
_qualified_evidence_bound(bound, mapping, namespace) = bound

function _qualified_observation_decorators(
    namespace::Symbol, observation::AbstractObservationDeclaration,
    declaration::AbstractObservationDeclaration, mapping)
    observation === base_observation(observation) && return declaration
    if observation isa WeightedObservation
        inner = _qualified_observation_decorators(
            namespace, observation.observation, declaration, mapping)
        weight = observation.weight
        source = _mapped_name(
            mapping, observation_weight_source(weight), namespace,
            "observation weight")
        return weighted_observation(
            inner, observation_weight(observation_weight_kind(weight), source))
    end
    inner = _qualified_observation_decorators(
        namespace, observation.observation, declaration, mapping)
    evidence = observation.evidence
    lower = _qualified_evidence_bound(
        evidence_lower(evidence), mapping, namespace)
    upper = _qualified_evidence_bound(
        evidence_upper(evidence), mapping, namespace)
    mapped = if evidence isa TruncatedEvidence
        truncated_evidence(; lower, upper)
    elseif evidence isa CensoredEvidence
        censored_evidence(; lower, upper)
    else
        interval_evidence(upper)
    end
    evidence_observation(inner, mapped)
end

function _observation_with_response(observation, response::Symbol)
    scalar = scalar_observation(observation)
    base = base_observation(scalar)
    dependencies = observation_dependencies(base)
    declaration = if base isa NormalObservation
        normal(response, dependencies...)
    elseif base isa BernoulliLogitObservation
        bernoulli_logit(response, only(dependencies))
    elseif base isa ZeroInflatedPoissonObservation
        zero_inflated_poisson(response, dependencies...)
    elseif base isa NegativeBinomial2Observation
        negative_binomial2(response, dependencies...)
    elseif base isa BetaBinomial2Observation
        beta_binomial2(response, dependencies...)
    elseif base isa BinomialLogitObservation
        binomial_logit(response, dependencies...)
    elseif base isa CategoricalLogitObservation
        categorical_logit(response, dependencies...)
    else
        poisson(response, only(dependencies))
    end
    declaration = _observation_decorators_with_response(scalar, declaration)
    is_broadcast_observation(observation) ? broadcasted(declaration) : declaration
end

function _observation_decorators_with_response(
    observation::AbstractObservationDeclaration,
    declaration::AbstractObservationDeclaration)
    observation === base_observation(observation) && return declaration
    if observation isa WeightedObservation
        return weighted_observation(
            _observation_decorators_with_response(
                observation.observation, declaration),
            observation.weight)
    end
    evidence_observation(
        _observation_decorators_with_response(
            observation.observation, declaration),
        observation.evidence)
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
    site_order_names = Symbol[]
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

        append!(site_order_names, (mapping[name]
                                   for name in declaration.site_order))

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
        resolved_values = Tuple(
            _resolved_reference(reference, resolved)
            for reference in Base.values(public_outputs))
        flattened_values = collect(resolved_values)
        occupied = Set((input_names..., parameter_names..., node_names...,
                        observation_names...))
        for index in eachindex(flattened_values)
            resolved_name = flattened_values[index]
            observation_index = findfirst(==(resolved_name), observation_names)
            observation_index === nothing && continue
            alias = names[index]
            alias === resolved_name || alias ∉ occupied || throw(ArgumentError(
                "native PPL public stochastic-site alias `$alias` collides " *
                "with an existing graph identity"))
            observation_names[observation_index] = alias
            observation_values[observation_index] = _observation_with_response(
                observation_values[observation_index], alias)
            for condition_index in eachindex(condition_names)
                condition_names[condition_index] === resolved_name &&
                    (condition_names[condition_index] = alias)
            end
            for site_index in eachindex(site_order_names)
                site_order_names[site_index] === resolved_name &&
                    (site_order_names[site_index] = alias)
            end
            delete!(occupied, resolved_name)
            push!(occupied, alias)
            flattened_values[index] = alias
        end
        NamedTuple{names}(Tuple(flattened_values))
    end

    declaration = model(
        inputs=NamedTuple{Tuple(input_names)}(Tuple(input_values)),
        parameters=NamedTuple{Tuple(parameter_names)}(Tuple(parameter_values)),
        nodes=NamedTuple{Tuple(node_names)}(Tuple(node_values)),
        observations=NamedTuple{Tuple(observation_names)}(
            Tuple(observation_values)),
        outputs=flattened_outputs,
        site_order=Tuple(site_order_names))
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
    inputs, parameters, nodes, observations, outputs, site_order)
    _check_named_declarations(
        inputs, "input", AbstractInputDeclaration)
    _check_named_declarations(
        parameters, "parameter", AbstractParameterDeclaration)
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

    observation_names = Set(keys(observations))
    available = union(input_names, parameter_names, observation_names)
    for (name, declaration) in pairs(nodes)
        dependencies = declaration isa Affine ?
            (node_inputs(declaration)..., affine_offsets(declaration)...) :
            declaration isa GroupGather ?
                (group_values(declaration), group_input(declaration)) :
            declaration isa RowProduct ? row_product_inputs(declaration) :
            declaration isa GroupedAffine ?
                (grouped_standardized(declaration),
                 grouped_scales(declaration),
                 grouped_correlation(declaration),
                 group_input(declaration),
                 (predictor for predictor in grouped_predictors(declaration)
                  if predictor !== nothing)...) :
                (node_input(declaration),)
        for dependency in dependencies
            dependency in available || throw(ArgumentError(
                "native PPL node `$name` references unavailable value " *
                "`$dependency`; deterministic nodes must be ordered among " *
                "themselves"))
        end
        declaration isa Affine &&
            affine_parameter(declaration) ∉ parameter_names &&
            throw(ArgumentError(
                "native PPL affine node `$name` references unknown parameter " *
                "`$(affine_parameter(declaration))`"))
        if declaration isa GroupGather
            group_values(declaration) in parameter_names || throw(ArgumentError(
                "native PPL group gather node `$name` references unknown " *
                "latent site `$(group_values(declaration))`"))
            group_input(declaration) in input_names || throw(ArgumentError(
                "native PPL group gather node `$name` references unknown " *
                "group input `$(group_input(declaration))`"))
        end
        if declaration isa GroupedAffine
            for parameter in (
                grouped_standardized(declaration),
                grouped_scales(declaration),
                grouped_correlation(declaration))
                parameter in parameter_names || throw(ArgumentError(
                    "native PPL grouped affine node `$name` references " *
                    "unknown parameter `$parameter`"))
            end
            group_input(declaration) in input_names || throw(ArgumentError(
                "native PPL grouped affine node `$name` references unknown " *
                "group input `$(group_input(declaration))`"))
        end
        push!(available, name)
    end

    available = union(input_names, parameter_names, node_names,
                      observation_names)
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

    if outputs !== nothing
        outputs isa NamedTuple || throw(ArgumentError(
            "native PPL model outputs must be a NamedTuple; got " *
            "$(typeof(outputs))"))
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

    site_order isa Tuple || throw(ArgumentError(
        "native PPL stochastic-site order must be a Tuple; got " *
        "$(typeof(site_order))"))
    all(name -> name isa Symbol, site_order) || throw(ArgumentError(
        "native PPL stochastic-site order must contain Symbols"))
    length(unique(site_order)) == length(site_order) || throw(ArgumentError(
        "native PPL stochastic-site order must contain unique identities"))
    expected_sites = union(parameter_names, Set(keys(observations)))
    Set(site_order) == expected_sites || throw(ArgumentError(
        "native PPL stochastic-site order must name each parameter/site " *
        "exactly once; expected $(sort!(collect(expected_sites))), got " *
        "$(collect(site_order))"))

    nothing
end

function model(;
    inputs, parameters=(;), nodes=(;), observations, outputs=nothing,
    site_order=_default_site_order(parameters, observations))
    _validate_model_components(
        inputs, parameters, nodes, observations, outputs, site_order)
    Model(inputs, parameters, nodes, observations, outputs, site_order)
end

function _validate_model(declaration::Model)
    _validate_model_components(
        declaration.inputs, declaration.parameters,
        declaration.nodes, declaration.observations, declaration.outputs,
        declaration.site_order)
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
    print(io, ", site_order=", declaration.site_order)
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
    _uses_factor_executor(declaration, conditions, bindings) &&
        return _bind_factor_plan(declaration, bindings, conditions)
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

function _lower_brmi_scalar_factor_dag(brmi::BRM.BRMI, response::Symbol,
                                       response_lhs, location::Symbol)
    haskey(brmi.operations, location) || return nothing
    _, location_rhs = BRM._native_ppl_sampling_rhs(brmi, location)
    location_rhs isa BRM.ExprColumn &&
        BRM.getf(location_rhs) === BRM.Normal || return nothing

    parameter_names = Symbol[]
    parameter_values = Any[]
    deferred_standard_names = Symbol[]
    deferred_standard_values = Any[]
    observation_names = Symbol[]
    observation_values = Any[]
    site_order = Symbol[]

    for name in keys(brmi.operations)
        name isa Symbol || continue
        lhs, rhs = BRM._native_ppl_sampling_rhs(brmi, name)
        rhs isa BRM.ExprColumn || throw(CapabilityError(
            :factor_dag,
            "scalar factor-DAG operation `$name` must have a distribution RHS"))
        isempty(BRM.getkwargs(rhs)) || throw(CapabilityError(
            :factor_dag,
            "scalar factor-DAG operation `$name` cannot have distribution keywords"))
        family = BRM.getf(rhs)
        arguments = BRM.getargs(rhs)
        push!(site_order, name)

        if name === response
            family === BRM.Normal && length(arguments) == 2 || throw(
                CapabilityError(
                    :factor_dag,
                    "scalar factor-DAG response `$response` must use Normal(location, scale)"))
            location_name = BRM._native_ppl_ref_name(arguments[1])
            scale_name = BRM._native_ppl_ref_name(arguments[2])
            (location_name === nothing || scale_name === nothing) && throw(
                CapabilityError(
                    :factor_dag,
                    "scalar factor-DAG response arguments must be named sites"))
            push!(observation_names, name)
            push!(observation_values, broadcasted(
                normal(name, location_name, scale_name)))
            continue
        end

        if family === BRM.Exponential
            length(arguments) == 1 && only(arguments) isa Real || throw(
                CapabilityError(
                    :factor_dag,
                    "Exponential site `$name` requires one literal scale"))
            push!(parameter_names, name)
            push!(parameter_values, parameter(
                PositiveSupport(), (name,); transform=Exp(),
                prior=Exponential(only(arguments))))
        elseif family === BRM.Normal && isempty(arguments)
            push!(deferred_standard_names, name)
            push!(deferred_standard_values, parameter(
                RealSupport(), (name,); transform=Identity(),
                prior=StandardNormal()))
        elseif family === BRM.Normal && length(arguments) == 2 &&
               all(argument -> argument isa Real, arguments)
            push!(parameter_names, name)
            push!(parameter_values, parameter(
                RealSupport(), (name,); transform=Identity(),
                prior=normal_prior(arguments...)))
        elseif family === BRM.Normal && length(arguments) == 2
            location_name = BRM._native_ppl_ref_name(arguments[1])
            scale_name = BRM._native_ppl_ref_name(arguments[2])
            (location_name === nothing || scale_name === nothing) && throw(
                CapabilityError(
                    :factor_dag,
                    "Normal site `$name` arguments must be literals or named sites"))
            push!(observation_names, name)
            push!(observation_values,
                  normal(name, location_name, scale_name))
        else
            throw(CapabilityError(
                :factor_dag,
                "unsupported scalar factor `$family` for site `$name`"))
        end
    end

    append!(parameter_names, deferred_standard_names)
    append!(parameter_values, deferred_standard_values)
    declaration = model(
        inputs=(;),
        parameters=_declaration_namedtuple(
            Tuple(parameter_names), Tuple(parameter_values)),
        nodes=(;),
        observations=_declaration_namedtuple(
            Tuple(observation_names), Tuple(observation_values)),
        outputs=NamedTuple{(response,)}((response,)),
        site_order=Tuple(site_order))
    response_values = parent(parent(response_lhs))
    conditions = NamedTuple{(response,)}((response_values,))
    (; declaration, bindings=(;), conditions)
end

function _brmi_literal_normal_prior(brmi::BRM.BRMI, name::Symbol)
    lhs, prior = BRM._native_ppl_sampling_rhs(brmi, name)
    BRM._native_ppl_ref_name(lhs) === name || throw(CapabilityError(
        :parameter_prior,
        "sampled offset `$name` must have a bare left-hand side"))
    prior isa BRM.ExprColumn && BRM.getf(prior) === BRM.Normal || throw(
        CapabilityError(
            :parameter_prior,
            "sampled offset `$name` must use Normal(location, scale)"))
    isempty(BRM.getkwargs(prior)) || throw(CapabilityError(
        :parameter_prior,
        "Normal prior for sampled offset `$name` cannot have keywords"))
    arguments = BRM.getargs(prior)
    length(arguments) == 2 && all(argument -> argument isa Real, arguments) ||
        throw(CapabilityError(
            :parameter_prior,
            "sampled offset `$name` requires literal Normal(location, scale)"))
    arguments[1] == 0 && arguments[2] == 1 ?
        StandardNormal() : normal_prior(arguments...)
end

function _brmi_group_sd_prior(brmi::BRM.BRMI, id::Symbol)
    matches = NamedTuple[]
    for (operation_name, named) in pairs(brmi.operations)
        operation = parent(named)
        operation isa BRM.ExprColumn && BRM.getf(operation) === (~) || continue
        lhs, prior = BRM.getargs(operation, 2)
        lhs isa BRM.ExprColumn && BRM.getf(lhs) === BRM.effect || continue
        address = BRM.getargs(lhs)
        length(address) >= 2 && address[1] === :sd && address[2] === id ||
            continue
        prior isa BRM.ExprColumn && BRM.getf(prior) === BRM.Exponential ||
            throw(CapabilityError(
                :group_prior,
                "varying-intercept SD for `|$id|` must use Exponential(scale)"))
        isempty(BRM.getkwargs(prior)) || throw(CapabilityError(
            :group_prior,
            "varying-intercept SD prior for `|$id|` cannot have keywords"))
        arguments = BRM.getargs(prior)
        length(arguments) == 1 && only(arguments) isa Real || throw(
            CapabilityError(
                :group_prior,
                "varying-intercept SD prior for `|$id|` requires one " *
                "literal scale"))
        scale = only(arguments)
        isfinite(scale) && scale > zero(scale) || throw(CapabilityError(
            :group_prior,
            "varying-intercept SD prior for `|$id|` must be positive"))
        push!(matches, (; operation=operation_name, scale))
    end
    length(matches) == 1 || throw(CapabilityError(
        :group_prior,
        "varying-intercept `|$id|` requires exactly one block-wide " *
        "`sd(:, $id) ~ Exponential(scale)` prior"))
    only(matches)
end

function _brmi_group_correlation_prior(brmi::BRM.BRMI, id::Symbol,
                                       dimension::Int)
    matches = NamedTuple[]
    for (operation_name, named) in pairs(brmi.operations)
        operation = parent(named)
        operation isa BRM.ExprColumn && BRM.getf(operation) === (~) || continue
        lhs, prior = BRM.getargs(operation, 2)
        lhs isa BRM.ExprColumn && BRM.getf(lhs) === BRM.effect || continue
        address = BRM.getargs(lhs)
        length(address) >= 2 && address[1] === :cor && address[2] === id ||
            continue
        prior isa BRM.ExprColumn && BRM.getf(prior) === BRM.LKJCholesky ||
            throw(CapabilityError(
                :group_prior,
                "correlation for `|$id|` must use LKJCholesky(K, eta)"))
        isempty(BRM.getkwargs(prior)) || throw(CapabilityError(
            :group_prior,
            "LKJ correlation prior for `|$id|` cannot have keywords"))
        arguments = BRM.getargs(prior)
        length(arguments) == 2 || throw(CapabilityError(
            :group_prior,
            "LKJCholesky prior for `|$id|` requires dimension and eta"))
        prior_dimension, eta = arguments
        prior_dimension isa Integer && prior_dimension == dimension || throw(
            CapabilityError(
                :group_prior,
                "LKJCholesky dimension for `|$id|` must equal $dimension"))
        eta isa Real && isfinite(eta) && eta > zero(eta) || throw(
            CapabilityError(
                :group_prior,
                "LKJ shape for `|$id|` must be finite and positive"))
        push!(matches, (; operation=operation_name, eta))
    end
    length(matches) == 1 || throw(CapabilityError(
        :group_prior,
        "correlated `|$id|` block requires exactly one " *
        "`cor(:, $id) ~ LKJCholesky($dimension, eta)` prior"))
    only(matches)
end

function _lower_brmi_distributional_gaussian(
    brmi::BRM.BRMI, response::Symbol, response_lhs,
    location::Symbol, scale_expression)
    scale_expression isa BRM.ExprColumn &&
        BRM.getf(scale_expression) === exp || return nothing
    isempty(BRM.getkwargs(scale_expression)) || throw(CapabilityError(
        :likelihood_link,
        "Normal scale `exp` link cannot have keywords"))
    scale_arguments = BRM.getargs(scale_expression)
    length(scale_arguments) == 1 || throw(CapabilityError(
        :likelihood_link,
        "Normal scale `exp` link needs one named log-scale predictor"))
    log_scale = BRM._native_ppl_ref_name(only(scale_arguments))
    log_scale === nothing && throw(CapabilityError(
        :likelihood_scale,
        "Normal `exp` scale must consume one named log-scale predictor"))
    location == log_scale && throw(CapabilityError(
        :graph_identity,
        "Normal location and log-scale predictors must be distinct"))

    specs = map((location, log_scale)) do predictor
        components = BRM._native_ppl_affine_components(brmi, predictor)
        isempty(components.offsets) || throw(CapabilityError(
            :distributional_predictor,
            "distributional predictor `$predictor` does not yet support " *
            "sampled offsets"))
        isempty(components.data_offsets) || throw(CapabilityError(
            :distributional_predictor,
            "distributional predictor `$predictor` does not yet support " *
            "data offsets"))
        terms = components.predictors
        isempty(terms) && !components.intercept && throw(CapabilityError(
            :distributional_predictor,
            "distributional predictor `$predictor` needs an intercept or " *
            "population predictor"))
        raw_names = Tuple(BRM.name(term.column) for term in terms)
        length(unique(raw_names)) == length(raw_names) || throw(
            CapabilityError(
                :distributional_predictor,
                "distributional predictor `$predictor` repeats a raw input"))
        coefficient_name = Symbol(:beta_, predictor)
        coefficient_keys = components.intercept ?
            (:Intercept, raw_names...) : raw_names
        transform_node_names = Symbol[]
        transform_node_declarations = Any[]
        affine_inputs = Symbol[]
        for (name, term) in zip(raw_names, terms)
            if term.transform === :identity
                push!(affine_inputs, name)
            else
                transform_name = Symbol(
                    term.transform, :_, name, :_for_, predictor)
                transform_declaration = term.transform === :center ?
                    center(name) : zscale(name)
                push!(transform_node_names, transform_name)
                push!(transform_node_declarations, transform_declaration)
                push!(affine_inputs, transform_name)
            end
        end
        group_specs = map(components.groups) do varying
            group_name = BRM.name(varying.group)
            group_predictor_names = Tuple(
                term === nothing ? nothing : begin
                    term.transform === :identity || throw(CapabilityError(
                        :group_term,
                        "distributional grouped predictors currently " *
                        "require raw, untransformed inputs"))
                    BRM.name(term.column)
                end
                for term in varying.predictors)
            local_keys = Tuple(name === nothing ? :Intercept : name
                               for name in group_predictor_names)
            correlated = length(local_keys) > 1
            prior = _brmi_group_sd_prior(brmi, varying.id)
            scale_name = Symbol(:tau_, varying.id, :_, group_name)
            correlation_name = Symbol(:L_, varying.id, :_, group_name)
            site_name = Symbol(:b_, varying.id, :_, group_name)
            gather_name = correlated ? Symbol(
                site_name, :_by_, group_name, :_for_, predictor) :
                Symbol(:r_, predictor, :_, varying.id, :_, group_name)
            predictor_value = length(group_predictor_names) == 1 ?
                only(group_predictor_names) : nothing
            product_name = !correlated && predictor_value !== nothing ?
                Symbol(gather_name, :_times_, predictor_value) : nothing
            (; id=varying.id, group_name, column=varying.group, prior,
               predictor_terms=varying.predictors,
               predictor_names=group_predictor_names,
               coefficient_keys=local_keys,
               coefficient_indices=ntuple(identity, length(local_keys)),
               coefficient_count=length(local_keys), correlated,
               correlation_prior=nothing, scale_name, correlation_name,
               site_name, gather_name, predictor_value, product_name,
               offset_name=product_name === nothing ?
                   gather_name : product_name)
        end
        group_names = Tuple(spec.group_name for spec in group_specs)
        length(unique(group_names)) == length(group_names) || throw(
            CapabilityError(
                :group_term,
                "distributional predictor `$predictor` repeats a grouping " *
                "input"))
        (; predictor, components, raw_names,
           raw_columns=Tuple(term.column for term in terms),
           coefficient_name, coefficient_keys,
           affine_inputs=Tuple(affine_inputs),
           transform_node_names=Tuple(transform_node_names),
           transform_node_declarations=Tuple(transform_node_declarations),
           group_specs=Tuple(group_specs))
    end

    group_buckets = Dict{Tuple{Symbol,Symbol},Vector{Tuple{Int,Int}}}()
    for (spec_index, spec) in pairs(specs)
        for (group_index, group_spec) in pairs(spec.group_specs)
            push!(get!(group_buckets,
                       (group_spec.id, group_spec.group_name),
                       Tuple{Int,Int}[]),
                  (spec_index, group_index))
        end
    end
    group_overrides = Dict{Tuple{Int,Int},Any}()
    for entries in values(group_buckets)
        first_entry = first(entries)
        first_spec_index, first_group_index = first_entry
        first_group = specs[first_spec_index].group_specs[first_group_index]
        if length(entries) == 1
            first_group.correlated || continue
            group_overrides[first_entry] = (;
                correlation_prior=_brmi_group_correlation_prior(
                    brmi, first_group.id,
                    length(first_group.coefficient_keys)))
            continue
        end
        coefficient_keys_buffer = Symbol[]
        entry_indices = Dict{Tuple{Int,Int},Tuple}()
        for entry in entries
            spec_index, group_index = entry
            spec = specs[spec_index]
            group_spec = spec.group_specs[group_index]
            group_spec.column == first_group.column || throw(CapabilityError(
                :group_term,
                "shared grouped id `$(group_spec.id)` must use one grouping " *
                "column across distributional predictors"))
            first_index = length(coefficient_keys_buffer) + 1
            append!(coefficient_keys_buffer, map(
                group_spec.coefficient_keys) do key
                    key === :Intercept ? spec.predictor :
                        Symbol(spec.predictor, :_, key)
                end)
            entry_indices[entry] = Tuple(
                first_index:length(coefficient_keys_buffer))
        end
        coefficient_keys = Tuple(coefficient_keys_buffer)
        length(unique(coefficient_keys)) == length(coefficient_keys) || throw(
            CapabilityError(
                :group_term,
                "shared grouped id `$(first_group.id)` produces duplicate " *
                "distributional coefficient identities"))
        coefficient_count = length(coefficient_keys)
        correlation_prior = _brmi_group_correlation_prior(
            brmi, first_group.id, coefficient_count)
        for entry in entries
            spec_index, group_index = entry
            spec = specs[spec_index]
            group_spec = spec.group_specs[group_index]
            gather_name = Symbol(
                group_spec.site_name, :_by_, group_spec.group_name,
                :_for_, spec.predictor)
            group_overrides[entry] = (;
                coefficient_keys,
                coefficient_indices=entry_indices[entry],
                coefficient_count, correlated=true, correlation_prior,
                gather_name, predictor_value=nothing, product_name=nothing,
                offset_name=gather_name)
        end
    end
    specs = ntuple(length(specs)) do spec_index
        spec = specs[spec_index]
        group_specs = ntuple(length(spec.group_specs)) do group_index
            group_spec = spec.group_specs[group_index]
            override = get(group_overrides, (spec_index, group_index), nothing)
            override === nothing ? group_spec : merge(group_spec, override)
        end
        merge(spec, (; group_specs))
    end

    predictor_names = Symbol[]
    predictor_columns = Any[]
    function add_input!(name, column)
        index = findfirst(==(name), predictor_names)
        if index === nothing
            push!(predictor_names, name)
            push!(predictor_columns, column)
        elseif predictor_columns[index] != column
            throw(CapabilityError(
                :graph_identity,
                "distributional input `$name` has conflicting sources"))
        end
        nothing
    end
    for spec in specs
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            add_input!(name, column)
        end
    end
    for spec in specs
        for group_spec in spec.group_specs
            for (name, term) in zip(
                    group_spec.predictor_names,
                    group_spec.predictor_terms)
                name === nothing || add_input!(name, term.column)
            end
            add_input!(group_spec.group_name, group_spec.column)
        end
    end

    parameter_names = Symbol[spec.coefficient_name for spec in specs]
    parameter_declarations = Any[parameter(
        RealSupport(), spec.coefficient_keys;
        transform=Identity(), prior=StandardNormal()) for spec in specs]
    other_parameter_names = Symbol[]
    other_parameter_declarations = Any[]
    function add_other_parameter!(name, declaration)
        index = findfirst(==(name), other_parameter_names)
        if index === nothing
            push!(other_parameter_names, name)
            push!(other_parameter_declarations, declaration)
        elseif other_parameter_declarations[index] != declaration
            throw(CapabilityError(
                :graph_identity,
                "distributional parameter `$name` has conflicting " *
                "declarations"))
        end
        nothing
    end

    node_names = Symbol[]
    node_declarations = Any[]
    for spec in specs
        append!(node_names, spec.transform_node_names)
        append!(node_declarations, spec.transform_node_declarations)
        for group_spec in spec.group_specs
            add_other_parameter!(group_spec.scale_name, parameter(
                PositiveSupport(), group_spec.correlated ?
                    group_spec.coefficient_keys : (group_spec.scale_name,);
                transform=Exp(), prior=Exponential(group_spec.prior.scale)))
            if group_spec.correlated
                add_other_parameter!(
                    group_spec.correlation_name,
                    cholesky_correlation(
                        group_spec.coefficient_keys,
                        group_spec.correlation_prior.eta))
            end
            add_other_parameter!(
                group_spec.site_name,
                group_spec.correlated ? grouped_standard_normal(
                    group_spec.group_name, group_spec.coefficient_keys) :
                    grouped_normal(
                        group_spec.group_name, 0.0, group_spec.scale_name))
            push!(node_names, group_spec.gather_name)
            push!(node_declarations,
                  group_spec.correlated ? grouped_affine(
                      group_spec.site_name, group_spec.scale_name,
                      group_spec.correlation_name, group_spec.group_name,
                      group_spec.predictor_names;
                      coefficient_indices=group_spec.coefficient_indices,
                      coefficient_count=group_spec.coefficient_count) :
                  group_gather(group_spec.site_name, group_spec.group_name))
            if group_spec.product_name !== nothing
                push!(node_names, group_spec.product_name)
                push!(node_declarations, row_product(
                    group_spec.gather_name, group_spec.predictor_value))
            end
        end
        push!(node_names, spec.predictor)
        push!(node_declarations, affine(
            spec.affine_inputs, spec.coefficient_name;
            offsets=Tuple(group_spec.offset_name
                          for group_spec in spec.group_specs),
            intercept=spec.components.intercept))
    end
    append!(parameter_names, other_parameter_names)
    append!(parameter_declarations, other_parameter_declarations)

    log_scale_spelling = String(log_scale)
    scale_name = startswith(log_scale_spelling, "log_") ?
        Symbol(log_scale_spelling[5:end]) : Symbol(:exp_, log_scale)
    scale_name in (predictor_names..., parameter_names..., node_names...,
                   response) && throw(CapabilityError(
        :graph_identity,
        "derived Normal scale `$scale_name` collides with another graph value"))
    push!(node_names, scale_name)
    push!(node_declarations, exp_link(log_scale))

    expected_values = Set((location, log_scale, response))
    union!(expected_values, predictor_names)
    for spec in specs
        for group_spec in spec.group_specs
            push!(expected_values, group_spec.prior.operation)
            group_spec.correlation_prior === nothing ||
                push!(expected_values,
                      group_spec.correlation_prior.operation)
        end
    end
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported distributional formula operations: " *
        join(sort!(collect(extras)), ", ")))

    input_declarations = _declaration_namedtuple(
        Tuple(predictor_names), Tuple(input() for _ in predictor_names))
    parameters = _declaration_namedtuple(
        Tuple(parameter_names), Tuple(parameter_declarations))
    nodes = _declaration_namedtuple(
        Tuple(node_names), Tuple(node_declarations))
    site_order_buffer = Symbol[]
    for spec in specs
        for group_spec in spec.group_specs
            group_spec.scale_name in site_order_buffer ||
                push!(site_order_buffer, group_spec.scale_name)
            if group_spec.correlated
                group_spec.correlation_name in site_order_buffer ||
                    push!(site_order_buffer, group_spec.correlation_name)
            end
            group_spec.site_name in site_order_buffer ||
                push!(site_order_buffer, group_spec.site_name)
        end
    end
    append!(site_order_buffer,
            Tuple(spec.coefficient_name for spec in specs))
    push!(site_order_buffer, response)
    declaration = model(
        inputs=input_declarations,
        parameters=parameters,
        nodes=nodes,
        observations=NamedTuple{(response,)}((broadcasted(
            normal(response, location, scale_name)),)),
        site_order=Tuple(site_order_buffer))
    bindings = _declaration_namedtuple(
        Tuple(predictor_names),
        Tuple(parent(parent(column)) for column in predictor_columns))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_linked_affine(
        brmi::BRM.BRMI, key::Symbol, link, graph_name::Symbol)
    components = BRM._native_ppl_affine_components(brmi, key; link)
    isempty(components.offsets) || throw(CapabilityError(
        :linked_predictor,
        "linked distributional predictor `$key` does not yet support " *
        "sampled offsets"))
    isempty(components.data_offsets) || throw(CapabilityError(
        :linked_predictor,
        "linked distributional predictor `$key` does not yet support data offsets"))
    isempty(components.groups) || throw(CapabilityError(
        :linked_predictor,
        "linked distributional predictor `$key` does not yet support grouped terms"))
    terms = components.predictors
    isempty(terms) && !components.intercept && throw(CapabilityError(
        :linked_predictor,
        "linked distributional predictor `$key` needs an intercept or predictor"))
    raw_names = Tuple(BRM.name(term.column) for term in terms)
    coefficient_name = Symbol(:beta_, graph_name)
    coefficient_keys = components.intercept ?
        (:Intercept, raw_names...) : raw_names
    parameter_declaration = parameter(
        RealSupport(), coefficient_keys;
        transform=Identity(), prior=StandardNormal())
    node_names = Symbol[]
    node_declarations = Any[]
    affine_inputs = Symbol[]
    for (name, term) in zip(raw_names, terms)
        if term.transform === :identity
            push!(affine_inputs, name)
        else
            transform_name = Symbol(
                term.transform, :_, name, :_for_, graph_name)
            push!(node_names, transform_name)
            push!(node_declarations,
                  term.transform === :center ? center(name) : zscale(name))
            push!(affine_inputs, transform_name)
        end
    end
    push!(node_names, graph_name)
    push!(node_declarations, affine(
        Tuple(affine_inputs), coefficient_name;
        intercept=components.intercept))
    (; key, graph_name, coefficient_name, parameter_declaration,
       raw_names, raw_columns=Tuple(term.column for term in terms),
       node_names=Tuple(node_names),
       node_declarations=Tuple(node_declarations))
end

function _lower_brmi_negative_binomial2(
        brmi::BRM.BRMI, response::Symbol, response_lhs, likelihood)
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "NegativeBinomial2 likelihood for `$response` has keywords"))
    arguments = BRM.getargs(likelihood)
    length(arguments) == 2 || throw(CapabilityError(
        :likelihood,
        "NegativeBinomial2 likelihood for `$response` needs mean and " *
        "precision arguments"))
    mean_key = BRM._native_ppl_ref_name(arguments[1])
    precision_key = BRM._native_ppl_ref_name(arguments[2])
    mean_key === nothing && throw(CapabilityError(
        :likelihood_mean,
        "NegativeBinomial2 mean must name one `log(mean) ~ ...` predictor"))
    precision_key === nothing && throw(CapabilityError(
        :likelihood_precision,
        "NegativeBinomial2 precision must name one `log(phi) ~ ...` predictor"))
    mean_key == precision_key && throw(CapabilityError(
        :graph_identity,
        "NegativeBinomial2 mean and precision predictors must be distinct"))

    log_mean = Symbol(:log_, mean_key)
    log_precision = Symbol(:log_, precision_key)
    mean_spec = _lower_brmi_linked_affine(brmi, mean_key, log, log_mean)
    precision_spec = _lower_brmi_linked_affine(
        brmi, precision_key, log, log_precision)
    predictor_names = Symbol[]
    predictor_columns = Any[]
    for spec in (mean_spec, precision_spec)
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            if name ∉ predictor_names
                push!(predictor_names, name)
                push!(predictor_columns, column)
            end
        end
    end
    expected_values = Set((
        response, mean_key, precision_key, predictor_names...))
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported NegativeBinomial2 formula operations: " *
        join(sort!(collect(extras)), ", ")))

    mean_name = Symbol(:exp_, log_mean)
    precision_name = Symbol(:exp_, log_precision)
    node_names = (
        mean_spec.node_names..., precision_spec.node_names...,
        mean_name, precision_name)
    length(unique(node_names)) == length(node_names) || throw(CapabilityError(
        :graph_identity,
        "NegativeBinomial2 predictor nodes have colliding identities"))
    node_declarations = (
        mean_spec.node_declarations..., precision_spec.node_declarations...,
        exp_link(log_mean), exp_link(log_precision))
    input_names = Tuple(predictor_names)
    declaration = model(
        inputs=_declaration_namedtuple(
            input_names, Tuple(input() for _ in input_names)),
        parameters=_declaration_namedtuple(
            (mean_spec.coefficient_name, precision_spec.coefficient_name),
            (mean_spec.parameter_declaration,
             precision_spec.parameter_declaration)),
        nodes=_declaration_namedtuple(node_names, node_declarations),
        observations=NamedTuple{(response,)}((broadcasted(
            negative_binomial2(
                response, mean_name, precision_name)),)))
    bindings = _declaration_namedtuple(
        input_names,
        Tuple(parent(parent(column)) for column in predictor_columns))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_beta_binomial2(
        brmi::BRM.BRMI, response::Symbol, response_lhs, likelihood)
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "BetaBinomial2 likelihood for `$response` has keywords"))
    arguments = BRM.getargs(likelihood)
    length(arguments) == 3 || throw(CapabilityError(
        :likelihood,
        "BetaBinomial2 likelihood for `$response` needs trials, mean, and " *
        "precision arguments"))
    trials_argument = arguments[1]
    trials_key = trials_argument isa BRM.NamedColumn &&
                 parent(trials_argument) isa BRM.DataColumn ?
        BRM.name(trials_argument) : BRM._native_ppl_ref_name(trials_argument)
    mean_key = BRM._native_ppl_ref_name(arguments[2])
    precision_key = BRM._native_ppl_ref_name(arguments[3])
    trials_key === nothing && throw(CapabilityError(
        :likelihood_trials,
        "BetaBinomial2 trials must name one data input"))
    mean_key === nothing && throw(CapabilityError(
        :likelihood_mean,
        "BetaBinomial2 mean must name one `logit(mean) ~ ...` predictor"))
    precision_key === nothing && throw(CapabilityError(
        :likelihood_precision,
        "BetaBinomial2 precision must name one `log(phi) ~ ...` predictor"))
    length(unique((trials_key, mean_key, precision_key))) == 3 || throw(
        CapabilityError(
            :graph_identity,
            "BetaBinomial2 trials, mean, and precision must be distinct"))

    logit_mean = Symbol(:logit_, mean_key)
    log_precision = Symbol(:log_, precision_key)
    mean_spec = _lower_brmi_linked_affine(
        brmi, mean_key, BRM.logit, logit_mean)
    precision_spec = _lower_brmi_linked_affine(
        brmi, precision_key, log, log_precision)
    predictor_names = Symbol[]
    predictor_columns = Any[]
    for spec in (mean_spec, precision_spec)
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            if name ∉ predictor_names
                push!(predictor_names, name)
                push!(predictor_columns, column)
            end
        end
    end
    trials_key in predictor_names && throw(CapabilityError(
        :graph_identity,
        "BetaBinomial2 trials input `$trials_key` also appears as a predictor"))
    expected_values = Set((
        response, trials_key, mean_key, precision_key, predictor_names...))
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported BetaBinomial2 formula operations: " *
        join(sort!(collect(extras)), ", ")))

    probability_name = Symbol(:logistic_, logit_mean)
    precision_name = Symbol(:exp_, log_precision)
    node_names = (
        mean_spec.node_names..., precision_spec.node_names...,
        probability_name, precision_name)
    length(unique(node_names)) == length(node_names) || throw(CapabilityError(
        :graph_identity,
        "BetaBinomial2 predictor nodes have colliding identities"))
    node_declarations = (
        mean_spec.node_declarations..., precision_spec.node_declarations...,
        logistic_link(logit_mean), exp_link(log_precision))
    input_names = (predictor_names..., trials_key)
    trials_column = getproperty(brmi.operations, trials_key)
    declaration = model(
        inputs=_declaration_namedtuple(
            input_names, Tuple(input() for _ in input_names)),
        parameters=_declaration_namedtuple(
            (mean_spec.coefficient_name, precision_spec.coefficient_name),
            (mean_spec.parameter_declaration,
             precision_spec.parameter_declaration)),
        nodes=_declaration_namedtuple(node_names, node_declarations),
        observations=NamedTuple{(response,)}((broadcasted(
            beta_binomial2(
                response, trials_key, probability_name, precision_name)),)))
    bindings = _declaration_namedtuple(
        input_names,
        (Tuple(parent(parent(column)) for column in predictor_columns)...,
         parent(parent(trials_column))))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_binomial_logit(
        brmi::BRM.BRMI, response::Symbol, response_lhs, likelihood)
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "BinomialLogit likelihood for `$response` has keywords"))
    arguments = BRM.getargs(likelihood)
    length(arguments) == 2 || throw(CapabilityError(
        :likelihood,
        "BinomialLogit likelihood for `$response` needs trials and logit " *
        "arguments"))
    trials_argument = arguments[1]
    trials_key = trials_argument isa BRM.NamedColumn &&
                 parent(trials_argument) isa BRM.DataColumn ?
        BRM.name(trials_argument) : BRM._native_ppl_ref_name(trials_argument)
    logit_key = BRM._native_ppl_ref_name(arguments[2])
    trials_key === nothing && throw(CapabilityError(
        :likelihood_trials,
        "BinomialLogit trials must name one data input"))
    logit_key === nothing && throw(CapabilityError(
        :likelihood_logit,
        "BinomialLogit logit must name one affine predictor"))
    trials_key == logit_key && throw(CapabilityError(
        :graph_identity,
        "BinomialLogit trials and logit predictor must be distinct"))

    logit_spec = _lower_brmi_linked_affine(
        brmi, logit_key, nothing, logit_key)
    predictor_names = collect(logit_spec.raw_names)
    trials_key in predictor_names && throw(CapabilityError(
        :graph_identity,
        "BinomialLogit trials input `$trials_key` also appears as a predictor"))
    expected_values = Set((
        response, trials_key, logit_key, predictor_names...))
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported BinomialLogit formula operations: " *
        join(sort!(collect(extras)), ", ")))

    input_names = (Tuple(predictor_names)..., trials_key)
    trials_column = getproperty(brmi.operations, trials_key)
    declaration = model(
        inputs=_declaration_namedtuple(
            input_names, Tuple(input() for _ in input_names)),
        parameters=NamedTuple{(logit_spec.coefficient_name,)}(
            (logit_spec.parameter_declaration,)),
        nodes=_declaration_namedtuple(
            logit_spec.node_names, logit_spec.node_declarations),
        observations=NamedTuple{(response,)}((broadcasted(
            binomial_logit(response, trials_key, logit_key)),)))
    bindings = _declaration_namedtuple(
        input_names,
        (Tuple(parent(parent(column)) for column in logit_spec.raw_columns)...,
         parent(parent(trials_column))))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_categorical_logit(
        brmi::BRM.BRMI, response::Symbol, response_lhs, likelihood)
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "CategoricalLogit likelihood for `$response` has keywords"))
    arguments = BRM.getargs(likelihood)
    isempty(arguments) && throw(CapabilityError(
        :likelihood,
        "CategoricalLogit likelihood for `$response` needs at least one " *
        "non-reference predictor"))
    logit_keys = Tuple(BRM._native_ppl_ref_name(argument)
                       for argument in arguments)
    any(isnothing, logit_keys) && throw(CapabilityError(
        :likelihood_logits,
        "CategoricalLogit arguments must name affine predictors"))
    length(unique(logit_keys)) == length(logit_keys) || throw(CapabilityError(
        :graph_identity,
        "CategoricalLogit predictors must be distinct"))

    specs = Tuple(_lower_brmi_linked_affine(
        brmi, key, nothing, key) for key in logit_keys)
    predictor_names = Symbol[]
    predictor_columns = Any[]
    for spec in specs
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            if name ∉ predictor_names
                push!(predictor_names, name)
                push!(predictor_columns, column)
            end
        end
    end
    expected_values = Set((response, logit_keys..., predictor_names...))
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported CategoricalLogit formula operations: " *
        join(sort!(collect(extras)), ", ")))

    node_names = Tuple(name for spec in specs for name in spec.node_names)
    length(unique(node_names)) == length(node_names) || throw(CapabilityError(
        :graph_identity,
        "CategoricalLogit predictor nodes have colliding identities"))
    node_declarations = Tuple(
        node for spec in specs for node in spec.node_declarations)
    parameter_names = Tuple(spec.coefficient_name for spec in specs)
    parameter_declarations = Tuple(spec.parameter_declaration for spec in specs)
    input_names = Tuple(predictor_names)
    declaration = model(
        inputs=_declaration_namedtuple(
            input_names, Tuple(input() for _ in input_names)),
        parameters=_declaration_namedtuple(
            parameter_names, parameter_declarations),
        nodes=_declaration_namedtuple(node_names, node_declarations),
        observations=NamedTuple{(response,)}((broadcasted(
            categorical_logit(response, logit_keys...)),)))
    bindings = _declaration_namedtuple(
        input_names,
        Tuple(parent(parent(column)) for column in predictor_columns))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_zero_inflated_poisson(
        brmi::BRM.BRMI, response::Symbol, response_lhs, likelihood)
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "ZeroInflatedPoisson likelihood for `$response` has keywords"))
    arguments = BRM.getargs(likelihood)
    length(arguments) == 2 || throw(CapabilityError(
        :likelihood,
        "ZeroInflatedPoisson likelihood for `$response` needs rate and " *
        "zero-inflation arguments"))
    rate_key = BRM._native_ppl_ref_name(arguments[1])
    probability_key = BRM._native_ppl_ref_name(arguments[2])
    rate_key === nothing && throw(CapabilityError(
        :likelihood_rate,
        "ZeroInflatedPoisson rate must name one `log(rate) ~ ...` predictor"))
    probability_key === nothing && throw(CapabilityError(
        :likelihood_probability,
        "ZeroInflatedPoisson probability must name one `logit(p) ~ ...` predictor"))
    rate_key == probability_key && throw(CapabilityError(
        :graph_identity,
        "zero-inflated rate and probability predictors must be distinct"))

    log_rate = Symbol(:log_, rate_key)
    probability_logit = Symbol(:logit_, probability_key)
    rate = _lower_brmi_linked_affine(brmi, rate_key, log, log_rate)
    probability = _lower_brmi_linked_affine(
        brmi, probability_key, BRM.logit, probability_logit)
    predictor_names = Symbol[]
    predictor_columns = Any[]
    for spec in (rate, probability)
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            if name ∉ predictor_names
                push!(predictor_names, name)
                push!(predictor_columns, column)
            end
        end
    end
    expected_values = Set((
        response, rate_key, probability_key, predictor_names...))
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported zero-inflated formula operations: " *
        join(sort!(collect(extras)), ", ")))

    rate_name = Symbol(:exp_, log_rate)
    probability_name = Symbol(:logistic_, probability_logit)
    node_names = (
        rate.node_names..., probability.node_names...,
        rate_name, probability_name)
    length(unique(node_names)) == length(node_names) || throw(CapabilityError(
        :graph_identity,
        "zero-inflated predictor nodes have colliding identities"))
    node_declarations = (
        rate.node_declarations..., probability.node_declarations...,
        exp_link(log_rate), logistic_link(probability_logit))
    input_names = Tuple(predictor_names)
    declaration = model(
        inputs=_declaration_namedtuple(
            input_names, Tuple(input() for _ in input_names)),
        parameters=_declaration_namedtuple(
            (rate.coefficient_name, probability.coefficient_name),
            (rate.parameter_declaration, probability.parameter_declaration)),
        nodes=_declaration_namedtuple(node_names, node_declarations),
        observations=NamedTuple{(response,)}((broadcasted(
            zero_inflated_poisson(
                response, rate_name, probability_name)),)))
    bindings = _declaration_namedtuple(
        input_names,
        Tuple(parent(parent(column)) for column in predictor_columns))
    conditions = NamedTuple{(response,)}((parent(parent(response_lhs)),))
    (; declaration, bindings, conditions)
end

function _lower_brmi_observation_weight(response::Symbol, likelihood)
    BRM.getf(likelihood) === BRM.weighted || return nothing, likelihood
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :observation_weights,
        "weighted likelihood for `$response` accepts no keywords"))
    arguments = BRM.getargs(likelihood)
    length(arguments) == 2 || throw(CapabilityError(
        :observation_weights,
        "weighted likelihood for `$response` needs a distribution and weights"))
    distribution, weight = arguments
    distribution isa BRM.ExprColumn || throw(CapabilityError(
        :observation_weights,
        "weighted likelihood for `$response` requires a distribution call"))
    weight isa BRM.ExprColumn || throw(CapabilityError(
        :observation_weights,
        "weighted likelihood for `$response` requires a typed StatsBase weight"))
    isempty(BRM.getkwargs(weight)) || throw(CapabilityError(
        :observation_weights,
        "weight constructor for `$response` accepts no keywords"))
    constructor = BRM.getf(weight)
    kind = constructor === BRM.aweights || constructor === BRM.AnalyticWeights ?
        :analytic :
        constructor === BRM.fweights || constructor === BRM.FrequencyWeights ?
        :frequency :
        constructor === BRM.weights || constructor === BRM.Weights ?
        :power :
        constructor === BRM.uweights || constructor === BRM.UnitWeights ?
        :unit : nothing
    (constructor === BRM.pweights ||
     constructor === BRM.ProbabilityWeights) && throw(CapabilityError(
        :observation_weights,
        "ProbabilityWeights semantics are not implemented for `$response`"))
    kind === nothing && throw(CapabilityError(
        :observation_weights,
        "unsupported observation-weight constructor `$constructor` for `$response`"))
    weight_arguments = BRM.getargs(weight)
    length(weight_arguments) == 1 || throw(CapabilityError(
        :observation_weights,
        "weight constructor for `$response` requires one data column"))
    column = only(weight_arguments)
    column isa BRM.NamedColumn && parent(column) isa BRM.DataColumn || throw(
        CapabilityError(
            :observation_weights,
            "weight constructor for `$response` requires one raw data column"))
    source = BRM.name(column)
    (; kind, source, column), distribution
end

function _lower_brmi_evidence_bound(response::Symbol, value,
                                    label::AbstractString)
    value === nothing && return (; value=nothing, source=nothing, column=nothing)
    if value isa Real
        isfinite(value) || throw(CapabilityError(
            :response_evidence,
            "$label for `$response` must be finite"))
        return (; value, source=nothing, column=nothing)
    end
    value isa BRM.NamedColumn && parent(value) isa BRM.DataColumn || throw(
        CapabilityError(
            :response_evidence,
            "$label for `$response` must be a finite literal, raw data " *
            "column, or `nothing`"))
    source = BRM.name(value)
    (; value=source, source, column=value)
end

function _lower_brmi_response_evidence(response::Symbol, likelihood)
    family = BRM.getf(likelihood)
    kind = family === BRM.truncated ? :truncated :
        family === BRM.censored ? :censored :
        family === BRM.interval_censored ? :interval_censored : nothing
    kind === nothing && return nothing, likelihood
    arguments = BRM.getargs(likelihood)
    isempty(arguments) && throw(CapabilityError(
        :response_evidence,
        "$family wrapper for `$response` needs a base distribution"))
    distribution = first(arguments)
    distribution isa BRM.ExprColumn || throw(CapabilityError(
        :response_evidence,
        "$family wrapper for `$response` requires a distribution call"))
    positional = arguments[2:end]
    keywords = BRM.getkwargs(likelihood)
    allowed = kind === :interval_censored ? (:upper,) : (:lower, :upper)
    unknown = Tuple(key for key in keys(keywords) if key ∉ allowed)
    isempty(unknown) || throw(CapabilityError(
        :response_evidence,
        "$family wrapper for `$response` has unsupported keywords $unknown"))

    lower = nothing
    upper = nothing
    if kind === :interval_censored
        isempty(positional) || throw(CapabilityError(
            :response_evidence,
            "interval_censored for `$response` accepts `upper` only as a keyword"))
        haskey(keywords, :upper) || throw(CapabilityError(
            :response_evidence,
            "interval_censored for `$response` requires `upper=`"))
        upper = keywords[:upper]
    else
        length(positional) <= 2 || throw(CapabilityError(
            :response_evidence,
            "$family for `$response` accepts at most lower and upper bounds"))
        !isempty(positional) && (lower = positional[1])
        length(positional) == 2 && (upper = positional[2])
        haskey(keywords, :lower) && begin
            isempty(positional) || throw(CapabilityError(
                :response_evidence,
                "$family for `$response` specifies `lower` twice"))
            lower = keywords[:lower]
        end
        haskey(keywords, :upper) && begin
            length(positional) < 2 || throw(CapabilityError(
                :response_evidence,
                "$family for `$response` specifies `upper` twice"))
            upper = keywords[:upper]
        end
        lower === nothing && upper === nothing && throw(CapabilityError(
            :response_evidence,
            "$family for `$response` needs at least one bound"))
    end
    lower = _lower_brmi_evidence_bound(
        response, lower, "$family lower bound")
    upper = _lower_brmi_evidence_bound(
        response, upper, "$family upper bound")
    if lower.value isa Real && upper.value isa Real &&
       lower.value >= upper.value
        throw(CapabilityError(
            :response_evidence,
            "$family lower bound for `$response` must be below its upper bound"))
    end
    (; kind, lower, upper), distribution
end

function _lower_brmi_multiple_outcomes(brmi::BRM.BRMI, outcomes)
    response_names = Tuple(outcome.response for outcome in outcomes)
    length(unique(response_names)) == length(response_names) || throw(
        CapabilityError(
            :outcomes,
            "multiple-outcome BRM responses must have distinct identities"))
    specs = map(outcomes) do outcome
        response = outcome.response
        response_lhs, likelihood = BRM._native_ppl_sampling_rhs(
            brmi, response)
        response_lhs isa BRM.NamedColumn &&
            parent(response_lhs) isa BRM.DataColumn || throw(
                CapabilityError(
                    :response_decorator,
                    "multiple-outcome response `$response` must be a bare " *
                    "observed data column"))
        likelihood isa BRM.ExprColumn || throw(CapabilityError(
            :likelihood,
            "multiple-outcome response `$response` must use a distribution call"))
        weight_spec, likelihood = _lower_brmi_observation_weight(
            response, likelihood)
        evidence_spec, likelihood = _lower_brmi_response_evidence(
            response, likelihood)
        weight_spec === nothing || evidence_spec === nothing || throw(
            CapabilityError(
                :response_evidence,
                "weighted response evidence for `$response` is not yet " *
                "lowered by the native PPL"))
        family = BRM.getf(likelihood)
        family === BRM.Normal || family === BRM.BernoulliLogit ||
            family === BRM.Poisson || throw(
            CapabilityError(
                :likelihood,
                "the mixed multiple-outcome native-PPL slice requires " *
                "Normal(location, scale), BernoulliLogit(logit), or " *
                "Poisson(exp(log_rate)) responses; " *
                "`$response` uses `$family`"))
        isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
            :likelihood_keywords,
            "$family likelihood for `$response` cannot have keywords"))
        arguments = BRM.getargs(likelihood)
        expected_arguments = family === BRM.Normal ? 2 : 1
        length(arguments) == expected_arguments || throw(CapabilityError(
            :likelihood,
            "$family response `$response` needs $expected_arguments argument(s)"))
        rate = nothing
        location = if family === BRM.Poisson
            rate_expression = only(arguments)
            rate_expression isa BRM.ExprColumn &&
                BRM.getf(rate_expression) === exp || throw(CapabilityError(
                    :likelihood_link,
                    "Poisson response `$response` must use " *
                    "`Poisson(exp(log_rate))`"))
            isempty(BRM.getkwargs(rate_expression)) || throw(CapabilityError(
                :likelihood_link,
                "Poisson `exp` link for `$response` cannot have keywords"))
            rate_arguments = BRM.getargs(rate_expression)
            length(rate_arguments) == 1 || throw(CapabilityError(
                :likelihood_link,
                "Poisson `exp` link for `$response` needs one named predictor"))
            log_rate = BRM._native_ppl_ref_name(only(rate_arguments))
            log_rate === nothing && throw(CapabilityError(
                :likelihood_location,
                "Poisson `exp` link for `$response` must consume one " *
                "named linear predictor"))
            rate = Symbol(:exp_, log_rate)
            log_rate
        else
            BRM._native_ppl_ref_name(arguments[1])
        end
        location === nothing && throw(CapabilityError(
            :likelihood_location,
            "$family response `$response` predictor must be one named " *
            "linear predictor"))
        scale = family === BRM.Normal ?
            BRM._native_ppl_ref_name(arguments[2]) : nothing
        family === BRM.Normal && scale === nothing && throw(CapabilityError(
            :likelihood_scale,
            "Normal response `$response` scale must be one named scalar " *
            "parameter"))
        identities = family === BRM.Normal ?
            (response, location, scale) :
            family === BRM.Poisson ? (response, location, rate) :
            (response, location)
        length(unique(identities)) == length(identities) ||
            throw(CapabilityError(
                :likelihood_parameters,
                "$family response and parameter identities for `$response` " *
                "must be distinct"))
        affine = BRM._native_ppl_affine_components(brmi, location)
        affine.intercept || throw(CapabilityError(
            :predictor_terms,
            "multiple-outcome predictor `$location` must include an intercept"))
        isempty(affine.offsets) && isempty(affine.data_offsets) || throw(
            CapabilityError(
                :predictor_terms,
                "multiple-outcome predictor `$location` currently accepts " *
                "population terms and grouped intercepts, but not offsets"))
        all(term -> term.transform === :identity, affine.predictors) || throw(
            CapabilityError(
                :predictor_terms,
                "multiple-outcome predictor `$location` currently accepts " *
                "only raw, untransformed predictors"))
        predictor_columns = Tuple(term.column for term in affine.predictors)
        predictor_names = Tuple(BRM.name(column)
                                for column in predictor_columns)
        length(unique(predictor_names)) == length(predictor_names) || throw(
            CapabilityError(
                :predictor_terms,
                "multiple-outcome predictor `$location` repeats a raw input"))
        group_specs = map(affine.groups) do varying
            group_name = BRM.name(varying.group)
            group_predictor_names = Tuple(
                predictor === nothing ? nothing : begin
                    predictor.transform === :identity || throw(
                        CapabilityError(
                            :group_term,
                            "multiple-outcome grouped predictors currently " *
                            "require raw, untransformed inputs"))
                    BRM.name(predictor.column)
                end
                for predictor in varying.predictors)
            coefficient_keys = Tuple(
                name === nothing ? :Intercept : name
                for name in group_predictor_names)
            correlated = length(coefficient_keys) > 1
            prior = _brmi_group_sd_prior(brmi, varying.id)
            # Validate the correlation prior only after cross-outcome `|ID|`
            # buckets have been assembled: a formula-local K can be a typed
            # projection of a larger shared block.
            correlation_prior = nothing
            scale_name = Symbol(:tau_, varying.id, :_, group_name)
            correlation_name = Symbol(:L_, varying.id, :_, group_name)
            site_name = Symbol(:b_, varying.id, :_, group_name)
            gather_name = correlated ? Symbol(
                site_name, :_by_, group_name, :_for_, location) :
                Symbol(:r_, location, :_, varying.id, :_, group_name)
            predictor_value = length(group_predictor_names) == 1 ?
                only(group_predictor_names) : nothing
            product_name = !correlated && predictor_value !== nothing ?
                Symbol(gather_name, :_times_, predictor_value) : nothing
            (; id=varying.id, group_name, column=varying.group, prior,
               predictor_terms=varying.predictors,
               predictor_names=group_predictor_names, coefficient_keys,
               coefficient_indices=ntuple(identity, length(coefficient_keys)),
               coefficient_count=length(coefficient_keys),
               correlated, correlation_prior,
               scale_name, correlation_name, site_name, gather_name,
               predictor_value, product_name,
               offset_name=product_name === nothing ?
                   gather_name : product_name)
        end
        group_names = Tuple(spec.group_name for spec in group_specs)
        length(unique(group_names)) == length(group_names) || throw(
            CapabilityError(
                :group_term,
                "multiple-outcome predictor `$location` repeats a grouping " *
                "input"))
        prior_scale = family === BRM.Normal ?
            BRM._native_ppl_exponential_prior(brmi, scale) : nothing
        coefficient = Symbol(:beta_, location)
        (; family, response, location, rate, scale, coefficient,
           predictor_columns, predictor_names, prior_scale,
           group_specs, weight_spec, evidence_spec,
           response_values=parent(parent(response_lhs)))
    end

    # A shared `(terms | id | group)` identifier denotes one latent block even
    # when its coefficients feed different outcome predictors.  Keep the
    # formula-local predictor tuples on each consumer node, but give every
    # consumer an explicit projection into one common coefficient axis.  This
    # is a typed graph connection; neither lowering nor execution relies on
    # predictor/site names happening to match.
    group_buckets = Dict{Tuple{Symbol,Symbol},Vector{Tuple{Int,Int}}}()
    for (spec_index, spec) in pairs(specs)
        for (group_index, group_spec) in pairs(spec.group_specs)
            key = (group_spec.id, group_spec.group_name)
            push!(get!(group_buckets, key, Tuple{Int,Int}[]),
                  (spec_index, group_index))
        end
    end
    shared_group_overrides = Dict{Tuple{Int,Int},Any}()
    for entries in values(group_buckets)
        first_spec_index, first_group_index = first(entries)
        first_group = specs[first_spec_index].group_specs[first_group_index]
        if length(entries) == 1
            first_group.correlated || continue
            shared_group_overrides[first(entries)] = (;
                correlation_prior=_brmi_group_correlation_prior(
                    brmi, first_group.id,
                    length(first_group.coefficient_keys)))
            continue
        end
        coefficient_keys_buffer = Symbol[]
        entry_indices = Dict{Tuple{Int,Int},Tuple}()
        for entry in entries
            spec_index, group_index = entry
            spec = specs[spec_index]
            group_spec = spec.group_specs[group_index]
            group_spec.column == first_group.column || throw(CapabilityError(
                :group_term,
                "shared grouped id `$(group_spec.id)` must use one grouping " *
                "column across outcome predictors"))
            first_index = length(coefficient_keys_buffer) + 1
            append!(coefficient_keys_buffer, map(
                group_spec.coefficient_keys) do key
                    key === :Intercept ? spec.location :
                        Symbol(spec.location, :_, key)
                end)
            last_index = length(coefficient_keys_buffer)
            entry_indices[entry] = Tuple(first_index:last_index)
        end
        coefficient_keys = Tuple(coefficient_keys_buffer)
        length(unique(coefficient_keys)) == length(coefficient_keys) || throw(
            CapabilityError(
                :group_term,
                "shared grouped id `$(first_group.id)` produces duplicate " *
                "cross-outcome coefficient identities"))
        coefficient_count = length(coefficient_keys)
        correlation_prior = _brmi_group_correlation_prior(
            brmi, first_group.id, coefficient_count)
        for entry in entries
            spec_index, group_index = entry
            spec = specs[spec_index]
            group_spec = spec.group_specs[group_index]
            gather_name = Symbol(
                group_spec.site_name, :_by_, group_spec.group_name,
                :_for_, spec.location)
            shared_group_overrides[entry] = (;
                coefficient_keys,
                coefficient_indices=entry_indices[entry],
                coefficient_count,
                correlated=true,
                correlation_prior,
                gather_name,
                predictor_value=nothing,
                product_name=nothing,
                offset_name=gather_name)
        end
    end
    specs = ntuple(length(specs)) do spec_index
        spec = specs[spec_index]
        group_specs = ntuple(length(spec.group_specs)) do group_index
            group_spec = spec.group_specs[group_index]
            override = get(shared_group_overrides,
                           (spec_index, group_index), nothing)
            override === nothing ? group_spec : merge(group_spec, override)
        end
        merge(spec, (; group_specs))
    end

    coefficient_names_buffer = Symbol[]
    coefficient_specs_buffer = Any[]
    other_parameter_names_buffer = Symbol[]
    other_parameter_declarations_buffer = Any[]
    node_names_buffer = Symbol[]
    node_declarations_buffer = Any[]
    function add_other_parameter!(name, declaration)
        index = findfirst(==(name), other_parameter_names_buffer)
        if index === nothing
            push!(other_parameter_names_buffer, name)
            push!(other_parameter_declarations_buffer, declaration)
        elseif other_parameter_declarations_buffer[index] != declaration
            throw(CapabilityError(
                :graph_identity,
                "multiple-outcome parameter `$name` has conflicting " *
                "declarations"))
        end
        nothing
    end
    for spec in specs
        if spec.coefficient ∉ coefficient_names_buffer
            push!(coefficient_names_buffer, spec.coefficient)
            push!(coefficient_specs_buffer, spec)
        end
        for group_spec in spec.group_specs
            add_other_parameter!(group_spec.scale_name, parameter(
                PositiveSupport(), group_spec.correlated ?
                    group_spec.coefficient_keys : (group_spec.scale_name,);
                transform=Exp(), prior=Exponential(group_spec.prior.scale)))
            if group_spec.correlated
                add_other_parameter!(
                    group_spec.correlation_name,
                    cholesky_correlation(
                        group_spec.coefficient_keys,
                        group_spec.correlation_prior.eta))
            end
            add_other_parameter!(group_spec.site_name,
                group_spec.correlated ? grouped_standard_normal(
                    group_spec.group_name, group_spec.coefficient_keys) :
                    grouped_normal(
                        group_spec.group_name, 0.0, group_spec.scale_name))
            push!(node_names_buffer, group_spec.gather_name)
            push!(node_declarations_buffer,
                group_spec.correlated ? grouped_affine(
                    group_spec.site_name, group_spec.scale_name,
                    group_spec.correlation_name, group_spec.group_name,
                    group_spec.predictor_names;
                    coefficient_indices=group_spec.coefficient_indices,
                    coefficient_count=group_spec.coefficient_count) : group_gather(
                    group_spec.site_name, group_spec.group_name))
            if group_spec.product_name !== nothing
                push!(node_names_buffer, group_spec.product_name)
                push!(node_declarations_buffer, row_product(
                    group_spec.gather_name, group_spec.predictor_value))
            end
        end
        if spec.scale !== nothing
            add_other_parameter!(spec.scale, parameter(
                PositiveSupport(), (spec.scale,);
                transform=Exp(), prior=Exponential(spec.prior_scale)))
        end
        if spec.location ∉ node_names_buffer
            push!(node_names_buffer, spec.location)
            push!(node_declarations_buffer, affine(
                spec.predictor_names, spec.coefficient;
                offsets=Tuple(group_spec.offset_name
                              for group_spec in spec.group_specs),
                intercept=true))
        end
        if spec.rate !== nothing && spec.rate ∉ node_names_buffer
            push!(node_names_buffer, spec.rate)
            push!(node_declarations_buffer, exp_link(spec.location))
        end
    end
    coefficient_names = Tuple(coefficient_names_buffer)
    coefficient_specs = Tuple(coefficient_specs_buffer)
    other_parameter_names = Tuple(other_parameter_names_buffer)
    other_parameter_declarations =
        Tuple(other_parameter_declarations_buffer)
    node_names = Tuple(node_names_buffer)
    all_graph_names = (response_names..., coefficient_names...,
                       other_parameter_names..., node_names...)
    length(unique(all_graph_names)) == length(all_graph_names) || throw(
        CapabilityError(
            :graph_identity,
            "multiple-outcome responses, predictors, coefficients, and " *
            "scales must have distinct graph identities"))

    input_names_buffer = Symbol[]
    input_columns_buffer = Any[]
    for spec in specs
        for (name, column) in zip(
                spec.predictor_names, spec.predictor_columns)
            name in input_names_buffer && continue
            push!(input_names_buffer, name)
            push!(input_columns_buffer, column)
        end
        for group_spec in spec.group_specs
            for (name, predictor) in zip(
                    group_spec.predictor_names,
                    group_spec.predictor_terms)
                name === nothing && continue
                name in input_names_buffer && continue
                push!(input_names_buffer, name)
                push!(input_columns_buffer, predictor.column)
            end
            group_spec.group_name in input_names_buffer && continue
            push!(input_names_buffer, group_spec.group_name)
            push!(input_columns_buffer, group_spec.column)
        end
        if spec.weight_spec !== nothing &&
           spec.weight_spec.source ∉ input_names_buffer
            push!(input_names_buffer, spec.weight_spec.source)
            push!(input_columns_buffer, spec.weight_spec.column)
        end
        if spec.evidence_spec !== nothing
            for bound in (spec.evidence_spec.lower, spec.evidence_spec.upper)
                bound.source === nothing && continue
                bound.source in input_names_buffer && continue
                push!(input_names_buffer, bound.source)
                push!(input_columns_buffer, bound.column)
            end
        end
    end
    input_names = Tuple(input_names_buffer)
    input_declarations = _declaration_namedtuple(
        input_names, map(_ -> input(), input_names))
    coefficient_declarations = Tuple(map(coefficient_specs) do spec
        parameter(
            RealSupport(), (:Intercept, spec.predictor_names...);
            transform=Identity(), prior=StandardNormal())
    end)
    parameter_declarations = _declaration_namedtuple(
        (coefficient_names..., other_parameter_names...),
        (coefficient_declarations..., other_parameter_declarations...))
    node_declarations = _declaration_namedtuple(
        node_names, Tuple(node_declarations_buffer))
    observation_values = map(specs) do spec
        observation = spec.family === BRM.Normal ? normal(
            spec.response, spec.location, spec.scale) :
            spec.family === BRM.BernoulliLogit ? bernoulli_logit(
                spec.response, spec.location) : poisson(
                spec.response, spec.rate)
        if spec.weight_spec !== nothing
            observation = weighted_observation(
                observation,
                observation_weight(
                    spec.weight_spec.kind, spec.weight_spec.source))
        end
        if spec.evidence_spec !== nothing
            evidence = if spec.evidence_spec.kind === :truncated
                truncated_evidence(;
                    lower=spec.evidence_spec.lower.value,
                    upper=spec.evidence_spec.upper.value)
            elseif spec.evidence_spec.kind === :censored
                censored_evidence(;
                    lower=spec.evidence_spec.lower.value,
                    upper=spec.evidence_spec.upper.value)
            else
                interval_evidence(spec.evidence_spec.upper.value)
            end
            observation = evidence_observation(observation, evidence)
        end
        broadcasted(observation)
    end
    observation_declarations = _declaration_namedtuple(
        response_names, Tuple(observation_values))
    site_order_buffer = Symbol[]
    for spec in specs
        for group_spec in spec.group_specs
            group_spec.scale_name in site_order_buffer ||
                push!(site_order_buffer, group_spec.scale_name)
            if group_spec.correlated
                group_spec.correlation_name in site_order_buffer ||
                    push!(site_order_buffer, group_spec.correlation_name)
            end
            group_spec.site_name in site_order_buffer ||
                push!(site_order_buffer, group_spec.site_name)
        end
        spec.coefficient in site_order_buffer ||
            push!(site_order_buffer, spec.coefficient)
        spec.scale === nothing || spec.scale in site_order_buffer ||
            push!(site_order_buffer, spec.scale)
        push!(site_order_buffer, spec.response)
    end
    site_order = Tuple(site_order_buffer)
    declaration = model(
        inputs=input_declarations,
        parameters=parameter_declarations,
        nodes=node_declarations,
        observations=observation_declarations,
        site_order=site_order)

    expected_operations = Set(input_names)
    for spec in specs
        union!(expected_operations, (spec.location, spec.response))
        spec.scale === nothing || push!(expected_operations, spec.scale)
        for group_spec in spec.group_specs
            push!(expected_operations, group_spec.prior.operation)
            group_spec.correlation_prior === nothing ||
                push!(expected_operations,
                      group_spec.correlation_prior.operation)
        end
    end
    extras = setdiff(Set(keys(brmi.operations)), expected_operations)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported multiple-outcome formula operations: " *
        join(sort!(collect(extras)), ", ")))
    bindings = _declaration_namedtuple(
        input_names,
        Tuple(parent(parent(column)) for column in input_columns_buffer))
    conditions = _declaration_namedtuple(
        response_names, Tuple(spec.response_values for spec in specs))
    (; declaration, bindings, conditions)
end

function _lower_brmi_mixed_distributional_outcomes(
        brmi::BRM.BRMI, outcomes)
    length(outcomes) == 2 || throw(CapabilityError(
        :outcomes, "mixed distributional native-PPL slice expects two responses"))
    response_names = Tuple(outcome.response for outcome in outcomes)
    length(unique(response_names)) == 2 || throw(CapabilityError(
        :outcomes, "mixed distributional responses must be distinct"))
    specs = NamedTuple[]
    for outcome in outcomes
        response = outcome.response
        response_lhs, likelihood = BRM._native_ppl_sampling_rhs(brmi, response)
        response_lhs isa BRM.NamedColumn && parent(response_lhs) isa BRM.DataColumn ||
            throw(CapabilityError(:response_decorator,
                "mixed distributional response `$response` must be a bare data column"))
        likelihood isa BRM.ExprColumn || throw(CapabilityError(
            :likelihood, "mixed distributional response `$response` needs a distribution"))
        BRM.getf(likelihood) in (BRM.Normal, BRM.BernoulliLogit) ||
            throw(CapabilityError(:likelihood,
                "mixed distributional slice supports Normal and BernoulliLogit"))
        isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
            :likelihood_keywords, "mixed distributional likelihoods cannot have keywords"))
        args = BRM.getargs(likelihood)
        family = BRM.getf(likelihood)
        expected_arguments = family === BRM.Normal ? 2 : 1
        length(args) == expected_arguments || throw(CapabilityError(
            :likelihood,
            "mixed distributional $family response `$response` needs " *
            "$expected_arguments argument(s)"))
        location_expression = first(args)
        scale_expression = family === BRM.Normal ? last(args) : nothing
        log_scale = nothing
        if family === BRM.Normal
            scale_expression isa BRM.ExprColumn && BRM.getf(scale_expression) === exp ||
                throw(CapabilityError(:likelihood_link,
                    "mixed distributional Normal scale must be exp(log_scale)"))
            scale_args = BRM.getargs(scale_expression)
            length(scale_args) == 1 || throw(CapabilityError(
                :likelihood_link, "mixed distributional exp scale needs one predictor"))
            log_scale = BRM._native_ppl_ref_name(only(scale_args))
            log_scale === nothing && throw(CapabilityError(
                :likelihood_scale, "mixed distributional exp scale needs a named predictor"))
        end
        location = BRM._native_ppl_ref_name(location_expression)
        location === nothing && throw(CapabilityError(
            :likelihood_location, "mixed distributional predictors must be named"))
        push!(specs, (; response, response_lhs, family, location, log_scale,
                      response_value=parent(parent(response_lhs))))
    end
    normal_specs = filter(spec -> spec.family === BRM.Normal, specs)
    bernoulli_specs = filter(
        spec -> spec.family === BRM.BernoulliLogit, specs)
    length(normal_specs) == 1 && length(bernoulli_specs) == 1 || throw(
        CapabilityError(
            :likelihood,
            "mixed distributional slice requires exactly one Normal and " *
            "one BernoulliLogit response"))
    normal_spec = only(normal_specs)
    bernoulli_spec = only(bernoulli_specs)
    predictor_specs = NamedTuple[]
    for (predictor, consumer) in (
            (normal_spec.location, normal_spec.response),
            (normal_spec.log_scale, normal_spec.response),
            (bernoulli_spec.location, bernoulli_spec.response))
        components = BRM._native_ppl_affine_components(brmi, predictor)
        components.intercept || throw(CapabilityError(
            :predictor_terms, "mixed distributional predictor `$predictor` needs an intercept"))
        isempty(components.offsets) && isempty(components.data_offsets) ||
            throw(CapabilityError(:predictor_terms,
                "mixed distributional predictors do not yet accept offsets"))
        all(term -> term.transform === :identity, components.predictors) ||
            throw(CapabilityError(:predictor_terms,
                "mixed distributional predictors require raw predictors"))
        raw_names = Tuple(BRM.name(term.column) for term in components.predictors)
        length(unique(raw_names)) == length(raw_names) || throw(
            CapabilityError(
                :predictor_terms,
                "mixed distributional predictor `$predictor` repeats a " *
                "raw input"))
        coefficient_name = Symbol(:beta_, predictor)
        group_specs = map(components.groups) do varying
            group_name = BRM.name(varying.group)
            predictor_names = Tuple(term === nothing ? nothing :
                BRM.name(term.column) for term in varying.predictors)
            coefficient_keys = Tuple(name === nothing ? :Intercept : name
                                     for name in predictor_names)
            prior = _brmi_group_sd_prior(brmi, varying.id)
            correlated = length(coefficient_keys) > 1
            scale_name = Symbol(:tau_, varying.id, :_, group_name)
            correlation_name = Symbol(:L_, varying.id, :_, group_name)
            site_name = Symbol(:b_, varying.id, :_, group_name)
            gather_name = Symbol(site_name, :_by_, group_name, :_for_, predictor)
            (; id=varying.id, group_name, column=varying.group, prior,
               predictor_terms=varying.predictors, predictor_names,
               coefficient_keys, correlated, scale_name, correlation_name,
               site_name, gather_name, predictor, consumer)
        end
        push!(predictor_specs, (; predictor, consumer, components,
            raw_names, raw_columns=Tuple(term.column for term in components.predictors),
            coefficient_name, coefficient_keys=(:Intercept, raw_names...),
            group_specs))
    end
    predictor_identities = Tuple(spec.predictor for spec in predictor_specs)
    length(unique(predictor_identities)) == length(predictor_identities) ||
        throw(CapabilityError(
            :group_term,
            "mixed distributional responses require distinct location and " *
            "log-scale predictors so each shared group coefficient has one " *
            "typed consumer"))
    all(length(spec.group_specs) == 1 for spec in predictor_specs) || throw(
        CapabilityError(
            :group_term,
            "mixed distributional slice requires exactly one shared " *
            "grouped intercept on each predictor"))
    shared_group = only(first(predictor_specs).group_specs)
    for spec in predictor_specs
        group = only(spec.group_specs)
        group.id == shared_group.id &&
            group.group_name == shared_group.group_name &&
            group.column == shared_group.column || throw(CapabilityError(
                :group_term,
                "mixed distributional predictors must share one `|ID|` " *
                "and grouping column"))
        group.coefficient_keys == (:Intercept,) || throw(CapabilityError(
            :group_term,
            "mixed distributional K=3 slice currently supports grouped " *
            "intercepts; grouped slopes use the arbitrary-K path"))
    end
    group_buckets = Dict{Tuple{Symbol,Symbol},Vector{Int}}()
    for (index, spec) in pairs(predictor_specs)
        for group in spec.group_specs
            push!(get!(group_buckets, (group.id, group.group_name), Int[]), index)
        end
    end
    group_axis = Dict{Tuple{Int,Int},Any}()
    for indices in values(group_buckets)
        first_index = first(indices)
        first_group = first(predictor_specs[first_index].group_specs)
        keys_buffer = Symbol[]
        ranges = Dict{Int,Tuple}()
        for index in indices
            spec = predictor_specs[index]
            group = only(filter(g -> g.id == first_group.id &&
                g.group_name == first_group.group_name, spec.group_specs))
            start = length(keys_buffer) + 1
            append!(keys_buffer, map(group.coefficient_keys) do key
                key === :Intercept ? spec.predictor : Symbol(spec.predictor, :_, key)
            end)
            ranges[index] = Tuple(start:length(keys_buffer))
            group.column == first_group.column || throw(CapabilityError(
                :group_term, "shared mixed distributional groups need one column"))
        end
        axis = Tuple(keys_buffer)
        corr = _brmi_group_correlation_prior(brmi, first_group.id, length(axis))
        for index in indices
            group_axis[(index, 1)] = (; coefficient_keys=axis,
                coefficient_indices=ranges[index], coefficient_count=length(axis),
                correlation_prior=corr, correlated=true)
        end
    end
    input_names = Symbol[]
    input_columns = Any[]
    for spec in predictor_specs
        for (name, column) in zip(spec.raw_names, spec.raw_columns)
            name in input_names || (push!(input_names, name); push!(input_columns, column))
        end
    end
    for spec in predictor_specs, group in spec.group_specs
        group.group_name in input_names ||
            (push!(input_names, group.group_name); push!(input_columns, group.column))
    end
    coefficient_names = Symbol[spec.coefficient_name for spec in predictor_specs]
    coefficient_declarations = Any[parameter(
        RealSupport(), spec.coefficient_keys;
        transform=Identity(), prior=StandardNormal()) for spec in predictor_specs]
    other_names = Symbol[]
    other_declarations = Any[]
    add_other!(name, declaration) = begin
        index = findfirst(==(name), other_names)
        index === nothing ? (push!(other_names, name), push!(other_declarations, declaration)) :
            other_declarations[index] == declaration || throw(CapabilityError(
                :graph_identity, "mixed distributional parameter `$name` conflicts"))
    end
    node_names = Symbol[]
    node_declarations = Any[]
    log_scale_spelling = String(normal_spec.log_scale)
    scale_name = if startswith(log_scale_spelling, "log_")
        suffix = log_scale_spelling[5:end]
        isempty(suffix) && throw(CapabilityError(
            :likelihood_scale,
            "mixed distributional log-scale predictor `$(normal_spec.log_scale)` " *
            "does not leave a valid derived scale identity after `log_`"))
        Symbol(suffix)
    else
        Symbol(:exp_, normal_spec.log_scale)
    end
    for (index, spec) in pairs(predictor_specs)
        if index == 3
            push!(node_names, scale_name)
            push!(node_declarations, exp_link(normal_spec.log_scale))
        end
        for group in spec.group_specs
            override = get(group_axis, (index, 1), nothing)
            keys = override === nothing ? group.coefficient_keys : override.coefficient_keys
            indices = override === nothing ? ntuple(identity, length(keys)) : override.coefficient_indices
            add_other!(group.scale_name, parameter(
                PositiveSupport(), keys; transform=Exp(), prior=Exponential(group.prior.scale)))
            add_other!(group.correlation_name, cholesky_correlation(
                keys, (override === nothing ? _brmi_group_correlation_prior(
                    brmi, group.id, length(keys)) : override.correlation_prior).eta))
            add_other!(group.site_name, grouped_standard_normal(group.group_name, keys))
            push!(node_names, group.gather_name)
            push!(node_declarations, grouped_affine(
                group.site_name, group.scale_name, group.correlation_name,
                group.group_name, group.predictor_names;
                coefficient_indices=indices, coefficient_count=length(keys)))
        end
        push!(node_names, spec.predictor)
        push!(node_declarations, affine(
            spec.raw_names, spec.coefficient_name;
            offsets=Tuple(group.gather_name for group in spec.group_specs), intercept=true))
    end
    (scale_name in input_names ||
     scale_name in coefficient_names ||
     scale_name in other_names ||
     scale_name in response_names ||
     count(==(scale_name), node_names) != 1) && throw(CapabilityError(
        :graph_identity,
        "derived mixed distributional Normal scale node `$scale_name` " *
        "collides with another input, parameter, node, or observation identity"))
    params = _declaration_namedtuple(
        (coefficient_names..., other_names...),
        (coefficient_declarations..., other_declarations...))
    nodes = _declaration_namedtuple(Tuple(node_names), Tuple(node_declarations))
    observation_values = map(specs) do spec
        spec.family === BRM.Normal ? broadcasted(normal(
            spec.response, spec.location, scale_name)) :
            broadcasted(bernoulli_logit(spec.response, spec.location))
    end
    observations = _declaration_namedtuple(
        response_names, Tuple(observation_values))
    group_scale = shared_group.scale_name
    group_correlation = shared_group.correlation_name
    group_site = shared_group.site_name
    declaration = model(
        inputs=_declaration_namedtuple(Tuple(input_names), Tuple(input() for _ in input_names)),
        parameters=params, nodes=nodes, observations=observations,
        site_order=(group_scale, group_correlation, group_site,
                    Symbol(:beta_, normal_spec.location),
                    Symbol(:beta_, normal_spec.log_scale),
                    normal_spec.response,
                    Symbol(:beta_, bernoulli_spec.location),
                    bernoulli_spec.response))
    expected = Set(response_names)
    union!(expected, (normal_spec.location, normal_spec.log_scale,
                      bernoulli_spec.location))
    for spec in predictor_specs, group in spec.group_specs
        push!(expected, group.prior.operation)
    end
    push!(expected, _brmi_group_correlation_prior(
        brmi, shared_group.id, length(predictor_specs)).operation)
    union!(expected, input_names)
    extras = setdiff(Set(keys(brmi.operations)), expected)
    isempty(extras) || throw(CapabilityError(
        :additional_operations, "unsupported mixed distributional operations: " *
        join(sort!(collect(extras)), ", ")))
    bindings = _declaration_namedtuple(
        Tuple(input_names), Tuple(parent(parent(column)) for column in input_columns))
    conditions = _declaration_namedtuple(
        response_names, Tuple(spec.response_value for spec in specs))
    (; declaration, bindings, conditions)
end

function _lower_brmi(brmi::BRM.BRMI)
    observed = BRM.outcomes(brmi)
    isempty(observed) && throw(CapabilityError(
        :outcomes, "expected at least one observed response"))
    joint = [o.response for o in observed if o.response isa Tuple]
    isempty(joint) || throw(CapabilityError(
        :correlated_joint_outcomes,
        "correlated joint outcomes $(joint) are supported by the " *
        "StanBlocks backend only; NativePPL cannot silently replace one " *
        "multivariate density with independent scalar terms"))
    if length(observed) > 1
        normal_exp = 0
        bernoulli_outcomes = 0
        for outcome in observed
            _, likelihood = BRM._native_ppl_sampling_rhs(
                brmi, outcome.response)
            likelihood isa BRM.ExprColumn || continue
            family = BRM.getf(likelihood)
            arguments = BRM.getargs(likelihood)
            if family === BRM.Normal && length(arguments) == 2
                scale = arguments[2]
                normal_exp += scale isa BRM.ExprColumn &&
                    BRM.getf(scale) === exp
            elseif family === BRM.BernoulliLogit
                bernoulli_outcomes += 1
            end
        end
        mixed_distributional = length(observed) == 2 &&
            normal_exp == 1 && bernoulli_outcomes == 1
        return mixed_distributional ?
            _lower_brmi_mixed_distributional_outcomes(brmi, observed) :
            _lower_brmi_multiple_outcomes(brmi, observed)
    end
    outcome = only(observed)
    response = outcome.response
    response_lhs, likelihood = BRM._native_ppl_sampling_rhs(brmi, response)
    response_lhs isa BRM.NamedColumn &&
        parent(response_lhs) isa BRM.DataColumn || throw(CapabilityError(
            :response_decorator,
            "response `$response` must be a bare observed data column"))
    likelihood isa BRM.ExprColumn ||
        throw(CapabilityError(
            :likelihood,
            "response `$response` must use a distribution call"))
    weight_spec, likelihood = _lower_brmi_observation_weight(
        response, likelihood)
    evidence_spec, likelihood = _lower_brmi_response_evidence(
        response, likelihood)
    family = BRM.getf(likelihood)
    if family === BRM.ZeroInflatedPoisson
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :mixture_decorator,
                "weighted or censored/truncated ZeroInflatedPoisson is not " *
                "yet supported"))
        return _lower_brmi_zero_inflated_poisson(
            brmi, response, response_lhs, likelihood)
    end
    if family === BRM.NegativeBinomial2
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :count_decorator,
                "weighted or censored/truncated NegativeBinomial2 is not " *
                "yet supported"))
        return _lower_brmi_negative_binomial2(
            brmi, response, response_lhs, likelihood)
    end
    if family === BRM.BetaBinomial2
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :count_decorator,
                "weighted or censored/truncated BetaBinomial2 is not yet " *
                "supported"))
        return _lower_brmi_beta_binomial2(
            brmi, response, response_lhs, likelihood)
    end
    if family === BRM.BinomialLogit
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :count_decorator,
                "weighted or censored/truncated BinomialLogit is not yet " *
                "supported"))
        return _lower_brmi_binomial_logit(
            brmi, response, response_lhs, likelihood)
    end
    if family === BRM.CategoricalLogit
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :categorical_decorator,
                "weighted or censored/truncated CategoricalLogit is not yet " *
                "supported"))
        return _lower_brmi_categorical_logit(
            brmi, response, response_lhs, likelihood)
    end
    is_student_t = family === BRM.LocationScale
    family === BRM.Normal || family === BRM.BernoulliLogit ||
        family === BRM.Poisson || is_student_t || throw(CapabilityError(
            :likelihood,
            "expected `Normal(location, scale)`, `BernoulliLogit(logit)`, " *
            "`Poisson(exp(log_rate))`, or " *
            "`LocationScale(location, scale, TDist(df))`, got `$family`"))
    isempty(BRM.getkwargs(likelihood)) || throw(CapabilityError(
        :likelihood_keywords,
        "$family likelihood for `$response` has keywords"))
    likelihood_args = BRM.getargs(likelihood)
    degrees_freedom_name = nothing
    if is_student_t
        weight_spec === nothing && evidence_spec === nothing || throw(
            CapabilityError(
                :response_evidence,
                "weighted/evidence Student-t regression is not yet supported"))
        length(likelihood_args) == 3 || throw(CapabilityError(
            :likelihood,
            "LocationScale Student-t response `$response` needs location, " *
            "scale, and TDist(degrees_freedom)"))
        base = likelihood_args[3]
        base isa BRM.ExprColumn && BRM.getf(base) === BRM.TDist || throw(
            CapabilityError(
                :likelihood,
                "LocationScale response `$response` currently requires " *
                "TDist(degrees_freedom) as its base distribution"))
        isempty(BRM.getkwargs(base)) || throw(CapabilityError(
            :likelihood_keywords,
            "TDist base distribution for `$response` cannot have keywords"))
        base_arguments = BRM.getargs(base)
        length(base_arguments) == 1 || throw(CapabilityError(
            :likelihood,
            "TDist base distribution for `$response` needs one named " *
            "degrees-of-freedom parameter"))
        degrees_freedom_name =
            BRM._native_ppl_ref_name(only(base_arguments))
        degrees_freedom_name === nothing && throw(CapabilityError(
            :likelihood,
            "TDist degrees of freedom for `$response` must be one named " *
            "scalar parameter"))
        likelihood_args = (likelihood_args[1], likelihood_args[2])
    end
    expected_arguments =
        (family === BRM.Normal || is_student_t) ? 2 : 1
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
    if family === BRM.Normal
        distributional = _lower_brmi_distributional_gaussian(
            brmi, response, response_lhs, location, likelihood_args[2])
        if distributional !== nothing
            weight_spec === nothing && evidence_spec === nothing || throw(
                CapabilityError(
                    :response_evidence,
                    "weighted/evidence distributional Normal regression is " *
                    "not yet supported"))
            return distributional
        end
    end
    has_scale = family === BRM.Normal || is_student_t
    scale_name = has_scale ?
        BRM._native_ppl_ref_name(likelihood_args[2]) : nothing
    has_scale && scale_name === nothing &&
        throw(CapabilityError(
            :likelihood_scale,
            "$family scale must be one named scalar parameter"))
    is_student_t &&
        length(unique((degrees_freedom_name, location, scale_name))) != 3 &&
        throw(CapabilityError(
            :likelihood_parameters,
            "Student-t degrees of freedom, location, and scale must have " *
            "distinct graph identities"))

    if family === BRM.Normal && weight_spec === nothing &&
       evidence_spec === nothing
        factor_dag = _lower_brmi_scalar_factor_dag(
            brmi, response, response_lhs, location)
        factor_dag === nothing || return factor_dag
    end

    affine_components = BRM._native_ppl_affine_components(brmi, location)
    predictor_terms = affine_components.predictors
    sampled_offsets = affine_components.offsets
    data_offsets = affine_components.data_offsets
    varying_groups = affine_components.groups
    has_intercept = affine_components.intercept
    isempty(sampled_offsets) && isempty(data_offsets) &&
        isempty(varying_groups) && !has_intercept &&
        throw(CapabilityError(
        :predictor_terms,
        "the current native-PPL no-intercept affine slice requires a " *
        "sampled offset, data offset, or varying intercept"))
    predictor_columns = map(term -> term.column, predictor_terms)
    predictor_names = map(BRM.name, predictor_columns)
    input_predictor_names = Symbol[predictor_names...]
    input_predictor_columns = Any[predictor_columns...]
    for varying in varying_groups
        for predictor in varying.predictors
            predictor === nothing && continue
            column = predictor.column
            name = BRM.name(column)
            if name ∉ input_predictor_names
                push!(input_predictor_names, name)
                push!(input_predictor_columns, column)
            end
        end
    end
    for data_offset in data_offsets
        if data_offset.name ∉ input_predictor_names
            push!(input_predictor_names, data_offset.name)
            push!(input_predictor_columns, data_offset.column)
        end
    end
    group_names = map(varying_groups) do varying
        BRM.name(varying.group)
    end
    length(unique(group_names)) == length(group_names) || throw(
        CapabilityError(
            :group_term,
            "varying-intercept grouping inputs must be unique"))
    response in predictor_names && throw(CapabilityError(
        :input_roles,
        "predictor `$response` is also the observed response"))
    predictors = map(
        column -> parent(parent(column)), input_predictor_columns)
    groups = map(varying_groups) do varying
        parent(parent(varying.group))
    end
    weight_input_names = weight_spec !== nothing &&
        weight_spec.source ∉ (input_predictor_names..., group_names...) ?
        (weight_spec.source,) : ()
    weight_inputs = isempty(weight_input_names) ? () :
        (parent(parent(weight_spec.column)),)
    evidence_bounds = evidence_spec === nothing ? () :
        (evidence_spec.lower, evidence_spec.upper)
    evidence_input_specs = Tuple(bound for bound in evidence_bounds
        if bound.source !== nothing &&
           bound.source ∉ (input_predictor_names..., group_names...,
                            weight_input_names...))
    evidence_input_names = Tuple(unique(bound.source
                                        for bound in evidence_input_specs))
    evidence_inputs = map(evidence_input_names) do source
        bound = first(entry for entry in evidence_input_specs
                      if entry.source === source)
        parent(parent(bound.column))
    end
    response_values = parent(parent(response_lhs))

    group_specs = map(varying_groups, group_names) do varying, group_name
        prior = _brmi_group_sd_prior(brmi, varying.id)
        group_predictor_names = Tuple(predictor === nothing ? nothing :
            BRM.name(predictor.column) for predictor in varying.predictors)
        coefficient_keys = Tuple(predictor_name === nothing ? :Intercept :
            predictor_name for predictor_name in group_predictor_names)
        correlated = length(coefficient_keys) > 1
        correlation_prior = correlated ? _brmi_group_correlation_prior(
            brmi, varying.id, length(coefficient_keys)) : nothing
        group_scale_name = Symbol(:tau_, varying.id, :_, group_name)
        group_correlation_name = Symbol(:L_, varying.id, :_, group_name)
        group_site_name = Symbol(:b_, varying.id, :_, group_name)
        gather_name = correlated ?
            Symbol(group_site_name, :_by_, group_name, :_for_, location) :
            Symbol(:r_, location, :_, varying.id, :_, group_name)
        (; id=varying.id, group_name, prior,
           predictor_terms=varying.predictors,
           predictor_names=group_predictor_names, coefficient_keys,
           correlated, correlation_prior,
           scale_name=group_scale_name,
           correlation_name=group_correlation_name,
           site_name=group_site_name, gather_name,
           product_name=nothing, offset_name=gather_name)
    end

    scalar_location = isempty(predictor_names) &&
        isempty(sampled_offsets) && isempty(data_offsets) &&
        isempty(group_specs)
    intercept_prior = scalar_location ?
        BRM._native_ppl_intercept_normal_prior(brmi, location) :
        (; location=0.0, scale=1.0, operation=nothing)

    prior_scale = has_scale ?
        BRM._native_ppl_exponential_prior(brmi, scale_name) : nothing
    prior_degrees_freedom = is_student_t ?
        BRM._native_ppl_exponential_prior(brmi, degrees_freedom_name) : nothing
    expected_values = is_student_t ?
        Set((location, degrees_freedom_name, scale_name, response,
             input_predictor_names..., sampled_offsets..., group_names...)) :
        family === BRM.Normal ?
        Set((location, scale_name, response, input_predictor_names...,
             sampled_offsets..., group_names...)) :
        Set((location, response, input_predictor_names..., sampled_offsets...,
             group_names...))
    weight_spec === nothing || push!(expected_values, weight_spec.source)
    if evidence_spec !== nothing
        evidence_spec.lower.source === nothing ||
            push!(expected_values, evidence_spec.lower.source)
        evidence_spec.upper.source === nothing ||
            push!(expected_values, evidence_spec.upper.source)
    end
    for spec in group_specs
        push!(expected_values, spec.prior.operation)
        spec.correlation_prior === nothing ||
            push!(expected_values, spec.correlation_prior.operation)
    end
    intercept_prior.operation === nothing ||
        push!(expected_values, intercept_prior.operation)
    extras = setdiff(Set(keys(brmi.operations)), expected_values)
    isempty(extras) || throw(CapabilityError(
        :additional_operations,
        "unsupported formula operations: " *
        join(sort!(collect(extras)), ", ")))

    input_names = (
        input_predictor_names..., group_names..., weight_input_names...,
        evidence_input_names...)
    input_declarations = _declaration_namedtuple(
        input_names, map(_ -> input(), input_names))
    scalar_location && family !== BRM.Normal && !is_student_t && throw(
        CapabilityError(
            :likelihood,
            "the intercept-only BRM native-PPL slice supports Normal and " *
            "Student-t"))
    coefficient_name = scalar_location ? location : Symbol(:beta_, location)
    coefficient_keys = has_intercept ?
        (:Intercept, predictor_names...) : Tuple(predictor_names)
    coefficient_declaration = parameter(
        RealSupport(),
        scalar_location ? (location,) : coefficient_keys;
        transform=Identity(),
        prior=intercept_prior.operation === nothing ? StandardNormal() :
            normal_prior(intercept_prior.location, intercept_prior.scale))
    offset_declarations = map(
        name -> parameter(
            RealSupport(), (name,); transform=Identity(),
            prior=_brmi_literal_normal_prior(brmi, name)),
        sampled_offsets)
    group_scale_declarations = map(group_specs) do spec
        parameter(
            PositiveSupport(), spec.correlated ? spec.coefficient_keys :
                (spec.scale_name,);
            transform=Exp(), prior=Exponential(spec.prior.scale))
    end
    group_correlation_declarations = map(group_specs) do spec
        spec.correlated ? cholesky_correlation(
            spec.coefficient_keys, spec.correlation_prior.eta) : nothing
    end
    group_site_declarations = map(group_specs) do spec
        spec.correlated ? grouped_standard_normal(
            spec.group_name, spec.coefficient_keys) :
            grouped_normal(spec.group_name, 0.0, spec.scale_name)
    end
    group_parameter_names = foldl(
        (names, spec) -> spec.correlated ?
            (names..., spec.scale_name, spec.correlation_name,
             spec.site_name) :
            (names..., spec.scale_name, spec.site_name),
        group_specs; init=())
    group_parameter_declarations = foldl(
        (declarations, entry) -> begin
            spec, scale, correlation, site = entry
            spec.correlated ?
                (declarations..., scale, correlation, site) :
                (declarations..., scale, site)
        end,
        zip(group_specs, group_scale_declarations,
            group_correlation_declarations, group_site_declarations);
        init=())
    parameter_declarations = if is_student_t
        degrees_freedom_declaration = parameter(
            PositiveSupport(), (degrees_freedom_name,);
            transform=Exp(), prior=Exponential(prior_degrees_freedom))
        scale_declaration = parameter(
            PositiveSupport(), (scale_name,);
            transform=Exp(), prior=Exponential(prior_scale))
        _declaration_namedtuple(
            (coefficient_name, group_parameter_names...,
             degrees_freedom_name, scale_name, sampled_offsets...),
            (coefficient_declaration, group_parameter_declarations...,
             degrees_freedom_declaration, scale_declaration,
             offset_declarations...))
    elseif family === BRM.Normal
        scale_declaration = parameter(
            PositiveSupport(), (scale_name,);
            transform=Exp(), prior=Exponential(prior_scale))
        _declaration_namedtuple(
            (coefficient_name, group_parameter_names..., scale_name,
             sampled_offsets...),
            (coefficient_declaration, group_parameter_declarations...,
             scale_declaration, offset_declarations...))
    else
        _declaration_namedtuple(
            (coefficient_name, group_parameter_names..., sampled_offsets...),
            (coefficient_declaration, group_parameter_declarations...,
             offset_declarations...))
    end

    transform_names = Symbol[]
    transform_declarations = Any[]
    data_offset_names = map(data_offsets) do data_offset
        data_offset.transform === :identity && return data_offset.name
        transform_name = Symbol(
            :log_, data_offset.name, :_for_, location)
        push!(transform_names, transform_name)
        push!(transform_declarations, log_link(data_offset.name))
        transform_name
    end
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
    group_specs = map(group_specs) do spec
        predictor_values = map(
            spec.predictor_terms, spec.predictor_names) do predictor_term,
                                                           predictor_name
            predictor_term === nothing && return nothing
            predictor_term.transform === :identity && return predictor_name
            canonical = predictor_term.transform
            transform_name = Symbol(
                canonical, :_, predictor_name, :_for_, location)
            if transform_name ∉ transform_names
                transform_declaration = canonical === :center ?
                    center(predictor_name) : zscale(predictor_name)
                push!(transform_names, transform_name)
                push!(transform_declarations, transform_declaration)
            end
            transform_name
        end
        predictor_value = length(predictor_values) == 1 ?
            only(predictor_values) : nothing
        product_name = !spec.correlated && predictor_value !== nothing ?
            Symbol(spec.gather_name, :_times_, predictor_value) : nothing
        merge(spec, (; predictor_values, predictor_value, product_name,
                     offset_name=product_name === nothing ?
                         spec.gather_name : product_name))
    end
    gather_names = Tuple(spec.gather_name for spec in group_specs)
    gather_declarations = Tuple(
        spec.correlated ? grouped_affine(
            spec.site_name, spec.scale_name, spec.correlation_name,
            spec.group_name, spec.predictor_values) :
            group_gather(spec.site_name, spec.group_name)
        for spec in group_specs)
    product_specs = Tuple(
        spec for spec in group_specs if spec.product_name !== nothing)
    product_names = Tuple(spec.product_name for spec in product_specs)
    product_declarations = Tuple(
        row_product(spec.gather_name, spec.predictor_value)
        for spec in product_specs)
    affine_offsets = (
        sampled_offsets..., data_offset_names...,
        (spec.offset_name for spec in group_specs)...)
    node_names = scalar_location ? () : (
        Tuple(transform_names)..., gather_names..., product_names..., location)
    node_values = scalar_location ? () : (
        Tuple(transform_declarations)...,
        gather_declarations..., product_declarations...,
        affine(Tuple(affine_inputs), coefficient_name;
               offsets=affine_offsets, intercept=has_intercept))
    node_declarations = _declaration_namedtuple(node_names, node_values)

    observation_declaration = if is_student_t
        student_t(response, degrees_freedom_name, location, scale_name)
    elseif family === BRM.Normal
        normal(response, location, scale_name)
    elseif family === BRM.BernoulliLogit
        bernoulli_logit(response, location)
    else
        node_declarations = merge(
            node_declarations,
            NamedTuple{(rate_name,)}((exp_link(location),)))
        poisson(response, rate_name)
    end
    if weight_spec !== nothing
        evidence_spec === nothing || throw(CapabilityError(
            :response_evidence,
            "weighted response evidence is not yet lowered from BRM"))
        observation_declaration = weighted_observation(
            observation_declaration,
            observation_weight(weight_spec.kind, weight_spec.source))
    end
    if evidence_spec !== nothing
        evidence = if evidence_spec.kind === :truncated
            truncated_evidence(;
                lower=evidence_spec.lower.value,
                upper=evidence_spec.upper.value)
        elseif evidence_spec.kind === :censored
            censored_evidence(;
                lower=evidence_spec.lower.value,
                upper=evidence_spec.upper.value)
        else
            interval_evidence(evidence_spec.upper.value)
        end
        observation_declaration = evidence_observation(
            observation_declaration, evidence)
    end
    observation_declarations = NamedTuple{(response,)}(
        (broadcasted(observation_declaration),))
    declaration = if isempty(sampled_offsets) && isempty(group_specs)
        model(
            inputs=input_declarations,
            parameters=parameter_declarations,
            nodes=node_declarations,
            observations=observation_declarations)
    else
        group_site_order = foldl(
            (order, spec) -> spec.correlated ?
                (order..., spec.scale_name, spec.correlation_name,
                 spec.site_name) :
                (order..., spec.scale_name, spec.site_name),
            group_specs; init=())
        model(
            inputs=input_declarations,
            parameters=parameter_declarations,
            nodes=node_declarations,
            observations=observation_declarations,
            site_order=(sampled_offsets..., group_site_order...,
                        coefficient_name,
                        (is_student_t ?
                            (degrees_freedom_name, scale_name) :
                         family === BRM.Normal ? (scale_name,) : ())...,
                        response))
    end
    bindings = _declaration_namedtuple(
        input_names,
        (predictors..., groups..., weight_inputs..., evidence_inputs...))
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

prepare(brmi::BRM.BRMI; kwargs...) = prepare(compile(brmi); kwargs...)

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

function _syntax_symbol_tuple(expression, context::AbstractString)
    expression isa Expr && expression.head === :tuple || throw(ArgumentError(
        "native PPL @model $context requires a literal tuple of Symbol keys"))
    keys = map(expression.args) do entry
        entry isa QuoteNode && entry.value isa Symbol || throw(ArgumentError(
            "native PPL @model $context requires literal Symbol keys"))
        entry.value
    end
    isempty(keys) && throw(ArgumentError(
        "native PPL @model $context keys cannot be empty"))
    Tuple(keys)
end

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
        lhs isa Symbol ||
            (lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
             first(lhs.args) isa Symbol) || throw(ArgumentError(
            "native PPL @model Exponential parameter must have a bare name " *
            "or one literal axis tuple"))
        length(prior_arguments) == 1 || throw(ArgumentError(
            "native PPL @model Exponential(scale) needs one scale"))
        axis_keys = lhs isa Symbol ? (parameter_name,) :
            _syntax_symbol_tuple(lhs.args[2], "Exponential parameter")
        value = Expr(
            :call, _syntax_ref(:parameter),
            Expr(:parameters,
                 Expr(:kw, :transform, Expr(:call, _syntax_ref(:Exp))),
                 Expr(:kw, :prior,
                      Expr(:call, _syntax_ref(:Exponential),
                           only(prior_arguments)))),
            Expr(:call, _syntax_ref(:PositiveSupport)),
            QuoteNode(axis_keys))
        return parameter_name, value
    elseif prior_name === :LKJCholesky
        lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
            first(lhs.args) isa Symbol || throw(ArgumentError(
                "native PPL @model LKJCholesky parameter requires one " *
                "literal coefficient-axis tuple"))
        length(prior_arguments) == 2 || throw(ArgumentError(
            "native PPL @model LKJCholesky(K, eta) needs dimension and eta"))
        coefficient_keys = _syntax_symbol_tuple(
            lhs.args[2], "LKJCholesky parameter")
        dimension, eta = prior_arguments
        dimension isa Integer && dimension == length(coefficient_keys) ||
            throw(ArgumentError(
                "native PPL @model LKJCholesky dimension must match its " *
                "coefficient axis"))
        value = Expr(
            :call, _syntax_ref(:cholesky_correlation),
            QuoteNode(coefficient_keys), eta)
        return parameter_name, value
    end
    throw(ArgumentError(
        "native PPL @model unsupported parameter prior `$prior_name`"))
end

function _syntax_grouped_parameter(sampling,
                                   argument_names::Set{Symbol})
    sampling.broadcasted && return nothing
    lhs = sampling.lhs
    lhs isa Expr && lhs.head === :ref && length(lhs.args) in (2, 3) &&
        first(lhs.args) isa Symbol && lhs.args[2] isa Symbol || return nothing
    name, group = lhs.args[1:2]
    group in argument_names || throw(ArgumentError(
        "native PPL @model grouped site `$name` must index one function " *
        "argument; got `$group`"))
    family, arguments = _syntax_distribution_call(
        sampling.rhs, "grouped parameter prior")
    if family === :Normal
        length(lhs.args) == 2 || throw(ArgumentError(
            "native PPL @model scalar grouped Normal site `$name` takes only " *
            "its group index"))
        length(arguments) == 2 || throw(ArgumentError(
            "native PPL @model grouped site `$name` requires " *
            "Normal(location, scale)"))
        location, scale = arguments
        location isa Real || throw(ArgumentError(
            "native PPL @model grouped Normal location must be literal"))
        scale isa Union{Real,Symbol} || throw(ArgumentError(
            "native PPL @model grouped Normal scale must be literal or named"))
        value = Expr(
            :call, _syntax_ref(:grouped_normal), QuoteNode(group),
            location, scale isa Symbol ? QuoteNode(scale) : scale)
        return (; name, group, value, correlated=nothing)
    end
    family === :MvNormalCholesky && length(arguments) == 2 || throw(
        ArgumentError(
            "native PPL @model grouped vector site `$name` requires " *
            "MvNormalCholesky(scales, correlation)"))
    length(lhs.args) == 3 || throw(ArgumentError(
        "native PPL @model grouped MvNormalCholesky site `$name` requires " *
        "a literal coefficient axis after its group index"))
    scales, correlation = arguments
    scales isa Symbol && correlation isa Symbol || throw(ArgumentError(
        "native PPL @model grouped MvNormalCholesky arguments must name its " *
        "scale and correlation sites"))
    coefficients = _syntax_symbol_tuple(
        lhs.args[3], "grouped MvNormalCholesky site")
    value = Expr(
        :call, _syntax_ref(:grouped_standard_normal), QuoteNode(group),
        QuoteNode(coefficients))
    correlated = (; scales, correlation, coefficients)
    (; name, group, value, correlated)
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
    elseif function_name === :log || function_name === :log_link
        length(arguments) == 1 || throw(ArgumentError(
            "native PPL @model log needs one input"))
        :log_link
    elseif function_name === :logistic || function_name === :logistic_link
        length(arguments) == 1 || throw(ArgumentError(
            "native PPL @model logistic needs one input"))
        :logistic_link
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

function _syntax_group_gather_name(site::Symbol, group::Symbol,
                                   location::Symbol)
    spelling = String(site)
    suffix = string('_', group)
    if startswith(spelling, "b_") && endswith(spelling, suffix)
        id = chop(spelling; head=2, tail=length(suffix))
        isempty(id) || return Symbol(:r_, location, :_, id, :_, group)
    end
    Symbol(site, :_by_, group, :_for_, location)
end

function _syntax_affine_assignment(statement,
                                   scalar_priors::Set{Symbol},
                                   block_priors::Dict{Symbol,Tuple},
                                   grouped_sites::Dict{Symbol,Symbol},
                                   correlated_grouped_sites::Dict{Symbol,NamedTuple})
    lhs, rhs = statement.args
    lhs isa Symbol || return nothing
    terms = _syntax_affine_terms(rhs)
    if terms === nothing
        rhs isa Expr && rhs.head === :call &&
            _syntax_name(first(rhs.args)) === :dot || return nothing
        terms = Any[rhs]
    end
    sampled_offsets = Symbol[]
    data_offsets = Symbol[]
    offset_transform_names = Symbol[]
    offset_transform_values = Any[]
    gathered_offsets = NamedTuple[]
    grouped_products = NamedTuple[]
    correlated_groups = NamedTuple[]
    group_node_order = NamedTuple[]
    population_blocks = NamedTuple[]
    ordinary_terms = Any[]
    dot_transform_names = Symbol[]
    dot_transform_values = Any[]
    function dot_predictor(predictor, kind::AbstractString)
        predictor == 1 && return (; value=nothing, key=:Intercept)
        predictor isa Symbol && return (; value=predictor, key=predictor)
        predictor isa Expr && predictor.head === :call || throw(ArgumentError(
            "native PPL @model $kind predictors must be `1`, named row " *
            "values, or fitted center/zscale calls"))
        function_name, arguments = _syntax_call(
            predictor, "$kind predictor transform")
        function_name in (:center, :zscale, :standardize) || throw(
            ArgumentError(
                "native PPL @model $kind predictors support only " *
                "center(input) or zscale(input)"))
        length(arguments) == 1 && only(arguments) isa Symbol || throw(
            ArgumentError(
                "native PPL @model $function_name requires one named input"))
        raw_input = only(arguments)
        canonical = function_name === :center ? :center : :zscale
        transform_name = Symbol(canonical, :_, raw_input, :_for_, lhs)
        if transform_name ∉ dot_transform_names
            push!(dot_transform_names, transform_name)
            push!(dot_transform_values, Expr(
                :call, _syntax_ref(canonical), QuoteNode(raw_input)))
        end
        (; value=transform_name, key=raw_input)
    end
    for term in terms
        if term isa Expr && term.head === :call &&
           _syntax_name(first(term.args)) === :offset
            _, arguments = _syntax_call(term, "affine offset")
            length(arguments) == 1 || throw(ArgumentError(
                "native PPL @model offset needs one named scalar site or " *
                "row-valued input"))
            argument = only(arguments)
            if argument isa Symbol
                if argument in scalar_priors
                    push!(sampled_offsets, argument)
                else
                    push!(data_offsets, argument)
                end
            elseif argument isa Expr && argument.head === :call
                function_name, transform_arguments = _syntax_call(
                    argument, "affine data-offset transform")
                function_name in (:log, :log_link) || throw(ArgumentError(
                    "native PPL @model data offset supports only `log(input)`"))
                length(transform_arguments) == 1 &&
                    only(transform_arguments) isa Symbol || throw(
                    ArgumentError(
                        "native PPL @model log offset needs one named input"))
                raw_input = only(transform_arguments)
                transform_name = Symbol(:log_, raw_input, :_for_, lhs)
                push!(offset_transform_names, transform_name)
                push!(offset_transform_values, Expr(
                    :call, _syntax_ref(:log_link), QuoteNode(raw_input)))
                push!(data_offsets, transform_name)
            else
                throw(ArgumentError(
                    "native PPL @model offset needs one named scalar site or " *
                    "row-valued input"))
            end
        elseif term isa Expr && term.head === :ref &&
               length(term.args) == 2 && first(term.args) isa Symbol &&
               term.args[2] isa Symbol &&
               haskey(grouped_sites, first(term.args))
            site_name, group_name = term.args
            grouped_sites[site_name] === group_name || throw(ArgumentError(
                "native PPL @model grouped site `$site_name` must be " *
                "gathered with its declared group input"))
            gather_name = _syntax_group_gather_name(
                site_name, group_name, lhs)
            push!(gathered_offsets, (; site_name, group_name, gather_name))
            push!(group_node_order, (;
                kind=:gather, index=length(gathered_offsets)))
        elseif term isa Expr && term.head === :call &&
               first(term.args) in (:*, :.*) && length(term.args) == 3
            product_terms = term.args[2:end]
            grouped_indices = findall(product_terms) do product_term
                product_term isa Expr && product_term.head === :ref &&
                    length(product_term.args) == 2 &&
                    first(product_term.args) isa Symbol &&
                    product_term.args[2] isa Symbol &&
                    haskey(grouped_sites, first(product_term.args))
            end
            if isempty(grouped_indices)
                push!(ordinary_terms, term)
                continue
            end
            length(grouped_indices) == 1 || throw(ArgumentError(
                "native PPL @model row product must contain one grouped site"))
            grouped_index = only(grouped_indices)
            grouped_ref = product_terms[grouped_index]
            site_name, group_name = grouped_ref.args
            grouped_sites[site_name] === group_name || throw(ArgumentError(
                "native PPL @model grouped site `$site_name` must be " *
                "gathered with its declared group input"))
            predictor_name = product_terms[3 - grouped_index]
            predictor_name isa Symbol || throw(ArgumentError(
                "native PPL @model grouped slope must multiply its grouped " *
                "site by one named predictor"))
            gather_name = _syntax_group_gather_name(
                site_name, group_name, lhs)
            product_name = Symbol(gather_name, :_times_, predictor_name)
            push!(grouped_products, (;
                site_name, group_name, predictor_name, gather_name,
                product_name))
            push!(group_node_order, (;
                kind=:product, index=length(grouped_products)))
        elseif term isa Expr && term.head === :call &&
               _syntax_name(first(term.args)) === :dot
            _, arguments = _syntax_call(term, "affine dot product")
            length(arguments) == 2 || throw(ArgumentError(
                "native PPL @model dot needs a parameter or grouped site " *
                "and predictor tuple"))
            grouped_ref, predictor_tuple = arguments
            if grouped_ref isa Symbol && haskey(block_priors, grouped_ref)
                predictor_tuple isa Expr && predictor_tuple.head === :tuple ||
                    throw(ArgumentError(
                        "native PPL @model parameter dot predictors must " *
                        "be a tuple"))
                parsed_predictors = Tuple(map(
                    predictor -> dot_predictor(predictor, "parameter dot"),
                    predictor_tuple.args))
                predictors = Tuple(parsed.value for parsed in parsed_predictors)
                count(isnothing, predictors) <= 1 || throw(ArgumentError(
                    "native PPL @model parameter dot may contain one intercept"))
                any(isnothing, predictors) && first(predictors) !== nothing &&
                    throw(ArgumentError(
                        "native PPL @model parameter-dot intercept must be first"))
                coefficient_keys = Tuple(
                    parsed.key for parsed in parsed_predictors)
                coefficient_keys == block_priors[grouped_ref] || throw(
                    ArgumentError(
                        "native PPL @model parameter dot predictors must " *
                        "match its declared coefficient keys"))
                push!(population_blocks, (;
                    name=grouped_ref, predictors,
                    raw_predictors=coefficient_keys))
                continue
            end
            grouped_ref isa Expr && grouped_ref.head === :ref &&
                length(grouped_ref.args) in (2, 3) &&
                first(grouped_ref.args) isa Symbol &&
                grouped_ref.args[2] isa Symbol || throw(ArgumentError(
                    "native PPL @model grouped dot must start with " *
                    "`site[group]` or `site[group, coefficient_keys]`"))
            site_name, group_name = grouped_ref.args[1:2]
            haskey(correlated_grouped_sites, site_name) || throw(
                ArgumentError(
                    "native PPL @model grouped dot site `$site_name` must " *
                    "have an MvNormalCholesky declaration"))
            metadata = correlated_grouped_sites[site_name]
            metadata.group === group_name || throw(ArgumentError(
                "native PPL @model grouped dot must use the site's declared " *
                "group input"))
            predictor_tuple isa Expr && predictor_tuple.head === :tuple ||
                throw(ArgumentError(
                    "native PPL @model grouped dot predictors must be a tuple"))
            parsed_predictors = Tuple(map(
                predictor -> dot_predictor(predictor, "grouped dot"),
                predictor_tuple.args))
            predictors = Tuple(parsed.value for parsed in parsed_predictors)
            selected_coefficients = length(grouped_ref.args) == 2 ?
                metadata.coefficients : _syntax_symbol_tuple(
                    grouped_ref.args[3], "grouped dot coefficient selection")
            isempty(selected_coefficients) && throw(ArgumentError(
                "native PPL @model grouped dot coefficient selection cannot " *
                "be empty"))
            length(unique(selected_coefficients)) ==
                length(selected_coefficients) || throw(ArgumentError(
                    "native PPL @model grouped dot coefficient selection " *
                    "must be unique"))
            all(coefficient -> coefficient in metadata.coefficients,
                selected_coefficients) || throw(ArgumentError(
                    "native PPL @model grouped dot selects an unknown group " *
                    "coefficient"))
            coefficient_indices = Tuple(
                something(findfirst(==(coefficient), metadata.coefficients))
                for coefficient in selected_coefficients)
            length(predictors) == length(selected_coefficients) || throw(
                ArgumentError(
                    "native PPL @model grouped dot predictor count must " *
                    "match its selected coefficient axis"))
            coefficient_keys = Tuple(parsed.key for parsed in parsed_predictors)
            length(grouped_ref.args) == 2 &&
                coefficient_keys != metadata.coefficients && throw(ArgumentError(
                "native PPL @model grouped dot predictors must match its " *
                "declared coefficient keys"))
            node_name = Symbol(site_name, :_by_, group_name, :_for_, lhs)
            push!(correlated_groups, (;
                site_name, group_name, predictors, node_name,
                scales=metadata.scales, correlation=metadata.correlation,
                coefficient_indices,
                coefficient_count=length(metadata.coefficients)))
            push!(group_node_order, (;
                kind=:correlated, index=length(correlated_groups)))
        else
            push!(ordinary_terms, term)
        end
    end
    length(unique(sampled_offsets)) == length(sampled_offsets) || throw(
        ArgumentError(
            "native PPL @model affine offsets must be used once each"))
    length(unique(data_offsets)) == length(data_offsets) || throw(
        ArgumentError(
            "native PPL @model data offsets must be used once each"))
    isempty(intersect(Set(sampled_offsets), Set(data_offsets))) || throw(
        ArgumentError(
            "native PPL @model sampled and data offsets must be distinct"))
    length(population_blocks) <= 1 || throw(ArgumentError(
        "native PPL @model affine expression may use one parameter dot"))
    intercepts = [term for term in ordinary_terms
                  if term isa Symbol && term in scalar_priors]
    length(intercepts) <= 1 || return nothing
    intercept = isempty(intercepts) ? nothing : only(intercepts)
    intercept === nothing && isempty(sampled_offsets) &&
        isempty(data_offsets) &&
        isempty(gathered_offsets) && isempty(grouped_products) &&
        isempty(correlated_groups) && isempty(population_blocks) &&
        return nothing

    slopes = Symbol[]
    predictor_names = Symbol[]
    raw_predictor_names = Symbol[]
    transform_names = [offset_transform_names...; dot_transform_names...]
    transform_values = [offset_transform_values...; dot_transform_values...]
    population_block = isempty(population_blocks) ? nothing :
        only(population_blocks)
    if population_block !== nothing
        isempty(ordinary_terms) || throw(ArgumentError(
            "native PPL @model cannot mix a parameter dot with scalar " *
            "population coefficients"))
        for (predictor, raw_predictor) in zip(
                population_block.predictors,
                population_block.raw_predictors)
            predictor === nothing && continue
            push!(predictor_names, predictor)
            push!(raw_predictor_names, raw_predictor)
        end
    end
    for product in ordinary_terms
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
            if transform_name ∉ transform_names
                push!(transform_names, transform_name)
                push!(transform_values, Expr(
                    :call, _syntax_ref(canonical), QuoteNode(raw_input)))
            end
            transform_name
        else
            return nothing
        end
        push!(slopes, slope)
        push!(predictor_names, predictor_name)
        push!(raw_predictor_names, raw_predictor_name)
    end
    isempty(slopes) && population_block === nothing && return nothing
    length(unique(slopes)) == length(slopes) || throw(ArgumentError(
        "native PPL @model affine coefficients must be used once each"))
    length(unique(predictor_names)) == length(predictor_names) ||
        throw(ArgumentError(
            "native PPL @model affine predictor paths must be unique"))
    length(unique(raw_predictor_names)) == length(raw_predictor_names) ||
        throw(ArgumentError(
            "native PPL @model affine features must use distinct raw inputs"))

    coefficient_name = population_block === nothing ?
        Symbol(:beta_, lhs) : population_block.name
    has_intercept = intercept !== nothing ||
        (population_block !== nothing &&
         any(isnothing, population_block.predictors))
    parameter_value = if population_block === nothing
        coefficient_keys = intercept === nothing ?
            Tuple(slopes) : (intercept, slopes...)
        Expr(
            :call, _syntax_ref(:parameter),
            Expr(:parameters,
                 Expr(:kw, :transform, Expr(:call, _syntax_ref(:Identity))),
                 Expr(:kw, :prior, Expr(:call, _syntax_ref(:StandardNormal)))),
            Expr(:call, _syntax_ref(:RealSupport)),
            QuoteNode(coefficient_keys))
    else
        nothing
    end
    affine_value = Expr(
        :call, _syntax_ref(:affine),
        Expr(:parameters,
             Expr(:kw, :offsets, QuoteNode((
                 sampled_offsets...,
                 data_offsets...,
                 (entry.kind === :gather ?
                    gathered_offsets[entry.index].gather_name :
                  entry.kind === :product ?
                    grouped_products[entry.index].product_name :
                    correlated_groups[entry.index].node_name
                  for entry in group_node_order)...))),
             Expr(:kw, :intercept, has_intercept)),
        QuoteNode(Tuple(predictor_names)),
        QuoteNode(coefficient_name))
    grouped_nodes = (gathered_offsets..., grouped_products...)
    gather_names = Tuple(grouped.gather_name for grouped in grouped_nodes)
    length(unique(gather_names)) == length(gather_names) || throw(ArgumentError(
        "native PPL @model grouped terms must use distinct latent sites"))
    gather_values = Tuple(Expr(
        :call, _syntax_ref(:group_gather),
        QuoteNode(gather.site_name), QuoteNode(gather.group_name))
        for gather in grouped_nodes)
    product_names = Tuple(product.product_name for product in grouped_products)
    product_values = Tuple(Expr(
        :call, _syntax_ref(:row_product),
        QuoteNode(product.gather_name), QuoteNode(product.predictor_name))
        for product in grouped_products)
    correlated_names = Tuple(group.node_name for group in correlated_groups)
    correlated_values = Tuple(Expr(
        :call, _syntax_ref(:grouped_affine),
        Expr(:parameters,
             Expr(:kw, :coefficient_indices,
                  QuoteNode(group.coefficient_indices)),
             Expr(:kw, :coefficient_count, group.coefficient_count)),
        QuoteNode(group.site_name), QuoteNode(group.scales),
        QuoteNode(group.correlation), QuoteNode(group.group_name),
        QuoteNode(group.predictors)) for group in correlated_groups)
    group_names = Symbol[]
    group_values = Any[]
    for entry in group_node_order
        if entry.kind === :gather
            push!(group_names, gather_names[entry.index])
            push!(group_values, gather_values[entry.index])
        elseif entry.kind === :product
            push!(group_names, gather_names[
                length(gathered_offsets) + entry.index])
            push!(group_values, gather_values[
                length(gathered_offsets) + entry.index])
            push!(group_names, product_names[entry.index])
            push!(group_values, product_values[entry.index])
        else
            push!(group_names, correlated_names[entry.index])
            push!(group_values, correlated_values[entry.index])
        end
    end
    (; location=lhs, intercept, offsets=Tuple(sampled_offsets),
       gather_names, gather_values, product_names, product_values,
       correlated_names, correlated_values,
       group_names=Tuple(group_names), group_values=Tuple(group_values),
       slopes=Tuple(slopes), coefficient_name,
       parameter_value, transform_names=Tuple(transform_names),
       transform_values=Tuple(transform_values), affine_value)
end

function _syntax_observation_weight(rhs)
    family, arguments = _syntax_distribution_call(
        rhs, "observation family")
    family === :weighted || return nothing
    length(arguments) == 2 || throw(ArgumentError(
        "native PPL @model weighted(distribution, weight) needs two arguments"))
    distribution, weight = arguments
    constructor, weight_arguments = _syntax_distribution_call(
        weight, "observation-weight constructor")
    kind = constructor === :aweights ? :analytic :
        constructor === :fweights ? :frequency :
        constructor === :weights ? :power :
        constructor === :uweights ? :unit : nothing
    constructor === :pweights && throw(ArgumentError(
        "native PPL ProbabilityWeights semantics are not implemented"))
    kind === nothing && throw(ArgumentError(
        "native PPL @model unsupported observation-weight constructor " *
        "`$constructor`; use aweights, fweights, weights, or uweights"))
    length(weight_arguments) == 1 && only(weight_arguments) isa Symbol ||
        throw(ArgumentError(
            "native PPL @model observation weights require one named input"))
    (; distribution, kind, source=only(weight_arguments))
end

function _syntax_call_parts(expression, context::AbstractString)
    expression isa Expr || throw(ArgumentError(
        "native PPL @model $context must be a call; got `$expression`"))
    arguments = if expression.head === :call
        expression.args[2:end]
    elseif expression.head === :. && length(expression.args) == 2 &&
           expression.args[2] isa Expr && expression.args[2].head === :tuple
        expression.args[2].args
    else
        throw(ArgumentError(
            "native PPL @model $context must be a call; got `$expression`"))
    end
    name = _syntax_name(first(expression.args))
    name === nothing && throw(ArgumentError(
        "native PPL @model cannot identify the function in `$expression`"))
    positional = Any[]
    keywords = Dict{Symbol,Any}()
    for argument in arguments
        if argument isa Expr && argument.head === :parameters
            for keyword in argument.args
                keyword isa Expr && keyword.head === :kw &&
                    length(keyword.args) == 2 &&
                    first(keyword.args) isa Symbol || throw(ArgumentError(
                        "native PPL @model $context has malformed keywords"))
                key, value = keyword.args
                haskey(keywords, key) && throw(ArgumentError(
                    "native PPL @model $context specifies `$key` twice"))
                keywords[key] = value
            end
        else
            push!(positional, argument)
        end
    end
    (; name, positional, keywords)
end

function _syntax_evidence_bound(bound, context::AbstractString)
    bound === :nothing && return (; value=nothing, dependency=nothing)
    bound isa Real && isfinite(bound) || bound isa Symbol || throw(
        ArgumentError(
            "native PPL @model $context must be a finite literal, named " *
            "value, or `nothing`"))
    bound isa Real && !isfinite(bound) && throw(ArgumentError(
        "native PPL @model $context must be finite"))
    bound isa Symbol ? (; value=QuoteNode(bound), dependency=bound) :
        (; value=bound, dependency=nothing)
end

function _syntax_observation_evidence(rhs)
    parts = _syntax_call_parts(rhs, "observation family")
    kind = parts.name === :truncated ? :truncated :
        parts.name === :censored ? :censored :
        parts.name === :interval_censored ? :interval_censored : nothing
    kind === nothing && return nothing
    isempty(parts.positional) && throw(ArgumentError(
        "native PPL @model $(parts.name) needs a base distribution"))
    distribution = first(parts.positional)
    positional = parts.positional[2:end]
    allowed = kind === :interval_censored ? (:upper,) : (:lower, :upper)
    unknown = Tuple(key for key in keys(parts.keywords) if key ∉ allowed)
    isempty(unknown) || throw(ArgumentError(
        "native PPL @model $(parts.name) has unsupported keywords $unknown"))
    lower = nothing
    upper = nothing
    if kind === :interval_censored
        isempty(positional) || throw(ArgumentError(
            "native PPL @model interval_censored accepts `upper` only as a keyword"))
        haskey(parts.keywords, :upper) || throw(ArgumentError(
            "native PPL @model interval_censored requires `upper=`"))
        upper = parts.keywords[:upper]
    else
        length(positional) <= 2 || throw(ArgumentError(
            "native PPL @model $(parts.name) accepts at most lower and upper bounds"))
        !isempty(positional) && (lower = positional[1])
        length(positional) == 2 && (upper = positional[2])
        if haskey(parts.keywords, :lower)
            isempty(positional) || throw(ArgumentError(
                "native PPL @model $(parts.name) specifies `lower` twice"))
            lower = parts.keywords[:lower]
        end
        if haskey(parts.keywords, :upper)
            length(positional) < 2 || throw(ArgumentError(
                "native PPL @model $(parts.name) specifies `upper` twice"))
            upper = parts.keywords[:upper]
        end
        lower === nothing && upper === nothing && throw(ArgumentError(
            "native PPL @model $(parts.name) needs at least one bound"))
    end
    lower = _syntax_evidence_bound(lower === nothing ? :nothing : lower,
                                   "$(parts.name) lower bound")
    upper = _syntax_evidence_bound(upper === nothing ? :nothing : upper,
                                   "$(parts.name) upper bound")
    if lower.value isa Real && upper.value isa Real &&
       lower.value >= upper.value
        throw(ArgumentError(
            "native PPL @model $(parts.name) lower bound must be below its upper bound"))
    end
    (; kind, distribution, lower, upper)
end

function _syntax_observation(lhs, rhs; broadcasted::Bool)
    lhs isa Symbol || throw(ArgumentError(
        "native PPL @model stochastic-site left-hand side must be a bare " *
        "name; got `$lhs`"))
    weight = _syntax_observation_weight(rhs)
    weight === nothing || (rhs = weight.distribution)
    evidence = _syntax_observation_evidence(rhs)
    evidence === nothing || (rhs = evidence.distribution)
    family, arguments = _syntax_distribution_call(rhs, "observation family")
    extra_node_names = Symbol[]
    extra_node_values = Any[]
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
        push!(extra_node_names, extra_node_name)
        push!(extra_node_values, Expr(
            :call, _syntax_ref(:exp_link), QuoteNode(log_rate))
        )
        arguments = Any[extra_node_name]
    elseif family === :ZeroInflatedPoisson && length(arguments) == 2
        linked = Any[]
        for (argument, expected_link, prefix, builder) in zip(
                arguments, (:exp, :logistic), (:exp_, :logistic_),
                (:exp_link, :logistic_link))
            if argument isa Symbol
                push!(linked, argument)
                continue
            end
            argument isa Expr && argument.head in (:call, :.) || throw(
                ArgumentError(
                    "native PPL @model ZeroInflatedPoisson accepts named " *
                    "parameters or exp(named_log_rate), " *
                    "logistic(named_logit)"))
            link_name, link_arguments = _syntax_distribution_call(
                argument, "ZeroInflatedPoisson parameter link")
            link_name === expected_link && length(link_arguments) == 1 &&
                only(link_arguments) isa Symbol || throw(ArgumentError(
                    "native PPL @model ZeroInflatedPoisson expects " *
                    "exp(named_log_rate) and logistic(named_logit)"))
            input = only(link_arguments)
            name = Symbol(prefix, input)
            push!(extra_node_names, name)
            push!(extra_node_values, Expr(
                :call, _syntax_ref(builder), QuoteNode(input)))
            push!(linked, name)
        end
        arguments = linked
    elseif family === :NegativeBinomial2 && length(arguments) == 2
        linked = Any[]
        for argument in arguments
            if argument isa Symbol
                push!(linked, argument)
                continue
            end
            argument isa Expr && argument.head in (:call, :.) || throw(
                ArgumentError(
                    "native PPL @model NegativeBinomial2 accepts named " *
                    "parameters or exp(named_log_parameter)"))
            link_name, link_arguments = _syntax_distribution_call(
                argument, "NegativeBinomial2 parameter link")
            link_name === :exp && length(link_arguments) == 1 &&
                only(link_arguments) isa Symbol || throw(ArgumentError(
                    "native PPL @model NegativeBinomial2 expects " *
                    "exp(named_log_mean) and exp(named_log_precision)"))
            input = only(link_arguments)
            name = Symbol(:exp_, input)
            push!(extra_node_names, name)
            push!(extra_node_values, Expr(
                :call, _syntax_ref(:exp_link), QuoteNode(input)))
            push!(linked, name)
        end
        arguments = linked
    elseif family === :BetaBinomial2 && length(arguments) == 3
        trials = first(arguments)
        trials isa Symbol || throw(ArgumentError(
            "native PPL @model BetaBinomial2 trials must be a named input"))
        linked = Any[trials]
        for (argument, expected_link, prefix, builder) in zip(
                arguments[2:3], (:logistic, :exp),
                (:logistic_, :exp_), (:logistic_link, :exp_link))
            if argument isa Symbol
                push!(linked, argument)
                continue
            end
            argument isa Expr && argument.head in (:call, :.) || throw(
                ArgumentError(
                    "native PPL @model BetaBinomial2 accepts named " *
                    "parameters or linked mean/precision parameters"))
            link_name, link_arguments = _syntax_distribution_call(
                argument, "BetaBinomial2 parameter link")
            link_name === expected_link && length(link_arguments) == 1 &&
                only(link_arguments) isa Symbol || throw(ArgumentError(
                    "native PPL @model BetaBinomial2 expects " *
                    "logistic(named_logit_mean) and " *
                    "exp(named_log_precision)"))
            input = only(link_arguments)
            name = Symbol(prefix, input)
            push!(extra_node_names, name)
            push!(extra_node_values, Expr(
                :call, _syntax_ref(builder), QuoteNode(input)))
            push!(linked, name)
        end
        arguments = linked
    elseif family === :CategoricalLogit
        isempty(arguments) && throw(ArgumentError(
            "native PPL @model CategoricalLogit needs at least one " *
            "non-reference predictor"))
        all(argument -> argument isa Symbol, arguments) || throw(
            ArgumentError(
                "native PPL @model CategoricalLogit arguments must name " *
                "scalar predictors"))
        length(unique(arguments)) == length(arguments) || throw(ArgumentError(
            "native PPL @model CategoricalLogit predictors must be distinct"))
    elseif family === :LocationScale
        length(arguments) == 3 || throw(ArgumentError(
            "native PPL @model LocationScale Student-t needs location, " *
            "scale, and TDist(degrees_freedom)"))
        location, scale, base = arguments
        location isa Symbol && scale isa Symbol || throw(ArgumentError(
            "native PPL @model LocationScale location and scale must be " *
            "named graph values"))
        base_family, base_arguments = _syntax_distribution_call(
            base, "LocationScale base distribution")
        base_family === :TDist && length(base_arguments) == 1 &&
            only(base_arguments) isa Symbol || throw(ArgumentError(
                "native PPL @model currently supports " *
                "LocationScale(location, scale, TDist(named_df))"))
        arguments = Any[only(base_arguments), location, scale]
        family = :StudentT
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
    elseif family === :ZeroInflatedPoisson
        (:zero_inflated_poisson, 2)
    elseif family === :NegativeBinomial2
        (:negative_binomial2, 2)
    elseif family === :BetaBinomial2
        (:beta_binomial2, 3)
    elseif family === :BinomialLogit
        (:binomial_logit, 2)
    elseif family === :CategoricalLogit
        (:categorical_logit, length(arguments))
    elseif family === :StudentT
        (:student_t, 3)
    else
        throw(ArgumentError(
            "native PPL @model unsupported observation family `$family`"))
    end
    length(arguments) == expected || throw(ArgumentError(
        "native PPL @model $family observation needs $expected parameter(s)"))
    scalar_value = Expr(:call, _syntax_ref(builder), QuoteNode(lhs),
                        (QuoteNode(argument) for argument in arguments)...)
    if evidence !== nothing
        evidence_value = if evidence.kind === :truncated
            Expr(
                :call, _syntax_ref(:truncated_evidence),
                Expr(:parameters,
                     Expr(:kw, :lower, evidence.lower.value),
                     Expr(:kw, :upper, evidence.upper.value)))
        elseif evidence.kind === :censored
            Expr(
                :call, _syntax_ref(:censored_evidence),
                Expr(:parameters,
                     Expr(:kw, :lower, evidence.lower.value),
                     Expr(:kw, :upper, evidence.upper.value)))
        else
            Expr(:call, _syntax_ref(:interval_evidence), evidence.upper.value)
        end
        scalar_value = Expr(
            :call, _syntax_ref(:evidence_observation), scalar_value,
            evidence_value)
    end
    if weight !== nothing
        weight_value = Expr(
            :call, _syntax_ref(:observation_weight), QuoteNode(weight.kind),
            QuoteNode(weight.source))
        scalar_value = Expr(
            :call, _syntax_ref(:weighted_observation), scalar_value,
            weight_value)
    end
    value = broadcasted ?
        Expr(:call, _syntax_ref(:broadcasted), scalar_value) : scalar_value
    evidence_dependencies = evidence === nothing ? () : Tuple(
        dependency for dependency in
        (evidence.lower.dependency, evidence.upper.dependency)
        if dependency !== nothing)
    (; name=lhs, value,
       extra_node_names=Tuple(extra_node_names),
       extra_node_values=Tuple(extra_node_values),
       dependencies=(Tuple(arguments)..., evidence_dependencies...,
                     (weight === nothing ? () : (weight.source,))...))
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
        all(entry -> entry isa Symbol, entries) || throw(ArgumentError(
            "native PPL @model named return shorthand requires bare names"))
        names = Tuple(entries)
        return names, names
    end

    aliases = Symbol[]
    names = Symbol[]
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
    (:Normal, :StandardNormal, :Exponential, :LKJCholesky,
     :MvNormalCholesky, :BernoulliLogit, :Poisson, :LocationScale,
     :ZeroInflatedPoisson, :NegativeBinomial2, :BetaBinomial2,
     :BinomialLogit, :CategoricalLogit, :weighted,
     :truncated, :censored, :interval_censored)
const _SYNTAX_DETERMINISTIC_NAMES =
    (:center, :zscale, :standardize, :affine,
     :exp, :exp_link, :logistic, :logistic_link, :dot)
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
    affine_coefficient_names = Symbol[]
    scalar_prior_names = Symbol[]
    block_prior_axes = Dict{Symbol,Tuple}()
    grouped_sites = Dict{Symbol,Symbol}()
    correlated_grouped_sites = Dict{Symbol,NamedTuple}()
    consumed_scalar_priors = Set{Symbol}()
    node_names = Symbol[]
    node_values = Any[]
    observation_names = Symbol[]
    observation_values = Any[]
    factor_dependency_names = Set{Symbol}()
    source_site_names = Symbol[]
    packed_site_names = Dict{Symbol,Symbol}()
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
            source_site_name = sampling.lhs isa Symbol ? sampling.lhs :
                (sampling.lhs isa Expr && sampling.lhs.head === :ref &&
                 first(sampling.lhs.args) isa Symbol ?
                 first(sampling.lhs.args) : nothing)
            source_site_name === nothing || push!(
                source_site_names, source_site_name)
            grouped_parameter = _syntax_grouped_parameter(
                sampling, argument_name_set)
            if grouped_parameter !== nothing
                haskey(grouped_sites, grouped_parameter.name) && throw(
                    ArgumentError(
                        "native PPL @model grouped site " *
                        "`$(grouped_parameter.name)` is declared twice"))
                push!(parameter_names, grouped_parameter.name)
                push!(parameter_values, grouped_parameter.value)
                grouped_sites[grouped_parameter.name] = grouped_parameter.group
                if grouped_parameter.correlated !== nothing
                    correlated_grouped_sites[grouped_parameter.name] = merge(
                        (; group=grouped_parameter.group),
                        grouped_parameter.correlated)
                end
                continue
            end
            scalar_prior = sampling.broadcasted ? nothing :
                _syntax_standard_normal(normalized)
            prior_name = _syntax_name(
                sampling.rhs isa Expr ? first(sampling.rhs.args) : sampling.rhs)
            is_explicit_parameter = !sampling.broadcasted &&
                prior_name in (:StandardNormal, :Exponential, :LKJCholesky)
            if scalar_prior === nothing && !is_explicit_parameter
                observation = _syntax_observation(
                    sampling.lhs, sampling.rhs;
                    broadcasted=sampling.broadcasted)
                for index in eachindex(
                        observation.extra_node_names,
                        observation.extra_node_values)
                    push!(node_names, observation.extra_node_names[index])
                    push!(node_values, observation.extra_node_values[index])
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
                    if prior_name === :StandardNormal
                        block_prior_axes[name] = _syntax_symbol_tuple(
                            sampling.lhs.args[2],
                            "StandardNormal parameter")
                    end
                else
                    scalar_prior in scalar_prior_names && throw(ArgumentError(
                        "native PPL @model parameter `$scalar_prior` is declared twice"))
                    push!(scalar_prior_names, scalar_prior)
                end
            end
        elseif statement isa Expr && statement.head === :(=)
            affine_declaration = _syntax_affine_assignment(
                statement, Set(scalar_prior_names), block_prior_axes,
                grouped_sites,
                correlated_grouped_sites)
            if affine_declaration === nothing
                name, value = _syntax_node(statement)
                push!(node_names, name)
                push!(node_values, value)
            else
                # Coefficient blocks precede support-transformed scalar
                # parameters in the flat coordinate ABI, independent of where
                # the deterministic affine assignment appears in source.
                if affine_declaration.parameter_value !== nothing
                    push!(
                        parameter_names, affine_declaration.coefficient_name)
                    push!(
                        parameter_values, affine_declaration.parameter_value)
                else
                    coefficient_index = findfirst(
                        ==(affine_declaration.coefficient_name),
                        parameter_names)
                    coefficient_index === nothing && throw(ArgumentError(
                        "native PPL @model affine dot references an " *
                        "unavailable coefficient block"))
                end
                affine_declaration.coefficient_name in
                    affine_coefficient_names || push!(
                        affine_coefficient_names,
                        affine_declaration.coefficient_name)
                for (transform_name, transform_value) in zip(
                    affine_declaration.transform_names,
                    affine_declaration.transform_values)
                    push!(node_names, transform_name)
                    push!(node_values, transform_value)
                end
                for (group_name, group_value) in zip(
                    affine_declaration.group_names,
                    affine_declaration.group_values)
                    push!(node_names, group_name)
                    push!(node_values, group_value)
                end
                push!(node_names, affine_declaration.location)
                push!(node_values, affine_declaration.affine_value)
                if affine_declaration.intercept !== nothing
                    packed_site_names[affine_declaration.intercept] =
                        affine_declaration.coefficient_name
                    push!(consumed_scalar_priors,
                          affine_declaration.intercept)
                end
                for slope in affine_declaration.slopes
                    packed_site_names[slope] =
                        affine_declaration.coefficient_name
                end
                union!(consumed_scalar_priors, affine_declaration.slopes)
                for offset_name in affine_declaration.offsets
                    push!(parameter_names, offset_name)
                    push!(parameter_values,
                          _syntax_scalar_standard_normal_parameter(offset_name))
                    push!(consumed_scalar_priors, offset_name)
                end
            end
        else
            throw(ArgumentError(
                "native PPL @model unsupported statement `$statement`"))
        end
    end
    if !isempty(affine_coefficient_names)
        non_affine_parameter_names = filter(
            name -> name ∉ affine_coefficient_names, parameter_names)
        ordered_parameter_names = (
            affine_coefficient_names..., non_affine_parameter_names...)
        parameter_lookup = Dict(
            name => value for (name, value) in
            zip(parameter_names, parameter_values))
        parameter_names = collect(ordered_parameter_names)
        parameter_values = Any[
            parameter_lookup[name] for name in ordered_parameter_names
        ]
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
    site_names = Set((parameter_names..., observation_names...))
    source_site_order = Symbol[]
    for source_name in source_site_names
        canonical_name = get(packed_site_names, source_name, source_name)
        canonical_name in site_names || continue
        canonical_name in source_site_order || push!(
            source_site_order, canonical_name)
    end
    for name in (parameter_names..., observation_names...)
        name in source_site_order || push!(source_site_order, name)
    end
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
        Expr(:kw, :site_order, QuoteNode(Tuple(source_site_order))),
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
generates the explicit namespaced graph machinery. Broadcast observations may
retain `truncated`, `censored`, or `interval_censored` evidence semantics;
these are typed factor decorators rather than backend-specific syntax.
"""
macro model(definition)
    esc(_model_function_syntax(definition))
end

export Model, ModelInstance, GraphRef, Component, Composition, Input, Parameter
export AbstractParameterDeclaration, GroupedNormalParameter
export GroupedStandardNormalParameter, CholeskyCorrelationParameter
export SiteValue, InputValue, NodeValue, LiteralValue, StochasticSite
export SiteCoordinates, GroupCoordinateKey, GroupCoefficientKey
export CorrelationCoordinateKey
export FactorGraph
export CenterFactorNode, ZScaleFactorNode, AffineFactorNode, ExpFactorNode
export FittedCenter, FittedZScale
export LogFactorNode, LogisticFactorNode
export GroupGatherFactorNode, RowProductFactorNode, GroupedAffineFactorNode
export FactorPlan, FactorPrepared, FactorWorkspace, FactorLogDensityProblem
export StandardNormalSiteFactor, NormalSiteFactor, StudentTSiteFactor,
       ExponentialSiteFactor
export LKJCholeskySiteFactor
export BernoulliLogitSiteFactor, PoissonSiteFactor
export ZeroInflatedPoissonSiteFactor, NegativeBinomial2SiteFactor
export BetaBinomial2SiteFactor
export BinomialLogitSiteFactor
export CategoricalLogitSiteFactor
export WeightedSiteFactor
export EvidenceSiteFactor
export ScalarSiteShape, BlockSiteShape, BroadcastSiteShape
export FreeSite, ConditionedSite, GeneratedSite
export RealSupport, PositiveSupport, CholeskyCorrelationSupport
export IdentityTransform, ExpTransform, CholeskyCorrelationTransform
export StandardNormal, NormalPrior, ExponentialPrior
export Center, ZScale, Affine, ExpLink, LogLink, LogisticLink
export GroupGather, RowProduct, GroupedAffine
export NormalObservation, StudentTObservation, BernoulliLogitObservation,
       PoissonObservation
export ZeroInflatedPoissonObservation, NegativeBinomial2Observation
export BetaBinomial2Observation
export BinomialLogitObservation
export CategoricalLogitObservation
export AbstractObservationWeight, ObservationWeight
export AbstractObservationEvidence, TruncatedEvidence, CensoredEvidence
export IntervalEvidence, EvidenceObservation
export WeightedObservation, BroadcastObservation
export model, input, parameter, Identity, Exp, normal_prior, Exponential
export center, zscale, standardize, affine, exp_link, log_link, logistic_link
export grouped_normal, group_gather, group_values, group_input
export grouped_standard_normal, group_coefficients
export cholesky_correlation, correlation_coefficients
export row_product, row_product_inputs
export grouped_affine, grouped_standardized, grouped_scales
export grouped_correlation, grouped_predictors
export grouped_coefficient_indices, grouped_coefficient_count
export normal, student_t, bernoulli_logit, poisson, zero_inflated_poisson
export negative_binomial2
export beta_binomial2
export binomial_logit
export categorical_logit
export observation_weight
export observation_weight_kind, observation_weight_source
export weighted_observation, truncated_evidence, censored_evidence
export interval_evidence, evidence_kind, evidence_lower, evidence_upper
export evidence_observation, broadcasted
export instantiate, substitute, condition, component, output, compose, bind, lower
export factor_graph, factor_output_site, factor_output_sites
export site_factor_dependencies, factor_node_dependencies
export site_value_name, input_value_name, node_value_name
export graph_namespace, graph_name, graph_kind, component_namespace, qualified_name
export @model
