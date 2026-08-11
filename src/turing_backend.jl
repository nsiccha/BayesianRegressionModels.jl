# Core Turing-facing types and semantic validation. This file deliberately has
# no dependency on Turing, DynamicPPL, StanBlocks, SBBRMI, or emitted SLIC. The
# package extension supplies the executable model after Turing is loaded.

struct _TuringPopulationPlan{F,C<:_BRMBackendContext,D<:_BRMPopulationDesign,
                             Y<:AbstractVector,B<:AbstractVector,S}
    family::F
    context::C
    design::D
    response::Y
    beta_location::B
    beta_scale::B
    scale_prior::S
end

"""
    TuringBRMI

A [`BRMI`](@ref) lowered to the Turing backend. `plan` is the strict,
Stan-independent semantic plan; `model` is the concrete DynamicPPL model
provided by `BayesianRegressionModelsTuringExt` when Turing is loaded.
"""
struct TuringBRMI{P<:BRMI,PL,M}
    parent::P
    plan::PL
    model::M
end

Base.parent(x::TuringBRMI) = x.parent
Base.show(io::IO, x::TuringBRMI) = print(
    io, "TuringBRMI with ", size(x.plan.design.matrix, 2),
    " population coefficients and ", length(x.plan.response), " observations")

# Implemented only by the Turing package extension. Keeping the generic here
# lets the core validate and materialise plans without loading Turing.
function _brm_turing_model end

function _turing_direct_observation(brmi::BRMI)
    found = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(op, 2)
        isnothing(_brm_observation_name(lhs)) && continue
        push!(found, (; key, lhs, rhs))
    end
    length(found) == 1 || error(
        "Turing backend: the initial executable slice requires exactly one " *
        "observed likelihood; found $(length(found))")
    only(found)
end

function _turing_named_reference(x, role::AbstractString)
    x isa NamedColumn || error(
        "Turing backend: Gaussian $role must be a direct named predictor; " *
        "got $(typeof(x))")
    name(x)
end

function _turing_scalar_exponential_prior(brmi::BRMI, target::Symbol)
    op = linear_predictor_op(brmi, target)
    isnothing(op) && error(
        "Turing backend: Gaussian scale `$target` needs an explicit " *
        "`$target ~ Exponential(scale)` prior")
    lhs, rhs = getargs(op, 2)
    lhs isa NamedColumn && name(lhs) === target || error(
        "Turing backend: Gaussian scale prior must bind bare `$target`")
    rhs isa ExprColumn && getf(rhs) === Exponential || error(
        "Turing backend: `$target` currently requires `Exponential(scale)`; " *
        "got $(rhs isa ExprColumn ? getf(rhs) : typeof(rhs))")
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Exponential(scale)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Exponential(scale)` requires one scale argument")
    scale = _brm_numeric_constant(only(args))
    isnothing(scale) && error(
        "Turing backend: `Exponential(scale)` scale must be a numeric constant")
    isfinite(scale) && scale > 0 || error(
        "Turing backend: `Exponential(scale)` requires a finite positive scale")
    scale
end

function _turing_population_components(brmi::BRMI, observation, predictor::Symbol;
                                       extra_operations=())
    lp = linear_predictor_op(brmi, predictor)
    isnothing(lp) && error(
        "Turing backend: predictor `$predictor` has no population formula")
    lp_lhs, lp_rhs = getargs(lp, 2)
    lp_lhs isa NamedColumn && name(lp_lhs) === predictor || error(
        "Turing backend: linked/distributional predictor LHSs are not yet supported")

    isempty(ranef_effect_priors(brmi)) || error(
        "Turing backend: random-effect prior overrides are not yet supported")
    isempty(r2d2_priors(brmi)) || error(
        "Turing backend: R2D2 priors are not yet supported")
    isempty(term_priors(brmi)) || error(
        "Turing backend: term-parameter priors are not yet supported")

    context = _brm_backend_context(brmi)
    # Raw data columns are represented in `brmi.operations` too. They are
    # inputs to the backend plan, not additional model operations.
    allowed = union(Set((observation.key, predictor, extra_operations...)),
                    Set(keys(context.data)),
                    _brm_population_effect_operation_keys(brmi))
    extras = setdiff(Set(keys(brmi.operations)), allowed)
    isempty(extras) || error(
        "Turing backend: unsupported top-level operation(s): " *
        join(sort!(collect(extras)), ", "))

    design = _brm_simple_population_design(
        predictor, lp_rhs, context.data, get(context.target_obs, predictor, nothing);
        required=true)
    response = context.data[observation.key]
    response isa AbstractVector || error(
        "Turing backend: response `$(observation.key)` must be a vector")
    length(response) == size(design.matrix, 1) || error(
        "Turing backend: response and population design have different row counts")
    k = size(design.matrix, 2)
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix="Turing backend")
    beta_location, beta_scale = _brm_materialize_normal_effect_priors(
        overrides, k; prefix="Turing backend")
    (; context, design, response, beta_location, beta_scale)
end

function _turing_normal_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Normal(mu, sigma)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `Normal(mu, sigma)` requires exactly two arguments")
    predictor = _turing_named_reference(args[1], "location")
    scale = _turing_named_reference(args[2], "scale")
    predictor === :mu || error(
        "Turing backend: the initial Gaussian slice requires location predictor " *
        "name `mu`; got `$predictor`")
    scale === :sigma || error(
        "Turing backend: the initial Gaussian slice requires scale name `sigma`; " *
        "got `$scale`")
    parts = _turing_population_components(
        brmi, observation, predictor; extra_operations=(scale,))
    parts.response isa AbstractVector{<:Real} || error(
        "Turing backend: Gaussian response `$(observation.key)` must be real-valued")
    scale_prior = _turing_scalar_exponential_prior(brmi, scale)
    _TuringPopulationPlan(Val(:normal_identity), parts.context, parts.design,
                          collect(Float64, parts.response), parts.beta_location,
                          parts.beta_scale, scale_prior)
end

function _turing_bernoulli_logit_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BernoulliLogit(eta)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `BernoulliLogit(eta)` requires exactly one argument")
    predictor = _turing_named_reference(only(args), "logit")
    parts = _turing_population_components(brmi, observation, predictor)
    all(x -> x isa Bool || (x isa Integer && x in (0, 1)), parts.response) || error(
        "Turing backend: Bernoulli response `$(observation.key)` must contain only 0/1")
    _TuringPopulationPlan(Val(:bernoulli_logit), parts.context, parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, nothing)
end

function _turing_poisson_log_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Poisson(exp(log_rate))` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Poisson(exp(log_rate))` requires exactly one argument")
    link = only(args)
    link isa ExprColumn && getf(link) === exp && isempty(getkwargs(link)) || error(
        "Turing backend: Poisson population models require the log link " *
        "`Poisson(exp(log_rate))`")
    link_args = getargs(link)
    length(link_args) == 1 || error(
        "Turing backend: Poisson log link `exp(log_rate)` requires one argument")
    predictor = _turing_named_reference(only(link_args), "log-rate")
    parts = _turing_population_components(brmi, observation, predictor)
    all(x -> x isa Integer && x >= 0, parts.response) || error(
        "Turing backend: Poisson response `$(observation.key)` must be nonnegative integers")
    _TuringPopulationPlan(Val(:poisson_log), parts.context, parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, nothing)
end

function _brm_turing_plan(brmi::BRMI)
    observation = _turing_direct_observation(brmi)
    observation.key === :y || error(
        "Turing backend: the initial executable slice requires response name `y`; " *
        "got `$(observation.key)`")
    observation.lhs isa NamedColumn || error(
        "Turing backend: response decorators and response links are not yet supported")
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "Turing backend: observed likelihood must be a distribution call")
    family = getf(rhs)
    family === Normal && return _turing_normal_plan(brmi, observation, rhs)
    family === BernoulliLogit &&
        return _turing_bernoulli_logit_plan(brmi, observation, rhs)
    family === Poisson && return _turing_poisson_log_plan(brmi, observation, rhs)
    error("Turing backend: executable families are `Normal(mu, sigma)`, " *
          "`BernoulliLogit(eta)`, and `Poisson(exp(log_rate))`; got `$family`")
end

_brm_turing_gaussian_plan(brmi::BRMI) = begin
    plan = _brm_turing_plan(brmi)
    plan.family isa Val{:normal_identity} || error(
        "Turing backend: requested Gaussian plan for non-Gaussian likelihood")
    plan
end
