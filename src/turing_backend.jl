# Core Turing-facing types and semantic validation. This file deliberately has
# no dependency on Turing, DynamicPPL, StanBlocks, SBBRMI, or emitted SLIC. The
# package extension supplies the executable model after Turing is loaded.

struct _TuringPopulationPlan{F,C<:_BRMBackendContext,P<:_BRMPopulationPredictor,
                             D<:_BRMPopulationDesign,Y<:AbstractVector,
                             B<:AbstractVector,A,S,R<:Tuple,RM}
    family::F
    context::C
    predictor::P
    design::D
    response::Y
    beta_location::B
    beta_scale::B
    family_args::A
    scale_prior::S
    random_effects::R
    response_modifier::RM
end

struct _TuringPopulationComponent{P<:_BRMPopulationPredictor,
                                  D<:_BRMPopulationDesign,
                                  B<:AbstractVector,R<:Tuple}
    predictor::P
    design::D
    beta_location::B
    beta_scale::B
    random_effects::R
end

struct _TuringMeanPrecisionPlan{F,C<:_BRMBackendContext,M,Q,A,
                                Y<:AbstractVector,RM}
    family::F
    context::C
    mean::M
    precision::Q
    family_args::A
    response::Y
    response_modifier::RM
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
_turing_num_population_coefficients(plan::_TuringPopulationPlan) =
    size(plan.design.matrix, 2)
_turing_num_population_coefficients(plan::_TuringMeanPrecisionPlan) =
    size(plan.mean.design.matrix, 2) + size(plan.precision.design.matrix, 2)

Base.show(io::IO, x::TuringBRMI) = print(
    io, "TuringBRMI with ", _turing_num_population_coefficients(x.plan),
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
        "Turing backend: direct execution requires exactly one " *
        "observed likelihood; found $(length(found))")
    only(found)
end

function _turing_named_reference(x, role::AbstractString)
    x isa NamedColumn || error(
        "Turing backend: $role must be a direct named predictor; " *
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

function _turing_model_context(brmi::BRMI, observation, model_operations)
    isempty(ranef_effect_priors(brmi)) || error(
        "Turing backend: random-effect prior overrides are not yet supported")
    isempty(r2d2_priors(brmi)) || error(
        "Turing backend: R2D2 priors are not yet supported")
    isempty(term_priors(brmi)) || error(
        "Turing backend: term-parameter priors are not yet supported")

    context = _brm_backend_context(brmi)
    # Raw data columns are represented in `brmi.operations` too. They are
    # inputs to the backend plan, not additional model operations.
    allowed = union(Set((observation.key, model_operations...)),
                    Set(keys(context.data)),
                    _brm_population_effect_operation_keys(brmi))
    extras = setdiff(Set(keys(brmi.operations)), allowed)
    isempty(extras) || error(
        "Turing backend: unsupported top-level operation(s): " *
        join(sort!(collect(extras)), ", "))
    context
end

function _turing_predictor_component(brmi::BRMI, context::_BRMBackendContext,
                                     predictor::Symbol;
                                     available_predictors=(predictor,),
                                     allow_group_terms::Bool=false,
                                     allow_random_slopes::Bool=false,
                                     allow_zero_correlation::Bool=false)
    random_effects = _brm_simple_random_effect_plans(
        brmi, predictor, context; required=true)
    if !allow_group_terms && !isempty(random_effects)
        error("Turing backend: random effects for predictor `$predictor` are " *
              "not yet supported by this likelihood plan")
    end
    if allow_group_terms && length(random_effects) > 1
        error("Turing backend: the initial grouped slice supports one plain " *
              "random-effect block per predictor; found " *
              "$(length(random_effects)) for `$predictor`")
    end
    if !allow_random_slopes && any(!block.intercept_only for block in random_effects)
        error("Turing backend: random slopes for predictor `$predictor` are " *
              "not yet supported by this likelihood plan")
    end
    if !allow_zero_correlation && any(block.zero_correlation
                                      for block in random_effects)
        error("Turing backend: zero-correlation `||` random effects for " *
              "predictor `$predictor` are not yet supported by this likelihood plan")
    end
    predictor_plan = _brm_simple_population_predictor(
        brmi, predictor, context; required=true)
    design = predictor_plan.design
    k = size(design.matrix, 2)
    overrides = _brm_simple_population_effect_overrides(
        brmi, design; prefix="Turing backend", available_predictors)
    beta_location, beta_scale = _brm_materialize_normal_effect_priors(
        overrides, k; prefix="Turing backend")
    _TuringPopulationComponent(
        predictor_plan, design, beta_location, beta_scale, random_effects)
end

function _turing_population_components(brmi::BRMI, observation, predictor::Symbol;
                                       extra_operations=(),
                                       allow_random_effects::Bool=false,
                                       allow_random_slopes::Bool=false,
                                       allow_zero_correlation::Bool=false)
    context = _turing_model_context(
        brmi, observation, (predictor, extra_operations...))
    component = _turing_predictor_component(
        brmi, context, predictor; allow_group_terms=allow_random_effects,
        allow_random_slopes, allow_zero_correlation)
    random_effects = component.random_effects
    response = context.data[observation.key]
    response isa AbstractVector || error(
        "Turing backend: response `$(observation.key)` must be a vector")
    length(response) == size(component.design.matrix, 1) || error(
        "Turing backend: response and population design have different row counts")
    all(length(block.indices) == length(response)
        for block in random_effects) || error(
            "Turing backend: response and random-effect index have different row counts")
    (; context, predictor=component.predictor, design=component.design, response,
       beta_location=component.beta_location, beta_scale=component.beta_scale,
       random_effects)
end


function _turing_require_predictor_link(parts, expected, role::AbstractString)
    found = parts.predictor.link_lhs_fn
    found === expected || error(
        "Turing backend: $role requires predictor `$(parts.predictor.name)` " *
        "to use `$(nameof(expected))(...)` on the formula LHS; got " *
        "`$(nameof(found))(...)`")
    nothing
end

function _turing_mean_precision_components(brmi::BRMI, observation,
                                           mean_raw, precision_raw,
                                           family_name::AbstractString,
                                           mean_link, precision_link)
    mean_name = _turing_named_reference(mean_raw, "$family_name mean")
    precision_name = _turing_named_reference(
        precision_raw, "$family_name precision")
    mean_name === precision_name && error(
        "Turing backend: $family_name mean and precision need distinct predictors")

    predictor_names = (mean_name, precision_name)
    context = _turing_model_context(brmi, observation, predictor_names)
    mean = _turing_predictor_component(
        brmi, context, mean_name; available_predictors=predictor_names,
        allow_group_terms=true, allow_random_slopes=true)
    precision = _turing_predictor_component(
        brmi, context, precision_name; available_predictors=predictor_names,
        allow_group_terms=true, allow_random_slopes=true)
    _turing_require_predictor_link((; predictor=mean.predictor), mean_link,
                                   "$family_name mean link")
    _turing_require_predictor_link((; predictor=precision.predictor),
                                   precision_link, "$family_name precision link")

    response = context.data[observation.key]
    response isa AbstractVector || error(
        "Turing backend: response `$(observation.key)` must be a vector")
    n = length(response)
    size(mean.design.matrix, 1) == n || error(
        "Turing backend: $family_name mean design has a different row count")
    size(precision.design.matrix, 1) == n || error(
        "Turing backend: $family_name precision design has a different row count")
    all(length(block.indices) == n
        for component in (mean, precision)
        for block in component.random_effects) || error(
            "Turing backend: $family_name response and random-effect index " *
            "have different row counts")
    (; context, mean, precision, response)
end

function _turing_negative_binomial2_plan(brmi::BRMI, observation,
                                         rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `NegativeBinomial2(mu, phi)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `NegativeBinomial2(mu, phi)` requires exactly two arguments")
    parts = _turing_mean_precision_components(
        brmi, observation, args[1], args[2], "NegativeBinomial2", log, log)
    response = parts.response
    all(x -> x isa Integer && !(x isa Bool) && x >= 0, response) || error(
        "Turing backend: NegativeBinomial2 response `$(observation.key)` must " *
        "contain nonnegative integer counts")
    _TuringMeanPrecisionPlan(
        Val(:negative_binomial2), parts.context, parts.mean, parts.precision,
        NamedTuple(), Int.(response), nothing)
end

function _turing_beta_binomial2_plan(brmi::BRMI, observation,
                                    rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BetaBinomial2(trials, mean, precision)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 3 || error(
        "Turing backend: `BetaBinomial2(trials, mean, precision)` requires exactly three arguments")
    trials_raw, mean_raw, precision_raw = args
    parts = _turing_mean_precision_components(
        brmi, observation, mean_raw, precision_raw, "BetaBinomial2", logit, log)
    trials = _brm_materialize_count_argument(
        trials_raw, length(parts.response), "BetaBinomial2 trial count";
        prefix="Turing backend")
    response = _brm_validate_binomial_response(
        parts.response, trials, observation.key; prefix="Turing backend")
    _TuringMeanPrecisionPlan(
        Val(:beta_binomial2), parts.context, parts.mean, parts.precision,
        (; trials), response, nothing)
end

function _turing_normal_plan(brmi::BRMI, observation, rhs::ExprColumn;
                             response_modifier=nothing)
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
        brmi, observation, predictor; extra_operations=(scale,),
        allow_random_effects=true, allow_random_slopes=true,
        allow_zero_correlation=true)
    _turing_require_predictor_link(parts, identity, "Gaussian identity location")
    parts.response isa AbstractVector{<:Real} || error(
        "Turing backend: Gaussian response `$(observation.key)` must be real-valued")
    scale_prior = _turing_scalar_exponential_prior(brmi, scale)
    materialized_modifier = isnothing(response_modifier) ? nothing :
        _brm_materialize_bounded_response(
            response_modifier, observation.key, parts.response,
            parts.context.data; prefix="Turing backend")
    _TuringPopulationPlan(Val(:normal_identity), parts.context, parts.predictor,
                          parts.design,
                          collect(Float64, parts.response), parts.beta_location,
                          parts.beta_scale, NamedTuple(), scale_prior,
                          parts.random_effects, materialized_modifier)
end

function _turing_bernoulli_population_plan(brmi::BRMI, observation,
                                           predictor::Symbol, expected_link,
                                           role::AbstractString)
    parts = _turing_population_components(
        brmi, observation, predictor; allow_random_effects=true,
        allow_random_slopes=true, allow_zero_correlation=true)
    _turing_require_predictor_link(parts, expected_link, role)
    all(x -> x isa Bool || (x isa Integer && x in (0, 1)), parts.response) || error(
        "Turing backend: Bernoulli response `$(observation.key)` must contain only 0/1")
    _TuringPopulationPlan(Val(:bernoulli_logit), parts.context, parts.predictor,
                          parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, NamedTuple(), nothing,
                          parts.random_effects, nothing)
end

function _turing_bernoulli_logit_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BernoulliLogit(eta)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `BernoulliLogit(eta)` requires exactly one argument")
    predictor = _turing_named_reference(only(args), "logit")
    _turing_bernoulli_population_plan(
        brmi, observation, predictor, identity, "BernoulliLogit")
end

function _turing_bernoulli_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Bernoulli(p)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Bernoulli(p)` requires exactly one argument")
    predictor = _turing_named_reference(only(args), "probability")
    _turing_bernoulli_population_plan(
        brmi, observation, predictor, logit, "Bernoulli logit link")
end

function _turing_binomial_population_plan(brmi::BRMI, observation, trials_raw,
                                          predictor::Symbol, expected_link,
                                          role::AbstractString)
    parts = _turing_population_components(
        brmi, observation, predictor; allow_random_effects=true,
        allow_random_slopes=true, allow_zero_correlation=true)
    _turing_require_predictor_link(parts, expected_link, role)
    trials = _brm_materialize_count_argument(
        trials_raw, length(parts.response), "Binomial trial count";
        prefix="Turing backend")
    response = _brm_validate_binomial_response(
        parts.response, trials, observation.key; prefix="Turing backend")
    _TuringPopulationPlan(Val(:binomial_logit), parts.context, parts.predictor,
                          parts.design,
                          response, parts.beta_location,
                          parts.beta_scale, (; trials), nothing,
                          parts.random_effects, nothing)
end

function _turing_binomial_logit_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BinomialLogit(trials, eta)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `BinomialLogit(trials, eta)` requires exactly two arguments")
    trials_raw, eta_raw = args
    predictor = _turing_named_reference(eta_raw, "logit")
    _turing_binomial_population_plan(
        brmi, observation, trials_raw, predictor, identity, "BinomialLogit")
end

function _turing_binomial_plan(brmi::BRMI, observation, rhs::ExprColumn)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Binomial(trials, p)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `Binomial(trials, p)` requires exactly two arguments")
    trials_raw, probability_raw = args
    predictor = _turing_named_reference(probability_raw, "probability")
    _turing_binomial_population_plan(
        brmi, observation, trials_raw, predictor, logit, "Binomial logit link")
end

function _turing_poisson_log_plan(brmi::BRMI, observation, rhs::ExprColumn;
                                  response_modifier=nothing)
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Poisson(exp(log_rate))` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Poisson(exp(log_rate))` requires exactly one argument")
    rate = only(args)
    predictor, expected_link = if rate isa ExprColumn &&
                                  getf(rate) === exp &&
                                  isempty(getkwargs(rate))
        link_args = getargs(rate)
        length(link_args) == 1 || error(
            "Turing backend: Poisson log link `exp(log_rate)` requires one argument")
        (_turing_named_reference(only(link_args), "log-rate"), identity)
    elseif rate isa NamedColumn
        (_turing_named_reference(rate, "rate"), log)
    else
        error("Turing backend: Poisson population models require either " *
              "`log(lambda) ~ formula; y ~ Poisson(lambda)` or " *
              "`log_rate ~ formula; y ~ Poisson(exp(log_rate))`")
    end
    parts = _turing_population_components(
        brmi, observation, predictor; allow_random_effects=true,
        allow_random_slopes=true, allow_zero_correlation=true)
    _turing_require_predictor_link(parts, expected_link, "Poisson log link")
    all(x -> x isa Integer && x >= 0, parts.response) || error(
        "Turing backend: Poisson response `$(observation.key)` must be nonnegative integers")
    materialized_modifier = isnothing(response_modifier) ? nothing :
        _brm_materialize_bounded_response(
            response_modifier, observation.key, parts.response,
            parts.context.data; support_kind=:discrete,
            prefix="Turing backend")
    _TuringPopulationPlan(Val(:poisson_log), parts.context, parts.predictor,
                          parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, NamedTuple(), nothing,
                          parts.random_effects, materialized_modifier)
end

function _brm_turing_plan(brmi::BRMI)
    observation = _turing_direct_observation(brmi)
    observation.lhs isa NamedColumn || error(
        "Turing backend: response decorators and response links are not yet supported")
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "Turing backend: observed likelihood must be a distribution call")
    response_modifier = _brm_response_modifier_plan(rhs; prefix="Turing backend")
    if !isnothing(response_modifier)
        response_modifier.kind in (:truncated, :censored) || error(
            "Turing backend: response modifier `$(response_modifier.kind)` is " *
            "not yet executable; the current response-composition slice supports " *
            "bounded Normal observations")
        rhs = response_modifier.base
        rhs isa ExprColumn || error(
            "Turing backend: bounded response base must be a distribution call")
    end
    family = getf(rhs)
    family === Normal && return _turing_normal_plan(
        brmi, observation, rhs; response_modifier)
    family === Poisson && return _turing_poisson_log_plan(
        brmi, observation, rhs; response_modifier)
    isnothing(response_modifier) || error(
        "Turing backend: `$(response_modifier.kind)` currently supports only " *
        "`Normal(mu, sigma)` and `Poisson(rate)` observations; got `$family`")
    family === Bernoulli && return _turing_bernoulli_plan(brmi, observation, rhs)
    family === BernoulliLogit &&
        return _turing_bernoulli_logit_plan(brmi, observation, rhs)
    family === Binomial && return _turing_binomial_plan(brmi, observation, rhs)
    family === BinomialLogit &&
        return _turing_binomial_logit_plan(brmi, observation, rhs)
    family === NegativeBinomial2 &&
        return _turing_negative_binomial2_plan(brmi, observation, rhs)
    family === BetaBinomial2 &&
        return _turing_beta_binomial2_plan(brmi, observation, rhs)
    error("Turing backend: executable families are `Normal(mu, sigma)`, " *
          "`Bernoulli(p)` / `BernoulliLogit(eta)`, `Binomial(trials, p)` / " *
          "`BinomialLogit(trials, eta)`, " *
          "`Poisson(exp(log_rate))`, `NegativeBinomial2(mu, phi)`, and " *
          "`BetaBinomial2(trials, mean, precision)`, and bounded " *
          "`Normal(mu, sigma)`; got `$family`")
end

_brm_turing_gaussian_plan(brmi::BRMI) = begin
    plan = _brm_turing_plan(brmi)
    plan.family isa Val{:normal_identity} || error(
        "Turing backend: requested Gaussian plan for non-Gaussian likelihood")
    plan
end
