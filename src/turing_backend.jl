# Core Turing-facing types and semantic validation. This file deliberately has
# no dependency on Turing, DynamicPPL, StanBlocks, SBBRMI, or emitted SLIC. The
# package extension supplies the executable model after Turing is loaded.

struct _TuringPopulationPlan{F,C<:_BRMBackendContext,P<:_BRMPopulationPredictor,
                             D<:_BRMPopulationDesign,Y<:AbstractVector,
                             B<:AbstractVector,A,S,R<:Tuple,MR,RM,OW}
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
    missing_response::MR
    response_modifier::RM
    observation_weight::OW
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
                                Y<:AbstractVector,RM,OW}
    family::F
    context::C
    mean::M
    precision::Q
    family_args::A
    response::Y
    response_modifier::RM
    observation_weight::OW
end

struct _TuringMultiResponsePlan{N<:Tuple,P<:Tuple,O<:Tuple}
    responses::N
    plans::P
    owners::O
end

"""
    TuringBRMI

A [`BRMI`](@ref) lowered to the Turing backend. `plan` is the strict,
Stan-independent semantic plan; `model` is the concrete DynamicPPL model
provided by `BayesianRegressionModelsTuringExt` when Turing is loaded.
"""
struct _TuringReplayState{G<:Tuple}
    resample_groups::G
end
_TuringReplayState() = _TuringReplayState(())

struct TuringBRMI{P<:BRMI,PL,M,R<:_TuringReplayState}
    parent::P
    plan::PL
    model::M
    replay::R
end
TuringBRMI(parent::BRMI, plan, model) =
    TuringBRMI(parent, plan, model, _TuringReplayState())

Base.parent(x::TuringBRMI) = x.parent
_turing_num_population_coefficients(plan::_TuringPopulationPlan) =
    size(plan.design.matrix, 2)
_turing_num_population_coefficients(plan::_TuringMeanPrecisionPlan) =
    size(plan.mean.design.matrix, 2) + size(plan.precision.design.matrix, 2)
_turing_num_population_coefficients(plan::_TuringMultiResponsePlan) =
    sum(_turing_num_population_coefficients, plan.plans)

_turing_num_observations(plan::_TuringPopulationPlan) = length(plan.response)
_turing_num_observations(plan::_TuringMeanPrecisionPlan) = length(plan.response)
_turing_num_observations(plan::_TuringMultiResponsePlan) =
    sum(_turing_num_observations, plan.plans)

Base.show(io::IO, x::TuringBRMI{<:BRMI,<:_TuringMultiResponsePlan}) = print(
    io, "TuringBRMI with ", _turing_num_population_coefficients(x.plan),
    " population coefficients across ", length(x.plan.responses),
    " responses and ", _turing_num_observations(x.plan), " observations")
Base.show(io::IO, x::TuringBRMI) = print(
    io, "TuringBRMI with ", _turing_num_population_coefficients(x.plan),
    " population coefficients and ", _turing_num_observations(x.plan),
    " observations")

# Implemented only by the Turing package extension. Keeping the generic here
# lets the core validate and materialise plans without loading Turing.
function _brm_turing_model end

"""
    turing_pointwise_loglikelihoods(backend::TuringBRMI, parameters)

Evaluate the direct-BRMI Turing model's observation sites at one constrained
parameter draw and return a response-named `NamedTuple` of rowwise log
likelihoods. Responses with modelled missing values retain their original row
axis, with `missing` at latent (non-observed) rows.

The Turing extension implements this by re-running DynamicPPL's pointwise
likelihood accumulator, so response modifiers and observation weights use the
same executable distributions as `backend.model`.
"""
function turing_pointwise_loglikelihoods end

"""
    turing_predictive_model(backend::TuringBRMI)

Return the direct-BRMI DynamicPPL model with every response site unobserved.
For chain-level prediction, call `Turing.predict([rng], backend, chain)` so
latent response values from partially missing fitted responses are removed
before DynamicPPL conditions the predictive model. Use this model directly
with `Turing.predict` only when the chain contains no response parameters, or
call [`turing_posterior_predictive`](@ref) for one constrained parameter draw.
"""
function turing_predictive_model end

"""
    turing_generated_quantities(backend::TuringBRMI, parameters)

Return the deterministic quantities produced by the fitted Turing model at one
constrained parameter draw. This is the response-aware BRM entry point to
DynamicPPL's `returned` evaluation.
"""
function turing_generated_quantities end

"""
    turing_posterior_predictive([rng], backend::TuringBRMI, parameters)

Draw every modeled response at one constrained parameter draw. The result is a
response-named `NamedTuple`, including one entry per response in a
multi-response model. Response parameters in `parameters` (for example latent
values fitted through `mi(y)`) are deliberately not conditioned, so every row
is regenerated.
"""
function turing_posterior_predictive end

function _turing_unobserved_response(response)
    T = nonmissingtype(eltype(response))
    unobserved = Vector{Union{Missing,T}}(undef, length(response))
    fill!(unobserved, missing)
end

function _turing_predictive_plan(plan::_TuringPopulationPlan)
    _TuringPopulationPlan(
        plan.family, plan.context, plan.predictor, plan.design,
        _turing_unobserved_response(plan.response), plan.beta_location,
        plan.beta_scale, plan.family_args, plan.scale_prior,
        plan.random_effects, plan.missing_response, plan.response_modifier,
        plan.observation_weight)
end

function _turing_predictive_plan(plan::_TuringMeanPrecisionPlan)
    _TuringMeanPrecisionPlan(
        plan.family, plan.context, plan.mean, plan.precision, plan.family_args,
        _turing_unobserved_response(plan.response), plan.response_modifier,
        plan.observation_weight)
end

_turing_predictive_plan(plan::_TuringMultiResponsePlan) =
    _TuringMultiResponsePlan(
        plan.responses,
        Tuple(_turing_predictive_plan(child) for child in plan.plans),
        plan.owners)

function _turing_replay_component(
        training::_TuringPopulationComponent,
        fresh::_TuringPopulationComponent,
        context::_BRMBackendContext, resample_groups)
    predictor = _brm_replay_population_predictor(training.predictor, context)
    random_effects = _turing_replay_random_effects(
        training.random_effects, fresh.random_effects, context,
        resample_groups)
    _TuringPopulationComponent(
        predictor, predictor.design, training.beta_location,
        training.beta_scale, random_effects)
end

function _turing_replay_random_effects(
        training::Tuple, fresh::Tuple, context::_BRMBackendContext,
        resample_groups)
    length(training) == length(fresh) || error(
        "Turing backend: replay changed the random-effect block count")
    Tuple(map(zip(training, fresh)) do (old, new)
        old.predictor === new.predictor && old.id === new.id &&
        old.group === new.group || error(
            "Turing backend: replay changed random-effect block identity")
        old.group in resample_groups ? new :
            _brm_replay_random_effect_plan(old, context)
    end)
end

function _turing_replay_plan(
        training::_TuringPopulationPlan,
        fresh::_TuringPopulationPlan, resample_groups=Set{Symbol}())
    predictor = _brm_replay_population_predictor(
        training.predictor, fresh.context)
    random_effects = _turing_replay_random_effects(
        training.random_effects, fresh.random_effects, fresh.context,
        resample_groups)
    _TuringPopulationPlan(
        fresh.family, fresh.context, predictor, predictor.design,
        fresh.response, training.beta_location, training.beta_scale,
        fresh.family_args, fresh.scale_prior, random_effects,
        fresh.missing_response, fresh.response_modifier,
        fresh.observation_weight)
end


function _turing_replay_plan(
        training::_TuringMeanPrecisionPlan,
        fresh::_TuringMeanPrecisionPlan, resample_groups=Set{Symbol}())
    mean = _turing_replay_component(
        training.mean, fresh.mean, fresh.context, resample_groups)
    precision = _turing_replay_component(
        training.precision, fresh.precision, fresh.context, resample_groups)
    _TuringMeanPrecisionPlan(
        fresh.family, fresh.context, mean, precision, fresh.family_args,
        fresh.response, fresh.response_modifier, fresh.observation_weight)
end

function _turing_replay_plan(
        training::_TuringMultiResponsePlan,
        fresh::_TuringMultiResponsePlan, resample_groups=Set{Symbol}())
    training.responses == fresh.responses || error(
        "Turing backend: replay changed the response-name set")
    length(training.plans) == length(fresh.plans) || error(
        "Turing backend: replay changed the response-plan count")
    plans = Tuple(_turing_replay_plan(old, new, resample_groups)
                  for (old, new) in zip(training.plans, fresh.plans))
    _TuringMultiResponsePlan(training.responses, plans, training.owners)
end


_turing_group_names(plan::_TuringPopulationPlan) =
    Set(block.group for block in plan.random_effects)
_turing_group_names(plan::_TuringMeanPrecisionPlan) = union(
    Set(block.group for block in plan.mean.random_effects),
    Set(block.group for block in plan.precision.random_effects))
function _turing_group_names(plan::_TuringMultiResponsePlan)
    groups = Set{Symbol}()
    foreach(child -> union!(groups, _turing_group_names(child)), plan.plans)
    groups
end

function _turing_collect_factor_schemas!(schemas, column)
    preprocess = column.preprocess
    isnothing(preprocess) && return schemas
    if preprocess.kind === :population_factor_dummy
        levels = collect(preprocess.const_.levels)
        existing = get(schemas, column.source, levels)
        existing == levels || error(
            "Turing backend: fitted categorical source `$(column.source)` " *
            "has inconsistent level schemas across predictors")
        schemas[column.source] = levels
    end
    foreach(dependency -> _turing_collect_factor_schemas!(schemas, dependency),
            preprocess.dependencies)
    schemas
end

function _turing_collect_factor_schemas!(schemas, design::_BRMPopulationDesign)
    foreach(column -> _turing_collect_factor_schemas!(schemas, column),
            design.columns)
    schemas
end
function _turing_collect_factor_schemas!(schemas, plan::_TuringPopulationPlan)
    _turing_collect_factor_schemas!(schemas, plan.design)
end
function _turing_collect_factor_schemas!(schemas, plan::_TuringMeanPrecisionPlan)
    _turing_collect_factor_schemas!(schemas, plan.mean.design)
    _turing_collect_factor_schemas!(schemas, plan.precision.design)
end
function _turing_collect_factor_schemas!(schemas, plan::_TuringMultiResponsePlan)
    foreach(child -> _turing_collect_factor_schemas!(schemas, child), plan.plans)
    schemas
end

function _turing_replay_input(plan, new_data)
    schemas = _turing_collect_factor_schemas!(Dict{Symbol,Any}(), plan)
    isempty(schemas) && return new_data
    names = Tuple(propertynames(new_data))
    values = Tuple(getproperty(new_data, key) for key in names)
    prepared = NamedTuple{names}(values)
    for (source, levels) in schemas
        raw = _brm_df_column(new_data, source)
        raw_values = raw isa CA.CategoricalVector ? let raw_levels = CA.levels(raw)
            [raw_levels[code] for code in Int.(CA.levelcode.(raw))]
        end : collect(raw)
        unknown = unique(
            [value for value in raw_values if value ∉ levels])
        isempty(unknown) || error(
            "BRM replay: categorical predictor `$source` contains unseen " *
            "level(s) $(collect(unknown)); fitted levels are $levels")
        categorical = CA.categorical(raw_values; levels)
        prepared = merge(
            prepared, NamedTuple{(source,)}((categorical,)))
    end
    prepared
end

function _turing_direct_observations(brmi::BRMI)
    found = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(op, 2)
        isnothing(_brm_observation_name(lhs)) && continue
        push!(found, (; key, lhs, rhs))
    end
    isempty(found) && error(
        "Turing backend: direct execution requires at least one observed likelihood")
    Tuple(found)
end


function _turing_direct_observation(brmi::BRMI)
    found = _turing_direct_observations(brmi)
    length(found) == 1 || error(
        "Turing backend: requested one direct observation but found " *
        "$(length(found))")
    only(found)
end

_turing_collect_model_references!(_out, _x) = nothing
function _turing_collect_model_references!(out, x::NamedColumn)
    parent(x) isa DataColumn || push!(out, name(x))
    nothing
end
function _turing_collect_model_references!(out, x::ExprColumn)
    foreach(arg -> _turing_collect_model_references!(out, arg), getargs(x))
    foreach(value -> _turing_collect_model_references!(out, value),
            values(getkwargs(x)))
    nothing
end

function _turing_multi_model_operations(observations)
    out = Set{Symbol}(observation.key for observation in observations)
    foreach(observations) do observation
        _turing_collect_model_references!(out, observation.rhs)
    end
    Tuple(sort!(collect(out)))
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
    isempty(r2d2_priors(brmi)) || error(
        "Turing backend: R2D2 priors are not yet supported")
    isempty(term_priors(brmi)) || error(
        "Turing backend: term-parameter priors are not yet supported")

    context = _brm_backend_context(brmi)
    # Raw data columns are represented in `brmi.operations` too. They are
    # inputs to the backend plan, not additional model operations.
    allowed = union(Set((observation.key, model_operations...)),
                    Set(keys(context.data)),
                    _brm_population_effect_operation_keys(brmi),
                    _brm_ranef_effect_operation_keys(brmi))
    extras = setdiff(Set(keys(brmi.operations)), allowed)
    isempty(extras) || error(
        "Turing backend: unsupported top-level operation(s): " *
        join(sort!(collect(extras)), ", "))
    context
end

function _turing_with_ranef_prior(block::_BRMRandomEffectPlan,
                                  sd_family, sd_rate, lkj_eta)
    _BRMRandomEffectPlan(
        block.predictor, block.id, block.group, block.levels, block.indices,
        block.columns, block.matrix, block.intercept_only,
        block.zero_correlation, collect(Int, sd_family),
        collect(Float64, sd_rate), Float64(lkj_eta))
end

function _turing_apply_ranef_effect_priors(brmi::BRMI, components::Tuple)
    specs = ranef_effect_priors(brmi)
    isempty(specs) && return components
    margins = Dict{Tuple{Symbol,Symbol},Vector{NamedTuple}}()
    locations = Dict{Tuple{Int,Int},Tuple{Tuple{Symbol,Symbol},UnitRange{Int}}}()
    for (component_index, component) in enumerate(components)
        for (block_index, block) in enumerate(component.random_effects)
            isnothing(block.id) && continue
            key = (block.id, block.group)
            axis = get!(margins, key, NamedTuple[])
            first_index = length(axis) + 1
            append!(axis, ((; predictor=block.predictor,
                             coefficient=column.label)
                           for column in block.columns))
            locations[(component_index, block_index)] =
                (key, first_index:length(axis))
        end
    end
    resolved = _brm_resolve_ranef_effect_overrides(
        specs, margins; prefix="Turing backend")
    Tuple(map(enumerate(components)) do (component_index, component)
        blocks = Tuple(map(enumerate(component.random_effects)) do (block_index, block)
            location = get(locations, (component_index, block_index), nothing)
            isnothing(location) && return block
            key, range = location
            override = get(resolved, key, nothing)
            isnothing(override) && return block
            _turing_with_ranef_prior(
                block, override.sd_family[range], override.sd_rate[range],
                override.lkj_eta)
        end)
        _TuringPopulationComponent(
            component.predictor, component.design, component.beta_location,
            component.beta_scale, blocks)
    end)
end

function _turing_materialize_response_modifier(
        response_modifier, observation, response, context;
        support_kind::Symbol=:continuous)
    isnothing(response_modifier) && return nothing
    response_modifier.kind === :interval_censored ?
        _brm_materialize_interval_response(
            response_modifier, observation.key, response, context.data;
            support_kind, prefix="Turing backend") :
        _brm_materialize_bounded_response(
            response_modifier, observation.key, response, context.data;
            support_kind, prefix="Turing backend")
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
                                       allow_zero_correlation::Bool=false,
                                       response_override=nothing)
    context = _turing_model_context(
        brmi, observation, (predictor, extra_operations...))
    if !isnothing(response_override)
        context.data[observation.key] = response_override
    end
    component = _turing_predictor_component(
        brmi, context, predictor; allow_group_terms=allow_random_effects,
        allow_random_slopes, allow_zero_correlation)
    component = only(_turing_apply_ranef_effect_priors(brmi, (component,)))
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
                                           mean_link, precision_link;
                                           additional_model_operations=())
    mean_name = _turing_named_reference(mean_raw, "$family_name mean")
    precision_name = _turing_named_reference(
        precision_raw, "$family_name precision")
    mean_name === precision_name && error(
        "Turing backend: $family_name mean and precision need distinct predictors")

    predictor_names = (mean_name, precision_name)
    context = _turing_model_context(
        brmi, observation,
        (predictor_names..., additional_model_operations...))
    mean = _turing_predictor_component(
        brmi, context, mean_name; available_predictors=predictor_names,
        allow_group_terms=true, allow_random_slopes=true,
        allow_zero_correlation=true)
    precision = _turing_predictor_component(
        brmi, context, precision_name; available_predictors=predictor_names,
        allow_group_terms=true, allow_random_slopes=true,
        allow_zero_correlation=true)
    mean, precision = _turing_apply_ranef_effect_priors(
        brmi, (mean, precision))
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
                                         rhs::ExprColumn;
                                         observation_weight=nothing,
                                         additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `NegativeBinomial2(mu, phi)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `NegativeBinomial2(mu, phi)` requires exactly two arguments")
    parts = _turing_mean_precision_components(
        brmi, observation, args[1], args[2], "NegativeBinomial2", log, log;
        additional_model_operations)
    response = parts.response
    all(x -> x isa Integer && !(x isa Bool) && x >= 0, response) || error(
        "Turing backend: NegativeBinomial2 response `$(observation.key)` must " *
        "contain nonnegative integer counts")
    _TuringMeanPrecisionPlan(
        Val(:negative_binomial2), parts.context, parts.mean, parts.precision,
        NamedTuple(), Int.(response), nothing, observation_weight)
end

function _turing_beta_binomial2_plan(brmi::BRMI, observation,
                                    rhs::ExprColumn;
                                    observation_weight=nothing,
                                    additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BetaBinomial2(trials, mean, precision)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 3 || error(
        "Turing backend: `BetaBinomial2(trials, mean, precision)` requires exactly three arguments")
    trials_raw, mean_raw, precision_raw = args
    parts = _turing_mean_precision_components(
        brmi, observation, mean_raw, precision_raw, "BetaBinomial2", logit, log;
        additional_model_operations)
    trials = _brm_materialize_count_argument(
        trials_raw, length(parts.response), "BetaBinomial2 trial count";
        prefix="Turing backend")
    response = _brm_validate_binomial_response(
        parts.response, trials, observation.key; prefix="Turing backend")
    _TuringMeanPrecisionPlan(
        Val(:beta_binomial2), parts.context, parts.mean, parts.precision,
        (; trials), response, nothing, observation_weight)
end

function _turing_normal_plan(brmi::BRMI, observation, rhs::ExprColumn;
                             missing_response=nothing,
                             response_modifier=nothing,
                             observation_weight=nothing,
                             additional_model_operations=())
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
        brmi, observation, predictor;
        extra_operations=(scale, additional_model_operations...),
        allow_random_effects=true, allow_random_slopes=true,
        allow_zero_correlation=true,
        response_override=isnothing(missing_response) ? nothing :
            missing_response.values)
    _turing_require_predictor_link(parts, identity, "Gaussian identity location")
    all(x -> ismissing(x) || x isa Real, parts.response) || error(
        "Turing backend: Gaussian response `$(observation.key)` must contain " *
        "only real values and explicitly modelled missings")
    scale_prior = _turing_scalar_exponential_prior(brmi, scale)
    materialized_modifier = _turing_materialize_response_modifier(
        response_modifier, observation, parts.response, parts.context)
    response = isnothing(missing_response) ?
        collect(Float64, parts.response) :
        Union{Missing,Float64}[
            ismissing(x) ? missing : Float64(x) for x in parts.response]
    _TuringPopulationPlan(Val(:normal_identity), parts.context, parts.predictor,
                          parts.design,
                          response, parts.beta_location,
                          parts.beta_scale, NamedTuple(), scale_prior,
                          parts.random_effects, missing_response,
                          materialized_modifier,
                          observation_weight)
end

function _turing_bernoulli_population_plan(brmi::BRMI, observation,
                                           predictor::Symbol, expected_link,
                                           role::AbstractString;
                                           observation_weight=nothing,
                                           additional_model_operations=())
    parts = _turing_population_components(
        brmi, observation, predictor;
        extra_operations=additional_model_operations,
        allow_random_effects=true,
        allow_random_slopes=true, allow_zero_correlation=true)
    _turing_require_predictor_link(parts, expected_link, role)
    all(x -> x isa Bool || (x isa Integer && x in (0, 1)), parts.response) || error(
        "Turing backend: Bernoulli response `$(observation.key)` must contain only 0/1")
    _TuringPopulationPlan(Val(:bernoulli_logit), parts.context, parts.predictor,
                          parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, NamedTuple(), nothing,
                          parts.random_effects, nothing, nothing,
                          observation_weight)
end

function _turing_bernoulli_logit_plan(brmi::BRMI, observation, rhs::ExprColumn;
                                      observation_weight=nothing,
                                      additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BernoulliLogit(eta)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `BernoulliLogit(eta)` requires exactly one argument")
    predictor = _turing_named_reference(only(args), "logit")
    _turing_bernoulli_population_plan(
        brmi, observation, predictor, identity, "BernoulliLogit";
        observation_weight, additional_model_operations)
end

function _turing_bernoulli_plan(brmi::BRMI, observation, rhs::ExprColumn;
                                observation_weight=nothing,
                                additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Bernoulli(p)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Bernoulli(p)` requires exactly one argument")
    predictor = _turing_named_reference(only(args), "probability")
    _turing_bernoulli_population_plan(
        brmi, observation, predictor, logit, "Bernoulli logit link";
        observation_weight, additional_model_operations)
end

function _turing_binomial_population_plan(brmi::BRMI, observation, trials_raw,
                                          predictor::Symbol, expected_link,
                                          role::AbstractString;
                                          observation_weight=nothing,
                                          additional_model_operations=())
    parts = _turing_population_components(
        brmi, observation, predictor;
        extra_operations=additional_model_operations,
        allow_random_effects=true,
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
                          parts.random_effects, nothing, nothing,
                          observation_weight)
end

function _turing_binomial_logit_plan(brmi::BRMI, observation, rhs::ExprColumn;
                                     observation_weight=nothing,
                                     additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `BinomialLogit(trials, eta)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `BinomialLogit(trials, eta)` requires exactly two arguments")
    trials_raw, eta_raw = args
    predictor = _turing_named_reference(eta_raw, "logit")
    _turing_binomial_population_plan(
        brmi, observation, trials_raw, predictor, identity, "BinomialLogit";
        observation_weight, additional_model_operations)
end

function _turing_binomial_plan(brmi::BRMI, observation, rhs::ExprColumn;
                               observation_weight=nothing,
                               additional_model_operations=())
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Binomial(trials, p)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `Binomial(trials, p)` requires exactly two arguments")
    trials_raw, probability_raw = args
    predictor = _turing_named_reference(probability_raw, "probability")
    _turing_binomial_population_plan(
        brmi, observation, trials_raw, predictor, logit, "Binomial logit link";
        observation_weight, additional_model_operations)
end

function _turing_poisson_log_plan(brmi::BRMI, observation, rhs::ExprColumn;
                                  response_modifier=nothing,
                                  observation_weight=nothing,
                                  additional_model_operations=())
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
        brmi, observation, predictor;
        extra_operations=additional_model_operations,
        allow_random_effects=true,
        allow_random_slopes=true, allow_zero_correlation=true)
    _turing_require_predictor_link(parts, expected_link, "Poisson log link")
    all(x -> x isa Integer && x >= 0, parts.response) || error(
        "Turing backend: Poisson response `$(observation.key)` must be nonnegative integers")
    materialized_modifier = _turing_materialize_response_modifier(
        response_modifier, observation, parts.response, parts.context;
        support_kind=:discrete)
    _TuringPopulationPlan(Val(:poisson_log), parts.context, parts.predictor,
                          parts.design,
                          Int.(parts.response), parts.beta_location,
                          parts.beta_scale, NamedTuple(), nothing,
                          parts.random_effects, nothing, materialized_modifier,
                          observation_weight)
end

function _brm_turing_single_plan(brmi::BRMI, observation;
                                 additional_model_operations=())
    missing_response = _brm_missing_response_plan(
        observation.lhs; prefix="Turing backend")
    (observation.lhs isa NamedColumn || !isnothing(missing_response)) || error(
        "Turing backend: response decorators other than `mi(response)` and " *
        "response links are not yet supported")
    rhs = observation.rhs
    rhs isa ExprColumn || error(
        "Turing backend: observed likelihood must be a distribution call")
    raw_response = isnothing(missing_response) ? _brm_data_vec(
        observation.key, parent(parent(observation.lhs))) :
        missing_response.values
    observation_weight = _brm_observation_weight_plan(
        rhs, observation.key, raw_response; prefix="Turing backend")
    if !isnothing(observation_weight)
        rhs = observation_weight.distribution
    end
    response_modifier = _brm_response_modifier_plan(rhs; prefix="Turing backend")
    (isnothing(missing_response) || isnothing(observation_weight)) || error(
        "Turing backend: `mi(response)` cannot yet be composed with " *
        "observation weights")
    (isnothing(missing_response) || isnothing(response_modifier)) || error(
        "Turing backend: `mi(response)` cannot yet be composed with response " *
        "modifiers")
    (isnothing(observation_weight) || isnothing(response_modifier) ||
     observation_weight.kind !== :analytic) || error(
        "Turing backend: analytic/precision weights cannot yet be composed " *
        "with response modifiers; use frequency/power objective weights or " *
        "an unmodified Normal observation")
    if !isnothing(response_modifier)
        response_modifier.kind in (:truncated, :censored, :interval_censored) || error(
            "Turing backend: response modifier `$(response_modifier.kind)` is " *
            "not yet executable; the current response-composition slice supports " *
            "bounded Normal observations")
        rhs = response_modifier.base
        rhs isa ExprColumn || error(
            "Turing backend: bounded response base must be a distribution call")
    end
    family = getf(rhs)
    family === Normal && return _turing_normal_plan(
        brmi, observation, rhs; missing_response, response_modifier,
        observation_weight, additional_model_operations)
    isnothing(missing_response) || error(
        "Turing backend: `mi(response)` currently supports only " *
        "`Normal(mu, sigma)` observations; got `$family`")
    (isnothing(observation_weight) ||
     observation_weight.kind !== :analytic) || error(
        "Turing backend: `AnalyticWeights` currently support only " *
        "`Normal(mu, sigma)` observations; got `$family`")
    family === Poisson && return _turing_poisson_log_plan(
        brmi, observation, rhs; response_modifier, observation_weight,
        additional_model_operations)
    isnothing(response_modifier) || error(
        "Turing backend: `$(response_modifier.kind)` currently supports only " *
        "`Normal(mu, sigma)` and `Poisson(rate)` observations; got `$family`")
    family === Bernoulli && return _turing_bernoulli_plan(
        brmi, observation, rhs; observation_weight,
        additional_model_operations)
    family === BernoulliLogit &&
        return _turing_bernoulli_logit_plan(
            brmi, observation, rhs; observation_weight,
            additional_model_operations)
    family === Binomial && return _turing_binomial_plan(
        brmi, observation, rhs; observation_weight,
        additional_model_operations)
    family === BinomialLogit &&
        return _turing_binomial_logit_plan(
            brmi, observation, rhs; observation_weight,
            additional_model_operations)
    family === NegativeBinomial2 &&
        return _turing_negative_binomial2_plan(
            brmi, observation, rhs; observation_weight,
            additional_model_operations)
    family === BetaBinomial2 &&
        return _turing_beta_binomial2_plan(
            brmi, observation, rhs; observation_weight,
            additional_model_operations)
    error("Turing backend: executable families are `Normal(mu, sigma)`, " *
          "`Bernoulli(p)` / `BernoulliLogit(eta)`, `Binomial(trials, p)` / " *
          "`BinomialLogit(trials, eta)`, " *
          "`Poisson(exp(log_rate))`, `NegativeBinomial2(mu, phi)`, and " *
          "`BetaBinomial2(trials, mean, precision)`, and bounded " *
          "`Normal(mu, sigma)`; got `$family`")
end

_turing_parameter_sources(plan::_TuringPopulationPlan) =
    plan.family isa Val{:normal_identity} ?
        (plan.predictor.name, :sigma) : (plan.predictor.name,)
_turing_parameter_sources(plan::_TuringMeanPrecisionPlan) =
    (plan.mean.predictor.name, plan.precision.predictor.name)

_turing_share_class(plan::_TuringPopulationPlan{Val{:normal_identity}}) = :normal
_turing_share_class(plan::_TuringPopulationPlan{Val{:bernoulli_logit}}) = :logit
_turing_share_class(plan::_TuringPopulationPlan{Val{:binomial_logit}}) = :logit
_turing_share_class(plan::_TuringPopulationPlan{Val{:poisson_log}}) = :poisson
_turing_share_class(plan::_TuringMeanPrecisionPlan{Val{:negative_binomial2}}) =
    :negative_binomial2
_turing_share_class(plan::_TuringMeanPrecisionPlan{Val{:beta_binomial2}}) =
    :beta_binomial2

function _turing_same_random_effects(left, right)
    length(left) == length(right) || return false
    all(zip(left, right)) do (a, b)
        a.predictor === b.predictor && a.id === b.id && a.group === b.group &&
        a.levels == b.levels && a.indices == b.indices &&
        a.matrix == b.matrix && a.intercept_only == b.intercept_only &&
        a.zero_correlation == b.zero_correlation
    end
end

function _turing_same_component(left::_TuringPopulationComponent,
                                right::_TuringPopulationComponent)
    left.predictor.name === right.predictor.name &&
    left.predictor.link_lhs_fn === right.predictor.link_lhs_fn &&
    left.design.matrix == right.design.matrix &&
    left.design.fixed == right.design.fixed &&
    left.beta_location == right.beta_location &&
    left.beta_scale == right.beta_scale &&
    _turing_same_random_effects(left.random_effects, right.random_effects)
end

function _turing_same_component(left::_TuringPopulationPlan,
                                right::_TuringPopulationPlan)
    left.predictor.name === right.predictor.name &&
    left.predictor.link_lhs_fn === right.predictor.link_lhs_fn &&
    left.design.matrix == right.design.matrix &&
    left.design.fixed == right.design.fixed &&
    left.beta_location == right.beta_location &&
    left.beta_scale == right.beta_scale &&
    _turing_same_random_effects(left.random_effects, right.random_effects)
end

function _turing_shared_plans_compatible(left::_TuringPopulationPlan,
                                         right::_TuringPopulationPlan)
    _turing_share_class(left) === _turing_share_class(right) || return false
    _turing_same_component(left, right) || return false
    _turing_share_class(left) === :normal ?
        left.scale_prior == right.scale_prior : true
end
function _turing_shared_plans_compatible(left::_TuringMeanPrecisionPlan,
                                         right::_TuringMeanPrecisionPlan)
    _turing_share_class(left) === _turing_share_class(right) &&
    _turing_same_component(left.mean, right.mean) &&
    _turing_same_component(left.precision, right.precision)
end
_turing_shared_plans_compatible(_left, _right) = false

function _turing_response_owners(responses, plans)
    owners = Int[]
    source_sets = [Set(_turing_parameter_sources(plan)) for plan in plans]
    for i in eachindex(plans)
        owner = i
        for j in 1:(i - 1)
            overlap = intersect(source_sets[i], source_sets[j])
            isempty(overlap) && continue
            if source_sets[i] != source_sets[j]
                error("Turing backend: responses `$(responses[j])` and " *
                      "`$(responses[i])` partially overlap parameter sources " *
                      "$(join(sort!(collect(overlap)), ", ")); shared response " *
                      "blocks must currently match exactly")
            end
            _turing_shared_plans_compatible(plans[j], plans[i]) || error(
                "Turing backend: responses `$(responses[j])` and " *
                "`$(responses[i])` share parameter sources but require " *
                "incompatible predictor geometry, priors, or likelihood links")
            owner = owners[j]
            break
        end
        push!(owners, owner)
    end
    Tuple(owners)
end

function _brm_turing_plan(brmi::BRMI)
    observations = _turing_direct_observations(brmi)
    length(observations) == 1 &&
        return _brm_turing_single_plan(brmi, only(observations))

    response_names = Tuple(observation.key for observation in observations)
    length(unique(response_names)) == length(response_names) || error(
        "Turing backend: multi-response observation names must be unique")
    model_operations = _turing_multi_model_operations(observations)
    plans = Tuple(_brm_turing_single_plan(
        brmi, observation; additional_model_operations=model_operations)
        for observation in observations)
    owners = _turing_response_owners(response_names, plans)
    _TuringMultiResponsePlan(response_names, plans, owners)
end

_brm_turing_gaussian_plan(brmi::BRMI) = begin
    plan = _brm_turing_plan(brmi)
    plan.family isa Val{:normal_identity} || error(
        "Turing backend: requested Gaussian plan for non-Gaussian likelihood")
    plan
end
