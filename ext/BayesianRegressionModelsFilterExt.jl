module BayesianRegressionModelsFilterExt

import BayesianRegressionModels as BRM
import ReactiveKernelsPPL as PPL
import ReactiveKernels
import SequentialMonteCarlo
using Random: Xoshiro
using Serialization: serialize
using SHA: sha256

backend(model::BRM.RKBRMI) = model
backend(model::BRM.BRMI) = BRM.RKBRMI(model)
constrained(rk, point::AbstractVector) = BRM._rk_constrained_values(rk, point)
constrained(rk, point) = point

function frozen_values(rk)
    if BRM._rk_design_checked(rk.plan)
        names, _ = BRM._rk_fitted_value_names(rk; prefix="BRM filter")
        return BRM._rk_frozen_values(rk, names)
    end
    isempty(BRM._rk_plan_fit_inputs(rk.plan)) || throw(ArgumentError(
        "BRM filter: this model's fitted preprocessing cannot be exported"))
    PPL.FrozenValues()
end

function source_identity(rk, frozen, configuration)
    emitted = BRM._rk_retained_program(rk)
    io = IOBuffer()
    serialize(io, (format=1, program=BRM._rk_identity_statements(rk),
        callables=emitted.bindings, frozen, configuration))
    "brm-filter-v1:" * bytes2hex(sha256(take!(io)))
end

function prepare(rk, configuration)
    frozen = frozen_values(rk)
    id = source_identity(rk, frozen, configuration)
    target = PPL.prepare_filter(BRM.rkppl_model(rk), BRM.rk_bound_plan(rk);
        model_id=id, frozen, configuration...)
    target, id
end

function BRM.brm_filter(model::Union{BRM.BRMI,BRM.RKBRMI}; parameters=nothing,
        ensemble=nothing, weights=nothing, inputs=(), groups=(;), drifting=(),
        rng=Xoshiro(), particles=1000, history_limit=32)
    rk = backend(model)
    observations = Tuple(BRM._rk_observed_names(rk.plan))
    inputs = Tuple(Symbol.(inputs))
    source = BRM.rkppl_model(rk)
    columns = BRM.rk_bound_data(rk)
    select(point) = PPL.filter_parameters(source, constrained(rk, point);
        observations, inputs, data=columns)
    if ensemble === nothing
        weights === nothing || throw(ArgumentError("weights requires a parameter ensemble"))
        parameter_values = select(parameters === nothing ? (;) : parameters)
        members = nothing
    else
        parameters === nothing || throw(ArgumentError("supply parameters or ensemble, not both"))
        members = [select(point) for point in ensemble]
        isempty(members) && throw(ArgumentError("the parameter ensemble must be nonempty"))
        parameter_values = first(members)
    end
    configuration = (; parameters=parameter_values, observations, inputs,
        groups=deepcopy(groups), drifting=Tuple(Symbol.(drifting)))
    target, id = prepare(rk, configuration)
    state = PPL.initialize_filter(target; rng, particles, ensemble=members,
        weights, history_limit)
    BRM.BRMFilter(target, state, configuration, id)
end

function BRM.brm_filter(fit::Union{BRM.BRMSMCFit,BRM.BRMSMCWindowFit};
        learn=false, kwargs...)
    BRM._brm_validate_smc_fit(fit)
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM filter: complete the parameter posterior bridge before filtering"))
    learn && return BRM._brm_filter_learn(fit; kwargs...)
    BRM.brm_filter(fit.backend; ensemble=eachcol(fit.state.positions),
        weights=fit.state.weights, kwargs...)
end

function BRM.brm_filter_step(filter::BRM.BRMFilter, observation::NamedTuple;
        inputs=(;), groups=nothing, condition_state_only=false)
    state = PPL.filter_step(filter.state, observation; inputs, groups, condition_state_only)
    BRM.BRMFilter(filter.target, state, filter.configuration, filter.model_id)
end

function BRM.brm_filter_stream(filter::Union{BRM.BRMFilter,BRM.BRMFilterSMC}, records;
        steps=typemax(Int), condition_state_only=false, progress=nothing,
        stop=filter -> false)
    steps isa Integer && steps >= 0 || throw(ArgumentError("steps must be a nonnegative integer"))
    BRM._brm_with_progress(progress; description="BRM state filtering") do node
        current = filter
        iteration = ()
        for completed in 1:steps
            (BRM._brm_interrupt_requested(node) || stop(current)) && break
            item = iterate(records, iteration...)
            item === nothing && break
            record, cursor = item
            current = BRM.brm_filter_step(current, record.observation;
                inputs=get(record, :inputs, (;)), groups=get(record, :groups, nothing),
                condition_state_only)
            iteration = (cursor,)
            BRM._brm_update_progress!(node, "$completed observations")
        end
        current
    end
end

function BRM.brm_filter_checkpoint(filter::BRM.BRMFilter)
    (; format=1, model_id=filter.model_id,
       configuration=deepcopy(filter.configuration), state=PPL.filter_checkpoint(filter.state))
end

function BRM.brm_filter_resume(model::Union{BRM.BRMI,BRM.RKBRMI}, checkpoint)
    checkpoint.format == 2 && return BRM._brm_filter_learning_resume(model, checkpoint)
    checkpoint.format == 1 || throw(ArgumentError("unsupported BRM filter checkpoint format"))
    configuration = deepcopy(checkpoint.configuration)
    target, id = prepare(backend(model), configuration)
    id == checkpoint.model_id || throw(ArgumentError(
        "BRM filter: the authored model or frozen preprocessing has changed"))
    state = PPL.resume_filter(target, checkpoint.state)
    BRM.BRMFilter(target, state, configuration, id)
end

end
