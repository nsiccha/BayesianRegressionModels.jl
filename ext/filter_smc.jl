# Included in SMCExt: the optional SMC/AD dependencies are already present.
filter_extension() = Base.get_extension(BRM, :BayesianRegressionModelsFilterExt)

function filter_calibration(fit; scale, basis, inputs, max_reflections, maxiter, progress)
    rk = fit.backend
    inputs = Tuple(Symbol.(inputs))
    observations = Tuple(BRM._rk_observed_names(rk.plan))
    names = PPL.filter_parameter_names(BRM.rkppl_model(rk);
        observations, inputs, data=BRM.rk_bound_data(rk))
    isempty(names) && throw(ArgumentError("BRM filter learning: no static parameters to learn"))
    posterior = PPL.carry_smc(fit.state; parameters=names,
        calibration_id=fit.binding_id, scale, basis, max_reflections, maxiter, progress)
    preprocessing = BRM._BRMFilterPreprocessing(
        deepcopy(filter_extension().frozen_values(rk)),
        deepcopy(BRM._rk_fitted_constants(rk)), observations, inputs)
    carried_reference(rk, posterior, preprocessing, fit.model_id)
end

function prepare_filter_learning(calibration, model, ad_backend;
        groups=(;), drifting=(), boundary=nothing, configuration=nothing)
    check_carried(calibration)
    boundary === nothing || boundary isa BRM.BRMFilter || throw(ArgumentError(
        "BRM filter learning: boundary must be an ordinary BRMFilter"))
    preprocessing = calibration.preprocessing
    preprocessing isa BRM._BRMFilterPreprocessing || throw(ArgumentError(
        "BRM filter learning: use brm_smc_carry(fit; parameters=:filter)"))
    rk = model isa BRM.RKBRMI ? model : BRM.RKBRMI(model)
    isequal(BRM._rk_fitted_constants(rk), preprocessing.constants) || throw(ArgumentError(
        "BRM filter learning: frozen constants have changed"))
    streaming_source(rk, preprocessing) == calibration.source_id || throw(ArgumentError(
        "BRM filter learning: authored source or frozen preprocessing has changed"))
    prototype = PPL.smc_parameters(calibration.posterior,
        calibration.posterior.reference.location)
    if configuration === nothing
        parameters = boundary === nothing ? prototype : deepcopy(boundary.configuration.parameters)
        configuration = (; parameters, observations=preprocessing.observations,
            inputs=preprocessing.inputs, groups=deepcopy(groups),
            drifting=Tuple(Symbol.(drifting)))
    end
    id = filter_extension().source_identity(rk, preprocessing.frozen, configuration)
    boundary === nothing || boundary.model_id == id || throw(ArgumentError(
        "BRM filter learning: boundary source, preprocessing or configuration has changed"))
    ordinary = PPL.prepare_filter(BRM.rkppl_model(rk), BRM.rk_bound_plan(rk);
        model_id=id, frozen=preprocessing.frozen, configuration...)
    target = PPL.prepare_filter_smc(ordinary, calibration.posterior;
        backend=ad_backend, calibration_model_id=calibration.model_id)
    target, configuration, id
end

"""
    brm_filter(calibration::BRMSMCCarriedReference, model; members=64,
        particles=128, window_length=32, ess_fraction=0.5, options=SMCOptions(),
        boundary=nothing, groups=(;), drifting=(), rng=Xoshiro(), history_limit=32)

Initialize parameter learning from a selected static calibration, produced by
`brm_smc_carry(fit; parameters=:filter, inputs=...)`. `particles` is per member.
Below the member ESS threshold RKPPL replenishes and rejuvenates the ensemble,
rebuilding conditional states over the bounded window for changed parameters.
An optional ordinary `BRMFilter` boundary transfers state-only training warm-up
and its clocks, excluding its training score. Specify its same groups/drifting
configuration. New observations must be supplied once.
"""
function BRM.brm_filter(calibration::BRM.BRMSMCCarriedReference,
        model::Union{BRM.BRMI,BRM.RKBRMI}; rng=Xoshiro(), members=64, particles=128,
        window_length=32, ess_fraction=0.5, options=PPL.SMCOptions(), history_limit=32,
        boundary=nothing, groups=(;), drifting=(),
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    calibration = deepcopy(calibration)
    target, configuration, id = prepare_filter_learning(calibration, model, ad_backend;
        groups, drifting, boundary)
    state = PPL.initialize_filter_smc(target; rng, members, particles, window_length,
        ess_fraction, options, history_limit,
        boundary=boundary === nothing ? nothing : boundary.state)
    BRM.BRMFilterSMC(target, state, configuration, calibration, id)
end

function BRM._brm_filter_learn(fit; inputs=(), scale=nothing, basis=nothing, kwargs...)
    calibration = BRM.brm_smc_carry(fit; parameters=:filter, inputs, scale, basis)
    BRM.brm_filter(calibration, fit.backend; kwargs...)
end

function BRM.brm_filter_tune(calibration::BRM.BRMSMCCarriedReference,
        model::Union{BRM.BRMI,BRM.RKBRMI}; rng, init, groups=(;), drifting=(),
        boundary=nothing, ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), kwargs...)
    target, _, _ = prepare_filter_learning(calibration, model, ad_backend;
        groups, drifting, boundary)
    PPL.tune_filter_smc(target; rng, init, kwargs...)
end

function BRM.brm_filter_tune(fit::Union{BRM.BRMSMCFit,BRM.BRMSMCWindowFit};
        inputs=(), scale=nothing, basis=nothing, kwargs...)
    calibration = BRM.brm_smc_carry(fit; parameters=:filter, inputs, scale, basis)
    BRM.brm_filter_tune(calibration, fit.backend; kwargs...)
end

function BRM.brm_filter_step(filter::BRM.BRMFilterSMC, observation::NamedTuple;
        inputs=(;), groups=nothing, condition_state_only=false)
    condition_state_only && throw(ArgumentError(
        "BRM filter learning: warm an ordinary filter first, then pass boundary="))
    check_carried(filter.calibration)
    state = PPL.filter_step(filter.state, observation; inputs, groups)
    BRM.BRMFilterSMC(filter.target, state, filter.configuration,
        filter.calibration, filter.model_id)
end

function BRM.brm_filter_checkpoint(filter::BRM.BRMFilterSMC)
    check_carried(filter.calibration)
    (; format=2, model_id=filter.model_id,
       configuration=deepcopy(filter.configuration),
       calibration=deepcopy(filter.calibration), state=PPL.filter_checkpoint(filter.state))
end

function BRM._brm_filter_learning_resume(model, checkpoint;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    calibration = deepcopy(checkpoint.calibration)
    target, configuration, id = prepare_filter_learning(calibration, model, ad_backend;
        configuration=deepcopy(checkpoint.configuration))
    id == checkpoint.model_id || throw(ArgumentError(
        "BRM filter learning: checkpoint source or configuration has changed"))
    state = PPL.resume_filter(target, checkpoint.state)
    BRM.BRMFilterSMC(target, state, configuration, calibration, id)
end
