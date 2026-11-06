module BayesianRegressionModelsSMCExt

import BayesianRegressionModels as BRM
import ReactiveKernelsPPL as PPL
using ADTypes: AutoEnzyme
import Enzyme
import DifferentiationInterface
import ReactiveKernels
import SequentialMonteCarlo
import WarmupHMC
using Random: Xoshiro
import LogDensityProblems
using Statistics: mean, std
using StatsBase: sample, Weights
using Serialization: serialize
using SHA: sha256

# Hash the complete binding, not a truncated display or just its shape.
# The initial template deliberately identifies a fit lineage, including its
# training data. Updates keep this template; callers never author identity IDs.
binding_identity(rk::BRM.RKBRMI) = BRM._brm_rk_binding_identity(rk, "brm-smc-v1")

function check_identity(rk, expected, label)
    binding_identity(rk) == expected || throw(ArgumentError(
        "BRM SMC: $label has changed; retain the original model, preprocessing and data"))
end

function BRM._brm_validate_smc_fit(fit::BRM.BRMSMCFit)
    check_identity(fit.template, fit.model_id, "training template")
    check_identity(fit.backend, fit.binding_id, "current binding")
    fit.previous === nothing || check_identity(fit.previous, fit.previous_id, "previous binding")
    nothing
end

function prepare(rk::BRM.RKBRMI, ad_backend, model_id, binding_id)
    PPL.prepare_smc(rk.model, BRM.rk_bound_plan(rk),
        zeros(Float64, rk.model.layout.total);
        backend=ad_backend, model_id, binding_id)
end

smc_model(model::BRM.BRMI) = BRM.RKBRMI(model)
smc_model(model::BRM.RKBRMI) = model
smc_names(rk::BRM.RKBRMI) = PPL.coordinate_names(rk.model.layout)
smc_update_model(template::BRM.RKBRMI, data) =
    BRM.reprocess(template, data; freeze_constants=true)

include("smc_laplace.jl")

function finish(state, stages, node)
    stages isa Integer && stages >= 0 || throw(ArgumentError("stages must be a nonnegative integer"))
    for _ in 1:stages
        state.temperature == 1 && break
        BRM._brm_interrupt_requested(node) && break
        state = PPL.smc_step(state)
        BRM._brm_update_progress!(node, "temperature $(round(state.temperature; digits=3))")
    end
    state
end

function tuning_controls(; kwargs...)
    # Leave optional defaults with the producer that implements each pilot.
    (; (key => value for (key, value) in pairs(kwargs) if value !== nothing)...)
end

function BRM.brm_smc_tune(model::Union{BRM.BRMI,BRM.RKBRMI,BRM._BRMSMCLaplaceProblem};
        rng, init, reference=nothing,
        temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        reference_inflation=nothing, grid_start=nothing, progress=nothing, callback=nothing,
        checkpoint_dir=nothing, resume=false, overwrite=false,
        options=PPL.SMCOptions(), ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    rk = smc_model(model)
    id = binding_identity(rk)
    target = prepare(rk, ad_backend, id, id)
    controls = tuning_controls(; temperatures, pilot, max_reflections,
        recording_target, pilot_evaluations, reference_inflation, grid_start)
    PPL.tune_smc(target; rng, init, reference, controls..., pilot_steps, options,
        progress, callback, checkpoint_dir, resume, overwrite)
end

function BRM.brm_smc_tune(fit::BRM.BRMSMCFit, cumulative_data;
        rng, init, temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        grid_start=nothing, progress=nothing, callback=nothing, checkpoint_dir=nothing,
        resume=false, overwrite=false,
        options=PPL.SMCOptions())
    BRM._brm_validate_smc_fit(fit)
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM SMC: finish the current bridge before tuning a data update"))
    rk = smc_update_model(fit.template, cumulative_data)
    id = binding_identity(rk)
    target = prepare(rk, fit.ad_backend, fit.model_id, id)
    previous = prepare(fit.backend, fit.ad_backend, fit.model_id, fit.binding_id)
    controls = tuning_controls(; temperatures, pilot, max_reflections,
        recording_target, pilot_evaluations, grid_start)
    PPL.tune_smc(target; rng, init, previous, controls..., pilot_steps, options,
        progress, callback, checkpoint_dir, resume, overwrite)
end

function BRM.brm_smc_fit(model::Union{BRM.BRMI,BRM.RKBRMI,BRM._BRMSMCLaplaceProblem};
        rng=Xoshiro(), particles=1000, reference=nothing, options=PPL.SMCOptions(),
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), stages=typemax(Int), progress=nothing)
    BRM._brm_with_progress(progress; description="BRM SMC fit") do node
        BRM._brm_throw_if_interrupted(node)
        rk = smc_model(model)
        id = binding_identity(rk)
        target = prepare(rk, ad_backend, id, id)
        state = reference === nothing ?
            PPL.initialize_smc(target; rng, particles, options) :
            PPL.initialize_smc(target; rng, particles, reference, options)
        state = finish(state, stages, node)
        BRM.BRMSMCFit(rk, rk, nothing, state, ad_backend, id, id, nothing)
    end
end

function BRM.brm_smc_fit(fit::BRM.BRMSMCFit; stages=typemax(Int), progress=nothing)
    BRM._brm_validate_smc_fit(fit)
    BRM._brm_with_progress(progress; description="BRM SMC continuation") do node
        state = finish(fit.state, stages, node)
        BRM.BRMSMCFit(fit.template, fit.backend, fit.previous, state,
            fit.ad_backend, fit.model_id, fit.binding_id, fit.previous_id)
    end
end

function BRM.brm_smc_update(fit::BRM.BRMSMCFit, cumulative_data;
        options=fit.state.options, stages=typemax(Int), progress=nothing)
    BRM._brm_validate_smc_fit(fit)
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM SMC: finish the current bridge with brm_smc_fit before starting another update"))
    BRM._brm_with_progress(progress; description="BRM SMC data update") do node
        BRM._brm_throw_if_interrupted(node)
        rk = smc_update_model(fit.template, cumulative_data)
        id = binding_identity(rk)
        target = prepare(rk, fit.ad_backend, fit.model_id, id)
        state = PPL.update_smc(fit.state, target; options, stages=0)
        state = finish(state, stages, node)
        BRM.BRMSMCFit(fit.template, rk, fit.backend, state,
            fit.ad_backend, fit.model_id, id, fit.binding_id)
    end
end

function BRM.brm_smc_checkpoint(fit::BRM.BRMSMCFit)
    BRM._brm_validate_smc_fit(fit)
    (; format=1, model_id=fit.model_id, binding_id=fit.binding_id,
       previous_id=fit.previous_id, state=PPL.smc_checkpoint(fit.state))
end

function BRM.brm_smc_resume(template::BRM.RKBRMI, checkpoint;
        data=nothing, previous_data=nothing, nthreads=nothing,
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    checkpoint.format == 1 || throw(ArgumentError("unsupported BRM SMC checkpoint format"))
    check_identity(template, checkpoint.model_id, "training template")
    rk = data === nothing ? template : BRM.reprocess(template, data; freeze_constants=true)
    check_identity(rk, checkpoint.binding_id, "current binding")
    previous = if checkpoint.previous_id === nothing
        previous_data === nothing || throw(ArgumentError("an initial-fit checkpoint has no previous data"))
        nothing
    else
        old = previous_data === nothing ? template :
            BRM.reprocess(template, previous_data; freeze_constants=true)
        check_identity(old, checkpoint.previous_id, "previous binding")
        old
    end
    target = prepare(rk, ad_backend, checkpoint.model_id, checkpoint.binding_id)
    old_target = previous === nothing ? nothing :
        prepare(previous, ad_backend, checkpoint.model_id, checkpoint.previous_id)
    state = PPL.resume_smc(target, checkpoint.state; previous=old_target, nthreads)
    BRM.BRMSMCFit(template, rk, previous, state, ad_backend,
        checkpoint.model_id, checkpoint.binding_id, checkpoint.previous_id)
end

include("smc_streaming.jl")
include("filter_smc.jl")

# Weighted first and second moments of a d × n cloud with normalized weights.
function weighted_moments(positions, weights)
    means = positions * weights
    centered = positions .- means
    variances = vec(sum(centered .^ 2 .* transpose(weights); dims=2))
    (means, sqrt.(max.(variances, 0.0)))
end

function BRM.brm_smc_compare(fits::BRM.BRMSMCFit...)
    length(fits) >= 2 || throw(ArgumentError("brm_smc_compare needs at least two completed fits"))
    reference = first(fits)
    for (k, fit) in enumerate(fits)
        BRM._brm_validate_smc_fit(fit)
        fit.model_id == reference.model_id && fit.binding_id == reference.binding_id &&
            fit.previous_id == reference.previous_id || throw(ArgumentError(
            "fit $k targets a different model, data or previous posterior than fit 1"))
        fit.state.temperature == 1 || throw(ArgumentError(
            "fit $k is not complete (temperature $(fit.state.temperature)); finish it with brm_smc_fit first"))
    end
    names = smc_names(reference.backend)
    runs = map(fits) do fit
        means, sds = weighted_moments(fit.state.positions, fit.state.weights)
        history = fit.state.history
        (; means, sds, log_normalizer=fit.state.log_normalizer,
            lineages=length(Set(fit.state.lineage)), stages=fit.state.stage,
            history_stages=length(history),
            evaluations=sum(stage.evaluations for stage in history; init=0),
            divergences=sum(stage.divergences for stage in history; init=0))
    end
    d = length(names)
    mean_matrix = reduce(hcat, (run.means for run in runs))
    sd_matrix = reduce(hcat, (run.sds for run in runs))
    pooled_mean = [mean(view(mean_matrix, i, :)) for i in 1:d]
    pooled_sd = [sqrt(mean(abs2, view(sd_matrix, i, :))) for i in 1:d]
    ratio(x, s) = s > 0 ? x / s : (x == 0 ? 0.0 : Inf)
    mean_difference = [ratio(maximum(view(mean_matrix, i, :)) - minimum(view(mean_matrix, i, :)), pooled_sd[i]) for i in 1:d]
    sd_ratio_min = [ratio(minimum(view(sd_matrix, i, :)), pooled_sd[i]) for i in 1:d]
    sd_ratio_max = [ratio(maximum(view(sd_matrix, i, :)), pooled_sd[i]) for i in 1:d]
    values = [run.log_normalizer for run in runs]
    worst = d == 0 ? nothing :
        (i = argmax(mean_difference); (; coordinate=names[i], index=i, mean_difference=mean_difference[i]))
    (; names, runs=collect(runs), pooled_mean, pooled_sd, mean_difference, sd_ratio_min, sd_ratio_max,
        log_normalizer=(; values, mean=mean(values), sd=std(values)), worst)
end

function BRM.brm_smc_runs(model::Union{BRM.BRMI,BRM.RKBRMI,BRM._BRMSMCLaplaceProblem}; seeds, particles=1000, reference=nothing,
        options=PPL.SMCOptions(), ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), progress=nothing)
    length(seeds) >= 2 || throw(ArgumentError("brm_smc_runs needs at least two seeds"))
    all(seed -> seed isa Integer, seeds) || throw(ArgumentError("seeds must be integers; each run draws its own Xoshiro from one"))
    rk = smc_model(model)
    fits = [BRM.brm_smc_fit(rk; rng=Xoshiro(seed), particles, reference, options, ad_backend, progress)
        for seed in seeds]
    (; fits, comparison=BRM.brm_smc_compare(fits...))
end

end
