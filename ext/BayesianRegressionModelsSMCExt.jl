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
using Serialization: serialize
using SHA: sha256

# Hash the complete binding, not a truncated display or just its shape.
# The initial template deliberately identifies a fit lineage, including its
# training data. Updates keep this template; callers never author identity IDs.
function binding_identity(rk::BRM.RKBRMI)
    emitted = BRM._rk_retained_program(rk)
    io = IOBuffer()
    serialize(io, (format=1, program=BRM._rk_program_statements(emitted),
        callables=emitted.bindings, data=BRM.rk_bound_data(rk),
        coordinates=BRM._rk_bound_coordinate_names(rk)))
    "brm-smc-v1:" * bytes2hex(sha256(take!(io)))
end

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

function prepare(rk, ad_backend, model_id, binding_id)
    PPL.prepare_smc(rk.model, BRM.rk_bound_plan(rk),
        zeros(Float64, rk.model.layout.total);
        backend=ad_backend, model_id, binding_id)
end

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

function BRM.brm_smc_fit(model::Union{BRM.BRMI,BRM.RKBRMI};
        rng=Xoshiro(), particles=1000, reference=nothing, options=PPL.SMCOptions(),
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), stages=typemax(Int), progress=nothing)
    BRM._brm_with_progress(progress; description="BRM SMC fit") do node
        BRM._brm_throw_if_interrupted(node)
        rk = model isa BRM.RKBRMI ? model : BRM.RKBRMI(model)
        id = binding_identity(rk)
        target = prepare(rk, ad_backend, id, id)
        ref = reference === nothing ? PPL.SMCReference(
            zeros(rk.model.layout.total), ones(rk.model.layout.total)) : reference
        state = PPL.initialize_smc(target; rng, particles, reference=ref, options)
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
        rk = BRM.reprocess(fit.template, cumulative_data; freeze_constants=true)
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
        data=nothing, previous_data=nothing,
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
    state = PPL.resume_smc(target, checkpoint.state; previous=old_target)
    BRM.BRMSMCFit(template, rk, previous, state, ad_backend,
        checkpoint.model_id, checkpoint.binding_id, checkpoint.previous_id)
end

end
