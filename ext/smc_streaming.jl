# Included inside BayesianRegressionModelsSMCExt. Sampler mechanics and the
# carried Gaussian live in RKPPL; this file only binds BRM models and identity.

# Serialization is a transport format, not a value fingerprint: it preserves
# object sharing, dictionary iteration order and padding in isbits arrays.
# Walk the data-only records by value, with a fresh serializer for each leaf.
# Type/shape tags keep distinct coordinate records distinct; numerical arrays
# retain the bulk path only when their elements have no struct padding.
function streaming_value!(io, value)
    write(io, 0x01)
    serialize(io, typeof(value))
    if isprimitivetype(typeof(value))
        serialize(io, value)
        return nothing
    end
    for name in fieldnames(typeof(value))
        streaming_value!(io, getfield(value, name))
    end
    nothing
end

function streaming_value!(io, value::Union{Number,Char,Symbol,AbstractString,Type,Module,Nothing,Missing,GlobalRef})
    write(io, 0x00)
    serialize(io, value)
    nothing
end

function streaming_value!(io, value::AbstractArray)
    write(io, 0x02)
    serialize(io, typeof(value))
    serialize(io, map(axis -> (first(axis), length(axis)), axes(value)))
    if isprimitivetype(eltype(value))
        # Materializing also excludes a view's parent and indexing machinery.
        serialize(io, collect(value))
    else
        for element in value
            streaming_value!(io, element)
        end
    end
    nothing
end

function streaming_value!(io, value::AbstractDict)
    write(io, 0x03)
    serialize(io, typeof(value))
    serialize(io, length(value))
    entries = map(collect(pairs(value))) do (key, val)
        keyio = IOBuffer()
        streaming_value!(keyio, key)
        (take!(keyio), val)
    end
    sort!(entries; by=entry -> bytes2hex(first(entry)))
    for (key, val) in entries
        serialize(io, length(key))
        write(io, key)
        streaming_value!(io, val)
    end
    nothing
end

function streaming_identity(value)
    io = IOBuffer()
    streaming_value!(io, value)
    "brm-smc-window-v2:" * bytes2hex(sha256(take!(io)))
end

function streaming_source(rk, preprocessing)
    emitted = BRM._rk_retained_program(rk)
    streaming_identity((; program=BRM._rk_program_statements(emitted),
        callables=BRM._brm_stable_bindings(emitted.bindings), preprocessing))
end

function check_carried(carried::BRM.BRMSMCCarriedReference)
    identity = streaming_identity((; carried.posterior, carried.preprocessing,
        carried.model_id, carried.source_id))
    identity == carried.identity || throw(ArgumentError(
        "BRM SMC: the carried posterior or frozen preprocessing has changed"))
    nothing
end

function BRM._brm_validate_smc_fit(fit::BRM.BRMSMCWindowFit)
    check_carried(fit.previous)
    fit.model_id == fit.previous.model_id || throw(ArgumentError(
        "BRM SMC: the window belongs to another carried model"))
    check_identity(fit.backend, fit.binding_id, "current window binding")
    streaming_source(fit.backend, fit.previous.preprocessing) == fit.previous.source_id ||
        throw(ArgumentError("BRM SMC: the window source or frozen preprocessing has changed"))
    nothing
end

fitted_preprocessing(fit::BRM.BRMSMCFit) = BRM.brm_fitted_state(fit.template)
fitted_preprocessing(fit::BRM.BRMSMCWindowFit) = fit.previous.preprocessing

function check_selective_window(fit::BRM.BRMSMCWindowFit)
    fit.coordinates === :auto || throw(ArgumentError(
        "BRM SMC: coordinate inspection and recovery require coordinates=:auto"))
    BRM._brm_validate_smc_fit(fit)
    nothing
end

function BRM.brm_smc_coordinates(fit::BRM.BRMSMCWindowFit)
    check_selective_window(fit)
    PPL.smc_window_coordinates(fit.state)
end

function BRM.brm_smc_recover(fit::BRM.BRMSMCWindowFit, compact_position; rng=Xoshiro())
    check_selective_window(fit)
    PPL.smc_recover(fit.state, compact_position; rng)
end

function carried_reference(rk, posterior, preprocessing, model_id)
    source_id = streaming_source(rk, preprocessing)
    identity = streaming_identity((; posterior, preprocessing, model_id, source_id))
    BRM.BRMSMCCarriedReference(posterior, preprocessing, model_id, source_id, identity)
end

function BRM.brm_smc_carry(fit::Union{BRM.BRMSMCFit,BRM.BRMSMCWindowFit};
        scale=nothing, basis=nothing, parameters=nothing, inputs=(),
        max_reflections=16, maxiter=100, progress=nothing)
    BRM._brm_validate_smc_fit(fit)
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM SMC: complete the posterior bridge before carrying its summary"))
    parameters === :filter && return filter_calibration(fit; scale, basis, inputs,
        max_reflections, maxiter, progress)
    parameters === nothing || throw(ArgumentError(
        "BRM SMC: parameters must be nothing or :filter"))
    isempty(inputs) || throw(ArgumentError("BRM SMC: inputs requires parameters=:filter"))
    preprocessing = deepcopy(fitted_preprocessing(fit))
    posterior = PPL.carry_smc(fit.state; calibration_id=fit.binding_id, scale, basis,
        max_reflections, maxiter, progress)
    carried_reference(fit.backend, posterior, preprocessing, fit.model_id)
end

function BRM.brm_smc_carry(model::Union{BRM.BRMI,BRM.RKBRMI};
        reference::PPL.SMCReference, ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    rk = model isa BRM.RKBRMI ? model : BRM.RKBRMI(model)
    id = binding_identity(rk)
    target = prepare(rk, ad_backend, id, id)
    preprocessing = deepcopy(BRM.brm_fitted_state(rk))
    posterior = PPL.carry_smc(target, reference; calibration_id=id)
    carried_reference(rk, posterior, preprocessing, id)
end

function prepare_window(carried::BRM.BRMSMCCarriedReference, model::BRM.BRMI, ad_backend;
        coordinates=nothing)
    coordinates in (nothing, :auto) || throw(ArgumentError(
        "BRM SMC: coordinates must be nothing or :auto"))
    check_carried(carried)
    carried.preprocessing isa BRM.BRMFittedState || throw(ArgumentError(
        "BRM SMC: a selected filter calibration requires brm_filter, not a full-frame window bridge"))
    rk = coordinates === :auto ? BRM._rk_fitted_replay(model,
        BRM._rk_brmi_data(model), carried.preprocessing;
        coordinates=:growing, prefix="BRM SMC growing window") :
        BRM.RKBRMI(model, carried.preprocessing)
    streaming_source(rk, carried.preprocessing) == carried.source_id || throw(ArgumentError(
        "BRM SMC: the window source or frozen preprocessing differs from the carried posterior"))
    id = binding_identity(rk)
    selection = coordinates === nothing ? (;) : (; coordinates)
    target = PPL.prepare_smc_window(rk.model, BRM.rk_bound_plan(rk),
        zeros(Float64, rk.model.layout.total); previous=carried.posterior,
        backend=ad_backend, model_id=carried.model_id, binding_id=id, selection...)
    rk, id, target
end

"""
    brm_smc_tune(carried::BRMSMCCarriedReference, model::BRMI;
        rng, init, coordinates=nothing, ...)

Tune independent pilot kernels for the new window under the carried posterior
and frozen preprocessing. Controls have the same meaning as the initial-fit
`brm_smc_tune` method. Use the returned schedule in `SMCOptions(tuning=...)`
for this exact window and the same `coordinates` selection used by
[`brm_smc_update`](@ref). `reference=:pilot` learns an importance reference for
the new target; it does not replace the carried posterior in that target.
"""
function BRM.brm_smc_tune(carried::BRM.BRMSMCCarriedReference, model::BRM.BRMI;
        rng, init, reference=nothing, temperatures=nothing, pilot_steps=64,
        pilot=nothing, max_reflections=nothing, recording_target=nothing,
        pilot_evaluations=nothing, reference_inflation=nothing, grid_start=nothing,
        progress=nothing, callback=nothing, checkpoint_dir=nothing, resume=false,
        overwrite=false,
        options=PPL.SMCOptions(), coordinates=nothing,
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    _, _, target = prepare_window(deepcopy(carried), model, ad_backend; coordinates)
    controls = tuning_controls(; temperatures, pilot, max_reflections,
        recording_target, pilot_evaluations, reference_inflation, grid_start)
    PPL.tune_smc(target; rng, init, reference, controls..., pilot_steps, options,
        progress, callback, checkpoint_dir, resume, overwrite)
end

"""
    brm_smc_update(carried::BRMSMCCarriedReference, model::BRMI;
        coordinates=nothing, rng, particles=1000, options=SMCOptions(), ...)

Assimilate this window's new observations under a carried posterior and its
frozen preprocessing. The default `coordinates=nothing` uses the complete,
unchanged parameter frame.

Use `coordinates=:auto` for a selective growing window. RKPPL discovers old
coordinates needed by the likelihood, new coordinates, and the old parents
needed by those new coordinates' priors. The target is the carried marginal
on the selected old coordinates times the new coordinates' authored priors
and this window's likelihood. Old priors are not counted again. Fresh priors
include their transform Jacobians; the likelihood retains the model's
constrained transforms.

The particle state contains only selected old and new coordinates. The
remaining old coordinates retain their conditional distribution given the
selected old coordinates; their marginal can therefore change when they
are correlated. [`brm_smc_carry`](@ref) combines this conditional with the
completed compact posterior approximation for the next window.

The authored declaration and frozen preprocessing must stay the same.
Data-driven declared-vector extents may grow; fitted group levels and the
identity of existing coordinates must be preserved. Checkpoints remember
the coordinate mode and require the same window on resume.
"""
function BRM.brm_smc_update(carried::BRM.BRMSMCCarriedReference, model::BRM.BRMI;
        rng=Xoshiro(), particles=1000, options=PPL.SMCOptions(), coordinates=nothing,
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), stages=typemax(Int), progress=nothing)
    BRM._brm_with_progress(progress; description="BRM SMC new-window update") do node
        BRM._brm_throw_if_interrupted(node)
        # Own the retained summary so later caller edits cannot change a fit.
        previous = deepcopy(carried)
        rk, id, target = prepare_window(previous, model, ad_backend; coordinates)
        state = PPL.initialize_smc(target; rng, particles, options)
        state = finish(state, stages, node)
        BRM.BRMSMCWindowFit(rk, previous, state, ad_backend, previous.model_id, id, coordinates)
    end
end

function BRM.brm_smc_fit(fit::BRM.BRMSMCWindowFit; stages=typemax(Int), progress=nothing)
    BRM._brm_validate_smc_fit(fit)
    BRM._brm_with_progress(progress; description="BRM SMC window continuation") do node
        state = finish(fit.state, stages, node)
        BRM.BRMSMCWindowFit(fit.backend, fit.previous, state,
            fit.ad_backend, fit.model_id, fit.binding_id, fit.coordinates)
    end
end

function BRM.brm_smc_checkpoint(fit::BRM.BRMSMCWindowFit)
    BRM._brm_validate_smc_fit(fit)
    (; format=2, kind=:window, model_id=fit.model_id, binding_id=fit.binding_id,
       previous=deepcopy(fit.previous), coordinates=fit.coordinates,
       state=PPL.smc_checkpoint(fit.state))
end

function BRM.brm_smc_resume(model::BRM.BRMI, checkpoint;
        nthreads=nothing, ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    checkpoint.format == 2 && checkpoint.kind === :window || throw(ArgumentError(
        "BRM SMC: a window declaration requires a window checkpoint"))
    previous = deepcopy(checkpoint.previous)
    previous.model_id == checkpoint.model_id || throw(ArgumentError(
        "BRM SMC: the checkpoint belongs to another carried model"))
    coordinates = get(checkpoint, :coordinates, nothing)
    rk, id, target = prepare_window(previous, model, ad_backend; coordinates)
    id == checkpoint.binding_id || throw(ArgumentError(
        "BRM SMC: the checkpoint's window data has changed"))
    state = PPL.resume_smc(target, checkpoint.state; nthreads)
    BRM.BRMSMCWindowFit(rk, previous, state, ad_backend, previous.model_id, id, coordinates)
end
