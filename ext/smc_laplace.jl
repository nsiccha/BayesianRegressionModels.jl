# Prepared embedded marginals use the same producer SMC engine as RKBRMI.
# The backend and data are read-only. Every execution owns the Laplace modes,
# derivative buffers and factors (including the sparse factor's native handles).
laplace_backend(p::BRM.BRMLaplaceProblem) = p.model.backend
laplace_backend(p::BRM._BRMSMCLaplaceProblem) = p.backend

function copy_laplace(p)
    rk = laplace_backend(p)
    Base.deepcopy_internal(p, IdDict{Any,Any}(rk => rk))
end
smc_model(p::BRM._BRMSMCLaplaceProblem) = copy_laplace(p)
smc_names(p::BRM._BRMSMCLaplaceProblem) = copy(p.names)

laplace_partition(p::BRM.BRMLaplaceProblem) = p.model.view.subject_coordinates
laplace_partition(p::BRM.BRMStructuredLaplaceProblem) = p.blocks
laplace_partition(p::Union{BRM.BRMSparseLaplaceProblem,BRM._BRMSampledLaplaceProblem}) =
    (; p.integrate, p.integrated, declarations=[b.declaration for b in p.blocks],
       p.components)

laplace_base_configuration(p) = (; engine=Symbol(nameof(typeof(p))), p.tol, p.maxiter,
    partition=laplace_partition(p))
laplace_configuration(p) = laplace_base_configuration(p)
function laplace_configuration(p::BRM.BRMSparseLaplaceProblem)
    base = laplace_base_configuration(p)
    p.algebra === nothing && return base
    (; base..., algebra=(; p.algebra.solver, p.algebra.options))
end

# The fixed probes/reference define the finite target. Fingerprint their
# values, excluding scratch buffers and diagnostics that evaluations mutate.
# Encode sparse factors by their stored entries, never as dense arrays.
laplace_numerical_binding(p) = nothing
function laplace_numerical_binding(p::BRM.BRMSparseLaplaceProblem)
    p.algebra === nothing && return nothing
    references = map(BRM._brm_reference_binding, p.algebra.references)
    (; p.algebra.large, p.algebra.probes, references)
end

# Share the extension's value encoder with carried-window identities. It
# ignores object sharing and dictionary traversal, retaining type and shape.
function laplace_identity(x)
    io = IOBuffer()
    streaming_value!(io, x)
    "brm-smc-laplace-v1:" * bytes2hex(sha256(take!(io)))
end

function laplace_preprocessing(rk)
    data = BRM.rk_bound_data(rk)
    inputs = Dict(name => getproperty(data, name)
        for name in keys(BRM._rk_plan_fit_inputs(rk.plan)) if hasproperty(data, name))
    # A fit may retain training inputs (CAR adjacency, thin-plate knots, ...)
    # that cannot be exported by brm_fitted_state. SMC does not require a
    # data-free export: fingerprint those frozen bindings directly. A backend
    # reconstructed from a fitted state instead holds frozen summary values.
    frozen = rk.fitted === nothing ? nothing : rk.fitted.frozen
    (; inputs, frozen, fits=BRM._rk_recorded_fits(rk),
       constants=BRM._rk_fitted_constants(rk), held_out=BRM._rk_held_out_names(rk.plan))
end

function laplace_frame(p)
    rk = laplace_backend(p)
    emitted = BRM._rk_retained_program(rk)
    (; program=BRM._rk_program_statements(emitted),
       callables=BRM._brm_stable_bindings(emitted.bindings),
       preprocessing=laplace_preprocessing(rk), p.population, p.names,
       configuration=laplace_configuration(p))
end
function binding_identity(p::BRM._BRMSMCLaplaceProblem)
    binding = (; frame=laplace_frame(p), data=BRM.rk_bound_data(laplace_backend(p)))
    numerical = laplace_numerical_binding(p)
    laplace_identity(numerical === nothing ? binding : (; binding..., numerical))
end

function prepare(p::BRM._BRMSMCLaplaceProblem, ad_backend, model_id, binding_id)
    PPL.prepare_smc(() -> copy_laplace(p), zeros(LogDensityProblems.dimension(p));
        backend=ad_backend, model_id, binding_id, frame_id=laplace_identity(laplace_frame(p)))
end

function smc_update_model(template::BRM._BRMSMCLaplaceProblem,
        next::BRM._BRMSMCLaplaceProblem)
    laplace_identity(laplace_frame(template)) == laplace_identity(laplace_frame(next)) ||
        throw(ArgumentError("BRM SMC: the Laplace model, integration, solver controls or " *
            "coordinate/preprocessing frame differs; rebuild from frozen reprocess " *
            "of the training backend with the same integration and solver settings"))
    copy_laplace(next)
end
function smc_update_model(::BRM._BRMSMCLaplaceProblem, data)
    throw(ArgumentError("BRM SMC: a Laplace update requires a prepared brm_laplace_problem " *
        "over the complete cumulative data, built from frozen reprocess of the training backend"))
end

function BRM.brm_smc_resume(template::BRM._BRMSMCLaplaceProblem, checkpoint;
        current=template, previous=nothing, nthreads=nothing,
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    checkpoint.format == 1 || throw(ArgumentError("unsupported BRM SMC checkpoint format"))
    check_identity(template, checkpoint.model_id, "training Laplace target")
    rk = smc_update_model(template, current)
    check_identity(rk, checkpoint.binding_id, "current Laplace target")
    old = if checkpoint.previous_id === nothing
        previous === nothing || throw(ArgumentError("an initial-fit checkpoint has no previous target"))
        nothing
    else
        p = smc_update_model(template, previous === nothing ? template : previous)
        check_identity(p, checkpoint.previous_id, "previous Laplace target")
        p
    end
    target = prepare(rk, ad_backend, checkpoint.model_id, checkpoint.binding_id)
    old_target = old === nothing ? nothing :
        prepare(old, ad_backend, checkpoint.model_id, checkpoint.previous_id)
    state = PPL.resume_smc(target, checkpoint.state; previous=old_target, nthreads)
    BRM.BRMSMCFit(copy_laplace(template), rk, old, state, ad_backend,
        checkpoint.model_id, checkpoint.binding_id, checkpoint.previous_id)
end

"""
    brm_laplace_draws(fit::BRMSMCFit; rng=Xoshiro(), conditional=0)

Complete a finished embedded-Laplace SMC fit. Return full packed `draws`, their
normalized outer `weights`, and conditional `diagnostics`. Every population
column and weight is preserved; use the weights when summarizing these draws.
`conditional` controls importance resampling of the integrated effects and
does not correct the approximate outer marginal. The fit, its workspaces and
an explicitly supplied RNG are left unchanged.
"""
function BRM.brm_laplace_draws(fit::BRM.BRMSMCFit{S,A,T}; rng=Xoshiro(),
        conditional::Integer=0) where {S,A,T<:BRM._BRMSMCLaplaceProblem}
    BRM._brm_validate_smc_fit(fit)
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM SMC: complete the marginal bridge before completing conditional draws"))
    out = BRM.brm_laplace_draws(copy_laplace(fit.backend), fit.state.positions;
        rng=copy(rng), conditional)
    (; out.draws, weights=copy(fit.state.weights), out.diagnostics)
end

# A marginal cloud has no full-model parameter projection: integrated effects
# must first be completed, and a new-window marginal needs its own likelihood.
function BRM.brm_smc_carry(fit::BRM.BRMSMCFit{S,A,T}; kwargs...) where
        {S,A,T<:BRM._BRMSMCLaplaceProblem}
    throw(ArgumentError("BRM SMC: carrying a Laplace marginal into an RK full-model " *
        "window is not supported; use cumulative prepared-target brm_smc_update"))
end

function BRM._brm_ladder_samples(fit::BRM.BRMSMCFit{S,A,T},
        p::BRM._BRMSMCLaplaceProblem, rng) where {S,A,T<:BRM._BRMSMCLaplaceProblem}
    BRM._brm_validate_smc_fit(fit)
    check_identity(p, fit.binding_id, "ladder Laplace target")
    fit.state.temperature == 1 || throw(ArgumentError(
        "BRM ladder: the SMC callback must return a completed marginal bridge"))
    indices = sample(rng, axes(fit.state.positions, 2), Weights(fit.state.weights),
        size(fit.state.positions, 2))
    (; positions=fit.state.positions[:, indices], method=:smc)
end
