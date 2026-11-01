"""
    BRMSMCFit

An RK-only SMC fit, including its frozen training `template`, current `backend`
and RKPPL `state`. `state.positions` contains unconstrained draws as columns;
`state.weights` contains normalized weights. A completed fit has
`state.temperature == 1`. Inspect `state.history` for ESS before resampling,
ancestry and mutation diagnostics; equal final weights do not prove mixing.

Keep model code, bound data and the returned arrays read-only. Continue a
partial fit with [`brm_smc_fit`](@ref), and assimilate cumulative data with
[`brm_smc_update`](@ref).
"""
struct BRMSMCFit{S,A}
    template::RKBRMI
    backend::RKBRMI
    previous::Union{Nothing,RKBRMI}
    state::S
    ad_backend::A
    model_id::String
    binding_id::String
    previous_id::Union{Nothing,String}
end

# Shared by posterior consumers that must retain the fit's original binding.
function _brm_validate_smc_fit end

"""
    brm_smc_fit(model; rng=Xoshiro(), particles=1000, reference=nothing,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse),
        stages=typemax(Int), progress=nothing)
    brm_smc_fit(fit::BRMSMCFit; stages=typemax(Int), progress=nothing)

Fit a `BRMI` or `RKBRMI` through RKPPL's adaptive SMC bridge, or continue a
partial fit. Load ReactiveKernels, ReactiveKernelsPPL, SequentialMonteCarlo,
WarmupHMC, Enzyme, ADTypes and DifferentiationInterface. The model is evaluated
with its full prior, likelihood and unconstrained-coordinate Jacobian.

`reference` is RKPPL's `SMCReference(location, scale)`: a normalized diagonal
Gaussian in unconstrained coordinates, defaulting to standard Normal. These
are reference draws, not model-prior draws. Its actual density is included in
the importance ratio. `options` is RKPPL's `SMCOptions`; mutation uses fixed
diagonal-scale NUTS. The explicit `Random.Xoshiro` is copied, not advanced.

`stages` bounds the number of complete bridge stages in this call (zero only
initializes). `progress` accepts the usual BRM Treebars node or backend symbol.
An interrupt returns the last completed stage; resume the returned fit.
No other backend is fitted by this entry point.

Numerical failures propagate `ReactiveKernelsPPL.SMCNonfiniteError` (RKPPL
0.1.37+), with owned `position` and optional `gradient` vectors, `value`,
`quantity`, `endpoint` and `temperature`. Its compact display does not replace
the complete numerical fields. NaN/+Inf densities and nonfinite gradients at
finite density fail fast; `-Inf` still denotes points outside support. A failed
continuation leaves the input fit and its RNG unchanged, so it can be inspected
or checkpointed. This exception also propagates from [`brm_smc_update`](@ref).
"""
function brm_smc_fit end

"""
    brm_smc_update(fit::BRMSMCFit, cumulative_data;
        options=fit.state.options, stages=typemax(Int), progress=nothing)

Update a completed fit to the posterior for `cumulative_data`, which must
contain the complete data intended for the new posterior, not just the new
rows. Replays the original training template with frozen preprocessing and
the same coordinates. New group levels and incompatible layouts are refused.

The bridge is from the previous posterior to the new posterior: the prior
and Jacobian occur once. Both histories are evaluated after parameter moves,
including deterministic recurrences. This is sequential parameter inference
with full-history replay, not constant-cost streaming or latent-state filtering.
Returns a new fit; its input remains usable. A partial update is continued with
`brm_smc_fit`, not by starting another update.
"""
function brm_smc_update end

"""
    brm_smc_checkpoint(fit::BRMSMCFit)
    brm_smc_resume(template::RKBRMI, checkpoint;
        data=nothing, previous_data=nothing, ad_backend=AutoEnzyme(mode=Enzyme.Reverse))

Create a data-only checkpoint or restore it against an explicitly rebuilt
original training `template`. Persist the checkpoint with Julia Serialization
to a trusted file. It contains particle state and identity checks, not training
data, compiled targets or AD caches. Retain the model code and all required data.

For a checkpoint after a data update, `data` supplies the current cumulative
data and `previous_data` supplies the previous posterior's data. Omit
`previous_data` when that posterior is the original training fit. Resume checks
the training template, current and previous bindings, and RKPPL's program,
layout and endpoint-density checks. It restores without advancing; continue
with `brm_smc_fit`. Exact replay requires the same Julia/package versions,
model code, data and configuration. File writing and atomic replacement belong
to the caller.
"""
function brm_smc_checkpoint end
function brm_smc_resume end
