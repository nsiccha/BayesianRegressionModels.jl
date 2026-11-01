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
    brm_smc_tune(model; rng, init, reference=nothing,
        temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
    brm_smc_tune(completed_fit::BRMSMCFit, cumulative_data; rng, init,
        temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        options=SMCOptions())

Prepare a bridge-specific RKPPL `SMCTuningSchedule` with independent pilots,
before running production particles. Pass it to `SMCOptions(tuning=schedule)`
in [`brm_smc_fit`](@ref) or [`brm_smc_update`](@ref). Load the same packages as
for fitting. Both explicit `rng::Xoshiro` and unconstrained `init` are copied;
choose them independently of the production cloud, including the retained
cloud of a completed fit. BRM never derives pilot inputs from fitted particles.

The default `pilot=:continuous` adapts diagonal scales and step size at each
supplied temperature; `pilot_steps` counts transitions per knot.
`pilot=:adaptive` uses WarmupHMC's adaptive warmup and retains the active
portable scale together with its paired step size; `pilot_steps` is its
retained-draw floor. Optional `max_reflections` and `recording_target` control
geometry and the recording pool; `pilot_evaluations` sets the initial work
window, not a hard total-work budget. `nothing` leaves these controls and
`pilot` at RKPPL's defaults. Pilot draws are discarded; retained pilot
evaluation counts exclude initialization and the first step-size search.

Production uses the nearest frozen temperature knot,
with ties selecting the higher knot. `temperatures` must increase strictly
from zero to one; `nothing` uses RKPPL's default grid. Include small positive
knots for bridges that change rapidly near zero; a generic default need not
resolve a particular model's curvature. `options`
supplies the starting kernel and production move/depth settings and must not
already contain a schedule. Pilot draws are not posterior samples.

For an initial fit, the normalized reference defaults to standard Normal and
stays fixed. With `pilot=:adaptive, reference=:pilot`, a separate full-target
pilot first estimates the reference mean and scale; that normalized Gaussian
is then frozen before temperature-grid tuning and production. The scale is
stored as copied plain data, with no dense metric. Omit `reference` in the
subsequent fit to use the schedule's exact reference; an explicit conflicting
reference rejects. For an update, supply
the complete cumulative data. The original preprocessing is frozen and the
previous posterior is the source; there is no initial-reference argument.
Schedules belong to their exact endpoints and cannot silently transfer to new
data. Checkpoints retain the schedule without rerunning its pilots on resume.
The production temperature rule remains adaptive; this does not establish
exactly unbiased evidence at finite particle counts.
"""
function brm_smc_tune end

"""
    brm_smc_fit(model; rng=Xoshiro(), particles=1000, reference=nothing,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse),
        stages=typemax(Int), progress=nothing)
    brm_smc_fit(fit::BRMSMCFit; stages=typemax(Int), progress=nothing)

Fit a `BRMI` or `RKBRMI` through RKPPL's adaptive SMC bridge, or continue a
partial fit. Load ReactiveKernels, ReactiveKernelsPPL, SequentialMonteCarlo,
WarmupHMC, Enzyme, ADTypes and DifferentiationInterface. The model is evaluated
with its full prior, likelihood and unconstrained-coordinate Jacobian.

`reference` is RKPPL's `SMCReference(location, scale)`: a normalized Gaussian
in unconstrained coordinates. `scale` accepts a vector of diagonal scales or
a portable `WarmupHMC.scale_data(L)` record. The default is standard Normal or
the exact reference retained by `options.tuning`. These
are reference draws, not model-prior draws. Its actual density is included in
the importance ratio. `options` is RKPPL's `SMCOptions`; mutation uses fixed
NUTS, with diagonal scales or `metric=WarmupHMC.scale_data(L)`.
With `options.tuning=brm_smc_tune(...)`, its settings vary
between stages according to the precomputed schedule; they stay fixed during
each stage. History retains the selected kernel, evaluation count, mean
acceptance and mean squared unconstrained movement. These diagnostics do not prove adequate
exploration. The explicit `Random.Xoshiro` is copied, not advanced.

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

If the completed fit used tuning, its schedule belongs to the old bridge and
the default `options` rejects. Supply a new `brm_smc_tune(fit, cumulative_data;
rng, init, ...)` schedule in `SMCOptions(tuning=...)`, or explicitly choose
untuned `SMCOptions()`.
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

"""
    brm_smc_compare(fits::BRMSMCFit...)
    brm_smc_runs(model; seeds, particles=1000, reference=nothing,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse), progress=nothing)

Compare two or more completed fits of the same model, data and, for updates,
the same previous posterior; fits of different identities, or fits that have
not reached temperature one, are refused by name. Nothing is fitted or
reweighted: the comparison summarizes the retained clouds. Per unconstrained
coordinate (`names`) it returns each run's weighted mean and SD (`runs`), the
pooled mean and the root mean within-run variance (`pooled_mean`,
`pooled_sd`), the largest difference between run means in pooled-SD units
(`mean_difference`), the smallest and largest run SD in pooled-SD units
(`sd_ratio_min`, `sd_ratio_max`), and the coordinate with the largest mean
difference (`worst`). `log_normalizer` holds the runs' evidence estimates with
their mean and sample SD. Each run also reports its surviving initial
lineages, this bridge's stage count, and the evaluations and divergences summed
over its retained history, which `history_limit` may have truncated.

These are descriptive agreement statistics between independent runs, not
Monte Carlo standard errors and not an acceptance gate: close agreement does
not prove the runs explored the posterior, and disagreement localizes which
coordinates to inspect. `brm_smc_runs` fits one bridge per integer seed with
the same settings as [`brm_smc_fit`](@ref), in sequence, and returns
`(; fits, comparison)`.
"""
function brm_smc_compare end
function brm_smc_runs end
