"""
    BRMSMCFit

An RK-only SMC fit, including its frozen training `template`, current `backend`
and RKPPL `state`. For a prepared embedded-Laplace fit, `template` and `backend`
are Laplace problems; the particle rows follow `backend.names`.
`state.positions` contains unconstrained draws as columns;
`state.weights` contains normalized weights. A completed fit has
`state.temperature == 1`. Inspect `state.history` for ESS before resampling,
ancestry and mutation diagnostics; equal final weights do not prove mixing.

Keep model code, bound data and the returned arrays read-only. Continue a
partial fit with [`brm_smc_fit`](@ref), and assimilate cumulative data with
[`brm_smc_update`](@ref).
"""
struct BRMSMCFit{S,A,T,B,P}
    template::T
    backend::B
    previous::P
    state::S
    ad_backend::A
    model_id::String
    binding_id::String
    previous_id::Union{Nothing,String}
end

const _BRMSMCLaplaceProblem = Union{BRMLaplaceProblem,BRMStructuredLaplaceProblem,
    BRMSparseLaplaceProblem,_BRMSampledLaplaceProblem}

# Shared by posterior consumers that must retain the fit's original binding.
function _brm_validate_smc_fit end

"""
    brm_smc_tune(model; rng, init, reference=nothing,
        temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        reference_inflation=nothing, grid_start=nothing, progress=nothing, callback=nothing,
        checkpoint_dir=nothing, resume=false, overwrite=false,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
    brm_smc_tune(completed_fit::BRMSMCFit, cumulative_data; rng, init,
        temperatures=nothing, pilot_steps=64, pilot=nothing,
        max_reflections=nothing, recording_target=nothing, pilot_evaluations=nothing,
        grid_start=nothing, progress=nothing, callback=nothing, checkpoint_dir=nothing,
        resume=false, overwrite=false,
        options=SMCOptions())

Prepare bridge-specific independent pilots before running production particles.
A completed call returns RKPPL's `SMCTuningSchedule`; pass it to `SMCOptions(tuning=schedule)`
in [`brm_smc_fit`](@ref) or [`brm_smc_update`](@ref). Load the same packages as
for fitting. Both explicit `rng::Xoshiro` and unconstrained `init` are copied;
choose them independently of the production cloud, including the retained
cloud of a completed fit. BRM never derives pilot inputs from fitted particles.
`model` may be a prepared Laplace problem; for its updates pass the prepared
cumulative target in place of `cumulative_data`, as in [`brm_smc_update`](@ref).

The default `pilot=:continuous` adapts diagonal scales and step size at each
supplied temperature; `pilot_steps` counts transitions per knot.
`pilot=:adaptive` uses WarmupHMC's adaptive warmup and retains the active
portable scale together with its paired step size; `pilot_steps` is its
retained-draw floor. Optional `max_reflections` and `recording_target` control
geometry and the recording pool; `pilot_evaluations` sets the initial work
window, not a hard total-work budget. `nothing` leaves these controls and
`pilot` at RKPPL's defaults. Pilot draws are discarded; retained pilot
evaluation counts exclude initialization and the first step-size search.

Pilots run one after another by default (`grid_start=nothing`, RKPPL's
`:sequential`): each grid pilot starts at the previous knot's final position,
and no thread count shortens tuning. With `pilot=:adaptive`,
`grid_start=:independent` (ReactiveKernelsPPL 0.1.100) starts every grid pilot
at its own copy of `init` and runs up to `options.nthreads` of them at once,
capped by the available Julia threads and the unfinished knots. The reference
pilot still finishes first. Knot seeds are reserved in grid order, so the
completed schedule is the same for every worker count, but it generally
differs from the sequential schedule. The continuous pilot is sequential only.
Keep `grid_start` unchanged when resuming a tuning checkpoint; `options.nthreads`
may change there.

Production uses the nearest frozen temperature knot,
with ties selecting the higher knot. `temperatures` must increase strictly
from zero to one; `nothing` uses RKPPL's default grid. Include small positive
knots for bridges that change rapidly near zero; a generic default need not
resolve a particular model's curvature. `options`
supplies the starting kernel and production move/depth settings and must not
already contain a schedule. Pilot draws are not posterior samples.

For an initial fit, the normalized reference defaults to standard Normal and
stays fixed. With `pilot=:adaptive, reference=:pilot`, a separate full-target
pilot first estimates the reference mean and, from RKPPL 0.1.47, the spread
of its retained draws in the learned basis multiplied by `reference_inflation`
(RKPPL's default is 2; `nothing` leaves that default; 1 uses the uninflated
moments); that normalized Gaussian is then frozen before temperature-grid
tuning and production. The scale is stored as copied plain data, with no
dense metric. Cross-covariances in the learned basis are discarded, and
inflation does not guarantee posterior coverage: compare independent runs
(`brm_smc_compare`) and read each stage's `reference_sd` beside its
`coordinate_sd`. Omit `reference` in the
subsequent fit to use the schedule's exact reference; an explicit conflicting
reference rejects. For an update, supply
the complete cumulative data. The original preprocessing is frozen and the
previous posterior is the source; there is no initial-reference argument.
Schedules belong to their exact endpoints and cannot silently transfer to new
data. Checkpoints retain the schedule without rerunning its pilots on resume.
The production temperature rule remains adaptive; this does not establish
exactly unbiased evidence at finite particle counts.

With `pilot=:adaptive`, `progress` accepts a Treebars parent and reports nested
tuning, current pilot and sampler progress. `callback(snapshot, boundary)` can
return literal `true` to stop at `:init`, `:window` or `:pilot` boundaries.
The scalar snapshot describes `phase` (`:reference` or `:grid`), `knot`,
`temperature`, `boundary`, `completed_knots`, `window`, `retained_draws` and
`evaluations`. A stop before completion returns RKPPL's `SMCTuningPartial`,
including when no draw exists yet; it cannot tune production. A stop after the
last completed pilot returns the ready schedule. These boundaries are
cooperative: initialization or one window can exceed a wall-clock deadline.

`checkpoint_dir` atomically retains `tuning_latest.jls` before callbacks. Resume
with the original model/data/preprocessing and tuning keywords using either
`resume=partial` (also after trusted Serialization round-trip) or
`checkpoint_dir=dir, resume=true`. Use this same method for initial, prepared
Laplace, cumulative-update and carried-window targets; retain the same previous
fit/carry and coordinate selection for updates. BRM supplies endpoint identities;
RKPPL validates those, configuration and versions, skips completed pilots and
restores the active sampler and RNG. The required `rng` is ignored on resume.
Supply callback/progress afresh. Inputs and supplied partials remain unchanged.
A fresh process rebuilds the model as usual, including models whose laws are
bound in the generated RK program module (such as `gmrf(...)` fields); this
needs ReactiveKernelsPPL 0.1.98 or later. Checkpoints written with an earlier
ReactiveKernelsPPL are not migrated: resume them with their original versions
or write a new initialization checkpoint.

Fresh calls refuse an existing tuning checkpoint unless `overwrite=true`;
overwrite cannot accompany resume. Only the latest outer checkpoint is kept and
progress/history retention is bounded, but active sampler draws can grow.
Completed pilots retain summaries, not draws. A process failure loses work since
the last boundary; without a directory, only an orderly stop yields recoverable
work. Keep Julia/package versions, code, data and configuration unchanged and
deserialize only trusted files. Production [`brm_smc_checkpoint`](@ref) is
separate: it retains an already-completed schedule, not an unfinished pilot.
"""
function brm_smc_tune end

"""
    brm_smc_fit(model; rng=Xoshiro(), particles=1000, reference=nothing,
        options=SMCOptions(), ad_backend=AutoEnzyme(mode=Enzyme.Reverse),
        stages=typemax(Int), progress=nothing)
    brm_smc_fit(fit::BRMSMCFit; stages=typemax(Int), progress=nothing)

Fit a `BRMI`, `RKBRMI` or prepared RK-backed `brm_laplace_problem` through
RKPPL's adaptive SMC bridge, or continue a
partial fit. Load ReactiveKernels, ReactiveKernelsPPL, SequentialMonteCarlo,
WarmupHMC, Enzyme, ADTypes and DifferentiationInterface. The model is evaluated
with its full prior, likelihood and unconstrained-coordinate Jacobian.

For an embedded-Laplace problem, the supplied marginal gradient is consumed
directly, without differentiating through its mode search. Generic, structured,
sparse and zero-integration targets are supported (RKPPL 0.1.57+). Each sampler
execution, pilot and mutation worker owns independent mutable problem
workspaces. `ad_backend` prepares the reference; the supplied problem already
owns its target-gradient implementation. Complete a finished marginal cloud
with `brm_laplace_draws(fit; conditional, rng)`, retaining its returned weights.
The normalizer is for the approximate marginal, and conditional completion
does not correct that outer approximation.

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

For a Laplace fit, pass a prepared next problem in place of `cumulative_data`.
Build it with `brm_laplace_problem(reprocess(training_rk, cumulative_data); ...)`
using the same integration, engine, tolerance and iteration limit. BRM checks
those controls, ordered coordinates and frozen preprocessing before bridging.

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
        data=nothing, previous_data=nothing, nthreads=nothing,
        ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
    brm_smc_resume(laplace_problem, checkpoint;
        current=laplace_problem, previous=nothing, nthreads=nothing,
        ad_backend=AutoEnzyme(mode=Enzyme.Reverse))

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

The checkpoint keeps the worker cap `options.nthreads` it was written with.
`nthreads=n` (ReactiveKernelsPPL 0.1.100) replaces only that cap, for example
`nthreads=Threads.nthreads()` after moving to a process with more threads; the
workers used stay bounded by the available Julia threads and the particle
count. Particle positions, weights, temperatures and the normalizer do not
depend on it; per-worker sums in the stage diagnostics and the recorded worker
count do. The window-checkpoint method `brm_smc_resume(model::BRMI,
checkpoint)` takes the same keyword.

A Laplace checkpoint requires explicitly reconstructed equivalent problems:
`current` is the cumulative endpoint and `previous` the previous endpoint
(defaults to the training template for the first update). Approximation and
solver settings, data, coordinates and preprocessing must match. No problem,
factory or mutable solver workspace is stored in the checkpoint.
"""
function brm_smc_checkpoint end
function brm_smc_resume end

"""
    brm_smc_rescore(template::RKBRMI, checkpoint;
        data=nothing, previous_data=nothing, rtol=1e-10,
        ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
    brm_smc_rescore(model::BRMI, checkpoint;
        rtol=1e-10, ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
    brm_smc_rescore(laplace_problem, checkpoint;
        current=laplace_problem, previous=nothing, rtol=1e-10,
        ad_backend=AutoEnzyme(mode=Enzyme.Reverse))

Recompute a checkpoint's saved endpoint scores on the current host, explicitly
accepting only floating-point drift within `rtol * max(abs(saved_score), 1)`
per score. Requires ReactiveKernelsPPL 0.1.156. All version, model, data,
coordinate and checkpoint-state checks still apply. The reconstruction
arguments match `brm_smc_resume`, including cumulative updates,
carried-reference windows and embedded Laplace problems.

Returns a new checkpoint without changing the input. Each call appends a record
to `checkpoint.state.rescored` with `changed`, `max_abs`, `max_rel`, `rtol` and
`codegen`. Save this returned checkpoint, then pass it to `brm_smc_resume` on
the same host. Resume itself remains exact; it never rescores automatically.
This is not a version migration and does not rescore tuning-pilot checkpoints.
"""
function brm_smc_rescore end

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
