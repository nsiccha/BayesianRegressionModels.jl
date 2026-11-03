"""
    BRMFilter

An RK-only finite-state filter. `state` exposes the conditional particle clouds,
fixed-parameter masses, predictive log score, accumulated log normalizer and
bounded diagnostics. The filter retains its compiled target and finite state,
not its original `RKBRMI`, training rows or consumed stream.

Treat the target, state and returned arrays as read-only. Each successful step
returns a new filter; a failed step leaves its input usable.
"""
struct BRMFilter{T,S,C}
    target::T
    state::S
    configuration::C
    model_id::String
end

"""
    brm_filter(model; parameters=nothing, ensemble=nothing, weights=nothing,
        inputs=(), groups=(;), drifting=(), rng=Xoshiro(), particles=1000,
        history_limit=32)
    brm_filter(fit::Union{BRMSMCFit,BRMSMCWindowFit}; learn=false, kwargs...)

Initialize filtering before the first authored time slice of a `BRMI` or
`RKBRMI`. Load ReactiveKernels, ReactiveKernelsPPL and SequentialMonteCarlo.
Recurrences and observation laws come from the model's captured RKPPL source;
preprocessing remains frozen at its training values.

Supply a calibrated constrained parameter named tuple/dict, an unconstrained
coordinate vector, or an `ensemble` of such points. RKPPL selects only static
parameters, excluding historical latent seeds, innovations and trajectories.
An SMC fit supplies its completed posterior particles and weights. `particles`
is the number of conditional state particles **per parameter member**. A fixed
ensemble can lose parameter diversity. With `learn=true`, the completed fit
instead initializes a `BRMFilterSMC` through its selected static Gaussian
summary. This route adds `members`, `window_length`, `ess_fraction`, `options`
and `boundary` controls for bounded-window replenishment and rejuvenation.
Its Gaussian/state-pool approximation is documented on `BRMFilterSMC`.

`inputs` names the per-step covariate ports. `groups` fixes the authored plate
axis, and `drifting` names sampled carried states with an explicit authored
parameter transition. No parameter jitter is introduced automatically.

Initialization is cold even when parameters came from a posterior. To continue
after its training history, warm the states with
`brm_filter_stream(filter, training_records; condition_state_only=true)`;
this preserves calibrated parameter masses and does not count that likelihood
again. A joint checkpoint resumes directly without warm-up.

The initial RKPPL implementation supports scalar Float64 carries, bounded
literal lags and one fixed group plate. It refuses whole-history reductions,
growing state and unsupported trajectory transformations. The explicit Xoshiro
RNG is copied. Opaque callables must themselves have bounded work and storage.
"""
function brm_filter end

"""
    brm_filter_step(filter::BRMFilter, observation::NamedTuple;
        inputs=(;), groups=nothing, condition_state_only=false)

Advance one time slice. The new state's `predictive_logscore` scores the
observation before conditioning; `log_normalizer` accumulates scores from
ordinary steps. State-only warm-up preserves both the accumulated normalizer
and calibrated parameter masses. Missing observations advance the state
without conditioning. A selected group advances on its own clock; omitted
groups do not advance. Mixed global/group recurrences require synchronous
slices. Group-indexed arrays retain their original positions.
"""
function brm_filter_step end

"""
    brm_filter_stream(filter::BRMFilter, records; steps=typemax(Int),
        condition_state_only=false, progress=nothing, stop=filter->false)

Consume records `(observation=(;y=...), inputs=(;x=...), groups=nothing)`;
`inputs` and `groups` can be omitted. Check cooperative interruption and `stop`
before requesting each record, and return after the last completed step.
Use an `Iterators.Stateful` iterator to resume consumption of a live stream.
An ordinary filter retains no records or iterator. For `BRMFilterSMC`, the
producer retains at most `window_length` records for rejuvenation;
`condition_state_only=true` requires an ordinary filter first, then transfer
through `boundary`.
"""
function brm_filter_stream end

"""
    brm_filter_checkpoint(filter::BRMFilter)
    brm_filter_resume(model::Union{BRMI,RKBRMI}, checkpoint)

Copy a data-only checkpoint, including the finite state, RNG, bounded history
and preparation configuration. Restore against the same authored model and
frozen preprocessing, explicitly rebuilt after restart. Neither training rows
nor compiled functions are saved. Resume checks model/preprocessing identity
and RKPPL's configuration, package-version and state-shape checks.

Use Julia Serialization only with trusted checkpoints. File writing and atomic
replacement belong to the caller. Exact continuation requires the same code,
Julia/package versions and fixed model inputs.
"""
function brm_filter_checkpoint end
function brm_filter_resume end
