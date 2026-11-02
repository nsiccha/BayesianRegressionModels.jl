# A filter needs frozen graph values and constants, not the calibration's
# historical unconstrained layout. Native selected geometry lives in RKPPL.
struct _BRMFilterPreprocessing{F,C}
    frozen::F
    constants::C
    observations::Tuple
    inputs::Tuple
end

"""
    BRMFilterSMC

Bounded-window parameter learning over an authored finite-state filter.
Construct with `brm_filter(completed_smc_fit; learn=true, ...)`, or with a
`brm_smc_carry(fit; parameters=:filter, inputs=...)` calibration and a model.
The native parameter ensemble, conditional state clouds and bounded window
live in `state`. The calibration retains no historical latent coordinates.
`state.positions` contains unconstrained member columns and
`state.parameter_mass` their normalized weights. `state.records` is the bounded
window; `state.history` reports ESS, replenishment and proposal diagnostics.
Accumulate each step's latest `replenished` flag to count lifetime events;
summing bounded history counts only its retained steps.

Window boundaries replace the posterior with independent Gaussian parameter
marginals and a weighted state pool. This assumed-density approximation loses
parameter correlations and parameter–state dependence; it is not exact
full-history SMC². Treat all fields as read-only.
"""
struct BRMFilterSMC{T,S,C,R}
    target::T
    state::S
    configuration::C
    calibration::R
    model_id::String
end

"""
    brm_filter_tune(fit; rng, init, inputs=(), groups=(;), boundary=nothing, ...)
    brm_filter_tune(calibration, model; rng, init, ...)

Tune an independent fixed NUTS proposal for filter parameter learning, using
RKPPL's `tune_filter_smc`. `init` is in the selected static unconstrained frame.
Pass the resulting schedule as `SMCOptions(tuning=schedule)` to `brm_filter`.
Only the reference is differentiated; particle resampling is not. The final
temperature's kernel stays fixed throughout the stream, including boundaries.
"""
function brm_filter_tune end
function _brm_filter_learn end
function _brm_filter_learning_resume end
