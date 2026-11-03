"""
    BRMSMCCarriedReference

A data-only posterior summary for new-window SMC: RKPPL's owned normalized
Gaussian reference plus BRM's frozen preprocessing and source identity. It
holds no old window, particle cloud, compiled target or AD workspace.
Construct it with [`brm_smc_carry`](@ref), and treat its contents as read-only.
"""
struct BRMSMCCarriedReference{C,P}
    posterior::C
    preprocessing::P
    model_id::String
    source_id::String
    identity::String
end

"""
    BRMSMCWindowFit

An SMC bridge for one new window under a carried posterior approximation.
`backend` contains this window, `previous` contains only the carried summary,
and `state` is the producer's particle state. Continue a partial bridge with
[`brm_smc_fit`](@ref). Once complete, call [`brm_smc_carry`](@ref) to discard
this window before assimilating the next one.

With `coordinates=:auto`, particle rows contain only the old coordinates
needed by this window and the new coordinates it introduces. The backend's
layout still describes the full current model. Omitted old coordinates are
retained through their conditional Gaussian in the carried summary.
Use [`brm_smc_coordinates`](@ref) to inspect the coordinate mapping and
[`brm_smc_recover`](@ref) to recover a full joint draw from a compact position.

`state.log_normalizer` estimates this window's predictive normalizer under
the carried prior, not exact full-history evidence.
"""
struct BRMSMCWindowFit{B,C,S,A}
    backend::B
    previous::C
    state::S
    ad_backend::A
    model_id::String
    binding_id::String
    coordinates::Union{Nothing,Symbol}
end
BRMSMCWindowFit(backend, previous, state, ad_backend, model_id, binding_id) =
    BRMSMCWindowFit(backend, previous, state, ad_backend, model_id, binding_id, nothing)

"""
    brm_smc_coordinates(fit::BRMSMCWindowFit)

Inspect an initialized or completed selective window's compact, current-model
and full-joint coordinate mapping through RKPPL's `smc_window_coordinates`.
Particle rows use the compact order; [`brm_smc_recover`](@ref) uses the joint
order. Requires a fit initialized with `coordinates=:auto`.
"""
function brm_smc_coordinates end

"""
    brm_smc_recover(fit::BRMSMCWindowFit, compact_position; rng=Xoshiro())

Recover one full joint unconstrained position from a selective window's
compact position. Selected coordinates keep their supplied values; omitted
old coordinates are drawn from their carried conditional Gaussian. The
result follows the joint order of [`brm_smc_coordinates`](@ref). Recovery
does not add full-width particles to the fit. A supplied compact position
must have the order and length of a column of `fit.state.positions`.

Works for an initialized or completed `coordinates=:auto` window. To obtain
a posterior draw, use a completed fit and select a compact particle according
to `fit.state.weights` before recovery.
"""
function brm_smc_recover end

"""
    brm_smc_carry(fit::Union{BRMSMCFit,BRMSMCWindowFit}; scale=nothing, basis=nothing,
        parameters=nothing, inputs=(), max_reflections=16, maxiter=100,
        progress=nothing)
    brm_smc_carry(model::Union{BRMI,RKBRMI}; reference, ad_backend=AutoEnzyme())

Summarize a completed posterior with RKPPL's `carry_smc`, retaining BRM's
fitted preprocessing without training rows. For a complete-frame cloud, the
default reference has its weighted mean and marginal standard deviations.
A selective window updates its compact marginal and combines it with the
old inactive conditional to produce a full joint Gaussian summary.
A supplied portable
WarmupHMC `scale_data` record defines the Gaussian covariance; the caller must
calibrate it as a posterior approximation. A mutation/proposal scale alone does
not guarantee posterior moments. No dense covariance or mutation-kernel
adaptation is performed here.

Use `basis=:fit` to ask RKPPL to fit bounded correlation directions and moments
from the completed weighted cloud through WarmupHMC's public fitter.
`max_reflections` and `maxiter` bound the fit, and `progress` is forwarded to
the producer. The owned `carried.posterior.fit_diagnostics` reports reflections,
iterations, residuals, convergence and effective sample size. Finite budgets
can still discard covariance; convergence does not certify posterior accuracy.
The fitted reference is used by subsequent windows, without adapting this
completed run's production kernel.

Use `basis=smc_reference(schedule).scale` to ask RKPPL to fit the completed
cloud's weighted variances along an independently learned orientation. This
retains its bounded reflection directions and recalibrates their magnitudes,
without forming a dense covariance. Cross-covariances in that basis are
discarded, so its orientation must still be adequate for the posterior.
`basis` and an explicit `scale` are mutually exclusive.

For a finite-state filter, use `parameters=:filter` and name any per-step
covariate `inputs`. RKPPL selects the authored static parameters and retains
their native constrained transforms, discarding historical latent coordinates.
This calibration is consumed by `brm_filter(carried, model; ...)`; it cannot
initialize a full-frame window bridge. With `basis=:fit`, fitting uses only
the selected static cloud. Any supplied scale or portable basis must describe
the selected coordinates. A slice of a full correlated scale generally is not
the marginal Gaussian scale of those coordinates.

The model form instead carries an explicitly supplied normalized posterior
approximation calibrated for this model and data. `smc_reference(schedule)`
from `reference=:pilot` is an importance/proposal reference, not a fitted
posterior covariance; do not carry it as a posterior-moment summary without
separate calibration. An external reference's location and scale are copied
without retaining its observations, queries or draws.

This is an assumed-density approximation. It is exact only for a Gaussian
posterior with matching moments, with additional Monte Carlo error when those
moments come from a finite cloud. A diagonal reference discards correlations.

Pass the result and a newly bound declaration to
`brm_smc_update(carried, builder(new_window); rng, particles, options, ...)`.
Only new, nonoverlapping observations belong in each window. By default the
same source, coordinates and fitted preprocessing must apply. Selective
growing windows opt in with `coordinates=:auto`; see [`brm_smc_update`](@ref).
"""
function brm_smc_carry end
