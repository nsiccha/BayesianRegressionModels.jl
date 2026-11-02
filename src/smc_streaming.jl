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
end

"""
    brm_smc_carry(fit::Union{BRMSMCFit,BRMSMCWindowFit}; scale=nothing, basis=nothing,
        parameters=nothing, inputs=(), max_reflections=16, maxiter=100,
        progress=nothing)
    brm_smc_carry(model::Union{BRMI,RKBRMI}; reference, ad_backend=AutoEnzyme())

Summarize a completed posterior with RKPPL's `carry_smc`, retaining BRM's
fitted preprocessing without training rows. The default reference has the
cloud's weighted mean and marginal standard deviations. A supplied portable
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
Only new, nonoverlapping observations belong in each window. The same source,
coordinates and fitted preprocessing must apply; growing latent trajectories
require a state-filtering model instead of this static-parameter bridge.
"""
function brm_smc_carry end
