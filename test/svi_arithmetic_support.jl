# Included inside the analytic-target test modules; drive the same production
# run/advance/result path that public fits and optimizer checkpoints use.
function arithmetic_fit(rng, t, q; kwargs...)
    c = _brm_svi_controls(length(t.units); kwargs...)
    run = _brm_svi_run(rng, q, c, t)
    _brm_svi_fit!(run, t, c)
    (; q=_brm_svi_result_q(run), elbo=run.trace, run.converged,
        iterations=run.iteration, run.sampling, run.unit_gradient_calls)
end
