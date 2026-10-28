# Pathfinder for `brm_start`'s start from the data (src/laplace_start.jl): runs
# over a Laplace problem's population coordinates.
module BayesianRegressionModelsPathfinderExt

using BayesianRegressionModels
using Pathfinder
const BRM = BayesianRegressionModels

# The problem supplies its own gradient, so Optimization must not synthesize
# another one (`NoAD`), and the ELBO along the path takes one draw per iterate,
# and at most 100 L-BFGS iterations, as WarmupHMC's initializer does: a Laplace
# evaluation can cost many model gradients. One try, so a failed run raises
# instead of restarting from a random point.
function BRM._brm_pathfinder_draws(
        p::Union{BRM.BRMLaplaceProblem,BRM.BRMStructuredLaplaceProblem},
        φ0::AbstractVector; ndraws, rng)
    result = Pathfinder.pathfinder(p; init=collect(Float64, φ0), rng, ndraws,
        ndraws_elbo=1, maxiters=100, ntries=1, adtype=Pathfinder.SciMLBase.NoAD())
    # Pathfinder warns and returns draws from its starting Gaussian when its
    # run fails (no L-BFGS step taken, or no fit with a finite ELBO).
    steps = length(result.optim_trace) - 1
    elbos = [e.value for e in result.elbo_estimates]
    steps > 0 && !isempty(elbos) && isfinite(elbos[result.fit_iteration]) ||
        BRM._brm_nlme_error("brm_start: Pathfinder over the Laplace marginal failed " *
            "($steps L-BFGS steps; ELBO estimates $(elbos)); the Laplace problem " *
            "counted $(p.nonfinite) nonfinite and $(p.unconverged) unconverged points")
    Matrix{Float64}(result.draws)
end

end
