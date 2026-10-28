# Pathfinder for `brm_start`'s start from the data (src/laplace_start.jl): runs
# over a Laplace problem's population coordinates.
module BayesianRegressionModelsPathfinderExt

using BayesianRegressionModels
using Pathfinder
const BRM = BayesianRegressionModels

# The problem supplies its own gradient, so Optimization must not synthesize
# another one (`NoAD`), as WarmupHMC's initializer does; one try, so a failed
# run raises instead of restarting from a random point.
function BRM._brm_pathfinder_draws(
        p::Union{BRM.BRMLaplaceProblem,BRM.BRMStructuredLaplaceProblem},
        φ0::AbstractVector; ndraws, rng)
    result = Pathfinder.pathfinder(p; init=collect(Float64, φ0), rng, ndraws,
        ntries=1, adtype=Pathfinder.SciMLBase.NoAD())
    Matrix{Float64}(result.draws)
end

end
