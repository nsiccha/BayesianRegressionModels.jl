# L-BFGS for `brm_laplace_mode` (src/laplace_mode.jl): the problem supplies its
# own value and gradient. A point where the Laplace scores -Inf is +Inf here,
# which the backtracking line search steps back from.
module BayesianRegressionModelsOptimExt

using BayesianRegressionModels
using LogDensityProblems
using Optim
const BRM = BayesianRegressionModels

function BRM._brm_laplace_mode(p, φ0; iterations, g_abstol)
    nonfinite = p.nonfinite
    f(φ) = -LogDensityProblems.logdensity(p, φ)
    function fg!(G, φ)
        v, g = LogDensityProblems.logdensity_and_gradient(p, φ)
        G .= .-g
        -v
    end
    g!(G, φ) = (fg!(G, φ); G)
    objective = OnceDifferentiable(f, g!, fg!, φ0)
    method = LBFGS(; linesearch=Optim.LineSearches.BackTracking())
    result = optimize(objective, φ0, method, Optim.Options(; g_abstol, iterations))
    φ = Optim.minimizer(result)
    # the integrated effects' modes at the mode itself
    value = LogDensityProblems.logdensity(p, φ)
    u = copy(p.u)
    u[p.population] .= φ
    converged = Optim.converged(result)
    converged || @warn("brm_laplace_mode: L-BFGS stopped after " *
        "$(Optim.iterations(result)) iterations with gradient norm " *
        "$(Optim.g_residual(result)), above g_abstol = $g_abstol")
    (; draws=reshape(u, :, 1), diagnostics=(; value, converged,
        iterations=Optim.iterations(result), gradient_norm=Optim.g_residual(result),
        nonfinite=p.nonfinite - nonfinite))
end

end
