# L-BFGS for `brm_laplace_mode` (src/laplace_mode.jl): the problem supplies its
# own value and gradient. A point where the Laplace scores -Inf is +Inf here,
# which the backtracking line search steps back from.
module BayesianRegressionModelsOptimExt

using BayesianRegressionModels
using LogDensityProblems
using Optim
using LinearAlgebra: norm
const BRM = BayesianRegressionModels

# L-BFGS's first direction is the gradient itself, whose length carries no
# scale: at a Laplace marginal's prior median it can be 60 in a log scale, and a
# trial step that long puts the scales where the integrated effects' curvature
# is lost to rounding and each evaluation's mode search runs to its iteration
# limit (snag `brm-laplace-inte-e0991a0f`: 220 ms for one such trial on epil).
# So the first trial step is at most of unit length, as liblbfgs and LBFGS++
# start; later steps use the curvature L-BFGS has learned, at full length.
mutable struct _BRMUnitFirstStep
    first::Bool
end
function (s::_BRMUnitFirstStep)(ls, state, phi_0, dphi_0, df)
    n = norm(state.s)
    state.alpha = s.first && n > 1 ? 1 / n : 1.0
    s.first = false
    state.alpha
end

# Typed, so it adds a method beside src's fallback instead of overwriting it,
# which precompilation refuses.
function BRM._brm_laplace_mode(
        p::Union{BRM.BRMLaplaceProblem,BRM.BRMStructuredLaplaceProblem,
            BRM.BRMSparseLaplaceProblem,BRM._BRMSampledLaplaceProblem},
        φ0::AbstractVector; iterations, g_abstol)
    nonfinite = p.nonfinite
    f(φ) = -LogDensityProblems.logdensity(p, φ)
    function fg!(G, φ)
        v, g = LogDensityProblems.logdensity_and_gradient(p, φ)
        G .= .-g
        -v
    end
    g!(G, φ) = (fg!(G, φ); G)
    objective = OnceDifferentiable(f, g!, fg!, φ0)
    method = LBFGS(; alphaguess=_BRMUnitFirstStep(true),
        linesearch=Optim.LineSearches.BackTracking())
    result = optimize(objective, φ0, method, Optim.Options(; g_abstol, iterations))
    φ = Optim.minimizer(result)
    # the integrated effects' modes at the mode itself
    value = LogDensityProblems.logdensity(p, φ)
    u = copy(p.u)
    u[p.population] .= φ
    BRM._brm_laplace_public_point!(p, u)
    converged = Optim.converged(result)
    converged || @warn("brm_laplace_mode: L-BFGS stopped after " *
        "$(Optim.iterations(result)) iterations with gradient norm " *
        "$(Optim.g_residual(result)), above g_abstol = $g_abstol")
    (; draws=reshape(u, :, 1), diagnostics=(; value, converged,
        iterations=Optim.iterations(result), gradient_norm=Optim.g_residual(result),
        nonfinite=p.nonfinite - nonfinite))
end

end
