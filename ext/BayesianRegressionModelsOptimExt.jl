# L-BFGS for `brm_laplace_mode` (src/laplace_mode.jl): the problem supplies its
# own value and gradient. A point where the Laplace scores -Inf is +Inf here,
# with a NaN gradient, which the line search steps back from.
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

# L-BFGS on an objective to minimize: `f(φ)` its value, `fg!(G, φ)` its value
# with the gradient written to `G`. `stop` names what ended it: `:tol`, the
# gradient's largest component is at most `g_abstol`; `:iterations`, the
# iteration limit; `:stalled`, neither: no step changed the point or the
# value any more, or none was acceptable (`code` is Optim's name for it).
#
# A step is accepted on Hager and Zhang's approximate Wolfe conditions
# (`LineSearches.HagerZhang`, Optim's own default for L-BFGS): where the value
# changes by less than 1e-6 of its size, they test the slope along the step
# instead of the value. A Laplace marginal's value carries the rounding of
# its large sums (measured on a mesh field: 2e-11 at 1,800 integrated effects,
# 6e-10 at 21,000), and its gradient is accurate far below that. From a
# gradient near 1e-4 down, a quasi-Newton step lowers the value by less than
# that rounding, so a test on the value alone (Armijo backtracking, as before)
# accepted or refused the step by the rounding. After one refusal every later
# search backtracked to a null step, and the run ended above `g_abstol` on
# Optim's "no change" stop: 85 of 99 value evaluations at 1,800 effects, 38 of
# 47 at 21,000 (snag `brm-laplace-mode-fcec521b`). More and Thuente's search
# tests the value too, and failed outright at 21,000.
function _brm_lbfgs(f, fg!, φ0::AbstractVector; iterations, g_abstol)
    g!(G, φ) = (fg!(G, φ); G)
    objective = OnceDifferentiable(f, g!, fg!, φ0)
    method = LBFGS(; alphaguess=_BRMUnitFirstStep(true),
        linesearch=Optim.LineSearches.HagerZhang())
    result = optimize(objective, φ0, method, Optim.Options(; g_abstol, iterations))
    code = Optim.termination_code(result)
    gradient_norm = Optim.g_residual(result)
    # Optim's `converged` is also true when a step left the point or the value
    # exactly unchanged (its `x_abstol` and `f_abstol` default to 0), whatever
    # the gradient: that is `:stalled` here.
    stop = gradient_norm <= g_abstol ? :tol :
        code === Optim.TerminationCode.Iterations ? :iterations : :stalled
    (; minimizer=Optim.minimizer(result), stop, code, gradient_norm,
        iterations=Optim.iterations(result))
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
    search = _brm_lbfgs(f, fg!, φ0; iterations, g_abstol)
    φ = search.minimizer
    # the integrated effects' modes at the mode itself
    value = LogDensityProblems.logdensity(p, φ)
    u = copy(p.u)
    u[p.population] .= φ
    BRM._brm_laplace_public_point!(p, u)
    search.stop === :tol || @warn("brm_laplace_mode: L-BFGS " *
        (search.stop === :iterations ? "reached its limit of $iterations iterations" :
            "could make no further progress after $(search.iterations) iterations " *
            "(Optim: $(search.code))") *
        " with gradient norm $(search.gradient_norm), above g_abstol = $g_abstol")
    (; draws=reshape(u, :, 1), diagnostics=(; value, converged=search.stop === :tol,
        search.stop, search.iterations, search.gradient_norm,
        nonfinite=p.nonfinite - nonfinite))
end

end
