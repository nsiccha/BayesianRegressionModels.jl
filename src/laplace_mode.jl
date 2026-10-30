# src/laplace_mode.jl — the mode of the embedded-Laplace marginal (rung 2 of
# the staged ladder, design `1lxkp7y` prong `point`): the population point
# that maximizes a Laplace problem's log density, by L-BFGS on its own value
# and gradient, every integrated effect at its conditional mode there. A
# Gaussian over the population at that point would be a diagnostic only,
# never the delivered posterior (`no-full-laplace`); none is returned.

"""
    brm_laplace_mode(target; init=nothing, iterations=1000, g_abstol=1e-6)
        -> (; draws, diagnostics)

The mode of the embedded-Laplace marginal of `target`, any problem
[`brm_laplace_problem`](@ref) returns: the population point that maximizes its
log density, by L-BFGS (Optim) with a backtracking line search on the
problem's own value and gradient, from `init` (population coordinates, in
`target.names` order) or, by default, the population coordinates' prior
medians. It is a point fit: the mode of the marginal over the population, the
integrated effects at their conditional modes there. It never puts a
Gaussian over the population.

`draws` is one complete RK packed point (coordinates × 1, the shape of
[`brm_laplace_draws`](@ref)): the population coordinates at the mode and every
integrated effect at its conditional mode. `diagnostics` holds the log density
there (`value`), whether L-BFGS converged (`converged`; a warning names a run
that did not), its `iterations`, the gradient's norm (`gradient_norm`) and how
many evaluations scored `-Inf` (`nonfinite`).

Needs Optim loaded (`using WarmupHMC` loads it through Pathfinder).
"""
function brm_laplace_mode(p; init=nothing, iterations::Integer=1000,
        g_abstol::Real=1e-6)
    φ0 = init === nothing ? _brm_laplace_prior_median(p) : collect(Float64, init)
    length(φ0) == LogDensityProblems.dimension(p) || throw(DimensionMismatch(
        "brm_laplace_mode: init has length $(length(φ0)), the problem has " *
        "$(LogDensityProblems.dimension(p)) population coordinates"))
    _brm_laplace_mode(p, φ0; iterations=Int(iterations), g_abstol=Float64(g_abstol))
end

# Implemented by `BayesianRegressionModelsOptimExt`.
_brm_laplace_mode(p, φ0; iterations, g_abstol) = throw(ArgumentError(
    "brm_laplace_mode: L-BFGS comes from Optim; load it (`using Optim`, or " *
    "`using WarmupHMC`, which loads it through Pathfinder)"))
