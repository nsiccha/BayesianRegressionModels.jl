# src/laplace_start.jl — data-based starting points with no initial values:
# Pathfinder over the per-unit Laplace marginal of the population coordinates
# (`brm_laplace_problem`), then each unit's random effects drawn from its
# Laplace Gaussian at that population point (`brm_laplace_draws`).
#
# Pathfinder over the joint density optimizes the group effects together with
# their scales and correlations, and a scale or autocorrelation can absorb the
# differences between groups before the groups separate. The marginal has the
# effects integrated out at each unit's own mode, so that path does not exist
# there, and the population space is small.

"""
    brm_start(rk::RKBRMI; chains=nothing, rng=Random.default_rng(), ad_backend,
        subject_group=nothing, nested_groups=NamedTuple(), shared_groups=())
    brm_start(sb::SBBRMI, stan_model; chains=nothing, rng=Random.default_rng(),
        rk=RKBRMI(parent(sb)), ad_backend, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=())

A start computed from the data, with no initial values to give. Pathfinder
runs over the model's population coordinates with every unit's random effects
integrated out by a Laplace approximation ([`brm_laplace_problem`](@ref)),
from a point drawn uniformly within one unit of the population coordinates'
prior medians (unconstrained scale). The start is one draw from Pathfinder's
Gaussian fit, with every unit's random effects drawn from its own Laplace
Gaussian at that point ([`brm_laplace_draws`](@ref)): each group starts near
its own value, and the scales near what the groups' spread supports.

`chains = n` returns `n` such starts, one independent Pathfinder run each, for
a sampler's per-chain `init` (`completion_warmup_mcmc(rngs, lpdf; init =
brm_start(sb, model; chains = n))`); without it, one start. A Pathfinder run
that fails (no L-BFGS step, or no fit with a finite ELBO) raises. The first form returns RK packed
unconstrained points; the second, the Stan model's unconstrained points,
through [`brm_coordinate_transport`](@ref) from `rk`. Pass `rk` to reuse an
RKBRMI built from `parent(sb)`.

Initial values for some addresses go on top of a start through `base`:
`brm_start(sb, model, inits; base = brm_start(sb, model))`.

Needs Pathfinder loaded (WarmupHMC loads it) and, for the default
`ad_backend` (reverse-mode `AutoEnzyme`), Enzyme and DifferentiationInterface.
The units are those of [`brm_nlme_view`](@ref) (`subject_group`,
`nested_groups`, `shared_groups`); a model it does not partition into units,
one with no random-effect block among them, is refused with the view's
reason.
"""
function brm_start(rk::RKBRMI; chains::Union{Nothing,Integer}=nothing,
        rng::AbstractRNG=Random.default_rng(), ad_backend=nothing,
        subject_group=nothing, nested_groups=NamedTuple(), shared_groups=())
    U = _brm_laplace_start_draws(rk, _brm_start_count(chains); rng, ad_backend,
        subject_group, nested_groups, shared_groups)
    _brm_start_result(U, chains)
end

function brm_start(sb::SBBRMI, stan_model; chains::Union{Nothing,Integer}=nothing,
        rng::AbstractRNG=Random.default_rng(), rk=nothing, ad_backend=nothing,
        subject_group=nothing, nested_groups=NamedTuple(), shared_groups=())
    rk = rk === nothing ? RKBRMI(parent(sb)) : rk
    t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan_model))
    U = _brm_laplace_start_draws(rk, _brm_start_count(chains); rng, ad_backend,
        subject_group, nested_groups, shared_groups)
    S = reduce(hcat, [brm_rk_point_to_stan(t, view(U, :, k)) for k in axes(U, 2)])
    _brm_start_result(S, chains)
end

function _brm_start_count(chains)
    chains === nothing && return 1
    chains >= 1 || throw(ArgumentError("brm_start: chains = $chains; pass at least 1"))
    Int(chains)
end

_brm_start_result(U::AbstractMatrix, ::Nothing) = U[:, 1]
_brm_start_result(U::AbstractMatrix, ::Integer) = [U[:, k] for k in axes(U, 2)]

# `n` complete RK packed points (coordinates × n).
function _brm_laplace_start_draws(rk::RKBRMI, n; rng, ad_backend, subject_group,
        nested_groups, shared_groups)
    ad = ad_backend === nothing ? _brm_rk_default_ad("brm_start", rk) : ad_backend
    p = try
        brm_laplace_problem(rk; ad_backend=ad, subject_group, nested_groups,
            shared_groups)
    catch err
        err isa BRMNLMEViewError || rethrow()
        _brm_nlme_error("brm_start: the start from the data integrates each " *
            "unit's random effects by a Laplace approximation, which this model " *
            "does not admit; give initial values instead (`brm_start(x, inits)`). " *
            sprint(showerror, err))
    end
    φ0 = _brm_laplace_prior_median(p)
    # One Pathfinder run per start, each from its own point near the medians,
    # so the starts are independent; chain k's draws do not depend on `n`.
    reduce(hcat, map(1:n) do _
        init = φ0 .+ (2 .* rand(rng, length(φ0)) .- 1)
        brm_laplace_draws(p, _brm_pathfinder_draws(p, init; ndraws=1, rng); rng).draws
    end)
end

# The population coordinates' prior medians, every other population coordinate
# at the unit point (the start of `brm_nlme_start(m)`).
function _brm_laplace_prior_median(p)
    backend, view, unit = _brm_laplace_parts(p)
    first(_brm_nlme_prior_median_point(backend, view, unit))[p.population]
end
_brm_laplace_parts(p::BRMLaplaceProblem) = (p.model.backend, p.model.view, p.model.unit)
_brm_laplace_parts(p::BRMStructuredLaplaceProblem) = (p.backend, p.view,
    _brm_nlme_unit_point(p.backend, p.view; shared_point=nothing))

# Implemented by `BayesianRegressionModelsPathfinderExt`: Pathfinder over the
# Laplace problem `p` from `φ0`, `ndraws` draws (coordinates × draws) from its
# Gaussian fit; a failed run raises.
_brm_pathfinder_draws(p, φ0; ndraws, rng) = throw(ArgumentError(
    "brm_start: the start from the data runs Pathfinder; load it (`using " *
    "Pathfinder`, or `using WarmupHMC`, which loads it)"))
