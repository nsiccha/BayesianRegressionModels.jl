# src/nlme_laplace.jl — the per-unit (embedded) Laplace approximation of an
# RK-lowered population model.
#
# With f(u) the RK sampler density over the packed unconstrained point
# u = (φ, z_1, …, z_n), where z_i are unit i's standardized random effects
# (`BRMNLMEView.subject_coordinates`) and φ is every other coordinate, units are
# conditionally independent given φ: f is a sum of one term per unit in z_i
# plus a term in φ alone. Integrating every z_i out by a Laplace approximation
# at its conditional mode ẑ_i(φ) gives the marginal
#
#     L(φ) = f(φ, ẑ) + Σ_i d_i/2 log 2π − 1/2 Σ_i log det(−H_i),
#
# H_i = ∇²_{z_i z_i} f at (φ, ẑ). It is exact when f is Gaussian in z. Its
# gradient follows from the implicit-function theorem (the approach of TMB and
# of Stan's `laplace_marginal`):
#
#     ∇L = ∇_φ f + 1/2 Σ_m T_m[φ] + 1/2 (∇²f w)[φ],
#
# where T_m = D³f[v_m, v_m, ·], v_m holds column m of C_i⁻ᵀ in every unit's
# block (−H_i = C_i C_iᵀ; units never interact, so one colored direction per
# within-unit position serves every unit), and w_i = (−H_i)⁻¹ Σ_m T_m[z_i].

"""
    BRMLaplaceProblem
    brm_laplace_problem(backend::RKBRMI; ad_backend, subject_group=nothing,
        nested_groups=NamedTuple(), tol=1e-24, maxiter=50) -> BRMLaplaceProblem

Integrate every unit's standardized random effects out of an RK-lowered model
by a Laplace approximation at their conditional mode, leaving the population
coordinates for HMC. The units, and which coordinates are their random
effects, are the partition of [`brm_nlme_view`](@ref); every other packed
coordinate (population coefficients, scalars, shared latents and the
random-effect scales and correlation factors) is a population coordinate.

The returned problem implements the LogDensityProblems interface (order 1)
over the population coordinates, in packed order (`problem.names`). Its log
density is the Laplace approximation of the model's marginal log posterior
density in those unconstrained coordinates, exact when the model is Gaussian
in the random effects. Each evaluation finds every unit's mode by Newton's
method with a per-unit backtracking line search, from the modes of the
previous evaluation, until every unit's Newton decrement gᵀ(−H)⁻¹g is at
most `tol`, or stops shrinking at rounding level. The log determinant moves
with the modes to first order; the standardized draws' N(0, I) prior makes
−H ≥ I for log-concave likelihoods, so the default `tol` keeps the modes to
about 1e-12. A search that has not converged after `maxiter` iterations
raises an error. The gradient comes from the implicit-function theorem with
third-derivative contractions of the model density.

A population point where the model's density, gradient or unit Hessians
overflow during the mode search, so that no finite Newton step exists, has log
density `-Inf`, as the full model's density there would; a sampler treats it
as a divergent step. `problem.nonfinite` counts such points.

A problem holds the last modes as mutable state: use one per chain
(`deepcopy` gives a copy its own prepared AD workspaces).

`subject_group` and `nested_groups` are those of [`brm_nlme_view`](@ref).
Refused, with [`BRMNLMEViewError`](@ref): everything `brm_nlme_model`
refuses, including Student-t random-effect blocks.
"""
mutable struct BRMLaplaceProblem{M<:BRMNLMEModel,B}
    model::M
    ad_backend::B
    population::Vector{Int}
    names::Vector{Symbol}
    third::Base.RefValue{Any}
    u::Vector{Float64}
    tol::Float64
    maxiter::Int
    nonfinite::Int
end

Base.show(io::IO, p::BRMLaplaceProblem) = print(io, "BRMLaplaceProblem(",
    length(p.population), " population coordinates, ",
    length(p.model.view.levels), " units, ",
    sum(length, p.model.view.subject_coordinates), " integrated random effects)")

function brm_laplace_problem(backend::RKBRMI; ad_backend, subject_group=nothing,
        nested_groups=NamedTuple(), tol=1e-24, maxiter=50)
    m = brm_nlme_model(backend; ad_backend, subject_group, nested_groups)
    population = findall(!=(:subject_effect), m.view.role)
    BRMLaplaceProblem(m, ad_backend, population, m.view.coordinates[population],
        Ref{Any}(nothing), copy(m.unit), Float64(tol), Int(maxiter), 0)
end

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: a prepared
# third-order query of the sampler density, and its contractions
# `D³f(u)[v, v, ·]`, one full-length vector per direction `v`.
function _rk_laplace_third_query end
function _rk_laplace_third end

function _brm_laplace_third_query(p::BRMLaplaceProblem)
    p.third[] === nothing || return p.third[]
    p.third[] = _rk_laplace_third_query(p.model.backend, p.model.unit, p.ad_backend)
end

# Each unit's share of the sampler density, up to constants: its attributed
# observation densities and its standardized draws' N(0, I) prior.
function _brm_laplace_unit_values(m::BRMNLMEModel, pointwise, u)
    values = _brm_nlme_sum_by_subject(m, pointwise)
    for (i, block) in enumerate(m.view.subject_coordinates)
        values[i] -= 0.5 * sum(abs2, view(u, block))
    end
    values
end

_brm_laplace_finite(x) = all(isfinite, x)

# A population point whose mode search overflows: the model density is not
# finite near it, so it scores -Inf.
function _brm_laplace_nonfinite!(p::BRMLaplaceProblem, u, iterations)
    p.nonfinite += 1
    (; u, value=-Inf, finite=false, iterations)
end

function _brm_laplace_evaluate(m::BRMNLMEModel, u::Vector{Float64})
    gradient = Vector{Float64}(undef, length(u))
    value, gradient, pointwise = _rk_value_gradient_and_pointwise!(m.sampler, gradient, u)
    (; value, gradient, units=_brm_laplace_unit_values(m, pointwise, u))
end

# The Cholesky factor of −H; `nothing` when −H is not positive definite.
function _brm_laplace_factor(block)
    F = cholesky(Symmetric(-block); check=false)
    issuccess(F) ? F : nothing
end

# A Newton step for one unit. Away from its mode a unit's density need not be
# concave there; Levenberg damping then shifts −H until it is.
function _brm_laplace_step(block, g)
    F = _brm_laplace_factor(block)
    F === nothing || return F \ g
    shift = max(1.0, maximum(abs, block))
    for _ in 1:60
        F = cholesky(Symmetric(shift * I - block); check=false)
        issuccess(F) && return F \ g
        shift *= 2
    end
    _brm_nlme_error("brm_laplace_problem: a unit's Hessian has no positive-definite " *
        "shift; its density is not finite")
end

"""
    _brm_laplace_mode!(p::BRMLaplaceProblem, φ) -> state

Every unit's conditional mode at population point `φ`, by Newton's method
from the modes of the previous evaluation. `state` holds the packed point `u`
at the modes, the sampler density's `value` and `gradient` there, each unit's
Hessian block, the Cholesky factor of its negation, and the number of Newton
steps taken.
"""
function _brm_laplace_mode!(p::BRMLaplaceProblem, φ::AbstractVector)
    m = p.model
    blocks = m.view.subject_coordinates
    length(φ) == length(p.population) || throw(DimensionMismatch(
        "the population point has length $(length(φ)), the problem has " *
        "$(length(p.population)) population coordinates"))
    u = copy(p.u)
    u[p.population] .= φ
    current = _brm_laplace_evaluate(m, u)
    if !(isfinite(current.value) && _brm_laplace_finite(current.gradient))
        # The previous modes may be far from this point's; start again at zero.
        for block in blocks
            u[block] .= 0.0
        end
        current = _brm_laplace_evaluate(m, u)
        isfinite(current.value) && _brm_laplace_finite(current.gradient) ||
            return _brm_laplace_nonfinite!(p, u, 0)
    end
    previous = fill(Inf, length(blocks))
    history = Vector{Float64}[]
    for iteration in 1:p.maxiter
        hessians = _brm_nlme_sampler_hessian_blocks(m, u)
        all(_brm_laplace_finite, hessians) || return _brm_laplace_nonfinite!(p, u, iteration - 1)
        steps = [_brm_laplace_step(hessians[i], current.gradient[block])
            for (i, block) in enumerate(blocks)]
        decrements = [dot(current.gradient[block], steps[i])
            for (i, block) in enumerate(blocks)]
        # A unit is at its mode once its Newton decrement is below `tol`, or
        # once a full Newton step (below) no longer shrinks its decrement: in
        # Newton's quadratic region that is the decrement's rounding floor,
        # which the scale of the unit's density sets.
        converged = all(eachindex(blocks)) do i
            decrements[i] <= p.tol ||
                (decrements[i] <= 1e-8 && decrements[i] >= previous[i] / 4)
        end
        previous = decrements
        push!(history, decrements)
        if converged
            factors = map(_brm_laplace_factor, hessians)
            any(isnothing, factors) && _brm_nlme_error(
                "brm_laplace_problem: a unit's density has a stationary point that " *
                "is not a maximum at this population point")
            p.u = u
            return (; u, value=current.value, gradient=current.gradient, hessians,
                factors, finite=true, iterations=iteration - 1)
        end
        # Backtrack each unit on its own: units are conditionally independent,
        # so one evaluation scores every unit's trial. A unit whose decrement
        # is already small is in Newton's quadratic region, where its density
        # gain (half the decrement) can lie below the density's rounding
        # error; its full step is taken unchecked.
        scale = ones(length(blocks))
        trial_u = copy(u)
        for (i, block) in enumerate(blocks)
            trial_u[block] .+= steps[i]
        end
        trial = _brm_laplace_evaluate(m, trial_u)
        for _ in 1:60
            rejected = [i for i in eachindex(blocks) if !(isfinite(trial.units[i]) &&
                _brm_laplace_finite(view(trial.gradient, blocks[i])) &&
                (decrements[i] <= 1e-8 ||
                 trial.units[i] >= current.units[i] + 1e-4 * scale[i] * decrements[i]))]
            isempty(rejected) && break
            for i in rejected
                scale[i] /= 2
                trial_u[blocks[i]] .= view(u, blocks[i]) .+ scale[i] .* steps[i]
            end
            trial = _brm_laplace_evaluate(m, trial_u)
        end
        all(isfinite, trial.units) && _brm_laplace_finite(trial.gradient) ||
            return _brm_laplace_nonfinite!(p, u, iteration)
        u = trial_u
        current = trial
    end
    worst = argmax(last(history))
    _brm_nlme_error("brm_laplace_problem: the unit modes did not converge in " *
        "$(p.maxiter) Newton iterations at population point $(repr(collect(φ))). " *
        "Unit $(repr(m.view.levels[worst]))'s Newton decrements, last iterations " *
        "first: $(reverse([h[worst] for h in history[max(1, end - 5):end]])) " *
        "(tol = $(p.tol))")
end

_brm_laplace_value(state) = state.finite ? state.value +
    sum(F -> size(F, 1) / 2 * log(2π) - logdet(F) / 2, state.factors; init=0.0) :
    state.value

# `D³f[v_m, v_m, ·]` summed over the colored directions v_m: column m of every
# unit's C⁻ᵀ.
function _brm_laplace_third_sum(p::BRMLaplaceProblem, state)
    blocks = p.model.view.subject_coordinates
    query = _brm_laplace_third_query(p)
    total = zeros(length(state.u))
    for k in 1:maximum(length, blocks)
        direction = zeros(length(state.u))
        for (i, block) in enumerate(blocks)
            k <= length(block) || continue
            e = zeros(length(block))
            e[k] = 1.0
            direction[block] .= state.factors[i].U \ e
        end
        total .+= _rk_laplace_third(query, direction, state.u)
    end
    total
end

function _brm_laplace_gradient(p::BRMLaplaceProblem, state)
    blocks = p.model.view.subject_coordinates
    third = _brm_laplace_third_sum(p, state)
    w = zeros(length(state.u))
    for (i, block) in enumerate(blocks)
        w[block] .= state.factors[i] \ third[block]
    end
    implicit = _rk_eta_hvp(p.model.eta_hvp, w, state.u)
    [state.gradient[j] + (third[j] + implicit[j]) / 2 for j in p.population]
end

LogDensityProblems.capabilities(::Type{<:BRMLaplaceProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::BRMLaplaceProblem) = length(p.population)
LogDensityProblems.logdensity(p::BRMLaplaceProblem, φ::AbstractVector) =
    _brm_laplace_value(_brm_laplace_mode!(p, φ))
function LogDensityProblems.logdensity_and_gradient(p::BRMLaplaceProblem,
        φ::AbstractVector)
    state = _brm_laplace_mode!(p, φ)
    state.finite || return state.value, fill(NaN, length(p.population))
    _brm_laplace_value(state), _brm_laplace_gradient(p, state)
end

"""
    brm_laplace_draws(p::BRMLaplaceProblem, P::AbstractMatrix;
        rng=Random.default_rng(), conditional=0) -> (; draws, diagnostics)

Complete draws from population draws `P` (`p.names` rows × draws, e.g. a
sampler's `posterior_position`). At each population draw every unit's random
effects are drawn from its Laplace approximation, the Gaussian at its
conditional mode with covariance (−H_i)⁻¹, or, with `conditional = K > 0`, from
its exact conditional posterior by importance resampling: `K` proposals per
unit from that Gaussian, weighted by the model's conditional density, one
resampled. All units' `k`-th proposals are scored by one model evaluation.

- `draws` — the model's packed unconstrained coordinates (rows, the order of
  `rk_restore_draws`) × draws; its population rows are `P`. Pass it to
  [`rk_restore_draws`](@ref) and the other readers of a full RK fit.
- `diagnostics` — `newton_steps`, the Newton steps each draw's mode search
  took from the previous draw's modes; with `conditional > 0` also
  `importance_ess`, units × draws, each unit's effective number of proposals
  (`(Σw)²/Σw²`, at most `K`). A small value marks a unit whose conditional
  posterior is far from Gaussian, where more proposals are needed.
"""
function brm_laplace_draws(p::BRMLaplaceProblem, P::AbstractMatrix;
        rng=Random.default_rng(), conditional::Integer=0)
    size(P, 1) == length(p.population) || throw(DimensionMismatch(
        "population draws have $(size(P, 1)) rows, the problem has " *
        "$(length(p.population)) population coordinates"))
    conditional >= 0 || throw(ArgumentError(
        "brm_laplace_draws: conditional = $conditional proposals; pass 0 for " *
        "Laplace draws or a positive count for importance resampling"))
    blocks = p.model.view.subject_coordinates
    draws = Matrix{Float64}(undef, length(p.u), size(P, 2))
    newton_steps = Vector{Int}(undef, size(P, 2))
    importance_ess = Matrix{Float64}(undef, conditional > 0 ? length(blocks) : 0,
        size(P, 2))
    for k in axes(P, 2)
        state = _brm_laplace_mode!(p, view(P, :, k))
        state.finite || _brm_nlme_error(
            "brm_laplace_draws: population draw $k has a nonfinite density")
        newton_steps[k] = state.iterations
        draws[:, k] .= state.u
        if conditional == 0
            for (i, block) in enumerate(blocks)
                draws[block, k] .+= state.factors[i].U \ randn(rng, length(block))
            end
        else
            picks = _brm_laplace_resample(p, state, conditional, rng)
            for (i, block) in enumerate(blocks)
                draws[block, k] .= picks.z[i]
            end
            importance_ess[:, k] .= picks.ess
        end
    end
    diagnostics = conditional > 0 ? (; newton_steps, importance_ess) : (; newton_steps)
    (; draws, diagnostics)
end

# Importance resampling of every unit's conditional posterior with its Laplace
# Gaussian as the proposal. A proposal z = ẑ + C⁻ᵀε has log proposal density
# −|ε|²/2 up to a per-unit constant, so its log weight is the unit's model
# density there plus |ε|²/2.
function _brm_laplace_resample(p::BRMLaplaceProblem, state, K, rng)
    m = p.model
    blocks = m.view.subject_coordinates
    ε = [randn(rng, length(block), K) for block in blocks]
    proposals = [state.u[block] .+ state.factors[i].U \ ε[i]
        for (i, block) in enumerate(blocks)]
    logw = Matrix{Float64}(undef, length(blocks), K)
    u = copy(state.u)
    for j in 1:K
        for (i, block) in enumerate(blocks)
            u[block] .= view(proposals[i], :, j)
        end
        units = _brm_laplace_unit_values(m, _rk_pointwise_values(m.pointwise, u), u)
        for i in eachindex(blocks)
            logw[i, j] = units[i] + sum(abs2, view(ε[i], :, j)) / 2
        end
    end
    z = Vector{Vector{Float64}}(undef, length(blocks))
    ess = Vector{Float64}(undef, length(blocks))
    for i in eachindex(blocks)
        w = exp.(view(logw, i, :) .- maximum(view(logw, i, :)))
        ess[i] = sum(w)^2 / sum(abs2, w)
        pick = searchsortedfirst(cumsum(w), rand(rng) * sum(w))
        z[i] = proposals[i][:, min(pick, K)]
    end
    (; z, ess)
end
