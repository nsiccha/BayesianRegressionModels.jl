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
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=:subjects, tol=1e-24, maxiter=50,
        structured=:auto, integrate=nothing, max_cell_width=128,
        solver=:auto, matrixfree=MatrixFreeLaplace(), progress=nothing)

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
in the random effects and exact determinant algebra is selected (see `solver`
below). Each evaluation finds every unit's mode by continuation from the prior:
it starts the effects at their prior's mode and follows each unit's mode while
the likelihood's weight rises from 0 to 1 in 10 stages (one stage for a sparse
component or structured unit whose density is known to be log-concave). Each
stage uses Newton's method with a per-unit backtracking line search, until every
unit's Newton decrement gᵀ(−H)⁻¹g is at
most `tol`, or stops shrinking at rounding level. The log determinant moves
with the modes to first order; the standardized draws' N(0, I) prior makes
−H ≥ I for log-concave likelihoods, so the default `tol` keeps the modes to
about 1e-12. The gradient comes from the implicit-function theorem with
third-derivative contractions of the model density.

Two kinds of population point have log density `-Inf`, which a sampler treats
as a divergent step, as at the extreme points a diverging warm-up trajectory
reaches:

- the model's density, gradient or unit Hessians overflow during the mode
  search, so that no finite Newton step exists (the full model's density there
  is not finite either); `problem.nonfinite` counts them;
- a stage of the mode search has not converged after `maxiter` Newton steps, as when a
  start lies far up an exponential's slope, where Newton gains about one unit
  per step; `problem.unconverged` counts them, and a warning names the point,
  the unit and its last decrements. At a point a sampler would accept this
  marks a defect.

A mode search must not depend on preceding evaluations: in a non-concave
conditional density, a warm start can select a different local maximum and
change the target. Continuation from the prior defines a single-mode
approximation at the mode connected to the prior's mode, which changes only
where that mode ceases to exist; the marginal can jump there. `problem.folds`
counts evaluations whose followed mode needed a modified Newton step after the
first stage. This does not sum over multiple conditional modes or guarantee a
global maximum.

A problem holds the last result and prepared AD workspaces as mutable state: use one per chain
(`deepcopy` gives a copy its own prepared AD workspaces).

When every likelihood term's parameters depend on its unit's random effects
only through linear predictors (Poisson-log, Bernoulli/binomial-logit,
negative-binomial with log mean and log shape, Normal; unweighted,
uncensored, flat groupings), the problem is a
[`BRMStructuredLaplaceProblem`](@ref): each unit's mode is its own Newton
search on its own rows from closed-form per-row derivatives, and the gradient
is one reverse sweep of the likelihood parameters. Otherwise it is a
`BRMLaplaceProblem`, which differentiates the whole model. `structured=true`
requires the first (refusing with the reason), `false` forces the second.

Declared independent Gaussian vectors can also be named by declaration, e.g.
`integrate = (:field,)` for `field[1:n] .~ Normal.(mean, scale)`. Their means
and scales can depend on outer coordinates; each element retains its authored
coordinate and prior. Width screening applies to the whole vector.

`integrate = :auto` chooses the latent terms itself ([`brm_laplace_plan`](@ref)
with its defaults): every term only Normal rows read, and every term read by
counts, binomial rows or `@plate` cells whose per-block importance diagnostic
admits it. `integrate = plan` takes a [`BRMLaplacePlan`](@ref) as given; the
problem's `plan` field holds the plan it was built from. With `integrate`,
`progress` reports the preparation through Treebars as in
[`brm_laplace_plan`](@ref).

`integrate = :g` or `integrate = (:g, :h, …)` instead integrates every level
of the named grouping columns' latent blocks by one sparse Laplace, whatever
their nesting: random-effect blocks over the grouping, `cdar(step; by=g)` walks
(their standardized innovations) and `car(g; edges)` fields over an area column
(ICAR values under their graph precision, BYM2's iid parts too; the default
lowering, not `car_units`). Groupings may be crossed, nested with any number
of levels per parent, or cut across the NLME units. The problem is then a
[`BRMSparseLaplaceProblem`](@ref) over every other coordinate, the latent
draws of the other groupings included. A field's per-component soft
sum-to-zero stays out of the sparse factor (Woodbury); every component must be
read by some row. Every observation is a regression response of one of the
structured problem's four families, whose rows it shares, or sits in a
`@plate` cell. A cell is one term over the coordinates it reads, its gradient,
Hessian block and third derivatives by Enzyme over that selection
(ReactiveKernelsPPL's compact-cell AD), which needs the Enzyme dogfood line
from `7a886d9`. The integrated effects' negative
Hessian `I + Cᵀ W C` (with the cells' blocks added) has sparse local structure;
each group of integrated effects that share likelihood terms
(a unit, or the units a crossed grouping joins) takes its own Newton steps,
backtracking and stop. `subject_group`, `nested_groups`, `shared_groups` and
`structured` describe the per-unit Laplace and do not apply with it.

For the `integrate` path, `solver=:auto` keeps exact algebra on components of
at most `matrixfree.exact_limit` coordinates (default 256) and uses operator
solves and approximate determinants on larger components. `solver=:exact`
retains the sparse Cholesky/augmented-factor implementation, while
`solver=:matrixfree` forces operator algebra. [`MatrixFreeLaplace`](@ref)
configures fixed probes, numerical tolerances and iteration caps. The
large-component gradient differentiates the reported finite determinant
approximation, including its dependence on the conditional mode. Probes and
policy stay fixed throughout a prepared target's use and copies.
`problem.algebra.diagnostics` reports per-component determinant probe MCSE;
this is not a bound on the error of Laplace integration.
Operator curvature checks cover Krylov search directions; they do not certify
global positive definiteness. Failed capped linear solves raise an error.

`subject_group`, `nested_groups` and `shared_groups` are those of
[`brm_nlme_view`](@ref); the shared latents (a block over a grouping in
`shared_groups`, a `car(...)` field) are population coordinates.
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
    factors::Base.RefValue{Any}
    tol::Float64
    maxiter::Int
    nonfinite::Int
    unconverged::Int
    # evaluations in which a unit's followed mode needed a modified Newton step
    # after the continuation's first stage (`_brm_laplace_mode!`)
    folds::Int
end

Base.show(io::IO, p::BRMLaplaceProblem) = print(io, "BRMLaplaceProblem(",
    length(p.population), " population coordinates, ",
    length(p.model.view.levels), " units, ",
    sum(length, p.model.view.subject_coordinates), " integrated random effects)")

function brm_laplace_problem(backend::RKBRMI; ad_backend, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=:subjects, tol=1e-24, maxiter=50,
        structured=:auto, integrate=nothing, max_cell_width=128,
        solver=nothing, matrixfree=nothing, progress=nothing)
    max_cell_width = _brm_spl_width_limit(max_cell_width)
    integrate === nothing && progress !== nothing && throw(ArgumentError(
        "brm_laplace_problem: `progress` reports the sparse Laplace's preparation, which needs `integrate`"))
    policy = integrate isa BRMLaplacePlan ? integrate.algebra_policy :
        (; solver=:auto, matrixfree=MatrixFreeLaplace())
    solver = something(solver, policy.solver)
    matrixfree = something(matrixfree, policy.matrixfree)
    if integrate !== nothing
        subject_group === nothing && isempty(nested_groups) && isempty(shared_groups) &&
            structured === :auto && unit_partition === :subjects || throw(ArgumentError("brm_laplace_problem: " *
            "integrate = $(repr(integrate)) chooses the integrated effects itself; " *
            "subject_group, nested_groups, shared_groups and structured describe the " *
            "per-unit Laplace and do not apply with it"))
        integrate === :auto && return _brm_sparse_laplace_problem_from_plan(backend,
            _brm_laplace_plan(backend; ad_backend, integrate=:auto, sample=(),
                ess_floor=0.5, draws=200, at=nothing, rng=Random.Xoshiro(20261031), tol,
                maxiter, max_cell_width, solver, matrixfree, progress); ad_backend, tol, maxiter,
                solver, matrixfree, progress)
        integrate isa BRMLaplacePlan && return _brm_sparse_laplace_problem_from_plan(
            backend, (; plan=integrate, problem=nothing); ad_backend, tol, maxiter,
            solver, matrixfree, progress)
        return _brm_sparse_laplace_problem(backend, integrate; ad_backend, tol, maxiter,
            max_cell_width, solver, matrixfree, progress)
    end
    solver in (:auto, :exact) || throw(ArgumentError(
        "brm_laplace_problem: solver=:matrixfree applies to the integrate path"))
    structured in (:auto, true, false) || throw(ArgumentError(
        "brm_laplace_problem: structured = $(repr(structured)); pass :auto, true or false"))
    if structured !== false
        view = brm_nlme_view(backend; subject_group, nested_groups, shared_groups, unit_partition)
        refusal = _brm_slap_refusal(backend, view)
        refusal === nothing && return _brm_structured_laplace_problem(backend, view;
            ad_backend, tol, maxiter)
        structured === true && _brm_nlme_error(
            "brm_laplace_problem: structured = true, but the model has no row structure: " *
            refusal)
    end
    m = brm_nlme_model(backend; ad_backend, subject_group, nested_groups,
        shared_groups, unit_partition)
    population = findall(!=(:subject_effect), m.view.role)
    BRMLaplaceProblem(m, ad_backend, population, m.view.coordinates[population],
        Ref{Any}(nothing), copy(m.unit), Ref{Any}(nothing), Float64(tol), Int(maxiter),
        0, 0, 0)
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
# concave (a likelihood that is not log-concave in its effects, such as a
# random effect on a negative-binomial shape). The step then uses −H with its
# eigenvalues reflected to their magnitudes and floored at 1, the curvature of
# the standardized draws' prior: an ascent direction scaled by the curvature
# along each eigenvector.
function _brm_laplace_step(block, g)
    F = _brm_laplace_factor(block)
    F === nothing || return F \ g
    E = eigen(Symmetric(-block))
    E.vectors * ((E.vectors' * g) ./ max.(abs.(E.values), 1.0))
end

# The continuation density's unit value (`v` a unit's density) or gradient
# (`v` its gradient) at stage s, for the unit's standardized effects z.
_brm_laplace_blend(s, v::Real, z) = s * v - (1 - s) * sum(abs2, z) / 2
_brm_laplace_blend(s, v::AbstractVector, z) = s .* v .- (1 - s) .* z

# A Newton step for one unit (`_brm_laplace_step`) and whether it is the plain
# Newton step.
function _brm_laplace_step_exact(block, g)
    F = _brm_laplace_factor(block)
    F === nothing || return F \ g, true
    _brm_laplace_step(block, g), false
end

# A plain Newton step's decrement below 1e-8 that has stopped shrinking fourfold
# is either at the rounding floor the density's scale sets, or in a stretch where
# Newton converges only linearly: on an exponential's slope it falls by a factor
# e per step, along a stiff curved valley more slowly, and the gradient is far
# from zero there. The gradient tells them apart. Moved a small fraction ξ along
# the Newton direction, a gradient that curvature sets scales by 1 − ξ, up to
# order ξ²; one that rounding sets stays put or jumps. So a candidate is at its
# floor when r = g(z + ξδ) − (1 − ξ) g(z), in the decrement's norm, is at least
# half the expected response ξ√λ.
const _BRM_LAP_FLOOR_PROBE = 1 / 64
_brm_lap_floor_candidate(λ, previous, tol) = tol < λ <= 1e-8 && λ >= previous / 4
_brm_lap_at_floor(ρ, λ) = ρ >= (_BRM_LAP_FLOOR_PROBE / 2)^2 * λ

# Which of the candidate units are at their rounding floor (above), by one
# evaluation at every unit's probe point.
function _brm_laplace_at_floor(m, s, u, current, hessians, steps, decrements, candidate)
    blocks = m.view.subject_coordinates
    ξ = _BRM_LAP_FLOOR_PROBE
    probe_u = copy(u)
    for (i, block) in enumerate(blocks)
        probe_u[block] .+= ξ .* steps[i]
    end
    probe = _brm_laplace_evaluate(m, probe_u)
    map(eachindex(blocks)) do i
        candidate[i] || return false
        block = blocks[i]
        r = _brm_laplace_blend(s, view(probe.gradient, block), view(probe_u, block)) .-
            (1 - ξ) .* _brm_laplace_blend(s, view(current.gradient, block), view(u, block))
        all(isfinite, r) || return false
        direction, _ = _brm_laplace_step_exact(s * hessians[i] - (1 - s) * I, r)
        _brm_lap_at_floor(dot(r, direction), decrements[i])
    end
end

"""
    _brm_laplace_mode!(p::BRMLaplaceProblem, φ) -> state

Every unit's conditional mode at population point `φ`, by continuation from
zero standardized effects, independently of preceding evaluations: Newton's
method on the continuation density (1 − s)·(−|z|²/2) + s·f, which is the
standardized effects' prior plus s times the rest of the sampler density f,
for each stage s of `_brm_lap_stages` in turn, each from the last stage's
modes. The modes reached are the ones connected to the prior's mode.
`state` holds the packed point `u`
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
    # Previous modes can belong to another basin. Even a converged search
    # from them need not define the same density at this population point.
    u = zeros(length(p.u))
    u[p.population] .= φ
    current = _brm_laplace_evaluate(m, u)
    isfinite(current.value) && _brm_laplace_finite(current.gradient) ||
        return _brm_laplace_nonfinite!(p, u, 0)
    # The density is a black box here, so its concavity in the effects is
    # unknown: every evaluation takes all stages.
    stages = _brm_lap_stages(false)
    total = 0
    late = 0
    history = Vector{Float64}[]
    hessians = nothing
    for (stage, s) in enumerate(stages)
        tol = s == 1 ? p.tol : max(p.tol, _BRM_LAP_STAGE_TOL)
        previous = fill(Inf, length(blocks))
        converged = false
        for iteration in 1:p.maxiter
            hessians = _brm_nlme_sampler_hessian_blocks(m, u)
            all(_brm_laplace_finite, hessians) ||
                return _brm_laplace_nonfinite!(p, u, total)
            steps = Vector{Vector{Float64}}(undef, length(blocks))
            exact = falses(length(blocks))
            decrements = zeros(length(blocks))
            for (i, block) in enumerate(blocks)
                g = _brm_laplace_blend(s, view(current.gradient, block), view(u, block))
                steps[i], exact[i] = _brm_laplace_step_exact(
                    s * hessians[i] - (1 - s) * I, g)
                decrements[i] = dot(g, steps[i])
            end
            # A unit is at its mode once its Newton decrement is below `tol`, or
            # at its rounding floor on a plain Newton step
            # (`_brm_lap_floor_candidate`, `_brm_laplace_at_floor`). A modified
            # step's decrement can shrink slowly away from the mode. The search
            # stops only when every unit does, so only then is the floor measured.
            stopped = [decrements[i] <= tol for i in eachindex(blocks)]
            candidate = [exact[i] && _brm_lap_floor_candidate(decrements[i], previous[i], tol)
                for i in eachindex(blocks)]
            if any(candidate) && all(stopped .| candidate)
                stopped .|= _brm_laplace_at_floor(m, s, u, current, hessians, steps,
                    decrements, candidate)
            end
            converged = all(stopped)
            previous = [exact[i] ? decrements[i] : Inf for i in eachindex(blocks)]
            push!(history, decrements)
            converged && break
            stage > 1 && !all(exact) && (late += 1)
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
                     _brm_laplace_blend(s, trial.units[i], view(trial_u, blocks[i])) >=
                        _brm_laplace_blend(s, current.units[i], view(u, blocks[i])) +
                        1e-4 * scale[i] * decrements[i]))]
                isempty(rejected) && break
                for i in rejected
                    scale[i] /= 2
                    trial_u[blocks[i]] .= view(u, blocks[i]) .+ scale[i] .* steps[i]
                end
                trial = _brm_laplace_evaluate(m, trial_u)
            end
            all(isfinite, trial.units) && _brm_laplace_finite(trial.gradient) ||
                return _brm_laplace_nonfinite!(p, u, total + iteration)
            u = trial_u
            current = trial
            total += 1
        end
        converged ||
            return _brm_laplace_unconverged!(p, m, φ, u, history, stage, length(stages))
    end
    late > 0 && (p.folds += 1)
    factors = map(_brm_laplace_factor, hessians)
    any(isnothing, factors) && _brm_nlme_error(
        "brm_laplace_problem: a unit's density has a stationary point that " *
        "is not a maximum at this population point")
    p.u = u
    p.factors[] = factors
    (; u, value=current.value, gradient=current.gradient, hessians,
        factors, finite=true, iterations=total)
end

function _brm_laplace_unconverged!(p::BRMLaplaceProblem, m, φ, u, history, stage, nstages)
    worst = argmax(last(history))
    @warn("brm_laplace_problem: the unit modes did not converge in " *
        "$(p.maxiter) Newton iterations of continuation stage $stage of $nstages at " *
        "population point $(repr(collect(φ))); its log density is -Inf. Unit " *
        "$(repr(m.view.levels[worst]))'s Newton decrements, last iterations first: " *
        "$(reverse([h[worst] for h in history[max(1, end - 5):end]])) " *
        "(tol = $(p.tol))", maxlog=3)
    p.unconverged += 1
    (; u, value=-Inf, finite=false, iterations=p.maxiter)
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
  took from the prior's mode; with `conditional > 0` also
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
