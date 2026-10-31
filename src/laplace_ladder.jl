# src/laplace_ladder.jl — a staged fit over the sparse embedded Laplace.
#
# `brm_ladder_fit` climbs the rungs of design `1lxkp7y` (the review's brief
# `g1tlc0`): each rung is a complete result in BRM's usual shape that starts
# the next, and each rung's check goes into `fit.ladder`.
#
#   1  effect modes at a fixed population point          (the factor, the modes)
#   2  the mode of the embedded-Laplace marginal          (`brm_laplace_mode`)
#   3  SAEM from rung 2, sampling only what is not integrated (NLMEEstimation)
#   4  effects given the point: Laplace draws through the factor
#   5a the population's curvature at the point: a labelled diagnostic, never
#      the delivered posterior
#   5b HMC over the Laplace marginal, the effects drawn through the factor
#   6  exact MCMC on the full model, from rung 5
#
# With `integrate = :auto` the plan (`brm_laplace_plan`) is measured again at
# rung 2's point and after each SAEM phase. Where gated terms switch, the
# next rung or phase uses the rebuilt problem. A plan passed in is kept.

"""
    brm_ladder_fit(rk::RKBRMI; ad_backend, upto = 4, integrate = :auto,
        start = nothing, draws = 1000, conditional = 0, rng = Random.default_rng(),
        continue_from = nothing, subject_group = nothing, nested_groups = (;),
        shared_groups = (), saem_options = nothing, sampler = nothing) -> NamedTuple

A staged fit of `rk` over its sparse embedded Laplace ([`brm_laplace_problem`](@ref)
with `integrate`, by default [`brm_laplace_plan`](@ref)'s choice). It climbs the
rungs up to `upto` (1 to 6); every rung is a complete result
that starts the next:

1. **Effect modes** at the population start (`start`, a vector over the
   problem's population coordinates; default the prior medians).
2. **The mode of the embedded-Laplace marginal** ([`brm_laplace_mode`](@ref);
   needs Optim, which `using WarmupHMC` loads). With `integrate = :auto` the
   plan is measured again at this point, and where a gated term switches the
   problem is rebuilt and the mode found again from the same point.
3. **SAEM** from rung 2's point over the units the call names
   (`subject_group`, `nested_groups`, `shared_groups`, as for
   [`brm_nlme_model`](@ref)), integrating the families of the plan's admitted
   terms and sampling the rest (NLMEEstimation 0.2.4; Laplace-EM when every
   family is integrated; `saem_options` are its `SAEMOptions`, or a tuple or
   vector of options for successive phases). Each phase starts at the previous
   phase's packed point with fresh stochastic averaging. With `:auto`, its
   final point re-measures the plan for the next phase or rung. `fit.ladder`
   records its gap to rung 2: the difference of the Laplace target's values
   at the two points (rung 2 is its maximum; SAEM's MAP is taken in
   NLMEEstimation's covariance coordinates, so the gap holds that change of
   coordinates as well as the Laplace error) and the largest coordinate gap.
   Without named units, or without NLMEEstimation loaded, it is skipped and
   named.
4. **Effects given the point:** `draws` Laplace draws through the factor
   ([`brm_laplace_draws`](@ref), `conditional` proposals per draw for
   importance resampling).
5. **(5a) The population's curvature** at the point, from central differences
   of the Laplace gradient: a dense matrix over the population coordinates
   only, labelled a diagnostic. Its standard errors are never the delivered
   posterior. **(5b) HMC over the Laplace marginal:** `sampler(problem, init,
   rng)` draws the population coordinates from the sparse problem started at
   the point, and each draw's integrated effects are drawn through the factor.
   Its check's `comparison` reports the HMC mean and standard deviation,
   the mean shift in curvature standard errors, and the ratio of standard
   deviations, in `problem.population` order.
6. **Exact:** `sampler` on the full model ([`rk_logdensity_problem`](@ref)),
   started from the last draw of the highest rung below.

`sampler = (problem, init, rng) -> draws` (dimension × draws) runs rungs 5b
and 6, for example
`(p, init, rng) -> Matrix(adaptive_warmup_mcmc(rng, p; init).posterior_position)`
with WarmupHMC; without it they are skipped and named.

Returns `(; draws, point, problem, plan, rungs, ladder)`:
- `draws`: the packed draws of the highest sampling or point rung climbed
  (coordinates × draws); the curvature diagnostic keeps rung 4's draws;
- `point`: the population point (rung 3's, else rung 2's);
- `problem`, `plan`: the sparse problem and its plan;
- `rungs`: each rung's result;
- `ladder`: one check per rung, a rung skipped with its reason.

`continue_from = fit` continues an earlier fit of the same RK backend,
preserving its automatic or explicit integration choice. A previously skipped
rung runs when its units or sampler are supplied. Running SAEM on continuation
rebuilds the later rungs from its new point.
"""
function brm_ladder_fit(backend::RKBRMI; ad_backend, upto::Integer=4, integrate=:auto,
        start=nothing, draws::Integer=1000, conditional::Integer=0,
        rng=Random.default_rng(), continue_from=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), saem_options=nothing,
        sampler=nothing)
    1 <= upto <= 6 || throw(ArgumentError(
        "brm_ladder_fit: upto = $upto; the ladder's rungs are 1 to 6"))
    rungs = continue_from === nothing ? NamedTuple[] : copy(continue_from.rungs)
    ladder = continue_from === nothing ? NamedTuple[] : copy(continue_from.ladder)
    done(r) = any(x -> x.rung == r, rungs)
    p = continue_from === nothing ?
        brm_laplace_problem(backend; ad_backend, integrate) : continue_from.problem
    p.backend === backend || throw(ArgumentError(
        "brm_ladder_fit: continue_from belongs to a different RK backend"))
    automatic = continue_from === nothing ? integrate === :auto : continue_from.automatic
    plan = p.plan
    has_units = subject_group !== nothing || !isempty(nested_groups)
    if upto >= 3 && has_units && !isempty(methods(_brm_ladder_saem)) &&
            any(r -> r.rung == 3 && get(r, :skipped, false), rungs)
        # A new SAEM point invalidates every result built from the old point.
        filter!(r -> r.rung < 3, rungs)
        filter!(r -> r.rung < 3, ladder)
    end
    if sampler !== nothing
        keep(r) = !(get(r, :skipped, false) && r.name in (:population_hmc, :exact))
        filter!(keep, rungs)
        filter!(keep, ladder)
    end
    if !done(1)
        φ0 = start === nothing ? _brm_laplace_prior_median(p) : collect(Float64, start)
        state = _brm_spl_mode!(p, φ0)
        switched = Symbol[]
        if automatic && start !== nothing && state.finite
            u = copy(state.u)
            p, plan, switched = _brm_ladder_replan(p, u; ad_backend, rng)
            φ0 = u[p.population]
            state = _brm_spl_mode!(p, φ0)
        end
        push!(rungs, (; rung=1, name=:effect_modes, point=φ0,
            draws=_brm_ladder_packed(p, state)))
        push!(ladder, (; rung=1, name=:effect_modes, finite=state.finite,
            newton_steps=state.finite ? state.iterations : 0,
            components=length(p.components), switched))
        state.finite || _brm_nlme_error("brm_ladder_fit: rung 1: the Laplace is not " *
            "finite at the population start")
    end
    if upto >= 2 && !done(2)
        φ1 = only(r.point for r in rungs if r.rung == 1)
        mode = brm_laplace_mode(p; init=φ1)
        replanned = false
        switched = Symbol[]
        if automatic
            u = vec(mode.draws)
            p, plan, switched = _brm_ladder_replan(p, u; ad_backend, rng)
            if !isempty(switched)
                mode = brm_laplace_mode(p; init=u[p.population])
                replanned = true
            end
        end
        φ2 = mode.draws[p.population, 1]
        push!(rungs, (; rung=2, name=:marginal_mode, point=φ2, draws=mode.draws))
        push!(ladder, (; rung=2, name=:marginal_mode, mode.diagnostics.converged,
            mode.diagnostics.gradient_norm, mode.diagnostics.iterations, replanned,
            switched))
    end
    if upto >= 3 && !done(3)
        reason = subject_group === nothing && isempty(nested_groups) ?
            "the call names no units (`subject_group`, `nested_groups`), so there is " *
                "no SAEM over them" :
            isempty(methods(_brm_ladder_saem)) ?
                "SAEM comes from NLMEEstimation; load it (`using NLMEEstimation`)" : nothing
        if reason === nothing
            r2 = only(r for r in rungs if r.rung == 2)
            reference = p
            options = saem_options isa Union{Tuple,AbstractVector} ?
                saem_options : (saem_options,)
            isempty(options) && throw(ArgumentError(
                "brm_ladder_fit: saem_options must contain at least one phase"))
            phases = NamedTuple[]
            u = vec(r2.draws)
            for option in options
                phase_start = copy(u)
                out = _brm_ladder_saem(backend, u,
                    Set(b.declaration for b in p.blocks); ad_backend,
                    subject_group, nested_groups, shared_groups, rng, options=option)
                u = out.point
                switched = Symbol[]
                if automatic
                    p, plan, switched = _brm_ladder_replan(p, u; ad_backend, rng)
                end
                push!(phases, (; start=phase_start, out.point, out.fit,
                    out.integrated, switched, plan))
            end
            out = last(phases)
            φ3 = out.point[p.population]
            # Keep the gap on rung 2's target even when later phases change
            # the integrated set and hence the population coordinates.
            v2 = LogDensityProblems.logdensity(reference, r2.draws[reference.population, 1])
            v3 = LogDensityProblems.logdensity(reference, out.point[reference.population])
            push!(rungs, (; rung=3, name=:saem, point=φ3, draws=reshape(out.point, :, 1),
                saem=out.fit, phases))
            push!(ladder, (; rung=3, name=:saem, integrated=out.integrated,
                kernel=out.fit.kernel, iterations=sum(x.fit.iterations for x in phases),
                phases=length(phases), switched=[x.switched for x in phases],
                gap=v2 - v3, max_coordinate_gap=maximum(abs,
                    out.point[reference.population] .- r2.draws[reference.population, 1])))
        else
            push!(ladder, (; rung=3, name=:saem, skipped=true, reason))
            push!(rungs, (; rung=3, name=:saem, skipped=true))
        end
    end
    point = _brm_ladder_point(rungs)
    if upto >= 4 && !done(4)
        out = brm_laplace_draws(p, repeat(point, 1, draws); rng, conditional)
        push!(rungs, (; rung=4, name=:effect_draws, point, out.draws))
        check = (; rung=4, name=:effect_draws, draws, conditional,
            newton_steps=maximum(out.diagnostics.newton_steps))
        conditional > 0 && (check = (; check...,
            importance_ess_min=minimum(out.diagnostics.importance_ess)))
        push!(ladder, check)
    end
    named(r, n) = any(x -> x.rung == r && x.name === n, rungs)
    if upto >= 5 && !named(5, :population_curvature)
        curvature = _brm_ladder_curvature(p, point)
        push!(rungs, (; rung=5, name=:population_curvature, point,
            draws=_brm_ladder_packed(p, _brm_spl_mode!(p, point)), curvature...))
        push!(ladder, (; rung=5, name=:population_curvature, diagnostic=true,
            curvature.positive_definite,
            standard_errors=curvature.standard_errors))
    end
    no_sampler = "pass `sampler = (problem, init, rng) -> draws` (dimension × draws), " *
        "for example `Matrix(adaptive_warmup_mcmc(rng, problem; init).posterior_position)` " *
        "with WarmupHMC"
    if upto >= 5 && !named(5, :population_hmc)
        if sampler === nothing
            push!(ladder, (; rung=5, name=:population_hmc, skipped=true, reason=no_sampler))
            push!(rungs, (; rung=5, name=:population_hmc, skipped=true))
        else
            P = sampler(p, copy(point), rng)
            size(P, 1) == length(p.population) || throw(DimensionMismatch(
                "brm_ladder_fit: rung 5b: the sampler returned $(size(P, 1)) rows for " *
                "$(length(p.population)) population coordinates"))
            out = brm_laplace_draws(p, P; rng)
            push!(rungs, (; rung=5, name=:population_hmc, point, out.draws))
            curvature = only(r for r in rungs if r.name === :population_curvature)
            sample_mean = vec(mean(P; dims=2))
            sample_sd = vec(std(P; dims=2))
            comparison = (; mean=sample_mean, standard_deviations=sample_sd,
                mean_shift_in_standard_errors=(sample_mean .- point) ./ curvature.standard_errors,
                standard_deviation_ratio=sample_sd ./ curvature.standard_errors)
            push!(ladder, (; rung=5, name=:population_hmc, draws=size(P, 2),
                newton_steps=maximum(out.diagnostics.newton_steps), comparison))
        end
    end
    if upto >= 6 && !done(6)
        if sampler === nothing
            push!(ladder, (; rung=6, name=:exact, skipped=true, reason=no_sampler))
            push!(rungs, (; rung=6, name=:exact, skipped=true))
        else
            # started from the last draw of the highest rung below it
            below = last(sort([r for r in rungs if _brm_ladder_has_draws(r) && r.rung < 6];
                by=r -> (r.rung, r.name === :population_hmc)))
            init = below.draws[:, end]
            q = rk_logdensity_problem(backend; ad_backend, u0=copy(init))
            D = sampler(q, init, rng)
            size(D, 1) == length(init) || throw(DimensionMismatch(
                "brm_ladder_fit: rung 6: the sampler returned $(size(D, 1)) rows for " *
                "$(length(init)) packed coordinates"))
            push!(rungs, (; rung=6, name=:exact, point, draws=D))
            push!(ladder, (; rung=6, name=:exact, draws=size(D, 2), started_from=below.rung))
        end
    end
    top = last(sort([r for r in rungs if _brm_ladder_has_draws(r)];
        by=r -> (r.rung, r.name === :population_hmc)))
    (; draws=top.draws, point, problem=p, plan, automatic, rungs, ladder)
end

_brm_ladder_has_draws(r) = !get(r, :skipped, false) && r.name !== :population_curvature

# Re-measure at a complete packed point. Rebuild only when terms switch;
# callers map the packed point into the resulting population coordinates.
function _brm_ladder_replan(p, u; ad_backend, rng)
    r = _brm_laplace_plan(p.backend; ad_backend, integrate=:auto, sample=(),
        ess_floor=p.plan.ess_floor, draws=p.plan.draws, at=u, rng,
        tol=p.tol, maxiter=p.maxiter)
    before = Set(t.declaration for t in _brm_plan_admitted(p.plan))
    after = Set(t.declaration for t in _brm_plan_admitted(r.plan))
    switched = sort!(collect(symdiff(before, after)))
    if !isempty(switched)
        p = _brm_sparse_laplace_problem_from_plan(p.backend, r; ad_backend,
            tol=p.tol, maxiter=p.maxiter)
    end
    p.plan = r.plan
    p, r.plan, switched
end

# The highest point a rung reached: rung 3's SAEM estimate, else rung 2's mode,
# else rung 1's start.
_brm_ladder_point(rungs) = last(sort([r for r in rungs
    if !get(r, :skipped, false) && r.rung in (1, 2, 3)]; by=r -> r.rung)).point

# Implemented by `BayesianRegressionModelsNLMEEstimationExt`: SAEM on the
# model's units from a packed point, integrating the families whose terms are
# in `admitted`; returns its estimate as a packed point.
function _brm_ladder_saem end

# One complete packed point (coordinates × 1) at a mode state.
_brm_ladder_packed(p, state) = reshape(copy(state.u), :, 1)

# The Laplace marginal's curvature over the population coordinates at `φ`:
# central differences of its gradient (2p gradients), symmetrized. Its inverse
# negative is a Gaussian's covariance, reported as a diagnostic only.
function _brm_ladder_curvature(p::BRMSparseLaplaceProblem, φ; h=1e-4)
    n = length(φ)
    H = zeros(n, n)
    for j in 1:n
        e = zeros(n)
        e[j] = h * max(1.0, abs(φ[j]))
        gp = last(LogDensityProblems.logdensity_and_gradient(p, φ .+ e))
        gm = last(LogDensityProblems.logdensity_and_gradient(p, φ .- e))
        H[:, j] .= (gp .- gm) ./ (2e[j])
    end
    H = (H + H') / 2
    F = cholesky(Symmetric(-H); check=false)
    positive_definite = issuccess(F)
    covariance = positive_definite ? Matrix(inv(F)) : fill(NaN, n, n)
    (; hessian=H, covariance, positive_definite,
        standard_errors=positive_definite ? sqrt.(diag(covariance)) : fill(NaN, n))
end
