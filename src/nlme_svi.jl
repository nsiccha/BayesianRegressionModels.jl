# src/nlme_svi.jl — minibatch stochastic variational inference over an
# RK-lowered population model's own packed coordinates.
#
# The target is the RK sampler density p(u) over the packed unconstrained point
# u = (g, z_1, …, z_N). Structural likelihood and prior supports assign each
# estimated coordinate to one unit or the globals. Atomic prior/transform
# blocks stay together; a unit includes its standardized subject draws and
# ordinary sampled coordinates only it reads. Given g the units are
# conditionally independent:
#
#     log p(u) = P(g) + Σ_i [U_i(g, z_i) + ℓ_i(g, z_i)],
#
# with every authored prior and coordinate Jacobian in exactly one P or U_i
# bucket, and ℓ_i the unit's observation cells. Standardized-draw-only units
# retain the equivalent N(0, I) fast path. A batch of B units, its terms
# scaled by N/B, is an unbiased estimate of that sum and its gradient.
#
# The variational family is Gaussian and factorizes into a global factor and
# one factor per subject:
#
#     g   = μ + exp.(ρ) .* ε + Β η,         ε ~ N(0, I), η ~ N(0, I_r)
#     z_i ~ N(m_i, Λ_i⁻¹)
#
# The global covariance is diagonal (`MeanFieldGlobal`) or diagonal plus rank r
# (`LowRankGlobal(r)`, Β is D × r); Adam moves it every iteration along the
# minibatch ELBO gradient. A subject's factor moves only when the subject is in
# the batch, by a natural-gradient step: the improved Bayesian learning rule of
# Lin, Schmidt and Khan (2020), whose curvature estimate comes from the same
# antithetic pair of reparameterized draws (Stein's lemma) and whose precision
# update stays positive definite, inside a per-subject trust region on the
# precision's relative change and the mean's move (as Theis and Hoffman (2015)
# bound SVI's natural-gradient steps). That step is Newton-like, so a subject's
# factor settles in tens of visits, where plain gradient steps on its mean and
# scale need hundreds.
#
# An iteration's unit work (the terms at the antithetic pair and the factor's
# step) can run on several tasks, one prepared copy of the cell samplers per
# task. Its random draws stay on the run's one stream, in batch order, and each
# unit's result goes to its own slot and is summed in batch order, so a run is
# the same to the last bit at any number of tasks.

"""
    MeanFieldGlobal()
    LowRankGlobal(rank)

The global variational factor of [`brm_svi`](@ref): a diagonal Gaussian, or a
diagonal plus a rank-`rank` factor.
"""
abstract type BRMSVIFamily end
struct MeanFieldGlobal <: BRMSVIFamily end
struct LowRankGlobal <: BRMSVIFamily
    rank::Int
    LowRankGlobal(rank::Integer) = rank >= 1 ? new(rank) :
        throw(ArgumentError("LowRankGlobal: rank must be at least 1, got $rank"))
end
_brm_svi_rank(::MeanFieldGlobal) = 0
_brm_svi_rank(f::LowRankGlobal) = f.rank

include("nlme_svi_sampling.jl")
include("nlme_svi_workspace.jl")
include("nlme_svi_units.jl")

# ── The minibatch target ─────────────────────────────────────────────────────

# Packed positions of the global coordinates: everything the NLME model
# estimates as θ or σ, and the covariance coordinates the NLME reading leaves to
# its estimators. Fixed shared latents (`shared_point`) stay at `m.unit`.
_brm_svi_globals(m::BRMNLMEModel) = sort!(vcat(m.theta, m.sigma,
    findall(r -> r === :subject_scale || r === :subject_correlation, m.view.role)))

function _brm_svi_target(m::BRMNLMEModel; global_coordinates=nothing, previous=nothing,
        separator=nothing, max_unit_coordinates=4096, max_unit_work=1_000_000,
        max_promoted_coordinates=128)
    m.cell_query === nothing && _brm_nlme_error(
        "brm_svi: minibatch SVI evaluates one subject at a time, on the " *
        "subject's own observation cells, and this model's observations have " *
        "no cells attributable to single subjects")
    # A carried checkpoint brings its own separator policy, which a changed
    # partition cannot use: name that change, not the policy it invalidates.
    previous === nothing || _brm_svi_check_partition(m.view, previous)
    connected = m.view.unit_partition === :connected
    policy = something(separator, connected ? :auto : :none)
    !connected && policy !== :none && throw(ArgumentError(
        "brm_svi_fit: automatic separators require unit_partition=:connected"))
    ownership = connected ? _brm_svi_connected_ownership(m; global_coordinates, previous,
        separator=policy, max_unit_coordinates, max_unit_work, max_promoted_coordinates) :
        _brm_svi_ownership(m; global_coordinates, previous)
    globals, units = ownership.globals, ownership.units
    # One strict partition sampler per worker; sparse selection metadata shares
    # globals and retains only each unit's local permutation and merge slots.
    partition, partition_view = connected ? _brm_svi_partition(m, ownership) :
        (m.cell_gradient, m.view)
    plan = _brm_nlme_partition_plan(partition, m.unit, globals, partition_view; units)
    layouts = [(; selection.order, selection.slots) for selection in plan.indices]
    width = length(globals) + maximum(length, units; init=0)
    work = [(; values=zeros(width), gradient=zeros(width))]
    nz = count(==(:subject_effect), m.view.role)
    # Connected targets use selected authored buckets even for standardized
    # draws: the legacy global-prior path traverses every local prior at zero.
    prior = !connected && sum(length, units; init=0) == nz &&
        all(j -> m.view.role[j] === :subject_effect, Iterators.flatten(units)) ? nothing :
        _brm_svi_authored_prior(m, globals, units; factors=ownership.factors)
    (; m, globals, units, ownership, previous, partition, plan, plans=[plan], work, layouts, prior,
       z_constant=-nz * log(2π) / 2, prior_point=copy(m.unit))
end

# `t` with `n` partition plans, one per task that evaluates units. `t.plan` is
# the first; each further one owns its sampler, gradients and work point
# (`_brm_nlme_worker_plan`).
function _brm_svi_with_workers(t, n, progress=nothing)
    n == 1 && return t
    workers = _brm_with_counter(progress, n - 1; description="Worker partition samplers") do node
        map(1:n-1) do k
            plan = _brm_nlme_worker_plan(t.plan, t.partition, t.m.unit)
            _brm_update_progress!(node, k)
            plan
        end
    end
    width = length(first(t.work).values)
    work = vcat([first(t.work)],
        [(; values=zeros(width), gradient=zeros(width)) for _ in workers])
    merge(t, (; plans=vcat([t.plan], workers), work))
end

# `f(plan, k)` for every `k` in `1:n`: in order on this task with one plan, or
# with several on one task per plan, each taking the next `k` as it finishes
# the last, so uneven units balance. `f` writes only to what belongs to `k`.
function _brm_svi_each_unit(f, plans, n)
    if length(plans) == 1
        plan = only(plans)
        for k in 1:n
            f(plan, k)
        end
        return nothing
    end
    next = Threads.Atomic{Int}(1)
    tasks = map(plans) do plan
        Threads.@spawn begin
            k = Threads.atomic_add!(next, 1)
            while k <= n
                f(plan, k)
                k = Threads.atomic_add!(next, 1)
            end
        end
    end
    # Every task has stopped before a failure propagates: none of them is left
    # writing to the run's factors or a plan's samplers.
    failure = nothing
    for task in tasks
        try
            wait(task)
        catch e
            failure === nothing && (failure = e)
        end
    end
    failure === nothing || throw(failure)
    nothing
end

# P(g) and its gradient: the RK prior and log-Jacobian at the packed point with
# every subject draw at zero, minus those draws' N(0, 1) constants.
_brm_svi_global_term(t, g) = _brm_svi_global_term(t.prior, t, g)
function _brm_svi_global_term(::Nothing, t, g)
    u = t.prior_point
    u[t.globals] .= g
    value, gradient = _rk_nlme_population_prior_value_gradient(t.m.population_prior, u)
    value - t.z_constant, gradient[t.globals]
end

# Each worker owns its plan and one G + max(d_i) workspace. Unit layout
# metadata stores local positions only; globals are never repeated per unit.
function _brm_svi_unit_term!(gg, gz, t, i, g, z, plan=t.plan, work=first(t.work))
    D = length(t.globals)
    layout = t.layouts[i]
    values = view(work.values, 1:D+length(z))
    gradient = view(work.gradient, 1:D+length(z))
    _brm_svi_joint!(values, layout, g, z)
    fill!(gradient, 0.0)
    value = _brm_nlme_partition_sum!(gradient, plan, t.m.unit, i, values)
    _brm_svi_split_gradient!(gg, gz, layout, gradient)
    value + _brm_svi_unit_prior!(gg, gz, t.prior, t, i, g, z)
end

# Unit i's prior term, its gradients added into `gg` (the globals) and `gz`
# (the unit's block). A standardized subject draw's N(0, I) reads no global.
function _brm_svi_unit_prior!(gg, gz, ::Nothing, t, i, g, z)
    gz .-= z
    -sum(abs2, z) / 2 - length(z) * log(2π) / 2
end

# ── Authored unit priors ─────────────────────────────────────────────────────

# The exact prior split for units whose blocks hold ordinary sampled
# coordinates, not only standardized subject draws: RK's authored prior and
# coordinate-Jacobian factors (ReactiveKernelsPPL `density_factors`), each
# assigned once by the coordinates it owns or reads. A factor touching unit i's
# block is part of U_i(g, z_i), with its reads of the globals (a hyperparameter
# such as `z ~ Normal(m, tau)`); a factor touching no unit is part of G(g). A
# factor touching two units, such as a joint law or a coupled transform across
# them, has no split into unit terms and is refused when the prior is built.
#
# `queries[1]` is the global term and `queries[1 + i]` unit i's, `nothing` when
# empty. Each query differentiates its own structural support only. For its
# k-th active coordinate, `places[b][k]` is p > 0 for `globals[p]`, -p for the
# unit block's p-th coordinate, and 0 for a coordinate SVI does not move (a
# fixed shared value), whose value `active[b][k]` holds from `base`.
struct _BRMSVIAuthoredPrior{Q}
    queries::Vector{Q}
    places::Vector{Vector{Int}}
    active::Vector{Vector{Float64}}
    base::Vector{Float64}
end

# Implemented by the ReactiveKernels extension: the factor queries above for
# `owner` (the unit owning each packed coordinate, 0 for none), the packed
# coordinates each differentiates, and one query's value and compact gradient.
function _rk_svi_factor_queries end
function _rk_svi_factor_coordinates end
function _rk_svi_factor_value_gradient end

function _brm_svi_authored_prior(m::BRMNLMEModel, globals, units;
        factors=_rk_svi_factors(m.backend, m.unit))
    n = length(m.unit)
    owner, place = zeros(Int, n), zeros(Int, n)
    place[globals] .= eachindex(globals)
    for (i, block) in enumerate(units)
        allunique(block) || _brm_nlme_error("brm_svi: packed coordinate " *
            "$(block[findfirst(j -> count(==(j), block) > 1, block)]) appears twice in unit $i")
        clash = findfirst(j -> owner[j] != 0 || place[j] != 0, block)
        clash === nothing || _brm_nlme_error("brm_svi: packed coordinate " *
            "$(block[clash]) is assigned to unit $i and to " *
            (owner[block[clash]] != 0 ? "unit $(owner[block[clash]])" : "the globals"))
        owner[block] .= i
        place[block] .= .-eachindex(block)
    end
    queries = _rk_svi_factor_queries(m.backend, m.unit,
        m.population_prior.ad_backend, owner, length(units), factors)
    places = map(enumerate(queries)) do (b, query)
        query === nothing && return Int[]
        map(_rk_svi_factor_coordinates(query)) do j
            # The extension assigned each factor to one bucket by its reads, so a
            # bucket only reads the globals, fixed values and its own unit.
            owner[j] == 0 || owner[j] == b - 1 || error("brm_svi: the authored " *
                "prior bucket $(b - 1) reads unit $(owner[j])'s coordinate $j")
            place[j]
        end
    end
    active = map(queries) do query
        query === nothing ? Float64[] : m.unit[_rk_svi_factor_coordinates(query)]
    end
    _BRMSVIAuthoredPrior(queries, places, active, copy(m.unit))
end

# One bucket's value, its gradient added into `gg` and `gz`.
function _brm_svi_factor_term!(gg, gz, prior::_BRMSVIAuthoredPrior, b, g, z)
    query = prior.queries[b]
    query === nothing && return 0.0
    places, active = prior.places[b], copy(prior.active[b])
    for (k, p) in enumerate(places)
        p > 0 ? (active[k] = g[p]) : p < 0 && (active[k] = z[-p])
    end
    value, gradient = _rk_svi_factor_value_gradient(query, prior.base, active)
    for (k, p) in enumerate(places)
        p > 0 ? (gg[p] += gradient[k]) : p < 0 && (gz[-p] += gradient[k])
    end
    value
end

function _brm_svi_global_term(prior::_BRMSVIAuthoredPrior, t, g)
    gradient = zeros(length(g))
    _brm_svi_factor_term!(gradient, Float64[], prior, 1, g, Float64[]), gradient
end

_brm_svi_unit_prior!(gg, gz, prior::_BRMSVIAuthoredPrior, t, i, g, z) =
    _brm_svi_factor_term!(gg, gz, prior, 1 + i, g, z)

# ── Variational parameters ───────────────────────────────────────────────────

# Subject i's factor is `M[i]`, `Λ[i]` and `U[i]`, sized by its own block.
# Large units use bounded-rank operators; only small units store dense arrays.
function _brm_svi_params(t, μ, family::BRMSVIFamily; global_scale, unit_scale,
        unit_dense_limit=64, unit_rank=8)
    _brm_svi_unit_options(unit_dense_limit, unit_rank)
    isfinite(unit_scale) && unit_scale > 0 || throw(ArgumentError(
        "brm_svi_fit: unit_scale must be finite and positive"))
    D = length(t.globals)
    factors = [_brm_svi_unit_factors(length(b), unit_scale, unit_dense_limit, unit_rank)
        for b in t.units]
    (; family, μ=Vector{Float64}(μ), ρ=fill(log(global_scale), D),
       Β=zeros(D, _brm_svi_rank(family)), M=[zeros(length(b)) for b in t.units],
       Λ=Union{Matrix{Float64},_BRMSVIUnitPrecision}[f[1] for f in factors],
       U=Union{Matrix{Float64},_BRMSVIUnitRoot}[f[2] for f in factors],
       unit_opt=Union{Nothing,_BRMSVIAdam}[f[3] for f in factors])
end

_brm_svi_global_covariance(q) = Diagonal(exp.(2 .* q.ρ)) + q.Β * q.Β'

# Work with the diagonal plus low-rank representation throughout optimization.
# Only the r × r Gram matrix is dense; D may include thousands of field weights.
function _brm_svi_global_geometry(q)
    s = exp.(q.ρ)
    V = q.Β ./ s
    K = cholesky(Symmetric(Matrix{Float64}(I, size(V, 2), size(V, 2)) + V' * V))
    (; s, V, K)
end

function _brm_svi_global_precision(h, x)
    y = x ./ h.s
    (y - h.V * (h.K \ (h.V' * y))) ./ h.s
end

_brm_svi_global_logdet(h) = 2sum(log, h.s; init=0.0) + logdet(h.K)

# C = diag(s) (I + V A V'), C C' = diag(s²) + ΒΒ'. The stable divided
# difference 1/(sqrt(1+λ)+1) also works when a rank column is initially zero.
function _brm_svi_global_sqrt(h)
    E = eigen(Symmetric(h.V' * h.V))
    W = h.V * E.vectors
    a = 1 ./ (sqrt.(1 .+ max.(E.values, 0)) .+ 1)
    (; s=h.s, W, a)
end
_brm_svi_global_sqrt_mul(c, x) = c.s .* (x + c.W * (c.a .* (c.W' * x)))
_brm_svi_global_sqrt_tmul(c, x) = begin
    y = c.s .* x
    y + c.W * (c.a .* (c.W' * y))
end

# A draw of the globals from its two standard normal parts.
_brm_svi_draw_global(q, ε, η) = q.μ .+ exp.(q.ρ) .* ε .+ q.Β * η
# A draw of subject i from ε_i: m_i + U_i⁻ᵀ ε_i, with U_i the lower Cholesky
# factor of the precision Λ_i.
_brm_svi_unit_offset(U::AbstractMatrix, ε) = LowerTriangular(U)' \ ε
_brm_svi_unit_offset(q, i, εi) = _brm_svi_unit_offset(q.U[i], εi)
_brm_svi_draw_unit(q, i, εi) = q.M[i] .+ _brm_svi_unit_offset(q, i, εi)

# The entropy of q, up to the constant (D + Σ d_i)(1 + log 2π)/2.
_brm_svi_entropy(q) = _brm_svi_global_logdet(_brm_svi_global_geometry(q)) / 2 -
    sum(_brm_svi_unit_logdet_root, q.U; init=0.0)
_brm_svi_unit_entropy(q, i) = -_brm_svi_unit_logdet_root(q.U[i])

# ── The global factor: Adam ──────────────────────────────────────────────────

mutable struct _BRMSVIAdam
    lr::Float64
    β1::Float64
    β2::Float64
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end
_BRMSVIAdam(n, lr) = _BRMSVIAdam(lr, 0.9, 0.999, zeros(n), zeros(n), 0)

# One ascent step of `x` along `g`.
function _brm_svi_step!(opt::_BRMSVIAdam, x, g, lr=opt.lr)
    opt.t += 1
    @. opt.m = opt.β1 * opt.m + (1 - opt.β1) * g
    @. opt.v = opt.β2 * opt.v + (1 - opt.β2) * g^2
    b1, b2 = 1 - opt.β1^opt.t, 1 - opt.β2^opt.t
    @. x += lr * (opt.m / b1) / (sqrt(opt.v / b2) + 1e-8)
    x
end

# The global parameters as one vector [μ; ρ; vec(Β)].
_brm_svi_global_vector(q) = vcat(q.μ, q.ρ, vec(q.Β))
function _brm_svi_set_global!(q, x)
    D = length(q.μ)
    q.μ .= view(x, 1:D)
    q.ρ .= view(x, D+1:2D)
    vec(q.Β) .= view(x, 2D+1:length(x))
    q
end

# The gradient of the global vector from w, the ELBO's gradient in the global
# coordinates, the draw's (ε, η), and the entropy's own gradient.
function _brm_svi_global_gradient(q, w, ε, η)
    h = _brm_svi_global_geometry(q)
    # diag(I - V (I + V'V)^-1 V') without a D × D intermediate.
    diagonal = 1 .- vec(sum(abs2, h.V / h.K.U; dims=2))
    gρ = w .* ε .* h.s .+ diagonal
    gΒ = w * η' .+ _brm_svi_global_precision(h, q.Β)
    vcat(w, gρ, vec(gΒ))
end

# A scale-preconditioned ("whitened") global step for polishing: μ moves in
# units of a square root of the global covariance and each Β row in units of
# its scale, so Adam's step size means a fraction of the current posterior
# scale, which stays meaningful when a global's posterior sd shrinks like 1/√N.
function _brm_svi_whitened_global_step!(opt, q, x, grad, lr)
    D = length(q.μ)
    C = _brm_svi_global_sqrt(_brm_svi_global_geometry(q))
    scale = vcat(ones(D), vec(repeat(exp.(q.ρ), 1, size(q.Β, 2))))
    grad = copy(grad)
    grad[1:D] .= _brm_svi_global_sqrt_tmul(C, view(grad, 1:D))
    rest = D+1:length(x)
    grad[rest] .*= scale
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad, lr)
    x[1:D] .+= _brm_svi_global_sqrt_mul(C, view(step, 1:D))
    x[rest] .+= scale .* view(step, rest)
    x
end

# ── A subject's factor: one natural-gradient step ───────────────────────────

# The positive eigenpair `(λ, e)` of the negative-Hessian estimate in the
# factor's whitened coordinates. With w = -U⁻¹ gd the estimate reads sym(w εᵀ),
# whose only nonzero eigenvalues are (wᵀε ± ‖w‖‖ε‖)/2, on ŵ ± ε̂, so it needs no
# eigensolver and costs O(d²) for the triangular solve.
function _brm_svi_unit_curvature(U, gd, εi)
    w = -(U \ gd)
    a, b = norm(w), norm(εi)
    (iszero(a) || iszero(b)) && return 0.0, zero(w)
    λ = (dot(w, εi) + a * b) / 2
    isfinite(λ) || throw(ArgumentError(
        "brm_svi: non-finite gradient difference in a unit's curvature estimate"))
    e = w ./ a .+ εi ./ b
    n = norm(e)
    # ŵ = -ε̂: the estimate is negative semidefinite, its positive part zero.
    iszero(n) ? (0.0, zero(w)) : (λ, e ./ n)
end

# The largest relative change of a subject's precision in one step: no
# eigenvalue of U⁻¹ β G U⁻ᵀ exceeds it in magnitude, so along every direction
# the precision changes by a factor between ½ and 1 + τ + τ²/2 (2.5).
const _BRM_SVI_UNIT_TRUST = 1.0
# The longest step of a subject's mean, in sds of its factor before the step.
const _BRM_SVI_UNIT_MEAN_TRUST = 1.0

# The improved Bayesian learning rule step of size β for subject i, from the
# gradients of log p at the antithetic draws z± = m ± U⁻ᵀ ε: `gz` their mean,
# which estimates E_q ∇ log p, and `gd` half their difference. Stein's lemma
# gives E[∇ log p εᵀ] = E[∇² log p] U⁻ᵀ, so -sym(gd εᵀ Uᵀ) estimates the
# negative Hessian; the difference cancels the gradient at the mean, which
# otherwise dominates a single draw's estimate whenever m is many posterior sds
# from the subject's mode. With G = that estimate - Λ the precision moves to
# Λ + β G + β²/2 G Λ⁻¹ G = ½(Λ + (Λ + βG) Λ⁻¹ (Λ + βG)), positive definite for
# any G, and the mean to m + β Λ_new⁻¹ gz.
#
# A subject whose observations pin it far more tightly than its factor's
# start (hundreds of rows against the unit start) has a gradient of 1e4 and
# more, and an unbounded step multiplied its precision by up to 1e8 in one
# visit (β²/2 G Λ⁻¹ G dominates) and then lost positive definiteness in
# floating point. Besides the antithetic draws, two changes keep the step
# sound there:
# - a trust region: β shrinks for this step until ‖U⁻¹ β G U⁻ᵀ‖ ≤ τ
#   (`_BRM_SVI_UNIT_TRUST`), so the precision grows by at most 2.5× per visit
#   along the drawn direction, and the mean moves at most one sd of the
#   factor before the step (`_BRM_SVI_UNIT_MEAN_TRUST`): while the precision
#   is still far below the subject's, Λ_new⁻¹ gz is a Newton step with too
#   little curvature, which overshot by hundreds of sds where a log-scale
#   random effect makes the density steep. A subject's step never shortens
#   another subject's.
# - the square-root form: with Λ = U Uᵀ and A = Λ + βG,
#   Λ_new = ½ (Uᵀ; U⁻¹A)ᵀ (Uᵀ; U⁻¹A), so the R of that stacked matrix's QR
#   factorization is √2 times the new Cholesky factor, which is positive
#   definite in floating point too.
#
# G uses only the positive part of the estimate, in the factor's whitened
# coordinates (`_brm_svi_unit_curvature`). The full single-draw estimate is
# indefinite, and its negative eigenvalue traps a weak direction: where a
# direction v's precision is far below its target, v dominates the whitened
# gradient difference, the trust region sets β from the estimate's largest
# eigenvalue, and the negative one halves the precision along a vector holding
# about half of v. v's marginal variance then grows on average, on an exact
# Gaussian target too, and with it the draws, however strongly the target
# curves along v. At a matched factor the estimate is εεᵀ, whose negative part
# is zero, so the positive part changes nothing there. It is biased towards
# more precision wherever the two differ, including a target with negative
# curvature, whose true fixed point it no longer reaches.
_brm_svi_unit_step!(q, i, gz, gd, εi, β) =
    _brm_svi_unit_step!(q.Λ[i], q, i, gz, gd, εi, β)
function _brm_svi_unit_step!(::AbstractMatrix, q, i, gz, gd, εi, β)
    # Promoting all of a scientific subject's effects leaves an empty local
    # factor. Its observation term may still contribute to the globals.
    isempty(q.M[i]) && return q
    U = LowerTriangular(q.U[i])
    λ, e = _brm_svi_unit_curvature(U, gd, εi)
    v = U * e
    G = Symmetric(λ .* (v * v') .- q.Λ[i])
    # U⁻¹ G U⁻ᵀ = λ e eᵀ - I: its eigenvalues are λ - 1 and, above d = 1, -1.
    r = max(abs(λ - 1), length(e) > 1 ? 1.0 : 0.0)
    β = min(β, _BRM_SVI_UNIT_TRUST / r)
    R = qr(vcat(Matrix(U'), U \ (q.Λ[i] .+ β .* G))).R
    L = LowerTriangular(R' * Diagonal(sign.(diag(R))) ./ sqrt(2))
    Δ = β .* (L' \ (L \ gz))
    Δ .*= min(1.0, _BRM_SVI_UNIT_MEAN_TRUST / norm(U' * Δ))
    q.U[i] .= L
    q.Λ[i] .= L * L'
    q.M[i] .+= Δ
    q
end

# ── Non-finite evaluations ───────────────────────────────────────────────────

"""
    BRMSVINonfiniteError

Thrown by [`brm_svi_fit`](@ref) and [`brm_svi_resume`](@ref) when an
iteration evaluates the target to a non-finite value or gradient at its Monte
Carlo draws. It is raised after the draws are evaluated and before the
iteration changes the variational factors or the optimizer. `iteration` is the
iteration that failed, one past the last completed one. `terms` lists every
non-finite term in batch order, each a named tuple with:

- `term`: `:global` for the global prior term, `:unit` for a unit's term;
- `unit`: the unit's index, 0 for the global term;
- `quantities`: what was not finite, for example `:value_plus`,
  `:global_gradient_minus` or `:unit_gradient_plus` (a unit is evaluated at
  the antithetic pair of draws, plus and minus; the global term has `:value`
  and `:global_gradient`);
- `coordinates`: the names of the packed coordinates whose gradient entries
  were not finite;
- `draw_finite`: whether the point the term was evaluated at was finite.

The run in memory is not a checkpoint after this error, because its random
stream has advanced. Continue from the last checkpoint `save` received. A
continuation repeats the run's random stream exactly, so resuming that
checkpoint with unchanged controls reaches the same evaluation again.
"""
struct BRMSVINonfiniteError <: Exception
    iteration::Int
    terms::Vector{NamedTuple}
end

function Base.showerror(io::IO, e::BRMSVINonfiniteError)
    print(io, "BRMSVINonfiniteError: iteration ", e.iteration, " evaluated the ",
        "target to a non-finite value or gradient at its Monte Carlo draws, before ",
        "changing the variational factors or the optimizer")
    for p in e.terms
        print(io, "\n  ", p.term === :global ? "global prior term" : "unit $(p.unit)",
            ": ", join(p.quantities, ", "),
            p.draw_finite ? "" : " (the evaluated point is not finite)")
        isempty(p.coordinates) && continue
        print(io, "; non-finite gradient in ", length(p.coordinates), " coordinate(s): ",
            join(p.coordinates, ", "))
    end
    print(io, "\nThe run in memory is not a checkpoint after this error. Continue ",
        "from the last checkpoint `save` received; resuming it with unchanged ",
        "controls repeats the same random stream and reaches this evaluation again.")
end

# Unit i's terms at the antithetic pair `zs`: `nothing` when every value and
# gradient is finite, else what was not. Gradient entries map to `t.globals`
# (`ggs`) and `t.units[i]` (`gzs`).
function _brm_svi_nonfinite_unit(t, i, g, zs, values, ggs, gzs)
    all(isfinite, values) && all(x -> all(isfinite, x), ggs) &&
        all(x -> all(isfinite, x), gzs) && return nothing
    quantities, bad = Symbol[], Int[]
    for (side, v, gg, gz) in zip((:plus, :minus), values, ggs, gzs)
        isfinite(v) || push!(quantities, Symbol(:value_, side))
        all(isfinite, gg) || (push!(quantities, Symbol(:global_gradient_, side));
            append!(bad, t.globals[findall(!isfinite, gg)]))
        all(isfinite, gz) || (push!(quantities, Symbol(:unit_gradient_, side));
            append!(bad, t.units[i][findall(!isfinite, gz)]))
    end
    (; term=:unit, unit=i, quantities,
       coordinates=t.m.view.coordinates[sort!(unique!(bad))],
       draw_finite=all(isfinite, g) && all(z -> all(isfinite, z), zs))
end

# Throws `BRMSVINonfiniteError` (iteration 0, which the caller fills in) when
# the global term `P`, `gP` at `g` or a unit's terms (`problems`) are not finite.
function _brm_svi_check_finite(t, g, P, gP, problems)
    found = NamedTuple[]
    if !(isfinite(P) && all(isfinite, gP))
        quantities = Symbol[]
        isfinite(P) || push!(quantities, :value)
        all(isfinite, gP) || push!(quantities, :global_gradient)
        push!(found, (; term=:global, unit=0, quantities,
            coordinates=t.m.view.coordinates[t.globals[findall(!isfinite, gP)]],
            draw_finite=all(isfinite, g)))
    end
    for p in problems
        p === nothing || push!(found, p)
    end
    isempty(found) || throw(BRMSVINonfiniteError(0, found))
    nothing
end

# ── One minibatch iteration ──────────────────────────────────────────────────

# One draw of the globals, an antithetic pair of draws per batch subject, the
# minibatch ELBO gradient, an Adam step of the globals and a natural-gradient
# step of each batch subject. Returns the minibatch estimate of E_q log p
# (without the entropy).
#
# Whatever `t.plans` holds, the result is the same to the last bit: the draws
# come from `rng` here, in batch order, before any unit is evaluated, each unit's
# result stays in its own slot and is summed in batch order, and a unit's step
# reads and writes only that unit's factor.
function _brm_svi_iteration!(rng, t, q, global_opt, batch, lr_global, β;
        whiten=false, weights=fill(length(t.units) / length(batch), length(batch)))
    D = length(t.globals)
    ε, η = randn(rng, D), randn(rng, size(q.Β, 2))
    g = _brm_svi_draw_global(q, ε, η)
    P, gP = _brm_svi_global_term(t, g)
    draws = [_brm_svi_unit_noise(rng, q.U[i]) for i in batch]
    terms = Vector{@NamedTuple{value::Float64,gg::Vector{Float64},gz::Vector{Float64},
        gd::Vector{Float64}}}(undef, length(batch))
    problems = Vector{Union{Nothing,NamedTuple}}(nothing, length(batch))
    _brm_svi_each_unit(eachindex(t.plans), length(batch)) do worker, k
        plan, work = t.plans[worker], t.work[worker]
        i, εi = batch[k], draws[k]
        gg₊, gg₋ = zeros(D), zeros(D)
        gz₊, gz₋ = zeros(length(t.units[i])), zeros(length(t.units[i]))
        m, δ = q.M[i], _brm_svi_unit_offset(q, i, εi)
        z₊, z₋ = m .+ δ, m .- δ
        v₊ = _brm_svi_unit_term!(gg₊, gz₊, t, i, g, z₊, plan, work)
        v₋ = _brm_svi_unit_term!(gg₋, gz₋, t, i, g, z₋, plan, work)
        problems[k] = _brm_svi_nonfinite_unit(t, i, g, (z₊, z₋), (v₊, v₋),
            (gg₊, gg₋), (gz₊, gz₋))
        terms[k] = (; value=(v₊ + v₋) / 2, gg=gg₊ .+ gg₋, gz=(gz₊ .+ gz₋) ./ 2,
            gd=(gz₊ .- gz₋) ./ 2)
    end
    # Before anything changes q or the optimizer.
    _brm_svi_check_finite(t, g, P, gP, problems)

    w = copy(gP)
    batch_value = 0.0
    for (k, term) in enumerate(terms)
        batch_value += weights[k] * term.value
        w .+= (weights[k] / 2) .* term.gg
    end
    x = _brm_svi_global_vector(q)
    grad = _brm_svi_global_gradient(q, w, ε, η)
    whiten ? _brm_svi_whitened_global_step!(global_opt, q, x, grad, lr_global) :
        _brm_svi_step!(global_opt, x, grad, lr_global)
    _brm_svi_set_global!(q, x)
    _brm_svi_each_unit(t.plans, length(batch)) do _, k
        _brm_svi_unit_step!(q, batch[k], terms[k].gz, terms[k].gd, draws[k], β)
    end
    P + batch_value
end

const _BRM_SVI_PHASES = (lower="Lower to ReactiveKernels", prepare="Prepare NLME model",
    target="Minibatch target", fit="Stochastic optimization", draws="Draws from q")

# Stochastic optimization in two phases. Exploration runs at constant step
# sizes (`lr_global` for Adam, `unit_step` for the subjects' natural-gradient
# steps) until the ELBO stops improving detectably: the mean over a window (one
# pass over the subjects, at least 100 iterations) has risen, over the last
# `patience` windows, by less than twice the standard error of that rise, after
# at least `min_windows`. Comparing across several windows rather than
# neighbouring ones lets a slow drift show above the minibatch noise. Polishing then restarts Adam on scale-preconditioned global
# steps (`polish_lr`, a fraction of the current scale), decays every step size
# as 1/√k over `polish_windows` windows, and returns the average of q over
# those iterations (Polyak–Ruppert), which damps the noise a stochastic step
# leaves.
#
# Everything an iteration reads and writes besides the target lives in one
# `_BRMSVIRun`, so a run stops after any completed iteration and continues from
# a copy of it (`BRMSVIOptimizerState`, src/nlme_svi_optimizer.jl) exactly as
# if it had not stopped: the raw iterate `q` (during polishing too, where the
# result is the running `average` instead), Adam's moments and step count, the
# random stream, the iteration, the ELBO trace and window statistics the
# stopping rule reads, and where polishing started. A field added here is
# checkpointed with it.
mutable struct _BRMSVIRun{Q}
    rng::AbstractRNG
    q::Q
    adam::_BRMSVIAdam
    iteration::Int
    trace::Vector{Float64}
    windows::Vector{NamedTuple{(:mean, :var),Tuple{Float64,Float64}}}
    polish_start::Int
    average::Union{Nothing,Q}
    converged::Bool
    sampler::_BRMSVIUnitSampler
    sampling::NamedTuple
    unit_gradient_calls::Int
    averaged_through::Vector{Int}
    unit_entropy::Vector{Float64}
    entropy_sum::Float64
end

_brm_svi_adam(q, lr) = _BRMSVIAdam(length(_brm_svi_global_vector(q)), lr)
function _brm_svi_run(rng, q, c, t)
    N = length(t.units)
    # Structural work proxy, not a measured runtime.
    work = hasproperty(t, :ownership) && hasproperty(t.ownership, :work) ?
        copy(t.ownership.work) :
        [Float64(length(t.units[i]) + length(t.plan.footprints[i])) for i in 1:N]
    sampler = _brm_svi_unit_sampler(c.unit_sampling, work)
    sampling = (; method=c.unit_sampling, work, probabilities=sampler.probabilities,
        visits=zeros(Int, N), draws=zeros(Int, N))
    entropy = [_brm_svi_unit_entropy(q, i) for i in 1:N]
    _BRMSVIRun(rng, q, _brm_svi_adam(q, c.lr_global), 0, Float64[],
        NamedTuple{(:mean, :var),Tuple{Float64,Float64}}[], 0, nothing, false,
        sampler, sampling, 0, zeros(Int, N), entropy, sum(entropy))
end

# The optimizer controls of a run over `N` subjects, with the batch size and
# window it actually uses. They are fixed for the run and its continuations.
function _brm_svi_controls(N; batch_size, max_iterations, lr_global, unit_step,
        polish_lr, window, patience, min_windows, polish_windows, unit_sampling=:uniform)
    B = min(batch_size, N)
    (; batch_size=B, window=window === nothing ? max(cld(N, B), 100) : window,
       max_iterations, lr_global, unit_step, polish_lr, patience, min_windows,
       polish_windows, unit_sampling)
end

_brm_svi_finished(run, c) = run.converged || run.iteration >= c.max_iterations
_brm_svi_polishing(run) = run.polish_start > 0

# One iteration of the run, including the stopping rule's bookkeeping at the
# end of a window. Returns `:converged` when polishing completed with it,
# `:window` when it completed a window, `:iteration` otherwise.
function _brm_svi_advance!(run, t, c)
    N, W = length(t.units), c.window
    it = run.iteration + 1
    polishing = _brm_svi_polishing(run)
    scale = polishing ? 1 / sqrt(1 + (it - run.polish_start) / W) : 1.0
    q = run.q
    batch = _brm_svi_batch(run.rng, run.sampler, c.batch_size)
    polishing && _brm_svi_average_units!(run.average, q, run.averaged_through,
        batch.indices, it - run.polish_start - 1)
    for (k, i) in enumerate(batch.indices)
        run.sampling.visits[i] += 1
        run.sampling.draws[i] += batch.multiplicities[k]
    end
    run.unit_gradient_calls += 2length(batch.indices)
    value = try
        _brm_svi_iteration!(run.rng, t, q, run.adam, batch.indices,
            scale * (polishing ? c.polish_lr : c.lr_global), scale * c.unit_step;
            whiten=polishing, weights=batch.weights)
    catch e
        e isa BRMSVINonfiniteError || rethrow()
        throw(BRMSVINonfiniteError(it, e.terms))
    end
    for i in batch.indices
        entropy = _brm_svi_unit_entropy(q, i)
        run.entropy_sum += entropy - run.unit_entropy[i]
        run.unit_entropy[i] = entropy
    end
    push!(run.trace, value + run.entropy_sum +
        _brm_svi_global_logdet(_brm_svi_global_geometry(q)) / 2)
    run.iteration = it
    if polishing
        k = it - run.polish_start
        for name in (:μ, :ρ, :Β)
            _brm_svi_average!(getfield(run.average, name), getfield(q, name), k)
        end
        k >= c.polish_windows * W && (run.converged = true; return :converged)
    end
    it % W == 0 || return :iteration
    w = view(run.trace, it-W+1:it)
    push!(run.windows, (; mean=Statistics.mean(w), var=Statistics.var(w)))
    windows = run.windows
    if !polishing && length(windows) >= max(c.min_windows, c.patience + 1)
        a, b = windows[end-c.patience], windows[end]
        if b.mean - a.mean < 2 * sqrt((a.var + b.var) / W)
            run.polish_start = it
            run.average = deepcopy(q)
            run.adam = _brm_svi_adam(q, c.polish_lr)
        end
    end
    :window
end

_brm_svi_progress_message(run, c) = string(
    _brm_svi_polishing(run) && run.polish_start < run.iteration ?
        "polishing $(run.iteration - run.polish_start)/$(c.polish_windows * c.window)" :
        "exploring, window $(length(run.windows))",
    " · iteration $(run.iteration) · ELBO ≈ ", round(last(run.windows).mean; sigdigits=6))

# Advance `run` until it converges, reaches `max_iterations`, or has taken
# `stop_after` iterations in this call. `save(checkpoint)` receives a detached
# `BRMSVIOptimizerState` (`snapshot(run)`) after every iteration that is a
# multiple of `checkpoint_every`, before an interrupt requested at the end of
# a window propagates, and when this call stops. A checkpoint copies the run;
# it never advances the random stream.
function _brm_svi_fit!(run, t, c; stop_after=typemax(Int), checkpoint_every=c.window,
        save=nothing, snapshot=identity, node=nothing)
    commit() = save === nothing || save(snapshot(run))
    taken = 0
    while !_brm_svi_finished(run, c) && taken < stop_after
        status = _brm_svi_advance!(run, t, c)
        taken += 1
        if status === :window
            try
                _brm_throw_if_interrupted(node)
            catch
                commit()
                rethrow()
            end
            _brm_update_progress!(node, _brm_svi_progress_message(run, c))
        end
        save === nothing || run.iteration % checkpoint_every != 0 ||
            _brm_svi_finished(run, c) || taken >= stop_after || commit()
    end
    commit()
    run
end

# The approximation a run currently stands for: its raw iterate, or while and
# after polishing the average of the polished iterates with the Cholesky
# factors of the averaged precisions. The run itself is left unchanged, so it
# stays exactly continuable and its checkpoints do not depend on whether a
# result was read from it.
function _brm_svi_result_q(run)
    run.average === nothing && return run.q
    average = deepcopy(run.average)
    _brm_svi_average_units!(average, run.q, copy(run.averaged_through),
        eachindex(run.q.M), run.iteration - run.polish_start)
    _brm_svi_factor!(average)
end

# The running mean `a` of iterates `x`, at the k-th: an array, or one array
# per subject.
_brm_svi_average!(a::AbstractArray{<:Real}, x, k) = (@. a += (x - a) / k; a)
_brm_svi_average!(a::AbstractVector{<:AbstractArray}, x, k) =
    (foreach((ai, xi) -> _brm_svi_average!(ai, xi, k), a, x); a)

# Weighted running averages over stretches during which a unit was unchanged.
# Dense factors average precision; structured factors average log scales and
# factor loadings, retaining their bounded rank and positive covariance.
# This equals visiting every unit at every polishing iteration, but touches
# only the selected units until the single final flush.
function _brm_svi_average_units!(average, q, through, indices, iteration)
    for i in indices
        count = iteration - through[i]
        count == 0 && continue
        weight = count / iteration
        a, x = average.M[i], q.M[i]
        @. a += weight * (x - a)
        _brm_svi_unit_average_precision!(average.Λ[i], q.Λ[i], weight)
        through[i] = iteration
    end
    average
end

# The subjects' Cholesky factors of an averaged precision.
function _brm_svi_factor!(q)
    for i in eachindex(q.Λ)
        _brm_svi_unit_refactor!(q.U[i], q.Λ[i])
    end
    q
end

# `n` draws from q as packed RK points, one per column.
function _brm_svi_draws(rng, t, q, n)
    U = repeat(t.m.unit, 1, n)
    for j in 1:n
        U[t.globals, j] .= _brm_svi_draw_global(q, randn(rng, length(q.μ)),
            randn(rng, size(q.Β, 2)))
        for (i, block) in enumerate(t.units)
            U[block, j] .= _brm_svi_draw_unit(q, i, _brm_svi_unit_noise(rng, q.U[i]))
        end
    end
    _brm_svi_fill_absent_draws!(rng, U, t, t.previous)
end

_brm_svi_fill_absent_draws!(rng, draws, t, ::Nothing) = draws

# The model through its preparation phases, as for `brm_nlme_fit`: lowering to
# RK and `brm_nlme_model`. A stage `x` has already passed is reused.
_brm_svi_lowered(x::BRMI, phase) = _brm_run_phase(_ -> RKBRMI(x), phase)
_brm_svi_lowered(x::Union{RKBRMI,BRMNLMEModel}, phase) = (_brm_reuse_phase!(phase); x)
_brm_svi_lowered(x, phase) = throw(ArgumentError(
    "brm_svi_fit: expected a @brm model (BRMI), an RKBRMI or a BRMNLMEModel, " *
    "got $(typeof(x))"))

_brm_svi_model(x::RKBRMI, phase; ad_backend, shared_point, subject_group,
        nested_groups, shared_groups, unit_partition) =
    _brm_run_phase(phase) do node
        _brm_nlme_model(x, node; ad_backend=ad_backend === nothing ?
            _brm_rk_default_ad("brm_svi_fit", x) : ad_backend, observation_ad_backend=nothing,
            shared_point, subject_group, nested_groups, shared_groups, unit_partition=something(unit_partition, :subjects))
    end
function _brm_svi_model(m::BRMNLMEModel, phase; ad_backend, shared_point,
        subject_group, nested_groups, shared_groups, unit_partition)
    ad_backend === nothing && shared_point === nothing && subject_group === nothing &&
        isempty(nested_groups) && isempty(shared_groups) && unit_partition === nothing || throw(ArgumentError(
        "brm_svi_fit: a prepared BRMNLMEModel already fixes its ad_backend, " *
        "shared_point and subject groupings; pass them to brm_nlme_model instead"))
    _brm_reuse_phase!(phase)
    m
end

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: reverse-mode
# `AutoEnzyme`, the backend `brm_nlme_model` requires. `caller` names the entry
# point in the error when that extension is not loaded.
_brm_rk_default_ad(caller, x) = throw(ArgumentError(
    "$caller: pass `ad_backend` (reverse-mode AutoEnzyme), or load Enzyme " *
    "and DifferentiationInterface for that default"))

# The packed point the global and unit means start from: BRM's default prior
# starting point, a full packed `start`, or address-keyed `inits` over it.
function _brm_svi_start(m::BRMNLMEModel, start, inits)
    start === nothing || inits === nothing || throw(ArgumentError(
        "brm_svi_fit: give either `start` or `inits`, not both"))
    median = first(_brm_nlme_prior_median_point(m))
    start === nothing || return (length(start) == length(median) ||
        throw(DimensionMismatch("brm_svi_fit: `start` has $(length(start)) " *
            "coordinates, the RK layout has $(length(median))")); Vector{Float64}(start))
    inits === nothing && return median
    brm_start(m.backend, inits; base=median)
end

"""
    brm_svi_fit([save,] x; laplace=nothing, batch_size=32, rng=Random.default_rng(),
        family=MeanFieldGlobal(), unit_sampling=nothing, ndraws=1000, start=nothing, inits=nothing,
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=nothing,
        global_coordinates=nothing, progress=nothing,
        stop_after=nothing, checkpoint_every=nothing, threads=nothing,
        kwargs...) -> (; draws, coordinates, diagnostics)

Minibatch stochastic variational inference for a model whose subjects are
conditionally independent given the population (the subjects of
[`brm_nlme_view`](@ref)), over the RK backend's packed unconstrained
coordinates. Each iteration evaluates `batch_size` subjects, each at an
antithetic pair of draws through their own observation cells, and scales them
to all subjects, so its cost follows the batch size, not the number of
subjects.

With `laplace=GaussianBlockLaplace()`, jointly integrate certified Gaussian
coefficients, unit effects and shared fields inside the variational target.
This opt-in evaluates the entire current window, uses a fixed update budget
with convergence unassessed, and returns a correlated `BRMSVILaplaceState`
through [`brm_svi_checkpoint`](@ref). See [`GaussianBlockLaplace`](@ref) for
eligibility, numerical limits and the additional carry approximation.
The minibatch controls and per-subject factors below describe `laplace=nothing`.

- `x` — a `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref), or a prepared
  [`BRMNLMEModel`](@ref). For a BRMI or RKBRMI the model is prepared with
  `ad_backend` (default reverse-mode `AutoEnzyme`, which needs Enzyme loaded),
  `shared_point`, `subject_group`, `nested_groups` and `shared_groups`, as
  [`brm_nlme_model`](@ref) takes them; a nested model passes `subject_group`
  and `nested_groups`, a model with a grouping crossed with the subjects
  `shared_groups`. Every observation must belong to one subject's cells: a
  cell of a `@plate for` over the subjects or over their rows, or a row of a
  flat-row response grouped by the subjects.
- `family` — the global factor: `MeanFieldGlobal()` (diagonal) or
  `LowRankGlobal(r)` (diagonal plus rank `r`).
- `unit_dense_limit` — units with at most this many coordinates use a dense
  Gaussian and natural-gradient updates (default 64, allowed 0:256).
  Larger units use a diagonal-plus-low-rank covariance with `unit_rank`
  columns (default 8, allowed 0:32, clipped to the unit dimension). Rank zero
  gives a diagonal factor. Their reparameterized ELBO gradients use Adam with
  posterior-scale preconditioning and bounded relative steps. Polishing
  averages their log scales and loadings; dense units average precisions.
  `diagnostics.unit_families` reports each unit's family, dimension and rank.
  A streaming update preserves existing units' families and ranks; these
  controls apply to new units. The factor restriction changes the
  approximation, not the model density.
- `separator` — defaults to `:auto` in explicitly connected mode and `:none`
  otherwise. Automatic promotion uses bounded structural trials if a unit
  exceeds `max_unit_coordinates=4096` or `max_unit_work=1_000_000`, with at
  most `max_promoted_coordinates=128` extra globals including block closure.
  `separator=:none` retains explicit `global_coordinates` promotion. Prior,
  transform and partition-axis blocks remain intact. Inspect `diagnostics.separator`
  for selected blocks, reasons and whether budgets were achieved; the heuristic
  cannot promise an optimal cut or satisfy every budget. Changing ownership
  changes the variational approximation, never the model density.
- `unit_sampling` — defaults to `:work` in explicitly connected mode and
  `:uniform` otherwise. `:uniform` selects distinct units with weight `N/B`;
  `:work` makes `B` draws with replacement, with probabilities
  `pᵢ = (1/N + wᵢ/sum(w))/2`. The structural work proxy `wᵢ` counts local
  coordinates and structural reads in the unit's likelihood/prior factors;
  it is not measured runtime. All-zero work gives uniform probabilities.
  Repeated selections share one antithetic evaluation and one local update,
  with multiplicity retained in the global density and gradient weight
  `countᵢ/(B*pᵢ)`. Every unit has positive probability. This changes the visit
  schedule and estimator variance; it does not guarantee faster convergence.
- Inference ownership follows structural likelihood and authored-prior
  supports, including dormant branches and zero coefficients. Ordinary
  sampled coordinates read by only one unit join that unit; shared and
  prior-only coordinates stay global. Joint prior and transform blocks stay
  together. Each authored prior and Jacobian is counted once, including
  gradients through global hyperparameters. Scientific NLME roles are unchanged.
- `global_coordinates` — packed coordinate indices or names to keep global.
  Promotion closes over joint blocks and dependent priors. It can include
  standardized subject effects. In connected mode the observation partition
  is rebuilt after promotion; the scientific model view stays unchanged.
  Diagnostics expose `globals`, `units`, `unit_levels` and `owner`
  (zero for globals or fixed coordinates, otherwise the unit index).
- `start` / `inits` — where the global and unit means start: BRM's prior-median
  starting point by default, a full RK packed point `start`, or values keyed
  by prior address as [`brm_start`](@ref) takes them. Declarations without a
  recorded median, including ordinary model-scope vectors, retain the packed
  unit point; entries not supplied in `inits` retain that default point.
- `progress` — a Treebars node or backend symbol: one `BRM SVI fit` node with
  the phases Lower to ReactiveKernels, Prepare NLME model, Minibatch target
  (the unit partition gradient and the start), Stochastic optimization (its
  message names the window, iteration and ELBO) and Draws from q. An interrupt raises `Treebars.ProgressInterrupted` at the
  next window.
- `threads` — how many tasks evaluate a batch's subjects and step their factors,
  capped at the batch size. Each further task prepares its own copy of the
  compact cell samplers before the first iteration, which costs time and retained
  memory. The default is `Threads.nthreads()`, or 1 on an Enzyme without the
  concurrent-GC repair, which refuses an explicit `threads` above one.
  The result does not depend on it: the draws come from the one `rng` in batch
  order and the subjects' results are summed in batch order, so a run and its
  checkpoints are the same to the last bit at any `threads`, and
  [`brm_svi_resume`](@ref) can continue at another. The fit holds BLAS at one
  thread while it optimizes and draws, and restores it afterwards. Not with
  `laplace=GaussianBlockLaplace()`, whose updates evaluate the whole window.
- Optimizer controls: `max_iterations`, `lr_global` (Adam on the globals),
  `unit_step` (the subjects' natural-gradient step; a step that would change a
  subject's precision more than 2.5-fold along some direction, or move its
  mean more than one sd of its factor, is shortened for that subject),
  `polish_lr`, `window`,
  `patience`, `min_windows`, `polish_windows`, `global_scale` and `unit_scale`
  (initial factor scales).
- Interruptible runs: `stop_after` ends this call after that many iterations
  (the run continues with [`brm_svi_resume`](@ref) on the same model and
  data; the stopping rule and step-size schedule are unaffected). An optional
  first argument `save` (a function, so a `do` block works) receives a
  [`BRMSVIOptimizerState`](@ref) after every iteration that is a multiple of
  `checkpoint_every` (default: once per window), when this call stops, and
  before an interrupt propagates. With `laplace=GaussianBlockLaplace()` the
  default is every 50 updates, and the interrupt is checked every 50 updates.

Returns `draws`, `ndraws` draws from the fitted approximation as RK packed
unconstrained points (coordinates × draws, one column per draw, the order of
`coordinate_names(rk.model.layout)`), their `coordinates`, and `diagnostics`:
the ELBO trace (`elbo`, `elbo_windows`, up to a constant), `converged`,
`finished` (converged or `max_iterations` reached; `false` after `stop_after`),
`iterations`, the variational parameters `q`, the packed positions of the
global coordinates (`globals`), `seconds` of optimization in this call,
`threads` (the tasks this call used), the prepared `model`, and the run's
`optimizer` state, sampling probabilities/work/visits/draw counts (`sampling`),
and actual `unit_gradient_calls`. The
([`brm_svi_optimizer_checkpoint`](@ref) copies it). The draws come from a copy
of the optimizer's random stream: they do not advance `rng`, and a continued
run is unaffected by them. The approximation is variational: posterior means
are close, and the spread of the population, scale and correlation coordinates
is too narrow.
"""
brm_svi_fit(x; kwargs...) = _brm_svi_initial_fit(nothing, x; kwargs...)
brm_svi_fit(save::Function, x; kwargs...) = _brm_svi_initial_fit(save, x; kwargs...)
_brm_svi_initial_fit(save, x; laplace=nothing, kwargs...) = laplace === nothing ?
    _brm_svi_fit(save, x, nothing; kwargs...) :
    _brm_svi_laplace_dispatch(save, x, nothing, laplace; kwargs...)

function _brm_svi_fit(save, x, previous; rng=Random.default_rng(), batch_size=32,
        max_iterations=100_000, family=nothing, unit_sampling=nothing,
        unit_dense_limit=64, unit_rank=8,
        lr_global=0.01, unit_step=0.1, start=nothing, inits=nothing,
        global_scale=0.1, unit_scale=1.0, polish_lr=0.05, window=nothing,
        patience=10, min_windows=40, polish_windows=20, kwargs...)
    family = _brm_svi_family(previous, family)
    _brm_svi_unit_options(unit_dense_limit, unit_rank)
    unit_sampling in (nothing, :uniform, :work) || throw(ArgumentError(
        "brm_svi_fit: unit_sampling must be :uniform or :work"))
    batch_size >= 1 || throw(ArgumentError("brm_svi_fit: batch_size must be positive"))
    max_iterations >= 0 || throw(ArgumentError("brm_svi_fit: max_iterations must be nonnegative"))
    window === nothing || window >= 1 || throw(ArgumentError("brm_svi_fit: window must be positive"))
    previous === nothing || (start === nothing && inits === nothing) || throw(ArgumentError(
        "brm_svi_fit: a streaming update starts from the previous factors; omit start and inits"))
    _brm_svi_drive(save, x, previous; caller="brm_svi_fit", description="BRM SVI fit",
            kwargs...) do t
        t, q = _brm_svi_initialize(t, previous, family; start, inits, global_scale, unit_scale,
            unit_dense_limit, unit_rank)
        c = _brm_svi_controls(length(t.units); batch_size, max_iterations, lr_global,
            unit_step, polish_lr, window, patience, min_windows, polish_windows,
            unit_sampling=something(unit_sampling,
                t.m.view.unit_partition === :connected ? :work : :uniform))
        t, _brm_svi_optimizer_state(t, previous, c, _brm_svi_run(rng, q, c, t))
    end
end

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: whether a
# reverse gradient of `ad_backend` (`nothing`: the default, reverse-mode Enzyme)
# stays sound while another thread allocates. Only Enzyme's have that defect.
_brm_concurrent_gradients(ad_backend) = true

# The tasks a call uses: `threads`, or by default every thread of this process.
# Without the repair, an Enzyme refuses several tasks a caller asked for and keeps
# the serial loop for a caller who asked for none.
function _brm_svi_threads(caller, threads, ad_backend;
        repaired=_brm_concurrent_gradients(ad_backend))
    threads === nothing && return repaired ? Threads.nthreads() : 1
    threads isa Integer && !(threads isa Bool) && threads >= 1 || throw(ArgumentError(
        "$caller: threads must be a positive integer, got $(repr(threads))"))
    threads == 1 || repaired || throw(ArgumentError(
        "$caller: threads = $threads takes reverse gradients on several threads, and " *
        "this Enzyme lacks the concurrent-GC repair, so a gradient can use freed " *
        "memory; use Enzyme's dogfood line (from 72d5d25) or pass threads = 1"))
    Int(threads)
end

# BLAS's thread count is process-global, so the pin is shared: concurrent or
# nested calls hold it together and the last to leave restores what the first
# found.
const _BRM_BLAS_PIN = (; lock=ReentrantLock(), depth=Ref(0), saved=Ref(1))

# `f()` with one BLAS thread. Tasks that each run LAPACK on a multi-threaded BLAS
# oversubscribe the host, and a BLAS thread count can change a factorization's
# last bits, so a fit pins it whatever its number of tasks.
function _brm_with_serial_blas(f)
    lock(_BRM_BLAS_PIN.lock) do
        if _BRM_BLAS_PIN.depth[] == 0
            _BRM_BLAS_PIN.saved[] = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
        end
        _BRM_BLAS_PIN.depth[] += 1
    end
    try
        f()
    finally
        lock(_BRM_BLAS_PIN.lock) do
            _BRM_BLAS_PIN.depth[] -= 1
            _BRM_BLAS_PIN.depth[] == 0 && BLAS.set_num_threads(_BRM_BLAS_PIN.saved[])
        end
    end
end

function _brm_svi_check_interruption(caller, save, stop_after, checkpoint_every)
    stop_after === nothing || stop_after isa Integer && stop_after >= 0 || throw(ArgumentError(
        "$caller: stop_after must be a nonnegative integer (iterations in this call) or nothing"))
    checkpoint_every === nothing || checkpoint_every isa Integer && checkpoint_every >= 1 ||
        throw(ArgumentError("$caller: checkpoint_every must be a positive integer or nothing"))
    save === nothing && checkpoint_every !== nothing && throw(ArgumentError(
        "$caller: checkpoint_every needs a checkpoint function as the first argument, " *
        "e.g. $caller(state -> save_checkpoint(path, state), ...)"))
    nothing
end

# A fit or resume of the minibatch engine through its progress phases. After
# preparing `x` and its target, `init(t)` returns the target (with a streaming
# prior, if any) and the `BRMSVIOptimizerState` to advance; it refuses an
# incompatible checkpoint before any optimization step. The result's draws
# come from a copy of the run's random stream, so drawing them never moves the
# stream a later continuation reads.
function _brm_svi_drive(init, save, x, previous; caller, description, ndraws=1000,
        stop_after=nothing, checkpoint_every=nothing, progress=nothing,
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=nothing, threads=nothing,
        global_coordinates=nothing, separator=nothing, max_unit_coordinates=4096,
        max_unit_work=1_000_000, max_promoted_coordinates=128, check_lowered=Returns(nothing))
    ndraws isa Integer && ndraws >= 0 || throw(ArgumentError(
        "$caller: ndraws must be a nonnegative integer"))
    _brm_svi_check_interruption(caller, save, stop_after, checkpoint_every)
    threads = _brm_svi_threads(caller, threads, ad_backend)
    _brm_with_progress(progress; description) do node
        _brm_with_phases(node, _BRM_SVI_PHASES) do phases
            lowered = _brm_svi_lowered(x, phases.lower)
            check_lowered(lowered)
            m = _brm_svi_model(lowered, phases.prepare;
                ad_backend, shared_point, subject_group, nested_groups, shared_groups, unit_partition)
            t, state = _brm_run_phase(phases.target) do phase
                # SVI's partition sampler, on first use, reports under this phase.
                t, state = init(_brm_with_cell_progress(
                    () -> _brm_svi_target(m; global_coordinates, previous, separator,
                        max_unit_coordinates, max_unit_work, max_promoted_coordinates), phase))
                _brm_svi_with_workers(t, min(threads, state.controls.batch_size), phase), state
            end
            run, c = state.run, state.controls
            seconds, draws, q = _brm_with_serial_blas() do
                seconds = @elapsed _brm_enter_phase(phases.fit) do phase
                    _brm_svi_fit!(run, t, c; stop_after=something(stop_after, typemax(Int)),
                        checkpoint_every=something(checkpoint_every, c.window), save,
                        snapshot=_ -> _brm_svi_detach(state), node=phase)
                end
                q = _brm_svi_result_q(run)
                draws = _brm_run_phase(phases.draws) do _
                    _brm_svi_draws(copy(run.rng), t, q, ndraws)
                end
                seconds, draws, q
            end
            (; draws, coordinates=copy(m.view.coordinates), diagnostics=(;
                threads=length(t.plans),
                elbo=run.trace, elbo_windows=first.(run.windows), run.converged,
                finished=_brm_svi_finished(run, c), iterations=run.iteration, q,
                globals=t.globals, units=t.units, owner=t.ownership.owner,
                global_coordinates=t.ownership.promoted,
                unit_levels=_brm_svi_unit_levels(t),
                subject_unit=hasproperty(t.ownership, :subject_unit) ?
                    t.ownership.subject_unit : m.view.subject_unit,
                separator=hasproperty(t.ownership, :separator) ? t.ownership.separator : nothing,
                unit_families=_brm_svi_unit_description.(q.U),
                seconds, model=m, source_id=_brm_svi_source_id(m.backend),
                sampling=run.sampling, unit_gradient_calls=run.unit_gradient_calls,
                streaming=_brm_svi_carried_state(previous, t, q), optimizer=state))
        end
    end
end

_brm_svi_family(::Nothing, ::Nothing) = MeanFieldGlobal()
_brm_svi_family(::Nothing, family::BRMSVIFamily) = family
function _brm_svi_initialize(t, ::Nothing, family; start, inits, global_scale, unit_scale, unit_dense_limit=64, unit_rank=8)
    point = _brm_svi_start(t.m, start, inits)
    q = _brm_svi_params(t, point[t.globals], family; global_scale, unit_scale, unit_dense_limit, unit_rank)
    for (i, block) in enumerate(t.units)
        q.M[i] .= point[block]
    end
    t, q
end
_brm_svi_carried_state(::Nothing, m, q) = nothing
