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
# Lin, Schmidt and Khan (2020), whose curvature estimate comes from the
# reparameterized draws themselves (Stein's lemma) and whose precision
# update stays positive definite, inside a per-subject trust region on the
# precision's relative change and the mean's move (as Theis and Hoffman (2015)
# bound SVI's natural-gradient steps). By default a dense subject evaluates d
# mutually orthogonal antithetic pairs per visit, which estimate its whole
# curvature (exactly for a quadratic term), and takes the whole step; a single
# pair, BRM 0.2.242's step, sees one direction per visit, and units of tens of
# coordinates did not settle in the visits a minibatch run gives them
# (`unit_curvature`, below). Either way only the positive part of the estimate
# enters the step.
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
include("nlme_svi_sites.jl")
include("nlme_svi_centering.jl")

# ── The minibatch target ─────────────────────────────────────────────────────

# Packed positions of the global coordinates: everything the NLME model
# estimates as θ or σ, and the covariance coordinates the NLME reading leaves to
# its estimators. Fixed shared latents (`shared_point`) stay at `m.unit`.
_brm_svi_globals(m::BRMNLMEModel) = sort!(vcat(m.theta, m.sigma,
    findall(r -> r === :subject_scale || r === :subject_correlation, m.view.role)))

# One independent Gaussian per mesh state drives a learned field's lengths to their
# limit, because a shorter field makes that factorization more nearly exact (on toys
# with closed-form optima, a spatial length of 0.017 where the truth was 0.5). A read-pair
# field's states sit where the data is, so the same factor stays close to the posterior.
function _brm_svi_check_fields(m::BRMNLMEModel)
    for record in _brm_nlme_records(m.backend.plan)
        record.kind === :gmrf || continue
        _brm_nlme_error("brm_svi_fit: `$(record.predictor)` is a mesh `gmrf` field. Its " *
            "independent factors per mesh state drive the learned lengths to collapse, so " *
            "this fit would mislead; use `gmrf(...; neighbors=m)`, whose states are its reads")
    end
    nothing
end

function _brm_svi_target(m::BRMNLMEModel; global_coordinates=nothing, previous=nothing,
        separator=nothing, max_unit_coordinates=4096, max_unit_work=1_000_000,
        max_promoted_coordinates=128, cell_units=nothing, field_states=:global)
    field_states in (:global, :units) || throw(ArgumentError(
        "brm_svi_fit: field_states must be :global or :units, got $(repr(field_states))"))
    field_states === :units && previous !== nothing && throw(ArgumentError(
        "brm_svi_fit: field_states=:units has no streaming form: a new window adds states the " *
        "carried factors cannot name. Fit the window with field_states=:global, or refit"))
    m.cell_query === nothing && _brm_nlme_error(
        "brm_svi: minibatch SVI evaluates one subject at a time, on the " *
        "subject's own observation cells, and this model's observations have " *
        "no cells attributable to single subjects")
    _brm_svi_check_fields(m)
    # A carried checkpoint brings its own separator policy, which a changed
    # partition cannot use: name that change, not the policy it invalidates.
    previous === nothing || _brm_svi_check_partition(m.view, previous)
    connected = m.view.unit_partition === :connected
    policy = something(separator, connected ? :auto : :none)
    !connected && policy !== :none && throw(ArgumentError(
        "brm_svi_fit: automatic separators require unit_partition=:connected"))
    cell_units = _brm_svi_cell_units_option(cell_units, connected)
    inventory =field_states === :units ? _brm_svi_field_inventory(m) : ()
    ownership = connected ? _brm_svi_connected_ownership(m; global_coordinates, previous,
        separator=policy, max_unit_coordinates, max_unit_work, max_promoted_coordinates,
        cell_units, fields=inventory) :
        _brm_svi_ownership(m; global_coordinates, previous, fields=inventory)
    globals, units = ownership.globals, ownership.units
    # One strict partition sampler per worker; sparse selection metadata shares
    # globals and retains only each unit's local permutation and merge slots.
    partition, partition_view = connected ? _brm_svi_partition(m, ownership) :
        (m.cell_gradient, m.view)
    plan = _brm_nlme_partition_plan(partition, m.unit, globals, partition_view; units)
    places = _brm_svi_footprint_places(plan.footprints, globals, units, length(m.unit))
    fields = isempty(inventory) ? nothing : _brm_svi_unit_fields(m, ownership, inventory, globals, units)
    work = [_brm_svi_worker_buffers(globals, fields)]
    nz = count(==(:subject_effect), m.view.role)
    # Connected targets use selected authored buckets even for standardized
    # draws: the legacy global-prior path traverses every local prior at zero.
    prior = !connected && sum(length, units; init=0) == nz &&
        all(j -> m.view.role[j] === :subject_effect, Iterators.flatten(units)) ? nothing :
        _brm_svi_authored_prior(m, globals, units; factors=ownership.factors)
    # Worker plans build their samplers from the partition's bound context, which is all
    # the target keeps of it: a connected partition's value query, its footprints (the
    # plan holds its own) and its evaluator pool are per unit and served the plan.
    t = (; m, globals, units, ownership, previous, partition=(; partition.context), plan,
       plans=[plan], work, places,
       touched=_brm_svi_field_touched!(_brm_svi_touched_globals(places, prior), fields), prior,
       fields, z_constant=-nz * log(2π) / 2, prior_point=copy(m.unit))
    # From the model's own prior, before a streaming fit puts its carried factor
    # there: what the sd bound, a stopgap, multiplies (src/nlme_svi_sd_bound.jl).
    merge(t, (; sd_prior=_brm_svi_sd_prior(t)))
end

# `t` with `n` partition plans, one per task that evaluates units. `t.plan` is
# the first; each further one owns its sampler, gradients and work point
# (`_brm_nlme_worker_plan`). `workers` holds plans an earlier call on this target
# prepared; the ones this call still needs are added to it.
function _brm_svi_with_workers(t, n, progress=nothing; workers=typeof(t.plan)[])
    n == 1 && return t
    missing_plans = n - 1 - length(workers)
    missing_plans > 0 && append!(workers, _brm_with_counter(progress, missing_plans;
            description="Worker partition samplers") do node
        map(1:missing_plans) do k
            plan = _brm_nlme_worker_plan(t.plan, t.partition, t.m.unit)
            _brm_update_progress!(node, k)
            plan
        end
    end)
    # Each task's global gradient and field buffers; the first is the target's own.
    work = vcat([first(t.work)], [_brm_svi_worker_buffers(t.globals, t.fields) for _ in 1:n-1])
    merge(t, (; plans=vcat([t.plan], workers[1:n-1]), work))
end

# ── A model's retained target ───────────────────────────────────────────────

# The target and worker plans for one call on `m`: the ones the model holds when
# they were prepared for the same ownership, separator policy and streaming carry,
# otherwise new ones. Nothing in them depends on the optimizer's state: `t` holds the model's
# read-only tables and work arrays each evaluation overwrites, and a run's factors,
# Adam moments and random stream live in `_BRMSVIRun`. So a continuation on them
# is the same run as one that never returned.
#
# The call takes them out of the slot, because its samplers and work arrays belong
# to one call at a time: a fit running at the same time on the same model finds
# the slot empty and prepares its own. `_brm_svi_checkin!` puts them back, so the
# slot holds at most one target, the last call's.
#
# A connected target also depends on its separator policy, which promotes more
# coordinates than were requested: its key is the request and the policy, which
# is what a resume passes (`_brm_svi_identity`). A strict target's promoted set
# is a closure of its request, so a request equal to it finds the same target.
# `field_states` is part of the key too: under `:units` a read-pair field's states
# sit in units' blocks and their prior is split by owner, so the two targets
# differ in their units and prior.
function _brm_svi_checkout!(m; global_coordinates, previous, separator=nothing,
        max_unit_coordinates=4096, max_unit_work=1_000_000, max_promoted_coordinates=128,
        cell_units=nothing, field_states=:global)
    requested = _brm_svi_global_coordinates(m, global_coordinates)
    carry = previous === nothing ? nothing : _brm_value_digest(previous)
    connected = m.view.unit_partition === :connected
    cell_units = _brm_svi_cell_units_option(cell_units, connected)
    policy = connected ? (; separator=something(separator, :auto), max_unit_coordinates,
        max_unit_work, max_promoted_coordinates, cell_units) :
        (; separator=something(separator, :none))
    slot = m.svi_target
    prepared = lock(slot.lock) do
        held = slot.prepared
        held !== nothing && isequal(held.carry, carry) && isequal(held.policy, policy) &&
            held.field_states === field_states && (requested == held.requested || !connected &&
                requested == held.target.ownership.promoted) || return nothing
        slot.prepared = nothing
        held
    end
    prepared !== nothing && return prepared, true
    t = _brm_svi_target(m; global_coordinates, previous, separator, max_unit_coordinates,
        max_unit_work, max_promoted_coordinates, cell_units, field_states)
    (; requested, carry, policy, field_states, target=t, workers=typeof(t.plan)[]), false
end

function _brm_svi_checkin!(m, prepared)
    slot = m.svi_target
    lock(() -> (slot.prepared = prepared), slot.lock)
    nothing
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
# With unit-owned field states, the conditionals of states no single unit owns.
function _brm_svi_global_term(t::NamedTuple, g)
    value, gradient = _brm_svi_global_term(t.prior, t, g)
    get(t, :fields, nothing) === nothing && return value, gradient
    value + _brm_svi_field_term!(gradient, Float64[], t, 0, Float64[], first(t.work)), gradient
end
function _brm_svi_global_term(::Nothing, t, g)
    u = t.prior_point
    u[t.globals] .= g
    value, gradient = _rk_nlme_population_prior_value_gradient(t.m.population_prior, u)
    value - t.z_constant, gradient[t.globals]
end

# Unit i's term with its gradients in `gg` (the globals) and `gz` (its block), both
# overwritten: the whole of `gg`, so this costs the globals' length. An iteration
# evaluates through `_brm_svi_unit_local!`, which does not.
function _brm_svi_unit_term!(gg, gz, t, i, g, z, plan=t.plan, work=first(t.work))
    fill!(gg, 0.0)
    fill!(gz, 0.0)
    _brm_svi_unit_local!(gg, gz, t, i, g, z, plan, work)
end
# An analytic test target has only the term above.
_brm_svi_unit_local!(gg, gz, t, i, g, z, plan, work) =
    _brm_svi_unit_term!(gg, gz, t, i, g, z, plan, work)

# The same, costing the unit's footprint and prior support, not the globals. It adds
# into `gg` and `gz`, reads and writes only `t.places[i]` and `t.touched[i]`, and
# restores the plan's point. The partition reads nothing else (`locality=:error`).
function _brm_svi_unit_local!(gg, gz, t::NamedTuple, i, g, z, plan, work)
    value = _brm_svi_unit_partition!(gg, gz, t, i, g, z, plan)
    value + _brm_svi_unit_prior!(gg, gz, t.prior, t, i, g, z) +
        _brm_svi_field_term!(gg, gz, t, i, z, work)
end

# The unit's observation cells alone, added into `gg` and `gz` as above.
_brm_svi_unit_partition!(gg, gz, t, i, g, z, plan) = i in plan.labels ?
    _brm_svi_partition_local!(gg, gz, plan, t.m.unit, plan.footprints[i], t.places[i], i, g, z) : 0.0

function _brm_svi_partition_local!(gg, gz, plan, unit, footprint, places, i, g, z)
    point, gradient = plan.point, plan.gradients[i]
    @inbounds for (k, j) in enumerate(footprint)
        p = places[k]
        p > 0 ? (point[j] = g[p]) : p < 0 && (point[j] = z[-p])
    end
    try
        value = first(_rk_nlme_partition_value_and_gradient!(plan.sampler, gradient, point, i))
        @inbounds for (k, p) in enumerate(places)
            p > 0 ? (gg[p] += gradient[k]) : p < 0 && (gz[-p] += gradient[k])
        end
        value
    finally
        @inbounds for (k, j) in enumerate(footprint)
            places[k] == 0 || (point[j] = unit[j])
        end
    end
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
    clipped::Int
end
_BRMSVIAdam(n, lr) = _BRMSVIAdam(lr, 0.9, 0.999, zeros(n), zeros(n), 0, 0)

# How far a global's gradient may enter Adam's moments, in running rms of that
# coordinate, for a batch that is the whole data; `_brm_svi_iteration!` widens it
# by the square root of the scale-up N / B. A draw that reaches a steep part of
# the density can give a gradient of 1e20 where the running rms is 1. The step
# itself is bounded, but β2 = 0.999 keeps g² in `v` for ln(g²/1000)/0.001
# iterations: a gradient 1e6 times the rms slows its coordinate for about 21,000
# iterations, 1e20 for about 85,000, and a draw reaches every coordinate its term
# reads. Measured on a run of 7,379 globals, 95% of them ended polishing with a
# step more than ten times too small. Nothing in the trace shows it: the ELBO
# then looks flat, as for a converged fit.
#
# A coordinate read by only a few units has a zero gradient in most batches, so
# its gradient when one is drawn is about √(N/B) times its rms. A tighter bound
# would shrink the rms and with it the next gradient's limit, without end. One
# whose readers are drawn less than once per batch on average takes them from
# their sites instead (src/nlme_svi_sites.jl), so the bound only meets its prior.
#
# A clipped gradient is not unbiased: a term with a heavy-tailed gradient pushes
# less than its expectation says. Where such a term's expectation is infinite
# (`_brm_svi_window`), no estimate of it exists to be biased.
const _BRM_SVI_GLOBAL_CLIP = 10.0

# `g` with each coordinate limited to `clip` times its running rms before this
# step. A coordinate with no gradient so far has no scale to compare with.
function _brm_svi_clip(opt::_BRMSVIAdam, g, clip)
    isfinite(clip) && opt.t > 0 || return g
    b2 = 1 - opt.β2^opt.t
    bounded = map(g, opt.v) do gi, vi
        iszero(vi) && return gi
        limit = clip * sqrt(vi / b2)
        clamp(gi, -limit, limit)
    end
    bounded == g || (opt.clipped += 1)
    bounded
end

# One ascent step of `x` along `g`, limited by `clip` (`_brm_svi_clip`).
function _brm_svi_step!(opt::_BRMSVIAdam, x, g, lr=opt.lr; clip=Inf)
    g = _brm_svi_clip(opt, g, clip)
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

# A scale-preconditioned ("whitened") step: μ moves in units of a square root of
# the factor's covariance and each Β row in units of its scale, so Adam's step
# size means a fraction of the current posterior scale, which stays meaningful
# when a global's posterior sd shrinks like 1/√N. It polishes a mean-field global
# factor (`_brm_svi_polish_global_step!`) and steps a structured unit factor
# (`_brm_svi_lowrank_step!`).
function _brm_svi_whitened_global_step!(opt, q, x, grad, lr; clip=Inf, rank=1.0)
    D = length(q.μ)
    C = _brm_svi_global_sqrt(_brm_svi_global_geometry(q))
    scale = vcat(ones(D), rank .* vec(repeat(exp.(q.ρ), 1, size(q.Β, 2))))
    grad = copy(grad)
    grad[1:D] .= _brm_svi_global_sqrt_tmul(C, view(grad, 1:D))
    rest = D+1:length(x)
    grad[rest] .*= scale
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad, lr; clip)
    x[1:D] .+= _brm_svi_global_sqrt_mul(C, view(step, 1:D))
    x[rest] .+= scale .* view(step, rest)
    x
end

# The exploration step: Adam on [μ; ρ; Β] at a constant step size, with each Β
# row in units of its coordinate's scale exp(ρ_k), as polishing steps it. μ and
# ρ step as they always have, so a `MeanFieldGlobal` run is unchanged to the
# last bit.
#
# Β's gradient w ηᵀ carries the minibatch noise of w, all of it independent of
# Β, so in most coordinates Adam's normalized step is a random walk of about a
# step size per iteration. A step of `lr_global` in absolute units, the same for
# every coordinate, moved Β's entries by about 0.2 in 400 iterations and 0.45 in
# 15,000, with rows several times the scale of a coordinate whose sd is 0.03:
# marginal variances 10 to 25 times the factor's own and correlations the target
# has no use for. On a model whose log-scale hyperparameters overflow a few sds
# out, the trimmed ELBO then fell after some 8,000 iterations and typical draws
# reached a non-finite density (4 of 4 fits, at 12,000 to 21,000 iterations; ρ
# moves on the log scale, so a mean-field factor never walks off that way). In
# units of exp(ρ_k) the walk is a fraction of each coordinate's own scale, and Β
# grows only where the gradient keeps pushing it.
function _brm_svi_exploration_global_step!(opt, q, x, grad, lr; clip=Inf)
    r = size(q.Β, 2)
    r == 0 && return _brm_svi_step!(opt, x, grad, lr; clip)
    D = length(q.μ)
    scale = repeat(exp.(q.ρ), r)
    rank = 2D+1:length(x)
    grad = copy(grad)
    grad[rank] .*= scale
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad, lr; clip)
    x[1:2D] .+= view(step, 1:2D)
    x[rank] .+= scale .* view(step, rank)
    x
end

# The settling and polishing step of the global factor. With no rank part it is
# `_brm_svi_whitened_global_step!`, unchanged. With one, μ moves in units of its
# diagonal scale exp(ρ_k), as Β's rows do, instead of a square root of the whole
# covariance.
#
#
function _brm_svi_polish_global_step!(opt, q, x, grad, lr; clip=Inf, rank=1.0)
    r = size(q.Β, 2)
    r == 0 && return _brm_svi_whitened_global_step!(opt, q, x, grad, lr; clip)
    D = length(q.μ)
    s = exp.(q.ρ)
    scale = vcat(s, ones(D), rank .* repeat(s, r))
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad .* scale, lr; clip)
    x .+= scale .* step
    x
end

# How far a low-rank factor's Β rows step in settling and polishing relative to
# μ, so that they never step further than at exploration's `lr_global`.
_brm_svi_rank_step_ratio(c) = min(1.0, c.lr_global / c.polish_lr)

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
# about half of v. In units above about 20 coordinates, measured on an exact
# Gaussian target, v's sd then stays wide or grows, and with it the draws,
# however strongly the target curves along v; at 13 and 15 coordinates it
# recovered. At a matched factor the estimate is εεᵀ, whose negative part
# is zero, so the positive part changes nothing there. It is biased towards
# more precision wherever the two differ, including a target with negative
# curvature, whose true fixed point it no longer reaches.
_brm_svi_unit_step!(q, i, gz, gd, εi, β; rank=1.0) =
    _brm_svi_unit_step!(q.Λ[i], q, i, gz, gd, εi, β; rank)
function _brm_svi_unit_step!(::AbstractMatrix, q, i, gz, gd, εi, β; rank=1.0)
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

# ── A dense unit's curvature on an orthogonal design (`unit_curvature`) ──────
#
# The single antithetic pair above estimates the curvature along one drawn
# direction of the factor per visit, so a dense unit of d coordinates corrects
# its precision about one direction per visit, at `unit_step` 0.1 a tenth of the
# way, and a minibatch run gives each unit tens to a few hundred visits. Units of
# 64-129 coordinates then stop unsettled, and the hyperparameters follow them:
# on an OU chain cut into window units the fitted length scale came out 1.4 and
# 11.4 against about 61 (centered and non-centered states), 113-411 nats below
# the family's optimum, while the same fits with the unit's exact curvature
# reached it (snag brm-svi-fit-hype-47cb328e).
#
# `unit_curvature = :orthogonal` (the default) evaluates the same estimator on d
# antithetic pairs whose whitened draws are the columns of E = √d Q, Q a uniformly
# random rotation: a randomly rotated third-degree spherical-radial cubature rule
# (Genz & Monahan 1998, "Stochastic integration rules for infinite regions";
# Arasaratnam & Haykin 2009, "Cubature Kalman filters"). The pairs' mean gradient estimates
# E_q ∇ log p, and (1/d) Σ_k sym(w_k ε_kᵀ), w_k = -U⁻¹ gd_k, estimates the
# whitened E_q[-∇² log p] by Stein's lemma: both exactly when the unit's term is
# quadratic in its coordinates (EEᵀ = d I), and unbiased over Q for terms of
# degree three. Every draw has radius √d, the typical radius of a pair's N(0, I)
# draw, so the design pushes no coordinate further out than the pair does. It
# costs 2d gradient evaluations of the unit per visit instead of 2.
# `unit_curvature = :pair` is the single pair, BRM 0.2.242's step.

# Whether unit i has a dense factor with coordinates (the step above), as
# opposed to a structured one (above `unit_dense_limit`) or none.
_brm_svi_dense_unit(q, i) = !(q.U[i] isa _BRMSVIUnitRoot) && !isempty(q.M[i])

# The whitened draws of unit i for one iteration: a vector for a pair, the
# d × d design E = √d Q for an orthogonal design. Drawn in batch order from the
# run's `rng`, before any unit is evaluated.
_brm_svi_unit_draws(rng, q, i, curvature) =
    curvature === :orthogonal && _brm_svi_dense_unit(q, i) ?
        _brm_svi_unit_design(rng, length(q.M[i])) : _brm_svi_unit_noise(rng, q.U[i])

function _brm_svi_unit_design(rng, d)
    F = qr(randn(rng, d, d))
    # Q from Householder QR is Haar-distributed once each column takes the sign
    # of R's diagonal (Mezzadri 2007, "How to generate random matrices from the
    # classical compact groups").
    Matrix(F.Q) .* (sqrt(d) .* [r < 0 ? -1.0 : 1.0 for r in diag(F.R)]')
end

# Unit gradient evaluations per visit of unit i under the run's controls.
_brm_svi_unit_calls(q, i, c) =
    _brm_svi_unit_curvature_mode(c) === :orthogonal && _brm_svi_dense_unit(q, i) ?
        2length(q.M[i]) : 2
# Controls written before `unit_curvature` existed (a checkpoint of BRM 0.2.242
# or earlier) keep the single pair.
_brm_svi_unit_curvature_mode(c) = get(c, :unit_curvature, :pair)
# Controls written before the orthogonal step took its estimate's positive part
# keep the whole estimate, so such a checkpoint resumes the run it was.
_brm_svi_unit_positive_part(c) = get(c, :unit_positive_part, false)

# The natural-gradient step of `_brm_svi_unit_step!` with a full curvature
# estimate `S` (whitened, symmetric): G = U (S₊ - I) Uᵀ for S₊ the estimate's
# positive part, with the same trust regions and square-root update.
#
# On a term quadratic in the unit's coordinates S is the exact curvature and S₊ is
# S. On a saturating or redescending term, a pair's gradient difference is not
# proportional to its draw, and while a stiff direction is far from resolved S is
# mostly noise whose eigenvalues come in nearly symmetric pairs ±λ with λ ≫ 1. The
# trust region then sets β ≈ 1/λ from their size, so the whole estimate halves the
# precision along each visit's negative eigenvector while it leaves that direction's
# true curvature unread: the same trap as the single pair's negative eigenvalue
# above, and it compounds over a different direction each visit. Measured at a
# saturating toy's exact unit precision of 1, the factor fell to 3e-4 and stayed
# there (test/nlme_svi_unit_curvature.jl); on a stiff unit of 17 coordinates a
# direction fell to 0.013 by its 18th visit while S itself read 223 along it on
# average (1 = the factor already matches), and logp fell by up to 2,500 nats within
# 4 of the factor's sds along it. With S₊ the same β leaves those directions alone
# (a factor of about 1 - β).
#
# A direction whose curvature really lies below the factor's still widens: S₊ reads
# its estimate, or zero where that is negative, and the precision halves per visit
# at most, once no other direction shortens the step. `positive = false` is the
# whole estimate, which checkpoints without the control keep
# (`_brm_svi_unit_positive_part`).
function _brm_svi_unit_step_full!(q, i, gz, S, β; positive=true)
    isempty(q.M[i]) && return q
    U = LowerTriangular(q.U[i])
    all(isfinite, S) || throw(ArgumentError(
        "brm_svi: non-finite gradient difference in a unit's curvature estimate"))
    Ŝ = Symmetric((S .+ S') ./ 2)
    Ĝ = Symmetric((positive ? _brm_svi_positive_part(Ŝ) : Ŝ) - I)
    r = maximum(abs, eigvals(Ĝ))
    β = r > 0 ? min(β, _BRM_SVI_UNIT_TRUST / r) : β
    # U⁻¹(Λ + βG) = (I + βĜ) Uᵀ.
    R = qr(vcat(Matrix(U'), (I + β .* Ĝ) * Matrix(U'))).R
    L = LowerTriangular(R' * Diagonal(sign.(diag(R))) ./ sqrt(2))
    Δ = β .* (L' \ (L \ gz))
    Δ .*= min(1.0, _BRM_SVI_UNIT_MEAN_TRUST / norm(U' * Δ))
    q.U[i] .= L
    q.Λ[i] .= L * L'
    q.M[i] .+= Δ
    q
end

# A symmetric matrix's positive part; itself, to the last bit, when it has no
# negative eigenvalue.
function _brm_svi_positive_part(S::Symmetric)
    F = eigen(S)
    first(F.values) >= 0 ? S : Symmetric(F.vectors * Diagonal(max.(F.values, 0.0)) * F.vectors')
end

# Unit i's terms on the orthogonal design `E` (columns ε_k): the value and the
# gradients averaged over the 2d draws, `gg` in the pair's convention (the sum
# over a pair, so half of it is the mean gradient), and the whitened curvature
# estimate. `problems` holds the first pair that was not finite, if any; the
# worker's global buffers are zero again at `touched` on return. Under learned
# centering every draw is the centered term (src/nlme_svi_centering.jl), and
# `centered` holds the controls' gradient averaged over the draws and every
# draw's adjoints, each weighted 1/d where a pair's are weighted 1
# (`_brm_svi_centering_batch_gradient`).
function _brm_svi_unit_design_terms(t, q, i, g, E, plan, work, touched, D;
        centering=nothing, frame=nothing)
    d = size(E, 1)
    U, m = LowerTriangular(q.U[i]), q.M[i]
    gg₊, gg₋ = _brm_svi_global_buffers(work, D)
    gz₊, gz₋ = zeros(d), zeros(d)
    value, gz, gg, W = 0.0, zeros(d), zeros(length(touched)), zeros(d, d)
    gc, adjoints = nothing, Any[]
    problem = nothing
    for k in 1:d
        δ = U' \ view(E, :, k)
        z₊, z₋ = m .+ δ, m .- δ
        fill!(gz₊, 0.0)
        fill!(gz₋, 0.0)
        if centering === nothing
            v₊ = _brm_svi_unit_local!(gg₊, gz₊, t, i, g, z₊, plan, work)
            v₋ = _brm_svi_unit_local!(gg₋, gz₋, t, i, g, z₋, plan, work)
        else
            v₊, gc₊, a₊ = _brm_svi_centered_unit_term!(gg₊, gz₊, centering, frame, t, q, i,
                g, z₊, plan, work)
            v₋, gc₋, a₋ = _brm_svi_centered_unit_term!(gg₋, gz₋, centering, frame, t, q, i,
                g, z₋, plan, work)
            gc = something(gc, zero(gc₊))
            gc .+= (gc₊ .+ gc₋) ./ 2d
            push!(adjoints, a₊, a₋)
        end
        problem = _brm_svi_nonfinite_unit(t, i, g, (z₊, z₋), (v₊, v₋),
            (gg₊, gg₋), (gz₊, gz₋), touched; packed=_brm_svi_packed_draw(centering, frame, q, i, g))
        for (entry, p) in enumerate(touched)
            gg[entry] += (gg₊[p] + gg₋[p]) / d
            gg₊[p] = gg₋[p] = 0.0
        end
        problem === nothing || break
        value += (v₊ + v₋) / 2d
        gz .+= (gz₊ .+ gz₋) ./ 2d
        W[:, k] .= -(U \ ((gz₊ .- gz₋) ./ 2))
    end
    S = (W * E' .+ E * W') ./ 2d
    centered = centering === nothing ? nothing :
        (; gc=something(gc, Float64[]), adjoints=Tuple(adjoints), weight=1 / d)
    (; value, gg, gz, S, problem, centered)
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
- `draw_finite`: whether the point the term was evaluated at was finite;
- `points`: the packed point(s) the term was evaluated at, in the order of
  `coordinate_names(rk.model.layout)`, as the columns of a fit's `draws`: `plus`
  and `minus` for a unit (under `unit_curvature = :orthogonal`, the pair that
  failed), `draw` for the global term. A coordinate the term does not read holds
  the value the fit holds for it there: the model's start, or a field state's
  draw of this iteration. Evaluating the model's density and gradient at a point
  replays the term's evaluation, so a consumer can look for the cause in its
  own model;
- `largest`: the term's coordinate of largest magnitude across `points`, among
  the globals and the unit's own, as `(; coordinate, value)` (`nothing` for a term
  with no coordinates).

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
        isempty(p.coordinates) || print(io, "; non-finite gradient in ",
            length(p.coordinates), " coordinate(s): ", join(p.coordinates, ", "))
        p.largest === nothing ||
            print(io, "; largest coordinate: ", p.largest.coordinate, " = ", p.largest.value)
    end
    print(io, "\nEach term's `points` holds the packed point(s) it was evaluated at, to ",
        "replay in the model. The run in memory is not a checkpoint after this error. ",
        "Continue from the last checkpoint `save` received; resuming it with unchanged ",
        "controls repeats the same random stream and reaches this evaluation again.")
end

# The packed point a term was evaluated at, packed as `_brm_svi_draws` packs a draw.
# What the term does not read is what the fit holds for it: the model's start or, with
# unit-owned field states, this iteration's reading of them.
function _brm_svi_term_point(t, g, block=Int[], z=Float64[])
    fields = hasproperty(t, :fields) ? t.fields : nothing
    u = copy(fields === nothing ? t.m.unit : fields.others)
    u[t.globals] .= g
    u[block] .= z
    u
end

# The largest coordinate in magnitude, among `positions`, of the term's `points`.
function _brm_svi_term_largest(t, points, positions)
    isempty(positions) && return nothing
    (_, entry), side = findmax(u -> findmax(p -> abs(u[p]), positions), points)
    (; coordinate=t.m.view.coordinates[positions[entry]], value=points[side][positions[entry]])
end

# A draw of unit i's factor → its packed coordinates, as `_brm_svi_draws` maps it.
_brm_svi_packed_draw(::Nothing, frame, q, i, g) = identity
_brm_svi_packed_draw(centering, frame, q, i, g) =
    z -> _brm_svi_centered_coordinates(centering, frame, q, i, g, z)

# Unit i's terms at the antithetic pair `zs`: `nothing` when every value and
# gradient is finite, else what was not. Gradient entries map to `t.globals`
# (`ggs`, read only at `touched`) and `t.units[i]` (`gzs`). `packed` maps a draw of
# the unit's factor to its packed coordinates (learned centering's factors are over
# other ones).
function _brm_svi_nonfinite_unit(t, i, g, zs, values, ggs, gzs, touched=eachindex(t.globals);
        packed=identity)
    finite(gg) = all(p -> isfinite(gg[p]), touched)
    all(isfinite, values) && all(finite, ggs) &&
        all(x -> all(isfinite, x), gzs) && return nothing
    quantities, bad = Symbol[], Int[]
    for (side, v, gg, gz) in zip((:plus, :minus), values, ggs, gzs)
        isfinite(v) || push!(quantities, Symbol(:value_, side))
        finite(gg) || (push!(quantities, Symbol(:global_gradient_, side));
            append!(bad, t.globals[filter(p -> !isfinite(gg[p]), collect(touched))]))
        all(isfinite, gz) || (push!(quantities, Symbol(:unit_gradient_, side));
            append!(bad, t.units[i][findall(!isfinite, gz)]))
    end
    points = NamedTuple{(:plus, :minus)}(map(z -> _brm_svi_term_point(t, g, t.units[i], packed(z)), zs))
    (; term=:unit, unit=i, quantities,
       coordinates=t.m.view.coordinates[sort!(unique!(bad))],
       draw_finite=all(isfinite, g) && all(z -> all(isfinite, z), zs), points,
       largest=_brm_svi_term_largest(t, points, vcat(t.globals, t.units[i])))
end

# Throws `BRMSVINonfiniteError` (iteration 0, which the caller fills in) when
# the global term `P`, `gP` at `g` or a unit's terms (`problems`) are not finite.
function _brm_svi_check_finite(t, g, P, gP, problems)
    found = NamedTuple[]
    if !(isfinite(P) && all(isfinite, gP))
        quantities = Symbol[]
        isfinite(P) || push!(quantities, :value)
        all(isfinite, gP) || push!(quantities, :global_gradient)
        points = (; draw=_brm_svi_term_point(t, g))
        push!(found, (; term=:global, unit=0, quantities,
            coordinates=t.m.view.coordinates[t.globals[findall(!isfinite, gP)]],
            draw_finite=all(isfinite, g), points,
            largest=_brm_svi_term_largest(t, points, t.globals)))
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
_brm_svi_field_prepare!(rng, t, q, g, batch) = nothing

function _brm_svi_iteration!(rng, t, q, global_opt, batch, lr_global, β;
        whiten=false, weights=fill(length(t.units) / length(batch), length(batch)),
        sites=nothing, site_scale=1.0, rank_step=1.0, curvature=:pair, site_step=β,
        structured_step=β, structured_rank=1.0, positive_part=true, sd_bound=nothing)
    D = length(t.globals)
    ε, η = randn(rng, D), randn(rng, size(q.Β, 2))
    g = _brm_svi_draw_global(q, ε, η)
    _brm_svi_field_prepare!(rng, t, q, g, batch)
    P, gP = _brm_svi_global_term(t, g)
    draws = [_brm_svi_unit_draws(rng, q, i, curvature) for i in batch]
    terms = Vector{@NamedTuple{value::Float64,gg::Vector{Float64},gz::Vector{Float64},
        gd::Vector{Float64}}}(undef, length(batch))
    # The whitened curvature estimate of a unit on an orthogonal design.
    curvatures = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(batch))
    problems = Vector{Union{Nothing,NamedTuple}}(nothing, length(batch))
    # Learned centering: the factors are over each unit's learned coordinates u,
    # and the unit terms read y = T(u) (src/nlme_svi_centering.jl).
    centering = _brm_svi_centering(t)
    frame = centering === nothing ? nothing : _brm_svi_centering_frame(centering, t, g)
    centered = Vector{Any}(nothing, length(batch))
    _brm_svi_each_unit(eachindex(t.plans), length(batch)) do worker, k
        plan, work = t.plans[worker], t.work[worker]
        i, εi = batch[k], draws[k]
        touched = _brm_svi_unit_touched(t, i)
        if εi isa AbstractMatrix
            r = _brm_svi_unit_design_terms(t, q, i, g, εi, plan, work, touched, D;
                centering, frame)
            problems[k] = r.problem
            terms[k] = (; r.value, r.gg, r.gz, gd=Float64[])
            curvatures[k] = r.S
            centered[k] = r.centered
            return
        end
        gg₊, gg₋ = _brm_svi_global_buffers(work, D)
        gz₊, gz₋ = zeros(length(t.units[i])), zeros(length(t.units[i]))
        m, δ = q.M[i], _brm_svi_unit_offset(q, i, εi)
        z₊, z₋ = m .+ δ, m .- δ
        if centering === nothing
            v₊ = _brm_svi_unit_local!(gg₊, gz₊, t, i, g, z₊, plan, work)
            v₋ = _brm_svi_unit_local!(gg₋, gz₋, t, i, g, z₋, plan, work)
        else
            v₊, gc₊, a₊ = _brm_svi_centered_unit_term!(gg₊, gz₊, centering, frame, t, q, i,
                g, z₊, plan, work)
            v₋, gc₋, a₋ = _brm_svi_centered_unit_term!(gg₋, gz₋, centering, frame, t, q, i,
                g, z₋, plan, work)
            centered[k] = (; gc=(gc₊ .+ gc₋) ./ 2, adjoints=(a₊, a₋))
        end
        problems[k] = _brm_svi_nonfinite_unit(t, i, g, (z₊, z₋), (v₊, v₋),
            (gg₊, gg₋), (gz₊, gz₋), touched; packed=_brm_svi_packed_draw(centering, frame, q, i, g))
        gg = [gg₊[p] + gg₋[p] for p in touched]
        # The buffers are the worker's, zero again for its next unit.
        for p in touched
            gg₊[p] = gg₋[p] = 0.0
        end
        terms[k] = (; value=(v₊ + v₋) / 2, gg, gz=(gz₊ .+ gz₋) ./ 2, gd=(gz₊ .- gz₋) ./ 2)
    end
    # Before anything changes q or the optimizer.
    _brm_svi_check_finite(t, g, P, gP, problems)

    # A sparse global takes its readers from their sites (src/nlme_svi_sites.jl),
    # not from N / B times this batch's.
    w = copy(gP)
    batch_value = 0.0
    for (k, term) in enumerate(terms)
        batch_value += weights[k] * term.value
        scale = weights[k] / 2
        for (entry, p) in enumerate(_brm_svi_unit_touched(t, batch[k]))
            _brm_svi_minibatch_global(sites, p) && (w[p] += scale * term.gg[entry])
        end
    end
    # Learned centering's chain terms through the cells' locations and scales: the
    # batch's one pullback steps the minibatch globals, and a sparse global's site
    # takes its reader's own (src/nlme_svi_centering.jl).
    if centering !== nothing
        chain = _brm_svi_centering_batch_gradient(centering, t, frame, centered, weights)
        for p in eachindex(chain)
            _brm_svi_minibatch_global(sites, p) && (w[p] += chain[p])
        end
        _brm_svi_centering_site_terms!(terms, sites, centering, t, frame, centered, batch,
            weights, chain)
    end
    _brm_svi_update_sites!(sites, t, q, g, ε, batch, terms, site_step, site_scale)
    x = _brm_svi_global_vector(q)
    grad = _brm_svi_add_site_gradient!(_brm_svi_global_gradient(q, w, ε, η), sites, q)
    # `N / B` for a uniform batch; under work sampling a unit drawn with chance π
    # carries weight about 1 / π, so its coordinates' gradients are about √w times
    # their rms when drawn, and the batch's largest weight bounds every unit in it.
    clip = _BRM_SVI_GLOBAL_CLIP * sqrt(maximum(weights))
    whiten ? _brm_svi_polish_global_step!(global_opt, q, x, grad, lr_global; clip, rank=rank_step) :
        _brm_svi_exploration_global_step!(global_opt, q, x, grad, lr_global; clip)
    _brm_svi_set_global!(q, x)
    _brm_svi_bound_sd!(q, _brm_svi_sd_prior_of(t), sd_bound)
    _brm_svi_each_unit(t.plans, length(batch)) do _, k
        S = curvatures[k]
        S === nothing ?
            _brm_svi_unit_step!(q, batch[k], terms[k].gz, terms[k].gd, draws[k],
                q.U[batch[k]] isa _BRMSVIUnitRoot ? structured_step : β;
                rank=structured_rank) :
            _brm_svi_unit_step_full!(q, batch[k], terms[k].gz, S, β; positive=positive_part)
    end
    # The controls move while exploring; settling and polishing step q in one fixed
    # frame, which polishing averages.
    centering === nothing || whiten || _brm_svi_control_steps!(q, batch, centered)
    P + batch_value
end

const _BRM_SVI_PHASES = (lower="Lower to ReactiveKernels", prepare="Prepare NLME model",
    target="Minibatch target", fit="Stochastic optimization", draws="Draws from q")

# Stochastic optimization in three phases. Exploration runs at constant step
# sizes (`lr_global` for Adam, `unit_step` for the subjects' natural-gradient
# steps) until the ELBO stops improving detectably: the trimmed mean over a
# window (one pass over the subjects, at least 100 iterations) has risen, over
# the last `patience` windows, by less than twice the standard error of that
# rise, after at least `min_windows` (`_brm_svi_window`). Comparing across several windows rather than
# neighbouring ones lets a slow drift show above the minibatch noise. Since
# BRM 0.2.264 the slope over the last `trend_windows` windows (default
# `4 patience + 1`) must also not be significantly positive (`_brm_svi_rising`):
# two windows alone miss a rise below about twice their noise. While those windows
# still reach back into the first `patience`, whose climb from the start is no line,
# the slope is also read without them, and either reading rising holds exploration.
#
#
# The phase is not stored: settling began at the first window at which
# exploration's test passed (`_brm_svi_explored`), which the run's windows
# record, so a checkpoint's layout is unchanged. A run whose controls predate
# `settle_windows` (a checkpoint of BRM 0.2.240 or 0.2.241) does not settle, and
# one whose controls predate `trend_windows` (BRM 0.2.263 and earlier) ends
# exploration on the two-window test alone, and one whose controls predate
# `trend_skip` reads the trend's line through the first windows alone, so its phases
# read as they were run.
#
const _BRMSVIWindow = NamedTuple{(:mean, :center, :spread),NTuple{3,Float64}}

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
    windows::Vector{_BRMSVIWindow}
    polish_start::Int
    average::Union{Nothing,Q}
    converged::Bool
    sampler::_BRMSVIUnitSampler
    sampling::NamedTuple
    unit_gradient_calls::Int
    averaged_through::Vector{Int}
    unit_entropy::Vector{Float64}
    entropy_sum::Float64
    sites::Union{Nothing,_BRMSVISites}
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
        _BRMSVIWindow[], 0, nothing, false,
        sampler, sampling, 0, zeros(Int, N), entropy, sum(entropy),
        _brm_svi_sites(t, c.batch_size, sampler.probabilities))
end

# The share of a window's iterations left out at each end of its statistics.
const _BRM_SVI_WINDOW_TRIM = 0.1

# What a window of ELBO estimates contributes to the stopping rule: `center`, its
# trimmed mean, and `spread`, so that `spread / W` estimates the variance of the
# center (the Winsorized variance over (1 - 2γ)², Yuen 1974). `mean` is the plain
# mean, which `diagnostics.elbo_windows` reports.
#
# A single draw can move a mean by any amount: the estimate is finite, but its
# expectation need not be. Under a Gaussian q over x and z, a density with a term
# like -exp(exp(x) z) has E_q log p = -Inf. A log link over a field or random
# effect whose sd sits on the log scale has such a term. So neither the mean nor
# the variance of a window estimates anything there: on one run a single draw
# took a window's mean to -1.25e19 while its median stayed at -12,076.
#
# Below 10 iterations per window nothing is trimmed, and this is the mean and
# variance.
function _brm_svi_window(w)
    n = length(w)
    k = floor(Int, _BRM_SVI_WINDOW_TRIM * n)
    x = sort(w)
    inner = view(x, k+1:n-k)
    winsorized = vcat(fill(first(inner), k), inner, fill(last(inner), k))
    (; mean=Statistics.mean(w), center=Statistics.mean(inner),
       spread=Statistics.var(winsorized) / (1 - 2k / n)^2)
end

# Whether the ELBO has failed to rise detectably from window `a` to window `b`.
_brm_svi_stalled(a, b, W) = b.center - a.center < 2 * sqrt((a.spread + b.spread) / W)

# Whether the ELBO still rises over the `L` windows ending at window `w` (all of them
# when there are fewer): the least-squares slope of their centres exceeds twice its
# standard error, read from the residuals about that line (each window one batch), so
# the error includes the iterate's wander between windows. Fewer than 3 windows (or
# `L < 3`, `0` being off) never rise.
#
# Exploration ends only where this is false too. Two windows `patience` apart cannot
# see a rise below about twice their noise, and a global that meets every unit on a
# ridge climbs that slowly: a non-centred step `m + s z_i` that each unit's data pin
# only in sum, so the spread `s` can fall only as fast as every unit's `z_i` grows.
# On a downstream model of that shape (8 and 16 units, batch 4) the two-window test
# passed at windows 42 and 41 while the 21-window slope read t = 7.2 and 6.8. The
# trimmed ELBO went on rising until about windows 157 and 208, by about 90 and 420
# nats, and only there did the step's mean settle where 400 windows put it (NUTS's
# median, on 8 units; snag brm-svi-fit-rkbr-f41cb255). Over 41 windows (`4 patience + 1`, the default
# `trend_windows`) the test stopped at those windows; over 21 it stopped at 95 and
# 70, still rising.
#
# With `first`, the line reads no window before it.
function _brm_svi_rising(windows, w, L, first=1)
    L = min(L, w - first + 1)
    L >= 3 || return false
    c = [windows[k].center for k in w-L+1:w]
    t = (1:L) .- (L + 1) / 2
    stt = sum(abs2, t)
    slope = sum(t .* c) / stt
    residual = c .- Statistics.mean(c) .- slope .* t
    se = sqrt(sum(abs2, residual) / (L - 2) / stt)
    slope > 2 * se
end

# Whether the ELBO's trend still rises at window `w`: the line over the last
# `trend_windows` windows rises, or the same line without the first `trend_skip`
# windows of the run (by default `patience`) does. The two differ only while the last
# `trend_windows` windows still reach back into those first ones.
#
# The first windows climb from the start, which a line does not describe: a window
# far below the rest is a residual so large that the error swamps any slope, so the
# line through it cannot read a rise, and exploration ended at the `min_windows` floor
# whatever the trend. On the downstream model above at 32 units the first window's
# trimmed ELBO sat 46,000 nats below the next 39, the slope over windows 1 to 40 read
# t = 1.9 while over 11 to 40 it read 7.2, and the fit stopped at window 40 with
# the step's mean on the ridge, exactly where the two-window test alone left it (snag
# brm-svi-fit-stop-4840e476). Leaving them out alone is not enough either: a start
# whose climb is smooth is evidence of the rise, and on such a start the line
# without it stopped at window 40 where the line with it held to 46. So exploration
# goes on while either line rises, and never ends earlier than the line through the
# first windows alone ended it.
function _brm_svi_trending(windows, w, c)
    L, skip = _brm_svi_trend_windows(c), _brm_svi_trend_skip(c)
    _brm_svi_rising(windows, w, L) || (skip > 0 && _brm_svi_rising(windows, w, L, skip + 1))
end

# Controls written before the trend test existed (a resumed checkpoint) keep the
# two-window test alone.
_brm_svi_trend_windows(c) = get(c, :trend_windows, 0)
# Controls written before `trend_skip` existed read the line through the first
# windows alone, as they ran.
_brm_svi_trend_skip(c) = get(c, :trend_skip, 0)

# Whether exploration ends at window `w`: the ELBO has not risen detectably over its
# last `patience` windows, and its trend is not upwards (`_brm_svi_trending`).
_brm_svi_exploration_stalled(windows, w, c) =
    _brm_svi_stalled(windows[w-c.patience], windows[w], c.window) &&
    !_brm_svi_trending(windows, w, c)

# The optimizer controls of a run over `N` subjects, with the batch size and
# window it actually uses. They are fixed for the run and its continuations.
function _brm_svi_controls(N; batch_size, max_iterations, lr_global, unit_step,
        polish_lr, window, patience, min_windows, polish_windows, unit_sampling=:uniform,
        settle_windows=_BRM_SVI_SETTLE_WINDOWS, unit_curvature=:pair,
        structured_unit_step=unit_step, structured_rank_step=1.0, unit_positive_part=true,
        sd_bound=_BRM_SVI_SD_BOUND, trend_windows=_brm_svi_default_trend_windows(patience))
    B = min(batch_size, N)
    (; batch_size=B, window=window === nothing ? max(cld(N, B), 100) : window,
       max_iterations, lr_global, unit_step, polish_lr, patience, min_windows,
       polish_windows, unit_sampling, settle_windows, unit_curvature, structured_unit_step,
       structured_rank_step, unit_positive_part, sd_bound, trend_windows,
       trend_skip=patience)
end

# The windows exploration's trend test reads when the caller names none
# (`_brm_svi_rising`): 41 at the default `patience` of 10.
_brm_svi_default_trend_windows(patience) = 4patience + 1

# A dense unit's step when the caller names none: the whole natural-gradient
# step on the orthogonal design's full curvature (the trust regions still bound
# it), a tenth of it on the single pair's rank-one estimate (BRM 0.2.242).
_brm_svi_default_unit_step(unit_curvature) = unit_curvature === :orthogonal ? 1.0 : 0.1

# A structured unit's step when the caller names none. `unit_curvature` does not
# reach these units: they have no natural-gradient step to take whole, only Adam
# on [μ; ρ; Β] in units of their scale (src/nlme_svi_units.jl), and this is Adam's
# step size. Above about 0.1 the unit leaves its target. On an exact Gaussian unit of
# 300 coordinates, 15 of them tight (`test/nlme_svi_units.jl`), Β's rows reached 27
# target sds within 100 iterations at step 1 and stayed there, then steps 1 and 0.3
# diverged as the gradient vanished near the optimum and Adam's normalized step did
# not (KL 1e5 and 1e10 at 5,000 iterations), where 0.1 reached 2e-8. On a real
# station's unit, step 1 reached draws 571 sds out within 569 iterations (snag
# svi-nonfinite-er-0923561d).
const _BRM_SVI_STRUCTURED_UNIT_STEP = 0.1

# How far a structured unit's Β rows step relative to its μ and ρ: at the default step,
# 0.01 of the coordinate's scale, the rate of the global factor's rows
# (`lr_global`, `_brm_svi_exploration_global_step!`). Β's gradient is the same noise
# here, and at the unit's whole step it walks: rows pass the marginal the target wants,
# ρ falls below them to make up for it, and the row's own step, in units of exp(ρ_k),
# falls with it, so it cannot come back. On three stations' units of 295-304
# coordinates under a known population law, a mean-field fit's factor read ELBOs of
# 481, 455 and 470; at the whole step rank 8 read 343, -204 and 425 with coordinates at
# marginal sd 8, 31 and 4 under an N(0, 1) prior, and at this ratio 486, 466 and 476
# with every sd near 1 (same snag).
const _BRM_SVI_STRUCTURED_RANK_STEP = 0.1

# The step of each family of units from `brm_svi_fit`'s `unit_step` and
# `unit_curvature`. A step the caller names is both families'.
_brm_svi_unit_steps(unit_step, unit_curvature) = (;
    unit_step=something(unit_step, _brm_svi_default_unit_step(unit_curvature)),
    structured_unit_step=something(unit_step, _BRM_SVI_STRUCTURED_UNIT_STEP),
    structured_rank_step=_BRM_SVI_STRUCTURED_RANK_STEP)

# Controls written before structured units had their own steps keep the ones they ran
# with, so a checkpoint's resume stays exact.
_brm_svi_structured_unit_step(c) = get(c, :structured_unit_step, c.unit_step)
_brm_svi_structured_rank_step(c) = get(c, :structured_rank_step, 1.0)

# At most this many windows of settling by default.
const _BRM_SVI_SETTLE_WINDOWS = 100

_brm_svi_finished(run, c) = run.converged || run.iteration >= c.max_iterations
_brm_svi_polishing(run) = run.polish_start > 0
# Controls written before settling existed (a resumed checkpoint) never settle.
_brm_svi_settle_windows(c) = get(c, :settle_windows, 0)

# The window at which exploration's test first passed, or 0 while exploring. A
# pure function of the run's windows, so it needs no field of its own.
function _brm_svi_explored(run, c)
    windows = run.windows
    for w in max(c.min_windows, c.patience + 1):length(windows)
        _brm_svi_exploration_stalled(windows, w, c) && return w
    end
    0
end
_brm_svi_settling(run, c) = !_brm_svi_polishing(run) && _brm_svi_explored(run, c) > 0

# Whether a settling run that has completed its windows up to `length(windows)`
# stops settling: the test on windows of settling alone, or the cap.
function _brm_svi_settled(windows, explored, c)
    n = length(windows) - explored
    n >= c.patience + 1 &&
        _brm_svi_stalled(windows[end-c.patience], windows[end], c.window)
end

# Where the run's phases began and whether settling ended on its test, from
# its windows: `exploring`, `settling` and `polishing` iterations, and
# `settled` (`false` when settling reached `settle_windows`, or has not ended).
function _brm_svi_stages(run, c)
    W, E = c.window, _brm_svi_explored(run, c)
    E == 0 && return (; exploring=run.iteration, settling=0, polishing=0,
        settled=false, settle_windows=_brm_svi_settle_windows(c))
    P = _brm_svi_polishing(run) ? run.polish_start ÷ W : 0
    settled = P == E || (P > E && _brm_svi_settled(view(run.windows, 1:P), E, c))
    (; exploring=E * W, settling=(P == 0 ? run.iteration : P * W) - E * W,
       polishing=P == 0 ? 0 : run.iteration - P * W, settled,
       settle_windows=_brm_svi_settle_windows(c))
end

# One iteration of the run, including the stopping rule's bookkeeping at the
# end of a window. Returns `:converged` when polishing completed with it,
# `:window` when it completed a window, `:iteration` otherwise.
function _brm_svi_advance!(run, t, c)
    N, W = length(t.units), c.window
    it = run.iteration + 1
    polishing = _brm_svi_polishing(run)
    settling = !polishing && _brm_svi_settling(run, c)
    scale = polishing ? 1 / sqrt(1 + (it - run.polish_start) / W) : 1.0
    q = run.q
    batch = _brm_svi_batch(run.rng, run.sampler, c.batch_size)
    polishing && _brm_svi_average_units!(run.average, q, run.averaged_through,
        batch.indices, it - run.polish_start - 1)
    for (k, i) in enumerate(batch.indices)
        run.sampling.visits[i] += 1
        run.sampling.draws[i] += batch.multiplicities[k]
    end
    run.unit_gradient_calls += sum(i -> _brm_svi_unit_calls(q, i, c), batch.indices; init=0)
    value = try
        _brm_svi_iteration!(run.rng, t, q, run.adam, batch.indices,
            scale * (polishing || settling ? c.polish_lr : c.lr_global), scale * c.unit_step;
            whiten=polishing || settling, weights=batch.weights, sites=run.sites,
            site_scale=scale, curvature=_brm_svi_unit_curvature_mode(c),
            positive_part=_brm_svi_unit_positive_part(c),
            site_step=scale * _brm_svi_site_step(c),
            structured_step=scale * _brm_svi_structured_unit_step(c),
            structured_rank=_brm_svi_structured_rank_step(c),
            rank_step=polishing || settling ? _brm_svi_rank_step_ratio(c) : 1.0,
            sd_bound=_brm_svi_sd_bound_factor(c))
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
    polished = false
    if polishing
        k = it - run.polish_start
        for name in (:μ, :ρ, :Β)
            _brm_svi_average!(getfield(run.average, name), getfield(q, name), k)
        end
        polished = k >= c.polish_windows * W
    end
    it % W == 0 && push!(run.windows, _brm_svi_window(view(run.trace, it-W+1:it)))
    polished && (run.converged = true; return :converged)
    it % W == 0 || return :iteration
    windows = run.windows
    polishing && return :window
    if settling
        explored = _brm_svi_explored(run, c)
        # Polishing continues settling's step and its Adam moments.
        (_brm_svi_settled(windows, explored, c) ||
            length(windows) - explored >= _brm_svi_settle_windows(c)) &&
            _brm_svi_start_polishing!(run, q, it)
    elseif length(windows) >= max(c.min_windows, c.patience + 1) &&
            _brm_svi_exploration_stalled(windows, length(windows), c)
        # Exploration ends: Adam restarts on the scale-preconditioned step.
        clipped = run.adam.clipped
        run.adam = _brm_svi_adam(q, c.polish_lr)
        run.adam.clipped = clipped
        _brm_svi_settle_windows(c) > 0 || _brm_svi_start_polishing!(run, q, it)
    end
    :window
end

function _brm_svi_start_polishing!(run, q, it)
    run.polish_start = it
    run.average = _brm_svi_average_start(q)
    run
end

_brm_svi_progress_message(run, c) = string(_brm_svi_phase_text(run, c),
    " · iteration $(run.iteration) · ELBO ≈ ", round(last(run.windows).center; sigdigits=6))

_brm_svi_phase_text(run, c) =
    _brm_svi_polishing(run) && run.polish_start < run.iteration ?
        "polishing $(run.iteration - run.polish_start)/$(c.polish_windows * c.window)" :
    _brm_svi_settling(run, c) ?
        "settling, window $(length(run.windows) - _brm_svi_explored(run, c))" :
        "exploring, window $(length(run.windows))"

_brm_svi_phase(run, c) = _brm_svi_polishing(run) ? :polishing :
    _brm_svi_settling(run, c) ? :settling : :exploring

# Advance `run` until it converges, reaches `max_iterations`, or has taken
# `stop_after` iterations in this call. `save(checkpoint)` receives a detached
# `BRMSVIOptimizerState` (`checkpoint(run)`) after every iteration that is a
# multiple of `checkpoint_every`, before an interrupt requested at the end of
# a window propagates, and when this call stops. `observe(status)` follows at the
# same three points, at multiples of `observe_every` (`_brm_svi_observer`).
# Neither advances the random stream or writes to the run.
function _brm_svi_fit!(run, t, c; stop_after=typemax(Int), checkpoint_every=c.window,
        save=nothing, checkpoint=identity, node=nothing, observe=nothing,
        observe_every=c.window)
    commit() = save === nothing || save(checkpoint(run))
    taken = 0
    while !_brm_svi_finished(run, c) && taken < stop_after
        status = _brm_svi_advance!(run, t, c)
        taken += 1
        if status === :window
            try
                _brm_throw_if_interrupted(node)
            catch
                commit()
                observe === nothing || observe(:interrupted)
                rethrow()
            end
            _brm_update_progress!(node, _brm_svi_progress_message(run, c))
        end
        (_brm_svi_finished(run, c) || taken >= stop_after) && continue
        save === nothing || run.iteration % checkpoint_every != 0 || commit()
        observe === nothing || run.iteration % observe_every != 0 || observe(:running)
    end
    commit()
    observe === nothing || observe(_brm_svi_end_status(run, c))
    run
end

_brm_svi_end_status(run, c) = !_brm_svi_finished(run, c) ? :stopped :
    run.converged ? :converged : :max_iterations

# ── Snapshots of a running fit ───────────────────────────────────────────────

const _BRM_SVI_SNAPSHOT_FORMAT = 1

# Before any preparation, so that a mistake here costs no model or target.
function _brm_svi_check_snapshot(caller, snapshot, every, draws)
    snapshot === nothing && (every !== nothing || draws != 0) && throw(ArgumentError(
        "$caller: snapshot_every and snapshot_draws need a `snapshot` function or path"))
    every === nothing || every isa Integer && every >= 1 || throw(ArgumentError(
        "$caller: snapshot_every must be a positive integer or nothing"))
    draws isa Integer && draws >= 0 || throw(ArgumentError(
        "$caller: snapshot_draws must be a nonnegative integer"))
    snapshot === nothing || snapshot isa AbstractString || !isempty(methods(snapshot)) ||
        throw(ArgumentError("$caller: snapshot must be a function or a file path, " *
            "got $(repr(snapshot))"))
    snapshot isa AbstractString && mkpath(dirname(abspath(snapshot)))
    nothing
end

# Called with a status at each boundary of the run in `state` on target `t`. The
# fields below are plain Julia data, so a saved snapshot loads without BRM or the
# model. Nothing here writes to what the run reads next, and draws come from a
# copy of the stream: an observed run is the run an unobserved one is.
_brm_svi_observer(::Nothing, t, state, draws) = nothing
function _brm_svi_observer(snapshot, t, state, draws)
    started = time()
    function (status)
        _brm_svi_deliver(snapshot, merge((; format=_BRM_SVI_SNAPSHOT_FORMAT, status,
                seconds=time() - started, threads=hasproperty(t, :plans) ? length(t.plans) : 1,
                model_identity=state.identity.binding,
                coordinates=copy(t.m.view.coordinates),
                # A connected fit's inference units, not its scientific subjects.
                levels=copy(hasproperty(t, :ownership) ? _brm_svi_unit_levels(t) : t.m.view.levels),
                globals=copy(t.globals), units=copy.(t.units)),
            _brm_svi_snapshot_run(state.run, t, state.controls, draws)))
    end
end

_brm_svi_deliver(f, snapshot) = (f(snapshot); nothing)
function _brm_svi_deliver(path::AbstractString, snapshot)
    temporary = path * ".partial"
    open(io -> serialize(io, snapshot), temporary, "w")
    mv(temporary, path; force=true)
    nothing
end

function _brm_svi_snapshot_run(run::_BRMSVIRun, t, c, draws)
    q = _brm_svi_result_q(run)
    (; iteration=run.iteration, max_iterations=c.max_iterations,
       phase=_brm_svi_phase(run, c), stages=_brm_svi_stages(run, c),
       q_kind=run.average === nothing ? :raw_iterate : :polished_average,
       averaged_iterations=run.average === nothing ? 0 : run.iteration - run.polish_start,
       q=(; μ=copy(q.μ), ρ=copy(q.ρ), Β=copy(q.Β), M=copy.(q.M),
          Λ=_brm_svi_plain_units(q.Λ), U=_brm_svi_plain_units(q.U)),
       elbo_windows=first.(run.windows),
       elbo_windows_trimmed=[w.center for w in run.windows],
       clipped_steps=run.adam.clipped,
       elbo=isempty(run.trace) ? nothing : last(run.trace),
       draws=_brm_svi_draws(copy(run.rng), t, q, draws))
end

# The approximation a run currently stands for: its raw iterate, or while and
# after polishing the average of the polished iterates, a dense unit's precision
# being the inverse of its averaged covariance. The run itself is left unchanged, so it
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

# Where polishing's running average starts: a copy of q in which each dense
# unit's `Λ` holds its covariance (`_brm_svi_unit_average!`).
function _brm_svi_average_start(q)
    average = deepcopy(q)
    foreach(_brm_svi_unit_average_start!, average.Λ)
    average
end

# Weighted running averages over stretches during which a unit was unchanged.
# Dense factors average covariance (the inverse of their precision, src/
# nlme_svi_units.jl says why); structured factors average log scales and
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
        _brm_svi_unit_average!(average.Λ[i], q.Λ[i], weight)
        through[i] = iteration
    end
    average
end

# The subjects' precisions and Cholesky factors from a running average (a dense
# unit's averaged covariance in its `Λ`), in place.
function _brm_svi_factor!(q)
    for i in eachindex(q.Λ)
        _brm_svi_unit_from_average!(q.U[i], q.Λ[i])
    end
    q
end

# `n` draws from q as packed RK points, one per column. Under learned centering
# each unit's draw of its learned coordinates is mapped back to the packed ones
# at that column's globals.
function _brm_svi_draws(rng, t, q, n)
    U = repeat(t.m.unit, 1, n)
    centering = _brm_svi_centering(t)
    for j in 1:n
        g = _brm_svi_draw_global(q, randn(rng, length(q.μ)), randn(rng, size(q.Β, 2)))
        U[t.globals, j] .= g
        frame = centering === nothing ? nothing : _brm_svi_centering_frame(centering, t, g)
        for (i, block) in enumerate(t.units)
            u = _brm_svi_draw_unit(q, i, _brm_svi_unit_noise(rng, q.U[i]))
            U[block, j] .= centering === nothing ? u :
                _brm_svi_centered_coordinates(centering, frame, q, i, g, u)
        end
    end
    _brm_svi_fill_absent_draws!(rng, U, t, t.previous)
end

_brm_svi_fill_absent_draws!(rng, draws, t, ::Nothing) = draws

# The model through its preparation phases, as for `brm_nlme_fit`: lowering to
# RK and `brm_nlme_model`. A stage `x` has already passed is reused. A resume
# passes its checkpoint's completion parametrization (`completions`), which
# fixes the lowering without reading the groupings.
_brm_svi_lowered(x::BRMI, phase; completions=nothing, groups...) =
    _brm_run_phase(phase) do _
        completions === nothing ? _brm_nlme_lowered(x; groups...) :
            _brm_svi_checkpoint_lowered(x, completions)
    end
_brm_svi_lowered(x::Union{RKBRMI,BRMNLMEModel}, phase; completions=nothing, groups...) =
    (_brm_reuse_phase!(phase); x)
_brm_svi_lowered(x, phase; completions=nothing, groups...) = throw(ArgumentError(
    "brm_svi_fit: expected a @brm model (BRMI), an RKBRMI or a BRMNLMEModel, " *
    "got $(typeof(x))"))

# An `RKBRMI` keeps the model this prepared for it (`x.prepared.nlme`), so a later call on
# the same backend with the same options reuses it, and the target it holds,
# as a call on the returned `diagnostics.model` does. The options are its key
# because they decide the model; nothing a fit changes lives in it.
function _brm_svi_model(x::RKBRMI, phase; ad_backend, shared_point, subject_group,
        nested_groups, shared_groups, unit_partition)
    ad_backend = ad_backend === nothing ? _brm_rk_default_ad("brm_svi_fit", x) : ad_backend
    unit_partition = something(unit_partition, :subjects)
    key = (; ad_backend, shared_point=shared_point === nothing ? nothing :
        Vector{Float64}(shared_point), subject_group, nested_groups, shared_groups,
        unit_partition)
    slot = x.prepared
    held = lock(() -> slot.nlme, slot.lock)
    if held !== nothing && isequal(held.key, key)
        _brm_reuse_phase!(phase)
        return held.model
    end
    m = _brm_run_phase(phase) do node
        _brm_nlme_model(x, node; ad_backend, observation_ad_backend=nothing,
            shared_point, subject_group, nested_groups, shared_groups, unit_partition)
    end
    lock(() -> (slot.nlme = (; key, model=m)), slot.lock)
    m
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
  `shared_groups`. A BRMI's `mi(...)` completions whose law reads the units'
  draws are lowered non-centered, as by [`brm_nlme_fit`](@ref). Every
  observation must belong to one subject's cells: a
  cell of a `@plate for` over the subjects or over their rows, or a row of a
  flat-row response grouped by the subjects.
- `family` — the global factor: `MeanFieldGlobal()` (diagonal) or
  `LowRankGlobal(r)` (diagonal plus rank `r`).
- `centering` — fresh fits default to [`LearnedCentering`](@ref), jointly
  learning each supported unit cell's coordinates and variational factor.
  This includes unit-owned Gaussian random effects, Gaussian vector elements
  and grouped HSGP weights. `diagnostics.centering` reports the cells and
  eligibility exclusions; authored `@plate` cell latents are outside this
  mechanism and its report. Pass `centering=nothing` for the plain independent
  factorization. Streaming updates and optimizer resume inherit the saved
  policy, including an older plain fit's. A factor-level start is a fresh fit;
  its unit Gaussians are transported into the newly selected coordinates.
- `unit_dense_limit` — units with at most this many coordinates use a dense
  Gaussian and natural-gradient updates (default 64, allowed 0:256).
- `unit_curvature` — how a dense unit's step estimates its curvature.
  `:orthogonal` (default): on each visit the unit is evaluated at d mutually
  orthogonal antithetic pairs of draws from its factor (2d gradient
  evaluations for d coordinates; a randomly rotated spherical-radial cubature
  rule), whose Stein estimate is its whole expected curvature, exact when the
  unit's term is quadratic in its coordinates, and the step defaults to
  `unit_step = 1`. The step reads the estimate's positive part, which on a
  quadratic term is all of it: on a saturating term the estimate is mostly noise
  with nearly symmetric eigenvalues until the unit's stiff directions are resolved,
  and its negative part would shrink the factor's precision far below the target's
  curvature. `:pair`: one antithetic pair (2 evaluations) and its
  rank-one estimate's positive part, with `unit_step = 0.1`, BRM 0.2.242's step;
  a unit of tens of coordinates then needs far more visits to settle than a
  minibatch run gives it, and the hyperparameters follow its unsettled factor.
  Structured units (above `unit_dense_limit`) are unaffected: they evaluate one
  pair, and `unit_step` defaults to 0.1 for them either way.
  Larger units use a diagonal-plus-low-rank covariance with `unit_rank`
  columns (default 8, allowed 0:32, clipped to the unit dimension). Rank zero
  gives a diagonal factor. Their reparameterized ELBO gradients use Adam with
  posterior-scale preconditioning and bounded relative steps. Polishing
  averages their log scales and loadings; dense units average covariances.
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
- `cell_units` — where connected mode puts likelihood cells that read no
  subject's effect. `:components`, its default, gives them inference units of
  their own, appended after the subjects' units, so a batch subsamples them:
  cells reading a common ordinary coordinate that no subject's cell reads share
  a unit, transitively, and cells with no such coordinate are grouped into about
  √m units of √m cells. A coordinate that joins nearly all of them, such as a
  shared population mean, joins them into one unit until it is promoted through
  `global_coordinates` or `separator`. `:first_unit` keeps them all in the first
  unit and counts unit work without rows, as earlier versions did, and is the
  only choice with `unit_partition=:subjects`. Splitting these cells changes the
  variational factorization and adds gradient variance from subsampling them,
  never the density. Checkpoints keep the layout they were made with.
- `unit_sampling` — defaults to `:work` in explicitly connected mode and
  `:uniform` otherwise. `:uniform` selects distinct units with weight `N/B`;
  `:work` makes `B` draws with replacement, with probabilities
  `pᵢ = (1/N + wᵢ/sum(w))/2`. The structural work proxy `wᵢ` counts local
  coordinates, structural reads in the unit's likelihood/prior factors and
  the rows of its vector-valued observation cells; it is not measured runtime.
  All-zero work gives uniform probabilities.
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
- `field_states` — how a read-pair field (`gmrf(...; neighbors=m)`) is held. `:global`
  (default) gives each state its own global Gaussian factor. `:units` puts a state read
  by one unit into that unit's factor, next to its effects, so its cost and storage follow
  the unit's reads, and a state read by several units stays global. It reads the field's
  lengths from the globals, and has no streaming form. A mesh `gmrf` is refused either way:
  one factor per mesh state collapses the learned lengths.
- `start` / `inits` — where the global and unit means start: BRM's prior-median
  starting point by default, a full RK packed point `start`, or values keyed
  by prior address as [`brm_start`](@ref) takes them. Declarations without a
  recorded median, including ordinary model-scope vectors, retain the packed
  unit point; entries not supplied in `inits` retain that default point.
  `start` may instead be a factor-level start: a [`BRMSVIStart`](@ref) from
  [`brm_svi_combine`](@ref), or fits or states of the same declaration on
  slices of the data (a vector, or one), combined as `brm_svi_combine` does
  with its defaults. It sets the mean and scale of every global and the factor
  of every unit it holds, matched by semantic coordinate identity; the rest
  keeps the default start, the global rank part starts at zero and Adam's
  moments start fresh. It is a start, not a prior: the target is unchanged.
  `diagnostics.start` holds its receipt, with `applied` counting the globals
  and units it started (`nothing` for any other start).
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
- Optimizer controls: `max_iterations`, `lr_global` (Adam on the globals while
  exploring, in the coordinates' own units),
  `unit_step` (the subjects' natural-gradient step, by default 1 under
  `unit_curvature = :orthogonal` and 0.1 under `:pair`; a step that would change a
  subject's precision more than 2.5-fold along some direction, or move its
  mean more than one sd of its factor, is shortened for that subject. A structured
  unit's Adam takes it as the step size of its mean and log scales, by default 0.1
  under either curvature, and its rank part steps a tenth as far; a value you pass
  applies to both families),
  `polish_lr` (Adam on the globals while settling and polishing, as a fraction
  of each global's current scale), `window`,
  `patience`, `min_windows`, `trend_windows`, `settle_windows`, `polish_windows`,
  `global_scale` and `unit_scale` (initial factor scales). The run explores at
  constant steps until the trimmed ELBO has not risen detectably over `patience`
  windows (after at least `min_windows`) and its least-squares slope over the last
  `trend_windows` windows (default `4 patience + 1`; `0` turns this off, the rule of
  BRM 0.2.263 and earlier) is not above twice its standard error (read both with and
  without the first `patience` windows, which climb from the start), settles at the
  constant scale-relative step until the
  same test passes on settling's own windows (for at most `settle_windows`
  windows; `0` skips settling, the schedule of BRM 0.2.241 and earlier), then
  polishes for `polish_windows` windows with decaying steps and returns the
  average of the polished iterates.
- `sd_bound` (default `2`), **a stopgap**: after every global step,
  a global whose prior reads no other coordinate has its marginal sd under q held
  at most `sd_bound` times the sd of its prior's Gaussian projection, so a global
  whose gradient is mostly noise cannot random-walk wide and overflow the density.
  It treats the symptom, not the optimizer's blindness to the prior under that
  noise, and it holds q too narrow where a posterior is wider than the bound. A
  fit that ends with a global at its bound warns and names it
  (`diagnostics.sd_bound`); `sd_bound=nothing` turns the bound off. A resumed
  checkpoint keeps its own factor, `2` for one written before `sd_bound` existed.
- Interruptible runs: `stop_after` ends this call after that many iterations
  (the run continues with [`brm_svi_resume`](@ref) on the same model and
  data; the stopping rule and step-size schedule are unaffected). An optional
  first argument `save` (a function, so a `do` block works) receives a
  [`BRMSVIOptimizerState`](@ref) after every iteration that is a multiple of
  `checkpoint_every` (default: once per window), when this call stops, and
  before an interrupt propagates. With `laplace=GaussianBlockLaplace()` the
  default is every 50 updates, and the interrupt is checked every 50 updates.
- Watching a running fit: `snapshot` is a function, or a file path that the run
  rewrites atomically so it always holds the latest complete snapshot. It
  receives a detached copy of the current approximation and diagnostics, without
  the call returning and without decoding a checkpoint, after every iteration
  that is a multiple of `snapshot_every` (default: once per window, or every 50
  updates with `laplace=GaussianBlockLaplace()`), when this call stops, and before
  an interrupt propagates. `snapshot_draws`
  adds that many draws (default none); each snapshot costs drawing them plus a
  copy of the factors. The run is the same to the last bit whether it is observed
  or not: draws come from a copy of the random stream, and a snapshot's `q` and
  `draws` equal those of a result stopped at its iteration. A snapshot is one
  named tuple of plain arrays, so a saved one loads without BRM or the model:

  - `status`: `:running` while the call continues. Only the boundary where the
    call ends says more: `:converged`, `:max_iterations`, `:stopped` (`stop_after`
    was reached; the run can continue) or `:interrupted`.
  - `iteration`, `max_iterations`, `seconds` (since optimization started in this
    call), `threads`, `format`, and `model_identity`, which names the emitted
    program, bound data and packed coordinates.
  - `phase`: `:exploring`, `:settling` or `:polishing`; for GaussianBlockLaplace,
    `:fixed_budget`. `stages`, as in `diagnostics`.
  - `q`: `μ`, `ρ`, `Β`, `M`, `Λ` and `U`, as in `diagnostics.q`. `q_kind` says
    what it is: `:raw_iterate`, `:polished_average` (the Polyak–Ruppert average,
    `averaged_iterations` of them, 0 at the iteration polishing starts) or, for
    GaussianBlockLaplace, `:window_average`. A unit above `unit_dense_limit` appears
    in `Λ` and `U` as `(; family=:diagonal_lowrank, ρ, Β)`, its covariance being
    `Diagonal(exp.(2ρ)) + Β * Β'`, so that no BRM type is needed to read it.
  - `coordinates`, `levels`, `globals` and `units`: `q.μ[k]` is coordinate
    `coordinates[globals[k]]`, `q.M[i][k]` is coordinate `coordinates[units[i][k]]`,
    of unit `levels[i]`.
  - `elbo_windows` (the mean ELBO per completed window; for GaussianBlockLaplace
    per 50 updates), `elbo` (the last iteration's value, `nothing` before the
    first). Without GaussianBlockLaplace also `elbo_windows_trimmed`, which the
    stopping rule reads, and `clipped_steps`, as in `diagnostics`.
  - `draws`: coordinates × `snapshot_draws`, as the result's `draws`.

  BRM keeps no snapshot: the size of one is that of `q` and `draws` plus two numbers
  per window (one for GaussianBlockLaplace), and a path holds only the latest. To
  keep more, pass a function.

Returns `draws`, `ndraws` draws from the fitted approximation as RK packed
unconstrained points (coordinates × draws, one column per draw, the order of
`coordinate_names(rk.model.layout)`), their `coordinates`, and `diagnostics`:
the ELBO trace, up to a constant (`elbo`, per iteration; `elbo_windows`, the
plain mean of each window, which a single draw can dominate; and
`elbo_windows_trimmed`, the 10% trimmed mean of each window, which the stopping
rule reads), `clipped_steps` (how many global steps had a coordinate's gradient
limited to `10 √w` times its running rms, with `w` the batch's largest unit
weight, `N / B` for uniform sampling, so that one extreme draw does not stop that
coordinate's optimization), `sd_bound` (`factor`, `nothing` when off, and per
global `coordinates` and `bound`, `Inf` where it has none, `at_bound`, the
globals whose marginal sd in the last iterate is at it, and `stopgap = true`),
`converged` (the optimizer completed its stopping and polishing rules, not a
certificate of posterior accuracy),
`finished` (converged or `max_iterations` reached; `false` after `stop_after`),
`iterations`, `stages` (the iterations spent `exploring`, `settling` and
`polishing`, and `settled`: whether settling ended on its test rather than at
`settle_windows`; `converged` with `settled = false` means polishing averaged
iterates that were still rising), the variational parameters `q`, the packed positions of the
global coordinates (`globals`), `seconds` of optimization in this call,
`threads` (the tasks this call used), `target` (below), the units' sampling
probabilities, work, visits and draw counts (`sampling`), the actual
`unit_gradient_calls` (2 per visit of a unit, 2d per visit of a dense unit of
d coordinates under `unit_curvature = :orthogonal`), the prepared `model`, and the run's `optimizer` state
([`brm_svi_optimizer_checkpoint`](@ref) copies it). The draws come from a copy
of the optimizer's random stream: they do not advance `rng`, and a continued
run is unaffected by them. Learned centering can recover population spread
lost by the plain factorization; neither the factor family nor optimizer
completion guarantees accurate means or calibrated intervals. Compare important
population uncertainty with a feasible full-MCMC or exact reference.

The prepared `model` keeps its minibatch target and worker samplers for its next
fit or continuation in this process, so passing `diagnostics.model` (or the
[`BRMNLMEModel`](@ref) you prepared) prepares neither again. An `RKBRMI` keeps
the model a call prepared from it, the last one: a later call on the same backend
with the same `ad_backend`, `shared_point`, groupings and `unit_partition` takes
that model and its target, and other options prepare another model, which the
backend then keeps instead. A copy or a serialized backend keeps none. A `BRMI`
is lowered to a new `RKBRMI`, and so prepared, on every call.
`diagnostics.target` reports what this call did: `reused` (it took the model's
target), `workers_built` and the `seconds` of that phase. A connected model's
target is reused only for the same `global_coordinates`, separator policy,
budgets and `cell_units`, since those decide its units, and any model's only for
the same `field_states`. The model holds one
target, the size of its partition samplers and work arrays, for as long as the
model lives; a copy or a serialized model holds none. A call that throws while
optimizing does not put its target back, and concurrent calls on one model each
use their own. Not with `laplace=GaussianBlockLaplace()`, which prepares its
target on every call.
"""
brm_svi_fit(x; kwargs...) = _brm_svi_initial_fit(nothing, x; kwargs...)
brm_svi_fit(save::Function, x; kwargs...) = _brm_svi_initial_fit(save, x; kwargs...)
function _brm_svi_initial_fit(save, x; laplace=nothing, kwargs...)
    laplace === nothing && return _brm_svi_fit(save, x, nothing; kwargs...)
    get(kwargs, :centering, nothing) === nothing || throw(ArgumentError(
        "brm_svi_fit: learned centering moves the minibatch engine's unit factors; " *
        "laplace=GaussianBlockLaplace() integrates them instead, so pass only one of them"))
    _brm_svi_laplace_dispatch(save, x, nothing, laplace; kwargs...)
end

function _brm_svi_fit(save, x, previous; rng=Random.default_rng(), batch_size=32,
        max_iterations=100_000, family=nothing, unit_sampling=nothing,
        unit_dense_limit=64, unit_rank=8,
        lr_global=0.01, unit_step=nothing, unit_curvature=:orthogonal, start=nothing,
        inits=nothing, global_scale=0.1, unit_scale=1.0, polish_lr=0.05, window=nothing,
        patience=10, min_windows=40, polish_windows=20,
        settle_windows=_BRM_SVI_SETTLE_WINDOWS, centering=missing, sd_bound=_BRM_SVI_SD_BOUND,
        trend_windows=nothing, kwargs...)
    family = _brm_svi_family(previous, family)
    sd_bound = _brm_svi_check_sd_bound(sd_bound)
    settle_windows isa Integer && settle_windows >= 0 || throw(ArgumentError(
        "brm_svi_fit: settle_windows must be a nonnegative integer (0: polish straight " *
        "after exploration)"))
    trend_windows === nothing || trend_windows isa Integer &&
        (trend_windows == 0 || trend_windows >= 3) || throw(ArgumentError(
        "brm_svi_fit: trend_windows must be nothing (4 patience + 1), 0 (off: the " *
        "two-window test alone) or an integer of at least 3, got $(repr(trend_windows))"))
    trend_windows = Int(something(trend_windows, _brm_svi_default_trend_windows(patience)))
    unit_curvature in (:orthogonal, :pair) || throw(ArgumentError(
        "brm_svi_fit: unit_curvature must be :orthogonal or :pair, got $(repr(unit_curvature))"))
    steps = _brm_svi_unit_steps(unit_step, unit_curvature)
    centering = _brm_svi_resolve_centering(previous, centering)
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
        centering === nothing || ((t, q) = _brm_svi_with_centering(t, q, centering))
        c = _brm_svi_controls(length(t.units); batch_size, max_iterations, lr_global,
            steps..., polish_lr, window, patience, min_windows, polish_windows,
            settle_windows=Int(settle_windows), unit_curvature, sd_bound, trend_windows,
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
        snapshot=nothing, snapshot_every=nothing, snapshot_draws=0,
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=nothing, threads=nothing,
        global_coordinates=nothing, separator=nothing, max_unit_coordinates=4096,
        max_unit_work=1_000_000, max_promoted_coordinates=128, cell_units=nothing,
        check_lowered=Returns(nothing), checkpoint_completions=nothing, field_states=:global)
    ndraws isa Integer && ndraws >= 0 || throw(ArgumentError(
        "$caller: ndraws must be a nonnegative integer"))
    _brm_svi_check_interruption(caller, save, stop_after, checkpoint_every)
    _brm_svi_check_snapshot(caller, snapshot, snapshot_every, snapshot_draws)
    threads = _brm_svi_threads(caller, threads, ad_backend)
    _brm_with_progress(progress; description) do node
        _brm_with_phases(node, _BRM_SVI_PHASES) do phases
            lowered = _brm_svi_lowered(x, phases.lower; completions=checkpoint_completions,
                subject_group, nested_groups, shared_groups, unit_partition)
            check_lowered(lowered)
            m = _brm_svi_model(lowered, phases.prepare;
                ad_backend, shared_point, subject_group, nested_groups, shared_groups, unit_partition)
            prepared = Ref{Any}(nothing)
            optimizing = false
            try
                started = time()
                t, state, reuse = _brm_run_phase(phases.target) do phase
                    # SVI's partition sampler, on first use, reports under this phase.
                    prepared[], reused = _brm_with_cell_progress(() -> _brm_svi_checkout!(m;
                        global_coordinates, previous, separator, max_unit_coordinates,
                        max_unit_work, max_promoted_coordinates, cell_units, field_states), phase)
                    reused && _brm_update_progress!(phase, "reusing this model's prepared target")
                    t, state = init(prepared[].target)
                    workers = prepared[].workers
                    before = length(workers)
                    t = _brm_svi_with_workers(t, min(threads, state.controls.batch_size), phase;
                        workers)
                    t, state, (; reused, workers_built=length(workers) - before)
                end
                target_seconds = time() - started
                optimizing = true
                run, c = state.run, state.controls
                seconds, draws = _brm_with_serial_blas() do
                    seconds = @elapsed _brm_enter_phase(phases.fit) do phase
                        _brm_svi_fit!(run, t, c; stop_after=something(stop_after, typemax(Int)),
                            checkpoint_every=something(checkpoint_every, c.window), save,
                            checkpoint=_ -> _brm_svi_detach(state), node=phase,
                            observe=_brm_svi_observer(snapshot, t, state, snapshot_draws),
                            observe_every=something(snapshot_every, c.window))
                    end
                    seconds, _brm_run_phase(phases.draws) do _
                        _brm_svi_draws(copy(run.rng), t, _brm_svi_result_q(run), ndraws)
                    end
                end
                optimizing = false
                q = _brm_svi_result_q(run)
                centering = _brm_svi_centering(t)
                centering_maps = _brm_svi_centering_maps(t, q)
                bound = _brm_svi_sd_bound_report(t, run.q, _brm_svi_sd_bound_factor(c))
                _brm_svi_warn_sd_bound(bound; caller)
                (; draws, coordinates=copy(m.view.coordinates), diagnostics=(;
                    threads=length(t.plans), target=(; reuse..., seconds=target_seconds),
                    elbo=run.trace, elbo_windows=first.(run.windows),
                    elbo_windows_trimmed=[w.center for w in run.windows],
                    clipped_steps=run.adam.clipped, sd_bound=bound,
                    run.converged,
                    finished=_brm_svi_finished(run, c), iterations=run.iteration,
                    stages=_brm_svi_stages(run, c), q,
                    globals=t.globals, units=t.units, owner=t.ownership.owner,
                    global_coordinates=t.ownership.promoted,
                    unit_levels=_brm_svi_unit_levels(t),
                    subject_unit=hasproperty(t.ownership, :subject_unit) ?
                        t.ownership.subject_unit : m.view.subject_unit,
                    separator=hasproperty(t.ownership, :separator) ? t.ownership.separator : nothing,
                    unit_families=_brm_svi_unit_description.(q.U),
                    seconds, model=m, source_id=_brm_svi_source_id(m.backend),
                    sampling=run.sampling, unit_gradient_calls=run.unit_gradient_calls,
                    sparse_globals=_brm_svi_sites_diagnostics(run.sites, t),
                    start=get(t, :start_receipt, nothing),
                    streaming=_brm_svi_carried_state(previous, t, q, centering_maps), optimizer=state,
                    centering_maps,
                    centering=centering === nothing ? nothing :
                        _brm_svi_centering_report(centering, q, _brm_svi_unit_levels(t))))
            finally
                # A refused checkpoint leaves the target as it found it; an exception
                # while optimizing may have left an AD workspace mid-evaluation.
                prepared[] === nothing || optimizing || _brm_svi_checkin!(m, prepared[])
            end
        end
    end
end

_brm_svi_family(::Nothing, ::Nothing) = MeanFieldGlobal()
_brm_svi_family(::Nothing, family::BRMSVIFamily) = family
function _brm_svi_initialize(t, ::Nothing, family; start, inits, global_scale, unit_scale, unit_dense_limit=64, unit_rank=8)
    _brm_svi_factor_start(start) && return _brm_svi_factor_initialize(t, start, family;
        inits, global_scale, unit_scale, unit_dense_limit, unit_rank)
    point = _brm_svi_start(t.m, start, inits)
    q = _brm_svi_params(t, point[t.globals], family; global_scale, unit_scale, unit_dense_limit, unit_rank)
    for (i, block) in enumerate(t.units)
        q.M[i] .= point[block]
    end
    t, q
end
_brm_svi_carried_state(::Nothing, m, q, maps=nothing) = nothing
