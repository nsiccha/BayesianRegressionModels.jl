# src/nlme_laplace_structured.jl — the per-unit Laplace on the model's rows.
#
# The generic `BRMLaplaceProblem` sees the RK density as one function of every
# coordinate, so it extracts each unit's Hessian and third derivatives from
# whole-model AD sweeps (one per within-unit position), and its units step in
# lockstep. Most models need none of that: a regression row depends on its
# unit's standardized draws z_i only through its linear predictors. For every
# likelihood term t (one response at one row) of unit i, the term's link-scale
# parameters are
#
#     ψ_t = A_t(φ) + Z_t b_i,      b_i = M(φ) z_i,
#
# with A_t(φ) the parameters at z = 0, Z_t the row's random-effect design
# (constant data) and M = diag(scales) L each random-effect block's draws map.
# The term's log density ℓ_t(ψ_t) and its first three ψ-derivatives have
# closed forms (`_brm_lap_row`), so each unit's mode is its own Newton search
# over d numbers on its own rows: no AD at all, and nothing shared with other
# units (its own iterations, line search, stop and restart).
#
# With C_t = Z_t M, h_i = I + Σ_t C_tᵀ W_t C_t (W_t = −∂²ℓ_t) at the mode,
#
#     L(φ) = Σ_t ℓ_t(ψ̂_t) − |ẑ|²/2 + prior(φ) + Σ_i (d/2 log 2π − ½ log det h_i)
#
# and the gradient is (∂A/∂φ)ᵀ w plus a d × d contraction. For a regression
# A is affine in φ (design-matrix products, factor effects, log of a sampled
# shape), so ∂A/∂φ is constant data recovered once (`J`, verified at other
# points like Z) and no AD runs per evaluation beyond the prior's gradient and
# the d-dimensional draws maps; otherwise the RK parameters are evaluated and
# pulled back per evaluation:
#
#     ∇L = (∂A/∂φ)ᵀ w + ⟨Γ, ∂M/∂φ⟩ + ∇prior,
#     w_t = ℓ′_t + ½ τ_t + ℓ″_t C_t v_i,   τ_t,s = Σ_pq (C_t h_i⁻¹ C_tᵀ)_pq ℓ‴_t,pqs,
#     v_i = h_i⁻¹ s_i,   s_i = ½ Σ_t C_tᵀ τ_t,
#     Γ = Σ_t Z_tᵀ (w_t ẑ_iᵀ + ℓ″_t C_t h_i⁻¹ + ℓ′_t v_iᵀ).
#
# τ is the log determinant's derivative through the row parameters, v the
# implicit-function correction for the modes' movement (dẑ_i/dφ = h_i⁻¹
# ∂²f/∂z∂φ), and Γ collects everything that reaches φ through M.

const _BRM_LAP_POISSON = Int8(1)   # ψ = (log μ,)
const _BRM_LAP_BINOMIAL = Int8(2)  # ψ = (logit p,), n trials (1 for Bernoulli)
const _BRM_LAP_NB2 = Int8(3)       # ψ = (log μ, log φ)
const _BRM_LAP_NORMAL = Int8(4)    # ψ = (μ, log σ)

_brm_lap_width(f::Int8) = f === _BRM_LAP_NB2 || f === _BRM_LAP_NORMAL ? 2 : 1

# One term's log density (without its constant, `_brm_lap_constant`) and its
# derivatives in its link-scale parameters (a, b), symmetric entries once.
# `third` adds the third derivatives the gradient needs. A term whose second
# parameter does not move with the unit's effects (`moving = false`, a scalar
# negative-binomial shape) skips what the mode search and gradient never read:
# its b-derivatives beyond ∂b and ∂a∂b, and, in the mode search (`third =
# false`), the log density's terms in b alone.
struct _BRMLapRow
    v::Float64
    g1::Float64
    g2::Float64
    h11::Float64
    h12::Float64
    h22::Float64
    t111::Float64
    t112::Float64
    t122::Float64
    t222::Float64
end

# ψ_n(y + P) − ψ_n(P) for n = 0, 1, 2 (pg is its negative for n = 2). For a
# small count y they are exact sums over the integers below y (ψ(x + 1) =
# ψ(x) + 1/x and its derivatives), cheaper than the special functions.
@inline function _brm_lap_polygammas(y::Float64, P::Float64, third::Bool)
    if 0 <= y <= 50 && isinteger(y)
        s1 = s2 = s3 = 0.0
        for k in 0:(Int(y) - 1)
            x = inv(P + k)
            x2 = x * x
            s1 += x
            s2 += x2
            s3 += x2 * x
        end
        return s1, -s2, -2s3
    end
    yP = y + P
    dg = digamma(yP) - digamma(P)
    tg = trigamma(yP) - trigamma(P)
    dg, tg, third ? polygamma(2, P) - polygamma(2, yP) : 0.0
end

@inline function _brm_lap_row(f::Int8, y::Float64, n::Float64, a::Float64, b::Float64,
        third::Bool, moving::Bool=true)
    if f === _BRM_LAP_POISSON
        μ = exp(a)
        return _BRMLapRow(y * a - μ, y - μ, 0.0, -μ, 0.0, 0.0, -μ, 0.0, 0.0, 0.0)
    elseif f === _BRM_LAP_BINOMIAL
        p = logistic(a)
        w = n * p * (1 - p)
        return _BRMLapRow(y * a - n * log1pexp(a), y - n * p, 0.0, -w, 0.0, 0.0,
            -w * (1 - 2p), 0.0, 0.0, 0.0)
    elseif f === _BRM_LAP_NB2
        # μ = exp(a), shape P = exp(b), q = μ/(μ + P); log(1 - q) = b - log(μ + P).
        P = exp(b)
        q = logistic(a - b)
        r = 1 - q
        l1q = -log1pexp(a - b)
        yP = y + P
        g1 = y - yP * q
        h11 = -q * r * yP
        if !moving
            third || return _BRMLapRow(P * l1q + y * (a - b) + y * l1q, g1, 0.0, h11,
                q * g1, 0.0, 0.0, 0.0, 0.0, 0.0)
            dg = first(_brm_lap_polygammas(y, P, false))
            return _BRMLapRow(loggamma(yP) - loggamma(P) + P * l1q + y * (a - b) + y * l1q,
                g1, P * (l1q + 1 + dg) - yP * r, h11, q * g1, 0.0,
                -q * r * (1 - 2q) * yP, q * r * (y - 2q * yP), 0.0, 0.0)
        end
        dg, tg, pg = _brm_lap_polygammas(y, P, third)
        v = loggamma(yP) - loggamma(P) + P * l1q + y * (a - b) + y * l1q
        g2 = P * (l1q + 1 + dg) - yP * r
        h12 = q * g1
        h22 = P * (P * tg + l1q + dg + 2) + yP * r^2 - (3P + y) * r
        third || return _BRMLapRow(v, g1, g2, h11, h12, h22, 0.0, 0.0, 0.0, 0.0)
        t111 = -q * r * (1 - 2q) * yP
        t112 = q * r * (y - 2q * yP)
        t122 = -q * (P + 2yP * r^2 - (3P + y) * r)
        t222 = -P * (P^2 * pg - 3P * tg - l1q - dg - 3) - 2yP * r^3 +
            3(2P + y) * r^2 - (7P + y) * r
        return _BRMLapRow(v, g1, g2, h11, h12, h22, t111, t112, t122, t222)
    else
        # Normal with mean a and log standard deviation b.
        e = exp(-2b)
        d = y - a
        return _BRMLapRow(-0.5 * d^2 * e - b, d * e, d^2 * e - 1, -e, -2d * e, -2d^2 * e,
            0.0, 2e, 4d * e, 4d^2 * e)
    end
end

# A term's log density constant, which no parameter moves.
function _brm_lap_constant(f::Int8, y::Float64, n::Float64)
    f === _BRM_LAP_POISSON && return -loggamma(y + 1)
    f === _BRM_LAP_BINOMIAL && return loggamma(n + 1) - loggamma(y + 1) - loggamma(n - y + 1)
    f === _BRM_LAP_NB2 && return -loggamma(y + 1)
    -0.5 * log(2π)
end

# An RK value's link-scale parameter, and its derivative: linear predictors
# are link-scale values already (RK's families take them, `PoissonLog`,
# `NegativeBinomial2Log`, …, and a scale predictor's link inverts at its use),
# a sampled scale is its natural value.
_brm_lap_link(transform::Symbol, θ) = transform === :log ? log(θ) : θ
_brm_lap_dlink(transform::Symbol, θ) = transform === :log ? inv(θ) : one(θ)

"""
    BRMStructuredLaplaceProblem

The per-unit Laplace of a regression-shaped RK model, built on its rows; see
[`brm_laplace_problem`](@ref). Same interface as [`BRMLaplaceProblem`](@ref):
LogDensityProblems order 1 over `names` (packed positions `population`),
[`brm_laplace_draws`](@ref), and the counters `nonfinite` and `unconverged`.
"""
mutable struct BRMStructuredLaplaceProblem{K}
    backend::RKBRMI
    view::BRMNLMEView
    population::Vector{Int}
    names::Vector{Symbol}
    blocks::Vector{Vector{Int}}
    d::Int
    kernels::K
    # Likelihood terms, grouped by unit (`unit_terms[i]`), and their first
    # parameter entry (`entry[t]`, a second at `entry[t] + 1` for width 2).
    family::Vector{Int8}
    y::Vector{Float64}
    trials::Vector{Float64}
    constant::Vector{Float64}
    entry::Vector{Int}
    moving::Vector{Bool}     # the second parameter moves with the unit's effects
    unit_terms::Vector{UnitRange{Int}}
    # Parameter entries: the RK value (`port`, `index` 0 for a scalar) each
    # reads and its transform to the link scale, and the row's random-effect
    # design `Z[:, e]` on b_i.
    port::Vector{Int}
    index::Vector{Int}
    link::Vector{Symbol}
    fixed::Vector{Float64}   # a literal parameter's value (`port` 0)
    Z::Matrix{Float64}
    # Z by entry, its nonzeros only: rows `zptr[e]:zptr[e+1]-1` of (`zrow`,
    # `zval`). A row reads one or a few of its unit's effects.
    zptr::Vector{Int}
    zrow::Vector{Int}
    zval::Vector{Float64}
    # Coordinates that move M (random-effect scales and correlation factors).
    omega::Vector{Int}
    # A(φ) = A0 + J (φ − φ0) when it is affine (`affine`): J by entry, rows
    # `jptr[e]:jptr[e+1]-1` of (`jcol` population index, `jval`).
    affine::Bool
    φ0::Vector{Float64}
    A0::Vector{Float64}
    jptr::Vector{Int}
    jcol::Vector{Int}
    jval::Vector{Float64}
    # Warm start: the last converged modes and their factors.
    u::Vector{Float64}
    factors::Base.RefValue{Any}
    tol::Float64
    maxiter::Int
    nonfinite::Int
    unconverged::Int
end

Base.show(io::IO, p::BRMStructuredLaplaceProblem) = print(io,
    "BRMStructuredLaplaceProblem(", length(p.population), " population coordinates, ",
    length(p.blocks), " units, ", p.d, " effects each, ", length(p.family),
    " likelihood terms)")

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: the
# prepared queries the structured Laplace evaluates (see that file).
function _rk_slap_kernels end
function _rk_slap_values end
function _rk_slap_pullback end
function _rk_slap_effects end
function _rk_slap_pointwise end

# Families the row structure covers: the family, its location link, and each
# parameter slot's RK value with its transform to the link scale.
function _brm_slap_family(spec)
    spec.family === :poisson_log &&
        return _BRM_LAP_POISSON, :log, ((spec.predictor, :identity),)
    spec.family in (:bernoulli_logit, :binomial_logit) &&
        return _BRM_LAP_BINOMIAL, :logit, ((spec.predictor, :identity),)
    if spec.family === :nb2_log || spec.family === :gaussian
        scale = spec.scale_predictor === nothing ? (spec.scale, :log) :
            (spec.scale_predictor, :identity)
        first(scale) === nothing && return nothing
        f = spec.family === :nb2_log ? _BRM_LAP_NB2 : _BRM_LAP_NORMAL
        return f, spec.family === :nb2_log ? :log : :identity,
            ((spec.predictor, :identity), scale)
    end
    nothing
end

_brm_slap_responses(plan::_RKStructuralPlan) = plan.responses
_brm_slap_responses(plan::_RKValuePlan) = plan.regression.responses
_brm_slap_responses(plan) = nothing
_brm_slap_predictors(plan::_RKStructuralPlan) = plan.predictors
_brm_slap_predictors(plan::_RKValuePlan) = plan.regression.predictors

# Why a model has no row structure, or `nothing` when it has one.
function _brm_slap_refusal(backend::RKBRMI, view::BRMNLMEView)
    _brm_nlme_ragged(view) && return "nested groupings give units ragged effect blocks"
    any(==(:subject_mixing), view.role) && return "a random-effect block is Student-t"
    responses = _brm_slap_responses(backend.plan)
    responses === nothing && return "the RK plan has no likelihood specs"
    for spec in responses
        _brm_slap_family(spec) === nothing &&
            return "response `$(spec.response)` has family `$(spec.family)`"
        spec.weights === nothing || return "response `$(spec.response)` is weighted"
        spec.evidence.kind === :none ||
            return "response `$(spec.response)` is $(spec.evidence.kind)"
        spec.mi_jobs === nothing || return "response `$(spec.response)` has missing values"
        spec.link === _brm_slap_family(spec)[2] ||
            return "response `$(spec.response)` has link `$(spec.link)`"
        if spec.scale_predictor !== nothing
            predictor = findfirst(q -> q.name === spec.scale_predictor,
                _brm_slap_predictors(backend.plan))
            predictor === nothing && return "response `$(spec.response)`'s scale " *
                "predictor `$(spec.scale_predictor)` has no predictor spec"
            _brm_slap_predictors(backend.plan)[predictor].link === :log ||
                return "response `$(spec.response)`'s scale predictor is not on the log scale"
        end
    end
    nothing
end

function _brm_structured_laplace_problem(backend::RKBRMI, view::BRMNLMEView; ad_backend,
        tol, maxiter)
    responses = _brm_slap_responses(backend.plan)
    population = findall(!=(:subject_effect), view.role)
    blocks = view.subject_coordinates
    d = length(first(blocks))
    all(b -> length(b) == d, blocks) || _brm_nlme_error(
        "brm_laplace_problem: units have effect blocks of different sizes")
    omega = findall(r -> r === :subject_scale || r === :subject_correlation, view.role)
    records = [r for r in _rk_coordinate_records(backend.plan) if r.kind === :ranef]
    effects = Tuple(Symbol(chopsuffix(String(r.z), ".z")) for r in records)
    families = map(_brm_slap_family, responses)
    ports = unique(Symbol[name for (_, _, slots) in families for (name, _) in slots
        if name isa Symbol])
    data = rk_bound_data(backend)
    u0 = _brm_nlme_unit_point(backend, view; shared_point=nothing)
    kernels = _rk_slap_kernels(backend, Tuple(ports), effects, omega, u0, ad_backend)
    L = sum(length, view.rows)
    row_unit = zeros(Int, L)
    for (i, rows) in enumerate(view.rows), r in rows
        row_unit[r] = i
    end
    # Terms, unit by unit; each response contributes one per row.
    family = Int8[]; ys = Float64[]; trials = Float64[]; entry = Int[]
    port = Int[]; index = Int[]; link = Symbol[]; fixed = Float64[]
    unit_terms = UnitRange{Int}[]; term_response = Symbol[]; term_row = Int[]
    values0 = _rk_slap_values(kernels, u0)
    for (i, rows) in enumerate(view.rows)
        first_term = length(family) + 1
        for r in rows, (spec, (f, _, slots)) in zip(responses, families)
            push!(term_response, spec.response)
            push!(term_row, r)
            push!(family, f)
            push!(ys, Float64(data[spec.response][r]))
            n = spec.trials === nothing ? 1.0 :
                spec.trials isa Symbol ? Float64(data[spec.trials][r]) : Float64(spec.trials)
            push!(trials, n)
            push!(entry, length(port) + 1)
            for (name, l) in slots
                if name isa Symbol
                    k = findfirst(==(name), ports)
                    v = values0[k]
                    push!(port, k)
                    push!(index, v isa Number ? 0 : r)
                    push!(fixed, NaN)
                else
                    push!(port, 0)
                    push!(index, 0)
                    push!(fixed, Float64(name))
                end
                push!(link, l)
            end
        end
        push!(unit_terms, first_term:length(family))
    end
    length(row_unit) == L && all(!iszero, row_unit) || _brm_nlme_error(
        "brm_laplace_problem: internal: a row belongs to no unit")
    constant = map(_brm_lap_constant, family, ys, trials)
    p = BRMStructuredLaplaceProblem(backend, view, population, view.coordinates[population],
        blocks, d, kernels, family, ys, trials, constant, entry, fill(true, length(family)),
        unit_terms, port, index,
        link, fixed, zeros(d, length(port)), Int[], Int[], Float64[], omega, false,
        u0[population],
        zeros(length(port)), Int[], Int[], Float64[], copy(u0), Ref{Any}(nothing),
        Float64(tol), Int(maxiter), 0, 0)
    _brm_slap_design!(p, u0)
    _brm_slap_population_design!(p, u0)
    _brm_slap_verify!(p, u0, term_response, term_row)
    p
end

# J = ∂A/∂φ by unit steps from u0 (exact for an affine A), kept when A is
# affine at two other population points; the random-effect scales and
# correlations move only M, so their columns must be zero.
function _brm_slap_population_design!(p::BRMStructuredLaplaceProblem, u0)
    nψ = length(p.port)
    base = copy(u0)
    for block in p.blocks
        base[block] .= 0.0
    end
    A0 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, base))
    columns = Vector{Vector{Float64}}(undef, length(p.population))
    ψ = zeros(nψ)
    for (j, k) in enumerate(p.population)
        u = copy(base)
        u[k] += 1.0
        _brm_slap_psi!(ψ, p, _rk_slap_values(p.kernels, u))
        columns[j] = ψ .- A0
    end
    jptr = [1]; jcol = Int[]; jval = Float64[]
    for e in 1:nψ
        for j in eachindex(columns)
            v = columns[j][e]
            v == 0.0 && continue
            push!(jcol, j); push!(jval, v)
        end
        push!(jptr, length(jcol) + 1)
    end
    p.φ0 = base[p.population]
    p.A0 = A0
    p.jptr, p.jcol, p.jval = jptr, jcol, jval
    rng = Random.Xoshiro(20261028)
    affine = all(isfinite, A0) && all(all(isfinite, c) for c in columns)
    for _ in 1:2
        affine || break
        u = copy(base)
        u[p.population] .+= 0.3 .* randn(rng, length(p.population))
        exact = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, u))
        predicted = _brm_slap_affine_A(p, u[p.population])
        affine = all(isapprox.(predicted, exact; rtol=1e-9, atol=1e-9))
    end
    p.affine = affine
    p
end

function _brm_slap_affine_A(p::BRMStructuredLaplaceProblem, φ)
    Δ = φ .- p.φ0
    A = copy(p.A0)
    @inbounds for e in eachindex(A)
        for k in p.jptr[e]:(p.jptr[e + 1] - 1)
            A[e] += p.jval[k] * Δ[p.jcol[k]]
        end
    end
    A
end

# Jᵀ w over the population coordinates.
function _brm_slap_affine_pullback(p::BRMStructuredLaplaceProblem, w)
    g = zeros(length(p.population))
    @inbounds for e in eachindex(w)
        we = w[e]
        we == 0.0 && continue
        for k in p.jptr[e]:(p.jptr[e + 1] - 1)
            g[p.jcol[k]] += p.jval[k] * we
        end
    end
    g
end

# Every term's log density at two points, against RK's own pointwise
# densities: the families, their parametrization and constants are the
# model's, or the row structure does not apply.
function _brm_slap_verify!(p::BRMStructuredLaplaceProblem, u0, response, row)
    rng = Random.Xoshiro(20261027)
    for u in (u0, u0 .+ 0.1 .* randn(rng, length(u0)))
        ψ = _brm_slap_psi!(zeros(length(p.port)), p, _rk_slap_values(p.kernels, u))
        pointwise = _rk_slap_pointwise(p.kernels, u)
        for t in eachindex(p.family)
            e = p.entry[t]
            b = _brm_lap_width(p.family[t]) == 2 ? ψ[e + 1] : 0.0
            mine = _brm_lap_row(p.family[t], p.y[t], p.trials[t], ψ[e], b, false).v +
                p.constant[t]
            theirs = pointwise[response[t]][row[t]]
            isapprox(mine, theirs; rtol=1e-9, atol=1e-9) || _brm_nlme_error(
                "brm_laplace_problem: response `$(response[t])` row $(row[t]) has log " *
                "density $theirs in the model and $mine in the row-structured Laplace, " *
                "which therefore does not apply")
        end
    end
    p
end

# The link-scale parameters of every entry at a packed point.
function _brm_slap_psi!(ψ, p::BRMStructuredLaplaceProblem, values)
    for e in eachindex(ψ)
        k = p.port[e]
        θ = k == 0 ? p.fixed[e] : p.index[e] == 0 ? values[k] : values[k][p.index[e]]
        ψ[e] = _brm_lap_link(p.link[e], θ)
    end
    ψ
end

# Unit l's draws vector b_l, concatenated over the effect blocks, from the
# effects readout (one levels × K matrix per block).
function _brm_slap_bvector(effects, l)
    b = Float64[]
    for value in effects
        value isa AbstractVector ? push!(b, value[l]) : append!(b, view(value, l, :))
    end
    b
end

# M (b = M z per unit) and, per `omega` coordinate, ∂M: unit l carries e_l.
function _brm_slap_effect_maps(p::BRMStructuredLaplaceProblem, u)
    usyn = copy(u)
    for block in p.blocks
        usyn[block] .= 0.0
    end
    length(p.blocks) >= p.d || _brm_nlme_error(
        "brm_laplace_problem: fewer units than effects per unit")
    for l in 1:p.d
        usyn[p.blocks[l][l]] = 1.0
    end
    value, tangents = _rk_slap_effects(p.kernels, usyn)
    columns(draws) = [_brm_slap_bvector(draws, l) for l in 1:p.d]
    b = columns(value)
    all(c -> length(c) == p.d, b) || _brm_nlme_error(
        "brm_laplace_problem: internal: the effect blocks hold $(length(first(b))) draws " *
        "per unit, the view $(p.d)")
    M = [b[l][k] for k in 1:p.d, l in 1:p.d]
    dM = map(tangents) do t
        bt = columns(t)
        [bt[l][k] for k in 1:p.d, l in 1:p.d]
    end
    M, dM
end

# Recover Z from the affine map z ↦ ψ at u0, and verify the structure at a
# second point: every entry must be A(φ) + Z M(φ) z with Z constant.
function _brm_slap_design!(p::BRMStructuredLaplaceProblem, u0)
    nψ = length(p.port)
    ψ0 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, u0))
    M, _ = _brm_slap_effect_maps(p, u0)
    C = zeros(p.d, nψ)
    ψk = zeros(nψ)
    for k in 1:p.d
        uk = copy(u0)
        for block in p.blocks
            uk[block[k]] += 1.0
        end
        _brm_slap_psi!(ψk, p, _rk_slap_values(p.kernels, uk))
        C[k, :] .= ψk .- ψ0
    end
    # C_e = Mᵀ Z_e for every entry.
    p.Z .= M' \ C
    for t in eachindex(p.family)
        e = p.entry[t]
        p.moving[t] = _brm_lap_width(p.family[t]) == 2 && any(!iszero, view(p.Z, :, e + 1))
    end
    p.zptr = [1]; p.zrow = Int[]; p.zval = Float64[]
    for e in axes(p.Z, 2)
        for a in axes(p.Z, 1)
            p.Z[a, e] == 0.0 && continue
            push!(p.zrow, a)
            push!(p.zval, p.Z[a, e])
        end
        push!(p.zptr, length(p.zrow) + 1)
    end
    # Check at another population point and other draws, unit by unit.
    rng = Random.Xoshiro(20261026)
    u1 = copy(u0)
    u1[p.population] .+= 0.1 .* randn(rng, length(p.population))
    for block in p.blocks
        u1[block] .= randn(rng, length(block))
    end
    uA = copy(u1)
    for block in p.blocks
        uA[block] .= 0.0
    end
    A1 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, uA))
    ψ1 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, u1))
    M1, _ = _brm_slap_effect_maps(p, u1)
    for (i, terms) in enumerate(p.unit_terms)
        b = M1 * u1[p.blocks[i]]
        for t in terms, e in p.entry[t]:(p.entry[t] + _brm_lap_width(p.family[t]) - 1)
            predicted = A1[e] + dot(view(p.Z, :, e), b)
            isapprox(predicted, ψ1[e]; rtol=1e-8, atol=1e-8) || _brm_nlme_error(
                "brm_laplace_problem: the model's rows are not affine in their unit's " *
                "random effects (unit $i, term $t: $(ψ1[e]) vs $predicted), so the " *
                "row-structured Laplace does not apply")
        end
    end
    p
end

# Σ_k f(k) over 1:d as a plain loop: Base's `sum` over a range goes through
# `mapreduce`, which costs more than the handful of products it adds here.
@inline function _brm_slap_sum(f::F, d::Int) where {F}
    s = 0.0
    @inbounds for k in 1:d
        s += f(k)
    end
    s
end

# Z_eᵀ b, x += α Z_e, X += α Z_e1 Z_e2ᵀ and Z_e1ᵀ Q Z_e2 over Z's nonzeros.
@inline function _brm_slap_zdot(p::BRMStructuredLaplaceProblem, e, b)
    s = 0.0
    @inbounds for k in p.zptr[e]:(p.zptr[e + 1] - 1)
        s += p.zval[k] * b[p.zrow[k]]
    end
    s
end
@inline function _brm_slap_zadd!(x, p::BRMStructuredLaplaceProblem, e, α)
    @inbounds for k in p.zptr[e]:(p.zptr[e + 1] - 1)
        x[p.zrow[k]] += α * p.zval[k]
    end
    x
end
@inline function _brm_slap_zouter!(X, p::BRMStructuredLaplaceProblem, e1, e2, α)
    @inbounds for k1 in p.zptr[e1]:(p.zptr[e1 + 1] - 1)
        x1 = α * p.zval[k1]
        for k2 in p.zptr[e2]:(p.zptr[e2 + 1] - 1)
            X[p.zrow[k1], p.zrow[k2]] += x1 * p.zval[k2]
        end
    end
    X
end
@inline function _brm_slap_zquad(p::BRMStructuredLaplaceProblem, e1, e2, Q)
    s = 0.0
    @inbounds for k1 in p.zptr[e1]:(p.zptr[e1 + 1] - 1), k2 in p.zptr[e2]:(p.zptr[e2 + 1] - 1)
        s += p.zval[k1] * Q[p.zrow[k1], p.zrow[k2]] * p.zval[k2]
    end
    s
end

# One unit's buffers, reused across units: the mode search's (g, H, their
# trial copies, the step, the trial point, the mode, a factor) and the draws
# space's (b = M z, the likelihood's gradient gb and Hessian Hb in b, Hb M).
struct _BRMSlapWork
    g::Vector{Float64}
    H::Matrix{Float64}
    gt::Vector{Float64}
    Ht::Matrix{Float64}
    step::Vector{Float64}
    trial::Vector{Float64}
    z::Vector{Float64}
    L::Matrix{Float64}
    b::Vector{Float64}
    gb::Vector{Float64}
    Hb::Matrix{Float64}
    T::Matrix{Float64}
end
_BRMSlapWork(d) = _BRMSlapWork(zeros(d), zeros(d, d), zeros(d), zeros(d, d), zeros(d),
    zeros(d), zeros(d), zeros(d, d), zeros(d), zeros(d), zeros(d, d), zeros(d, d))

# Unit i's density f_i(z) = Σ_t ℓ_t − |z|²/2 (less its terms in φ alone, see
# `_brm_lap_row`), gradient g and Hessian H. Each term reads b = M z through
# its few Z entries; the likelihood's gradient and Hessian accumulate in b
# (gb, Hb) and map back once: g = Mᵀ gb − z, H = Mᵀ Hb M − I.
function _brm_slap_unit!(g, H, w::_BRMSlapWork, p::BRMStructuredLaplaceProblem, A, M, i, z)
    d = p.d
    b, gb, Hb, T = w.b, w.gb, w.Hb, w.T
    f = 0.0
    @inbounds for a in 1:d
        f -= 0.5 * z[a]^2
        b[a] = _brm_slap_sum(k -> M[a, k] * z[k], d)
        gb[a] = 0.0
        for c in 1:d
            Hb[a, c] = 0.0
        end
    end
    @inbounds for t in p.unit_terms[i]
        e = p.entry[t]
        moving = p.moving[t]
        two = _brm_lap_width(p.family[t]) == 2
        a = A[e] + _brm_slap_zdot(p, e, b)
        bb = two ? A[e + 1] + (moving ? _brm_slap_zdot(p, e + 1, b) : 0.0) : 0.0
        r = _brm_lap_row(p.family[t], p.y[t], p.trials[t], a, bb, false, moving)
        f += r.v
        _brm_slap_zadd!(gb, p, e, r.g1)
        _brm_slap_zouter!(Hb, p, e, e, r.h11)
        if moving
            _brm_slap_zadd!(gb, p, e + 1, r.g2)
            _brm_slap_zouter!(Hb, p, e, e + 1, r.h12)
            _brm_slap_zouter!(Hb, p, e + 1, e, r.h12)
            _brm_slap_zouter!(Hb, p, e + 1, e + 1, r.h22)
        end
    end
    @inbounds for k in 1:d
        g[k] = _brm_slap_sum(a -> M[a, k] * gb[a], d) - z[k]
        for l in 1:d
            T[k, l] = _brm_slap_sum(c -> Hb[k, c] * M[c, l], d)
        end
    end
    @inbounds for k in 1:d, l in 1:d
        H[k, l] = _brm_slap_sum(a -> M[a, k] * T[a, l], d) - (k == l)
    end
    f
end

# In-place lower Cholesky factor of a small d × d matrix (its lower triangle);
# false unless the matrix is positive definite. Units' blocks are a handful of
# effects, where LAPACK's call overhead and allocations would dominate.
function _brm_slap_chol!(A::AbstractMatrix{Float64})
    d = size(A, 1)
    @inbounds for j in 1:d
        s = A[j, j]
        for k in 1:(j - 1)
            s -= A[j, k]^2
        end
        s > 0 || return false
        l = sqrt(s)
        A[j, j] = l
        for i in (j + 1):d
            r = A[i, j]
            for k in 1:(j - 1)
                r -= A[i, k] * A[j, k]
            end
            A[i, j] = r / l
        end
    end
    true
end

# x = (L Lᵀ)⁻¹ b, L the lower factor from `_brm_slap_chol!`.
function _brm_slap_cholsolve!(x, L, b)
    d = length(b)
    @inbounds for i in 1:d
        r = b[i]
        for k in 1:(i - 1)
            r -= L[i, k] * x[k]
        end
        x[i] = r / L[i, i]
    end
    @inbounds for i in d:-1:1
        r = x[i]
        for k in (i + 1):d
            r -= L[k, i] * x[k]
        end
        x[i] = r / L[i, i]
    end
    x
end

# The Newton step (−H)⁻¹ g, or where −H is not positive definite the step along
# −H with reflected, floored eigenvalues (`_brm_laplace_step`).
function _brm_slap_step!(w::_BRMSlapWork, H, g)
    w.L .= .-H
    _brm_slap_chol!(w.L) ? _brm_slap_cholsolve!(w.step, w.L, g) :
        copyto!(w.step, _brm_laplace_step(H, g))
end

# Unit i's mode by Newton's method from z0 into `w.z` (its Hessian there in
# `w.H`): its own iterations, backtracking, stop (Newton decrement at most
# `tol`, or at its rounding floor) and restart. Returns (status, steps).
function _brm_slap_unit_mode!(w::_BRMSlapWork, p::BRMStructuredLaplaceProblem, A, M, i, z0)
    z, g, H, gt, Ht = w.z, w.g, w.H, w.gt, w.Ht
    copyto!(z, z0)
    f = _brm_slap_unit!(g, H, w, p, A, M, i, z)
    if !(isfinite(f) && all(isfinite, g))
        fill!(z, 0.0)
        f = _brm_slap_unit!(g, H, w, p, A, M, i, z)
        isfinite(f) && all(isfinite, g) || return :nonfinite, 0
    end
    previous = Inf
    for iteration in 1:p.maxiter
        all(isfinite, H) || return :nonfinite, iteration - 1
        step = _brm_slap_step!(w, H, g)
        λ = dot(g, step)
        (λ <= p.tol || (λ <= 1e-8 && λ >= previous / 4)) && return :converged, iteration - 1
        previous = λ
        t = 1.0
        accepted = false
        for _ in 1:60
            w.trial .= z .+ t .* step
            ft = _brm_slap_unit!(gt, Ht, w, p, A, M, i, w.trial)
            if isfinite(ft) && all(isfinite, gt) && (λ <= 1e-8 || ft >= f + 1e-4 * t * λ)
                copyto!(z, w.trial)
                copyto!(g, gt)
                copyto!(H, Ht)
                f = ft
                accepted = true
                break
            end
            t /= 2
        end
        accepted || return :nonfinite, iteration
    end
    :unconverged, p.maxiter
end

# Every unit's mode at population point φ, from the previous modes, and the
# lower Cholesky factor of each unit's −H there (`L[:, :, i]`).
function _brm_slap_mode!(p::BRMStructuredLaplaceProblem, φ::AbstractVector)
    length(φ) == length(p.population) || throw(DimensionMismatch(
        "the population point has length $(length(φ)), the problem has " *
        "$(length(p.population)) population coordinates"))
    uA = copy(p.u)
    uA[p.population] .= φ
    for block in p.blocks
        uA[block] .= 0.0
    end
    values = p.affine ? nothing : _rk_slap_values(p.kernels, uA)
    A = p.affine ? _brm_slap_affine_A(p, φ) :
        _brm_slap_psi!(zeros(length(p.port)), p, values)
    M, dM = _brm_slap_effect_maps(p, uA)
    if !(all(isfinite, A) && all(isfinite, M))
        p.nonfinite += 1
        return (; finite=false, value=-Inf, iterations=0)
    end
    u = copy(uA)
    L = zeros(p.d, p.d, length(p.blocks))
    w = _BRMSlapWork(p.d)
    iterations = 0
    for (i, block) in enumerate(p.blocks)
        status, steps = _brm_slap_unit_mode!(w, p, A, M, i, view(p.u, block))
        if status !== :converged
            status === :nonfinite ? (p.nonfinite += 1) : (p.unconverged += 1)
            status === :unconverged && @warn("brm_laplace_problem: unit " *
                "$(repr(p.view.levels[i]))'s mode did not converge in $(p.maxiter) " *
                "Newton iterations at population point $(repr(collect(φ))); its log " *
                "density is -Inf", maxlog=3)
            return (; finite=false, value=-Inf, iterations=steps)
        end
        u[block] .= w.z
        Li = view(L, :, :, i)
        Li .= .-w.H
        _brm_slap_chol!(Li) || _brm_nlme_error(
            "brm_laplace_problem: a unit's density has a stationary point that is not " *
            "a maximum at this population point")
        iterations = max(iterations, steps)
    end
    p.u = u
    p.factors[] = L
    (; finite=true, u, uA, values, A, M, dM, L, iterations)
end

# log det(L Lᵀ) for a lower factor.
_brm_slap_logdet(L) = 2 * sum(k -> log(L[k, k]), axes(L, 1))

# The mode search's unit densities leave out terms that do not move with the
# effects; the value sums every term in full.
function _brm_slap_value(p::BRMStructuredLaplaceProblem, state)
    state.finite || return -Inf
    v = sum(p.constant)
    for (i, block) in enumerate(p.blocks)
        z = view(state.u, block)
        v += _brm_slap_unit_value(p, state.A, state.M, i, z) +
            p.d / 2 * log(2π) - _brm_slap_logdet(view(state.L, :, :, i)) / 2
    end
    v + _rk_slap_prior(p.kernels, state.uA)
end

function _brm_slap_unit_value(p::BRMStructuredLaplaceProblem, A, M, i, z)
    v = -0.5 * sum(abs2, z)
    b = M * z
    for t in p.unit_terms[i]
        e = p.entry[t]
        two = _brm_lap_width(p.family[t]) == 2
        a = A[e] + _brm_slap_zdot(p, e, b)
        bb = two ? A[e + 1] + _brm_slap_zdot(p, e + 1, b) : 0.0
        v += _brm_lap_row(p.family[t], p.y[t], p.trials[t], a, bb, true, p.moving[t]).v
    end
    v
end

# Log density and gradient over the population coordinates (see the header),
# in each unit's draws space b = M z: with Q = M h⁻¹ Mᵀ, (C h⁻¹ Cᵀ)_pq =
# Z_pᵀ Q Z_q, C v = Zᵀ (M v), and Γ's sum over a unit's terms is
# Hb (M h⁻¹) + gb vᵀ + wb ẑᵀ, with Hb, gb, wb the terms' ℓ″, ℓ′ and w in b.
function _brm_slap_value_and_gradient(p::BRMStructuredLaplaceProblem, state)
    d = p.d
    A, M, u = state.A, state.M, state.u
    w = zeros(length(p.port))
    Γ = zeros(d, d)
    value = sum(p.constant)
    rows = _BRMLapRow[]
    taus = Float64[]
    hinv = zeros(d, d); unit = zeros(d); T = zeros(d, d); Q = zeros(d, d)
    b = zeros(d); sb = zeros(d); v = zeros(d); Mv = zeros(d)
    gb = zeros(d); wb = zeros(d); Hb = zeros(d, d)
    for (i, block) in enumerate(p.blocks)
        Li = view(state.L, :, :, i)
        for k in 1:d
            fill!(unit, 0.0)
            unit[k] = 1.0
            _brm_slap_cholsolve!(view(hinv, :, k), Li, unit)
        end
        z = view(u, block)
        value += d / 2 * log(2π) - _brm_slap_logdet(Li) / 2 - 0.5 * sum(abs2, z)
        @inbounds for a in 1:d
            b[a] = _brm_slap_sum(k -> M[a, k] * z[k], d)
            for l in 1:d
                T[a, l] = _brm_slap_sum(k -> M[a, k] * hinv[k, l], d)       # M h⁻¹
            end
        end
        @inbounds for a in 1:d, c in 1:d
            Q[a, c] = _brm_slap_sum(l -> T[a, l] * M[c, l], d)               # M h⁻¹ Mᵀ
        end
        fill!(sb, 0.0); fill!(gb, 0.0); fill!(wb, 0.0); fill!(Hb, 0.0)
        empty!(rows); empty!(taus)
        terms = p.unit_terms[i]
        @inbounds for t in terms
            e = p.entry[t]
            moving = p.moving[t]
            two = _brm_lap_width(p.family[t]) == 2
            a = A[e] + _brm_slap_zdot(p, e, b)
            bb = two ? A[e + 1] + (moving ? _brm_slap_zdot(p, e + 1, b) : 0.0) : 0.0
            r = _brm_lap_row(p.family[t], p.y[t], p.trials[t], a, bb, true, moving)
            push!(rows, r)
            value += r.v
            m11 = _brm_slap_zquad(p, e, e, Q)
            m12 = moving ? _brm_slap_zquad(p, e, e + 1, Q) : 0.0
            m22 = moving ? _brm_slap_zquad(p, e + 1, e + 1, Q) : 0.0
            τ1 = m11 * r.t111 + 2m12 * r.t112 + m22 * r.t122
            τ2 = two ? m11 * r.t112 + 2m12 * r.t122 + m22 * r.t222 : 0.0
            push!(taus, τ1, τ2)
            _brm_slap_zadd!(sb, p, e, 0.5 * τ1)
            _brm_slap_zadd!(gb, p, e, r.g1)
            _brm_slap_zouter!(Hb, p, e, e, r.h11)
            if moving
                _brm_slap_zadd!(sb, p, e + 1, 0.5 * τ2)
                _brm_slap_zadd!(gb, p, e + 1, r.g2)
                _brm_slap_zouter!(Hb, p, e, e + 1, r.h12)
                _brm_slap_zouter!(Hb, p, e + 1, e, r.h12)
                _brm_slap_zouter!(Hb, p, e + 1, e + 1, r.h22)
            end
        end
        # s = Mᵀ sb, v = h⁻¹ s, and M v for the terms' C v.
        fill!(v, 0.0)
        @inbounds for l in 1:d
            sl = _brm_slap_sum(a -> M[a, l] * sb[a], d)
            for k in 1:d
                v[k] += hinv[k, l] * sl
            end
        end
        @inbounds for a in 1:d
            Mv[a] = _brm_slap_sum(k -> M[a, k] * v[k], d)
        end
        @inbounds for (j, t) in enumerate(terms)
            e = p.entry[t]
            moving = p.moving[t]
            r = rows[j]
            cv1 = _brm_slap_zdot(p, e, Mv)
            cv2 = moving ? _brm_slap_zdot(p, e + 1, Mv) : 0.0
            w1 = r.g1 + 0.5 * taus[2j - 1] + r.h11 * cv1 + r.h12 * cv2
            w[e] = w1
            _brm_slap_zadd!(wb, p, e, w1)
            if _brm_lap_width(p.family[t]) == 2
                w2 = r.g2 + 0.5 * taus[2j] + r.h12 * cv1 + r.h22 * cv2
                w[e + 1] = w2
                moving && _brm_slap_zadd!(wb, p, e + 1, w2)
            end
        end
        @inbounds for a in 1:d, l in 1:d
            Γ[a, l] += _brm_slap_sum(c -> Hb[a, c] * T[c, l], d) + gb[a] * v[l] + wb[a] * z[l]
        end
    end
    gradient = p.affine ? _brm_slap_affine_pullback(p, w) :
        _rk_slap_pullback(p.kernels, state.uA,
            _brm_slap_cotangents(p, state.values, w))[p.population]
    for (j, k) in enumerate(p.omega)
        gradient[findfirst(==(k), p.population)] += dot(Γ, state.dM[j])
    end
    prior, prior_gradient = _rk_slap_prior_gradient(p.kernels, state.uA)
    value + prior, gradient .+ prior_gradient[p.population]
end

# Per-port cotangents for the RK pullback from the entries' cotangents.
function _brm_slap_cotangents(p::BRMStructuredLaplaceProblem, values, w)
    cot = map(v -> v isa Number ? Ref(0.0) : zeros(length(v)), values)
    for e in eachindex(w)
        k = p.port[e]
        k == 0 && continue
        θ = p.index[e] == 0 ? values[k] : values[k][p.index[e]]
        x = w[e] * _brm_lap_dlink(p.link[e], θ)
        p.index[e] == 0 ? (cot[k][] += x) : (cot[k][p.index[e]] += x)
    end
    map(x -> x isa Ref ? x[] : x, cot)
end

function _rk_slap_prior end
function _rk_slap_prior_gradient end

LogDensityProblems.capabilities(::Type{<:BRMStructuredLaplaceProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::BRMStructuredLaplaceProblem) = length(p.population)
LogDensityProblems.logdensity(p::BRMStructuredLaplaceProblem, φ::AbstractVector) =
    _brm_slap_value(p, _brm_slap_mode!(p, φ))
function LogDensityProblems.logdensity_and_gradient(p::BRMStructuredLaplaceProblem,
        φ::AbstractVector)
    state = _brm_slap_mode!(p, φ)
    state.finite || return -Inf, fill(NaN, length(p.population))
    _brm_slap_value_and_gradient(p, state)
end

function brm_laplace_draws(p::BRMStructuredLaplaceProblem, P::AbstractMatrix;
        rng=Random.default_rng(), conditional::Integer=0)
    size(P, 1) == length(p.population) || throw(DimensionMismatch(
        "population draws have $(size(P, 1)) rows, the problem has " *
        "$(length(p.population)) population coordinates"))
    conditional >= 0 || throw(ArgumentError(
        "brm_laplace_draws: conditional = $conditional proposals; pass 0 for " *
        "Laplace draws or a positive count for importance resampling"))
    draws = Matrix{Float64}(undef, length(p.u), size(P, 2))
    newton_steps = Vector{Int}(undef, size(P, 2))
    importance_ess = Matrix{Float64}(undef, conditional > 0 ? length(p.blocks) : 0,
        size(P, 2))
    g = zeros(p.d); H = zeros(p.d, p.d); work = _BRMSlapWork(p.d)
    for k in axes(P, 2)
        state = _brm_slap_mode!(p, view(P, :, k))
        state.finite || _brm_nlme_error(
            "brm_laplace_draws: population draw $k has a nonfinite density")
        newton_steps[k] = state.iterations
        draws[:, k] .= state.u
        for (i, block) in enumerate(p.blocks)
            # −H = L Lᵀ, so ẑ + L⁻ᵀ ε has covariance (−H)⁻¹.
            U = LowerTriangular(state.L[:, :, i])'
            if conditional == 0
                draws[block, k] .+= U \ randn(rng, p.d)
                continue
            end
            # Importance resampling of the unit's exact conditional, its own
            # rows only: proposal ẑ + U⁻¹ε has log density −|ε|²/2 + const.
            ε = randn(rng, p.d, conditional)
            proposals = state.u[block] .+ U \ ε
            logw = [_brm_slap_unit!(g, H, work, p, state.A, state.M, i, view(proposals, :, j)) +
                sum(abs2, view(ε, :, j)) / 2 for j in 1:conditional]
            wts = exp.(logw .- maximum(logw))
            importance_ess[i, k] = sum(wts)^2 / sum(abs2, wts)
            pick = searchsortedfirst(cumsum(wts), rand(rng) * sum(wts))
            draws[block, k] .= view(proposals, :, min(pick, conditional))
        end
    end
    diagnostics = conditional > 0 ? (; newton_steps, importance_ess) : (; newton_steps)
    (; draws, diagnostics)
end
