# Adaptive partial centering for ordinary and R2D2-scaled random-effect blocks,
# ungrouped and grouped squared-exponential HSGP basis weights, ungrouped
# periodic HSGP cosine/sine weights, cdar walk innovations and areal `car`
# fields.
#
# This file owns only BRM/Stan emission semantics: which unconstrained
# coordinates are one block's effects, optional LKJ-Cholesky free values, and
# positive scales, plus the pure-Julia reconstruction of C.  WarmupHMC
# owns the online candidate accumulators and window lifecycle through the
# optional extension method of `adaptive_centering_problem`.

function _adaptive_log_sample_std(values)
    n = length(values)
    n >= 2 || error("HSGP centeredness selection needs at least two pilot draws")
    mean_value = sum(values) / n
    variance = sum(abs2(value - mean_value) for value in values) / (n - 1)
    variance > 0 ? 0.5log(variance) : -Inf
end

"""
    select_hsgp_centeredness(unit_weights, log_scales;
                             candidates=0:0.1:1,
                             underflow_log=log(floatmin(Float64)))

Choose one fixed partial-centering exponent per HSGP basis weight from a pilot
fit. Rows are pilot draws and columns are basis frequencies. `unit_weights`
contains the noncentered standard-normal coordinates and `log_scales` contains
the corresponding log spectral standard deviations.

For candidate `c`, the loss is

```
log(std(unit_weights .* exp.(c .* log_scales))) - mean(c .* log_scales)
```

evaluated with a per-column exponent shift. A candidate is inadmissible when
its centered coordinate scale would fall below `underflow_log`; importantly,
the `c=0` endpoint is evaluated without multiplying by `log_scales`, so it
remains valid even when a physical high-frequency scale is `-Inf`.

The result is a named tuple with `centeredness`, the candidate `losses`, an
`admissible` mask, and the normalized `candidates`. This is an offline
pilot-then-refit rule. It does not update geometry during warmup.
"""
function select_hsgp_centeredness(unit_weights::AbstractMatrix,
                                  log_scales::AbstractMatrix;
                                  candidates=0:0.1:1,
                                  underflow_log=log(floatmin(Float64)))
    size(unit_weights) == size(log_scales) || throw(DimensionMismatch(
        "unit_weights and log_scales must have the same draws × basis shape"))
    size(unit_weights, 1) >= 2 || error(
        "HSGP centeredness selection needs at least two pilot draws")
    all(isfinite, unit_weights) || error(
        "HSGP centeredness selection needs finite pilot unit weights")
    underflow_log isa Real && isfinite(underflow_log) || error(
        "underflow_log must be finite")
    cs = Float64[candidates...]
    !isempty(cs) || error("HSGP centeredness candidates cannot be empty")
    all(c -> isfinite(c) && 0 <= c <= 1, cs) || error(
        "HSGP centeredness candidates must lie in [0, 1]")

    n_draws, n_basis = size(unit_weights)
    losses = fill(Inf, length(cs), n_basis)
    admissible = falses(length(cs), n_basis)
    for b in 1:n_basis, (ci, c) in enumerate(cs)
        centered_log_scale = if iszero(c)
            zeros(Float64, n_draws)
        else
            c .* @view(log_scales[:, b])
        end
        all(isfinite, centered_log_scale) || continue
        minimum(centered_log_scale) >= underflow_log || continue
        offset = maximum(centered_log_scale)
        transformed = @view(unit_weights[:, b]) .* exp.(centered_log_scale .- offset)
        all(isfinite, transformed) || continue
        losses[ci, b] = _adaptive_log_sample_std(transformed) + offset -
                        sum(centered_log_scale) / n_draws
        admissible[ci, b] = isfinite(losses[ci, b])
    end
    selected = Vector{Float64}(undef, n_basis)
    selected_index = Vector{Int}(undef, n_basis)
    for b in 1:n_basis
        any(@view admissible[:, b]) || error(
            "no numerically admissible centeredness candidate for basis weight $b")
        selected_index[b] = argmin(@view losses[:, b])
        selected[b] = cs[selected_index[b]]
    end
    (; centeredness=selected, selected_index, candidates=cs, losses, admissible,
       underflow_log=Float64(underflow_log))
end

"""
    select_ranef_centeredness(unit_effects, log_scales;
                              candidates=0:0.1:1,
                              underflow_log=log(floatmin(Float64)))

Apply the scalar offline partial-centering criterion to ordinary random-effect
coordinates. Rows are pilot draws; columns are one scalar `(term, group)` cell.
`unit_effects` holds noncentered standardized effects and `log_scales` the
matching log standard deviations. The criterion and stable shifted-exponent
evaluation are identical to [`select_hsgp_centeredness`](@ref); only the
coordinate interpretation differs. The result is a fixed-partial specification
for a fresh fit and does not adapt during warmup.
"""
select_ranef_centeredness(unit_effects::AbstractMatrix, log_scales::AbstractMatrix;
                          candidates=0:0.1:1,
                          underflow_log=log(floatmin(Float64))) =
    select_hsgp_centeredness(unit_effects, log_scales;
                             candidates, underflow_log)

const _ADAPTIVE_CORRELATED_FAMILIES = Set((
    :ranef_correlated,
    :ranef_slope,
    :ranef_correlated_draws,
    :ranef_correlated_draws_effect,
    :ranef_correlated_draws_generic,
    :ranef_correlated_centered,
    :ranef_correlated_draws_centered,
    :ranef_correlated_draws_centered_effect,
    :ranef_correlated_draws_centered_generic,
    # The sampled active margins of a marginalized `|ID|` block; its generated
    # completion has no coordinates and stays outside every set.
    :ranef_correlated_draws_marginal,
    :ranef_correlated_draws_centered_marginal,
))

const _ADAPTIVE_INTERCEPT_FAMILIES = Set((
    :ranef_intercept,
    :ranef_intercept_centered,
    # Multi-membership intercepts sample `log_scale` exactly like an ordinary
    # `(1 | g)`; only the downstream gather differs, and it is linear in the
    # draws, so the scalar `u = s^c * z` path is exact here (snag
    # adaptive-centeri-953d87e0).
    :ranef_intercept_draws,
))

const _ADAPTIVE_STRATIFIED_FAMILIES = Set((
    :ranef_correlated_by,
    :ranef_correlated_by_draws,
    :ranef_correlated_by_draws_generic,
))

# R2D2 blocks DERIVE their marginal scales (`tau[j] = reference_scale[j] *
# sqrt(phi[j] * R2 / (1 - R2))`) instead of sampling them, so the compiled
# model has no unconstrained `tau`/`log_scale` coordinates. They become
# `R2D2AdaptiveCenteringBlock`s whose scales are compiled from the emitted
# assignment (`_adaptive_r2d2_block`), never silently left fixed (snag
# adaptive-centeri-953d87e0).
const _ADAPTIVE_R2D2_FAMILIES = Set((
    :ranef_intercept_r2d2,
    :ranef_correlated_r2d2,
    :ranef_correlated_draws_r2d2,
))

"""
    AbstractAdaptiveCenteringBlock

Supertype of the random-effect blocks [`adaptive_centering_blocks`](@ref)
returns: [`AdaptiveCenteringBlock`](@ref) for a block whose marginal scales are
sampled coordinates, [`R2D2AdaptiveCenteringBlock`](@ref) for one whose scales
an R2D2 prior derives. Both carry `ranef`, `target_c`, `effects` and
`cholesky_free` with the same meaning, and
`_adaptive_block_cholesky(x, block)` rebuilds `diag(tau) * L` for either.
"""
abstract type AbstractAdaptiveCenteringBlock end

"""
    AdaptiveCenteringBlock

The unconstrained-coordinate description needed to adapt one ordinary
random-effect block without parsing generated Stan.

- `ranef` is BRM's emitted [`RanefBlock`](@ref).
- `target_c` is the parametrization of the compiled model: `0.0` for BRM's
  standardised `z` emission and `1.0` for a `centered_groups=` emission.
- `effects[k,g]` indexes term `k`, group `g` in the compiled model's
  unconstrained vector.
- `cholesky_free` indexes Stan's `K(K-1)/2` unconstrained values for a
  correlated block's `cholesky_factor_corr[K] L`; it is empty at `K=1`.
- `log_scales` indexes the unconstrained values underlying the positive scale:
  `tau` for a correlated emission, `log_scale` for the `(1 | g)` fast path, or
  one stratum's column of the sampled `tau_s_tau` for a stratum of a
  `gr(g, by=b)` block.

Construct with [`adaptive_centering_blocks`](@ref).  Indices are resolved by
name against the caller-supplied unconstrained-name vector; no ordering of the
whole parameter vector is assumed.
"""
struct AdaptiveCenteringBlock <: AbstractAdaptiveCenteringBlock
    ranef::RanefBlock
    target_c::Float64
    effects::Matrix{Int}
    cholesky_free::Vector{Int}
    log_scales::Vector{Int}
end

Base.show(io::IO, b::AdaptiveCenteringBlock) = print(
    io,
    "AdaptiveCenteringBlock(", b.ranef.binding,
    ", target_c=", b.target_c,
    ", ", b.ranef.n_terms, "×", b.ranef.n_groups, ")",
)

# ---- R2D2: derived marginal scales -----------------------------------------
#
# An R2D2 block's `tau[k]` is not a coordinate: the emitted model assigns it
# from other parameters (`ref * sqrt(phi[j] * R2 / (1 - R2))` for the
# `sd(...) ~ r2d2(...)` grammar, `sqrt((1 - R2) * tau_bsv^2)` for the
# whole-predictor form, a free half-normal for a margin a partial ICC statement
# leaves unaddressed). Every such assignment is a product of powers of positive
# quantities, so its LOG is linear in a handful of log-atoms of unconstrained
# coordinates. `_AdaptiveLogScale` is that normal form, compiled once from the
# emitted assignment and evaluated on every gradient call:
#
#     log tau = constant
#             + sum(w * x[i])                                  # log of a lower=0 parameter
#             + sum(a * log(inv_logit(x[i])) + b * log1m(inv_logit(x[i])))  # unit-interval parameter
#             + sum(w * log(simplex(x[coords])[j]))            # Dirichlet share
#
# `_AdaptiveLogScale` is the inspectable form. The generic Enzyme path reads
# the block's `_AdaptiveLogScaleTable` instead: the same atoms, flattened into
# term-indexed arrays, so a differentiated row loop loads only numbers. On Julia
# 1.10, reading a term's `_AdaptiveLogScale` (a struct holding vectors) inside
# that loop is a use-after-free in Enzyme's reverse pass once another thread
# runs GC; flat reads of the same data are clean (see `_block_location` in
# BayesianRegressionModelsWarmupHMCExt). No recursion, closures or per-call
# dispatch either.

struct _AdaptiveSimplexLog
    weight::Float64
    # The simplex's `n - 1` unconstrained coordinates, in Stan order.
    coordinates::Vector{Int}
    entry::Int
end

struct _AdaptiveLogScale
    constant::Float64
    linear::Vector{Tuple{Float64,Int}}
    unit::Vector{Tuple{Float64,Float64,Int}}
    simplex::Vector{_AdaptiveSimplexLog}
end

# A block's `_AdaptiveLogScale`s, term `k` owning `linear[k]:linear[k+1]-1` of
# the linear atoms, likewise `unit` and `simplex`; simplex atom `q` owns
# `simplex_coordinates[q]:simplex_coordinates[q+1]-1` of `simplex_coordinate`.
# Atoms keep their order within each term.
struct _AdaptiveLogScaleTable
    constant::Vector{Float64}
    linear::Vector{Int}
    linear_weight::Vector{Float64}
    linear_index::Vector{Int}
    unit::Vector{Int}
    unit_log::Vector{Float64}
    unit_log1m::Vector{Float64}
    unit_index::Vector{Int}
    simplex::Vector{Int}
    simplex_weight::Vector{Float64}
    simplex_entry::Vector{Int}
    simplex_coordinates::Vector{Int}
    simplex_coordinate::Vector{Int}
end

function _AdaptiveLogScaleTable(scales::AbstractVector{_AdaptiveLogScale})
    t = _AdaptiveLogScaleTable(Float64[], [1], Float64[], Int[], [1], Float64[],
        Float64[], Int[], [1], Float64[], Int[], [1], Int[])
    for s in scales
        push!(t.constant, s.constant)
        for (w, i) in s.linear
            push!(t.linear_weight, w); push!(t.linear_index, i)
        end
        push!(t.linear, length(t.linear_index) + 1)
        for (a, b, i) in s.unit
            push!(t.unit_log, a); push!(t.unit_log1m, b); push!(t.unit_index, i)
        end
        push!(t.unit, length(t.unit_index) + 1)
        for atom in s.simplex
            push!(t.simplex_weight, atom.weight); push!(t.simplex_entry, atom.entry)
            append!(t.simplex_coordinate, atom.coordinates)
            push!(t.simplex_coordinates, length(t.simplex_coordinate) + 1)
        end
        push!(t.simplex, length(t.simplex_entry) + 1)
    end
    t
end

"""
    R2D2AdaptiveCenteringBlock

A random-effect block whose marginal scales are not raw log-scale coordinates,
described for adaptive partial centering: an R2D2-scaled block
(`ranef_intercept_r2d2`, `ranef_correlated_r2d2`,
`ranef_correlated_draws_r2d2`), or a known-value block that holds some scales
or its correlation at known values (`sd(...) = value`, `cor(:, ID) = R`).
`ranef`, `target_c`, `effects` and `cholesky_free` mean exactly what they mean
on [`AdaptiveCenteringBlock`](@ref). There is no `log_scales` coordinate
vector: `scales[k]` is the compiled log-scale of term `k` as a function of the
unconstrained vector — the model's own `R2`, Dirichlet share and
reference/total scale coordinates, a known block's sampled scale coordinate, or
a known scale's constant. Those coordinates are read, never transformed. A
known correlation is the constant Cholesky factor `cholesky_known` (empty when
the block samples `L`), and `cholesky_free` is then empty.

A Student-t block (`ranef(:, ID) ~ TDist(nu)`) is described in this form too:
`mixing[g]` indexes level `g`'s mixing weight `w[g]` (its unconstrained value is
`log w[g]`), and level `g`'s factor is `sqrt(w[g]) * diag(tau) * L`. The weights
are read, never transformed. `mixing` is empty for a Gaussian block.

Construct with [`adaptive_centering_blocks`](@ref).
"""
struct R2D2AdaptiveCenteringBlock <: AbstractAdaptiveCenteringBlock
    ranef::RanefBlock
    target_c::Float64
    effects::Matrix{Int}
    cholesky_free::Vector{Int}
    scales::Vector{_AdaptiveLogScale}
    # `scales`, flattened for the differentiated accessors.
    table::_AdaptiveLogScaleTable
    # A known correlation's Cholesky factor, and the canonical partial
    # correlations the differentiated stick walk reads in place of
    # `tanh(x[cholesky_free])` (`_adaptive_cholesky_cpc`); both empty when the
    # block samples `L`.
    cholesky_known::Matrix{Float64}
    cholesky_cpc::Vector{Float64}
    # A Student-t block's per-level mixing-weight coordinates; empty otherwise.
    mixing::Vector{Int}
    function R2D2AdaptiveCenteringBlock(ranef, target_c, effects, cholesky_free,
                                        scales; cholesky_known=zeros(0, 0),
                                        mixing=Int[])
        isempty(cholesky_known) || isempty(cholesky_free) || error(
            "BRM adaptive centering: block `$(ranef.binding)` has both a known ",
            "correlation and Cholesky coordinates")
        isempty(mixing) || length(mixing) == ranef.n_groups || error(
            "BRM adaptive centering: Student-t block `$(ranef.binding)` has ",
            "$(ranef.n_groups) levels but $(length(mixing)) mixing weights")
        new(ranef, target_c, effects, cholesky_free, scales,
            _AdaptiveLogScaleTable(scales), cholesky_known,
            _adaptive_cholesky_cpc(cholesky_known), mixing)
    end
end

# An ordinary block in derived-scale form: term `k`'s log-scale is the single
# atom `1.0 * x[log_scales[k]]`, which every accessor evaluates bit-identically
# to `x[log_scales[k]]`. `ranef` and `mixing` turn a Gaussian frame into its
# Student-t block.
_adaptive_derived_block(b::AdaptiveCenteringBlock; ranef=b.ranef, mixing=Int[]) =
    R2D2AdaptiveCenteringBlock(
        ranef, b.target_c, b.effects, b.cholesky_free,
        [_AdaptiveLogScale(0.0, [(1.0, i)], Tuple{Float64,Float64,Int}[],
                           _AdaptiveSimplexLog[]) for i in b.log_scales];
        mixing)
_adaptive_derived_block(b::R2D2AdaptiveCenteringBlock; ranef=b.ranef, mixing=Int[]) =
    R2D2AdaptiveCenteringBlock(ranef, b.target_c, b.effects, b.cholesky_free,
                               b.scales; cholesky_known=b.cholesky_known, mixing)

# Level `g`'s log mixing scale, `log(sqrt(w[g])) = x[mixing[g]] / 2`, of a
# Student-t block; zero for a Gaussian block.
_adaptive_log_mixing(x::AbstractVector, ::AdaptiveCenteringBlock, g) = zero(eltype(x))
_adaptive_log_mixing(x::AbstractVector, block::R2D2AdaptiveCenteringBlock, g) =
    isempty(block.mixing) ? zero(eltype(x)) : x[block.mixing[g]] / 2

# Every unconstrained coordinate a derived-scale block reads without
# transforming it: its scales' inputs and a Student-t block's weights.
_adaptive_read_coordinates(block::R2D2AdaptiveCenteringBlock) =
    vcat(Int[], (_adaptive_scale_coordinates(s) for s in block.scales)...,
         block.mixing)

# The canonical partial correlations of a correlation Cholesky factor `L`, in
# Stan's `cholesky_factor_corr` coordinate order: row `i`'s `z[l] = L[i, l] /
# stick`, with the stick updated as the walk in the WarmupHMC extension does,
# so that walk rebuilds `L`.
function _adaptive_cholesky_cpc(L::AbstractMatrix)
    cpc = Float64[]
    for i in 2:size(L, 1)
        stick = 1.0
        for l in 1:i-1
            z = L[i, l] / stick
            push!(cpc, z)
            stick *= sqrt(1.0 - z * z)
        end
    end
    cpc
end

Base.show(io::IO, b::R2D2AdaptiveCenteringBlock) = print(
    io,
    "R2D2AdaptiveCenteringBlock(", b.ranef.binding,
    ", target_c=", b.target_c,
    ", ", b.ranef.n_terms, "×", b.ranef.n_groups, ")",
)

# `log1p(exp(y))` without overflow on either side.
_adaptive_softplus(y) = y > zero(y) ? y + log1p(exp(-y)) : log1p(exp(y))

# Entry `j` of the isometric log-ratio sum-to-zero vector `z` of `y` in Stan's
# (>= 2.37) `stan::math::simplex_constrain`, rebuilt with exactly the operations
# that transform's loop applies to `z[j]`: `0 + sum(w[N:-1:j])`, then
# `- w[j-1] * (j-1)`, with `w[i] = y[i] / sqrt(i * (i + 1))`. `y[i]` is
# `x[coordinate[offset + i]]`.
function _adaptive_simplex_ilr(x::AbstractVector, coordinate, offset, N, j)
    T = eltype(x)
    1 <= j <= N + 1 || throw(BoundsError(1:N+1, j))
    sum_w = zero(T)
    for i in N:-1:j
        sum_w += x[coordinate[offset + i]] * inv(sqrt(T(i * (i + 1))))
    end
    z = zero(T) + sum_w
    if j > 1
        i = j - 1
        z -= (x[coordinate[offset + i]] * inv(sqrt(T(i * (i + 1))))) * i
    end
    z
end

# `log(simplex_constrain(y)[entry])`: the softmax of the isometric log-ratio
# vector at `entry`. Each `z[j]` is recomputed instead of stored — O(N^2) over a
# handful of Dirichlet components, bit-identical to filling a vector: a heap
# work vector here is the same Julia 1.10 Enzyme use-after-free.
function _adaptive_log_simplex(x::AbstractVector, coordinate, offset, N, entry)
    T = eltype(x)
    N == 0 && return zero(T)
    m = _adaptive_simplex_ilr(x, coordinate, offset, N, 1)
    for j in 2:N+1
        m = max(m, _adaptive_simplex_ilr(x, coordinate, offset, N, j))
    end
    total = zero(T)
    for j in 1:N+1
        total += exp(_adaptive_simplex_ilr(x, coordinate, offset, N, j) - m)
    end
    _adaptive_simplex_ilr(x, coordinate, offset, N, entry) - (m + log(total))
end

# Term `k`'s log-scale, atom by atom in `_AdaptiveLogScale` order.
function _adaptive_log_scale(x::AbstractVector, t::_AdaptiveLogScaleTable, k::Int)
    acc = convert(eltype(x), t.constant[k])
    for q in t.linear[k]:t.linear[k + 1] - 1
        acc += t.linear_weight[q] * x[t.linear_index[q]]
    end
    for q in t.unit[k]:t.unit[k + 1] - 1
        # log(inv_logit(y)) = -softplus(-y); log(1 - inv_logit(y)) = -softplus(y)
        y = x[t.unit_index[q]]
        acc -= t.unit_log[q] * _adaptive_softplus(-y) +
            t.unit_log1m[q] * _adaptive_softplus(y)
    end
    for q in t.simplex[k]:t.simplex[k + 1] - 1
        first = t.simplex_coordinates[q]
        acc += t.simplex_weight[q] * _adaptive_log_simplex(x, t.simplex_coordinate,
            first - 1, t.simplex_coordinates[q + 1] - first, t.simplex_entry[q])
    end
    acc
end

# Every unconstrained coordinate a compiled scale reads.
function _adaptive_scale_coordinates(s::_AdaptiveLogScale)
    out = Int[]
    append!(out, (i for (_, i) in s.linear))
    append!(out, (i for (_, _, i) in s.unit))
    for atom in s.simplex
        append!(out, atom.coordinates)
    end
    out
end

# What the emitted model knows, gathered once per `adaptive_centering_blocks`
# call and only when an R2D2 block is present (every other model keeps its
# historical metadata path untouched). `parameters` are the descriptor's
# parameter outputs — their `type`/`constraints` are StanBlocks' declared Stan
# types, family-implied supports included — and `assignments` the emitted
# body's top-level `name = rhs` statements, where the derived `tau` lives.
function _adaptive_r2d2_context(model, pos)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    parameters = Dict{Symbol,BRMOutput}(
        o.name => o for o in descriptor.outputs if o.kind === :parameter)
    assignments = Dict{Symbol,Any}()
    ambiguous = Set{Symbol}()
    body = plan.model.model
    body isa Expr && body.head === :block || error(
        "BRM adaptive centering: the emitted model body is not a statement " *
        "block; R2D2 scales cannot be resolved.")
    for st in body.args
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol || continue
        lhs = st.args[1]
        haskey(assignments, lhs) && push!(ambiguous, lhs)
        assignments[lhs] = st.args[2]
    end
    for lhs in ambiguous
        delete!(assignments, lhs)
    end
    (; descriptor, plan, data=plan.data, parameters, assignments, ambiguous, pos)
end

# Accumulates `w * log(expr)` while compiling; `_adaptive_log_scale_program`
# freezes it into the sorted, concrete `_AdaptiveLogScale`.
mutable struct _AdaptiveLogScaleBuilder
    constant::Float64
    linear::Dict{Int,Float64}
    unit::Dict{Int,Tuple{Float64,Float64}}
    simplex::Dict{Tuple{Vector{Int},Int},Float64}
end

_AdaptiveLogScaleBuilder() = _AdaptiveLogScaleBuilder(
    0.0, Dict{Int,Float64}(), Dict{Int,Tuple{Float64,Float64}}(),
    Dict{Tuple{Vector{Int},Int},Float64}())

function _adaptive_r2d2_refuse(owner, expr, why)
    error("BRM adaptive centering: R2D2 block `$owner` derives its scale as ",
          "`$expr`; $why. Only products, quotients, square roots and literal ",
          "powers of positive scalars, `1 - R2`, and Dirichlet shares compile ",
          "to an online log-scale.")
end

function _adaptive_unc_index(ctx, owner, name)
    i = get(ctx.pos, name, 0)
    i == 0 && error(
        "BRM adaptive centering: R2D2 block `$owner` reads the unconstrained ",
        "coordinate `$name`, which this compiled model does not have. The ",
        "model and `unc_names` do not describe the same emission.")
    i
end

function _adaptive_scalar_constraint(ctx, owner, o::BRMOutput)
    extra = setdiff(collect(keys(o.constraints)), [:lower, :upper])
    isempty(extra) || error(
        "BRM adaptive centering: R2D2 block `$owner` reads `$(o.name)`, whose ",
        "Stan type carries $(Tuple(extra)); only lower/upper bounds are ",
        "supported.")
    function bound(key)
        haskey(o.constraints, key) || return nothing
        value = _brm_constraint_value(ctx.descriptor, o, key)
        value isa Real || error(
            "BRM adaptive centering: R2D2 block `$owner` reads `$(o.name)`, ",
            "whose `$key` bound `$(o.constraints[key])` is not one number ",
            "fixed by the data.")
        Float64(value)
    end
    (bound(:lower), bound(:upper))
end

# Add `w * log(sym)` for a scalar the scale reads by name.
function _adaptive_log_symbol!(acc, sym::Symbol, w, ctx, owner, depth)
    sym in ctx.ambiguous && _adaptive_r2d2_refuse(owner, sym,
        "`$sym` is assigned more than once in the emitted body")
    if haskey(ctx.data, sym)
        v = ctx.data[sym]
        v isa Real || _adaptive_r2d2_refuse(owner, sym,
            "data `$sym` is not a scalar")
        return _adaptive_log_form!(acc, v, w, ctx, owner, depth)
    end
    haskey(ctx.assignments, sym) &&
        return _adaptive_log_form!(acc, ctx.assignments[sym], w, ctx, owner, depth + 1)
    o = get(ctx.parameters, sym, nothing)
    isnothing(o) && _adaptive_r2d2_refuse(owner, sym,
        "`$sym` is neither data, a top-level assignment, nor a parameter")
    o.type === :real && isempty(o.size) || _adaptive_r2d2_refuse(owner, sym,
        "parameter `$sym` is a `$(o.type)`, not a scalar")
    lower, upper = _adaptive_scalar_constraint(ctx, owner, o)
    unit = lower == 0.0 && upper == 1.0
    lower == 0.0 && (isnothing(upper) || unit) || _adaptive_r2d2_refuse(
        owner, sym,
        "parameter `$sym` has bounds ($(something(lower, "none")), " *
        "$(something(upper, "none"))); only `lower=0` and the unit interval " *
        "are supported")
    i = _adaptive_unc_index(ctx, owner, String(sym))
    if unit
        a, b = get(acc.unit, i, (0.0, 0.0))
        acc.unit[i] = (a + w, b)
    else
        acc.linear[i] = get(acc.linear, i, 0.0) + w
    end
    acc
end

# Add `w * log(expr)` to `acc`.
function _adaptive_log_form!(acc, expr, w, ctx, owner, depth)
    depth <= 16 || _adaptive_r2d2_refuse(owner, expr,
        "its assignment chain is deeper than 16")
    expr = _adaptive_stan_expr_value(expr)
    if expr isa Real
        isfinite(expr) && expr > 0 || _adaptive_r2d2_refuse(owner, expr,
            "the constant $expr is not finite and positive")
        acc.constant += w * log(Float64(expr))
        return acc
    end
    expr isa Symbol && return _adaptive_log_symbol!(acc, expr, w, ctx, owner, depth)
    expr isa Expr || _adaptive_r2d2_refuse(owner, expr, "it is not an expression")
    if expr.head === :call
        f, args = expr.args[1], expr.args[2:end]
        if f === :* && !isempty(args)
            for a in args
                _adaptive_log_form!(acc, a, w, ctx, owner, depth)
            end
            return acc
        elseif f === :/ && length(args) == 2
            _adaptive_log_form!(acc, args[1], w, ctx, owner, depth)
            return _adaptive_log_form!(acc, args[2], -w, ctx, owner, depth)
        elseif f === :sqrt && length(args) == 1
            return _adaptive_log_form!(acc, args[1], w / 2, ctx, owner, depth)
        elseif f === :^ && length(args) == 2 && _adaptive_stan_expr_value(args[2]) isa Real
            p = Float64(_adaptive_stan_expr_value(args[2]))
            return _adaptive_log_form!(acc, args[1], w * p, ctx, owner, depth)
        elseif f === :- && length(args) == 2
            return _adaptive_log1m!(acc, args[1], args[2], w, ctx, owner)
        end
    elseif expr.head === :ref && length(expr.args) == 2
        return _adaptive_log_entry!(acc, expr.args[1], expr.args[2], w, ctx, owner)
    end
    _adaptive_r2d2_refuse(owner, expr, "this operation is not supported")
end

# `w * log(1 - s)` for a unit-interval parameter `s`: the residual share of R2.
function _adaptive_log1m!(acc, one_, s, w, ctx, owner)
    c = _adaptive_stan_expr_value(one_)
    s = _adaptive_stan_expr_value(s)
    expr = :($one_ - $s)
    c isa Real && c == 1 || _adaptive_r2d2_refuse(owner, expr,
        "only `1 - R2` subtracts")
    o = s isa Symbol ? get(ctx.parameters, s, nothing) : nothing
    (isnothing(o) || s in ctx.ambiguous) && _adaptive_r2d2_refuse(owner, expr,
        "`$s` is not a parameter")
    o.type === :real && isempty(o.size) &&
        _adaptive_scalar_constraint(ctx, owner, o) == (0.0, 1.0) ||
        _adaptive_r2d2_refuse(owner, expr, "`$s` is not a unit-interval scalar")
    i = _adaptive_unc_index(ctx, owner, String(s))
    a, b = get(acc.unit, i, (0.0, 0.0))
    acc.unit[i] = (a, b + w)
    acc
end

# `w * log(v[j])`: a Dirichlet share, an entry of a positive vector parameter,
# or a data entry.
function _adaptive_log_entry!(acc, v, j, w, ctx, owner)
    v = _adaptive_stan_expr_value(v)
    j = _adaptive_stan_expr_value(j)
    expr = :($v[$j])
    v isa Symbol && j isa Integer && j >= 1 || _adaptive_r2d2_refuse(owner, expr,
        "only a literal index into a named vector is supported")
    v in ctx.ambiguous && _adaptive_r2d2_refuse(owner, expr,
        "`$v` is assigned more than once in the emitted body")
    if haskey(ctx.data, v)
        d = ctx.data[v]
        d isa AbstractVector{<:Real} && j <= length(d) ||
            _adaptive_r2d2_refuse(owner, expr, "data `$v` has no real entry $j")
        return _adaptive_log_form!(acc, d[j], w, ctx, owner, 0)
    end
    o = get(ctx.parameters, v, nothing)
    isnothing(o) && _adaptive_r2d2_refuse(owner, expr, "`$v` is not a parameter")
    if o.type === :simplex
        n = _adaptive_simplex_length(ctx, owner, v)
        j <= n || _adaptive_r2d2_refuse(owner, expr, "`$v` has $n entries")
        coords = [_adaptive_unc_index(ctx, owner, "$v.$i") for i in 1:n-1]
        haskey(ctx.pos, "$v.$n") && _adaptive_r2d2_refuse(owner, expr,
            "simplex `$v` has more unconstrained coordinates than its $n-entry " *
            "Dirichlet declaration")
        key = (coords, Int(j))
        acc.simplex[key] = get(acc.simplex, key, 0.0) + w
    elseif o.type === :vector
        lower, upper = _adaptive_scalar_constraint(ctx, owner, o)
        lower == 0.0 && isnothing(upper) || _adaptive_r2d2_refuse(owner, expr,
            "vector `$v` is not `lower=0`")
        i = _adaptive_unc_index(ctx, owner, "$v.$j")
        acc.linear[i] = get(acc.linear, i, 0.0) + w
    else
        _adaptive_r2d2_refuse(owner, expr, "parameter `$v` is a `$(o.type)`")
    end
    acc
end

# A simplex's length, read from the Dirichlet declaration that samples it.
function _adaptive_simplex_length(ctx, owner, v)
    decls = [d for d in ctx.plan.declarations
             if d.role === :prior && d.target === v]
    length(decls) == 1 && decls[1].family === :dirichlet &&
        length(decls[1].arguments) == 1 || _adaptive_r2d2_refuse(owner, v,
        "simplex `$v` is not sampled by exactly one `dirichlet(alpha)` declaration")
    alpha = _adaptive_stan_expr_value(only(decls[1].arguments))
    alpha = alpha isa Symbol ? get(ctx.data, alpha, nothing) : alpha
    alpha isa AbstractVector{<:Real} || _adaptive_r2d2_refuse(owner, v,
        "the Dirichlet concentration of `$v` is not a data vector")
    length(alpha)
end

function _adaptive_log_scale_program(expr, ctx, owner)
    acc = _adaptive_log_form!(_AdaptiveLogScaleBuilder(), expr, 1.0, ctx, owner, 0)
    _AdaptiveLogScale(
        acc.constant,
        [(acc.linear[i], i) for i in sort!(collect(keys(acc.linear)))],
        [(acc.unit[i]..., i) for i in sort!(collect(keys(acc.unit)))],
        [_AdaptiveSimplexLog(acc.simplex[k], k[1], k[2])
         for k in sort!(collect(keys(acc.simplex)))],
    )
end

const _ADAPTIVE_R2D2_SCALE_KEYWORD = Dict(
    :ranef_intercept_r2d2 => :scale,
    :ranef_correlated_r2d2 => :tau,
    :ranef_correlated_draws_r2d2 => :tau,
)

function _adaptive_r2d2_block(ranef::RanefBlock, unc_names, pos, ctx)
    binding = ranef.binding
    decls = [d for d in ctx.plan.declarations
             if d.role === :prior && d.target === binding]
    length(decls) == 1 || error(
        "BRM adaptive centering: R2D2 block `$binding` has $(length(decls)) ",
        "prior declarations in the emitted body; expected exactly one.")
    key = _ADAPTIVE_R2D2_SCALE_KEYWORD[ranef.family]
    sym = get(only(decls).keywords, key, nothing)
    sym isa Symbol && haskey(ctx.assignments, sym) || error(
        "BRM adaptive centering: R2D2 block `$binding` passes `$key=$(repr(sym))`, ",
        "which is not a top-level assignment of the emitted body; its derived ",
        "scale cannot be resolved.")
    rhs = ctx.assignments[sym]
    margins = key === :scale ? Any[rhs] :
        (rhs isa Expr && rhs.head === :vect ? rhs.args : error(
            "BRM adaptive centering: R2D2 block `$binding` assigns `$sym = $rhs`; ",
            "expected one derived expression per margin."))
    K = ranef.n_terms
    length(margins) == K || error(
        "BRM adaptive centering: R2D2 block `$binding` has $K terms but `$sym` ",
        "derives $(length(margins)) scales.")
    scales = [_adaptive_log_scale_program(m, ctx, binding) for m in margins]
    cholesky_names = ranef.family === :ranef_intercept_r2d2 ? String[] :
        ["$(binding)_L.$i" for i in 1:(K * (K - 1) ÷ 2)]
    R2D2AdaptiveCenteringBlock(
        ranef,
        ranef.noncentered ? 0.0 : 1.0,
        ranef_coordinates(ranef, unc_names),
        _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
        scales,
    )
end

function _adaptive_named_indices(pos, names, binding, role)
    missing = String[]
    out = Vector{Int}(undef, length(names))
    for (i, name) in enumerate(names)
        idx = get(pos, name, 0)
        idx == 0 ? push!(missing, name) : (out[i] = idx)
    end
    isempty(missing) || error(
        "BRM adaptive centering: block `$binding` expects $role unconstrained ",
        "coordinates that this compiled model does not have: ",
        join(missing, ", "), ". The model and `unc_names` do not describe the ",
        "same emission.",
    )
    out
end

# One block's frame: the same name spells the whole-file loop uses, factored
# so centered replay (prediction.jl) resolves ONE block's hyperparameters
# without tripping over a sibling this contract skips or refuses. Returns
# `nothing` for a family outside every set below. A stratified block has one
# frame per stratum, which `_adaptive_stratified_blocks` resolves; it raises
# here rather than being read as one frame. An R2D2 block needs the emitted
# model's scale context `r2d2` (`_adaptive_r2d2_context`), which only
# `adaptive_centering_blocks` builds: R2D2 blocks are never centered, so the
# replay caller never reaches one. A known-value block needs its binding
# record `known` (`_adaptive_known_record`), which carries the known values.
# A Student-t family resolves its Gaussian family's frame over its own effect
# coordinates (`effects`, which the replay caller also passes) and returns it
# in derived-scale form with its mixing weights.
function _adaptive_block(ranef::RanefBlock, unc_names, pos; r2d2=nothing,
                         known=nothing, effects=nothing)
    ranef.family in _ADAPTIVE_STRATIFIED_FAMILIES && error(
        "BRM adaptive centering: stratified block `$(ranef.binding)` ",
        "(`$(ranef.family)`, group `$(ranef.group)`, by `$(ranef.by)`) has one ",
        "scale/Cholesky frame per stratum, not one frame; resolve it through ",
        "`adaptive_centering_blocks(model, unc_names)`, which returns one block ",
        "per stratum.",
    )
    _ranef_level_correlated(ranef.family) && error(
        "BRM adaptive centering: block `$(ranef.binding)` correlates the levels ",
        "of `$(ranef.group)` with each other (`gr(...; cor=...)`), so its levels ",
        "share one factor across levels; per-level centering of such a block is ",
        "not implemented, and it is not silently left fixed either.",
    )
    # A Student-t block is its Gaussian family's frame with level `g`'s factor
    # scaled by `sqrt(w[g])`: the Gaussian frame is resolved over the block's
    # own effect coordinates, then carried in derived-scale form with the
    # weights.
    base = get(_RANEF_STUDENT_T_BASES, ranef.family, nothing)
    if !isnothing(base)
        gaussian = RanefBlock((f === :family ? base : getfield(ranef, f)
                               for f in fieldnames(RanefBlock))...)
        frame = _adaptive_block(gaussian, unc_names, pos; r2d2, known,
            effects=isnothing(effects) ? ranef_coordinates(ranef, unc_names) : effects)
        isnothing(frame) && error(
            "BRM adaptive centering: Student-t block `$(ranef.binding)` ",
            "(`$(ranef.family)`) has a Gaussian family `$base` that adaptive ",
            "centering does not describe.")
        return _adaptive_derived_block(frame; ranef,
            mixing=_ranef_mixing_coordinates(ranef, unc_names))
    end
    if get(get(_RANEF_FAMILIES, ranef.family, (;)), :known, false)
        isnothing(known) && error(
            "BRM adaptive centering: known-value block `$(ranef.binding)` ",
            "needs its binding record; resolve it through ",
            "`adaptive_centering_blocks(model, unc_names)`.")
        return _adaptive_known_block(ranef, unc_names, pos, known; effects)
    end
    if ranef.family in _ADAPTIVE_R2D2_FAMILIES
        isnothing(r2d2) && error(
            "BRM adaptive centering: R2D2 block `$(ranef.binding)` derives its ",
            "scales from the emitted model; resolve it through ",
            "`adaptive_centering_blocks(model, unc_names)`.")
        return _adaptive_r2d2_block(ranef, unc_names, pos, r2d2)
    end
    is_intercept = ranef.family in _ADAPTIVE_INTERCEPT_FAMILIES
    is_correlated = ranef.family in _ADAPTIVE_CORRELATED_FAMILIES
    (is_intercept || is_correlated) || return nothing
    K = ranef.n_terms
    n_cholesky = K * (K - 1) ÷ 2
    binding = ranef.binding
    cholesky_names = is_intercept ? String[] :
        ["$(binding)_L.$i" for i in 1:n_cholesky]
    scale_names = is_intercept ? ["$(binding)_log_scale"] :
        ["$(binding)_tau.$i" for i in 1:K]
    target_c = ranef.noncentered ? 0.0 : 1.0
    AdaptiveCenteringBlock(
        ranef,
        target_c,
        isnothing(effects) ? ranef_coordinates(ranef, unc_names) : effects,
        _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
        _adaptive_named_indices(pos, scale_names, binding, "scale"),
    )
end

# The binding record of a known-value block in `plan` (its `known_scales` and
# `known_cor`), or `nothing` for any other block.
_adaptive_known_record(plan, ranef::RanefBlock) =
    get(get(_RANEF_FAMILIES, ranef.family, (;)), :known, false) ?
        plan.bindings[ranef.binding] : nothing

# A known-value block (`sd(...) = value`, `cor(:, ID) = R`;
# `_sb_emit_known_id_bucket!`) in derived-scale form: term `k`'s log-scale is
# the constant `log(value)` for a known SD, or the log of its sampled scale,
# `<binding>_tau_free` when some scale is known and `<binding>_tau` otherwise
# (each `lower=0`, so its unconstrained value is that log). A known correlation
# is a constant Cholesky factor in place of the `<binding>_L` coordinates.
function _adaptive_known_block(ranef::RanefBlock, unc_names, pos, record;
                               effects=nothing)
    K = ranef.n_terms
    binding = ranef.binding
    known = collect(record.known_scales)
    free = findall(isnothing, known)
    carrier = length(free) < K ? "$(binding)_tau_free" : "$(binding)_tau"
    free_index = _adaptive_named_indices(pos, ["$carrier.$j" for j in eachindex(free)],
                                         binding, "scale")
    scales = map(1:K) do k
        constant, linear = isnothing(known[k]) ?
            (0.0, [(1.0, free_index[only(findall(==(k), free))])]) :
            (log(known[k]), Tuple{Float64,Int}[])
        _AdaptiveLogScale(constant, linear, Tuple{Float64,Float64,Int}[],
                          _AdaptiveSimplexLog[])
    end
    known_cor = record.known_cor
    cholesky_names = isnothing(known_cor) ?
        ["$(binding)_L.$i" for i in 1:(K * (K - 1)) ÷ 2] : String[]
    R2D2AdaptiveCenteringBlock(
        ranef,
        ranef.noncentered ? 0.0 : 1.0,
        isnothing(effects) ? ranef_coordinates(ranef, unc_names) : effects,
        _adaptive_named_indices(pos, cholesky_names, binding, "Cholesky"),
        scales;
        cholesky_known=isnothing(known_cor) ? zeros(0, 0) :
            Matrix{Float64}(cholesky(Symmetric(known_cor)).L),
    )
end

# A stratified `gr(g, by=b)` block (`ranef_correlated_by*`) as one ordinary
# block per stratum. Stratum `s` samples its own scales `tau_s[:, s]` and
# correlation factor `L_s[s]`, and only its levels read them: level `g` draws
# `b[g] = diag(tau_s[:, s]) * L_s[s] * z[g]` with `s = stratum_idx[g]`. So
# stratum `s` is exactly an ordinary block over its own levels, and no two
# strata share a coordinate. Each stratum's block keeps the block's binding,
# group and `by`, and holds the stratum's levels in block order, the order
# `brm_ranef_draws` reports them; the blocks follow the stratum order. These
# families are always non-centred: `centered_groups` refuses a stratified
# group at emission.
function _adaptive_stratified_blocks(ranef::RanefBlock, unc_names, pos, plan)
    spec = _RANEF_FAMILIES[ranef.family]
    S, stratum = _ranef_stratum_index(plan, ranef, "BRM adaptive centering")
    K, binding = ranef.n_terms, ranef.binding
    effects = ranef_coordinates(ranef, unc_names)
    # The sampled scales are a `matrix<lower=0>[K, S]`, named and stored
    # column-major.
    tau = reshape(_adaptive_named_indices(pos,
        ["$(binding)_$(_adaptive_stratum_carrier(spec, :scale)).$k.$s"
         for s in 1:S for k in 1:K], binding, "scale"), K, S)
    cholesky = _adaptive_stratum_cholesky(pos, binding,
        _adaptive_stratum_carrier(spec, :correlation_cholesky), S, K)
    blocks = AdaptiveCenteringBlock[]
    for s in 1:S
        members = findall(==(s), stratum)
        part = RanefBlock(binding, ranef.family, ranef.group, ranef.id, ranef.by,
            ranef.levels[members], K, length(members), ranef.z, ranef.noncentered,
            ranef.generated, ranef.population)
        push!(blocks, AdaptiveCenteringBlock(part, ranef.noncentered ? 0.0 : 1.0,
            effects[:, members], cholesky[s], tau[:, s]))
    end
    blocks
end

# The parameter a stratified family samples in each stratum's plate cell for
# `component` (`:scale`, `:correlation_cholesky`): `tau_s_tau` and `L_s_L`,
# which the collected `tau_s` and `L_s` copy. `_RANEF_FAMILIES` labels them.
_adaptive_stratum_carrier(spec, component) =
    only(carrier for (carrier, c) in pairs(spec.internals) if c === component)

# The Cholesky coordinates of each stratum's correlation factor, stratum by
# stratum in Stan's `cholesky_factor_corr` order; none for a one-margin block,
# whose single-margin form samples no factor (`_sb_single_margin_ranef`).
# The factors are an `array[S] cholesky_factor_corr[K]`, which Stan stores
# element-major (stratum `s` at `(s-1)*m+1 : s*m`, `m = K(K-1)/2`) but names
# array-index-fastest, as for the `:group_term` carrier in `ranef_coordinates`:
# the names locate the carrier's contiguous segment, and positions within it
# follow the storage order (measured against BridgeStan `param_constrain`).
function _adaptive_stratum_cholesky(pos, binding, carrier, S, K)
    m = K * (K - 1) ÷ 2
    m == 0 && return [Int[] for _ in 1:S]
    found = _adaptive_named_indices(pos,
        ["$(binding)_$(carrier).$s.$i" for i in 1:m for s in 1:S], binding, "Cholesky")
    start = minimum(found)
    sort(found) == start:(start + S * m - 1) || error(
        "BRM adaptive centering: stratified block `$binding` resolves its ",
        "`array[$S] cholesky_factor_corr[$K]` coordinates to a non-contiguous ",
        "unconstrained segment. The model and `unc_names` do not describe the ",
        "same emission.")
    [collect(start + (s - 1) * m .+ (0:m-1)) for s in 1:S]
end

"""
    adaptive_centering_blocks(model, unc_names) -> Vector{AbstractAdaptiveCenteringBlock}

Describe every random-effect block in `model` for adaptive partial centering:
an [`AdaptiveCenteringBlock`](@ref) for each block whose marginal scales are
sampled, and an [`R2D2AdaptiveCenteringBlock`](@ref) for each block whose
scales an R2D2 prior derives or that holds a scale or its correlation at a
known value. `model` is an [`SBBRMI`](@ref) or
[`GenerativePlan`](@ref), and `unc_names` is the compiled model's unconstrained
parameter-name vector (for example BridgeStan's `param_unc_names`).

The result supports both BRM endpoints: the default noncentered emission is the
compiled target `c=0`, while `SBBRMI(...; centered_groups=...)` is target `c=1`.
Every intermediate source uses the triangular term-wise map

```
u[k] = c[k] * sum(C[k,l] * z[l] for l < k) + C[k,k]^c[k] * z[k]
```

with `C = diag(tau) * L`. At `K=1`, this reduces to `u = s^c * z` with
`s = exp(log_scale)` and no Cholesky coordinates. Consequently `c=0` is the
literal standardised draw and `c=1` is the literal model-scale effect.
[`adaptive_centering_problem`](@ref) additionally centres the cells of a block
compiled non-centred on their predictor's population coefficient where one
matches (`u[k]` gains `c[k] * beta[k]`, so `c=1` is the level's total);
that location is not part of this metadata, and
[`adaptive_centering_controls`](@ref) reports it per cell.

R2D2 blocks use the same map; only `tau` differs. It is not a coordinate but the
emitted model's own derived scale — `ref * sqrt(phi[j] * R2 / (1 - R2))` for
`sd(...) ~ r2d2(...)` (block-wide, per-margin ICC, or joint `include=`), with
a free half-normal for a margin a partial ICC statement leaves unaddressed, and
`sqrt((1 - R2) * tau_bsv^2)` for the whole-predictor `effect(lp, :) ~ r2d2(...)`
form — re-derived from the `R2`, Dirichlet share and reference/total scale
coordinates on every evaluation. Those coordinates are read, never transformed.
A reference scale must be a positive constant, data scalar, or `lower=0`
parameter (or a product, quotient, square root or literal power of them);
anything else raises naming the block and the expression.

Known-value blocks (`sd(...) = value`, `cor(:, ID) = R`) use the same map
under either emission and are returned as `R2D2AdaptiveCenteringBlock`s too: a
known SD is a constant log-scale, a sampled SD of the same block its
`<binding>_tau` (or `<binding>_tau_free`) coordinate, and a known correlation
the constant Cholesky factor `cholesky_known`, with no Cholesky coordinates.
When every scale and the correlation are known, every control of the block is
a fixed per-coordinate rescaling of the compiled draw, so it cannot change the
sampling geometry under an adapted diagonal metric.

Student-t blocks (`ranef(:, ID) ~ TDist(nu)`), with sampled, configured or
known scales and either emission, are `R2D2AdaptiveCenteringBlock`s whose
level `g` has the factor `C_g = sqrt(w[g]) * diag(tau) * L`, with `w[g]` the
block's mixing weight. The same map then reads `C_g` in place of `C`, so `c=0`
is the emitted standardised draw and `c=1` the level's effect `b[g]`. The
weights are read, never transformed.

A stratified `gr(g, by=b)` block samples one `tau`, `L` frame per stratum of
`b`, and each group reads its own stratum's frame. It is returned as one
[`AdaptiveCenteringBlock`](@ref) per stratum, in stratum order: the stratum's
`tau` and `L` coordinates over the effects of its groups. Each such block keeps
the stratified block's `binding`, `group` and `by`, and its `ranef.levels` are
the stratum's groups in block order, as [`brm_ranef_draws`](@ref) lists them.
Intercept-only `(1 | mm(...))` blocks adapt through the
ordinary scalar path (their downstream gather is linear); an `mm` block with
any slope term, including a slope-only `(0 + x | mm(...))`, shares the ordinary
correlated emission and adapts with it.

A block emitted in generated quantities ([`RanefBlock`](@ref)`.generated`: no
observed likelihood reaches it, for example because its response is omitted)
has no sampler coordinate: Stan re-draws its effects, so it has nothing to
adapt and is skipped, as [`transport_draws`](@ref) and
[`population_draws`](@ref) skip it. Its scale or correlation parameters, if
they remain sampled, stay untouched coordinates of the wrapped density.

Correlated `cdar(step; by=group, cor=C)` walks are not ordinary random-effect
blocks either; they have their own metadata contract in
[`_adaptive_cdar_centering_blocks`](@ref) and join the online plan through the
WarmupHMC extension exactly like the ungrouped-HSGP companion below.
"""
function adaptive_centering_blocks(model, unc_names)
    pos = _ranef_name_positions(unc_names)
    plan = _ranef_plan(model)
    # A generated block has no sampler coordinate; see the docstring.
    ranefs = filter(r -> !r.generated, ranef_blocks(plan))
    r2d2 = any(r -> r.family in _ADAPTIVE_R2D2_FAMILIES, ranefs) ?
        _adaptive_r2d2_context(model, pos) : nothing
    out = AbstractAdaptiveCenteringBlock[]
    for ranef in ranefs
        if ranef.family in _ADAPTIVE_STRATIFIED_FAMILIES
            append!(out, _adaptive_stratified_blocks(ranef, unc_names, pos, plan))
            continue
        end
        blk = _adaptive_block(ranef, unc_names, pos; r2d2,
                              known=_adaptive_known_record(plan, ranef))
        isnothing(blk) || push!(out, blk)
    end

    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects))
        append!(claimed, block.cholesky_free)
        block isa AdaptiveCenteringBlock && append!(claimed, block.log_scales)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted correlated blocks claim overlapping ",
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    # A derived scale must not read a coordinate the transform rewrites:
    # the map's triangular Jacobian assumes every block's scale is constant
    # in every block's effects. Scale inputs may be shared (one `R2` or
    # reference across margins), so they are checked against effects only.
    effects = Set{Int}(Iterators.flatten(vec(b.effects) for b in out))
    for block in out
        block isa R2D2AdaptiveCenteringBlock || continue
        isdisjoint(_adaptive_read_coordinates(block), effects) || error(
            "BRM adaptive centering: block `$(block.ranef.binding)` ",
            "derives its scale from a random-effect coordinate this ",
            "transform rewrites; refusing an inexact transform.")
    end
    out
end

# The HSGP path deliberately has its own metadata type rather than widening
# `AdaptiveCenteringBlock`.  The latter feeds a small-block Enzyme
# specialization whose exact dispatch and arithmetic are part of the ordinary
# random-effect contract.  Keeping the two paths separate makes adding HSGPs a
# zero-change operation for that hot path.
struct _HSGPAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    # The compiled model may use a different fixed exponent for every basis.
    # Both the transform and retrospective scorer must start in that frame.
    target_c::Vector{Float64}
    effects::Vector{Int}
    # One length scale per `omega2` column, or the single shared one of an
    # isotropic multi-axis term (`_brm_hsgp_omega2`). Coordinate `0` (here or
    # in `sd`) is a value held known by `=` on its address, carried as both of
    # its bounds (`_adaptive_hsgp_hyper`).
    length_scales::Vector{Int}
    # Compiler-declared bounds of each length scale and the marginal SD; an
    # upper bound is `Inf` unless declared (`_adaptive_bounds`).
    length_scale_lower::Vector{Float64}
    length_scale_upper::Vector{Float64}
    sd::Int
    sd_lower::Float64
    sd_upper::Float64
    omega2::Matrix{Float64}
    # Periodic spectral geometry: the compiler-owned harmonic index of each
    # basis column (`[1, 2, ..., K, 1, 2, ..., K]` for cosine/sine pairs).
    # Empty for squared-exponential blocks, whose geometry is `omega2` (which
    # is in turn empty for periodic blocks); the two are never populated
    # together, and `_adaptive_hsgp_log_scale` selects the spectrum on this
    # field.
    harmonics::Vector{Float64}
end

Base.show(io::IO, b::_HSGPAdaptiveCenteringBlock) = print(
    io,
    "HSGPAdaptiveCenteringBlock(", b.logical, ", ", b.term,
    ", target_c=", b.target_c, ", ", length(b.effects), " basis weights",
    isempty(b.harmonics) ? ")" : ", periodic)",
)

_adaptive_stan_expr_value(x::StanBlocks.StanExpr) =
    _adaptive_stan_expr_value(x.expr)
_adaptive_stan_expr_value(x) = x

# A declared constraint's value. StanBlocks evaluates every output's
# `constraints` against the model's current data as `constraint_values`
# (StanBlocks b233afbc, stanblocks-use §30), with Stan's semantics: a literal,
# a data value, the `fmax`/`fmin` sbimpl writes where an explicit bound meets a
# structural one (`_sb_bound_intersection`), arithmetic over data, and one
# entry per coordinate for a per-coordinate bound (an anisotropic term's
# `fmax(0.0, rho_lower_…)`, elementwise since StanBlocks c7293b88). A bound
# that reads a parameter has no value without a draw and comes back `nothing`
# for the caller to refuse. A descriptor without a Stan view (Turing) declares
# its bounds as numbers already. A caller holding only the `GenerativePlan`
# gets the view from its model, at the cost of one more `stan_descriptor`.
# Shared by adaptive centering and `parameter_draws`.
_brm_stan_view(d) = d.stan
_brm_stan_view(plan::GenerativePlan) =
    Base.invokelatest(StanBlocks.stan_descriptor, plan.model)

function _brm_constraint_value(d, output::BRMOutput, key::Symbol)
    raw = output.constraints[key]
    view = _brm_stan_view(d)
    isnothing(view) && return raw
    i = findfirst(o -> o.name === output.name, view.outputs)
    isnothing(i) && error(
        "BRM: output `$(output.name)` is not in the descriptor's StanBlocks " *
        "view, so its `$key` bound cannot be read.",
    )
    stan = view.outputs[i]
    hasproperty(stan, :constraint_values) || error(
        "BRM: reading the `$key` bound of `$(output.name)` needs StanBlocks' " *
        "`ModelOutput.constraint_values` (StanBlocks b233afbc or later); the " *
        "loaded StanBlocks predates it.",
    )
    getproperty(stan, :constraint_values)[key]
end

# The scalar hyperparameters an adaptive cell's spread reads (HSGP length
# scales and marginal SDs, cdar marginal SDs and persistences) are declared by
# the compiler in exactly two shapes: `<lower=a>` (a default or non-Uniform
# scale prior) and `<lower=a, upper=b>` (an explicit `Uniform(a, b)` or
# two-sided `truncated` prior, see `_sb_gp_scale_prior` / `_sb_dar_ar_prior`,
# and every cdar persistence). Both are Stan's ordinary scalar transforms, so
# the physical value is a closed-form function of the unconstrained
# coordinate, reproduced by `_adaptive_constrain`. Each end is fixed by the
# data (`_brm_constraint_value`). An absent upper bound reads as `Inf`. A
# declaration without a finite lower bound, with an offset/multiplier, or with a
# bound that reads a parameter still raises.
function _adaptive_bounds(descriptor, output::BRMOutput, n::Int, owner, role)
    constraints = output.constraints
    unsupported = setdiff(collect(keys(constraints)), [:lower, :upper])
    isempty(unsupported) || error(
        "BRM adaptive centering: `$owner` $role uses unsupported Stan " *
        "constraint(s) $(Tuple(unsupported)); this contract supports a finite " *
        "lower bound, an optional finite upper bound, and no " *
        "offset/multiplier transform.",
    )
    haskey(constraints, :lower) || error(
        "BRM adaptive centering: `$owner` $role is not lower-bounded; " *
        "the compiled unconstrained-to-physical transform is unsupported.",
    )
    lower = _adaptive_bound_values(descriptor, output, :lower, n, owner, role)
    upper = haskey(constraints, :upper) ?
        _adaptive_bound_values(descriptor, output, :upper, n, owner, role) :
        fill(Inf, n)
    all(lower .< upper) || error(
        "BRM adaptive centering: `$owner` $role has lower bounds $lower not " *
        "strictly below its upper bounds $upper.",
    )
    lower, upper
end

function _adaptive_bound_values(descriptor, output::BRMOutput, side::Symbol,
                                n::Int, owner, role)
    value = _brm_constraint_value(descriptor, output, side)
    value isa Real || value isa AbstractArray{<:Real} || error(
        "BRM adaptive centering: `$owner` $role has the $side bound " *
        "`$(output.constraints[side])`, which reads a parameter; this contract " *
        "needs a bound fixed by the data.",
    )
    values = value isa Real ? fill(Float64(value), n) : collect(Float64, value)
    length(values) == n || error(
        "BRM adaptive centering: `$owner` $role has $(length(values)) " *
        "$side bounds for $n compiler-owned coordinates.",
    )
    all(isfinite, values) || error(
        "BRM adaptive centering: `$owner` $role $side bounds must be finite.",
    )
    values
end

# Stan Math's `inv_logit` (`stan/math/prim/fun/inv_logit.hpp`), so a
# lower-upper-bounded coordinate reads the value the compiled model constrains
# to rather than a reassociated approximation of it.
const _ADAPTIVE_LOG_EPSILON = log(eps(Float64))
function _adaptive_inv_logit(x)
    if x < 0
        e = exp(x)
        x < _ADAPTIVE_LOG_EPSILON && return e
        return e / (1 + e)
    end
    inv(1 + exp(-x))
end

# Stan's `lb_constrain` (`exp(x) + lower`) and finite `lub_constrain`
# (`(upper - lower) * inv_logit(x) + lower`) for a `_adaptive_bounds` pair.
_adaptive_constrain(x, lower, upper) =
    isfinite(upper) ? (upper - lower) * _adaptive_inv_logit(x) + lower :
                      exp(x) + lower

function _adaptive_same_hsgp_owner(resolved, owner, role)
    declaration = resolved.output.declaration
    (!isnothing(declaration) && declaration.target === owner.target) || error(
        "BRM adaptive centering: HSGP `$(owner.target)` $role resolved to a " *
        "different compiler declaration; refusing crossed term metadata.",
    )
    resolved
end

# An HSGP term's length scales and marginal SD as its adaptive blocks read
# them: each sampled one's unconstrained coordinate with its declared bounds,
# or, held at a known value by `=` on its address
# (`length_scale(lp, hsgp(x)) = value`), coordinate 0 with both bounds at that
# value (`_adaptive_hsgp_hyper`).
function _adaptive_hsgp_hypers(descriptor, logical, unc_names, entry, owner)
    rho = _adaptive_hsgp_hyper_coordinates(descriptor, logical, unc_names,
        entry, owner, :length_scale, "length scale")
    sigma = _adaptive_hsgp_hyper_coordinates(descriptor, logical, unc_names,
        entry, owner, :sd, "marginal SD")
    length(rho.coordinates) >= 1 || error(
        "BRM adaptive centering: HSGP `$(entry.term)` has no length-scale coordinate.",
    )
    length(sigma.coordinates) == 1 || error(
        "BRM adaptive centering: HSGP `$(entry.term)` must have one marginal-SD coordinate.",
    )
    rho, sigma
end

function _adaptive_hsgp_hyper_coordinates(descriptor, logical, unc_names, entry,
                                          owner, parameter, what)
    known = _brm_term_known_value(descriptor.plan.parent, logical, entry.value, parameter)
    if !isnothing(known)
        values = collect(Float64, known isa Real ? (known,) : known)
        return (; coordinates=zeros(Int, length(values)), lower=values, upper=copy(values))
    end
    resolved = _adaptive_same_hsgp_owner(brm_term_coordinates(
        descriptor, logical, unc_names; term=entry.term, parameter), owner, what)
    lower, upper = _adaptive_bounds(descriptor, resolved.output,
        length(resolved.coordinates), entry.term, what)
    (; coordinates=collect(resolved.coordinates), lower, upper)
end

# One length scale or marginal SD of an HSGP block at the unconstrained point
# `x`: its constrained coordinate, or its known value at coordinate 0.
_adaptive_hsgp_hyper(x, coordinate::Int, lower, upper) =
    iszero(coordinate) ? lower : _adaptive_constrain(x[coordinate], lower, upper)

# Whether a formula term is emitted in generated quantities: no observed
# likelihood reaches it (for example, its response is omitted). Its weights or
# innovations then have no sampler coordinate, so it has no adaptive cell, like
# a generated `RanefBlock` in `adaptive_centering_blocks`. Carrier lookup
# prefers a sampled carrier (`_brm_carrier_indices`), so a generated answer
# means none is sampled. Only the term families with adaptive cells answer.
_adaptive_term_generated(::Any, descriptor, logical, entry) = false
_adaptive_term_generated(::typeof(cdar), descriptor, logical, entry) =
    _brm_term_parameter_output(descriptor, logical, entry, :innovations).kind ===
        :generated_quantity
_adaptive_term_generated(::typeof(car), descriptor, logical, entry) =
    _brm_term_parameter_output(descriptor, logical, entry, :structured).kind ===
        :generated_quantity
function _adaptive_term_generated(::typeof(hsgp), descriptor, logical, entry)
    kw = getkwargs(entry.value)
    haskey(kw, :by) || return _brm_term_parameter_output(
        descriptor, logical, entry, :basis_weights).kind === :generated_quantity
    # A grouped term's flat weight carrier has no parameter role.
    flat = last(_adaptive_grouped_hsgp_weights(
        descriptor.plan.parent, logical, entry))
    idx = _brm_carrier_indices(descriptor.outputs, o -> o.name === flat)
    !isempty(idx) && descriptor.outputs[first(idx)].kind === :generated_quantity
end

# Everything the adaptive builders skip because it is emitted in generated
# quantities: random-effect blocks, exact-total blocks and HSGP/cdar terms. The
# WarmupHMC wrapper names them when it finds no cell at all, so a model whose
# every block is generated says why instead of claiming it has none.
function _adaptive_generated_labels(model)
    labels = Symbol[b.binding for b in ranef_blocks(model) if b.generated]
    append!(labels, (b.binding for b in _total_blocks_by_placement(model).generated))
    descriptor = brm_descriptor(model)
    for predictor in linear_predictors(descriptor.plan.parent)
        for entry in _brm_term_coordinate_entries(descriptor.plan.parent, predictor.name)
            _adaptive_term_generated(getf(entry.value), descriptor,
                                     predictor.name, entry) &&
                push!(labels, entry.term)
        end
    end
    labels
end

"""
    _adaptive_hsgp_centering_blocks(model, unc_names)

Resolve the compiled coordinates and fixed spectral geometry for every
ungrouped and grouped squared-exponential HSGP and every ungrouped periodic
HSGP in an `SBBRMI` or `GenerativePlan`. A grouped term contributes one block
per group level; all of them share the term's spectral scales, so each
level's basis weights adapt as independent scalar cells around the same
per-basis frame. A periodic term contributes one block whose `harmonics`
carry the cosine/sine spectrum; its cells join the same HSGP pair order as
squared-exponential cells. A grouped term whose levels are correlated
(`cor=`) or scaled per level (a hyper-predictor) is resolved by
[`_adaptive_hsgp_level_centering_blocks`](@ref) instead.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref).
It follows BRM's formula-term descriptor to declaration-owned parameter roles,
then reads the declaration's compiler-owned spectral data binding (`omega2`
for squared-exponential terms, `harmonics` for periodic ones).  It never
parses generated Stan or assumes a global parameter order.  A multi-axis
squared-exponential term carries one length scale per axis (`iso=false`) or
its single isotropic length scale, which every axis's frequencies read, as in
the compiled spectrum.  Length scales and marginal SDs may be `<lower=a>` or `<lower=a, upper=b>` (an explicit
`Uniform(a, b)` prior); the cells read them through Stan's own transform. The
metadata is intentionally fail-closed: unknown covariances, grouped periodic
terms, a scale without a finite lower bound or with an offset/multiplier, and
declaration/artifact coordinate drift raise before a reparametrizer is built.
A term emitted in generated quantities (no observed likelihood reaches it) has
no sampler coordinate and is skipped, like a generated random-effect block.
"""
function _adaptive_hsgp_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    pos = _ranef_name_positions(unc_names)
    out = _HSGPAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === hsgp || continue
            _adaptive_term_generated(hsgp, descriptor, logical, entry) && continue
            kw = getkwargs(entry.value)
            covariance = _sb_gp_cov(kw, :hsgp)
            if covariance === :periodic
                _adaptive_periodic_hsgp_block!(
                    out, descriptor, plan, unc_names, logical, entry, kw)
                continue
            end
            covariance === :exp_quad || error(
                "BRM adaptive centering: HSGP `$(entry.term)` on predictor " *
                "`$logical` uses covariance `$covariance`; this contract " *
                "supports the squared-exponential and periodic geometries.",
            )
            if haskey(kw, :by)
                _adaptive_grouped_hsgp_blocks!(
                    out, descriptor, plan, pos, unc_names, logical, entry, kw)
                continue
            end

            weights = brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:basis_weights,
            )
            owner = weights.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: HSGP `$(entry.term)` basis weights " *
                "have no compiler declaration owner.",
            )
            if owner.family isa Symbol
                owner.family in (
                    :_sb_hsgp, :_sb_hsgp_aniso,
                    :_sb_hsgp_partial, :_sb_hsgp_partial_aniso,
                    :_sb_hsgp_latent, :_sb_hsgp_latent_orthogonal, :_sb_hsgp_latent_centered,
                ) || error(
                    "BRM adaptive centering: HSGP `$(entry.term)` resolved to " *
                    "unsupported emitted family `$(owner.family)`.",
                )
            end
            isempty(weights.output.constraints) || error(
                "BRM adaptive centering: HSGP `$(entry.term)` basis weights " *
                "are constrained, so they are not the emitted standard-normal " *
                "`beta_raw` coordinates this transform requires.",
            )

            rho, sigma = _adaptive_hsgp_hypers(
                descriptor, logical, unc_names, entry, owner)

            omega2 = _brm_hsgp_omega2(plan, owner,
                "BRM adaptive centering: HSGP `$(entry.term)`",
                length(weights.coordinates), length(rho.coordinates))
            push!(out, _HSGPAdaptiveCenteringBlock(
                logical, entry.term,
                _brm_hsgp_centeredness(kw, length(weights.coordinates)),
                collect(weights.coordinates),
                rho.coordinates, rho.lower, rho.upper,
                only(sigma.coordinates), only(sigma.lower), only(sigma.upper),
                omega2, Float64[],
            ))
        end
    end

    claimed_effects = Int[]
    claimed_scales = Int[]
    seen_terms = Set{Tuple{Symbol,Symbol}}()
    for block in out
        append!(claimed_effects, block.effects)
        # One grouped term contributes one block per group level; all of them
        # share the term's length-scale and marginal-SD coordinates, so scales
        # are claimed once per (predictor, term), while every weight cell is
        # claimed exactly once.
        (block.logical, block.term) in seen_terms && continue
        push!(seen_terms, (block.logical, block.term))
        # A known value (coordinate 0) claims no coordinate.
        append!(claimed_scales, filter(!iszero, block.length_scales))
        iszero(block.sd) || push!(claimed_scales, block.sd)
    end
    length(unique(claimed_effects)) == length(claimed_effects) &&
        length(unique(claimed_scales)) == length(claimed_scales) || error(
        "BRM adaptive centering: emitted HSGP blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

# Ungrouped periodic HSGP weights for online adaptation. The cosine/sine
# basis weights are conditionally independent given their spectral scales
# exactly like squared-exponential weights, so one term contributes one
# ordinary `_HSGPAdaptiveCenteringBlock` with `harmonics` populated and
# `omega2` empty; `_adaptive_hsgp_log_scale` selects the Bessel spectrum on
# that field. The compiled periodic emission is always fully non-centered
# (sbimpl refuses a periodic `centeredness`), so `target_c` is zeros by
# construction and checked here. Grouped periodic HSGP is not emitted by the
# compiler; a `by=` keyword reaching this path refuses rather than
# misrouting into the grouped squared-exponential layout.
function _adaptive_periodic_hsgp_block!(
        out, descriptor, plan, unc_names, logical, entry, kw)
    haskey(kw, :by) && error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` on predictor " *
        "`$logical` carries `by=`; the grouped periodic basis is not " *
        "implemented.",
    )

    weights = brm_term_coordinates(
        descriptor, logical, unc_names;
        term=entry.term, parameter=:basis_weights,
    )
    owner = weights.output.declaration
    isnothing(owner) && error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` basis weights " *
        "have no compiler declaration owner.",
    )
    if owner.family isa Symbol
        owner.family === :_sb_hsgp_periodic || error(
            "BRM adaptive centering: periodic HSGP `$(entry.term)` resolved " *
            "to unsupported emitted family `$(owner.family)`.",
        )
    end
    isempty(weights.output.constraints) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` basis weights " *
        "are constrained, so they are not the emitted standard-normal " *
        "`beta_raw` coordinates this transform requires.",
    )

    rho, sigma = _adaptive_hsgp_hypers(descriptor, logical, unc_names, entry, owner)
    length(rho.coordinates) == 1 || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$(length(rho.coordinates)) length-scale coordinates; the cosine/sine " *
        "spectrum needs exactly one.",
    )

    harmonics_key = get(owner.keywords, :harmonics, nothing)
    harmonics_key isa Symbol || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` declaration " *
        "does not expose its compiler-owned `harmonics` data binding.",
    )
    harmonics_raw = get(plan.data, harmonics_key, nothing)
    harmonics_raw isa AbstractVector{<:Real} || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` compiler data " *
        "`$harmonics_key` is not a real harmonic-index vector.",
    )
    harmonics = Vector{Float64}(harmonics_raw)
    n_weights = length(weights.coordinates)
    length(harmonics) == n_weights || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$n_weights basis weights, but `$harmonics_key` carries " *
        "$(length(harmonics)) harmonic indices.",
    )
    iseven(n_weights) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` owns " *
        "$n_weights basis weights; the cosine/sine basis needs an even count.",
    )
    harmonics == _brm_hsgp_periodic_harmonics(n_weights ÷ 2) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` compiler data " *
        "`$harmonics_key` is not the cosine/sine harmonic index vector; " *
        "refusing crossed term metadata.",
    )

    target_c = _brm_hsgp_centeredness(kw, n_weights)
    all(iszero, target_c) || error(
        "BRM adaptive centering: periodic HSGP `$(entry.term)` carries a " *
        "non-zero compiled centeredness; partial centering supports only " *
        "the exp_quad HSGP spectrum.",
    )
    push!(out, _HSGPAdaptiveCenteringBlock(
        logical, entry.term, target_c,
        collect(weights.coordinates),
        rho.coordinates, rho.lower, rho.upper,
        only(sigma.coordinates), only(sigma.lower), only(sigma.upper),
        Matrix{Float64}(undef, 0, 0), harmonics,
    ))
    out
end

# The grouped term's per-group weight block and its flat standardized weight
# carrier: the block `logical`'s own occurrence draws, read from the emitter's
# block plan (`_sb_group_block_plan`), so a repeat on another predictor
# resolves to that predictor's block.
function _adaptive_grouped_hsgp_weights(brmi, logical, entry)
    base = _sb_group_block_base(
        only(_sb_structured_fields(_sb_term_group_block(hsgp, entry.value), hsgp)),
        entry.value)
    suffixes = unique(gb.suffix for gb in
                      _sb_group_block_plan(_sb_collect_group_block_terms(brmi))
                      if gb.key === logical && gb.base === base)
    length(suffixes) == 1 || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` on predictor " *
        "`$logical` matches $(length(suffixes)) per-group weight blocks; " *
        "refusing an ambiguous weight layout.",
    )
    Symbol(:b_, only(suffixes)), Symbol(:zflat_, only(suffixes))
end

# Grouped squared-exponential HSGP weights for online adaptation. Spectral
# hyperparameters stay shared across groups, so one term contributes one
# ordinary `_HSGPAdaptiveCenteringBlock` per group level: each block's
# `effects` are that level's basis-weight coordinates and every block shares
# the term's per-basis compiled frame, length scales, marginal SD, and
# spectral frequencies. The flat index rule mirrors the emitted
# `to_matrix(zflat, n_basis, n_groups)'` layout: flat position `(g-1)*B+b`
# addresses frequency `b` in level `g`.
function _adaptive_grouped_hsgp_blocks!(
        out, descriptor, plan, pos, unc_names, logical, entry, kw)
    # A term whose levels are correlated or scaled per level is mapped across
    # its levels (`_adaptive_hsgp_level_centering_blocks`), not here.
    _adaptive_hsgp_level_mapped(plan, logical, entry, kw) && return out
    K, _ = _sb_hsgp_options(kw, length(getargs(entry.value)))
    B = prod(K)
    gname = name(kw[:by])
    n_key = Symbol(:n_, gname)
    G = get(plan.data, n_key, nothing)
    G isa Integer || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` on predictor " *
        "`$logical` has no integer group count under `$n_key`; refusing " *
        "an ambiguous weight layout.",
    )
    block, flat_name = _adaptive_grouped_hsgp_weights(plan.parent, logical, entry)
    flat_names = ["$(flat_name).$i" for i in 1:(G*B)]
    flat = _adaptive_named_indices(pos, flat_names, entry.term,
        "grouped HSGP basis weights")

    owner = _brm_term_owner_output(descriptor, logical, entry).declaration
    if owner.family isa Symbol
        owner.family in (:_sb_hsgp_by, :_sb_hsgp_by_aniso) || error(
            "BRM adaptive centering: grouped HSGP `$(entry.term)` resolved to " *
            "unsupported emitted family `$(owner.family)`.",
        )
    end
    rho, sigma = _adaptive_hsgp_hypers(descriptor, logical, unc_names, entry, owner)
    get(owner.keywords, :beta, nothing) === block || error(
        "BRM adaptive centering: grouped HSGP `$(entry.term)` declaration " *
        "does not carry the expected per-group weight block; refusing " *
        "crossed term metadata.",
    )
    omega2 = _brm_hsgp_omega2(plan, owner,
        "BRM adaptive centering: grouped HSGP `$(entry.term)`",
        B, length(rho.coordinates))
    target_c = _brm_hsgp_centeredness(kw, B)
    for g in 1:G
        push!(out, _HSGPAdaptiveCenteringBlock(
            logical, entry.term, target_c,
            flat[(g-1)*B+1:g*B],
            rho.coordinates, rho.lower, rho.upper,
            only(sigma.coordinates), only(sigma.lower), only(sigma.upper),
            omega2, Float64[],
        ))
    end
    out
end

# ---- Grouped HSGPs mapped across their levels ------------------------------
#
# `hsgp(x...; by=g, cor=Omega)` draws basis column `b` of its levels x basis
# weights as `w_b = diag(s_b) * L * z_b`: `z_b` the column's standardized
# coordinates (one per level), `L` the lower Cholesky factor of the level
# correlation and `s_b[g]` level `g`'s spectral scale `sqrt(S_g(omega_b))`
# (snag adaptive-centeri-11062238). Per basis function that is a correlated
# block whose margins are the levels, with factor `C_b = diag(s_b) * L`, so the
# levels of one column adapt exactly like the margins of one group of a
# correlated random-effect block: level `k` is centred on its conditional mean
# given levels `1..k-1` (the grouping's level order). A hyper-predictor
# (`log(sd(hsgp(x))) ~ 1 + (1 | g)`, or the length scale's) gives each level
# its own amplitude or length scale, so `s_b[g]` is per level; without `cor=`
# the levels are independent (`L = I`) and each weight is a scalar cell around
# 0 with its own level's scale. A grouped term with neither keeps the shared
# per-level blocks of `_adaptive_grouped_hsgp_blocks!`.

# One spectral hyperparameter of such a term (its marginal SD or length
# scale): shared, one coordinate per axis or exactly one (coordinate `0` a
# known value, both bounds at it), or predicted per level, where level `g`'s
# log value is `intercept + sd * z[g]` (each piece absent as `0`/empty).
struct _HSGPLevelScale
    predicted::Bool
    coordinates::Vector{Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    intercept::Int
    sd::Int
    sd_lower::Float64
    sd_upper::Float64
    z::Vector{Int}
end

_HSGPLevelScale(shared::NamedTuple) = _HSGPLevelScale(false, shared.coordinates,
    shared.lower, shared.upper, 0, 0, 0.0, Inf, Int[])

struct _HSGPLevelAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    group::Symbol
    # The grouping's labels in its index order, which is the order of the
    # triangular map.
    levels::Vector{Any}
    # `effects[g, b]`: the standardized weight of level `g`, basis `b`
    # (flat carrier position `(g - 1) * n_basis + b`).
    effects::Matrix{Int}
    # `:none` (independent levels), `:parameter` (a sampled `LKJCholesky`,
    # whose Stan `cholesky_factor_corr` coordinates are `cholesky_free`) or
    # `:data` (a known correlation, its canonical partial correlations in
    # `cholesky_cpc`).
    correlation::Symbol
    cholesky_free::Vector{Int}
    cholesky_cpc::Vector{Float64}
    sigma::_HSGPLevelScale
    rho::_HSGPLevelScale
    # A predicted length scale's validity floor, `fmax(rho[g], rho_lower)` in
    # the compiled spectrum; unused for a shared one, whose bound carries it.
    rho_floor::Float64
    omega2::Matrix{Float64}
end

Base.show(io::IO, b::_HSGPLevelAdaptiveCenteringBlock) = print(
    io,
    "HSGPLevelAdaptiveCenteringBlock(", b.logical, ", ", b.term, ", ",
    size(b.effects, 1), " levels of ", b.group, " × ", size(b.effects, 2),
    " basis weights, correlation=", b.correlation, ")",
)

# Whether a grouped exp-quad term is mapped across its levels: its `cor=` is a
# constant level correlation, or a hyper-predictor scales its levels. A
# multigroup coherence is refused: its level correlation depends on the basis
# frequency and the length scale, or its rows are an inducing grid.
function _adaptive_hsgp_level_mapped(plan, logical, entry, kw)
    haskey(kw, :by) || return false
    predicted = !isempty(_brm_term_hyper_plans(plan, logical, entry.value))
    haskey(kw, :cor) || return predicted
    lc = _adaptive_hsgp_level_correlation(logical, entry, kw)
    lc isa _BRMLevelCorrelation && return true
    error("BRM adaptive centering: grouped HSGP `$(entry.term)` on predictor ",
          "`$logical` correlates the levels of `$(name(kw[:by]))` through ",
          "`cor = multigroup(...)`, whose level correlation changes with each ",
          "basis frequency (or whose weights sit on a shared grid of inducing ",
          "rows), so a per-basis level map with one correlation factor does not ",
          "describe it; adaptive centering of such a term is not implemented, ",
          "and it is not silently left fixed either. `cor = Omega` (a sampled ",
          "`LKJCholesky`) and a known correlation matrix are supported.")
end

_adaptive_hsgp_level_correlation(logical, entry, kw) = _brm_level_correlation(
    kw[:cor]; what="hsgp(...; by=$(name(kw[:by])), cor=...) on `$logical`",
    prefix="BRM adaptive centering")

"""
    _adaptive_hsgp_level_centering_blocks(model, unc_names)

Resolve every grouped squared-exponential HSGP whose levels are correlated
(`cor = Omega` or a known correlation matrix) or scaled per level (a
hyper-predictor on its `sd` or `length_scale`) in an `SBBRMI` or
`GenerativePlan`: one `_HSGPLevelAdaptiveCenteringBlock` per term, read from
the descriptor's term roles, the emitter's weight-block plan and the term
declaration's compiler data (`omega2`, `rho_lower`, `L_level`), never from a
parsed name. The companion of [`_adaptive_hsgp_centering_blocks`](@ref), which
resolves every other HSGP; together they cover each term exactly once. A term
emitted in generated quantities is skipped, and a multigroup coherence, a
non-zero compiled `centeredness` and declaration drift raise.
"""
function _adaptive_hsgp_level_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    pos = _ranef_name_positions(unc_names)
    out = _HSGPLevelAdaptiveCenteringBlock[]
    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === hsgp || continue
            _adaptive_term_generated(hsgp, descriptor, logical, entry) && continue
            kw = getkwargs(entry.value)
            _sb_gp_cov(kw, :hsgp) === :exp_quad || continue
            _adaptive_hsgp_level_mapped(plan, logical, entry, kw) || continue
            push!(out, _adaptive_hsgp_level_block(
                descriptor, plan, pos, unc_names, logical, entry, kw))
        end
    end
    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects))
    end
    allunique(claimed) || error(
        "BRM adaptive centering: emitted HSGP level blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.")
    out
end

function _adaptive_hsgp_level_block(descriptor, plan, pos, unc_names, logical,
                                    entry, kw)
    what = "BRM adaptive centering: grouped HSGP `$(entry.term)` on predictor `$logical`"
    K, _ = _sb_hsgp_options(kw, length(getargs(entry.value)))
    B = prod(K)
    gname = name(kw[:by])
    G = get(plan.data, Symbol(:n_, gname), nothing)
    G isa Integer || error(
        "$what has no integer group count under `n_$gname`; refusing an " *
        "ambiguous weight layout.")
    block, flat_name = _adaptive_grouped_hsgp_weights(plan.parent, logical, entry)
    flat = _adaptive_named_indices(pos, ["$(flat_name).$i" for i in 1:(G*B)],
        entry.term, "grouped HSGP basis weights")
    owner = _brm_term_owner_output(descriptor, logical, entry).declaration
    get(owner.keywords, :beta, nothing) === block || error(
        "$what declaration does not carry the expected per-group weight " *
        "block; refusing crossed term metadata.")
    all(iszero, _brm_hsgp_centeredness(kw, B)) || error(
        "$what carries a non-zero compiled `centeredness`; the grouped weights " *
        "are emitted standardized.")

    plans = _brm_term_hyper_plans(plan, logical, entry.value)
    sigma_plan = _sb_hyper_plan_for(plans, :sd)
    rho_plan = _sb_hyper_plan_for(plans, :length_scale)
    sigma = isnothing(sigma_plan) ?
        _HSGPLevelScale(_adaptive_hsgp_hyper_coordinates(descriptor, logical,
            unc_names, entry, owner, :sd, "marginal SD")) :
        _adaptive_hsgp_level_predicted(descriptor, logical, unc_names, entry, owner,
            sigma_plan, :sd, G)
    rho = isnothing(rho_plan) ?
        _HSGPLevelScale(_adaptive_hsgp_hyper_coordinates(descriptor, logical,
            unc_names, entry, owner, :length_scale, "length scale")) :
        _adaptive_hsgp_level_predicted(descriptor, logical, unc_names, entry, owner,
            rho_plan, :length_scale, G)
    sigma.predicted || length(sigma.coordinates) == 1 || error(
        "$what must have one marginal-SD coordinate.")
    rho.predicted || !isempty(rho.coordinates) || error(
        "$what has no length-scale coordinate.")
    omega2 = _brm_hsgp_omega2(plan, owner, what, B,
        rho.predicted ? 1 : length(rho.coordinates))
    rho_floor = 0.0
    if rho.predicted
        floor_key = get(owner.keywords, :rho_lower, nothing)
        floor = floor_key isa Symbol ? get(plan.data, floor_key, nothing) : floor_key
        floor isa Real || error(
            "$what declaration does not expose its predicted length scales' " *
            "scalar `rho_lower` floor.")
        rho_floor = Float64(floor)
    end

    correlation, cholesky_free, cholesky_cpc = :none, Int[], Float64[]
    if haskey(kw, :cor)
        lc = _adaptive_hsgp_level_correlation(logical, entry, kw)
        lc.K == G || error("$what correlates $(lc.K) levels, but `$gname` has $G.")
        factor = get(owner.keywords, :L_level, nothing)
        if lc.kind === :parameter
            factor === lc.name || error(
                "$what declaration does not read the level correlation " *
                "`$(lc.name)`; refusing crossed term metadata.")
            correlation = :parameter
            cholesky_free = _adaptive_named_indices(pos,
                ["$(lc.name).$i" for i in 1:(G*(G-1)÷2)], lc.name, "Cholesky")
        else
            L = factor isa Symbol ? get(plan.data, factor, nothing) : nothing
            L isa AbstractMatrix{<:Real} && size(L) == (G, G) || error(
                "$what declaration does not expose its known level correlation " *
                "factor as a $G × $G data matrix.")
            correlation = :data
            cholesky_cpc = _adaptive_cholesky_cpc(Matrix{Float64}(L))
        end
    end
    _HSGPLevelAdaptiveCenteringBlock(logical, entry.term, gname,
        _grouped_hsgp_levels(plan, plan.parent, gname),
        permutedims(reshape(flat, B, G)), correlation, cholesky_free, cholesky_cpc,
        sigma, rho, rho_floor, omega2)
end

# A hyper-predicted SD or length scale: level `g`'s log value is
# `intercept + sd * z[g]` (`_sb_hyper_param_stmts!`), each piece the plan's,
# read through the term's hyper roles (`:sd_intercept`, `:sd_ranef_sd`,
# `:sd_ranef_z`, and the `length_scale_` ones).
function _adaptive_hsgp_level_predicted(descriptor, logical, unc_names, entry, owner,
                                        hyper_plan, hyper, G)
    prefix = hyper === :sd ? "sd" : "length_scale"
    what = hyper === :sd ? "marginal-SD" : "length-scale"
    resolve(role) = _adaptive_same_hsgp_owner(brm_term_coordinates(descriptor,
        logical, unc_names; term=entry.term, parameter=Symbol(prefix, :_, role)),
        owner, "$what $role")
    intercept = 0
    if hyper_plan.intercept
        r = resolve(:intercept)
        isempty(r.output.constraints) || error(
            "BRM adaptive centering: HSGP `$(entry.term)` $what intercept is " *
            "constrained; it is read as the log value's unconstrained intercept.")
        intercept = only(r.coordinates)
    end
    sd, sd_lower, sd_upper, z = 0, 0.0, Inf, Int[]
    if !isempty(hyper_plan.ranefs)
        r = resolve(:ranef_sd)
        lower, upper = _adaptive_bounds(descriptor, r.output, 1, entry.term,
                                        "$what level-effect SD")
        sd, sd_lower, sd_upper = only(r.coordinates), only(lower), only(upper)
        zr = resolve(:ranef_z)
        isempty(zr.output.constraints) && length(zr.coordinates) == G || error(
            "BRM adaptive centering: HSGP `$(entry.term)` $what level effects " *
            "resolve to $(length(zr.coordinates)) constrained coordinates for " *
            "$G levels.")
        z = collect(zr.coordinates)
    end
    _HSGPLevelScale(true, Int[], Float64[], Float64[], intercept, sd, sd_lower,
                    sd_upper, z)
end

# A predicted hyper's log value at one level, `intercept + sd * x[z]`, each
# piece absent as `0` (`_sb_hyper_param_stmts!`). Shared by the reference
# below and the WarmupHMC extension's level cells.
@inline function _adaptive_hsgp_predicted_log(x, intercept, sd, sd_lower, sd_upper, z)
    v = intercept == 0 ? zero(eltype(x)) : x[intercept]
    sd == 0 && return v
    v + _adaptive_constrain(x[sd], sd_lower, sd_upper) * x[z]
end

# Level `g`'s marginal SD (on the log scale) and length scale on `axis` at the
# unconstrained point `x`. A predicted length scale is floored at `rho_lower`,
# as `brm_hsgp_by_hyper_sqrt_spd` floors it.
function _adaptive_hsgp_level_log_sigma(x, block::_HSGPLevelAdaptiveCenteringBlock, g)
    s = block.sigma
    s.predicted || return log(_adaptive_hsgp_hyper(x, only(s.coordinates),
                                                   only(s.lower), only(s.upper)))
    _adaptive_hsgp_predicted_log(x, s.intercept, s.sd, s.sd_lower, s.sd_upper,
                                 s.sd == 0 ? 0 : s.z[g])
end

function _adaptive_hsgp_level_rho(x, block::_HSGPLevelAdaptiveCenteringBlock, g, axis)
    s = block.rho
    s.predicted && return max(exp(_adaptive_hsgp_predicted_log(x, s.intercept, s.sd,
        s.sd_lower, s.sd_upper, s.sd == 0 ? 0 : s.z[g])), block.rho_floor)
    j = length(s.coordinates) == 1 ? 1 : axis
    _adaptive_hsgp_hyper(x, s.coordinates[j], s.lower[j], s.upper[j])
end

# Level `g`'s log spectral scale `log sqrt(S_g(omega_b))` for basis `b`, as
# `brm_hsgp_sqrt_spd` (`brm_hsgp_by_hyper_sqrt_spd` under a hyper-predictor)
# computes it, with the adaptive log-scale floor of `_adaptive_hsgp_log_scale`.
function _adaptive_hsgp_level_log_scale(x::AbstractVector,
                                        block::_HSGPLevelAdaptiveCenteringBlock, g, b)
    value = _adaptive_hsgp_level_log_sigma(x, block, g)
    for axis in axes(block.omega2, 2)
        value = _adaptive_hsgp_axis_log_scale(value,
            _adaptive_hsgp_level_rho(x, block, g, axis), block.omega2[b, axis])
    end
    _adaptive_hsgp_floor(value)
end

# Every coordinate a level block reads without rewriting it: its correlation,
# scales and their predictors.
_adaptive_read_coordinates(block::_HSGPLevelAdaptiveCenteringBlock) = filter(!iszero,
    vcat(block.cholesky_free,
         (vcat(s.coordinates, s.intercept, s.sd, s.z) for s in (block.sigma, block.rho))...))

# The level correlation's lower Cholesky factor at `x`: Stan's
# `cholesky_factor_corr` of the sampled coordinates, a known factor rebuilt
# from its partial correlations, or the identity for independent levels.
function _adaptive_hsgp_level_correlation_factor(x::AbstractVector,
                                                 block::_HSGPLevelAdaptiveCenteringBlock)
    G = size(block.effects, 1)
    block.correlation === :parameter &&
        return _adaptive_cholesky_corr(x[block.cholesky_free], G)
    block.correlation === :data &&
        return _adaptive_cholesky_from_cpc(identity, block.cholesky_cpc, G)
    Matrix{eltype(x)}(I, G, G)
end

# Basis `b`'s factor `C_b = diag(s_b) * L` over the levels at `x`.
function _adaptive_hsgp_level_factor(x::AbstractVector,
                                     block::_HSGPLevelAdaptiveCenteringBlock, b)
    L = _adaptive_hsgp_level_correlation_factor(x, block)
    s = [exp(_adaptive_hsgp_level_log_scale(x, block, g, b)) for g in axes(L, 1)]
    s .* L
end

# The compiler-owned squared-exponential frequencies of an ungrouped or
# grouped HSGP: `omega2[b, axis]` for basis row `b`. `brm_hsgp_sqrt_spd` reads
# one length scale per axis; an isotropic term (`_sb_hsgp`, `_sb_hsgp_partial`,
# `_sb_hsgp_by`) samples one `rho_iso` and passes `rep_vector(rho_iso, n_axes)`,
# so a term owns either one length scale per `omega2` column or exactly one,
# which every column reads. `what` prefixes each error with the caller.
function _brm_hsgp_omega2(plan, owner, what, n_weights, n_scales)
    omega_key = get(owner.keywords, :omega2, nothing)
    omega_key isa Symbol || error(
        "$what declaration does not expose its compiler-owned `omega2` data " *
        "binding.",
    )
    omega_raw = get(plan.data, omega_key, nothing)
    omega_raw isa AbstractMatrix{<:Real} || error(
        "$what compiler data `$omega_key` is not a real spectral-frequency matrix.",
    )
    omega2 = Matrix{Float64}(omega_raw)
    size(omega2, 1) == n_weights && n_scales in (1, size(omega2, 2)) || error(
        "$what owns $n_weights basis weights and $n_scales length scales, but " *
        "`$omega_key` has size $(size(omega2)); expected $n_weights rows and " *
        "one length scale per axis or a single isotropic one.",
    )
    omega2
end

# The log-scale an HSGP cell reads is its exact spectral weight while that is
# above `exp(-110)`, and a smooth floor at `exp(-150)` below. A long length
# scale drives a basis weight's exact log-scale towards `-rho^2 * omega2 / 4`
# (measured -473 on a 2-D field at `rho = 11`, snag adaptive-hsgp-ce-024ae486).
# A cell partially centred on such a scale has a source coordinate `s^c * z`
# whose spread underflows to zero, and its candidate frames underflow and
# overflow to NaN losses. Floored, every frame of such a weight is a fixed
# rescaling of `z` (the non-centred geometry), and `exp(±2c * log_scale)` stays
# a normal Float64 for every control `c`. Because those frames cannot be told
# apart, the HSGP scoring frame scores a floored observation in the non-centred
# frame for every candidate, so a weight scored only on the floor keeps
# `c = 0` (snag adaptive-centeri-eca5a77c). The transform is exact for any
# function of coordinates it does not rewrite, so the floor moves coordinates,
# never the posterior. Between the two bounds the softplus differs from the
# exact value by less than `exp(-40)`; above `exp(-110)` they are identical.
const _ADAPTIVE_HSGP_LOG_SCALE_FLOOR = -150.0
const _ADAPTIVE_HSGP_LOG_SCALE_EXACT = _ADAPTIVE_HSGP_LOG_SCALE_FLOOR + 40.0

_adaptive_hsgp_floor(value) = value > _ADAPTIVE_HSGP_LOG_SCALE_EXACT ? value :
    _ADAPTIVE_HSGP_LOG_SCALE_FLOOR +
    _adaptive_softplus(value - _ADAPTIVE_HSGP_LOG_SCALE_FLOOR)

function _adaptive_hsgp_log_scale(x::AbstractVector,
                                  block::_HSGPAdaptiveCenteringBlock,
                                  basis::Int)
    1 <= basis <= length(block.effects) || throw(BoundsError(block.effects, basis))
    if !isempty(block.harmonics)
        length(block.length_scales) == 1 || error(
            "BRM adaptive centering: periodic HSGP `$(block.term)` owns " *
            "$(length(block.length_scales)) length scales; the cosine/sine " *
            "spectrum needs exactly one.",
        )
        sigma = _adaptive_hsgp_hyper(x, block.sd, block.sd_lower, block.sd_upper)
        rho = _adaptive_hsgp_hyper(x, block.length_scales[1],
            block.length_scale_lower[1], block.length_scale_upper[1])
        return _adaptive_hsgp_floor(
            _brm_hsgp_periodic_log_scale(block.harmonics[basis], sigma, rho))
    end
    sigma = _adaptive_hsgp_hyper(x, block.sd, block.sd_lower, block.sd_upper)
    value = log(sigma)
    iso = length(block.length_scales) == 1
    for axis in axes(block.omega2, 2)
        j = iso ? 1 : axis
        rho = _adaptive_hsgp_hyper(x, block.length_scales[j],
            block.length_scale_lower[j], block.length_scale_upper[j])
        value = _adaptive_hsgp_axis_log_scale(value, rho, block.omega2[basis, axis])
    end
    _adaptive_hsgp_floor(value)
end

# One axis's term of an exp-quad HSGP's log spectral scale, `log(sqrt(2 pi) *
# rho) / 2 - rho^2 * omega2 / 4` (`brm_hsgp_log_sqrt_spd`), added to `value`
# one summand at a time. Shared by the per-basis and the per-level HSGP cells.
@inline _adaptive_hsgp_axis_log_scale(value, rho, omega2) =
    (value + 0.5 * log(rho * 2.5066282746310002)) - 0.25 * rho * rho * omega2

# The cdar path has its own metadata type for the same reason the HSGP path
# does: `AdaptiveCenteringBlock` feeds the small-block Enzyme specialization
# for ordinary random effects, and walk cells must not perturb that dispatch.
# Cells are scalar and independent like HSGP basis weights — location zero and
# one marginal prior spread each — so per-pair transport stays O(1) and the
# whole walk stays linear, unlike a time-triangular map whose per-cell history
# reread would cost O((P*W)^2) per gradient with no legal Enzyme cache.
struct _CDARAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    n_groups::Int
    n_steps::Int
    # Column-major `eta` unconstrained indices: position `p + (w - 1) * P`
    # addresses group `p`, step `w`, matching the emitted `eta.1`, ... order.
    effects::Vector{Int}
    # Compiler-declared bounds (`_adaptive_bounds`): the marginal SD's upper
    # bound is `Inf` unless declared; the persistence is always an interval
    # inside `[0, 1]`.
    sigma::Int
    sigma_lower::Float64
    sigma_upper::Float64
    rho::Int
    rho_lower::Float64
    rho_upper::Float64
    # Frozen per-group marginal variances `C[p, p]` of `C = L * L'`.
    cdiag::Vector{Float64}
end

Base.show(io::IO, b::_CDARAdaptiveCenteringBlock) = print(
    io,
    "CDARAdaptiveCenteringBlock(", b.logical, ", ", b.term,
    ", ", b.n_groups, "×", b.n_steps, " walk cells)",
)

function _adaptive_cdar_physical(block::_CDARAdaptiveCenteringBlock, x::AbstractVector)
    sigma = _adaptive_constrain(x[block.sigma], block.sigma_lower, block.sigma_upper)
    rho = _adaptive_constrain(x[block.rho], block.rho_lower, block.rho_upper)
    sigma, rho
end

function _adaptive_cdar_log_scale(x::AbstractVector,
                                  block::_CDARAdaptiveCenteringBlock,
                                  pair::Int)
    1 <= pair <= length(block.effects) || throw(BoundsError(block.effects, pair))
    p = mod(pair - 1, block.n_groups) + 1
    w = div(pair - 1, block.n_groups) + 1
    sigma, rho = _adaptive_cdar_physical(block, x)
    isfinite(sigma) && sigma > 0 || error(
        "BRM adaptive centering: cdar `$(block.term)` marginal scale is not " *
        "finite-positive at this position.",
    )
    0 <= rho <= 1 || error(
        "BRM adaptive centering: cdar `$(block.term)` persistence left " *
        "[0, 1] at this position.",
    )
    one_minus_rho2 = 1 - rho * rho
    spread = if one_minus_rho2 <= 1e-12
        Float64(w)
    else
        exp(log1p(-rho^(2w)) - log(one_minus_rho2))
    end
    log(sigma) + 0.5 * log(block.cdiag[p]) + 0.5 * log(spread)
end

function _adaptive_cdar_rho_bounds(descriptor, output::BRMOutput, owner)
    lower, upper = only.(_adaptive_bounds(descriptor, output, 1, owner, "persistence"))
    0 <= lower && upper <= 1 || error(
        "BRM adaptive centering: cdar `$owner` persistence is declared on " *
        "[$lower, $upper], outside the stationary interval [0, 1].",
    )
    lower, upper
end

"""
    _adaptive_cdar_centering_blocks(model, unc_names)

Resolve the compiled coordinates and frozen walk geometry for every
`cdar(step; by=group, cor=C)` term in an `SBBRMI` or `GenerativePlan`.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref),
following the same descriptor-to-declaration route as
[`_adaptive_hsgp_centering_blocks`](@ref). Each of the `P * W` innovations is
one scalar cell with zero location and its marginal prior spread
`sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`; `c=0` is the emitted
`eta` frame. The marginal SD may be `<lower=a>` or `<lower=a, upper=b>` and the
persistence any declared interval inside `[0, 1]` (an explicit `Uniform(a, b)`
prior narrows either); the cells read both through Stan's own transform. The
metadata is intentionally fail-closed: a persistence interval leaving `[0, 1]`,
a scale without a finite lower bound, a missing or misshapen frozen factor, and
declaration/artifact coordinate drift all raise before a reparametrizer is
built. A walk emitted in generated quantities (no observed likelihood reaches
it) has no sampler coordinate and is skipped.
"""
function _adaptive_cdar_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    out = _CDARAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === cdar || continue
            _adaptive_term_generated(cdar, descriptor, logical, entry) && continue

            innovations = brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:innovations,
            )
            owner = innovations.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: cdar `$(entry.term)` innovations " *
                "have no compiler declaration owner.",
            )
            if owner.family isa Symbol
                owner.family === :_sb_cdar || error(
                    "BRM adaptive centering: cdar `$(entry.term)` resolved to " *
                    "unsupported emitted family `$(owner.family)`.",
                )
            end
            isempty(innovations.output.constraints) || error(
                "BRM adaptive centering: cdar `$(entry.term)` innovations " *
                "are constrained, so they are not the emitted standard-normal " *
                "`eta` coordinates this transform requires.",
            )

            sigma = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:sd,
            ), owner, "marginal SD")
            rho = _adaptive_same_hsgp_owner(brm_term_coordinates(
                descriptor, logical, unc_names;
                term=entry.term, parameter=:ar,
            ), owner, "persistence")
            length(sigma.coordinates) == 1 || error(
                "BRM adaptive centering: cdar `$(entry.term)` must have one marginal-SD coordinate.",
            )
            length(rho.coordinates) == 1 || error(
                "BRM adaptive centering: cdar `$(entry.term)` must have one persistence coordinate.",
            )

            l_key = get(owner.keywords, :L, nothing)
            l_key isa Symbol || error(
                "BRM adaptive centering: cdar `$(entry.term)` declaration does " *
                "not expose its frozen correlation-factor `L` data binding.",
            )
            l_raw = get(plan.data, l_key, nothing)
            l_raw isa AbstractMatrix{<:Real} || error(
                "BRM adaptive centering: cdar `$(entry.term)` compiler data " *
                "`$l_key` is not a real correlation-factor matrix.",
            )
            ng_key = get(owner.keywords, :n_groups, nothing)
            ns_key = get(owner.keywords, :n_steps, nothing)
            n_groups = ng_key isa Symbol ? get(plan.data, ng_key, nothing) : nothing
            n_steps = ns_key isa Symbol ? get(plan.data, ns_key, nothing) : nothing
            n_groups isa Integer && n_steps isa Integer || error(
                "BRM adaptive centering: cdar `$(entry.term)` declaration does " *
                "not expose integer group/step counts.",
            )
            P, W = Int(n_groups), Int(n_steps)
            L = Matrix{Float64}(l_raw)
            size(L) == (P, P) || error(
                "BRM adaptive centering: cdar `$(entry.term)` owns a " *
                "$(size(L)) factor for $P groups.",
            )
            length(innovations.coordinates) == P * W || error(
                "BRM adaptive centering: cdar `$(entry.term)` owns " *
                "$(length(innovations.coordinates)) innovations for a " *
                "$P×$W walk.",
            )
            cdiag = vec(sum(abs2, L; dims=2))
            all(c -> isfinite(c) && c > 0, cdiag) || error(
                "BRM adaptive centering: cdar `$(entry.term)` frozen factor " *
                "has a non-finite or non-positive marginal variance.",
            )

            sigma_lower, sigma_upper = only.(_adaptive_bounds(
                descriptor, sigma.output, 1, entry.term, "marginal SD",
            ))
            rho_lower, rho_upper =
                _adaptive_cdar_rho_bounds(descriptor, rho.output, entry.term)
            push!(out, _CDARAdaptiveCenteringBlock(
                logical, entry.term, P, W,
                collect(innovations.coordinates),
                only(sigma.coordinates), sigma_lower, sigma_upper,
                only(rho.coordinates), rho_lower, rho_upper,
                cdiag,
            ))
        end
    end

    claimed = Int[]
    for block in out
        append!(claimed, block.effects)
        push!(claimed, block.sigma)
        push!(claimed, block.rho)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted cdar blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

# ---- Areal car fields ---------------------------------------------------------
#
# `car(area; edges, type=:icar)` adds `sigma * phi[struct_idx[k]]` for area `k`,
# `type=:bym2` adds `b[k] = a_k * phi[struct_idx[k]] + s_k * theta[k]` with
# `a_k = sigma * sqrt(rho / s_c)` and `s_k = sigma * sqrt(1 - rho)` (an island,
# `struct_idx[k] == 0`, has `b[k] = sigma * theta[k]`), see `_sb_car_bym2`.
# Both `phi` and `theta` are emitted standardized, so the compiled frame is
# `c=0` throughout. Under `:icar` every area with neighbours is one cell, its
# `phi` (location 0, scale `sigma`), so `c=1` is its effect `sigma * phi`; the
# ICAR density couples neighbouring cells, which leaves the transform exact: it
# only rescales each `phi` coordinate. Under `:bym2` every area is one cell, its
# `theta` (scale `s_k`, location `intercept + a_k * phi`, see
# `_adaptive_car_intercept`), so `c=1` is the area's total `intercept + b[k]`
# (Gelfand, Sahu and Carlin 1995), and `phi` stays the emitted coordinate. A
# BYM2 `phi` cell would be centred on `a_k`, which vanishes with `rho`: on a
# mostly unstructured field the per-cell score partly centred such cells, and
# the funnel they opened at `rho -> 0` stopped a chain (910 of 1000 draws
# divergent), while keeping `phi` standardized mixed `rho` better in both
# regimes measured.

struct _CARAdaptiveCenteringBlock
    logical::Symbol
    term::Symbol
    bym2::Bool
    # `phi` unconstrained indices in emitted order (areas with neighbours,
    # component by component).
    structured::Vector{Int}
    # `theta` unconstrained indices, one per area in sorted label order;
    # empty for `:icar`.
    unstructured::Vector{Int}
    # Per area, its position in `phi`, 0 for an island (the emitted `struct_idx`),
    # and per `phi` position, its area.
    struct_idx::Vector{Int}
    phi_area::Vector{Int}
    # Per `phi` position, `log(1 / s_c)` of its component (`:bym2`; zeros for
    # `:icar`, whose `phi` scale is `sigma`).
    log_inv_scale::Vector{Float64}
    # Compiler-declared bounds (`_adaptive_bounds`); `rho` is 0 for `:icar`.
    sigma::Int
    sigma_lower::Float64
    sigma_upper::Float64
    rho::Int
    rho_lower::Float64
    rho_upper::Float64
    # The predictor's sampled population intercept, which every `theta` cell's
    # location adds (`_adaptive_car_intercept`); 0 when it has none.
    intercept::Int
    # The fitted graph's area labels in sorted order (`effects` order).
    areas::Vector{Any}
end

Base.show(io::IO, b::_CARAdaptiveCenteringBlock) = print(
    io,
    "CARAdaptiveCenteringBlock(", b.logical, ", ", b.term, ", ",
    b.bym2 ? ":bym2, $(length(b.unstructured)) theta cells)" :
        ":icar, $(length(b.structured)) phi cells)",
)

# Cells of one block: under `:icar` the `phi` cells in emitted order, under
# `:bym2` one `theta` cell per area in sorted order.
_adaptive_car_n_cells(block::_CARAdaptiveCenteringBlock) =
    block.bym2 ? length(block.unstructured) : length(block.structured)
# The area of a cell, an index into `areas`.
_adaptive_car_area(block::_CARAdaptiveCenteringBlock, cell::Int) =
    block.bym2 ? cell : block.phi_area[cell]

# One cell as plain numbers. The differentiated accessors read only a vector of
# these, never the block: on Julia 1.10 an Enzyme reverse pass that reads a
# struct holding vectors corrupts GC once another thread collects (see
# `_block_location` in BayesianRegressionModelsWarmupHMCExt), and the joint
# wrapper's accessor crashed `test/adaptive_centering_threads.jl` while its car
# branch read the block. The accessors below are `@inline`: passing a cell to an
# outlined call aborted that gate's car-free `joint` case the same way, with or
# without the intercept read.
struct _CARAdaptiveCell
    index::Int
    # false: scale `sigma` (an `:icar` `phi` cell, an island's `theta` cell);
    # true: the `theta` cell of an area with neighbours, `sigma * sqrt(1 - rho)`.
    share::Bool
    sigma::Int
    sigma_lower::Float64
    sigma_upper::Float64
    rho::Int
    rho_lower::Float64
    rho_upper::Float64
    # A `theta` cell's location `intercept + sigma * sqrt(rho / s_c) * phi`: the
    # area's `phi` coordinate (0 for an island or a `phi` cell), `log(1 / s_c)`
    # of its component, and the intercept coordinate (0 when there is none).
    phi::Int
    log_inv_scale::Float64
    intercept::Int
end

function _adaptive_car_cell(block::_CARAdaptiveCenteringBlock, cell::Int)
    hyper = (block.sigma, block.sigma_lower, block.sigma_upper,
             block.rho, block.rho_lower, block.rho_upper)
    block.bym2 || return _CARAdaptiveCell(block.structured[cell], false, hyper...,
                                          0, 0.0, 0)
    j = block.struct_idx[cell]
    j == 0 && return _CARAdaptiveCell(block.unstructured[cell], false, hyper...,
                                      0, 0.0, block.intercept)
    _CARAdaptiveCell(block.unstructured[cell], true, hyper..., block.structured[j],
                     block.log_inv_scale[j], block.intercept)
end
_adaptive_car_index(block::_CARAdaptiveCenteringBlock, cell::Int) =
    _adaptive_car_cell(block, cell).index

# `(log(rho), log(1 - rho))` of a share declared on `[lower, upper]` inside
# `[0, 1]`, through Stan's `lub_constrain`. An end at 0 or 1 is taken in closed
# form, `log(inv_logit(x)) == -softplus(-x)`, so neither log reaches `-Inf` as
# the share approaches that end.
@inline function _adaptive_car_log_shares(x, lower, upper)
    rho = _adaptive_constrain(x, lower, upper)
    log_rho = iszero(lower) ? log(upper) - _adaptive_softplus(-x) : log(rho)
    log1m_rho = isone(upper) ? log(upper - lower) - _adaptive_softplus(x) :
        log1p(-rho)
    log_rho, log1m_rho
end

@inline _adaptive_car_log_sigma(x::AbstractVector, cell::_CARAdaptiveCell) =
    log(_adaptive_constrain(x[cell.sigma], cell.sigma_lower, cell.sigma_upper))

@inline function _adaptive_car_log_scale(x::AbstractVector, cell::_CARAdaptiveCell)
    log_sigma = _adaptive_car_log_sigma(x, cell)
    cell.share || return log_sigma
    _, log1m_rho = _adaptive_car_log_shares(x[cell.rho], cell.rho_lower, cell.rho_upper)
    log_sigma + log1m_rho / 2
end
_adaptive_car_log_scale(x::AbstractVector, block::_CARAdaptiveCenteringBlock,
                        cell::Int) = _adaptive_car_log_scale(x, _adaptive_car_cell(block, cell))

# A cell's location: 0 for a `phi` cell, the intercept plus the area's spatial
# component for a `theta` cell. It reads only coordinates no cell rewrites, so it
# is the same in every frame.
@inline function _adaptive_car_location(x::AbstractVector, cell::_CARAdaptiveCell)
    m = cell.intercept == 0 ? zero(eltype(x)) : x[cell.intercept]
    cell.phi == 0 && return m
    log_rho, _ = _adaptive_car_log_shares(x[cell.rho], cell.rho_lower, cell.rho_upper)
    m + exp(_adaptive_car_log_sigma(x, cell) + (log_rho + cell.log_inv_scale) / 2) *
        x[cell.phi]
end
_adaptive_car_location(x::AbstractVector, block::_CARAdaptiveCenteringBlock,
                       cell::Int) = _adaptive_car_location(x, _adaptive_car_cell(block, cell))

"""
    _adaptive_car_centering_blocks(model, unc_names)

Resolve the compiled coordinates and frozen graph numbering of every
`car(area; edges, type=:icar | :bym2)` term in an `SBBRMI` or `GenerativePlan`.

This is the backend-internal companion to [`adaptive_centering_blocks`](@ref),
following the same descriptor-to-declaration route as
[`_adaptive_cdar_centering_blocks`](@ref). Under `:icar` each area with
neighbours has one `phi` cell (location 0, scale `sigma`); under `:bym2` each
area has one `theta` cell with scale `sigma * sqrt(1 - rho)` (`sigma` for an
island) and location `intercept + sigma * sqrt(rho / s_c) * phi`, the
predictor's intercept plus the area's spatial component, so its `c=1`
coordinate is the area's total. The marginal SD may be
`<lower=a>` or `<lower=a, upper=b>` and the mixing share any declared interval
inside `[0, 1]`. The metadata is fail-closed: a missing or misnumbered
`struct_idx`, a non-positive scaling factor, a share interval leaving `[0, 1]`
and declaration/artifact coordinate drift all raise before a reparametrizer is
built. A field emitted in generated quantities (no observed likelihood reaches
it) has no sampler coordinate and is skipped.
"""
function _adaptive_car_centering_blocks(model, unc_names)
    descriptor = brm_descriptor(model)
    plan = descriptor.plan
    out = _CARAdaptiveCenteringBlock[]

    for predictor in linear_predictors(plan.parent)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(plan.parent, logical)
            getf(entry.value) === car || continue
            _adaptive_term_generated(car, descriptor, logical, entry) && continue
            bym2 = getkwargs(entry.value).type === :bym2
            term = entry.term
            coordinates(parameter) = brm_term_coordinates(
                descriptor, logical, unc_names; term, parameter)

            structured = coordinates(:structured)
            owner = structured.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: car `$term` structured field has no " *
                "compiler declaration owner.",
            )
            if owner.family isa Symbol
                owner.family === (bym2 ? :_sb_car_bym2 : :_sb_car_icar) || error(
                    "BRM adaptive centering: car `$term` resolved to " *
                    "unsupported emitted family `$(owner.family)`.",
                )
            end
            unstructured = bym2 ? _adaptive_same_hsgp_owner(
                coordinates(:unstructured), owner, "unstructured field") : nothing
            for field in (structured, unstructured)
                isnothing(field) || isempty(field.output.constraints) || error(
                    "BRM adaptive centering: car `$term` field " *
                    "`$(field.output.name)` is constrained, so it is not the " *
                    "emitted standardized coordinate this transform requires.",
                )
            end
            sigma = _adaptive_same_hsgp_owner(coordinates(:sd), owner, "marginal SD")
            rho = bym2 ? _adaptive_same_hsgp_owner(
                coordinates(:mixing), owner, "mixing share") : nothing
            length(sigma.coordinates) == 1 || error(
                "BRM adaptive centering: car `$term` must have one marginal-SD coordinate.",
            )
            isnothing(rho) || length(rho.coordinates) == 1 || error(
                "BRM adaptive centering: car `$term` must have one mixing-share coordinate.",
            )

            data(key) = (k = get(owner.keywords, key, nothing);
                         k isa Symbol ? get(plan.data, k, nothing) : nothing)
            struct_idx = data(:struct_idx)
            struct_idx isa AbstractVector{<:Integer} || error(
                "BRM adaptive centering: car `$term` declaration does not expose " *
                "its integer `struct_idx` data binding.",
            )
            struct_idx = collect(Int, struct_idx)
            n_struct = length(structured.coordinates)
            sort!(filter(>(0), struct_idx)) == 1:n_struct && all(>=(0), struct_idx) ||
                error("BRM adaptive centering: car `$term` `struct_idx` does not " *
                      "number its $n_struct structured areas once each.")
            area_of = zeros(Int, n_struct)
            for (k, j) in enumerate(struct_idx)
                j > 0 && (area_of[j] = k)
            end
            graph_entry = get(plan.preproc, get(owner.keywords, :row_idx, nothing), nothing)
            areas = isnothing(graph_entry) || graph_entry.kind !== :car ? nothing :
                graph_entry.const_.graph.nodes
            !isnothing(areas) && length(areas) == length(struct_idx) || error(
                "BRM adaptive centering: car `$term` has no fitted graph record " *
                "naming its $(length(struct_idx)) areas.",
            )
            log_inv_scale = zeros(n_struct)
            if bym2
                length(unstructured.coordinates) == length(struct_idx) || error(
                    "BRM adaptive centering: car `$term` owns " *
                    "$(length(unstructured.coordinates)) unstructured coordinates " *
                    "for $(length(struct_idx)) areas.",
                )
                inv_scale = data(:inv_scale)
                inv_scale isa AbstractVector{<:Real} &&
                    length(inv_scale) == length(struct_idx) || error(
                    "BRM adaptive centering: car `$term` declaration does not " *
                    "expose one `inv_scale` per area.",
                )
                values = Float64[inv_scale[k] for k in area_of]
                all(v -> isfinite(v) && v > 0, values) || error(
                    "BRM adaptive centering: car `$term` has a non-finite or " *
                    "non-positive BYM2 scaling factor.",
                )
                log_inv_scale .= log.(values)
            end

            sigma_lower, sigma_upper = only.(_adaptive_bounds(
                descriptor, sigma.output, 1, term, "marginal SD"))
            rho_lower, rho_upper = isnothing(rho) ? (0.0, 1.0) :
                only.(_adaptive_bounds(descriptor, rho.output, 1, term, "mixing share"))
            0 <= rho_lower && rho_upper <= 1 || error(
                "BRM adaptive centering: car `$term` mixing share is declared on " *
                "[$rho_lower, $rho_upper], outside [0, 1].",
            )
            push!(out, _CARAdaptiveCenteringBlock(
                logical, term, bym2,
                collect(structured.coordinates),
                isnothing(unstructured) ? Int[] : collect(unstructured.coordinates),
                struct_idx, area_of, log_inv_scale,
                only(sigma.coordinates), sigma_lower, sigma_upper,
                isnothing(rho) ? 0 : only(rho.coordinates), rho_lower, rho_upper,
                bym2 ? _adaptive_car_intercept(descriptor, logical, unc_names) : 0,
                collect(Any, areas),
            ))
        end
    end

    claimed = Int[]
    for block in out
        append!(claimed, block.structured, block.unstructured)
        push!(claimed, block.sigma)
        block.bym2 && push!(claimed, block.rho)
    end
    length(unique(claimed)) == length(claimed) || error(
        "BRM adaptive centering: emitted car blocks claim overlapping " *
        "unconstrained coordinates; refusing an ambiguous transform.",
    )
    out
end

# The sampled population intercept of predictor `logical`, or 0. A BYM2 `theta`
# cell is centred on `intercept + a * phi`, so its `c=1` coordinate is the
# area's total (Gelfand, Sahu and Carlin 1995) rather than its deviation `b[k]`:
# beside a free intercept, centring `b[k]` ties every centred area to the
# intercept along the direction `(intercept + d, b .- d)`, which a likelihood
# that pins each area's total does not resolve. An intercept absorbed into exact
# totals or S2Z, or one declared with bounds (its coordinate is then not the
# value), is not read. `:icar` `phi` cells are not centred on it: the ICAR's sum-to-zero
# already separates the field from the intercept.
_adaptive_car_intercept(descriptor, logical, unc_names) =
    _adaptive_population_coordinate(descriptor, logical, :Intercept, unc_names)

# The unconstrained coordinate of predictor `logical`'s sampled population
# coefficient `coefficient`, or 0 when the predictor has no such coefficient,
# or it is absorbed into exact totals or S2Z (recovered, not sampled), or it is
# declared with bounds (its coordinate is then not the value). Adaptive
# centring reads it as a cell's location, never transforms it. An absorbed
# coefficient is skipped before the lookup, which refuses an S2Z-absorbed one
# when `unc_names` carries no generated quantities.
function _adaptive_population_coordinate(descriptor, logical, coefficient, unc_names)
    coefficient in popcoefnames(descriptor.plan.parent, logical) || return 0
    any(b -> coefficient in b.population_columns,
        _brm_absorbing_blocks(descriptor, logical)) && return 0
    resolved = brm_population_effect_coordinates(descriptor, logical, unc_names;
                                                 coefficient)
    (resolved.recovered || length(resolved.coordinates) != 1 ||
     resolved.output.kind !== :parameter ||
     !isempty(resolved.output.constraints)) && return 0
    only(resolved.coordinates)
end

# Hierarchical centring of random-effect cells (Gelfand, Sahu and Carlin 1995;
# snag fitted-seven-hie-76d07973, decision 1xdxiv1). A cell of level `j`, term
# `k` is centred on its predictor's population coefficient for that term, so
# its `c=1` coordinate is the level's total `beta_k + b[k, j]` rather than the
# deviation `b[k, j]`. Centring the deviation on 0 leaves the direction
# `(beta + d, b .- d)` to the prior alone: with strongly identified levels on
# few groups that ridge has width about `tau / sqrt(J)` and no control removes
# it. The coefficient is read, never transformed, so the map stays exact.
#
# Returns, per block, one coordinate per term (0 = centred on 0). A block gets
# none when it is compiled centred (`centered_groups`: the compiled coordinate
# is then the deviation, which WarmupHMC's shared-location pair map cannot keep
# while the source samples the total), groups by multi-membership, or has no
# margin labels. A term gets none when its predictor has no matching sampled,
# unbounded population coefficient (`_adaptive_population_coordinate`), and
# when the block's grouping is nested in another grouping of the same predictor
# (`_adaptive_nested_bindings`): a nested level's parent is the enclosing
# level's total, a coordinate another cell rewrites, so it stays centred on 0.
function _adaptive_ranef_locations(model, blocks, unc_names)
    out = IdDict{Any,Vector{Int}}()
    isempty(blocks) && return out
    d = model isa BRMDescriptor ? model : brm_descriptor(model)
    margins = Dict(m.block.binding => m.margins for m in _brmd_ranef_metadata(d))
    nested = _adaptive_nested_bindings(d.plan, blocks, margins)
    for block in blocks
        K = block.ranef.n_terms
        labels = get(margins, block.ranef.binding, ())
        locs = zeros(Int, K)
        if block.target_c == 0.0 && block.ranef.group isa Symbol && length(labels) == K
            for k in 1:K
                (block.ranef.binding, labels[k].predictor) in nested && continue
                locs[k] = _adaptive_population_coordinate(d, labels[k].predictor,
                                                          labels[k].coefficient,
                                                          unc_names)
            end
        end
        out[block] = locs
    end
    out
end

# `(binding, predictor)` pairs whose grouping is nested in a coarser grouping of
# another block on the same predictor: every level of the first occurs with one
# level of the second on the predictor's rows, and the second has fewer levels.
# Rows are compared through the blocks' group-index data, which index the same
# rows when they have the same length; blocks whose rows cannot be aligned
# (different lengths, a resampled index) are not compared.
function _adaptive_nested_bindings(plan, blocks, margins)
    index = Dict{Symbol,AbstractVector}()
    for block in blocks
        binding = block.ranef.binding
        haskey(index, binding) && continue
        decl = only(x for x in plan.declarations
                    if x.role === :prior && x.target === binding)
        key = get(decl.keywords, :group_idx, nothing)
        value = key isa Symbol ? get(plan.data, key, nothing) : nothing
        value isa AbstractVector && (index[binding] = value)
    end
    by_predictor = Dict{Symbol,Vector{Symbol}}()
    for (binding, labels) in margins, label in labels
        haskey(index, binding) || continue
        list = get!(by_predictor, label.predictor, Symbol[])
        binding in list || push!(list, binding)
    end
    nested = Set{Tuple{Symbol,Symbol}}()
    for (predictor, bindings) in by_predictor, a in bindings, b in bindings
        a === b && continue
        _adaptive_nested_in(index[a], index[b]) && push!(nested, (a, predictor))
    end
    nested
end

# Whether the levels coded by `inner` nest in those coded by `outer` on the same
# rows: each inner level occurs with one outer level, and the outer grouping is
# strictly coarser (a relabelling of the same grouping is not nesting).
function _adaptive_nested_in(inner::AbstractVector, outer::AbstractVector)
    length(inner) == length(outer) || return false
    parent = Dict{eltype(inner),eltype(outer)}()
    for (a, b) in zip(inner, outer)
        get!(parent, a, b) == b || return false
    end
    length(unique(values(parent))) < length(parent)
end

# Stan's native `cholesky_factor_corr[K]` constrain transform.  Its
# K(K-1)/2 free coordinates are row-major over the strict lower triangle; each
# is mapped with tanh and multiplied by the remaining row stick.  This exact
# spelling is checked against BridgeStan in the adaptive-centering acceptance
# probe, including a nonzero K=3 discriminator.
_adaptive_cholesky_corr(raw::AbstractVector, K::Int) =
    _adaptive_cholesky_from_cpc(tanh, raw, K)

# The same factor from values that `partial` maps to the canonical partial
# correlations: `tanh` of Stan's free coordinates, or `identity` of known ones
# (`_adaptive_cholesky_cpc`).
function _adaptive_cholesky_from_cpc(partial, raw::AbstractVector, K::Int)
    length(raw) == K * (K - 1) ÷ 2 || throw(DimensionMismatch(
        "cholesky_factor_corr[$K] needs $(K * (K - 1) ÷ 2) free values, got $(length(raw))",
    ))
    T = eltype(raw)
    L = zeros(T, K, K)
    L[1, 1] = one(T)
    p = 1
    for i in 2:K
        stick = one(T)
        for j in 1:i-1
            z = partial(raw[p])
            p += 1
            L[i, j] = stick * z
            stick *= sqrt(one(T) - z * z)
        end
        L[i, i] = stick
    end
    L
end

_adaptive_block_taus(x::AbstractVector, block::AdaptiveCenteringBlock) =
    exp.(x[block.log_scales])
_adaptive_block_taus(x::AbstractVector, block::R2D2AdaptiveCenteringBlock) =
    [exp(_adaptive_log_scale(x, block.table, k)) for k in 1:block.ranef.n_terms]

function _adaptive_block_cholesky(x::AbstractVector, block::AbstractAdaptiveCenteringBlock)
    L = _adaptive_block_correlation(x, block)
    tau = _adaptive_block_taus(x, block)
    tau .* L
end

# Level `g`'s factor: `diag(tau) * L`, scaled by `sqrt(w[g])` for a Student-t
# block. A Gaussian block's factor is the same for every level.
function _adaptive_block_cholesky(x::AbstractVector, block::AbstractAdaptiveCenteringBlock, g)
    C = _adaptive_block_cholesky(x, block)
    block isa R2D2AdaptiveCenteringBlock && !isempty(block.mixing) || return C
    exp(_adaptive_log_mixing(x, block, g)) .* C
end

_adaptive_block_correlation(x::AbstractVector, block::AbstractAdaptiveCenteringBlock) =
    _adaptive_cholesky_corr(x[block.cholesky_free], block.ranef.n_terms)
# A known correlation (`cor(:, ID) = R`) is its constant factor.
_adaptive_block_correlation(x::AbstractVector, block::R2D2AdaptiveCenteringBlock) =
    isempty(block.cholesky_known) ?
        _adaptive_cholesky_corr(x[block.cholesky_free], block.ranef.n_terms) :
        block.cholesky_known

# The per-draw model-scale factor `C` with fresh effect `b = C * z`,
# reconstructed from one unconstrained draw with no BridgeStan call. Correlated
# blocks rebuild `tau .* L` above; a scalar `(1 | g)` intercept block has no
# Cholesky and its scale is `exp(log_scale)`, so it is the 1×1 factor. Shared
# by centered replay (prediction.jl); the WarmupHMC online path calls
# `_adaptive_block_cholesky` directly and never reaches the intercept branch.
function _adaptive_block_cov_chol(x::AbstractVector, block::AdaptiveCenteringBlock)
    if block.ranef.family in _ADAPTIVE_INTERCEPT_FAMILIES
        block.ranef.n_terms == 1 || error(
            "BRM adaptive centering: intercept block `$(block.ranef.binding)` ",
            "has $(block.ranef.n_terms) terms; the scalar scale path needs one.")
        return fill(exp(x[only(block.log_scales)]), 1, 1)
    end
    _adaptive_block_cholesky(x, block)
end
# A centered known-value block (the only derived-scale block that is centered).
_adaptive_block_cov_chol(x::AbstractVector, block::R2D2AdaptiveCenteringBlock) =
    _adaptive_block_cholesky(x, block)

"""
    adaptive_centering_problem(model, problem, ad_backend; kwargs...)

Wrap a BRM `problem` in WarmupHMC's strictly-online adaptive-centering
transform. Compiled models use BRM's WarmupHMC extension. A `TuringBRMI` uses
the separate joint Turing+WarmupHMC extension and, for now, accepts only the
identity-link, fixed-scale Gaussian model with one default noncentered scalar
random-intercept geometry documented by that method.
"""
function adaptive_centering_problem end

"""
    adaptive_centering_controls(adaptive, model; unc_names=nothing) -> Vector{NamedTuple}

One row per adaptive cell of `adaptive`, the wrapper
[`adaptive_centering_problem`](@ref) returned for `model`, in the wrapper's
pair order (the order `centeredness` and checkpoints use). Every row has

- `cell`, the pair number; `index` and `name`, its unconstrained coordinate;
- `family`: `:ranef` (ordinary or R2D2 random-effect cell), `:hsgp` (basis
  weight), `:scalar` (exact total or S2Z contrast), `:car` (areal field cell)
  or `:cdar` (walk cell);
- `c`, the current source control, and `target`, the compiled frame's.

Random-effect rows also carry `binding`, `group`, `margin` (its
`(; predictor, coefficient)`, as [`brm_ranef_draws`](@ref) labels it),
`level`, `term` (the margin's declared position in its block), `rank`, its
position in the group's current triangular order (`term` itself under
`ranef_order=:declared`, the learned position under `ranef_order=:learned`),
and `location`, the unconstrained name of the population coefficient the cell
is centred on (so its `c=1` coordinate is the level's total), or `nothing` for
a cell centred on 0 (see [`adaptive_centering_problem`](@ref)).
HSGP and cdar rows carry `binding`, `margin = (; predictor, coefficient=term)`
and `term` (the basis or walk cell); a grouped HSGP mapped across its levels
also carries `group`, `level` (the level's label) and `rank` (its position in
the basis column's triangular map). Car rows carry `binding`, `margin`, `level`
(the area's label) and `term`: 1 for an `:icar` area's structured `phi` cell, 2
for a `:bym2` area's unstructured `theta` cell. Fields that do not apply are
`nothing`.

Read it after warm-up from the same object the sampler adapted, for example a
single-chain `adaptive_warmup_mcmc` run; samplers that copy their density per
chain leave the passed object at its initial controls (warmuphmc-use §5c).
Pass `unc_names` when `adaptive` does not wrap a BridgeStan model.
"""
function adaptive_centering_controls end

function adaptive_centering_blocks(::TuringBRMI, _unc_names)
    error(
        "Turing backend: `adaptive_centering_blocks` describes compiled Stan " *
        "unconstrained coordinates and does not apply to DynamicPPL models. " *
        "Choose the executable endpoint with " *
        "`TuringBRMI(brmi; centered_groups=...)`.")
end

function adaptive_centering_problem(::TuringBRMI, _problem, _ad_backend;
                                    _kwargs...)
    error(
        "Turing backend: native online adaptive centering requires both " *
        "Turing and WarmupHMC, a DynamicPPL.LogDensityFunction, " *
        "and the supported fixed-scale Gaussian model with one default " *
        "noncentered `(1 | group)` geometry. " *
        "The compiled-Stan coordinate bridge does not apply to DynamicPPL models.")
end
