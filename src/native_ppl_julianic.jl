# ---------------------------------------------------------------------------
# Julianic NativePPL — a "run-the-body" authoring surface.
#
# Goal (user, 2026-08-04): the `@model` body should read as valid SSA Julia that
# actually computes the log density, with `~` the ONLY magic — and `~` reducing
# to "prep + logpdf": constrain the unconstrained coordinate (+ log-Jacobian),
# then accumulate the distribution's log density.
#
# This is the additive Option-1 path (decision `0tuhucl`, resolved "use your
# judgment"): a new `@jmodel` entry point that lowers `~` and RUNS the body as
# the primal density `θ -> Real`, verified model-by-model against the existing
# declarative executor as a differential oracle. Once the gallery reaches
# parity, `@model` itself flips onto this lowering.
#
# Design
# ------
# * Only `~` / `.~` statements are rewritten; every other statement is kept
#   VERBATIM — that is exactly "valid Julia except ~".
# * A latent `sym ~ dist` lowers to `sym = _sample_site!(ctx, Val(:sym), dist)`.
# * A broadcast observation `@. lhs ~ dist` lowers to accumulating
#   `sum(@. logpdf(dist, <conditioned data for lhs>))`. `lhs .~ dist` is the
#   same site in the spelling the declarative `@model` uses, where the RHS is
#   already written in broadcast form (`y .~ Poisson.(exp(r))`) and so is NOT
#   re-dotted.
# * The body runs twice against a context `ctx`:
#     - TRACE (compile): discover the total unconstrained dimension by counting
#       the coordinates each latent `~` consumes. Returns in-support placeholder
#       values so deterministic code and observations run without error.
#     - PRIMAL (density): a cursor walks `θ`; each latent `~` consumes its slice,
#       preps it (constrain + logjac), and accumulates prior logpdf; each
#       observation accumulates its data logpdf. The scalar `ctx.buffers.acc` is the
#       primal `θ -> Real` that Enzyme differentiates (same AD path the
#       declarative executor already uses — DifferentiationInterface + AutoEnzyme
#       straight through the primal kernel).
#
# Latent-vs-observation is decided SYNTACTICALLY (broadcast = observation,
# scalar = latent), so BOTH directions are checked against the conditioned data
# and fail closed. Without that check a scalar `~` on a conditioned name would
# silently score a fresh latent and never look at the observation — and the
# declarative `@model` accepts exactly that spelling (`macro_scalar_gaussian`,
# test/native_ppl.jl:238), so the same source text would mean two different
# models depending on which macro read it.
#
# Supported latent site families, each one `_sample!` method pair (trace +
# primal) — the surface grows by ADDING methods, never by touching the lowering:
#   * scalar `Normal` (real support, identity) and `Exponential` (positive, exp);
#   * real-support multivariate (`MvNormal`, `product_distribution` of Normals) —
#     one contiguous block, identity transform;
#   * positive-support multivariate (`product_distribution` of `Exponential`s) —
#     per-coordinate exp transform, for a vector of grouped scales;
#   * `LKJCholesky` — the constrained correlation manifold, K(K-1)/2 raw coords.
# Observations take any `Distributions` likelihood in broadcast form. A
# multivariate family with NO method of its own hits the real-support probe and
# is REFUSED rather than scored against the wrong measure.
#
# Gradients run under the plain `DI.AutoEnzyme()` the declarative executor uses:
# the conditioned data is threaded through the body as a separate `DI.Constant`
# argument rather than stored on the differentiated context, so STATIC activity
# analysis suffices and no `set_runtime_activity` workaround is needed.
# ---------------------------------------------------------------------------

# NOTE: `Normal`/`Exponential`/`StandardNormal` are ALSO names inside NativePPL
# (prior/declaration constructors), so we must NOT `using Distributions`-import
# them here — fully qualify the Distributions types/functions instead. In the
# macro-generated model body, `Normal`/`logpdf` deliberately resolve in the
# CALLER's scope (the user's `using Distributions`), keeping the body real Julia.
import Distributions
# `weighted`'s typed weight-vector methods dispatch on the StatsBase weight
# kinds, and `rand` on a wrapper distribution needs the RNG abstract type.
import StatsBase
import Random

# 0-alloc buffer-threading substrate (user directive 2026-08-04): every
# array-valued intermediate in the run-the-body primal is filled IN PLACE into a
# preallocated buffer via MutatingFunctions' `apply!!`, so the differentiated
# kernel allocates nothing per gradient eval — the same discipline the declarative
# FactorWorkspace uses (preallocated `FactorBuffers` threaded as a DI.Cache).
using MutatingFunctions: apply!!
# Destination SHAPES for those in-place fills, declared rather than discovered by a
# pilot run: `broadcast_size` for broadcast nodes (function-first, so it is its own
# entry point) and `output_size(getindex, …)` for gathers.
using OutputSignatures: broadcast_size, output_size

# --- Buffer pool ----------------------------------------------------------
#
# The pool holds one reusable flat `Vector{T}` per array-valued intermediate the
# body produces, walked by a cursor in body-execution order. It is threaded through
# the primal kernel as a `DI.Cache` (Enzyme shadows it; the in-place fills are
# differentiated through), exactly mirroring the executor's `FactorBuffers`.
#
# Storage is deliberately FLAT and concretely typed (`Vector{Vector{T}}`): a pool of
# heterogeneously-shaped arrays would be an abstract container and cost a dynamic
# dispatch per access. An N-D intermediate (`tau .* (L*z)` is K×ngroups) is served by
# `reshape`ing its flat buffer to the traced shape — a view, so still no allocation,
# and MutatingFunctions' `apply!!(cache::AbstractArray, broadcast, …)` fills any-rank
# destinations in place (snag `apply-0alloc-ppl-fb5b821f`, landed `0134a6c`).
#
# Sizing: a TRACE pass runs the body once with in-support placeholders and records
# each intermediate's SHAPE (`buffer_sizes::Vector{Dims}`); `jworkspace` then
# preallocates the pool at the workspace eltype. `_next_buffer!` grows the pool on
# demand for the unprepared path (`jlogdensity` without a workspace), so a
# preallocated pool never grows inside a differentiated call.
# It also carries the primal's own mutable state — the θ cursor and the density
# accumulator. That state is preallocated with the pool rather than constructed per
# call for the same reason the executor keeps everything in `FactorWorkspace`: a
# per-call mutable context ESCAPES into the body closure, so Julia must heap-allocate
# it, and one such box per gradient eval is exactly the allocation this whole piece
# exists to remove. `JulianicPrimal` is therefore an immutable view over this struct.
mutable struct JulianicBuffers{T}
    vectors::Vector{Vector{T}}
    vcursor::Int
    cursor::Int         # position in θ
    acc::T              # the accumulated log density
end
JulianicBuffers{T}() where {T} =
    JulianicBuffers{T}(Vector{Vector{T}}(), 0, 0, zero(T))
function JulianicBuffers{T}(sizes::AbstractVector) where {T}
    JulianicBuffers{T}([zeros(T, prod(s)) for s in sizes], 0, 0, zero(T))
end

# The next flat buffer, grown to `n` elements if the pool was not presized.
@inline function _next_flat!(pool::JulianicBuffers{T}, n::Int) where {T}
    pool.vcursor += 1
    if pool.vcursor > length(pool.vectors)
        push!(pool.vectors, zeros(T, n))    # grow lazily (unprepared eval only)
    end
    buffer = @inbounds pool.vectors[pool.vcursor]
    length(buffer) == n || resize!(buffer, n)
    return buffer
end

# The next buffer, shaped to `shape`. A 1-D shape returns the flat vector itself;
# an N-D shape returns a `reshape` view of it (no allocation either way).
@inline _next_buffer!(pool::JulianicBuffers, shape::Tuple{Int}) =
    _next_flat!(pool, @inbounds shape[1])
@inline _next_buffer!(pool::JulianicBuffers, shape::Dims) =
    reshape(_next_flat!(pool, prod(shape)), shape)

# --- Contexts -------------------------------------------------------------

# PRIMAL: walks θ with a cursor and accumulates the log density. Both the cursor
# and the accumulator live on the preallocated `buffers` (see `JulianicBuffers`),
# so this context is IMMUTABLE — two fields, both references — and costs nothing to
# construct per call. A mutable context here would be heap-boxed on every gradient
# eval, because it escapes into the body closure.
#
# The observation DATA is deliberately NOT a field here: it is a run-time
# constant, and mixing a constant array with the active `theta`/`acc` inside one
# struct is exactly what forced Enzyme's `set_runtime_activity`. Instead the body's
# `run` closure takes the data as a SEPARATE argument (passed through DI as a
# `Constant`), so this struct holds only active state and STATIC activity
# suffices — no runtime-activity workaround.
# A context that WALKS θ: it constrains each site's coordinates and returns the
# natural-space value, so every deterministic in the body computes for real.
# `JulianicPrimal` accumulates the log density while doing it; the postprocessing
# contexts (below) do the identical walk and record instead. Sharing the abstract
# type is what makes postprocessing structurally unable to disagree with the
# density — there is one walk, not a walk and a reimplementation of it.
abstract type JulianicRun end

struct JulianicPrimal{TH<:AbstractVector,B} <: JulianicRun
    theta::TH
    buffers::B          # JulianicBuffers{eltype(theta)} — the preallocated pool
end

# TRACE: runs the body once to size the unconstrained coordinate vector AND record
# each array-valued intermediate's SHAPE (`buffer_sizes`), in body order, so the
# primal pool can be preallocated to match.
mutable struct JulianicTrace
    dim::Int
    buffer_sizes::Vector{Dims}
end
JulianicTrace() = JulianicTrace(0, Dims[])

# POSTPROCESSING: the same θ walk as the primal, recording instead of scoring.
# It reuses `JulianicBuffers` verbatim — including the cursor and the (unread)
# accumulator — so every `_sample!` method above works here with no second
# implementation. That is the whole point: a postprocessing pass that re-derived
# what a site means could disagree with the density, and this one cannot.
#
# `sites` collects `name => constrained value` in body order, and `observations`
# collects whatever the observation hook produced per observed site. No pooled
# buffer discipline: postprocessing is not on the gradient hot path, and the
# 0-alloc constraint is deliberately scoped to logdensity + gradient (user,
# 2026-08-04). Values are COPIED out, because a `_sample!` returning a `@view`
# into θ (the multivariate path) would otherwise alias a caller's position — and
# a pooled observation buffer is overwritten by the next site.
struct JulianicRecord{TH<:AbstractVector,B,K} <: JulianicRun
    theta::TH
    buffers::B
    kernel::K           # what an observation site computes (logpdf / rand / …)
    sites::Vector{Pair{Symbol,Any}}
    observations::Vector{Pair{Symbol,Any}}
end
JulianicRecord(theta::AbstractVector, buffers, kernel) =
    JulianicRecord(theta, buffers, kernel,
                   Pair{Symbol,Any}[], Pair{Symbol,Any}[])

# The recording hook on the latent path. Inlined to the identity for every
# context that is not recording, so the primal keeps its 0-alloc property.
@inline _record_site!(_ctx, ::Val, value) = value
@inline function _record_site!(ctx::JulianicRecord, ::Val{name}, value) where {name}
    push!(ctx.sites, name => _julianic_recorded(value))
    return value
end
# `copy` on anything that is a view of, or a pooled buffer over, live state.
@inline _julianic_recorded(value) = value
@inline _julianic_recorded(value::AbstractArray) = copy(value)

# --- `~` runtime: prep + logpdf ------------------------------------------
#
# Each latent method advances the cursor by the site's unconstrained dimension,
# applies the support's transform (prep), accumulates prior logpdf + logjac, and
# returns the CONSTRAINED value (so downstream body code sees the natural-space
# quantity). Trace methods only grow `dim` and return an in-support placeholder.

# Is `name` bound to conditioned data? Both arguments are compile-time
# constants (the site name rides a `Val`, the data is a `NamedTuple` whose field
# names are in its type), so this folds to a literal `true`/`false` and the
# guarded branch disappears from the primal.
@inline _julianic_is_conditioned(::NamedTuple{names}, ::Val{name}) where {names,name} =
    name in names
@inline _julianic_is_conditioned(_data, ::Val) = false

@noinline _julianic_conditioned_latent_error(::Val{name}) where {name} =
    throw(ArgumentError(
        "julianic @jmodel: `" * String(name) * "` is conditioned data, so " *
        "`" * String(name) * " ~ dist` would silently score a fresh LATENT " *
        "and never look at the observation. Write the observation in " *
        "broadcast form — `@. " * String(name) * " ~ dist` or `" *
        String(name) * " .~ dist` — or drop it from `jcondition`."))

# Every lowered scalar `~` routes through here, so the fail-closed check covers
# present AND future `_sample!` methods. `data` is the same constant value the
# observation sites read; it is threaded in rather than stored on `ctx` so the
# differentiated context keeps holding only active state (see `JulianicPrimal`).
@inline function _sample_site!(ctx, data, name::Val, dist)
    _julianic_is_conditioned(data, name) &&
        _julianic_conditioned_latent_error(name)
    return _record_site!(ctx, name, _sample!(ctx, name, dist))
end

# Normal: real support, identity transform (log-Jacobian 0). Covers `Normal()`
# (standard normal) and `Normal(mu, sigma)` with latent-dependent parameters.
# NOTE: no `@inbounds` on the cursor read. The trace and the primal must agree
# on how many coordinates each site consumes; if they ever drift, an elided
# bounds check turns that into an out-of-bounds READ (silent garbage under
# Enzyme) instead of a `BoundsError`. The check is free next to a `logpdf`.
@inline function _sample!(ctx::JulianicRun, ::Val, dist::Distributions.Normal)
    ctx.buffers.cursor += 1
    v = ctx.theta[ctx.buffers.cursor]
    ctx.buffers.acc += Distributions.logpdf(dist, v)
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Normal) = (ctx.dim += 1; 0.0)

# Exponential: positive support, exp transform. constrain u -> exp(u); the
# log-Jacobian of that transform is exactly u, so accumulate logpdf + u.
@inline function _sample!(ctx::JulianicRun, ::Val, dist::Distributions.Exponential)
    ctx.buffers.cursor += 1
    u = ctx.theta[ctx.buffers.cursor]
    v = exp(u)
    ctx.buffers.acc += Distributions.logpdf(dist, v) + u
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Exponential) = (ctx.dim += 1; 1.0)

# Multivariate real-support latent (e.g. MvNormal, `product_distribution` of
# real-support univariates): identity transform on every coordinate (support is
# all of R^k, so the log-Jacobian is 0). Consumes `length(dist)` contiguous
# coordinates and returns the value vector for the body to index. This covers
# grouped/blocked latents authored as `b ~ product_distribution(fill(D, k))`
# or `b ~ MvNormal(...)`, then referenced as `b[group]`.
@inline function _sample!(ctx::JulianicRun, ::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += k
    v = @view ctx.theta[lo:ctx.buffers.cursor]   # view, not a copy — avoids an allocation
    ctx.buffers.acc += Distributions.logpdf(dist, v)
    return v
end
@inline function _sample!(ctx::JulianicTrace, name::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    _julianic_require_real_support(name, dist, k)
    ctx.dim += k
    return zeros(k)
end

# The `MultivariateDistribution` primal method above dispatches on the whole
# abstract type but implements only the REAL-support case (identity transform,
# zero log-Jacobian). A constrained multivariate — `Dirichlet`, `MvLogNormal` —
# would silently get the wrong measure: no constraining transform and no
# log-Jacobian, exactly the asymmetry the scalar `Exponential` method exists to
# avoid. Probe the support once, in TRACE mode (so the primal pays nothing), and
# refuse rather than mis-score. Constrained families that DO have their own
# `_sample!` (positive vectors, LKJ, below) never reach this method at all — a
# more specific signature wins dispatch — so the guard only ever fires on a
# family nobody has implemented yet.
@noinline function _julianic_require_real_support(::Val{name}, dist, k) where {name}
    Distributions.insupport(dist, fill(-1.0, k)) && return nothing
    throw(ArgumentError(
        "julianic @jmodel: multivariate site `" * String(name) * "` has a " *
        "distribution of type " * string(nameof(typeof(dist))) * " whose " *
        "support is not all of R^" * string(k) * ". The run-the-body `~` " *
        "applies an identity transform with zero log-Jacobian to multivariate " *
        "sites, which would score the wrong measure. Constrained-manifold " *
        "sites (LKJ, simplex, positive) need their own `_sample!` method."))
end

# Multivariate POSITIVE-support latent: a `product_distribution` of `Exponential`
# marginals — used for a vector of grouped scales `tau ~
# product_distribution(fill(Exponential(1), K))`. Positive support → exp
# transform per coordinate (constrain u -> exp(u), log-Jacobian u), so accumulate
# per-marginal logpdf + sum(u). More specific than the real-support method above,
# so it wins for the `Product{…,Exponential}` type (a product of Normals resolves
# to `DiagNormal <: MvNormal`, which correctly stays on the identity path).
@inline function _sample!(ctx::JulianicRun, ::Val,
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential})
    k = length(dist)
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += k
    u = @view ctx.theta[lo:ctx.buffers.cursor]   # view, not a copy
    v = exp.(u)
    ctx.buffers.acc += sum(Distributions.logpdf.(dist.v, v)) + sum(u)
    return v
end
@inline function _sample!(ctx::JulianicTrace, ::Val,
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential})
    k = length(dist)
    ctx.dim += k
    return ones(k)
end

# LKJ-Cholesky correlation latent: `L ~ LKJCholesky(K, eta)` returns the K×K
# lower-triangular Cholesky factor of a correlation matrix, so the body can write
# the non-centered effect `b = diag(tau) * (L * z)` in ordinary Julia. The `~`
# magic reduces to prep + logpdf exactly as everywhere else — it just prepares a
# constrained-manifold coordinate:
#   * prep: consume K(K-1)/2 unconstrained coords, build L via the tanh/sech
#     parametrization (row-normalized Cholesky-of-correlation).
#   * logpdf (fused with the manifold log-Jacobian): the LKJ density in raw
#     coords, `sum_j (log_normalizer_j + alpha_j * logsech2(raw))`.
# This reuses the executor's exact leaf math (`_factor_logsech2` / `_factor_sech`
# / the `LKJCholeskySiteFactor` normalizer formula), so it matches bit-for-bit —
# but with RUNTIME K (no `Val{K}` / `@generated`) so the primal kernel stays
# type-stable when K comes from the distribution object's `d` field.

# LKJ log-density in raw coords (mirrors `_factor_lkj_logdensity`, runtime K).
@inline function _julianic_lkj_logdensity(K::Int, eta, coords::AbstractVector{T}) where {T}
    density = zero(T)
    offset = 0
    for column in 1:(K - 1)
        alpha = eta + (K - column - 1) / 2
        logc = BRM.loggamma(alpha + 0.5) - BRM.loggamma(alpha) - 0.5 * log(BRM.pi)
        for _row in (column + 1):K
            raw = coords[begin + offset]
            density += logc + alpha * _factor_logsech2(raw)
            offset += 1
        end
    end
    return density
end

# Build the K×K lower-triangular correlation Cholesky factor from the raw coords
# (mirrors the implicit construction in `_factor_grouped_affine_value`): column s
# of row c is `(prod_{t<s} sech(raw_ct)) * tanh(raw_cs)`, diagonal is the residual
# `prod_{t<c} sech(raw_ct)`. Coord addressing matches `_factor_correlation_raw`.
@inline function _julianic_corr_cholesky(K::Int, coords::AbstractVector{T}) where {T}
    L = zeros(T, K, K)
    @inbounds L[1, 1] = one(T)
    @inbounds for c in 2:K
        residual = one(T)
        for s in 1:(c - 1)
            raw = coords[(s - 1) * K - (s - 1) * s ÷ 2 + c - s]
            L[c, s] = residual * tanh(raw)
            residual = residual * _factor_sech(raw)
        end
        L[c, c] = residual
    end
    return L
end

@inline function _sample!(ctx::JulianicRun, ::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    m = K * (K - 1) ÷ 2
    lo = ctx.buffers.cursor + 1
    ctx.buffers.cursor += m
    coords = @view ctx.theta[lo:lo + m - 1]   # view, not a copy
    ctx.buffers.acc += _julianic_lkj_logdensity(K, dist.η, coords)
    return _julianic_corr_cholesky(K, coords)
end
@inline function _sample!(ctx::JulianicTrace, name::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    _julianic_require_lower_uplo(name, dist)
    ctx.dim += K * (K - 1) ÷ 2
    L = zeros(Float64, K, K)                # in-support placeholder (L = I)
    for i in 1:K
        L[i, i] = 1.0
    end
    return L
end

# `LKJCholesky` carries an `uplo` field, and `LKJCholesky(K, eta, :U)` promises
# the UPPER factor — but `_julianic_corr_cholesky` always builds the LOWER one,
# and hands the body a bare `Matrix` that cannot carry the distinction. The body
# would then compute `L * z` against a silently transposed factor. The density is
# `uplo`-invariant (it is a function of the raw coords), so this can never
# surface as a wrong number at the `~` itself — only as wrong DOWNSTREAM math,
# which is the hardest kind to notice. Checked once in TRACE mode, like the
# real-support probe.
@noinline function _julianic_require_lower_uplo(::Val{name}, dist) where {name}
    dist.uplo === 'L' && return nothing
    throw(ArgumentError(
        "julianic @jmodel: LKJ site `" * String(name) * "` was declared with " *
        "uplo=" * repr(dist.uplo) * ", but the run-the-body `~` returns the " *
        "LOWER-triangular correlation factor as a plain matrix — the body " *
        "would silently use a transposed factor. Declare it as " *
        "`LKJCholesky(" * string(dist.d) * ", " * string(dist.η) * ")` " *
        "(uplo='L') and transpose in the body if you need the upper factor."))
end

# Observation log density. The `~` magic (not the user) supplies `logpdf`, so we
# route through this runtime helper — the model body needs no `logpdf` in scope,
# only the distribution constructor the user actually wrote. Broadcast-safe.
@inline _obs_logpdf(dist, x) = Distributions.logpdf(dist, x)

# --- Observation weights --------------------------------------------------
#
# The declarative surface spells a weighted observation
# `y ~ weighted(Normal(mu, sigma), fweights(n))`, and the weight KIND — carried
# by the StatsBase constructor's NAME — selects the semantics at lowering time:
# `:analytic` scales precision (`Normal(mu, sigma/sqrt(w))`), `:frequency` and
# `:power` MULTIPLY the log density by `w`, `:unit` is a no-op. That taxonomy is
# a FORMULA-surface device: a declaration cannot say `sigma/sqrt(w)`, so the
# weight type has to say it for the author.
#
# A julianic body is ordinary Julia, so it says it directly:
#
#   analytic:  `@. y ~ Normal(mu, sigma / sqrt(w))`     — already expressible
#   unit:      `@. y ~ Normal(mu, sigma)`               — just omit the weight
#
# What plain Julia has no spelling for is the MULTIPLY: `logpdf` returns a
# number, and there is no distribution whose density is another's raised to a
# power. That is the one real gap, and this closes it — as a runtime wrapper
# distribution, per decision `06le2au`. `weighted` is BRM's existing formula
# marker (`src/macro.jl`, declared with no methods); this gives it an executable
# method, so ONE name means the same thing on both surfaces.
#
# The result is a genuine `Distributions.Distribution` — the point of julianic
# is that the body is real Julia, so `logpdf(weighted(d, 2.0), y)` must work
# outside the macro too. It is deliberately NOT normalized: `w * logpdf` is a
# pseudo-likelihood for any `w != 1`, exactly as the declarative `:frequency` /
# `:power` kinds are.
struct WeightedDistribution{VF<:Distributions.VariateForm,
                            VS<:Distributions.ValueSupport,
                            D<:Distributions.Distribution{VF,VS},
                            W<:Real} <: Distributions.Distribution{VF,VS}
    distribution::D
    weight::W
end

Distributions.params(d::WeightedDistribution) =
    (Distributions.params(d.distribution)..., d.weight)

# The whole contract, both variate forms. `logpdf` on the base distribution is
# whatever the author's distribution defines, so this composes with every family
# — including the `truncated`/`censored` response evidence.
@inline Distributions.logpdf(d::WeightedDistribution{Distributions.Univariate},
                             x::Real) =
    d.weight * Distributions.logpdf(d.distribution, x)
@inline Distributions.logpdf(d::WeightedDistribution{<:Union{Distributions.Multivariate,
                                                             Distributions.Matrixvariate}},
                             x::AbstractArray) =
    d.weight * Distributions.logpdf(d.distribution, x)

# Sampling is UNWEIGHTED on purpose, matching the declarative backend: a
# frequency/power weight scales the log-density contribution but leaves the base
# distribution's predictive RNG intact (`src/macro.jl`'s `weighted` docstring).
Base.rand(rng::Random.AbstractRNG, d::WeightedDistribution) =
    rand(rng, d.distribution)
Base.length(d::WeightedDistribution) = length(d.distribution)
Distributions.insupport(d::WeightedDistribution, x) =
    Distributions.insupport(d.distribution, x)

"""
    weighted(distribution, weight::Real)

A distribution whose log density is `weight * logpdf(distribution, x)` — the
executable form of BRM's observation-weight marker, for a julianic `@jmodel`
body:

    @. y ~ weighted(Poisson(exp(eta)), n)

This is the `:frequency` / `:power` semantics (the weight MULTIPLIES the log
density). The other two kinds need no wrapper, because a julianic body writes
them directly: `:analytic` weights scale precision — `@. y ~ Normal(mu, sigma /
sqrt(w))` — and `:unit` weights are the unweighted model.

`weight * logpdf` is a pseudo-likelihood, not a normalized density, for any
`weight != 1`. `rand` is deliberately UNWEIGHTED, so the predictive RNG stays
the base distribution's.

!!! note
    Passing a `StatsBase` weight VECTOR (`weighted(d, fweights(n))`) applies
    elementwise and is checked: `aweights` is rejected with the precision
    spelling above, because broadcasting it here would silently multiply. Under
    `@.` the vector is broadcast to plain numbers BEFORE this function sees it,
    so that check cannot fire — write the weight kind you mean.
"""
BRM.weighted(distribution::Distributions.Distribution, weight::Real) =
    WeightedDistribution(distribution, weight)

# The un-dotted vector spellings, kept identical to the declarative surface so a
# model ports across unchanged. Each maps to the kind's real semantics, and the
# two that this wrapper CANNOT express fail closed rather than silently
# multiplying.
BRM.weighted(distribution::Distributions.Distribution,
             weight::StatsBase.AbstractWeights) =
    WeightedDistribution.(distribution, weight)
@noinline function BRM.weighted(::Distributions.Distribution,
                                ::StatsBase.AnalyticWeights)
    throw(ArgumentError(
        "julianic `weighted`: analytic weights scale PRECISION, not the log " *
        "density, so they are not a `weighted(...)` wrapper here — write them " *
        "directly, e.g. `@. y ~ Normal(mu, sigma / sqrt(w))`. Use " *
        "`weighted(dist, w)` only for frequency/power weights, which multiply " *
        "the log density."))
end
@noinline function BRM.weighted(::Distributions.Distribution,
                                ::StatsBase.ProbabilityWeights)
    throw(ArgumentError(
        "julianic `weighted`: ProbabilityWeights semantics are not implemented " *
        "(the declarative surface rejects them too)."))
end

# Accumulate an already-computed observation log-density contribution.
@inline _accumulate!(ctx::JulianicRun, contribution) =
    (ctx.buffers.acc += contribution; nothing)
@inline _accumulate!(::JulianicTrace, _contribution) = nothing

# Fetch conditioned data for an observation site. The data is passed to the body
# as a separate argument (not stored in the differentiated context — see
# `JulianicPrimal`), so this reads straight from that constant `data` value.
# Available in both modes, because tracing happens after conditioning.
#
# Fails closed on a site that was never conditioned — the mirror of
# `_sample_site!`'s check, and it surfaces at `jprepare` because the trace runs
# the same body. The check folds away exactly like that one: `name` rides a
# `Val` and `data`'s field names are in its type.
@noinline function _julianic_unconditioned_observation_error(data, ::Val{name}) where {name}
    conditioned = isempty(propertynames(data)) ? "none" :
        join(propertynames(data), ", ")
    throw(ArgumentError(
        "julianic @jmodel: observation site `" * String(name) * "` has no " *
        "conditioned data (conditioned sites: " * conditioned * "). Pass it " *
        "to `jcondition`, or write a latent as `" * String(name) * " ~ dist` " *
        "without the broadcast."))
end

@inline function _observation(data, name::Val{sitename}) where {sitename}
    _julianic_is_conditioned(data, name) ||
        _julianic_unconditioned_observation_error(data, name)
    return getproperty(data, sitename)
end

# --- Cached array ops (the 0-alloc rewrite targets) -----------------------
#
# The macro rewrites each array-valued intermediate in the body into one of these,
# decomposing a fused broadcast per operation node. PRIMAL fills a pooled buffer in
# place via `apply!!`; TRACE runs the op allocating (once, at prepare) and records
# the result SHAPE so the pool can be presized. Both passes execute the same lowered
# body, so the push order (trace) and the walk order (primal) stay in lockstep.
#
# Shapes come from OutputSignatures — `broadcast_size` for a broadcast node (it is
# function-first, so it has its own entry point rather than the `x`-first
# `output_size`) and `output_size(getindex, …)` for a gather — which is exactly what
# that package is for: the destination shape from immediately-available information,
# no pilot run (snag `broadcast-gather-3d8a0849`, landed `121de31`).
#
# `broadcast` a single operator/function `f` over already-materialized args:
@inline function _cached_broadcast!(ctx::JulianicRun, f::F, args...) where {F}
    buffer = _next_buffer!(ctx.buffers, broadcast_size(args...))
    apply!!(buffer, broadcast, f, args...)
end
@inline function _cached_broadcast!(ctx::JulianicTrace, f::F, args...) where {F}
    push!(ctx.buffer_sizes, broadcast_size(args...))
    broadcast(f, args...)
end

# `getindex` gather `A[idx]` (a whole-vector index, e.g. `b[group]`):
@inline function _cached_gather!(ctx::JulianicRun, A, idx)
    buffer = _next_buffer!(ctx.buffers, output_size(getindex, size(A), idx))
    apply!!(buffer, getindex, A, idx)
end
@inline function _cached_gather!(ctx::JulianicTrace, A, idx)
    push!(ctx.buffer_sizes, output_size(getindex, size(A), idx))
    getindex(A, idx)
end

# Broadcast observation, accumulated 0-alloc. The macro brackets the obs with these
# so the SAME generated `@. buf = logpdf(Dist(params), data)` runs in both passes
# (no closure — the fused fill is inline generated code, keeping Enzyme's activity
# analysis clean). PRIMAL fills a pooled, right-sized buffer (the distribution
# vector stays lazy, never materialized) and sums it into `acc`; TRACE records the
# obs length and hands back a throwaway scratch so the same `.=` fill is harmless.
@inline _obs_begin!(ctx::JulianicRun, n::Int) = _next_buffer!(ctx.buffers, (n,))
@inline function _obs_begin!(ctx::JulianicTrace, n::Int)
    push!(ctx.buffer_sizes, (n,))
    Vector{Float64}(undef, n)           # one-shot trace scratch for the .= fill
end
@inline _obs_end!(ctx::JulianicPrimal, ::Val, buf) = (ctx.buffers.acc += sum(buf); nothing)
@inline _obs_end!(::JulianicTrace, ::Val, _buf) = nothing
@inline function _obs_end!(ctx::JulianicRecord, ::Val{name}, buf) where {name}
    push!(ctx.observations, name => identity.(buf))   # narrows `Any` to the real eltype
    return nothing
end

# WHAT an observation site computes, chosen by the context rather than baked into
# the lowered body. The body's fused fill calls this function elementwise over
# `(distribution, datum)`, so one body serves the density, the pointwise
# log-likelihood and the posterior predictive with no second lowering — the
# observation distribution is built by the author's own code either way.
@inline _obs_kernel(_ctx) = _obs_logpdf
@inline _obs_kernel(ctx::JulianicRecord) = ctx.kernel

# Posterior-predictive kernel: ignores the observed datum — redrawing it is the
# point — and carries its own RNG so a predictive pass is reproducible.
struct JulianicPredictKernel{R<:Random.AbstractRNG}
    rng::R
end
@inline (kernel::JulianicPredictKernel)(dist, _x) = rand(kernel.rng, dist)

# Postprocessing collects into a FRESH `Any` buffer rather than the pooled one:
# the pool is typed at θ's eltype, which would silently coerce a discrete
# predictive draw (a `Poisson` count, a `Bernoulli` outcome) to `Float64`.
# `_obs_end!` narrows it back to the concrete eltype the kernel actually
# produced. Nothing here is on the gradient path, so the allocation is free of
# consequence — the 0-alloc constraint is scoped to logdensity + gradient.
@inline _obs_begin!(::JulianicRecord, n::Int) = Vector{Any}(undef, n)

# --- Model objects --------------------------------------------------------

# A julianic model: the lowered body (`run(ctx)`, closing over the model's
# input arguments) plus the input argument names (for introspection/error text).
struct JulianicModel{R}
    run::R
    input_names::Tuple{Vararg{Symbol}}
end

# After conditioning: the run closure + the conditioned observation data.
struct JulianicConditioned{R,D}
    run::R
    data::D
end

# After the trace: adds the discovered unconstrained dimension AND the ordered
# lengths of the array-valued intermediate buffers (so a primal pool can be
# preallocated to match — see `JulianicBuffers`).
struct JulianicPrepared{R,D}
    run::R
    data::D
    dimension::Int
    buffer_sizes::Vector{Dims}
end

"""
    jcondition(model::JulianicModel; observations...)

Bind observed data to a julianic model, yielding a `JulianicConditioned`.
"""
jcondition(model::JulianicModel; observations...) =
    JulianicConditioned(model.run, NamedTuple(observations))

"""
    jprepare(conditioned::JulianicConditioned)

Run the body once in TRACE mode to discover the unconstrained dimension.
"""
function jprepare(conditioned::JulianicConditioned)
    trace = JulianicTrace()
    conditioned.run(trace, conditioned.data)
    return JulianicPrepared(
        conditioned.run, conditioned.data, trace.dim, trace.buffer_sizes)
end

dimension(prepared::JulianicPrepared) = prepared.dimension

# The primal kernel `θ -> Real`. Top-level (not a closure over θ) so the AD
# backend sees `prepared` as a constant, `theta` as the sole active input, and the
# preallocated `buffers` pool as scratch (a DI.Cache in the gradient path). The
# body fills those buffers in place, so no per-eval allocation.
function _julianic_kernel(theta::AbstractVector{T}, prepared::JulianicPrepared,
                          buffers::JulianicBuffers{T}) where {T}
    buffers.vcursor = 0                 # rewind the pool + the primal's own state
    buffers.cursor = 0
    buffers.acc = zero(T)
    prepared.run(JulianicPrimal(theta, buffers), prepared.data)
    return buffers.acc
end

@noinline _julianic_dimension_error(prepared::JulianicPrepared, theta) =
    throw(DimensionMismatch(
        "julianic @jmodel: model has " * string(prepared.dimension) *
        " unconstrained coordinate(s) but the position has length " *
        string(length(theta))))

@inline _julianic_check_dimension(prepared::JulianicPrepared, theta::AbstractVector) =
    length(theta) == prepared.dimension || _julianic_dimension_error(prepared, theta)

"""
    jlogdensity(prepared::JulianicPrepared, theta::AbstractVector)

Evaluate the log density by running the body as the primal.
"""
function jlogdensity(prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    buffers = JulianicBuffers{eltype(theta)}(prepared.buffer_sizes)
    return _julianic_kernel(theta, prepared, buffers)
end

# Gradient plumbing. DifferentiationInterface lives in a weak-dep extension, so
# these are declared here and given methods in
# ext/BayesianRegressionModelsDifferentiationInterfaceExt.jl.
function _julianic_prepare_gradient end
function _julianic_value_and_gradient! end

struct JulianicWorkspace{P,G,B}
    di_preparation::P
    gradient::G
    buffers::B          # the preallocated pool, threaded as a DI.Cache
end

"""
    jworkspace(prepared, T, backend)

Allocate a gradient workspace (AD preparation + gradient buffer + the preallocated
0-alloc buffer pool) for a julianic prepared model at element type `T`.

`backend` is the plain `DI.AutoEnzyme()` the declarative executor uses — the
conditioned data reaches the body as a separate `DI.Constant` argument instead
of riding on the differentiated context, so static activity analysis resolves
the primal and no `set_runtime_activity` mode is required.
"""
function jworkspace(prepared::JulianicPrepared, ::Type{T}, backend) where {T<:AbstractFloat}
    position = zeros(T, prepared.dimension)
    buffers = JulianicBuffers{T}(prepared.buffer_sizes)
    di_preparation = _julianic_prepare_gradient(prepared, backend, position, buffers)
    JulianicWorkspace(di_preparation, zeros(T, prepared.dimension), buffers)
end

"""
    jlogdensity!(workspace, prepared, theta)

Log density evaluated against the workspace's preallocated pool — the 0-alloc
primal, and the exact counterpart of the declarative executor's
`logdensity!(workspace, …)`. `jlogdensity(prepared, theta)` computes the same
number but allocates a pool per call, so it is the convenience entry point, not
the one to put in a sampler's inner loop.
"""
function jlogdensity!(
    workspace::JulianicWorkspace, prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    return _julianic_kernel(theta, prepared, workspace.buffers)
end

"""
    jlogdensity_and_gradient!(workspace, prepared, theta) -> (density, gradient)

Value + gradient of the log density via Enzyme straight through the run-the-body
primal, with all array intermediates filled in place into the preallocated pool.
"""
function jlogdensity_and_gradient!(
    workspace::JulianicWorkspace, prepared::JulianicPrepared, theta::AbstractVector)
    _julianic_check_dimension(prepared, theta)
    return _julianic_value_and_gradient!(
        prepared, workspace.di_preparation, workspace.gradient, theta,
        workspace.buffers)
end

# --- Postprocessing ---------------------------------------------------------
#
# The same body, the same θ walk, a different sink. Everything a consumer asks of
# a fitted model — what are the parameters in natural space, how well is each
# observation predicted, what would new data look like — is answered by RUNNING
# the model rather than by a parallel description of it. The declarative surface
# needs `output`/`outputs`/`simulate` machinery precisely because its executor
# discarded the body; julianic still has the body.
function _julianic_record(prepared::JulianicPrepared, theta::AbstractVector, kernel)
    _julianic_check_dimension(prepared, theta)
    buffers = JulianicBuffers{eltype(theta)}(prepared.buffer_sizes)
    context = JulianicRecord(theta, buffers, kernel)
    prepared.run(context, prepared.data)
    return context
end

"""
    jconstrained(prepared, theta) -> NamedTuple

Every latent site's value in NATURAL (constrained) space, at the unconstrained
position `theta`, keyed by site name in body order.

This is the same constraining transform the density applies — one walk, not a
reimplementation — so a value here can never disagree with the log density that
scored it.
"""
jconstrained(prepared::JulianicPrepared, theta::AbstractVector) =
    NamedTuple(_julianic_record(prepared, theta, _obs_logpdf).sites)

"""
    jpointwise(prepared, theta) -> NamedTuple

Per-observation log-likelihood at `theta`, keyed by observation site — the
elementwise terms whose sum the density accumulates, which is what PSIS-LOO and
WAIC consume.
"""
jpointwise(prepared::JulianicPrepared, theta::AbstractVector) =
    NamedTuple(_julianic_record(prepared, theta, _obs_logpdf).observations)

"""
    jpredict([rng], prepared, theta) -> NamedTuple

One posterior-predictive draw per observation site at `theta`: the body's own
observation distributions, resampled instead of scored. Discrete families keep
their natural element type.
"""
jpredict(rng::Random.AbstractRNG, prepared::JulianicPrepared, theta::AbstractVector) =
    NamedTuple(_julianic_record(
        prepared, theta, JulianicPredictKernel(rng)).observations)
jpredict(prepared::JulianicPrepared, theta::AbstractVector) =
    jpredict(Random.default_rng(), prepared, theta)

# --- Draws ------------------------------------------------------------------
#
# The batch forms. `positions` is a matrix of unconstrained draws with ROWS =
# draws, matching the declarative `evaluate_draws!` convention so a sampler's
# output feeds either surface unchanged.
#
# Each site's draws are stacked into one dense array whose FIRST axis is the
# draw, so a scalar site becomes an `ndraws` vector, a length-k site an
# `ndraws × k` matrix, and the LKJ factor an `ndraws × K × K` array. That layout
# is what a labelled container wants (`TreeData(store, :draw, …)` — see the
# TreeArrays extension) and what a plain reduction over draws wants.
@inline _julianic_draw_store(value::Number, ndraws::Int) =
    Vector{typeof(value)}(undef, ndraws)
@inline _julianic_draw_store(value::AbstractArray, ndraws::Int) =
    Array{eltype(value)}(undef, ndraws, size(value)...)

@inline _julianic_store_draw!(store, draw::Int, value::Number) =
    (store[draw] = value; nothing)
@inline _julianic_store_draw!(store, draw::Int, value::AbstractArray) =
    (selectdim(store, 1, draw) .= value; nothing)

@noinline _julianic_draw_shape_error(draw::Int, expected, got) =
    throw(ArgumentError(
        "julianic draws: draw " * string(draw) * " recorded sites " *
        string(got) * " but draw 1 recorded " * string(expected) * ". The body " *
        "takes a different path per position, so the draws cannot be stacked."))

function _julianic_draws(prepared::JulianicPrepared, positions::AbstractMatrix,
                         kernel, selector::F) where {F}
    ndraws = size(positions, 1)
    ndraws >= 1 || throw(ArgumentError(
        "julianic draws: `positions` needs at least one row (rows are draws)"))
    recorded = selector(
        _julianic_record(prepared, view(positions, 1, :), kernel))
    names = Tuple(pair.first for pair in recorded)
    stores = Tuple(_julianic_draw_store(pair.second, ndraws) for pair in recorded)
    for (store, pair) in zip(stores, recorded)
        _julianic_store_draw!(store, 1, pair.second)
    end
    for draw in 2:ndraws
        recorded = selector(
            _julianic_record(prepared, view(positions, draw, :), kernel))
        # A body whose site set depends on the position would stack garbage under
        # the first draw's names, so check rather than trust the shape.
        Tuple(pair.first for pair in recorded) == names ||
            _julianic_draw_shape_error(
                draw, names, Tuple(pair.first for pair in recorded))
        for (store, pair) in zip(stores, recorded)
            _julianic_store_draw!(store, draw, pair.second)
        end
    end
    return NamedTuple{names}(stores)
end

_julianic_sites(context::JulianicRecord) = context.sites
_julianic_observations(context::JulianicRecord) = context.observations

"""
    jconstrained_draws(prepared, positions) -> NamedTuple

`jconstrained` over a matrix of unconstrained draws (ROWS = draws). Each site's
draws are stacked with the draw as the FIRST axis.
"""
jconstrained_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, _obs_logpdf, _julianic_sites)

"""
    jpointwise_draws(prepared, positions) -> NamedTuple

`jpointwise` over a matrix of unconstrained draws (ROWS = draws), stacked as
`ndraws × nobservations` per observation site — the layout PSIS-LOO expects.
"""
jpointwise_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, _obs_logpdf, _julianic_observations)

"""
    jpredict_draws([rng], prepared, positions) -> NamedTuple

`jpredict` over a matrix of unconstrained draws (ROWS = draws): the posterior
predictive, one redrawn replicate per draw per observation.
"""
jpredict_draws(rng::Random.AbstractRNG, prepared::JulianicPrepared,
               positions::AbstractMatrix) =
    _julianic_draws(prepared, positions, JulianicPredictKernel(rng),
                    _julianic_observations)
jpredict_draws(prepared::JulianicPrepared, positions::AbstractMatrix) =
    jpredict_draws(Random.default_rng(), prepared, positions)

# Labelled-container forms, given methods in
# ext/BayesianRegressionModelsTreeArraysExt.jl. They are a weak-dep extension
# rather than core because a labelled container is a POSTPROCESSING convenience:
# what the axes mean, attached to the very arrays the functions above already
# return. A consumer who only wants the density never loads TreeArrays.
function jconstrained_tree end
function jpointwise_tree end
function jpredict_tree end

# --- LogDensityProblems -----------------------------------------------------
#
# The interface every downstream consumer actually reaches for (WarmupHMC and
# the samplers take a `LogDensityProblem`, not a `Prepared`), so julianic cannot
# stand in for the declarative surface without it. Deliberately the same shape as
# `FactorLogDensityProblem`: the problem OWNS its workspace, so `logdensity` and
# `logdensity_and_gradient` hit the 0-alloc pooled paths rather than allocating
# per call.
struct JulianicLogDensityProblem{P,W}
    prepared::P
    workspace::W
end

"""
    JulianicLogDensityProblem(prepared, backend; eltype=Float64)

Wrap a prepared julianic model as a `LogDensityProblems` problem of order 1,
allocating the gradient workspace (and the 0-alloc buffer pool) once.
"""
function JulianicLogDensityProblem(prepared::JulianicPrepared, backend;
                                   eltype::Type{T}=Float64) where {T<:AbstractFloat}
    work = jworkspace(prepared, T, backend)
    JulianicLogDensityProblem{typeof(prepared),typeof(work)}(prepared, work)
end

Base.eltype(problem::JulianicLogDensityProblem) = eltype(problem.workspace.gradient)
BRM.LogDensityProblems.capabilities(::Type{<:JulianicLogDensityProblem}) =
    BRM.LogDensityProblems.LogDensityOrder{1}()
BRM.LogDensityProblems.dimension(problem::JulianicLogDensityProblem) =
    problem.prepared.dimension
BRM.LogDensityProblems.logdensity(problem::JulianicLogDensityProblem,
                                  position::AbstractVector) =
    jlogdensity!(problem.workspace, problem.prepared, position)
# `copy` because the workspace's gradient buffer is reused on the next call —
# the same contract the declarative problem keeps.
function BRM.LogDensityProblems.logdensity_and_gradient(
        problem::JulianicLogDensityProblem, position::AbstractVector)
    density, gradient = jlogdensity_and_gradient!(
        problem.workspace, problem.prepared, position)
    density, copy(gradient)
end

# --- Macro ----------------------------------------------------------------

# Detect a broadcast sampling statement: `@. lhs ~ rhs` or `@__dot__ lhs ~ rhs`.
function _julianic_broadcast_sample(statement)
    statement isa Expr && statement.head === :macrocall || return nothing
    macro_name = statement.args[1]
    (macro_name === Symbol("@__dot__") || macro_name === Symbol("@.")) || return nothing
    inner = statement.args[end]
    inner isa Expr && inner.head === :call && length(inner.args) == 3 &&
        inner.args[1] === :~ || return nothing
    return (lhs=inner.args[2], rhs=inner.args[3])
end

# Detect a scalar sampling statement: `lhs ~ rhs`.
function _julianic_scalar_sample(statement)
    statement isa Expr && statement.head === :call && length(statement.args) == 3 &&
        statement.args[1] === :~ || return nothing
    return (lhs=statement.args[2], rhs=statement.args[3])
end

# Detect a dotted sampling statement: `lhs .~ rhs`. This is the spelling the
# declarative `@model` uses for observations (`y .~ Poisson.(exp(r))`), where
# the RHS is ALREADY written in broadcast form — so unlike `@. lhs ~ rhs` the
# RHS must NOT be re-dotted. Binary `.~` has no meaning in Base, so matching it
# here cannot shadow ordinary Julia (unary `~x` is `length(args) == 2` and is
# left alone).
function _julianic_dotted_sample(statement)
    statement isa Expr && statement.head === :call && length(statement.args) == 3 &&
        statement.args[1] === :.~ || return nothing
    return (lhs=statement.args[2], rhs=statement.args[3])
end

# --- 0-alloc rewrite of deterministic array expressions -------------------
#
# Every array-valued intermediate the body computes is routed through a pooled
# buffer instead of a fresh allocation, so the differentiated kernel is literally
# 0-alloc (user directive 2026-08-04; decision `08w0buk` option B). The rewrite is
# per-NODE, bottom-up:
#
#   `beta[1] .+ beta[2] .* x .+ b[group]`
#     ->  t1 = _cached_gather!(ctx, b, group)             # b[group]
#         t2 = _cached_broadcast!(ctx, *, beta[2], x)     # beta[2] .* x
#         t3 = _cached_broadcast!(ctx, +, beta[1], t2)    # ... .+ ...
#         t4 = _cached_broadcast!(ctx, +, t3, t1)
#
# which is exactly `apply!!(buffer, broadcast|getindex, …)` — MutatingFunctions'
# registered 0-alloc forms — with the buffers presized from the trace pass.
# Scalar-only expressions are left VERBATIM (nothing to buffer), so the body stays
# ordinary Julia wherever buffering buys nothing.

# A dotted operator (`.+`) or dot-call (`exp.(v)`) is a broadcast node; `A[idx]`
# with a non-literal index is a gather node. Anything else is left alone.
_julianic_dotted_operator(f) =
    f isa Symbol && length(string(f)) > 1 && startswith(string(f), ".") ?
        Symbol(string(f)[2:end]) : nothing

# Rewrite one expression bottom-up, emitting pooled-buffer ops into `prelude`.
# Returns the expression to use in place of `ex`.
function _julianic_buffer_expression!(prelude, ex, ctx)
    ex isa Expr || return ex
    # `exp.(v)` / `f.(a, b)` — an explicit dot-call.
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        f = ex.args[1]
        args = [_julianic_buffer_expression!(prelude, a, ctx) for a in ex.args[2].args]
        tmp = gensym(:bc)
        push!(prelude, :($tmp = $(_cached_broadcast!)($ctx, $f, $(args...))))
        return tmp
    end
    if ex.head === :call
        operator = _julianic_dotted_operator(ex.args[1])
        if operator !== nothing
            args = [_julianic_buffer_expression!(prelude, a, ctx) for a in ex.args[2:end]]
            tmp = gensym(:bc)
            push!(prelude, :($tmp = $(_cached_broadcast!)($ctx, $operator, $(args...))))
            return tmp
        end
        # Non-dotted call (`L * z`, `reshape(z, 2, n)`, `maximum(group)`, …) —
        # recurse into the arguments but keep the call itself verbatim. `*` on
        # arrays already has a registered mutating form, but it needs an
        # output-shaped (possibly matrix) buffer, so it is handled by the
        # matrix-aware path in a later milestone rather than forced through the
        # vector pool here.
        return Expr(:call, ex.args[1],
                    (_julianic_buffer_expression!(prelude, a, ctx)
                     for a in ex.args[2:end])...)
    end
    # `b[group]` — a gather when the index is not a literal scalar.
    if ex.head === :ref && length(ex.args) == 2 && !(ex.args[2] isa Integer)
        A = _julianic_buffer_expression!(prelude, ex.args[1], ctx)
        idx = _julianic_buffer_expression!(prelude, ex.args[2], ctx)
        tmp = gensym(:gather)
        push!(prelude, :($tmp = $(_cached_gather!)($ctx, $A, $idx)))
        return tmp
    end
    return ex
end

# Lower one statement. Only `~` statements change MEANING; deterministic
# statements keep their semantics exactly and are only re-routed through pooled
# buffers (still ordinary Julia, just allocation-free).
# `ctx` names the (mutable) sampling context; `data` names the conditioned
# observation data — both are parameters of the generated `run` closure.
function _julianic_lower_statement(statement, ctx, data)
    # `end`/`begin` inside `A[...]` are POSITIONAL markers — they mean nothing
    # outside the indexing expression that encloses them. The buffer rewrite
    # hoists a gather into its own statement (`t = _cached_gather!(ctx, A, idx)`),
    # which carries the index OUT of that context, so `z[1:2:end]` — ordinary
    # Julia the surface promises to run — would lower to a bare undefined `end`.
    # Resolve them to `lastindex`/`firstindex` calls FIRST, exactly as Base's own
    # `@views`/`@view` do, so the index is a self-contained expression before
    # anything moves it.
    statement = Base.replace_ref_begin_end!(statement)
    broadcast_sample = _julianic_broadcast_sample(statement)
    if broadcast_sample !== nothing
        lhs, rhs = broadcast_sample.lhs, broadcast_sample.rhs
        lhs isa Symbol || throw(ArgumentError(
            "julianic @jmodel broadcast observation LHS must be a name; got `$lhs`"))
        data_var = gensym(lhs)
        buffer_var = gensym(:obs)
        # `@__dot__` only dots calls whose head is a *symbol*, so bind the
        # runtime logpdf to a local name before broadcasting. Writing the fused
        # result INTO the pooled buffer (`@. buf = lp(D(a,b), y)`) keeps the
        # vector of distributions lazy — it is never materialized — so the whole
        # observation term costs no allocation.
        logpdf_var = gensym(:logpdf)
        fill = Expr(:macrocall, Symbol("@__dot__"), LineNumberNode(0),
                    :($buffer_var = $(Expr(:call, logpdf_var, rhs, data_var))))
        return quote
            $(logpdf_var) = $(_obs_kernel)($ctx)
            $(data_var) = $(_observation)($data, $(Val(lhs)))
            $(buffer_var) = $(_obs_begin!)($ctx, length($data_var))
            $fill
            $(_obs_end!)($ctx, $(Val(lhs)), $buffer_var)
        end
    end
    dotted_sample = _julianic_dotted_sample(statement)
    if dotted_sample !== nothing
        lhs, rhs = dotted_sample.lhs, dotted_sample.rhs
        lhs isa Symbol || throw(ArgumentError(
            "julianic @jmodel broadcast observation LHS must be a name; got `$lhs`"))
        data_var = gensym(lhs)
        logpdf_var = gensym(:logpdf)
        buffer_var = gensym(:obs)
        # The user already wrote the RHS dotted, so score elementwise WITHOUT
        # re-dotting it: `y .~ Poisson.(exp(r))` fills the pooled buffer with
        # `lp.(Poisson.(exp(r)), y)` and sums it — same 0-alloc path as `@.`.
        fill = :($buffer_var .= $(Expr(:., logpdf_var, Expr(:tuple, rhs, data_var))))
        return quote
            $(logpdf_var) = $(_obs_kernel)($ctx)
            $(data_var) = $(_observation)($data, $(Val(lhs)))
            $(buffer_var) = $(_obs_begin!)($ctx, length($data_var))
            $fill
            $(_obs_end!)($ctx, $(Val(lhs)), $buffer_var)
        end
    end
    scalar_sample = _julianic_scalar_sample(statement)
    if scalar_sample !== nothing
        lhs, rhs = scalar_sample.lhs, scalar_sample.rhs
        lhs isa Symbol || throw(ArgumentError(
            "julianic @jmodel scalar site LHS must be a name (milestone 1); got `$lhs`"))
        return :($(lhs) = $(_sample_site!)($ctx, $data, $(Val(lhs)), $rhs))
    end
    # Deterministic assignment `lhs = rhs`: buffer every array-valued node in rhs.
    if statement isa Expr && statement.head === :(=)
        prelude = Any[]
        rewritten = _julianic_buffer_expression!(prelude, statement.args[2], ctx)
        isempty(prelude) && return statement
        return Expr(:block, prelude..., Expr(:(=), statement.args[1], rewritten))
    end
    return statement
end

# Any `~` still present AFTER lowering is a sampling statement nested inside
# control flow (`for` / `if` / `begin`), which milestone 1 does not lower — it
# would survive verbatim and fail at run time with an `UndefVarError` naming the
# site, far from its cause. Reject it at macro-expansion time instead.
function _julianic_assert_lowered(expr)
    expr isa Expr || return nothing
    if _julianic_scalar_sample(expr) !== nothing ||
       _julianic_dotted_sample(expr) !== nothing ||
       _julianic_broadcast_sample(expr) !== nothing
        throw(ArgumentError(
            "julianic @jmodel: `~` is only lowered at the TOP LEVEL of the " *
            "model body, but found a nested sampling statement `" *
            string(expr) * "`. Hoist it to the top level; a looped/vector site " *
            "is written as one multivariate `~` (`b ~ product_distribution(...)`) " *
            "and indexed afterwards in ordinary Julia."))
    end
    for argument in expr.args
        _julianic_assert_lowered(argument)
    end
    return nothing
end

function _julianic_model_syntax(definition)
    definition isa Expr && definition.head === :function ||
        throw(ArgumentError("julianic @jmodel must wrap a function definition"))
    signature, body = definition.args
    signature isa Expr && signature.head === :call ||
        throw(ArgumentError("julianic @jmodel requires a named function definition"))
    function_name = first(signature.args)
    arguments = signature.args[2:end]
    input_names = Tuple(Symbol[_julianic_argument_name(a) for a in arguments])

    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    ctx = gensym(:ctx)
    data = gensym(:data)
    lowered = Any[]
    for statement in statements
        statement isa LineNumberNode && (push!(lowered, statement); continue)
        # Drop an explicit `return` — the density is the accumulator, not the
        # body's return value (the return only named outputs in `@model`).
        if statement isa Expr && statement.head === :return
            continue
        end
        lowered_statement = _julianic_lower_statement(statement, ctx, data)
        _julianic_assert_lowered(lowered_statement)
        push!(lowered, lowered_statement)
    end

    run_closure = Expr(:function, Expr(:call, gensym(:run), ctx, data),
                       Expr(:block, lowered..., nothing))
    model_expr = :($(JulianicModel)($run_closure, $(input_names)))
    return Expr(:function, signature, Expr(:block, model_expr))
end

_julianic_argument_name(argument::Symbol) = argument
function _julianic_argument_name(argument::Expr)
    argument.head === :(::) && return _julianic_argument_name(argument.args[1])
    argument.head === :kw && return _julianic_argument_name(argument.args[1])
    throw(ArgumentError("julianic @jmodel unsupported argument form `$argument`"))
end

"""
    @jmodel function name(inputs...)
        latent ~ Dist(...)          # prep + logpdf, consumes an unconstrained slice
        block  ~ product_distribution(...)   # k contiguous coordinates
        deterministic = f(latent)   # ordinary Julia, kept verbatim
        @. observed ~ Dist(...)     # accumulate data logpdf (RHS gets dotted)
        other .~ Dist.(...)         # same, RHS already dotted by the author
    end

Julianic NativePPL model: the body is valid SSA Julia except `~`, which reduces
to prep + logpdf. Calling `name(inputs...)` returns a `JulianicModel`; condition
it with `jcondition`, `jprepare` it, then evaluate with `jlogdensity` /
`jlogdensity_and_gradient!` (see `jworkspace` for the required AD backend).

A site is a LATENT when written scalar (`s ~ D`) and an OBSERVATION when written
broadcast (`@. y ~ D` or `y .~ D.(...)`). Both spellings are checked against the
names passed to `jcondition` and fail closed on a mismatch, so a conditioned
name can never be silently resampled as a latent. `~` is lowered at the TOP
LEVEL of the body only; a nested one is a macro-expansion error rather than an
opaque runtime failure.
"""
macro jmodel(definition)
    esc(_julianic_model_syntax(definition))
end
