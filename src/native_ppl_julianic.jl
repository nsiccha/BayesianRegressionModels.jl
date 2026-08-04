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
# * A latent `sym ~ dist` lowers to `sym = _sample!(ctx, Val(:sym), dist)`.
# * A broadcast observation `@. lhs ~ dist` lowers to accumulating
#   `sum(@. logpdf(dist, <conditioned data for lhs>))`.
# * The body runs twice against a context `ctx`:
#     - TRACE (compile): discover the total unconstrained dimension by counting
#       the coordinates each latent `~` consumes. Returns in-support placeholder
#       values so deterministic code and observations run without error.
#     - PRIMAL (density): a cursor walks `θ`; each latent `~` consumes its slice,
#       preps it (constrain + logjac), and accumulates prior logpdf; each
#       observation accumulates its data logpdf. The scalar `ctx.acc` is the
#       primal `θ -> Real` that Enzyme differentiates (same AD path the
#       declarative executor already uses — DifferentiationInterface + AutoEnzyme
#       straight through the primal kernel).
#
# Milestone 1 supports scalar `Normal`/`Exponential` latents (real & positive
# support) and broadcast `Normal` observations — enough for the walking-skeleton
# Gaussian and the centered-hierarchical model. Later milestones add vector /
# grouped / constrained-manifold sites by adding `_sample!` methods.
# ---------------------------------------------------------------------------

# NOTE: `Normal`/`Exponential`/`StandardNormal` are ALSO names inside NativePPL
# (prior/declaration constructors), so we must NOT `using Distributions`-import
# them here — fully qualify the Distributions types/functions instead. In the
# macro-generated model body, `Normal`/`logpdf` deliberately resolve in the
# CALLER's scope (the user's `using Distributions`), keeping the body real Julia.
import Distributions

# --- Contexts -------------------------------------------------------------

# PRIMAL: walks θ with a cursor and accumulates the log density into `acc`.
# `acc` is parameterized on θ's element type (concrete, type-stable) so Enzyme
# differentiates through the accumulation cleanly.
#
# The observation DATA is deliberately NOT a field here: it is a run-time
# constant, and mixing a constant array with the active `theta`/`acc` inside one
# mutable struct is exactly what forced Enzyme's `set_runtime_activity`. Instead
# the body's `run` closure takes the data as a SEPARATE argument (passed through
# DI as a `Constant`), so this struct holds only active/inactive-scalar state and
# STATIC activity suffices — no runtime-activity workaround.
mutable struct JulianicPrimal{TH<:AbstractVector,A}
    theta::TH
    cursor::Int
    acc::A
end
JulianicPrimal(theta::AbstractVector) = JulianicPrimal(theta, 0, zero(eltype(theta)))

# TRACE: runs the body once to size the unconstrained coordinate vector.
mutable struct JulianicTrace
    dim::Int
end
JulianicTrace() = JulianicTrace(0)

# --- `~` runtime: prep + logpdf ------------------------------------------
#
# Each latent method advances the cursor by the site's unconstrained dimension,
# applies the support's transform (prep), accumulates prior logpdf + logjac, and
# returns the CONSTRAINED value (so downstream body code sees the natural-space
# quantity). Trace methods only grow `dim` and return an in-support placeholder.

# Normal: real support, identity transform (log-Jacobian 0). Covers `Normal()`
# (standard normal) and `Normal(mu, sigma)` with latent-dependent parameters.
@inline function _sample!(ctx::JulianicPrimal, ::Val, dist::Distributions.Normal)
    ctx.cursor += 1
    v = @inbounds ctx.theta[ctx.cursor]
    ctx.acc += Distributions.logpdf(dist, v)
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Normal) = (ctx.dim += 1; 0.0)

# Exponential: positive support, exp transform. constrain u -> exp(u); the
# log-Jacobian of that transform is exactly u, so accumulate logpdf + u.
@inline function _sample!(ctx::JulianicPrimal, ::Val, dist::Distributions.Exponential)
    ctx.cursor += 1
    u = @inbounds ctx.theta[ctx.cursor]
    v = exp(u)
    ctx.acc += Distributions.logpdf(dist, v) + u
    return v
end
@inline _sample!(ctx::JulianicTrace, ::Val, ::Distributions.Exponential) = (ctx.dim += 1; 1.0)

# Multivariate real-support latent (e.g. MvNormal, `product_distribution` of
# real-support univariates): identity transform on every coordinate (support is
# all of R^k, so the log-Jacobian is 0). Consumes `length(dist)` contiguous
# coordinates and returns the value vector for the body to index. This covers
# grouped/blocked latents authored as `b ~ product_distribution(fill(D, k))`
# or `b ~ MvNormal(...)`, then referenced as `b[group]`.
@inline function _sample!(ctx::JulianicPrimal, ::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    lo = ctx.cursor + 1
    ctx.cursor += k
    v = @view ctx.theta[lo:ctx.cursor]   # view, not a copy — avoids an allocation
    ctx.acc += Distributions.logpdf(dist, v)
    return v
end
@inline function _sample!(ctx::JulianicTrace, ::Val, dist::Distributions.MultivariateDistribution)
    k = length(dist)
    ctx.dim += k
    return zeros(k)
end

# Multivariate POSITIVE-support latent: a `product_distribution` of `Exponential`
# marginals — used for a vector of grouped scales `tau ~
# product_distribution(fill(Exponential(1), K))`. Positive support → exp
# transform per coordinate (constrain u -> exp(u), log-Jacobian u), so accumulate
# per-marginal logpdf + sum(u). More specific than the real-support method above,
# so it wins for the `Product{…,Exponential}` type (a product of Normals resolves
# to `DiagNormal <: MvNormal`, which correctly stays on the identity path).
@inline function _sample!(ctx::JulianicPrimal, ::Val,
        dist::Distributions.Product{Distributions.Continuous,<:Distributions.Exponential})
    k = length(dist)
    lo = ctx.cursor + 1
    ctx.cursor += k
    u = @view ctx.theta[lo:ctx.cursor]   # view, not a copy
    v = exp.(u)
    ctx.acc += sum(Distributions.logpdf.(dist.v, v)) + sum(u)
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

@inline function _sample!(ctx::JulianicPrimal, ::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    m = K * (K - 1) ÷ 2
    lo = ctx.cursor + 1
    ctx.cursor += m
    coords = @view ctx.theta[lo:lo + m - 1]   # view, not a copy
    ctx.acc += _julianic_lkj_logdensity(K, dist.η, coords)
    return _julianic_corr_cholesky(K, coords)
end
@inline function _sample!(ctx::JulianicTrace, ::Val, dist::Distributions.LKJCholesky)
    K = dist.d
    ctx.dim += K * (K - 1) ÷ 2
    L = zeros(Float64, K, K)                # in-support placeholder (L = I)
    for i in 1:K
        L[i, i] = 1.0
    end
    return L
end

# Observation log density. The `~` magic (not the user) supplies `logpdf`, so we
# route through this runtime helper — the model body needs no `logpdf` in scope,
# only the distribution constructor the user actually wrote. Broadcast-safe.
@inline _obs_logpdf(dist, x) = Distributions.logpdf(dist, x)

# Accumulate an already-computed observation log-density contribution.
@inline _accumulate!(ctx::JulianicPrimal, contribution) =
    (ctx.acc += contribution; nothing)
@inline _accumulate!(::JulianicTrace, _contribution) = nothing

# Fetch conditioned data for an observation site. The data is passed to the body
# as a separate argument (not stored in the differentiated context — see
# `JulianicPrimal`), so this reads straight from that constant `data` value.
@inline _observation(data, ::Val{name}) where {name} = getproperty(data, name)

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

# After the trace: adds the discovered unconstrained dimension.
struct JulianicPrepared{R,D}
    run::R
    data::D
    dimension::Int
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
    return JulianicPrepared(conditioned.run, conditioned.data, trace.dim)
end

dimension(prepared::JulianicPrepared) = prepared.dimension

# The primal kernel `θ -> Real`. Top-level (not a closure over θ) so the AD
# backend sees `prepared` as a constant and `theta` as the sole active input.
function _julianic_kernel(theta::AbstractVector, prepared::JulianicPrepared)
    ctx = JulianicPrimal(theta)
    prepared.run(ctx, prepared.data)
    return ctx.acc
end

"""
    jlogdensity(prepared::JulianicPrepared, theta::AbstractVector)

Evaluate the log density by running the body as the primal.
"""
jlogdensity(prepared::JulianicPrepared, theta::AbstractVector) =
    _julianic_kernel(theta, prepared)

# Gradient plumbing. DifferentiationInterface lives in a weak-dep extension, so
# these are declared here and given methods in
# ext/BayesianRegressionModelsDifferentiationInterfaceExt.jl.
function _julianic_prepare_gradient end
function _julianic_value_and_gradient! end

struct JulianicWorkspace{P,G}
    di_preparation::P
    gradient::G
end

"""
    jworkspace(prepared, T, backend)

Allocate a gradient workspace (AD preparation + gradient buffer) for a julianic
prepared model at element type `T`.
"""
function jworkspace(prepared::JulianicPrepared, ::Type{T}, backend) where {T<:AbstractFloat}
    position = zeros(T, prepared.dimension)
    di_preparation = _julianic_prepare_gradient(prepared, backend, position)
    JulianicWorkspace(di_preparation, zeros(T, prepared.dimension))
end

"""
    jlogdensity_and_gradient!(workspace, prepared, theta) -> (density, gradient)

Value + gradient of the log density via Enzyme straight through the run-the-body
primal.
"""
function jlogdensity_and_gradient!(
    workspace::JulianicWorkspace, prepared::JulianicPrepared, theta::AbstractVector)
    return _julianic_value_and_gradient!(
        prepared, workspace.di_preparation, workspace.gradient, theta)
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

# Lower one statement. Only `~` statements are rewritten; everything else is
# returned verbatim (that is the whole point — the body is ordinary Julia).
# `ctx` names the (mutable) sampling context; `data` names the conditioned
# observation data — both are parameters of the generated `run` closure.
function _julianic_lower_statement(statement, ctx, data)
    broadcast_sample = _julianic_broadcast_sample(statement)
    if broadcast_sample !== nothing
        lhs, rhs = broadcast_sample.lhs, broadcast_sample.rhs
        lhs isa Symbol || throw(ArgumentError(
            "julianic @jmodel broadcast observation LHS must be a name; got `$lhs`"))
        data_var = gensym(lhs)
        # `@__dot__` only dots calls whose head is a *symbol*, so bind the
        # runtime logpdf to a local name before broadcasting. `@. lp(D(a,b), y)`
        # then lowers to `lp.(D.(a,b), y)` — the constructor is dotted (so vector
        # parameters broadcast row-wise) and each element's logpdf is scored.
        logpdf_var = gensym(:logpdf)
        dotted = Expr(:macrocall, Symbol("@__dot__"), LineNumberNode(0),
                      Expr(:call, logpdf_var, rhs, data_var))
        return quote
            $(logpdf_var) = $(_obs_logpdf)
            $(data_var) = $(_observation)($data, $(Val(lhs)))
            $(_accumulate!)($ctx, sum($(dotted)))
        end
    end
    scalar_sample = _julianic_scalar_sample(statement)
    if scalar_sample !== nothing
        lhs, rhs = scalar_sample.lhs, scalar_sample.rhs
        lhs isa Symbol || throw(ArgumentError(
            "julianic @jmodel scalar site LHS must be a name (milestone 1); got `$lhs`"))
        return :($(lhs) = $(_sample!)($ctx, $(Val(lhs)), $rhs))
    end
    return statement
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
        push!(lowered, _julianic_lower_statement(statement, ctx, data))
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
        deterministic = f(latent)   # ordinary Julia, kept verbatim
        @. observed ~ Dist(...)     # accumulate data logpdf
    end

Julianic NativePPL model: the body is valid SSA Julia except `~`, which reduces
to prep + logpdf. Calling `name(inputs...)` returns a `JulianicModel`; condition
it with `jcondition`, `jprepare` it, then evaluate with `jlogdensity` /
`jlogdensity_and_gradient!`.
"""
macro jmodel(definition)
    esc(_julianic_model_syntax(definition))
end
