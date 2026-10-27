# ---- Cell latents with a given total: `T::shares(total) ~ Law(…)` ------------
#
# In a `@plate` cell, `T::shares(land[s]) ~ LogNormal(location, scale)` declares
# a positive latent that sums to `land[s]` in every draw, with `LogNormal` as
# its law on that surface. The sampler moves the K − 1 isometric coordinates
# `y` of the shares `T ./ total`, the coordinates Stan's `simplex[K]` uses:
#
#     T = exp.(log(total) - logsumexp(z) .+ z),    z = sum_to_zero_constrain(y).
#
# The density of `y` is the law's at `T` plus `sum(log.(T)) - log(total)`: the
# joint of `T` and its total, conditioned on `sum(T) == total`, in these
# coordinates. Write `T = total .* w`: the surface element of `y -> T` is
# `total^(K - 1)` times that of `y -> w`, which Stan's simplex transform adds as
# `sum(log.(w))` up to a constant; together `sum(log.(T)) - log(total)`. For a
# total from data this is the exact conditional given the observed total (the
# limit of an observation of `sum(T)` whose error goes to 0), and the last term
# is a constant. For a total that is a model value, the law of `T` and the
# total's own law both act on the total (snag `exact-sum-constr-7ba55c76`, user
# decisions `18wcqyi`, `10kbiuy` and `1a5fvjo`).
#
# The typed left-hand side is rewritten, before the formula is parsed, into
# internal law keywords `_brm_bijection = :shares, _brm_total = total` that the
# backends read. SBBRMI declares `y` as an ordinary per-cell latent
# `<name>_coordinates` (K − 1 per cell) whose law is the BRM family below (the
# authored law as its argument) and binds `<name>` to the mapped values, the
# cell local the rest of the cell reads. RKBRMI declares `<name>` as one joint
# latent per cell whose law, `_BRMShares`, carries this geometry (the
# ReactiveKernels extension).

# The bijections a cell latent's typed left-hand side names.
const _BRM_CELL_BIJECTIONS = (:shares,)

# `@plate` cells' `name::shares(total) ~ Law(args…)` become `name ~ Law(args…;
# _brm_bijection = :shares, _brm_total = total)`; everything else is unchanged.
function _brm_desugar_cell_bijections(x)
    x isa Expr || return x
    _is_brm_plate(x) && return Expr(x.head, map(_brm_desugar_plate_loop, x.args)...)
    Expr(x.head, map(_brm_desugar_cell_bijections, x.args)...)
end
_brm_desugar_plate_loop(x) = Meta.isexpr(x, :for, 2) ?
    Expr(:for, x.args[1], _brm_desugar_cell_body(x.args[2])) : x
_brm_desugar_cell_body(x) = Meta.isexpr(x, :block) ?
    Expr(:block, map(_brm_desugar_cell_statement, x.args)...) : _brm_desugar_cell_statement(x)

function _brm_desugar_cell_statement(x)
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ && Meta.isexpr(x.args[2], :(::), 2) ||
        return x
    name, type = x.args[2].args
    law = x.args[3]
    name isa Symbol || error(
        "@brm: a typed left-hand side `$(x.args[2])` names a cell latent and its " *
        "bijection, `T::shares(total) ~ …`; `$name` is not a bare name")
    bijection = Meta.isexpr(type, :call) ? first(type.args) : type
    bijection in _BRM_CELL_BIJECTIONS || error(
        "@brm: `$name::$type` names no bijection of a cell latent; the typed " *
        "left-hand side takes " * join(("`$b(total)`" for b in _BRM_CELL_BIJECTIONS), ", ") *
        " (`T::shares(total) ~ LogNormal(…)`: positive, summing to `total` in every draw)")
    Meta.isexpr(type, :call) && length(type.args) == 2 || error(
        "@brm: `$name::$type` takes the total the latent sums to: `$name::shares(total)`")
    Meta.isexpr(law, :call) || error(
        "@brm: `$name::$type ~ $law` needs a law call such as `LogNormal(location, scale)`")
    Expr(:call, :~, name, Expr(:call, law.args...,
        Expr(:kw, :_brm_bijection, QuoteNode(bijection)), Expr(:kw, :_brm_total, type.args[2])))
end

# The bijection a law carries, its total and the law without them; `nothing`
# for a law without one.
function _brm_cell_bijection(law)
    Meta.isexpr(law, :call) || return nothing
    bijection = total = nothing
    rest = Any[]
    for arg in law.args[2:end]
        if Meta.isexpr(arg, :kw, 2) && arg.args[1] === :_brm_bijection
            bijection = arg.args[2] isa QuoteNode ? arg.args[2].value : arg.args[2]
        elseif Meta.isexpr(arg, :kw, 2) && arg.args[1] === :_brm_total
            total = arg.args[2]
        else
            push!(rest, arg)
        end
    end
    bijection === nothing && return nothing
    (; bijection, total, law=Expr(:call, law.args[1], rest...))
end

# `shares` takes a positive latent: its law is a continuous univariate family
# on `(0, Inf)`.
function _brm_check_shares_family(family, lhs)
    family isa Type && family <: UnivariateDistribution &&
        family <: ContinuousDistribution || error(
        "@brm: `$lhs::shares(…)` needs a continuous univariate Distributions " *
        "family such as `LogNormal(…)` or `Gamma(…)`; got `$family`")
    lo, hi = _brm_total_family_support(family, lhs)
    lo == 0 && hi == Inf || error(
        "@brm: `$lhs::shares(total)` declares a positive latent summing to " *
        "`total`; `$family` lives on [$lo, $hi]. A log-scale latent " *
        "`eta ~ Normal(location, scale)` is the same model as " *
        "`T ~ LogNormal(location, scale)` with `T = exp.(eta)`: write that one")
    nothing
end

# A family's support, read from an instance with unit arguments.
function _brm_total_family_support(family, lhs)
    d = nothing
    for n in 0:3
        d = try
            family(ntuple(_ -> 1.0, n)...)
        catch
            nothing
        end
        d === nothing || break
    end
    d === nothing && error(
        "@brm: cannot read the support of cell latent `$lhs`'s family `$family`")
    (minimum(d), maximum(d))
end

# The coordinate latent's name for a latent `name` (SBBRMI).
_brm_total_coordinates_name(name::Symbol) = Symbol(name, "_coordinates")

# ---- Julia maps (brm_start, the RK geometry, tests) ---------------------------

# Free coordinates of a sum-to-zero vector `z` (length K) in Stan's isometric
# basis, and back.
function _brm_sum_to_zero_free(z::AbstractVector)
    K = length(z)
    s = cumsum(z)
    [(s[j] - j * z[j+1]) / sqrt(j * (j + 1)) for j in 1:K-1]
end
function _brm_sum_to_zero_values(y::AbstractVector)
    K = length(y) + 1
    w = [y[j] / sqrt(j * (j + 1)) for j in 1:K-1]
    tails = reverse(cumsum(reverse(w)))
    [(i <= K - 1 ? tails[i] : 0.0) - (i >= 2 ? (i - 1) * w[i-1] : 0.0) for i in 1:K]
end

_brm_logsumexp(x) = (m = maximum(x); m + log(sum(exp.(x .- m))))

# One cell's coordinates of positive values `T` (they depend on the shares
# only), the log values of coordinates `y` with total `total`, and the values.
function _brm_shares_coordinates(T::AbstractVector{<:Real})
    isempty(T) && return Float64[]
    eta = log.(T)
    _brm_sum_to_zero_free(eta .- sum(eta) / length(eta))
end
function _brm_shares_log_values(y::AbstractVector{<:Real}, total::Real)
    z = _brm_sum_to_zero_values(y)
    log(total) .- _brm_logsumexp(z) .+ z
end
_brm_shares_values(y::AbstractVector{<:Real}, total::Real) = exp.(_brm_shares_log_values(y, total))

# ---- StanBlocks families -----------------------------------------------------
#
# `y` holds a cell's K − 1 coordinates. The law's arguments follow `family`
# and `total`; the first vector-valued one sizes the latent (its other
# arguments may be scalars or vectors of that length), as the authored law
# broadcasts over them.
StanBlocks.@deffun begin
    # `log.(T)` from the coordinates and the total; a one-entry cell has no
    # coordinate (`sum_to_zero_constrain` of an empty vector is `[0.0]`).
    brm_shares_log_values(y::vector[m], total::real)::vector[m + 1] = begin
        z = sum_to_zero_constrain(y)
        log(total) - log_sum_exp(z) + z
    end
    brm_shares_values(y::vector[m], total::real)::vector[m + 1] =
        exp(brm_shares_log_values(y, total))

    # The density of the coordinates: the law at `T` plus
    # `sum(log(T)) - log(total)`.
    @lpxf brm_shares_lpdf(y::vector[m], family, total::real, a::vector[n], args...)::real = begin
        eta = brm_shares_log_values(y, total)
        density(family, exp(eta), a, args...) + sum(eta) - log(total)
    end
    brm_shares_lpdf(y::vector[m], family, total::real, a::real, b::vector[n], args...)::real = begin
        eta = brm_shares_log_values(y, total)
        density(family, exp(eta), a, b, args...) + sum(eta) - log(total)
    end
    # Pointwise, the cell's density spread over its coordinates (none in a
    # one-entry cell).
    brm_shares_lpdfs(y::vector[m], family, args...)::vector[m] =
        rep_vector(brm_shares_lpdf(y, family, args...) / max(m, 1), m)
    # The law conditioned on its sum has no draw in closed form; the latent's
    # size is all a draw supplies.
    brm_shares_rng(family, total::real, a::vector[n], args...)::vector[n - 1] = begin
        @stan_assert false "brm_shares_rng: a draw of a `shares(total)` cell latent is not built"
        rep_vector(0.0, n - 1)
    end
    brm_shares_rng(family, total::real, a::real, b::vector[n], args...)::vector[n - 1] = begin
        @stan_assert false "brm_shares_rng: a draw of a `shares(total)` cell latent is not built"
        rep_vector(0.0, n - 1)
    end
end

# ---- SBBRMI: the cell statements ----------------------------------------------
#
# `name ~ Family(args…; _brm_bijection = :shares, _brm_total = total)`
# becomes, in StanBlocks spelling,
#
#     name_coordinates ~ brm_shares(family, total, args…)
#     name = brm_shares_values(name_coordinates, total)
#
# `nothing` for any other statement.
function _sb_cell_total_coordinates(x, scope)
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ || return nothing
    spec = _brm_cell_bijection(x.args[3])
    spec === nothing && return nothing
    lhs = x.args[2]
    lhs isa Symbol || error(
        "@brm: internal: a `shares` latent `$lhs` is not a bare cell name")
    _brm_check_shares_family(_sb_cell_binding(scope, first(spec.law.args)), lhs)
    call = _sb_cell_family(spec.law, scope)
    Meta.isexpr(call, :call) && first(call.args) isa Symbol &&
        isdefined(StanBlocks, first(call.args)) || error(
        "@brm: internal: cell latent `$lhs`'s law has no StanBlocks family token")
    family = getfield(StanBlocks, first(call.args))
    # The total in the cell's StanBlocks spelling.
    total = _sb_cell_value(spec.total, scope)
    coordinates = _brm_total_coordinates_name(lhs)
    Any[Expr(:call, :~, coordinates, Expr(:call, brm_shares, family, total, call.args[2:end]...)),
        Expr(:(=), lhs, Expr(:call, brm_shares_values, coordinates, total))]
end

# The cell latents a kernel body samples through a bijection, name =>
# bijection. Shared by the backends' checks, the descriptor and `brm_start`.
function _brm_cell_total_latents(body_stmts, scope)
    found = Dict{Symbol,Symbol}()
    for x in body_stmts
        Meta.isexpr(x, :call, 3) && x.args[1] === :~ || continue
        spec = _brm_cell_bijection(x.args[3])
        spec === nothing && continue
        lhs = x.args[2]
        lhs isa Symbol || continue
        _brm_check_shares_family(_sb_cell_binding(scope, first(spec.law.args)), lhs)
        found[lhs] = spec.bijection
    end
    found
end

# Such a latent is a cell latent: a bare name that is no sliced input of the
# cell and no `=` binding of it.
function _brm_check_cell_total_latents(body_stmts, params, scope; prefix="@brm")
    for (lhs, bijection) in _brm_cell_total_latents(body_stmts, scope)
        lhs in params && error(
            "$prefix: `$lhs::$bijection(…) ~ …` observes the cell's input `$lhs`; a " *
            "typed left-hand side declares a latent of the cell")
        any(s -> Meta.isexpr(s, :(=)) && first(s.args) === lhs, body_stmts) && error(
            "$prefix: `$lhs::$bijection(…) ~ …` needs a name no `=` in the cell binds")
    end
end

# The latents sampled through a bijection across a model's kernel cells,
# name => bijection.
function _brm_total_latent_kinds(brmi)
    found = Dict{Symbol,Symbol}()
    for (_, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs_raw = getargs(op, 2)
        rhs = _as_expr_column(rhs_raw); isnothing(rhs) && continue
        getf(rhs) === kernel || continue
        args = getargs(rhs)
        isempty(args) && continue
        lam = _brm_inline_expr(first(args))
        (lam isa Expr && length(lam.args) >= 2) || continue
        body = lam.args[2]
        stmts = Meta.isexpr(body, :block) ? body.args : Any[body]
        merge!(found, _brm_cell_total_latents(stmts, _brm_inline_scope(first(args))))
    end
    found
end

# `brm_start` on SBBRMI gives such a latent its values, one vector per cell or
# one flat vector; the start writes the coordinates of their shares, and
# `brm_start` then checks that the model's values are the given ones (the
# given values must sum to each cell's total).
function _brm_total_start_target(t, bijection, d)
    coordinates = _brm_total_coordinates_name(t.name)
    isempty(brm_outputs(d; logical=coordinates)) && error(
        "brm_start: cell latent `$(t.name)::$bijection(…)` has no sampled coordinates " *
        "`$coordinates` in this model")
    values = _brm_start_flat_values(t.name, brm_output(d, t.name), t.value)
    any(<=(0), values) && error(
        "brm_start: cell latent `$(t.name)::$bijection(…)` is positive; its start needs positive values")
    ends = something(brm_output(d, t.name).segments, [length(values)])
    starts = [1; ends[1:end-1] .+ 1]
    coords = reduce(vcat, (_brm_shares_coordinates(values[a:b])
        for (a, b) in zip(starts, ends)); init=Float64[])
    all(isfinite, coords) || error(
        "brm_start: cell latent `$(t.name)::$bijection(…)`'s values have no finite coordinates")
    merge(t, (; name=coordinates, value=coords))
end

# ---- RKBRMI: the law of one cell's latent --------------------------------------
#
# RKBRMI declares `name[i] ~ _BRMShares{Family}(n, total, args…)` in the cell:
# RKPPL's joint latent per index (rkppl-use §13, "One joint latent per index").
# `n` is the latent's length, `length` of the first law argument that reads
# only data. The family is a type parameter, as RKPPL needs for an inferable
# density; the ReactiveKernels extension gives the law its geometry and density.
struct _BRMShares{F,A<:Tuple}
    args::A
end
(::Type{_BRMShares{F}})(n::Integer, total::Real, args...) where {F} =
    _BRMShares{F,typeof(args)}(args)

# The law's arguments at entry `k` (a number is every entry's), and the check
# that its array arguments have the latent's length.
@inline _brm_shares_entries(args::Tuple, k) =
    map(arg -> arg isa AbstractArray ? arg[k] : arg, args)
@inline function _brm_shares_check_lengths(args::Tuple, T)
    foreach(args) do arg
        arg isa AbstractArray && length(arg) != length(T) &&
            _brm_shares_length_error(length(arg), length(T))
    end
    nothing
end
@noinline _brm_shares_length_error(m, n) = throw(DimensionMismatch(
    "a `shares(total)` cell latent has $n entries in this cell; its law has an " *
    "argument of length $m"))
