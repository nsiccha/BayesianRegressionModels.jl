# ---- Cell latents in total coordinates: `coordinates = :logsumexp | :sum` ----
#
# `eta ~ Normal(location, scale; coordinates = :logsumexp)` in a `@plate` cell
# declares the same latent as without the keyword, sampled in other
# coordinates. Per cell, `c[1] = logsumexp(eta)` and `c[2:K]` are the
# isometric sum-to-zero coordinates of `eta .- mean(eta)` (the basis of Stan's
# `sum_to_zero_vector`), so
#
#     eta = c[1] - logsumexp(z) .+ z,    z = V * c[2:K].
#
# The map has a constant Jacobian determinant (√K per cell): `eta -> (mean,
# V' * centred)` is orthogonal up to √K, and replacing the mean by
# `logsumexp(eta)` shifts it by a function of the centred part alone. So the
# density of `c` is the law's density at `eta(c)` with nothing added, and an
# observation of `sum(exp.(eta))` with a small error pins `c[1]` alone instead
# of a curved ridge across all K entries (snag `exact-sum-constr-7ba55c76`,
# user decision `18wcqyi`).
#
# `:sum` makes the latent's own total the coordinate. A positive latent `T`
# (support `(0, Inf)`) is `exp.(eta)` with `eta` in the coordinates above, so
# `c[1] = log(sum(T))`, and its density gains `sum(eta)`, the term Stan's
# positivity transform adds. A real latent `x` is `c[1] / K .+ V * c[2:K]`, a
# linear map, so `c[1] = sum(x)`.
#
# SBBRMI declares `c` as an ordinary per-cell latent `<name>_coordinates`
# whose law is the BRM family below (the authored law as its argument) and
# binds `<name>` to the mapped values, the cell local the rest of the cell
# reads. RKPPL has no per-cell vector latent with a joint law yet, so RKBRMI
# refuses the keyword, naming the ReactiveKernels snag that tracks it.

const _BRM_TOTAL_COORDINATE_MODES = (:logsumexp, :sum)

# The coordinates a law asks for, and the law without the keyword; `nothing`
# when the law has no `coordinates` keyword.
function _brm_total_coordinates_keyword(law)
    Meta.isexpr(law, :call) || return nothing
    mode = nothing
    rest = Any[]
    for arg in law.args[2:end]
        if Meta.isexpr(arg, :parameters)
            kept = Any[]
            for kw in arg.args
                key = Meta.isexpr(kw, :kw, 2) ? kw.args[1] : kw
                if key === :coordinates
                    Meta.isexpr(kw, :kw, 2) || error(
                        "@brm: `coordinates` takes a value: `coordinates = :logsumexp` or `:sum`")
                    mode = kw.args[2]
                else
                    push!(kept, kw)
                end
            end
            isempty(kept) || push!(rest, Expr(:parameters, kept...))
        elseif Meta.isexpr(arg, :kw, 2) && arg.args[1] === :coordinates
            mode = arg.args[2]
        else
            push!(rest, arg)
        end
    end
    mode === nothing && return nothing
    mode isa QuoteNode && (mode = mode.value)
    mode in _BRM_TOTAL_COORDINATE_MODES || error(
        "@brm: `coordinates = $(repr(mode))` is not a coordinate system of a cell " *
        "latent; use `:logsumexp` (the log of `sum(exp.(x))`) or `:sum` (the " *
        "latent's own sum)")
    (; mode, law=Expr(:call, law.args[1], rest...))
end

# Which total the coordinates take, from the law's support: `:logsumexp` and
# `:sum` on a real latent, `:sum` on a positive one. Anything else has no
# total coordinate.
function _brm_total_coordinates_kind(mode, family, lhs)
    family isa Type && family <: UnivariateDistribution &&
        family <: ContinuousDistribution || error(
        "@brm: cell latent `$lhs` with `coordinates = :$mode` needs a continuous " *
        "univariate Distributions family such as `Normal(…)` or `LogNormal(…)`; got `$family`")
    lo, hi = _brm_total_family_support(family, lhs)
    if lo == -Inf && hi == Inf
        mode === :logsumexp ? :logsumexp : :sum
    elseif lo == 0 && hi == Inf
        mode === :sum || error(
            "@brm: cell latent `$lhs` is positive (`$family`), so its total is its " *
            "own sum: use `coordinates = :sum`, which samples `log(sum($lhs))`")
        :logsum
    else
        error("@brm: cell latent `$lhs` lives on [$lo, $hi] (`$family`); total " *
            "coordinates need a real or a positive latent")
    end
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

# The coordinate latent's name for a latent `name`.
_brm_total_coordinates_name(name::Symbol) = Symbol(name, "_coordinates")

# ---- Julia maps (brm_start, tests) -------------------------------------------

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

# Coordinates of one cell's latent values, and the values of coordinates.
function _brm_total_coordinates(kind::Symbol, v::AbstractVector{<:Real})
    isempty(v) && return Float64[]
    kind === :logsum && return _brm_total_coordinates(:logsumexp, log.(v))
    centred = v .- sum(v) / length(v)
    total = kind === :logsumexp ? _brm_logsumexp(v) : sum(v)
    vcat(total, _brm_sum_to_zero_free(centred))
end
function _brm_total_values(kind::Symbol, c::AbstractVector{<:Real})
    isempty(c) && return Float64[]
    kind === :logsum && return exp.(_brm_total_values(:logsumexp, c))
    z = _brm_sum_to_zero_values(c[2:end])
    kind === :logsumexp ? c[1] .- _brm_logsumexp(z) .+ z : c[1] / length(c) .+ z
end

# ---- StanBlocks families -----------------------------------------------------
#
# `c` holds the coordinates. The law's arguments follow `family`; the first
# vector-valued one sizes the latent (its other arguments may be scalars or
# vectors of that length), as the authored law broadcasts over them.
StanBlocks.@deffun begin
    # Stan's isometric sum-to-zero vector from the free coordinates `c[2:n]`
    # (Stan's own `sum_to_zero_constrain`, vectorised: `w - cumulative_sum(w)
    # + sum(w)` holds the suffix sums of `w`). The padded `segment` keeps a
    # one-entry cell, whose total is the latent itself, in range.
    brm_latent_split(c::vector[n])::vector[n] = begin
        j = cumulative_sum(rep_vector(1.0, n - 1))
        w = segment(append_row(c, 0.0), 2, n - 1) ./ sqrt(j .* (j + 1))
        append_row(w - cumulative_sum(w) + sum(w), 0.0) - append_row(0.0, j .* w)
    end
    # The coordinates of `x`: `total`, then the free coordinates of `x`'s
    # centred values (the inverse of `brm_latent_split`).
    brm_latent_coordinates(total::real, x::vector[n])::vector[n] = begin
        z = x - mean(x)
        j = cumulative_sum(rep_vector(1.0, n - 1))
        s = segment(cumulative_sum(z), 1, n - 1)
        append_row(total, (s - j .* segment(append_row(z, 0.0), 2, n - 1)) ./ sqrt(j .* (j + 1)))
    end

    brm_latent_logsumexp_values(c::vector[n])::vector[n] = begin
        z = brm_latent_split(c)
        c[1] - log_sum_exp(z) + z
    end
    brm_latent_sum_values(c::vector[n])::vector[n] = c[1] / n + brm_latent_split(c)
    brm_latent_logsum_values(c::vector[n])::vector[n] = exp(brm_latent_logsumexp_values(c))

    brm_latent_logsumexp_coordinates(x::vector[n])::vector[n] =
        brm_latent_coordinates(log_sum_exp(x), x)
    brm_latent_sum_coordinates(x::vector[n])::vector[n] = brm_latent_coordinates(sum(x), x)
    brm_latent_logsum_coordinates(x::vector[n])::vector[n] =
        brm_latent_logsumexp_coordinates(log(x))

    # Densities of the coordinates: the law at the mapped values (the
    # Jacobian is constant); a positive latent adds `sum(log(T))`.
    @lpxf brm_latent_logsumexp_lpdf(c::vector[n], family, a::vector[n], args...)::real =
        density(family, brm_latent_logsumexp_values(c), a, args...)
    brm_latent_logsumexp_lpdf(c::vector[n], family, a::real, b::vector[n], args...)::real =
        density(family, brm_latent_logsumexp_values(c), a, b, args...)
    brm_latent_logsumexp_lpdfs(c::vector[n], family, args...)::vector[n] =
        append_row(brm_latent_logsumexp_lpdf(c, family, args...), rep_vector(0.0, n - 1))
    brm_latent_logsumexp_rng(family, a::vector[n], args...)::vector[n] =
        brm_latent_logsumexp_coordinates(to_vector(predictive(family, a, args...)))
    brm_latent_logsumexp_rng(family, a::real, b::vector[n], args...)::vector[n] =
        brm_latent_logsumexp_coordinates(to_vector(predictive(family, a, b, args...)))

    @lpxf brm_latent_sum_lpdf(c::vector[n], family, a::vector[n], args...)::real =
        density(family, brm_latent_sum_values(c), a, args...)
    brm_latent_sum_lpdf(c::vector[n], family, a::real, b::vector[n], args...)::real =
        density(family, brm_latent_sum_values(c), a, b, args...)
    brm_latent_sum_lpdfs(c::vector[n], family, args...)::vector[n] =
        append_row(brm_latent_sum_lpdf(c, family, args...), rep_vector(0.0, n - 1))
    brm_latent_sum_rng(family, a::vector[n], args...)::vector[n] =
        brm_latent_sum_coordinates(to_vector(predictive(family, a, args...)))
    brm_latent_sum_rng(family, a::real, b::vector[n], args...)::vector[n] =
        brm_latent_sum_coordinates(to_vector(predictive(family, a, b, args...)))

    @lpxf brm_latent_logsum_lpdf(c::vector[n], family, a::vector[n], args...)::real = begin
        eta = brm_latent_logsumexp_values(c)
        density(family, exp(eta), a, args...) + sum(eta)
    end
    brm_latent_logsum_lpdf(c::vector[n], family, a::real, b::vector[n], args...)::real = begin
        eta = brm_latent_logsumexp_values(c)
        density(family, exp(eta), a, b, args...) + sum(eta)
    end
    brm_latent_logsum_lpdfs(c::vector[n], family, args...)::vector[n] =
        append_row(brm_latent_logsum_lpdf(c, family, args...), rep_vector(0.0, n - 1))
    brm_latent_logsum_rng(family, a::vector[n], args...)::vector[n] =
        brm_latent_logsum_coordinates(to_vector(predictive(family, a, args...)))
    brm_latent_logsum_rng(family, a::real, b::vector[n], args...)::vector[n] =
        brm_latent_logsum_coordinates(to_vector(predictive(family, a, b, args...)))
end

const _BRM_TOTAL_FAMILIES = (; logsumexp=(brm_latent_logsumexp, brm_latent_logsumexp_values),
    sum=(brm_latent_sum, brm_latent_sum_values), logsum=(brm_latent_logsum, brm_latent_logsum_values))

# ---- SBBRMI: the cell statements ----------------------------------------------
#
# `name ~ Family(args…; coordinates = mode)` becomes, in StanBlocks spelling,
#
#     name_coordinates ~ brm_latent_<kind>(family, args…)
#     name = brm_latent_<kind>_values(name_coordinates)
#
# `nothing` for any other statement.
function _sb_cell_total_coordinates(x, scope)
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ || return nothing
    spec = _brm_total_coordinates_keyword(x.args[3])
    spec === nothing && return nothing
    lhs = x.args[2]
    lhs isa Symbol || error(
        "@brm: `coordinates = :$(spec.mode)` declares the coordinates of a cell " *
        "latent; `$lhs` is not a bare cell name")
    callable = _sb_cell_binding(scope, first(spec.law.args))
    kind = _brm_total_coordinates_kind(spec.mode, callable, lhs)
    call = _sb_cell_family(spec.law, scope)
    Meta.isexpr(call, :call) && first(call.args) isa Symbol &&
        isdefined(StanBlocks, first(call.args)) || error(
        "@brm: internal: cell latent `$lhs`'s law has no StanBlocks family token")
    family = getfield(StanBlocks, first(call.args))
    density, values = getproperty(_BRM_TOTAL_FAMILIES, kind)
    coordinates = _brm_total_coordinates_name(lhs)
    Any[Expr(:call, :~, coordinates, Expr(:call, density, family, call.args[2:end]...)),
        Expr(:(=), lhs, Expr(:call, values, coordinates))]
end

# The cell latents a kernel body declares in total coordinates, with the
# coordinate kind each one takes. Shared by the backends' checks and
# `brm_start`.
function _brm_cell_total_latents(body_stmts, scope)
    found = Dict{Symbol,Symbol}()
    for x in body_stmts
        Meta.isexpr(x, :call, 3) && x.args[1] === :~ || continue
        spec = _brm_total_coordinates_keyword(x.args[3])
        spec === nothing && continue
        lhs = x.args[2]
        lhs isa Symbol || continue
        callable = _sb_cell_binding(scope, first(spec.law.args))
        found[lhs] = _brm_total_coordinates_kind(spec.mode, callable, lhs)
    end
    found
end

# A latent in total coordinates is a cell latent: a bare name that is no
# sliced input of the cell and no `=` binding of it.
function _brm_check_cell_total_latents(body_stmts, params, scope; prefix="@brm")
    for (lhs, _) in _brm_cell_total_latents(body_stmts, scope)
        lhs in params && error(
            "$prefix: `$lhs ~ …(…; coordinates = …)` observes the cell's input `$lhs`; " *
            "total coordinates are a way to sample a latent of the cell")
        any(s -> Meta.isexpr(s, :(=)) && first(s.args) === lhs, body_stmts) && error(
            "$prefix: `$lhs ~ …(…; coordinates = …)` needs a name no `=` in the cell binds")
    end
end

# The latents in total coordinates across a model's kernel cells, name => kind.
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

# `brm_start` gives such a latent its values, one vector per cell or one flat
# vector; the start writes their coordinates.
function _brm_total_start_target(t, kind, d)
    coordinates = _brm_total_coordinates_name(t.name)
    isempty(brm_outputs(d; logical=coordinates)) && error(
        "brm_start: cell latent `$(t.name)` in total coordinates has no sampled " *
        "coordinates `$coordinates` in this model")
    output = brm_output(d, coordinates)
    values = _brm_start_flat_values(t.name, output, t.value)
    kind === :logsum && any(<=(0), values) && error(
        "brm_start: cell latent `$(t.name)` is positive; its start needs positive values")
    ends = something(output.segments, [length(values)])
    starts = [1; ends[1:end-1] .+ 1]
    coords = reduce(vcat, (_brm_total_coordinates(kind, values[a:b])
        for (a, b) in zip(starts, ends)); init=Float64[])
    all(isfinite, coords) || error(
        "brm_start: cell latent `$(t.name)`'s values have no total coordinates" *
        (kind === :logsum ? " (a positive latent needs positive values)" : ""))
    merge(t, (; name=coordinates, value=coords))
end
