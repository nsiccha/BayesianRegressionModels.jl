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

# ---- A completion with group totals: `mi(x)::shares(total, rows) ~ Law(…)` ---
#
# At model scope, with a binding `rows = groups(g => labels)` (or the call
# written in place), `mi(x)::shares(total, rows) ~ Law(args…)` says that the
# completed `x[rows[k]]` sums to `total[k]` for every entry `k` of `labels`
# (snag `mi-completion-wi-54222432`, design decision `0guq37f`). Per group, the
# missing entries are the group's residual, `total[k]` minus its observed
# entries, times shares: the cell latent's bijection above, with the law's
# density at the values plus `sum(log.(x[missing])) - log(residual)`. A group
# with one missing entry sets it to the residual (no coordinate; the law's
# density there still counts). Missing entries of no group stay free.
#
# The typed left-hand side is rewritten, before parsing, into internal `mi`
# keywords naming the three data columns,
# `mi(x; _brm_shares_group = g, _brm_shares_cells = labels, _brm_shares_total = total)`,
# which `_brm_missing_response_plan` turns into `_BRMMissingTotals`.
const _BRM_MI_TOTAL_KEYS = (:_brm_shares_group, :_brm_shares_cells, :_brm_shares_total)

function _brm_desugar_mi_totals(x, groupings, grouped_reads)
    Meta.isexpr(x, :block) && return Expr(:block,
        map(s -> _brm_desugar_mi_totals(s, groupings, grouped_reads), x.args)...)
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ && Meta.isexpr(x.args[2], :(::), 2) &&
        isxcall(x.args[2].args[1], :mi) || return x
    shown = sprint(Base.show_unquoted, x.args[2])
    completion, type = x.args[2].args
    length(completion.args) == 2 && completion.args[2] isa Symbol || error(
        "@brm: `$shown`: a completion with group totals names one data column, " *
        "`mi(x)::shares(total, rows)`")
    bijection = Meta.isexpr(type, :call) ? first(type.args) : type
    bijection === :shares || error(
        "@brm: `$shown` names no bijection of a completion; the typed left-hand side " *
        "is `mi(x)::shares(total, rows)`: the completed `x[rows[k]]` sums to `total[k]`")
    Meta.isexpr(type, :call, 3) && type.args[2] isa Symbol || error(
        "@brm: `$shown`: `mi(x)::shares(total, rows)` takes the data vector of the " *
        "groups' totals and their rows, a binding `rows = groups(g => labels)`")
    total, rows = type.args[2], type.args[3]
    binding = if rows isa Symbol
        haskey(groupings, rows) || error(
            "@brm: `$shown`: `$rows` is no model-scope binding `$rows = groups(g => labels)`")
        push!(grouped_reads, rows)
        groupings[rows]
    else
        call = isxcall(rows, :groups) ? _brm_plate_binding(rows) : nothing
        call === nothing && error(
            "@brm: `$shown`: the rows of `mi(x)::shares(total, rows)` are a binding " *
            "`rows = groups(g => labels)` or that call with data names `g` and `labels`")
        call
    end
    Expr(:call, :~, Expr(:call, :mi, Expr(:parameters,
            Expr(:kw, :_brm_shares_group, binding.group),
            Expr(:kw, :_brm_shares_cells, binding.cells),
            Expr(:kw, :_brm_shares_total, total)), completion.args[2]),
        x.args[3])
end

# The completion statement `mi(x; _brm_shares_… = …)`: its inner column and the
# data names its keywords read, or `nothing` for any other left-hand side.
function _brm_mi_totals_lhs(lhs)
    isxcall(lhs, :mi) && length(lhs.args) == 3 && Meta.isexpr(lhs.args[2], :parameters) ||
        return nothing
    keywords = Dict(kw.args[1] => kw.args[2] for kw in lhs.args[2].args)
    Set(keys(keywords)) == Set(_BRM_MI_TOTAL_KEYS) || return nothing
    (; inner=lhs.args[3], reads=[keywords[k] for k in _BRM_MI_TOTAL_KEYS])
end

# The three data columns of a completion's group totals, or `nothing`.
function _brm_mi_totals_spec(lhs::ExprColumn; prefix)
    kwargs = getkwargs(lhs)
    present = filter(in(keys(kwargs)), _BRM_MI_TOTAL_KEYS)
    isempty(present) && return nothing
    length(present) == length(_BRM_MI_TOTAL_KEYS) || error(
        "$prefix: internal: a completion with group totals carries $present only")
    columns = map(k -> _brm_mi_totals_column(kwargs[k], lhs; prefix), _BRM_MI_TOTAL_KEYS)
    (; group=columns[1], cells=columns[2], total=columns[3])
end
function _brm_mi_totals_column(column, lhs; prefix)
    column isa NamedColumn && parent(column) isa DataColumn || error(
        "$prefix: `mi($(name(only(getargs(lhs)))))::shares(total, rows)` reads data " *
        "columns for the totals and the grouping; got $(column isa NamedColumn ? "`$(name(column))`, which is no data column" : typeof(column))")
    (; name=name(column), values=parent(parent(column)))
end

"""
    _BRMMissingTotals

The group totals of a completion `mi(x)::shares(total, rows)`: the three data
columns (`group`, `cells`, `total`), the number of free missing entries (first
in the plan's `missing_indices`), the inclusive ends of each constrained group
among the entries after them, and each group's residual.
"""
struct _BRMMissingTotals
    group::Symbol
    cells::Symbol
    total::Symbol
    nfree::Int
    ends::Vector{Int}
    residuals::Vector{Float64}
end

# The plan's missing indices, free first and then group by group, and the
# totals. Every group's residual must be positive where entries are missing;
# a group without one must already sum to its total.
function _brm_missing_totals(source, raw, missing_indices, spec; prefix)
    group, cells, total = spec.group, spec.cells, spec.total
    shown = "`mi($source)::shares($(total.name), groups($(group.name) => $(cells.name)))`"
    length(group.values) == length(raw) || error(
        "$prefix: $shown: `$(group.name)` has $(length(group.values)) entries for the " *
        "$(length(raw)) of `$source`")
    length(total.values) == length(cells.values) || error(
        "$prefix: $shown: `$(total.name)` has $(length(total.values)) totals for the " *
        "$(length(cells.values)) groups of `$(cells.name)`")
    all(t -> t isa Real && isfinite(t), total.values) || error(
        "$prefix: $shown: `$(total.name)` must hold finite numbers")
    rows = groups(group.values => cells.values)
    missing = falses(length(raw))
    missing[missing_indices] .= true
    grouped, ends, residuals = Int[], Int[], Float64[]
    ingroup = falses(length(raw))
    for (k, r) in enumerate(rows)
        ingroup[r] .= true
        observed = sum((Float64(raw[i]) for i in r if !missing[i]); init=0.0)
        residual = Float64(total.values[k]) - observed
        entries = filter(i -> missing[i], r)
        label = repr(cells.values[k])
        if isempty(entries)
            isapprox(observed, total.values[k]; rtol=1e-8, atol=1e-8) || error(
                "$prefix: $shown: group $label has no missing entry, and its entries sum " *
                "to $observed, not to its total $(total.values[k])")
            continue
        end
        residual > 0 || error(
            "$prefix: $shown: the observed entries of group $label sum to $observed, " *
            "which leaves $residual of its total $(total.values[k]) for its " *
            "$(length(entries)) missing entries; their law is positive")
        append!(grouped, entries)
        push!(ends, length(grouped))
        push!(residuals, residual)
    end
    free = filter(i -> !ingroup[i], missing_indices)
    (vcat(free, grouped), _BRMMissingTotals(group.name, cells.name, total.name,
        length(free), ends, residuals))
end

# Whether a completion's totals constrain any missing entry.
_brm_mi_grouped(::Nothing) = false
_brm_mi_grouped(t::_BRMMissingTotals) = !isempty(t.ends)

# The layout of a completion's missing entries from the four data vectors,
# as RKBRMI's data preparations compute it in the graph
# (`brm_covariate_total_*`, statistical_models.jl): free and grouped row
# indices, all of them in the plan's order, the groups' ends and residuals.
function _brm_totals_layout(values, group, cells, totals)
    spec = (; group=(; name=:group, values=group), cells=(; name=:cells, values=cells),
        total=(; name=:total, values=totals))
    ordered, t = _brm_missing_totals(:x, values, findall(ismissing, values), spec;
        prefix="RK backend")
    (; ordered, free=ordered[1:t.nfree], grouped=ordered[t.nfree+1:end],
        ends=t.ends, residuals=t.residuals)
end

# The values of the grouped entries from their coordinates (`K - 1` per group
# of `K`), and back (brm_start, the RK geometry, tests).
function _brm_grouped_shares_values(y::AbstractVector{<:Real}, ends, residuals)
    out = similar(y, ends[end])
    start, coordinate = 1, 1
    for (k, stop) in enumerate(ends)
        K = stop - start + 1
        out[start:stop] = _brm_shares_values(view(y, coordinate:coordinate+K-2), residuals[k])
        start, coordinate = stop + 1, coordinate + K - 1
    end
    out
end
function _brm_grouped_shares_coordinates(x::AbstractVector{<:Real}, ends)
    start = 1
    coordinates = Float64[]
    for stop in ends
        append!(coordinates, _brm_shares_coordinates(x[start:stop]))
        start = stop + 1
    end
    coordinates
end

# ---- StanBlocks families of the grouped entries ---------------------------------
#
# `y` holds every constrained group's coordinates, group after group; `ends`
# are the groups' inclusive ends among the entries; the law's arguments are
# already gathered onto the entries (or scalars). The coordinates carry no
# density of their own (`flat()`): the groups' residuals are observed, and
# their density given the coordinates is the law at the entries plus each
# group's `sum(log.(x)) - log(residual)`, so the pair has the joint density of
# the entries and their sums. Written as a latent with that density instead,
# the coordinates would read as a prior no likelihood reaches whenever nothing
# reads the completion, and StanBlocks would re-draw them, dropping what the
# residuals say about the law.
StanBlocks.@deffun begin
    brm_grouped_shares_values(y::vector[m], ends::int[G], totals::vector[G])::vector[m + G] = begin
        x = rep_vector(0., m + G)
        start = 1
        coordinate = 0
        for k in 1:G
            K = ends[k] - start + 1
            coordinates = rep_vector(0., K - 1)
            for j in 1:(K - 1)
                coordinates[j] = y[coordinate + j]
            end
            values = brm_shares_values(coordinates, totals[k])
            for j in 1:K
                x[start + j - 1] = values[j]
            end
            start = ends[k] + 1
            coordinate = coordinate + K - 1
        end
        return x
    end

    @lpxf brm_group_residuals_lpdf(r::vector[G], y::vector[m], family, ends::int[G],
                                   rows::int[N], args...)::real = begin
        x = brm_grouped_shares_values(y, ends, r)
        density(family, x, args...) + sum(log(x)) - sum(log(r))
    end
    # Group by group.
    brm_group_residuals_lpdfs(r::vector[G], y::vector[m], family, ends::int[G],
                              rows::int[N], args...)::vector[G] = begin
        x = brm_grouped_shares_values(y, ends, r)
        lp = pointwise(family, x, args...)
        out = rep_vector(0., G)
        start = 1
        for k in 1:G
            for j in start:ends[k]
                out[k] = out[k] + lp[j] + log(x[j])
            end
            out[k] = out[k] - log(r[k])
            start = ends[k] + 1
        end
        return out
    end
    # A new draw of the residuals: the groups' sums of new entries drawn from
    # the law (`rows`, the entries' row indices, sizes the draw).
    brm_group_residuals_rng(vector[G], y::vector[m], family, ends::int[G],
                            rows::int[N], args...)::vector[G] = begin
        draws = predictive(family, vector[N], args...)
        out = rep_vector(0., G)
        start = 1
        for k in 1:G
            for j in start:ends[k]
                out[k] = out[k] + draws[j]
            end
            start = ends[k] + 1
        end
        return out
    end
end

# SBBRMI's completion with group totals: `_sb_mi_response` with the missing
# entries split into free ones (`y_free`, the law as written) and the grouped
# ones' coordinates (`y_shares`), whose law the observed `residuals` carry
# (`brm_group_residuals`). Their values, free first, are the completion's
# missing entries `y_mis` in `Jmis` order.
_sb_mi_shares_response = StanBlocks.@slic begin
    n_free = num_elements(Jfree)
    n_shares = num_elements(Jgrp) - num_elements(residuals)
    y_free :: vector[n_free] ~ dummy()
    y_shares :: vector[n_shares] ~ flat()
    residuals ~ dummy()
    y_obs ~ dummy()
    y_mis = append_row(y_free, brm_grouped_shares_values(y_shares, ends, residuals))
    return mi_merge(y_obs, y_mis, Jobs, Jmis, num_elements(Jobs) + num_elements(Jmis))
end

# The data keys of a completion's group totals.
_sb_mi_totals_keys(source::Symbol) = (; free=Symbol(:Jfree_, source), grouped=Symbol(:Jgrp_, source),
    ends=Symbol(source, :_shares_ends), residuals=Symbol(source, :_shares_residuals))

# Binds the group totals' data and returns the submodel's extra keywords.
function _sb_bind_mi_totals!(data, source, plan)
    totals = plan.totals
    keys = _sb_mi_totals_keys(source)
    data[keys.free] = plan.missing_indices[1:totals.nfree]
    data[keys.grouped] = plan.missing_indices[totals.nfree+1:end]
    data[keys.ends] = totals.ends
    data[keys.residuals] = totals.residuals
    keys
end

# ---- RKBRMI: the law of the grouped entries ------------------------------------
#
# RKBRMI declares the grouped entries as one joint latent of the completion,
# `y_shares[axes(rows, 1)] ~ _BRMGroupedShares{Family}(ends, residuals, args…)`
# (rk_missing_values.jl): its packed coordinates are SBBRMI's `y_shares`, the
# geometry maps them onto the entries (the ReactiveKernels extension), and the
# density is the law at the entries. The family is a type parameter, as RKPPL
# needs for an inferable density. RKPPL keeps a latent's density whether or
# not anything reads it, so the residuals inform the law either way.
struct _BRMGroupedShares{F,A<:Tuple}
    args::A
end
(::Type{_BRMGroupedShares{F}})(ends, residuals, args...) where {F} =
    _BRMGroupedShares{F,typeof(args)}(args)
