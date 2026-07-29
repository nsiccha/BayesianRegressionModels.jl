# Introspection helpers on a BRMI -- "given a parsed @brm formula, what
# outcomes / linear predictors / groups / continuous / categorical
# predictors does it expose?". Used by the web-macro auto-PPC detector
# (replaces the ad-hoc walker that opened up ExprColumn / NamedColumn /
# DataColumn by hand). Convenient for any downstream consumer that
# needs the model shape without re-parsing the formula text.
#
# Design idiom: every kind-specific question is dispatched on the column
# type (NamedColumn / ExprColumn / Number / ...), not on a Symbol-tag
# switch. That way new column shapes plug in by adding methods, not by
# editing branches.

# ---- arg-role classification ------------------------------------------------

# Tagged NamedTuples returned by `_classify_arg`: one of
#   (; role=:data, name)
#   (; role=:linear_predictor, link_fn, link_lp)
#   (; role=:constant, value)
#   (; role=:expression, expr)
# `outcomes` uses these to expose every family arg, so distributional
# likelihoods like `Normal(loc, err)` surface both LPs.

_classify_arg(a::Number) = (; role=:constant, value=a)

_classify_arg(a::NamedColumn) = _classify_named(a, parent(a))
_classify_named(a, ::DataColumn)  = (; role=:data, name=name(a))
# `~` (formula) and `assign` (literal `=`) are both LP-binding ops as far
# as outcome introspection cares: each names a latent the response depends
# on. Stan-emit treats them very differently (popefs vs literal binding),
# but the PPC kind detector + plot-builder only needs "what's the LP that
# feeds this outcome arg?".
_classify_named(a, op::ExprColumn) =
    (getf(op) === (~) || getf(op) === assign) ?
        (; role=:linear_predictor, link_fn=identity, link_lp=name(a)) :
        (; role=:expression, expr=a)
_classify_named(a, _) = (; role=:expression, expr=a)

# A unary wrapper around exactly one LP -- bare like `exp(loc)` or with
# inline offset terms like `exp(loc + log(exposure))` -- classifies as
# `:linear_predictor` with the outer function as `link_fn`. Anything
# more entangled (multiple LPs, LP buried inside a non-`+` op) stays
# opaque. Walks the subtree and counts LP NamedColumns; >1 or 0 -> not
# a clean wrapped LP.
_classify_arg(a::ExprColumn) = let lps = _collect_lps(a)
    length(getargs(a)) == 1 && length(lps) == 1 ?
        (; role=:linear_predictor, link_fn=getf(a), link_lp=lps[1]) :
        (; role=:expression, expr=a)
end

_collect_lps(::Any) = Symbol[]
_collect_lps(x::NamedColumn) = _collect_lps_named(x, parent(x))
_collect_lps_named(x, p::ExprColumn) =
    (getf(p) === (~) || getf(p) === assign) ? Symbol[name(x)] : Symbol[]
_collect_lps_named(_, _) = Symbol[]
_collect_lps(x::ExprColumn) = let out = Symbol[]
    for a in getargs(x); append!(out, _collect_lps(a)); end
    out
end

_classify_arg(a) = (; role=:expression, expr=a)

# Two-level dispatch helpers used by every introspection walker. Each returns
# either the narrowed value (the NamedColumn / the inner ExprColumn) or
# `nothing`, so callers compose with `isnothing(...)` rather than carrying
# Bool predicates and `&&`/`||` control flow internally.

_named_op(v::NamedColumn) = _named_op_inner(parent(v))
_named_op(_) = nothing
_named_op_inner(p::ExprColumn) = p
_named_op_inner(_) = nothing

# Narrow to ExprColumn (or nothing); split a sum into its summands (or wrap as
# a 1-tuple so callers don't have to special-case non-`+` RHS shapes).
_as_expr_column(x::ExprColumn) = x
_as_expr_column(_) = nothing
_sum_summands(x::ExprColumn{typeof(+)}) = getargs(x)
_sum_summands(x) = (x,)

# A NamedColumn whose backing storage is a DataColumn → the column itself,
# else nothing. Two methods peel the wrappers; the previous Bool predicate
# `_is_data_named`/`_is_data_backing` collapsed into this single Maybe form.
_data_backed_or_nothing(x::NamedColumn) = _data_backed_inner(x, parent(x))
_data_backed_or_nothing(_) = nothing
_data_backed_inner(x, ::DataColumn) = x
_data_backed_inner(_x, _) = nothing

# ---- outcomes ---------------------------------------------------------------

"""
    outcomes(brmi::BRMI) -> Vector{NamedTuple}

Return one entry per `<response> ~ Family(args...)` likelihood (i.e. every
`~` op whose LHS is a NamedColumn over a `DataColumn`):

    (; response::Symbol, family, args::Vector{NamedTuple})

`args` is the family-arg list, classified per arg into one of:
- `(; role=:data, name)` -- e.g. `Binomial`'s trials column.
- `(; role=:linear_predictor, link_fn, link_lp)` -- a latent LP referenced
  bare (link_fn=identity) or via a unary wrap like `exp(eta)` /
  `logistic(odds)`. Distributional models with multiple LPs (e.g.
  `Normal(loc, err)` where both `loc ~ ...` and `log(err) ~ ...`)
  produce one entry per LP.
- `(; role=:constant, value)` -- numeric literal.
- `(; role=:expression, expr)` -- catch-all opaque ExprColumn.
"""
function outcomes(brmi::BRMI)
    out = NamedTuple[]
    for (k, v) in pairs(brmi.operations)
        op = _named_op(v)
        op === nothing && continue
        getf(op) === (~) || continue
        lhs, rh = getargs(op, 2)
        isnothing(_data_backed_or_nothing(lhs)) && continue
        isnothing(_as_expr_column(rh)) && continue
        family = getf(rh)
        args = [_classify_arg(a) for a in getargs(rh)]
        push!(out, (; response=k, family, args))
    end
    out
end

# Convenience accessors so callers don't recompute filters by hand.
"""All `:linear_predictor` args from an outcome (>1 means distributional)."""
linear_predictor_args(o) = [a for a in o.args if a.role === :linear_predictor]
"""All `:data` args from an outcome (e.g. Binomial trials column)."""
data_args(o) = [a for a in o.args if a.role === :data]
"""The first `:linear_predictor` arg from an outcome, or `nothing`."""
primary_lp(o) = begin
    for a in o.args
        a.role === :linear_predictor && return a
    end
    nothing
end

# ---- linear predictors (intermediate `~` ops, non-data LHS) ---------------

"""
    linear_predictor_op(brmi::BRMI, name::Symbol) -> Union{ExprColumn,Nothing}

The `~` ExprColumn that defines `<name> ~ <rhs>`, or `nothing` if
`name` isn't a `~`-bound entry in `brmi.operations`.
"""
function linear_predictor_op(brmi::BRMI, n::Symbol)
    haskey(brmi.operations, n) || return nothing
    op = _named_op(brmi.operations[n])
    op === nothing && return nothing
    # Either `<n> ~ <rhs>` (formula) or `<n> = <rhs>` (literal assign).
    # See `_classify_named` -- both bind a latent that downstream
    # outcomes can reference as a linear-predictor arg.
    f = getf(op)
    (f === (~) || f === assign) || return nothing
    op
end

# Helper: peel the optional unary link wrap on a LP's LHS. `loc ~ ...`
# returns (identity, :loc); `log(err) ~ ...` returns (log, :err).
_peel_lp_lhs(lhs::NamedColumn) = (identity, name(lhs))
_peel_lp_lhs(lhs::ExprColumn) = begin
    args = getargs(lhs)
    length(args) == 1 || return nothing
    inner = _as_named_column_intro(args[1])
    isnothing(inner) ? nothing : (getf(lhs), name(inner))
end
_as_named_column_intro(x::NamedColumn) = x
_as_named_column_intro(_) = nothing
_peel_lp_lhs(_) = nothing

"""
    linear_predictors(brmi::BRMI) -> Vector{NamedTuple}

One entry per `~` op whose LHS is NOT a data column -- the intermediate
linear-predictor binds (`loc ~ 1 + a`, `log(err) ~ 1 + b`, ...). Each
entry is `(; name::Symbol, link_lhs_fn::Function)` where `name` is the
underlying parameter name (e.g. `:err` for `log(err) ~ ...`) and
`link_lhs_fn` is the link applied to the LHS (e.g. `log`; `identity`
when bare).
"""
function linear_predictors(brmi::BRMI)
    out = NamedTuple[]
    for (k, v) in pairs(brmi.operations)
        op = _named_op(v)
        op === nothing && continue
        getf(op) === (~) || continue
        lhs, _ = getargs(op, 2)
        # Skip data-bound LHS (those are likelihoods).
        isnothing(_data_backed_or_nothing(lhs)) || continue
        peeled = _peel_lp_lhs(lhs)
        isnothing(peeled) && continue
        link_lhs_fn, lp_name = peeled
        push!(out, (; name=lp_name, link_lhs_fn))
    end
    out
end

# ---- population-effect prior statements -----------------------------------

"""
    effect_priors(brmi::BRMI) -> Vector{NamedTuple}

Return the population-coefficient prior statements captured by [`@brm`](@ref),
in formula order. Each entry has

```julia
(; predictor, coefficient, family, arguments, keywords, expression)
```

`predictor` is a `Symbol` for the explicit
`effect(linear_predictor, coefficient)` form and `nothing` for the concise
`effect(coefficient)` form. `coefficient` uses the same labels as
[`popcoefnames`](@ref), including `:Intercept`. `expression` is the exact
parsed RHS [`ExprColumn`](@ref); `family`, `arguments`, and `keywords` are its
decomposed, directly inspectable parts.

The SBBRMI backend resolves a concise address only when the label belongs to
exactly one linear predictor, and rejects ambiguous or unknown labels.
"""
function effect_priors(brmi::BRMI)
    out = NamedTuple[]
    for op_nc in values(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        lhs_e = _as_expr_column(lhs)
        isnothing(lhs_e) && continue
        getf(lhs_e) === effect || continue
        address = getargs(lhs_e)
        length(address) in (1, 2) || error(
            "effect_priors: malformed effect address with $(length(address)) arguments")
        rhs_e = _as_expr_column(rhs)
        isnothing(rhs_e) && error(
            "effect_priors: `effect(...)` RHS is not a distribution expression")
        predictor = length(address) == 2 ? address[1] : nothing
        coefficient = address[end]
        push!(out, (; predictor, coefficient, family=getf(rhs_e),
                     arguments=getargs(rhs_e), keywords=getkwargs(rhs_e),
                     expression=rhs_e))
    end
    out
end

# ---- predictors of a single LP --------------------------------------------

"""
    predictors(brmi::BRMI, lhs::Symbol) -> Union{NamedTuple,Nothing}

For a linear-predictor LHS, classify the RHS into:

- `intercept::Bool`
- `continuous::Vector{Symbol}` -- bare predictors over `Real` data columns.
- `categorical::Vector{Symbol}` -- bare predictors over `Integer` data columns.
- `re_terms::Vector{(; group, inner)}` -- random-effects blocks, with
  `inner` recursively the same shape so callers can reach RE-internal
  predictors.

Works for both bare-LHS LPs (`loc ~ 1 + a`) and link-transformed-LHS
LPs (`log(err) ~ 1 + b`); the link is captured separately via
`linear_predictors`. `nothing` only when the RHS is malformed.
"""
function predictors(brmi::BRMI, lhs::Symbol)
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return nothing
    _, rhs_expr = getargs(op, 2)
    # Normalise the RHS to a list of summands: `loc ~ 1 + a` ->
    # `getargs(+)`, `loc ~ 1` (intercept-only) or `loc ~ a` (single
    # predictor) -> singleton wrap. _walk_predictors handles each
    # summand kind (Number / NamedColumn / RE) the same way.
    _walk_predictors(_sum_summands(rhs_expr))
end

# Find the first data-column NamedColumn reachable in `x`'s subtree
# (returning the NamedColumn itself), or `nothing` if no data leaf is
# reachable. Used by the predictors walker to:
#   - decide whether a wrapper expression (`mo(c)`, `protect(a^2)`,
#     `log(exposure)`, ...) is a valid predictor (it must touch data),
#   - label the resulting predictor column by the inner data name
#     instead of the outermost function name,
#   - and pick the right (continuous vs categorical) bucket from the
#     leaf's eltype (so e.g. `factor(c1, ref=2)` lands in categorical
#     instead of continuous).
_walk_pred_leaf(x::NamedColumn) = _data_backed_or_nothing(x)
function _walk_pred_leaf(x::ExprColumn)
    for a in getargs(x)
        leaf = _walk_pred_leaf(a)
        leaf === nothing || return leaf
    end
    nothing
end
_walk_pred_leaf(_) = nothing
_walk_pred_label(x) = let leaf = _walk_pred_leaf(x); leaf === nothing ? nothing : name(leaf) end

function _walk_predictors(summands)
    intercept = Ref(false)
    continuous = Symbol[]
    categorical = Symbol[]
    re_terms = NamedTuple[]
    for s in summands
        _walk_pred_summand!(intercept, continuous, categorical, re_terms, s) || return nothing
    end
    (; intercept=intercept[], continuous, categorical, re_terms)
end

_walk_pred_summand!(intercept, _, _, _, ::Number) = (intercept[] = true; true)
_walk_pred_summand!(_, continuous, categorical, _, s::NamedColumn) =
    _bucket_named_pred!(continuous, categorical, s, parent(s))
_bucket_named_pred!(continuous, categorical, s, pcol::DataColumn) = let elt = eltype(parent(pcol))
    elt <: Integer ? (push!(categorical, name(s)); true) :
    elt <: Real    ? (push!(continuous,  name(s)); true) :
    false
end
_bucket_named_pred!(args...) = false
_walk_pred_summand!(intercept, continuous, categorical, re_terms, s::ExprColumn) =
    _walk_pred_expr!(intercept, continuous, categorical, re_terms, getf(s), s)
_walk_pred_summand!(_, _, _, _, _) = false

_walk_pred_expr!(intercept, continuous, categorical, re_terms, ::typeof(+), s) = begin
    sub = _walk_predictors(getargs(s))
    isnothing(sub) && return false
    intercept[] |= sub.intercept
    append!(continuous, sub.continuous)
    append!(categorical, sub.categorical)
    append!(re_terms, sub.re_terms)
    true
end
# Zero-correlation ranef `(... || g)` -- treat structurally like `(... | g)`
# for PPC enumeration: the group + inner predictors matter, the variance-
# correlation structure doesn't show up in PPC plots.
_walk_pred_expr!(_, _, _, re_terms, ::typeof(doublepipe), s) =
    _walk_pred_grouped!(re_terms, s)
_walk_pred_expr!(_, _, _, re_terms, ::typeof(|), s) =
    _walk_pred_grouped!(re_terms, s)
# Skip without bailing so the marginal `a` / `b` (when present in the same
# formula) still drive PPC kind detection. The interaction itself is a
# derived column the user wouldn't recognise as a separate predictor anyway.
_walk_pred_expr!(_, _, _, _, ::typeof(&), _) = true
# Generic wrapped/derived predictor: `mo(c)`, `s(a)`, `log(exposure)`,
# `factor(c1, ref=2)`, `a^2`, ... -- any subtree that touches a data leaf
# becomes a predictor. Bucket follows the leaf's eltype: integer ->
# categorical, real -> continuous. Bail when no data leaf is reachable
# (opaque expression with only sampled-parameter references can't drive
# a PPC axis).
_walk_pred_expr!(_, continuous, categorical, _, _, s) = let leaf = _walk_pred_leaf(s)
    leaf === nothing && return false
    elt = eltype(parent(parent(leaf)))
    push!(elt <: Integer ? categorical : continuous, name(leaf))
    true
end

_walk_pred_grouped!(re_terms, s) = begin
    ra = getargs(s)
    length(ra) >= 2 || return false
    last_arg = ra[end]
    isnothing(_data_backed_or_nothing(last_arg)) && return false
    grp = name(last_arg)
    inner = _walk_predictors(_sum_summands(ra[1]))
    isnothing(inner) && return false
    push!(re_terms, (; group=grp, inner))
    true
end

# `hsgp(x, by=g)` is a per-group random effect, so its `by=` grouping column is a
# genuine grouping factor (web-macro auto-PPC facets on these), even though it
# rides on a predictor summand rather than a `(... | g)` term. Surfaced via a
# dispatched helper (no `getf(...) === :sym` branch inside the walker).
_gp_by_group(s::ExprColumn) = _gp_by_group_f(getf(s), s)
_gp_by_group(_s) = nothing
_gp_by_group_f(::typeof(hsgp), s) = _gp_by_group_kw(get(getkwargs(s), :by, nothing))
_gp_by_group_f(::Any, _s) = nothing
_gp_by_group_kw(by::NamedColumn) = name(by)
_gp_by_group_kw(_by) = nothing

"""
    grouping_factors(brmi::BRMI, lhs::Symbol) -> Vector{Symbol}

The grouping-factor symbols in `<lhs> ~ ...`: the last arg of every
`(... | g)` term plus the `by=` group of every `hsgp(x, by=g)` term. Empty if
no grouping term or the LP isn't introspectable.
"""
function grouping_factors(brmi::BRMI, lhs::Symbol)
    p = predictors(brmi, lhs)
    re_groups = isnothing(p) ? Symbol[] : Symbol[t.group for t in p.re_terms]
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return unique!(re_groups)
    _, rhs_expr = getargs(op, 2)
    gp_groups = Symbol[g for g in (_gp_by_group(s) for s in _sum_summands(rhs_expr)) if !isnothing(g)]
    unique!(vcat(re_groups, gp_groups))
end

# ---- DAG traversal --------------------------------------------------------

"""
    dependencies(brmi::BRMI, name::Symbol) -> NamedTuple

Recursively walk the `~` body of `name`, collecting every data column
+ intermediate LP it transitively references:

    (; data::Vector{Symbol}, intermediates::Vector{Symbol})

For `y1 ~ Normal(loc, err)` with `loc ~ 1 + a + (1|g1)` and
`log(err) ~ 1 + b`, `dependencies(brmi, :y1)` returns
`(; data=[:a, :g1, :b], intermediates=[:loc, :err])`. RE grouping
factors count as data deps.
"""
function dependencies(brmi::BRMI, n::Symbol)
    data = OrderedSet{Symbol}()
    intermediates = OrderedSet{Symbol}()
    seen = Set{Symbol}()
    _trace!(brmi, n, data, intermediates, seen)
    (; data=collect(data), intermediates=collect(intermediates))
end

function _trace!(brmi, n::Symbol, data, intermediates, seen)
    n in seen && return
    push!(seen, n)
    op = linear_predictor_op(brmi, n)
    if isnothing(op)
        # Maybe a likelihood node -- look for `~` op even if LHS is data.
        haskey(brmi.operations, n) || return
        outer = _named_op(brmi.operations[n])
        outer === nothing && return
        getf(outer) === (~) || return
        op = outer
    end
    _, rhs = getargs(op, 2)
    _trace_expr!(brmi, rhs, data, intermediates, seen)
end

# Dispatch on argument type for traversal -- no Symbol switch.
_trace_expr!(_, ::Number, _, _, _) = nothing
function _trace_expr!(brmi, expr::NamedColumn, data, intermediates, seen)
    inner = parent(expr)
    _trace_named!(brmi, name(expr), inner, data, intermediates, seen)
end
_trace_named!(_, n::Symbol, ::DataColumn, data, _, _) = (push!(data, n); nothing)
_trace_named!(brmi, n::Symbol, op::ExprColumn, data, intermediates, seen) = begin
    if getf(op) === (~)
        push!(intermediates, n)
        _trace!(brmi, n, data, intermediates, seen)
    else
        for a in getargs(op)
            _trace_expr!(brmi, a, data, intermediates, seen)
        end
    end
end
_trace_named!(_, _, _, _, _, _) = nothing
function _trace_expr!(brmi, expr::ExprColumn, data, intermediates, seen)
    for a in getargs(expr)
        _trace_expr!(brmi, a, data, intermediates, seen)
    end
end
_trace_expr!(_, _, _, _, _) = nothing

# ---- column-level helpers -------------------------------------------------

"""
    column_data(brmi::BRMI, name::Symbol)

Underlying data vector for a NamedColumn-over-DataColumn entry.
`nothing` if `name` doesn't resolve to a DataColumn.
"""
function column_data(brmi::BRMI, n::Symbol)
    haskey(brmi.operations, n) || return nothing
    _column_data_value(brmi.operations[n])
end
_column_data_value(v::NamedColumn) = _column_data_inner(parent(v))
_column_data_value(_) = nothing
_column_data_inner(d::DataColumn) = parent(d)
_column_data_inner(_) = nothing

"""
    data_columns(brmi::BRMI) -> Vector{Symbol}

Every name in `brmi.operations` that is a NamedColumn over a DataColumn
(i.e. every actual data column referenced by the formula).
"""
function data_columns(brmi::BRMI)
    out = Symbol[]
    for (k, v) in pairs(brmi.operations)
        _push_if_data_name!(out, k, v)
    end
    out
end
_push_if_data_name!(out, k, v::NamedColumn) = _push_if_data_inner!(out, k, parent(v))
_push_if_data_name!(_out, _k, _v) = nothing
_push_if_data_inner!(out, k, ::DataColumn) = (push!(out, k); nothing)
_push_if_data_inner!(_out, _k, _p) = nothing

"""
    hierarchical_outcomes(brmi::BRMI)

`outcomes(brmi)` filtered to those whose linear predictor has any RE
term.
"""
function hierarchical_outcomes(brmi::BRMI)
    [o for o in outcomes(brmi) if _has_re(brmi, o)]
end
function _has_re(brmi::BRMI, o)
    lp = primary_lp(o)
    isnothing(lp) && return false
    p = predictors(brmi, lp.link_lp)
    !isnothing(p) && !isempty(p.re_terms)
end
