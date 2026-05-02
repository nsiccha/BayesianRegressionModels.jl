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

# Each family arg gets classified into one of these tagged NamedTuples by
# `_classify_arg`. Used by `outcomes` to expose every family arg, not
# just the first linear predictor (so distributional likelihoods like
# `Normal(loc, err)` where both `loc` and `err` are LPs surface both).

# Number / literal / constant.
_classify_arg(a::Number) = (; role=:constant, value=a)

# NamedColumn -> dispatch on its parent's role.
_classify_arg(a::NamedColumn) = _classify_named(a, parent(a))
_classify_named(a, ::DataColumn)  = (; role=:data, name=name(a))
_classify_named(a, op::ExprColumn) = getf(op) === (~) ?
    (; role=:linear_predictor, link_fn=identity, link_lp=name(a)) :
    (; role=:expression, expr=a)
_classify_named(a, _) = (; role=:expression, expr=a)

# ExprColumn -> peel a unary link wrap if it's of the form `link(NC{lp,~})`.
_classify_arg(a::ExprColumn) = _classify_expr(a, getargs(a))
_classify_expr(a, args) =
    if length(args) == 1 && args[1] isa NamedColumn &&
       parent(args[1]) isa ExprColumn && getf(parent(args[1])) === (~)
        (; role=:linear_predictor, link_fn=getf(a), link_lp=name(args[1]))
    else
        (; role=:expression, expr=a)
    end

# Catch-all.
_classify_arg(a) = (; role=:expression, expr=a)

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
        v isa NamedColumn || continue
        op = parent(v)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, rh = getargs(op, 2)
        lhs isa NamedColumn && parent(lhs) isa DataColumn || continue
        rh isa ExprColumn || continue
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
    v = brmi.operations[n]
    v isa NamedColumn || return nothing
    op = parent(v)
    op isa ExprColumn || return nothing
    getf(op) === (~) || return nothing
    op
end

# Helper: peel the optional unary link wrap on a LP's LHS. `loc ~ ...`
# returns (identity, :loc); `log(err) ~ ...` returns (log, :err).
_peel_lp_lhs(lhs::NamedColumn) = (identity, name(lhs))
_peel_lp_lhs(lhs::ExprColumn) = begin
    args = getargs(lhs)
    if length(args) == 1 && args[1] isa NamedColumn
        return (getf(lhs), name(args[1]))
    end
    return nothing
end
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
        v isa NamedColumn || continue
        op = parent(v)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, _ = getargs(op, 2)
        # Skip data-bound LHS (those are likelihoods).
        lhs isa NamedColumn && parent(lhs) isa DataColumn && continue
        peeled = _peel_lp_lhs(lhs)
        isnothing(peeled) && continue
        link_lhs_fn, lp_name = peeled
        push!(out, (; name=lp_name, link_lhs_fn))
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
    rhs_expr isa ExprColumn || return nothing
    getf(rhs_expr) === (+) || return nothing
    _walk_predictors(getargs(rhs_expr))
end

function _walk_predictors(summands)
    intercept = false
    continuous = Symbol[]
    categorical = Symbol[]
    re_terms = NamedTuple[]
    for s in summands
        if s isa Number
            intercept = true
        elseif s isa NamedColumn
            pcol = parent(s)
            pcol isa DataColumn || return nothing
            elt = eltype(parent(pcol))
            if elt <: Integer
                push!(categorical, name(s))
            elseif elt <: Real
                push!(continuous, name(s))
            else
                return nothing
            end
        elseif s isa ExprColumn && getf(s) === (+)
            sub = _walk_predictors(getargs(s))
            isnothing(sub) && return nothing
            intercept |= sub.intercept
            append!(continuous, sub.continuous)
            append!(categorical, sub.categorical)
            append!(re_terms, sub.re_terms)
        elseif s isa ExprColumn && getf(s) === (|)
            ra = getargs(s)
            length(ra) >= 2 || return nothing
            ra[end] isa NamedColumn && parent(ra[end]) isa DataColumn || return nothing
            grp = name(ra[end])
            inner_summands = (ra[1] isa ExprColumn && getf(ra[1]) === (+)) ?
                             getargs(ra[1]) : (ra[1],)
            inner = _walk_predictors(inner_summands)
            isnothing(inner) && return nothing
            push!(re_terms, (; group=grp, inner))
        else
            return nothing
        end
    end
    (; intercept, continuous, categorical, re_terms)
end

"""
    grouping_factors(brmi::BRMI, lhs::Symbol) -> Vector{Symbol}

The grouping-factor symbols (last arg of every `(... | g)` term) in
`<lhs> ~ ...`. Empty if no RE term or LP isn't introspectable.
"""
function grouping_factors(brmi::BRMI, lhs::Symbol)
    p = predictors(brmi, lhs)
    isnothing(p) ? Symbol[] : Symbol[t.group for t in p.re_terms]
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
        v = brmi.operations[n]
        v isa NamedColumn || return
        outer = parent(v)
        outer isa ExprColumn && getf(outer) === (~) || return
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
    v = brmi.operations[n]
    v isa NamedColumn || return nothing
    inner = parent(v)
    inner isa DataColumn ? parent(inner) : nothing
end

"""
    data_columns(brmi::BRMI) -> Vector{Symbol}

Every name in `brmi.operations` that is a NamedColumn over a DataColumn
(i.e. every actual data column referenced by the formula).
"""
function data_columns(brmi::BRMI)
    out = Symbol[]
    for (k, v) in pairs(brmi.operations)
        v isa NamedColumn && parent(v) isa DataColumn && push!(out, k)
    end
    out
end

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
