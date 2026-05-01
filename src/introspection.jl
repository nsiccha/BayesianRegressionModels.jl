# Introspection helpers on a BRMI -- "given a parsed @brm formula, what
# outcomes / linear predictors / groups / continuous / categorical
# predictors does it expose?". Used by the web-macro auto-PPC detector
# (replaces the ad-hoc walker that opened up ExprColumn / NamedColumn /
# DataColumn by hand). Convenient for any downstream consumer that
# needs the model shape without re-parsing the formula text.

# A NamedColumn's `parent` distinguishes its role:
#   DataColumn{<vec>}  -- bound to an actual observed/data vector
#   ExprColumn{~}      -- a derived linear predictor (`loc ~ 1 + a`)
#   ExprColumn{<link>} -- a link-transformed bind (`log(err) ~ 1`)
#   MissingColumn      -- referenced but not yet bound (transient)

"""
    outcomes(brmi::BRMI) -> Vector{NamedTuple}

Walk every entry in `brmi.operations` and return one entry per
`<response> ~ Family(args...)` likelihood, as a NamedTuple with fields:

- `response::Symbol` — the outcome key (LHS of `~`).
- `family` — the family Type (`Normal`, `Poisson`, `Bernoulli`, ...).
- `link_fn::Function` — the unary wrap applied to the linear-predictor
  arg (e.g. `exp` for `Poisson(exp(eta))`); `identity` if the arg is
  bare.
- `link_lp::Symbol` — the latent-parameter name the loc-arg refers to
  (e.g. `:loc` in `Normal(loc, sigma)`, `:log_rate` in
  `Poisson(exp(log_rate))`).
- `family_args::Vector` — the family's other args (e.g. trials column
  for `Binomial`, sigma for `Normal`).

Heuristic for "which arg is the linear predictor": the first arg that
is NOT a NamedColumn over a `DataColumn` (i.e. not a data column).
That picks `loc` for `Normal(loc, sigma)`, the link-wrap for
`Poisson(exp(eta))` / `Bernoulli(logistic(odds))`, and `args[2]` for
`Binomial(N, logistic(odds))`. Skips formulas whose RHS doesn't fit
the `Family(...)` pattern.
"""
function outcomes(brmi::BRMI)
    out = NamedTuple[]
    for (k, v) in pairs(brmi.operations)
        v isa NamedColumn || continue
        op = parent(v)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, rh = getargs(op, 2)
        # Likelihood: LHS bound to actual observed data.
        lhs isa NamedColumn && parent(lhs) isa DataColumn || continue
        rh isa ExprColumn || continue
        family = getf(rh)
        family isa Type || continue
        args = getargs(rh)
        isempty(args) && continue

        # Locate the linear-predictor arg (first non-data arg).
        loc_idx = findfirst(a -> !(a isa NamedColumn && parent(a) isa DataColumn), args)
        isnothing(loc_idx) && continue
        target = args[loc_idx]

        # Peel optional unary link wrap.
        link_fn, link_lp = if target isa NamedColumn
            (identity, name(target))
        elseif target isa ExprColumn && length(getargs(target)) == 1 &&
               getargs(target)[1] isa NamedColumn
            (getf(target), name(getargs(target)[1]))
        else
            continue
        end

        family_args = Any[args[i] for i in eachindex(args) if i != loc_idx]
        push!(out, (; response=k, family, link_fn, link_lp, family_args))
    end
    out
end

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

"""
    predictors(brmi::BRMI, lhs::Symbol) -> Union{NamedTuple,Nothing}

For a linear-predictor LHS (`:loc`, `:log_rate`, ...), classify the
RHS into:

- `intercept::Bool` — whether a literal `1` is summed in.
- `continuous::Vector{Symbol}` — bare predictors over `Real` data
  columns.
- `categorical::Vector{Symbol}` — bare predictors over `Integer` data
  columns (treatment-coded).
- `re_terms::Vector{NamedTuple}` — random-effects blocks, each with
  `(; group::Symbol, inner::NamedTuple)` where `inner` is recursively
  the same shape (so callers can pull RE-internal predictors).

Returns `nothing` if the LHS itself is link-transformed (e.g.
`log(err) ~ ...`) — that path is intentionally excluded since the
caller usually wants predictors on the natural scale, and the
link-transformed LHS is rarely the head linear predictor.
"""
function predictors(brmi::BRMI, lhs::Symbol)
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return nothing
    op_lhs, rhs_expr = getargs(op, 2)
    op_lhs isa NamedColumn || return nothing
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
the RHS of `<lhs> ~ ...`. Empty when there's no RE term or the LHS
isn't introspectable.
"""
function grouping_factors(brmi::BRMI, lhs::Symbol)
    p = predictors(brmi, lhs)
    isnothing(p) ? Symbol[] : Symbol[t.group for t in p.re_terms]
end

"""
    column_data(brmi::BRMI, name::Symbol)

Return the underlying data vector for a `name` whose `brmi.operations`
entry is a `NamedColumn` over a `DataColumn`. Returns `nothing`
otherwise.
"""
function column_data(brmi::BRMI, n::Symbol)
    haskey(brmi.operations, n) || return nothing
    v = brmi.operations[n]
    v isa NamedColumn || return nothing
    inner = parent(v)
    inner isa DataColumn ? parent(inner) : nothing
end

"""
    hierarchical_outcomes(brmi::BRMI)

`outcomes(brmi)` filtered to the entries whose linear predictor has at
least one RE term. Convenient for "does this formula need group-level
faceting / pooled draws?" decisions.
"""
function hierarchical_outcomes(brmi::BRMI)
    [o for o in outcomes(brmi) if begin
        p = predictors(brmi, o.link_lp)
        !isnothing(p) && !isempty(p.re_terms)
    end]
end
