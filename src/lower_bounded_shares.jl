# Lower-bounded completion laws retain their normalization independently of
# the shares geometry. The first numeric argument is the effective lower
# support; the remaining arguments belong to the authored base family.
struct _BRMLowerLaw{F,N,A<:Tuple}
    args::A
end
(::Type{_BRMLowerLaw{F,N}})(args...) where {F,N} =
    _BRMLowerLaw{F,N,typeof(args)}(args)
Base.minimum(d::_BRMLowerLaw) = first(d.args)
Base.maximum(::_BRMLowerLaw) = Inf
function Distributions.logpdf(d::_BRMLowerLaw{F,N}, x::Real) where {F,N}
    lo, args = first(d.args), Base.tail(d.args)
    x >= lo && isfinite(lo) || return -Inf
    base = F(args...)
    # At the base support's lower endpoint the surviving probability is
    # exactly one. Keep that constant explicit, including its zero derivative.
    normalizer = N && lo > minimum(base) ? logccdf(base, lo) : zero(x)
    logpdf(base, x) - normalizer
end

_brm_lower_family(::Any) = false
_brm_lower_family(::Type{<:_BRMLowerLaw}) = true
_brm_lower_normalized(::Type{<:_BRMLowerLaw{F,N}}) where {F,N} = N
_brm_lower_base(::Type{<:_BRMLowerLaw{F}}) where {F} = F

# As with grouped shares, put a standard normal on the unconstrained
# coordinate and observe the exact density remainder. This changes only the
# prior/likelihood split: x = lo + exp(z), with log-Jacobian z.
struct _BRMLowerEntryWeight{F} end
function (::_BRMLowerEntryWeight{F})(z, x, args...) where {F}
    logpdf(F(args...), x) + z + (z * z + log(2pi)) / 2
end

# Any other continuous univariate completion law, non-centered: the law's own
# support transform carries a standard normal draw to its value, `x = T(z)`, by
# the standard unconstraining maps read from the law's support at the point
# (the identity on the real line, `lo + exp(z)` above a lower bound,
# `hi - exp(z)` below an upper one, `lo + (hi - lo) * logistic(z)` on an
# interval). Each entry observes the exact density remainder
# `logpdf(law, x) + log|T'(z)| - log N(z; 0, 1)`. These are the coordinates the
# centred lowering samples (and SBBRMI's unconstrained entries), so the density
# is the centred one; only the prior/likelihood split differs, which puts a law
# that reads the units' draws into their units' conditionals.
#
# `Vararg{Any,N}` specializes on the arguments' count: a bare `args...`
# leaves the law's constructor dispatched at run time.
struct _BRMSupportEntryValue{F} end
(::_BRMSupportEntryValue{F})(z, args::Vararg{Any,N}) where {F,N} =
    first(_brm_support_transform(F(args...), z))
struct _BRMSupportEntryWeight{F} end
function (::_BRMSupportEntryWeight{F})(z, x, args::Vararg{Any,N}) where {F,N}
    d = F(args...)
    logpdf(d, x) + last(_brm_support_transform(d, z)) + (z * z + log(2pi)) / 2
end
# `(T(z), log|T'(z)|)` for the support of `d`.
function _brm_support_transform(d, z)
    lo, hi = minimum(d), maximum(d)
    if isfinite(lo) && isfinite(hi)
        width = hi - lo
        (lo + width * LogExpFunctions.logistic(z),
            log(width) - LogExpFunctions.log1pexp(-z) - LogExpFunctions.log1pexp(z))
    elseif isfinite(lo)
        (lo + exp(z), z)
    elseif isfinite(hi)
        (hi - exp(z), z)
    else
        (z, zero(z))
    end
end

function _brm_lower_parts(f, args, kwargs)
    f in (truncated, restricted) || return nothing
    length(args) in (1, 3) || error("@brm: a bounded completion needs a base law and lower/upper bounds")
    all(k -> k in (:lower, :upper), keys(kwargs)) || error(
        "@brm: a bounded completion takes only lower/upper keywords")
    length(args) == 3 && !isempty(kwargs) && error(
        "@brm: a bounded completion cannot mix positional and keyword bounds")
    lo, hi = length(args) == 3 ? args[2:3] :
        (get(kwargs, :lower, nothing), get(kwargs, :upper, nothing))
    (hi === nothing || hi === Inf) || error(
        "@brm: shares with a finite upper bound need a box-constrained simplex; " *
        "the lower-bounded shares geometry requires an open upper end")
    lo === nothing && (lo = -Inf)
    (; base=first(args), lower=lo, normalized=f === truncated)
end
_brm_lower_parts(d::ExprColumn) = _brm_lower_parts(getf(d), getargs(d), getkwargs(d))
_brm_lower_parts(d::_BRMPreparedExpr) = _brm_lower_parts(d.callable, d.args, d.kwargs)

function _brm_cell_lower_parts(law, scope)
    f = _sb_cell_binding(scope, first(law.args))
    f in (truncated, restricted) || return nothing
    args, kwargs = Any[], Pair{Symbol,Any}[]
    for arg in law.args[2:end]
        if Meta.isexpr(arg, :parameters)
            append!(kwargs, (a.args[1] => a.args[2] for a in arg.args))
        elseif Meta.isexpr(arg, :kw)
            push!(kwargs, arg.args[1] => arg.args[2])
        else
            push!(args, arg)
        end
    end
    _brm_lower_parts(f, Tuple(args), (; kwargs...))
end

function _brm_lower_floor(F)
    F isa Type && F <: ContinuousUnivariateDistribution || error(
        "@brm: a bounded completion needs a continuous univariate base family")
    # These supports do not depend on the base law's sampled arguments.
    # Inferring a support from unit arguments silently gives the wrong map
    # for, for example, Pareto with a live scale or a shifted distribution.
    F in (Normal, Cauchy, TDist, Logistic, Laplace, Gumbel) && return -Inf
    F in (LogNormal, Gamma, Exponential, Weibull, InverseGamma,
        InverseGaussian, Rayleigh, Chi, Chisq, FDist, BetaPrime) && return 0.
    error("@brm: lower-bounded completions need a registered fixed, open-upper " *
        "base support; `$F` may have finite or parameter-dependent endpoints")
end
_brm_effective_lower(lo::Real, floor::Real) = max(lo, floor)
_brm_effective_lower(lo, floor::Real) = max.(lo, floor)
_brm_lower_at(lo::Real, _) = lo
_brm_lower_at(lo, j) = lo[j]
_brm_lower_slice(lo::Real, _) = lo
_brm_lower_slice(lo, rows) = view(lo, rows)
_brm_lower_sum(lo::Real, n) = n * lo
_brm_lower_sum(lo, _) = sum(lo)

function _rk_lower_completion_law(d::_BRMPreparedExpr)
    spec = _brm_lower_parts(d)
    spec === nothing && return d
    base = spec.base
    base isa _BRMPreparedExpr && isempty(base.kwargs) || error(
        "RK backend: a bounded completion needs a positional base distribution call")
    floor = _brm_lower_floor(base.callable)
    lower = floor == -Inf ? spec.lower : spec.lower isa Real ? max(spec.lower, floor) :
        _BRMPreparedExpr(_brm_effective_lower, (spec.lower, floor), (;))
    _BRMPreparedExpr(_BRMLowerLaw{base.callable,spec.normalized},
        (lower, base.args...), (;))
end

function _brm_validate_lower_completion(plan, lo)
    lo isa Real || length(lo) == length(plan.values) || throw(DimensionMismatch(
        "mi($(plan.source)): lower bounds must have one entry per response row"))
    all(isfinite, lo isa Real ? (lo,) : lo) || throw(ArgumentError(
        "mi($(plan.source)): lower bounds must be finite"))
    for j in plan.observed_indices
        plan.values[j] >= _brm_lower_at(lo, j) || throw(ArgumentError(
            "mi($(plan.source)): observed row $j lies below its lower bound $(_brm_lower_at(lo, j))"))
    end
    _brm_mi_grouped(plan.totals) || return nothing
    t = plan.totals
    rows = plan.missing_indices[t.nfree+1:end]
    start = 1
    for (k, stop) in enumerate(t.ends)
        lower = lo isa Real ? lo : view(lo, rows[start:stop])
        _brm_lower_shares_valid(t.residuals[k], lower, stop-start+1) || throw(ArgumentError(
            "mi($(plan.source))::shares: group $k has residual $(t.residuals[k]) " *
            "and lower-bound sum $(_brm_lower_sum(lower, stop-start+1)); " *
            "multiple missing entries require positive slack, a singleton requires nonnegative slack"))
        start = stop + 1
    end
    nothing
end
function _brm_validate_positive_residuals(plan)
    _brm_mi_grouped(plan.totals) || return nothing
    all(>(0), plan.totals.residuals) || throw(ArgumentError(
        "mi($(plan.source))::shares: the positive law requires positive group residuals"))
    nothing
end

function _rk_validate_lower_completion(plan, distribution, columns)
    _brm_lower_family(distribution.callable) || return nothing
    lower = first(distribution.args)
    refs = _brm_prepared_references(lower)
    all(r -> haskey(columns, r), refs) || return nothing
    values = (; (r => columns[r] for r in refs)...)
    lo = [_brm_eval_prepared_expr(lower, values, j) for j in eachindex(plan.values)]
    _brm_validate_lower_completion(plan, lo)
end

# A singleton is exactly its residual, including an endpoint admitted by the
# law. With two or more entries the slack must be strictly positive: there is
# no full-dimensional continuous simplex at zero slack.
function _brm_lower_shares_valid(total, lo, n)
    slack = total - _brm_lower_sum(lo, n)
    isfinite(slack) && (n == 1 ? slack >= 0 : slack > 0)
end
function _brm_shares_values(y::AbstractVector, total::Real, lo)
    n = length(y) + 1
    n == 1 && return [total + zero(eltype(y))]
    slack = total - _brm_lower_sum(lo, n)
    # Outside the proposal domain provide finite values; the law and Jacobian
    # reject the point before evaluating logarithms at the empty simplex.
    slack > 0 && isfinite(slack) || return [zero(eltype(y)) +
        (isfinite(_brm_lower_at(lo, j)) ? _brm_lower_at(lo, j) : 0.) for j in 1:n]
    _brm_shares_values(y, slack) .+ lo
end
function _brm_lower_shares_logjac(y, total, lo)
    n = length(y) + 1
    _brm_lower_shares_valid(total, lo, n) || return -Inf
    n == 1 && return zero(eltype(y))
    slack = total - _brm_lower_sum(lo, n)
    sum(_brm_shares_log_values(y, slack)) - log(slack)
end
function _brm_shares_coordinates(x, lo)
    if length(x) == 1
        x[1] >= _brm_lower_at(lo, 1) || throw(ArgumentError(
            "a singleton shares value must respect its lower bound"))
        return Float64[]
    end
    all(>(0), x .- lo) || throw(ArgumentError(
        "shares values must be strictly above their row lower bounds"))
    _brm_shares_coordinates(x .- lo)
end
function _brm_grouped_shares_values(y::AbstractVector, ends, residuals, lo)
    out = similar(y, last(ends))
    start, coordinate = 1, 1
    for (k, stop) in enumerate(ends)
        n = stop - start + 1
        out[start:stop] = _brm_shares_values(view(y, coordinate:coordinate+n-2),
            residuals[k], _brm_lower_slice(lo, start:stop))
        start, coordinate = stop + 1, coordinate + n - 1
    end
    out
end
function _brm_grouped_shares_coordinates(x, ends, lo)
    out = Float64[]
    start = 1
    for stop in ends
        append!(out, _brm_shares_coordinates(view(x, start:stop), _brm_lower_slice(lo, start:stop)))
        start = stop + 1
    end
    out
end
function _brm_lower_grouped_logjac(y, ends, residuals, lo)
    result = zero(eltype(y))
    start, coordinate = 1, 1
    for (k, stop) in enumerate(ends)
        n = stop - start + 1
        result += _brm_lower_shares_logjac(view(y, coordinate:coordinate+n-2),
            residuals[k], _brm_lower_slice(lo, start:stop))
        start, coordinate = stop + 1, coordinate + n - 1
    end
    result
end

# Bounds are ordinary law arguments, so replay and live parameter dependencies
# travel through the same graph as the distribution's location and scale.
struct _BRMLowerShares{F,A<:Tuple,L,T}
    args::A
    lower::L
    total::T
end
(::Type{_BRMLowerShares{F}})(n, total, lo, args...) where {F} =
    _BRMLowerShares{F,typeof(args),typeof(lo),typeof(total)}(args, lo, total)
struct _BRMLowerGroupedShares{F,A<:Tuple,L,E,R}
    args::A
    lower::L
    ends::E
    residuals::R
end
(::Type{_BRMLowerGroupedShares{F}})(ends, residuals, lo, args...) where {F} =
    _BRMLowerGroupedShares{F,typeof(args),typeof(lo),typeof(ends),typeof(residuals)}(
        args, lo, ends, residuals)

function (::_BRMGroupShareWeight{F})(z, x, total, lo, args::Vararg{Any,N}) where {F<:_BRMLowerLaw,N}
    _brm_lower_shares_valid(total, lo, length(x)) || return -Inf
    weight = _brm_lower_shares_logjac(z, total, lo)
    for j in eachindex(x)
        weight += logpdf(F(_brm_lower_at(lo, j), _brm_shares_entries(args, j)...), x[j])
    end
    for c in eachindex(z)
        weight += z[c]^2 / 2 + log(2π) / 2
    end
    weight
end

StanBlocks.@deffun begin
    @lpxf brm_lower_lpdf(x::real, family, normalized::int, lo::real, args...)::real = begin
        if is_inf(lo) || is_nan(lo) || x < lo
            return negative_infinity()
        end
        if normalized == 1
            return density(family, x, args...) - logccdf(family, lo, args...)
        end
        density(family, x, args...)
    end
    brm_lower_lpdf(x::anything[n], family, normalized::int, lo, args...)::real =
        sum(brm_lower_lpdfs(x, family, normalized, lo, args...))
    brm_lower_lpdfs(x::real, family, normalized::int, lo, args...)::real =
        brm_lower_lpdf(x, family, normalized, lo, args...)
    brm_lower_lpdfs(x::anything[n], family, normalized::int, lo, args...)::vector[n] =
        jbroadcasted(brm_lower_lpdf, x, family, normalized, lo, args...)
    brm_lower_rng(family, normalized::int, lo::real, args...)::real =
        lower_conditioning_rng(family, lo, args...)
    brm_lower_draw_rng(dummy, family, normalized::int, lo, args...)::real =
        brm_lower_rng(family, normalized, lo, args...)
    brm_lower_rng(vector[n], family, normalized::int, lo, args...)::vector[n] =
        jbroadcasted(brm_lower_draw_rng, rep_vector(0., n), family, normalized, lo, args...)

    brm_lower_vector(lo::real, n::int)::vector[n] = rep_vector(lo, n)
    brm_lower_vector(lo::vector[n], size::int)::vector[n] = begin
        @stan_assert n == size "lower bounds must have one entry per shares value"
        lo
    end
    brm_lower_shares_values(y::vector[m], total::real, lo::vector[n])::vector[n] = begin
        if n == 1
            return rep_vector(total, 1)
        end
        slack = total - sum(lo)
        if is_inf(slack) || is_nan(slack)
            return rep_vector(0., n)
        end
        if slack <= 0
            return lo
        end
        lo + brm_shares_values(y, slack)
    end
    brm_lower_shares_logjac(y::vector[m], total::real, lo::vector[n])::real = begin
        slack = total - sum(lo)
        if is_inf(slack) || is_nan(slack) || slack < 0 || (n > 1 && slack == 0)
            return negative_infinity()
        end
        if n == 1
            return 0.
        end
        sum(brm_shares_log_values(y, slack)) - log(slack)
    end
    brm_lower_grouped_values(y::vector[m], ends::int[G], r::vector[G], lo::vector[N])::vector[N] = begin
        x = rep_vector(0., N)
        start = 1
        coordinate = 1
        for k in 1:G
            K = ends[k] - start + 1
            if K == 1
                x[start] = r[k]
            else
                x[start:ends[k]] = brm_lower_shares_values(segment(y, coordinate, K-1),
                    r[k], segment(lo, start, K))
            end
            start = ends[k] + 1
            coordinate += K - 1
        end
        x
    end
    brm_lower_cell_values(y::vector[m], total::real, lo)::vector[m+1] =
        brm_lower_shares_values(y, total, brm_lower_vector(lo, m+1))
    brm_lower_cell_from_excess(excess::vector[n], total::real, lo::vector[n])::vector[n] = begin
        if n == 1
            return rep_vector(total, 1)
        end
        lo + excess
    end
    @lpxf brm_lower_cell_lpdf(y::vector[m], family, normalized::int, total::real, lo, args...)::real = begin
        lower_values = brm_lower_vector(lo, m+1)
        slack = total - sum(lower_values)
        if is_inf(slack) || is_nan(slack) || slack < 0 || (m > 0 && slack == 0)
            return negative_infinity()
        end
        x = brm_lower_shares_values(y, total, lower_values)
        brm_lower_lpdf(x, family, normalized, lower_values, args...) +
            brm_lower_shares_logjac(y, total, lower_values)
    end
    brm_lower_cell_lpdfs(y::vector[m], family, args...)::vector[m] =
        rep_vector(brm_lower_cell_lpdf(y, family, args...) / max(m, 1), m)
    brm_lower_cell_rng(family, normalized::int, total::real, lo::vector[n], args...)::vector[n-1] = begin
        @stan_assert false "shares: a conditional cell draw is not available"
        rep_vector(0., n-1)
    end
    brm_lower_cell_rng(family, normalized::int, total::real, lo::real, a::vector[n], args...)::vector[n-1] = begin
        @stan_assert false "shares: a conditional cell draw is not available"
        rep_vector(0., n-1)
    end
    brm_lower_cell_rng(family, normalized::int, total::real, lo::real, a::real, b::vector[n], args...)::vector[n-1] = begin
        @stan_assert false "shares: a conditional cell draw is not available"
        rep_vector(0., n-1)
    end
    @lpxf brm_lower_residuals_lpdf(r::vector[G], y::vector[m], family, normalized::int,
            ends::int[G], lo::vector[N], args...)::real =
        sum(brm_lower_residuals_lpdfs(r, y, family, normalized, ends, lo, args...))
    brm_lower_residuals_lpdfs(r::vector[G], y::vector[m], family, normalized::int,
            ends::int[G], lo::vector[N], args...)::vector[G] = begin
        x = brm_lower_grouped_values(y, ends, r, lo)
        lp = brm_lower_lpdfs(x, family, normalized, lo, args...)
        out = rep_vector(0., G)
        start = 1
        coordinate = 1
        for k in 1:G
            K = ends[k] - start + 1
            slack = r[k] - sum(segment(lo, start, K))
            if is_inf(slack) || is_nan(slack) || slack < 0 || (K > 1 && slack == 0)
                out[k] = negative_infinity()
            else
                out[k] = sum(segment(lp, start, K))
                if K > 1
                    out[k] += brm_lower_shares_logjac(
                        segment(y, coordinate, K-1), r[k], segment(lo, start, K))
                end
            end
            start = ends[k] + 1
            coordinate += K - 1
        end
        out
    end
    brm_lower_residuals_rng(vector[G], y::vector[m], family, normalized::int,
            ends::int[G], lo::vector[N], args...)::vector[G] = begin
        x = brm_lower_rng(vector[N], family, normalized, lo, args...)
        out = rep_vector(0., G)
        start = 1
        for k in 1:G
            out[k] = sum(segment(x, start, ends[k] - start + 1))
            start = ends[k] + 1
        end
        out
    end
end

function _sb_lower_cell_coordinates(lhs, spec, bounded, scope)
    F = _sb_cell_binding(scope, first(bounded.base.args))
    floor = _brm_lower_floor(F)
    call = _sb_cell_family(bounded.base, scope)
    family = _sb_stan_family(first(call.args))
    lower = _sb_cell_value(bounded.lower, scope)
    floor == -Inf || (lower = Expr(:call, :fmax, floor, lower))
    total = _sb_cell_value(spec.total, scope)
    coordinates = _brm_total_coordinates_name(lhs)
    lower_name = Symbol(lhs, :_lower)
    excess_name = Symbol(lhs, :_shares_excess)
    Any[Expr(:call, :~, coordinates, Expr(:call, brm_lower_cell, family,
            Int(bounded.normalized), total, lower, call.args[2:end]...)),
        Expr(:(=), lower_name, Expr(:call, brm_lower_vector, lower,
            Expr(:call, :+, Expr(:call, :num_elements, coordinates), 1))),
        Expr(:(=), excess_name, Expr(:call, :-,
            Expr(:call, brm_lower_cell_values, coordinates, total, lower_name), lower_name)),
        Expr(:(=), lhs, Expr(:call, brm_lower_cell_from_excess, excess_name, total, lower_name))]
end

_sb_mi_lower_shares_response = StanBlocks.@slic begin
    n_free = num_elements(Jfree)
    n_shares = num_elements(Jgrp) - num_elements(residuals)
    y_free::vector[n_free] ~ dummy()
    y_shares::vector[n_shares] ~ flat()
    residuals ~ dummy()
    y_obs ~ dummy()
    y_mis = append_row(y_free, brm_lower_grouped_values(y_shares, ends, residuals, shares_lower))
    return mi_merge(y_obs, y_mis, Jobs, Jmis, num_elements(Jobs) + num_elements(Jmis))
end

function _sb_emit_lower_mi!(stmts, data, plan, spec)
    source = plan.source
    base = _as_expr_column(spec.base)
    base === nothing && error("sbimpl: a bounded completion needs a base distribution call")
    floor = _brm_lower_floor(getf(base))
    raw_lower = _brm_materialize_data_expression(spec.lower)
    raw_lower === nothing || _brm_validate_lower_completion(plan,
        _brm_effective_lower(raw_lower, floor))
    translated = Any[]
    _sb_likelihood!(translated, :y_obs, base, data)
    length(translated) == 1 || error("sbimpl: a bounded completion needs one elementwise base law")
    call = only(translated).args[3]
    family = _sb_stan_family(first(call.args))
    lower = _sb_scalar_expr(spec.lower, data)
    floor == -Inf || (lower = Expr(:call, :fmax, floor, lower))
    lowered = (lower, call.args[2:end]...)
    names = ntuple(i -> Symbol(:mi_arg_, i), length(lowered))
    at(rows) = [Expr(:call, :maybe_index, arg, rows) for arg in names]
    observed(rows) = Expr(:call, brm_lower, family, Int(spec.normalized), at(rows)...)
    # Native wrappers own independent draws and their activity analysis. In
    # particular, an unread restricted entry still carries its support mass
    # about the law's sampled parents; an unread normalized entry integrates out.
    sampled(rows) = Expr(:call, spec.normalized ? :truncated : :restricted,
        Expr(:parameters, Expr(:kw, :lower, first(at(rows)))),
        Expr(:call, first(call.args), at(rows)[2:end]...))
    grouped = _brm_mi_grouped(plan.totals)
    submodel = if grouped
        Base.merge(_sb_mi_lower_shares_response,
            Expr(:(=), :shares_lower, Expr(:call, brm_lower_vector,
                first(at(:Jgrp)), Expr(:call, :num_elements, :Jgrp))),
            Expr(:call, :~, :y_free, sampled(:Jfree)),
            Expr(:call, :~, :residuals, Expr(:call, brm_lower_residuals,
                :y_shares, family, Int(spec.normalized), :ends, :shares_lower, at(:Jgrp)[2:end]...)),
            Expr(:call, :~, :y_obs, observed(:Jobs)))
    else
        Base.merge(_sb_mi_response,
            Expr(:call, :~, :y_mis, sampled(:Jmis)),
            Expr(:call, :~, :y_obs, observed(:Jobs)))
    end
    obs, jobs, jmis = Symbol(source, :_obs), Symbol(:Jobs_, source), Symbol(:Jmis_, source)
    data[obs], data[jobs], data[jmis] = plan.observed_values, plan.observed_indices, plan.missing_indices
    totals = grouped ? _sb_bind_mi_totals!(data, source, plan) : nothing
    entry = _sb_missing_preproc(plan, jobs, jmis)
    entry = PreprocEntry(entry.kind, merge(entry.const_, (; lower_bounded=true)), entry.raw_ref, true)
    if raw_lower !== nothing
        entry = PreprocEntry(entry.kind, merge(entry.const_,
            (; lower_bound=spec.lower, lower_floor=floor)), entry.raw_ref, true)
    end
    _sb_record_preproc!(data, obs, entry)
    kwargs = Expr(:parameters, Expr(:kw, :y_obs, obs), Expr(:kw, :Jobs, jobs), Expr(:kw, :Jmis, jmis),
        (grouped ? (Expr(:kw, :Jfree, totals.free), Expr(:kw, :Jgrp, totals.grouped),
            Expr(:kw, :ends, totals.ends), Expr(:kw, :residuals, totals.residuals)) : ())...,
        (Expr(:kw, arg, value) for (arg, value) in zip(names, lowered))...)
    push!(stmts, Expr(:call, :~, source, Expr(:call, submodel, kwargs)))
end
