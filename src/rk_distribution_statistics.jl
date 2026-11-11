# Resolve authored distribution statistics before their constructors become
# opaque ordinary calls. This file owns syntax and callable identity only;
# the RK extension supplies the producer-owned numerical graphs.
struct _RKLogNormalStatistic{Statistic,Side} end

_rk_statistic_call_parts(ex::_BRMPreparedExpr, _scope) =
    (; ex.callable, ex.args, ex.kwargs)
_rk_statistic_call_parts(ex, scope) = _sb_distribution_call_parts(ex, scope)

_rk_statistic_absent_bound(value) = value === nothing || value === :nothing

function _rk_lognormal_statistic_law(ex, scope)
    call = _rk_statistic_call_parts(ex, scope)
    call === nothing && return nothing
    if call.callable isa Type && call.callable <: LogNormal
        all(==(:check_args), keys(call.kwargs)) || return nothing
        length(call.args) <= 2 || return nothing
        location = isempty(call.args) ? 0.0 : call.args[1]
        scale = length(call.args) < 2 ? 1.0 : call.args[2]
        return (; location, scale, side=:base, bound=nothing)
    end
    call.callable === Distributions.truncated || return nothing
    isempty(call.args) && return nothing
    base = _rk_lognormal_statistic_law(first(call.args), scope)
    base === nothing && return nothing
    base.side === :base || return nothing
    if length(call.args) == 1
        all(k -> k in (:lower, :upper), keys(call.kwargs)) || return nothing
        lower = get(call.kwargs, :lower, nothing)
        upper = get(call.kwargs, :upper, nothing)
    elseif length(call.args) == 3 && isempty(call.kwargs)
        lower, upper = call.args[2:3]
    else
        return nothing
    end
    has_lower = !_rk_statistic_absent_bound(lower)
    has_upper = !_rk_statistic_absent_bound(upper)
    # Two-sided Truncated has distinct Julia dispatch and is outside the
    # one-sided graph contract, including an explicitly supplied infinite end.
    has_lower && has_upper && return nothing
    !has_lower && !has_upper && return base
    (; base.location, base.scale, side=has_upper ? :upper : :lower,
       bound=has_upper ? upper : lower)
end

function _rk_distribution_statistic!(lower, bindings, callable, args, kwargs,
        taken, scope)
    isempty(kwargs) || return nothing
    tail = callable === logcdf || callable === logccdf
    (callable === mean || callable === var || tail) || return nothing
    length(args) == (tail ? 2 : 1) || return nothing
    law = _rk_lognormal_statistic_law(first(args), scope)
    law === nothing && return nothing
    if tail
        law.side === :base || return nothing
        statistic = callable === logcdf ? :logcdf : :logccdf
        side, bound = :base, args[2]
    else
        statistic = callable === mean ? :mean : :var
        # The producer's ordinary LogNormal moment route is the lower object
        # with zero bound; its density/tail object has no moment endpoints.
        side, bound = law.side === :base ? (:lower, 0.0) : (law.side, law.bound)
    end
    callee = _rk_value_callee!(bindings, _RKLogNormalStatistic{statistic,side}(), taken)
    Expr(:call, callee, lower(law.location), lower(law.scale), lower(bound))
end
