# Distribution statistics are deterministic operations, not sampling laws.
# Recognize the ordinary Julia call by identity and unpack its constructor
# without materializing a Julia Distribution inside a Stan program. Numerical
# evaluation belongs to Distributions (RK) and StanBlocks (Stan).

_sb_distribution_callable(scope, head) =
    head isa Function || head isa Type ? head : _sb_cell_binding(scope, head)

_sb_statistic_keyword(arg) = arg isa Symbol
_sb_statistic_keyword(arg::Expr) =
    Meta.isexpr(arg, :kw, 2) ? first(arg.args) isa Symbol :
    Meta.isexpr(arg, :parameters) && all(_sb_statistic_keyword, arg.args)

function _sb_distribution_call_parts(ex, scope)
    Meta.isexpr(ex, :call) || return nothing
    callable = _sb_distribution_callable(scope, first(ex.args))
    args = Tuple(filter(!_sb_cell_keyword_argument, ex.args[2:end]))
    keyword_args = filter(_sb_cell_keyword_argument, ex.args[2:end])
    all(_sb_statistic_keyword, keyword_args) || return nothing
    keywords = foldl(_sb_cell_keywords!, keyword_args;
        init=Pair{Symbol,Any}[])
    (; callable, args, kwargs=NamedTuple(keywords))
end

function _sb_lognormal_statistic_law(ex, scope)
    call = _sb_distribution_call_parts(ex, scope)
    call === nothing && return nothing
    if call.callable isa Type && call.callable <: LogNormal
        all(==(:check_args), keys(call.kwargs)) || return nothing
        length(call.args) <= 2 || return nothing
        mu = isempty(call.args) ? 0.0 : call.args[1]
        sigma = length(call.args) < 2 ? 1.0 : call.args[2]
        return (; mu, sigma, lower=0.0, truncated=false)
    elseif call.callable === Distributions.truncated
        isempty(call.args) && return nothing
        base = _sb_lognormal_statistic_law(first(call.args), scope)
        base === nothing && return nothing
        base.truncated && return nothing
        if length(call.args) == 1
            all(k -> k in (:lower, :upper), keys(call.kwargs)) || return nothing
            lower = get(call.kwargs, :lower, nothing)
            upper = get(call.kwargs, :upper, nothing)
        elseif length(call.args) == 3 && isempty(call.kwargs)
            lower, upper = call.args[2:3]
        else
            return nothing
        end
        # Julia distinguishes an absent upper bound from an explicit Inf:
        # the latter constructs a two-sided Truncated, whose moments are not
        # implemented. Keep the same accepted family on both backends.
        unbounded_upper = upper === nothing || upper === :nothing
        unbounded_upper || error("sbimpl: LogNormal mean/var currently support lower truncation only; " *
            "omit the upper bound or pass upper=nothing")
        lower = lower === nothing || lower === :nothing ? 0.0 : lower
        return (; base.mu, base.sigma, lower, truncated=true)
    end
    nothing
end

function _sb_distribution_statistic(callable, args, scope)
    is_logcdf = callable === logcdf || callable === logccdf
    (callable === mean || callable === var || is_logcdf) || return nothing
    length(args) == (is_logcdf ? 2 : 1) || return nothing
    law = _sb_lognormal_statistic_law(first(args), scope)
    law === nothing && return nothing
    if is_logcdf
        law.truncated && return nothing
        name = callable === logcdf ? :lognormal_lcdf_stable : :lognormal_lccdf_stable
        return Expr(:call, name, args[2], law.mu, law.sigma)
    end
    name = callable === mean ? :lognormal_truncated_mean : :lognormal_truncated_variance
    Expr(:call, name, law.mu, law.sigma, law.lower)
end
