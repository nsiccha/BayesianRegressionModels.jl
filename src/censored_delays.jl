# ---- Primary-event and double interval censored delays ---------------------
#
# Delays between a primary and a secondary event (infection -> symptoms,
# onset -> report) are usually recorded as two calendar windows: the primary
# event happened somewhere in [p_L, p_L + w_P], the secondary one somewhere in
# [s_L, s_L + w_S], and only pairs whose secondary event happened before the
# observation time T are in the data. The delay X has a parametric law F.
#
# Measured from the start of the primary window, the secondary event is at
# D = U + X with U ~ Uniform(0, w_P). Its law, the primary-censored delay, has
#
#     F_pc(d) = (1 / w_P) ∫_{d - w_P}^{d} F(t) dt
#             = (G(d) - G(d - w_P)) / w_P,      G(q) = ∫_0^q F(t) dt = q F(q) - E[X; X <= q]
#     S_pc(d) = 1 - F_pc(d)
#             = (H(d - w_P) - H(d)) / w_P,      H(q) = ∫_q^∞ S(t) dt = E[X; X > q] - q S(q)
#
# with closed forms of the partial expectations for a lognormal, gamma or
# Weibull delay (the primarycensored / CensoredDistributions.jl analytical
# solutions). One observation contributes the probability of its secondary
# window, conditioned on the truncation window [L, U]:
#
#     log P(D in [y, y + w_S] | L <= D <= U)
#       = log(F_pc(y + w_S) - F_pc(y)) - log(F_pc(U) - F_pc(L))
#
# (epidist's marginal model; Park et al. 2024, Charniga et al. 2024). Without a
# secondary window the observation contributes the density f_pc(y) =
# (F(y) - F(y - w_P)) / w_P instead.
#
# Accuracy: every interval probability is the difference of the two CDF values
# when the lower end sits below the median of D, and the difference of the two
# survival values otherwise, so the upper tail never subtracts two numbers close
# to one. The survival forms use complementary CDFs (`normccdf`, the upper
# regularised incomplete gamma) directly.
#
# The same mathematics is the observation law of both backends: SBBRMI emits
# the Stan functions in `sb_censored_delays.jl`, RKBRMI calls these Julia
# methods. `test/censored_delays.jl` checks them against an independent
# 256-bit quadrature of the convolution.

"""
    PrimaryCensored(dist, primary_event)

The law of `U + X`: a delay `X ~ dist` that starts at a primary event
`U ~ primary_event` known only to lie in a window. Construct it with
[`primary_censored`](@ref).
"""
struct PrimaryCensored{D<:Distributions.ContinuousUnivariateDistribution,
                       P<:Distributions.Uniform} <:
       Distributions.ContinuousUnivariateDistribution
    dist::D
    primary_event::P
end

"""
    primary_censored(dist, primary_event)

The distribution of a delay `X ~ dist` measured from the start of the time axis
when its primary event happened at an unknown time `U ~ primary_event`: the law
of `U + X`. `dist` is a `LogNormal`, `Gamma` or `Weibull`; `primary_event` is a
`Uniform(a, b)` window. Same signature and meaning as CensoredDistributions.jl's
`primary_censored` with its analytical solutions.

In a `@brm` model, `delay ~ primary_censored(LogNormal(mu, sigma), Uniform(0, w))`
observes exact secondary-event times (measured from each primary window's
start) with a primary-event window of width `w`, a number or a data column. For
secondary events known only to a window, and for truncation by the observation
time, use [`double_interval_censored`](@ref).
"""
function primary_censored(dist::Distributions.ContinuousUnivariateDistribution,
                          primary_event::Distributions.Uniform)
    _cd_check_delay(dist)
    PrimaryCensored(dist, primary_event)
end
primary_censored(dist, primary_event) = throw(ArgumentError(
    "primary_censored: the delay must be a LogNormal, Gamma or Weibull distribution " *
    "and the primary event a Uniform(a, b) window; got $(typeof(dist)) and " *
    "$(typeof(primary_event))"))

"""
    DoubleIntervalCensored(pc, lower, upper, interval)

A [`PrimaryCensored`](@ref) delay truncated to `[lower, upper]` and observed
through a secondary window of width `interval` starting at the observed value.
Construct it with [`double_interval_censored`](@ref).
"""
struct DoubleIntervalCensored{PC<:PrimaryCensored,T<:Real} <:
       Distributions.ContinuousUnivariateDistribution
    pc::PC
    lower::T
    upper::T
    interval::T   # 0: the secondary event time is observed exactly
end

"""
    double_interval_censored(dist; primary_event = Uniform(0, 1), lower = nothing,
                             upper = nothing, interval = nothing)

A delay with a censored primary event, truncation and a censored secondary
event: `X ~ dist` starts at an unknown time `U ~ primary_event`, only delays
`U + X` in `[lower, upper]` are observed (right truncation by the observation
time is `upper`), and each observation `y` is the start of its secondary-event
window `[y, y + interval]`. Its log density at `y` is

    log(F_pc(y + interval) - F_pc(y)) - log(F_pc(upper) - F_pc(lower))

with `F_pc` the CDF of [`primary_censored`](@ref)`(dist, primary_event)`. This
is epidist's marginal delay model. The keywords follow CensoredDistributions.jl's
`double_interval_censored` (primary censoring, then truncation, then the
secondary window), except that `y` is the window's start as recorded rather
than rounded down onto a grid of width `interval`; for data on that grid the
two coincide. Without `interval` the observation is an exact secondary-event
time and contributes the truncated density.

In a `@brm` model:

```julia
delay ~ double_interval_censored(LogNormal(mu, sigma);
    primary_event = Uniform(0, pwindow), upper = obs_time, interval = swindow)
```

where `delay` is the start of the secondary window measured from the start of
the primary window, and `pwindow`, `obs_time` and `swindow` are numbers or data
columns. Wrap it in `weighted(..., fweights(n))` for aggregated rows.
"""
function double_interval_censored(dist; primary_event=Distributions.Uniform(0, 1),
                                  lower=nothing, upper=nothing, interval=nothing)
    pc = primary_censored(dist, primary_event)
    lo = something(lower, -Inf)
    hi = something(upper, Inf)
    width = something(interval, 0)
    lo < hi || throw(ArgumentError(
        "double_interval_censored: `lower` must be below `upper`, got $lo and $hi"))
    width >= 0 || throw(ArgumentError(
        "double_interval_censored: `interval` must be nonnegative, got $width"))
    lo, hi, width = promote(float(lo), float(hi), float(width))
    DoubleIntervalCensored(pc, lo, hi, width)
end

# ---- the delay families ------------------------------------------------------
_cd_check_delay(::Distributions.LogNormal) = nothing
_cd_check_delay(::Distributions.Gamma) = nothing
_cd_check_delay(::Distributions.Weibull) = nothing
_cd_check_delay(dist) = throw(ArgumentError(
    "primary_censored: the delay must be a LogNormal, Gamma or Weibull " *
    "distribution (closed-form primary censoring); got $(typeof(dist))"))

# (F(q), S(q), E[X; X <= q], E[X; X > q]) of the delay at q > 0, from the
# family parameters. Kept parameter-level (not `Distributions` objects) so the
# RK law and the Stan functions share one argument order.
function _cd_lognormal_parts(q, mu, sigma)
    z = (log(q) - mu) / sigma
    m = exp(mu + sigma^2 / 2)
    (StatsFuns.normcdf(z), StatsFuns.normccdf(z),
     m * StatsFuns.normcdf(z - sigma), m * StatsFuns.normccdf(z - sigma))
end
function _cd_gamma_parts(q, shape, scale)
    x = q / scale
    p, c = SpecialFunctions.gamma_inc(shape, x, 0)
    p1, c1 = SpecialFunctions.gamma_inc(shape + 1, x, 0)
    (p, c, shape * scale * p1, shape * scale * c1)
end
function _cd_weibull_parts(q, shape, scale)
    x = (q / scale)^shape
    s = exp(-x)
    a = 1 + 1 / shape
    p1, c1 = SpecialFunctions.gamma_inc(a, x, 0)
    m = scale * SpecialFunctions.gamma(a)
    (-expm1(-x), s, m * p1, m * c1)
end
_cd_mean(::typeof(_cd_lognormal_parts), mu, sigma) = exp(mu + sigma^2 / 2)
_cd_mean(::typeof(_cd_gamma_parts), shape, scale) = shape * scale
_cd_mean(::typeof(_cd_weibull_parts), shape, scale) =
    scale * SpecialFunctions.gamma(1 + 1 / shape)

# G(q) = ∫_0^q F, H(q) = ∫_q^∞ S.
function _cd_G(parts, q, a, b)
    q <= 0 && return zero(float(promote_type(typeof(q), typeof(a), typeof(b))))
    F, _, M, _ = parts(q, a, b)
    q * F - M
end
function _cd_H(parts, q, a, b)
    q <= 0 && return _cd_mean(parts, a, b) - q
    _, S, _, Mc = parts(q, a, b)
    Mc - q * S
end
# Primary-censored CDF and survival at d (measured from the window start).
_cd_cdf(parts, d, w, a, b) = d <= 0 ? zero(float(promote_type(typeof(d), typeof(a)))) :
    isinf(d) ? one(float(promote_type(typeof(w), typeof(a)))) :
    (_cd_G(parts, d, a, b) - _cd_G(parts, d - w, a, b)) / w
_cd_ccdf(parts, d, w, a, b) = d <= 0 ? one(float(promote_type(typeof(d), typeof(a)))) :
    isinf(d) ? zero(float(promote_type(typeof(w), typeof(a)))) :
    (_cd_H(parts, d - w, a, b) - _cd_H(parts, d, a, b)) / w

# P(l <= D <= u), l <= u, from whichever side subtracts small numbers.
function _cd_mass(parts, l, u, w, a, b)
    lower = _cd_cdf(parts, l, w, a, b)
    lower <= 0.5 ? _cd_cdf(parts, u, w, a, b) - lower :
        _cd_ccdf(parts, l, w, a, b) - _cd_ccdf(parts, u, w, a, b)
end
# Density of D at d: (F(d) - F(d - w)) / w, or the survival form above the median.
function _cd_density(parts, d, w, a, b)
    d <= 0 && return zero(float(promote_type(typeof(d), typeof(a))))
    F, S, _, _ = parts(d, a, b)
    if F <= 0.5
        lo = d - w <= 0 ? zero(F) : first(parts(d - w, a, b))
        (F - lo) / w
    else
        hi = d - w <= 0 ? one(S) : parts(d - w, a, b)[2]
        (hi - S) / w
    end
end

"""
    censored_delay_logpdf(parts, y, primary_width, lower, upper, interval, a, b)

Log density of one double interval censored delay observation, shared by
RKBRMI and the Julia methods: `parts` names the delay family
(`_cd_lognormal_parts`, `_cd_gamma_parts`, `_cd_weibull_parts`) with parameters
`(a, b)`; `y`, `lower` and `upper` are measured from the primary window's start.
"""
function censored_delay_logpdf(parts, y, primary_width, lower, upper, interval, a, b)
    T = float(promote_type(typeof(y), typeof(a), typeof(b), typeof(primary_width)))
    hi = interval > 0 ? y + interval : y
    (lower <= y && hi <= upper) || return T(-Inf)
    numerator = interval > 0 ? _cd_mass(parts, y, hi, primary_width, a, b) :
        _cd_density(parts, y, primary_width, a, b)
    normaliser = lower <= 0 && isinf(upper) ? one(T) :
        _cd_mass(parts, max(lower, zero(lower)), upper, primary_width, a, b)
    (numerator > 0 && normaliser > 0) || return T(-Inf)
    log(numerator) - log(normaliser)
end

_cd_parts(::Distributions.LogNormal) = _cd_lognormal_parts
_cd_parts(::Distributions.Gamma) = _cd_gamma_parts
_cd_parts(::Distributions.Weibull) = _cd_weibull_parts
_cd_params(d::Distributions.LogNormal) = Distributions.params(d)
_cd_params(d::Distributions.Gamma) = Distributions.params(d)         # (shape, scale)
_cd_params(d::Distributions.Weibull) = Distributions.params(d)       # (shape, scale)

_cd_shift(d::PrimaryCensored) = Distributions.minimum(d.primary_event)
_cd_width(d::PrimaryCensored) = Distributions.maximum(d.primary_event) - _cd_shift(d)

Distributions.minimum(d::PrimaryCensored) = _cd_shift(d)
Distributions.maximum(::PrimaryCensored) = Inf
Distributions.insupport(d::PrimaryCensored, x::Real) = x >= _cd_shift(d)
Distributions.cdf(d::PrimaryCensored, x::Real) =
    _cd_cdf(_cd_parts(d.dist), x - _cd_shift(d), _cd_width(d), _cd_params(d.dist)...)
Distributions.ccdf(d::PrimaryCensored, x::Real) =
    _cd_ccdf(_cd_parts(d.dist), x - _cd_shift(d), _cd_width(d), _cd_params(d.dist)...)
Distributions.logcdf(d::PrimaryCensored, x::Real) = log(Distributions.cdf(d, x))
Distributions.logccdf(d::PrimaryCensored, x::Real) = log(Distributions.ccdf(d, x))
Distributions.pdf(d::PrimaryCensored, x::Real) =
    _cd_density(_cd_parts(d.dist), x - _cd_shift(d), _cd_width(d), _cd_params(d.dist)...)
Distributions.logpdf(d::PrimaryCensored, x::Real) = log(Distributions.pdf(d, x))
Base.rand(rng::Random.AbstractRNG, d::PrimaryCensored) =
    rand(rng, d.primary_event) + rand(rng, d.dist)

Distributions.minimum(d::DoubleIntervalCensored) = max(d.lower, Distributions.minimum(d.pc))
Distributions.maximum(d::DoubleIntervalCensored) = d.upper
Distributions.insupport(d::DoubleIntervalCensored, x::Real) =
    d.lower <= x && x + d.interval <= d.upper
function Distributions.logpdf(d::DoubleIntervalCensored, x::Real)
    shift = _cd_shift(d.pc)
    censored_delay_logpdf(_cd_parts(d.pc.dist), x - shift, _cd_width(d.pc),
        d.lower - shift, d.upper - shift, d.interval, _cd_params(d.pc.dist)...)
end
Distributions.pdf(d::DoubleIntervalCensored, x::Real) = exp(Distributions.logpdf(d, x))
# A draw from the truncated primary-censored law (rejection), reported as the
# start of its secondary window on the grid of width `interval` from zero.
function Base.rand(rng::Random.AbstractRNG, d::DoubleIntervalCensored)
    for _ in 1:1_000_000
        x = rand(rng, d.pc)
        d.lower <= x <= d.upper || continue
        d.interval > 0 || return x
        start = floor(x / d.interval) * d.interval
        start + d.interval <= d.upper && start >= d.lower && return start
    end
    error("double_interval_censored: rejection sampling found no draw in " *
          "[$(d.lower), $(d.upper)] in 10^6 tries")
end

# ---- formula plan shared by both backends ------------------------------------
#
# `y ~ primary_censored(Base(a, b), Uniform(p_lo, p_hi))` and
# `y ~ double_interval_censored(Base(a, b); primary_event, lower, upper, interval)`
# plan to one record: the family id (1 LogNormal, 2 Gamma, 3 Weibull, in their
# Julia parameterizations), the base arguments and the window/bound
# expressions, each a number, a data column or (bounds only) `nothing`.
struct _BRMCensoredDelayPlan
    family::Int
    base::Any
    a::Any
    b::Any
    primary_lo::Any
    primary_hi::Any
    lower::Any
    upper::Any
    interval::Any
end

_brm_censored_delay_family_id(::Type{<:Distributions.LogNormal}) = 1
_brm_censored_delay_family_id(::Type{<:Distributions.Gamma}) = 2
_brm_censored_delay_family_id(::Type{<:Distributions.Weibull}) = 3
_brm_censored_delay_family_id(f) = nothing
_brm_is_censored_delay(f) = f === primary_censored || f === double_interval_censored

function _brm_censored_delay_plan(f, args, kwargs::NamedTuple; prefix="BRM backend lowering")
    name = f === primary_censored ? "primary_censored" : "double_interval_censored"
    if f === primary_censored
        length(args) == 2 && isempty(kwargs) || error(
            "$prefix: `primary_censored` takes a delay distribution and a primary-event " *
            "window, `primary_censored(LogNormal(mu, sigma), Uniform(0, w))`")
        base, primary = args
        lower = upper = interval = nothing
    else
        length(args) == 1 || error(
            "$prefix: `double_interval_censored` takes one delay distribution and " *
            "keywords: `double_interval_censored(LogNormal(mu, sigma); " *
            "primary_event = Uniform(0, w), upper = obs_time, interval = 1)`")
        unknown = setdiff(collect(keys(kwargs)), [:primary_event, :lower, :upper, :interval])
        isempty(unknown) || error(
            "$prefix: `double_interval_censored` accepts the keywords `primary_event`, " *
            "`lower`, `upper` and `interval`; got $unknown")
        base = only(args)
        primary = get(kwargs, :primary_event, nothing)
        lower = _brm_normalize_response_bound(get(kwargs, :lower, nothing))
        upper = _brm_normalize_response_bound(get(kwargs, :upper, nothing))
        interval = _brm_normalize_response_bound(get(kwargs, :interval, nothing))
    end
    base_call = _as_expr_column(base)
    family = isnothing(base_call) ? nothing : _brm_censored_delay_family_id(getf(base_call))
    isnothing(family) && error(
        "$prefix: `$name` needs a `LogNormal`, `Gamma` or `Weibull` delay as its " *
        "first argument (the closed-form primary censoring); got " *
        "$(isnothing(base_call) ? typeof(base) : getf(base_call))")
    isempty(getkwargs(base_call)) && length(getargs(base_call)) == 2 || error(
        "$prefix: `$name` needs the delay's two positional parameters, " *
        "e.g. `LogNormal(mu, sigma)`, `Gamma(shape, scale)`, `Weibull(shape, scale)`")
    a, b = getargs(base_call)
    primary_lo, primary_hi = if isnothing(primary)
        (0.0, 1.0)
    else
        window = _as_expr_column(primary)
        !isnothing(window) && getf(window) <: Distributions.Uniform &&
            length(getargs(window)) == 2 && isempty(getkwargs(window)) || error(
            "$prefix: `$name` needs the primary event as a `Uniform(a, b)` window " *
            "(numbers or data columns); got $(isnothing(window) ? typeof(primary) : getf(window))")
        Tuple(getargs(window))
    end
    _BRMCensoredDelayPlan(family, base_call, a, b, primary_lo, primary_hi,
                          lower, upper, interval)
end

# Row checks on the observed values: each window lies inside the truncation
# bounds and the primary window has positive width. `values(x)` materializes a
# plan field (number, data column or `nothing`) to a number or a vector.
function _brm_censored_delay_check(plan::_BRMCensoredDelayPlan, target, y::AbstractVector,
                                   values; prefix="BRM backend lowering")
    at(x, i) = x isa AbstractVector ? x[i] : x
    lo, hi = values(plan.primary_lo), values(plan.primary_hi)
    lower = isnothing(plan.lower) ? -Inf : values(plan.lower)
    upper = isnothing(plan.upper) ? Inf : values(plan.upper)
    interval = isnothing(plan.interval) ? 0.0 : values(plan.interval)
    for x in (lo, hi, lower, upper, interval)
        x isa Real || x isa AbstractVector{<:Real} || error(
            "$prefix: censored-delay windows and bounds of `$target` must be numbers " *
            "or numeric data columns")
        x isa AbstractVector && length(x) != length(y) && error(
            "$prefix: a censored-delay window or bound of `$target` has $(length(x)) " *
            "rows; expected $(length(y))")
    end
    for i in eachindex(y)
        at(hi, i) > at(lo, i) || error(
            "$prefix: `$target` row $i: the primary-event window must have positive width")
        at(interval, i) >= 0 || error(
            "$prefix: `$target` row $i: the secondary window `interval` must be nonnegative")
        at(lower, i) <= y[i] && y[i] + at(interval, i) <= at(upper, i) || error(
            "$prefix: `$target` row $i: the observation window " *
            "[$(y[i]), $(y[i] + at(interval, i))] lies outside the truncation bounds " *
            "[$(at(lower, i)), $(at(upper, i))]")
    end
    nothing
end

# The authored censored-delay family as the backend family call
# `brm_censored_delay(fam, p_lo, p_hi, lower, upper, interval, a, b)`, through
# a `weighted(...)` wrapper. SBBRMI emits it as the Stan family of that name
# (`sb_censored_delays.jl`); RKBRMI observes it through the `LogDensity` law
# its `_rk_observation_source!` provider defines.
function _brm_canonical_censored_delay(x)
    x isa ExprColumn || return x
    f = getf(x)
    if f === weighted
        args = getargs(x)
        isempty(args) && return x
        inner = _brm_canonical_censored_delay(first(args))
        inner === first(args) && return x
        return ExprColumn(weighted, inner, args[2:end]...; getkwargs(x)...)
    end
    _brm_is_censored_delay(f) || return x
    plan = _brm_censored_delay_plan(f, getargs(x), getkwargs(x); prefix="BRM preparation")
    ExprColumn(brm_censored_delay, plan.family, plan.primary_lo, plan.primary_hi,
        something(plan.lower, -Inf), something(plan.upper, Inf),
        something(plan.interval, 0.0), plan.a, plan.b)
end

# The censored-delay plan under an optional `weighted(...)`, or `nothing`.
function _brm_censored_delay_observation_plan(x; prefix="BRM preparation")
    x isa ExprColumn || return nothing
    getf(x) === weighted && !isempty(getargs(x)) &&
        return _brm_censored_delay_observation_plan(first(getargs(x)); prefix)
    _brm_is_censored_delay(getf(x)) || return nothing
    _brm_censored_delay_plan(getf(x), getargs(x), getkwargs(x); prefix)
end

# The RKBRMI observation law: the Julia closed forms, selected by family id.
function _brm_censored_delay_logpdf(y, fam, p_lo, p_hi, lower, upper, interval, a, b)
    w, d, lo, hi = p_hi - p_lo, y - p_lo, lower - p_lo, upper - p_lo
    fam == 1 ? censored_delay_logpdf(_cd_lognormal_parts, d, w, lo, hi, interval, a, b) :
    fam == 2 ? censored_delay_logpdf(_cd_gamma_parts, d, w, lo, hi, interval, a, b) :
               censored_delay_logpdf(_cd_weibull_parts, d, w, lo, hi, interval, a, b)
end
