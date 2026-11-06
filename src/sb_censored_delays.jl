# SBBRMI's observation law for `primary_censored` / `double_interval_censored`
# (`censored_delays.jl`): the same mathematics as the Julia methods, as Stan
# functions. The delay family is an integer id (1 LogNormal(mu, sigma),
# 2 Gamma(shape, scale), 3 Weibull(shape, scale), the Julia parameterizations),
# as primarycensored selects its closed forms by `dist_id`.
#
# `brm_censored_delay(fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)` observes
# the start `y` of a secondary window of width `interval` (0: an exact time)
# Gamma and Weibull survival terms use `1 - gamma_p`, not `gamma_q`: Stan Math
# differentiates `gamma_q` in its shape argument only to about 1e-6, which the
# window differences amplify to ~1e-3 relative gradient error (measured in
# `test/censored_delays.jl`); `gamma_p`'s shape derivative is accurate to ~1e-10.
# This costs relative precision only where a survival value is itself tiny.
# for a delay whose primary event lies in [p_lo, p_hi], truncated to
# [trunc_lo, trunc_hi]; all positions on the response's time axis.
StanBlocks.@deffun begin
    brm_cd_F_base(q::real, fam::int, a::real, b::real)::real =
        fam == 1 ? 0.5 * erfc(-((log(q) - a) / b) / sqrt(2.0)) :
        fam == 2 ? gamma_p(a, q / b) : -expm1(-((q / b) ^ a))
    brm_cd_S_base(q::real, fam::int, a::real, b::real)::real =
        fam == 1 ? 0.5 * erfc(((log(q) - a) / b) / sqrt(2.0)) :
        fam == 2 ? 1 - gamma_p(a, q / b) : exp(-((q / b) ^ a))
    # E[X; X <= q] and E[X; X > q].
    brm_cd_lower_moment(q::real, fam::int, a::real, b::real)::real =
        fam == 1 ? exp(a + 0.5 * b ^ 2) * 0.5 * erfc(-((log(q) - a) / b - b) / sqrt(2.0)) :
        fam == 2 ? a * b * gamma_p(a + 1, q / b) :
        b * tgamma(1 + 1 / a) * gamma_p(1 + 1 / a, (q / b) ^ a)
    brm_cd_upper_moment(q::real, fam::int, a::real, b::real)::real =
        fam == 1 ? exp(a + 0.5 * b ^ 2) * 0.5 * erfc(((log(q) - a) / b - b) / sqrt(2.0)) :
        fam == 2 ? a * b * (1 - gamma_p(a + 1, q / b)) :
        b * tgamma(1 + 1 / a) * (1 - gamma_p(1 + 1 / a, (q / b) ^ a))
    brm_cd_mean(fam::int, a::real, b::real)::real =
        fam == 1 ? exp(a + 0.5 * b ^ 2) : fam == 2 ? a * b : b * tgamma(1 + 1 / a)
    # G(q) = ∫_0^q F and H(q) = ∫_q^∞ S.
    brm_cd_G(q::real, fam::int, a::real, b::real)::real =
        q <= 0 ? 0.0 : q * brm_cd_F_base(q, fam, a, b) - brm_cd_lower_moment(q, fam, a, b)
    brm_cd_H(q::real, fam::int, a::real, b::real)::real =
        q <= 0 ? brm_cd_mean(fam, a, b) - q :
        brm_cd_upper_moment(q, fam, a, b) - q * brm_cd_S_base(q, fam, a, b)
    # Primary-censored CDF and survival at d (from the primary window's start).
    brm_cd_F_pc(d::real, w::real, fam::int, a::real, b::real)::real =
        d <= 0 ? 0.0 : is_inf(d) ? 1.0 :
        (brm_cd_G(d, fam, a, b) - brm_cd_G(d - w, fam, a, b)) / w
    brm_cd_S_pc(d::real, w::real, fam::int, a::real, b::real)::real =
        d <= 0 ? 1.0 : is_inf(d) ? 0.0 :
        (brm_cd_H(d - w, fam, a, b) - brm_cd_H(d, fam, a, b)) / w
    brm_cd_mass(l::real, u::real, w::real, fam::int, a::real, b::real)::real = begin
        lower_cdf = brm_cd_F_pc(l, w, fam, a, b)
        if lower_cdf <= 0.5
            return brm_cd_F_pc(u, w, fam, a, b) - lower_cdf
        end
        return brm_cd_S_pc(l, w, fam, a, b) - brm_cd_S_pc(u, w, fam, a, b)
    end
    brm_cd_density(d::real, w::real, fam::int, a::real, b::real)::real =
        d <= 0 ? 0.0 :
        brm_cd_F_base(d, fam, a, b) <= 0.5 ?
            (brm_cd_F_base(d, fam, a, b) -
             (d - w <= 0 ? 0.0 : brm_cd_F_base(d - w, fam, a, b))) / w :
            ((d - w <= 0 ? 1.0 : brm_cd_S_base(d - w, fam, a, b)) -
             brm_cd_S_base(d, fam, a, b)) / w

    brm_cd_logpdf(y::real, fam::int, p_lo::real, p_hi::real, trunc_lo::real,
            trunc_hi::real, interval::real, a::real, b::real)::real = begin
        w = p_hi - p_lo
        d = y - p_lo
        hi = interval > 0 ? d + interval : d
        lo_t = trunc_lo - p_lo
        hi_t = trunc_hi - p_lo
        numerator = interval > 0 ? brm_cd_mass(d, hi, w, fam, a, b) :
            brm_cd_density(d, w, fam, a, b)
        normaliser = lo_t <= 0 && is_inf(hi_t) ? 1.0 :
            brm_cd_mass(fmax(lo_t, 0.0), hi_t, w, fam, a, b)
        (y < trunc_lo || y + interval > trunc_hi || numerator <= 0 || normaliser <= 0) ?
            negative_infinity() : log(numerator) - log(normaliser)
    end
    # The family's scalar density (the Stan `_lpdf` name must return a real, so
    # the vectorised companions broadcast `brm_cd_logpdf`, not this name).
    @lpxf brm_censored_delay_lpdf(y::real, fam::int, p_lo::real, p_hi::real, trunc_lo::real,
            trunc_hi::real, interval::real, a::real, b::real)::real =
        brm_cd_logpdf(y, fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)
    brm_censored_delay_lpdf(y::anything[n], fam::int, p_lo, p_hi, trunc_lo, trunc_hi, interval,
                            a, b)::real =
        sum(brm_censored_delay_lpdfs(y, fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b))
    brm_censored_delay_lpdfs(y::real, fam::int, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)::real =
        brm_cd_logpdf(y, fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)
    brm_censored_delay_lpdfs(y::anything[n], fam::int, p_lo, p_hi, trunc_lo, trunc_hi, interval,
                             a, b)::vector[n] =
        jbroadcasted(brm_cd_logpdf, y, fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)
    # A draw of the truncated primary-censored delay (rejection), reported as
    # the start of its secondary window on the grid of width `interval`.
    brm_censored_delay_rng(fam::int, p_lo::real, p_hi::real, trunc_lo::real, trunc_hi::real,
                           interval::real, a::real, b::real)::real = begin
        x = 0.0
        found = 0
        tries = 0
        while found == 0
            tries += 1
            if tries > 1000000
                reject("brm_censored_delay_rng: no draw inside the truncation window in 10^6 tries")
            end
            x = uniform_rng(p_lo, p_hi) +
                (fam == 1 ? lognormal_rng(a, b) : fam == 2 ? gamma_rng(a, 1 / b) : weibull_rng(a, b))
            if interval > 0
                x = floor(x / interval) * interval
            end
            if x >= trunc_lo && x + interval <= trunc_hi
                found = 1
            end
        end
        x
    end
    brm_censored_delay_cell_rng(dummy, fam::int, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)::real =
        brm_censored_delay_rng(fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b)
    brm_censored_delay_rng(vector[n], fam::int, p_lo, p_hi, trunc_lo, trunc_hi, interval,
                           a, b)::vector[n] =
        jbroadcasted(brm_censored_delay_cell_rng, rep_vector(0., n), fam, p_lo, p_hi,
                     trunc_lo, trunc_hi, interval, a, b)
end

function _sb_censored_delay_likelihood!(stmts, target, f, args, kwargs::NamedTuple, data)
    plan = _brm_censored_delay_plan(f, args, kwargs; prefix="sbimpl")
    y = get(data, target, nothing)
    if !isnothing(y)
        values(x) = x isa Real ? x : _sb_bound_data(x, data)
        _brm_censored_delay_check(plan, target, _sb_composed_values(y), values; prefix="sbimpl")
    end
    bound(x, default) = isnothing(x) ? default : _sb_scalar_expr(x, data)
    rhs = Expr(:call, brm_censored_delay, plan.family,
        _sb_scalar_expr(plan.primary_lo, data), _sb_scalar_expr(plan.primary_hi, data),
        bound(plan.lower, :(negative_infinity())), bound(plan.upper, :(positive_infinity())),
        bound(plan.interval, 0.0),
        _sb_scalar_expr(plan.a, data), _sb_scalar_expr(plan.b, data))
    push!(stmts, Expr(:call, :~, target, rhs))
    nothing
end
_sb_lik_family!(stmts, target, f::typeof(primary_censored), args, kwargs::NamedTuple, data) =
    _sb_censored_delay_likelihood!(stmts, target, f, args, kwargs, data)
_sb_lik_family!(stmts, target, f::typeof(double_interval_censored), args,
                kwargs::NamedTuple, data) =
    _sb_censored_delay_likelihood!(stmts, target, f, args, kwargs, data)

# RKBRMI: the observation law of `brm_censored_delay`, the Julia methods of
# `censored_delays.jl` bound as one numerical leaf.
function _rk_observation_source!(definitions, bindings, entry, ::typeof(brm_censored_delay))
    law = Symbol(entry, :_law)
    push!(bindings, law => _brm_censored_delay_logpdf)
    push!(definitions, :(ReactiveKernels.@kernel function $entry(value, fam, p_lo, p_hi,
            lower, upper, interval, a, b)
        return $law(value, fam, p_lo, p_hi, lower, upper, interval, a, b)
    end))
    :done
end
