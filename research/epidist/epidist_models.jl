# epidist's marginal delay model (epinowcast/epidist 1.0.0, `as_epidist_marginal_model`)
# written three ways on this stack:
#
#   1. one `@brm` declaration, lowered by BOTH backends: SBBRMI (Stan) and RKBRMI (RKPPL);
#   2. a hand-written `@rkppl` program (RKPPL proper);
#   3. epidist's own brms-generated Stan program (the reference), read through BridgeStan.
#
# The likelihood (primarycensored, uniform primary event window, lognormal delay):
#
#   F_pc(d) = (1/w_P) ∫_{d-w_P}^{d} F(t) dt                     (uniform primary on [0, w_P])
#   p_i     = [F_pc(d_upr) - F_pc(d_lwr)] / F_pc(D)              (D: observation time relative to
#                                                                  the primary window start)
#
# with the closed form ∫_0^q F(t) dt = q Φ(z) - exp(μ + σ²/2) Φ(z - σ), z = (log q - μ)/σ.
# Aggregated rows enter with frequency weights n.
using BayesianRegressionModels, StanBlocks
using StatsFuns: normcdf

# ---------------------------------------------------------------------------
# Julia implementation (RKBRMI and the hand-written @rkppl program call it)
# ---------------------------------------------------------------------------
function pcens_lognormal_G(q, mu, sigma)
    T = promote_type(typeof(float(q)), typeof(mu), typeof(sigma))
    q <= 0 && return zero(T)
    z = (log(q) - mu) / sigma
    return q * normcdf(z) - exp(mu + sigma^2 / 2) * normcdf(z - sigma)
end
function pcens_lognormal_cdf(d, mu, sigma, pw)
    T = promote_type(typeof(float(d)), typeof(mu), typeof(sigma))
    isinf(d) && return one(T)
    return (pcens_lognormal_G(d, mu, sigma) - pcens_lognormal_G(d - pw, mu, sigma)) / pw
end
"Probability of one marginal row: delay in [l, u], primary window width pw, truncated at D."
epidist_marginal_p(l, u, pw, D, mu, sigma) =
    (pcens_lognormal_cdf(u, mu, sigma, pw) - pcens_lognormal_cdf(l, mu, sigma, pw)) /
    pcens_lognormal_cdf(D, mu, sigma, pw)
"Frequency-weighted log probability of one row (the hand-written @rkppl law)."
function epidist_marginal_wlpmf(l, mu, sigma, u, pw, D, n)
    p = epidist_marginal_p(l, u, pw, D, mu, sigma)
    return p > 0 ? n * log(p) : oftype(p, -Inf)   # the CDF difference can round to ≤ 0 deep in a tail
end

# ---------------------------------------------------------------------------
# Stan implementation (SBBRMI traces it), same name as the Julia methods below
# ---------------------------------------------------------------------------
StanBlocks.@deffun begin
    pcens_lognormal_G(q::real, mu::real, sigma::real)::real = begin
        if q <= 0
            return 0.0
        end
        z = (log(q) - mu) / sigma
        return q * Phi(z) - exp(mu + 0.5 * sigma^2) * Phi(z - sigma)
    end
    pcens_lognormal_cdf(d::real, mu::real, sigma::real, pw::real)::real = begin
        if is_inf(d)
            return 1.0
        end
        return (pcens_lognormal_G(d, mu, sigma) - pcens_lognormal_G(d - pw, mu, sigma)) / pw
    end
    epidist_marginal_prob(l::vector[n], u::vector[n], pw::vector[n], D::vector[n], mu::vector[n], sigma::vector[n])::vector[n] = begin
        p::vector[n]
        for i in 1:n
            p[i] = (pcens_lognormal_cdf(u[i], mu[i], sigma[i], pw[i]) - pcens_lognormal_cdf(l[i], mu[i], sigma[i], pw[i])) /
                   pcens_lognormal_cdf(D[i], mu[i], sigma[i], pw[i])
        end
        return p
    end
    epidist_marginal_prob(l::vector[n], u::vector[n], pw::vector[n], D::vector[n], mu::real, sigma::real)::vector[n] = begin
        p::vector[n]
        for i in 1:n
            p[i] = (pcens_lognormal_cdf(u[i], mu, sigma, pw[i]) - pcens_lognormal_cdf(l[i], mu, sigma, pw[i])) /
                   pcens_lognormal_cdf(D[i], mu, sigma, pw[i])
        end
        return p
    end
end
# RKBRMI runs the Julia methods of the same function object.
epidist_marginal_prob(l::AbstractVector, u::AbstractVector, pw::AbstractVector, D::AbstractVector, mu, sigma) =
    epidist_marginal_p.(l, u, pw, D, mu, sigma)

# ---------------------------------------------------------------------------
# 1. The BRM declaration. `mu ~ 1` / `log(sigma) ~ 1` are brms's `bf(mu ~ 1, sigma ~ 1)`;
#    covariates go on these lines exactly as in epidist. Today the bespoke likelihood
#    enters as a computed row probability p_i observed through a Binomial whose
#    response equals its trial count (`count = n`): log Binomial(n_i | n_i, p_i) = n_i log p_i,
#    the frequency-weighted row log probability. (`weighted(Bernoulli(p), fweights(n))`
#    lowers on SBBRMI but RKBRMI refuses weights on a response whose law reads an
#    assigned value: "value-based response weights need an authored response".)
# ---------------------------------------------------------------------------
epidist_brm = @brm begin
    mu ~ 1
    log(sigma) ~ 1
    effect(mu, Intercept) ~ Normal(prior_mu_location, prior_mu_scale)
    effect(sigma, Intercept) ~ Normal(prior_sigma_location, prior_sigma_scale)
    p = epidist_marginal_prob(delay_lwr, delay_upr, pwindow, relative_obs_time, mu, sigma)
    count ~ Binomial(n, p)
end

# ---------------------------------------------------------------------------
# 2. The hand-written RKPPL program
# ---------------------------------------------------------------------------
using ReactiveKernelsPPL
epidist_rkppl = @rkppl begin
    Intercept ~ Normal(prior_mu_location, prior_mu_scale)
    Intercept_sigma ~ Normal(prior_sigma_location, prior_sigma_scale)
    sigma = exp(Intercept_sigma)
    delay_lwr .~ LogDensity.(epidist_marginal_wlpmf, Intercept, sigma, delay_upr, pwindow, relative_obs_time, n)
end
