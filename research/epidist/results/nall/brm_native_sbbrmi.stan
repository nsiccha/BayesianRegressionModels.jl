functions {
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
real weighted_brm_censored_delay_lpdf(
    vector y,
    vector weight,
    int args1,
    int args2,
    vector args3,
    real args4,
    vector args5,
    vector args6,
    vector args7,
    vector args8
) {
    int n = dims(y)[1];
    if (dims(weight)[1] != n) reject("weighted_lpdf: dim mismatch — `weight` dim 1 (= ", dims(weight)[1], ") does not match `n` (= ", n, "), inferred from `y` dim 1. `n` sizes: `y` dim 1 (= ", dims(y)[1], "), `weight` dim 1 (= ", dims(weight)[1], ").");
    vector[dims(y)[1]] lp = brm_censored_delay_lpdfs(y, args1, args2, args3, args4, args5, args6, args7, args8);
    return sum((weight .* lp));
}
vector brm_censored_delay_lpdfs(
    vector y,
    int fam,
    int p_lo,
    vector p_hi,
    real trunc_lo,
    vector trunc_hi,
    vector interval,
    vector a,
    vector b
) {
    int n = dims(y)[1];
    return jbroadcasted_brm_cd_logpdf(y, fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b);
}
vector jbroadcasted_brm_cd_logpdf(
    vector x1,
    int x2,
    int x3,
    vector x4,
    real x5,
    vector x6,
    vector x7,
    vector x8,
    vector x9
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = brm_cd_logpdf(
            broadcasted_getindex(x1, i),
            x2,
            x3,
            broadcasted_getindex(x4, i),
            x5,
            broadcasted_getindex(x6, i),
            broadcasted_getindex(x7, i),
            broadcasted_getindex(x8, i),
            broadcasted_getindex(x9, i)
        );
    }
    return rv;
}
real brm_cd_logpdf(
    real y,
    int fam,
    int p_lo,
    real p_hi,
    real trunc_lo,
    real trunc_hi,
    real interval,
    real a,
    real b
) {
    real w = (p_hi - p_lo);
    real d = (y - p_lo);
    real hi = ((interval > 0) ? (d + interval) : d);
    real lo_t = (trunc_lo - p_lo);
    real hi_t = (trunc_hi - p_lo);
    real numerator = ((interval > 0) ? brm_cd_mass(d, hi, w, fam, a, b) : brm_cd_density(d, w, fam, a, b));
    real normaliser = (((lo_t <= 0) && is_inf(hi_t)) ? 1.0 : brm_cd_mass(fmax(lo_t, 0.0), hi_t, w, fam, a, b));
    return (((y < trunc_lo) || (((y + interval) > trunc_hi) || ((numerator <= 0) || (normaliser <= 0)))) ? negative_infinity() : (log(numerator) - log(normaliser)));
}
real brm_cd_mass(
    real l,
    real u,
    real w,
    int fam,
    real a,
    real b
) {
    real lower_cdf = brm_cd_F_pc(l, w, fam, a, b);
    if((lower_cdf <= 0.5)) {
        return (brm_cd_F_pc(u, w, fam, a, b) - lower_cdf);
    }
    return (brm_cd_S_pc(l, w, fam, a, b) - brm_cd_S_pc(u, w, fam, a, b));
}
real brm_cd_F_pc(
    real d,
    real w,
    int fam,
    real a,
    real b
) {
    return ((d <= 0) ? 0.0 : (is_inf(d) ? 1.0 : ((brm_cd_G(d, fam, a, b) - brm_cd_G((d - w), fam, a, b)) / w)));
}
real brm_cd_G(
    real q,
    int fam,
    real a,
    real b
) {
    return ((q <= 0) ? 0.0 : ((q * brm_cd_F_base(q, fam, a, b)) - brm_cd_lower_moment(q, fam, a, b)));
}
real brm_cd_F_base(
    real q,
    int fam,
    real a,
    real b
) {
    return ((fam == 1) ? (0.5 * erfc(((-((log(q) - a) / b)) / sqrt(2.0)))) : ((fam == 2) ? gamma_p(a, (q / b)) : (-expm1((-((q / b) ^ a))))));
}
real brm_cd_lower_moment(
    real q,
    int fam,
    real a,
    real b
) {
    return ((fam == 1) ? (exp((a + (0.5 * (b ^ 2)))) * 0.5 * erfc(((-(((log(q) - a) / b) - b)) / sqrt(2.0)))) : ((fam == 2) ? (a * b * gamma_p((a + 1), (q / b))) : (b * tgamma((1 + (1 / a))) * gamma_p((1 + (1 / a)), ((q / b) ^ a)))));
}
real brm_cd_S_pc(
    real d,
    real w,
    int fam,
    real a,
    real b
) {
    return ((d <= 0) ? 1.0 : (is_inf(d) ? 0.0 : ((brm_cd_H((d - w), fam, a, b) - brm_cd_H(d, fam, a, b)) / w)));
}
real brm_cd_H(
    real q,
    int fam,
    real a,
    real b
) {
    return ((q <= 0) ? (brm_cd_mean(fam, a, b) - q) : (brm_cd_upper_moment(q, fam, a, b) - (q * brm_cd_S_base(q, fam, a, b))));
}
real brm_cd_mean(
    int fam,
    real a,
    real b
) {
    return ((fam == 1) ? exp((a + (0.5 * (b ^ 2)))) : ((fam == 2) ? (a * b) : (b * tgamma((1 + (1 / a))))));
}
real brm_cd_upper_moment(
    real q,
    int fam,
    real a,
    real b
) {
    return ((fam == 1) ? (exp((a + (0.5 * (b ^ 2)))) * 0.5 * erfc(((((log(q) - a) / b) - b) / sqrt(2.0)))) : ((fam == 2) ? (a * b * (1 - gamma_p((a + 1), (q / b)))) : (b * tgamma((1 + (1 / a))) * (1 - gamma_p((1 + (1 / a)), ((q / b) ^ a))))));
}
real brm_cd_S_base(
    real q,
    int fam,
    real a,
    real b
) {
    return ((fam == 1) ? (0.5 * erfc((((log(q) - a) / b) / sqrt(2.0)))) : ((fam == 2) ? (1 - gamma_p(a, (q / b))) : exp((-((q / b) ^ a)))));
}
real brm_cd_density(
    real d,
    real w,
    int fam,
    real a,
    real b
) {
    return ((d <= 0) ? 0.0 : ((brm_cd_F_base(d, fam, a, b) <= 0.5) ? ((brm_cd_F_base(d, fam, a, b) - (((d - w) <= 0) ? 0.0 : brm_cd_F_base((d - w), fam, a, b))) / w) : (((((d - w) <= 0) ? 1.0 : brm_cd_S_base((d - w), fam, a, b)) - brm_cd_S_base(d, fam, a, b)) / w)));
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
vector weighted_brm_censored_delay_lpdfs(
    vector y,
    vector weight,
    int args1,
    int args2,
    vector args3,
    real args4,
    vector args5,
    vector args6,
    vector args7,
    vector args8
) {
    int n = dims(y)[1];
    if (dims(weight)[1] != n) reject("weighted_lpdfs: dim mismatch — `weight` dim 1 (= ", dims(weight)[1], ") does not match `n` (= ", n, "), inferred from `y` dim 1. `n` sizes: `y` dim 1 (= ", dims(y)[1], "), `weight` dim 1 (= ", dims(weight)[1], ").");
    vector[dims(y)[1]] lp = brm_censored_delay_lpdfs(y, args1, args2, args3, args4, args5, args6, args7, args8);
    return (weight .* lp);
}
vector weighted_vector_brm_censored_delay_rng(
    int anontok__1,
    vector weight,
    int args1,
    int args2,
    vector args3,
    real args4,
    vector args5,
    vector args6,
    vector args7,
    vector args8
) {
    int n = anontok__1;
    return brm_censored_delay_vector_rng(n, args1, args2, args3, args4, args5, args6, args7, args8);
}
vector brm_censored_delay_vector_rng(
    int anontok__1,
    int fam,
    int p_lo,
    vector p_hi,
    real trunc_lo,
    vector trunc_hi,
    vector interval,
    vector a,
    vector b
) {
    int n = anontok__1;
    return jbroadcasted_brm_censored_delay_cell_rng_rng(
        rep_vector(0.0, n),
        fam,
        p_lo,
        p_hi,
        trunc_lo,
        trunc_hi,
        interval,
        a,
        b
    );
}
vector jbroadcasted_brm_censored_delay_cell_rng_rng(
    vector x1,
    int x2,
    int x3,
    vector x4,
    real x5,
    vector x6,
    vector x7,
    vector x8,
    vector x9
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = brm_censored_delay_cell_rng(
            broadcasted_getindex(x1, i),
            x2,
            x3,
            broadcasted_getindex(x4, i),
            x5,
            broadcasted_getindex(x6, i),
            broadcasted_getindex(x7, i),
            broadcasted_getindex(x8, i),
            broadcasted_getindex(x9, i)
        );
    }
    return rv;
}
real brm_censored_delay_cell_rng(
    real dummy,
    int fam,
    int p_lo,
    real p_hi,
    real trunc_lo,
    real trunc_hi,
    real interval,
    real a,
    real b
) {
    return brm_censored_delay_rng(fam, p_lo, p_hi, trunc_lo, trunc_hi, interval, a, b);
}
real brm_censored_delay_rng(
    int fam,
    int p_lo,
    real p_hi,
    real trunc_lo,
    real trunc_hi,
    real interval,
    real a,
    real b
) {
    real x = 0.0;
    int found = 0;
    int tries = 0;
    while((found == 0)) {
        tries += 1;
        if((tries > 1000000)) {
            reject("brm_censored_delay_rng: no draw inside the truncation window in 10^6 tries");
        }
        x = (
            uniform_rng(p_lo, p_hi) +
            ((fam == 1) ? lognormal_rng(a, b) : ((fam == 2) ? gamma_rng(a, (1 / b)) : weibull_rng(a, b)))
        );
        if((interval > 0)) {
            x = (floor((x / interval)) * interval);
        }
        if(((x >= trunc_lo) && ((x + interval) <= trunc_hi))) {
            found = 1;
        }
    }
    return x;
}
}
data {
    int delay_lwr_n;
    vector[delay_lwr_n] delay_lwr;
    real prior_mu_location;
    real prior_mu_scale;
    real prior_sigma_location;
    real prior_sigma_scale;
    int brm_weight_delay_lwr_n;
    vector[brm_weight_delay_lwr_n] brm_weight_delay_lwr;
    int pwindow_n;
    vector[pwindow_n] pwindow;
    int relative_obs_time_n;
    vector[relative_obs_time_n] relative_obs_time;
    int swindow_n;
    vector[swindow_n] swindow;
}
transformed data {
    matrix[num_elements(delay_lwr), 1] X_mu = hcat(rep_vector(1.0, num_elements(delay_lwr)));
    int pop_mu_n_covariates = 1;
    matrix[num_elements(delay_lwr), 1] X_log_sigma = hcat(rep_vector(1.0, num_elements(delay_lwr)));
    int pop_log_sigma_n_covariates = 1;
}
parameters {
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
    vector[pop_log_sigma_n_covariates] pop_log_sigma_beta_pop;
}
transformed parameters {
    vector[num_elements(delay_lwr)] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[num_elements(delay_lwr)] mu = pop_mu;
    vector[num_elements(delay_lwr)] pop_log_sigma = (X_log_sigma * pop_log_sigma_beta_pop);
    vector[num_elements(delay_lwr)] log_sigma = pop_log_sigma;
    vector[num_elements(delay_lwr)] sigma = exp(log_sigma);
}
model {
    pop_mu_beta_pop ~ normal([prior_mu_location]', [prior_mu_scale]');
    pop_log_sigma_beta_pop ~ normal([prior_sigma_location]', [prior_sigma_scale]');
    delay_lwr ~ weighted_brm_censored_delay(
        brm_weight_delay_lwr,
        1,
        0,
        pwindow,
        negative_infinity(),
        relative_obs_time,
        swindow,
        mu,
        sigma
    );
}
generated quantities {
    vector[brm_weight_delay_lwr_n] delay_lwr_likelihood = weighted_brm_censored_delay_lpdfs(
        delay_lwr,
        brm_weight_delay_lwr,
        1,
        0,
        pwindow,
        negative_infinity(),
        relative_obs_time,
        swindow,
        mu,
        sigma
    );
    vector[delay_lwr_n] delay_lwr_gen = weighted_vector_brm_censored_delay_rng(
        delay_lwr_n,
        brm_weight_delay_lwr,
        1,
        0,
        pwindow,
        negative_infinity(),
        relative_obs_time,
        swindow,
        mu,
        sigma
    );
}