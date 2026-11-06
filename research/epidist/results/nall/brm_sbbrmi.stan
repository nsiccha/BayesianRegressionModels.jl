functions {
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
vector epidist_marginal_prob(
    vector l,
    vector u,
    vector pw,
    vector D,
    vector mu,
    vector sigma
) {
    int n = dims(l)[1];
    if (dims(u)[1] != n) reject("epidist_marginal_prob: dim mismatch — `u` dim 1 (= ", dims(u)[1], ") does not match `n` (= ", n, "), inferred from `l` dim 1. `n` sizes: `l` dim 1 (= ", dims(l)[1], "), `u` dim 1 (= ", dims(u)[1], "), `pw` dim 1 (= ", dims(pw)[1], "), `D` dim 1 (= ", dims(D)[1], "), `mu` dim 1 (= ", dims(mu)[1], "), `sigma` dim 1 (= ", dims(sigma)[1], ").");
    if (dims(pw)[1] != n) reject("epidist_marginal_prob: dim mismatch — `pw` dim 1 (= ", dims(pw)[1], ") does not match `n` (= ", n, "), inferred from `l` dim 1. `n` sizes: `l` dim 1 (= ", dims(l)[1], "), `u` dim 1 (= ", dims(u)[1], "), `pw` dim 1 (= ", dims(pw)[1], "), `D` dim 1 (= ", dims(D)[1], "), `mu` dim 1 (= ", dims(mu)[1], "), `sigma` dim 1 (= ", dims(sigma)[1], ").");
    if (dims(D)[1] != n) reject("epidist_marginal_prob: dim mismatch — `D` dim 1 (= ", dims(D)[1], ") does not match `n` (= ", n, "), inferred from `l` dim 1. `n` sizes: `l` dim 1 (= ", dims(l)[1], "), `u` dim 1 (= ", dims(u)[1], "), `pw` dim 1 (= ", dims(pw)[1], "), `D` dim 1 (= ", dims(D)[1], "), `mu` dim 1 (= ", dims(mu)[1], "), `sigma` dim 1 (= ", dims(sigma)[1], ").");
    if (dims(mu)[1] != n) reject("epidist_marginal_prob: dim mismatch — `mu` dim 1 (= ", dims(mu)[1], ") does not match `n` (= ", n, "), inferred from `l` dim 1. `n` sizes: `l` dim 1 (= ", dims(l)[1], "), `u` dim 1 (= ", dims(u)[1], "), `pw` dim 1 (= ", dims(pw)[1], "), `D` dim 1 (= ", dims(D)[1], "), `mu` dim 1 (= ", dims(mu)[1], "), `sigma` dim 1 (= ", dims(sigma)[1], ").");
    if (dims(sigma)[1] != n) reject("epidist_marginal_prob: dim mismatch — `sigma` dim 1 (= ", dims(sigma)[1], ") does not match `n` (= ", n, "), inferred from `l` dim 1. `n` sizes: `l` dim 1 (= ", dims(l)[1], "), `u` dim 1 (= ", dims(u)[1], "), `pw` dim 1 (= ", dims(pw)[1], "), `D` dim 1 (= ", dims(D)[1], "), `mu` dim 1 (= ", dims(mu)[1], "), `sigma` dim 1 (= ", dims(sigma)[1], ").");
    vector[n] p;
    for(i in 1:n) {
        p[i] = (
            (
                pcens_lognormal_cdf(u[i] | mu[i], sigma[i], pw[i]) -
                pcens_lognormal_cdf(l[i] | mu[i], sigma[i], pw[i])
            ) /
            pcens_lognormal_cdf(D[i] | mu[i], sigma[i], pw[i])
        );
    }
    return p;
}
real pcens_lognormal_cdf(
    real d,
    real mu,
    real sigma,
    real pw
) {
    if(is_inf(d)) {
        return 1.0;
    }
    return ((pcens_lognormal_G(d, mu, sigma) - pcens_lognormal_G((d - pw), mu, sigma)) / pw);
}
real pcens_lognormal_G(
    real q,
    real mu,
    real sigma
) {
    if((q <= 0)) {
        return 0.0;
    }
    real z = ((log(q) - mu) / sigma);
    return ((q * Phi(z)) - (exp((mu + (0.5 * (sigma ^ 2)))) * Phi((z - sigma))));
}
vector binomial_lpmfs(
    array[] int y,
    array[] int args1,
    vector args2
) {
    return jbroadcasted_binomial_lpmfs(y, args1, args2);
}
vector jbroadcasted_binomial_lpmfs(
    array[] int x1,
    array[] int x2,
    vector x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = binomial_lpmfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i)
        );
    }
    return rv;
}
real binomial_lpmfs(
    int args1,
    int args2,
    real args3
) {
    return binomial_lpmf(args1 | args2, args3);
}
int broadcasted_getindex(array[] int x, int i) {
    return x[i];
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
array[] int binomial_int_rng(
    int anontok__1,
    array[] int N,
    vector p
) {
    int n = anontok__1;
    if (dims(N)[1] != n) reject("binomial_rng: dim mismatch — `N` dim 1 (= ", dims(N)[1], ") does not match `n` (= ", n, "), inferred from `anontok__1` dim 1. `n` sizes: `anontok__1` dim 1 (= ", anontok__1, "), `N` dim 1 (= ", dims(N)[1], ").");
    if((n == 0)) {
        array[n] int rv;
        return rv;
    } else {
        return binomial_rng(N, p);
    }
}
}
data {
    int count_n;
    array[count_n] int count;
    real prior_mu_location;
    real prior_mu_scale;
    real prior_sigma_location;
    real prior_sigma_scale;
    int delay_lwr_n;
    vector[delay_lwr_n] delay_lwr;
    int delay_upr_n;
    vector[delay_upr_n] delay_upr;
    int pwindow_n;
    vector[pwindow_n] pwindow;
    int relative_obs_time_n;
    vector[relative_obs_time_n] relative_obs_time;
    int n_n;
    array[n_n] int n;
}
transformed data {
    matrix[num_elements(count), 1] X_mu = hcat(rep_vector(1.0, num_elements(count)));
    int pop_mu_n_covariates = 1;
    matrix[num_elements(count), 1] X_log_sigma = hcat(rep_vector(1.0, num_elements(count)));
    int pop_log_sigma_n_covariates = 1;
}
parameters {
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
    vector[pop_log_sigma_n_covariates] pop_log_sigma_beta_pop;
}
transformed parameters {
    vector[num_elements(count)] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[num_elements(count)] mu = pop_mu;
    vector[num_elements(count)] pop_log_sigma = (X_log_sigma * pop_log_sigma_beta_pop);
    vector[num_elements(count)] log_sigma = pop_log_sigma;
    vector[num_elements(count)] sigma = exp(log_sigma);
    vector[num_elements(count)] p = epidist_marginal_prob(delay_lwr, delay_upr, pwindow, relative_obs_time, mu, sigma);
}
model {
    pop_mu_beta_pop ~ normal([prior_mu_location]', [prior_mu_scale]');
    pop_log_sigma_beta_pop ~ normal([prior_sigma_location]', [prior_sigma_scale]');
    count ~ binomial(n, p);
}
generated quantities {
    vector[count_n] count_likelihood = binomial_lpmfs(count, n, p);
    array[n_n] int count_gen = binomial_int_rng(count_n, n, p);
}