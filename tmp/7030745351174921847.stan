functions {
matrix hcat(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return append_col(x, y);
}
vector normal_lpdfs(
    vector obs,
    vector loc,
    real scale
) {
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    vector x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = normal_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector normal_vector_rng(
    int anontok__1,
    vector a,
    real b
) {
    int n = anontok__1;
    return to_vector(normal_rng(a, b));
}
}
data {
    int x_n;
    vector[x_n] x;
    int n_g;
    int g_idx_n;
    array[g_idx_n] int g_idx;
    int y_n;
    vector[y_n] y;
    int z_n;
    vector[z_n] z;
}
transformed data {
    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);
    int pop_mu_n_covariates = 2;
}
parameters {
    real<lower=0.0> sigma;
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
    real r_mu_g_log_scale;
    vector[n_g] r_mu_g_xi;
}
transformed parameters {
    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[g_idx_n] r_mu_g = (exp(r_mu_g_log_scale) * r_mu_g_xi[g_idx]);
    vector[x_n] mu = (pop_mu + r_mu_g);
}
model {
    sigma ~ exponential(1);
    pop_mu_beta_pop ~ std_normal();
    r_mu_g_log_scale ~ std_normal();
    r_mu_g_xi ~ std_normal();
    y ~ normal(mu, sigma);
    z ~ normal(mu, (2 .* sigma));
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, mu, sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, mu, sigma);
    vector[z_n] z_likelihood = normal_lpdfs(z, mu, (2 .* sigma));
    vector[z_n] z_gen = normal_vector_rng(z_n, mu, (2 .* sigma));
}