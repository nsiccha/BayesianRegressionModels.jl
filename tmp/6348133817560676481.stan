functions {
real multi_normal_cholesky0_lpdf(
    array[] vector x,
    matrix scale
) {
    int n = dims(x)[2];
    if (dims(scale)[1] != n) reject("multi_normal_cholesky0_lpdf: dim mismatch — `scale` dim 1 (= ", dims(scale)[1], ") does not match `n` (= ", n, ")");
    if (dims(scale)[2] != n) reject("multi_normal_cholesky0_lpdf: dim mismatch — `scale` dim 2 (= ", dims(scale)[2], ") does not match `n` (= ", n, ")");
    return multi_normal_cholesky_lpdf(x | rep_vector(0.0, n), scale);
}
matrix ranef_b_matrix(
    array[] vector b
) {
    int m = dims(b)[1];
    int n = dims(b)[2];
    matrix[m, n] rv;
    for(i in 1:m) {
        rv[i, :] = (b[i]');
    }
    return rv;
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
    int n_terms;
    int n_groups;
    int group_idx_n;
    int Z_m;
    int Z_n;
    matrix[Z_m, Z_n] Z;
    array[group_idx_n] int group_idx;
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
    cholesky_factor_corr[n_terms] L;
    vector<lower=0.0>[n_terms] tau;
    array[n_groups] vector[n_terms] b;
}
transformed parameters {
    matrix[n_groups, n_terms] bm = ranef_b_matrix(b);
    vector[group_idx_n] mu = rows_dot_product(Z, bm[group_idx, :]);
}
model {
    L ~ lkj_corr_cholesky(1.0);
    tau ~ std_normal();
    b ~ multi_normal_cholesky0(diag_pre_multiply(tau, L));
    y ~ normal(mu, 1.0);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, mu, 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, mu, 1.0);
}