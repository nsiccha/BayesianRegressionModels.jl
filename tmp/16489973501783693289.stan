functions {
vector multi_normal_cholesky_vector_rng(
    int anontok__1,
    vector loc,
    matrix scale
) {
    int n = anontok__1;
    if (dims(loc)[1] != n) reject("multi_normal_cholesky_rng: dim mismatch — `loc` dim 1 (= ", dims(loc)[1], ") does not match `n` (= ", n, ")");
    return multi_normal_cholesky_rng(loc, scale);
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
    matrix[n_terms, n_groups] b_cols_bc;
}
transformed parameters {
    matrix[n_terms, n_groups] b_cols;
    for(plate_i__pl_1 in 1:n_groups) {
        b_cols[:, plate_i__pl_1] = b_cols_bc[:, plate_i__pl_1];
    }
    matrix[n_groups, n_terms] b = (b_cols');
    vector[group_idx_n] mu = rows_dot_product(Z, b[group_idx, :]);
}
model {
    L ~ lkj_corr_cholesky(1.0);
    tau ~ std_normal();
    for(plate_i__pl_1 in 1:n_groups) {
        b_cols_bc[:, plate_i__pl_1] ~ multi_normal_cholesky(rep_vector(0.0, n_terms), diag_pre_multiply(tau, L));
    }
    y ~ normal(mu, 1.0);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, mu, 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, mu, 1.0);
}