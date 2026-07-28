functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
matrix hcat(vector x, vector y, vector z) {
    return hcat(hcat(x, y), z);
}
matrix hcat(
    matrix x,
    vector y
) {
    int m = dims(x)[1];
    int n = dims(x)[2];
    if (dims(y)[1] != m) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `m` (= ", m, ")");
    return append_col(x, y);
}
matrix hcat(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return append_col(x, y);
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
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
    int n_terms_p_subject;
    int n_subject;
    int diseased_n;
    int male_n;
    vector[male_n] male;
    int zage_n;
    vector[zage_n] zage;
    vector[diseased_n] diseased;
    int subject_idx_n;
    array[subject_idx_n] int subject_idx;
    int y_n;
    vector[y_n] y;
}
transformed data {
    matrix[diseased_n, (2 + 1)] X_eta_CL = hcat(male, zage, diseased);
    int pop_eta_CL_n_covariates = (2 + 1);
    matrix[diseased_n, (2 + 1)] X_eta_Vc = hcat(male, zage, diseased);
    int pop_eta_Vc_n_covariates = (2 + 1);
    matrix[diseased_n, 1] X_eta_Q = hcat(diseased);
    int pop_eta_Q_n_covariates = 1;
}
parameters {
    cholesky_factor_corr[n_terms_p_subject] b_p_subject_L;
    vector<lower=0.0>[n_terms_p_subject] b_p_subject_tau;
    matrix[n_terms_p_subject, n_subject] b_p_subject_b_cols_z;
    vector[pop_eta_CL_n_covariates] pop_eta_CL_beta_pop;
    vector[pop_eta_Vc_n_covariates] pop_eta_Vc_beta_pop;
    vector[pop_eta_Q_n_covariates] pop_eta_Q_beta_pop;
}
transformed parameters {
    matrix[n_terms_p_subject, n_subject] b_p_subject_b_cols;
    for(b_p_subject_plate_i__pl_1 in 1:n_subject) {
        b_p_subject_b_cols[:, b_p_subject_plate_i__pl_1] = (
            diag_pre_multiply(b_p_subject_tau, b_p_subject_L) *
            b_p_subject_b_cols_z[:, b_p_subject_plate_i__pl_1]
        );
    }
    matrix[n_subject, n_terms_p_subject] b_p_subject = (b_p_subject_b_cols');
    vector[diseased_n] pop_eta_CL = (X_eta_CL * pop_eta_CL_beta_pop);
    vector[subject_idx_n] r_eta_CL_p_subject = b_p_subject[subject_idx, 1];
    vector[diseased_n] eta_CL = (pop_eta_CL + r_eta_CL_p_subject);
    vector[diseased_n] pop_eta_Vc = (X_eta_Vc * pop_eta_Vc_beta_pop);
    vector[subject_idx_n] r_eta_Vc_p_subject = b_p_subject[subject_idx, 2];
    vector[diseased_n] eta_Vc = (pop_eta_Vc + r_eta_Vc_p_subject);
    vector[diseased_n] pop_eta_Q = (X_eta_Q * pop_eta_Q_beta_pop);
    vector[subject_idx_n] r_eta_Q_p_subject = b_p_subject[subject_idx, 3];
    vector[diseased_n] eta_Q = (pop_eta_Q + r_eta_Q_p_subject);
    vector[subject_idx_n] r_eta_ka_p_subject = b_p_subject[subject_idx, 4];
    vector[subject_idx_n] eta_ka = r_eta_ka_p_subject;
}
model {
    b_p_subject_L ~ lkj_corr_cholesky(1.0);
    b_p_subject_tau ~ std_normal();
    for(b_p_subject_plate_i__pl_1 in 1:n_subject) {
        b_p_subject_b_cols_z[:, b_p_subject_plate_i__pl_1] ~ std_normal();
    }
    pop_eta_CL_beta_pop ~ std_normal();
    pop_eta_Vc_beta_pop ~ std_normal();
    pop_eta_Q_beta_pop ~ std_normal();
    y ~ normal((eta_CL + eta_Vc + eta_Q + eta_ka), 1.0);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, (eta_CL + eta_Vc + eta_Q + eta_ka), 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, (eta_CL + eta_Vc + eta_Q + eta_ka), 1.0);
}