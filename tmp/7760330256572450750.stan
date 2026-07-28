functions {
matrix reshape(vector v, int m, int k) {
    return to_matrix(v, m, k);
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
}
data {
    int n_terms_p_subject;
    int n_subject;
    int subject_idx_n;
    array[subject_idx_n] int subject_idx;
    int kernel_nsub_pred;
    int dose_n;
    vector[dose_n] dose;
    int dv_n;
    vector[dv_n] dv;
}
transformed data {
    matrix[num_elements(subject_idx), 1] X_log_scale = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_scale_n_covariates = 1;
}
parameters {
    cholesky_factor_corr[n_terms_p_subject] b_p_subject_L;
    vector<lower=0.0>[n_terms_p_subject] b_p_subject_tau;
    vector[(n_terms_p_subject * n_subject)] b_p_subject_z_flat;
    real<lower=0.0> sigma;
    vector[pop_log_scale_n_covariates] pop_log_scale_beta_pop;
}
transformed parameters {
    matrix[n_terms_p_subject, n_subject] b_p_subject_z = reshape(b_p_subject_z_flat, n_terms_p_subject, n_subject);
    matrix[n_subject, n_terms_p_subject] b_p_subject = ((diag_pre_multiply(b_p_subject_tau, b_p_subject_L) * b_p_subject_z)');
    vector[num_elements(subject_idx)] pop_log_scale = (X_log_scale * pop_log_scale_beta_pop);
    vector[subject_idx_n] r_log_scale_p_subject = b_p_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_scale = (pop_log_scale + r_log_scale_p_subject);
    vector[kernel_nsub_pred] pred;
    vector[kernel_nsub_pred] pred_mu;
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        pred_mu[plate_i__pl_1] = ((dose[plate_i__pl_1] / 10.0) * exp(log_scale[plate_i__pl_1]));
        pred[plate_i__pl_1] = pred_mu[plate_i__pl_1];
    }
}
model {
    b_p_subject_L ~ lkj_corr_cholesky(1.0);
    b_p_subject_tau ~ std_normal();
    b_p_subject_z_flat ~ std_normal();
    sigma ~ exponential(1);
    pop_log_scale_beta_pop ~ std_normal();
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        dv[plate_i__pl_1] ~ normal(pred_mu[plate_i__pl_1], sigma);
    }
}
generated quantities {
    vector[dv_n] dv_gen;
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        dv_gen[plate_i__pl_1] = normal_rng(pred_mu[plate_i__pl_1], sigma);
    }
}