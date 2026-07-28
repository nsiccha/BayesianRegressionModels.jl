functions {
matrix reshape(vector v, int m, int k) {
    return to_matrix(v, m, k);
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
int ragged_end(array[] int ends, int i) {
    return ends[i];
}
int ragged_start(
    array[] int ends,
    int i
) {
    if((i == 1)) {
        return 1;
    } else {
        return (1 + ends[(i - 1)]);
    }
}
vector normal_lpdfs(
    vector obs,
    vector loc,
    vector scale
) {
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    vector x2,
    vector x3
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
vector normal_vector_rng(
    int anontok__1,
    vector a,
    vector b
) {
    int n = anontok__1;
    return to_vector(normal_rng(a, b));
}
vector addprop(
    vector loc,
    real add,
    real prop
) {
    int n = dims(loc)[1];
    return sqrt(((add ^ 2) + ((loc .* prop) .^ 2)));
}
}
data {
    int n_terms_p_subject;
    int n_subject;
    int subject_idx_n;
    array[subject_idx_n] int subject_idx;
    int kernel_nsub_pred;
    int t_ends_n;
    int t_mem_n;
    tuple(vector[t_mem_n], array[t_ends_n] int) t;
    int dose_n;
    vector[dose_n] dose;
    int dv_ends_n;
    int dv_mem_n;
    tuple(vector[dv_mem_n], array[dv_ends_n] int) dv;
}
transformed data {
    matrix[num_elements(subject_idx), 1] X_log_CL = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_CL_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_V = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_V_n_covariates = 1;
    array[kernel_nsub_pred] int pred__pl_len_1;
    array[kernel_nsub_pred] int pred_mu__pl_len_1;
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        pred__pl_len_1[plate_i__pl_1] = (1 + (ragged_end(t.2, plate_i__pl_1) - ragged_start(t.2, plate_i__pl_1)));
        pred_mu__pl_len_1[plate_i__pl_1] = (1 + (ragged_end(t.2, plate_i__pl_1) - ragged_start(t.2, plate_i__pl_1)));
    }
    array[kernel_nsub_pred] int pred__pl_end_1 = cumulative_sum(pred__pl_len_1);
    array[kernel_nsub_pred] int pred_mu__pl_end_1 = cumulative_sum(pred_mu__pl_len_1);
}
parameters {
    cholesky_factor_corr[n_terms_p_subject] b_p_subject_L;
    vector<lower=0.0>[n_terms_p_subject] b_p_subject_tau;
    vector[(n_terms_p_subject * n_subject)] b_p_subject_z_flat;
    real<lower=0.0> sigma_a;
    real<lower=0.0> sigma_p;
    vector[pop_log_CL_n_covariates] pop_log_CL_beta_pop;
    vector[pop_log_V_n_covariates] pop_log_V_beta_pop;
}
transformed parameters {
    matrix[n_terms_p_subject, n_subject] b_p_subject_z = reshape(b_p_subject_z_flat, n_terms_p_subject, n_subject);
    matrix[n_subject, n_terms_p_subject] b_p_subject = ((diag_pre_multiply(b_p_subject_tau, b_p_subject_L) * b_p_subject_z)');
    vector[num_elements(subject_idx)] pop_log_CL = (X_log_CL * pop_log_CL_beta_pop);
    vector[subject_idx_n] r_log_CL_p_subject = b_p_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_CL = (pop_log_CL + r_log_CL_p_subject);
    vector[num_elements(subject_idx)] pop_log_V = (X_log_V * pop_log_V_beta_pop);
    vector[subject_idx_n] r_log_V_p_subject = b_p_subject[subject_idx, 2];
    vector[num_elements(subject_idx)] log_V = (pop_log_V + r_log_V_p_subject);
    vector[sum(pred__pl_len_1)] pred__pl_mem_1;
    vector[sum(pred_mu__pl_len_1)] pred_mu__pl_mem_1;
    vector[kernel_nsub_pred] pred_CL;
    vector[kernel_nsub_pred] pred_V;
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        pred_CL[plate_i__pl_1] = exp(log_CL[plate_i__pl_1]);
        pred_V[plate_i__pl_1] = exp(log_V[plate_i__pl_1]);
        pred_mu__pl_mem_1[
            ragged_start(pred_mu__pl_end_1, plate_i__pl_1):ragged_end(pred_mu__pl_end_1, plate_i__pl_1)
        ] = (
            (dose[plate_i__pl_1] / pred_V[plate_i__pl_1]) *
            exp(
                (
                    (-(pred_CL[plate_i__pl_1] / pred_V[plate_i__pl_1])) *
                    t.1[ragged_start(t.2, plate_i__pl_1):ragged_end(t.2, plate_i__pl_1)]
                )
            )
        );
        pred__pl_mem_1[
            ragged_start(pred__pl_end_1, plate_i__pl_1):ragged_end(pred__pl_end_1, plate_i__pl_1)
        ] = pred_mu__pl_mem_1[
            ragged_start(pred_mu__pl_end_1, plate_i__pl_1):ragged_end(pred_mu__pl_end_1, plate_i__pl_1)
        ];
    }
}
model {
    b_p_subject_L ~ lkj_corr_cholesky(1.0);
    b_p_subject_tau ~ std_normal();
    b_p_subject_z_flat ~ std_normal();
    sigma_a ~ exponential(1);
    sigma_p ~ exponential(1);
    pop_log_CL_beta_pop ~ std_normal();
    pop_log_V_beta_pop ~ std_normal();
    for(plate_i__pl_1 in 1:kernel_nsub_pred) {
        dv.1[ragged_start(dv.2, plate_i__pl_1):ragged_end(dv.2, plate_i__pl_1)] ~ normal(
            pred_mu__pl_mem_1[
                ragged_start(pred_mu__pl_end_1, plate_i__pl_1):ragged_end(pred_mu__pl_end_1, plate_i__pl_1)
            ],
            addprop(
                pred_mu__pl_mem_1[
                    ragged_start(pred_mu__pl_end_1, plate_i__pl_1):ragged_end(pred_mu__pl_end_1, plate_i__pl_1)
                ],
                sigma_a,
                sigma_p
            )
        );
    }
}
generated quantities {
}