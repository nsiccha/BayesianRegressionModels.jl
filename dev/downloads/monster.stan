functions {
// value UDF brm_vector_prior_180e10f5bbee3641_lpdf
real brm_vector_prior_180e10f5bbee3641_lpdf(
    vector x,
    real arg_1,
    real arg_2
) {
    if((x[1] < 0.0)) {
        return negative_infinity();
    }
    return scaled_inv_chi_scale_lpdf(x[1] | arg_1, arg_2);
}
real scaled_inv_chi_scale_lpdf(
    real tau,
    real nu,
    real s
) {
    return (scaled_inv_chi_square_lpdf(square(tau) | nu, s) + log((2.0 * tau)));
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
vector monster_experiment(
    vector times,
    real concentration_exposure,
    real lean_body_mass,
    real mass_fraction_fat,
    real volume_flow_pulmonary,
    real VPR,
    vector unit_volume_flow,
    vector volume_fraction,
    real partition_coefficient_alveolar,
    vector partition_coefficient,
    real VMI_per_kg,
    real KMI_per_l,
    int no_sub_steps
) {
    int n_time = dims(times)[1];
    int no_exposure_times = 1;
    real body_mass = (lean_body_mass / (1 - mass_fraction_fat));
    real volume_fat = ((mass_fraction_fat * body_mass) / 0.92);
    real volume_flow_alveolar = (0.7 * volume_flow_pulmonary);
    real mass_flow_exposure = (volume_flow_alveolar * concentration_exposure);
    real volume_flow_venous = (volume_flow_alveolar / VPR);
    vector[dims(unit_volume_flow)[1]] volume_flow = (unit_volume_flow * volume_flow_venous);
    vector[4] volume;
    volume[1] = (lean_body_mass * volume_fraction[1]);
    volume[2] = (lean_body_mass * volume_fraction[2]);
    volume[3] = volume_fat;
    volume[4] = (lean_body_mass * volume_fraction[3]);
    vector[dims(partition_coefficient)[1]] effective_volume = (volume .* partition_coefficient);
    real VMI = (((lean_body_mass ^ 0.7) * VMI_per_kg) / effective_volume[4]);
    real KMI = (KMI_per_l / effective_volume[4]);
    vector[4] concentration_out = rep_vector(monster_min_concentration(), 4);
    vector[dims(partition_coefficient)[1]] FVP = (volume_flow ./ effective_volume);
    real FPF = (volume_flow_venous + (volume_flow_alveolar / partition_coefficient_alveolar));
    matrix[dims(partition_coefficient)[1], dims(partition_coefficient)[1]] A = add_diag((FVP * ((volume_flow / FPF)')), (-FVP));
    vector[dims(partition_coefficient)[1]] A_source = ((mass_flow_exposure / FPF) * (A \ FVP));
    vector[4] last_concentration_out = concentration_out;
    real dt = (times[1] / no_sub_steps);
    matrix[dims(partition_coefficient)[1], dims(partition_coefficient)[1]] transition_matrix = matrix_exp((dt * A));
    vector[dims(partition_coefficient)[1]] exp_A_source = ((transition_matrix * A_source) - A_source);
    array[n_time] vector[4] all_concentration_out;
    real last_time = 0.0;
    real next_time = 0.0;
    int time_idx = 1;
    real next_checkpoint = times[time_idx];
    while((time_idx <= n_time)) {
        next_time = (last_time + dt);
        concentration_out[4] = monster_michaelis_menten_step((dt / 2), concentration_out[4], (-VMI), KMI);
        if((time_idx <= no_exposure_times)) {
            concentration_out = ((transition_matrix * concentration_out) + exp_A_source);
        } else {
            concentration_out = (transition_matrix * concentration_out);
        }
        concentration_out[4] = monster_michaelis_menten_step((dt / 2), concentration_out[4], (-VMI), KMI);
        while((next_time >= next_checkpoint)) {
            all_concentration_out[time_idx] = monster_interpolate(((next_checkpoint - last_time) / dt), last_concentration_out, concentration_out);
            if((time_idx == no_exposure_times)) {
                concentration_out = all_concentration_out[time_idx];
                next_time = times[time_idx];
            }
            time_idx += 1;
            if((time_idx <= n_time)) {
                next_checkpoint = times[time_idx];
            } else {
                break;
            }
        }
        last_time = next_time;
        last_concentration_out = concentration_out;
    }
    vector[(n_time + n_time)] prediction;
    for(t in 1:n_time) {
        real concentration_venous = dot_product(unit_volume_flow, all_concentration_out[t]);
        real concentration_inhale = ((t <= no_exposure_times) ? concentration_exposure : 0.0);
        real concentration_alveolar = ((concentration_inhale + concentration_venous) / (VPR + partition_coefficient_alveolar));
        real concentration_exhale = ((0.7 * concentration_alveolar) + (0.3 * concentration_inhale));
        prediction[t] = (monster_min_concentration() + concentration_venous);
        prediction[(n_time + t)] = (monster_min_concentration() + concentration_exhale);
    }
    return prediction;
}
real monster_min_concentration() {
    return 1.0e-12;
}
real monster_michaelis_menten_step(
    real dt,
    real C,
    real V,
    real K
) {
    if((C <= monster_min_concentration())) {
        return monster_min_concentration();
    }
    if((K == 0)) {
        return (C - (dt * V));
    }
    real earg = ((((dt * V) + C) / K) + log((C / K)));
    return (K * monster_lambert_w0_exp(earg));
}
real monster_lambert_w0_exp(
    real earg
) {
    if(is_nan(earg)) {
        return earg;
    }
    if((earg > 700)) {
        real y0 = lambert_w0(exp(700.0));
        return (y0 + (((earg - 700.0) * y0) / (y0 + 1)));
    }
    if((earg < -40)) {
        return (lambert_w0(exp(-40.0)) * exp((earg + 40.0)));
    }
    return lambert_w0(exp(earg));
}
vector monster_interpolate(
    real xi,
    vector left,
    vector right
) {
    int n = dims(left)[1];
    if (dims(right)[1] != n) reject("monster_interpolate: dim mismatch — `right` dim 1 (= ", dims(right)[1], ") does not match `n` (= ", n, "), inferred from `left` dim 1. `n` sizes: `left` dim 1 (= ", dims(left)[1], "), `right` dim 1 (= ", dims(right)[1], ").");
    return exp(
        (
            ((1 - xi) * log((monster_min_concentration() + left))) +
            (xi * log((monster_min_concentration() + right)))
        )
    );
}
vector lognormal_lpdfs(
    vector obs,
    vector loc,
    real scale
) {
    return jbroadcasted_lognormal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_lognormal_lpdfs(
    vector x1,
    vector x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = lognormal_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), x3);
    }
    return rv;
}
real lognormal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return lognormal_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    return x[i];
}
vector lognormal_vector_rng(
    int anontok__1,
    vector a,
    real b
) {
    int n = anontok__1;
    if((n == 0)) {
        vector[n] rv;
        return rv;
    } else {
        return to_vector(lognormal_rng(a, b));
    }
}
}
data {
    int n_terms_VPR_subject;
    int n_subject;
    int n_terms_Fwp_subject;
    int n_terms_Fpp_subject;
    int n_terms_Ff_subject;
    int n_terms_Fl_subject;
    int n_terms_Vwp_subject;
    int n_terms_Vpp_subject;
    int n_terms_Vl_subject;
    int n_terms_Pba_subject;
    int n_terms_Pwp_subject;
    int n_terms_Ppp_subject;
    int n_terms_Pf_subject;
    int n_terms_Pl_subject;
    int n_terms_VMI_subject;
    int n_terms_KMI_subject;
    int subject_idx_n;
    array[subject_idx_n] int subject_idx;
    int kernel_nsub_pred_144;
    int times_72_ends_n;
    int times_72_mem_n;
    tuple(vector[times_72_mem_n], array[times_72_ends_n] int) times_72;
    int times_144_ends_n;
    int times_144_mem_n;
    tuple(vector[times_144_mem_n], array[times_144_ends_n] int) times_144;
    int venous_72_mem_n;
    int venous_72_ends_n;
    tuple(vector[venous_72_mem_n], array[venous_72_ends_n] int) venous_72;
    int exhaled_72_mem_n;
    int exhaled_72_ends_n;
    tuple(vector[exhaled_72_mem_n], array[exhaled_72_ends_n] int) exhaled_72;
    int venous_144_mem_n;
    int venous_144_ends_n;
    tuple(vector[venous_144_mem_n], array[venous_144_ends_n] int) venous_144;
    int exhaled_144_mem_n;
    int exhaled_144_ends_n;
    tuple(vector[exhaled_144_mem_n], array[exhaled_144_ends_n] int) exhaled_144;
    real exposure_72;
    int lean_body_mass_n;
    vector[lean_body_mass_n] lean_body_mass;
    int fat_fraction_n;
    vector[fat_fraction_n] fat_fraction;
    int pulmonary_flow_n;
    vector[pulmonary_flow_n] pulmonary_flow;
    int n_substeps;
    real exposure_144;
    int venous_72_index_ends_n;
    int venous_72_index_mem_n;
    tuple(array[venous_72_index_mem_n] int, array[venous_72_index_ends_n] int) venous_72_index;
    int exhaled_72_index_ends_n;
    int exhaled_72_index_mem_n;
    tuple(array[exhaled_72_index_mem_n] int, array[exhaled_72_index_ends_n] int) exhaled_72_index;
    int venous_144_index_ends_n;
    int venous_144_index_mem_n;
    tuple(array[venous_144_index_mem_n] int, array[venous_144_index_ends_n] int) venous_144_index;
    int exhaled_144_index_ends_n;
    int exhaled_144_index_mem_n;
    tuple(array[exhaled_144_index_mem_n] int, array[exhaled_144_index_ends_n] int) exhaled_144_index;
}
transformed data {
    matrix[num_elements(subject_idx), 1] X_log_VPR = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_VPR_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Fwp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Fwp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Fpp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Fpp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Ff = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Ff_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Fl = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Fl_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Vwp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Vwp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Vpp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Vpp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Vl = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Vl_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Pba = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Pba_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Pwp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Pwp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Ppp = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Ppp_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Pf = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Pf_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_Pl = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_Pl_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_VMI = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_VMI_n_covariates = 1;
    matrix[num_elements(subject_idx), 1] X_log_KMI = hcat(rep_vector(1.0, num_elements(subject_idx)));
    int pop_log_KMI_n_covariates = 1;
    array[kernel_nsub_pred_144] int pred_144_pred_72__pl_len_1;
    array[kernel_nsub_pred_144] int pred_144_pred_144__pl_len_1;
    array[kernel_nsub_pred_144] int pred_144__pl_len_1;
    for(plate_i__pl_1 in 1:kernel_nsub_pred_144) {
        pred_144_pred_72__pl_len_1[plate_i__pl_1] = (
            (1 + (ragged_end(times_72.2, plate_i__pl_1) - ragged_start(times_72.2, plate_i__pl_1))) +
            (1 + (ragged_end(times_72.2, plate_i__pl_1) - ragged_start(times_72.2, plate_i__pl_1)))
        );
        pred_144_pred_144__pl_len_1[plate_i__pl_1] = (
            (1 + (ragged_end(times_144.2, plate_i__pl_1) - ragged_start(times_144.2, plate_i__pl_1))) +
            (1 + (ragged_end(times_144.2, plate_i__pl_1) - ragged_start(times_144.2, plate_i__pl_1)))
        );
        pred_144__pl_len_1[plate_i__pl_1] = (
            (1 + (ragged_end(times_144.2, plate_i__pl_1) - ragged_start(times_144.2, plate_i__pl_1))) +
            (1 + (ragged_end(times_144.2, plate_i__pl_1) - ragged_start(times_144.2, plate_i__pl_1)))
        );
    }
    array[kernel_nsub_pred_144] int pred_144_pred_72__pl_end_1 = cumulative_sum(pred_144_pred_72__pl_len_1);
    array[kernel_nsub_pred_144] int pred_144_pred_144__pl_end_1 = cumulative_sum(pred_144_pred_144__pl_len_1);
    array[kernel_nsub_pred_144] int pred_144__pl_end_1 = cumulative_sum(pred_144__pl_len_1);
}
parameters {
    cholesky_factor_corr[n_terms_VPR_subject] b_VPR_subject_L;
    vector<lower=0.0>[n_terms_VPR_subject] b_VPR_subject_tau;
    vector[(n_terms_VPR_subject * n_subject)] b_VPR_subject_z_flat;
    cholesky_factor_corr[n_terms_Fwp_subject] b_Fwp_subject_L;
    vector<lower=0.0>[n_terms_Fwp_subject] b_Fwp_subject_tau;
    vector[(n_terms_Fwp_subject * n_subject)] b_Fwp_subject_z_flat;
    cholesky_factor_corr[n_terms_Fpp_subject] b_Fpp_subject_L;
    vector<lower=0.0>[n_terms_Fpp_subject] b_Fpp_subject_tau;
    vector[(n_terms_Fpp_subject * n_subject)] b_Fpp_subject_z_flat;
    cholesky_factor_corr[n_terms_Ff_subject] b_Ff_subject_L;
    vector<lower=0.0>[n_terms_Ff_subject] b_Ff_subject_tau;
    vector[(n_terms_Ff_subject * n_subject)] b_Ff_subject_z_flat;
    cholesky_factor_corr[n_terms_Fl_subject] b_Fl_subject_L;
    vector<lower=0.0>[n_terms_Fl_subject] b_Fl_subject_tau;
    vector[(n_terms_Fl_subject * n_subject)] b_Fl_subject_z_flat;
    cholesky_factor_corr[n_terms_Vwp_subject] b_Vwp_subject_L;
    vector<lower=0.0>[n_terms_Vwp_subject] b_Vwp_subject_tau;
    vector[(n_terms_Vwp_subject * n_subject)] b_Vwp_subject_z_flat;
    cholesky_factor_corr[n_terms_Vpp_subject] b_Vpp_subject_L;
    vector<lower=0.0>[n_terms_Vpp_subject] b_Vpp_subject_tau;
    vector[(n_terms_Vpp_subject * n_subject)] b_Vpp_subject_z_flat;
    cholesky_factor_corr[n_terms_Vl_subject] b_Vl_subject_L;
    vector<lower=0.0>[n_terms_Vl_subject] b_Vl_subject_tau;
    vector[(n_terms_Vl_subject * n_subject)] b_Vl_subject_z_flat;
    cholesky_factor_corr[n_terms_Pba_subject] b_Pba_subject_L;
    vector<lower=0.0>[n_terms_Pba_subject] b_Pba_subject_tau;
    vector[(n_terms_Pba_subject * n_subject)] b_Pba_subject_z_flat;
    cholesky_factor_corr[n_terms_Pwp_subject] b_Pwp_subject_L;
    vector<lower=0.0>[n_terms_Pwp_subject] b_Pwp_subject_tau;
    vector[(n_terms_Pwp_subject * n_subject)] b_Pwp_subject_z_flat;
    cholesky_factor_corr[n_terms_Ppp_subject] b_Ppp_subject_L;
    vector<lower=0.0>[n_terms_Ppp_subject] b_Ppp_subject_tau;
    vector[(n_terms_Ppp_subject * n_subject)] b_Ppp_subject_z_flat;
    cholesky_factor_corr[n_terms_Pf_subject] b_Pf_subject_L;
    vector<lower=0.0>[n_terms_Pf_subject] b_Pf_subject_tau;
    vector[(n_terms_Pf_subject * n_subject)] b_Pf_subject_z_flat;
    cholesky_factor_corr[n_terms_Pl_subject] b_Pl_subject_L;
    vector<lower=0.0>[n_terms_Pl_subject] b_Pl_subject_tau;
    vector[(n_terms_Pl_subject * n_subject)] b_Pl_subject_z_flat;
    cholesky_factor_corr[n_terms_VMI_subject] b_VMI_subject_L;
    vector<lower=0.0>[n_terms_VMI_subject] b_VMI_subject_tau;
    vector[(n_terms_VMI_subject * n_subject)] b_VMI_subject_z_flat;
    cholesky_factor_corr[n_terms_KMI_subject] b_KMI_subject_L;
    vector<lower=0.0>[n_terms_KMI_subject] b_KMI_subject_tau;
    vector[(n_terms_KMI_subject * n_subject)] b_KMI_subject_z_flat;
    real log_sigma_venous;
    real log_sigma_exhaled;
    vector[pop_log_VPR_n_covariates] pop_log_VPR_beta_pop;
    vector[pop_log_Fwp_n_covariates] pop_log_Fwp_beta_pop;
    vector[pop_log_Fpp_n_covariates] pop_log_Fpp_beta_pop;
    vector[pop_log_Ff_n_covariates] pop_log_Ff_beta_pop;
    vector[pop_log_Fl_n_covariates] pop_log_Fl_beta_pop;
    vector[pop_log_Vwp_n_covariates] pop_log_Vwp_beta_pop;
    vector[pop_log_Vpp_n_covariates] pop_log_Vpp_beta_pop;
    vector[pop_log_Vl_n_covariates] pop_log_Vl_beta_pop;
    vector[pop_log_Pba_n_covariates] pop_log_Pba_beta_pop;
    vector[pop_log_Pwp_n_covariates] pop_log_Pwp_beta_pop;
    vector[pop_log_Ppp_n_covariates] pop_log_Ppp_beta_pop;
    vector[pop_log_Pf_n_covariates] pop_log_Pf_beta_pop;
    vector[pop_log_Pl_n_covariates] pop_log_Pl_beta_pop;
    vector[pop_log_VMI_n_covariates] pop_log_VMI_beta_pop;
    vector[pop_log_KMI_n_covariates] pop_log_KMI_beta_pop;
}
transformed parameters {
    matrix[n_terms_VPR_subject, n_subject] b_VPR_subject_z = to_matrix(b_VPR_subject_z_flat, n_terms_VPR_subject, n_subject);
    matrix[n_subject, n_terms_VPR_subject] b_VPR_subject = ((diag_pre_multiply(b_VPR_subject_tau, b_VPR_subject_L) * b_VPR_subject_z)');
    matrix[n_terms_Fwp_subject, n_subject] b_Fwp_subject_z = to_matrix(b_Fwp_subject_z_flat, n_terms_Fwp_subject, n_subject);
    matrix[n_subject, n_terms_Fwp_subject] b_Fwp_subject = ((diag_pre_multiply(b_Fwp_subject_tau, b_Fwp_subject_L) * b_Fwp_subject_z)');
    matrix[n_terms_Fpp_subject, n_subject] b_Fpp_subject_z = to_matrix(b_Fpp_subject_z_flat, n_terms_Fpp_subject, n_subject);
    matrix[n_subject, n_terms_Fpp_subject] b_Fpp_subject = ((diag_pre_multiply(b_Fpp_subject_tau, b_Fpp_subject_L) * b_Fpp_subject_z)');
    matrix[n_terms_Ff_subject, n_subject] b_Ff_subject_z = to_matrix(b_Ff_subject_z_flat, n_terms_Ff_subject, n_subject);
    matrix[n_subject, n_terms_Ff_subject] b_Ff_subject = ((diag_pre_multiply(b_Ff_subject_tau, b_Ff_subject_L) * b_Ff_subject_z)');
    matrix[n_terms_Fl_subject, n_subject] b_Fl_subject_z = to_matrix(b_Fl_subject_z_flat, n_terms_Fl_subject, n_subject);
    matrix[n_subject, n_terms_Fl_subject] b_Fl_subject = ((diag_pre_multiply(b_Fl_subject_tau, b_Fl_subject_L) * b_Fl_subject_z)');
    matrix[n_terms_Vwp_subject, n_subject] b_Vwp_subject_z = to_matrix(b_Vwp_subject_z_flat, n_terms_Vwp_subject, n_subject);
    matrix[n_subject, n_terms_Vwp_subject] b_Vwp_subject = ((diag_pre_multiply(b_Vwp_subject_tau, b_Vwp_subject_L) * b_Vwp_subject_z)');
    matrix[n_terms_Vpp_subject, n_subject] b_Vpp_subject_z = to_matrix(b_Vpp_subject_z_flat, n_terms_Vpp_subject, n_subject);
    matrix[n_subject, n_terms_Vpp_subject] b_Vpp_subject = ((diag_pre_multiply(b_Vpp_subject_tau, b_Vpp_subject_L) * b_Vpp_subject_z)');
    matrix[n_terms_Vl_subject, n_subject] b_Vl_subject_z = to_matrix(b_Vl_subject_z_flat, n_terms_Vl_subject, n_subject);
    matrix[n_subject, n_terms_Vl_subject] b_Vl_subject = ((diag_pre_multiply(b_Vl_subject_tau, b_Vl_subject_L) * b_Vl_subject_z)');
    matrix[n_terms_Pba_subject, n_subject] b_Pba_subject_z = to_matrix(b_Pba_subject_z_flat, n_terms_Pba_subject, n_subject);
    matrix[n_subject, n_terms_Pba_subject] b_Pba_subject = ((diag_pre_multiply(b_Pba_subject_tau, b_Pba_subject_L) * b_Pba_subject_z)');
    matrix[n_terms_Pwp_subject, n_subject] b_Pwp_subject_z = to_matrix(b_Pwp_subject_z_flat, n_terms_Pwp_subject, n_subject);
    matrix[n_subject, n_terms_Pwp_subject] b_Pwp_subject = ((diag_pre_multiply(b_Pwp_subject_tau, b_Pwp_subject_L) * b_Pwp_subject_z)');
    matrix[n_terms_Ppp_subject, n_subject] b_Ppp_subject_z = to_matrix(b_Ppp_subject_z_flat, n_terms_Ppp_subject, n_subject);
    matrix[n_subject, n_terms_Ppp_subject] b_Ppp_subject = ((diag_pre_multiply(b_Ppp_subject_tau, b_Ppp_subject_L) * b_Ppp_subject_z)');
    matrix[n_terms_Pf_subject, n_subject] b_Pf_subject_z = to_matrix(b_Pf_subject_z_flat, n_terms_Pf_subject, n_subject);
    matrix[n_subject, n_terms_Pf_subject] b_Pf_subject = ((diag_pre_multiply(b_Pf_subject_tau, b_Pf_subject_L) * b_Pf_subject_z)');
    matrix[n_terms_Pl_subject, n_subject] b_Pl_subject_z = to_matrix(b_Pl_subject_z_flat, n_terms_Pl_subject, n_subject);
    matrix[n_subject, n_terms_Pl_subject] b_Pl_subject = ((diag_pre_multiply(b_Pl_subject_tau, b_Pl_subject_L) * b_Pl_subject_z)');
    matrix[n_terms_VMI_subject, n_subject] b_VMI_subject_z = to_matrix(b_VMI_subject_z_flat, n_terms_VMI_subject, n_subject);
    matrix[n_subject, n_terms_VMI_subject] b_VMI_subject = ((diag_pre_multiply(b_VMI_subject_tau, b_VMI_subject_L) * b_VMI_subject_z)');
    matrix[n_terms_KMI_subject, n_subject] b_KMI_subject_z = to_matrix(b_KMI_subject_z_flat, n_terms_KMI_subject, n_subject);
    matrix[n_subject, n_terms_KMI_subject] b_KMI_subject = ((diag_pre_multiply(b_KMI_subject_tau, b_KMI_subject_L) * b_KMI_subject_z)');
    real sigma_venous = exp(log_sigma_venous);
    real sigma_exhaled = exp(log_sigma_exhaled);
    vector[num_elements(subject_idx)] pop_log_VPR = (X_log_VPR * pop_log_VPR_beta_pop);
    vector[subject_idx_n] r_log_VPR_VPR_subject = b_VPR_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_VPR = (pop_log_VPR + r_log_VPR_VPR_subject);
    vector[num_elements(subject_idx)] pop_log_Fwp = (X_log_Fwp * pop_log_Fwp_beta_pop);
    vector[subject_idx_n] r_log_Fwp_Fwp_subject = b_Fwp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Fwp = (pop_log_Fwp + r_log_Fwp_Fwp_subject);
    vector[num_elements(subject_idx)] pop_log_Fpp = (X_log_Fpp * pop_log_Fpp_beta_pop);
    vector[subject_idx_n] r_log_Fpp_Fpp_subject = b_Fpp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Fpp = (pop_log_Fpp + r_log_Fpp_Fpp_subject);
    vector[num_elements(subject_idx)] pop_log_Ff = (X_log_Ff * pop_log_Ff_beta_pop);
    vector[subject_idx_n] r_log_Ff_Ff_subject = b_Ff_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Ff = (pop_log_Ff + r_log_Ff_Ff_subject);
    vector[num_elements(subject_idx)] pop_log_Fl = (X_log_Fl * pop_log_Fl_beta_pop);
    vector[subject_idx_n] r_log_Fl_Fl_subject = b_Fl_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Fl = (pop_log_Fl + r_log_Fl_Fl_subject);
    vector[num_elements(subject_idx)] pop_log_Vwp = (X_log_Vwp * pop_log_Vwp_beta_pop);
    vector[subject_idx_n] r_log_Vwp_Vwp_subject = b_Vwp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Vwp = (pop_log_Vwp + r_log_Vwp_Vwp_subject);
    vector[num_elements(subject_idx)] pop_log_Vpp = (X_log_Vpp * pop_log_Vpp_beta_pop);
    vector[subject_idx_n] r_log_Vpp_Vpp_subject = b_Vpp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Vpp = (pop_log_Vpp + r_log_Vpp_Vpp_subject);
    vector[num_elements(subject_idx)] pop_log_Vl = (X_log_Vl * pop_log_Vl_beta_pop);
    vector[subject_idx_n] r_log_Vl_Vl_subject = b_Vl_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Vl = (pop_log_Vl + r_log_Vl_Vl_subject);
    vector[num_elements(subject_idx)] pop_log_Pba = (X_log_Pba * pop_log_Pba_beta_pop);
    vector[subject_idx_n] r_log_Pba_Pba_subject = b_Pba_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Pba = (pop_log_Pba + r_log_Pba_Pba_subject);
    vector[num_elements(subject_idx)] pop_log_Pwp = (X_log_Pwp * pop_log_Pwp_beta_pop);
    vector[subject_idx_n] r_log_Pwp_Pwp_subject = b_Pwp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Pwp = (pop_log_Pwp + r_log_Pwp_Pwp_subject);
    vector[num_elements(subject_idx)] pop_log_Ppp = (X_log_Ppp * pop_log_Ppp_beta_pop);
    vector[subject_idx_n] r_log_Ppp_Ppp_subject = b_Ppp_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Ppp = (pop_log_Ppp + r_log_Ppp_Ppp_subject);
    vector[num_elements(subject_idx)] pop_log_Pf = (X_log_Pf * pop_log_Pf_beta_pop);
    vector[subject_idx_n] r_log_Pf_Pf_subject = b_Pf_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Pf = (pop_log_Pf + r_log_Pf_Pf_subject);
    vector[num_elements(subject_idx)] pop_log_Pl = (X_log_Pl * pop_log_Pl_beta_pop);
    vector[subject_idx_n] r_log_Pl_Pl_subject = b_Pl_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_Pl = (pop_log_Pl + r_log_Pl_Pl_subject);
    vector[num_elements(subject_idx)] pop_log_VMI = (X_log_VMI * pop_log_VMI_beta_pop);
    vector[subject_idx_n] r_log_VMI_VMI_subject = b_VMI_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_VMI = (pop_log_VMI + r_log_VMI_VMI_subject);
    vector[num_elements(subject_idx)] pop_log_KMI = (X_log_KMI * pop_log_KMI_beta_pop);
    vector[subject_idx_n] r_log_KMI_KMI_subject = b_KMI_subject[subject_idx, 1];
    vector[num_elements(subject_idx)] log_KMI = (pop_log_KMI + r_log_KMI_KMI_subject);
    vector[sum(pred_144_pred_72__pl_len_1)] pred_144_pred_72__pl_mem_1;
    vector[sum(pred_144_pred_144__pl_len_1)] pred_144_pred_144__pl_mem_1;
    matrix[(2 + 1), kernel_nsub_pred_144] pred_144_volume_fraction;
    vector[kernel_nsub_pred_144] pred_144_Vl;
    matrix[4, kernel_nsub_pred_144] pred_144_unit_volume_flow;
    matrix[4, kernel_nsub_pred_144] pred_144_partition_coefficient;
    for(plate_i__pl_1 in 1:kernel_nsub_pred_144) {
        pred_144_unit_volume_flow[:, plate_i__pl_1] = softmax(
            [log_Fwp[plate_i__pl_1], log_Fpp[plate_i__pl_1], log_Ff[plate_i__pl_1], log_Fl[plate_i__pl_1]]'
        );
        pred_144_Vl[plate_i__pl_1] = exp(log_Vl[plate_i__pl_1]);
        pred_144_volume_fraction[:, plate_i__pl_1] = append_row(
            ((0.837 - pred_144_Vl[plate_i__pl_1]) * softmax([log_Vwp[plate_i__pl_1], log_Vpp[plate_i__pl_1]]')),
            pred_144_Vl[plate_i__pl_1]
        );
        pred_144_partition_coefficient[:, plate_i__pl_1] = exp([log_Pwp[plate_i__pl_1], log_Ppp[plate_i__pl_1], log_Pf[plate_i__pl_1], log_Pl[plate_i__pl_1]]');
        pred_144_pred_72__pl_mem_1[
            ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
        ] = monster_experiment(
            times_72.1[ragged_start(times_72.2, plate_i__pl_1):ragged_end(times_72.2, plate_i__pl_1)],
            exposure_72,
            lean_body_mass[plate_i__pl_1],
            fat_fraction[plate_i__pl_1],
            pulmonary_flow[plate_i__pl_1],
            exp(log_VPR[plate_i__pl_1]),
            pred_144_unit_volume_flow[:, plate_i__pl_1],
            pred_144_volume_fraction[:, plate_i__pl_1],
            exp(log_Pba[plate_i__pl_1]),
            pred_144_partition_coefficient[:, plate_i__pl_1],
            exp(log_VMI[plate_i__pl_1]),
            exp(log_KMI[plate_i__pl_1]),
            n_substeps
        );
        pred_144_pred_144__pl_mem_1[
            ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
        ] = monster_experiment(
            times_144.1[ragged_start(times_144.2, plate_i__pl_1):ragged_end(times_144.2, plate_i__pl_1)],
            exposure_144,
            lean_body_mass[plate_i__pl_1],
            fat_fraction[plate_i__pl_1],
            pulmonary_flow[plate_i__pl_1],
            exp(log_VPR[plate_i__pl_1]),
            pred_144_unit_volume_flow[:, plate_i__pl_1],
            pred_144_volume_fraction[:, plate_i__pl_1],
            exp(log_Pba[plate_i__pl_1]),
            pred_144_partition_coefficient[:, plate_i__pl_1],
            exp(log_VMI[plate_i__pl_1]),
            exp(log_KMI[plate_i__pl_1]),
            n_substeps
        );
    }
}
model {
    b_VPR_subject_L ~ lkj_corr_cholesky(1.0);
    b_VPR_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_VPR_subject_z_flat ~ std_normal();
    b_Fwp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Fwp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.1823215567939546);
    b_Fwp_subject_z_flat ~ std_normal();
    b_Fpp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Fpp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.1823215567939546);
    b_Fpp_subject_z_flat ~ std_normal();
    b_Ff_subject_L ~ lkj_corr_cholesky(1.0);
    b_Ff_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.1823215567939546);
    b_Ff_subject_z_flat ~ std_normal();
    b_Fl_subject_L ~ lkj_corr_cholesky(1.0);
    b_Fl_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.09531017980432493);
    b_Fl_subject_z_flat ~ std_normal();
    b_Vwp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Vwp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.1823215567939546);
    b_Vwp_subject_z_flat ~ std_normal();
    b_Vpp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Vpp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.1823215567939546);
    b_Vpp_subject_z_flat ~ std_normal();
    b_Vl_subject_L ~ lkj_corr_cholesky(1.0);
    b_Vl_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.09531017980432493);
    b_Vl_subject_z_flat ~ std_normal();
    b_Pba_subject_L ~ lkj_corr_cholesky(1.0);
    b_Pba_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_Pba_subject_z_flat ~ std_normal();
    b_Pwp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Pwp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_Pwp_subject_z_flat ~ std_normal();
    b_Ppp_subject_L ~ lkj_corr_cholesky(1.0);
    b_Ppp_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_Ppp_subject_z_flat ~ std_normal();
    b_Pf_subject_L ~ lkj_corr_cholesky(1.0);
    b_Pf_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_Pf_subject_z_flat ~ std_normal();
    b_Pl_subject_L ~ lkj_corr_cholesky(1.0);
    b_Pl_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.26236426446749106);
    b_Pl_subject_z_flat ~ std_normal();
    b_VMI_subject_L ~ lkj_corr_cholesky(1.0);
    b_VMI_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.6931471805599453);
    b_VMI_subject_z_flat ~ std_normal();
    b_KMI_subject_L ~ lkj_corr_cholesky(1.0);
    b_KMI_subject_tau ~ brm_vector_prior_180e10f5bbee3641(4.0, 0.4054651081081644);
    b_KMI_subject_z_flat ~ std_normal();
    pop_log_VPR_beta_pop ~ normal([0.47000362924573563]', [0.26236426446749106]');
    pop_log_Fwp_beta_pop ~ normal([-0.7339691750802004]', [0.1823215567939546]');
    pop_log_Fpp_beta_pop ~ normal([-1.6094379124341003]', [0.1823215567939546]');
    pop_log_Ff_beta_pop ~ normal([-2.659260036932778]', [0.1823215567939546]');
    pop_log_Fl_beta_pop ~ normal([-1.3862943611198906]', [0.09531017980432493]');
    pop_log_Vwp_beta_pop ~ normal([-1.2729656758128873]', [0.1823215567939546]');
    pop_log_Vpp_beta_pop ~ normal([-0.579818495252942]', [0.1823215567939546]');
    pop_log_Vl_beta_pop ~ normal([-3.4112477175156566]', [0.09531017980432493]');
    pop_log_Pba_beta_pop ~ normal([2.4849066497880004]', [0.4054651081081644]');
    pop_log_Pwp_beta_pop ~ normal([1.5686159179138452]', [0.4054651081081644]');
    pop_log_Ppp_beta_pop ~ normal([0.47000362924573563]', [0.4054651081081644]');
    pop_log_Pf_beta_pop ~ normal([4.8283137373023015]', [0.4054651081081644]');
    pop_log_Pl_beta_pop ~ normal([1.5686159179138452]', [0.4054651081081644]');
    pop_log_VMI_beta_pop ~ normal([-3.170085660698769]', [2.302585092994046]');
    pop_log_KMI_beta_pop ~ normal([2.772588722239781]', [2.302585092994046]');
    for(plate_i__pl_1 in 1:kernel_nsub_pred_144) {
        venous_72.1[ragged_start(venous_72.2, plate_i__pl_1):ragged_end(venous_72.2, plate_i__pl_1)] ~ lognormal(
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    venous_72_index.1[
                        ragged_start(venous_72_index.2, plate_i__pl_1):ragged_end(venous_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        exhaled_72.1[ragged_start(exhaled_72.2, plate_i__pl_1):ragged_end(exhaled_72.2, plate_i__pl_1)] ~ lognormal(
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_72_index.1[
                        ragged_start(exhaled_72_index.2, plate_i__pl_1):ragged_end(exhaled_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
        venous_144.1[ragged_start(venous_144.2, plate_i__pl_1):ragged_end(venous_144.2, plate_i__pl_1)] ~ lognormal(
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    venous_144_index.1[
                        ragged_start(venous_144_index.2, plate_i__pl_1):ragged_end(venous_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        exhaled_144.1[ragged_start(exhaled_144.2, plate_i__pl_1):ragged_end(exhaled_144.2, plate_i__pl_1)] ~ lognormal(
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_144_index.1[
                        ragged_start(exhaled_144_index.2, plate_i__pl_1):ragged_end(exhaled_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
    }
}
generated quantities {
    vector[sum(pred_144__pl_len_1)] pred_144__pl_mem_1;
    vector[num_elements(venous_72.1)] venous_72_gen;
    vector[num_elements(venous_72.2)] venous_72_likelihood;
    vector[num_elements(exhaled_72.1)] exhaled_72_gen;
    vector[num_elements(exhaled_72.2)] exhaled_72_likelihood;
    vector[num_elements(venous_144.1)] venous_144_gen;
    vector[num_elements(venous_144.2)] venous_144_likelihood;
    vector[num_elements(exhaled_144.1)] exhaled_144_gen;
    vector[num_elements(exhaled_144.2)] exhaled_144_likelihood;
    for(plate_i__pl_1 in 1:kernel_nsub_pred_144) {
        venous_72_gen[ragged_start(venous_72.2, plate_i__pl_1):ragged_end(venous_72.2, plate_i__pl_1)] = lognormal_vector_rng(
            (1 + (ragged_end(venous_72.2, plate_i__pl_1) - ragged_start(venous_72.2, plate_i__pl_1))),
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    venous_72_index.1[
                        ragged_start(venous_72_index.2, plate_i__pl_1):ragged_end(venous_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        venous_72_likelihood[plate_i__pl_1] = lognormal_lpdf(venous_72.1[ragged_start(venous_72.2, plate_i__pl_1):ragged_end(venous_72.2, plate_i__pl_1)] | 
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    venous_72_index.1[
                        ragged_start(venous_72_index.2, plate_i__pl_1):ragged_end(venous_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        exhaled_72_gen[ragged_start(exhaled_72.2, plate_i__pl_1):ragged_end(exhaled_72.2, plate_i__pl_1)] = lognormal_vector_rng(
            (1 + (ragged_end(exhaled_72.2, plate_i__pl_1) - ragged_start(exhaled_72.2, plate_i__pl_1))),
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_72_index.1[
                        ragged_start(exhaled_72_index.2, plate_i__pl_1):ragged_end(exhaled_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
        exhaled_72_likelihood[plate_i__pl_1] = lognormal_lpdf(exhaled_72.1[ragged_start(exhaled_72.2, plate_i__pl_1):ragged_end(exhaled_72.2, plate_i__pl_1)] | 
            log(
                pred_144_pred_72__pl_mem_1[
                    ragged_start(pred_144_pred_72__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_72__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_72_index.1[
                        ragged_start(exhaled_72_index.2, plate_i__pl_1):ragged_end(exhaled_72_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
        venous_144_gen[ragged_start(venous_144.2, plate_i__pl_1):ragged_end(venous_144.2, plate_i__pl_1)] = lognormal_vector_rng(
            (1 + (ragged_end(venous_144.2, plate_i__pl_1) - ragged_start(venous_144.2, plate_i__pl_1))),
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    venous_144_index.1[
                        ragged_start(venous_144_index.2, plate_i__pl_1):ragged_end(venous_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        venous_144_likelihood[plate_i__pl_1] = lognormal_lpdf(venous_144.1[ragged_start(venous_144.2, plate_i__pl_1):ragged_end(venous_144.2, plate_i__pl_1)] | 
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    venous_144_index.1[
                        ragged_start(venous_144_index.2, plate_i__pl_1):ragged_end(venous_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_venous
        );
        exhaled_144_gen[ragged_start(exhaled_144.2, plate_i__pl_1):ragged_end(exhaled_144.2, plate_i__pl_1)] = lognormal_vector_rng(
            (1 + (ragged_end(exhaled_144.2, plate_i__pl_1) - ragged_start(exhaled_144.2, plate_i__pl_1))),
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_144_index.1[
                        ragged_start(exhaled_144_index.2, plate_i__pl_1):ragged_end(exhaled_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
        exhaled_144_likelihood[plate_i__pl_1] = lognormal_lpdf(exhaled_144.1[ragged_start(exhaled_144.2, plate_i__pl_1):ragged_end(exhaled_144.2, plate_i__pl_1)] | 
            log(
                pred_144_pred_144__pl_mem_1[
                    ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
                ][
                    exhaled_144_index.1[
                        ragged_start(exhaled_144_index.2, plate_i__pl_1):ragged_end(exhaled_144_index.2, plate_i__pl_1)
                    ]
                ]
            ),
            sigma_exhaled
        );
        pred_144__pl_mem_1[
            ragged_start(pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144__pl_end_1, plate_i__pl_1)
        ] = pred_144_pred_144__pl_mem_1[
            ragged_start(pred_144_pred_144__pl_end_1, plate_i__pl_1):ragged_end(pred_144_pred_144__pl_end_1, plate_i__pl_1)
        ];
    }
}