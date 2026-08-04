data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  vector[N] w;
  array[N] int<lower=1, upper=G> group;
  vector[N] y;
  array[N] int<lower=0, upper=1> z;
}
transformed data {
  matrix[N, 2] X_x;
  matrix[N, 2] X_w;
  X_x[, 1] = rep_vector(1, N);
  X_x[, 2] = x;
  X_w[, 1] = rep_vector(1, N);
  X_w[, 2] = w;
}
parameters {
  cholesky_factor_corr[3] b_p_group_L;
  vector<lower=0>[3] b_p_group_tau;
  vector[3 * G] b_p_group_z_flat;
  vector[2] pop_mu_y_beta_pop;
  vector[2] pop_log_sigma_y_beta_pop;
  vector[2] pop_eta_z_beta_pop;
}
model {
  matrix[G, 3] b_p_group = (
    diag_pre_multiply(b_p_group_tau, b_p_group_L) *
    to_matrix(b_p_group_z_flat, 3, G))';
  vector[N] mu_y = X_x * pop_mu_y_beta_pop + b_p_group[group, 1];
  vector[N] log_sigma_y =
    X_w * pop_log_sigma_y_beta_pop + b_p_group[group, 2];
  vector[N] eta_z = X_x * pop_eta_z_beta_pop + b_p_group[group, 3];

  b_p_group_L ~ lkj_corr_cholesky(2);
  b_p_group_tau ~ exponential(1);
  b_p_group_z_flat ~ std_normal();
  pop_mu_y_beta_pop ~ std_normal();
  pop_log_sigma_y_beta_pop ~ std_normal();
  pop_eta_z_beta_pop ~ std_normal();
  y ~ normal(mu_y, exp(log_sigma_y));
  z ~ bernoulli_logit(eta_z);
}
