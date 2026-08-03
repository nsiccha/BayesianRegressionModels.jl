data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  array[N] int<lower=1, upper=G> group;
  vector[N] y;
}

transformed data {
  vector[N] x_scaled = (x - mean(x)) / sd(x);
  matrix[N, 1] X;
  matrix[N, 2] Z;
  X[, 1] = x_scaled;
  Z[, 1] = rep_vector(1.0, N);
  Z[, 2] = x_scaled;
}

parameters {
  cholesky_factor_corr[2] L;
  vector<lower=0>[2] tau;
  matrix[2, G] z;
  real<lower=0> sigma;
  vector[1] beta;
}

model {
  matrix[2, G] b = diag_pre_multiply(tau, L) * z;
  vector[N] group_offset;
  for (n in 1:N) {
    group_offset[n] = dot_product(Z[n], b[, group[n]]);
  }

  L ~ lkj_corr_cholesky(2);
  tau ~ exponential(1);
  to_vector(z) ~ std_normal();
  sigma ~ exponential(0.5);
  beta ~ std_normal();
  target += normal_id_glm_lpdf(y - group_offset | X, 0.0, beta, sigma);
}
