data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  vector[N] w;
  array[N] int<lower=1, upper=G> group;
  vector[N] y;
}

transformed data {
  matrix[N, 2] X;
  matrix[N, 3] Z;
  X[, 1] = x;
  X[, 2] = w;
  Z[, 1] = rep_vector(1.0, N);
  Z[, 2] = x;
  Z[, 3] = w;
}

parameters {
  cholesky_factor_corr[3] L;
  vector<lower=0>[3] tau;
  matrix[3, G] z;
  real<lower=0> sigma;
  vector[2] beta;
}

model {
  matrix[3, G] b = diag_pre_multiply(tau, L) * z;
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
