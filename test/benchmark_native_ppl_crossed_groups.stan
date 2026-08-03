data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> I;
  vector[N] x;
  array[N] int<lower=1, upper=S> subject;
  array[N] int<lower=1, upper=I> item;
  vector[N] y;
}

transformed data {
  matrix[N, 1] X;
  X[, 1] = x;
}

parameters {
  cholesky_factor_corr[2] L_p;
  vector<lower=0>[2] tau_p;
  matrix[2, S] z_p;
  real<lower=0> tau_q;
  vector[I] z_q;
  real<lower=0> sigma;
  vector[1] beta;
}

model {
  matrix[2, S] b_p = diag_pre_multiply(tau_p, L_p) * z_p;
  vector[I] b_q = tau_q * z_q;
  vector[N] group_offset;
  for (n in 1:N) {
    group_offset[n] = b_p[1, subject[n]]
                      + b_p[2, subject[n]] * x[n]
                      + b_q[item[n]];
  }

  L_p ~ lkj_corr_cholesky(2);
  tau_p ~ exponential(1);
  to_vector(z_p) ~ std_normal();
  tau_q ~ exponential(1);
  z_q ~ std_normal();
  sigma ~ exponential(0.5);
  beta ~ std_normal();
  target += normal_id_glm_lpdf(y - group_offset | X, 0.0, beta, sigma);
}
