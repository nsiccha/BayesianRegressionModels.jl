data {
  int<lower=1> N;
  vector[N] x;
  vector<lower=0>[N] exposure;
  array[N] int<lower=0> y;
}
transformed data {
  matrix[N, 2] X;
  vector[N] log_exposure = log(exposure);
  X[, 1] = rep_vector(1.0, N);
  X[, 2] = x;
}
parameters {
  vector[2] beta;
}
model {
  beta ~ std_normal();
  target += poisson_log_glm_lpmf(y | X, log_exposure, beta);
}
