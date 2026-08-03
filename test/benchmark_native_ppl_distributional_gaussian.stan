data {
  int<lower=1> N;
  vector[N] x;
  vector[N] z;
  vector[N] y;
}
parameters {
  vector[2] beta_mu;
  vector[2] beta_log_sigma;
}
model {
  beta_mu ~ std_normal();
  beta_log_sigma ~ std_normal();
  target += normal_lpdf(
    y |
    beta_mu[1] + beta_mu[2] * x,
    exp(beta_log_sigma[1] + beta_log_sigma[2] * z));
}
