data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  array[N] int<lower=1, upper=G> group;
  vector<lower=0>[N] weight;
  vector[N] y;
}

transformed data {
  matrix[N, 2] X;
  real half_log_weight_sum = 0.5 * sum(log(weight));
  X[, 1] = rep_vector(1.0, N);
  X[, 2] = x;
}

parameters {
  real<lower=0> tau;
  vector[G] z;
  real<lower=0> sigma;
  vector[2] beta;
}

model {
  vector[N] mu = X * beta + tau * z[group];
  vector[N] residual = y - mu;

  tau ~ exponential(1);
  z ~ std_normal();
  sigma ~ exponential(0.5);
  beta ~ std_normal();
  target += half_log_weight_sum - N * log(sigma)
            - 0.5 * dot_product(weight, square(residual)) / square(sigma)
            - 0.5 * N * log(2 * pi());
}
