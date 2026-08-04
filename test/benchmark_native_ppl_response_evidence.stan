data {
  int<lower=1> N;
  int<lower=1> G;
  vector[N] x;
  array[N] int<lower=1, upper=G> group;
  vector[N] y;
  real lower_bound;
  real upper_bound;
}

transformed data {
  matrix[N, 2] X;
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

  tau ~ exponential(1);
  z ~ std_normal();
  sigma ~ exponential(0.5);
  beta ~ std_normal();
  for (n in 1:N) {
    if (y[n] == lower_bound) {
      target += log(erfc(-(lower_bound - mu[n]) / sigma / sqrt(2))) - log(2);
    } else if (y[n] == upper_bound) {
      target += log(erfc((upper_bound - mu[n]) / sigma / sqrt(2))) - log(2);
    } else {
      target += normal_lpdf(y[n] | mu[n], sigma);
    }
  }
}
