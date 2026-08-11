# Backend lowering

One `BRMI` can feed more than one backend, but the routes are deliberately not
the same. The Stan route lowers through `SBBRMI` and StanBlocks; the Turing
extension consumes the backend-neutral BRMI plan directly. Use the tabs below
for a compact reading view or **Compare all** for the three-column view.

The comparison component is not tied to these backend names: a docs example can
provide any two or more code fences and a matching `data-pane-labels` list.

## BRM → StanBlocks → Stan

This is one ordinary Gaussian BRMI shown at the three useful abstraction
levels. The middle pane is a readable view of the generated StanBlocks
declaration; the last pane shows the model-bearing blocks of the emitted Stan.

```@raw html
<div class="backend-comparison" data-backend-comparison data-comparison-title="BRM to Stan lowering" data-pane-labels="BRM|StanBlocks IR|Generated Stan">
```

```julia
using BayesianRegressionModels, Distributions

data = (;
    x = [-1.0, 0.5, 2.0],
    y = [0.2, 1.1, -0.4],
)

brmi = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    effect(mu, x) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)(data)
```

```text
SlicModel (untraced)
  bound inputs: y, x
  declaration:
    begin
        sigma ~ exponential(1.0 ./ 2)
        X_mu = hcat(rep_vector(1.0, num_elements(x)), x)
        pop_mu ~ _popefs_normal_coefs(
            X=X_mu,
            beta_loc=[0.0, 0],
            beta_scale=[1.0, 0.25],
        )
        y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma)
        mu = X_mu * pop_mu
    end
```

```stan
data {
    int x_n;
    vector[x_n] x;
    int y_n;
    vector[y_n] y;
}
transformed data {
    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);
    int pop_mu_n_covariates = 2;
}
parameters {
    real<lower=0.0> sigma;
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
}
transformed parameters {
    vector[pop_mu_n_covariates] pop_mu = pop_mu_beta_pop;
}
model {
    sigma ~ exponential((1.0 ./ 2));
    pop_mu_beta_pop ~ normal([0.0, 0]', [1.0, 0.25]');
    y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma);
}
generated quantities {
    vector[x_n] y_likelihood = normal_id_glm_lpdfs(
        y, X_mu, 0.0, pop_mu, sigma
    );
    vector[x_n] y_gen = normal_id_glm_vector_rng(
        y_n, X_mu, 0.0, pop_mu, sigma
    );
    vector[x_n] mu = X_mu * pop_mu;
}
```

```@raw html
</div>
```

The generated file also contains the Stan helper definitions for `hcat`,
`normal_id_glm_lpdfs`, and `normal_id_glm_vector_rng`; the comparison keeps the
model-bearing blocks visible instead of repeating those library helpers.

## Turing ← BRM → StanBlocks

This view makes the architectural fork explicit. Both wrappers receive the
same `BRMI`; `TuringBRMI` does not construct or inspect the StanBlocks model.

```@raw html
<div class="backend-comparison" data-backend-comparison data-comparison-title="One BRMI, sibling executors" data-pane-labels="Turing|BRM|StanBlocks">
```

```julia
using Turing

turing = TuringBRMI(brmi)
model = turing.model  # DynamicPPL.Model

chain = sample(model, NUTS(), 1_000)
```

```julia
brmi = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    effect(mu, x) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)(data)
```

```julia
stanblocks = SBBRMI(brmi)
model = stanblocks.model  # StanBlocks.SlicModel

source = stan_code(stanblocks)
```

```@raw html
</div>
```

The supported Turing surface and its fail-closed boundary are tracked in the
[Turing backend parity matrix](turing-backend.md#parity-contract). The
StanBlocks route continues to cover the wider formula catalogue.
