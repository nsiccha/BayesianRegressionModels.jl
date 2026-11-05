# BayesianRegressionModels.jl

Bayesian regression models in Julia, inspired by brms and bambi. Write a model
with `@brm`, then execute it with StanBlocks (`SBBRMI`) or
ReactiveKernels/RKPPL (`RKBRMI`).

The formula language covers population and group effects, distributional
regression, structured terms, priors, missing values, and custom calculations
inside `@plate` cells. Both backends start from the same `BRMI` model.

See the [documentation](https://nsiccha.github.io/BayesianRegressionModels.jl/dev/)
for executable examples, backend contracts, fitting, and prediction. The
[feature atlas](docs/src/feature-atlas.md) shows declarations alongside their
generated representations. Development checks use the
[test environment](test/README.md).
