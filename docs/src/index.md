---
layout: home

hero:
  name: BayesianRegressionModels.jl
  text: brms-style formula DSL on top of Stan
  tagline: Compose linear-predictor formulas, custom priors, and likelihoods; transpile to Stan via StanBlocks; fit via Pathfinder / NUTS.
  actions:
    - theme: brand
      text: Gallery
      link: /gallery
    - theme: alt
      text: GitHub
      link: https://github.com/JuliaBayes/BayesianRegressionModels.jl
---

# BayesianRegressionModels.jl

A formula DSL for Bayesian regression. The macro `@brm` parses brms-style
syntax (`y ~ 1 + a + (1 | g)`, `log(err) ~ 1 + b`,
`y ~ Normal(loc, err)`) into an intermediate representation, then a
StanBlocks backend transpiles it to Stan code that fits via Pathfinder
or full warmup HMC.

See the [Gallery](/gallery) for live, interactive examples — input
formula, the SLIC submodel body, the transpiled Stan source, and the
auto-generated posterior-predictive check, all in one card.
