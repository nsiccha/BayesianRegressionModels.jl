# Changelog

All notable changes to BayesianRegressionModels.jl are documented here, newest
first.

Every landing that adds or changes public API (an export, a macro, a keyword, a
method, or documented behaviour a dependant can rely on) bumps the patch digit
in `Project.toml` and adds its entry under a new heading at the top of this
file. An entry that breaks something a dependant relies on says **Breaking:**
and what to change. A package or app that relies on an entry states its version
as the floor, for example `[compat] BayesianRegressionModels = "0.2.2"`. An
environment that pins an older BRM commit is then refused when it resolves
instead of failing at run time.

## 0.2.2

- `brm_start(sb, stan_model, inits)` takes a value for every level of an
  intercept-free predictor's cell-means block (`0 + g`, `0 + factor(g)`),
  `lp = (; g = ["a" => 1.1, "b" => 0.4, ...])`, on the link scale of a linked
  predictor, as `brm_start(rk, inits)` already did. Before, the Stan method
  found no coefficient for any level of such a block. Snag
  `brm-start-sees-n-5b61087d`.
