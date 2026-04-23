# label: 3.7 autoregressive submodels ar(), ar1()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**What it is.** Add an AR(p) structure to the residuals: `y_t = eta_t + phi * (y_{t-1} - eta_{t-1}) + epsilon_t`. brms's `ar(time, p=1)` specifies the order and the time variable.

**Why it matters.** Time-series and repeated-measures data routinely have autocorrelated errors. Ignoring AR structure inflates the effective sample size and gives overconfident posteriors.

**Status: `ar(time, p=1)` first-pass done in sbimpl.** Parameterized as an additive AR(1) latent noise process `u[t] = phi * u[t-1] + epsilon[t]` (u[1] = epsilon[1], no stationary init), with `phi = tanh(phi_raw)` under `std_normal(phi_raw)` and `epsilon ~ std_normal(n)`. Returned as one column; popefs multiplies by an overall beta. `time` arg is a length probe only -- rows are assumed already in time order. Only `p=1` supported; `ar(time; p>1)` / `ar1(time)` still error.

**Gaps.**
- No sort by `time`; feeding a non-monotonic column silently produces an AR process over the row order, not time order.
- No stationary initialization; `u[1]` has marginal variance 1 instead of `1/(1 - phi^2)`.
- No per-group structure yet (composes with (2.5) for per-subject AR).
- Likelihood is still per-row independent -- the AR lives in the predictor, matching brms's "random-effects" encoding but not its `autocor = ar(..., cov=TRUE)` dense-covariance path.

**Implementation.** Bigger than it looks because the likelihood is no longer per-row independent -- it's a chain. Need a new `LikelihoodColumn`-like type that holds the time index and walks the data in time order, accumulating the AR contribution row by row. The current sbimpl path is the simpler "AR on the predictor" encoding; a true AR-on-residuals variant needs either a specialized likelihood or a dense-covariance formulation.

Composes with (2.5 grouped random effects) for per-subject AR structure.

**Verification.** Preset against synthetic AR(1) data. Recover phi.

=#

loc ~ 1 + a + ar(g1, p=1)
y1 ~ Normal(loc, 1)
