# label: 3.2 inferred predictors / measurement error me()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**What it is.** brms's `me(x_obs, sd_x)` says "the predictor `x_obs` is itself measured with error of size `sd_x`; sample the latent true value during inference". The model sees both the observed value and the latent.

**Why it matters.** Standard regression treats predictors as fixed/known. When predictors are themselves estimates (e.g. from a previous study or a noisy sensor), ignoring measurement error biases the slope estimates toward zero. `me()` is the principled fix.

**Status: done in sbimpl.** `me(x_obs, sd_x)` allocates a length-N latent `x_true` vector with prior `std_normal`. The measurement-error likelihood `x_obs ~ normal(x_true, sd_x)` is emitted by the walker at the call site -- StanBlocks' activity analysis drops sampling statements with a kwarg LHS inside submodel bodies, so it lives in the walker output rather than the `_sb_me` submodel. `x_true` feeds popefs just like a regular continuous covariate. Only a constant `sd_x` is supported; per-row sd vectors and hierarchical priors on `x_true` are follow-ups. vimpl side is still a bigger refactor (see below).

**Implementation (vimpl, for reference).** Bigger architectural change: predictor columns become latent variables sampled during inference, not data columns evaluated once. Needs:
- A new column type analogous to `MissingColumn` but with an observation-driven prior `Normal(x_obs, sd_x)`.
- The latent column gets a slot in the population block (one parameter per row).
- `lprior!` adds the per-row Normal prior contribution.
- `vbroadcasted` resolves the column to the latent values, not the observed ones.

**Verification.** Preset against synthetic data where the true `x` is known but only a noisy observed version is in the dataframe. Compare slope estimates with and without `me()`.

=#

loc ~ 1 + me(a, 0.1)
y1 ~ Normal(loc, 1)
