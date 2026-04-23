# label: sb.1 linear regression (verified working)
# tier: 1
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**Status: should work.** Baseline sbimpl coverage: pop fixed effects (intercept + numeric columns), `log(x) ~ ...` link, `Normal(loc, scale)` likelihood.

**What sbimpl emits.**
- Prepass stashes all data-backed NamedColumns into the SlicModel's data dict (so `num_elements(<probe>)` resolves even when an intercept-only predictor precedes the first data reference).
- `loc ~ 1 + a + b` becomes `X_loc = hcat(rep_vector(1., num_elements(a)), a, b); loc ~ popefs(; X=X_loc)`.
- `log(err) ~ 1` becomes `X_log_err = reshape(rep_vector(1., num_elements(a)), :, 1); log_err ~ popefs(; X=X_log_err); err = exp(log_err)`.
- `y1 ~ Normal(loc, err)` becomes `y1 ~ normal(loc, err)` with `data[:y1] = df.y1`.

**Verification.** Click "sbimpl (Stan) ▶" — should print a compilable Stan program. "cimpl (bench) ▶" should also finish with a healthy FD check.

=#

loc ~ 1 + a + b
log(err) ~ 1
y1 ~ Normal(loc, err)
