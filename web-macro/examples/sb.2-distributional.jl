# label: sb.2 distributional (loc + scale predictors)
# tier: 1
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**Status: should work.** Two separate linear predictors — one for the location, one (via `log` link) for the scale — both pop-only.

**Why it matters.** Distributional regression is the simplest place where the "activity analysis routes statements to the right Stan block" benefit shows up: `X_loc` / `X_log_err` are built in `transformed data`, `beta_pop`s land in `parameters`, the `~ popefs(...)` calls land in `model`.

**Verification.** Stan source should show two `popefs` calls with independent `X_...` design matrices.

=#

loc ~ 1 + a + b
log(err) ~ 1 + c + d
y1 ~ Normal(loc, err)
