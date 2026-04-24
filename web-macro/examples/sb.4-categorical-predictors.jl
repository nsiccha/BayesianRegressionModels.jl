# label: sb.4 categorical predictors (c1, c2, ...) — needs impl
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a + c1 + c2
log(err) ~ 1
y1 ~ Normal(loc, err)
