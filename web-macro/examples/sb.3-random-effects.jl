# label: sb.3 random effects (1 | g) — needs impl
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_fit_warmup,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a + (1 + a | g1)
log(err) ~ 1
y1 ~ Normal(loc, err)
