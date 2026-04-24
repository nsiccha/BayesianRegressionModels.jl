# label: sb.2 distributional (loc + scale predictors)
# tier: 1
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a + b
log(err) ~ 1 + c + d
y1 ~ Normal(loc, err)
