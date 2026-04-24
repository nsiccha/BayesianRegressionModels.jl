# label: 3.2 inferred predictors / measurement error me()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + me(a, 0.1)
y1 ~ Normal(loc, 1)
