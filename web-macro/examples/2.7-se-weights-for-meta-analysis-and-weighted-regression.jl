# label: 2.7 se() / weights() — already works (just a different distribution)
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a
y1 ~ Normal(loc, exposure)
