# label: 3.7 autoregressive submodels ar(), ar1()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a + ar(g1, p=1)
y1 ~ Normal(loc, 1)
