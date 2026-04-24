# label: 3.3 ordinal predictors mo() (monotonic effects)
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + mo(c3)
y1 ~ Normal(loc, 1)
