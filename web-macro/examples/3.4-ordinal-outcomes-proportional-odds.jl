# label: 3.4 ordinal outcomes (proportional odds)
# tier: 3
# status: open
# flag: sb
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a
c1 ~ OrderedLogistic(loc)
