# label: 3.6 splines / GP submodels s(), bs(), gp(), hsgp(), t2()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + s(a) + t2(a, b; k=(5, 5), basis=(:cr, :cr), full=false)
y1 ~ Normal(loc, 1)
