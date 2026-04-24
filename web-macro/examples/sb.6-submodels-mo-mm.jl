# label: sb.6 submodels (mo, mm, spline) as SlicModels — needs impl
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + a + mo(c1)
log(err) ~ 1
y1 ~ Normal(loc, err)
