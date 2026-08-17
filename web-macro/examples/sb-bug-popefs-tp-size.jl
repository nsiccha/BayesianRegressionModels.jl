# label: SB bug: popefs TP matrix size scope
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap

loc ~ 1 + ztime
log(y_scale) ~ 0 + assay_idx
y ~ Normal(loc, y_scale)
