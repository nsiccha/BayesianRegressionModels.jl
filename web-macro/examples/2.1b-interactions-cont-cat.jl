# label: 2.1b interactions cont x cat (sbimpl)
# tier: 2
# status: sbbrm
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,transform,wrap

loc ~ 1 + a + c1 + a&c1
y1 ~ Normal(loc, 1)
