# label: 2.1c interactions cat x cat (sbimpl)
# tier: 2
# status: sbbrm
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,transform,wrap

loc ~ 1 + c1 + c2 + c1&c2
y1 ~ Normal(loc, 1)
