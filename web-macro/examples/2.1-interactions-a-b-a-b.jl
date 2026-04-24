# label: 2.1 interactions a&b (cont x cont done; cat cases open)
# tier: 2
# status: partial (cont x cont done; cont x cat / cat x cat open)
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,transform,wrap

loc ~ 1 + a + b + a&b
y1 ~ Normal(loc, 1)
