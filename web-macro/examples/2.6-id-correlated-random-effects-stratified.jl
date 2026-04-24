# label: 2.6 cross-formula `|ID|` ranefs, optionally stratified via `gr(g, by=b)`
# tier: 2
# status: open
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile

loc1 ~ 1 + a + (1 | p | g1)
loc2 ~ 0 + a + (0 + a | p | g1)
log(err) ~ 1
y1 ~ Normal(loc1 + loc2, err)
