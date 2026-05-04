# label: random intercept
# tier: 0
# status: open

loc ~ 1 + a + (1 | g1)
log(err) ~ 1
y1 ~ Normal(loc, err)
