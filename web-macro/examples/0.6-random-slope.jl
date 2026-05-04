# label: random slope
# tier: 0
# status: open

loc ~ 1 + (1 + a | g1)
log(err) ~ 1
y1 ~ Normal(loc, err)
