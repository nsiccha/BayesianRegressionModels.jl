# label: categorical random slope
# tier: 0
# status: open

loc ~ 1 + (1 + c1 | g1)
log(err) ~ 1
y1 ~ Normal(loc, err)
