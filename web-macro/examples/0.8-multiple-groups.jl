# label: multiple groups
# tier: 0
# status: open

loc ~ 1 + a + (1 | g1) + (1 | g2)
log(err) ~ 1
y1 ~ Normal(loc, err)
