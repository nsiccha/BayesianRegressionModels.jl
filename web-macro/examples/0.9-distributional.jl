# label: distributional
# tier: 0
# status: open

loc ~ 1 + a
log(err) ~ 1 + b
y1 ~ Normal(loc, err)
