# label: exact GP
# tier: 0
# status: done (sbimpl)

loc ~ 1 + a + gp(b)
y1 ~ Normal(loc, 1)
