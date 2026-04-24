# label: 2.6 multi-membership random effects mm()
# tier: 2
# status: deprioritized

loc ~ 1 + (1 | mm(g1, g2))
y1 ~ Normal(loc, 1)
