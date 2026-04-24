# label: 2.4 centered / non-centered parameterization toggle
# tier: 2
# status: deprioritized

loc ~ 1 + (1 + a | centered(g1))
y1 ~ Normal(loc, 1)
