# label: 3.10 Dirichlet process / non-parametric models
# tier: 3
# status: deprioritized

loc ~ 1 + (1 | dp(g1))
y1 ~ Normal(loc, 1)
