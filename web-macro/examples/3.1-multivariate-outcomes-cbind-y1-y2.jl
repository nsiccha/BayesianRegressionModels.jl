# label: 3.1 multivariate outcomes — uncorrelated free, correlated needs MvNormal
# tier: 3
# status: deprioritized

loc ~ 1 + a + (1 | g1)
y1 ~ Normal(loc, 1)
y2 ~ Normal(loc, 1)
