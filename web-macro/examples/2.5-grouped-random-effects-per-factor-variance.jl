# label: 2.5 grouped random effects (gr(g, by=b)) — per-stratum covariance
# tier: 2
# status: done (vimpl); sb backend open

loc ~ 1 + a + (1 + a | gr(g1, by=g2))
log(err) ~ 1
y1 ~ Normal(loc, err)
