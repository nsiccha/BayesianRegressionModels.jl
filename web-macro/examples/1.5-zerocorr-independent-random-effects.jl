# label: 1.5 zerocorr — `(terms || group)` for independent random effects
# tier: 1
# status: done (vimpl)

loc ~ 1 + (1 + a || g1)
y1 ~ Normal(loc, 1)
