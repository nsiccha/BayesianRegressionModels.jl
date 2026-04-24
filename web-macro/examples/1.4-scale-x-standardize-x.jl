# label: 1.4 zscale(x) / center(x) / standardize(x)
# tier: 1
# status: done (vimpl)

loc ~ 1 + zscale(a) + center(b)
y1 ~ Normal(loc, 1)
