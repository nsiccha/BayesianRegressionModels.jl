# label: 1.2 offset / fixed exposure — already works without a wrapper
# tier: 1
# status: open
# flag: sbbrm

loc ~ 1 + a
k1 ~ Poisson(exp(loc + log(exposure)))
