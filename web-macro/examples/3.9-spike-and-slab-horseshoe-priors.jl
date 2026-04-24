# label: 3.9 spike-and-slab / Horseshoe priors
# tier: 3
# status: open
# flag: sbbrm

coef_a ~ Horseshoe()
loc ~ coef_a * a
y1 ~ Normal(loc, 1)
