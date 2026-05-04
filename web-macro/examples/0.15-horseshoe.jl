# label: Horseshoe
# tier: 0
# status: open

coef_a ~ Horseshoe()
loc = coef_a * a
y1 ~ Normal(loc, 1)
