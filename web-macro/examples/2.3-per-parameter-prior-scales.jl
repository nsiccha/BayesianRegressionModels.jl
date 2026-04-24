# label: 2.3 per-parameter prior scales
# tier: 2
# status: deprioritized

coef_a ~ Normal(0, 0.1)
loc ~ coef_a * a
y1 ~ Normal(loc, 1)
