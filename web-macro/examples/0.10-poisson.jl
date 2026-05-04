# label: Poisson
# tier: 0
# status: open

log_rate ~ 1 + a + (1 | g1)
k1 ~ Poisson(exp(log_rate))
