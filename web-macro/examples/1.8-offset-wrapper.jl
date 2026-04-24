# label: 1.8 offset(x) wrapper — brms-compatible fixed-slope term
# tier: 1
# status: done (vimpl)

log_rate ~ 1 + a + offset(log(exposure))
k1 ~ Poisson(exp(log_rate))
