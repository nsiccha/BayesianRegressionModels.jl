# label: ZIP
# tier: 0
# status: open

log_rate ~ 1 + a
logit(zi) ~ 1
k1 ~ ZeroInflatedPoisson(exp(log_rate), zi)
