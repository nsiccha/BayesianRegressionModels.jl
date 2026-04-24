# label: 3.11 zero-inflated / hurdle likelihoods
# tier: 3
# status: open
# flag: sbbrm

log_rate ~ 1 + a
zi_logit ~ 1
k1 ~ ZeroInflatedPoisson(exp(log_rate), logistic(zi_logit))
