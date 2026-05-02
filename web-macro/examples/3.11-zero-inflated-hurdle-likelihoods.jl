# label: 3.11 zero-inflated / hurdle likelihoods
# tier: 3
# status: open
# flag: sbbrm

log_rate ~ 1 + a
logit(zi) ~ 1
k1 ~ ZeroInflatedPoisson(exp(log_rate), zi)
