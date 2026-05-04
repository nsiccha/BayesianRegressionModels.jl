# label: Bernoulli
# tier: 0
# status: open

log_odds ~ 1 + a + (1 | g1)
bin_y ~ BernoulliLogit(log_odds)
