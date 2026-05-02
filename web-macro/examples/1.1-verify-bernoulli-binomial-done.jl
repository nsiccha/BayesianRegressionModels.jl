# label: 1.1 verify Bernoulli/Binomial
# tier: 1
# status: open
# flag: sb

log_odds_bin ~ 1 + c2 + (1 | g1)
bin_succ ~ BinomialLogit(bin_n, log_odds_bin)

log_odds_b ~ 1 + (1 | g1)
bin_y ~ BernoulliLogit(log_odds_b)
