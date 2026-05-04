# label: cbpp + therapeutic touch
# tier: 0
# status: open

log_odds_bin ~ 1 + c1 + (1 | g1)
bin_succ ~ BinomialLogit(bin_n, log_odds_bin)

log_odds_b ~ 1 + (1 | g1)
bin_y ~ BernoulliLogit(log_odds_b)
