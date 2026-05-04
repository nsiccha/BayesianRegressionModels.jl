# label: Binomial
# tier: 0
# status: open

log_odds ~ 1 + a + (1 | g1)
bin_succ ~ BinomialLogit(bin_n, log_odds)
