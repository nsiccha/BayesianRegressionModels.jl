# label: 1.1 verify Bernoulli/Binomial
# tier: 1
# status: done
#=
**Status: done.** ✓ Confirmed that the existing `FBroadcasted{<:Type{<:Distribution}}` pass-through in `vimpl.jl` handles both Bernoulli and Binomial cleanly.

**Verification.** The form below loads the **cbpp + therapeutic touch** model — a faithful translation of `brms::cbpp_binomial` (categorical predictor + random intercept + Binomial with per-row trial counts) and `kruschke::therapeutic_touch` (hierarchical Bernoulli) into one multi-likelihood model. brms's `incidence | trials(size) ~ ...` sidecar collapses to a plain positional argument: `bin_succ ~ Binomial(bin_n, logistic(η))`. If the gradient sanity check stays green every other `Distribution` family (Beta, Gamma, NegBinomial, …) should be free as well.

=#

log_odds_bin ~ 1 + c1 + (1 | g1)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + (1 | g1)
bin_y ~ Bernoulli(logistic(log_odds_b))
