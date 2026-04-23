# label: 3.11 zero-inflated / hurdle likelihoods
# tier: 3
# status: open
# flag: sbbrm
#=
**What it is.** ZI Poisson, ZI Negative Binomial, hurdle Poisson, hurdle Gamma, … — likelihoods that mix a point mass at zero (or a separate "is zero" Bernoulli) with a continuous/count distribution for the nonzero values.

**Why it matters.** Count data with excess zeros (insurance claims, species abundance, healthcare utilization) is everywhere. Standard Poisson / NegBin underfits the zero count.

**Implementation.** Should mostly work through the existing `Distribution` pass-through once we use `Distributions.jl`'s ZI distributions (or write small wrappers). The mixing weight needs its own linear predictor, which is just another distributional-regression-style `~` line.

**Verification.** Preset against synthetic ZI Poisson data. Confirm the recovered zero-inflation probability matches the synthetic generator.

=#

log_rate ~ 1 + a
zi_logit ~ 1
k1 ~ ZeroInflatedPoisson(exp(log_rate), logistic(zi_logit))
