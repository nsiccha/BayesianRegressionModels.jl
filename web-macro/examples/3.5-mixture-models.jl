# label: 3.5 mixture models
# tier: 3
# status: open
# flag: sbbrm
#=
**What it is.** brms's `mixture(Normal, Normal)` lets the likelihood be a weighted mixture of K component distributions, with mixing weights estimated as parameters.

**Why it matters.** Heterogeneous populations, latent class analysis, robust regression (Normal + heavy-tailed component), zero-inflated outcomes, …

**Implementation.** Extends the existing `Distribution` pass-through. Need a new `MixtureModel` wrapper that holds component distributions plus a weights parameter block. `llikelihood!` uses `logsumexp(log_weights .+ logpdf.(components, y))` per row.

Composes with (3.4 ordered priors) for the mixing weights' Dirichlet-like prior.

**Verification.** Preset against synthetic two-component-Normal data. Confirm the recovered mixture weights and component parameters.

=#

loc ~ 1 + a
y1 ~ MixtureModel(Normal[Normal(loc, 0.5), Normal(0, 5)], [0.9, 0.1])
