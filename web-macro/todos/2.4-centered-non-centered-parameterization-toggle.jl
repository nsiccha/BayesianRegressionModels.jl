# label: 2.4 centered / non-centered parameterization toggle
# tier: 2
# status: open
#=
**What it is.** Currently every random-effect block uses non-centered parameterization (we sample standard normals and apply `mul!(vi, C.L, xi)`). brms / Stan let you choose centered (sample directly from `Normal(0, σ)` per group) on a per-factor basis.

**Why it matters.** Non-centered is the default for "weak data per group" cases (Neal's funnel pathology), but for "strong data per group" cases centered samples better. Letting users choose is a meaningful sampling speedup for the latter regime.

**Implementation.** Small change to `lprior!` and `growblock!!`. Add a `centered::Bool` flag to `meta.block_data[group]`. In `lprior!`'s non-population branch, if the block is centered, sample directly from `Normal(0, exp(log_scale))` instead of `Normal(0, 1)` then multiplying by `L`. The Cholesky machinery for off-diagonal correlations still applies in the centered case — just on the column before the variance scaling rather than after.

Once (2.3) is in place, the centered/non-centered choice could be encoded as `(1 | g) ~ Normal(0, σ)` (centered) vs the implicit non-centered default — but for now a per-block kwarg or a wrapper function (e.g. `centered((1 + a | g))`) is simpler.

**Verification.** Same model with both parameterizations should produce the same logdensity at the same parameter values (after the appropriate change of variables). Sampling efficiency on a known-funnel dataset should differ.

=#

loc ~ 1 + (1 + a | centered(g1))
y1 ~ Normal(loc, 1)
