# label: 3.7 autoregressive submodels ar(), ar1()
# tier: 3
# status: open
#=
**What it is.** Add an AR(p) structure to the residuals: `y_t = η_t + φ * (y_{t-1} - η_{t-1}) + ε_t`. brms's `ar(time, p=1)` specifies the order and the time variable.

**Why it matters.** Time-series and repeated-measures data routinely have autocorrelated errors. Ignoring AR structure inflates the effective sample size and gives overconfident posteriors.

**Implementation.** Bigger than it looks because the likelihood is no longer per-row independent — it's a chain. Need a new `LikelihoodColumn`-like type that holds the time index and walks the data in time order, accumulating the AR contribution row by row.

Composes with (2.5 grouped random effects) for per-subject AR structure.

**Verification.** Preset against synthetic AR(1) data. Recover φ.

=#

loc ~ 1 + a + ar(g1, p=1)
y1 ~ Normal(loc, 1)
