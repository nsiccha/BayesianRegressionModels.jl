# label: 2.6 cross-formula `|ID|` ranefs, optionally stratified via `gr(g, by=b)`
# tier: 2
# status: open
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile
#=
**Cross-formula correlated random effects (brms `|ID|`).** All `(... | ID | g)`
terms across all sub-formulas sharing the same `ID` pool into one correlated
block. Here `loc1`'s intercept and `loc2`'s slope on `a` are drawn jointly
per level of `g1` from a shared 2x2 LKJ covariance.

**Stratified variant.** Wrap the group in `gr(g, by=b)` (or use the
`gr(..., id="p")` function form) to get per-stratum covariance: each level
of `b` gets its own 2x2 LKJ, but within a stratum the two columns still
share correlation. Every level of `g` must belong to exactly one level of
`b` — the walker rejects data that violates this.

Swap line A (plain) with line B (stratified) to toggle.
=#

# A. plain cross-formula correlated:
loc1 ~ 1 + a + (1 | p | g1)
loc2 ~ 0 + a + (0 + a | p | g1)

# B. stratified cross-formula correlated (uncomment to use):
# loc1 ~ 1 + a + (1 | p | gr(g1, by=g2))
# loc2 ~ 0 + a + (0 + a | p | gr(g1, by=g2))

log(err) ~ 1
y1 ~ Normal(loc1 + loc2, err)
