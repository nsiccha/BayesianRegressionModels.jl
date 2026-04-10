# label: 2.6 multi-membership random effects mm()
# tier: 2
# status: deprioritized
#=
**What it is.** brms's `mm(g1, g2, ...)` lets one observation belong to **multiple** levels of the same random factor simultaneously, with weights summing to 1. Standard use: a student belongs to multiple schools across the year, and we want their random effect to be a weighted average of the per-school effects.

**Why it matters.** Standard random effects assume each observation belongs to exactly one group. Multi-membership is the only clean way to handle observations that span groups (mobile students, patients seen by multiple clinicians, etc.).

**Implementation.** `_gc_idx` would have to return a row-of-vectors instead of a single Int per row. Two paths:
- **Sparse design matrix**: replace the `gc_idx` lookup with a sparse `(N × n_levels)` matrix where each row's nonzero entries are the membership weights. The materialized random effect becomes `sparse_membership * random_effects_vector`.
- **Per-row lookup loop**: keep the row-major view but make `_re_lookup` iterate the membership list per row, summing weighted contributions.

The sparse matrix approach is more memory-efficient and SIMD-friendly. Needs a new wrapper in the formula syntax: `(1 | mm(g1, g2; weights=...))`.

**Verification.** Preset against synthetic data where each observation has 2 random group memberships with weights summing to 1. Compare against the equivalent "fully observed in primary group only" model.

=#

loc ~ 1 + (1 | mm(g1, g2))
y1 ~ Normal(loc, 1)
