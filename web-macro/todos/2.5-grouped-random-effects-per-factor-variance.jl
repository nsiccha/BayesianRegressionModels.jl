# label: 2.5 grouped random effects (per-factor variance)
# tier: 2
# status: open
#=
**What it is.** Peter's "different variance by diagnosis" pattern: `(1 | subject) gr(diagnosis)` says "the random intercept by subject has a different variance per diagnosis level". In brms this is a custom group structure where the variance hyperparameter itself depends on a second factor.

**Why it matters.** Common in clinical data where treatment groups have intrinsically different between-subject variability. Without this, you have to fit separate models per diagnosis or accept a single pooled variance.

**Implementation.** Bigger than it looks because the variance is no longer a single scalar but a length-`n_levels(diagnosis)` vector that needs its own prior and its own gradient.

Proposed shape:
- A new `gr(group_factor)` wrapper recognized in the `~` RHS via a `function gr end` stub (already exists in `macro.jl`).
- The wrapped block stores `n_levels(group_factor)` log-scale parameters instead of one. `lprior!` walks them, multiplying each subject's random intercept by the diagnosis-specific scale.
- Requires the gc_idx for the inner factor (subject) AND for the outer factor (diagnosis) — both vectors of length N.

This composes naturally with (1.6 caching) and (2.3 per-parameter priors).

**Verification.** Preset against synthetic data with two grouping factors, one nested inside the other, with intentionally different per-outer-level variance. Confirm the fitted scales recover the synthetic values.

=#

loc ~ 1 + (1 | gr(g1, g2))
y1 ~ Normal(loc, 1)
