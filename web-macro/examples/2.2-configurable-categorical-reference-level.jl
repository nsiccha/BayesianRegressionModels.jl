# label: 2.2 configurable categorical reference level
# tier: 2
# status: open
#=
**What it is.** Currently the reference level for treatment-coded categoricals is `sort(unique(x))[1]`. brms / lme4 let you override this via `factor(x, ref="some_level")` or by reordering the factor's levels.

**Why it matters.** The reference level changes the interpretation of the intercept (it becomes "the mean for the reference level") and of the coefficients (each becomes "the difference from reference"). For some analyses, changing the reference is the only way to make the coefficients directly answer the research question.

**Implementation.**
1. Add `function factor end` to `macro.jl`.
2. Either store the override at parse time (rewrite `factor(x, ref=:level3)` into a wrapper that the materializer recognizes) or at materialization time via a `meta.factor_ref` NamedTuple keyed by column name.
3. The categorical-predictor path's `levels = sort(unique(x))` becomes `levels = sort(unique(x), by=l -> l == ref ? -Inf : l)` so the chosen reference always sorts first.

**Verification.** Preset: `loc ~ 1 + factor(c1, ref=2); y1 ~ Normal(loc, 1)`. Compare the fitted coefficients against the default-reference version — they should differ by the level-2-vs-level-1 mean shift but produce the same logdensity.

=#

loc ~ 1 + factor(c1, ref=2)
y1 ~ Normal(loc, 1)
