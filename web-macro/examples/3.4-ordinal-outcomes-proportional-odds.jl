# label: 3.4 ordinal outcomes (proportional odds)
# tier: 3
# status: deprioritized
#=
**What it is.** When `y` is itself ordered categorical (Likert response, severity grades, …), use a cumulative-link model: `Pr(y ≤ k) = logistic(α_k - η)` where `α_k` are K-1 cutpoints and η is the linear predictor. The likelihood is the difference of consecutive CDFs.

**Why it matters.** Ordinal outcomes are common in survey data, clinical scoring, and any "rating" task. Treating them as continuous is statistically wrong; treating them as nominal categorical loses the ordering information.

**Implementation.** New likelihood family with a vector of cutpoints as additional parameters. `Distributions.jl` has `OrderedLogistic` already — the pass-through path should mostly handle it once the parser knows to extract cutpoints from a `cumulative` wrapper.

The cutpoints need an ordered prior (e.g. ordered transform of unconstrained reals), which means a new prior block type — similar to (3.3)'s simplex.

**Verification.** Preset against synthetic Likert data. Compare cutpoints against `polr` from R's MASS package.

=#
