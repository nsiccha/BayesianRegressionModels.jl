# label: 3.4 ordinal outcomes (proportional odds)
# tier: 3
# status: open
# flag: sb
#=
**What it is.** When `y` is itself ordered categorical (Likert response, severity grades, ...), use a cumulative-link model: `Pr(y <= k) = logistic(alpha_k - eta)` where `alpha_k` are K-1 cutpoints and eta is the linear predictor. The likelihood is the difference of consecutive CDFs.

**Why it matters.** Ordinal outcomes are common in survey data, clinical scoring, and any "rating" task. Treating them as continuous is statistically wrong; treating them as nominal categorical loses the ordering information.

**Status: sbimpl design staged, blocked on StanBlocks.** BRM defines a local `OrderedLogistic` marker + a cumulative-sum-of-positive-increments cutpoints submodel (`_sb_ordered_cuts`) that would drive `y ~ ordered_logistic(eta, cuts)` -- but two StanBlocks-side gaps hold it up:
1. `ordered_logistic_lpmf` is not in StanBlocks' `@builtin_module`, so `~ ordered_logistic(...)` fails symbol resolution.
2. `type=ordered` is not surfaced by `@slic` sampling statements (`ordered` isn't resolvable via `forward!(Symbol)` in the caller's module).
Workarounds tried: @deffun-wrapping `ordered_logistic_lpmf` inside a custom `*_lpdf` (the inner call still hits (1)); `@defsig` declarations for external Stan functions (the generated code references StanBlocks-internal `stan_size`, not reachable from user modules). Ask StanBlocks to add `ordered_logistic_lpmf` to `@builtin_module` (plus the lpxf/rng/likelihood hookups) and/or expose `type=ordered` as a sampling-statement kwarg. Once either lands, flip the flag to `sbbrm`.

**Implementation.** New likelihood family with a vector of cutpoints as additional parameters. `Distributions.jl` has `OrderedLogistic` already -- the pass-through path should mostly handle it once the parser knows to extract cutpoints from a `cumulative` wrapper.

The cutpoints need an ordered prior (e.g. ordered transform of unconstrained reals), which means a new prior block type -- similar to (3.3)'s simplex.

**Verification.** Preset against synthetic Likert data. Compare cutpoints against `polr` from R's MASS package.

=#

loc ~ 1 + a
c1 ~ OrderedLogistic(loc)
