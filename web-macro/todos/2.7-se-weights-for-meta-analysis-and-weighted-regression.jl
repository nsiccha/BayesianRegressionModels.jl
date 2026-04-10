# label: 2.7 se() / weights() — already works (just a different distribution)
# tier: 2
# status: done
#=
**Status: already works without any new code.** Same insight as 1.2 (offset) and 1.3 (`I()`): brms needs sidecar syntax (`y | se(sigma_y) ~ ...`, `y | weights(w) ~ ...`) because R's formula DSL has no other way to attach extra info to the LHS. Our DSL has no such constraint — the likelihood is a free-form Julia expression, so anything brms expresses via sidecar syntax we can express via constructor arguments or a wrapper distribution.

**Meta-analysis (per-observation known SE).** Just pass the SE column as the scale argument to `Normal`:

```julia
loc ~ 1 + a
y1 ~ Normal(loc, exposure)
```

`exposure` is a length-N data column; the broadcasted `Normal(loc, exposure)` constructs a per-row Normal at materialization time, and `logpdf(Normal(loc[i], exposure[i]), y1[i])` is what `llikelihood!` ends up summing. No new code in `vimpl.jl`.

**Weighted regression (per-observation likelihood weights).** Define a thin wrapper distribution that multiplies the underlying logpdf by a weight, and use the existing `FBroadcasted{<:Type{<:Distribution}}` pass-through:

```julia
struct WeightedLikelihood{D, W} <: Distributions.ContinuousUnivariateDistribution
    base::D
    weight::W
end
Distributions.logpdf(w::WeightedLikelihood, y) = w.weight * logpdf(w.base, y)

# Then in the formula:
loc ~ 1 + a
y1 ~ WeightedLikelihood(Normal(loc, 1), exposure)
```

The wrapper is ~5 lines of plain Julia in the user's own scope, no changes to the DSL or to `vimpl.jl`. The pass-through walks the broadcasted `WeightedLikelihood` constructor, materializes `loc`/`exposure` per row, and `llikelihood!` calls the user-defined `logpdf` per row.

**Conclusion.** Neither feature needs special parser or vimpl support. `se()` is a constructor argument; `weights()` is a one-page user-defined distribution. The general principle: anything brms expresses as an LHS sidecar, our DSL expresses as a positional argument or a user-supplied distribution.

**Verification.** The form below loads the meta-analysis case directly. The gradient sanity check should be all-active and the per-row Normal scales should pick up the `exposure` column.

=#

loc ~ 1 + a
y1 ~ Normal(loc, exposure)
