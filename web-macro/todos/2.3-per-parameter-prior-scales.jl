# label: 2.3 per-parameter prior scales
# tier: 2
# status: deprioritized
#=
**What it is.** Currently every parameter is `Normal(0, 1)` in `lprior!`. brms / Stan-style models routinely set custom priors per coefficient: `b ~ Normal(0, 0.5)` for tight priors on slopes, `b ~ Cauchy(0, 1)` for heavy-tailed priors, etc.

**Why it matters.** Default `Normal(0, 1)` is fine after standardization but poor on raw scales. Allowing per-parameter prior scales is the prerequisite for spike-and-slab, Horseshoe, and most prior sensitivity analyses. Without it, users have no way to express domain knowledge about parameter magnitudes.

**Implementation.** This is the largest design decision in Tier 2 because it has knock-on effects for every other prior-related TODO.

Two storage candidates:
- **Per-block scales**: extend `meta.block_data[group]` with a per-column scale vector. `lprior!` multiplies the standard-normal draw by the scale before storing. Simple but only handles Normal-with-scale priors.
- **Per-block prior distributions**: store a vector of `Distribution` objects per block. `lprior!` calls `logpdf(prior_i, xi)` for each parameter. More general; handles Cauchy, StudentT, Horseshoe, etc.

Recommend the second — it's strictly more powerful and the runtime cost is identical (one `logpdf` call per parameter). Default value is `Normal(0, 1)` for backward compatibility.

**Implementation sketch.**
1. Extend `meta.block_data` with a `priors` field per block.
2. The macro syntax `b ~ Normal(0, 0.5)` parses as a sampling statement with a Distribution-typed RHS. Currently this is reserved for likelihood declarations; it would need a new "is this a prior or a likelihood?" branch in `vmeta_sampling`. Likelihood: LHS is a data column. Prior: LHS is a maybelocal (parameter).
3. `lprior!` reads the per-column prior and calls `logpdf(prior, value)` instead of the hard-coded `logpdf(Normal(), value)`.

**Verification.** Preset: `loc ~ 1 + a; b ~ Normal(0, 0.1); y1 ~ Normal(loc, 1)` — confirm the gradient is dampened on `b` compared to the default-prior version, and that the dead-param check still passes.

=#

coef_a ~ Normal(0, 0.1)
loc ~ coef_a * a
y1 ~ Normal(loc, 1)
