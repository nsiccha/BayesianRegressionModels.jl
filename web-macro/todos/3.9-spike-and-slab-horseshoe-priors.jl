# label: 3.9 spike-and-slab / Horseshoe priors
# tier: 3
# status: open
#=
**What it is.** Sparsity-inducing priors for high-dimensional regression. Spike-and-slab puts a delta-spike at zero plus a wide slab; Horseshoe uses a half-Cauchy hyperprior on a per-coefficient scale, producing a heavy-tailed shrinkage prior.

**Why it matters.** Without sparsity priors, high-dimensional regressions overfit. These are the standard solution in Bayesian variable selection and high-dim genomics / finance.

**Implementation.** Depends entirely on (2.3 per-parameter priors). Once that's in place, spike-and-slab is `prior = Mixture(Normal(0, ε), Normal(0, slab_scale))` per coefficient, and Horseshoe is `Normal(0, λ_i * τ)` with `λ_i ~ HalfCauchy(0, 1)` and `τ ~ HalfCauchy(0, 1)` — both expressible in the existing Distribution pass-through once per-coefficient priors are wired up.

**Verification.** Preset on a sparse synthetic regression (mostly-zero true coefficients with a few large ones). Confirm the Horseshoe-fitted coefficients shrink the noise toward zero and preserve the signal.

=#

coef_a ~ Horseshoe()
loc ~ coef_a * a
y1 ~ Normal(loc, 1)
