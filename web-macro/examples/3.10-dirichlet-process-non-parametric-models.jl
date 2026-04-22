# label: 3.10 Dirichlet process / non-parametric models
# tier: 3
# status: open
#=
**What it is.** Models where the number of components / clusters / random-effect levels is itself inferred during sampling, via a Dirichlet process or stick-breaking prior.

**Why it matters.** When you don't know how many clusters are in your data, fixing K is itself a strong assumption. DP priors let the model decide.

**Implementation.** The heaviest item on the list. Needs sampling-time level inference, a stick-breaking parameter block, and a different `growblock!!` that grows during sampling rather than at `VBRMI` build time. Probably requires a different `lprior!` interface entirely.

Defer until everything else is solid.

**Verification.** Preset against synthetic data with an unknown number of latent clusters. Compare recovered K against the truth.

=#

loc ~ 1 + (1 | dp(g1))
y1 ~ Normal(loc, 1)
