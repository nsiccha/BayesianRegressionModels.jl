# label: sb.6 submodels (mo, mm, spline) as SlicModels — needs impl
# tier: 3
# status: open
#=
**Status: not implemented.** The main architectural bet behind sbimpl: terms that introduce their own parameters (`mo(x)` monotonic effects, `mm(x)` multi-membership, `s(x)` splines, etc.) should map to their own `@slic` submodels with their own `~`-sampled hyperparameters, composed into the overall linear predictor via `popranefs`-style addition.

**Why it matters.** Each submodel lives as a standalone `SlicModel`, so StanBlocks' activity analysis routes its parameters / priors / transforms to the right Stan block automatically. No per-term case analysis in sbimpl.

**Prototype term: `mo(x)`** (Bürkner & Charpentier 2018). Emit:

```julia
const mo = StanBlocks.@slic begin
    n_levels = maximum(x)
    simplex_incr ~ dirichlet(rep_vector(1., n_levels - 1))
    return cumulative_sum(append_row(0., simplex_incr))[x]
end
```

Then `loc ~ 1 + mo(c1)` walks to `mo_c1 = mo(; x=c1); loc ~ popefs(; X=hcat(..., mo_c1))`.

**Verification.** `cimpl` has `Mo1Part` already (todo `3.3-ordinal-predictors-mo-monotonic-effects`); Stan dim should match.

=#

loc ~ 1 + a + mo(c1)
log(err) ~ 1
y1 ~ Normal(loc, err)
