# label: 1.2 offset / fixed exposure — already works without a wrapper
# tier: 1
# status: done
#=
**Status: already works without any new code.** brms needs `offset(z)` because R's formula syntax has no other way to put a "no-coefficient term" into the linear predictor — the only thing on the RHS of `~` is the formula DSL. In our DSL the linear predictor and the likelihood are *separate* `~` lines, and the second one (the likelihood) takes a free-form Julia expression. Anything inside that expression gets evaluated as plain code at materialization time via `vbroadcasted` — function calls dispatch to whatever Julia function the symbol resolves to, and data column references are pulled from the dataframe.

So instead of `count ~ x + offset(log(exposure))`, you write the offset directly inside the likelihood:

```julia
loc ~ 1 + a
k1 ~ Poisson(exp(loc + log(exposure)))
```

The `log(exposure)` here is just `Base.log` applied to the `exposure` data column, broadcasted across rows and added to `loc` (which is the materialized linear predictor). No parameter is allocated for it because `growblock!!` is never called for that branch — there's no `~` on the data side, just an argument to `Poisson(...)`.

**Verification.** Form below loads exactly that model. The VBRMI dim should match the offset-free version (only the population intercept + slope on `a`); the gradient sanity check should stay green; and the materialized `k1` likelihood should incorporate the row-specific exposure shift.

=#

loc ~ 1 + a
k1 ~ Poisson(exp(loc + log(exposure)))
