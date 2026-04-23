# label: 3.1 multivariate outcomes — uncorrelated free, correlated needs MvNormal
# tier: 3
# status: deprioritized
#=
**What it is.** brms's `mvbind(y1, y2) ~ x + (1 | g)` declares two outcomes sharing the same linear predictor structure with **correlated residuals** (joint MvNormal with a covariance matrix to estimate). brms also supports combining multiple per-outcome formulas via `bf(y1 ~ x) + bf(y2 ~ x)`, but **`+` on the LHS of `~` is not a thing in brms's formula DSL** — both their multivariate syntaxes are sidecars (`mvbind`) or formula-object combinators (`bf + bf`).

**Half of this is already free in our DSL.** Multiple `~` lines on different data columns produce independent likelihoods that `llikelihood!` sums. If they reference the same intermediate variable, they share that linear predictor automatically — no parser changes needed:

```julia
loc ~ 1 + a + (1 | g1)
y1 ~ Normal(loc, 1)
y2 ~ Normal(loc, 1)
```

That's "multiple uncorrelated outcomes with the same regression formula", already supported. Mixed-family is also free:

```julia
loc ~ 1 + a + (1 | g1)
y1 ~ Normal(loc, 1)
k1 ~ Poisson(exp(loc))
```

**What's actually missing.** Only the **correlated residuals** case — when the off-diagonal entries of the residual covariance matrix are themselves parameters to estimate (because you want to learn how much shared latent noise the outcomes have). For that you need:

1. A new likelihood path that takes a *tuple* of data columns and computes a single joint `logpdf(MvNormal(loc_vec, Σ), [y1[i], y2[i], ...])` per row instead of summing marginal logpdfs.
2. A new parameter block holding the Cholesky factor of `Σ` over the outcomes (separate from the random-effects Cholesky on the random-effects block).
3. A way to spell this in our DSL. Two candidate syntaxes:
   - `(y1, y2) ~ MvNormal(loc, Sigma)` — Julia parses `(y1, y2)` as a tuple expression, the `~` parser would need a new branch for tuple LHS.
   - `mvbind(y1, y2) ~ MvNormal(loc, Sigma)` — explicit wrapper, parser-friendly because it's just a function call (and `mvbind` could be a stub like `gr`/`doublepipe`).

**Use cases for correlated:** `mcelreath::waffle_divorce_multivariate` is the canonical example. Joint pharmacology/efficacy modeling. Mediation analysis where you model both the mediator and the outcome and want to know how much shared subject-level variance they have.

**Verification.** The form below loads the **uncorrelated** case (already works). Two `Normal` likelihoods on `y1` and `y2` sharing one `loc` formula. The gradient sanity check should be all-active. For the correlated case, `mvbind(y1, y2) ~ MvNormal(loc, Sigma)` would currently error — that's the half this todo still tracks.

=#

loc ~ 1 + a + (1 | g1)
y1 ~ Normal(loc, 1)
y2 ~ Normal(loc, 1)
