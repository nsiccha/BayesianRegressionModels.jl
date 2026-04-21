# label: 1.5 zerocorr — `(terms || group)` for independent random effects
# tier: 1
# status: done (vimpl)
#=
**Status: done in vimpl.** brms (via lme4 syntax) lets you opt out of the LKJ
correlation between multiple random terms on the same grouping factor:
`(1 + x || g)` says "fit a random intercept and a random slope, but treat them
as independent — don't estimate a 2x2 Cholesky between them". Useful when
there's not enough data to estimate the correlation, or when you have prior
reason to believe the terms are uncorrelated.

**Semantics.** `(1 + a + b || g)` splits into three independent 1-column
grouped_normal blocks, each keyed distinctly from plain `(... | g)` so the
uncorrelated and correlated variants never accidentally coalesce. Per-term
1x1 Cholesky reduces to a single log_scale parameter (same `chol` machinery
used for the scalar `(1 | g)` case).

**Implementation.** Our `_x` walker wraps `||` as `ExprColumn{typeof(doublepipe)}`.
We added a `vmeta_sampling_rhs` overload that flattens the LHS on `+`, then
foldl's over the terms with a synthetic per-term group key
(`Symbol(name(rhs), :__nocor__, i)`) so each term lands in its own block.

**Verification.** Preset below vs. correlated counterpart
`loc ~ 1 + (1 + a | g1)`:
- correlated variant: 1 + `chol(2)` (3 params) + `grouped_normal(8, 2)` (16 params) = 20 params
- uncorrelated variant: 1 + 2 * (`chol(1)` (1 param) + `grouped_normal(8, 1)` (8 params)) = 19 params

So the only difference is one fewer Cholesky parameter (the off-diagonal).

=#

loc ~ 1 + (1 + a || g1)
y1 ~ Normal(loc, 1)
