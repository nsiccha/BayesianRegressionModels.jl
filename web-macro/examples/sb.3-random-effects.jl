# label: sb.3 random effects (1 | g) — needs impl
# tier: 2
# status: open
#=
**Status: not implemented.** sbimpl currently errors on `ExprColumn{typeof(|)}` in `_sb_collect_terms_expr!`.

**What to implement.** Emit `ranefs` / `popranefs` SB submodels that sample a non-centered grouped-normal block and add it to the popefs output.

**Walker work.** When `_sb_collect_terms_expr!` sees `(expr | group)`:
1. Split terms into `pop_terms` and `ran_terms`, collect `Z` as the one-hot / index matrix for `group`.
2. Swap the emitted `loc ~ popefs(; X)` for `loc ~ popranefs(; X_pop, X_ran, Z)`.
3. Data: add `Z_<group>` (or group indices) to the data dict. Design matrix widening still happens in Stan so data size doesn't bake in.

**Verification.** Reuse the `bernoulli/binomial` smoke test once `sb.5` lands (non-Normal likelihoods), but a Normal version works as a first pass.

=#

loc ~ 1 + a + (1 + a | g1)
log(err) ~ 1
y1 ~ Normal(loc, err)
