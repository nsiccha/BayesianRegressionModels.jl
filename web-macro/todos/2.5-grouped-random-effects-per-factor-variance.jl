# label: 2.5 grouped random effects (gr(g, by=b)) — per-stratum covariance
# tier: 2
# status: done (vimpl); sb backend open
#=
**Status: done in vimpl.** `gr(g, by=b)` on the RHS of `|` allocates an
independent LKJ-Cholesky + grouped-normal block per level of `b`, so each
stratum gets its own full covariance structure (not just its own variance).

**Semantics.** `(1 + x | gr(g, by=b))` says: random intercept + slope on `x`
across `g`-levels, but with a separate covariance matrix per `b`-level. If
`b` has 3 levels you get 3 independent (K, K) Cholesky factors (and 3
independent tau vectors) — brms's canonical "different correlations per
diagnosis" pattern. `gr(g)` with no kwargs collapses to the plain `(... | g)`
path (same block, fully correlated shared prior).

**Constraint.** Every level of `g` must belong to exactly one level of `b`.
The walker (`_stratum_idx`) checks this at VBRMI time and errors if any
group-level straddles strata.

**Implementation sketch** (vimpl.jl):
- `GrGroup(group, by)` walker-side tag, produced by `_normalize_group`.
- New Part kinds: `grouped_normal_by` (values + stratum_idx + n_strata),
  `chol_by` (Ls::Array{Float64, 3}).
- `finalize(::Part{grouped_normal_by})` allocates the (n, n, n_strata) Ls
  and prepends a `chol_by` sibling sharing the tensor.
- `lprior!(::Part{chol_by})` loops strata, reuses `_chol_lprior!` per slice.
- `lprior!(::Part{grouped_normal_by})` loops rows, picks
  `Ls[:, :, stratum_idx[i]]` for the per-row mul.

**sb backend.** See sibling TODO (TBD) — needs an SB loop over groups and
array-of-cholesky allocation, then the same `rows_dot_product` form.

**Verification.** Synthetic data with two grouping factors (one nested) and
intentionally different per-outer-level correlation structure. Confirm the
fitted scales + correlations recover the synthetic values.

=#

loc ~ 1 + a + (1 + a | gr(g1, by=g2))
log(err) ~ 1
y1 ~ Normal(loc, err)
