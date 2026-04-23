# label: 3.8 decompositions (QR, orthogonal polar)
# tier: 3
# status: open
# flag: both
#=
**What it is.** Numerical-stability transformations of the population design matrix. brms uses QR decomposition on the design matrix internally so the sampler sees an orthogonal-columns version, then transforms back at the end. Stan does the same.

**Why it matters.** When population covariates are correlated (which is the norm), the unrotated design matrix gives a poorly-conditioned posterior that NUTS struggles with. QR fixes this with no statistical change.

**Status: needs its own implementation per backend.** VBRMI and SBBRMI are orthogonal, so a QR option has to be implemented twice -- once on each side -- not routed through a shared layer.

For sbimpl: precompute `Q, R = qr(X_pop)` in Julia (inside `_sb_linear_predictor!`), stash `Q` under the popefs X-name (instead of the raw hcat), stash `R` as a small upper-triangular data matrix, and in generated quantities expose `beta = R \ beta_tilde` so coefficients are reported on the original scale. `popefs` itself does not need to change -- it keeps sampling a `std_normal` on its beta vector; the rotated design matrix gives NUTS a better-conditioned posterior for free. Gate behind a `qr_transform` kwarg on `SBBRMI(...)` so the default-off behaviour is preserved while verifying; flip default-on once gradient conditioning is confirmed.

For vimpl: mirror the same approach at `VBRMI` build time -- store both the original X and the QR factor per block, run sampling on the rotated parameter space, transform back when extracting coefficients.

**Implementation.** Mostly orthogonal to the formula DSL -- happens at backend build time, not in the parser. Could be implemented today without any parser changes, as a `qr_transform=true` flag on each backend. Lift to a default-on once verified.

**Verification.** Same model with and without QR should produce identical logdensity values (up to constants), but the gradient should be better-conditioned (smaller condition number on the Hessian). Easy smoke test: compare sbimpl's `ldp` between the two flag settings on a wide-correlated-X preset.

=#
