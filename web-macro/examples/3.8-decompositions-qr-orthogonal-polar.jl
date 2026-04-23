# label: 3.8 decompositions (QR, orthogonal polar)
# tier: 3
# status: open
# flag: sbbrm
#=
**What it is.** Numerical-stability transformations of the population design matrix. brms uses QR decomposition on the design matrix internally so the sampler sees an orthogonal-columns version, then transforms back at the end. Stan does the same.

**Why it matters.** When population covariates are correlated (which is the norm), the unrotated design matrix gives a poorly-conditioned posterior that NUTS struggles with. QR fixes this with no statistical change.

**Status: sbimpl-only work; no StanBlocks changes needed.** Sketch for `_sb_linear_predictor!`: after building `X_pop` via `hcat`, run `Q, R = qr(X_pop)` in Julia, stash `Q` as the popefs `X` kwarg (instead of raw hcat'd columns), and stash `R` as an upper-triangular data matrix. popefs itself stays unchanged -- it keeps sampling `std_normal` on its beta vector and returning `X * beta_pop`, but now `X = Q`, so the rotated design matrix gives NUTS a better-conditioned posterior for free. Expose the original-scale coefficients in generated quantities via `beta_original = inverse(R) * beta_pop` (`inverse` is already a StanBlocks builtin, so no upstream changes are required). Gate behind a `qr_transform` kwarg on `SBBRMI(...)` so default-off behaviour is preserved while verifying; flip default-on once gradient conditioning is confirmed.

**Implementation.** Mostly orthogonal to the formula DSL -- happens at `SBBRMI` build time, not in the parser. Could be implemented today without any parser changes. Lift to a default-on once verified.

**Verification.** Same model with and without QR should produce identical logdensity values (up to constants), but the gradient should be better-conditioned (smaller condition number on the Hessian). Easy smoke test: compare sbimpl's `ldp` between the two flag settings on a wide-correlated-X preset.

=#
