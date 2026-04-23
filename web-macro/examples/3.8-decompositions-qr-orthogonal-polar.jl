# label: 3.8 decompositions (QR, orthogonal polar)
# tier: 3
# status: open
#=
**What it is.** Numerical-stability transformations of the population design matrix. brms uses QR decomposition on the design matrix internally so the sampler sees an orthogonal-columns version, then transforms back at the end. Stan does the same.

**Why it matters.** When population covariates are correlated (which is the norm), the unrotated design matrix gives a poorly-conditioned posterior that NUTS struggles with. QR fixes this with no statistical change.

**Status: not sbimpl scope.** This is a `VBRMI` / cimpl concern -- no formula DSL surface, no SB-backend gap. For the SB path the equivalent would be a `qr_transform` flag on `SBBRMI` that swaps `popefs(; X)` for a QR-rotated analog and emits an inverse transform in `generated quantities`, but the decision logic and the "default-on once verified" lift live alongside VBRMI's block transforms. Re-flag for SB only if/when an example surfaces a gap that sbimpl must solve (e.g. very wide design matrices where the Stan fit diverges until rotated).

**Implementation.** Mostly orthogonal to the formula DSL -- happens at `VBRMI` build time. Add a per-block transform: store both the original design matrix and the QR factor, run sampling on the rotated parameter space, transform back when extracting coefficients.

Could be implemented today without any parser changes, as a `qr_transform=true` flag on `VBRMI`. Lift to a default-on once verified.

**Verification.** Same model with and without QR should produce identical logdensity values, but the gradient should be better-conditioned (smaller condition number on the Hessian).

=#
