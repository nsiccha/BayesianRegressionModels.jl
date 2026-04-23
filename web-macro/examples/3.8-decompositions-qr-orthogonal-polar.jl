# label: 3.8 decompositions (QR, orthogonal polar)
# tier: 3
# status: deprioritized
#=
**What it is.** Numerical-stability reparameterization of the population design matrix. brms exposes this as `decomp = "QR"` on the formula; it reparameterizes `X * beta` as `(Q*) * (R* * beta)` where `Q*` has orthonormal columns (up to `sqrt(N-1)` scaling) and sampling happens on `beta_tilde = R* * beta`, with `beta = R*^{-1} * beta_tilde` exposed as a transformed parameter.

**Why it matters.** When population covariates are correlated, the unrotated posterior has correlated coordinates and NUTS struggles; sampling on `beta_tilde` instead sees an orthogonal geometry.

**Status: deprioritized.** brms treats this as an opt-in, whole-response `decomp = "QR"` flag, not an always-on transform and not a per-term marker. Carrying that shape through to BRM would be a formula-surface addition, not a sbimpl-internal knob -- the right framing is "add `decomp = "QR"` to the formula DSL", which is a macro-side decision that we don't need right now. Revisit once there's a concrete model with correlated population covariates whose NUTS fit is actually painful enough to justify the prior-semantics tradeoff.

**Prior-semantics tradeoff.** A `std_normal` prior on the rotated coefficient is NOT the same as a `std_normal` prior on the original coefficient -- the implicit prior on `beta` becomes `N(0, (X'X)^{-1})`. brms acknowledges this; for weakly-informative priors it's usually fine, for tight informative priors it's a real gotcha.

**Sketch if/when we resurrect.** `qr(X)` in Julia, stash `Q` as popefs's X kwarg and `L = R'` (lower-tri) as a data matrix; sample `beta_rot ~ std_normal`; recover `beta_orig` via `transpose(mdivide_right_tri_low(to_row_vector(beta_rot), L))` (canonical Stan transpose trick -- no `inverse()` anywhere). Only kicks in when all pop terms are "simple" (intercept + raw data columns); mixed cases (`mo`, `s`, `me`, etc.) fall through.

=#
