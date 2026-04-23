# label: 3.6 splines / GP submodels s(), bs(), gp(), t2()
# tier: 3
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_generate,stan_instantiate,stan_shapes,transform,wrap
#=
**What it is.** Smoothers in the linear predictor: `s(x)` for a generic spline, `bs(x, knots=...)` for a B-spline basis, `gp(x)` for a Gaussian process, `t2(x, y)` for a tensor-product spline. brms / mgcv use these heavily.

**Why it matters.** Nonlinear effects without committing to a specific functional form. The de facto way to model dose-response curves, time effects, growth curves, spatial trends, ...

**Status: `s(x)` first-pass done in sbimpl.** Emits a natural-cubic-spline truncated-power basis with 2 interior knots at 1/3 and 2/3 quantiles of `x` (5 basis columns: x, x^2, x^3, (x - k1)^3_+, (x - k2)^3_+). The submodel `_sb_s` puts a `std_normal` prior on the basis coefficients and returns `X_basis * coefs` as a single length-N column that popefs multiplies by an overall beta. The basis is built inside the slic via `hcat` of per-column data kwargs (passing a raw Julia Matrix as a data kwarg currently triggers a StanBlocks type-inference bug -- see sbimpl.jl comment on `_sb_predictor_term!(::typeof(s), ...)`).

**Gaps.**
- `bs(x, knots=...)`, `t2(x, y)`, `gp(x)` dispatches are not yet added.
- No smoothness penalty on the coefficients -- `std_normal` only, so the fit is closer to unpenalized polynomial regression than to a properly regularized spline. Hooking a difference-penalty prior in wants (2.3 per-parameter priors) first.
- Extra overall `beta` from popefs is harmless but redundant; a direct-summand variant would drop it.

**Implementation.** Each smoother is a basis-matrix builder that grows a population block by `n_basis` columns and stores the basis matrix as part of `meta`. Function stubs (`s`, `bs`, `t2`, `gp`) already exist in `scripts/parsing.jl` so the parser side is partly done.

For each smoother type, the materialization is `basis_matrix * coefficients` (length-N output). The smoothness prior is a structured prior on the coefficients (typically a Gaussian prior with a banded or 2D-difference penalty matrix), which requires (2.3 per-parameter priors) as a prerequisite.

**Verification.** Preset against synthetic curve data. Compare fitted smoother against `mgcv::gam`.

=#

loc ~ 1 + s(a)
y1 ~ Normal(loc, 1)
