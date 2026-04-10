# label: 3.6 splines / GP submodels s(), bs(), gp(), t2()
# tier: 3
# status: open
#=
**What it is.** Smoothers in the linear predictor: `s(x)` for a generic spline, `bs(x, knots=...)` for a B-spline basis, `gp(x)` for a Gaussian process, `t2(x, y)` for a tensor-product spline. brms / mgcv use these heavily.

**Why it matters.** Nonlinear effects without committing to a specific functional form. The de facto way to model dose-response curves, time effects, growth curves, spatial trends, …

**Implementation.** Each smoother is a basis-matrix builder that grows a population block by `n_basis` columns and stores the basis matrix as part of `meta`. Function stubs (`s`, `bs`, `t2`, `gp`) already exist in `scripts/parsing.jl` so the parser side is partly done.

For each smoother type, the materialization is `basis_matrix * coefficients` (length-N output). The smoothness prior is a structured prior on the coefficients (typically a Gaussian prior with a banded or 2D-difference penalty matrix), which requires (2.3 per-parameter priors) as a prerequisite.

**Verification.** Preset against synthetic curve data. Compare fitted smoother against `mgcv::gam`.

=#

loc ~ 1 + s(a)
y1 ~ Normal(loc, 1)
