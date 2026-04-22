# label: 1.4 scale(x) / center(x) / standardize(x)
# tier: 1
# status: done (vimpl)
#=
**Status: done in vimpl.** Three data-transform wrappers that z-transform (or
just center) a column at VBRMI-materialization time. brms does `scale(x)`
automatically in most vignettes for sampler stability + prior scale-invariance;
making it available as a formula-level wrapper avoids forcing users to
pre-transform their DataFrame columns.

**Semantics.**
- `center(x)`      -> `x - mean(x)`
- `scale(x)`       -> `(x - mean(x)) / std(x)`
- `standardize(x)` -> alias for `scale(x)`

Each fires inside `vbroadcasted`, so transforms compose with every downstream
consumer: pop predictor, ranef slope, link functions, interactions, etc.
`scale(a):scale(b)` works as expected (z-transformed both operands before the
elementwise product).

**Implementation.** Three-line `vbroadcasted` overloads in vimpl.jl using
inlined `_mean`/`_std` helpers (no new dependency). The transform evaluates
once at VBRMI construction and caches the resulting vector.

**Verification.** Same model spelled two ways:
- `loc ~ 1 + scale(a) + scale(b)` -- standardized at formula level
- `loc ~ 1 + a_z + b_z` (where a_z, b_z were pre-standardized in the DataFrame)

VBRMI dim + log-density should be identical; fitted coefficients ≈ the
unscaled coefficients × `std(x)`.

=#

loc ~ 1 + scale(a) + center(b)
y1 ~ Normal(loc, 1)
