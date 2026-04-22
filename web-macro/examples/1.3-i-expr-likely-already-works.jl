# label: 1.3 protect(expr) — brms literal-escape
# tier: 1
# status: done (vimpl)
#=
**Status: done in vimpl.** Our DSL already supports arbitrary Julia function
calls on the RHS (e.g. `a^2`, `sqrt(abs(b))`, `log(exposure)`), so `protect()` is
functionally unnecessary. But brms users who have internalized the `protect()`
convention can now write it literally — we added a passthrough overload
(`vbroadcasted(::ExprColumn{typeof(protect)}) = inner`) so `protect(expr)` behaves the same
as `expr` alone, no parameter penalty.

**Why brms needs protect().** In R, `a:b`, `a*b`, `|` etc. have DSL meaning inside a
formula — `protect()` escapes them so the inner expression is interpreted as plain
arithmetic. Our formula DSL is parsed by Julia first, so `a^2` is already a
plain function call and `protect()` is a no-op.

**Verification.** The form below mixes the "naked" spelling (`a^2`,
`sqrt(abs(b))`, `log(exposure)`) with protect()-wrapped forms — the VBRMI dim,
gradient check, and materialized predictor values must be identical.

=#

loc ~ 1 + a + protect(a^2) + protect(sqrt(abs(b))) + log(exposure)
y1 ~ Normal(loc, 1)
