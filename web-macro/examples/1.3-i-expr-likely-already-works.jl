# label: 1.3 I(expr) — brms literal-escape
# tier: 1
# status: done (vimpl)
#=
**Status: done in vimpl.** Our DSL already supports arbitrary Julia function
calls on the RHS (e.g. `a^2`, `sqrt(abs(b))`, `log(exposure)`), so `I()` is
functionally unnecessary. But brms users who have internalized the `I()`
convention can now write it literally — we added a passthrough overload
(`vbroadcasted(::ExprColumn{typeof(I)}) = inner`) so `I(expr)` behaves the same
as `expr` alone, no parameter penalty.

**Why brms needs I().** In R, `a:b`, `a*b`, `|` etc. have DSL meaning inside a
formula — `I()` escapes them so the inner expression is interpreted as plain
arithmetic. Our formula DSL is parsed by Julia first, so `a^2` is already a
plain function call and `I()` is a no-op.

**Verification.** The form below mixes the "naked" spelling (`a^2`,
`sqrt(abs(b))`, `log(exposure)`) with I()-wrapped forms — the VBRMI dim,
gradient check, and materialized predictor values must be identical.

=#

loc ~ 1 + a + I(a^2) + I(sqrt(abs(b))) + log(exposure)
y1 ~ Normal(loc, 1)
