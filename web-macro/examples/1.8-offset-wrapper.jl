# label: 1.8 offset(x) wrapper — brms-compatible fixed-slope term
# tier: 1
# status: done (vimpl)
#=
**Status: done in vimpl.** Sibling 1.2 showed the functional pattern (put
`log(exposure)` directly inside the likelihood expression). This adds the
brms-style `offset(x)` wrapper on the RHS of a linear predictor, for users
who prefer the familiar formula-language spelling.

**Semantics.** `loc ~ 1 + a + offset(z)` allocates one population intercept
and one slope on `a`, but NO parameter for `z` — the raw `z` vector is added
directly to the predictor (fixed slope = 1). Equivalent to writing
`loc ~ 1 + a` and then manually `loc .+ z` before the likelihood, just inline.

**Implementation sketch.** Single parser method in vimpl.jl:
`vmeta_sampling_rhs(meta, x::ExprColumn{typeof(offset)}; group)` returns
`vbroadcasted(inner; meta)` without going through `_scale_by_beta`. No Part,
no parameter slot. Pop-level only (brms restriction); group-level is unusual
and would need a walker tweak.

**Verification.** Equivalent Poisson-with-exposure model spelled two ways:
the VBRMI dim, gradient check, and materialized log_rate values should all
match the sibling 1.2 version bit-for-bit.

=#

log_rate ~ 1 + a + offset(log(exposure))
k1 ~ Poisson(exp(log_rate))
