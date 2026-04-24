# label: 1.3 protect(expr) — brms literal-escape
# tier: 1
# status: done (vimpl)

loc ~ 1 + a + protect(a^2) + protect(sqrt(abs(b))) + log(exposure)
y1 ~ Normal(loc, 1)
