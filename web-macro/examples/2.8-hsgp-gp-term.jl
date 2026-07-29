# label: 2.8 HSGP hsgp(x; k, c) — Hilbert-space approximate Gaussian process
# tier: 2
# status: done (sbimpl)

loc ~ 1 + a + hsgp(b; k=20, c=1.5)
y1 ~ Normal(loc, 1)
