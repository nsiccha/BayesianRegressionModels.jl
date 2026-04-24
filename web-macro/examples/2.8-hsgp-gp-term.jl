# label: 2.8 HSGP gp(x; k, c) — Hilbert-space approx Gaussian process
# tier: 2
# status: done (vimpl); sb backend open

loc ~ 1 + a + gp(b; k=20, c=1.5)
y1 ~ Normal(loc, 1)
