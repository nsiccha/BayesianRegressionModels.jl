# label: GP (HSGP)
# tier: 0
# status: open

loc ~ 1 + a + gp(b; k=20, c=1.5)
y1 ~ Normal(loc, 1)
