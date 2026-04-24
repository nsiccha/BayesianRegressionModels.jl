# label: 2.2 configurable categorical reference level
# tier: 2
# status: open
# flag: sbbrm

loc ~ 1 + factor(c1, ref=2)
y1 ~ Normal(loc, 1)
