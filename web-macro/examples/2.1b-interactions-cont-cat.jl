# label: 2.1b interactions cont x cat (sbimpl)
# tier: 2
# status: sbbrm
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,transform,wrap
#=
**cont x cat interaction on the sb backend.**

Sibling to `2.1-interactions-a-b-a-b.jl` (cont x cont) and `2.1c-interactions-cat-cat.jl` (cat x cat).

Treatment coding, reference level = 1. `a&c1` with K levels allocates K-1
columns, each `a .* (c1 == k ? 1 : 0)` for k=2..K. The main effect `c1`
is a separate term that still emits K-1 betas via the `_sb_cat`
direct-summand path.

Parameter dim: 1 (intercept) + 1 (a) + (K-1) (cat_c1) + (K-1) (int_a_c1).

Interactions with `mo(...)`, `me(...)`, `s(...)`, `ar(...)` error at walk
time (no broadcast expansion defined for them yet).
=#

loc ~ 1 + a + c1 + a&c1
y1 ~ Normal(loc, 1)
