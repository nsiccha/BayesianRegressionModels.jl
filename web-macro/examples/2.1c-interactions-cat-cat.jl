# label: 2.1c interactions cat x cat (sbimpl)
# tier: 2
# status: sbbrm
# flag: sbbrm
# stages_pass: brmi,parse,slic_model,transform,wrap
#=
**cat x cat interaction on the sb backend.**

Sibling to `2.1-interactions-a-b-a-b.jl` (cont x cont) and `2.1b-interactions-cont-cat.jl` (cont x cat).

Treatment coding, reference level = 1 for each operand. `c1&c2` with
levels K1, K2 allocates (K1-1)*(K2-1) indicator-product columns, one per
non-reference pair (k1, k2). Main effects `c1` and `c2` are separate
terms and emit (K1-1) + (K2-1) betas via the `_sb_cat` direct-summand
path.

Parameter dim: 1 + (K1-1) + (K2-1) + (K1-1)*(K2-1).
=#

loc ~ 1 + c1 + c2 + c1&c2
y1 ~ Normal(loc, 1)
