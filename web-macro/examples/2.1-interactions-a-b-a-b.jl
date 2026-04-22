# label: 2.1 interactions a:b (cont x cont done; cat cases open)
# tier: 2
# status: partial (cont x cont done; cont x cat / cat x cat open)
#=
**Status: partial in vimpl.** Continuous-by-continuous interactions via `a:b`
are implemented -- one free beta times the elementwise product. Cont x cat and
cat x cat still error out (with a pointer to this TODO).

**Julia syntax quirk.** In Julia, `a:b` parses as `Expr(:call, :(:), :a, :b)`
-- i.e. the `Colon()` function applied to `a` and `b`. At BRMI-walk time it
lands as `ExprColumn{Colon}(a, b)`. Precedence is tighter than `+` but looser
than `*`, so `1 + a + a:b` behaves correctly. `a*b` / `*`-interaction sugar is
left out for now because `*` already means scalar/array multiplication in most
Julia contexts -- keeping its surface meaning avoids DSL ambiguity.

**Implementation.** `vmeta_sampling_rhs(meta, x::ExprColumn{Colon}; group)`
broadcasts the two operands, multiplies them elementwise, and hands off to
`_scale_by_beta` like any other continuous predictor. A small
`_check_cont_interaction` guard errors out if either operand is an integer /
categorical column.

**Open (cat expansions).** For cont x cat, allocate `(K-1)` slots multiplied
by the treatment dummies. For cat x cat, allocate `(K1-1)*(K2-1)` slots via a
2D level-index lookup. Both reuse the existing `_cat_broadcast` / `_cat_lookup`
machinery but need a slightly different indexing path.

**Verification presets.**
- cont x cont (working):  `loc ~ 1 + a + b + a:b; y1 ~ Normal(loc, 1)`   -> dim 4
- cont x cat  (open):     `loc ~ 1 + a + c1 + a:c1; y1 ~ Normal(loc, 1)` -> dim 6
- cat x cat   (open):     `loc ~ 1 + c1 + c2 + c1:c2; y1 ~ Normal(loc, 1)` -> dim 5

=#

loc ~ 1 + a + b + a:b
y1 ~ Normal(loc, 1)
