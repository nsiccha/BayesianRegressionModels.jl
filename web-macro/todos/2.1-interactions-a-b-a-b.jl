# label: 2.1 interactions a:b, a*b
# tier: 2
# status: open
#=
**What it is.** brms's `a:b` is the elementwise interaction term (a single coefficient multiplying `a[i] * b[i]`). `a*b` is the "main effects + interaction" shorthand: it desugars to `a + b + a:b`.

**Why it matters.** Interactions are the most commonly missed feature in regression DSLs. Without them, every model that needs `a:b` has to manually create the interaction column in the input DataFrame.

**Implementation.**
1. **Parser side.** Add a `:` case to `_x` so that `a:b` becomes `ExprColumn(:, NamedColumn(:a), NamedColumn(:b))` instead of falling through to a Symbol/Range parse error.
2. **Materialization side.** Add `vmeta_sampling_rhs(meta, x::ExprColumn{typeof(:)}; group)` that elementwise-multiplies the operands and dispatches to the float-vector path. For continuous × continuous it's a single coefficient on `a .* b`; for categorical × continuous it's `(k-1)` coefficients (one per non-reference level of the categorical, multiplied by the continuous); for categorical × categorical it's `(k₁-1)*(k₂-1)` coefficients via a 2D `_cat_lookup`.
3. **`a*b` desugaring.** At parse time in `_x`, rewrite `*` between formula terms as `+(a, b, :(a:b))`. This needs care because `*` also means multiplication elsewhere (e.g. `Normal(0, 2*sigma)`); the rewrite should only apply at formula-RHS top-level.

**Verification.** Presets exercising each interaction type:
- continuous×continuous: `loc ~ 1 + a + b + a:b; y1 ~ Normal(loc, 1)` → dim 4
- continuous×categorical: `loc ~ 1 + a + c1 + a:c1; y1 ~ Normal(loc, 1)` → dim 6 (1 + 1 + 2 + 2)
- categorical×categorical: `loc ~ 1 + c1 + c2 + c1:c2; y1 ~ Normal(loc, 1)` → dim 5 (1 + 2 + 1 + 2)
- shorthand: `loc ~ 1 + a*b; y1 ~ Normal(loc, 1)` should match `loc ~ 1 + a + b + a:b` exactly.

=#

loc ~ 1 + a + b + a:b
y1 ~ Normal(loc, 1)
