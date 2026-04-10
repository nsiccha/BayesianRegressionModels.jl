# label: 1.3 I(expr) — likely already works
# tier: 1
# status: done
#=
**What it is.** brms's `I()` is a literal-escape: `I(x^2)` says "compute `x^2` from the data and treat it as a single column". brms needs it because `+`, `*`, `:`, `|`, … all have special meaning inside an R formula.

**Why we probably don't need it.** Our DSL is parsed by Julia first, then walked by `_x`. `_x` recursively wraps every `Expr(:call, f, args...)` in an `ExprColumn`, regardless of whether `f` is special. So `loc ~ a + x^2` becomes `+(a, ^(x, 2))` → `ExprColumn(+, NamedColumn(:a), ExprColumn(^, NamedColumn(:x), 2))`. The `^` is just another function call, no special handling needed.

The only operators that have DSL meaning in our system are `~` (sampling), `=` (assignment), and `|` / `||` inside random-effects specs. Everything else (`^`, `/`, `sqrt`, `log`, `exp`, `mod`, `min`, `max`, …) is a regular function call resolved at materialization time via `vbroadcasted`.

**Verification.** The form below loads a model with three nonlinear terms (`a^2`, `sqrt(abs(b))`, `log(exposure)`) directly as population-level covariates. The VBRMI dim should match the number of distinct terms; the gradient sanity check should be all-active. If it works, that confirms `I()` is unnecessary because Julia function calls are first-class on the formula RHS.

=#

loc ~ 1 + a + a^2 + sqrt(abs(b)) + log(exposure)
y1 ~ Normal(loc, 1)
