# Formula terms

## Penalized smooth: `s(x)`

[`s`](@ref) adds a penalized one-dimensional thin-plate regression spline to a
linear predictor. It is available only in the `SBBRMI` StanBlocks backend.

```julia
using BayesianRegressionModels, Distributions

x = collect(range(-2, 2; length=50))
df = (; x, y=sin.(x))

brmi = @brm df begin
    y ~ Normal(mu, sigma)
    mu ~ s(x)
    sigma ~ Exponential(1)
end

sb = SBBRMI(brmi)
```

The supported public call has exactly one numeric predictor and no keyword
arguments. Training values must be finite and contain at least 10 unique
values. Internally, `s(x)` uses a fixed rank-10 basis: two unpenalized
null-space columns for `{1, x}` and eight penalty-whitened range columns. The
range coefficients share a smoothing standard deviation with a standard
half-normal prior. The term contributes this complete smooth directly to the
linear predictor, so it does not receive an additional population coefficient.

For prediction or posterior replay on new data, the default
`reprocess(sb, new_df)` and `restan_data(sb, new_df)` calls evaluate `x` against
the frozen training centers and basis. Passing `freeze_constants=false`
re-estimates the basis from the new data and therefore has fresh-fit rather
than prediction semantics.

### Difference from `bs(...)`

Bambi/Formulae formulas such as `bs(x, knots=knots)` create a deterministic
B-spline design matrix whose dimension and knots are controlled by the formula.
BayesianRegressionModels' `s(x)` instead represents a penalized thin-plate
smooth with the fixed rank and smoothing prior described above. They are not a
one-for-one syntax translation: `s(x; k=...)`, `s(x; knots=...)`, `bs(...)`, and
`t2(...)` are not supported by this term.
