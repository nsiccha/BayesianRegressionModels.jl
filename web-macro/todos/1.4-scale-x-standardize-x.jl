# label: 1.4 scale(x) / standardize(x)
# tier: 1
# status: open
#=
**What it is.** brms's `scale(x)` z-transforms a column at parse time: `scale(x) = (x - mean(x)) / std(x)`. The model sees the standardized column. Crucial for default priors (which are scale-invariant only after standardization) and sampler stability (well-conditioned linear predictors).

**Why it matters.** Most brms vignettes do `scale(x)` automatically as a convenience. Without it, every formula has to either manually z-transform the data or accept poorly-scaled coefficients.

**Implementation.**
1. Add `function scale end` (and `function center end`, `function standardize end`) to `macro.jl`.
2. Add a `vmeta_sampling_rhs` overload in `vimpl.jl`:
```julia
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(scale)}; group) = begin
    inner = vbroadcasted(only(getargs(x)); meta)
    materialized = Base.materialize(inner)
    z = (materialized .- Statistics.mean(materialized)) ./ Statistics.std(materialized)
    vmeta_sampling_rhs(meta, z; group)
end
```
The standardization happens once when the BRMI is materialized into a VBRMI. Composes with the existing dense-map caching TODO.
3. Add `Statistics` to `vimpl.jl`'s using-list (or vendor `mean`/`std` inline).

**Verification.** Preset: `loc ~ 1 + scale(a) + scale(b); y1 ~ Normal(loc, 1)`. Compare against the unscaled version: same dim, different posterior geometry. The fitted coefficients should be ≈ the unscaled coefficients × std(x).

=#

loc ~ 1 + scale(a) + scale(b)
y1 ~ Normal(loc, 1)
