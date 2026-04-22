# label: 1.7 CategoricalArrays / PooledArrays integration
# tier: 1
# status: open
#=
**What it is.** When the input column is already a `CategoricalVector` or a `PooledArray`, the dense level mapping is already computed and stored in the column's `.refs` field. Use it directly instead of rebuilding via `Dict`.

**Why it matters.** Most real-world DataFrames use `CategoricalArrays.jl` for factor columns. Skipping the rebuild eliminates allocation entirely for the common case and gets us "for free" interop with the standard categorical-data ecosystem.

**Implementation.** Two design choices:
- **Hard dep**: add `CategoricalArrays` to vimpl.jl's deps, dispatch on `CategoricalVector`, read `levelcode.(col)` and `levels(col)` directly.
- **Duck-typed**: sniff for the `.refs` field and `levels` method without importing the package, falling back to the generic Dict path.

Recommend hard dep — it's the standard for tabular Julia code, and the duck-type path is more code with no real win. Same for `PooledArrays`.

The actual integration is small once the design is picked: a method specialization in `_gc_idx` and in the categorical-predictor path. Composes with the caching TODO above.

**Verification.** Preset (or test) that builds a DataFrame with a `CategoricalVector` column and uses it as a grouping factor / categorical predictor. Confirm the gradient sanity check stays green and the per-VBRMI allocation count drops.

=#
