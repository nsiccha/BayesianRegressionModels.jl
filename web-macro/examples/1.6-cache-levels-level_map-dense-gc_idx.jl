# label: 1.6 cache levels / level_map / dense / gc_idx
# tier: 1
# status: open
#=
**What it is.** Stop rebuilding the dense level mapping (`Dict(level => row_index)`) and the gc_idx vector on every `VBRMI(brmi)` call. Cache them once per source data column.

**Why it matters.** Today every `VBRMI` build re-traces the categorical / grouping columns, sorts unique values, builds a Dict, and walks the column to dense-encode it. For models with many categorical columns or many `VBRMI` rebuilds (e.g. during AD), this adds up.

**Implementation.** Pick a storage layout for the per-column metadata. Two candidates:
- A new `meta.factor` NamedTuple keyed by source column name, holding `(; levels, level_map, dense, gc_idx)` per column. Built lazily on first reference, indexed via `name(column)`.
- Attach the metadata to `meta.materialized[column_name]` directly. More tightly coupled but avoids a parallel NamedTuple.

The TODO already lives at `vimpl.jl:78–86`. Once a layout is picked, refactor `_gc_idx` and the inline dense map in the categorical path to read from the cache, falling back to a build-on-miss helper.

**Verification.** No behavioral change — the gradient sanity check should stay green. Benchmark `VBRMI(brmi)` with Chairmarks before/after and confirm a measurable speedup on a model with multiple categorical/grouping columns.

=#
