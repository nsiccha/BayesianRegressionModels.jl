# Historical BRM model inventory

This directory recovers the removed 2026-05 historical catalogue, preserves
its claims and receipts, translates each row conservatively to current BRM
semantics, and records staged compiler/runtime evidence. Historical-source
fidelity and current-BRM capability are intentionally separate axes.

## Reproduce

Run from a checkout that contains the historical Git object and whose current
environment has been resolved with the ecosystem's canonical local-dependency
resolver.

```sh
julia --startup-file=no research/historical_model_inventory/extract.jl \
  05c3f465e7987e8d7caa7e214fedddd90415a922
julia --startup-file=no research/historical_model_inventory/translate.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/probe.jl
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/probe.jl --runtime
julia --startup-file=no --project=web-macro \
  research/historical_model_inventory/runtime_controls.jl
julia --startup-file=no research/historical_model_inventory/assemble.jl
```

The static probe builds BRMI and SBBRMI objects, derives the executable
descriptor, generates Stan, and runs stanc. `--runtime` additionally compiles
each accepted unique program with BridgeStan and checks a deterministic density
and gradient. It does not sample every synthetic analogue. Sampling is reserved
for the real-data sleepstudy control and executable kernel/plate route controls
in `runtime_controls.jl`.

Receipt reproduction has separate instructions in `receipts/README.md` because
it performs timestamped network retrieval and requires an exact historical
source checkout.

## Evidence products

- `historical_catalog.tsv`: lossless source claims, source lines, formula
  return values, and hidden/deployed scope.
- `translations.tsv`: exact-metadata and inferred-family translations, each
  with provenance and an explicit ordinary/kernel/plate/unsupported route.
- `capability_results.tsv`: row-expanded stage evidence; inherited results are
  only from an identical normalized probe.
- `runtime_controls.tsv`: real-data sleepstudy plus executable BRM-kernel and
  StanBlocks-plate controls.
- `receipts/`: model and dataset retrieval evidence, conservative source
  fidelity verdicts, and reproduction scripts.
- `model_matrix.tsv`: one row per deployed historical model, combining the
  independent fidelity, translation, compiler/runtime, and semantic-mount axes.
- `final_summary_by_source.tsv`: final manual-overridden fidelity and current
  capability counts for each of the 19 historical sources.
- `all_source_rows_matrix.tsv`: the same schema including the one hidden
  non-Bayesian baseline row.
- `PROVENANCE.md`: exact historical/current/SbPMX branch, commit, path, and blob
  provenance.
- `REPORT.md`: findings, gaps, and interpretation boundaries.

## Interpretation boundaries

- An HTTP 200 is reachability, not semantic support.
- A generated program passing stanc is not a finite density or a correct model.
- A finite synthetic density is not a validated fit to the historical dataset.
- Family inference is reported beside, never in place of, exact metadata.
- Kernel/plate controls establish route capability only; they are not fanned
  onto historical rows without an exact row-specific implementation.
- The current SbPMX pattern is descriptor-driven, but the BRM-to-HTMXObjects
  mount adapter is not shipped. Mount projections here are experimental.
