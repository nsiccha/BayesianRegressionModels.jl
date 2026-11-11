# Synthetic station field

This directory supplies the data, model, fit and summaries for the
[spatial field example](../../docs/src/spatial-fields.md). The
[spatial guide](../../docs/src/space-time-fields.md) explains the other field
representations. The data are entirely synthetic.

## Reproduce

From the repository root:

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/space_time_fields/fields.jl
```

Then build the site as `docs/README.md` describes.

The test setup pins the model, sampler and differentiation dependencies.
Use Julia 1.10, as the documentation CI does.

`fields.jl` generates the station data, fits the model once, and writes small
TSV summaries. `figures.jl` builds AlgebraOfVega specifications during the
documentation build. It does not fit, write images, or write Vega-Lite files.
A missing summary is a build failure, not an instruction to sample in CI.

`receipt.tsv` is written before sampling and completed after all summaries
have been written. It names the content key, source commit, working-tree
status, start time, environment, seed, settings and sampler outcome. The tuple
hashed by
`DynamicObjects.canonical_hash` is `(program, data, settings, seed, harness_version)`,
matching DocsFits 0.1; its hash can differ across Julia minor versions.
The program is the generated Stan form of the same BRM declaration.

The transitional TSV format is used while DocsFits is unpublished. Its
quantity table already has the DocsFits columns, and `fit_station_field`
returns a coordinate × draw × chain array plus sampler information.
`summarize_station_field` accepts those draws and information and returns named
tables. Adopting the shared adapter replaces the transitional persistence;
it does not require a second scientific model or another set of figure data.

Optional interrupted-run checkpoints stay outside git:

```sh
FIELD_CHECKPOINT_DIR=/path/to/local/checkpoints \
  julia --startup-file=no --project=test research/space_time_fields/fields.jl
FIELD_CHECKPOINT_DIR=/path/to/local/checkpoints FIELD_RESUME=1 \
  julia --startup-file=no --project=test research/space_time_fields/fields.jl
```

Use a checkpoint directory only with the same program, data and settings.
The fit script never commits or publishes its outputs automatically.

The sampled draws are saved before any table is written, to
`$BRM_DOCS_CACHE/space_time_fields/<key>/draws.jls` (`BRM_DOCS_CACHE` defaults
to `brm-docs-examples` below the system temporary directory). A second run
under the same key rebuilds the tables from them without sampling; the page
never reads that file. A run under a different key within 30 days of the
receipt's start time stops unless `BRM_DOCS_FIT_FORCE=1` is set: a docs fit
is rerun when its model, data or settings change, at most about once a month.

## Summary tables

Every table must be smaller than 100 KB. Quantiles and diagnostics are
computed from the full-precision draws, then rounded for display: field values
to 0.01 response units, field SDs to 0.001, and each quantity to two
significant digits of its 95% interval's width. Draw arrays, checkpoints and
generated images are not committed.

| File | Content |
| --- | --- |
| `data.tsv` | Generated counts, domain, times, neighbor setting and noise SD |
| `field_grid.tsv` | `east,north,time,truth,q025,q25,q50,q75,q975,sd` |
| `observations.tsv` | `station,east,north,time,y,truth` |
| `quantities.tsv` | Noise SD, field SD and both lengths as `quantity,unit,description,median,q025,q975,rhat,ess_bulk,ess_tail` |
| `diagnostics.tsv` | Chain counts, omissions, truncation, coordinate diagnostics and divergences |
| `map_checks.tsv` | Grid RMSE, pointwise coverage and mapping draw count |
| `receipt.tsv` | Content key and run provenance as `key,value` rows |

The figure functions also accept row tables directly, so a plotting check can
use an explicitly synthetic fixture without writing it into `results/`.

## Simulation and inference

An analytic pair of moving Gaussian features defines the generating field.
The field is not drawn from the inference prior. Stratified station locations,
replicate measurements and independent Gaussian measurement noise are
reproducible from the data seed in `station_data`.

The fitted `neighbors=` GMRF uses position–time reads, a Matérn-1 spatial
covariance and OU temporal dependence. BRM's spatial length ℓ sets
`kappa = sqrt(2) / ℓ`, so the spatial correlation falls to about 0.14 at a
distance of 2ℓ; the temporal length is the exponential correlation time. Amplitude, both lengths and observation-noise
SD are inferred. The source function `station_field_model` is the single model
used by the fit and the build-generated BRM/IR/StanBlocks/Stan panes.

WarmupHMC uses independent RK density/gradient workspaces, prior-median
initialization and a completion quorum. The returned draw array contains equal
prefixes of retained chains, with their original lengths and omitted indices
in the receipt. Cross-chain diagnostics and the quantity table use the equal
prefixes. The divergence count covers all draws in retained chains, including
those beyond the prefixes. Runtime-based chain selection can introduce bias
that retained-chain diagnostics cannot detect.

Mapping uses an equal, evenly spaced subset of each stored chain.
`brm_gmrf_predict` receives a random-number generator so each new point includes
conditional field variation. New points are conditionally independent given
the fitted states; they have the intended marginal laws but not their joint
spatial covariance. The maps and slice therefore support pointwise uncertainty,
not uncertainty in spatial totals or simultaneous coverage claims.

A grid RMSE and coverage calculation describe this dataset and grid only.
They are not a repeated-simulation calibration study. More grid points do
not improve the neighbor approximation; changing its budget requires refitting.

## Validate

```sh
julia --startup-file=no --project=test test/space_time_fields_docs_page.jl
```

The page check executes its build-time blocks, validates generated model panes
and Stan syntax, and checks that the figures embed Vega-Lite. Run it after the
fit summaries exist, against the same shared docs helper as the integrated site.
The site review also checks the rendered page in light and dark modes, including
figure hover and generated-model panes.
