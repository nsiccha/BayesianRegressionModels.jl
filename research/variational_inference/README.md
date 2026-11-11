# Minibatch and streaming variational inference

Sources behind the docs page `docs/src/variational-inference.md`. All data are
simulated: counts at monitoring sites over repeated visits, with a site-specific
intercept and slope (`site_counts_model` in `svi.jl`).

| file | what |
| --- | --- |
| `svi.jl` | the data, the model, and the fits: SVI against NUTS on the full data (`fit`), the stream of windows with NUTS and batch SVI on the accumulated data as references (`stream`), and a full fit started from fits on four slices of the sites (`combine`). Writes the tables in `results/`. |
| `figures.jl` | AlgebraOfVega specifications built from `results/`; the documentation renders these as Vega-Lite figures. Running the file directly writes optional, ignored PNG previews. |
| `results/` | the tables every number on the page comes from. |

The docs build evaluates the part of `svi.jl` above `site_counts_model` (the data
simulation) and shows the model and the fitting calls read from the file, so keep
the fitting packages' `using` lines below the model and the displayed call sites
as they are.

Run from the repository root, in the test environment (`julia --project=test
test/setup_env.jl` builds it):

```sh
julia --threads=1 --project=test research/variational_inference/svi.jl probe
julia --threads=1 --project=test research/variational_inference/svi.jl fit,stream,combine /abs/cache-dir
julia --project=test research/variational_inference/figures.jl
```

Run Julia with one thread for the Enzyme-backed NUTS reference. The cache directory
keeps completed SVI fits, their data-only checkpoints, NUTS draws and diagnostics,
and receipts with the source hash, package versions and run time. Re-running a
section reads its cached fits and regenerates tables; it does not refit. The key
hashes the scientific function definitions, data, settings, seeds, harness version,
Julia version and fitting package versions. Comments, prose and summary writers
are excluded. A changed key requires an explicit new cache directory; changing the
NUTS draw count preserves the old reference under its old filename. Keep this
cache on the machine that computed it.

The streaming reference is fitted on each window's accumulated observations.
Late sites have no observations before arrival, so the last window cannot reuse
the full-data NUTS reference. `combine` reuses the full-data cold SVI and NUTS fits
from `fit`. Each section can run separately with the same cache directory.

Only small summary TSVs and receipts travel with this example. Each fit has its
own directory, receipt and `population.tsv` with the shared quantity schema.
Streaming windows sit below `stream_svi/`. Cross-fit comparison tables live at the results root.
Raw draws,
checkpoints and rendered images are gitignored. `receipt.tsv` is written before
the requested sections and refreshed on success; it records the recipe, settings,
versions, input digests and fit source hashes. Documentation builds read summaries
and never sample. These tables and the receipt form the eventual DocsFits summary;
the page currently uses committed tables until the shared adapter is published.
The first measured caches used a recipe-and-data key. Their receipts retain that
legacy key and record the explicit migration to the content key; no fit was rerun
or timing replaced during this migration.

The fit writes a Treebars progress snapshot to `progress.txt` in the cache
directory, unless the run environment supplies a progress-file destination.
