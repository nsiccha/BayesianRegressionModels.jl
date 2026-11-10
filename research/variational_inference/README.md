# Minibatch and streaming variational inference

Sources behind the docs page `docs/src/variational-inference.md`. All data are
simulated: counts at monitoring sites over repeated visits, with a site-specific
intercept and slope (`site_counts_model` in `svi.jl`).

| file | what |
| --- | --- |
| `svi.jl` | the data, the model, and the fits: SVI against NUTS on the full data (`fit`), the stream of windows with NUTS and batch SVI on the accumulated data as references (`stream`), and a full fit started from fits on four slices of the sites (`combine`). Writes the tables in `results/`. |
| `figures.jl` | the figures of the docs page, drawn from `results/` into `docs/src/assets/variational-inference/`. |
| `results/` | the tables every number on the page comes from. |

The docs build evaluates the part of `svi.jl` above `site_counts_model` (the data
simulation) and shows the model and the fitting calls read from the file, so keep
the fitting packages' `using` lines below the model and the displayed call sites
as they are.

Run from the repository root, in the test environment (`julia --project=test
test/setup_env.jl` builds it):

```sh
julia --project=test research/variational_inference/svi.jl probe            # a 12-site smoke run, nothing written
julia --project=test research/variational_inference/svi.jl fit,stream,combine /abs/cache-dir
julia --project=test research/variational_inference/figures.jl
```

The cache directory keeps the NUTS draws of the `fit` section for `stream` and
`combine`; the sections can run one at a time with the same directory. On a fleet
host every run takes a compute token.
