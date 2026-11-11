# Epidemic renewal models on the `@brm` formula surface

## Documentation fit

`docs/src/renewal.md` now teaches one 56-day renewal model with fixed generation and reporting delays. `fit.jl` runs four WarmupHMC chains through the shared MCMC harness and writes one fit under `results/docs/`: its receipt, natural-scale quantities with between-chain R-hat and bulk/tail ESS, expected-report bands and predictive intervals. These replace the earlier single-chain figures in the docs; the older multi-patch, forecast and delay experiments below remain research material. The related interval-censored, right-truncated delay likelihood is demonstrated separately in `research/epidist/`.

## Earlier renewal research

The research series extends the documentation example
(`docs/src/renewal.md`): renewal-equation models of daily case counts whose
statistical parts are formula lines (`log_R ~ 1 + rw(time) + cdar(week; by=patch, cor=C)`,
`log_I0 ~ 0 + offset(seed_mean) + factor(patch)`) and whose mechanics — renewal
recursion, gravity mixing, reporting-delay convolution — are Stan functions
declared with `StanBlocks.@deffun`. All data are simulated, so every estimate
is compared with a known truth.

## Research files

| File | Role |
|---|---|
| `renewal.jl` | Self-contained: simulation, Stan functions, the three `@brm` models (reporting delay, one population, six coupled patches), fit and summaries. The docs page reads its code blocks from this file at build time. |
| `renewal_figures.jl` | Draws the earlier research figures from `results/renewal/*.json` into `results/figures/` (AlgebraOfVega specs rendered with `vl-convert`). |
| `results/renewal/*.json` | Posterior summaries (quantile rows next to the truth), parameter table rows (`scalars.json`) and sampler diagnostics (`fits.json`) of the earlier research fits. The page now reads `results/docs/*.tsv`. |
| `../../test/renewal_docs_page.jl` | Evaluates the page's build-time blocks as Documenter does, `stanc`-checks the Stan panes, and checks a finite BridgeStan density and gradient for every model. No sampling. |

```sh
julia --startup-file=no --project=test research/epi_renewal/renewal.jl            # fits, about 20 minutes
julia --startup-file=no --project=<plot env> research/epi_renewal/renewal_figures.jl
julia --startup-file=no --project=test test/renewal_docs_page.jl
```

## Earlier stages

The other scripts in this directory (`fixtures.jl`, `single_patch.jl`,
`multi_patch.jl`, `recover.jl`, `prior_predictive.jl`, `delay_model.jl`,
`forecast.jl`, `vector_params.jl`, `wren16.jl`, `formula_lines.jl`, `plots.jl`,
`build_brief.py`) and the summaries directly under `results/` are the
development record of the same models in earlier spellings (explicit innovation
vectors, `kernel(...)` cells, a dedicated six-patch family). `renewal.jl`
supersedes them; they are not used by the documentation.
