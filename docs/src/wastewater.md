# Wastewater-based Rt inference (EpiSewer / `ww-inference-model`)

This page writes the semi-mechanistic renewal model behind
[EpiSewer](https://github.com/adrian-lison/EpiSewer) and the CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model) as one
`@brm` declaration and shows what it emits. The build reads the declaration
from the checked-in script, so the page cannot drift from the code.

A StanBlocks-native port of the EpiSewer library is at
[StanBlocks: EpiSewer case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/episewer),
and the larger multi-stream structural port of the CDC model is on the
[CDC ww-inference page](wastewater-cdc.md).

The model has EpiSewer's three modules.

1. Infections follow a renewal process, `I(t) = Rt(t) · Σ_s g(s) · I(t − s)`,
   seeded by a per-site initial level over the first window. `g` is the
   known generation-interval PMF, passed as data and captured into the plate
   cell. The reproduction number is a smooth latent over time, here an
   `hsgp` prior on `log Rt`.
2. Shedding: infections are convolved with a fecal shedding-load PMF `sh`,
   also data, `load(t) = Σ_s sh(s) · I(t − s + 1)`.
3. Measurement: the observed log concentration is normal around the log load
   plus a load-to-concentration scale,
   `log C(t) ~ Normal(log load(t) + log_scale, sigma_obs)`.

The sites share `sigma_obs`, `log_scale` and the smooth `log Rt`; each site
has its own seeding. The renewal and shedding recurrences carry state from
one day to the next, which a plate cell cannot express, so they are the
`@deffun` Stan functions `ww_renewal` and `ww_shed`, called from the cell as
the PK/PD pages call an ODE solver.

The hidden setup below evaluates those two helpers and the data fixtures from
the checked-in script before the declaration is rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:wastewater),
    "research/wastewater/renewal.jl";
    before=:wastewater_brm_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

## The model

`g` and `sh` are data columns the cell reads by name; the cell's data-vector
capture registers them as shared Stan data, so they can be rebound without
editing the model. The smooth `log Rt` lives on the flat time frame
(`time_x`, one row per site and day): `rows = groups(time_site => site)` binds
each site's rows of it, and the cell reads `log_rt[rows[i]]`. Each site's
seeding is the `(1 | site)` random effect of `log_I0`.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:wastewater),
    Main.BRMDocsComparisons.source_function(
        "research/wastewater/renewal.jl", :wastewater_brm_model,
    ),
    :wastewater_brm_model;
    title="Wastewater renewal Rt-inference (EpiSewer / ww-inference)",
    require_stan=true,
)
```

## Provenance and scope

The script
[`research/wastewater/renewal.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/wastewater/renewal.jl)
and its
[README](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/wastewater/README.md)
also carry a pure-StanBlocks `@slic` form (`wastewater_model` and
`wastewater_censored_model`, the latter left-censoring non-detects below a
log limit of detection), the layer where EpiSewer's own hand-written Stan
lives. The renewal structure is EpiSewer's and the CDC model's; the PMFs and
priors here are illustrative defaults, not a deployment's calibrated values.

`test/wastewater_model.jl` (the `@slic` forms) and `test/kernel_capture.jl`
(the `@brm` form) check that the models transpile, pass `stanc`, and give a
finite BridgeStan density and gradient. The model is not fitted on this page.

## Run the checks

From the repository root, with the test environment bootstrapped:

```sh
julia --startup-file=no --project=test test/wastewater_model.jl
julia --startup-file=no --project=test test/kernel_capture.jl
```

`BRM_KERNEL_RUNTIME=0` runs lowering and `stanc` without BridgeStan.
