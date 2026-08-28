# Wastewater-based Rt inference (EpiSewer / `ww-inference-model`)

This page ports the semi-mechanistic renewal model behind
[EpiSewer](https://github.com/adrian-lison/EpiSewer) and the CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model) onto the
`@brm` formula surface, and renders it in the standard four-pane view: the exact
BRM authoring source, the StanBlocks model it emits, the generated Stan, and the
selected Turing model. The build reads the declaration directly from the
checked-in reproduction script, so the displayed BRM source cannot drift from
the executable source.

A **parallel StanBlocks-native port** of the EpiSewer library is at
[StanBlocks — EpiSewer case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/episewer);
the larger multi-stream structural port is on the
[CDC ww-inference page](wastewater-cdc.md).

The model has three EpiSewer modules:

1. **Infection model** — a renewal process. Latent incidence follows
   `I(t) = Rt(t) · Σ_{s} g(s) · I(t-s)`, seeded by a per-site initial level over
   the first window. `g` is the (fixed, known) generation-interval PMF, passed as
   **data** and captured into the kernel cell. The reproduction number is a smooth
   latent over time (here an `hsgp` prior on `log Rt`).
2. **Shedding-load model** — infections are convolved with a fecal shedding-load
   PMF `sh` (also data): `load(t) = Σ_{s} sh(s) · I(t-s+1)`.
3. **Measurement model** — observed log-concentration is Normal on the log load
   plus a load→concentration scale: `log C(t) ~ Normal(log load(t) + log_scale, sigma_obs)`.

Multiple sites share `sigma_obs` / `log_scale` and the smooth `log Rt`; each site
carries its own seeding. The renewal and shedding recurrences are **carried-state
sequential scans** — which `plate` deliberately cannot express — so they live in
`@deffun` Stan functions (`ww_renewal`, `ww_shed`), called loop-free from the
`kernel(...)` cell exactly as the PMX pages call `ode_rk45`.

The hidden setup below evaluates those two `@deffun` helpers and the fixtures from
the checked-in source before the declaration is rendered.

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

The generation-interval PMF `g` and the shedding PMF `sh` are ordinary data
columns referenced by name inside the `kernel(...)` do-block; the kernel-cell
data-vector capture registers them as shared Stan data, so they are re-bindable
without editing the model — no baked-in constants. A per-timepoint smooth `log Rt`
enters the cell as a `ragged(log_rt, time_site)` secondary axis; each site's
seeding is a `(1 | site)` random effect.

The declaration below is extracted verbatim from `wastewater_brm_model` in the
reviewed source. Its `kernel(...)` cell threads one ragged concentration-time
course per site through the renewal and shedding scans.

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

The Turing pane is intentionally retained even though this structural kernel is
outside the current Turing executor; its build-time construction error is part of
the comparison rather than being hidden.

## Provenance and scope

The reproduction lives at
[`research/wastewater/renewal.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/837f8018/research/wastewater/renewal.jl)
alongside its
[README](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/837f8018/research/wastewater/README.md).
The same file also carries a pure-StanBlocks `@slic` form (`wastewater_model` /
`wastewater_censored_model`, the latter left-censoring non-detects below a log
limit of detection) — the layer where EpiSewer's own hand-written Stan lives, and
where the generation-interval / shedding PMFs are likewise data. "Faithful" refers
to the EpiSewer / `ww-inference-model` renewal structure; the fixed epidemiological
PMFs and priors here are illustrative defaults, not a claim about any specific
deployment's calibrated values.

The model artifact is verified — transpile, `stanc`, and finite BridgeStan
density/gradient — by `test/wastewater_model.jl` (the `@slic` forms) and
`test/kernel_capture.jl` (the `@brm` form). It is not sampled here; the deliverable
is the model.

## Run the reproduction

After bootstrapping the repository's test environment:

```sh
julia --startup-file=no --project=test test/wastewater_model.jl
julia --startup-file=no --project=test test/kernel_capture.jl
```

Set `BRM_KERNEL_RUNTIME=0` to run lowering and `stanc` without BridgeStan
instantiation.
