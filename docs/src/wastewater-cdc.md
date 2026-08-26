# The full CDC `ww-inference-model`

This page ports the complete CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model) — a **joint
wastewater + hospital-admissions** renewal model — onto BRM's StanBlocks backend.
Unlike the [reduced single-catchment example](/wastewater), this is the full model
from the upstream `model_definition.md`, and it is verified end to end (transpile +
`stanc` + finite BridgeStan density/gradient, dim ≈ 380) by
`test/cdc_ww_inference.jl`.

A **parallel StanBlocks-native port** of the same CDC model lives at
[StanBlocks — wastewater case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/wastewater):
same math, expressed in StanBlocks' own idiom (see the note on the subpopulation
axis below).

## Why this one is NOT the four-pane view

Every other model page uses the four-pane split — *`@brm` authoring → StanBlocks →
Stan → Turing*. That split is, by definition, a comparison of a **`@brm` formula
model** across backends. This model is not a `@brm` formula model, and cannot be:
its structure exceeds the formula DSL. Specifically —

- it has a **multi-level** shape: independent per-subpopulation renewal cells whose
  outputs are **aggregated** (population-weighted) and then feed a **second**
  observation stream (hospitalizations) at the jurisdiction level; and
- it carries several **latent parameter processes** — a differenced-AR weekly
  log-Rᵘ, per-subpopulation AR(1) deviations, an AR(1)-logit IHR, a day-of-week
  Dirichlet — that are not formula terms.

So it lives at BRM's **backend layer** (`@slic` + `@deffun` over StanBlocks), which
*can* host it. The honest rendering below is therefore the two panes that apply: the
exact model source, and the Stan it emits (whose `functions {}` block contains the
`@deffun` scan recipes). This is the concrete answer to "how well can BRM support
these models": the backend hosts the full model; the formula surface does not reach
it.

## The model

Four coupled pieces, faithful to `model_definition.md`:

1. **Infection process.** Reference-subpopulation weekly unadjusted log-Rᵘ follows a
   differenced autoregression; the realized reproduction number carries **infection
   feedback** `R(t)=Rᵘ(t)·exp(−γ Σ_τ I(t−τ)g(τ))`; incidence follows the renewal
   equation with a 50-day exponential seeding phase. (`dar_logru`,
   `renewal_feedback`.)
2. **Multi-subpopulation hierarchy.** K catchments (+ a reference) each carry
   `log Rᵘ_k = log Rᵘ_0 + m + δ_k(t)` with δ_k an AR(1) deviation (`ar1_dev`); each
   runs its own renewal, and jurisdiction incidence is the population-weighted
   aggregate `I(t)=Σ_k w_k I_k(t)`.
3. **Wastewater observation (per subpopulation).** Genome concentration is the
   shedding-kinetics convolution of that subpopulation's infections scaled by
   genomes-per-infection / per-person volume (`shed_convolve`); site-lab
   measurements are log-normal with a site scaling and a hierarchical observation
   SD, **left-censored below the limit of detection**.
4. **Hospital admissions (jurisdiction).** Expected admissions convolve aggregate
   infections with an infection-to-hospitalization delay (`delay_convolve`), scaled
   by a time-varying AR(1)-logit IHR and a day-of-week Dirichlet effect;
   **negative-binomial** counts.

The carried-state parts (renewal + feedback, the AR processes, all convolutions) are
`@deffun` Stan functions — `plate` broadcasts the per-subpopulation renewal +
wastewater cell over **independent** subpopulations, and the aggregate +
hospitalizations are ordinary top-level statements on the collected infection matrix.

The hidden setup below evaluates the `@deffun` helpers and the fixture from the
checked-in source before the model is rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    "research/wastewater/cdc_ww_inference.jl";
    before=:cdc_ww_inference_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

The declaration below is extracted verbatim from `cdc_ww_inference_model` in the
reviewed source, followed by the Stan it emits (the `@deffun` renewal / AR /
convolution scans appear in the generated `functions {}` block).

```@eval
Main.BRMDocsComparisons.slic_model_panes(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    Main.BRMDocsComparisons.source_function(
        "research/wastewater/cdc_ww_inference.jl", :cdc_ww_inference_model,
    ),
    :cdc_ww_inference_model;
    title="Full CDC ww-inference model (@slic backend)",
)
```

## Relation to the StanBlocks port

The [StanBlocks port](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/wastewater)
expresses the *same math* with the same module decomposition (a renewal scan, AR
processes, shedding/delay convolutions), and differs in **one** place — how the
subpopulation axis is written:

- **here (BRM):** `I_mat ~ plate(ww; outer=(K,)) do wwi … end`, one independent cell
  per subpopulation, aggregated as `I_agg = I_mat * w`. `plate` is BRM's idiomatic
  per-group construct, and it is *load-bearing*: the per-subpopulation **sampled**
  parameters (`eps_d`, `log_I0`, `logM_k`, …) are introduced inside the cell, which a
  `@deffun` cannot do (`@deffun` bodies have no `~`). This mirrors the CDC original's
  per-subpopulation *loop*.
- **StanBlocks:** an internal `@deffun` subpopulation loop returning a
  `tuple(matrix, matrix)` — closer to the CDC original's per-subpopulation *tuple
  return*.

Both are faithful; the difference is each package expressing the same loop in its
natural shape, and is intentional rather than an artifact.

## Provenance

Reproduction: [`research/wastewater/cdc_ww_inference.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/research/wastewater/cdc_ww_inference.jl),
gated by `test/cdc_ww_inference.jl`. The generation-interval, shedding-kinetics and
delay PMFs and the priors are illustrative discretizations, not calibrated values;
"faithful" refers to the CDC model's structure. Model artifact only — it is not
sampled here.
