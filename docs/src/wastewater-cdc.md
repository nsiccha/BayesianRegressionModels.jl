# The full CDC `ww-inference-model`

This page ports the complete CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model) — a **joint
wastewater + hospital-admissions** renewal model — onto the `@brm` **formula
surface**, and renders it in the standard four-pane view: the exact BRM authoring
source, the StanBlocks model it emits, the generated Stan, and the selected Turing
model. Unlike the [reduced single-catchment example](/wastewater), this is the full
multi-subpopulation model from the upstream `model_definition.md`, and it is verified
end to end (transpile + `stanc` + finite BridgeStan density/gradient, dim ≈ 160) by
`test/cdc_ww_inference.jl`.

A **parallel StanBlocks-native port** of the same CDC model lives at
[StanBlocks — wastewater case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/wastewater):
same math, expressed in StanBlocks' own idiom (see the note on the subpopulation
axis below).

## The full model on the `@brm` formula surface

This is a genuinely multi-level, multi-stream renewal model, and every coupled piece
is a formula-level statement — nothing here is authored one layer down. The seams:

- **Shared reference log-Rᵘ** — `log_ru ~ 1 + ar(time_grid; p=1)`, an autoregressive
  daily reproduction number, **closed over** inside the per-subpopulation cell (a
  top-level latent is in scope for a `kernel(...)` cell exactly as a top-level
  parameter is).
- **Multi-subpopulation hierarchy** — `I_mat ~ kernel(t_grid, ww, log_I0) do … end`
  broadcasts the per-subpopulation renewal + wastewater cell over the K catchments;
  the per-subpopulation AR(1) deviation `δ_k` is drawn *inside* the cell
  (`eps_d ~ std_normal()` + `ar1_dev`), and the `(1 | site)` seeding intercept
  derives the kernel's grouping. Each cell returns its infection trajectory, and the
  cells collect into a matrix `I_mat`.
- **Infection feedback + renewal** — `renewal_feedback` (a carried-state `@deffun`
  scan) inside the cell: `R(t)=Rᵘ(t)·exp(−γ Σ_τ I(t−τ)g(τ))`.
- **Per-subpopulation wastewater** — a **left-censored** log-normal `~` statement
  *inside* the cell, on that subpopulation's shedding-convolved infections.
- **Jurisdiction aggregate** — `I_agg = wsum(I_mat, w)`, the population-weighted sum
  across subpopulation columns (`I(t)=Σ_k w_k I_k(t)`).
- **Hospital admissions (jurisdiction)** — on the dense aggregate: a delay
  convolution, an AR(1)-logit IHR (`logit_ihr ~ 1 + ar(time_grid; p=1)`), a
  day-of-week effect (`log_dow ~ 0 + factor(dow)`), and a **negative-binomial**
  count likelihood.

Two small things live in `@deffun`s rather than the formula for good reasons, and
they are visible in the generated `functions {}` block, not hidden: the carried-state
sequential scans (`renewal_feedback`, `ar1_dev`, the shedding/delay convolutions),
which `plate`/`kernel` deliberately cannot express; and `wsum` — the cross-cell
population-weighted matmul — because `*` at the **formula** level is the Wilkinson
interaction operator, not matrix multiplication.

**One faithful re-parameterization** vs. the pure-backend `@slic` form (see
Provenance): the reference log-Rᵘ uses the built-in `ar(time; p=1)` autoregressive
term in place of the upstream **differenced**-AR, which needs a top-level innovation
vector the formula surface does not yet declare. It is an autoregressive-Rt prior in
the same spirit; the four coupled pieces above are structurally faithful.

The hidden setup below evaluates the `@deffun` helpers and the fixture from the
checked-in source before the declaration is rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    "research/wastewater/cdc_ww_inference.jl";
    before=:cdc_ww_brm_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

The declaration below is extracted verbatim from `cdc_ww_brm_model` in the reviewed
source, followed by the StanBlocks model it emits, the generated Stan (whose
`functions {}` block carries the `@deffun` renewal / AR / convolution scans), and the
Turing pane.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:cdc_ww),
    Main.BRMDocsComparisons.source_function(
        "research/wastewater/cdc_ww_inference.jl", :cdc_ww_brm_model,
    ),
    :cdc_ww_brm_model;
    title="Full CDC ww-inference model (@brm formula surface)",
    require_stan=true,
)
```

The Turing pane is intentionally retained even though this structural kernel is
outside the current Turing executor; its build-time construction error is part of the
comparison rather than being hidden.

## The four coupled pieces, faithful to `model_definition.md`

1. **Infection process.** Reference-subpopulation daily log-Rᵘ is autoregressive; the
   realized reproduction number carries **infection feedback**
   `R(t)=Rᵘ(t)·exp(−γ Σ_τ I(t−τ)g(τ))`; incidence follows the renewal equation with an
   exponential seeding phase. (`renewal_feedback`.)
2. **Multi-subpopulation hierarchy.** K catchments each carry
   `log Rᵘ_k = log Rᵘ_0 + δ_k(t)` with δ_k an AR(1) deviation (`ar1_dev`); each runs
   its own renewal, and jurisdiction incidence is the population-weighted aggregate
   `I(t)=Σ_k w_k I_k(t)` (`wsum`).
3. **Wastewater observation (per subpopulation).** Genome concentration is the
   shedding-kinetics convolution of that subpopulation's infections scaled by a site
   scaling (`shed_convolve`, `log_scale`); measurements are log-normal,
   **left-censored below the limit of detection**.
4. **Hospital admissions (jurisdiction).** Expected admissions convolve aggregate
   infections with an infection-to-hospitalization delay (`delay_convolve`), scaled by
   a time-varying AR(1)-logit IHR and a day-of-week `factor` effect;
   **negative-binomial** counts.

## Relation to the StanBlocks port and the `@slic` form

The same reproduction file also carries a pure-StanBlocks **`@slic`** form
(`cdc_ww_inference_model`) — the backend layer, where the reference log-Rᵘ uses the
exact differenced-AR (`dar_logru`) and the per-subpopulation cell is a `plate`. The
[StanBlocks port](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/wastewater)
expresses the *same math* with the same module decomposition (a renewal scan, AR
processes, shedding/delay convolutions) and the **same subpopulation-axis idiom**:
`plate` over the K subpopulations (`I_mat ~ plate(…) do … end`, then `I_mat * w`), with
the within-subpopulation time recurrence in a `@deffun` the cell calls. `plate` is
load-bearing there for the same reason `kernel(...)` is here: the per-subpopulation
**sampled** parameters (`eps_d`, `log_I0`, …) are introduced inside the cell, which a
`@deffun` cannot do. So the `@brm` form, the `@slic` form, and the StanBlocks port are
structurally aligned.

## Provenance

Reproduction: [`research/wastewater/cdc_ww_inference.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/research/wastewater/cdc_ww_inference.jl),
gated by `test/cdc_ww_inference.jl` (both the `@slic` and the `@brm` forms:
transpile + `stanc` + finite BridgeStan density/gradient, plus re-bind to a different
K / horizon). The generation-interval, shedding-kinetics and delay PMFs and the priors
are illustrative discretizations, not calibrated values; "faithful" refers to the CDC
model's structure. Model artifact only — it is not sampled here.

## Run the reproduction

After bootstrapping the repository's test environment:

```sh
julia --startup-file=no --project=test test/cdc_ww_inference.jl
```

Set `BRM_KERNEL_RUNTIME=0` to run lowering and `stanc` without BridgeStan
instantiation.
