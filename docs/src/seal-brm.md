# Grey-seal IPM on the `@brm` formula surface

The [verbatim SlicTranspiler grey-seal IPM](/seal-ipm) is a research-grade state-space
model authored on BRM's StanBlocks backend (`@slic` + `@deffun`): a Leslie matrix, a
within-year hunting ODE, a multinomial-allocation, and eight observation streams. This
page is the companion demonstration that the seal's **regression / random-effect /
time-series / multi-stream structure is a `@brm` formula model** — every such piece is
a formula-level statement — with the mechanistic demographic dynamics in a `@deffun`
scan, exactly as the [CDC ww-inference model](/wastewater-cdc) keeps its renewal scan in
a `@deffun`. It is rendered in the standard four-pane view and verified (transpile +
`stanc` + finite BridgeStan density/gradient) by `test/seal_brm.jl`.

The dynamics here are **illustrative** (a compact pup/juvenile/adult projection), the
same way the CDC port uses illustrative PMFs; "faithful" refers to the *structure*, not
to the upstream's calibrated demography. The full research model stays in its `@slic`
form.

## Each seam is a formula statement

- **Birth-rate covariate regression** — `eta_birth ~ 1 + herring_index`. The upstream
  seal's `compute_baseline_birth_rate` is a bounded logistic regression of the birth
  rate on a herring-index covariate; that is a formula linear predictor.
- **Per-year random effect** — `re_birth ~ 0 + (1 | year)`. The upstream `epsilon_*`
  non-centered per-year process innovations are `(1|year)` random effects; the
  mechanistic scaling is applied where they are *consumed* in the scan, not in the prior.
- **Mechanistic age-structured scan** — `state = seal_state(...)`, a `@deffun` that
  runs the demographic recurrence ONCE and returns a **NamedTuple** of carriers, each
  observation reading a **named field** (`state.population_total`, …) — the same shape
  the `@slic` seal uses (`run_state_process` returns a struct). This is `@brm` body
  **field access** on a `@deffun` struct return.
- **Standard-family observations** — `NegativeBinomial2` aerial pup counts and a
  `Binomial` pregnancy stream, on scan carriers.
- **Per-row composition observation** — `Multinomial(comp_sample, state.composition)`,
  a **per-year** age-composition simplex (`matrix[T, 3]` carrier). This is the seal's
  hunting / reproductive-signs composition shape: a `int[T, K]` count response with a
  per-row simplex, which the `Multinomial` family lowers to a per-row multinomial.

The mechanistic core (the Leslie projection, density dependence, and — in the full
research model — the hunting ODE and multinomial-allocation) stays a `@deffun` scan.
That is the right home for a carried-state recurrence, not a limitation: it is the same
pattern CDC's renewal uses on the `@brm` surface.

The hidden setup below evaluates the `@deffun` scan and the fixture from the checked-in
source before the declaration is rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:seal_brm),
    "research/seal/grey_seal_brm.jl";
    before=:grey_seal_brm_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

The declaration below is extracted verbatim from `grey_seal_brm_model` in the reviewed
source, followed by the StanBlocks model it emits, the generated Stan (whose `functions
{}` block carries the `@deffun` scan and the `brm_multinomial` composition density), and
the Turing pane.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:seal_brm),
    Main.BRMDocsComparisons.source_function(
        "research/seal/grey_seal_brm.jl", :grey_seal_brm_model,
    ),
    :grey_seal_brm_model;
    title="Grey-seal IPM (@brm formula surface)",
    require_stan=true,
)
```

The Turing pane is intentionally retained even though this structural model is outside
the current Turing executor; its build-time construction error is part of the comparison
rather than being hidden.

## Relation to the `@slic` port

The [verbatim `@slic` grey-seal IPM](/seal-ipm) and its
[StanBlocks-native case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/grey-seal-ipm)
carry the full research model (ODE, multinomial-allocation, all eight streams). This
`@brm` form is the structural companion — it shows the regression, random-effect,
time-series and multi-stream skeleton on the formula surface — not a line-by-line port.

## Provenance

Reproduction: [`research/seal/grey_seal_brm.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/research/seal/grey_seal_brm.jl),
gated by `test/seal_brm.jl` (transpile + `stanc` + finite BridgeStan density/gradient,
plus re-bind to a different horizon). Illustrative dynamics; "faithful" refers to the
IPM's structure. Model artifact only — it is not sampled here.

## Run the reproduction

After bootstrapping the repository's test environment:

```sh
julia --startup-file=no --project=test test/seal_brm.jl
```

Set `BRM_KERNEL_RUNTIME=0` to run lowering and `stanc` without BridgeStan instantiation.
