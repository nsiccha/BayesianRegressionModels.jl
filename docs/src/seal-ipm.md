# Grey-seal integrated population model (IPM)

This page ports the **grey-seal IPM** — the spotlight example from
[SlicTranspiler](https://github.com/nsiccha/SlicTranspiler) — onto BRM's StanBlocks
backend. It is a Baltic grey-seal integrated population model: a joint state-space
model of the population's age/sex structure fit to eight heterogeneous observation
streams. It is the most complete demonstration that BRM's backend hosts a real,
research-grade state-space model, and it is verified (assembled + `stanc`-clean, ~52 KB
of Stan) by `test/seal_ipm.jl`.

Like the [full CDC ww-inference model](/wastewater-cdc), it is authored one layer down
— directly on BRM's StanBlocks backend (`@slic` + `@deffun`) rather than the `@brm`
formula surface — because the multi-stream, parameter-carrying state-space shape isn't
something today's formula terms compose directly. That is an authoring choice about the
current surface (which is extensible), not a fundamental limit; there is simply no
`@brm` source pane, so the view below is the two panes that apply: the model source and
the Stan it emits.

A **parallel StanBlocks-native port** of the same spotlight is at
[StanBlocks — grey-seal IPM case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/grey-seal-ipm):
same verbatim SlicTranspiler source, so the two ports are faithful to each other and to
the upstream.

## The model

A **year-recursive, age/sex-structured** population process, observed through eight
streams:

- **State process** (`run_state_process`, a `@deffun` scan over years): each year ages
  the population through a Leslie matrix, applies size-structured mortality and a
  density-dependent birth rate, solves the within-year hunting dynamics with an ODE
  (`ode_rk45_tol` on `dH_dt`), and allocates each cohort's fate — survived / bycatch /
  hunted-Sweden / hunted-Finland — by a `multinomial_allocation` (a logistic-normal
  approximation to the Dirichlet-multinomial). It carries population totals, hunting
  bags, bycatch expectations, pregnancy and reproductive-signs probabilities forward.
- **Observation streams** (Form-A submodels + direct family calls on the state
  carriers): aerial pup counts (**negative binomial**), Swedish/Finnish harvest bags
  (**normal**), Swedish/Finnish hunting age-composition (**multinomial**), bycatch
  composition (**multinomial**), pregnancy counts (**binomial**), and reproductive-signs
  composition (**multinomial**). Deleting one `~` line deactivates a whole stream (and a
  submodel node drops its own parameters) — the modular "one removable node per stream"
  shape.

It is authored as inert SLIC source — a parent `@slic` body + 31 `@deffun` UDF cards +
2 anonymous observation-stream submodels — and assembled with StanBlocks'
`compile_slic_bundle`. The parent body is shown below; the 31 `@deffun` cards (the
state-process scan, the ODE right-hand side, the multinomial allocation, and the
observation-family densities) appear in the generated Stan `functions {}` block.

The hidden setup below loads the model source + example data from the checked-in
reproduction and builds it.

```@eval
Main.BRMDocsComparisons.bundle_model_panes(
    Main.BRMDocsComparisons.example_module(:grey_seal),
    "research/seal/grey_seal_ipm.jl",
    :GREY_SEAL_IPM_SOURCE,
    :grey_seal_ipm_bundle;
    title="Grey-seal IPM — parent @slic body, and the Stan it emits",
)
```

## Provenance

Verbatim port of
[`SlicTranspiler/src/grey_seal_ipm.jl`](https://github.com/nsiccha/SlicTranspiler)
(the parent body + 31 UDF cards + 2 observation-stream submodels) and its example
dataset (3 age classes × 2 sexes × 3 state years). The BRM side is only the fixture and
the `compile_slic_bundle` builder; see
[`research/seal/grey_seal_ipm.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/research/seal/grey_seal_ipm.jl),
gated by `test/seal_ipm.jl`. The model is a real research IPM; it is assembled and
`stanc`-verified here, not sampled.
