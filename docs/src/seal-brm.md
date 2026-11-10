# Grey-seal IPM on the `@brm` formula surface

This page writes the full Baltic grey-seal integrated population model, a
state-space model of an age- and sex-structured population fitted to eight
observation streams, as one `@brm` declaration. It is a port of the
SlicTranspiler research model that the
[StanBlocks case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/grey-seal-ipm)
renders in StanBlocks' own idiom: every regression, random effect and
covariate becomes a brms-style formula, and the mechanistic core stays a
`@deffun` scan, as the [CDC ww-inference model](wastewater-cdc.md) keeps its
renewal scan. `test/seal_brm.jl` checks that the model transpiles, passes
`stanc` and compiles.

## What is a formula here

Hoisted from the upstream `@slic` parent onto the formula surface:

- the birth-rate covariate regression,
  `bbr_logit ~ 1 + herring_index_1 + herring_index_2` (the upstream
  `compute_baseline_birth_rate`, logistic in the herring indices), bounded
  inside the scan;
- per-year process and effort random effects with estimated group SDs,
  `eps_birth ~ 0 + (1|year)`, `eps_sex`, `eps_h_sw`, `eps_h_fi`, `eps_ca`,
  `eps_placental`. The upstream `epsilon_*` innovations have a fixed unit
  scale; here they are `(1|year)` random effects whose SD is estimated, which
  is why the `@brm` form has a few more parameters than the `@slic` one;
- per-demographic-class hunting selectivity and bycatch bias,
  `hs_sw ~ 0 + (1|demo)`, `hs_fi`, `bycatch_bias`;
- per-cell transition process noise, `tnoise ~ 0 + (1|noise_cell)`, a crossed
  cell random effect that the scan reshapes into the transition-noise matrix.

These live on four frames at once (`year`, `demo`, `noise_cell` and the
herring covariate frame), which the formula surface allows.

## What stays a `@deffun` scan

The carried-state recurrence: the verbatim `run_state_process` (Leslie aging,
size-structured mortality, density-dependent births, a within-year hunting
ODE solved with `ode_rk45_tol`, and a `multinomial_allocation`
logistic-normal split into survived, bycaught and hunted) with its twelve
helper functions. A `seal_state` wrapper runs the parent's transformed
quantities and the scan and returns a named tuple of carriers; each
observation reads a named field (`state.population_total`,
`state.hunted_sweden`, ...) through field access in the `@brm` body. One scan
serves all eight streams, as the `@slic` seal reads its `run_state_process`
struct.

## The eight observation streams

Each is a `@brm` family likelihood of a `state.<field>` indexed to that
stream's observation years, on its own frame:

- aerial pup counts: `NegativeBinomial2` on `state.population_total`;
- Swedish and Finnish harvest bags: `Normal` with a cv-scaled SD on the
  hunting-bag totals;
- Swedish and Finnish hunting age composition, bycatch composition and
  reproductive-signs composition: per-row `Multinomial` on the per-year
  simplices derived from the hunted, bycatch and reproductive carriers;
- pregnancy counts: `Binomial` on `state.pregnancy_rate`.

The hidden setup below evaluates the scan, its helpers and the fixture from
the checked-in source before the declaration is rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:seal_brm),
    "research/seal/grey_seal_brm.jl";
    before=:grey_seal_brm_model,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

The declaration is read from `grey_seal_brm_model`; the Stan program's
`functions {}` block carries the scan, the helpers and the composition
densities.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:seal_brm),
    Main.BRMDocsComparisons.source_function(
        "research/seal/grey_seal_brm.jl", :grey_seal_brm_model,
    ),
    :grey_seal_brm_model;
    title="Full grey-seal IPM (@brm formula surface)",
    require_stan=true,
)
```

## Provenance and scope

The script
[`research/seal/grey_seal_brm.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/seal/grey_seal_brm.jl)
is checked by `test/seal_brm.jl`. The thirteen state-process functions and
`run_state_process` are verbatim from the SlicTranspiler research model; the
fixture is its example dataset (3 age classes × 2 sexes × 3 state years). The
checks are the same as for the `@slic` reference: transpile, `stanc`,
compile. Like the reference, the model is not fitted on this page: the
scan's simplex constraints make a naive initial value degenerate for both
forms (the `@slic` reference is non-finite at the origin too).

## Run the checks

From the repository root, with the test environment bootstrapped:

```sh
julia --startup-file=no --project=test test/seal_brm.jl
```
