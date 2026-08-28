# Grey-seal IPM on the `@brm` formula surface

This page ports the **full** Baltic grey-seal integrated population model — a joint
state-space model of an age/sex-structured population fit to **eight** heterogeneous
observation streams — onto the `@brm` **formula surface**. It is a faithful port of the
verbatim SlicTranspiler research model (the same one rendered as a
[StanBlocks-native case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/grey-seal-ipm)):
every regression / random-effect / covariate piece is hoisted onto a brms-style
formula, while the mechanistic core stays a `@deffun` scan — exactly as the
[CDC ww-inference model](wastewater-cdc.md) keeps its renewal scan in a `@deffun`. It is
rendered in the standard four-pane view and verified (transpile + `stanc` + `compiles`)
by `test/seal_brm.jl`.

## What is a brms formula here

Hoisted from the upstream `@slic` parent onto the formula surface:

- **Birth-rate covariate regression** — `bbr_logit ~ 1 + herring_index_1 + herring_index_2`
  (the upstream `compute_baseline_birth_rate` logistic-on-herring), bounded inside the
  scan.
- **Per-year process/effort random effects, with *estimated* group SDs** —
  `eps_birth ~ 0 + (1|year)`, `eps_sex`, `eps_h_sw`, `eps_h_fi`, `eps_ca`,
  `eps_placental`. The upstream `epsilon_*` fixed-unit `std_normal` innovations become
  proper `(1|year)` random effects; the estimated group SD is the brms upgrade (and the
  reason the `@brm` form has a few more parameters than the `@slic` one).
- **Per-demographic-class hunting selectivity + bycatch bias** — `hs_sw ~ 0 + (1|demo)`,
  `hs_fi`, `bycatch_bias`.
- **Per-cell transition process noise** — `tnoise ~ 0 + (1|noise_cell)`, a crossed
  cell random effect reshaped into the transition-noise matrix in the scan.

These live on **four different frames** (`year`, `demo`, `noise_cell`, plus the herring
covariate frame) simultaneously — multi-frame is fine on the formula surface.

## What stays a `@deffun` scan

The carried-state recurrence — the right home for it, as CDC's renewal is: the
**verbatim** `run_state_process` (Leslie aging, size-structured mortality,
density-dependent births, a within-year hunting ODE `ode_rk45_tol` on `dH_dt`, and a
`multinomial_allocation` logistic-normal survived/bycatch/hunted split) and its twelve
verbatim UDFs. A `seal_state` wrapper runs the parent's transformed quantities + the
scan and returns a **NamedTuple** of carriers, each observation reading a **named
field** (`state.population_total`, `state.hunted_sweden`, …) via `@brm` body field
access — one scan, exactly as the `@slic` seal reads its `run_state_process` struct.

## The eight observation streams

Each is a `@brm` family likelihood reading a `state.<field>` indexed to that stream's
observation years, on its own frame:

- **aerial pup counts** — `NegativeBinomial2` on `state.population_total`;
- **Swedish / Finnish harvest bags** — `Normal` (cv-scaled sd) on the hunting-bag totals;
- **Swedish / Finnish hunting age-composition, bycatch composition, reproductive-signs
  composition** — per-row `Multinomial` on the per-year simplices derived from the
  hunted / bycatch / reproductive carriers;
- **pregnancy counts** — `Binomial` on `state.pregnancy_rate`.

The hidden setup below evaluates the `@deffun` scan + helpers and the fixture from the
checked-in source before the declaration is rendered.

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
{}` block carries the verbatim scan + UDFs + the composition densities), and the Turing
pane.

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

The Turing pane is intentionally retained even though this structural model is outside
the current Turing executor; its build-time construction error is part of the comparison
rather than being hidden.

## Provenance and scope

Reproduction: [`research/seal/grey_seal_brm.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/main/research/seal/grey_seal_brm.jl),
gated by `test/seal_brm.jl`. The 13 state-process UDFs and `run_state_process` are
verbatim from the SlicTranspiler research model; the fixture is its example dataset
(3 age classes × 2 sexes × 3 state years). Verified at the **same level as the `@slic`
reference** — transpile + `stanc` + `compiles`. Like the reference it is **not sampled**:
the mechanistic scan's simplex constraints make a naive init degenerate for both forms
(the `@slic` reference is non-finite at the origin too), so the deliverable is the model.
The [StanBlocks-native case study](https://nsiccha.github.io/StanBlocks.jl/dev/examples/case-studies/grey-seal-ipm)
carries the same model in its own idiom.

## Run the reproduction

After bootstrapping the repository's test environment:

```sh
julia --startup-file=no --project=test test/seal_brm.jl
```
