# Wastewater renewal examples

The documentation page is `docs/src/wastewater.md`. It fits the CDC-inspired
BRM declaration in `cdc_ww_inference.jl` to the public simulated data selected
by `data.jl`; `fit.jl` writes chain diagnostics, natural-scale population
summaries, the reference reproduction-number trajectory and wastewater
posterior-predictive quantiles. Input provenance and the exported CDC commit
are in `data/README.md`. These are synthetic observations, not public-health
estimates for an actual location.

## Structural port

The shared weekly reference log reproduction number is a differenced AR(1)
process. Each covered subpopulation adds a stationary daily AR(1) deviation;
the uncovered reference does not. A renewal recurrence, with infection
feedback, produces incidence fractions from an exponentially seeded history.
An inferred triangular shedding profile is normalized and convolved with
incidence; sparse laboratory observations have their own scale, noise and
row-specific detection limit. Independent count frames map the same infection
matrix through population weights, delay PMFs, time-varying observation rates
and weekday effects.

The source preserves CDC's causal order and several parameterizations,
including seeding from first-observed incidence, but does not reproduce its
exact hospitalization-rate mean reversion, R-interface hyperprior values or
component switches. It is therefore a structural port, not a claim of
posterior equivalence. CDC's authoritative source is
[wwinference.stan](https://github.com/CDCgov/ww-inference-model/blob/3e92fea29a0b96212e1690dbdcaeb8abfe19cc4f/inst/stan/wwinference.stan).
The data source is frozen to that revision independently of later changes
to CDC's repository.

The documentation fit retains one measured catchment and the remainder of
its state as a reference, one laboratory, one admissions stream, 35 observed
days, 50 seeded days and seven forecast days. The generation and admission
delay PMFs come from CDC's packaged data. Dilution is 757,000 mL per person
per day, consistent with observations in genome copies per mL. The forecast
is not used as calibration data. The displayed reproduction number is the
unadjusted shared reference, not the catchment's feedback-adjusted value.

## Earlier EpiSewer example

`renewal.jl` retains the smaller single-catchment/multiple-site example:
renewal incidence, shedding convolution, and lognormal wastewater error,
with an optional censoring likelihood. The generation and shedding PMFs are
data captured by the plate, and the formula version uses an HSGP predictor
for log reproduction number. The CDC example supersedes that page in the
reader's path; the smaller model remains useful source for studying those
three components alone.

Earlier `test/wastewater_model.jl` and `test/cdc_ww_inference.jl` established
transpilation, Stan syntax and finite runtime density/gradients. Those are
execution checks, not fitted posteriors. The new documentation fit adds the
sampling and diagnostic evidence in `results/receipt.tsv` and its tables.
The old authoring limitation around shared data captured by a plate and the
later addition of `dar` are history, not restrictions on current models.

## Reproduce

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia -t 4 --startup-file=no --project=test research/wastewater/fit.jl
```

Four chains request 1,000 retained draws each under WarmupHMC's completion
quorum. The receipt is written before sampling, and the draws are cached
outside git by program, data, settings, seed and sampling-harness content.
A repeated invocation with the same key regenerates summaries from those
draws. The page reads small TSVs and builds its Vega-Lite figure during
Documenter evaluation; no sampling or raster figure generation runs in CI.
