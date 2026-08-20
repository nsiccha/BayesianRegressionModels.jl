# Exact public Warfarin PK/PD reproduction

`reproduce.jl` lowers the strongest public match for Sebastian Weber's
Warfarin example into BRM and StanBlocks without approximating the statistical
model. It contains a public two-subject data slice so the translation can be
parsed, lowered, checked by `stanc`, and instantiated independently of R.

## What is reproduced

The source is a two-stage population model:

1. A one-compartment oral PK model with first-order absorption and a bounded
   lag time. Clearance and volume have fixed allometric weight effects. Four
   independent subject effects modify lag, absorption, clearance, and volume.
2. A turnover PD ODE, fitted after PK and conditioned on each subject's fixed
   posterior-median PK log parameters (already transformed and including the
   PK weight effects). Three independent PD subject effects modify baseline
   response, inverse turnover rate, and EC50.
3. Both stages use the source's `gamma2_overdisp` observation distribution,
   whose variance is `sigma^2 + mu^2 / kappa`.

The implementation preserves the original priors, `tlagMax = 1`, PK and PD
overdispersion scalings (`5^2` and `25^2`), allometric exponents, and the PD
solver settings (`rk45`, `t0=-1e-4`, relative tolerance `1e-5`, absolute
tolerance `1e-3`, maximum 500 steps).

## Provenance boundary

The brms issue says only that Weber was looking at “the PK/PD model for
Warfarin”; it does not attach equations, code, data, or a citation. The model
here is therefore the strongest identifiable public match, not a provably
identical recovery from the issue alone: it is Weber's public StanCon 2018
Warfarin PK/PD program, and the Stan forum later calls its attached generated
model the “warfarin ODE example.”

Primary sources:

- <https://github.com/paul-buerkner/brms/issues/1509#issuecomment-1598639613>
- <https://github.com/stan-dev/stancon_talks/tree/master/2018-helsinki/Contributed-Talks/weber/stancon18-master>
- <https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pk_tlagMax.stan>
- <https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pd_tlagMax_2.stan>
- <https://discourse.mc-stan.org/t/map-rect-threading/8802/3>
- <https://discourse.mc-stan.org/t/parallel-dynamic-hmc-merits/10895/3>

The current public data file contains 32 subjects, 32 dose records, 253 retained
PK observations, five time-0.5 zero PK observations discarded before fitting,
and 232 PD observations. The accompanying `munge.R` comment says four PK rows
were discarded, but the executable predicate and current data discard five;
the reproduction follows the executable data transformation.

## Capability audit and blockers

No new StanBlocks primitive is required. Its typed UDFs, custom distribution
triad, variadic `ode_rk45_tol`, and ragged `kernel(...)` lowering express the
exact fit. This work adds the two BRM prior-lowering pieces the source exposed:
bounded scalar parameters and independent half-normal group standard deviations.

There is one confirmed StanBlocks authoring limitation (`plate-cell-rejec-87663e11`):
a plate cell cannot hoist an intermediate `array[] vector` ODE trajectory. Both
the direct `to_vector(ode_rk45_tol(...)[:, 1])` spelling and the typed-helper
spelling used here stanc-check and preserve the same solver equation. General
support for two-dimensional cell-local values would improve the diagnostic and
remove the need for either inline selection or helper encapsulation; it is not
a fit blocker.

Two limitations remain outside the fitted log density:

- Identity is not provable from the brms issue because its model specification
  is absent. A private source or confirmation from the author is required to
  turn the strong public match into an exact provenance claim.
- The public Stan generated quantities emit both elementwise and patient-summed
  log likelihood, plus conditional and new-population predictive draws. BRM's
  descriptor can produce the exact fit and separate replay operations, but one
  compiled model cannot currently expose all four arrays simultaneously.
  `resample_groups` provides the new-subject-effects operation separately.

There is no PK/PD residual-covariance blocker in the executable public model:
PK and PD are separate scalar Gamma likelihoods fitted in separate stages. The
unknown covariance sketched in the brms discussion belongs to its generic
multi-response example, not to this published Warfarin program. BRM's
`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L_res)` / `LKJCovarianceFactor`
surface resolves the fitted-model side if that joint variant is nevertheless
desired. Its row-level pointwise log likelihood is the joint vector density;
it deliberately does not invent per-outcome decompositions of a correlated
likelihood. None of this affects the exact public reproduction.

## Run

From the repository root after bootstrapping `test/Project.toml`:

```sh
BRM_WARFARIN_RUNTIME=1 julia --startup-file=no --project=test \
  research/warfarin/reproduce.jl
```

Set `BRM_WARFARIN_RUNTIME=0` for lowering and `stanc` only.
