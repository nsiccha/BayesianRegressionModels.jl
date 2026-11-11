# Warfarin PK/PD

Sebastian Weber's StanCon 2018 warfarin example in BRM, on its public data,
behind `docs/src/warfarin.md`.

| file | holds |
| --- | --- |
| `data/warfarin_data.txt`, `data/README.md` | the 32-subject data and their provenance |
| `data.jl` | `warfarin_data()`: the data in the per-subject form the models take; `warfarin_pd_data()`: the same plus each subject's PK medians from `results/` |
| `reproduce.jl` | Weber's two-stage models: the StanBlocks helpers (closed-form PK, turnover ODE, the `gamma2_overdisp` family), `warfarin_pk_brmi` and `warfarin_pd_brmi`, and their lowering, `stanc` and BridgeStan checks |
| `joint_reproduce.jl` | `warfarin_joint_brmi`, one posterior over both channels (`JOINT.md`), with its checks |
| `fit.jl` | the WarmupHMC fits of the three models; writes `results/` |
| `figures.jl` | the figures of the docs page from `results/` |
| `results/` | one directory per fit (`pk/`, `pd/`, `joint/`), each with `population.tsv` (estimates and diagnostics), `fits.tsv` (sampler report), `ppc.tsv` (posterior predictive quantiles per observation) and `receipt.tsv` (what the numbers are the numbers of); `pk_subject_medians.tsv` (the second stage's inputs) |

## What is reproduced

Weber's two Stan programs, statement by statement:

1. a one-compartment oral PK model with first-order absorption and a lag of
   at most one hour (`tlagMax = 1`), clearance and volume scaled by body weight
   with the allometric exponents 0.75 and 1, four independent subject effects
   (lag, absorption, clearance, volume), and his `gamma2_overdisp`
   observation family, whose variance is `sigma^2 + mu^2 / kappa` with
   `kappa` scaled by `5^2`;
2. a turnover PD model solved with `ode_rk45_tol` (`t0 = -1e-4`, relative
   tolerance `1e-5`, absolute tolerance `1e-3`, at most 500 steps),
   conditional on each subject's PK posterior medians, with three independent
   subject effects (baseline, inverse turnover rate, EC50) and the same
   observation family, `kappa` scaled by `25^2`.

All priors are his. One spelling differs: `log_inv_logit(x)` is written
`-log1pexp(-x)`, the same function. The data follow his `munge.R`: the five
zero concentrations at 0.5 h are dropped. His notebook prints no estimates,
so the fits in `results/` have nothing published to be compared with.

The joint model is not part of his example; `JOINT.md` states its contract.

## Provenance

- <https://github.com/stan-dev/stancon_talks/tree/master/2018-helsinki/Contributed-Talks/weber/stancon18-master>
  (the data, `warfarin_pk_tlagMax.stan`, `warfarin_pd_tlagMax_2.stan`,
  `munge.R`, `warfarin_pk.R`, `warfarin_pd.R`)

The brms issue that once motivated this directory
(<https://github.com/paul-buerkner/brms/issues/1509#issuecomment-1598639613>)
names a warfarin PK/PD model without equations, code or data; Weber's public
program is the identifiable public match, not a provably identical model.

## Run

From the repository root, with `test/Project.toml` bootstrapped
(`julia --project=test test/setup_env.jl`):

```sh
julia -t 4 --startup-file=no --project=test research/warfarin/fit.jl        # the fits, results/
julia --startup-file=no --project=research/adaptive_centering/plots research/warfarin/figures.jl
BRM_WARFARIN_RUNTIME=0 julia --startup-file=no --project=test research/warfarin/reproduce.jl
BRM_WARFARIN_JOINT_RUNTIME=0 julia --startup-file=no --project=test research/warfarin/joint_reproduce.jl
```

`fit.jl` takes the draw count and chain count as arguments (default 1000 and
4). The last two commands check lowering and `stanc` without a fit; with the
variable at `1` they also instantiate BridgeStan and check a finite density
and gradient.
