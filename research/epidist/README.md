# epidist's marginal delay model on BRM and RKPPL: parity and timing

[epidist](https://epidist.epinowcast.org/) (epinowcast, 1.0.0) estimates epidemiological
delay distributions from line lists with double interval censoring (daily primary and
secondary event windows) and right truncation (the observation time). It is a brms
extension: a custom brms family per model, Stan through brms. Its recommended model,
the *marginal* model, integrates the primary event time out analytically
(primarycensored's closed form for a uniform primary window and a lognormal, gamma or
Weibull delay) and aggregates identical rows into weighted ones.

This directory compares that model across four implementations on epidist's own
simulated data:

| Target | What it is |
|---|---|
| `epidist (brms Stan)` | epidist's generated Stan program and Stan data, read through BridgeStan |
| `BRM SBBRMI` | one `@brm` declaration lowered to Stan by StanBlocks |
| `BRM RKBRMI` | the same `@brm` declaration lowered to an RKPPL program (ReactiveKernels, Enzyme reverse) |
| `RKPPL @rkppl` | a hand-written `@rkppl` program with a caller-owned `LogDensity` law |

All four use the same closed-form primary-censored lognormal likelihood and epidist's
default priors, which are read from its generated program.

## Files

| File | Role |
|---|---|
| `epidist_reference.R` | Simulates epidist's "approximate inference" data (`simulate_gillespie(seed = 101)`, lognormal delay meanlog 1.8, sdlog 0.5, observation time 25, daily windows), prepares the marginal model, writes epidist's Stan program, Stan data and rows, and times epidist's own fit (brms + cmdstanr, 4 chains × 1000 + 1000, one core). |
| `epidist_models.jl` | The likelihood in Julia and as StanBlocks `@deffun`s, the `@brm` declaration and the `@rkppl` program. |
| `epidist_parity.jl` | Density and gradient parity of all four targets at 200 points, density-plus-gradient timing, and (with `fit`) WarmupHMC fits of all four. |
| `results/n200/`, `results/nall/` | The vignette's 200 cases (94 aggregated rows) and every observed case (2287 cases, 221 rows): epidist's program and data, `epidist_fit.json`, `parity.json`, `fits.json`, BRM's emitted Stan. `epidist_marginal.stan` is epidist's generated program verbatim; it embeds primarycensored 1.6.0's Stan functions (both packages MIT). |

```sh
# R library with epidist 1.0.0, brms 2.23.0, cmdstanr 0.9.0 (R_EPIDIST_LIB), CmdStan 2.40.0 (CMDSTAN)
Rscript research/epidist/epidist_reference.R research/epidist/results/n200 200 101
Rscript research/epidist/epidist_reference.R research/epidist/results/nall 1000000 101
julia --project=test test/setup_env.jl
julia --project=test research/epidist/epidist_parity.jl research/epidist/results/n200 fit
julia --project=test research/epidist/epidist_parity.jl research/epidist/results/nall fit
```

## Results (gordito, 2026-11-05, one CPU per run, loaded host)

Density parity in the posterior bulk (25 of 200 random points), against epidist's program:
max |Δ log density| 2.4e-12 (200 cases) and 1.2e-10 (2287 cases), gradients to 2.3e-12 and 2.4e-11.
The same holds for BRM SBBRMI and RKBRMI and for `@rkppl`. SBBRMI sits a constant 0.921586 higher:
Stan's `~` drops the two normal priors' normalising constants.

| | epidist (brms Stan) | BRM SBBRMI | BRM RKBRMI | RKPPL `@rkppl` |
|---|---|---|---|---|
| density + gradient, 94 rows | 200 µs | 106 µs | 45 µs | 36 µs |
| density + gradient, 221 rows | 514 µs | 246 µs | 132 µs | 120 µs |
| WarmupHMC, 4 × (1000 + 1000), 200 cases | 4.4 s | 2.0 s | 1.1 s | 0.9 s |
| WarmupHMC, 4 × (1000 + 1000), 2287 cases | 8.5 s | 4.6 s | 2.7 s | 2.2 s |

The fits run after one untimed fit per target, with chains in sequence and every chain kept
(`min_completed = 4`). epidist's own fit (brms + cmdstanr NUTS, 4 × (1000 + 1000), chains in
sequence) spent 10.6 s and 23.4 s in Stan's sampler on the same data. All fits recover meanlog
1.8 and log sdlog −0.693 inside their 90 % intervals, with R-hat at most 1.003 and bulk ESS
1800–3500 per 4000 draws. Numbers: `results/*/parity.json`, `results/*/fits.json`,
`results/*/epidist_fit.json`.

## The BRM spelling today

BRM has no primary-censored response family yet, so the declaration computes each
row's probability in a top-level assignment and observes it through a Binomial whose
response equals its trial count (`log Binomial(n | n, p) = n log p`, the
frequency-weighted row log probability):

```julia
@brm begin
    mu ~ 1
    log(sigma) ~ 1
    effect(mu, Intercept) ~ Normal(prior_mu_location, prior_mu_scale)
    effect(sigma, Intercept) ~ Normal(prior_sigma_location, prior_sigma_scale)
    p = epidist_marginal_prob(delay_lwr, delay_upr, pwindow, relative_obs_time, mu, sigma)
    count ~ Binomial(n, p)
end
```

`epidist_marginal_prob` is a StanBlocks `@deffun` for SBBRMI with Julia methods of the
same name for RKBRMI. The natural spelling,
`delay ~ double_interval_censored(LogNormal(mu, sigma); …)` composed with
`weighted(…, fweights(n))`, needs BRM work first.

## Limits of this comparison

- The Julia/Stan helper is the plain closed form, `(G(d) − G(d − w)) / w` differenced
  in linear space. It matches epidist to 1e-10 or better in the posterior bulk, but
  loses precision deep in the tails, where epidist's log-space primarycensored code
  is more careful (and, at a few extreme points, returns NaN where this one returns −Inf).
- Timings come from a shared, loaded fleet host (`gordito`) and one CPU per run; compare
  them within one run.
