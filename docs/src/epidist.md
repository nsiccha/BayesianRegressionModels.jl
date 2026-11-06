# Epidemiological delays

How long from infection to symptom onset, from onset to a positive test, from
admission to discharge? Line lists answer such questions only indirectly. Each
row records two calendar windows: the primary event happened on some day (a
window of width one), the secondary event on another. And only pairs whose
secondary event happened before the day of analysis are in the list. Read
naively, the list gives biased delays. Two corrections remove the bias:

- **double interval censoring**: the delay is only known to lie between the
  earliest and the latest times the two windows allow;
- **right truncation**: during a growing epidemic most primary events are
  recent, and a recent event is in the data only if its delay was short, so
  short delays are over-represented.

The R package [epidist](https://epidist.epinowcast.org/) fits such delays on
top of brms. Its recommended *marginal* model integrates the unknown primary
event time out in closed form, as do the
[primarycensored](https://primarycensored.epinowcast.org/) package and its Julia
counterpart [CensoredDistributions.jl](https://censoreddistributions.epiaware.org/).
BRM has the same likelihood as a response family, spelled as in
CensoredDistributions.jl:

```julia
delay ~ double_interval_censored(LogNormal(mu, sigma);
    primary_event = Uniform(0, pwindow), upper = obs_time, interval = swindow)
```

- `delay` is the start of the secondary event's window, counted from the start
  of the primary event's window, and `swindow` is its width.
- `primary_event = Uniform(0, pwindow)` says the primary event happened
  uniformly within its window.
- `upper = obs_time` truncates at the observation time (counted from the same
  start). `lower` truncates from below.
- The delay itself is `LogNormal`, `Gamma` or `Weibull`, and its parameters may
  be linear predictors, as anywhere in BRM.

Writing ``F_{pc}(d) = \frac{1}{w_P}\int_{d-w_P}^{d} F(t)\,dt`` for the CDF of
the delay plus a uniform primary offset (``F`` the delay's CDF, ``w_P`` the
primary window), each row contributes

```math
\log\bigl(F_{pc}(y + w_S) - F_{pc}(y)\bigr) - \log\bigl(F_{pc}(T) - F_{pc}(L)\bigr),
```

with ``y`` the response, ``w_S`` the secondary window, and ``[L, T]`` the
truncation window. `primary_censored(LogNormal(mu, sigma), Uniform(0, w))` is the
same law for exactly observed secondary times.

## epidist's example

The data are epidist's own. Its "approximate inference" article simulates an
outbreak with a lognormal delay (meanlog 1.8, sdlog 0.5), observes it on day 25,
keeps the pairs whose secondary window ended by then, and samples 200 of them.
epidist merges rows with identical windows into 94 rows with a count each, the
`n` the model reads through `weighted(..., fweights(n))`. The priors are
epidist's defaults for a lognormal delay.

```@eval
let mod = Main.BRMDocsComparisons.example_module(:epidist)
    Main.BRMDocsComparisons.evaluate_source_prelude(
        mod, "research/epidist/epidist_page.jl"; before=:epidist_delay_model)
end
nothing
```

```@eval
Main.BRMDocsComparisons.source_function("research/epidist/epidist_page.jl", :epidist_rows)
```

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:epidist),
    Main.BRMDocsComparisons.source_function(
        "research/epidist/epidist_page.jl", :epidist_delay_model,
    ),
    :epidist_delay_model;
    title="epidist's marginal delay model",
    require_stan=true,
)
```

The StanBlocks backend emits the family as the Stan family `brm_censored_delay`.
The ReactiveKernels backend observes the same mathematics, written in Julia,
and differentiates it with Enzyme.

## The same density as epidist

At 200 points around the posterior, both backends reproduce the log density
and its gradient of the Stan program epidist itself generates for these data,
read through BridgeStan. The comparison runs on the 200-case data set and on
all 2287 observed cases (221 rows). The StanBlocks program differs by a
constant, the two normal priors' normalising constants that Stan's `~` drops,
so the columns below compare log densities up to that constant. Times are one
density and gradient evaluation on one CPU of a shared machine; only their
ratios carry over.

```@eval
using JSON, Markdown, Printf
root = Main.BRMDocsComparisons.REPOSITORY_ROOT
rows = String[]
for (label, dir) in (("200 cases (94 rows)", "n200"), ("2287 cases (221 rows)", "nall"))
    p = JSON.parsefile(joinpath(root, "research", "epidist", "results", dir, "parity.json"))
    g = p["gradient_seconds"]
    worst = maximum(p["parity"][k]["bulk_max_abs_logdensity_diff_after_offset"]
                    for k in ("BRM native SBBRMI", "BRM native RKBRMI"))
    push!(rows, @sprintf("| %s | %.1e | %.0f µs | %.0f µs | %.0f µs |", label, worst,
        1e6 * g["epidist (brms Stan)"], 1e6 * g["BRM native SBBRMI"], 1e6 * g["BRM native RKBRMI"]))
end
Markdown.parse("""
| Data | largest log-density difference | epidist's Stan program | BRM, StanBlocks | BRM, ReactiveKernels |
|:-----|-------:|-------:|-------:|-------:|
$(join(rows, "\n"))
""")
```

In the far tails the two differ in accuracy. BRM takes a window's probability
from survival values when the window lies above the median, so it stays finite
where epidist's program returns NaN (at 2 of the 200 points on the 200-case
data).

## Fits

All three programs are fitted with the same sampler (WarmupHMC, four chains of
1000 warm-up iterations and 1000 draws), so the posteriors agree to Monte Carlo
error and recover the simulated delay.

```@eval
using JSON, Markdown, Printf
root = Main.BRMDocsComparisons.REPOSITORY_ROOT
rows = String[]
for (label, dir) in (("200 cases", "n200"), ("2287 cases", "nall"))
    f = JSON.parsefile(joinpath(root, "research", "epidist", "results", dir, "fits.json"))
    for (name, key) in (("epidist's Stan program", "epidist (brms Stan)"),
                        ("BRM, StanBlocks", "BRM native SBBRMI"),
                        ("BRM, ReactiveKernels", "BRM native RKBRMI"))
        r = f["fits"][key]
        push!(rows, @sprintf("| %s | %s | %.2f [%.2f, %.2f] | %.3f [%.3f, %.3f] | %.0f |",
            label, name, r["mu"]["mean"], r["mu"]["q5"], r["mu"]["q95"],
            exp(r["log_sigma"]["mean"]), exp(r["log_sigma"]["q5"]), exp(r["log_sigma"]["q95"]),
            min(r["mu"]["ess"], r["log_sigma"]["ess"])))
    end
end
Markdown.parse("""
| Data | Program | meanlog (truth 1.8) | sdlog (truth 0.5) | smallest ESS |
|:-----|:--------|:-------|:-------|-----:|
$(join(rows, "\n"))
""")
```

Means with 90 % intervals; sdlog is shown as ``\exp`` of the posterior of
``\log`` sdlog.

## Not built yet

- Primary windows other than uniform, such as epidist's exponential-growth
  primary.
- Delay laws other than lognormal, gamma and Weibull, which need numerical
  integration.
- Truncation bounds that are model values. epidist's *latent* model samples
  each row's event times and truncates at a bound that depends on them.
- Predictive draws on the ReactiveKernels backend. The StanBlocks backend draws
  each row from the truncated law and reports the start of its secondary
  window.
