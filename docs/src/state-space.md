# Continuous-time latent states

The data are 40 synthetic measurements at irregular times over a period of
several weeks. A latent Ornstein–Uhlenbeck process fluctuates around a
stationary mean; independent Gaussian measurement error separates the
observations from that process. The simulation uses seed 144 and retains
the generating states for comparison. Time is measured in days, and the
response is in arbitrary units. The fit sees the observation times and
measurements; the generating states are used only to assess and display
the result.

## Continuous-time model

The process returns towards its mean at a positive rate. Its stochastic
differential equation can be written in terms of the stationary standard
deviation, rather than the diffusion coefficient:

```math
dX_t = \lambda(\mu-X_t)\,dt + \sqrt{2\lambda}\,s\,dW_t,
\qquad Y_j\sim\operatorname{Normal}(X_{t_j},\sigma).
```

The exact transition over a gap of length `dt` has persistence
`exp(-rate * dt)` and innovation standard deviation
`stationary_sd * sqrt(1 - exp(-2rate * dt))`. Small gaps retain most of the
previous state; long gaps approach the stationary distribution. The first
state is drawn from that stationary distribution. This is a substantive
initial-condition assumption, not an estimate of a pre-observation path.

The same transition is used to simulate the data and to construct the
fitted path. No numerical integration grid is introduced, and there is no
Euler discretization error for this model. `expm1` evaluates the innovation
variance accurately when time gaps are small.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:state_space),
    "research/ema_ctsem/docs_model.jl"; before=:state_space_model,
)
Main.BRMDocsComparisons.source_code_region(
    "research/ema_ctsem/docs_model.jl";
    starting_at="StanBlocks.@deffun @juliacompat begin", ending_before="\"A public synthetic series",
)
```

## Sampled states

The declaration samples independent standard-normal innovations `z` and
constructs the latent state as a deterministic function of those innovations
and the population parameters. This is a non-centred representation of the
Gaussian path. BRM exposes `state` as a named derived output, so posterior
summaries use its public descriptor address without identifying generated
Stan coordinates by hand.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:state_space),
    Main.BRMDocsComparisons.source_function("research/ema_ctsem/docs_model.jl", :state_space_model),
    :state_space_model; title="Irregularly observed Ornstein–Uhlenbeck process", require_stan=true,
)
```

The mean has a normal prior. The reversion rate, stationary process SD and
measurement SD have lognormal priors on positive values. The prior on the
rate is centred around a response to changes over a few days; the prior
on measurement noise reflects the intended scale of the synthetic example.
These values are visible in the declaration and are part of this model,
not defaults supplied by BRM.

Stationary process variation and measurement error affect the data
differently. Process variation is correlated across nearby observations,
whereas measurement errors are independent. Irregular gaps help express
that difference, although a short series may still leave the two scales
weakly separated. The fitted state between observations is not part of this
output: only states at the supplied times are sampled. Adding a plotting
grid requires conditional path recovery or additional latent times, rather
than interpreting straight line segments as posterior draws of a continuous
path.

## Fit and diagnostics

Four independent WarmupHMC chains start from different random seeds, each
requesting 1,000 retained draws. The completion sampler requires three chains
to finish and then lets the remaining chain finish its current warmup round.
The table records how many chains contributed, which were omitted, the total
retained draws, divergences and sampling time. These are results of the
cached run; rebuilding this page reads its tables without sampling again.

```@eval
Main.BRMDocsComparisons.results_table(
    "research/ema_ctsem/results/docs/fits.tsv";
    columns=[:chains_requested => "started", :chains_kept => "retained",
             :omitted_chains => "omitted", :draws => "draws",
             :divergent => "divergences", :divergent_pct => "%", :seconds => "seconds"],
)
```

Population summaries use the retained chains on the quantities' natural
scales. The intervals contain the central 95 % of their pooled draws.
Split R-hat compares chains; bulk and tail effective sample sizes describe
the information in the centre and tails. A small divergence count alone is
insufficient evidence that the chains agree. The completion rule can also
select chains by speed, so the omitted-chain record belongs with the result.

```@eval
Main.BRMDocsComparisons.results_table(
    "research/ema_ctsem/results/docs/population.tsv";
    columns=[:quantity => "quantity", :unit => "unit", :median => "median",
             :q025 => "2.5 %", :q975 => "97.5 %", :rhat => "R-hat",
             :ess_bulk => "bulk ESS", :ess_tail => "tail ESS"],
)
```

## Latent states and observations

The retained run has 3,000 draws across three chains and no divergences.
R-hat is 1.004–1.011 for the reported population quantities; the smallest
bulk ESS is 218 for the observation SD. All four generating values fall
inside their reported 95 % intervals. The uncertainty in the measurement
scale remains visible: its interval is 0.067–0.269 response units around
a generating value of 0.15.

```@eval
if !isdefined(Main, :BRMExampleFigures)
    Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "example_figures.jl"))
end
nothing
```
```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMExampleFigures.trajectory("research/ema_ctsem/results/docs/trajectory.tsv";
        xlabel="Time (days)", ylabel="Latent response", observed=true);
    caption="Latent states at the observed times: posterior median with 50 % and 95 % credible bands in blue; noisy measurements in orange.",
)
```

The blue intervals describe uncertainty in the latent state after
conditioning on the entire series. They are smoothing intervals: later
observations can inform an earlier state through the process model. A
filtering interval would instead condition only on observations available
up to that time. The distinction matters when using a fitted trajectory
for retrospective interpretation versus real-time monitoring.

Orange points include measurement noise, so their position outside a
state interval is not itself a predictive failure. The script also writes
posterior predictive quantiles for a new measurement at each supplied time.
Those intervals include both state uncertainty and observation noise and
are wider for that reason. The generating path is retained in the trajectory
table, allowing a simulation check without treating it as observed data.

## Marginalized states

For this linear Gaussian process, a Kalman filter can integrate the latent
path out of the likelihood exactly. HMC then needs only the population
parameters, followed by conditional state recovery. The [filters guide](filtering.md)
explains that likelihood representation and the boundary between exact
Gaussian filtering and approximations for nonlinear or non-Gaussian models.
The sampled version here keeps the state construction explicit and provides
a direct example of posterior state summaries. It is not a claim that
sampling every state is preferable for long series.

The earlier multivariate continuous-time examples, including hierarchical
stress and mood models and approximate filters, remain in the research
directory. They add scientific structure and numerical questions beyond
the single exact transition taught here.

## Reproduce

::: details Commands and fit receipt

The repository's test environment supplies the pinned BRM, StanBlocks,
BridgeStan, WarmupHMC and TreeArrays dependencies. These shell commands are
manual: the docs build executes the model declarations and figures but reads
the completed fit's summaries. Run the fit deliberately when its model,
data or settings change; routine documentation edits reuse the cache.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia -t 4 --startup-file=no --project=test research/ema_ctsem/fit.jl
```

The script writes its receipt before sampling and saves the draws outside the
repository. Its key includes the emitted program, data, settings, seed and
sampling harness. TreeArrays reduces the named chain and draw axes to small
summary tables. A second invocation with the same key rebuilds those tables
from the saved draws. The receipt identifies the program behind these numbers.

On a fresh checkout the draws may be absent. An existing receipt guards
against repeating a recent fit; set `BRM_DOCS_FIT_FORCE=1` for a deliberate
rerun. `BRM_DOCS_CACHE` selects a persistent draw-cache directory; its default
is under the system temporary directory. Only the small summary tables belong
in git.

```@eval
Main.BRMDocsComparisons.results_table(
    "research/ema_ctsem/results/docs/receipt.tsv";
    columns=[:key => "field", :value => "value"],
)
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/ema_ctsem/README.md).
:::
