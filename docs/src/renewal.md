# Epidemic renewal and reporting delays

This example uses 56 simulated days of epidemic counts, generated with seed
132. The generating reproduction number starts at 1.3 and follows a daily
log random walk. A gamma generation interval links past infections to new
infections; a lognormal reporting delay separates infection from observation.
Daily reports follow a negative-binomial distribution. Simulation makes the
expected report trajectory available for comparison, while the fit receives
only the counts, observation days and fixed delay distributions. There are
no patient records in this example.

## Model

The renewal equation separates transmission from observation. Infectiousness
is a weighted sum of previous incidence, and the reproduction number scales
that sum to give today's infections. Reports are a second convolution:

```math
I_t=R_t\sum_{s\geq1}g_s I_{t-s},\qquad
m_t=\sum_{d\geq0}h_d I_{t-d},\qquad
Y_t\sim\operatorname{NegBinomial}_2(m_t,\kappa^{-2}).
```

The generation-interval weights `g` start at lag one; reporting-delay weights
`h` include lag zero. The distinction matters: same-day reports are possible,
but an infection does not generate itself. The expected-case function carries
the infection history through both convolutions. Before day one, the model
uses an exponential history whose initial growth rate satisfies the
discretized Euler–Lotka equation for the initial reproduction number.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:renewal),
    "research/epi_renewal/renewal.jl"; before=:renewal_single_model,
)
Main.BRMDocsComparisons.source_code_region(
    "research/epi_renewal/renewal.jl";
    starting_at="    expected_cases(log_R", ending_before="    @lhs @lpxf nb_cases_lpmf",
)
```

BRM's `rw(time)` term supplies the daily innovations. The intercept prior
sets the first log reproduction number; the term's SD prior sets the scale
of changes from one day to the next. `log_I0` determines the initial infection
scale. `cluster` is a positive overdispersion scale: the observation variance
is `m + cluster² m²`. These quantities are inferred jointly, so uncertainty
in initial incidence propagates through the entire reporting trajectory.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:renewal),
    Main.BRMDocsComparisons.source_function("research/epi_renewal/renewal.jl", :renewal_single_model),
    :renewal_single_model; title="Renewal with delayed reports", require_stan=true,
)
```

The `observed` mask in the custom count family distinguishes observed from
unobserved days. This fit observes all 56 days. Setting later entries to
zero removes those likelihood contributions while retaining the latent
process and predictive outputs for those days. It does not set their counts
to zero or treat missing reports as evidence of no infections. A forecast
must therefore be constructed by changing the mask, not by appending zero
counts to the fitted observation vector.

## Delay distributions

A continuous delay is not rounded independently of the primary event time.
An infection occurs somewhere within its calendar day. The data preparation
integrates over a uniform within-day offset before assigning probability to
successive report days, then normalizes over the chosen support. The
probability vectors enter the model as data, so another generation interval
or reporting-delay distribution changes the fit without changing the model
body. This example conditions on those distributions; their uncertainty is
not included in the posterior bands.

A line list would require a separate delay likelihood. If both event dates
are recorded as intervals and recent events are missing long delays, BRM's
`double_interval_censored` family integrates the primary event window and
conditions on the observation cutoff. The [epidist research example](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/epidist)
contains that model and its comparison with epidist. Estimating delays from
such a list is a different inferential step from fitting counts with fixed
delay weights; the research notes describe how the two relate.

## Fit and diagnostics

Four independent WarmupHMC chains start from different random seeds, each
requesting 1,000 retained draws. The completion sampler requires three chains
to finish and then lets the remaining chain finish its current warmup round.
The table records how many chains contributed, which were omitted, the total
retained draws, divergences and sampling time. These are results of the
cached run; rebuilding this page reads its tables without sampling again.

```@eval
Main.BRMDocsComparisons.results_table(
    "research/epi_renewal/results/docs/fits.tsv";
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
    "research/epi_renewal/results/docs/population.tsv";
    columns=[:quantity => "quantity", :unit => "unit", :median => "median",
             :q025 => "2.5 %", :q975 => "97.5 %", :rhat => "R-hat",
             :ess_bulk => "bulk ESS", :ess_tail => "tail ESS"],
)
```

## Expected reports

The retained three-chain run has no divergences. R-hat is 1.001–1.008 for
the reported population quantities, with bulk ESS at least 445. Their 95 %
intervals contain the generating initial reproduction number of 1.3,
initial incidence of 50 infections per day and overdispersion scale of 0.1.
This checks recovery in this simulated series, conditional on the fixed
generation and reporting delays.

```@eval
if !isdefined(Main, :BRMExampleFigures)
    Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "example_figures.jl"))
end
nothing
```
```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMExampleFigures.trajectory("research/epi_renewal/results/docs/trajectory.tsv";
        xlabel="Day", ylabel="Expected reports per day", observed=true);
    caption="Expected daily reports: posterior median with 50 % and 95 % credible bands in blue; observed counts in orange.",
)
```

The bands describe the expected reporting intensity `Y` in the declaration.
They exclude negative-binomial observation noise, so there is no requirement
that 95 % of the orange counts lie within the outer band. The separate
`ppc.tsv` table contains predictive intervals that include this noise.
Reporting delays spread the effect of a change in transmission over later
days; the plot should be read as a trajectory of reports, not of infection
onset. The random-walk prior regularizes changes that the delayed counts
cannot identify individually.

## Reproduce

::: details Commands and fit receipt

The repository's test environment supplies the pinned BRM, StanBlocks,
BridgeStan, WarmupHMC and TreeArrays dependencies. These shell commands are
manual: the docs build executes the model declarations and figures but reads
the completed fit's summaries. Run the fit deliberately when its model,
data or settings change; routine documentation edits reuse the cache.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia -t 4 --startup-file=no --project=test research/epi_renewal/fit.jl
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
    "research/epi_renewal/results/docs/receipt.tsv";
    columns=[:key => "field", :value => "value"],
)
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/epi_renewal/README.md).
:::
