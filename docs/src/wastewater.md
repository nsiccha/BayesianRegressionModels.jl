# Wastewater and infection renewal

The data are CDC's public simulated example: wastewater RNA concentrations
from a catchment of 400,000 people and daily hospital admissions in a state
of 3,000,000 people. We use site 1 and the first 35 calendar days, from
1 September through 5 October 2023. Each wastewater record carries its
sampling date, laboratory and detection limit; hospital records occupy a
separate daily axis. The remaining state population has no wastewater
observations. These are synthetic observations published by
[CDC's ww-inference-model](https://github.com/CDCgov/ww-inference-model), not
measurements from a named community. The input tables and their exact source
revision are linked under Reproduce.

## Model

A shared weekly log reproduction number drives daily infection renewal.
Each subpopulation follows that reference with its own stationary AR(1)
deviation. Recent infections contribute infectiousness through a fixed
generation-interval probability mass function. A feedback term attenuates
the reproduction number as recent incidence increases:

```math
\Lambda_{k,t}=\sum_{s\geq1}g_s I_{k,t-s},\qquad
I_{k,t}=\exp(\ell_t+\delta_{k,t}-\gamma\Lambda_{k,t})\Lambda_{k,t}.
```

Here incidence is a daily population fraction. The reference has
zero subpopulation deviation; its trajectory therefore anchors the
hierarchy. Fifty unobserved days provide an exponentially seeded history
before the first observation. Seven further days allow the fitted process
to continue into a forecast. Neither the unobserved history nor the
forecast contributes an extra observation likelihood.

The renewal recurrence is an ordinary StanBlocks function. This is the
code the model calls, read from its source during the build:

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:wastewater),
    "research/wastewater/cdc_ww_inference.jl";
    before=:cdc_ww_brm_model, starting_at="StanBlocks.@deffun begin",
)
Base.include(Main.BRMDocsComparisons.example_module(:wastewater),
    joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research/wastewater/data.jl"))
Main.BRMDocsComparisons.source_code_region(
    "research/wastewater/cdc_ww_inference.jl";
    starting_at="    renewal_feedback(logRt::vector", ending_before="    # CDC parameterizes incidence",
)
```

The `@brm` declaration keeps the population hierarchy, priors and observation
families together. Its plate produces the infection trajectory of each
subpopulation. Separate indexing arrays then connect each observed sample
to the appropriate day, catchment and laboratory. Repeated measurements
can share one latent infection value without duplicating that state.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:wastewater),
    Main.BRMDocsComparisons.source_function("research/wastewater/cdc_ww_inference.jl", :cdc_ww_brm_model) *
    "\n\n" * Main.BRMDocsComparisons.source_function("research/wastewater/data.jl", :wastewater_public_model),
    :wastewater_public_model; title="Wastewater and hospital admissions", require_stan=true,
)
```

Wastewater concentration is proportional to infection incidence convolved
with the shedding profile. The profile has an inferred peak and duration
and is normalized before convolution. A laboratory multiplier changes the
concentration scale, and a laboratory noise scale controls log-concentration
error. Values at their detection limits contribute a left-censoring
probability. Concentrations retain their source units, genome copies per mL;
the dilution constant uses mL per person per day.

Hospital admissions constrain a different view of the same infections.
The model averages incidence with population weights, convolves it with
the infection-to-hospital delay distribution, and applies a varying
admission probability and a mean-one weekday effect. A negative-binomial
likelihood accounts for count variation beyond the Poisson variance. The
wastewater and admission frames remain separate throughout this calculation;
there is no interpolation of sparse samples into invented daily measurements.

With one measured catchment, the state admissions series supplies information
about infections outside that catchment. The shared reference and deviation
priors determine how those signals are pooled. Changing the population weights
therefore changes the scientific model, not merely the labels on its output.

## Fit and diagnostics

Four independent WarmupHMC chains start from different random seeds, each
requesting 1,000 retained draws. The completion sampler requires three chains
to finish and then lets the remaining chain finish its current warmup round.
The table records how many chains contributed, which were omitted, the total
retained draws, divergences and sampling time. These are results of the
cached run; rebuilding this page reads its tables without sampling again.

```@eval
Main.BRMDocsComparisons.results_table(
    "research/wastewater/results/fits.tsv";
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
    "research/wastewater/results/population.tsv";
    columns=[:quantity => "quantity", :unit => "unit", :median => "median",
             :q025 => "2.5 %", :q975 => "97.5 %", :rhat => "R-hat",
             :ess_bulk => "bulk ESS", :ess_tail => "tail ESS"],
)
```

This run retained three chains and recorded 20 divergences. The initial
reference reproduction number has R-hat 1.084 and bulk ESS 49, so agreement
on the other reported quantities does not establish convergence of the whole
fit. The initial level belongs to the unobserved seeding history, but its
uncertainty is coupled to the later trajectory. Treat the bands below as
provisional Monte Carlo summaries, not validated transmission estimates.

## Reproduction number

```@eval
if !isdefined(Main, :BRMExampleFigures)
    Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "example_figures.jl"))
end
nothing
```
```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMExampleFigures.trajectory("research/wastewater/results/trajectory.tsv";
        xlabel="Day (1 = 1 September 2023)", ylabel="Reference reproduction number",
        select=r -> 1 <= r.time <= 35);
    caption="Unadjusted reference reproduction number over the observed window: posterior median, 50 % and 95 % bands.",
)
```

The plotted quantity is the exponentiated shared reference process,
before infection feedback and catchment-specific deviations. It is not the
realized reproduction number of the measured catchment. Its bands describe
uncertainty in that latent reference, rather than a prediction interval for
a future concentration. Convolution and sparse sampling mean that a short
change in wastewater need not imply an equally sharp change in transmission.

This example uses BRM's structural port of the CDC model. The recurrence,
subpopulation hierarchy, shedding and count streams follow the same causal
order; the priors and hospitalization-rate parameterization are those of
the checked-in BRM declaration. This fit is not a reproduction of CDC's
posterior estimates. The research notes give the differences and retain the
smaller EpiSewer-style model for comparison.

## Reproduce

::: details Commands and fit receipt

The repository's test environment supplies the pinned BRM, StanBlocks,
BridgeStan, WarmupHMC and TreeArrays dependencies. These shell commands are
manual: the docs build executes the model declarations and figures but reads
the completed fit's summaries. Run the fit deliberately when its model,
data or settings change; routine documentation edits reuse the cache.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia -t 4 --startup-file=no --project=test research/wastewater/fit.jl
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
    "research/wastewater/results/receipt.tsv";
    columns=[:key => "field", :value => "value"],
)
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/wastewater/README.md).
:::
