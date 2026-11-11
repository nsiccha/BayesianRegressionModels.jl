# Map a field through space and time

Fit a moving field from station measurements, read it at new locations, and
separate uncertainty in the field from observation noise. The example uses a
`gmrf` term with nearest-neighbor conditioning. The
[space-time fields guide](space-time-fields.md) covers the other field terms
and when each fits the data.

## Data and model

The data are synthetic measurements on a square measured in kilometers.
A positive feature moves east and a negative feature moves north. Each
station has repeated measurements at each time, which help distinguish
observation noise from variation in the latent field. The generating surface
is an analytic function, not a draw from the fitted prior.

```@eval
let mod = Main.BRMDocsComparisons.example_module(:spatial_fields)
    Main.BRMDocsComparisons.evaluate_source_prelude(
        mod, "research/space_time_fields/fields.jl"; before=:station_field_model)
end
nothing
```

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/data.tsv";
    columns=[:quantity => "Synthetic data", :value => "Value"])
```

`neighbors` selects a Gaussian conditional representation of the observed
position–time pairs. Repeated rows at the same coordinates and time read the
same state. The amplitude prior uses response units, the spatial length prior
uses kilometers, and the temporal length prior uses the observation time unit.
Project longitude and latitude into suitable planar coordinates before fitting.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:spatial_fields),
    Main.BRMDocsComparisons.source_function("research/space_time_fields/fields.jl", :station_field_model),
    :station_field_model; title="A station field with unknown lengths and noise", require_stan=true)
```

The panes show one declaration and its generated backends. The field contributes
directly to the predictor; `sd` is its amplitude, without another population
coefficient. Add `1 + gmrf(...)` when the model needs an intercept.

## Run and diagnose

The [fit script](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/space_time_fields/fields.jl)
samples the RK density with WarmupHMC. Each chain gets its own log-density
problem, and so its own gradient workspace, and starts with the scalar
hyperparameters at their prior medians:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/space_time_fields/fields.jl";
    starting_at="    # One RK log density per chain", ending_before="    chains = [Matrix(")
```

The fit runs once, outside the documentation build; this page reads its
summaries, and the Reproduce section records the run.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/diagnostics.tsv";
    columns=[:quantity => "Fit diagnostic", :value => "Value"])
```

R-hat and effective sample sizes cover all sampler coordinates, using equally
long prefixes of the retained chains. The completion quorum can omit slower
chains; these diagnostics do not rule out bias from selecting chains by runtime.
Inspect divergences and chain agreement before interpreting the maps.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/quantities.tsv";
    columns=[:quantity => "Quantity", :unit => "Units", :median => "Median",
             :q025 => "2.5%", :q975 => "97.5%", :description => "Truth or prior",
             :rhat => "R-hat", :ess_bulk => "Bulk ESS", :ess_tail => "Tail ESS"])
```

The noise SD's interval contains its generating value. The generating surface
is not a Gaussian process, so the field SD and the two lengths have no true
values; compare them with their priors. A long spatial length couples distant
stations, and a long temporal length keeps the field similar across nearby
times. The data move the temporal length above most of its prior range: each
feature moves a few kilometers per time unit, less than its own width. The
spatial length stays near its prior median, with a narrower interval.

## Read and map the field

Sampler coordinates are rows of the draw matrix; posterior draws are columns.
Read fitted locations with `rk_draw_tree(rk, draws, (:field,)).field`, then
reduce its named `:draw` axis with `quantile` or `std` from TreeArrays.
Raw sampler coordinates are not field values.

For new coordinates and times, `brm_gmrf_predict` returns one row per point
and one column per draw. TreeArrays reduces the named draw axis to the map
quantiles:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/space_time_fields/fields.jl";
    starting_at="    # rng adds the conditional uncertainty", ending_before="    # Field values on the response scale")
```

Passing a random-number generator includes conditional variation at new
locations. Without it, the result contains conditional means; their spread
understates pointwise uncertainty away from observations.

```@eval
let mod = Main.BRMDocsComparisons.example_module(:spatial_figures)
    Base.include(mod, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT,
        "research", "space_time_fields", "figures.jl"))
end
nothing
```

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:spatial_figures).field_maps();
    caption="Generating field and posterior median at each time. Dots mark stations; all panels share a color scale.")
```

The maps compare the fitted surface with the generating function on the same
grid. Changes between times represent the field's evolution, not different
sets of stations. Hover over a cell to read its coordinates and value.

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:spatial_figures).uncertainty_maps();
    caption="Posterior standard deviation of the latent field, on a shared scale starting at zero. Observation noise is excluded.")
```

Each new point has its conditional marginal law, but points are drawn
independently given the fitted states. These draws support pointwise intervals;
they do not supply the joint covariance needed for uncertainty in a spatial total.

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:spatial_figures).field_slices();
    caption="An east–west slice through the square's center. Blue: posterior median and 50%/95% intervals. Dashed orange: generating field.")
```

The bands show uncertainty at individual locations. They are not simultaneous
coverage bands for an entire curve, and predicting a future measurement would
also require observation noise.

## Checks against the generating surface

The truth is known here because the data are synthetic. On the map grid at the
three observation times:

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/map_checks.tsv";
    columns=[:quantity => "Grid check", :value => "Value"])
```

Every grid point's 95% interval contains the generating value, so on this
surface the intervals are wide rather than tight. One simulated field is not a
calibration study: nearby cells share information, so repeat the simulation
across datasets before making a coverage claim. In an application, held-out
stations or times check the fit between stations; residuals at the fitted rows
do not.

The neighbor count sets a covariance approximation, not the map's resolution,
and a denser map grid does not improve it. Refit with more neighbors when
conclusions depend on fine spatial structure. A change of representation, mesh
or basis also needs a new fit; replay only evaluates the fitted model at new
inputs.

For areal counts, smooth channel surfaces, mesh priors and minibatch SVI, see
the [space-time fields guide](space-time-fields.md).

## Reproduce

::: details Commands and fit receipt

The repository's test environment supplies the pinned BRM, ReactiveKernels,
WarmupHMC and TreeArrays dependencies. These shell commands are manual: the
docs build executes the model declaration and the figure code but reads the
completed fit's summaries.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/space_time_fields/fields.jl
```

The receipt's key hashes the emitted program, the data, the settings, the seed
and the harness version. The draws stay outside git, under `BRM_DOCS_CACHE`
(by default a directory below the system temporary directory); a second run
with the same key rebuilds the tables from them. A run under a new key within
30 days of the last one needs `BRM_DOCS_FIT_FORCE=1`.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/receipt.tsv";
    columns=[:key => "field", :value => "value"])
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/space_time_fields/README.md).
:::
