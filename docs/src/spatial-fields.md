# Map a field through space and time

Fit a moving field from station measurements, read it at new locations, and
separate uncertainty in the field from observation noise. The example uses
`gmrf`; the final sections show when an area graph (`car`) or a smooth basis
(`hsgp`) is a better description of the data.

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

Run the [fit script](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/space_time_fields/fields.jl)
once, then build the page from its summaries. WarmupHMC adapts independent
chains on the RK density, each with its own gradient workspace and initial
values from `brm_start(...; base=:prior_median)`. Sampling stays outside the
documentation build; the Reproduce footer records the exact run.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/diagnostics.tsv";
    columns=[:quantity => "Fit diagnostic", :value => "Value"])
```

R-hat and effective sample sizes cover all sampler coordinates, using equally
long prefixes of the retained chains. The completion quorum can omit slower
chains; these diagnostics do not rule out bias from selecting chains by runtime.
Inspect divergences and chain agreement before interpreting the maps.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/noise.tsv";
    columns=[:quantity => "Quantity", :unit => "Units", :median => "Median",
             :q025 => "2.5%", :q975 => "97.5%", :rhat => "R-hat", :ess_bulk => "Bulk ESS", :ess_tail => "Tail ESS"])
```

## Read and map the field

Sampler coordinates are rows of the draw matrix; posterior draws are columns.
Read fitted locations with `rk_draw_tree(rk, draws, (:field,)).field`, then
reduce its named `:draw` axis with `quantile` or `std` from TreeArrays.
Raw sampler coordinates are not field values.

For new coordinates and times, use
`brm_gmrf_predict(rk, draws, :field; east, north, time, rng)`.
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

## What this example checks

The generating surface has two features with different signs and directions
of motion. A fit can reproduce the station measurements while misrepresenting
the field between stations; compare the maps, rather than using agreement at
the observed rows as the only check. The truth is available here because the
data are synthetic. In an application, held-out locations or times provide a
more useful check of extrapolation than fitted residuals alone.

The spatial and temporal lengths govern different kinds of borrowing. A long
spatial length couples distant stations; a long temporal length keeps nearby
times similar. Replication helps identify observation noise, but does not create
new spatial locations. Compare the posterior lengths with their priors when
stations are sparse or the observed time span is short.

The maps summarize the latent field, not a complete future dataset. Their
posterior standard deviation combines uncertainty about fitted states and
hyperparameters with conditional variation at each new location. It excludes
the extra variation from a new noisy measurement. The slice makes the
pointwise intervals easier to read than the SD color scale alone.

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/map_checks.tsv";
    columns=[:quantity => "Grid check", :value => "Value"])
```

These checks summarize this generating surface on this grid. Pointwise
coverage across a single simulated field is not a calibration study: nearby
cells share information and are not independent replications. Repeat the
simulation across datasets before making a general coverage claim.

The neighbor count controls a covariance approximation, not just display
resolution. Increasing the map grid density does not improve that approximation.
Refit with different neighbor settings when scientific conclusions depend on
fine spatial structure. Changing the representation, mesh, or basis budget
also requires a new fit; replay evaluates the fitted model at new inputs.

For areal counts, smooth channel surfaces, mesh priors and minibatch SVI, see
[Spatial and space-time fields](space-time-fields.md). That guide explains
when `car`, `hsgp`, or another `gmrf` representation matches the data.

<details>
<summary>Reproduce</summary>

The [research README](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/space_time_fields/README.md)
records the model conventions, environment, summary files and validation.
Sampling runs locally; the documentation build reads those summaries and emits
Vega-Lite specifications with AlgebraOfVega.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/space_time_fields/fields.jl
julia --startup-file=no --project=docs docs/make.jl
```

```@eval
Main.BRMDocsComparisons.results_table("research/space_time_fields/results/receipt.tsv";
    columns=[:key => "Run", :value => "Value"])
```

</details>
