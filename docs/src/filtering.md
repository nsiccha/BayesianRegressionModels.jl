```@raw html
---
title: Sequential Monte Carlo and filtering
description: "Fit a count time series with SMC, update it as observations arrive, carry a posterior between windows, and compare filtering and parameter uncertainty with MCMC."
---
```

# Sequential Monte Carlo and filtering

Fit an initial count series with SMC, update it with new counts, and compare
online state estimates and parameter uncertainty with a full-data MCMC fit.

| Method | Retained information | Target of an update |
| --- | --- | --- |
| Cumulative SMC | Particles, old and new data | Full posterior; moves replay histories. |
| Carried windows | Gaussian posterior summary | Carried Gaussian × new window likelihood. |
| Ordinary filter | Parameter members and conditional state clouds | New observations reweight fixed members. |
| Learning filter | Members, state clouds and a bounded window | Replenishment within the window; Gaussian boundaries. |

## Daily counts

The synthetic data contain daily Poisson counts from units with separate
log-rate offsets and latent paths, with shared autoregressive parameters. The generator
records the latent paths, so filtered intervals can be compared with their
known values. Persistence is `rho`; innovation standard deviation is `tau`.

```@example filtering
using BayesianRegressionModels, Distributions
model_file = "research/filtering/model.jl"
model_file = joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, model_file) # hide
include(model_file)
fixture = filtering_fixture()
data = fixture.data
training = filtering_window(data, 1, 20)
@assert all(all(iszero, y[21:end]) for y in training.y)
@assert all(all(iszero, e[21:end]) for e in training.exposure)
(units=data.U, days=data.T, observed_counts=sum(sum, data.y))
```

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:filtering),
    "research/filtering/model.jl";
    before=:filtering_model,
)
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:filtering),
    Main.BRMDocsComparisons.source_function("research/filtering/model.jl", :filtering_model),
    :filtering_model;
    title="Poisson counts with a latent state per unit",
    require_stan=true,
)
```

`T` fixes the latent horizon. Outside a supplied observation window,
`exposure` and the count are zero: the Poisson likelihood is one. Cumulative
updates therefore retain the same coordinates without observing future counts.
Prior settings are data inputs.

The inference snippets below run in the repository's pinned `test` environment;
the docs environment lacks the inference and differentiation dependencies.
The build executes the model, data checks, tables and figures. Sampling runs
separately, with its cached receipt in Reproduce.

```julia
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using SequentialMonteCarlo, WarmupHMC, Enzyme, ADTypes, DifferentiationInterface
using Random, Serialization, TreeArrays
rk = RKBRMI(filtering_model(training))
```

## Fit and tune the first window

An independent pilot learns a reference distribution and mutation kernels.
Production SMC then moves a weighted cloud from that normalized reference
toward the posterior, resampling as the temperature increases.

```julia
d = length(coordinate_names(rk.model.layout))
schedule = brm_smc_tune(rk;
    rng=Xoshiro(1201), init=zeros(d),
    pilot=:adaptive, reference=:pilot, pilot_steps=64,
    grid_start=:independent,
    options=SMCOptions(moves=3, max_tree_depth=7))

fit = brm_smc_fit(rk; rng=Xoshiro(201), particles=1024,
    options=SMCOptions(tuning=schedule))
@assert fit.state.temperature == 1   # the bridge is complete
```

Pilot draws are discarded. `reference=:pilot` fits and inflates a Gaussian
importance distribution; the bridge includes its density. The reference is
not the prior, and it does not ensure exploration of every posterior mode.
A tuning schedule belongs to its particular bridge. `grid_start=:independent`
parallelizes its pilots; production mutation uses `SMCOptions(nthreads=...)`.

The cloud is stored in unconstrained coordinates. Query model quantities
through the backend, retaining the weights:

```julia
post = rk_draw_tree(rk, fit.state.positions, (:level, :rho, :tau))
rho = vec(collect(TreeActualArray(post.rho)))
rho_mean = sum(rho .* fit.state.weights)
```

Retain the weights when extracting quantities. Resampling can give equal
weights even when the cloud explores poorly; inspect stage history and
compare independent runs and posterior spread.

## Update with cumulative data

```julia
cumulative = filtering_window(data, 1, 40)
bridge = brm_smc_tune(fit, cumulative;
    rng=Xoshiro(1202), init=zeros(d), pilot=:adaptive, pilot_steps=64,
    grid_start=:independent,
    options=SMCOptions(moves=3, max_tree_depth=7))
updated = brm_smc_update(fit, cumulative;
    options=SMCOptions(tuning=bridge))
```

Supply all counts observed so far. The update counts the prior once and
continues the saved random-number stream. Moves still evaluate the old and
new histories. BRM checks coordinate compatibility but cannot detect an
accidentally omitted observation. Growing a sampled path changes its coordinate
frame; use a fixed horizon or selective windows.

## Carry a posterior into a new window

```julia
previous = brm_smc_carry(fit; basis=:fit, max_reflections=16)
window = filtering_window(data, 21, 40)

partial = brm_smc_update(previous, filtering_model(window);
    coordinates=:auto, rng=Xoshiro(301), particles=1024, stages=0,
    options=SMCOptions(moves=3, max_tree_depth=7))
window_schedule = brm_smc_tune(previous, filtering_model(window);
    coordinates=:auto, rng=Xoshiro(1301),
    init=zeros(size(partial.state.positions, 1)),
    pilot=:adaptive, reference=:pilot, pilot_steps=64,
    grid_start=:independent,
    options=SMCOptions(moves=3, max_tree_depth=7))
window_fit = brm_smc_update(previous, filtering_model(window);
    coordinates=:auto, rng=Xoshiro(301), particles=1024,
    options=SMCOptions(tuning=window_schedule))
next = brm_smc_carry(window_fit; basis=:fit, max_reflections=16)
```

Supply each observation in one window. A boundary replaces the posterior
with a Gaussian, losing non-Gaussian features. `basis=:fit` retains bounded
correlation directions; the default diagonal carry discards correlations.

`coordinates=:auto` selects coordinates needed by the likelihood, fresh
coordinates and their prior parents. Omitted coordinates retain their carried
conditional distribution. A scan may still require most of its earlier path.

```julia
mapping = brm_smc_coordinates(window_fit)
full_draw = brm_smc_recover(window_fit,
    window_fit.state.positions[:, 1]; rng=Xoshiro(309))
```

Recovery preserves the compact particle and samples omitted coordinates
conditionally. For posterior draws, select compact columns with their weights.
Existing indices and fitted categorical levels must retain their meanings.

### When a new unit appears

A smaller count model makes the selection visible. Each unit has its own
log rate, with a shared level:

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:filtering),
    Main.BRMDocsComparisons.source_function("research/filtering/model.jl", :growing_model),
    :growing_model;
    title="Unit-specific counts with a shared level",
    require_stan=true,
)
```

```julia
initial_data = (U=2, unit=[1, 2], y=[6, 14],
    level_location=2.0, level_scale=1.0, offset_scale=0.5)
initial = brm_smc_fit(growing_model(initial_data); rng=Xoshiro(701),
    particles=1024, options=SMCOptions(moves=3, max_tree_depth=7))
carried = brm_smc_carry(initial; basis=:correlation)
next_counts = (; initial_data..., U=3, unit=[3], y=[9])
selected = brm_smc_update(carried, growing_model(next_counts);
    coordinates=:auto, rng=Xoshiro(702), particles=1024,
    options=SMCOptions(moves=3, max_tree_depth=7))
brm_smc_coordinates(selected)
```

The new count belongs to a third unit that the first fit never saw. Its
likelihood reads the fresh coordinate `log_rate[3]`, whose prior reads the
shared `level`, so those two coordinates are active. The first two units'
rates remain in the carried conditional distribution given `level`, and
`brm_smc_recover` returns all four values. The target combines the carried
Gaussian marginal of `level`, the prior of `log_rate[3]` and the new count's
likelihood; it does not count the old priors or observations again.

## Filter the next observation

The filter derives its transition and observation law from the same model.
There is no second hand-written Poisson likelihood or state transition to
keep in agreement with it.

```julia
records = [(observation=(y=[ys[t] for ys in data.y],),
    inputs=(exposure=ones(data.U),)) for t in 1:data.T]
filter = brm_filter(fit; inputs=(:exposure,),
    groups=(i=1:data.U,), particles=64, rng=Xoshiro(401))
warm = brm_filter_stream(filter, records[1:20]; condition_state_only=true)
next = brm_filter_step(warm, records[21].observation; inputs=records[21].inputs)
next.state.predictive_logscore
```

The latest latent states are in the retained carries. For the first unit:

```julia
states = [getproperty(carry, Symbol("i.state"))[end, 1]
    for carry in next.state.carries]
state_mean = sum(states .* next.state.weights)
```

Initialization starts before the first time slice, even when the parameters
come from a fit. The state-only pass warms the latent states using the
calibration observations without counting their likelihood again in the
parameter weights or new-data score. Each subsequent score is computed
before conditioning on its observation.

```@eval
Base.include(Main.BRMDocsComparisons.example_module(:filtering),
    joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research/filtering/figures.jl"))
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:filtering).filtered_states();
    caption="Blue: filtered median and 50% / 90% intervals after calibration. Orange dashed: generating latent state. Each interval uses only counts available that day.")
```

The parameter members come from the calibration cloud. Each has its own
conditional state particles, so the total state-particle count is the number
of members times `particles`. Ordinary filtering reweights those members but
does not replenish them. Inspect `state.parameter_mass` for concentration as
the stream grows.

```@eval
using Markdown
let rows = Main.BRMDocsComparisons.example_module(:filtering).filtering_table("filter_scores")
    final = last(rows)
    Markdown.parse("At the final day, the effective parameter-member count is $(round(final.parameter_ess; digits=1)). This describes this run, not a calibration study.")
end
```

With fixed group and particle counts, bounded lag and bounded history,
filter storage stays bounded. Here the latent state is continuous:
*finite-state* means retained computational state. Missing observations advance
selected groups without conditioning; omitted groups do not advance. Declare
new per-step covariates in `inputs`.

## Learn the static parameters while filtering

```julia
calibration = brm_smc_carry(fit; parameters=:filter,
    inputs=(:exposure,), basis=:correlation)
learning = brm_filter(calibration, filtering_model(training);
    boundary=warm, groups=(i=1:data.U,), rng=Xoshiro(402),
    members=256, particles=128, window_length=10,
    options=SMCOptions(moves=3, max_tree_depth=7))
learning = brm_filter_stream(learning, records[21:end])
```

`parameters=:filter` retains only static coordinates, dropping the historical
latent path. `basis=:correlation` fits their correlation structure. The warmed
boundary supplies state particles and clocks independently of the new
parameter members. A learning filter cannot itself perform state-only warm-up.

New observations reweight members; replenishment and particle-Metropolis
moves restore diversity within the retained window. At a boundary, a Gaussian
summarizes the parameter posterior and an independent pool summarizes the
states. That discards parameter–state dependence. This is assumed-density
inference, rather than exact full-history SMC². Read weighted parameter
positions from `state.positions` and `state.parameter_mass`, and inspect
`state.history` for replenishment and boundary resets.

## Compare with MCMC

Every row uses the same observations and model. Batch and cumulative SMC
target the full posterior. The carried windows and learning filter introduce
the boundary approximations described above. Compare location and spread;
agreement of a posterior mean alone can hide lost uncertainty.

The MCMC reference uses WarmupHMC's completion sampler on the full-data RK
log density:

```julia
rk_full = RKBRMI(filtering_model(data))
problem = rk_logdensity_problem(rk_full;
    ad_backend=AutoEnzyme(mode=Enzyme.Reverse))
mcmc = completion_warmup_mcmc(Xoshiro.([601, 602, 603, 604]), problem;
    min_completed=3, n_draws=1000,
    init=(position=zeros(d), squared_scale=ones(d)))
```

The retained chains are checked across all unconstrained coordinates:

```@eval
Main.BRMDocsComparisons.results_table("research/filtering/results/diagnostics.tsv";
    columns=[:max_rhat=>"Maximum R-hat", :min_bulk_ess=>"Minimum bulk ESS",
        :min_tail_ess=>"Minimum tail ESS", :divergences=>"Divergences", :chains=>"Retained chains"])
```

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:filtering).marginal_comparison();
    caption="Medians with thick 50% and thin 90% intervals after all observations. Orange methods approximate earlier posteriors at window boundaries; blue methods retain the full history.")
```

```@eval
using Markdown
let rows = Main.BRMDocsComparisons.example_module(:filtering).filtering_table("marginals")
    sd(method, parameter) = only(r.sd for r in rows if r.method == method && r.parameter == parameter)
    carried = round(sd("Carried windows", "tau") / sd("MCMC", "tau"); digits=1)
    learning = round(Int, 100 * sd("Learning filter", "tau") / sd("MCMC", "tau"))
    Markdown.parse("Batch and cumulative SMC agree closely with the MCMC reference here. " *
        "Carried windows produce much wider uncertainty: the innovation-scale SD is $(carried) times the MCMC SD. " *
        "The learning filter's innovation-scale SD is $(learning)% of the MCMC value. " *
        "These differences describe this dataset and configuration; they are not accuracy guarantees.")
end
```

Independent SMC runs check agreement without an MCMC reference:

```julia
full_schedule = brm_smc_tune(rk_full;
    rng=Xoshiro(1500), init=zeros(d),
    pilot=:adaptive, reference=:pilot, pilot_steps=64,
    grid_start=:independent,
    options=SMCOptions(moves=3, max_tree_depth=7))
replicates = brm_smc_runs(rk_full; seeds=(501, 502, 503), particles=1024,
    options=SMCOptions(tuning=full_schedule))
replicates.comparison.worst
replicates.comparison.sd_ratio_min
replicates.comparison.sd_ratio_max
```

The comparison reports differences between run means in pooled-SD units and
ratios of within-run standard deviations. These are descriptive agreement
measures, not Monte Carlo standard errors. Independent runs can agree while
missing the same region of the posterior.

## Checkpoints

Bound work by completed stages, save a checkpoint, and reconstruct the
original training model to resume:

```julia
partial = brm_smc_update(fit, cumulative; stages=1,
    options=SMCOptions(tuning=bridge))
serialize("smc-checkpoint.jls", brm_smc_checkpoint(partial))

rk = RKBRMI(filtering_model(training))
restored = brm_smc_resume(rk, deserialize("smc-checkpoint.jls"); data=cumulative)
completed = brm_smc_fit(restored)
```

After later cumulative updates, also supply `previous_data` for the
immediately preceding cumulative dataset. A carried-window checkpoint needs
only that window and its model declaration:

```julia
restored_window = brm_smc_resume(filtering_model(window), brm_smc_checkpoint(window_fit))
restored_filter = brm_filter_resume(rk, brm_filter_checkpoint(next))
```

Checkpoints retain particle state and identity records. Keep model code and
data separately and retain the numerical environment: endpoint densities must
reproduce exactly. Only deserialize trusted
files. Production checkpoints preserve completed tuning; interrupted pilots
need their own `checkpoint_dir` and tuning-resume path.

If moving a production checkpoint to another CPU changes endpoint scores only
by roundoff, explicitly rescore it on the destination host before resuming
(BRM 0.2.279, ReactiveKernelsPPL 0.1.156):

```julia
checkpoint = deserialize("smc-checkpoint.jls")
rescored = brm_smc_rescore(rk, checkpoint; data=cumulative, rtol=1e-10)
last(rescored.state.rescored)  # changed, max_abs, max_rel, rtol, codegen
serialize("smc-checkpoint-rescored.jls", rescored)
restored = brm_smc_resume(rk, rescored; data=cumulative)
```

Each score must differ by at most `rtol * max(abs(saved_score), 1)`.
The returned checkpoint records the change in `rescored.state.rescored`;
the input stays unchanged. Resume still requires exact scores. Rescore keeps
all model, data, version and state checks and cannot migrate an older-version
checkpoint. It uses the same reconstruction arguments as resume:
`previous_data` for later cumulative updates, `model::BRMI` for a carried
window (including compact coordinates), or a Laplace problem with `current`
and, when needed, `previous`. Tuning-pilot checkpoints have no rescore adapter.

## A whole series as a likelihood

A filter can also run *inside* a model likelihood. A plate cell then observes
one unit's whole series under a law that integrates its time-local states
out. Outer MCMC or variational inference fits the remaining parameters, and
each density evaluation runs the filter over that unit's series. This differs
from `brm_filter_step`, which advances an already fitted model online.

The [continuous-time state-space example](state-space.md) shows the
model declarations and filter bodies for this approach, including Kalman and
nonlinear approximations and checks of the approximation's accuracy.

```@raw html
<details><summary>Reproduce</summary>
```

The [research notes and scripts](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/filtering)
record settings, seeds, checkpoint handling and the numerical comparison.
The saved main comparison uses ReactiveKernelsPPL 0.1.145. Its checkpoints
require their original environment; rescore does not migrate them. To run the
example with the current pinned versions, choose a fresh checkpoint directory.

```sh
julia --project=test test/setup_env.jl
julia --project=test research/filtering/fit.jl /path/to/new/filtering-checkpoints
julia --project=docs docs/make.jl
```

```@eval
Main.BRMDocsComparisons.results_table("research/filtering/results/receipt.tsv";
    columns=[:quantity=>"Run", :value=>"Value"])
```

```@raw html
</details>
```
