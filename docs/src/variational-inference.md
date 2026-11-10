````@raw html
---
title: Variational inference
description: "Minibatch stochastic variational inference for hierarchical models on BRM's ReactiveKernels backend: brm_svi_fit, what a unit is, how a fit stops, continuing a fit as data arrive in windows, a start from fits on slices of the data, and a comparison with full MCMC on a synthetic count model."
---
````

# Variational inference

Fit a hierarchical count model with `brm_svi_fit`, compare its Gaussian approximation
with NUTS, continue the fit as new observations arrive, and combine fits on slices
into a starting point. This example shows where the approximation loses population
uncertainty even when its site-effect estimates agree with NUTS.

## The model and the data

A monitoring network of sites is visited repeatedly. At each visit a count is recorded
together with a covariate `x`, and the count rate of a site depends on `x` through the
site's own intercept and slope. The two site effects are correlated. The data are a
NamedTuple of nested vectors, one entry per site holding that site's visits; no table
with one row per observation is needed.

```@eval
using Markdown
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# The values the data are simulated from.",
    ending_before="site_counts_data(; kwargs...)",
)
```

The model has a `@plate` over sites. The population part,
`a ~ 1 + (1 | s | site)` and `b ~ 1 + (1 | s | site)`, gives each site an intercept and
a slope drawn from one correlated block `s`. The plate cell reads that site's two values
and observes its counts.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:variational_inference),
    "research/variational_inference/svi.jl";
    before=:site_counts_model,
)
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:variational_inference),
    Main.BRMDocsComparisons.source_function(
        "research/variational_inference/svi.jl", :site_counts_model,
    ),
    :site_counts_model;
    title="Correlated site intercepts and slopes for Poisson counts",
    require_stan=true,
)
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let svi = vi_row("fits.tsv", r -> r.method == "SVI")
    Markdown.parse("The fits below use $(svi.sites) sites with $(svi.observations) counts in all. " *
        "The posterior has $(svi.coordinates) coordinates: the two population intercepts, the two " *
        "random-effect sds, the correlation, and two standardized effects per site.")
end
```

## Fit the model on minibatches

`brm_svi_fit` takes the model on the ReactiveKernels backend, `RKBRMI(brmi)`, with
Enzyme loaded for the gradients. Its units are those identified by `brm_nlme_view`:
here one site with its visits. In this model, every observation belongs to exactly one unit,
which is what makes a batch of units an unbiased estimate of the whole likelihood.

The approximation has two parts. The coordinates every unit reads, here the population
intercepts, the two sds and the correlation, are the globals; their factor is a
diagonal Gaussian (`family=MeanFieldGlobal()`, the default) or a diagonal plus a
low-rank part (`LowRankGlobal(r)`). Each unit's own coordinates, here a site's two
standardized effects, get a full Gaussian of their own. The family has no dependence
between a unit and the globals, or between two units.

One iteration draws the globals once and a batch of 32 sites. With the default
`unit_curvature=:orthogonal`, each selected two-coordinate site is evaluated at two
orthogonal antithetic pairs, giving four unit gradients. Adam updates the global
factor and a natural-gradient step updates each selected site's factor. Sites not
in the batch do not move. The NUTS reference uses `rk_logdensity_problem` through
WarmupHMC's quorum sampler: four chains are requested, with at least three required
to finish. Each starts from its own jittered point near the origin of the
unconstrained coordinates.

Quorum selection can favour faster chains; the retained-chain diagnostics cannot
rule out that selection bias.

These fitting calls run in the test environment and produce the cached results;
the documentation build does not sample.

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# SVI: one call.",
    ending_before="# ── Summaries",
)
```

Both return draws in the packed coordinates of the RK model, one column per draw, so
the same readers take them. `rk_restore_draws(rk, draws)` restores the named
parameters on their natural scale; the random-effect block comes back as its margin
sds, the Cholesky factor of their correlation and the standardized site draws:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# The population parameters of `site_counts_model`",
    ending_before="const POPULATION_LABELS",
)
```

### How the fit stops

The optimization runs in windows of about `N / B` iterations, at least 100, and
stops on its own in three phases. Exploring steps the globals at a constant rate
until a window's 10 % trimmed ELBO mean has stopped rising, with no upward trend over
the last windows. Settling steps by a fraction of each global's own scale, until
the same test passes again or its window limit is reached. Polishing decays the step
over 20 windows and returns the average of its iterates. `diagnostics.converged`
reports that the rule fired, which is not a proof that a slow drift has settled.

```@eval
let figures = Main.BRMDocsComparisons.example_module(:variational_inference_figures)
    Base.include(figures, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "figures.jl"))
    Main.BRMDocsComparisons.figure(Base.invokelatest(figures.elbo);
        caption="Trimmed ELBO means over optimization windows: blue exploring, orange settling, neutral ink polishing.")
end
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let svi = vi_row("fits.tsv", r -> r.method == "SVI"), nuts = vi_row("fits.tsv", r -> r.method == "NUTS"),
    elbo = vi_table("elbo.tsv")
    stages = [(s, count(r -> r.stage == s, elbo)) for s in ("exploring", "settling", "polishing")]
    Markdown.parse("The fit ran $(svi.iterations) iterations in " *
        join(("$n windows $s" for (s, n) in stages), ", ", " and ") *
        ", evaluating $(svi.unit_gradient_calls) site gradients, in $(vi_seconds(svi.seconds)). " *
        "The $(nuts.chains) retained NUTS chains took $(vi_seconds(nuts.seconds)) for $(nuts.draws) draws " *
        "($(nuts.gradient_evaluations) gradient evaluations after warm-up): " *
        "maximum R-hat $(vi_sig(nuts.max_rhat, 4)), minimum bulk ESS $(vi_int(nuts.min_bulk_ess)), " *
        "minimum tail ESS $(vi_int(nuts.min_tail_ess)), $(nuts.divergent) divergent transitions " *
        "($(vi_sig(nuts.divergent_percent)) % of retained draws). " *
        "Both ran on one CPU of a shared host; the times are what that run took, not a benchmark.")
end
```

### Compare with NUTS

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:variational_inference_figures).marginals();
    caption="Population medians with 50 % and 95 % intervals. Dashed lines mark generating values; SVI intervals are narrower than NUTS intervals.")
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let rows = vi_table("population.tsv")
    order = ["intercept a", "slope b", "sd of site intercepts", "sd of site slopes",
             "correlation of intercepts and slopes"]
    body = map(order) do p
        nuts = only(filter(r -> r.parameter == p && r.method == "NUTS", rows))
        svi = only(filter(r -> r.parameter == p && r.method == "SVI", rows))
        (p, vi_sig(nuts.generating), "$(vi_sig(nuts.median)) $(vi_interval(nuts))",
         "$(vi_sig(svi.median)) $(vi_interval(svi))", vi_sig(svi.sd / nuts.sd))
    end
    vi_markdown_table(["parameter", "generating value", "NUTS: median [95 % interval]",
                       "SVI: median [95 % interval]", "sd ratio SVI / NUTS"], body)
end
```

The population medians are close, but every SVI marginal is narrower. The factorized
Gaussian omits dependence between the globals and the site effects. Nothing in this
fit corrects or reweights its draws. Use full MCMC on a feasible reference dataset
to assess population uncertainty.

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:variational_inference_figures).site_means();
    caption="Each point is one site's intercept or slope deviation. Posterior means lie close to the equality line; the table also compares their uncertainty.")
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let rows = vi_table("approximation.tsv")
    vi_markdown_table(["coordinates", "count", "mean error, median (NUTS sds)", "mean error, largest",
                       "sd ratio SVI / NUTS: 5 %, median, 95 %"],
        [(r.group, r.coordinates, vi_sig(r.mean_error_median), vi_sig(r.mean_error_max),
          "$(vi_sig(r.sd_ratio_q05)), $(vi_sig(r.sd_ratio_median)), $(vi_sig(r.sd_ratio_q95))") for r in rows])
end
```

## Continue the fit as data arrive

A fitted approximation is a data-free summary of what was seen. `brm_svi_checkpoint(fit)`
returns it as a `BRMSVIState`: the global factor, one factor per unit keyed by the
unit's label, and the frozen preprocessing of the model. It holds no observations and
no compiled model, so it serializes and travels. `brm_svi_fit(state, builder(window))`
then fits a new window of observations with that state as the prior and the start:
the target is `q_previous(globals, units) * p(window | globals, units)`.

Units are matched by label. A site seen before continues from its carried factor; a
site that is new in a window starts from its authored prior under the carried global
factor; a site absent from a window keeps its factor in the state and recovers it when
it returns. Each observation enters once: the history is represented by the carried
Gaussians and never revisited, so a stream of windows is a sequential approximation
that can depend on the order of the windows and need not equal a fit of everything at
once.

The stream here cuts the eight visits into four windows of two. The windows contain
300, 340, 380 and 400 sites. A site contributes observations only after it arrives;
its earlier visits are absent from the accumulated reference data too:

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# Window `w` holds visits",
    ending_before="# Every observation seen up to",
)
```

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# The stream: calibrate on the first window",
    ending_before="checkpoint_bytes(state) =",
)
```

After each window, the streaming estimate of every population parameter is compared
with two fits of all the data seen up to that window: NUTS and a batch `brm_svi_fit`.

```@eval
Main.BRMDocsComparisons.figure(
    Main.BRMDocsComparisons.example_module(:variational_inference_figures).streaming();
    caption="Population medians with 50 % and 95 % intervals: blue NUTS, orange streaming SVI, neutral ink batch SVI. All methods use exactly the observations seen so far.")
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let rows = vi_table("streaming_cost.tsv")
    vi_markdown_table(["window", "sites", "new counts", "counts so far", "streaming fit", "iterations",
                       "checkpoint", "batch SVI on everything", "NUTS on everything"],
        [(r.window, r.sites, r.new_observations, r.accumulated_observations, vi_seconds(r.streaming_seconds),
          r.streaming_iterations, "$(vi_sig(r.checkpoint_bytes / 1000, 3)) kB", vi_seconds(r.batch_seconds),
          vi_seconds(r.nuts_seconds)) for r in rows])
end
```

The work of a streaming update follows the window, not the history: each update
minibatches only the units of its window, and the checkpoint grows with the number of
distinct sites, not with the observations. The narrow global sds of the batch fit carry
over to the stream; carrying the factors does not widen them.

## Start from fits on slices of the data

Where the units fall into slices that can be fitted independently, regions whose sites
each lie in one region for example, `brm_svi_combine` turns the slices' checkpoints into a
start for the full fit. The global factors multiply as Gaussian sites with the prior
counted once, and each unit takes its slice's factor. The result is a start and not a
prior: the full fit's target is the full model, and the start only decides where its
optimizer begins. Slices that measure something else than the full data, such as the
amplitude of a spatial field fitted inside one region, are named in `exclude` and start
cold.

```@eval
Main.BRMDocsComparisons.source_code_region(
    "research/variational_inference/svi.jl";
    starting_at="# Fits of four slices of the sites",
    ending_before="function section_combine",
)
```

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let rows = vi_table("combine.tsv")
    vi_markdown_table(["start of the full fit", "optimization", "total including slices", "iterations", "of them exploring",
                       "site gradients", "largest population mean error (NUTS sds)"],
        [(r.start, vi_seconds(r.seconds), vi_seconds(r.total_seconds), r.iterations, r.exploring_iterations, r.unit_gradient_calls,
          vi_sig(r.max_population_mean_error_sds)) for r in rows])
end
```

## Two-stage fits and Laplace

[`brm_two_stage_fit`](@ref) estimates the population by SAEM (minibatches of units
in the E-step) and then samples every unit's conditional posterior at those estimates
with a sampler of your choice, usually WarmupHMC's `adaptive_warmup_mcmc` on the
unit's own problem. This targets each unit's conditional posterior rather than a
Gaussian approximation to it. Population uncertainty is not propagated. It needs
the NLMEEstimation extension.

[`brm_svi_fit`](@ref) with `laplace=GaussianBlockLaplace()` integrates a shared
Gaussian block, such as a Hilbert-space Gaussian process the units read, jointly with
Gaussian population coefficients and unit effects, conditional on the remaining
variational globals. It applies to affine Gaussian likelihoods and visits the whole
window at each update instead of a batch. The per-unit Laplace route outside SVI,
`brm_laplace_problem`, integrates every unit's effects out at their conditional modes
and runs NUTS on the population coordinates alone.

```@raw html
<details><summary>Reproduce</summary>
```

The [research recipe](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/variational_inference)
contains the synthetic data generator, fit commands, cache rules and figure source.
Fits run once locally; the documentation build renders their summary tables and
AlgebraOfVega specifications. It never samples.

```@eval
Main.BRMDocsComparisons.results_table("research/variational_inference/results/receipt.tsv";
    columns=[:field => "receipt", :value => "value"],
    select=r -> r.field in ("recipe", "commit", "uncommitted_changes", "started", "host",
                            "Julia", "BayesianRegressionModels", "WarmupHMC", "Enzyme", "threads"))
```

The [complete receipt](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/variational_inference/results/receipt.tsv)
also records settings, seeds, input keys and fit source hashes.

```@raw html
</details>
```
