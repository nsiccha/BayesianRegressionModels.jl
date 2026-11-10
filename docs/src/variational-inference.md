````@raw html
---
title: Variational inference
description: "Minibatch stochastic variational inference for hierarchical models on BRM's ReactiveKernels backend: brm_svi_fit, what a unit is, how a fit stops, continuing a fit as data arrive in windows, a start from fits on slices of the data, and a comparison with full MCMC on a synthetic count model."
---
````

# Variational inference

`brm_svi_fit` fits a Gaussian approximation to the posterior of a hierarchical model
from random batches of its units. It is for the models where NUTS over every unit is
too slow, and for data that arrive in windows: the fit of one window continues into the
next without touching the old observations again. This page fits one synthetic model
both ways, with `brm_svi_fit` and with NUTS, and shows where the two agree and where the
approximation is narrower than the posterior.

Everything shown is built or measured from
[`research/variational_inference/svi.jl`](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/variational_inference/svi.jl):
the code blocks are read from it at build time, and every number and figure comes from
the tables it writes.

## The model and the data

A monitoring network of sites is visited repeatedly. At each visit a count is recorded
together with a covariate `x`, and the count rate of a site depends on `x` through the
site's own intercept and slope. The two site effects are correlated. The data are a
NamedTuple of nested vectors, one entry per site holding that site's visits; no table
with one row per observation is needed.

```@eval
using Markdown
Markdown.MD([Markdown.Code("julia",
    Main.BRMDocsComparisons.source_function("research/variational_inference/svi.jl", :simulate_sites))])
```

The model is a `@brm` declaration with a `@plate` over the sites. The population part,
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
Enzyme loaded for the gradients. Its **units** are the groups of the model's plate:
here one site with its visits. Every observation has to belong to exactly one unit,
which is what makes a batch of units an unbiased estimate of the whole likelihood.

The approximation has two parts. The coordinates every unit reads, here the population
intercepts, the two sds and the correlation, are the **globals**; their factor is a
diagonal Gaussian (`family=MeanFieldGlobal()`, the default) or a diagonal plus a
low-rank part (`LowRankGlobal(r)`). Each unit's own coordinates, here a site's two
standardized effects, get a full Gaussian of their own. The family has no dependence
between a unit and the globals, or between two units.

One iteration draws the globals once and a batch of 32 sites, evaluates each site's
term at an antithetic pair of draws of its effects, steps the global factor by Adam
and each selected site's factor by a natural-gradient step. Sites not in the batch do
not move. The four NUTS chains run on the same log density, `rk_logdensity_problem`,
through WarmupHMC's adaptive sampler, each from its own jittered start near the
centre of the prior.

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
stops on its own in three phases. **Exploring** steps the globals at a constant rate
until a window's 10 % trimmed ELBO mean has stopped rising, with no upward trend over
the last windows. **Settling** continues at a smaller step that is a fraction of each
global's own scale, until the same test passes again. **Polishing** decays the step
over 20 windows and returns the average of its iterates. `diagnostics.converged`
reports that the rule fired, which is not a proof that a slow drift has settled.

![The ELBO of the fit by window, coloured by stage](assets/variational-inference/elbo.png)

```@eval
include(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "variational_inference", "tables.jl"))
let svi = vi_row("fits.tsv", r -> r.method == "SVI"), nuts = vi_row("fits.tsv", r -> r.method == "NUTS"),
    elbo = vi_table("elbo.tsv")
    stages = [(s, count(r -> r.stage == s, elbo)) for s in ("exploring", "settling", "polishing")]
    Markdown.parse("The fit ran $(svi.iterations) iterations in " *
        join(("$n windows $s" for (s, n) in stages), ", ", " and ") *
        ", evaluating $(svi.unit_gradient_calls) site gradients, in $(vi_seconds(svi.seconds)). " *
        "The four NUTS chains took $(vi_seconds(nuts.seconds)) for $(nuts.draws) retained draws " *
        "($(nuts.gradient_evaluations) gradient evaluations after warm-up): " *
        "maximum R-hat $(vi_sig(nuts.max_rhat, 3)), minimum bulk ESS $(vi_int(nuts.min_bulk_ess)), " *
        "minimum tail ESS $(vi_int(nuts.min_tail_ess)), $(nuts.divergent) divergent transitions. " *
        "Both ran on one CPU of a shared host; the times are what that run took, not a benchmark.")
end
```

### What the approximation gets right

The marginal posteriors of the five population parameters under NUTS and under the
fitted approximation:

![Marginal posteriors of the population parameters under NUTS and SVI](assets/variational-inference/population-marginals.png)

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

The medians agree to within a fraction of a posterior sd. The spread does not: the
mean-field factor is narrower than the posterior, most for the two random-effect sds,
whose posteriors are skewed and coupled with the site effects. This is the known
behaviour of a factorized Gaussian approximation and nothing in `brm_svi_fit`
corrects or reweights it. There is deliberately no importance-sampling check of the
approximation. Where the width of the population posterior is the result, use
`brm_two_stage_fit` or NUTS, below.

The site effects fare better. Each point is one site's intercept or slope deviation,
its posterior mean under SVI against its posterior mean under NUTS:

![Site effects: SVI posterior mean against NUTS posterior mean](assets/variational-inference/site-effects.png)

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

The stream here cuts the eight visits into four windows of two. Three hundred sites
report in the first window and forty more arrive with each later window:

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

![Streaming estimates per window beside NUTS and batch SVI on the accumulated data](assets/variational-inference/streaming.png)

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
    vi_markdown_table(["start of the full fit", "optimization", "iterations", "of them exploring",
                       "site gradients", "largest population mean error (NUTS sds)"],
        [(r.start, vi_seconds(r.seconds), r.iterations, r.exploring_iterations, r.unit_gradient_calls,
          vi_sig(r.max_population_mean_error_sds)) for r in rows])
end
```

## Two-stage fits and Laplace

Two other routes share the unit factorization and return draws in the same packed
shape, so they can stand in for each other.

[`brm_two_stage_fit`](@ref) estimates the population by SAEM (minibatches of units
in the E-step) and then samples every unit's conditional posterior at those estimates
with a sampler of your choice, usually WarmupHMC's `adaptive_warmup_mcmc` on the
unit's own problem. Each unit's posterior is exact given the population values; the
population's own uncertainty is not propagated, which matters with few units and not
with many. It needs the NLMEEstimation extension.

[`brm_svi_fit`](@ref) with `laplace=GaussianBlockLaplace()` integrates a shared
Gaussian block, such as a Hilbert-space Gaussian process the units read, jointly with
Gaussian population coefficients and unit effects, conditional on the remaining
variational globals. It applies to affine Gaussian likelihoods and visits the whole
window at each update instead of a batch. The per-unit Laplace route outside SVI,
`brm_laplace_problem`, integrates every unit's effects out at their conditional modes
and runs NUTS on the population coordinates alone.
