# BRM feature atlas

Every executable example in the user guide has the same four semantic views:
the BRM declaration, the StanBlocks model BRM emitted, the complete Stan source
emitted from that model, and the Turing model selected directly from the BRM.
Choose a tab for one view, or **Compare** and select any two or more views for a
side-by-side reading. If a backend does not support an example, its pane remains
present and shows the exact construction error produced by the current build.

Nothing below is copied output. The docs build evaluates the displayed source
and derives all three emission panes from it.

## Gaussian population model and effect priors

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
gaussian = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    effect(:, :) ~ Normal(0, 2)
    effect(mu, x) ~ Normal(0, 0.25)
    y ~ Normal(mu, sigma)
end)((; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4]))
""", :gaussian; title="Gaussian population model")
```

## Partly missing Gaussian response

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
missing_gaussian = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    mi(y) ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    y=Union{Missing,Float64}[0.2, missing, -0.4, missing],
))
""", :missing_gaussian; title="Partly missing Gaussian response")
```

## Shared and independent multi-response model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
multi_response = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)

    log(rate) ~ 1 + z
    count ~ Poisson(rate)

    y_replicate ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0],
    z=[0.25, -0.5, 1.0],
    y=[0.2, 1.1, -0.4],
    count=[0, 2, 4],
    y_replicate=[-0.1, 0.3, 0.7],
))
""", :multi_response; title="Shared and independent responses")
```

## Canonical Binomial-logit model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
binomial = (@brm begin
    logit(p) ~ 1 + x
    successes ~ Binomial(trials, p)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    trials=[2, 4, 6, 3],
    successes=[0, 2, 5, 1],
))
""", :binomial; title="Binomial logit")
```

## Poisson model with transformed data and an offset

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
poisson_offset = (@brm begin
    log(rate) ~ 1 + center(x) + offset(log(exposure))
    counts ~ Poisson(rate)
end)((;
    x=[1.0, 2.0, 4.0, 8.0],
    exposure=[2.0, 4.0, 8.0, 16.0],
    counts=[0, 2, 5, 7],
))
""", :poisson_offset; title="Poisson exposure model")
```

## Distributional count model

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
negative_binomial = (@brm begin
    log(mu) ~ 1 + x
    log(phi) ~ 1
    y ~ NegativeBinomial2(mu, phi)
end)((; x=[-1.0, 0.5, 2.0, 0.25], y=[0, 2, 5, 1]))
""", :negative_binomial; title="Negative-binomial mean and precision")
```

## Distributional count model with independent group slopes

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped_negative_binomial = (@brm begin
    log(mu) ~ 1 + x + (1 + x || subject)
    log(phi) ~ 1 + z + (0 + z || batch)
    y ~ NegativeBinomial2(mu, phi)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    z=[0.0, 1.0, -0.5, 0.75],
    subject=["b", "a", "b", "c"],
    batch=[2, 1, 1, 2],
    y=[0, 2, 5, 1],
))
""", :grouped_negative_binomial;
title="Independent mean and precision group slopes")
```

## Shared distributional group covariance

The repeated `joint` ID makes the mean and precision slopes slices of one
four-dimensional covariance block for each subject. Removing the ID would make
the two predictor blocks independent.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
shared_group_negative_binomial = (@brm begin
    log(mu) ~ 1 + x + (1 + x | joint | subject)
    log(phi) ~ 1 + z + (1 + z | joint | subject)
    y ~ NegativeBinomial2(mu, phi)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25, -0.75, 1.25],
    z=[0.0, 1.0, -0.5, 0.75, 0.25, -1.0],
    subject=["b", "a", "b", "c", "a", "c"],
    y=[0, 2, 5, 1, 3, 4],
))
""", :shared_group_negative_binomial;
title="Shared mean and precision group covariance")
```

## Categorical population terms

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
categorical = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + factor(group; ref=3) + x + x & group
    effect(mu, group) ~ Normal(0, 0.5)
    y ~ Normal(mu, sigma)
end)((;
    group=[1, 2, 3, 1, 2, 3],
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
))
""", :categorical; title="Categorical population design")
```

## Correlated group effects

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
grouped = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | subject)
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=[2, 1, 2, 3],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :grouped; title="Correlated random slopes")
```

## Multiple crossed group effects

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
crossed_groups = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 + x | subject) + (1 | item) + (0 + x || site)
    outcome ~ Normal(mu, sigma)
end)((;
    x=[-1.0, 0.5, 2.0, 0.25],
    subject=["b", "a", "b", "c"],
    item=[2, 1, 1, 2],
    site=[10, 10, 20, 20],
    outcome=[0.2, 1.1, -0.4, 0.7],
))
""", :crossed_groups;
title="Crossed correlated and independent group blocks")
```

## Interval-censored evidence

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
interval_normal = (@brm begin
    sigma ~ Exponential(2)
    mu ~ 1 + x
    lower ~ interval_censored(Normal(mu, sigma); upper=upper)
end)((;
    x=[-1.0, 0.0, 1.0],
    lower=[-0.4, 0.1, 0.8],
    upper=[-0.1, 0.4, 1.2],
))
""", :interval_normal; title="Interval-censored Normal evidence")
```

## A StanBlocks-only surface

The Turing pane is deliberately not removed for unsupported features. This
smooth example exercises the same build path and preserves the current
fail-closed reason beside the successful StanBlocks and Stan emissions.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
smooth = (@brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(x)
    y ~ Normal(mu, sigma)
end)((;
    x=collect(range(-2, 2; length=20)),
    y=sin.(collect(range(-2, 2; length=20))),
))
""", :smooth; title="Spline term with an unsupported Turing executor")
```
