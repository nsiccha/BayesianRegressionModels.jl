# Model syntax

Build a regression with several predictors, shared group effects and reusable
components from one `@brm` declaration. The examples use named tuples; a data
frame works too. Columns may have different lengths when they describe
different observation or group axes.

## Formulas and links

This model predicts both the mean and the standard deviation. The subject's
intercept and slope are correlated, and the scale stays positive through its
explicit log link.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:SyntaxGuide), raw"""
using BayesianRegressionModels

builder = @brm begin
    mu ~ 1 + x + (1 + x | person | subject)
    log(sigma) ~ 1 + x
    y ~ Normal(mu, sigma)

    effect(mu, :) ~ Normal(0, 1)
    effect(sigma, Intercept) ~ Normal(-1, 0.5)
    effect(sigma, x) ~ Normal(0, 0.2)
    sd(:, person) ~ Exponential(0.5)
    cor(:, person) ~ LKJCholesky(2, 2)
end
data = (; subject=["a", "a", "b", "b", "c", "c"],
          x=[-1., 1., -1., 1., -1., 1.],
          y=[-0.6, 0.9, -0.2, 1.4, -0.8, 0.5])
model = builder(data)
""", :model; title="Mean and scale regression", require_stan=true)
```

`~` introduces a formula or a probability law; `=` computes a value. A formula
allocates coefficients for its terms. Write `1` for an intercept: BRM does not
insert one. The name `sigma` above holds the response-scale value
`exp(linear predictor)`. Its prior address is `effect(sigma, ...)`; applying
`exp` to `sigma` again would apply the link twice.

The BRM, intermediate-representation, StanBlocks and Stan panes all come from
the same bound model. The following inspection checks the authored inputs and
keeps the prior declarations visible.

```@example syntax
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:SyntaxGuide).model
(inputs=data_columns(model), priors=string(priors_of(model)))
```

| Familiar formula | BRM meaning |
| --- | --- |
| `1 + x + z` | An intercept and two population effects |
| `x * z` / `x & z` | Main effects plus interaction / interaction alone |
| `(1 + x \| subject)` | Correlated subject intercepts and slopes |
| `(1 + x \|\| subject)` | Independent subject intercepts and slopes |
| `(1 \| p \| subject)` in two predictors | One correlation block spanning both predictors |
| `(1 \| gr(subject; by=arm))` | Separate group-effect distributions by stratum |
| Two response statements | A joint model; shared names share parameters |
| `y ~ Normal(@brm(1 + x), exp(@brm(1)))` | Nested mean and log-scale formulas |

Integer, string and categorical columns are factors; use floating-point values
for a numeric trend. With an intercept, factors use treatment contrasts.
Without one, the first eligible factor uses one coefficient per level.
`factor(g; cmc=false)` keeps treatment coding; `factor(g; ref=k)` chooses a
reference level. [The reference](terms-reference.md#columns-and-formula-operators) gives
the interaction and ordinal exceptions.

## Nonlinear models and plates

Ordinary assignments combine sampled values without adding coefficients.
This exponential-decay model has one rate per unit and several observations
per unit. A `groups` binding connects the two axes.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:PlateGuide), raw"""
decay = @brm begin
    log(rate) ~ 1 + (1 | unit)
    rows = groups(observation_unit => unit)
    @plate for i in eachindex(unit)
        mean_y = amplitude * exp.(-rate[i] .* time[rows[i]])
        y[rows[i]] ~ Normal(mean_y, noise)
    end

    amplitude ~ LogNormal(0, 0.5)
    noise ~ Exponential(0.2)
    effect(rate, Intercept) ~ Normal(-1, 0.5)
    sd(rate, unit) ~ Exponential(0.3)
end
model = decay((; unit=["a", "b"],
    observation_unit=["a", "b", "a", "b", "b"],
    time=[0., 0., 1., 1., 2.], y=[1.1, 0.9, 0.7, 0.6, 0.4]))
""", :model; title="Two axes connected by a plate", require_stan=true)
```

Inside the cell, `rate[i]` is a unit's value, `time[rows[i]]` its observations,
and bare `amplitude` is shared. Cell code uses Julia broadcasting explicitly.
A bare left-hand name is local; an indexed left-hand name exposes a model
output. Observe the response slice directly: an assignment that aliases it
does not become a new observation.

For long-format series, `@plate for p in groups(unit; order=time)` iterates
each unit's ordered row indices. An `@scan` within the cell states a recurrence;
`AR1Normal(mu[p], sigma, phi; time=time[p])` supplies an AR(1) residual law
that accounts for gaps. [Response families](response-families.md) distinguishes
residual dependence from a latent `rw`, `dar` or `cdar` term.

## Reusable components

A function-style component gives each call its own parameters. Shared values
are explicit arguments. This calibration is used twice with separate local
intercepts and slopes.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ComponentGuide), raw"""
"A calibration curve; every call owns its intercept and slope."
@brm function calibration(x, shared_scale)
    value = a + b * x
    a ~ Normal(0, shared_scale)
    b ~ Normal(1, 0.2)
    return value
end

joint = @brm begin
    left ~ calibration(x_left, shared_scale)
    right ~ calibration(x_right, shared_scale)
    y_left ~ Normal(left, noise)
    y_right ~ Normal(right, noise)
    shared_scale ~ Exponential(1)
    noise ~ Exponential(0.5)
end
model = joint((; x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
    y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1]))
""", :model; title="Independent calls with a shared prior scale", require_stan=true)
```

```@example components
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:ComponentGuide).model
calls = brm_components(model)
@assert length(calls) == 2
[call.path for call in calls]
```

Components accept positional and keyword inputs, including defaults, and resolve
functions in their defining module. A named return such as
`return (; recovery, decay)` exposes `kin.recovery` and `kin.decay` after
`kin ~ kinetics(...)`. Before indexing a returned field in a plate, bind it
at model scope, `recovery = kin.recovery`, then read `recovery[i]`.
`brm_components` discovers the definitions and call paths; descriptor queries
use `submodel=:left` rather than generated names.

Use `merge(structure_a, structure_b, priors_a, priors_b)` for fragments that
share one namespace. Put structure first and priors last within each scope;
moving a cell-local prior outside the plate changes the model. A retained
builder can bind new data; `reprocess(backend, new_data)` instead preserves
the fit's preprocessing. These are different operations.

The [Warfarin example](warfarin.md) combines group predictors, nonlinear
time courses and observations on several axes.
