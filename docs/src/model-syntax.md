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
| :--- | :--- |
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
reference level. [The reference](terms-reference.md#Columns-and-formula-operators) gives
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
    @plate for i in eachindex(rate)
        mean_y = amplitude * exp.(-rate[i] .* time[rows[i]])
        y[rows[i]] .~ Normal.(mean_y, noise)
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

The loop follows the unit-level `rate` vector. Inside the cell, `rate[i]` is a unit's value, `time[rows[i]]` its observations,
and bare `amplitude` is shared. Cell code uses Julia broadcasting explicitly.
An assignment to a bare name is local. An assignment to `name[i]` exposes a
model output. Observe the response slice directly, as in `y[rows[i]]` above.

For four independent standard-normal draws in each cell, write
`z[1:4] .~ Normal.(0, 1)` and read the vector as `z` inside the cell.
Use `z[1:K]` when an integer data field sets the width. When the location
is a vector, `z .~ Normal.(mu[i], s)` follows its length in each cell,
including when cells differ in length. A scalar draw uses `a ~ Normal(0, 1)`.
A single joint draw also uses `~`, as in `p ~ Dirichlet(alpha[i])`.

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

### Repeating a component over records

Bind one member per record when the channels in a measurement vary between
datasets. An `if` on the record's settings selects which declarations enter
the model. Here the calibrated channels each get their own shift.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:RecordComponentGuide), raw"""
@brm function sensor(settings)
    if settings.calibrated
        shift ~ Normal(0, settings.scale)
        reading = settings.offset + shift
    else
        reading = settings.offset
    end
    return reading
end

signals = @brm begin
    for item in sensors
        channels[item.id] ~ sensor(item)
    end
    y ~ Normal(channels, 1)
end

sensors = [(id=:left, calibrated=true, offset=0.4, scale=0.7),
           (id=:right, calibrated=false, offset=1.2, scale=0.5)]
data = (; sensors, y=[0.2, 1.0])
model = signals(data)
""", :model; title="Sensor shifts selected by record", require_stan=true)
```

```@example record_components
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:RecordComponentGuide).model
[call.path for call in brm_components(model)]
```

`channels` follows the order of the supplied records. Fitted parameters keep
their labels when `reprocess` receives the same members in a different order.
For example, `brm_outputs(brm_descriptor(SBBRMI(model)); submodel=(:channels,
:left), logical=:shift)` finds the calibrated channel's shift. Binding with
`channels[item]` uses positions as keys. For a fit with a different selection
of members or declaration flags, call the builder on the new records.

When instruments measure different covariates, store the fixed-effect formula
on each record. In this calibration model the left sensor adjusts for
temperature and the right sensor fits only an intercept. Its offset vector
gives each instrument the rows on which to evaluate its calibration.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:RecordFormulaGuide), raw"""
@brm function calibration(entry)
    level ~ rhs(entry.formula) + offset(entry.offset)
    effect(level, :) ~ Normal(0, entry.width)
    effect(level, Intercept) ~ Normal(0, 1)
    return sum(level)
end

survey = @brm begin
    for sensor in sensors
        channels[sensor.id] ~ calibration(sensor)
    end
    y ~ Normal(channels, 1)
end

sensors = [(id=:left, formula="1 + standardize(temp)", temp=[8., 10., 12.],
            offset=zeros(3), width=0.4),
           (id=:right, formula=1, offset=zeros(2), width=0.7)]
model = survey((; sensors, y=[0.2, 1.0]))
""", :model; title="Calibration covariates selected by instrument", require_stan=true)
```

```@example record_formulas
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:RecordFormulaGuide).model
sb = SBBRMI(model)
left = brm_transform_constants(sb, :standardize_entry_temp; submodel=(:channels, :left))
(left.center, left.scale, brm_transform_constants(sb; submodel=(:channels, :right)))
```

A formula may be text, a Julia expression such as `:(1 + standardize(temp))`,
a column symbol, or the integer `0` or `1`. Combine record columns with `+`
and `&`, transform a column with `center`, `standardize` or `zscale`, or index
by a record column. BRM uses the same coefficient labels as the written-out
member formula. For example, `temp` is `:entry_temp` and `standardize(temp)`
is `:standardize_entry_temp`. To give that term its own prior wherever it
appears, write `optional(effect(level, :standardize_entry_temp)) ~ Normal(0, 0.3)`.
Pass the label as `coefficient` and the same
`submodel` key to `brm_population_effect_coordinates`. Each predictor can
read a different formula field. A change to the selected terms needs a new fit.

To give selected instruments their own smooth time effects, put the formula
inside the guard. Each enabled instrument supplies its time grid and basis
size. The fixed instrument below contributes its known mean.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:SmoothComponentGuide), raw"""
@brm function drift(instrument)
    if instrument.smooth
        trend ~ 0 + hsgp(instrument.time; k=instrument.k, c=1.5)
        prediction = sum(trend)
    else
        prediction = instrument.offset
    end
    return prediction
end

survey = @brm begin
    for item in instruments
        estimates[item.id] ~ drift(item)
    end
    reading ~ Normal(estimates, 1)
end

instruments = [(id=:early, smooth=true, time=[0.1, 0.4, 0.7], k=3),
               (id=:fixed, smooth=false, offset=0.2),
               (id=:late, smooth=true, time=[1.0, 1.3, 1.8, 2.1], k=4)]
model = survey((; instruments, reading=[0.5, 0.3, 0.7]))
""", :model; title="Instruments with independent time smooths", require_stan=true)
```

```@example smooth_components
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:SmoothComponentGuide).model
[call.path for call in brm_components(model)]
```

Here `early` and `late` each fit a marginal scale and a length scale,
with three and four basis weights respectively. A second guard can add a
seasonal term with `cov=:periodic, period=instrument.period`, or deviations
by subject with `by=instrument.subject, anchor=0.0`. Each term reads the
same instrument's time grid.

For prediction on new time grids, keep the data-bound backend and call
`reprocess` with the same member keys. The RK fitted-state export currently
requires fitted summaries at model scope, so these per-member smooths need
the original fitted backend for replay.

An observation clause lets each member use a trajectory computed in a
parent plate. Here each sensor supplies its readings as one vector per
subject, in the order of `input`. An empty vector represents a subject
with no readings from that sensor.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ObservedComponentGuide), raw"""
@brm function instrument(setting)
    bias ~ Normal(0, setting.scale)
    @observe s trajectory member begin
        setting.y[s] ~ Normal(trajectory .+ member, 1)
    end
    return bias
end

experiment = @brm begin
    for setting in instruments
        meters[setting.id] ~ instrument(setting)
    end
    @plate for s in eachindex(input)
        trajectory[s] = 0.5 * input[s]
        observe(meters, s, trajectory[s])
    end
end

instruments = [(id=:left, scale=0.5, y=[[1.2, 1.0], Float64[]]),
               (id=:right, scale=0.8, y=[[0.9], [1.7]])]
model = experiment((; instruments, input=[2.0, 3.0]))
""", :model; title="Sensors observing a shared subject trajectory", require_stan=true)
```

```@example observed_components
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:ObservedComponentGuide).model
[call.path for call in brm_components(model)]
```

`member` receives the component's return value for each observed row.
If the component returns a vector on its own time grid,
`member[setting.read_index[s]]` selects one grid position per reading.
For a named return such as `(; curve, scale)`, use
`member.curve[setting.read_index[s]]` and `member.scale`.
The clause can sit inside a declaration guard, so different members can
use different likelihoods in the same subject plate. A grouped intercept
`(1 | p | setting.site)` belongs to that member. Use
`(1 | parentblock(:p) | setting.site)` to put its margin in the parent's
`p` correlation block. The site labels match rows across members and the
parent, including when their rows have different orders or lengths. The
member's formula can carry its own terms beside that margin, as in
`shift ~ 1 + (1 | parentblock(:p) | setting.site)`, and
`effect(shift, Intercept) ~ Normal(0, 1)` then sets that member's intercept
prior.

Query a member's observation draws with
`brm_output_coordinates(descriptor, :y, names; role=:posterior_predictive,
submodel=(:meters, :left))`. For a member with a formula predictor, use
`brm_term_coordinates` with the same `submodel` address to select its
term parameters, such as `parameter=:basis_weights` on a time smooth.
For a shared intercept margin, `brm_ranef_sd_coordinates(descriptor,
:shift, names; id=:p, submodel=(:meters, :left))` selects the fitted
between-site standard deviation for the member's `shift` predictor.

Use `merge(structure_a, structure_b, priors_a, priors_b)` for fragments that
share one namespace. Put structure first and priors last within each scope;
moving a cell-local prior outside the plate changes the model. A retained
builder can bind new data; `reprocess(backend, new_data)` instead preserves
the fit's preprocessing. These are different operations.

The [Warfarin example](warfarin.md) combines group predictors, nonlinear
time courses and observations on several axes.
