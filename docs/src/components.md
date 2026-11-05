# Reusable model components

A function-style `@brm` definition declares explicit inputs, local model
parameters and one returned value, or several named values. Call it with `~`
inside another BRM model.
StanBlocks and ReactiveKernels compile the same scientific declaration into
their native submodel scopes.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ScopedComponentsDocs), raw"""
using BayesianRegressionModels, Distributions

"A calibration curve with independent coefficients for each call."
@brm function calibration(x, shared_scale)
    value = a + b * x
    "Local intercept using the parent's shared scale."
    a ~ Normal(0, shared_scale)
    b ~ Normal(1, 0.2)
    return value
end

joint = @brm begin
    left ~ calibration(x_left, shared_scale)
    right ~ calibration(x_right, shared_scale)
    y_left ~ Normal(left, obs_sd)
    y_right ~ Normal(right, obs_sd)
    shared_scale ~ Exponential(1)
    obs_sd ~ Exponential(1)
end

data = (x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
        y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1])
model = joint(data)
""", :model; title="Independent calibration components", require_stan=true)
```

There are six sampled scalars: each call owns its own `a` and `b`, both calls
read the parent's `shared_scale`, and the likelihoods share `obs_sd`. A component call introduces no
regression coefficient. The two input vectors need not have the same length.
The result can be scalar or array valued; ordinary component statements can
declare vector priors and call other components.

An elementwise vector result retains its input row frame. It can be used in a formula
offset or selected by a row index, including repeated or reordered indices:

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ScopedComponentsDocs), raw"""
rows = @brm begin
    field ~ calibration(x, scale)
    log(quantity) ~ 1 + offset(field) + offset(fixed)
    effect(quantity, Intercept) ~ Normal(0, 1)
    y ~ Normal(quantity, 0.7)
    scale ~ Exponential(1)
end
row_model = rows((; x=[-0.4, 0.2, 0.7], fixed=[0.1, -0.1, 0.2],
                   y=[1.1, 1.4, 1.8]))
""", :row_model; title="Component output in a formula offset", require_stan=true)
```

An internal plate also retains its cell axis when each returned cell is a
scalar. Vector-valued cells require explicit row selection; matching the
flattened length to another predictor does not establish row alignment.

Inputs are positional names or keyword names, with optional defaults. Model
values outside the component must be passed explicitly. Functions resolve in
the component's defining module, which can differ from its caller's module.
The function-shaped declaration is a `BRMComponent`, a declarative model
definition. It is called at a model binding, `result ~ component(...)`.

Correlation dimensions and concentrations can be explicit data inputs. The
same component definition can serve different sizes, including different
sizes in one model:

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:LKJComponentsDocs), raw"""
using BayesianRegressionModels, Distributions

@brm function correlation(K, eta)
    L ~ LKJCholesky(K, eta)
    return L
end

paired = @brm begin
    small ~ correlation(small_K, small_eta)
    large ~ correlation(large_K, large_eta)
    y ~ Normal(small[2, 1] + large[2, 1], 0.3)
end
model = paired((; small_K=2, small_eta=1.3,
                  large_K=4, large_eta=2.3, y=0.1))
""", :model; title="One correlation component at two sizes", require_stan=true)
```

`K` must resolve from known data to an integer at least one; `eta` must be
finite and strictly positive. Data-derived expressions such as
`length(channels)` also work. Both backends preserve the normalized LKJ
prior and its `K * (K - 1) / 2` unconstrained coordinates. Native RK requires
ReactiveKernelsPPL 0.1.97 or later and retains the named inputs in emitted
component source.

## Returning several named values

One call can return several related quantities as a named tuple. The call keeps
one scope: its priors and correlation blocks are declared once, and the parent
reads each returned value as a property of the call. Here two linear predictors
share one correlated unit effect, and the parent uses one in a likelihood and
the other in a formula offset:

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:NamedComponentsDocs), raw"""
using BayesianRegressionModels, Distributions

"Recovery and decay rates whose unit effects share one correlation block."
@brm function kinetics(x, unit)
    recovery ~ 1 + x + (1 | p | unit)
    decay ~ 1 + (1 | p | unit)
    effect(recovery, :) ~ Normal(0, 1)
    effect(decay, :) ~ Normal(0, 1)
    return (; recovery, decay)
end

joint = @brm begin
    kin ~ kinetics(x, unit)
    y_recovery ~ Normal(kin.recovery, 0.5)
    log(q) ~ 1 + offset(kin.decay)
    effect(q, Intercept) ~ Normal(0, 1)
    y_q ~ Normal(q, 0.5)
end

data = (; x=[-0.4, 0.2, 0.7, 0.1], unit=["a", "b", "a", "c"],
        y_recovery=[0.1, 0.4, 0.8, 0.0], y_q=[1.2, 0.9, 1.4, 1.1])
model = joint(data)
""", :model; title="Two correlated predictors returned by one call", require_stan=true)
```

A field can be named after a local, `(; recovery)`, or computed in the return,
`(; decay = exp(log_decay))`. Each field keeps its own row axis, so it can enter
a formula offset or a likelihood, or be bound at model scope. A plate cell
slices names, so bind the field first, `recovery = kin.recovery`, and read
`recovery[i]` in the cell. Stan returns the values as one tuple; ReactiveKernels
reads each from the call's own scope. The call itself is not a value: reading
bare `kin`, an unknown field, or `kin.recovery[i]` inside a plate is refused
with the supported spelling. A component called inside a plate cell returns one
value per cell. A positional tuple return is refused, because the parent reads
values by name.

The existing `@brm model(data) = begin … end` form still defines a data-to-model
builder. `merge(builder_a, builder_b)` still combines statements in one
namespace. Use a component when repeated calls should own separate locals.

## Inspection and replay

`brm_source(calibration)` returns a copy of the authored function expression;
`brm(source; mod)` reconstructs the component. `brm_columns(calibration)` lists
its explicit inputs. Backend source inspection retains native submodel calls
and their priors.

`brm_components(model)` discovers every reachable component call recursively.
Each record includes its logical `path`, shared `definition` identity, authored
`source`, defining `scope`, and copied `call` expression without bound data.
Repeated instances have separate
paths and share a definition. Viewers can use this inventory to include real
component code in HTML, PDF and snapshots without a source registry. It also
accepts a backend, a component definition, or `(builder, data)`.

Julia docstrings before the component and before its declarations remain
authored metadata. `brm_documentation(calibration)` uses path `()` for the
component header and `(:a,)` for its local intercept. In the parent model,
`brm_documentation(model; path=(:left, :a))` selects that call's intercept
documentation; `(:right, :a)` selects the other call. Nested calls extend the
same logical paths. `brm_source` preserves the docstrings for reconstruction,
and redefining a Julia binding does not change a retained component's docs.

```julia
sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
rk = RKBRMI(model)
d = brm_descriptor(sb)
d.submodels
brm_output(d, :a; submodel=:left)
brm_output(d, :a; submodel=:right)
```

Each descriptor submodel retains its defining module, call target, own prior
declarations and logical outputs. Use `submodel=` to address an internal
value, including a tuple path for nested calls. A returned field is addressed
by its name, `brm_output(d, :decay; submodel=:kin)`. Use
`brm_coordinate_transport(rk, sb, stan_names)` to pair sampled coordinates
between the backends.

`reprocess(backend, new_data)` rebinds explicit data inputs while preserving
the fitted model and its preprocessing. Both calls may receive new row
counts independently. The usual `freeze_constants=false` option requests
fresh calibration of preprocessing constants.

`state = brm_fitted_state(rk)` also retains each call's fitted preprocessing
under its own path. `RKBRMI(builder(new_data), state)` can bind that state
without the original training rows, using the builder that authored `rk`.
