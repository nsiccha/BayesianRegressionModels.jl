# Backends and model inspection

Lower one declaration to a Julia density, a Stan program or a Python/JAX
module. Inspect the intermediate representation and effective priors before
choosing a fitting method.

## The lowering chain

An `@brm` builder binds data into a [`BRMI`](@ref), a typed collection of
declarations, assignments, prior selectors and data. It has not fitted the
model. Each backend consumes these same operations and records the fitted
preprocessing needed to evaluate new rows consistently.

| Entry point | Result | Uses |
| :--- | :--- | :--- |
| `SBBRMI(model)` | StanBlocks representation and emitted Stan | BridgeStan density and gradients; Stan generated quantities |
| `RKBRMI(model)` | ReactiveKernelsPPL program | Julia density and gradients; MCMC, variational inference, filters and JAX export |

The backends share formula interpretation; their supported terms, custom
functions and prediction surfaces differ. See the [terms reference](terms-reference.md)
for individual contracts. Start with RK for a Julia workflow and use Stan
when a needed term or observation mechanism is supported there.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:BackendGuide), raw"""
model = (@brm begin
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)

    effect(mu, Intercept) ~ Normal(0, 1)
    effect(mu, x) ~ Normal(0, 0.5)
    sd(mu, subject) ~ Exponential(0.5)
    sigma ~ Exponential(0.3)
end)((; x=[-1., 0., 1., -1., 0., 1.],
    subject=["a", "a", "a", "b", "b", "b"], y=[-0.5, 0.2, 0.8, -0.2, 0.4, 1.1]))
""", :model; title="One declaration, two backends and a JAX export", require_stan=true, jax=true)
```

Read the IR pane when a result surprises you. A wrong response, data column
or prior selector there reaches every backend. The generated panes come
from this bound object during the documentation build.

```@example backend_inspection
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
model = Main.BRMDocsComparisons.example_module(:BackendGuide).model
sb = SBBRMI(model; total_groups=())
rk = RKBRMI(model)
(columns=data_columns(model), priors=priors_of(model),
    stan_characters=length(stan_code(sb)))
```

[`rk_logdensity_problem`](@ref) provides the `LogDensityProblems` interface.
Choose an AD backend when constructing it; the sampler then consumes that
interface. `stan_instantiate` compiles the Stan density through
BridgeStan. Neither lowering call alone runs MCMC. The
[adaptive-centering guide](adaptive-centering.md) explains coordinate choices.

## Replay and outputs

Keep the prepared model with its fit. `reprocess` binds new rows while
retaining fitted factor levels, centering and scaling constants, group
structure and basis information. Rebuilding from an unbound formula can
change those constants and therefore change the meaning of a posterior draw.

Use declared output paths when reading predictions or latent quantities.
[`brm_outputs`](@ref) and [`brm_output`](@ref) expose a Stan descriptor's
output inventory. RK's `rk_draw_values` and `rk_draw_tree`
evaluate requested quantities from model-coordinate draws. A latent field,
a linear predictor and a predictive response are different outputs; adding
observation noise belongs to prediction.

## Descriptions and inspection

`brm_descriptor` supplies a Stan-backed description of the prepared model.
`brm_description` converts that inventory into equations, component prose
and effective priors. Check `complete` and `diagnostics` before treating
the result as a complete scientific description.

```@example backend_description
using BayesianRegressionModels
model = Main.BRMDocsComparisons.example_module(:BackendGuide).model
description = brm_description(brm_descriptor(SBBRMI(model; total_groups=())))
@assert description.complete
(complete=description.complete, components=length(description.components),
    priors=length(description.priors), diagnostics=description.diagnostics)
```

Pass `labels` for scientific meanings, symbols and units; names alone do
not establish them. `description_texts(:de)` selects German throughout the
generated prose. Presentation metadata does not change model identity,
data or inference caches. Custom description hooks and inventory fields
are documented in the [API](api.md); the
[description protocol](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/model_descriptions)
retains the detailed extension and rendering contracts.

With HTMXObjects loaded, [`brm_model_view`](@ref) displays source, roles,
ranks and documentation as one semantic tree. An unbound builder shows
source and inputs; a bound model supplies data shapes; an explicitly
prepared backend can add parameter dimensions. Inspection never fits a
model or evaluates a custom helper to guess its shape. Unknown dimensions
remain unknown. Component source and declaration docstrings are retained
under their logical call paths. The view supports HTML, Markdown and text;
HTMXObjects can also export it to a PDF.

## Python/JAX export

Export the RK program together with its bound plan. The generated module
owns its data preparation and unconstrained-coordinate layout. Keep the
bound inputs with it; the Python source alone does not contain the data.

```@example backend_export
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
model = Main.BRMDocsComparisons.example_module(:BackendGuide).model
rk = RKBRMI(model)
source = jax_source(rk.model, rk_bound_plan(rk))
inputs = rk_bound_data(rk)
path = joinpath(mktempdir(), "model.py")
write(path, source)
@assert occursin("def log_density", read(path, String))
(source_bytes=filesize(path), input_names=sort!(collect(keys(inputs)); by=string))
```

In Python, call the module's `prepare_data` with its declared `DATA` inputs,
then `log_density` with an unconstrained position. JAX can differentiate and
compile that density. A Julia callable needs a supported translation;
inspect the generated `UNTRANSLATED` report before using an export. Compare
values and gradients at common positions before comparing fits.

The [JAX example](jax.md) performs that parity check and fits the exported
model with NumPyro, with reproduction instructions for both runtimes.
