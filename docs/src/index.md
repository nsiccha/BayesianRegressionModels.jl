# BayesianRegressionModels.jl

BayesianRegressionModels.jl describes hierarchical regression models with brms-style formulas and executes them in Julia, Stan or Python/JAX.

```sh
git clone -b ns/devibe https://github.com/nsiccha/BayesianRegressionModels.jl
cd BayesianRegressionModels.jl
julia --project=test test/setup_env.jl
```

```@eval
using Markdown
mod = Main.BRMDocsComparisons.example_module(:start)
view = Main.BRMDocsComparisons.comparison(mod,
    "using BayesianRegressionModels\n\n" *
    Main.BRMDocsComparisons.source_function("research/start/reproduce.jl", :intro_model) *
    "\n\nmodel = intro_model()",
    :model; title="Gaussian regression", require_stan=true, jax=true)
observed = Base.invokelatest(mod.column_data, mod.model, :y)
subjects = Base.invokelatest(mod.column_data, mod.model, :subj)
push!(view.content, Markdown.Code("text", "$(length(observed)) observations in $(length(unique(subjects))) subjects; response: y"))
view
```

The six synthetic observations have population effects for age and a numeric
sex code, plus subject-specific intercepts and age slopes. The tabs show the
same declaration as an intermediate representation, a StanBlocks model,
Stan source and a JAX module.

The fit uses four WarmupHMC chains, retaining results when three complete.
These estimates come from the cached run; the build renders the model and its
results without sampling again.

```@eval
Main.BRMDocsComparisons.results_table("research/start/results/population.tsv";
    columns=[:quantity => "Quantity", :median => "Median", :q025 => "2.5%", :q975 => "97.5%",
             :rhat => "R̂", :ess_bulk => "Bulk ESS"])
```

```@eval
using AlgebraOfVega
lines = readlines(joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research/start/results/ppc.tsv"))
header = Tuple(Symbol.(split(first(lines), '\t')))
cell(value) = something(tryparse(Float64, value), String(value))
rows = [merge(NamedTuple{header}(Tuple(cell.(split(line, '\t')))), (; observation=string(i)))
        for (i,line) in enumerate(lines[2:end])]
intervals = data(rows) * mapping(:q50 => "Response"; y=:observation => "Observation") *
    pointinterval(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color="#2a78d6")
observed = data(rows) * mapping(:observed => "Response"; y=:observation => "Observation") *
    visual(Scatter; color="#eb6834", markersize=9)
Main.BRMDocsComparisons.figure((intervals + observed) * config(width=500, height=210);
    caption="Observed responses (orange) and posterior predictive medians with central 50% and 95% intervals (blue). Six observations leave substantial predictive uncertainty.")
```

```@eval
Main.BRMDocsComparisons.results_table("research/start/results/fits.tsv";
    columns=[:chains_kept => "Completed chains", :omitted_chains => "Omitted chains",
             :draws => "Retained draws", :divergent => "Divergences"])
```

## Documentation

| Part | Purpose |
|:--|:--|
| Start | Install the source environment and inspect a fitted model. |
| [Guides](model-syntax.md) | Model syntax, priors, response families, fields, backends and centering. |
| [Examples](warfarin.md) | Worked models with fits, diagnostics and figures. |
| [Reference](api.md) | Public functions and the [term catalogue](terms-reference.md). |

```@raw html
<details><summary>Reproduce</summary>
```

The [reproduction notes](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/start/README.md)
explain the fit key, priors and summary files. Run sampling deliberately from
the checkout; selecting chains by completion time can affect inference.

```sh
julia --startup-file=no --threads=1 --project=test research/start/fit.jl
```

```@eval
Main.BRMDocsComparisons.results_table("research/start/results/receipt.tsv";
    columns=[:key => "Field", :value => "Value"])
```

```@raw html
</details>
```
