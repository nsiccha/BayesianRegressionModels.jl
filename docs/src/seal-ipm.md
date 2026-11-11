# Grey-seal integrated population model

An integrated population model connects several observations to one changing
population. Here, aerial counts, hunting bags, demographic samples and
reproductive observations share an age- and sex-structured state process.
The example uses synthetic data: three years, three age classes and two sexes.
It illustrates the model's construction, not the size or condition of today's
Baltic grey-seal population.

::: warning Exploratory fit: all four chains rejected

The saved run has 18.7–23.7% divergent transitions per chain. None passed the
sampler's unchanged 10% admission limit. Its simulations below illustrate the
model and the failed run; their bands are not reliable posterior uncertainty.
They do not establish convergence or support population-management conclusions.

:::

## Density-dependent reproduction

Carrying capacity changes reproduction through population density. At reference
herring conditions, let ``b_0`` be the birth probability at zero density and
``b_K`` the birth probability that replaces demographic losses. The model uses

```math
b(N) = b_0\exp\{-\theta_0[\exp(\theta_1N)-1]\},\qquad
\theta_1 = \frac{1}{K}\log\left(1-\frac{\log(b_K/b_0)}{\theta_0}\right).
```

Thus ``b(0)=b_0`` and ``b(K)=b_K``. Holding the demographic parameters fixed,
a larger ``K`` shifts the decline to larger population sizes. These curves use
the model's compiled birth-rate helpers at fixed illustrative values
``b_0=0.7``, ``b_K=0.2``, maximum birth rate 0.9 and scaled density parameter
0.4. They are a deterministic comparison, independent of the failed fit.

```@eval
if !isdefined(Main, :SealFigures)
    Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "research", "seal", "figures.jl"))
end
Main.BRMDocsComparisons.figure(Main.SealFigures.density_response();
    caption="Birth probability falls as population increases. Each orange point marks its panel's carrying capacity, where the blue curve reaches the same replacement birth probability, 0.2. These are fixed-parameter examples, not fitted estimates.")
```

The implementation follows the density-dependence equations in
[Vanko et al. (2026), Appendix C](https://doi.org/10.1016/j.ecolmodel.2026.111666)
and the paper's [archived code, v1.1](https://doi.org/10.5281/zenodo.20381625).
Niko Siccha wrote this BRM implementation. Its herring regression, several
priors and random effects, and the small synthetic fixture differ from the
published analysis; this page is not a replication of that study.

## State process

The state contains six demographic classes. Survivors age through a
Leslie-style transition; mature females produce pups, with variation in the
number of births and sex allocation. Survival depends on age and sex.
Herring covariates and population density affect birth probabilities. A
short initialization burn-in sets the starting composition before scaling
it to the initial population-size parameter.

Within a year, competing hunting risks for Sweden and Finland are integrated
with an ODE. A logistic-normal allocation then divides each class among
survivors, deaths and the two hunting bags. This recurrence is shared by
all observation streams:

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:seal_ipm),
    "research/seal/grey_seal_brm.jl";
    before=:grey_seal_brm_model, starting_at="StanBlocks.@deffun begin",
)
Main.BRMDocsComparisons.source_code_region(
    "research/seal/grey_seal_brm.jl";
    starting_at="run_state_process(n_state_years", ending_before="    # ---- wrapper: transformed quantities",
)
```

## Observation model

BRM formulas declare the birth-rate regression and the random effects for
years, demographic classes and transition noise. The mechanistic helper
returns named state quantities. Each likelihood reads them at its own
observation years; the input need not force all streams into one table.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:seal_ipm),
    Main.BRMDocsComparisons.source_function("research/seal/grey_seal_brm.jl", :grey_seal_brm_model),
    :grey_seal_brm_model; title="Grey-seal population and eight observation streams", require_stan=true,
)
```

The aerial count has a negative-binomial likelihood. Its mean is an
observation fraction times total population: counts alone cannot separate
abundance from detectability. Two normal likelihoods describe national
hunting totals, with standard deviations proportional to their means.
Four multinomial likelihoods describe demographic or reproductive-sign
composition, and a binomial likelihood describes pregnancy counts.

These observations are conditionally independent given the shared state
and parameters. Their combination can resolve some ambiguities of a single
stream, but three synthetic years provide little information about many
latent effects. Female adult survival is displayed on its natural scale,
``0.8+0.2\,\mathtt{phi\_a\_sc}``, rather than as its unit-interval parameter.

## Saved run and diagnostics

Four WarmupHMC chains each saved 300 post-warmup states. The run used target
acceptance 0.99, a 10% warmup divergence gate and two restarts per chain.
The completion sampler required three admissible chains; it obtained zero.
Saved states remain available for diagnosis even when a chain is rejected.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/chains.tsv";
    columns=[:chain => "Chain", :saved_states => "Saved states",
             :accepted_states => "Accepted states", :divergences => "Divergences",
             :divergence_pct => "% divergent", :status => "Status"])
```

Every saved state has a finite density and gradient. That is a numerical
check, not evidence of adequate exploration. Split R-hat and effective
sample sizes below describe the rejected chains, with adult survival
transformed before the calculation. They do not override the divergences.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/diagnostics.tsv";
    columns=[:quantity => "Quantity", :rhat => "R-hat",
             :ess_bulk => "Bulk ESS", :ess_tail => "Tail ESS"])
```

## Exploratory aerial-count simulations

Each panel uses only its named chain. The blue line and bands summarize
new aerial counts simulated from its saved states; orange points are the
three synthetic observations. The logarithmic count axis keeps the medians
visible alongside the wide simulation ranges. The count scale includes the observation
fraction and negative-binomial variation, so it is not a population-size
trajectory. Chains are shown separately to avoid concealing disagreement
in a pooled summary.

```@eval
Main.BRMDocsComparisons.figure(Main.SealFigures.aerial_simulations();
    caption="Exploratory simulations from rejected chains. Blue: median and central 50%/95% ranges of simulated aerial counts; orange points: synthetic observations. These descriptive ranges are not reliable posterior predictive intervals, even where they cover the observations.")
```

A broad simulated range can cover the data despite poor sampling. This
figure therefore explains the observation model and exposes the saved run;
it does not validate a fit. Density-response curves above remain useful
without a successful fit because their inputs are fixed explicitly.

## Reproduce

::: details Commands, summaries and receipt

The committed files are small summary tables. Raw checkpoints stay outside
git; the docs build reads the tables and generates interactive figures
without sampling. `summarize.jl` reads checkpoints without changing them,
maps their working coordinates back through the exact adaptive-centering
target, and simulates observations with fixed seeds. It records the
emitted program, data and checkpoint hashes.

With the original strict-run checkpoint directory available:

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/seal/summarize.jl CHECKPOINT_DIRECTORY
```

A fresh checkout can reproduce the numerical equation checks without
those checkpoints:

```sh
julia --startup-file=no --project=test test/seal_density_dependence.jl
```

The separate manual sampling check is `test/seal_density_fit.jl`. It retains
the original admission limits and fails when the quorum is not met; setting
`BRM_SEAL_FIT_CHECKPOINT_DIR` preserves its diagnostic states. Run it only
when deliberately revisiting the fit, not for routine documentation edits.
The test environment pins StanBlocks 0.2.36, including the row-vector repair
needed by this model; the docs environment requires at least that version.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/receipt.tsv";
    columns=[:key => "Field", :value => "Value"])
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/seal/README.md).

:::
