# Grey-seal integrated population model

An integrated population model connects several observations to one changing
population. Here, aerial counts, hunting bags, demographic samples and
reproductive observations share an age- and sex-structured state process.
The example uses synthetic data: three years, three age classes and two sexes.
It illustrates the model's construction, not the size or condition of today's
Baltic grey-seal population.

The fitted example retains all four chains and 6,000 draws. One transition
is divergent (0.017%); across all 123 sampling coordinates, maximum R-hat is
1.009 and minimum bulk/tail effective sample sizes are 814/995. These are
Monte Carlo diagnostics for the synthetic fixture, not evidence about a
real seal population. The settings and remaining divergence are reported below.

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
0.4. They are a deterministic comparison, independent of the fitted results.

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

The calibration also requires
``\theta_0+\log(b_0/b_K)>0``. Ordered parameter bounds enforce this joint
support, including the mortality-dependent replacement rate. They retain the
original prior density kernels; normalizing each conditional restriction
would change the prior. The earlier parameterization allowed Hamiltonian
trajectories to cross this boundary and produced 251 divergences in 1,200
saved states. The current fit uses the same admission thresholds.

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
    before=:grey_seal_brm_model,
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

Four WarmupHMC chains each retained 1,500 post-warmup draws, using seed
20261111, target acceptance 0.99 and a 10% warmup divergence gate. The
completion sampler required three admissible chains under its unchanged
10% divergence limit; all four finished and were admitted. No chain was
omitted. One transition in chain 3 diverged.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/chains.tsv";
    columns=[:chain => "Chain", :saved_states => "Saved states",
             :accepted_states => "Accepted states", :divergences => "Divergences",
             :divergence_pct => "% divergent", :status => "Status"])
```

Every retained draw has a finite density and gradient. Across all 123
sampling coordinates, maximum split R-hat is 1.00898, minimum bulk ESS is
813.99 and minimum tail ESS is 994.80. The table reports selected model
quantities on their natural scales, including transformed adult survival.
These checks support using the fit for this worked example; they do not
establish that the model or its illustrative priors describe real data.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/diagnostics.tsv";
    columns=[:quantity => "Quantity", :rhat => "R-hat",
             :ess_bulk => "Bulk ESS", :ess_tail => "Tail ESS"])
```

## Posterior predictive aerial counts

Each panel uses only its named chain. The blue line and bands summarize
new aerial counts simulated from its retained draws; orange points are the
three synthetic observations. The logarithmic count axis keeps the medians
visible alongside the wide simulation ranges. The count scale includes the observation
fraction and negative-binomial variation, so it is not a population-size
trajectory. Separate panels make agreement between the chains visible.

```@eval
Main.BRMDocsComparisons.figure(Main.SealFigures.aerial_simulations();
    caption="Posterior predictive aerial counts by admitted chain. Blue: median and central 50%/95% predictive intervals; orange points: synthetic observations. All four chains are retained; one of 6,000 transitions diverged. These are predictions for the synthetic fixture, not estimates for the Baltic population.")
```

The predictive intervals combine parameter uncertainty and negative-binomial
observation variation. Their breadth reflects the limited synthetic data as
well as the observation model. Coverage of these three observations alone
does not validate the scientific model.

## Reproduce

::: details Commands, summaries and receipt

The committed files are small summary tables. Raw checkpoints stay outside
git; the docs build reads the tables and generates interactive figures
without sampling. `summarize_accepted.jl` checks the saved source and program
hashes, chain admission and checkpoint reconstruction against the saved
model-frame draws. It leaves checkpoints unchanged and simulates observations
with fixed seeds. The receipt records the program, data and draw hashes.

With the admitted run's checkpoint directory available, create a new output
directory:

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/seal/summarize_accepted.jl CHECKPOINT_DIRECTORY NEW_RESULTS_DIRECTORY
```

A fresh checkout can reproduce the numerical equation checks without
those checkpoints:

```sh
julia --startup-file=no --project=test test/seal_density_dependence.jl
```

To deliberately reproduce the longer fit, use a fresh checkpoint directory:

```sh
julia --threads=4 --startup-file=no --project=test research/seal/fit.jl NEW_CHECKPOINT_DIRECTORY
```

The shorter manual sampling check remains `test/seal_density_fit.jl`. Neither
command runs during routine documentation builds. Both preserve the original
admission limits and fail if the quorum is not met. The test environment pins
StanBlocks 0.2.40, including native evaluation of algebraic support bounds;
the docs environment requires at least that version.

The earlier rejected run remains in `research/seal/rejected-results/` with
its original receipt. Its checkpoints require the frozen model at `dd5f2ca8`;
they must not be interpreted using the new parameter coordinates.

```@eval
Main.BRMDocsComparisons.results_table("research/seal/results/receipt.tsv";
    columns=[:key => "Field", :value => "Value"])
```

[Research notes and source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/seal/README.md).

:::
