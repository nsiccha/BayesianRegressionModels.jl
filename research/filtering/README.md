# Count-series SMC and filtering example

The public [filtering example](../../docs/src/filtering.md) uses one synthetic
Poisson state-space model to compare cumulative SMC, Gaussian carried windows,
ordinary state filtering, a learning filter, and a full-data MCMC reference.
A second small model demonstrates selective coordinates when one existing unit
receives new counts.

## Reproduction

From the repository root, bootstrap the pinned inference environment and choose
a fresh directory for the serialized intermediate fits:

```sh
julia --project=test test/setup_env.jl
julia --project=test research/filtering/fit.jl /path/to/new/filtering-checkpoints
julia --project=docs docs/make.jl
```

`model.jl` generates the data and declares both models. `fit.jl` writes the
small summary tables under `results/`; the documentation build reads them and
executes `figures.jl` to emit AlgebraOfVega specifications. No inference runs in
the docs build. Draws, serialized checkpoints and rendered figures are not
committed. The summary TSVs and `receipt.tsv` are the cached docs inputs.

## Experiment

The generator uses seed 20261110, three units and 60 days. Shared level,
persistence and innovation SD are 2, 0.8 and 0.3. Unit offsets span −0.4 to 0.4.
All prior hyperparameters are data inputs. Exposures are one where a count is
observed and zero elsewhere, paired with zero counts. The latent horizon stays
at 60 days: unobserved future states follow their priors, not future data.

Training uses days 1–20 and 1,024 SMC particles. Cumulative updates reveal days
1–40 and 1–60. Carried windows instead introduce days 21–40 and 41–60, replacing
earlier posteriors with a Gaussian with up to 16 fitted correlation directions.
Each bridge has its own independent pilot and schedule; each adaptive pilot
uses 64 steps, with three mutation moves and maximum tree depth seven.

The ordinary filter uses the training parameter cloud and 64 conditional state
particles per member. A state-only pass over training observations initializes
its boundary without adding those likelihoods again. The learning filter uses
256 parameter members, 128 state particles, a 10-day window and a correlated
Gaussian calibration restricted to static parameters. Its boundary resets
approximate both the parameter posterior and parameter–state dependence.

Three independent full-data SMC runs use seeds 501–503 and one separately tuned
reference (pilot seed 1500). WarmupHMC fits the same full-data ReactiveKernels
log density using completion sampling: four requested chains, seeds 601–604,
at least three completed chains, and 1,000 requested draws per chain. The
completion table records what actually finished. Diagnostics use the common
retained draw count across completed chains and all unconstrained coordinates.

## Interpreting the comparison

`marginals.tsv` reports weighted means, SDs, medians and 50%/90% intervals. Inspect
spread as well as location: agreeing means can conceal lost uncertainty.
`comparison.tsv` compares independent SMC run means in pooled-SD units and
within-run SD ratios; these are descriptive checks, not Monte Carlo errors.
`diagnostics.tsv` reports maximum R-hat, minimum bulk/tail ESS and divergences
for the MCMC reference. Poor diagnostics weaken that reference.

In the saved run, batch and cumulative SMC agree closely with the MCMC
reference. The carried-window approximation substantially broadens persistence
and innovation-scale uncertainty. The learning filter instead narrows those
marginals. The page computes their scale comparisons from `marginals.tsv`;
these differences are properties of this dataset and configuration, not
general accuracy guarantees for either approximation.

`filtered_state.tsv` contains filtering summaries after calibration, evaluated
only against observations available on each day. `filter_scores.tsv` contains
predictive scores and effective parameter-member counts. Ordinary filtering
can follow states while its fixed parameter cloud concentrates. Coverage of one
simulated path is descriptive, not a calibration study.

`summary.tsv` records active-coordinate counts and checks of checkpoint
continuation. The selective example samples the observed unit's log rate while
recovering the shared level and other rate from their carried conditional law.
The scan example need not become low-dimensional under automatic selection.

## Checkpoint handling

Intermediate objects are cached atomically in the supplied directory. Keep the
original model, data and package versions with them. Exact continuation also
checks reproduced endpoint densities bit for bit; a different CPU or numerical
environment may fail that check even with the same package versions. Keep the
original execution environment for exact resume. Use a fresh directory when
changing any model, input, setting or seed. Deserialization is only for trusted
files. A completed production schedule is saved; interrupting a tuning pilot
requires its separate tuning checkpoint mechanism. The public page's checkpoint
example is exercised by the experiment before the full fits continue.

The receipt's `started` field records the experiment's first invocation;
`commit`, source hashes and `seconds_this_invocation` describe the invocation
that exports the tables, which can resume previously completed fits. Sampling
completion and elapsed time are retained separately in `completion.tsv`.
The saved main comparisons used harness version 1 and BRM 0.2.272, as their
receipt records. Version 2 adds the separately verified existing-unit selection
example and includes the inference program in the cache key; its main comparison
settings are unchanged. Use a fresh checkpoint directory for version 2.
