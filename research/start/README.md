# Start example

The landing-page example is a Gaussian regression with population age and
numeric sex effects, correlated subject intercepts and age slopes, and an
exponential prior on the residual standard deviation. Its six observations
are synthetic and illustrate the interface rather than a scientific inference.

The Start page's `docs/start/setup.jl` installs a public environment for model
declarations and Stan export. The cached fit used the separate test environment,
whose pinned Enzyme compiler build currently requires the fleet resolver named
in `test/setup_env.jl`. That installer is not portable to an ordinary fresh
checkout. The receipt records the exact fitted package versions; the data and
model themselves are public synthetic examples.

With that research environment available, run the fit deliberately from the
repository root:

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --threads=1 --project=test research/start/fit.jl
```

The script uses four independent WarmupHMC chains, a completion quorum of
three and 1,000 retained draws per completed chain. The receipt records
omitted chains and divergences; selecting chains by completion time may
affect inference. The posterior predictive intervals describe replicated
observations, not uncertainty intervals for the population regression line.

The content key covers the emitted Stan program, bound data, sampler
settings, seed and harness version. A matching completed summary is reused.
The receipt is written before sampling. Rebuilding the documentation only
reads the small summary tables and creates the figures; it never runs the fit.

The quantity table follows the DocsFits summary schema. Small TSV summaries
are the temporary transport until that package is publicly available; raw
draws and rendered figures are not committed.
