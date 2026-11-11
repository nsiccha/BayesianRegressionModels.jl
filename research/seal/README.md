# Grey-seal integrated population model

Niko Siccha's implementation of the grey-seal IPM, with a synthetic fixture
of three ages, two sexes and three years. `grey_seal_brm.jl` declares the
regression, random effects and eight observation likelihoods with BRM;
`grey_seal_ipm.jl` retains the SLIC bundle form.

The scientific reference is Vanko et al. (2026), *Ecological Modelling* 519,
111666, [doi:10.1016/j.ecolmodel.2026.111666](https://doi.org/10.1016/j.ecolmodel.2026.111666).
Density dependence follows Appendix C, equations 42, 48, 51 and 53, and
the [archived v1.1 implementation](https://doi.org/10.5281/zenodo.20381625)
at `52622e4c2393893799fd8319eff745380abf6bcc`.
The exact demographic equilibrium calibrates carrying capacity; this is not
the later soft replacement prior in the live `n-kall/sealIPM` repository.
The BRM herring regression and several priors and random effects differ
from the paper. The fixture is not the paper's data, and the example does
not reproduce its fitted analysis.

## Numerical correction

Birth probability is `b0 * exp(-theta0 * expm1(theta1 * N))`.
The slope is calibrated so that births at `N = K`, at reference herring
conditions, replace demographic losses. Pregnancy loss accumulates over
elapsed time since implantation. The BRM regression exposes its reference
intercept as a scalar shared with this calibration.

`test/seal_density_dependence.jl` checks both compiled helper implementations
against an independent pregnancy-hazard calculation, boundary identities,
finite differences and the full model's response to changing carrying
capacity. `test/seal_brm.jl` and `test/seal_ipm.jl` check compiler behavior.

## Saved diagnostic run

The run associated with `21c3e34ab807c812dbfd546252415ff43313a77a` used
four chains, 300 saved states each, seed 20261111, target acceptance 0.99
and a 10% divergence gate. Each chain exhausted two warmup restarts.
Divergences were 66, 71, 56 and 58 per 300 states. Zero chains passed the
unchanged 10% admission limit, so there is no accepted posterior fit.
The native density and gradient are finite at all 1,200 saved states;
that check does not establish adequate exploration.

The `results/` tables are descriptive summaries of these rejected states.
`summarize.jl` restores their exact adaptive-centering target and uses
`WarmupHMC.back_transform` before constraining them. It leaves checkpoints
unchanged and records their hashes. Aerial simulations use the model's
generated predictive draws with seeds 20261112 through 20261115; the
50% and 95% ranges are empirical simulation ranges, not reliable posterior
predictive intervals. Adult survival uses `0.8 + 0.2 * phi_a_sc`.

```sh
julia --startup-file=no --project=test test/setup_env.jl
julia --startup-file=no --project=test research/seal/summarize.jl CHECKPOINT_DIRECTORY
julia --startup-file=no --project=test test/seal_density_dependence.jl
```

`CHECKPOINT_DIRECTORY` is the saved strict run, not a path committed to the
repository. The docs build reads only small tables and constructs figures
through `docs/backend_comparisons.jl`; it never samples. Raw checkpoints
and rendered images stay outside git.

For a deliberate new diagnostic fit, set `BRM_SEAL_FIT_CHECKPOINT_DIR` and
run `test/seal_density_fit.jl`. It raises if the quorum fails. Routine page
edits must reuse the existing results rather than repeat sampling.
