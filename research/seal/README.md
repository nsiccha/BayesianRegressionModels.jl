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

## Joint support of the calibrated equation

The calibration requires a positive pregnancy-loss rate at carrying capacity.
Let `c` be the replacement birth rate, `m` the maximum baseline birth rate,
`l` its minimum-to-maximum ratio, `z` the reference herring logit, and `u`
the scaled density-dependence parameter. With
`r = l + (1-l)*logistic(z)` and `q = m + (1-m)*u`, the required margin is
`-log(q) + log(m*r/c) > 0`. This implies `c < r < 1`.

The models now encode that existing joint support in their parameter bounds.
For adult survival `a`, pup-to-adult survival ratio `h`, age count `A`, and
survival shape `s`, the mortality rates are
`mu_j = exp((1-w_j)*log(-log(a*h)) + w_j*log(-log(a)))`, with
`w_j = (j/(A-1))^s`. Thus `c = 2*(1-a)*exp(sum(mu_j, j=0:A-2))`.
These are interpolated mortality rates, not interpolated survival rates.

The ordered bounds are:

1. Adult survival exceeds both 0.8 and the root of
   `2*(1-a) = a^(A-1)`. Below this root even `h=1` would give `c>=1`.
2. The pup ratio exceeds the unique root at which `c=1`. The solver works
   in `log(-log(a*h))`; its residual is smooth and strictly increasing.
3. `l > max(0, (c-logistic(z))/(1-logistic(z)))` ensures `r>c`.
4. `u < min(1, m*(r-c)/(c*(1-m)))` ensures the calibration margin is positive.

Every feasible point of the original model satisfies these bounds, and every
interior point satisfying them has a defined calibration. The sampled priors
use `restricted`, retaining their original density kernels on this support.
Normalizing each conditional restriction would change the joint prior.
Stan supplies the coordinate Jacobians, including the implicit derivatives
through the mortality root. Bounds refer directly to earlier sampled
parameters because Stan declares parameters before transformed parameters.

`test/seal_joint_support.jl` compares both authored helper sets with an
independent calculation of the original density, a finite-difference
coordinate Jacobian, and finite-difference gradients for 2, 3, 6 and 12 ages.
This check is separate from fitting and does not establish convergence.

## Saved fitted example

The current `results/` tables use the support-repaired model at
`52978e90b0c81db1de10613dc9994375a0eb3897`. Four chains retained 1,500 draws
each with seed 20261111 and target acceptance 0.99. All four passed the
unchanged 10% divergence admission limit and three-of-four quorum. One of
6,000 transitions diverged (0.01667%); per-chain counts are 0, 0, 1, 0.
All retained draws have finite density and gradients. Across 123 sampling
coordinates, maximum R-hat is 1.00898 and minimum bulk/tail ESS is 813.99/994.80.
These diagnostics support the synthetic worked example; this is not a fit
of the paper's data or a population-management estimate.

`summarize_accepted.jl` checks source/program/data hashes, chain admission,
and equality between the saved model-frame draws and the checkpoint draws
restored with `WarmupHMC.back_transform`. It leaves checkpoints unchanged.
Aerial posterior predictive draws use seeds 20261112 through 20261115;
central 50% and 95% intervals include observation variation. Adult survival
uses `0.8 + 0.2 * phi_a_sc`.

```sh
julia --startup-file=no --project=test test/setup_env.jl
# Deliberate new sampling only; refuses an existing checkpoint directory:
julia --threads=4 --startup-file=no --project=test research/seal/fit.jl NEW_CHECKPOINT_DIRECTORY
# Reuse the saved admitted run; never samples and refuses existing output:
julia --startup-file=no --project=test research/seal/summarize_accepted.jl CHECKPOINT_DIRECTORY NEW_RESULTS_DIRECTORY
```

The checkpoint directory contains the saved model-frame draws and provenance
files produced by `fit.jl`; it is not committed. The docs build reads only
small tables and constructs figures through `docs/backend_comparisons.jl`;
it never samples. Raw checkpoints and rendered images stay outside git.
The shorter `test/seal_density_fit.jl` remains a deliberate 300-draw sampling
check with the same admission thresholds. Routine page edits reuse results.

## Preserved rejected run

`rejected-results/` preserves the tables for the earlier model at
`21c3e34ab807c812dbfd546252415ff43313a77a`: four chains of 300 states with
66, 71, 56 and 58 divergences. Zero chains passed the same 10% admission
limit. Those tables describe rejected states, not reliable posterior intervals.

Reproduce the old tables from frozen commit `dd5f2ca8`, with its original
parameterization and summarizer. The support repair preserves the joint law
but changes stored coordinates. The retained `summarize.jl` checks the old
program fingerprint before reading those checkpoints and rejects the current
model. Never pass old checkpoints to the new accepted-fit summarizer.
