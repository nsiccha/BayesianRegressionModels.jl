# Matrix-free conditional Laplace measurements

> **Superseded in part (BRM 0.2.159).** This record measures the rational
> quadrature (65 shifted solves per probe) that BRM has since replaced by one
> Lanczos recurrence per probe, so `logdet_step` and `logdet_extent` no longer
> exist; `refine` now compares solve tolerances. A `gmrf` field's component also
> has its own reference (`src/gmrf_preconditioner.jl`). The numbers below stay as
> recorded and do not describe the current code.


These are synthetic controls for the nonlinear embedded-Laplace implementation.
They do not import an ODL model, graph or observation. Results below were measured
on `a Linux fleet host` on 2026-11-02 with one CPU per run. Host load varied; allocation
counts are cumulative allocations, not peak resident memory.

## Reproducers

Use the repository's prepared `test` environment and the required compute-token
wrapper for Julia runs.

```sh
julia --project=test test/nlme_laplace_matrixfree_algebra.jl
julia --project=test test/nlme_laplace_matrixfree.jl
julia --project=test test/nlme_laplace_matrixfree_subset.jl
julia --project=test test/smc_laplace_matrixfree.jl
julia --project=test research/matrixfree_conditional/scaling.jl 44 38 24 det
julia --project=test research/matrixfree_conditional/nonlinear.jl 10 10 10 exact fd
julia --project=test research/matrixfree_conditional/nonlinear.jl 44 38 24 exact fd
julia --project=test research/matrixfree_conditional/nonlinear.jl 10 10 10 auto exact fd
julia --project=test research/matrixfree_conditional/production.jl 8 8 8 refine draw
julia --project=test research/matrixfree_conditional/production.jl 44 38 24 refine draw
julia --project=test research/matrixfree_conditional/production.jl 44 38 336
```

`scaling.jl` uses a rectangular spatial precision `(I + L)'(I + L)` and an
AR(1) temporal precision with correlation 0.9, plus diagonal observation
curvature 0.1. It compares operator preparation and a solve with warmed exact
refactorization and a solve on the same matrix. Its determinant measurement
includes selected derivative contractions, but not a complete model gradient.
The fixed reference is prepared once and reported separately. Add
`unpreconditioned iterations=2048` to reproduce the original operator baseline.

`nonlinear.jl` prepares a full Poisson model with a three-dimensional ICAR
effect and three population coordinates. Its warmed timings include both the
target value and population gradient. Finite differences recompute the mode,
so they also check the implicit mode correction.
The default measurement forces operator algebra; `auto` exercises the public
automatic policy instead.

`production.jl` uses the production determinant and selected contractions,
with an independent separable eigenvalue oracle instead of an exact sparse
factor. `refine` compares quadrature spacing 1 versus 0.5 with the same eight
probes and fixed reference, then uses 32 probes at spacing 0.5. `draw` measures
a conditional draw and reports its precision quadratic form; one such value
does not establish distributional accuracy. `allocations.jl` samples warmed
allocation sites in BRM's implementation.

## Measured baseline, before fixed sparse preconditioning

Implementation checkpoint: `c06923cd`. Its algebra suite passed 24 assertions,
its nonlinear suite passed 63, and the focused exact cell/sum/plan/ladder
regressions passed 613. These checks preceded the next canonical refresh.
The refreshed checkpoint `4c24316a`, based on canonical `f98f5757`, passed
70 nonlinear assertions (including moving declared Gaussian priors) and 72
focused exact declared-Gaussian assertions.

The determinant uses eight retained Rademacher probes and 65 quadrature nodes.

| Control | Result |
| --- | --- |
| 40,128 coordinates, default 512-iteration cap | Failed the first linear solve; relative residual 1.40e-5 exceeds 1e-10 |
| 40,128 coordinates, explicit 2,048 cap | Converged; at most 1,150 Newton-solve and 1,168 shifted-solve iterations |
| Operator preparation and solve at 40,128 | 1.9495 s; 135,352,208 allocated bytes; 63,394,321 retained bytes |
| Exact refactorization and solve at 40,128 | 1.7589 s |
| Warm approximate determinant at 40,128 | 41.4336 s; 896,875,144 allocated bytes |
| Approximate / exact log determinant | 183252.86377 / 183333.852; probe MCSE 214.32 |
| Full nonlinear Poisson/ICAR, 1,000 coordinates | Warm value and gradient 0.03406 s; exact 0.06696 s |
| Nonlinear finite-difference gradient check | Relative discrepancy 4.04e-9 |
| Nonlinear approximation error against exact | Log-density difference -3.40743; gradient difference norm 3.14115 |

The successful nonlinear check establishes consistency of the approximate
density and gradient, not accuracy against the exact determinant. The large
proxy establishes bounded storage and convergence with an increased cap; it
does not yet establish a speed advantage. The 1,000-coordinate nonlinear
control has a different, easier precision than the squared spatial precision.
This baseline does not establish performance at 561,792 coordinates.

The retained run receipts are `algebra-inplace.log`,
`scaling-40128-default.log`, `scaling-40128-inplace-2048.log` and
`nonlinear-scaling-1000-standard.log` under
`<local path omitted>` on the measurement host.

## Fixed preconditioner experiment

`preconditioned.jl` measures the reference independently of the BRM integration. It uses the
documented factor interface of LimitedLDLFactorizations 0.5.2 to construct a
bounded-fill reference precision `P = C C'`. With the reference held fixed,

```text
A = C^-1 H C^-T
logdet(H) = logdet(P) + tr(log(A))
x = (A + t I)^-1 r
d[-t r' (A + t I)^-1 r] = t (C^-T x)' dH (C^-T x)
```

Applying the same fixed probes and finite quadrature to `A` preserves the
density-gradient relationship when sensitivities are transformed back with
`C^-T`. Refreshing `P` as population coordinates change would require additional
derivatives and is outside this experiment. The operator still applies the
original precision; this does not remove model couplings.

The first two-by-two noncommuting derivative control passed (analytic
0.543182722604, finite difference 0.543182722601). The first 512-coordinate
experiment with four extra factor entries per column required 855 iterations
and failed its original-system residual check: 1.047e-9, despite a centered
residual of 4.12e-11. With 32 extra entries per column and a stricter centered
tolerance, the same 512-coordinate solve passed at residual 9.68e-14 in 12
iterations. The warmed determinant took 0.03331 s and 585,184 allocated bytes,
with at most 11 shifted iterations. Its log-determinant estimate was 2166.30392
versus exact 2166.75161, with probe MCSE 0.12965. The factor retained 19,364
entries and required no diagonal shift. The larger-support results below
measure scaling separately from this small control.

At 40,128 coordinates, the same fill budget passed: 169 solve iterations and
physical residual 3.93e-13. Reference preparation took 0.5247 s with 1,853,844
factor entries. The warmed determinant took 14.7389 s and 43,676,512 allocated
bytes, with at most 150 shifted iterations. Its value was 183330.06174 versus
exact 183333.85164, with probe MCSE 8.036. The fixed-probe derivative for an
identity perturbation was 3188.99726 versus exact trace inverse 3228.48666.
This is an improvement over the baseline, but the experiment does not include
BRM's selected-gradient contractions, and timings were taken at different host
loads. No production or end-to-end speedup is established.

The same unregularized reference failed at 120,384 coordinates: its factor
built in 1.789 s with 5,629,515 entries, but the centered solve exhausted 2,048
iterations. No determinant result was produced. A fixed reference-only
regularization `P_reference = H_reference + 0.01 diag(H_reference)` was then
measured; the model's precision remains unchanged. The unregularized
preconditioner is not an accepted default.

With fixed relative ridge 0.01, the 120,384-coordinate case passed: 65 solve
iterations, original-system residual 1.34e-13, and at most 53 shifted
iterations. Reference preparation took 1.300 s; the warmed determinant took
22.096 s and 130,995,792 allocated bytes. Its estimate was 555497.83807 versus
exact 555543.33333, with probe MCSE 55.033. The identity-perturbation derivative
was 9568.39173 versus exact 9511.09730.

The unchanged settings also passed at 561,792 coordinates: 71 solve iterations,
original-system residual 1.03e-13, and 57 shifted iterations. Preparation took
4.500 s; the warmed determinant took 158.534 s and 611,247,696 allocated bytes.
The retained operator/reference size was 790,507,456 bytes, not the peak memory
of a full model. The estimate was 2602608.42155 versus exact 2602695.43665,
with probe MCSE 103.440; the trace derivative was 44151.91865 versus exact
44065.11355. Host load rose from 18.37 at grant to 52.36 at release, so the
timing increase cannot be attributed to dimension alone. These standalone
measurements exclude BRM's selected contractions and full nonlinear target;
integrated controls follow below.

The 24-node mapped Gauss-Legendre trial at 40,128 took 6.673 s, but its estimate
183315.79423 differed by -14.2675 from the same-probe log-trapezoidal reference.
Its trace derivative also moved from 3188.99726 to 3162.89893. It was rejected
as the default despite passing its own finite-functional derivative check.

To run the optional experiment, create an isolated Julia environment containing
LimitedLDLFactorizations 0.5.2 and set `BRM_PRECONDITIONER_PROJECT` to that
environment's absolute path. The optional fifth argument selects a positive
Gauss-Legendre quadrature after mapping `t=(1+u)/(1-u)`; omitting it uses the
65-node log-trapezoidal reference. Quadrature changes must be checked against
the same probes and fixed preconditioner, separately from stochastic error.
`BRM_REFERENCE_RIDGE` sets the fixed relative diagonal regularization in the
experiment (default zero).

## Integrated reference controls

The first integrated algebra check passed 33 assertions, including a
noncommuting determinant derivative, original-system residuals, transformed
covariance draws, reference immutability and iteration exhaustion.

At 512 coordinates, the integrated default (reference ridge 0.01, eight probes)
passed with 35 solve and 30 shifted-solve iterations. Its log determinant was
2162.85856 versus exact 2166.75161; probe MCSE was 2.367. Halving the quadrature
spacing changed the estimate by 3.160e-6 and the selected sensitivity by
8.431e-8 in relative norm. Using 32 probes reduced the log-determinant error
from -3.89305 to -1.20281; the trace-inverse derivative error changed from
1.98759 to 1.81841. More probes do not promise monotone improvement for every
derivative. The warmed draw took 0.0270 s; its precision quadratic form was
525.890 for 512 coordinates, 0.434 standard deviations above the Gaussian
expectation.

The initial production determinant allocated 73,890,936 bytes at 512
coordinates and took 0.1863 s. Allocation sampling located boxed scalar
arithmetic in the shifted residual/value loop. A function barrier reduced
allocations to 1,481,976 bytes and warmed time to 0.04045 s, with identical
reported determinant, sensitivity and draw results. The 33 algebra assertions
also passed again. Retained storage was 995,889 bytes; cumulative allocation
is not peak memory. Receipts: `production-512-refinement.log` (exit 0, 58 s),
`production-allocation-profile.log` (exit 0, 48 s),
`production-512-unboxed.log` (exit 0, 216 s including a live precompile wait)
and `algebra-preconditioned-unboxed.log` (exit 0, 221 s) in the scratch directory
above.

On the refreshed BRM 0.2.130 candidate `8d328b28`, the production
40,128-coordinate squared-precision proxy passed with the default
512-iteration cap: 57 solve iterations and 47 shifted-solve iterations.
Operator preparation plus a solve took 0.95959 s versus 2.65654 s for exact
refactorization plus a solve on the same matrix; original-system residual was
1.340e-13. The warmed determinant, including selected sensitivity contractions,
took 13.0857 s and allocated 95,147,288 bytes, with 94,984,321 retained bytes.
Its estimate was 183337.98647 versus exact 183333.85164, with probe MCSE
16.4748. The one-time reference preparation took 3.337 s including its first
compilation. Host load was 61.76 at grant and 67.36 at release. Receipt:
`scaling-40128-v130.log` (exit 0, elapsed 69 s).

The full 1,000-coordinate nonlinear Poisson control also passed on this
candidate. Its warmed value and gradient took 0.03987 s and allocated
13,060,680 bytes; exact algebra took 0.007664 s and allocated 5,179,888 bytes.
Thus this small control does not show a speed advantage for the operator
policy. The finite-difference relative gradient discrepancy was 3.949e-9.
The approximate log density differed from exact by -0.14851, and the gradient
difference norm was 0.60249. Solves used at most 24 iterations and shifted
solves 20. Receipt: `poisson-1000-v130.log` (exit 0, elapsed 225 s).

At 40,128 coordinates the full nonlinear Poisson target took 4.52281 s for
the warmed value and gradient, allocating 479,564,472 bytes. Exact algebra on
the same model took 5.47564 s and allocated 480,314,080 bytes. Finite
differences, recomputing the mode for all three population coordinates,
agreed to relative error 2.422e-9. The approximate value differed from exact
by +2.22632 and the gradient difference norm was 6.30392. Solves used at most
17 iterations and shifted solves 14; maximum checked residuals were
1.511e-13 and 9.575e-12, respectively. Setup took 209.43 s, including
compilation, and cumulatively allocated 8.690 GB; the first value and gradient
took 19.392 s. Receipt: `poisson-40128-v130.log` (exit 0, elapsed 354 s).
These paired timings do not establish a universal crossover. Based on the
small-model result, the automatic policy retains exact algebra through 1,024
coordinates per connected component; callers can override that bound.
The public `solver=:auto` control at 1,000 coordinates subsequently confirmed
that it selects exact algebra. Its value and gradient were identical to an
explicit exact target, finite differences agreed to relative error 4.005e-9,
and the warmed evaluation took 0.006109 s with 5,180,048 allocated bytes.
Receipt: `poisson-auto-1000-v130.log` (exit 0, elapsed 545 s, including a shared
compile-cache wait).

At 40,128 coordinates the production squared-precision proxy also passed
separate quadrature/probe refinement and draw controls. With the same eight
probes and fixed reference, halving the quadrature spacing changed the
determinant by only 0.000241573 and the selected sensitivity by 6.326e-8 in
relative norm. Increasing to 32 probes at the finer spacing changed the
log-determinant error from +4.13482 to -7.37158 (MCSE 11.3014), while the
trace-inverse derivative error improved from -42.1306 to +0.434782. Thus the
probe refinement did not monotonically improve every result. The 32-probe
fine-quadrature determinant took 145.564 s, versus 11.871 s for the default
eight-probe rule in this run. A warmed conditional draw took 23.0104 s and
allocated 129,983,296 bytes. Repeated draws with the same seed were identical
and finite; the precision quadratic form was 40426.18, 1.053 standard
deviations above its Gaussian expectation. This single diagnostic is not a
covariance certificate. Receipt: `production-40128-refinement-v130.log`
(exit 0, elapsed 451 s).

The production determinant and selected contractions at 561,792 coordinates
passed with the unchanged 512-iteration budget: 71 solve iterations and 57
shifted-solve iterations, with original-system residual 1.030e-13. The warmed
determinant took 203.056 s and allocated 1,260,147,800 bytes; the retained
operator factor, reference and selected contractions occupied 1,358,524,945
bytes. These are cumulative allocations and retained object sizes, not a
peak process measurement. The determinant was 2602608.42155 versus the
separable eigenvalue oracle 2602695.43665, an error of -87.0151 with probe
MCSE 103.440. The trace-inverse derivative was 44151.91865 versus exact
44065.11355, an error of +86.8051. Reference preparation took 14.013 s
including its first compilation; a warmed solve took 13.791 s. Host load was
31.77 at grant and 31.64 at release. Receipt: `production-561792-v130.log`
(exit 0, elapsed 655 s). This control measures conditional algebra; the
40,128-coordinate Poisson control above measures a full nonlinear target.
The first full Poisson run at 561,792 coordinates subsequently failed: its
conditional mode did not converge within 50 Newton iterations at population
point `[0.1,0.1,0.1]`, so both repeated evaluations returned a nonfinite result.
Finite differences were not reached. Receipt: `poisson-561792-v130.log`
(exit 1, elapsed 674 s, token held 606 s). This is not a successful
large-model result. `mode_trace.jl` records decrement, line-search step and
observed density gain using the existing mode primitives to diagnose it
without changing the policy or tolerances.
The trace localized the loss of progress: after three full Newton steps the
decrement was 1.964e-8, but sequential accumulation of the component density
obscured its final gain. Accepted step fractions fell from 1/32 to about
1e-10 while the decrement stalled at 1.843e-8. Linear solves remained accurate
(maximum 17 iterations, checked residual 9.723e-14). Receipt:
`poisson-561792-mode-trace.log` (exit 0, elapsed 310 s). Compensated component
density accumulation restored full steps with unchanged acceptance and
convergence tolerances. After the fourth full step the decrement dropped to
4.918e-19; the next was 6.686e-27, below the unchanged 1e-24 tolerance. The
gradient infinity norm fell to 2.220e-15. Linear solves used at most 17
iterations with checked residual 7.344e-14. Receipt:
`poisson-561792-mode-compensated.log` (exit 0, elapsed 543 s, token held 441 s).
The subsequent full Poisson target also passed at 561,792 coordinates. A
warmed value and gradient took 59.42895 s and cumulatively allocated
7,215,291,560 bytes; the first evaluation took 74.4110 s. Setup, including
compilation, took 231.385 s and cumulatively allocated 26,836,877,444 bytes.
These allocation totals are not peak process memory. Repeated evaluations
were identical, no mode failures were recorded, and finite differences over
all three population coordinates agreed to relative error 3.293e-9. Linear
solves used at most 17 iterations (checked residual 7.344e-14), shifted solves
14 (residual 9.694e-12). The log determinant was 1119555.02242, with probe
MCSE 25.4759 and 10,109,334 reference entries. This run did not compute an
exact sparse determinant, so it establishes density-gradient consistency,
not approximation accuracy at this size. Host load was 31.86 at grant and
37.56 at release. Receipt: `poisson-561792-compensated-fd.log` (exit 0,
elapsed 779 s, token held 776 s).

The refreshed exact plan-reuse controls passed 123 assertions: 61 for row
models and 62 for nonlinear sum factors, including strict subsets, comparison
with freshly prepared targets and finite-difference gradients. Receipt:
`exact-plan-reuse-v130.log` (exit 0, elapsed 509 s).

The first refreshed nonlinear run passed 50 assertions for crossed models,
policy retention and ordinary cells, then exposed a sum-factor bug: a closure's
local reduction variable shared the probe right-hand-side binding and replaced
that buffer with a shorter vector. Renaming the two bindings fixed it. The
focused overlapping-sum rerun passed all 10 assertions, but the process then
hit its 600-second limit before reporting the negative-binomial and declared
Gaussian sets; that run is not an overall pass. Receipt:
`nonlinear-sum-fix-v130.log` (exit 124, elapsed 603 s, including a shared compile
cache wait). The strict matrix-free plan-subset rerun passed all 32 assertions
for row and nonlinear-sum models, including comparison against fresh targets,
finite-difference gradients, rebuilt references and unchanged source plans.
Receipt: `matrixfree-subset-fix-v130.log` (exit 0, elapsed 291 s).

## Final focused acceptance

After the compensated density change and canonical dependency refresh, all
166 assertions in the four focused suites passed:

| Suite | Assertions | Exit | Elapsed | Retained receipt |
| --- | ---: | ---: | ---: | --- |
| Nonlinear targets, mode correction and plan policy | 74 | 0 | 456 s | `nonlinear-final-v131.log` |
| Finite determinant derivatives, solves and draws | 33 | 0 | 36 s | `algebra-final-v131.log` |
| Restricted row and nonlinear-sum plans | 32 | 0 | 280 s | `subset-final-v131.log` |
| SMC ownership, checkpoint replay and target identity | 27 | 0 | 202 s | `smc-matrixfree-v131.log` |

The nonlinear suite covers crossed effects, ordinary nonlinear cells,
overlapping sums, moving negative-binomial shape and declared Gaussian priors
with moving mean and scale. The SMC suite verifies replay from a fresh target,
rejection of changed policy/probes/reference values, independent workspaces,
conditional draws and cumulative-data references.

These runs started at `d28f513e` using ReactiveKernels 0.1.52 and
ReactiveKernelsPPL 0.1.57 from `1968438e6d72`, Enzyme 0.13.217 from
`93e89fc870d2`, Krylov 0.10.12, KrylovKit 0.10.4 and
LimitedLDLFactorizations 0.5.2. While they ran, `99684930` merged canonical
`fd30f300` and reserved version 0.2.132. That merge changed ragged draw labels,
documentation and the version; the Laplace/SMC numerical source and dependency
pins were unchanged. The test manifest's path dependency still recorded
version 0.2.131, but loaded this worktree. This is focused acceptance, not a
run of the entire repository suite.

The companion guide at `03a4e1923`, based on Claude `b75c517f`, passed all six
affected fixtures (exit 0, 22 s, `skills-fixtures-v132.log`). The source and
guide remain review candidates until the user's landing decision is resolved.
