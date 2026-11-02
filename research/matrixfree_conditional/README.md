# Matrix-free conditional Laplace measurements

These are synthetic controls for the nonlinear embedded-Laplace implementation.
They do not import an ODL model, graph or observation. Results below were measured
on `gordito` on 2026-11-02 with one CPU per run. Host load varied; allocation
counts are cumulative allocations, not peak resident memory.

## Reproducers

Use the repository's prepared `test` environment and the required compute-token
wrapper for Julia runs.

```sh
julia --project=test test/nlme_laplace_matrixfree_algebra.jl
julia --project=test test/nlme_laplace_matrixfree.jl
julia --project=test research/matrixfree_conditional/scaling.jl 44 38 24 det
julia --project=test research/matrixfree_conditional/nonlinear.jl 10 10 10 exact fd
julia --project=test research/matrixfree_conditional/production.jl 8 8 8 refine draw
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
`/home/niko/scratch/matrix-free-cond-148dea8d/` on the measurement host.

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
timing increase cannot be attributed to dimension alone. The production
integration and full nonlinear target remain to be validated.

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
