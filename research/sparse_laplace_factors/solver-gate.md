# Sum-cell sparse solver gate

This records the solver experiments and fitting integration on 2026-10-31.
The final section describes the public fitting implementation and its
acceptance checks; the earlier dated sections preserve the rejected paths
and intermediate evidence.

The derivative preparation and chain rule are in
`src/nlme_laplace_factors.jl`. Each reduction summand retains its own compact
selection and Hessian. The outer Hessian acts on sums and direct inputs.
The resulting precision has the form `H = D + B' W B`; neither the outer
clique nor the full inverse should be materialized.

## Algebra and verified checks

The augmented system before an orthogonal change of auxiliary coordinates is

```
K = [ D  B'  0
      B   0  I
      0   I  W ].
```

Its Schur complement is H. The auxiliary block has absolute determinant one
and equally many positive and negative eigenvalues, including for singular
or indefinite W. A 45-degree rotation gives the implemented two-copy B
pattern. Selected entries of its inverse provide local covariance blocks,
`S B'` and `B S B'` for the third derivative contraction.

The independent tests in `test/laplace_factors.jl` passed 73 checks: 34
augmented solve/log-determinant/selected-inverse checks, 34 covariance-root
checks, and five support-scaling checks at 32 and 4,096 summands. The batch
exited 0 in 79 seconds. Dense references occur only in small fixtures.
Covariance draws use KrylovKit's public reorthogonalized Lanczos recurrence,
with a fixed basis cap and an explicit error bound. Those helpers also remain
unconnected to public fitting.

## CHOLMOD cannot be the unconditional augmented factor

A valid positive-definite H can have a zero local pivot. For example, take
scalar D=0, B=1 and W=1. Then H=1, whereas the rotated augmented matrix is

```
[ 0          1/sqrt(2)  1/sqrt(2)
  1/sqrt(2)  1.5       -0.5
  1/sqrt(2) -0.5       -0.5 ].
```

CHOLMOD's unpivoted LDL factor fails with its observed ordering [1,2,3].
The scalar grid D in {-1,0,1,2}, B in {0.1,1,2}, W in {-2,-1,0,1,2,4}
found nine such failures among valid H>0 cases. A failed augmented LDL must
not be reported as proof that the model's stationary point is not a maximum.
Julia's [documented LDL requirements](https://docs.julialang.org/en/v1.10/stdlib/LinearAlgebra/)
explicitly require well-conditioned leading principal minors.

## MUMPS experiment: correctness, memory, and runtime are separate gates

MUMPS.jl 1.7.0 with the sequential backend handled the pivoting cases. The
43 valid scalar cases passed 172 checks of inertia, log determinant, solve
and selected inverse. The documented low-level solve-phase interface also
passed repeated solve/selected-inverse transitions. Use explicit JOB=3 for
that interface; the convenience solve following selected inversion did not
produce the expected result in this experiment.

The convenience selected-inverse function allocated about 134.8 MB at
4,096 effects: it is not suitable for this work's memory requirement.
The documented sparse RHS interface avoids that allocation. GC-preserved
CSC value/index buffers were supplied to `RHS_SPARSE`, `IRHS_SPARSE` and
`IRHS_PTR`, with `NRHS=N`, `NZ_RHS=nnz(request)`, ICNTL(30)=1 and ICNTL(21)=0.
The [MUMPS manual, section 5.18.4](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
states that dense RHS storage is unnecessary in this mode. Julia's
[public low-level interface](https://jso.dev/MUMPS.jl/stable/api/) exposes
that structure and phase invocation.

For D=diag(0,1,...,1), B=[1,0.01,...,0.01], W=1, the sparse interface gave:

| Effects | Factor entries (INFOG(35)) | Julia allocations | Selected-inverse seconds |
|---:|---:|---:|---:|
| 128 | 1,013 | 7,728 B | 0.000167 |
| 512 | 1,710 | 17,024 B | 0.000497 |
| 4,096 | 12,462 | 102,880 B | 0.00958 |
| 32,768 | 98,478 | 791,008 B | 0.4449 |

Requesting only inverse diagonals did not remove the runtime problem:
0.000419, 0.00867 and 0.4403 seconds at 512, 4,096 and 32,768 effects.
These are exploratory single-run measurements, not a formal complexity
proof. The roughly 50-fold growth for the final eightfold size increase
fails the intended scaling gate despite linear storage. Do not adopt MUMPS
selected inversion merely because the allocation problem is avoidable.

The solver choice is still open. It must handle valid zero/indefinite local
blocks, preserve inertia and log determinants, supply the required inverse
entries with suitable scaling, and support independent chain workspaces.
Do not replace a package solver with an unreviewed numerical kernel, silently
perturb the target, or form a whole-cell dense Hessian to bypass this gate.

## Pivoted LU selected-inverse feasibility

`pivoted_lu_probe.jl` is an executable research probe, not BRM implementation.
It uses Julia's existing UMFPACK factorization and the scalar version of
[Jacquelin, Lin and Yang (2017), Algorithm 1](https://arxiv.org/abs/1708.04539).
SelectedInversion.jl currently documents CHOLMOD and LDLFactorizations
inputs, not pivoted sparse LU. A reusable implementation would belong in
that numerical package rather than in BRM.

The probe passed 58 small correctness checks (43 valid augmented scalar
cases, 15 random nonsymmetric matrices) and 15 large-case analytic/trace
checks. The final run exited 0 in nine seconds at 17:48:52Z. It preserves
the symbolic closure of the requested pattern under UMFPACK's final row
and column permutations: the numerical factors alone omit exact
cancellations, and an original-coordinate zero diagonal still has to be
requested. Earlier probe versions failed these coverage cases; the current
checks retain them.

On the same zero-local-pivot arrow system used above, three runs after
warming gave these minima, including symbolic closure and inverse assembly:

| Effects | LU entries | Selected entries | Julia allocations | Seconds |
|---:|---:|---:|---:|---:|
| 512 | 2,565 | 3,075 | 1,834,016 B | 0.000676 |
| 4,096 | 24,581 | 24,579 | 15,094,560 B | 0.00514 |
| 32,768 | 163,845 | 196,611 | 116,180,272 B | 0.0865 |

Storage follows the sparse factor pattern. These results justify pursuing
the upstream capability; they do not establish scaling for arbitrary
overlapping cells. The unoptimized probe allocates many small sets.
Inertia remains a separate requirement: a pivoted LU determinant cannot
certify a positive-definite latent precision. A mature symmetric pivoting
solver such as the tested MUMPS backend can supply that check, but using
two factorizations and the required package lifecycle must still be assessed.

## Producer dependency resolved

RK 0.1.34 / RKPPL 0.1.28 at published revision
`30d676c97e0b8c95066b472d6267be2ffd174744` repairs the single-sum outer
exemplar. BRM now requires those versions and pins that revision. No BRM
shape adjustment was needed. The test environment resolved successfully
(166 seconds), then all 95 checks in `test/laplace_factors.jl` passed
(117 seconds, exit 0 at 2026-10-31T17:47:47Z). This includes the independent
gradient, Hessian and third-derivative references, production covariance
contraction, shared operator reuse, repeated/ragged/empty reductions, and
compact support at 32 versus 4,096 rows. The solver gate above remains open.

## Sparse assembly and covariance projections

Checkpoint `e7e5f3aa` assembles D, B and W directly from the compact factors
and obtains the local covariance contractions from selected augmented inverse
entries. Structural zeros are retained even when a current derivative vanishes;
third derivatives still need those coordinates. The complete factor file passed
137 checks in 216 seconds, exit 0 at 2026-10-31T18:02:22Z. This includes
independent full gradient/Hessian/third references, mixed population/effect
coordinates, covariance projections and linear sparse storage. These helpers
are still separate from the public fitting path.

`assembly_scaling.jl` measures the same prepared factor fixture, using three
warmed evaluations at each size. It exited 0 in 109 seconds at 18:03:08Z.

| Summands | Packed coordinates | D entries | B entries | W entries | Seconds | Allocated bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 35 | 164 | 70 | 4 | 0.00131 | 5,029,216 |
| 512 | 515 | 2,564 | 1,030 | 4 | 0.0174 | 79,718,160 |
| 4,096 | 4,099 | 20,484 | 8,198 | 4 | 0.175 | 638,749,888 |

Storage and allocation growth are linear in this fixture. The allocation
constant is still large and needs attention before large fits. An isolated
summand derivative call allocates exactly 26,528 bytes at all three sizes;
its warmed time stays between 4.8 and 7.5 microseconds. These measurements
cover assembly only, not factorization, selected inversion or fitting.

The shared Juliaup launcher failed before the final verification began. The
successful final runs used its already-installed Julia 1.10.12 executable,
the same default depot, native static Enzyme and the normal compute wrapper.
The host incident is tracked separately; no shared launcher files were edited.

## Preparation ownership and full factorization measurements

The allocation profile found that each term embedded the entire immutable
AD preparation (3,112 bytes). Holding that shared preparation by reference
reduces the term record from 3,176 to 72 bytes. At 512 summands, warmed
assembly allocation falls from 79,716,128 to 21,968,208 bytes (72% less).
The per-term derivative itself still allocates 26,528 bytes. All 137 factor
checks passed again after this change (119 seconds, exit 0 at 18:19:59Z).
`assembly_allocations.jl` reproduces the attribution.

`inertia_lifecycle.jl` tests a separate MUMPS inertia/determinant workspace,
discarding numeric factors (ICNTL(31)=1), with exact root inertia enabled
(ICNTL(13)=1). It reuses symbolic analysis through the public phase interface
while values and negative-pivot counts change. The 232 small checks and 13
scaling/solve checks passed (18 seconds, exit 0 at 18:12:40Z). Repeated
inertia factorization took 0.000296 / 0.00255 / 0.0198 seconds at 512 /
4,096 / 32,768 effects, with 0.126 / 0.985 / 7.87 MB of Julia allocations
and 1 / 2 / 14 MB reported MUMPS workspace. This verifies interleaved
independent instances, not concurrent entry from Julia threads.

The full factorization gate matters. The earlier selected-inverse probe
excluded LU factorization. `lu_factor_scaling.jl` found default UMFPACK
fresh factors took 0.000762 / 0.0344 / 2.10 seconds on the same arrow
fixture. Reusing its analysis took 0.000611 / 0.0285 / 1.80 seconds.
Selecting its documented symmetric AMD strategy did not resolve this.
Explicit leaf-first ordering improved numerical time but caused an initial
allocation estimate of about 6 GB at 32,768 effects; that configuration
does not satisfy the memory requirement. The unrotated augmentation did
not fix the default factorization timing. These are rejected configurations,
not a conclusion that all UMFPACK configurations must behave this way.

The alternative mature SuiteSparse solver KLU 0.6.0 passed the same sparse
solve checks and a small public-factor reconstruction. With its defaults,
`klu_factor_scaling.jl` measured:

| Effects | LU entries | Fresh seconds | Fresh allocated bytes | Refactor seconds |
|---:|---:|---:|---:|---:|
| 512 | 3,077 | 0.0000463 | 331,576 | 0.00000772 |
| 4,096 | 24,581 | 0.000464 | 2,636,720 | 0.0000627 |
| 32,768 | 196,613 | 0.00261 | 21,078,528 | 0.000568 |

This run exited 0 in six seconds at 18:19:46Z. KLU is a promising factor
source, not yet an adopted BRM dependency. Its public factor identity and
scaling differ from UMFPACK, and its BTF off-diagonal factor must be
accounted for in any general selected-inverse adapter. This symmetric
connected fixture has no such off-diagonal block. Inertia, selected-inverse
integration, pivot changes, overlapping cells and concurrency remain gates.

`pivoted_backend_checks.jl` subsequently passed 1,057 checks: 576 for
changing pivots and overlapping augmented cells, and 481 for four independent
workspaces on two Julia threads. KLU factorization/solves run independently;
MUMPS initialization, factorization and finalization are serialized through
one process lock. This verifies that specific lifecycle, not unrestricted
concurrent MUMPS entry. Every concurrent result is asserted in the parent
testset. The run exited 0 in eight seconds at 18:24:44Z. Its small dense
references check inertia, log determinant, physical solves, residuals, BTF
and the public factor identity. Selected inversion and public BRM fitting
are not part of this experiment.

The final assembly benchmark after sharing preparation passed at 18:24:03Z
(96 seconds). At 32 / 512 / 4,096 rows, allocations were 1,410,016 /
21,970,240 / 176,841,456 bytes and minimum warmed times were 0.000881 /
0.00950 / 0.0958 seconds. Sparse entry counts are unchanged. The 4,096-row
allocation is 72% below the original 638,749,888 bytes; it remains linear.
Four additional chain-copy checks verify that a copy owns independent AD
preparations, preserves its internal sharing, matches the full-model
gradient at another point and leaves the original evaluation unchanged.
All 54 checks in that affected test group passed (105 seconds, exit 0 at
18:25:08Z), after the full 137-check run recorded above.

Next dependency: the approved SelectedInversion package task `00d73lp`
must deliver and publish the supported factor/selected-inverse boundary.
Its task now carries the measured KLU alternative and the rejected default
UMFPACK configurations. After that, connect the validated numerical path
to Newton assembly, marginal gradients and draws, then verify actual fits
and overlapping-cell scaling before the BRM landing decision. Neither KLU
nor MUMPS was a BRM dependency at that checkpoint, and no sum-cell landing
had been requested.

## Public fitting integration and scaling

The implementation now routes sum-factored cells through the existing sparse
mode search, retaining `D`, `B` and `W` separately. KLU 0.6 supplies a fresh
pivoted factor at each curvature; `klu!` is unsuitable here because it retains
old numerical pivots and can fail on a nonsingular updated matrix. The
SelectedInversion workspace can still be reused as its documented API
reprepares after pivot changes. The required fleet revision is
`8b1ee1077cea94c1aff746e182681ae39ce91526`; registered version 0.2.2 lacks it.

MUMPS 1.7 uses a temporary sequential workspace under one process lock, with
exact root inertia, numerical null-pivot detection and discarded numeric
factors. Its native workspace is finalized before the lock is released.
Null-pivot detection matters because rotation can leave roundoff at an
exactly singular Schur complement. A problem copy resets the factor cache;
chains retain separate KLU factors and selected-inverse workspaces.

The gradient combines the local third-order chain rule with selected entries
of the augmented inverse. Stored Jacobian zeros must survive concatenation:
they request inverse entries even when the corresponding current Jacobian
entry vanishes. Ordinary sparse concatenation removed them and produced a
wrong nonlinear marginal gradient despite correct modes and Hessians. The
production assembly now preserves those requests explicitly.

Conditional draws use the bounded Krylov square-root action above. Automatic
term selection caches each cell's baseline sums and summand incidence, so
changing one level only re-evaluates its affected summands. Small covariance
blocks are read directly from their requested entries. General CSC submatrix
indexing allocated a model-sized row map for every small block: at 4,096
overlapping effects, removing those maps reduced the marginal derivative's
allocation from 3.96 GB to 0.584 GB, excluding mode finding. The corresponding
512-effect derivative allocates 73.2 MB.

`test/nlme_laplace_sums.jl` checks exact Gaussian marginal values and gradients,
conditional covariance and importance weights, automatic selection, mode
fitting, independent copies, nonlinear finite differences with two reductions
and direct inputs, ICAR soft constraints and local diagnostic densities,
and scalar singular/indefinite curvature. All 247 checks pass in the fresh
current-pin validation process. The factor suite's 333 checks also pass;
a separate two-thread process verified all 192 concurrent-workspace assertions
(exit 0 at 20:40:02Z). Fresh-process acceptance on source `cfe9bae1` and the reconciled pins exited
0 at 21:33:12Z: 247 sum-cell, 333 factor, 26 plan, 44 ladder, 158 existing
sparse-engine, four changelog and seven dependency-floor checks (819 total).
The command included those seven standalone test files in separate modules.
Its retained a Linux fleet host receipt is `kb-run-compact.2EF1fc` under
`<local path omitted>`.
The later block-fallback review restricted its auxiliary system to the rows
incident on that block. It previously retained all auxiliary rows even when
most had zero Jacobian on the block. Six added checks cover reordered block
indices, indefinite local curvature, damping with no incident rows, and
allocation growth below 2x when the model grows from 128 to 4,096 effects.
Those checks and the existing numerical factor, square-root and concurrent
workspace checks passed together: 268 checks, exit 0 at 21:39:11Z, receipt
`kb-run-compact.Y7OoND`. This follow-up includes the production helper in
isolation with the same numerical dependencies as the test manifest; it does
not claim a repeat of the full 819-check run. The factor suite now has 339
checks. The later complete run below includes these six checks and the
indexed outer-footprint regression.

`fitting_scaling.jl` measures the public value/gradient path, including Newton
steps, factorization and selected inversion. On a Linux fleet host with a one-CPU cap,
source `9b229e3a`, RK `23d26ca1`, Enzyme `0e251cfb` and SelectedInversion
`8b1ee107`:

| Shape | Effects | Gradient seconds | Allocated bytes | D/B/W entries | Augmented entries | Eight-draw per-level diagnostic seconds |
|---|---:|---:|---:|---:|---:|---:|
| One nonlinear sum | 512 | 0.0605 | 135,373,720 | 1,540 | 4,624 | 0.878 |
| One nonlinear sum | 4,096 | 0.556 | 1,077,497,672 | 12,292 | 36,880 | 1.09 |
| Overlapping triples | 512 | 0.173 | 400,422,008 | 5,612 | 20,912 | 0.101 |
| Overlapping triples | 4,096 | 1.63 | 3,212,182,184 | 45,036 | 167,856 | 0.927 |
| Two sums plus indexed direct read | 512 | 0.112 | 298,790,680 | 3,620 | 12,944 | 0.772 |
| Two sums plus indexed direct read | 4,096 | 1.03 | 2,365,994,264 | 28,708 | 102,544 | 1.22 |

The combined acceptance and benchmark run exited 0 at 00:02:47Z on
2026-11-01 (1,063 seconds); its allocation and sparse-storage growth
assertions pass. These are total allocations, not peak
memory. In the earlier RK `30d676c9` allocation profile, both warmed mode
searches in the overlapping comparison took four iterations; mode allocations
grew from 332 MB to 2.66 GB. Compilation and machine load affect the times.
The substantial allocation constant remains visible even after eliminating
the quadratic extraction cost. No sampling-efficiency comparison is claimed.
Scaling assumes compact summand and outer-input footprints; an outer function
that directly reads a broad effect vector has a correspondingly broad local
derivative block. It is separate from the sparse storage of the reduction's
combined effect support.

### Indexed outer-input footprint repair and acceptance

An indexed direct predictor read previously acquired a model-wide footprint.
On RK `81363978` (0.1.38 / PPL 0.1.32),
the outer query for `Normal(sum(exp.(eta[rows[k]])) + eta[anchor[k]], sigma)`
reports every packed coordinate: 19 for 16 effects and 67 for 64 effects,
although `anchor = [1]`. A summand's footprint stays at three coordinates.
The growing outer selection would make its local derivative block quadratic;
documenting compactness as an assumption does not repair that defect.

ReactiveKernels snag `sum-factor-outer-ca5cc80e` repaired this in published
`23d26ca1aa8aea14998a1fab32c451d1a14dd190` (RK 0.1.39 / PPL 0.1.33).
The consumer continues to use public factor and footprint queries.
`outer_footprint_scaling.jl` now passes at 16, 64 and 4,096 effects: the
outer query reads four packed coordinates and each summand reads three.
`test/nlme_laplace_sums.jl` permanently checks the same bounds with two sums
and a direct indexed read. The last two benchmark rows above exercise the
complete mixed value/gradient and per-level diagnostic paths.

On source `9b229e3a628d8c11064dc3f1c9985ad6f9b5eee7`, all 831 regression
assertions passed: 253 sum-cell, 339 factor, 26 plan, 44 ladder, 158 existing
sparse-engine, four changelog and seven dependency-floor checks. The separate
scaling script also repeats 201 scalar-curvature assertions. The retained
a Linux fleet host receipt is `rk139-acceptance.log` under
`<local path omitted>`.
The benchmark reports intermittent MUMPS out-of-core warnings despite the
requested in-core controls. Separate public-API probes completed 3,300
factorizations, including GC/allocation stress, with ICNTL(22)=0, ICNTL(16)=1
and ICNTL(48)=0 before and after factorization, with exact inertia and
determinant checks, without reproducing the warning. A two-thread probe
using this candidate's augmented matrices, KLU and selected inverse passed
400 further factorizations with the same control assertions. Receipts:
`mumps-control-warning.log`, `mumps-control-warning-stress.log` and
`mumps-augmented-warning.log`, all exit 0 (last at 00:16:27Z). These do not
establish why the acceptance process warned. The investigation is tracked
by KB todo `2026-11-01T01-16-48-166-0nfgko8` on this lane.

After the published SMC release landed as BRM 0.2.94, source `a860fa6d`
reconciled this candidate as 0.2.95 with RK `4ddd18e5` / PPL 0.1.34. The new
RK pin contains the indexed-footprint repair. Fresh setup and all 831
regression assertions passed again, together with the 16/64/4,096 footprint
probe. Receipt: `post-smc-acceptance.log`, exit 0 after 1,667 seconds at
2026-11-01T00:39:55Z. This run emitted no MUMPS out-of-core warning. It did
emit the sparse engine's normal rejection warning for a sampler trial whose
mode failed to converge; the sampler/ladder checks passed. The scaling table
remains attributed to the earlier source and dependency pair above.

The targeted control trace reproduced six warnings during 655 factorizations
(`trace-mumps-fitting.log`, exit 0 after 314 seconds at 00:47:11Z). Every
warning occurred inside the `Mumps` constructor, through its initialization
call to `invoke_mumps_unsafe!`. Every returned constructor had
ICNTL(22)/ICNTL(16)/ICNTL(48) = (0, 1, 0), and assertions confirmed those
values again after each factorization. All three fitting/scaling cases and
their numerical/storage checks passed. This establishes a constructor-time
diagnostic rather than out-of-core storage during the observed
factorizations; the cause within MUMPS remains for its maintainers to resolve.
`mumps_control_trace.jl` preserves that diagnostic driver with shortened stack
printing. It instruments only BRM's own wrapper in the research process;
production code does not suppress the warning or substitute solver controls.

## Stiff outer blocks: the rotation is gone (2026-11-06)

This section supersedes the 45-degree rotation described above. The rotation was
adopted for its two-copy B pattern, and stiffness accuracy was not among the
measured gates. On a bounded ODL target with outer curvature 1.9e7 it left a
residual of 0.83% in `h v = s`, which moved `odl_offset_Intercept`'s gradient from
395 to 19214: that coordinate is aliased with an integrated effect, so its explicit
and implicit terms cancel down to exactly this residual.

The rotated variables are (q ± p) / sqrt(2) with p = B x and q = W p, so x is recovered
from a difference of numbers of W's size and the residual grows as W². Residuals of
`h x = b` on a rank-deficient outer block, 30 effects, 13 outer inputs:

| outer curvature | rotated, first | rotated, refined | unrotated, first | unrotated, refined |
|---:|---:|---:|---:|---:|
| 1e7 | 49164 | 3.6e-9 (6 steps) | 1.57 | 4.4e-9 (2 steps) |
| 1e8 | 1.9e7 | 3.7e-8 (16 steps) | 277 | 3.1e-8 (3 steps) |
| 3e8 | 6.8e8 | 6.3e8 (no convergence) | 1907 | 9.0e-8 (3 steps) |
| 1e9 | 2.8e10 | 2.8e10 (no convergence) | 28152 | 4.2e-7 (4 steps) |
| 1e12 | 2.8e19 | 2.8e19 (no convergence) | 1.4e9 | 4.0e-4 (7 steps) |

K is now `[D Z' B'; Z W -I; B -I 0]` on (x, p, q), with Z a zero-valued copy of B's
pattern that requests [x,p] = S B'; [p,p] = B S B' sits in W's pattern. The unrotated
columns above were measured before adding Z, which holds no values; the final form's
residuals are asserted in `test/laplace_factors.jl` up to 1e10.

On the final form, definiteness matches dense on that family and on five small cases
(indefinite h, singular h, zero D, indefinite W, zero W), and selected blocks are as
accurate as the rotated form's at every curvature. Log-determinant errors against dense
are comparable up to 1e10 (5.3e-7 against 8.0e-7) and larger at 1e12 (1.1e-4 against
2.2e-6).

The UMFPACK rejection above does not apply: BRM factors with KLU. Costs at 512 / 4,096 /
32,768 effects are identical on the arrow fixture. On overlapping three-effect cells
at 32,768 effects, K has 524,288 entries against 819,200 and LU 983,023 against 1,212,394,
and KLU took 0.021 s against 0.028 s, MUMPS 0.046 s against 0.053 s and selected
inversion 0.028 s against 0.044 s.

Each solve also refines against h itself while a step at least halves the residual,
because the gradient keeps the absolute residual and even the unrotated first solve's
grows with W. K's own residual is not the measure: its q rows bottom out at roundoff of
W's size.
