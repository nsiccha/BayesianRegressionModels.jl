# Running the tests

`seal_density_dependence.jl` checks both grey-seal model forms against Vanko
et al. (2026): birth and pregnancy boundary identities, compiled gradients
against an independently integrated loss hazard, and a finite full-model start.
`seal_density_fit.jl` separately runs a short four-chain fitting check and
re-evaluates retained draws for finite densities and gradients. Its short fit
does not establish convergence or reproduce the paper's analysis.
`seal_joint_support.jl` checks both authored helper sets against the original
joint density, finite-difference coordinate Jacobians and native gradients
across four age counts. The test environment pins StanBlocks `dcabd24f`
(0.2.40), which includes differentiable algebraic roots and native evaluation
of the data-only support bound during descriptor construction.

`rk_distribution_statistics.jl` checks native LogNormal moments and log tails:
constructor/callable identity, one-sided bounds, values against 512-bit
references, ordinary Enzyme Reverse gradients and printed-source replay.
`rk_distribution_graphs.jl` checks completion-law normalization, support,
mapped-family coverage and gradients. Run either with `julia --project=test
test/<file>.jl`; both accept testset substring filters. The test pin includes
RK 0.1.161's allocation fix for one-sided variance.

The moment benchmark writes timings, allocations and public emitted source
to an output directory. Run both shared and row-varying bounds, then audit
the shared case's placement before the observation loop:

```sh
julia --project=test research/distribution_graphs/hoisting.jl /tmp/brm-moments-shared
julia --project=test research/distribution_graphs/hoisting.jl /tmp/brm-moments-varying --varying-bound
julia --project=test research/distribution_graphs/audit_hoisting.jl /tmp/brm-moments-shared
```

`linear_predictor_entries.jl` checks that an entry of `linear_predictors(brmi)`
is a predictor address: `popcoefnames`, `predictors`, `grouping_factors`,
`dependencies` and `linear_predictor_op` give for the `(; name, link_lhs_fn)`
record what they give for its name, on a formula predictor, a linked one and a
declared parameter, and the documented `popcoefnames` scan needs no filtering.
It builds no backend; run it with `julia --startup-file=no --project=.
test/linear_predictor_entries.jl`.

`rk_adaptive_centering.jl` checks `adaptive_centering_problem(rk, problem,
backend)` on RKBRMI random effects: the blocks name the RK layout's draws, scales
and correlation-factor coordinates; each cell is located on its population
coefficient, and a nested grouping on its parent's effect; the wrapped density,
Jacobian and native Reverse gradient match an independent map of the plain RK log
density (with central differences) at zero, one and random controls, for `||`,
correlated and nested blocks; and the blocks it does not build raise. At the
default start (`c = 0`) the wrapper's map, log density and gradient equal the
plain RK density's where a block's log scale is -750 (`exp` underflows to 0)
and that density is finite, for `||`, correlated, three-margin and nested
blocks, through BRM's own reverse passes and WarmupHMC's generic one (snag
`rkbrmi-adaptive-326587b2`). About 4 minutes warm; it accepts testset substring
filters.

`rk_data_expression_scalars.jl` checks a numeric scalar data field read by an RKBRMI
data expression: offsets of many shapes, a coefficient's column, fitted transforms,
an interaction operand, a random slope, a data-only assignment, a component's
argument and model-value offsets. It compares an independent likelihood oracle and
compiled Stan (value and native Reverse gradient), replays the printed source, checks
that another value emits the same program, and covers frozen and data-free replay, a
design binder, a held-out response and the forms that stay refused. It accepts testset
substring filters.
`runtime_numeric_settings.jl` checks that sizes and numeric settings are inputs of
the emitted programs (user principle `0ikeidi`): one Stan and one RK program across
a `gmrf` or `hsgp` setting, a data scalar a prior reads, a block's LKJ `eta`, a
factor's levels, a known scale and a known correlation, with one lowered program and `prepare_query`
type where RKPPL allows it and a literal as the positive control. One retained build
evaluates values A, B, A against an analytic density and gradient. It also checks
frozen and fresh replay and that checkpoint identities see a prior's numbers. The
`@test_broken` lines name the ReactiveKernels snags they wait on. About 15 minutes;
it accepts testset substring filters.

`rk_data_law_arguments.jl` checks RKBRMI formula responses whose law reads
data: a scalar data field, a `constant(...)` value, a plain alias and a per-row
column as a Normal scale, a censored Normal, a Student-t scale, a LogNormal scale
and a model-scope `gmrf` readout. It compares an independent likelihood oracle and
compiled Stan (value and native Reverse gradient, through the coordinate
transport), replays the printed source, checks that another value emits the same
program, that binding refuses an invalid value, and frozen `reprocess` of a
scalar field and of a constant. The further arguments of the other response
families are checked against the oracle and finite differences. A per-row `zi`
or Student-t `nu` outside its support is checked as a refusal, which BRM's
ReactiveKernelsPPL 0.1.105 compat floor guarantees. It accepts testset substring
filters.

`rk_value_weighted_observations.jl` checks formula-level `fweights` and
`weights` on RKBRMI responses whose law reads an assigned value: a Julia
reader's probability under `weighted(Bernoulli(p), fweights(n))` against
its `Binomial(n, p)` equivalent, power, frequency and censored Normal rows,
a predictor read through an assignment and a mixture with per-row component
weights. It compares normalized densities
and native Reverse gradients with independent oracles and compiled Stan,
checks printed-source replay, rebinding and value checks of a new design's
weights, and the refused analytic, `mi()` and invalid-weight cases. It
accepts testset substring filters.

`epidist_docs_page.jl` checks the delay model now retained in
`research/epidist/`: its source markers resolve, and its native declaration
produces a Stan pane that `stanc` accepts. The renewal page links the research
README. No sampling.

`censored_delays.jl` checks `primary_censored` and `double_interval_censored`
(epidist's marginal delay model). The closed-form primary-censored CDF and
survival of lognormal, gamma and Weibull delays are compared with an
independent 256-bit tanh-sinh quadrature of the convolution, including shapes
below one. It also checks upper-tail window probabilities, truncation, windows,
a shifted primary window and draws, and native Enzyme Reverse gradients. A
weighted `@brm` model on SBBRMI and RKBRMI is compared with the Julia law in
value and gradient for all three laws. The test also covers exact secondary
times and the row checks. Run it with `julia --project=test
test/censored_delays.jl`; about 3 minutes warm.

`offset_only_axes.jl` checks coefficient-free formulas on scalar plate and
component rows, fixed expressions, explicit gathers, changed-size replay and
in-cell consumers. It compares physical values, normalized densities and native
Reverse gradients with independent formulas and compiled Stan, and retains the
row-length and scalar-cell checks. It accepts testset substring filters.

`model_view_documentation.jl` checks authored Julia documentation in builder and
bound-model views, plus callable docs after binding: references become formatted labels,
ordinary URLs remain linked, doctest contents survive, and source/doc records
remain unchanged. The bootstrap selects HTMXObjects 0.1.65, which also renders
tabbed views as readable plain text without internal node representations.

`heterogeneous_mixtures.jl` checks native mixtures of different full-real
scalar families and custom component graphs against independent normalized
densities and compiled Stan. It covers native Reverse, pointwise likelihood,
printed-source replay, changed data, extreme log scales, Stan membership and
predictive outputs, and measured shared/row work placement. It accepts testset
substring filters through `testset_filter.jl`; empty-data rebinding has its own
testset so other cases remain independently runnable.

`lkj_data_inputs.jl` checks data-supplied correlation dimensions and concentrations
at model scope and in reusable and nested BRM components, including two sizes
in one model, derived data, frozen constants and correlated groupings. It compares normalized
densities and native Reverse gradients with literal controls, an independent
two-dimensional law and compiled Stan, and checks source replay, physical
coordinate transport and invalid inputs. The bootstrap supplies RKPPL 0.1.97.

`plate_selected_collection.jl` checks selections of vector cells passed to
ordinary helpers. Reordered and repeated selections, collected and non-last
outputs, analytic densities and gradients, and prior draws retain the same
cell semantics across distinct producer and reader frames.

`gmrf_multiresolution.jl` checks the sparse spatial basis, normalized shared OU
prior and analytic derivatives, compressed read maps, and the original
213-read temporal-covariance regression. `gmrf_multiresolution_backends.jl`
checks Stan emission and native Reverse derivatives, physical coordinate
transport, strict cell locality, frozen replay, multiple fields, descriptions,
and stationary prior draws. Both are standalone scripts in the test environment.
The backend file accepts testset-name substrings as command-line filters.
`Primal coordinate` independently compares native values and central finite
differences with compiled Stan, including sampled LKJ channels. Archive-scale
operator and cost controls are in
[`research/irregular_read_gmrf`](../research/irregular_read_gmrf/README.md).

`plate_direct_outputs.jl` checks direct reads of complete cells from another
plate, including fixed-width vectors, collected and non-last outputs, and
observed or omitted responses. Square and non-square shapes compare compiled
Stan densities and gradients with analytic references; prior draws check the
same cell values. Scalar, ragged-vector and matrix cells supply shape controls.
Run it as a standalone script in the test environment.
`gmrf_stan_plate_outputs.jl` reads such cells beside a `gmrf` formula in one
plate cell, and compares the compiled density and gradient with a dense oracle.

`component_start_descriptors.jl` checks exact initial values in repeated and
nested components through the public descriptor, BridgeStan and coordinate
transport APIs. It covers independent scalars, unequal-length vectors, positive
scales and simplexes, and preserves unaddressed values and top-level starts.
Run it as a standalone script in the test environment; it compiles Stan even
when the resulting point is consumed by RK.

`component_prior_inputs.jl` checks component formula priors that read the
caller's parameters through inputs: population-effect locations and scales and
a component-local `|ID|` block's SD prior. It compares compiled Stan densities
and gradients with an independent density, checks each input's contribution
along its coordinate, and covers a nested call and a call whose input is a
number. The SD case is also checked on `RKBRMI`. It accepts testset substring
filters; run it as a standalone script in the test environment.

`component_start_values.jl` checks native component-local starts, recursive
scalar prior medians, repeated and nested call paths, constrained values,
explicit precedence, numeric-base preservation and input immutability.
The default run only builds RK models. Add `--stan` for compiled Stan
readback and cross-backend physical-point parity.

`matrix_start_values.jl` checks top-level LKJ Cholesky starts at identity and
nonidentity factors, matrix shape and support, unchanged base values and input
immutability. It also combines top-level matrices with scalar/vector starts,
component-local factors and prior medians. Add `--stan` for independent
compiled-Stan readback and physical coordinate parity.

`component_formula_starts.jl` checks `component_inits` entries that name a
component call's own linear predictors and shared `|ID|` blocks: link-scale
coefficients, block scales, correlation, covariance and per-level effects in
repeated and nested calls over different levels, component-local coefficient
and scale prior medians, unchanged top-level values and refusals. The default
run only builds RK models; `--stan` adds compiled readback of each call's
predictor values and cross-backend physical-point parity.

`smc_tuning_lifecycle.jl` checks adaptive pilot interruption before any draw and
during a later grid pilot, nested Treebars progress, input ownership, endpoint
and configuration validation, and exact fresh-process replay from both a
serialized partial and the latest disk checkpoint. It covers initial,
cumulative-update, selective carried-window and prepared sparse-Laplace targets
using small synthetic models. Run `test/setup_env.jl` first for the pinned RKPPL recovery contract and
Treebars 0.1.5, then `julia --project=test test/smc_tuning_lifecycle.jl`.

`smc_worker_forwarding.jl` checks that BRM forwards ReactiveKernelsPPL 0.1.100's
worker controls on the same four targets: `grid_start=:independent` reaches every
`brm_smc_tune` route and gives the same schedule at one and several workers, and
`nthreads` on every `brm_smc_resume` route replaces a production checkpoint's
saved worker cap without changing the resumed cloud. Its assertions hold at any
thread count; run `julia -t 4 --project=test test/smc_worker_forwarding.jl` to
also exercise concurrent pilots and workers.

`smc_rescore.jl` checks explicit production-checkpoint rescoring for initial
fits, first and later cumulative updates, ordinary and compact carried windows,
and initial and updated Laplace problems. It perturbs saved endpoint scores by
one or two ulps, requires strict resume to refuse them, and checks that rescore
returns an unchanged-input copy whose continuation matches exact same-host
replay. It also checks tolerance forwarding, audit history and data/model guards.
Run `julia --project=test test/smc_rescore.jl` with RKPPL 0.1.156 or later.

`smc_generated_law_recovery.jl` stops an adaptive pilot of a small `gmrf(...)`
field model at initialization in one Julia process and resumes it in another,
through `smc_generated_law_process.jl`. The field's law is bound in the
generated RK program module, so the consumer first confirms that the raw law
expression cannot be deserialized, then requires exact window and schedule
replay from a serialized partial and from disk. It needs ReactiveKernelsPPL
0.1.98 or later. The `setup_env.jl` pin also carries the selective-window
capture-footprint repair, covered by `smc_window_selective_api.jl`.

`scoped_components.jl` checks function-style BRM components with independent
local priors, explicit shared inputs, nested calls, vector results, and plate
cells. It compares normalized densities and native Reverse gradients with
independent references and compiled Stan, and checks source/documentation
inspection, constrained-value and coordinate round trips, row selection,
offsets, and frozen or refitted preprocessing. Native RKPPL documentation
retains component headers and local provenance at each nested call site.
Run it as a standalone script in the test environment; StanBlocks 0.2.25
preserves keyword labels when a component is called inside a plate, and
ReactiveKernelsPPL 0.1.93 retains authored documentation.

`component_cell_returns.jl` checks returned plate collections through direct,
aliased and nested component calls. It
verifies row/channel selection, strict per-row dependency footprints, emitted
source replay, normalized densities and native gradients against compiled Stan.
Run it as a standalone script in the same test environment.

`component_direct_rows.jl` checks direct downstream reads of component cells
with shared matrix inputs, including nested calls and a component result as
the plate's only extent. It compares physical values, normalized densities,
native Reverse gradients, compiled Stan and strict selected-row dependencies.

`authored_documentation.jl`, `authored_documentation_binding.jl`,
`authored_documentation_printing.jl`, `authored_documentation_rk_source.jl`
and `authored_documentation_rkppl.jl` check ordinary Julia documentation through
binding, composition, readable source replay, scientific descriptions, artifact
metadata and native RKPPL lowering. They verify exact declaration scopes and
unchanged numerical results. Run each as a standalone script in the test
environment after `test/setup_env.jl`; native metadata requires RKPPL 0.1.93.

`adaptive_centering_bounds.jl` checks bounded random-effect scales against
BridgeStan's constrained values, correlated centered and noncentered maps,
their Jacobians, Enzyme gradients, centered replay, and stable log-scale tails.
Run it as a standalone script in the test environment.

`rk_interval_tails.jl` checks interval-censored Normal responses through
`RKBRMI`: pointwise and total log densities, native Enzyme Reverse location
and log-scale gradients, and input immutability. Synthetic intervals cover
both tails through 400 standard deviations and moderate controls at two
parameter points. Independent 256-bit MPFR masses, analytic derivatives and
central differences supply the references. Run `test/setup_env.jl` first to
select the RKPPL 0.1.60 / DistributionKernels 0.1.1 repair, then
`julia --startup-file=no --project=test test/rk_interval_tails.jl`.

`powerscale.jl` checks BRM's density assembly against the compiled Stan
target and its PowerScaling.jl adapter against the pre-extraction Gaussian
sensitivity table. Generic CJS, PSIS and sensitivity tests live in PowerScaling.
The bootstrap supplies its pinned 0.1.0 release on Julia 1.10 too.

`nlme_laplace_matrixfree_algebra.jl` checks the derivative of the fixed-probe
Lanczos determinant against independent two-dimensional identities, including
noncommuting curvature, mixed exact/operator components, ICAR soft constraints,
conditional draws and explicit solver failures. `nlme_laplace_matrixfree.jl`
checks nonlinear row/cell/sum gradients, evaluation-history and copy invariance,
plan/ladder policy retention and nonconcave mode searches.
`nlme_laplace_matrixfree_subset.jl` compares restricted row and nonlinear-sum
plans with fresh preparation, including gradients of newly sampled effects.
The synthetic drivers
in `research/matrixfree_conditional/` measure operator/exact algebra and a full
Poisson/ICAR target separately; neither represents a fitted consumer model.

`runtime_constants.jl` checks strict RK inference of runtime `constant(...)`
expressions from `brm(source; mod)`, changed fit inputs, frozen inputs, and
printed-source round trips. It exercises constants feeding a parameter-dependent
helper, so they remain graph recipes rather than bind-time data.

`revise_ar1.jl` loads Revise before BRM, checks that all 16 AR(1) submodels
match their anonymous `@slic` definitions, and rescans the package twice.
It also edits and restores an unrelated method in a temporary copy of the
cell source, checking that explicit rescans apply the edits and preserve the
registered models. Run it as a standalone process with
`julia --startup-file=no --project=test test/revise_ar1.jl`.

`nlme_laplace_values.jl` checks sparse integration of model-scope observations:
gathered predictors on distinct row axes against an exact Gaussian marginal,
and derived count likelihoods against equivalent plate cells and numerical
derivatives. Automatic plans retain these likelihoods and gate their latent
terms with the per-block importance diagnostic.
`ragged_factor_preparation.jl` checks that a wide predictor's mixed column
types prepare type-stable ragged factors whose values equal the full model's
pointwise densities.

`draw_trees.jl` fits a small hierarchical model with WarmupHMC, then checks
the optional TreeArrays draw APIs: descriptor labels under permuted names,
views and allocation size, unequal chain lengths, ragged random-effect blocks,
predictive and conditional outputs, RK values, and `nanquantile` through
`TreeTable` to AlgebraOfVega. `ranef_draws.jl` and `level_correlation.jl`
also check the labelled wrappers against their existing independent oracles.
Use `julia --project=test test/setup_env.jl` first; it pins the optional
plotting dependencies as well as the fitting stack.

`draw_tree_coordinates.jl` checks random-effect parameter trees for known scales,
correlated and centered effects, stratified blocks, Student-t mixing weights
and shared predictors. Distinct supplied values and shuffled constrained names
verify level/margin storage order. Unequal chains give the same R-hat and bulk
and tail effective sample sizes as direct column selection, without copying
draw storage. The fitted `draw_trees.jl` fixture also checks the standardized
tree against compiled Stan's scaled, correlated effects.
Pointwise model assignments and population contributions retain observation
coordinates; unrelated equal-length vectors cannot acquire them by shape.
Scalar model constants and scalar data inputs preserve those coordinates through
aliases and chained transforms. The fitted fixture checks their values against
compiled Stan and their pooled summaries against direct draw extraction.

`ragged_draw_trees.jl` checks ragged top-level and in-cell response axes,
group-aggregate likelihood widths, constrained-name permutations, shared
backing, quantiles over unequal chains, and changed-design replay with an
omitted response. It uses deterministic unconstrained points, not a fit.

The files in this directory are standalone scripts, not a `Pkg.test` suite.
`smc_window_identity.jl` checks carried-posterior identity across dictionary
rebuilding, array padding, shared values, copying and serialization, plus
window update/tuning, checkpoint replay and mutation rejection through the
public APIs. `smc_window.jl` additionally checks replay in a fresh process.
`plate_derived_inputs.jl` checks computed arrays read directly by plate cells:
transformed hierarchical predictors, chained assignments, gathers and
concatenations, including a derived array as the plate range. Independent
normalized densities and finite differences agree with native reverse and
compiled Stan; printed RK source and changed-data replay are checked too.
`rk_scalar_plate_locality.jl` checks collected scalar plate outputs through
broadcast observation locations: strict selected-cell queries, ordinary Enzyme
Reverse gradients, independent analytic values, and bounded allocations as
the number of observations grows. The test environment pins RK 0.1.51 /
RKPPL 0.1.53, including scalar-helper type inference and guarded captures.
`rk_field_offset.jl` checks coefficient-one offsets of declared scalar plate
outputs and explicitly gathered helper outputs. It covers both output orders,
reader-first composition, unequal axes, repeated and reordered reads, normalized
density, native reverse and Stan gradients, complete printed-source replay,
immutable inputs, and bounded selected-row allocations. Vector cells with an
accidentally matching flattened length must not establish a scalar row axis.
`laplace_plan_reuse.jl` counts actual row/cell derivative preparations and
compares retained plans with freshly prepared targets, including strict
subsets of row, plate and sum factors. It checks marginal values, gradients,
finite differences, independent workspaces and backend/AD changes.
`laplace_shared_prefix.jl` checks helpers shared by ordinary and sum factors:
exact mixed derivatives, strict local supports, retained subsets, independent
copies and matrix-free covariance contractions. Synthetic scaling fixtures
check that preparation retains only local supports and shared derivative
sweeps allocate approximately in proportion to the number of cells.
`laplace_cell_routes.jl` checks that both routes of a cell with no sums, compact
operators and the factor outer, give the same marginal value and gradient,
including a cell wider than the HVP batch, a retained plan and a copy. It forces
the factor route for one testset by overriding the locality report, and a helper
that hides its rows from both routes is refused by name. A stiff cell whose
population intercept is aliased with an integrated effect keeps the factor route's
gradient equal to the compact route's, whose Cholesky of h never sees the augmented
system (it read -50401 for -2.16 on the rotated form).
`plate_selected_outputs.jl` checks nested selection of shared vector outputs
across plates with different cell counts: data-only and parameter-dependent
outputs, fixed and varying widths, both output orders, reader-first composition,
and lexical binding preservation. Native reverse and compiled Stan match an
independent density and gradient.
`plate_capture_dependencies.jl` checks whole-value captures in composed plates:
late parameter dependencies, cell-local name collisions, scan-local names,
and compiled Stan/native density and gradient agreement with an independent oracle.
`rk_multigroup_replay.jl` checks coordinate multigroup HSGPs across changing
windows, including data-free state serialization, shared weight identities,
new and returning labels, reordered levels, cached binder validation, frozen
prior preprocessing, and dense tensor-field values and native gradients.
`rk_resample_binder.jl` checks a draw of the fit on designs that resample
groupings through `rk_design_binder` (snag `rk-replay-with-r-bf059724`): fewer,
as many and more new levels than the fit, a correlated block shared by two
predictors, a multi-membership block and a data-free fit, against the fit's own
values and closed forms at a draw whose coordinates all differ, and the refusal
of a `gr(g, by = s)` design with other level counts. Run it with
`julia --project=test test/rk_resample_binder.jl`; it needs no Enzyme.
`rk_grouped_anisotropic_hsgp.jl` checks per-axis length scales of grouped
HSGPs: independent, sampled, known and multigroup correlations, default and
explicit floor bounds, fixed scales, term starts, shared readers and frozen
replay. Compiled Stan densities and gradients agree with native Enzyme and
coordinate-wise finite differences.
`plate_vector_parameters.jl` checks declared vector priors read by plate cells:
literal and data-dependent extents, joint-normal and simplex axes, scalar
hyperparameters, standalone parameter axes, replay, and extent validation.
Downstream plates read complete vectors from earlier cells, including outputs
followed by another assignment in their cell.
Compiled Stan and native reverse gradients are compared with independent densities.
`hypot.jl` checks stable elementwise formula norms on RK and Stan: scalar and
vector operands, full densities and native gradients, replay, predictor terms,
offsets, and overflow/underflow controls. It uses the StanBlocks 0.2.12 pin
from `setup_env.jl`.

`nlme_laplace_sums.jl` checks sparse sum-cell integration against exact
Gaussian marginals, conditional covariance and full-model derivatives, plus
automatic selection, marginal optimization, independent copies and scalar
curvature sign changes. Indexed direct predictor reads beside sums retain
four outer coordinates at 16, 64 and 4,096 effects (RK 0.1.39 / RKPPL 0.1.33).
`laplace_factors.jl` checks the compact factor chain
rule and bounded covariance square-root action, and that outer curvature up to
1e10 costs the augmented solve neither its residual nor the selected covariance
blocks. The end-to-end scaling
driver is `research/sparse_laplace_factors/fitting_scaling.jl`.

Sum-cell tests require the fleet SelectedInversion revision recorded in
`setup_env.jl`; registered 0.2.2 does not include its KLU API. The bootstrap
reads that revision from the host mirror, as it does the Enzyme dogfood pin.

Run any one of them with:

```sh
julia --project=test test/adaptive_centering_bridgestan.jl
```

`bounded_lognormal_moments.jl` checks ordinary `mean`/`var` expressions over
a lower-truncated LogNormal and its separate support log probability in scalar
and plate models. It compares compiled Stan and native Enzyme gradients with
high-precision moment references and independent finite differences, including
deep tails, nearly certain survival, and a parameter-dependent bound. A grouped
model checks NLME unit attribution and unit/full-model density and gradient
agreement. The survival
event is an explicit likelihood factor; normalized moments alone do not add it.
The exact Distributions fork and StanBlocks revisions are pinned in
`setup_env.jl` and `Project.toml`; the bootstrap can materialize review
candidates from the fleet mirrors before their publication.

`smc_laplace_matrixfree.jl` checks fixed-target checkpoint replay, owned
operator buffers, rejection of changed numerical policy or retained probe and
reference values, conditional completion and cumulative-data reference reuse.

`smc_laplace.jl` checks outer SMC over all RK-backed embedded-Laplace
implementations against exact Gaussian marginals and completed joint moments,
with scalar quadrature for the generic/structured model's sampled scale,
independent pilots/workspaces, frozen cumulative updates, identity refusals,
non-Gaussian completion, zero integration, interruption, weighted ladder
resampling and fresh-process checkpoint replay. It uses the RKPPL 0.1.57
prepared-target adapter and Enzyme's concurrent-GC repair. Pass testset title
substrings to run a focused subset; run with two Julia threads for the
serial/threaded comparison.

`smc.jl` checks RK-only batch and cumulative-data SMC against analytic
Gaussian posteriors and evidence, deterministic continuation and checkpoint
replay, unchanged caller RNG, frozen training preprocessing, incompatible
groups and mutated-data refusals, and a synthetic deterministic recurrence.
It uses the optional SequentialMonteCarlo and WarmupHMC extensions.

`smc_nonfinite.jl` checks that batch fits and cumulative updates expose RKPPL's
`SMCNonfiniteError` directly, including with Treebars progress, with the exact
failing unconstrained point, compact rendering and serializable diagnostics.
Failed updates preserve the input fit's complete checkpoint, including its RNG.

`smc_tuning.jl` checks `brm_smc_tune` on the RK backend against the authored
correlated Gaussian oracle in `smc_tuning_fixture.jl` (a narrow sum direction
beside a wider difference direction, never supplied as a dense metric):
independent continuous pilots, bridge identities of initial and
cumulative-data schedules, exact continuation, and checkpoint replay across
Julia processes through `smc_tuning_checkpoint_process.jl`. `smc_adaptive.jl`
checks adaptive pilots with portable structured scales and an independently
learned, frozen normalized reference, replayed in fresh processes through
`smc_adaptive_checkpoint_process.jl`. `smc_correlated.jl` checks an explicit
correlated `SMCReference` and mutation `metric` built with
`WarmupHMC.scale_data`. `smc_tuning_frozen.jl` checks that update tuning
reprocesses cumulative data with frozen training preprocessing. They use the
RKPPL 0.1.55 and WarmupHMC 0.2.10 pins from `setup_env.jl`.

`smc_hierarchy_spread.jl` checks tuned SMC on a two-group hierarchy
`mu ~ 1 + (1 | g)` against an exact grid oracle of the population intercept
and log scale, with the group effects mixed analytically: the continuous pilot
and the adaptive pilot with the standard-normal reference reproduce the
oracle's spread. With RKPPL 0.1.47, the learned-reference regression checks
the retained-moment repair for ReactiveKernels snag `learned-smc-refe-cc5677a1`,
the forwarded `reference_inflation` control and reference/cloud spread diagnostics.
It retains the known population-intercept under-dispersion assertion: the
repaired-reference seed-101 run has SD 0.786 of the oracle, below the controls'
unchanged 0.8 lower bound. The reference repair is not an accuracy acceptance.

`smc_compare.jl` checks `brm_smc_runs` and `brm_smc_compare`: independent
bridges on an analytic Normal posterior and on the tuned correlated Gaussian
oracle agree with the exact means, SDs and evidence; the comparison is
deterministic for fixed seeds; fits of different data, of a different
previous posterior, or incomplete fits are refused by name.

`rk_captured_source.jl` checks that `rkppl_model(rk)` preserves submodel and
callable-provider densities, shares captured source across compatible replays,
and keeps frozen preprocessing in the binding, including training-free replay.

`filter.jl` checks the RK filtering adapter against deterministic recurrence
scores and a Kalman recursion, calibrated ensemble warm-up, independent group
updates, cooperative stream consumption, checkpoint recovery and bounded
retention. `filter_models.jl` contains the shared public synthetic models.
`filter_checkpoint_process.jl` is invoked in separate processes by that suite.
`filter_calibration.jl` additionally checks an explicitly drifting parameter
against analytic filtering, narrow-ensemble predictive calibration and the
completed-SMC-to-filter interface.
Its testsets accept the usual substring filters as trailing arguments.

`streaming_oracles_check.jl` validates the independent Gaussian and Kalman
references in `streaming_oracles.jl`. Sequential Gaussian windows must equal
the batch posterior and evidence, counting the prior once. The joint static
parameter/AR(1) oracle is checked against numerical integration of a separate
conditional scalar filter, including missing observations. These checks need
only Julia's standard libraries; they do not exercise the SMC/filter adapters.

`streaming_preprocessing.jl` checks a serialized fitted-state replay after
poisoning the training arrays, comparing its window likelihood with the
analytic density under the original scaling anchors. It also checks that the
serialized preprocessing size stays bounded as the training rows grow.

`filter_smc.jl` checks selected-static calibration from a latent-history fit,
owned RNGs and inputs, named identity refusals, warm state-only boundary
transfer and exact continuation of parameter learning. `filter_smc_process.jl`
runs as `write <path>` then `read <path>` in separate Julia processes; its
reader constructs only a one-slice model, never the old fit or training data.
`filter_smc_scaling.jl` measures fixed-work allocation and checkpoint retention
at 12 and 120 elapsed steps. These use `filter_smc_fixture.jl`.
`filter_smc_oracle.jl` compares the learned static parameter and predictive
normalizer with an independent joint Kalman recursion, using three seeds,
missing observations, an independent frozen pilot and forced replenishment.
It stays within a single window to separate particle accuracy from the
documented Gaussian/state-independence approximation at boundaries.

`smc_window_pilot.jl` tests correlated sequential windows against both their
actual carried Gaussian oracle and the full batch posterior. Independent
pilots tune kernels; completed clouds supply the next reference via `basis=:fit`.
Ownership, schedule identity, `reference_inflation` forwarding and exact
checkpoint continuation are also checked.

`smc_window_scaling.jl` checks bounded carry/checkpoint storage over 128 windows
and reports warmed initialization time at fixed particle/window dimensions.
Shared-host wall times are reported, not asserted.

`smc_window.jl` checks carried posterior summaries on new, nonoverlapping
windows: sequential Gaussian posteriors against a batch fit, unchanged caller
RNG and carried inputs, named source/data/preprocessing refusals, and bounded
summary size. `smc_window_process.jl` checks exact checkpoint continuation in
a fresh process that never constructs the training arrays.

`smc_window_selective.jl` exercises opt-in `coordinates=:auto` windows: old
priors are excluded, fresh priors include their old parents and transform
Jacobians, and omitted coordinates retain their carried conditional law.
Independent Gaussian and truncated-Normal references check posterior moments
and predictive normalizers. Storage checks compare 32 and 256 particles in a
10,002-coordinate growing frame, and lazy recovery checks a correlated omitted
coordinate without adding full-width particles. Its testsets accept substring
filters as trailing arguments. Repeated carries check that summaries stay
bounded after growth. `smc_window_selective_api.jl` checks owned coordinate maps,
compact pilot tuning, schedule identity and exact tuned continuation.
`smc_window_selective_process.jl` checks fresh-process checkpoint continuation,
coordinate maps, conditional recovery and the next carry while
preserving fitted standardization without the original training rows.

`rk_vector_extent.jl` checks data-driven declared-vector extents through emitted
RK source, fitted replay and design binding. It covers integral expressions,
growing versus fixed replay, shrinking refusal and an unrelated scalar input
whose value happens to equal the extent.

`test/Project.toml` is a superset of the root project: it carries every package
that any `test/*.jl` loads, so no test file fails at its own `using` line the way
three of them do under `--project=.` (see below).

One deliberate absence, and it is a standing rule rather than a gap:
**`ForwardDiff` is not in this environment and is not coming back.** BRM
differentiates with Enzyme only — every gradient in this suite goes through
`AutoEnzyme()` via `DifferentiationInterface`. No test file loads ForwardDiff,
so there is nothing here to work around; do not add it back to make a new
gradient site easier.

`plate_completed_values.jl` checks direct and gathered plate reads of
ordinary and total-constrained completions, including transformed gathers
and forward assignment dependencies. Compiled Stan densities agree with an
independent full-density calculation; RK values and reverse gradients agree
with Stan under centered and noncentered completion coordinates. The script
also checks replay, printed RK source, predictor gathers, row-axis validation
and preservation of in-cell response coverage errors. Run it with the test
environment above; its `@stestset` blocks accept substring filters.

`infinite_censoring_bounds.jl` checks exact `-Inf` and `Inf` response bounds
through SBBRMI and BridgeStan: lower, upper and two-sided Normal and Student-t
censoring, alongside interval-censored LogNormal evidence. Finite and infinite
bound scenarios compare normalized densities and gradients against independent
CDF, CCDF and density calculations. Run it with
`julia --project=. test/infinite_censoring_bounds.jl` or the test environment.

`kernel_ragged_broadcast.jl` checks Stan source emission and stanc acceptance
for modeled secondary-axis arguments in plain and nested dotted calls. It
covers interleaved rows, shared random effects, empty groups, raw-data controls
and preservation of field and keyword names. Run it with
`julia --project=. test/kernel_ragged_broadcast.jl` or the test environment above.

`plate_output_reads.jl` checks that the model reads every indexed output of an
`@plate for` cell by its name, not only the cell's last: a model-scope
observation of a non-last output (written after or above the cell), both
outputs of one cell read by the model, another cell reading it, and per-cell
reads (`sum(tot)`, `tot[2]`) of scalar and vector cells. Each has the
BridgeStan and RK log density and gradient of the spelling with that output
collected last (the RK gradient of vector-cell reads is `@test_broken`, todo
`1svm4l1`). It also covers the omitted response, `reprocess`, and
the refusal of a name two cells write. Snag `model-scope-read-3649ae8a`.

`plate_output_helpers.jl` checks model-scope whole-value helpers receiving
matrix, vector and higher-dimensional plate cells. Independent normalized
density and finite-difference gradient oracles cover positional and keyword
arguments, likelihood arguments, non-last outputs, reader-first order,
data-only cells, replay with empty cells and changed row counts, and ordinary
Reverse AD. Complete printed-source replay and flattened rowwise observation
controls guard the surrounding contracts, including `addprop` over a plate
output, assigned or in the law, with a shared or per-row additive scale. Run with
`julia --project=test test/plate_output_helpers.jl`; testset substring filters
are supported.

`plate_grouped_outputs.jl` checks grouped reads of both outputs from another
plate through `rows = groups(group => labels)` and `value[rows[k]]`.
Compiled Stan and native RK densities and gradients match an independent
oracle, including interleaved rows, an empty group, either output order, a
reader written before its producer, and replay onto changed groups and row
counts. Its testsets accept substring filters.

`walk_forecast.jl` checks forecasts of the walk terms `rw(time)` and
`cdar(step; by=g, cor=C)`: a replay whose rows reach past the fitted grid keeps
the fit's unconstrained coordinates and draws the appended steps' innovations
in generated quantities, so each level continues from its fitted state (the
exact recurrence per draw, `N(0, I)` innovations, the one-step covariance
`sigma^2 (1 - rho^2) C`), also through `resample_groups`; a step inside the
fitted grid is refused, and `freeze_constants=false` samples every step. Set
`BRM_WALK_FORECAST_RUNTIME=0` to skip the compiled-model sets. Snag
`per-group-statio-059ae6f6`. It also checks `resample_groups` naming a `cdar`
term's `by` column: the levels the replay rows name, unseen ones included, are
redrawn in generated quantities (the exact recurrence per draw, `N(0, I)`
innovations, the stationary marginal SD), given the fitted levels the rows do
not name for a correlated `C` (the conditional factors and the first state's
conditional mean and SD); the fit's coordinates stay, `C` over the union of
levels is read from the replay data and checked against the fit, and it
composes with a resampled random effect. Snag `cdar-replay-new-57df8982`.

`column_type_size.jl` checks that a column's type does not spell its
expression tree: a 60-deep chain of model-scope assignments gives `ExprColumn`
/ `NamedColumn` types as small as a 3-deep one, and building, planning and
emitting it compiles no signature holding a column or prepared-expression type
longer than 4 KB. Types that spelled the tree made a 479-statement model's
build and RK planning compile for about ten minutes. Snag
`rkbrmi-start-up-8ed0a306`.

`changelog.jl` checks that `CHANGELOG.md`'s newest heading is the
`Project.toml` version (`dev` §7.5).

`mi_group_totals.jl` checks `mi(x)::shares(total, rows)`, a completion whose
missing entries have known group sums: the plan's residuals and its refusals;
SBBRMI's density against an independent oracle (LogNormal, Gamma, the rows in
place, a completion nothing reads); RKBRMI against SBBRMI through the coordinate
transport, value and Enzyme gradient (the Gamma gradient is `@test_broken` on
registered Enzyme, repaired on Enzyme dogfood from `4c73c40`, Enzyme snag
`rkbrmi-gradient-519ec090`); frozen replay; the descriptor, the
description, `brm_start`; and refusals of real-valued laws.
Snag `mi-completion-wi-54222432`. It also checks
`RKBRMI(brmi; completions = :noncentered)` on such a completion: the same
coordinates and density as the centred lowering, its group terms in
`:pointwise`, SBBRMI agreement through the transport (LogNormal, an unread
completion, Gamma's value), and a Gamma law's free entries through their
support transform and density remainder (value, gradient and transport against
Stan). Snags `nlme-view-shares-f1cfaa71`, `brm-nlme-loglike-7c4b209b`.

`mi_lower_totals.jl` checks lower support declared by `truncated` or
`restricted` on shared completions, ordinary imputation and cell shares.
It compares the full Stan density, RK native Enzyme gradients and coordinate
transport with independent LogNormal/Normal density and Jacobian oracles;
covers scalar, row and live bounds, free rows, singleton boundaries, complete
groups, invalid budgets, source/frozen replay, output descriptors and starts.
The emitted group map is also checked for linear coordinate reads, allocation
and layout storage as group size doubles. Cell checks include Exponential
laws whose vector of lower bounds supplies their dimension.
Free-row checks require strict cell locality and compare native gradients with
analytic scores for default Exponential, scaled Exponential and Normal laws.
Run it directly in the test environment. `nlme_lower_shares.jl` runs in the
NLMEEstimation scratch environment below and checks bounded groups' fixed
unit variance, lockstep/unit densities and gradients, `check_protocol` and
per-cell coordinate locality. The RK 0.1.38 / RKPPL 0.1.32 test pin preserves
grouped-design shape facts (`group-plate-desi-9ffa5d79`) and binds data-computed
entry axes and tuple-geometry metadata before shape inference
(`prepare-a-strict-13a9e1b8`, `prepare-one-resp-b1f58f91`). Enzyme dogfood
`0e251cfb` with its declared libEnzyme `782c9d95` repairs the native HVP
(`custom-geometry-8640cc9d`). Both locality and `check_protocol` remain required
acceptance in the full consumer fixture. Snag `atlas-shifted-co-e72a3041`.

`rk_mi_bound_types.jl` checks that RKBRMI binds every `mi()` column as
concretely typed values (`<x>_raw`) and a `Vector{Bool}` presence mask
(`<x>_present`), never as a column of `Missing` unions: for a completed
covariate, an unread one, an `mi(y)` response, a completion with group totals
and a complete column, on the fit, on frozen `reprocess` replays and through
`rk_design_binder`, with the values equal to the data at the observed rows and
the mask equal to the data's presence. Snag `mi-raw-columns-r-2b85d8f9`.

`noncentered_completion_transport.jl` checks `brm_coordinate_transport` on
completions `RKBRMI(brmi; completions = :noncentered)` samples as standardized
draws: RK's `<column>.z_mis` against SBBRMI's centred `<column>_y_mis` through
`s = location + scale * z`, with the law's location and scale at the point.
Both point maps, the log density with the log-Jacobian `sum(log, scale)`, the
native Enzyme gradient against the Stan gradient pulled back through the
location and scale, and the physical check, for a law of data columns, a law
of parameters, a LogNormal law (Stan's entries are the logs) and a law reading
another completion. Snag `coordinate-trans-23993804`; the free entries of
`mi(x)::shares(total, rows)` are covered in `mi_group_totals.jl`, and NLME
unit draws reaching Stan in `nlme_joint_start.jl`.

`nlme_shares_units.jl` (test environment) reads such a completion with region
units: the centred reading refused, the non-centered grouped coordinates as
unit draws of the region owning each group (free entries beside them), the
group terms attributed to it, a group spanning two regions refused, lockstep
and unit-problem densities against the full RK density, and per-unit Laplace
and SVI. Its panel lives in `nlme_shares_fixture.jl`.
`nlme_shares_units_fit.jl` runs NLMEEstimation on that panel: the grouped
coordinates' family in `NLMEEstimation.fixed_omega`, `check_protocol` (with
the η Hessian blocks) and a short SAEM fit. Run it in the NLMEEstimation
scratch environment described below. Snag `nlme-view-shares-f1cfaa71`.

`nlme_completion_laws.jl` (test environment) checks `mi(...)` completions
whose law reads the units' draws on the NLME paths: `brm_nlme_view` refuses
their centred reading by name; the fits' `@brm` entry
(`BRM._brm_nlme_lowered`) non-centers exactly those completions, any
continuous univariate law (an interval Beta law here) keeping the centred
coordinates and density; one-unit and lockstep values, gradients and η Hessian
blocks agree with each other and with central differences, on the reported
panel (a region with no observed entry) among others; a completion whose law
reads no unit's draws stays centred and is accepted; and a regression-plan
`mi(y)` response lowers the same under every spelling. Snag
`brm-nlme-loglike-7c4b209b`.

`nlme_completion_axes.jl` checks gathered completion ownership on a shorter
response and on a reordered response with the same length as the subject
column, and ordinary completions with repeated rows on a longer response.
Shares, free entries, observed entries and downstream responses follow the
gathered subjects; groups spanning subjects are refused. Lower-bounded free entries
are unit draws with their exact density remainder. Lockstep and unit
densities and native gradients are compared with the full RK model.
Run it in the test environment. `mi_lower_totals.jl` also covers the bounded
coordinate transport, density, gradients, replay and per-row locality.
`nlme_completion_axes_fit.jl` uses the same panel in the NLMEEstimation
scratch environment below: both completion families stay fixed at unit
variance through `check_protocol` (including native Hessians) and a short
SAEM fit.

`nlme_offaxis_observations.jl` (test environment) checks `brm_nlme_model`'s
attribution of observations on rows other than the subject column's, on a
regional panel with a fragment of its own (`nlme_offaxis_fixture.jl`): its
published counts read through an index (`exp(log_f[f_obs])`), its suppressed
counts' residuals as a plate over its own groups, and a count that reads no
random effect. Each density's measured region is checked against the region
of its rows in the data, and the per-region likelihoods against the
pointwise densities summed by those regions; lockstep, per-subject cells and
unit problems against the full RK density; a residual whose rows lie in two
regions is refused, naming them; and per-unit Laplace and SVI run.
`nlme_offaxis_observations_fit.jl` runs `check_protocol` and a short SAEM fit
on that panel in the NLMEEstimation scratch environment described below.
Snag `nlme-model-obser-dd665aca`.

`plate_subset_response_sums.jl` checks a response on some of a predictor's
rows, the others observed only through group sums (suppressed cells of a count
table): SBBRMI against an independent density for Poisson and NB2 cells, a
frozen replay, and RKBRMI against SBBRMI. Snag `mi-completion-wi-54222432`.

`forward_reads.jl` checks that a statement may read a name declared below it:
a formula term computed from a parameter (with an `effect(...)` prior on its
coefficient), from an `mi()` completion or from data alone, a grouping, a chain
of assignments and the completed `mi()` column itself, each with the
declaration moved below its readers, give the same Stan program and the same
RK density and gradient as in source order; `merge` composes fragments that
read each other in either order, builders and built `BRMI`s alike; and a
fragment that only reads a data column no longer replaces another fragment's
likelihood of it. Snag `formula-covariat-5673c470`.

`rk_inline_response_location.jl` checks a response whose location is computed
inline from model values (`y ~ NegativeBinomial2(exp(dev) * (…), phi)`) against
the same location assigned first: SBBRMI gives both spellings one density, and
RKBRMI agrees with SBBRMI (value and Enzyme gradient) on the assigned spelling
and, for a Normal location and scale and a Bernoulli probability, on the inline
one. The inline Poisson and NegativeBinomial2 cases are `@test_broken` until
the test environment's ReactiveKernels pin carries an RKPPL that lowers an
inline count rate (ReactiveKernelsPPL 0.1.2, RK main `f9a5c9c41`, ReactiveKernels
snag `rkppl-logistic-a-b6e04821`). Snag `rkbrmi-on-a-resp-017c6f44`.

`plate_gathered_predictor.jl` checks a predictor on its own rows that reads
another part's values through model-scope gathers (`c2 = c[pos2]` of an `mi()`
completion, `z2 = z[pos2]` where `z` reads part-1 data columns), read by a
`@plate` over a grouping of those rows: the gather is sized by its index, both
backends equal the twin with the part-1 columns gathered by hand (RK centered
and non-centered completions), RK equals Stan, and a part-1 column read directly
is still refused. Snag `rk-backend-predi-9a0c2121`.

`plate_grouping.jl` checks model-scope `rows = groups(group => labels)`
bindings that `@plate for` cells read as `x[rows[i]]`: `groups` itself,
identical Stan source, descriptor outputs and RK log density to the legacy
`kernel(ragged(x, group, labels), …) do` spelling, the omitted-response path,
the boundary refusals (including the retired `ragged(...)` statement and
in-cell read), and a per-subject linear predictor without a random effect read
as `x[i]` (BridgeStan and RK density parity with the old one-element ragged
gather).

`plate_grouped_response.jl` checks a flat response observed in the cell through
a binding, `y[obs[i]] ~ …`: density and gradient equal to the response grouped
by hand, bounds read through the binding (censored, scalar, truncated),
invalid vectors, the refusal of the retired formula-level `ragged(y, group) ~ …`,
and replay/resample regathering a grouped response and its bound.

`dirichlet_multinomial.jl` checks joint count vectors in plate cells:
shared concentration, ragged category vectors, model-derived probabilities,
normalized densities, Enzyme gradients, complete printed-source replay,
one pointwise contribution per vector, and NLME subject likelihoods,
gradients and Hessians. A multivariate Normal case checks the common whole-event path.
The support checks require `-Inf` for mismatched totals, negative counts and
noninteger counts, both pointwise and in the model density, without changing
caller data. The test bootstrap explicitly selects the public Distributions
fix `83cec5e3a8a35167d54f19da9dc5afb54f59273e`; the registered 0.25.132
release lacks that check, and an RKPPL version floor does not select the fix.
SBBRMI uses StanBlocks 0.2.8's native family. Compiled BridgeStan checks
compare normalized densities, gradients and pointwise units against RK;
ragged predictive vectors must contain nonnegative integer counts summing
to each declared total.

`plate_matrix_data.jl` checks per-cell matrices in a `@plate for` cell:
- per-cell matrix data, a pre-grouped `Vector{Matrix}` read as `X[i]`: RKBRMI
  and SBBRMI (BridgeStan) density and gradient against an independent oracle
  and each other, the descriptor's single required `X` input, and the
  flattened `reshape(Xf[i], n[i], k)` twin equal to it on both backends;
- a per-cell matrix bound to a local (a covariance whose size follows its
  cell): SBBRMI density and gradient equal to the inline spelling and an
  oracle, and the local addressed by `brm_output(d, :ccov)` with the right
  segments, with and without an empty cell.

Both need StanBlocks `c9504e99` or later (snags `ragged-matrix-da-163255e9`
and `binding-a-per-ce-e16efa8d`).

`plate_level_axis_keywords.jl` checks a predictor read through a `@plate`
grouping whose term keeps data off its rows: the levels' coordinates in
`cor = multigroup((x, y))`, a known `cor = C` on `hsgp`, on `gr` and on
`cdar`, and `car(area; edges = (from, to))`. Each plate model's RKBRMI and
SBBRMI (BridgeStan) density and gradient equal its plate-free twin's (the
`cdar` cases on SBBRMI only, since RKBRMI does not lower `cdar` yet; there
RKBRMI is only checked to pass the row check), `cdar` summed per step through a
`groups(t => steps)` binding equals the same `C` given by `constant(...)`, the term as a
predictor of its own beside a random-effect predictor equals the term inside
that predictor (on SBBRMI with `total_groups=()`, since under
`total_groups=:auto` only the first keeps an exact-total block; that program
is checked to evaluate), an RK
replay keeps the density, and a covariate with one value per level is still
refused as a row column (snags `rkbrmi-on-a-mode-0bd31c04`,
`cdar-cor-matrix-c8278be6`). Set
`BRM_PLATE_LEVEL_AXIS_RUNTIME=0` to skip the compiled-model sets.

`plate_mvnormal_observation.jl` checks a cell observing its slice jointly,
`y[rows[i]] ~ MvNormal(location, covariance)`, with the covariance computed in
the cell. The model is an aggregate benchmark: unit-level means and variances
summed to district totals, a grand total, and the observed district totals
given it. It checks:

- the SBBRMI density and gradient against a plain-Julia joint `MvNormal`,
  including a cell with an empty slice (log density 0);
- `Diagonal(v)` and `diagm(v)` both lowering to `diag_matrix(v)`;
- a covariance bound to a cell local against the inline spelling;
- the unit-level spelling against district sums precomputed as data;
- the RKBRMI density and native Enzyme gradient against the same Stan program.

`fixed_ranef_scales.jl` checks random-effect scales and correlations held at
known values, `sd(lp, ID) = value` and `cor(:, ID) = R`. It covers:

- parsing, `priors_of`/`merge` composition and the refusals;
- SBBRMI emission and stanc acceptance for mixed, all-known, correlation-only,
  one-margin, Student-t, `cv_groups` and `@plate` models;
- BridgeStan densities: each known model equals its sampled twin evaluated at
  the known values, up to exactly the twin's prior density of those values;
- RK/Stan coordinate transport with density and gradient parity;
- `population_draws`, `reprocess(...; resample_groups=...)`, an
  omitted-response prior program, descriptors, descriptions, the NLME refusal,
  and the adaptive-centering description of known-value blocks, Student-t
  included.

Set `BRM_FIXED_RANEF_RUNTIME=0` to skip the compiled-model sets.

`hsgp_intercept_projection.jl` checks `hsgp(x...; orthogonal_to=:intercept)`:
- every basis column centered over the term's rows, for two raw axes
  (isotropic, anisotropic, an explicit `domain=`) and over the masked rows of
  a `rows=` term, with the fitted projection recorded;
- frozen SBBRMI replay evaluating the training projection on a subset of rows
  (`:linear` too), and `freeze_constants=false` re-fitting it;
- `by=g` projected within each level (decision 15hn91t), with and without
  `cor=Omega`, with a per-level amplitude (`log(sd(hsgp(...))) ~ 1 + (1 | g)`),
  with `rows=`, and `:linear`: each group's basis is its raw
  basis minus that group's column means (and orthogonal to its `1` and `x`),
  one projection per level is recorded, a frozen replay over a subset of the
  groups keeps each level's projection, and a fresh replay re-fits it;
- a model-derived axis centered in Stan (`brm_hsgp_center_columns`);
- the refusals (`anchor=`, `:linear` over two axes, an unknown value, a
  projection fitted on no rows, a group with no row in `rows=`) and the
  description of the projection (per group for `by=`);
- RKBRMI density and gradient equal to SBBRMI (BridgeStan) through the
  coordinate transport, including a field that multiplies a positive weight
  and the grouped models above.

Set `BRM_HSGP_PROJECTION_RUNTIME=0` to skip the compiled parity sets.

`level_correlation.jl` checks correlated levels of a grouping factor:
`gr(g; cor=Omega)` and `hsgp(x...; by=g, cor=Omega)`, with a declared
`Omega ~ LKJCholesky(K, eta)` or a known data matrix. It covers:

- SBBRMI emission and stanc acceptance;
- the descriptor, the generative plan and the description;
- refusals of wrong shapes;
- the SB density against a hand-written raw-Stan intrinsic-coregionalisation
  program;
- RK density and gradient equal to SB, through `brm_coordinate_transport`,
  the grouped HSGP included;
- `brm_ranef_draws`' `level_cor`, and replay.

Combinations not implemented yet are `@test_broken`. Set
`BRM_LEVEL_CORRELATION_RUNTIME=0` to skip the compiled-model sets.

`grouped_hsgp_transport_readers.jl` checks that an RKBRMI fit of a grouped
multigroup HSGP, `hsgp(day; by = site, cor = multigroup((site_x, site_y); k))`,
is read by the multigroup readers through `brm_coordinate_transport`: a daily
panel with `|ID|` random effects, a per-site residual SD and AR(1) residuals in
a `groups(site; order = day)` plate. The transport pairs every coordinate
(RK density, gradient and physical values equal to BridgeStan's), and
`brm_svi_fit` draws mapped to Stan's constrained names give
`brm_multigroup_coherence` ranges and `brm_multigroup_field` values equal to
RK's own length scale, spatial scale and term value. Snag
`read-brm-svi-fit-bb7a9be0`.

`level_correlation_hyper.jl` checks per-level amplitudes and length scales
(the hyper-predictors `log(sd(hsgp(x))) ~ 1 + (1 | g)` and
`log(length_scale(hsgp(x))) ~ 1 + (1 | g)`) on a grouped HSGP with correlated
levels, `cor=Omega`, `cor=multigroup(Omega, tau)` or
`cor=multigroup((x, y), lambda; k)`: SBBRMI emission and stanc acceptance,
one- and two-axis terms, the refused per-level length scale under either
`multigroup`, descriptor, description and replay, the SB density against
hand-written raw-Stan programs, and RK density and gradient equal to SB. The
same `BRM_LEVEL_CORRELATION_RUNTIME=0` skips its compiled-model sets.

`sd_prior_input_names.jl` checks that a scale prior's caller scalar may share a
name with a binding of the submodel the prior is merged into, such as `tau` in
`sd(:, ID) ~ Exponential(tau)`. It covers Gaussian, general-family,
per-margin, centered, `cv_groups` and Student-t blocks, smoothing scales, and
the completion submodel of a split block. Each spelling must emit exactly the
Stan program of a twin whose scalar is called `lam`; the twin must pass stanc.

`model_family_composition.jl` checks composing a family of related models
without string templating:

- `optional(<address>)` priors: every address class is dropped when the model
  lacks it, and the result is byte-identical to the plain twin on SBBRMI and RK;
- the typed builder: `brm_columns(builder)` equals the descriptor's columns, and
  `merge(builders...)`, `Expr` composition and `show(builder)` work;
- source views: `show(::BRMI)` prints the authored statements, and the printed
  source re-parses to the same Stan program;
- `String` categorical columns on both backends, with stanc acceptance and a
  BridgeStan density matching the integer-coded twin;
- order-free structural `merge`, and the refused `effect(<unknown predictor>, :)`.

`builder_without_eval.jl` checks `brm(source; mod)`, the builder of a runtime
`Expr` that evaluates no code, and `merge` of builders, which goes through it:

- nothing is defined (the world counter does not move), against the
  `Core.eval(mod, _brm(expr))` control, which defines a method per model;
- the same model as the compiled builder: equal BRMIs, byte-identical Stan and
  data, family members and split fragments, and a sweep of every `@brm` block
  of a list of test files (`BRM_BUILDER_SWEEP=all` sweeps every file,
  `BRM_BUILDER_SWEEP=a.jl,b.jl` names files);
- the Julia of `constant(expr)` evaluates as compiled code would, and
  definitions and mutation are refused;
- fragments a package defines compose in another package's precompile workload,
  where the evaluating route fails with "closed module".

`kernel_population_lp.jl` checks per-subject linear predictors without a random
effect (a typical value plus covariates) read as `x[i]`, beside a random-effect
predictor and in a no-random-effects panel, against an independent density
oracle with finite-difference RK gradients and matched BridgeStan coordinates.
It also covers a panel whose cell slices only such predictors and the refusal of
one declared on another row axis. A predictor whose only covariate is a string
categorical (`Vector{String}` or a string-leveled `CategoricalVector`) needs no
zero `offset(...)` witness: its Stan and RK densities equal the witnessed
twin's on both spellings.

`string_categorical_coordinates.jl` checks `brm_population_effect_coordinates`
for a predictor whose categorical column is a plain `Vector{String}`, in a flat
and an `@plate` model. `Intercept` and the column address resolve to the same
result shape as a `CategoricalVector` column. A `CategoricalVector` point
carried to the String model through the resolved coordinates gives the same
pointwise log likelihood, and the same point with two levels swapped does not.
It also covers exact totals, cell-mean coding, and `brm_start` against the RK
start.

`zero_element_coordinates.jl` checks `brm_output_coordinates` on carriers with
no elements: a one-level treatment factor's contrast vector and a one-level
ordinal outcome's thresholds return `Int[]` and carry `dims == (0,)`, every
unconstrained BridgeStan name has exactly one sampled owner, and a carrier
that owns elements but is absent from the names, or whose `dims` are
unresolved, still fails closed.

`output_predictors.jl` checks `BRMOutput.predictors`, `term` and the
population-level `component`s: every term family, a column used by two
predictors, a shared `|ID|` block listing both predictors, R2D2 and horseshoe
priors, exact totals, S2Z and stratified blocks, all found through
`brm_outputs(d; predictor=, term=, component=)`. Its compiled half resolves
every address back to its own output through
`brm_population_effect_coordinates` and `brm_term_coordinates`, and checks that
grouping by predictor reaches every sampled coordinate. Set
`BRM_OUTPUT_PREDICTORS_RUNTIME=0` to skip the compiled half.

`descriptor_trace_reuse.jl` checks that `ranef_blocks(d)`,
`brm_ranef_sd_coordinates` and `brm_ranef_draws` read the descriptor's own
traced program: a copy of the descriptor whose plan holds no traceable program
answers each query identically, in a fitted and in a prior program, while
`ranef_blocks` on that plan fails.

`transform_constants.jl` checks `brm_transform_constants`, the fitted
`center`/`zscale`/`standardize` constants. On SBBRMI each record reproduces the
model's transformed Stan data bit for bit, follows frozen and refitted
`reprocess`, and reports an `mi(...)` column's observed-only fit under the
coefficient's name. On RKBRMI the records equal the graph's own transformed
values, a frozen replay keeps them, and an `&` operand is refused by name.

`plate_index_into.jl` checks `indexin(idx[rows[i]], target[i])`, an index
vector into another grouping's entries: the per-subject positions equal
hand-rebased pre-grouped columns on BridgeStan and RK,
`something.(indexin(…), 0)` keeps `0` as "no index" while plain `indexin`
refuses it, `reprocess` rebases new data again, and an index that leaves its
subject or the target vector is refused, as are bindings with different labels
or an `order`.

`dose_axis_operation_gather.jl` checks a predictor read only at the doses of an
operation stream, declared over the dose vectors and placed onto the subject's
operations in the cell, `vcat(0.0, log_F[doses[i]])` read at
`something.(indexin(op_dose[events[i]], doses[i]), 0) .+ 1`. Its BridgeStan and
RK density and gradient equal the same model on the operation stream with its
read rows filled, with a subject that has no doses. A frozen `reprocess` onto a
new schedule gives the same density on RK and the same HSGP basis rows at the
doses on Stan, and both backends replay onto new subjects. Both models have
complete scientific descriptions.

`plate_empty_axis.jl` checks a `groups(group => labels)` binding over an empty
`group` vector: a formula LP, a data vector and an observed response (against a
cell expression or a collected output) each give every cell an empty slice. Each model's BridgeStan and
RK density and gradient equal a twin that never reads the empty axis.
`reprocess` moves the same Stan program onto data with rows and back,
recomputing the row count of a declared axis that sizes an intercept-only
predictor, and a missing group label is still refused. It also checks a plate
with no cells at all, over `groups(g)` or `groups(g => cells)` with an empty
column: each backend's density and gradient equal an independent formula for the
priors, with the prior scale from data or written as a literal, and SBBRMI emits
the same program as with rows and replays between the two in both directions.

`hsgp_empty_axis_gather.jl` checks an HSGP with an explicit `domain=` (and a
periodic one) over an axis with no rows, gathered per subject through
`ragged(drift, d_subject)` over an empty group: each model's BridgeStan and RK
density and gradient equal a twin that declares the same term but never reads
it, and `reprocess` moves the Stan program onto data with rows.

`hsgp_constant_axis.jl` checks that a fresh HSGP fit refuses a constant axis at
every row count, including those whose computed mean rounds an ulp away from
the axis value, on the shared fit, a fresh
`SBBRMI` build and a `freeze_constants=false` reprocess. A frozen replay keeps
the fitted interval on that axis, and an explicit `domain=` fits it fresh.

`plate_ragged_omitted_response.jl` checks a censored ragged observation inside
`@plate` with its data column bound, omitted, and reprocessed. It requires the
StanBlocks revision pinned by `setup_env.jl` and checks the public flat
predictive segments, bound pointwise likelihood, and refusal of a missing
non-response ragged input.

`s2z_omitted_response.jl` checks `s2z_groups` when a response is omitted. A
predictor that no observed likelihood reaches keeps its block conventional, in
generated quantities. It covers an intercept-only predictor, a grouping that a
reached predictor shares, and an omitted `@plate` response. Each program
matches its conventional or renamed-grouping twin, and the sampled BridgeStan
density equals that of `prune_responses` of the same model.

`omitted_plate_observations.jl` checks the grouped `pk_conc[i]` spelling with
an omitted censored response and a joint QT likelihood. It checks the
predictive descriptor and a finite generated draw above the censoring floor.

`ragged_rowwise_arguments.jl` checks that SBBRMI evaluates a row-wise helper
(`addprop`) or elementwise arithmetic over a ragged plate or kernel result once
per subject when it is observed outside the cell. The observation never reads
the helper over the whole flattened result. It checks BridgeStan density and
gradient parity with the in-cell spelling. For a response observed in the cell
through a binding it also checks the omitted response, its replay, and the
predictive segments. For the
model-scope form over pre-grouped data (`y ~ censored(Normal(loc, addprop(loc,
…)); lower=lloq)` with `y` omitted, and its uncensored twin) it checks the same
observation statement, an argument plate over the plate result's own cells,
predictive segments before and after `reprocess`, a finite BridgeStan draw
above the censoring floor, and that a bound `qt` beside the omitted `y` has
exactly the `qt`-only model's coordinates and density.

`omitted_censored_response.jl` checks a flat `y ~ censored(Normal(mu, sigma);
lower=lloq)` with `y` omitted: the warning, the observation role, the
`y_gen`-only output, the bound twin's predictive law, BridgeStan draws clamped
at `lloq`, and a fitted response beside it whose density equals the
uncensored twin's.

`omitted_discrete_response.jl` checks omitted responses with a discrete law.
A `BetaBinomial2` and a `Binomial` count with data trials and a `logistic`
mean give a `fixed_param` program drawing the fitted twin's law, within the
trials; an omitted literal `BetaBinomial2` matches its mean/precision pmf; a
fitted `Binomial` beside an omitted `BetaBinomial2` has the binomial-only
density. Products, `logistic`, `truncated` and the `Bernoulli`-type families
draw their bound twin's law, a continuous omitted response spells
`logistic` as `inv_logit`, and a `@plate` cell's `BetaBinomial2` has the
density of its `beta_binomial` spelling and lowers omitted.

`omitted_interval_response.jl` checks response-omitted interval observations:
zero-dimensional joint prior draws, uncoarsened base-law moments, predictive
twins, retained row reads and scalar/dense/ragged bound shapes. It also checks
the observed interval mass and gradient, a fitted response beside an omitted
interval, bound validation, and the sole-term interval predictor spelling.
Requires StanBlocks 0.2.21; the bootstrap pins its published omission fix.

`truncated_low_mass_draws.jl` draws omitted `truncated` responses whose window
holds almost none of the base law's mass: a two-sided `NegativeBinomial2`
count class in [1, 9] and [250, 1e7] around a mean of 70, its one-sided tails,
and a `Normal` above 6 sd. Every draw lies in its window and follows the exact
truncated law (pmf and mean from Distributions). It needs StanBlocks 0.2.6 or
later, whose `truncated` draws invert the base CDF on the window.

`adaptive_multiaxis_hsgp_centering.jl` checks online adaptive centering of
two-axis squared-exponential HSGPs: isotropic (one length scale every axis
reads) and `iso=false`, ungrouped and grouped, beside a shared random-effect
block. Each cell's spectral scale is pinned against BridgeStan's constrained
hyperparameters, and the wrapped non-centered model at `c = 1` reproduces the
density and gradient of the compiled `centeredness=1.0` twin.
`hsgp_coordinate_draws` returns the same log scales on those draws.

`benchmark_emission_latency.jl` reports package loading, first model construction,
first RK artifact emission and repeated calls separately, including compiler time,
allocation and garbage collection. It uses independent synthetic regression and
ragged panel examples, with unchanged cold/warm artifact source checks. Run each
case in a fresh process; there is no timing threshold on a shared host:

```sh
julia --startup-file=no --project=test test/benchmark_emission_latency.jl regression
julia --startup-file=no --project=test test/benchmark_emission_latency.jl panel
```

A second argument saves the complete artifact and printed source for comparison
across revisions, plus a source copy with location nodes removed so checkout paths
do not affect the comparison. Dependency pins, Julia version and host load should
accompany timing reports; package-image rebuilding belongs to the separate
loading phase.

`benchmark_emission_breadth.jl` measures compiler work across many independently
named but structurally identical BRMs. The names deliberately give each model a
different `BRMI` carrier type while the statistical work stays fixed, exposing
planner or emitter specialization that grows with corpus breadth. It reports RK
artifact and StanBlocks construction separately, then repeats the first model as
a warm control. As above, compare cumulative compiler time and allocation from
fresh processes on the same host; the script has no shared-host timing threshold:

```sh
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 both
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 rk
julia --startup-file=no --project=. test/benchmark_emission_breadth.jl 12 sb
```

`rk_planning_specialization.jl` pins the same property without timing. After
one warm member of a value-route family (caller-owned submodel, ordinary
callable assignment, grouped, population and log-scale predictors), building
another member, emitting its RK artifact and building its kernel must not
compile any BRM method, or generic method over BRM values, whose signature
names that member's operations. Only constructing the model's own
`NamedTuple`, `BRMI` and `RKBRMI` may. It records compilation with
`jl_dump_compiles` and also checks that the reused compilation emits the same
program up to the names. A second testset builds a member whose submodel
callable differs, so its expression types are new throughout (a named column
embeds its definition): the source walkers that collect data, references,
target axes, roles and additive terms must still reuse their compilation, while
constructing the member's values, the caller-owned hook and the prepared
representation may specialize:

```sh
julia --startup-file=no --project=test test/rk_planning_specialization.jl
```

## NLME estimators

`nlme_connected.jl` checks explicit `unit_partition=:connected` with generic
Gaussian models: structural transitive connectivity, dormant branches and zero
coefficients, indivisible joint observations, exact likelihood/gradient and
prior accounting, complete-unit SVI minibatches, optimizer continuation,
streaming membership checks, and a Laplace marginal against an analytic
Gaussian marginal. `nlme_connected_fit.jl`, in the NLMEEstimation scratch
environment below, checks repeated covariance families, protocol agreement,
start transport, scientific labels and empirical Bayes modes against the
analytic conditional Gaussian, plus a short SAEM call. Neither rewrites the
authored model or uses numerical derivatives as an oracle.

`nlme_laplace_coloring.jl` checks the whole-model Laplace's structural
coloring on connected units: a unit of subjects that all read one hub subject
needs the same few colors at 5 and 13 subjects, and on hub and chain units with
Gaussian and Poisson cells the colored unit Hessians, Laplace values and
gradients match the same problem under the position coloring. A pattern cut
short of the cells' reads is caught by the first assembly's check, which warns
and falls back to the position coloring. Subjects no cell couples, and the
strict partition, keep the position coloring.

`nlme_joint_start.jl` checks a complete NLME estimate against the same
BRMI's compiled Stan model: non-unit covariance and natural subject effects
round-trip through Stan's unconstrained coordinates, fixed/free indices
partition the full vector, subject labels are checked, and a direct coordinate
copy is shown to be a different physical point.

`nlme_conditional.jl` conditions that compiled model, and the RK program of
the same BRMI, on a checked start for flat, fixed-shared-HSGP and nested
models: values, gradients and Hessians equal the full target's at the embedded
point restricted to the subject draws; the RK and Stan problems agree on free
gradients and log-density differences at the same physical points; fixed
coordinates survive every evaluation and short WarmupHMC runs on both
backends; provenance names each free draw's subject and nested unit; and
mismatched or tampered starts and foreign backends are refused. It runs in the
test environment above.

`nlme_view.jl` pins `brm_nlme_view`, lockstep evaluation
(`brm_nlme_loglikelihoods[_and_gradients]`), and single-subject cell
evaluation (`brm_nlme_loglikelihood[_and_gradient]`) against an independent
oracle with analytic η gradients. It also compares the deprecated
`kernel(...) do` spelling with the indexed cell's partition, likelihoods and
η gradients. The single-subject test checks that
value and gradient allocations stay flat when unrelated subjects grow from
3 to 9 to 27. A response observed in the cell through a `groups` binding,
including a left-censored one with a per-row limit and an `addprop` scale
beside another in-cell observation, has the same single-subject cell, checked
against independent oracles. Both joins keep one subject's work flat at 81 and
243 subjects; the censored join's `addprop` scale is written by the kernel's
plate, so its cell does not run the kernel for every subject. A completed
covariate's observed entries join their subject's cell.
Flat-row grouped models retain lockstep evaluation and refuse
the single-subject method by name. It runs in the test environment above.

`nlme_component_blocks.jl` checks that `brm_nlme_view` reads a random-effect
block declared inside a function-style component call (`kin ~ kinetics(subject)`
with `(1 | pk | subject)` in its body) as the same block at model scope, over
the column the call passes. A model whose subject block and a crossed device
block are declared in components has its model-scope twin's partition under
the calls' paths, except that the components' population coefficients stay
shared latents; its lockstep likelihoods equal the twin's at the same point and
the full RK density, with η gradients, and its SVI split is the RK target. A
component grouping named nowhere is refused by name; either grouping can be the
units. A nested call through a renamed input composes paths and names, the
blocks' scale laws read the calls' own data for prior-median starts, and
`brm_nlme_estimates`, `brm_svi_fit` with connected units and
`brm_nlme_joint_map` (a compiled Stan model and a checked full start) read the
blocks. It runs in the test environment above. Snag `nlme-view-does-n-4a5e64ce`.

A single-level treatment-coded factor owns no packed coordinate: `nlme_view.jl`
checks that such a model has the partition, likelihoods and η gradients of the
same model without the factor, on flat rows and in a `@plate for` cell, and that
a declaration the layout does not hold at zero length stays an internal error.
`nlme_fit.jl` checks `check_protocol`, SAEM and the named estimates on those
models, and `nlme_conditional.jl` checks RK and Stan parity of their conditional
targets.

`nlme_monotonic_cell.jl` reads a `mo()` or `mo1()` term in a subject's `@plate`
cell through `groups`, as a bare term, beside a population block and as a joint
block over two predictors. RK prepares each strictly with the subjects' effects
declared partitioned, and the level table, which has no row form, is computed
once per population point. Cell values and full partition gradients match the
whole model's pointwise densities and central differences, and no subject reads
another subject's effect. `brm_nlme_model` matches the same reference. A planned
sparse Laplace problem integrates the subjects' effects past the same term, one
component per subject: its value matches a Laplace marginal computed from the
whole model's density, and its gradient matches central differences. It runs in
the test environment above.

`rk_batched_field_cells.jl` checks a paired Fourier field with channel mixing written as array cells (BRM todo `07y7ghy`, from RK snag `plate-of-per-row-d1b4ed43`). The cell is `phase = frequency_t * (coordinates[s] ./ lengths)`, then products of shared coefficient matrices with `cos.(phase)` and `sin.(phase)`, then `correlation * raw`. It is checked against the same field written with one `@deffun @juliacompat` call per row, on three counts:

- RK gives both the same coordinates, density and native Reverse gradient.
- RK plans the array cell's plate as matrix products (log group `:plate_batching`) and the per-row plate as a cell loop.
- SBBRMI's compiled Stan density and gradient equal RK's.

ReactiveKernels batches plates from 0.1.127 (RK `fa3deb83`). Below it the batched-plan check is `broken`; this environment's RK pin `23fbef437` (RK 0.1.164) requires it. Everything else passes on both. Each spelling rule in the file's header was found by a refusal. It runs in the test environment, about 3 minutes on one CPU.

`rk_plate_invariant_output.jl` checks an RKBRMI `@plate` output whose cell reads
no per-index value, `decay[g] = exp(log_a)`, over three cells (BRM snag
`retained-pointwi-fa27bc37`). At model scope `decay`, `sum(decay)` and
`length(decay)` see every cell, and the likelihood matches `Distributions.logpdf`.
A second plate reads `decay[g]` per subject. Its pointwise densities match the
same oracle. Its whole-model density and native Reverse gradient match a
model-scope `fill` of the same values and central differences. `brm_nlme_model`
gives each subject's likelihood and η gradient. A one-cell binding is the
positive control. ReactiveKernelsPPL fixed the shape in 0.1.129 (RK `b6196f114`,
ReactiveKernels snag `plate-output-of-c4a0ca87`), which this environment's RK pin
`23fbef437` (RKPPL 0.1.156) carries. On older RKPPL the output is one number. There
the three-cell checks are `broken`: a later `decay[2]` raises a `BoundsError`,
and `sum` and `length` return one cell's value. The one-cell count, total and
likelihood pass on both. Its `decay` is `broken` there too, because one number is
not a one-entry array. It accepts
testset substring filters; about 5 minutes warm on a Linux fleet host.

`nlme_nested.jl` uses a public synthetic model with interleaved observations
and 1, 2, and 3 nested units per subject. It checks explicit ownership,
compact η likelihoods and gradients against an independent Normal oracle,
the ragged NLME protocol, posthoc/SAEM smoke paths, a full Stan-start
round trip with reordered subjects, and the end-to-end chain from an SAEM fit
through `brm_nlme_joint_start` to WarmupHMC on the conditional Stan and RK
subject-draw targets. Run it in the NLMEEstimation scratch
environment described below.

`nlme_fit.jl` checks the prior-median default start against independently
computed medians (authored, default, truncated, hierarchical and `Flat()`
laws), keeps random-effect blocks without an ID independent, reads named
natural-unit estimates of a hand-set point (and passes them back as
address-keyed `inits`), and runs `brm_nlme_fit` with `empirical_bayes`,
`saem` and `focei` on flat and nested models. Run it in the NLMEEstimation
scratch environment described below.

`nlme_start.jl` reads complete RK and Stan points as NLMEEstimation starting
values with `brm_nlme_start`. It covers a correlated population-PK model and a
nested model with a correlated subject family, nested units and a shared HSGP,
in estimated and fixed shared modes, and a partial fit withholding one endpoint
of a shared block. Subject log-likelihoods sum to the RK
observation likelihood at the point, and `brm_nlme_joint_start` rebuilds the
same Stan point from the natural values. Initial estimates keyed by prior
address (`brm_nlme_start(m, inits)`) give the addressed θ, σ and Ω with zero
η, and `brm_start` on the joint map's Stan model gives the same point as
`brm_nlme_joint_start` of that start. Run it in the NLMEEstimation scratch
environment described below.

`nlme_two_stage.jl` checks one unit's conditional problem
(`brm_nlme_unit_problem`) on flat and nested models. Its log-density
differences equal the full RK model's at points assembled by `brm_nlme_draws`,
and its gradient matches finite differences. The file also runs
`brm_two_stage_fit` end to end, full-batch and minibatch, with a WarmupHMC
unit sampler, and checks that units sampled on several threads give exactly
the single-threaded draws. Run it in the NLMEEstimation scratch environment
described below, with `--threads` above 1 so the threading check exercises
concurrency.

`nlme_units_panel.jl` (test environment) reads a public synthetic regional
panel (`nlme_panel_fixture.jl`: regions ⊃ districts ⊃ municipalities and
district-years) as region units. It checks that a stratified
`gr(muni, by=sclass)` block is one covariance family per stratum, with every
municipality in its class's family and its region; that
`RKBRMI(...; completions=:noncentered)` makes each missing cell of an `mi(...)`
completion a unit draw in a family fixed at unit variance (and that the
default centred reading, whose law reads the regions' effects, is refused by
name while the fits' `@brm` entry lowers it non-centered; a Gamma law is
non-centered by its support transform, with the centred density and agreeing
lockstep and one-unit routes); that
district-year totals cells are attributed to the region owning their rows;
and that every region's unit problem differs from the full RK density by its
own unit alone, in value and gradient. It also checks each region's η Hessian
pattern (`BRM._brm_nlme_eta_sparsity(m)`, what `eta_hessian_sparsity` declares): it covers
every nonzero of the region's dense η-HVP block and leaves most pairs of the
region's effects out, and a flat model declares none.
`nlme_units_panel_fit.jl` runs NLMEEstimation on the same panel: the fixed
completion family (`NLMEEstimation.fixed_omega`), prior-median starts in every
stratum, the population prior's ω gradient against finite differences with a
fixed family present, stratified estimates by stratum, sparse η Hessians from
the declared patterns (`check_protocol` against η-gradient differences,
`NewtonEBE` by default, and `empirical_bayes`'s sparse Hessians equal to the
dense blocks less the prior precision; NLMEEstimation 0.1.3 or later), and a
timed `brm_two_stage_fit`. Run it in the NLMEEstimation scratch environment
described below; NLMEEstimation master needs Treebars `a2f9e41` or later,
newer than this environment's pin.

`nlme_grouped_hsgp.jl` (test environment) reads a grouped HSGP,
`hsgp(x; by=g)`, whose levels are the subjects: on a flat-row model and on a
subject plate that reads `hsgp(ecg_time; by=ecg_subject)` through
`groups(ecg_subject => subject)`, each level's basis weights are one fixed
family of unit draws in its subject's block, the length scale and amplitude
stay shared, and lockstep, single-subject and unit-problem densities match the
full RK density. It also covers a subject without a level (ragged blocks), a
hyper-predicted amplitude's `(1 | g)` effects as an estimated family, the
shared readings (levels that are not subjects, `cor = C`, a declared
`Omega ~ LKJCholesky`) and the refusal of labels that name only some
subjects. Its data and models live in `nlme_grouped_hsgp_fixture.jl`.
`nlme_grouped_hsgp_fit.jl` runs NLMEEstimation on those models: the weight
family in `NLMEEstimation.fixed_omega`, `check_protocol` on flat and ragged
layouts, the population prior's ω gradient beside the fixed family, and short
SAEM, FOCE(I) and empirical-Bayes fits read back by name. Run it in the
NLMEEstimation scratch environment described below. The lockstep and
unit-problem checks live in `nlme_unit_checks.jl`, shared with `rk_cdar.jl`.

`rk_cdar.jl` (test environment) checks `cdar(step; by=g, cor=C)` on RKBRMI:
RK density and gradient equal SBBRMI's at transported coordinates for
`cor = I`, a correlated `cor`, stated and known priors and a walk grouped by
the units, the RK walk equals an independent stationary recurrence, and the
printed source replays. Its NLME half reads districts nested in region units
(`nested_groups=(kreis=:region,)`): each district's innovations are a fixed
identity family in its region's block, `sigma` and `rho` stay shared, a `cor=`
within the units keeps that reading and one across the units keeps the walk
shared, lockstep and unit-problem densities match the full RK density, and
`brm_laplace_problem` and `brm_svi_fit` take the model. A frozen RK replay is
`@test_broken`. Its data and models live in `rk_cdar_fixture.jl`.
`rk_cdar_fit.jl` runs NLMEEstimation on the nested model: the innovation
family in `NLMEEstimation.fixed_omega`, `check_protocol` on the ragged layout,
and short SAEM and empirical-Bayes fits read back by name. Run it in the
NLMEEstimation scratch environment described below.

`nlme_crossed.jl` (test environment) reads a random-effect block over a
grouping crossed with the subjects (`shared_groups = (:class,)`, every class
in several subjects) as shared latents in θ: roles, the refusal naming a
grouping named in neither `nested_groups` nor `shared_groups` and the
refusals of a bad `shared_groups`, lockstep, single-subject and unit-problem
densities against the full RK density, the per-unit Laplace (row-structured
and whole-model) against the analytic Gaussian marginal, the minibatch SVI
split against the full RK target, `brm_nlme_estimates` naming the block for
every block form (plain, stratified, Student-t, level-correlated, known
scale) against the model's own predictor values, and a `car(...)` field,
whose edges cross the subjects, read as shared latents. Its data and models
live in `nlme_crossed_fixture.jl`. `nlme_crossed_fit.jl` runs NLMEEstimation
on them: the population prior against RK's prior and log-Jacobian queries,
prior-median starts of the crossed scales, a short `brm_nlme_fit(saem)` naming
the block, and a two-stage fit holding it at its stage-1 estimate. Run it in
the NLMEEstimation scratch environment described below. Snag
`crossed-grouping-39d39fcc`.

`population_grouped_hsgp.jl` checks the population and transport routes on a
grouped HSGP, `hsgp(t; by = g)`, whose weights `ranef_blocks` does not list.
It runs a shared trend, per-subject deviations with a per-subject amplitude
and a `(1 | subject)` intercept on one grouping column. `population_draws`
zeroes the weights and the amplitude effects with the intercept (BridgeStan:
the deviation is 0 on every row). `population_model` keeps the fitted
coordinates, agrees with it on everything the program computes, is
idempotent and survives `reprocess`. A grouping whose only effects are a
grouped HSGP is accepted. `transport_draws` copies the weights by level label
onto a model with a level dropped, and refuses a new or resampled level with
the `resample_groups` route. It also checks the ambiguity message for an
unqualified hyper-predictor. `BRM_PRED_RUNTIME=0` skips the compiled-model
sets.

`nlme_svi_sampling.jl` checks work-weighted batches by exact enumeration of
density and gradient expectations, including repeated draws, the alias
distribution, positive probabilities and bounded batch allocations.
`nlme_svi_weighted_iteration.jl` exercises the production optimizer against
an analytic target, including duplicate evaluation counts, reported visits,
and lazy local averages against averages of every iterate. It also checks that
a non-finite unit stops dense and structured factors under uniform and work
batches before any update, with an unpoisoned control run that does update.
`nlme_svi_workspace.jl` checks compact local layouts against independent dense
gather/scatter arithmetic and storage independent of the global dimension.
These three arithmetic scripts need only Julia's standard libraries.
`nlme_svi_units.jl` checks the bounded local Gaussian operators, exact
reparameterized gradients, linear allocation and checkpoint growth at fixed
rank, covariance preservation under coordinate reordering, and an independent
correlated Gaussian fitting oracle. It also checks a structured unit's own steps:
the default does not follow `unit_curvature`, controls saved before them keep what
they ran with, on an exact Gaussian with tight coordinates the default converges
where step 1 leaves the target (the control), and on one with low-rank covariance
the rank part at its default rate beats the best diagonal, where at the unit's whole
step it loses to it. It also needs only standard libraries.

`nlme_svi_unit_curvature.jl` checks a dense unit's step on an orthogonal design
(`unit_curvature = :orthogonal`): the design is d pairs at radius √d
under a uniformly random rotation; its Stein estimate is a quadratic unit's whole
curvature and its gradient and value the exact expectations; the full step reaches
such a unit's optimum within the trust regions; and, through the production run
loop, a prior scale that reads 32 units of 48 coordinates reaches its profile
optimum with the units' factors at theirs, where the single pair (`:pair`, the
control) leaves both off. Controls without `unit_curvature` keep the pair. The step
takes the estimate's positive part: on saturating rows, whose target has precision
exactly 1 off their span, the whole estimate (the control) has nearly symmetric
eigenvalues and falls below 0.01 where the positive part stays at 1 and reaches the
rows' stiffness; through the production loop, controls without `unit_positive_part`
keep the whole estimate, so an old checkpoint resumes as it ran. A direction whose
curvature lies below the factor's still widens, exactly on a Gaussian. About a
minute; standard libraries only.

`nlme_svi.jl` checks minibatch stochastic variational inference
(`brm_svi_fit`). The minibatch split of the RK target is checked first: the
global prior and Jacobian term plus every subject's cell term equal the full
RK log density and gradient, at a point with non-unit covariance. The scaled
batches of a partition of the subjects average to the full density and
gradient, and each subject's term equals `brm_nlme_unit_problem`'s density.
On synthetic Gaussian (intercept and slope) and Poisson (intercept) models
it compares the fitted draws with four NUTS chains: posterior means agree, and
the global coordinates' sds may be narrower, as variational approximations
are, but not wider. The split is also checked on a flat-row grouped model
(each row a cell of its subject), a nested model whose subject blocks have
three different lengths, and the regional panel of `nlme_panel_fixture.jl`
(nested units, non-centred completions and totals cells); the nested model's
fit is compared with NUTS too. Starts (prior medians, a packed point,
`inits`) and their refusals are covered as well. It runs in the test
environment above.

`nlme_svi_settling.jl` checks the settling stage between exploration and
polishing (BRM 0.2.242). Settling starts at the window where exploration's test
passed, both schedules explore identically to the bit until then, it ends on its
own test or at `settle_windows` (and `stages.settled` says which), and
`settle_windows=0` polishes straight away. A run stopped while settling continues
exactly, and a checkpoint whose controls predate `settle_windows` keeps the earlier
schedule. On an exactly Gaussian target with 32 tight globals every unit reads,
where q* is known, settling brings the returned factor closer to q* and its global
means within half an optimal sd (the earlier schedule: about one). About 6 minutes
warm, most of it the exact Hessian; it accepts testset substring filters.

`nlme_svi_trend.jl` checks exploration's trend test (BRM 0.2.264): on window
statistics alone, a rise too slow for two windows `patience` apart is read by the
slope over `trend_windows` windows, a stationary ELBO ends exploration where the
two-window test does in all but about 1% of seeds, and controls without
`trend_windows` keep the two-window test. On a non-centred step that every unit's
data pin (8 units, batch 4), the two-window stop ends at the 40-window floor with
the step's mean far off, and the trend stop explores on to within 1 nat of a
400-window run, both identical to the bit until the earlier stop. A checkpoint
continues on its own rule, and one whose controls predate `trend_windows` on the
two-window test; the keyword is validated and `GaussianBlockLaplace` refuses it.
It accepts testset substring filters.

`nlme_svi_heavy_tails.jl` checks what one extreme ELBO draw does to minibatch
SVI, whose estimate can have an infinite expectation with every draw finite.
With no bound, one gradient of 1e21 freezes its Adam coordinate (the control), and
with `_brm_svi_clip` that coordinate keeps its step. A coordinate that most batches
do not read keeps its scale under the `√(N / B)`-wide bound and loses it under a
tighter one (the control). `_brm_svi_window`'s `spread / W` is the variance of its
trimmed mean on Gaussian and skewed draws, and one draw moves neither. The stopping
test neither stops a rising fit nor hides a stalled one on an extreme draw, where a
test of means and variances does stop the rising fit.
A structured local factor's own Adam (above `unit_dense_limit`) is bounded the
same way, and a work-weighted fit (`unit_sampling=:work`), whose bound widens by
the batch's largest unit weight, keeps its globals moving on the same model.

`nlme_svi_rank_step.jl` checks `LowRankGlobal`'s exploration step
(`_brm_svi_exploration_global_step!`): two coordinates with scales 1 and 0.01
given the same gradient on their rows of Β move them 1 : 0.01, μ and ρ step
exactly as `_brm_svi_step!` steps them, and with no rank part the step is
`_brm_svi_step!` to the last bit. Its polishing step
(`_brm_svi_polish_global_step!`) moves μ 1 : 0.01 too where a rank column couples
the two coordinates (the whitened step moves the second along the column, more
than ten times as far), and with no rank part it is
`_brm_svi_whitened_global_step!` to the last bit. In settling and polishing Β's
rows step `_brm_svi_rank_step_ratio` (0.2 at the defaults) times as far as μ's, and
a mean-field step ignores that factor. On `svi_gauss_model` (200 subjects, batches of
4), 15,000 iterations of exploration leave every global's sd below 1.2 times the
NUTS sd and its mean within one NUTS sd; with the absolute step (0.2.242 and earlier) one
sd read 1.53 times (2.30, 2.39 and 1.00 on seeds 2-4, against 0.66-0.86).

`nlme_svi_authored_factors.jl` checks the public RKPPL authored-factor API
and production SVI ownership. On a
six-subject model whose coefficient pair `z[2i-1], z[2i]` is read only by
subject `i`, and on a variant with hyperparameter-dependent Normal, Exponential
and Beta priors, it gives each pair (and each constrained coordinate) to its
subject, keeps the hyperparameters' priors global although every subject reads
them, and checks that each prior and Jacobian event lands in exactly one
bucket. At three nonzero packed points the bucket values and compact gradients
plus the likelihood equal the full RK sampler's value and Reverse gradient.
It then checks SVI's authored prior term (`_BRMSVIAuthoredPrior`): on the
strict subject units it equals the standard-normal split term by term, and
with the subjects also owning their coefficients the global term plus every
unit term equals RK's whole prior and Jacobian, with the units' reads of
`tau` carried in the globals' gradient. A `Dirichlet` simplex, one atomic
law and coupled transform, is refused once its two coordinates are given to
different units, as is a coordinate assigned twice, while the strict subject
units build with the simplex global. That needs RKPPL 0.1.72, which reports
the constant Dirichlet's own block as its support (snag
`ReactiveKernels/factor-support-r-2b8c2968`; 0.1.70 reported the whole packed
vector). It also checks the production ownership split, public fits and
draw packing. Run it alone in the test environment.

`nlme_svi_owned.jl` covers structural ownership through dormant branches and
zero coefficients, shared and prior-only coordinates, connected units,
explicit global promotions, optimizer resume and streaming absent/returning
units. Its density and native Reverse gradient oracle is the full RK sampler.
The dormant scalar/vector branch's ordinary Reverse needs Enzyme dogfood from
`0fa85df3` (repairs `74abe90` and `7ac58eb`, contained in the bootstrap's
`cecb571a`); earlier pins raise `IllegalTypeAnalysisException` there.
`nlme_svi_ownership_algebra.jl` isolates atomic-block and directed-prior
closure, including fixed coordinates and factors that own no coordinates.

`nlme_cell_latents.jl` checks `@plate` cell latents (vector and scalar) and a
component called in a cell on the NLME paths: their coordinates are
`:shared_latent` in θ, each unit's conditional likelihood is the likelihood
query's share and moves only with its own cell's latents, `shared_point` fixes
them, and the SVI split equals the full RK sampler's density and native Reverse
gradient. SVI gives a cell component's locals and a dotted cell latent's
entries to their unit (the latter from RKPPL 0.1.127, RK snags
`rkppl-dotted-pla-1629a747` and `dotted-cell-late-3789b8ea`); a `LogNormal`
cell latent's per-unit ownership is a `@test_broken` until ReactiveKernels reads
a transformed block per cell (RK todo `0sr6ilc`). A latent
whose law reads a per-cell model value is refused naming its rewrite, which
builds (RK todo `1olsblw`). Run it in the test environment; about 8 minutes on
one CPU.

`nlme_svi_atomic_partition.jl` checks a two-margin correlated soil hierarchy
with no trajectory parent. Its local atomic Cholesky parameter has one
packed coordinate but a two-by-two constrained shape. Both local ownership
and explicit global promotion must preserve the scientific grouping and
the full native density, Reverse gradient and authored-factor accounting.
Run this standalone script in the test environment.

`nlme_svi_running.jl` checks a prepared model's reuse of its target and workers
across calls (`diagnostics.target`), and `snapshot` / `snapshot_every` /
`snapshot_draws`, against a run on a fresh model that nothing observes.

`nlme_svi_combine.jl` checks `brm_svi_combine` and `brm_svi_fit(x; start)` from
fits on regional slices of a synthetic panel (subjects nested in regions, a
class grouping crossed with them): the prior projection against the closed form,
the product of sites against an independent oracle, informed and dropped sites,
units matched by label and carried to the combined hyperparameters (an exact
one-margin oracle with a mu-referenced intercept, and totals kept for
correlated, stratified and unit-held shared blocks, read back through the NLME
view), a global held inside two slices' units, frozen-replay slices, a low-rank
slice, a joint site's factor and its widening against closed forms, the between-slice
variance, slices that contradict each other in either product (the plain product as
the positive control), ridges that cross under priors they sit far from or of very
different widths and from fits that left them too narrow, the fallback when a joint
mean leaves what supports it, coordinates `exclude` leaves out (each start equals the
one from the same slices with those coordinates uninformed in every slice, joint sites
and carried draws included), the refusals, `brm_start` effects addressed by level on a frozen replay
that lacks some levels, slices whose scales differ 40-fold under a log link
(carried, the first window beats a cold start and the old copy), and that real
slice fits start the full fit near its optimum. About 15 minutes with loading
on one CPU; it accepts testset substring filters.

`nlme_svi_threads.jl` checks `brm_svi_fit(...; threads)` with real threads (run
it with `--threads=4`; it refuses to run without them). The state after every
iteration, whose Adam moments are the gradient trajectory, and the draws must equal
the one-task run to the last bit: through exploration and polishing, beside a thread
that allocates and collects, on nested blocks of different lengths, flat rows, a
low-rank global factor, the default random stream, a resume at another count and a
window fit. It also covers each worker's own samplers over shared read-only tables,
how a failing unit stops the tasks, the BLAS pin and the refusals.
`research/svi_threads/benchmark.jl` measures per-iteration time, GC share and retained
worker memory against `threads`.

`nlme_threads.jl` checks one-unit NLME calls from several tasks at once, which
take their own evaluators from the model's pool. On flat, plate and nested
models, the protocol's one-unit values, η gradients and population gradients,
called from every thread in shuffled order, equal serial calls to the last bit,
and the pool prepares more than one evaluator, and at most one per Julia thread
even when tasks holding one yield. `check_protocol(...; threads =
true)` passes on a nested model. A seeded `WarmupHMCNUTS(threads = true)` SAEM fit
of a 24-subject plate PK model gives the same bits in child processes with 1, 4
and N threads (`BRM_NLME_THREADS_N`, default 8). Over six seeds its estimates
agree with lockstep fits (`threads = false`) within twice the lockstep fits'
spread, and both recover the simulated population values. Run it in the
NLMEEstimation scratch environment below (NLMEEstimation 0.2.5 or later) with
`--threads=4`; it refuses to run without threads.
`research/nlme_threads/benchmark.jl` measures a round of one-unit calls and paired
SAEM fits of both routes at the thread count it is started with.
`research/nlme_threads/unit_cost.jl` measures what one subject's one-unit value and
gradient cost against its share of the whole-cohort sweep, on a cheap plate PK model
and a joint model with a second response, and how much of it is RK's partition call
(brm-use nlme.md, "Subjects on threads"). Run it in the same environment.

`nlme_svi_laplace_plan.jl` checks structural eligibility for
`GaussianBlockLaplace`, including fixed HSGP fitting inputs, nonstandard
Gaussian coefficient priors, and named refusals for nonlinear expressions.
`nlme_svi_laplace_carry.jl` and `nlme_svi_laplace_prior.jl` compare the
bounded common-factor carry and its conditional precision with small dense
Gaussian controls. `nlme_svi_laplace.jl` checks native mode, curvature,
marginal gradients, numerical budgets, and reproducible public fits.
`nlme_svi_laplace_streaming.jl` adds frozen replay across new, absent and
returning units, checkpoint serialization, and identity refusals. Its short
fits exercise transport; posterior accuracy is checked separately by
`research/shared_field_laplace/public_acceptance.jl` against the retained
exact Gaussian-marginal reference. Run these files separately in the test
environment; the native file also includes the structural checks.
`nlme_svi_laplace_identity.jl` checks that requesting additional output draws
does not change the checkpoint, absent-unit marginals follow the updated
common factor, and within-unit coordinate order follows semantic identities.
The public acceptance and cost drivers share `public_fixture.jl`.
`public_scaling.jl` measures five predeclared size cases with fixed numerical
limits. `public_streaming_cost.jl` keeps the active window fixed and extends
only the absent Gaussian state; it checks identical active targets and
separately reports checkpoint copying, construction and serialization costs.
These cost controls do not establish posterior accuracy on larger panels or
across repeated windows.

`nlme_laplace.jl` checks the per-unit Laplace marginal (`brm_laplace_problem`,
`brm_laplace_draws`) against oracles written from the model equations: Gaussian
random-intercept and correlated intercept-and-slope models equal their analytic
marginals and that marginal's gradient; a Poisson random intercept equals a
per-unit Laplace formula from a scalar Newton search; and NB and binomial
gradients match fourth-order differences of the Laplace marginal. Laplace draws
of a Gaussian unit match its exact conditional, and importance-resampled
(`conditional = K`) Poisson unit draws match each unit's conditional mean and
SD by quadrature. It also covers a unit density that is not concave
everywhere, and the two kinds of population point that score `-Inf`. Every
oracle runs against both implementations, the row-structured
`BRMStructuredLaplaceProblem` these models get by default and the whole-model
`BRMLaplaceProblem` (`structured=false`); a panel of an NB2 count with
correlated effects, an NB2 count with a shape predictor and a binomial share
checks that the two agree in value and gradient, and a LogNormal model that
only the whole-model path takes it. An NB2 model with a monotonic effect and a
shape predictor, whose parameters are not affine in the population
coordinates, checks the structured gradient's pullback of both parameters
through RK against the whole-model gradient and differences. It needs
ReactiveKernels with `prepare_ad_third` and pullbacks over several WANT ports
(the test pin) and runs in the test environment above.

`nlme_laplace_history.jl` checks a nonlinear Normal cell with two conditional
modes. The generic and sparse targets return exactly the same density,
gradient and modes after either of two distant proposals. A scalar conditional
mode oracle checks the value, and finite differences check both population
derivatives. This checks independence from previous evaluations; the
approximation still uses one mode.

`laplace_continuation.jl` checks which conditional mode the marginal uses:
the one continuation from the prior reaches (BRM 0.2.147). One unit with a
squared-mean and a linear Normal observation has two conditional modes on part
of the population line. On both the whole-model and the sparse target:
- the value and the selected mode match a scalar oracle of the followed mode
  on both sides of the point where that mode ceases to exist, where the
  selection switches branch;
- an evaluation whose followed mode ceases to exist mid-way is counted in
  `folds`;
- finite differences match the gradient on each smooth piece;
- the two targets agree;
- the sparse target's `search` record names each component's stop and counts
  its refused full steps.

A log-concave Poisson model takes one stage. Run it in the test environment
with `julia --project=test test/laplace_continuation.jl`.

`start_terms.jl` checks authored term initializers and deterministic prior bases
on StanBlocks and ReactiveKernels, with constrained read-back and coordinate
transport, effective bounds, known values, invalid addresses and overrides.
Run it directly in the test environment.

`gmrf_preconditioner.jl` checks a `gmrf` field's references against dense oracles,
without model lowering: each is one congruence with the determinant it reports,
in both channel combinations and through a star over coupled fields, and its blocks
between neighbouring times, or between a star's hub and another field, equal the
precision's. With a complete orthogonal probe set the rebuilt reference's exact
derivative cancels its estimate entry by entry, and with few probes the control
variate's derivative matches a central difference of its value.

`gmrf_laplace.jl` also compares a 640-state field and two coupled fields under
matrix-free algebra with exact algebra: value within four probe standard errors,
gradient against a central difference of the returned value.
`gmrf_vecchia.jl` checks the footprint representation's numerical core without a
backend: the exact Gaussian on a fully read grid with irregular gaps, the exact
time factor on synchronized reads at any neighbour count, separable channels, shared solves, gap
linkage, reverse derivatives against finite differences, the same footprint under
any row order on a tied lattice, and refusals. Outside the support (a zero or
infinite length, or lengths whose conditioning systems are numerically singular,
including the line-search proposal of snag `gmrf-neighbors-d-952fd2fb`) the density
and its derivative are `-Inf` with zero gradients, prediction's strict solve still
raises, and long lengths the system resolves stay finite.
`gmrf_vecchia_rk.jl` checks the term on RKBRMI: density and native reverse rule,
the `-Inf` answer of `rk_logdensity_problem` outside the support,
sampled channel factor, state plates under `locality = :error`, a strict
partition, replay from a fitted state, the same coordinates and density under
another row order, and refusals. `gmrf_rk_numerical.jl` checks the `mesh=` density
graph and its native Reverse gradient against a dense covariance, its sparse LDL,
and `-Inf` where the factor loses a pivot or a length is zero or infinite, with
long lengths it resolves exact against a 512-bit precision oracle; the other
`gmrf_rk_*` density files include it. `gmrf_vecchia_stan.jl` checks
valid Stan, the shared helper, RK/Stan parity through coordinate transport, replay
and descriptions.

`gmrf_laplace_start.jl` checks that the Laplace default start of a model-scope
`gmrf` field equals its `brm_start(rk, (;); base=:prior_median)` medians, for
declared laws, a known hyperparameter and a law reading a model-scope parameter.
It also checks that `brm_laplace_mode` and the `brm_laplace_plan` diagnostic use
that point, with the diagnostic at the unit point as a positive control.

`gmrf_small_length.jl` compares the normalized GMRF density, precision action
and diagonal against an independent 256-bit dense oracle on a five-vertex
mesh, through spatial lengths of `exp(-365)`. `gmrf_small_length_backends.jl`
checks the RK graph and Stan's Julia-compatible helper against that oracle,
including native Reverse gradients. `gmrf_small_length_runtime.jl` checks
the public RK and compiled Stan models' sampler values and gradients, and
distinguishes a finite tail density from a genuine `-Inf` overflow. The latter
two run directly with `julia --project=test test/<file>.jl`; the first needs
only Julia's standard libraries.

`start_values.jl` checks `brm_start`, initial values keyed by prior address,
on both backends. For a correlated population-PK model, a linked predictor,
a categorical predictor with a `Dirichlet` simplex, and an exact-total block,
it checks the addressed constrained values, the semantic transport's
agreement between the RK and Stan points, unaddressed coordinates kept at a
random base, and refusals of unknown addresses and invalid values. A model
with one lone `|ID|` per predictor checks that the default exact totals keep
each `|ID|`'s scale address, for `brm_start` and `brm_ranef_sd_coordinates`,
including the generated-quantities block of its prior program. It runs in
the test environment above.

`start_effects_scaling.jl` checks that `brm_start`'s explicit random-effect
solve on the Stan backend is linear in the model's size and in the given
levels: it counts the points constrained (`margins + 2` for an independent
block and `levels + margins + 1` for a level-correlated `gr(g; cor=...)`
block, whatever the number of given levels) and checks that adding margins
to a block costs less than one further constrained-plus-transformed vector,
on models whose transformed parameters dwarf the block. It runs in the test
environment above.

`nlme_estimation_ext.jl` covers the NLMEEstimation.jl protocol extension:
batched and single-subject evaluation (including `check_protocol` and online
SAEM on ragged joins and flat grouped rows), unit-partition population gradients,
analytic η Hessian blocks, Gaussian observation moments
and colored η Jacobians, `check_protocol`, and Newton posthoc against
independent oracles. Its Gaussian checks include arrays observed within plate
cells and a modeled scale shared by the observations in each cell. A periodic
`hsgp` read in a subject cell passes `check_protocol` with every preparation
checked at `type_stability = :error`, the sparse Laplace probe's data prefix
included. `runtime_numeric_settings.jl` checks that model's cell factors against
its whole likelihood in the test environment, so a landing that runs only that
environment still reaches them.
NLMEEstimation.jl is not part of `test/Project.toml`. Run the file from a
copy of this environment that develops a local NLMEEstimation checkout:

```sh
cp test/Project.toml test/Manifest.toml <scratch-env>/
julia --project=<scratch-env> -e 'using Pkg; Pkg.develop(path="<NLMEEstimation checkout>")'
julia --project=<scratch-env> test/nlme_estimation_ext.jl
```

Without NLMEEstimation the file stops with an error rather than skipping.
Shared fixtures live in `nlme_fixtures.jl`. Its sparse-Laplace testsets check a
cell that reduces its data with `unique!(sort!(...))`, with both responses and
with one omitted: `check_protocol` and the Laplace value and gradient against the
same lengths computed by `sort!(unique(...))`. A stub product checks that a cell
group moves to reverse over forward on `EnzymeRuntimeActivityError` alone, whatever
Enzyme can take today.

`nlme_partition.jl` checks strict structural ownership before returning an
NLME model, including a cross-unit dependency in a dormant branch, and a valid
unit with no observations in one group. It also checks exact cell-wise
curvature edges and bounds allocations when observations repeat a support.
Its dense-point testset checks that a one-unit η gradient (RK's local-only
`shared=false` route) equals the η entries of the full gradient, that the
query and plan work points equal the unit point after every call, and that one
unit's allocations do not grow with the number of units (3 against 300). A
one-unit gradient allocates within 1 KB of RK's own call, so it does not copy
the model to the heap.
`nlme_units_panel.jl` covers nested
partition likelihoods, gradients, cell-based Hessian sparsity and allocation
scaling. The `nlme_svi.jl` filters `minibatch split` and
`structural footprints and target split` check the wider SVI gradients without
running its longer sampling comparisons.

`nlme_car_separator.jl` checks the separated CAR law, coordinate transport,
replays and sparse unit supports. Its `car identity geometry` filter checks
strict cell values and structural ownership independently of unit plans.
The `strict car partitions` filter compares
both shared and unit gradients with independent per-cell reverse passes for
ICAR and BYM2, including CAR unit terms. Graph-visible identity geometry needs
ReactiveKernels 0.1.33 / ReactiveKernelsPPL 0.1.27 or later; strict partition
evaluation requires 0.1.36 / 0.1.30 for uniform empty/nonempty plans and
scheduled-loop resets. Lower-bounded shares completions evaluated
through strict partitions (`nlme_lower_shares.jl`) require 0.1.42 / 0.1.39,
which keeps shapes through nested data-only helpers inside sampled
expressions. Scan-captured sigma rows and censored ragged factors need the
0.1.47 / 0.1.43 floors. Shared-prefix value queries and gradients need
0.1.48 / 0.1.51; packed gathered-column shapes need 0.1.50 / 0.1.52.
RKPPL 0.1.54 or later also keeps shared-prefix binding in the
latest world and preserves custom packed-vector storage and local reads.
The grouped HSGP variants in `nlme_svi.jl` check
value/gradient parity and shared-cache refresh across changed points.
Both partition fixtures use ordinary Enzyme Reverse
with the repaired dogfood revision pinned by `setup_env.jl`.

`nlme_laplace_capability.jl` checks an optional Laplace refusal using a toy
recurrence whose selected outputs are summed. It verifies that the capability
probe returns `false` and caches the RK locality diagnostic, ordinary protocol
checks and a short SAEM fit succeed, explicit integration rethrows the same
diagnostic, and unrelated errors propagate. A supported nested model remains
Laplace-capable. Run it in the same NLMEEstimation scratch environment.

`nlme_integrated.jl` covers NLMEEstimation's integrated families on BRM
models: SAEM integrates the chosen covariance families by BRM's sparse Laplace
and samples the rest. `check_protocol` with each set of families integrated
on a Normal and a Poisson nested model compares the values, modes and
covariance blocks with NLMEEstimation's own Laplace and the gradients with
central differences. Laplace-EM on a Gaussian nested panel reaches the direct
L-BFGS maximum of the same objective, and SAEM sampling the subjects with the
units integrated ends within half a log-likelihood unit of it.
`brm_ladder_fit`'s rung 3 runs two SAEM phases with replanning between them on
that model written as formula rows, whose affine Normal reads the automatic plan
certifies, so both families are integrated and SAEM is Laplace-EM. Written with
a `@plate` cell (`nested_model`), the automatic plan samples both families (it
does not certify a cell's log-concavity), so rung 3 is ordinary SAEM and needs
`burnin < iterations`; naming the groupings in `integrate` opts them in. Run it
in the same scratch environment. NLMEEstimation 0.2.4 needs WarmupHMC 0.2.8 and a
Treebars with `ProgressInterrupted`, so develop those checkouts in the same
`Pkg.develop` call as NLMEEstimation.

## Description PDF rendering

`description_tex.jl` checks Student-t sampling equations for standard, affine,
long-expression and embedded kernel models while preserving their source,
data, priors and model identity. It also runs under the root project. Add
`--pdf` to compile the complete public Markdown through Quarto and LuaLaTeX:

```sh
julia --project=. test/description_tex.jl --pdf
```

This explicit PDF mode requires Quarto and a working LuaLaTeX installation;
render failures propagate. Outputs use a temporary directory, or the directory
specified by `BRM_DESCRIPTION_TEX_OUTPUT`.

## Standard parametrization by default

`standard_default.jl` checks that `SBBRMI`, `generative_plan` and
`brm_descriptor` emit the standard (non-centered) parametrization when no
`total_groups` is given, and that an explicit `total_groups=:auto` builds exact
totals without a warning. The totals files below pass `total_groups=:auto`
explicitly:

```sh
julia --project=test test/standard_default.jl
```

## Mean-located exact totals under adaptive centering

`total_effects_mean_location.jl` checks that `adaptive_centering_problem`
centres each total on its column's least-squares fit on the column's location
design (the mean of the totals for a plain block; group-level predictors widen
the design), on a plain block and on one with group-level predictors: the
column table, the exact map (round trip, log-Jacobian against finite
differences and its closed form `sum(log a) + logdet(W'W) - logdet(W'AW)`,
aliasing, wrapped density and gradient), the `c = 1` and `c = 0` frames, the
window rule's per-cell candidates against a direct rebuild of each cell's
frame, `select_total_centeredness` against those candidates, the controls
round trip and report, an online fit, and totals beside an ordinary correlated
block (declared and learned order), whose controls keep the joint pair order:

```sh
julia --project=test test/total_effects_mean_location.jl
```

## Per-predictor S2Z weights

`s2z_per_predictor_rho.jl` checks `s2z_rho = [:mu => w_mu, :sigma => w_sigma]`
on two predictors with different coefficient counts on one S2Z grouping: each
block gets its own weights; the Stan source equals the scalar build's, with only
the `s2z_rho_*` data changed; the compiled model runs; unknown, missing,
duplicate and non-Symbol predictor names and a wrong per-predictor length are
refused; and a single `predictor => w` pair works:

```sh
julia --project=test test/s2z_per_predictor_rho.jl
```

## Exact totals and sum-to-zero priors that read data

`integrated_prior_data_fields.jl` builds each exact-total and sum-to-zero model
twice, its population priors as numbers and as data fields, and checks one source
across the values and equal blocks, coordinates, log density, gradient and
generated quantities against the literal twin, the refusals of what cannot be
integrated, and replay, rebuild and transport of the fitted prior:

```sh
julia --project=test test/integrated_prior_data_fields.jl
```

## Exact totals beside offsets

`total_effects_offset.jl` checks that `offset(...)` terms, of data and of a
model value, leave an otherwise eligible block on exact totals under `:auto`.
Its BridgeStan density equals the conventional program's marginal over the
integrated intercept at random points. It also checks that `reprocess` rebinds
the offset column, that `population_model` keeps the offset, and that the
description stays complete. An explicit `total_groups=[:g]` must raise an error
naming every predictor whose block on `g` would stay conventional:

```sh
julia --project=test test/total_effects_offset.jl
```

## Exact totals for some blocks of a group

`total_effects_selection.jl` checks `total_groups=[:g => [:b]]` on several
per-unit linear predictors that share one grouping and one `@plate` cell:

- `:auto` stays conventional;
- a whole-group request names each skipped predictor, why, and the eligible
  subset;
- the pair form integrates exactly the named blocks;
- the malformed or ineligible requests are refused.

Its BridgeStan density equals the conventional program's marginal over the
integrated intercept, also for a block whose scale prior is
`truncated(Normal(0, 1); lower=0)`. It checks that `generative_plan(plan,
new_df)` keeps the fitted selection, and that the joint WarmupHMC wrapper maps
the totals beside the group's conventional blocks:

```sh
julia --project=test test/total_effects_selection.jl
```

## Exact-total descriptions

`description_totals.jl` describes each synthetic model under the default
automatic exact totals and under `total_groups=()`. Shared logical prior
identities must carry identical families, arguments and support in both
programs, and the totals description must be complete. It also covers remaining
population columns, independent total columns, Student-t and flat population
priors, a kernel cell, coordinatewise generated vector priors with a consumer
family gap, and the retained sum-to-zero gap. It runs under the root project:

```sh
julia --startup-file=no --project=. test/description_totals.jl
```

## Group-level predictors in exact totals

`total_effects.jl` checks the exact-total Stan density, its gradient and
recovery against a Julia reference, with and without per-group design values
`W`. The reference subtracts beta's exact conditional density at two different
betas, which agree only for the true conditional moments.
`total_effects_group_predictors.jl` checks that a population column constant
within the block's groups, or equal to a random slope times a per-group value,
is integrated with the matched coefficients. Each totals program's log density
equals the conventional (`total_groups=()`) program's with those coefficients
integrated out. It also covers generated deviations, recovery,
`brm_population_effect_coordinates`, `population_draws`, `brm_start`,
centering locations, `population_model`, descriptions, frozen replay and rebuild
refusals for changed group-level values, transport for a new group, and the
legacy rebuild of a fit recorded without per-group values. Columns that vary
within a group, or that would make the totals' mean design rank-deficient,
stay sampled. `string_categorical_coordinates.jl` checks that a group-constant
`String` column stays sampled, like a `CategoricalVector` column.

```sh
julia --project=test test/total_effects.jl
julia --project=test test/total_effects_group_predictors.jl
```

## In-cell distribution combinators

`description_cell_combinators.jl` describes `@plate for` cells whose
observations use StanBlocks' `censored`, `truncated`, `weighted` and
`interval_censored` combinators. A call-form family is an argument child; a
token-form family is a `:family` child over the token's arguments, which a
family hook sees by its actual callable. Bounds are keyword-only, so a token's
arguments never render as bounds. The in-cell censoring prose matches the
formula wrapper's. A combinator token in `weighted`'s family position takes
`weighted`'s bounds, so `weighted(censored, w, normal, m, s; lower=lo)`
describes the same censoring as the call form
`weighted(censored(Normal(m, s); lower=lo), w)`. It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_cell_combinators.jl
```

`description_concatenation.jl` describes `vcat` in its two roles. As a value in
a `@plate for` cell, such as the dose-axis gather's `vcat(0.0, log_F[doses[i]])`,
it is a covered built-in rendered as a stacked vector. As the distribution of
`~`, here in an included `@slic` submodel, it stays the block-wise prior family.
It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_concatenation.jl
```

`description_known_values.jl` describes random-effect blocks held at known
values. A correlation held at the identity (`cor(:, ID) = I`) leaves the
margins independent, with covariance D²; a known matrix is described as such.
A known `sd(...) = v` or `cor(:, ID) = R` is the point mass `Dirac`, a
described law with a matrix shown as a matrix, and the sampled scales of such
a block are listed once, as its `sd` priors. It also checks the mean and
dispersion of `NegativeBinomial2`, at top level and in a `@plate` cell. It runs
under the test project:

```sh
julia --startup-file=no --project=test test/description_known_values.jl
```

`description_cell_functions.jl` describes the elementary functions a `@plate`
cell's statements use: `log1p` and `expm1` by their standard names,
StanBlocks' `reshape(v, m, n)` as the column-major fill, and StatsFuns'
`normcdf`/`norminvcdf` as Phi and its inverse. It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_cell_functions.jl
```

`description_comparisons.jl` describes comparisons: model-scope numeric
indicators `1.0 * (vessel == 2)` with each of `==`, `!=`, `<`, `<=`, `>` and
`>=` written as the indicator `\mathbf 1\{…\}`, in both languages, and `|`
and `&` of comparisons in a `@plate` cell as one indicator of their logical
or/and, a nested connective parenthesized. A formula's `(1 | g)` keeps its
group term. It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_comparisons.jl
```

`description_plate_cells.jl` checks that a `@plate for` cell is described as
written, in its own index. It covers:

- entry reads `x[i]`, row reads `x[rows[i]]` with each row set stated once,
  and `indexin` positions;
- a `groups(g; order = t)` plate's group rows;
- a sum over one row set, also when a long relation names it as one term;
- a collected output `y[i] = …`;
- a cell latent among the cell's statements and in the prior table.

It also checks that no input binding gets a symbol, that each part's sentences
come before its own equations on the page, and the incomplete-page line in both
of its variants and in both languages. It runs under the test project:

```sh
julia --startup-file=no --project=test test/description_plate_cells.jl
```

## ODE solvers in kernel cells

`rk_kernel_ode_cell.jl` runs the population-PK `@plate for` cell of
`nlme_fixtures.jl` with its closed-form concentration replaced by Stan's
`ode_rk45`, `ode_rk45_tol`, `ode_bdf` and `ode_bdf_tol`, spelled as
stanblocks-use §29 writes them (`rep_vector`, `to_array_1d`, `[:, 1]`,
`to_vector`) with a `@deffun @juliacompat` right-hand side. On RKBRMI each
solve is checked against the closed-form model's density and Reverse gradient
at the same coordinates, and the tolerance-controlled forms against the
compiled Stan density and gradient. It needs ReactiveKernels with
`rk_ode_rk45_tol` (`ce48fc77` or later, which the test environment's pin
contains) and the solver packages, which the test environment does not carry,
so under `--project=test` it fails at its `using` line by design. Run it from a
copy of this environment that adds the solvers:

```sh
cp test/Project.toml test/Manifest.toml test/LocalPreferences.toml <scratch-env>/
julia --project=<scratch-env> -e 'using Pkg
    Pkg.add(["OrdinaryDiffEqLowOrderRK", "OrdinaryDiffEqBDF", "SciMLBase", "SciMLSensitivity"])'
julia --startup-file=no --project=<scratch-env> test/rk_kernel_ode_cell.jl
```

On BRM 0.2.64 with ReactiveKernels `3deb61ae` that passes 171/171 (a Linux fleet host,
2026-10-29); adding the solvers precompiles 56 packages in about 9 minutes.

`kernel_ode_cell.jl` remains
the Stan-only gate for the closure-style `kernel(...) do` spelling.

## Native GP and dual HSGP adoption

`rk_gp_covariance.jl` checks BRM-owned squared-exponential and periodic
covariance construction, multivariate locations, duplicate-location jitter,
input ownership, graph composition, independent density and native Reverse
coordinates. `rk_dual_hsgp.jl` checks the exact public motorcycle dual HSGP
against independent normalized densities and all 44 gradients at three points
and four centeredness settings; it reuses the existing hashed public data.
Both load the real ReactiveKernels extension.

`rk_gp_multiaxis.jl` checks exact formula GPs with multiple isotropic axes
and one or more anisotropic axes. It compares covariance, normalized density
and every native Enzyme gradient coordinate with an independent oracle and
generated Stan. It also checks printed-source replay, duplicate-location
jitter, known per-axis scales, and two readers sharing a named latent draw.

These focused scripts use RK's public `recipe_inventory` and
`rk_cholesky_lower` APIs. Run them
in an environment carrying RK `db4f16aa0f3d768e376269429ec613691c743b13`
or a descendant containing both APIs and the precompiled graph allocation
repair and native prepared-callback Reverse repair. The matching nested
packages share the same checkout. The standard
`test/setup_env.jl` pin and `test/Project.toml` source revision carry that floor:

```sh
julia --project=test test/setup_env.jl
julia --project=test test/rk_dual_hsgp.jl
julia --project=test test/rk_gp_covariance.jl
julia --project=test test/rk_gp_multiaxis.jl
```

## Python/JAX modules

`rk_jax.jl` translates RKBRMI fits and replays with
`jax_source(rk.model, rk_bound_plan(rk))` and checks that BRM's preprocessing
(level codes, fitted `zscale`/`center` anchors, cell flattens) is the module's
`prepare_data`. With `RK_JAX_PYTHON` naming a Python that has `jax`, it also
compares each module's `log_density` and `jax.grad` with the native density
and its central differences, through the pinned ReactiveKernels checkout's
`test/jax_python.jl`; without it those comparisons are skipped. It needs an RK
with `jax_source` and `prepare_data` (`888bdfd42` or later). Its grouped-read
case (`groups(group => cells)` read in a cell, ordered or not) checks that the
row partition is `prepare_data` host code and that the densities match: that
needs RK 0.1.138 (`51e6404f`), whose host code writes a list element in place
and translates only a decided arm, and whose reductions over a gather through
a ragged row mask the padding. The standard pin carries both:

```sh
python -m venv <jaxenv> && <jaxenv>/bin/pip install "jax[cpu]"
RK_JAX_PYTHON=<jaxenv>/bin/python julia --project=test test/rk_jax.jl
```

`rk_jax_models.jl` is the model-level parity suite. Through
`jax_module_driver.jl` it runs each module once in Python over several
datasets and points and compares, against the native RKPPL queries of the
same binding: `log_density`, `log_prior`, `log_likelihood` and `log_jacobian`
(eager, `jax.jit` with the prepared data closed over, and `jax.jit` with them
traced as arguments), `jax.grad` against the native Enzyme gradient, and
`constrain` and `values=` outputs. One module also evaluates a second dataset
with other sizes and other prior values given as data, whose RKPPL program is
identical. Constructs that do not translate yet are `@test_broken`, each
naming its gap. `BRM_JAX_REQUIRE=1` turns a missing `RK_JAX_PYTHON` into a
failure instead of a skip; `BRM_JAX_WORKDIR=<dir>` keeps each run's module,
inputs and Python stderr.

```sh
RK_JAX_PYTHON=<jaxenv>/bin/python BRM_JAX_REQUIRE=1 julia --project=test test/rk_jax_models.jl
```

The file requires ReactiveKernels' JAX lowering as it landed on RK main
(RK 0.1.132 / RKPPL 0.1.140 or later; the test pin is `23fbef437`) and refuses
to run below it. The CDF-based laws, jit with the prepared data traced as
arguments, size-generic modules, per-group AR(1) scans, nested `constrain` and
the multi-axis and level-correlated HSGP terms are required checks. Its testset "every definition BRM emits is documented"
also checks, natively, that every definition RKBRMI emits carries a docstring
and that `rkppl_documentation` keeps each submodel's.

`rk_jax_inference.jl` takes one representative model (a synthetic station
network: ragged per-station series, a covariate with gaps through `mi()`, a
seasonal `hsgp`, correlated station levels and sensitivities, log-normal
readings). It checks parity first, then samples the posterior natively
(WarmupHMC's quorum sampler) and in Python (NumPyro's NUTS on the module's
`log_density`, which needs `numpyro` in the same Python). It compares every
scalar constrained quantity: the two means must agree within 4.5 combined
Monte Carlo standard errors. It also compares predictions at posterior draws.
About 10 minutes on one CPU; `BRM_JAX_RUN_DIR=<dir>` writes the fixture,
versions and results as a run directory (`manifest.json`, `parity.json`,
`fit.json`, `predictive.json`).

```sh
<jaxenv>/bin/pip install numpyro
RK_JAX_PYTHON=<jaxenv>/bin/python BRM_JAX_REQUIRE=1 julia --project=test test/rk_jax_inference.jl
```

## Chunking heavy suites

`rk_retained_transport.jl` attaches `RKBRMI(brmi, artifact.plan, built)` to
the artifact route's existing build. It checks normalized compiled Stan
densities, every ordinary Reverse coordinate and the physical-check return
schema, with all responses and a held-out response. A centered SBBRMI maps
exactly onto the same build (densities and gradients included) while the
permutation-only helpers refuse it; invalid input and a corrupted permutation
remain errors. Public block metadata identifies known parameterization mismatches
before compiling Stan. `rk_coordinate_transport.jl` covers the broader semantic inventory,
including the typed unsupported-coverage diagnostic for HSGP internals.
Run either file with `julia --project=test test/<file>.jl`.

Files with dozens of testsets (`rk_parity.jl`, `rk_emitter.jl`) OOM a squeezed
host single-process. Those files spell their blocks `@stestset` (defined in
`testset_filter.jl`, covered by `testset_filter_check.jl`) instead of
`@testset`, so lanes can run them in fresh-process chunks: pass substring
filters matching testset names as trailing args, or comma-separated via
`BRM_TEST_FILTER` (union). An empty filter runs everything, exactly as before;
a non-empty filter that matches nothing exits 1 rather than reporting a
hollow green.

```sh
julia --project=test test/rk_emitter.jl "group-C" "fail closed: group-C"
BRM_TEST_FILTER="von-Mises,mixture" julia --project=test test/rk_parity.jl
```

A new heavy file adopts the same contract with one `include` plus the macro:
`include(joinpath(@__DIR__, "testset_filter.jl"))` after `using Test`, then
`@stestset "name" begin ... end` per chunkable block.

`hsgp_hyper_prior_support.jl` checks explicit priors on authored log HSGP
intercepts through both RK and Stan. The reported grouped model and an
ungrouped control retain unbounded Normal intercepts; bounded and normalized
truncated priors keep their own support and density constants. The checks
compare independent normalized densities, every ordinary Reverse coordinate,
compiled same-BRMI Stan, group-specific floors and full printed-source replay.

`missing_covariates.jl` checks joint continuous `mi(x)` predictors through
StanBlocks. It compiles Normal and LogNormal models and compares normalized
densities and all unconstrained gradients with an independent explicit
observed/missing split, then resolves completed covariates and derived values
through public logical output coordinates. Separate blocks cover complete
conditional observations, observed-only transform anchors and their degenerate
cases, frozen row masks, and retention
of fitted missing coordinates during subject kernel CV. It uses only root
dependencies, so `julia --project=. test/missing_covariates.jl` also works;
trailing filters such as `anchors` or `kernel CV` select focused blocks.

`mi_intercept_only.jl` checks an intercept-only predictor that only an `mi(...)`
likelihood sizes: `log(sigma) ~ 1` or `mu ~ 1` read by `mi(y) ~ Normal(...)` or
`LogNormal(...)`, and the scale of a covariate model `mi(x) ~ LogNormal(mx, sx)`.
SBBRMI sizes it by the column's row count `n_rows_y` and RKBRMI fills it over
the column's rows; replay recomputes both. Densities and gradients on both
backends equal a twin
that declares the intercept as a scalar parameter. Run
`julia --project=test test/mi_intercept_only.jl`.

`rk_single_level_factor.jl` checks a single-level treatment-coded factor on the
RK route. Its K=1 source equals the K=2 source apart from the level list and the
zero-width coefficient declaration, as SBBRMI declares `vector[0]`. Densities and
every native Reverse coordinate equal the factor-free model, including R2D2 and
Horseshoe predictors, and same-BRMI compiled Stan agrees through the coordinate
transport. Run `julia --project=test test/rk_single_level_factor.jl`.

`rk_missing_value_predictors.jl` checks native completed covariates with fixed
observed-only standardization anchors, shared correlated subject effects and
subject-kernel composition. It compares full normalized densities and all
inference gradients with independent and compiled same-BRMI Stan controls,
and replays complete printed definitions/data. Observed-only declarations and
complete real columns retain their modeled observations without missing
inference coordinates. Run `julia --project=test
test/rk_missing_value_predictors.jl`; trailing filters select individual
blocks. These are public producer controls, not original application or
performance acceptance.

`rk_completed_covariate_axes.jl` checks a completed subject covariate feeding
ragged kernel cells with their own likelihood, scalar covariate hyperpriors,
and two correlated subject predictors. It compares normalized densities and
all native Reverse gradients with an independent oracle and compiled
same-BRMI Stan, including cell-row permutations and separate pointwise terms.
Printed source replay, completion graph ports, multiple missing entries and
callable name collisions are included. Run `julia --project=test
test/rk_completed_covariate_axes.jl`. The verified native closure is RK/PPL
`3bdbfbbbb1c82caf4acf7b6ecb5fa28e3a089c58`, StanBlocks
`e355994384d44b86108105264ae3e34e0f2a0625`, and Enzyme 0.13.209.

`rk_component_prior_inputs.jl` checks caller values used by group and HSGP
priors when their names collide with the component's own parameters. It uses
independent normalized laws, all-coordinate finite differences and full printed
source replay with native Reverse gradients. Run `julia --project=test
test/rk_component_prior_inputs.jl`.

`rk_modeled_transform_predictors.jl` checks named pointwise assignments from
modeled and completed columns as formula predictors. It retains the original
subject axis across ragged kernel likelihoods and checks independent normalized
densities, every ordinary Reverse coordinate, compiled same-model Stan, named
graph values, row permutations, bit-exact printed-source values, and
all-coordinate Reverse agreement (absolute/relative tolerance `2e-13`). A second model covers assignment chains, scalar sampled parents and
an explicit transformed-coefficient prior without completion. Run
`julia --project=test test/rk_modeled_transform_predictors.jl`; trailing filters
select the standalone blocks. It uses the same native closure listed above.

`rk_modeled_hsgp_axes.jl` checks HSGP terms over model-derived axes (an
assignment such as `x = exp(eta)` and a log-link predictor) with a fixed
`domain=`, and linear orthogonalization on model-derived and raw axes. The RK
graph evaluates the basis, centering and projection from the current axis
values. It checks the actual built graph intermediates, the emitted basis
law at fresh axes, independent normalized densities, every ordinary Reverse
coordinate, compiled same-model Stan, pointwise queries and their replay, row
permutations, artifact replay and complete printed-source replay. Run
`julia --project=test test/rk_modeled_hsgp_axes.jl`.

`missing_joint_covariates.jl` checks the native correlated
`mi([x, z]) ~ MvNormalCholesky(...)` block. Independent normalized density,
all-coordinate gradient and pointwise likelihood oracles cover all four
bivariate row-specific missing patterns, formula priors and frozen replay.
Other blocks check positive log-space assignments, same-named dataframe
columns, observed-only anchors, complete or entirely missing columns,
row-aligned vector locations, and fitted completion during group resampling.
Run `julia --project=. test/missing_joint_covariates.jl`; trailing filters
such as `independent` or `log-space` select focused blocks.

## Shared preparation and backend lowering

`rk_addprop.jl` checks the shared additive/proportional scale law as visible
native graph source. Unequal grouped responses, empty subjects, callable
aliases, top-level assignments and in-cell calls retain normalized densities,
all ordinary Reverse gradients, censoring and row order against independent
oracles and compiled same-BRMI Stan. Per-location scales read in the cell,
including through a binding and a scale computed from parameters and per-row
data, have the same checks. Complete printed source and artifact re-entry are
included; a scalar native-location control is checked separately.

`rk_r2d2_priors.jl` checks shared-margin R2D2M2 and joint population/contrast
budgets through native RK execution. It compares absolute normalized densities,
every ordinary Reverse coordinate, pointwise responses and complete printed
source replay with independent oracles and compiled same-BRMI Stan at physical
simplex parameters. Controls cover unequal categorical frequencies, explicit
factor design columns, sampled latent references, partial per-margin ICCs and
conflicting allocation statements. The two original synthetic formulas retain
their separate active sampled reference scales.

`rk_coordinate_transport.jl` checks the public complete semantic RK/Stan
coordinate map: scalar and population coefficients, categorical reference and
cell-mean coding, grouping levels, random-effect scales and standardized draws,
and whole correlation factors, including independent predictor blocks on the
same grouping column, and centered SBBRMI blocks (intercept, correlated, single
slope, shared `|ID|` and configured-scale families) through their exact
per-point map. Distinct nonzero points compare constrained
values, normalized densities and every ordinary Reverse gradient with compiled
same-BRMI Stan. Point/gradient round trips, input preservation and incomplete
coverage diagnostics are included. HSGP internals remain an expected capability
gap. `rk_modeled_random_slopes.jl` additionally compares the generated map with
its independent hand-written mapping. Run either file through the test project;
trailing filters select individual blocks.

`generated_prior_transport.jl` composes the existing public APIs to reuse a
single response-omitted Stan prior draw in an observed Stan/RK target. Runtime
constrained names select the target parameters, BridgeStan applies the target's
inverse transforms, and BRM maps the resulting point into RK coordinates.
The fixture checks four joint draws containing a GMRF with learned space/time
scales and channel correlation, correlated ordinary effects, positive/bounded
scalars, a simplex and a declared coefficient vector. Complete parameter
round trips and graph values must agree with the saved generated draw. Run
`julia --startup-file=no --project=test test/generated_prior_transport.jl`.

`spline_basis_signs.jl` checks canonical TPS/t2 projection signs, input
ownership, and frozen projections directly. The spline blocks in
`rk_parity.jl` compare RK densities and Enzyme gradients with independent
oracles; `spline_parity_models.jl` holds their shared BridgeStan fixtures.
`spline_sb_parity.jl` compiles those fixtures independently and checks the
full-posterior anchors and gradients, including the distributional spline toy.

The focused preparation gates are `preparation_program.jl`,
`preparation_replay.jl`, `preparation_assignments.jl`, and `backend_plan.jl`.
`rk_modeled_random_slopes.jl` keeps modeled formula values active in both
population and subject slope designs. It covers six rows from three subjects,
censored LogNormal and Normal observations, sampled covariance and authored
priors, row permutations, independent full-density/native Reverse oracles,
compiled same-BRMI Stan, and complete source/artifact replay. Direct formula
slopes and partially completed covariates also keep their original row axes.
`rk_declared_consumer_axes.jl` checks that an explicit ragged consumer supplies
an intercept predictor's row axis even when its endpoint data are omitted.
Its observed model compares complete printed RK replay, ordinary Reverse and
an independent normalized oracle with compiled same-BRMI Stan on distinct
subject, ECG-row and response axes. Omitted-data geometry checks do not claim
native prior-draw execution.
`rk_monotonic_consumer.jl` combines sampled monotonic contrasts and an HSGP
on operation rows consumed by a subject kernel. It checks complete source and
artifact replay, independent density/all-gradient oracles, compiled same-BRMI
Stan at physical simplex parameters, and an empty subject's retained prior.
The assignment gate checks dependency ordering and distinct response row axes.
`preparation_replay.jl` checks fitted-transform replay through StanBlocks.
`shared_preparation.jl` checks categorical and group levels, callable term
shape dispatch, and explicit refusal of unsupported structured group forms.
`sbimpl_generation_concurrency.jl` checks concurrent SBBRMI construction and
immediate consumption in compiled callers, including cold and warm vector
priors, continuous and discrete mixtures, horseshoe labels and source ownership,
same-named custom modules, and valid construction after rejected input. It also
checks that generated families add no module bindings or support methods. Run
`julia --threads=4 --project=test test/sbimpl_generation_concurrency.jl` in a
fresh process.
`rk_construction_concurrency.jl` checks independent RK builds with distinct
data and layouts, then compares their first executions against analytic
densities. Run `test/setup_env.jl` to install the fixed RK pin, then
`julia --threads=4 --project=test test/rk_construction_concurrency.jl`.

`rk_bounded_source.jl` checks scalar declaration bounds and ordinary value
observations. `rk_bounded_runtime.jl` compares the original family kernels,
coordinate Jacobians, standard Enzyme gradients and full printed-source replay
with independent oracles and emitted Stan. These use the exact published
`restricted` capability pinned by `setup_env.jl`.

`rk_computed_predictor.jl` covers ordinary formula predictors consumed by a
whole-array gather, with and without an intercept and with equal or distinct
predictor/observation row axes. It checks full normalized densities, analytic
and finite-difference gradients, mapped compiled Stan, caller ownership and
complete printed-source replay on the published computed-matrix repair.

`rk_wildcard_ownership.jl` checks that wildcard coefficient defaults reach
owning predictors, skip nonowners, and retain unmatched-target and equal-
specificity errors. Its unchanged linked multi-axis model compares normalized
Normal/censored densities, standard native Enzyme gradients, mapped compiled
Stan and complete public printed-source replay.

`rk_held_out.jl` checks response-level likelihood withholding with shared and
separate parameter branches. It preserves every fitted prior and coordinate,
compares normalized native densities and reverse gradients against independent
oracles and compiled Stan, and replays the complete emitted source and artifact.

`held_out_rows.jl` checks that a per-row `held_out` mask is refused on both
backends and on every replay (a fit receives only the rows it conditions on,
user decision `1bwqnxs`), that a replay's `held_out=` replaces the fit's
whole-response hold-outs through `reprocess`, `restan_data`, `generative_plan`
and the descriptor, and that a per-subject window fitted on its kept rows
replays onto the full data, a denser grid and a re-drawn subject with its
frozen `zscale` constants.

`rk_callable_terms.jl` checks positional/keyword formula broadcasting, including
singleton input axes, against independent and compiled Stan targets.
`rk_kernel_values.jl` checks unequal and empty grouped cells, explicit ragged
joins, lexical native callables, distinct catalogue/response axes and in-cell
held-out aliases without rewriting the original BRMI.
`rk_cell_latents.jl` checks latents declared in a `@plate` cell on RKBRMI:
scalar, varying-length vector and bounded latents, and laws reading sampled
scalars or a per-cell formula predictor (`m[i]`, which needs the RK pin
`0470e9e13` or later). Each matches the compiled Stan model under `brm_coordinate_transport`
(density, gradient and the physical check), and the vector latent also matches
an independent oracle. The file also covers draws restored per cell, a latent
nothing reads (not sampled on either backend), `brm_start` addressing the
latent by name on both backends, and frozen replay onto the fit's cells and
its refusal for new cells, the refusals, and one known gap: a law reading a
top-level vector parameter at the cell (`b[i]`), which BRM's RK preparation
refuses as a per-subject linear predictor.
`rk_grouped_bounded_response.jl` gathers observed rows before bounded-response
validation and keeps scalar or row-specific censoring, truncation and interval
bounds aligned with uneven/empty subject cells. Complete source replay,
ordinary Reverse, independent densities and gradients, compiled same-BRMI
Stan, invalid evidence and original bound-column ownership are covered.
`rk_submodel_values.jl`
checks paired source hooks for an empty marker with active predictor keywords,
no extra result coefficient, exact callable bindings and fresh-module replay;
an emitted entry and a bound leaf cannot share the same symbol. Two hook calls
appending one shared helper `@kernel` emit its definition once, with the
independent density/gradient oracle and artifact replay; a different definition
under that name fails before evaluation.

`rk_callable_source.jl` checks original nested callable identities supplied by
ordinary native source definitions. Ragged cells, live keyword arguments,
native Reverse, independent density/gradient oracles, ownership and complete
source replay are covered. Both ordinary assignments and uneven/empty ragged
nested helpers compare with compiled same-BRMI Stan on the published
StanBlocks caller-dimension fix `65cabef`, pinned by `setup_env.jl`.
These scoped checks do not certify complete application performance.

`rk_graph_source.jl` supplies explicit `@kernel` entries through the callable
source provider. It inspects the actual bound posterior for subject plates and
child scans, then checks complete printed and artifact replay, ordinary Reverse
and independent normalized oracles against compiled same-BRMI Stan. The
computed grouped predictor case retains an empty subject's prior and all five
coordinates while passing separately sliced inputs to the child object graph.
The pinned RK revision also supports those computed graph arguments; source
definitions or a separately prepared reader alone do not prove posterior
visibility.

`kernel_expr(bound, assign_layout(bound))` is pre-build replay text. Its calls
to qualified `KernelSpec` readers compose when `build_kernel` runs; textual
plate/scan counts therefore do not count the complete built graph. Use public
`kernel_graph(built.spec)`, `plate_body` and `scan_body` entry/body accessors.
The fixtures explicitly qualify their internal recipe predicates as frozen
revision diagnostics. The former `999451c` composition failure is repaired
on the pinned published RK `5184fe62`; provider-only controls still do not
certify the complete original application.

`rk_kernel_observation_families.jl` checks exact in-cell Student-t and
caller-owned scalar observation graphs through the original constructor
identity. Direct, grouped and separately bound child `KernelSpec` forms retain
two aligned vector arguments and a live shared scale, with normalized
independent values/gradients, ordinary Reverse, strict same-BRMI Stan,
complete source/data replay, original geometry/ownership and scalar algebra
inside the actual built observation plate. A Bernoulli-logit case preserves
integer response rows across uneven/empty cells.
`rk_observation_source_names.jl` checks the two routes by which a caller-owned
law calls a function the build module does not bind (`logsumexp`): a leaf named
from `entry`, and a law defined in the consumer's module bound as a child
graph. Each matches an independent mixture density with ordinary Reverse
gradients and printed source replay.
`rk_grouped_observation_arguments.jl` checks the equivalent outside-cell law,
grouped argument collection, explicit ragged joins, empty cells, broadcasting,
elementwise argument expressions and unchanged data ports. Each case checks
normalized independent values and every ordinary Reverse coordinate, strict
same-model compiled Stan parity, printed source replay and actual graph gathers.
Run the standalone file with optional case filters such as `joined` or `integer`.
`rk_kernel_weighted_observations.jl` checks in-cell
`weighted(family, weight, args...)` power likelihoods: SLIC token and call
forms, a cell-expression weight and a subject-shared weight over a native
family, plus a caller-owned `_rk_observation_source!` law, each with uneven
and empty cells, independent normalized values and Reverse gradients, strict
same-BRMI Stan parity and printed source replay. The weight reader must read
only data. A weighted censored Normal, in the call form
`weighted(censored(Normal(m, s); lower=lo[i]), w[i])` and the token form
`weighted(censored, w[i], normal, m, s; lower=lo[i])`, is checked against an
independent weighted clamp density, Reverse gradients, pointwise rows and
strict same-BRMI Stan parity. In-cell `interval_censored` remains
`@test_broken` on RK.
`rk_incell_censored.jl` checks an in-cell `censored(Normal(...); lower=...)`
over a ragged join against Stan, pointwise rows and NLME subject
log-likelihoods, then the same cell under a row weight against the
unweighted model: density, Reverse gradient, pointwise rows and NLME
log-likelihoods all scale by the row weights, and its density and gradient
match Stan.
`rk_observed_graph_law.jl` checks the public inclusive threshold law's lower
and upper branches and normalized interior in that built graph, against
independent native values/gradients. The retained ordinary-constructor fixture
`rk_observed_callable_law.jl` has three strict Stan-gradient failures caused
by the independently attributed upstream `normal_lcdf` derivative
approximation (`inclusive-normal-eff38407`, StanBlocks issue #59); this
source/graph bridge does not repair or relax that scientific comparison.

`rk_plain.jl` and `rk_statistical_library.jl` exercise BRM-owned statistical
declarations through ordinary RKPPL lowering. `rk_statistical_library.jl` also
checks the `rkppl_model(:car_icar)` / `(:car_bym2)` bodies on a graph with two
components, an island and an area without rows against an independent oracle
of the car term's law (the BYM2 scaling from each component Laplacian's dense
pseudo-inverse), and checks that the data-bound recipe brm-use documents gives
the same densities. The bodies need RKPPL 0.1.145 (the test pin is `23fbef437`).
Run its car testsets alone with `julia --project=test
test/rk_statistical_library.jl car`. `rk_statistical_source.jl`
compares emitted reusable block algebra with the former ordinary inline
source: every statistical submodel allocates its own parameters and returns
its computed value. Scoped names such as `b_g.tau.1` map bijectively
to the hand-inlined program's `b_g_tau.1`. Prior, likelihood and full
densities agree to rounding at three points, as do all standard Reverse
coordinates. Caller ownership is preserved and complete printed-source replay
is bit-identical. Optional testset filters select individual block cases.
Graph structure checks use RK's public `recipe_inventory`, including retained
child plates and scans.
`rk_prepared_data_source.jl` checks emitted categorical/ordinal preparation and
completed-covariate geometry, including declared level order, empty and
all-missing partitions, retained numerical recipes, printed replay and cached
columns that must not override source computations. Run it directly with
`julia --project=test test/rk_prepared_data_source.jl`.
`rk_factor_gather.jl` checks that categorical population effects gather their
coefficients from one level index per factor coding (treatment coding with a
leading reference zero, cell means directly, `ref=` listed first), that
predictors with identical design columns share one matrix, and the normalized
density and ordinary Reverse gradient against an independent oracle.
`rk_hsgp_domains.jl` checks fixed one-dimensional and tensor HSGP domains
against independent basis/frequency/floor calculations and normalized
same-BRMI compiled Stan values and every mapped ordinary Reverse coordinate.
It covers K=1, isotropic/anisotropic geometry, explicit normalized or bounded
hyperpriors, unchanged fixed boundaries under row rebinding, caller ownership,
actual built graph inspection, and complete printed-source fresh-module replay.
`rk_source_composition.jl` additionally combines those statistical definitions
with identity-resolved native entries and live keywords, retaining both kinds
of definitions in the ordinary source reference and complete printed replay.
`rk_leveled.jl` adds independent
categorical-reference and multinomial row-density, Jacobian and ordinary
reverse-gradient oracles. `rk_values.jl` checks whole
predictor arrays, callable readers and distinct predictor/observation axes.
`rk_source_roundtrip.jl` supplies the shared complete-program replay check:
definitions and model source are printed, reparsed and built afresh, then
compared bit-exactly for likelihood, prior and full density at three points.
`rk_parity.jl` adds independent statistical density and Jacobian references,
standard Enzyme reverse gradients and finite-difference checks.
`rk_mixture_weights.jl` checks hierarchical mixture probabilities against
independent normalized laws and compiled Stan, every ordinary Reverse
coordinate, pointwise likelihood, complete printed replay and changed-data
reprocessing. It also covers weight-only predictors, more than two components,
named and tuple expressions, computed shared weights, shared and row-valued
probability data, invalid rows and logistic endpoints.
Run it directly with `julia --project=test test/rk_mixture_weights.jl`.
`rk_stan_priors.jl` compares explicit shared scale/LKJ priors with emitted Stan
using a complete named coordinate map. `rk_gaussian.jl` checks the actual
ordinary Gaussian emission against analytic and mapped Stan densities and
gradients at three points and three data sizes, then reports warmed native
density and preallocated reverse allocations without machine-specific limits.

Prior support and density comparisons against BridgeStan live
in `truncated_priors.jl`, `hierarchical_prior_bounds.jl`,
`prior_expression_keywords.jl`, `term_prior_bounds.jl`, and
`von_mises_prior.jl`. `backend_comparisons.jl` executes the documentation's
four-pane helper and checks the emitted Stan.
`sbbrmi_display.jl` evaluates the displayed configured-submodel bindings and
body in a fresh module, requiring identical generated Stan and data. It covers
shared and distinct configurations, unchanged default calls, one-margin grouped
blocks (named through their family's single-margin form), and replay.
`callable_priors.jl` checks distribution-factory shape registration and a
complete-call Stan AST translation against the original Julia factory's
density.
`term_prior_semantics.jl` checks that both backends reject unmatched or
ambiguous term-prior addresses and that the most specific address resolves.
`stanblocks_term_prior_semantics.jl` checks custom simplex densities through
BridgeStan and prior-only RNG draws, alongside unchanged Dirichlet spellings.
`adaptive_centering_public_api.jl` exercises ordinary scalar, shared-ID `K=1`,
and correlated `K=3` blocks through the public wrapper at both compiled endpoints.
It checks scalar/per-cell controls, independent BridgeStan scale/correlation
queries, map/Jacobian/inverse/density and all gradients, restoration/copy isolation,
and every cell's 11 candidate losses against an independent weighted-correlation
oracle at zero, mixed and one source controls. It tests the fixed-frame scoring
proxy; it does not measure online window winners or sampler convergence.

`adaptive_centering_learned_order.jl` covers `ranef_order=:learned`, where
each group's triangular order follows its controls (snag
`correlated-kreis-15564811`). Its synthetic half needs no Stan model: it checks
the map at both compiled endpoints against a dense `cholesky(P*Sigma*P')`
reference, plus the inverse, a finite-difference log-Jacobian and the Enzyme
gradient. With every group in declaration order, the map, Jacobian, inverse
and scoring frame must equal the per-pair joint wrapper (ordinary, R2D2, HSGP
and scalar cells). The window rule is checked against an independent
re-implementation. The compiled half checks a `K=3` block at both endpoints
against BridgeStan's own `diag(tau) * L`, the `adaptive_centering_controls`
report, copy isolation, the declared fallback of a model without a correlated
block, and a short WarmupHMC run that is checkpointed and resumed.

`adaptive_centering_blockwise_gradient.jl` covers the block-level reverse pass
that BRM's WarmupHMC/Enzyme extension runs for an ordinary plan wider than
`K ≤ 2` and for a joint plan whose cells are all Gaussian random-effect pairs
(snag `warmuphmc-adapti-e35d58ad`). On compiled toy models (three margins,
a stratified `gr(g, by = b)` block, known correlations and SDs) it checks which
plans take that pass and which keep WarmupHMC's generic per-pair path (Student-t
mixing, HSGP cells), that the log density equals the generic path's exactly and
the gradient to rounding, and that the pass allocates under a quarter of what
the generic path does. It also checks the candidate-scoring frame, which builds
a Gaussian block's factor once per block, bit for bit against the per-level
frame.

`adaptive_centering_blockwise_widths.jl` sweeps that pass over block widths from
1 to 12 margins and four kinds of derived scale on a toy Gaussian density, with
no Stan toolchain. It checks that each plan takes the pass, that its log density
equals the generic path's exactly and its gradient to rounding, that both agree
with central differences, and that inputs are unchanged. A width has to be
checked on its own: the pass works at four margins, and at five Enzyme raised
`EnzymeNoTypeError` until the repair in Enzyme `6053067f`. The test bootstrap
pins `cecb571a`, which contains that repair and declares libEnzyme `7f23cb72`.

`adaptive_hsgp_centering.jl` checks pilot-selection under spectral underflow,
the partial-coordinate Jacobian, StanBlocks/BridgeStan density and gradient
consistency, physical constrained quantities, and the
distributional model's two distinct zero-mean HSGP bindings.

`adaptive_hsgp_level_centering.jl` checks online adaptive centering of a
grouped HSGP mapped across its levels: `hsgp(x...; by=g, cor=Omega)` (sampled
`LKJCholesky`), a known correlation matrix, per-level amplitudes and length
scales through hyper-predictors, independent levels with per-level
amplitudes, an anisotropic term, and the term beside a correlated
random-effect block and an ungrouped HSGP (snag `adaptive-centeri-11062238`).
Per compiled model: the metadata (one level block per term, weight layout,
correlation, scale roles), the reference factor `C_b = diag(s_b) * L` against
the compiled model's own level weights (`<term>_beta_scaled`), the wrapper's
map against a materialized triangular reference, `log|J|` by central
differences, the inverse, the density identity, the Enzyme gradient against
finite differences, and that `c = 1` samples the level weights themselves. It
also checks the learned-order wrapper, the controls report, the multigroup
refusal and a short `adaptive_warmup_mcmc` run on the two-axis shape.

`adaptive_bounded_scales.jl` checks that online adaptive centering reads
`Uniform(a, b)`-bounded HSGP length scales and SDs, and cdar SDs and
persistences, through Stan's own `lb`/`lub` transform. It also reads the
`<lower=fmax(0.0, L)>` declared by a `truncated(D; lower=L, ...)` scale prior
whose `L` is `length_scale_floor`, a data scalar or arithmetic over them
(`2 * length_scale_floor`), per axis for an anisotropic term, as StanBlocks
evaluates it (`ModelOutput.constraint_values`). Because the wrapper's density identities hold for any
function of those coordinates, each compiled model that adapts first compares
the physical value every cell reads with BridgeStan's `param_constrain`, then
checks the wrapper identities. A bound-independent oracle checks that a
`truncated(LogNormal(0, 1); lower=length_scale_floor)` prior wraps exactly like
the default floor plus one constant.

`adaptive_car_centering.jl` checks online adaptive centering of the areal
`car(area; edges, type=:icar | :bym2)` field: one cell per area, its `phi`
under `:icar` and its `theta` under `:bym2`, centred on the area's total (the
predictor's intercept plus the area's spatial component; `phi` stays
standardized). It resolves the cells and the intercept against the compiled
model, checks the transform, Jacobian and inverse against an independent map
written from the emitted law, checks that `c=1` is BridgeStan's transformed
`effects` (`:icar`) or the intercept plus them (`:bym2`; islands and areas
without rows included), checks candidate-frame invariance and Enzyme
and finite-difference gradients, checks car cells beside a correlated
random-effect block in the joint and learned-order wrappers (each family maps
as its own wrapper does), the controls report, the cdar refusal and the
generated-field skip, and runs one short online warmup on a data-rich lattice.

`adaptive_centering_threads.jl` differentiates the ordinary, joint (ordinary +
R2D2 + HSGP), learned-order (`ranef_order=:learned` over the joint cells),
S2Z, HSGP-only, cdar, scalar (total / S2Z cell) and car wrappers, car
cells and grouped HSGP level cells (a sampled and a known level correlation,
per-level amplitudes and length scales) inside the joint and learned-order
wrappers, and a Student-t block (each
level's factor scaled by its mixing weight) inside both, on
three concurrent tasks while a fourth forces collections, in a child Julia
started with four threads; the child must exit cleanly and every gradient must
bitwise-equal its single-threaded reference. On Julia 1.10, Enzyme's reverse
pass corrupted Julia's GC whenever an accessor allocated a work vector, read a
struct holding vectors in its row loop, reached two `_block_location`
instantiations, or passed a car cell to an outlined call (even in the car-free
`joint` case): the process segfaulted in `getindex` or aborted with "GC error
(probable corruption)". Synthetic blocks only; no Stan toolchain. The stress
loop and child launch live in `threaded_gradient_stress.jl`, shared with the
gate below.

`rk_enzyme_threads.jl` runs the same stress, 15 seconds per case, over
native-Reverse RK gradients (`rk_logdensity_problem` for correlated random
effects, an HSGP with a modeled log-scale, a logit GLM and an `@plate for`
population-PK cell; the NLME batched, compact-cell and η-Hessian entry points).
It is known broken: Enzyme's reverse pass corrupts memory when another thread
runs GC, measured on Julia 1.10 and 1.12 — the same upstream defect, reproduced
without BRM by a plain `@rkppl` random-effects model (Enzyme snag
`julia-1-10-concu-152a4415`; `reactivekernels-use` §7ao). It then records one
`@test_broken` without stressing; `BRM_STRESS_KNOWN_BROKEN=1` runs the stress,
which is the BRM acceptance check for that upstream fix.

`stanblocks_preservation_corpus.jl` compares fourteen representative models'
emitted SLIC/Stan, prepared data, metadata, and frozen replay against an external
baseline artifact. Capture the artifact on the implementation base, then use
that same file when checking the refactor:

```sh
BRM_SB_PRESERVATION=write BRM_SB_PRESERVATION_FILE="$TMPDIR/brm-sb.bin" \
  julia --project=test test/stanblocks_preservation_corpus.jl
BRM_SB_PRESERVATION=check BRM_SB_PRESERVATION_FILE="$TMPDIR/brm-sb.bin" \
  julia --project=test test/stanblocks_preservation_corpus.jl
```

The artifact is intentionally untracked. The comparison normalizes only the
absolute test-source path embedded by StanBlocks; it retains model text and
parameter identities.

`laplace_declared_gaussian.jl` checks automatic and explicit integration of
declared independent Normal vectors. It compares heterogeneous fixed priors
and sampled means/scales against closed-form Gaussian marginal values and
finite-difference gradients at several points. Nonlinear Normal observations
exercise the diagnostic and implicit gradient; mixed generated effects,
retained subsets, independent copies, prior dependencies, whole-vector width
screening and linear sparse storage cover structural behavior.

`laplace_mode_floor.jl` checks the mode search's rounding floor on the
whole-model, structured, sparse and matrix-free problems. One Poisson effect
starts far up an exponential's slope, where the decrement stops shrinking
fourfold with a gradient near 100: the value must be the Laplace formula at a
BigFloat oracle's mode, and the gradient must match central differences. An
effect beside an intercept of 1e6 has a real floor above `tol`, which must still
stop the search. Run it with `julia --project=test test/laplace_mode_floor.jl`.

`laplace_mode_overflow.jl` exercises finite-density trials with overflowing
derivatives in ordinary and sum cells, with exact and matrix-free solvers.
It checks local backtracking, sequential
fallback, conditional stationarity, marginal gradients, independent component
steps and deterministic repeated evaluations.

`laplace_gradient_nonfinite.jl` evaluates a finite marginal whose gradient is
not (an infinite third derivative at a converged mode) in an ordinary cell, a
sum cell and under matrix-free algebra. It checks the `nonfinite_gradient`
counter, the warning's named term, the closed-form value and the central
difference, and that a target materialized from a plan starts at zero. It also
pins a `log(erfc(t))` cell with its modes near `t = 19` in the same three
configurations: the gradient is finite, matches central differences and counts
nothing. That one needs an Enzyme whose libEnzyme build is `2b1f1cfe` or later
(`setup_env.jl` pins dogfood `07842f2`, libEnzyme `fce75191`); with an earlier
build its third order is NaN.

## One-time bootstrap

Use `test/setup_env.jl` as the single bootstrap entry:

```sh
julia --project=test test/setup_env.jl
```

The script materializes every external dependency at an exact full-SHA revision
in this host's package store,
`~/.local/share/BayesianRegressionModels/package-sets/<pkg>-<sha12>`
(`BRM_TEST_PACKAGE_STORE` overrides the directory). ReactiveKernels
contributes its monorepo root **and** the nested
`ReactiveKernelsDistributionKernels` and `ReactiveKernelsPPL` packages from
that same checkout; those nested packages have no standalone repositories. All
developed paths enter one resolve on Julia 1.10, where `[sources]` in
`test/Project.toml` is ignored. A dirty shared checkout is never used as the
source of truth: an exact revision is checked out deliberately, so the
environment does not drift with local branches.

Every pin first checks `~/.local/state/kb-agents/git-mirrors/<Name>.jl.git`.
When that mirror contains the exact revision, materialization uses it without
contacting GitHub. An absent mirror or missing revision falls back to the pin's
public origin. Mirror paths are derived from the pin list, so adding a dependency
needs no separate mirror entry. Existing exact cached checkouts are reused.

The store sits outside the checkout because Julia rejects a compiled image built
from another source path. With a dependency at one stable path, a fresh worktree
or job on the same pins reuses the images of those pins and their dependents, and
rebuilds only BRM and its extensions, whose source really is that checkout's. Four
things follow from sharing it across every run on a host:

- **A checkout is read-only and never rewritten.** A fresh mtime rejects an image
  even over identical bytes, and an edit would reach every run on the host. To
  try a change to a pinned package, develop your own checkout of it instead.
- **Each checkout holds one commit.** Its `.git` answers `git rev-parse HEAD`, and
  it has no history to ask about. Read history from the mirror or the package's
  own repository.
- **Two runs may materialize one pin at once.** Each builds its own scratch
  directory and renames it into place, and the later run validates the earlier
  run's checkout and discards its own.
- **An unused checkout is evicted.** Setup removes checkouts it has not used for 14
  days, recorded in the store's `.last-used/`, and anything a killed run left
  after a day. It touches no other name. If a test run says a package at a store
  path is missing, its checkout was evicted: re-run `setup_env.jl`.

A `test/.bootstrap/` left in an existing worktree is no longer read, and is safe
to delete once that worktree has re-run `setup_env.jl`.

The offline regression checks use small local Git repositories:

```sh
julia --project=test test/dependency_floors_test.jl
```

Running `setup_env.jl` writes `test/Manifest.toml`, which is deliberately **not** committed (the
root `.gitignore` covers `Manifest*.toml`). Re-run `setup_env.jl` on a new
machine, after changing a pin, or when an existing ignored manifest still points
at an older dependency checkout.

`BridgeStan` needs the BridgeStan C++ sources in addition to the Julia package.
`BridgeStan.jl` finds them via `$BRIDGESTAN`, falling back to
`~/.bridgestan/bridgestan-<version>`, and downloads them if neither exists.

## Why each constraint exists

Every one of these was paid for by a failed resolve; none is stylistic.

- **`WarmupHMC` must be a develop path, never a registry resolve.** The
  registry only carries WarmupHMC 0.1.x, and the root `Project.toml` bounds it
  at `WarmupHMC = "0.2.8"` — so a registry resolve is unsatisfiable, not merely
  stale. `[sources]` cannot fix this either: it is a Julia 1.11+ feature and is
  silently ignored on 1.10, which is this package's compat floor. The pinned
  revision must contain WarmupHMC `9c642178720d5c294b9cead86fc8c82da5a5db09`:
  that floor retains Pathfinder's use of the target's own
  `logdensity_and_gradient` and admits Pathfinder 0.10.7, the first registered
  release admitted by that dependency floor. Older and newer
  checkouts all report version 0.2.1, so the version number cannot express
  this. `setup_env.jl` resolves an exact published revision rather than a
  shared checkout's possibly stale local branch.
- **`StanBlocks` must be a checkout, not a release.** BRM does not precompile
  against registered StanBlocks; it fails inside a `@deffun` in `src/sbimpl.jl`.
  Configured `gp` / `hsgp` term priors additionally require StanBlocks
  `10529af04d42a330df383864059c2b61a11d9480` or later. Older checkouts trace
  the spliced term model before its keyword data are bound and fail on an
  unresolved `omega2`. StanBlocks is unregistered and both sides of this floor
  report version `0.1.5`, so verify the checkout SHA rather than its version.
  `hsgp(x; cov=:periodic, period=…)` additionally needs StanBlocks
  `bec23bc3c52303ebde60a026af48c435e4c81330` or later, which registers the
  `log_modified_bessel_first_kind` builtin its spectral weights call (older
  checkouts fail at transpile with `Could not find
  log_modified_bessel_first_kind …`); `test/gp_hsgp_periodic.jl` is the gate.
  Sampled values used only in another parameter's support additionally require
  `67767349bb1e9b05a50fdd1c05e0a8cf878e43e5` or later (landed in
  `e2b2fa952ebbd550016ff6cb85d11b0f4482aba1`). Earlier activity analysis drops
  those dependencies; `test/hierarchical_prior_bounds.jl` exercises this case.
- **`Treebars` is here even though no test uses it.** It is an unregistered
  *transitive* dependency of WarmupHMC, which pins it with a `[sources]` entry —
  ignored on 1.10, same as above. Without a path the resolve fails outright with
  `Treebars [e1e568c4] has no known versions!`. `Pkg.develop` can only fix a path
  for a *direct* dependency, so Treebars has to be listed in `test/Project.toml`
  as well; that entry is transitive plumbing, not a test dependency.
- **`Enzyme_jll` 0.0.302 is excluded.** That binary, which Enzyme 0.13.211
  requires, can segfault (signal 11) in `getindex` inside the reverse pass on
  Julia 1.10 when a differentiated function returns a freshly allocated array,
  such as an index-vector gather
  ([EnzymeAD/Enzyme.jl#3776](https://github.com/EnzymeAD/Enzyme.jl/issues/3776);
  the fix ships in a later `Enzyme_jll`). Native Reverse gradients that read a
  grouped effect as `draws[indices, margin]` crash with it; measured on the
  public random-effect block of `rk_coordinate_transport.jl`,
  `rk_retained_transport.jl`, `rk_modeled_random_slopes.jl` and five
  `rk_values.jl` testsets, all of which pass on 0.0.301. `Enzyme_jll` is a
  direct dependency here only to carry `Enzyme_jll = "< 0.0.302, >= 0.0.303"`,
  which resolves Enzyme 0.13.210 with `Enzyme_jll` 0.0.301 and admits the next
  binary without another edit; like Treebars, it is resolution plumbing, not a
  test dependency. A manifest resolved before this entry still holds 0.0.302:
  re-run `setup_env.jl`.
- **Enzyme is the ecosystem's `dogfood` line and needs its declared
  libEnzyme build.** BRM's RK cell gradients need repairs no registered Enzyme carries:
  a nested-NLME totals cell reads an `hcat` design built from views at its
  rows, which registered Enzyme rejects with `IllegalTypeAnalysisException`
  and dogfood `d37cb0b4` differentiates (Enzyme snag
  `native-reverse-t-6d51cae6`; user pick on BRM decision `0n0kgfh`). The
  sparse Laplace's ordinary `@plate` cells take their Hessian blocks through
  RKPPL's forward-over-reverse compact-cell AD. This requires both the
  static-activity repair from `7a886d9` (Enzyme snag
  `static-reverse-s-f9483e9c`) and the augmented-helper JIT-symbol repair
  (Enzyme snag `forward-over-rev-188051dc`), which preserves the `digamma`
  partials used by negative-binomial cells. Strict partition gradients
  need `a6458e9c` for constant plans and compact overlays; the ordinary
  Reverse fixtures do not enable runtime activity. The exact pin and declared
  libEnzyme build come from `setup_env.jl`. The pin includes `3a89b497`'s compact
  nested-AD overlay repairs for sum factors and the dynamic-call Hessian
  repair needed by lower-bounded completion geometry. It also repairs
  positive-tail `erfcx` derivatives through third order (Enzyme snag
  `erfcx-positive-t-db1c8c84`), required by scaled-tail likelihoods in the
  embedded Laplace. Its libEnzyme also repairs the integer-division fold
  that prevented grouped-HSGP partition gradients from compiling on Julia
  1.10 with default bounds (Enzyme snag `enzyme-type-anal-b401ef8f`).
  `setup_env.jl` materializes the pinned dogfood commit from the KB host's
  Enzyme mirror when available, develops its nested `lib/EnzymeCore`
  0.8.27 beside it (Pkg 1.10 ignores Enzyme's `[sources]`; registered
  EnzymeCore lacks `set_atomic_add`), and runs the KB resolver's
  dogfood-libEnzyme step: it points `Enzyme_jll` 0.0.301 at the declared
  libEnzyme build through the ignored `test/LocalPreferences.toml` and
  prints `KB-DOGFOOD-LIBENZYME <commit> llvm<M> <dir>`. The resolver checks
  that the build is available for the host and Julia version;
  otherwise setup refuses rather than load the stock library, whose Reverse
  gradients the build fixes (skill `enzyme-use`). Check a running process with
  `Enzyme.Compiler.Enzyme_jll.libEnzyme_path`.
- **All `develop` paths go in ONE `Pkg.develop` call.** Resolution has to
  satisfy them together: the unregistered BRM root, the external source
  pins, the nested ReactiveKernels packages and Enzyme's EnzymeCore. Developing
  StanBlocks by itself fails with `expected package
  BayesianRegressionModels to be registered`, while omitting a source-only or
  nested package produces the same error for that dependency.
- **Setup precompiles after resolution.** `setup_env.jl` suppresses
  instantiate's automatic precompile and explicitly runs `Pkg.precompile()`
  once after resolution. Host CPU and memory budgets still apply. See the
  `precompile` skill for current runtime policy and stall diagnostics.

## Why not `Pkg.test`

`Pkg.test()` cannot work here, so there is deliberately no `test/runtests.jl`
and no `[extras]`/`[targets]` in the root `Project.toml`:

- `Enzyme` and `WarmupHMC` are root `[weakdeps]`. `Pkg.test`'s sandbox resolves
  them from the registry, which lands on WarmupHMC 0.1.x and violates the
  `"0.2"` bound above. The only fixes are a committed absolute develop path or
  `[sources]`, and neither is available.
- A `test/Project.toml` already takes precedence over `[extras]`/`[targets]` on
  every supported Julia version, so carrying both would be two declarations of
  one dependency list.

These files illustrate why this environment exists — they fail at their own
`using` line under `julia --project=.`, before any BRM code runs:

| file | needs beyond the root project |
| --- | --- |
| `test/adaptive_centering_bridgestan.jl` | `WarmupHMC`, `Enzyme` |
| `test/adaptive_centering_warmuphmc.jl` | `WarmupHMC`, `Enzyme`, `DifferentiationInterface` |
| `test/plate_stress.jl` | `BridgeStan` |

There is no CI workflow for these on purpose: they need `stanc` and a BridgeStan
toolchain, so a GitHub Actions job would be red by construction.
# Prior-regime fitted geometry: `rk_prior_regimes.jl` retains the authoritative
# generative snapshot for omitted responses while native fitted execution has
# zero coordinates. Bound baseline observations keep their inference ancestors.
# Controls cover normalized compiled same-BRMI Stan, ordinary native Reverse,
# full source/artifact replay and input ownership; native generated draws are
# unavailable. Both backends start such a program at the empty point
# (`brm_start` with and without initial values, `base=:prior_median`, the start
# from the data), the coordinate transport pairs the two empty layouts, and an
# initial value is refused by name.

## Bounded sparse Laplace acceptance

`laplace_cell_hvp_width.jl` checks that a cell's Hessian block compiles at a
bounded Enzyme batch width: batched blocks and the whole target's value and
gradient against one batch over every direction, finite differences of the cell
gradient, a 129-direction cell at the same width, a batched group's move to
reverse over forward at that width, and the plan's and problem's Treebars
reports. It needs Treebars and the Enzyme dogfood line.
`laplace_width.jl` checks structural screening before derivative preparation,
deterministic whole-term selection, forced conflicts, correlated margins,
exact Gaussian retained-set values and native Enzyme gradients, local sum
directions, and the zero-integration target/draw/Pathfinder/ladder workflows.
Its independent Gaussian oracle integrates the same selected coordinates of
the full pointwise model; it does not compare different marginal targets.
`laplace_plan_reuse.jl` checks that prepared plans still avoid new AD preparation
and give independent targets when a strict subset is retained.

The measurements in `research/sparse_laplace_factors/*_2026-11-02.tsv` use
Julia 1.10.12 on a Linux fleet host, one CPU per process, RK
`5e9d2d03818ba5c8ef7d6d068fef42101f902d0e` (0.1.52 / PPL 0.1.55), Enzyme
`5325554cab66d13ab34c22ec502fd00759ffdcf7` and libEnzyme
`782c9d95efd9895e380c451ea9975cde1c197219` (LLVM 15). Each point runs in a
separate process under `/usr/bin/time -v`; `external_peak_rss_kib` is that
independent measurement. `allocated_bytes` and `seconds` cover the warmed
operation; process peak RSS includes Julia, package loading and compilation.
Concurrent host load makes these short timings descriptive, not a speedup
comparison.

Three probes separate the costs:

- `pattern_scaling.jl` measures only footprint deduplication and CSC union.
  Its `statistics_bytes` is just the small statistics tuple, not all metadata.
- `pattern_builder_scaling.jl` executes the complete, unchanged production
  `_brm_spl_pattern!` method in a minimal input container. It counts input cell
  storage and all retained index metadata, including linear `tptr`/`tz`/`rloc`,
  diagonals, prior maps and components. The source hash identifies the measured
  implementation. Repeated footprints test duplicate suppression; overlapping
  distinct footprints test retained structure separately from pair visits.
- `width_scaling.jl` calls the public planner on ordinary Gaussian cells using
  `dot` to preserve a genuinely coupled footprint. It passes no AD backend,
  sets the bound to 32, and asserts that all effects are sampled with no prepared
  conditional problem. This measures structural planning and generated query
  setup, not just the selection helper.

Complete production builder results (bytes, rounded to decimal MB):

| Shape | Width | Cells | Pair visits | Retained nnz | Allocated MB | Index metadata MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| repeated | 32 | 100 | 1,024 | 1,024 | 0.541 | 0.029 |
| repeated | 32 | 400 | 1,024 | 1,024 | 2.072 | 0.110 |
| repeated | 128 | 100 | 16,384 | 16,384 | 2.701 | 0.108 |
| repeated | 128 | 400 | 16,384 | 16,384 | 7.440 | 0.420 |
| overlapping | 32 | 400 | 409,600 | 26,161 | 5.221 | 0.120 |
| overlapping | 128 | 400 | 6,553,600 | 118,129 | 18.019 | 0.429 |

Every point retains zero per-cell quadratic pair-map entries. Width 128 with
400 repeated cells keeps 51,200 input incidences, a 263,336-byte CSC pattern
and a 1,122,952-byte complete fixture. Distinct overlapping blocks still visit
their required local pairs; those visits are not stored as duplicate triplets.
With a fixed local width bound, their work is bounded by that width times
footprint incidence. Storage follows input incidence and the retained sparse
pattern, including its diagonals and prior edges.

Public planning allocated 18.20/23.02 MB at width 64 with 100/400 cells, and
21.95/37.28 MB at width 256. All four plans retained zero integrated coordinates
and 66/258 sampled coordinates respectively; none prepared conditional AD.
Their isolated process peaks were 1.01–1.08 GB including compilation, while
the warmed planner took 0.033–0.046 seconds. The raw TSV retains each result
and its source hash.

To reproduce, run each parameter tuple in a fresh process inside the usual
compute-token wrapper (and use `JULIA_PKG_OFFLINE=true` when dependencies are
already present):

```sh
/usr/bin/time -v julia --project=test research/sparse_laplace_factors/pattern_builder_scaling.jl 128 400 repeated
/usr/bin/time -v julia --project=test research/sparse_laplace_factors/width_scaling.jl 256 400
```

The builder and microprobe matrix is `(width,cells,shape)` =
`(32,100,repeated)`, `(32,400,repeated)`, `(128,100,repeated)`,
`(128,400,repeated)`, `(32,400,overlap)`, `(128,400,overlap)`.
The public-planner matrix is widths `(64,256)` crossed with cells `(100,400)`.

These bounded fixtures do not establish full Atlas runtime, memory or fitting
speed. Sparse factorization can still incur fill-in, sampled targets can be
large, and raising `max_cell_width` permits genuinely dense local blocks.

### Connected SVI separators and weighted continuation

- `julia --startup-file=no test/nlme_svi_separators.jl`: bounded deterministic
  hub/chain policy, explicit overrides and atomic promotion budgets.
- `julia --project=test test/nlme_svi_separator_native.jl`: actual hub/chain
  repartition, full-density/native Reverse-gradient equality at nonzero points,
  automatic work sampling, optimizer continuation and structured streaming carry.
  A prepared model reuses its SVI target only under the same separator policy and
  budgets, a resume finds it again, and snapshots name the inference units.
- `julia --project=test --threads=2 test/nlme_svi_weighted_resume.jl`: exact
  trajectories for uniform/work batches and dense/structured local factors,
  lazy polishing, thread-count changes and a fresh-process checkpoint resume.
- `julia --project=test test/nlme_svi_cell_units.jl`: pure `cell_units`
  ownership on a hand-built layout: which subjectless cells share a unit, what
  may join them, `:first_unit`'s old layout, and cost linear in the supports.
- `julia --project=test test/nlme_svi_cell_units_native.jl`: the same on
  `research/svi_cell_units/toy.jl`'s sensor model: unit terms sum to the full
  density and native gradient, separators, zero-dimensional units, fits,
  resume from `:first_unit` checkpoints, streaming windows, row-counted `work`
  and the refusal under `:subjects`. About 15-20 minutes on a busy host.
- `julia --project=test --threads=2 test/nlme_svi_field_units.jl`: a read-pair
  field's states in units' own factors (`field_states=:units`). Unit and global
  terms add up to the whole model in value and gradient (`check_whole_model`).
  The public fit resumes to the same trajectory and is the same at one or two
  tasks. A held target is reused only under the same `field_states`, a resume
  finds it again, and a `:units` snapshot loads without BRM. Mesh fields and
  streaming windows are refused by name.

### A unit's whole series under one law; units under a known population law

- `julia --project=test test/nlme_svi_series_law.jl`: each station's series is
  one observation under a consumer-written law, an exact Kalman filter's log
  likelihood bound as a leaf of a whole-event observation provider. The filter equals the dense Gaussian of the
  readings, the RK target's native Reverse gradient equals central differences,
  each pointwise term is its station's oracle, and `brm_svi_fit` with the
  stations as units (factors of three statics each) is compared with NUTS on
  the same target. One station under a known population law (known means by
  `offset(...)`, every SD and the correlation known) is one unit whose block
  packs no scale or correlation coordinate: its density equals an independent
  oracle, its SVI fit is compared with NUTS, and a partly known block holds its
  known margin's SD. Two chained windows carry each station's filter state as
  data. The NUTS references make it the longest part.
- `julia --project=<NLMEEstimation scratch env> test/nlme_partly_known_blocks.jl`:
  a subject block that holds some of its values known, on a linear Gaussian
  model. Known scales under a sampled correlation are one family holding those
  margins' SDs at 1 (`FixedOmega(sd = ...)`, a known margin's η its effect over its
  known SD); a known correlation is a family at the identity with its sampled
  scales in θ. It checks conditional likelihoods and empirical Bayes modes against
  closed forms, `check_protocol`, the embedded Laplace hook on the family that
  holds an SD (its value against NLMEEstimation's own Laplace, its gradient
  against central differences, its marginals' sum against the exact marginal),
  a packed point
  read back, the population prior
  against the named densities over the free coordinates (value, gradient and that
  it integrates to one), FOCE(I) at the exact objective's stationary point and
  SAEM near it, `brm_nlme_joint_start`, `brm_svi_fit` against NUTS next to a known
  shared block, and that another held value is the same program.
- `julia --project=<NLMEEstimation scratch env> test/nlme_known_blocks_fit.jl`:
  NLMEEstimation on a linear Gaussian model whose block is fully known. The
  block is one `fixed_omega` family at the identity, the start and
  `brm_nlme_estimates` name its known values, conditional likelihoods equal
  the Normal oracle at η, `check_protocol` passes, the empirical Bayes modes
  equal the closed-form conditional modes, and FOCE(I) and SAEM fit the
  population with the block held.

## Reusable model view

`julia --project=test test/model_view.jl` checks declaration roles, scalar/vector/
matrix/higher ranks, zero-length axes, callable documentation and all three
semantic projections. It verifies that viewing a model never evaluates a custom
helper. The descriptor fixture checks constrained simplex dimensions and keeps
logical quantities separate from generated carriers, without fitting or compiling
Stan. Set `BRM_MODEL_VIEW_HTML` to an output path to save a browser preview.
The model-view files share `model_view_plain_text_checks.jl`: titles and tab
labels must occupy real lines, source remains multiline, default-input rows are
readable fields, and renderer implementation dumps are absent. These checks cover
builders, bound declarations, reusable components, descriptors and prepared RK
views, retaining the helper/default non-evaluation guards.

`julia --project=test test/model_view_edge_cases.jl` covers joint and transformed
response roles, callable documentation inside mixture-component lists, optional
component defaults and explicit empty-input states. Helper/default functions must
not run during any projection.

`julia --project=test test/model_view_rk.jl` checks the RK extension with RKPPL
0.1.93 or later: constrained simplex dimensions versus free-coordinate counts,
ragged/empty cells, matrix parameters, custom helper documentation, existing-build
wrappers without lazy binding, and both ordinary and data-free replay boundaries.
It builds toy models but does not prepare a density query or fit them.

`julia --project=web-macro web-macro/validate_model_view_export.jl <new-output-dir>`
exports the same view as a Quarto PDF and a static HTML site. It verifies the
PDF link and the documentation content without executing the fixture's helper.
Run `kb-playwright-python web-macro/validate_model_view_browser.py <output-dir>`
for real browser acceptance of bundled highlighting, touch/keyboard tabs,
documentation disclosures and the PDF download in light and dark themes.
The browser script owns and closes its temporary server and Chrome process.
For fresh RK, descriptor and component captures, also run
`julia --project=<rk-env> web-macro/validate_model_view_prepared_export.jl <output-dir>`
after the main export, then append `--prepared` to the browser command.
The current dependency floor includes Julia-documentation projection and the
shared Julia doctest grammar. Both the base and prepared bundles must pass strict
export and browser checks; the base checks remain independently runnable.
The RK environment must contain BRM, HTMXObjects, ReactiveKernels and
ReactiveKernelsPPL. This explicitly prepares toy objects, then exports the
supplied objects with helper evaluation guarded off. Browser checks cover
ragged element shapes, constrained simplex carriers and optional input rows.
