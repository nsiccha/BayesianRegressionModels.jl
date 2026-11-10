# Changelog

All notable changes to BayesianRegressionModels.jl are documented here, newest
first.

Every landing that adds or changes public API (an export, a macro, a keyword, a
method, or documented behaviour a dependant can rely on) bumps the patch digit
in `Project.toml` and adds its entry under a new heading at the top of this
file. An entry that breaks something a dependant relies on says **Breaking:**
and what to change. A package or app that relies on an entry states its version
as the floor, for example `[compat] BayesianRegressionModels = "0.2.2"`. An
environment that pins an older BRM commit is then refused when it resolves
instead of failing at run time.

## 0.2.262

- Added: a data-only assignment, a formula column and a model-scope
  expression apply the smooth Base functions `sin`, `cos`, `tan`, `sinpi`,
  `cospi`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `log2`, `exp2`,
  `cbrt` and `abs2` per element, as they already did `log`, `exp`, `sqrt` and
  `abs`. A cyclic covariate from a period held as data,
  `cycle = sinpi(2 * day / period)`, is now a data column at every site that
  takes one, on RKBRMI as on SBBRMI. Before, RKBRMI refused it as "not a data
  column". RKBRMI computes it in its graph, so `jax_source` puts it in the
  module's `prepare_data`.
- Fixed: RKBRMI writes a negated data-only expression (`exp2(-abs(t))`) as
  `-x`. Before, it wrote the dotted `.-x`, which ReactiveKernelsPPL refuses
  ("function (.-) not in the slice-1 assignment allowlist").
- Changed: an `mi()` column without missing entries emits the same RK program
  as one with gaps, its missing block over an empty row axis, so its missing
  count is data (user principle `0ikeidi`). A fit with gaps and a design
  without them now share one program and one `jax_source` module. Densities,
  gradients and coordinates are unchanged: the empty block adds no coordinate
  and contributes 0.
- Added: every definition RKBRMI emits into its RKPPL program carries a
  docstring that says what it computes. This covers component submodels
  (`brm_population_effects`, `brm_group_effects`, `brm_hsgp_effect`, …),
  blocks (`brm_cdar`, `brm_differenced_ar1`, `brm_gmrf`, `brm_car_icar`,
  `brm_missing_covariate`, …) and statistical helpers (`brm_prepared_indices`,
  `brm_covariate_mean`, the HSGP basis graphs, …). The text is written where
  BRM builds the definition's body, and the record names that line of BRM's
  source. `ReactiveKernelsPPL.rkppl_documentation(rk_bound_plan(backend))`
  now holds an `owner = :definition` record for each instantiated submodel,
  at its instance path (`(:pop_mu,)`), beside the authored declarations'
  records. ReactiveKernels' JAX export renders these records as the
  docstrings of the module's functions. The printed program shows the
  docstrings. Program identities (`_rk_program_statements`, fitted-state
  replay, SVI and SMC checkpoints) read the program without them, so they
  are unchanged, as are densities, gradients and coordinates.
- Tests: `test/rk_jax_models.jl` and `test/rk_jax_inference.jl` check RKBRMI's
  `jax_source` modules in a real Python process against the native model.
  The test environment pins ReactiveKernels `main` `ddc30859` (RK 0.1.130 /
  RKPPL 0.1.140), which carries RK's JAX lowering.

## 0.2.261

- Fixed: two bindings of one program now give `brm_nlme_model` the same subject
  cell groups in the same order, so a later binding reuses the code an earlier
  one compiled. Before, the groups followed the data. A plate observation whose
  density count equalled the subject grouping's row count was read by those rows
  and joined the plate group, which went first; in a binding where the counts
  differed it was measured and became a group of its own, after the observations
  before it. The groups' order set the partition keys' order (`brm_nlme_unit_<g>`
  and the partition's `by`), so the model's type changed and the later binding
  compiled its NLME model and first SVI fit again. On the test's toy, in one
  process, the second binding compiled 2.7 s preparing the model and 8.0 s in its
  first `brm_svi_fit`; it now compiles 0.01 s in each. A downstream report, on
  BRM 0.2.242, measured 118.9 s of compilation for such a second binding (76.7 s
  with a pending ReactiveKernels fix for the same order on its side), against
  1.1 s for one whose plates were empty in the same places.
- The groups are now the program's: the observations of one authored
  `@plate for` cell (with the series its `@scan`s observe, and the formula
  observations of its output observed per subject) share one group, measured or
  not; every other observation is a group of its own; the groups follow the
  program's pointwise order. A group whose observations attribute their cells
  differently is split into one group per observation. Two `@plate for` cells
  over the subjects are now two groups (before, every plate observation read by
  rows shared one), so a nested model's structural reads prepare one cell query
  per plate, and its η Hessian patterns are those of each plate's own cells.
  Values, gradients and attribution are unchanged.
- `test/nlme_binding_cell_groups.jl` covers both bindings: the same groups,
  order, partition keys and model type, and lockstep, per-subject and unit
  values against the full density.

## 0.2.260

- New, **a stopgap**: in minibatch SVI (`brm_svi_fit` and its resume), a global
  whose prior reads no other coordinate no longer drifts far wider than that
  prior. Adam steps a coordinate by about its step size whatever its gradient
  noise, so where the noise dwarfs the prior's pull the factor's scale and rank
  row random-walk, and where the gradient grows with the draw (a log length
  inside a random-feature phase) a wider factor meets larger noise: one such log
  length under a N(-1.386, 0.5) prior reached a marginal sd of 378 in
  exploration and overflowed the density. Its marginal sd under q, `√(exp(2ρ_k)
  + |Β_k|²)`, is now held at most `sd_bound` (a new keyword, default `2`) times
  the sd of the Gaussian closest to its prior (the factor a fit on no data would
  reach). This treats the symptom: the optimizer still does not see the prior
  through that noise, the mean can still wander, and where a posterior is wider
  than the bound (a heavy-tailed prior under prior-data conflict, a periodic or
  multimodal likelihood) q is held too narrow. So a fit that ends with a global
  at its bound warns and names it, `diagnostics.sd_bound` reports the `factor`,
  each global's `bound` and those `at_bound` (with `stopgap = true`), and
  `sd_bound=nothing` turns the bound off. Globals whose prior reads others
  (centered field states, their lengths) and flat priors have no bound. A fit
  whose factor stays inside the bound is unchanged to the last bit, checkpoints
  keep their layout, and a checkpoint written before this version resumes with
  the default bound. A global whose prior sd is below half of `global_scale`
  (0.1 by default), where every global starts, is at its bound from the first
  step on instead of shrinking there over hundreds of iterations, so such a fit
  differs from its first step: on one model a coordinate with prior sd 1e-6
  still had sd 0.008 after 2,000 iterations.
- Fixed: every `GaussianBlockLaplace` fit failed while building its optimizer
  state, with `MethodError: no method matching get(::_BRMSVILaplaceTarget,
  ...)`, since the unit-owned field states of `field_states=:units` (BRM
  0.2.237).

## 0.2.259

- Changed: `brm_svi_fit` gives a `@plate` cell latent's entries
  (`x ~ Normal(loc[i], s)` in a cell, one draw per cell or per entry) to the
  unit whose cells read them, as it already did a cell component's locals.
  Before, every unit's footprint read every cell's latent block, so the latents
  were globals of the fit (exact, but not per unit). This needs
  ReactiveKernelsPPL 0.1.127 (ReactiveKernels snags `rkppl-dotted-pla-1629a747`
  and `dotted-cell-late-3789b8ea`) and, beside a `gmrf(...; neighbors=)` field,
  ReactiveKernels 0.1.120 (snag `derivativerule-r-0d7e245c`), so the floors rise
  to ReactiveKernels 0.1.120 and ReactiveKernelsPPL 0.1.128. On Julia 1.10 these
  need Enzyme dogfood `bcefef27` or later, without which `brm_svi_fit` can abort
  the process (`LLVM ERROR: function failed verification`, Enzyme snag
  `native-reverse-p-1a4ac06d`); the test environment pins dogfood `9b573c63`,
  whose libEnzyme `48fe7ef5` is built for Linux x86_64 only. A cell latent under a transformed law
  (`LogNormal` and other positive or bounded families) still stays global, since
  ReactiveKernels reads its constrained block whole. A cell latent whose law
  reads a value the model computes per cell is still refused by name with its
  rewrite.

## 0.2.258

- New: `brm_svi_fit(x; centering=LearnedCentering())` also centres the weights of
  a grouped HSGP whose levels are the units (`hsgp(t; by=subject)`): each level's
  standardized weights are cells located at 0 and scaled by the term's spectral
  scales, per level when a hyper-predictor scales `sd` or `length_scale` by
  level. `fit.diagnostics.centering.cells` lists them with `kind=:hsgp`. A mode
  whose scale underflows is floored in log scale, as WarmupHMC's adaptive HSGP
  centering floors it, so its frame stays finite; the floor moves coordinates,
  not the target.
- An HSGP cell's control starts from `min(start, 0.1)`; other cells keep
  `LearnedCentering(; start)`. A high mode's log scale moves steeply with the
  length scale, so from 0.5 a centred weight swings with every draw of it and
  the ELBO answers by collapsing q(length scale). On a 12-unit toy (6 units with
  40 rows, 6 with 3, against 4×1000 NUTS draws) that left the length scale 1.7
  NUTS sds off with 0.02 of NUTS's sd, 5–6 nats below the plain fit. From 0.1,
  over three seeds and against centring the `(1 | subject)` cells alone, the
  length scale's mean moves from 0.52–0.58 NUTS sds low to 0.14–0.25, the HSGP
  `sd`'s from 0.83–0.86 to 0.54–0.69, the weights' rms error from 0.16–0.17 to
  0.10–0.13 NUTS sds, and the ELBO rises by 0.2–0.4 nats. The length scale's sd
  narrows, from 0.31–0.33 of NUTS's to 0.22–0.25.
- Still uncentred, and listed in `uncentered` with the reason: levels correlated
  by `cor=` (`cor=Omega`, `multigroup`) and an HSGP's weights shared by the units,
  which are global coordinates. `brm_svi_centeredness` hands WarmupHMC random-effect
  cells only; its HSGP rows keep their current values.
- **Breaking:** every HSGP term's emitted program changes, because its spectral
  scales are now a value of the program, `<id>.spectral_scale`. Values and
  gradients are unchanged, but an existing SVI or SMC optimizer checkpoint or
  streaming state of an HSGP model is refused, as for any changed program, whether
  or not it uses centering. Finish such a run on its own version or restart it.

## 0.2.257

- Added: `brm_svi_combine(x, slices; exclude)` leaves named global coordinates out
  of the product of the slices' sites. Each starts as a coordinate no slice
  informs, at the full model's default start (the prior-median point with the
  fit's `global_scale`), exactly as if every slice had marked it uninformed, its
  joint sites included, and draws are carried to that default where it is one of
  their scales. It is for a coordinate whose meaning depends on the slice's
  domain: a spatial field's amplitude fitted on one region measures the field's
  variation inside that region, so regional slices all put it low, and the
  between-slice variance and `spread`, which compare the slices with each other,
  cannot see a bias they share. On a downstream
  all-Germany model the 16 regional slices put such an amplitude at log −7.6
  against a prior projection of −1.2, and the full fit from that start climbed
  back only to −3.9 against −2.4 and −2.1 cold, with a higher ELBO but worse
  held-out prediction at new stations. A name selects that coordinate and every
  coordinate under it (`exclude=[:natural_field]` selects `natural_field.amplitude`
  and `natural_field.z.3`); a function of a name selects where it returns `true`.
  A name of no global coordinate, or a selection of nothing, is refused.
  `receipt.globals.excluded` marks them; `informed` still counts their slices'
  sites. Without `exclude` the start is unchanged.

## 0.2.256

- Added: `brm_nlme_view`, and so `brm_nlme_model`, `brm_nlme_fit`,
  `brm_two_stage_fit`, `brm_svi_fit` and `brm_nlme_joint_map`, reads a subject
  block that holds only some of its values known. They refused it before, naming
  BRM todo `0840h34`.
  - **Known scales under a sampled correlation** (`sd(a, p) = input` next to
    `sd(b, p) ~ Exponential(...)` and `cor(:, p) ~ LKJCholesky(...)`) are one
    covariance family that holds those margins' SDs, NLMEEstimation's
    `FixedOmega(sd = [1, nothing])`. The free SDs and the correlation are estimated,
    and SVI keeps them as globals and centres the block as it does an estimated one.
    A known margin's η, in `H`, `omega_matrix` and the EBEs, is its effect over its
    known SD, so its variance there is 1. Population coefficients stay mu-referenced
    through it. `population_logprior` is the authored prior over the free coordinates
    `(log sd, m)` that the estimators search.
  - **A known correlation next to sampled scales** is one family held at the
    identity, whose standardized draws are η. The sampled scales are
    `:shared_latent`, so θ in NLMEEstimation and globals in SVI; they stay fixed when
    the model gets a `shared_point`. No population coefficient is mu-referenced
    through it, and SVI leaves its draws uncentred. SAEM moves those scales by its
    numerical M-step alone. On a 30-subject linear Gaussian model at the default
    options its estimate's log marginal likelihood ended 0.10 to 0.97 units below
    that of FOCE(I)'s estimate, which sits at the exact log posterior's maximum, over
    four seeds (0.006 to 0.43 at 1600 iterations over three), against 0.01 to 0.12
    for known scales under a sampled correlation.
  - `brm_nlme_estimates` names such a block in natural units with `known_sd` (each
    margin's known SD, `nothing` at a sampled one) and `known_cor = true`, as they
    apply. Its `sd` and `cor` go back through `brm_start`.
  - A Student-t block holding known values stays refused: `brm_nlme_model` covers no
    Student-t block.
- Fixed: on RKBRMI, a known scale (`sd(...) = value`) is a generated input of the
  program instead of a literal in its source (user principle `0ikeidi`), as a known
  correlation's factor already was. Another value is the same emitted and lowered
  program, on subject blocks and on `shared_groups` blocks. A frozen replay with
  another value is refused as another fit.

## 0.2.255

- Fixed: under `unit_curvature = :orthogonal`, a dense unit's step takes the
  positive part of its curvature estimate. On a saturating or redescending term the
  estimate is mostly noise with nearly symmetric eigenvalues until the unit's stiff
  directions are resolved: the trust region shortened the step from their size, and
  the negative part halved the precision along a new direction each visit. A unit's
  factor could become far wider than its target in a direction, and its stiff
  directions came slowly. A saturating toy whose precision off its stiff subspace is
  exactly 1 fell to 3e-4 and stayed there, with a top eigenvalue of 75 against the
  target's 1.65e8; now it stays at 1 and reaches 1.65e8. On a downstream target of 67
  stiff units of up to 17 coordinates, a unit's smallest precision eigenvalue at
  update 60 was 0.013 where the estimate read 223 along that direction on average;
  now it is 0.6, and the largest eigenvalue at update 100 is 26,670 against 3,045.
  The estimate is exact on a quadratic term and has no negative part there, so such
  a unit's step is unchanged to the last bit. A direction whose curvature really lies
  below the factor's still widens, by up to half per visit. Against Gaussian optima
  known exactly, at 17 coordinates, no visit's estimate had a negative eigenvalue on
  Student-t (3 and 5 degrees of freedom), logistic or Poisson terms, so those steps
  are unchanged; a Cauchy term, negative on 45% of visits, moved the rms sd ratio
  from 0.951 to 0.950. The step adds one eigendecomposition per visit beside its 2d
  unit gradients. A fit with dense units whose estimate goes indefinite differs from
  earlier versions from that visit on. A checkpoint written before this resumes at
  the step it ran with.
## 0.2.254

- Fixed: `brm_svi_combine` no longer turns slices that contradict each other into a
  start the full fit cannot run from. The joint product put means far outside anything
  the slices or the prior support, and both products held a contradicted coordinate as
  narrow as the narrowest slice, where few slices put it. Ridges still cross where they
  did, and slices that do not contradict each other and hold no rank part give the
  start they gave.
  - A coordinate's sites widen by the slices' between-slice variance, as in a
    random-effects meta-analysis, in the joint and the coordinate-by-coordinate product
    alike, so the coordinate sits among the slices, and it starts no wider than a cold
    fit's default 0.1. It is a Paule-Mandel
    estimate over the slices that pin the coordinate by themselves, where Cochran's Q
    says they contradict (at 0.999, allowing for sds a quarter of the posterior's, as
    SVI's can be), and `receipt.globals.heterogeneity` reports it. A
    slice pins a coordinate if it narrows it (marginal variance below 0.8 of the prior
    projection's) and the rank part does not carry it (below 0.8 of that variance), or
    if its marginal variance is below 0.01 of the prior projection's.
  - A slice's joint site holds only the coordinates it narrows, or carries along a
    ridge (below 1.2 of the prior's variance, with at least 0.8 of it in the rank part).
    Other coordinates have a marginal site where they are narrower than the prior and
    none otherwise.
  - A joint mean outside the range that supports it (from the lowest to the highest
    of 3 sds around each counted site's mean and the prior's) makes the
    coordinate-by-coordinate product stand, with `receipt.globals.joint === :outlying`
    and `receipt.globals.outlying` marking the coordinates.
  - A unit starts from its carried draws unless its own density at the combined globals
    is clearly better (by over half a nat per draw) with its slice's draws as they
    were, and `receipt.units.uncarried` lists those. Carrying keeps a unit's totals, which
    can take draws no prior allows: a block whose scale a model pins at 1e-6 would need
    a draw of 1,000 for a location change of 0.001.
- Changed: `globals.correlated` holds only the coordinates in joint sites, so
  `joint_limit` now bounds those and not every informed global, and `joint_limit=0`
  still asks for the coordinate-by-coordinate product. A site's directions wider than
  the prior are now those of the prior-relative eigenproblem, which a rescaling of the
  coordinates leaves alone, and `clamped` follows them and the marginal sites.

## 0.2.253

- Fixed: `brm_svi_fit`'s structured units (above `unit_dense_limit`) default to
  `unit_step = 0.1` under either `unit_curvature`. BRM 0.2.248 through 0.2.252 gave them
  the dense units' default of 1 under `:orthogonal`, though the entry for 0.2.248
  called them unchanged: their step is Adam's step size, not a natural-gradient step
  to take whole. A step you pass still applies to both families.
- Changed: a structured unit's rank part steps a tenth as far as its mean and log
  scales, the rate of the global factor's rows at the default steps. At the unit's
  whole step Β's rows walk past the marginal the target wants, ρ falls below them
  to make up for it, and the rows' own step, in units of exp(ρ), falls with it, so
  they cannot come back. A rank-8 factor then ended below the diagonal it contains.
- Measured on one station's unit of 295-304 coordinates under a known population
  law, a whole series under a Julia leaf, three stations, ELBO of the returned factor
  over 1,000 draws. Before, rank 8 stopped with `BRMSVINonfiniteError` at iterations
  569, 802 and 2,128 on draws 571, 599 and 2,683 prior sds out, and mean-field
  converged. At `unit_step = 0.1` alone rank 8 converged at 343, -204 and 425 with
  marginal sds of 8, 31 and 4 under an N(0, 1) prior, against mean-field's 481, 455
  and 470. With both changes rank 8 reads 486, 466 and 476 with every sd near 1. On
  exact Gaussian units with a low-rank covariance the rank part at this rate beat
  the best diagonal in all 7 settings measured (500 to 8,000 visits per unit), and
  at the whole step lost to it in 6 of them.
- Changed: a fit of a model with a structured unit differs from its first visit of
  such a unit, its rank part from then on in every version, its mean and scales
  against 0.2.248 through 0.2.252. Mean-field fits read higher ELBOs too (one station's
  trimmed window 62-68 against 20-34). Dense units and checkpoints are unchanged: a
  checkpoint written before this version resumes at the steps it ran with.
- Added: `BRMSVINonfiniteError`'s `terms` carry the evaluated points. Each term has
  `points`, packed as a fit's `draws` (`plus` and `minus` for a unit, `draw` for the
  global term), so `LogDensityProblems.logdensity_and_gradient(problem, term.points.minus)`
  replays the evaluation in the model, and `largest`, its largest coordinate as
  `(; coordinate, value)`, which the message prints. Under `centering=LearnedCentering()`
  the points are the packed coordinates, not the factor's learned ones.

## 0.2.252

- Fixed: `brm_nlme_view`, and so `brm_nlme_model`, `brm_nlme_fit`,
  `brm_two_stage_fit`, `brm_svi_fit` and `brm_nlme_joint_map`, reads a
  random-effect block declared inside a function-style component call
  (`kin ~ kinetics(subject)` with `(1 | pk | subject)` in its body) as the same
  block at model scope, over the column the call passes for the grouping
  input. That column is one of the model's groupings: name it as
  `subject_group`, in `nested_groups` or in `shared_groups` like any other.
  The block's `|ID|` and margins carry the call's path (`kin.pk`, predictor
  `kin.CL`), as its coordinates already did (`kin.b_pk_subject.z`); nested
  calls compose their paths, and an input named otherwise than the model's
  column is followed to it. Every coordinate of a component call was a
  `:shared_latent` before, so such a grouping was no grouping of the model
  (naming it in `shared_groups` was refused: "no random-effect block ...
  groups by") and a model scoping its subject effects in a component had no
  subjects. The
  call's other sampled locals, its population coefficients included, stay
  `:shared_latent`; so does a block whose grouping input is passed an
  expression, or whose call reads other levels of the column than the model.
  Prior-median starts read the blocks' scale laws with the call's own data.
  **Breaking:** a component grouping other than the subject grouping must now
  be named in `nested_groups` or `shared_groups`; `shared_groups = (:g,)`
  restores the former, all-shared reading of its block.
- Fixed: `brm_coordinate_transport` and `brm_start(...; component_inits)` with a
  formula key read a random-effect block declared inside a component again.
  Both died with `type NamedTuple has no field fitted`.

## 0.2.251

- Changed: an `RKBRMI` keeps the `BRMNLMEModel` that `brm_svi_fit`,
  `brm_svi_resume` and their slice and Laplace forms prepared from it, the last
  one. A later call on the same backend with the same `ad_backend`,
  `shared_point`, groupings and `unit_partition` takes that model, and with it
  the target and worker samplers it holds, as a call on `diagnostics.model`
  does: `diagnostics.target.reused` is `true` and nothing is prepared
  (`laplace=GaussianBlockLaplace()` reuses the model and still prepares its
  target on every call). Before,
  each call on an `RKBRMI` prepared the model and its target again: on a
  downstream 33-unit target a repeated 20-update call took 58 s, and takes
  0.18 s now.
  Results are unchanged: a reused model's run equals a fresh model's to the
  last bit. Other options prepare another model, which the backend then keeps
  instead. A copy or a serialized backend keeps none. A `BRMI` is still
  lowered, and so prepared, on every call.
- Changed: an `RKBRMI` builds its authored density-factor catalogue (RKPPL
  `density_factors`) once, on first use, for every reader: `brm_nlme_view`'s
  scan of shared factors that read a unit and each minibatch SVI target's
  ownership. Each built it again before, planning the whole program each time
  (about 5 s warm on a 33-unit model).
- Changed: `brm_nlme_model` prepares the whole-model sampler that retains the
  pointwise densities, and each subject-cell group's cell query, the first time
  something reads them, instead of while preparing the model. The lockstep
  gradients (`brm_nlme_loglikelihoods_and_gradients`) and the per-unit Laplace
  read the sampler; a nested model's η sparsity and a refused partition's
  diagnosis read the cell queries. A unit's values and gradients, minibatch
  SVI's included, come from the observation partition, so a fit that reads
  neither prepares neither. On a downstream 33-unit, 890-coordinate model the
  two steps took 75 s and 84 s of a 413 s cold preparation, but most of that
  was compiling planner code the partition steps then compile instead: the
  cold `brm_nlme_model` fell from 471 s to 440 s, a warm rebuild by about 10 s
  of 23 s, and its allocation from 16.1 GB to 12.5 GB. The progress tree no
  longer has a `Cell queries` counter. Values, gradients and refusals are unchanged; an
  `ad_backend` the sampler cannot use is refused when the sampler is first
  read.

## 0.2.250

- New: a subject block whose every scale and whose correlation are known
  (`sd(...) = value` on every margin, and `cor(:, ID) = R` or one margin) has
  an NLME reading. `brm_nlme_view` reads it as one covariance family held at
  the identity (NLMEEstimation's `FixedOmega`): its standardized draws are η
  and the model maps them to the effects with the known `diag(sd) L`. So
  `brm_svi_fit` and NLMEEstimation's estimators fit a unit under a known
  population law (known means through `offset(...)`), and
  `brm_nlme_estimates` names the block with `known = true` and its known
  `covariance`, `sd` and `cor`. Before, the view refused every known-value
  block. A block that mixes known and sampled values (or a Student-t block
  with known values) is still refused, now naming what is sampled (todo
  `0840h34`).
- Tested: a unit's whole series under one consumer-written law, a filter's
  log likelihood bound as a leaf of a whole-event observation provider, is
  one observation of its unit under `brm_nlme_view` and `brm_svi_fit`
  (`test/nlme_svi_series_law.jl`, snag `odlplaus-margina-60bc212b`).

## 0.2.249

- Fixed: `brm_svi_fit(...; centering=LearnedCentering())` centres the levels of a
  random-effect block over a `shared_groups` grouping whose draws one unit owns,
  as it does a unit's own block, and lists the others in
  `fit.diagnostics.centering.uncentered` with the reason. Such a block was in
  neither `cells` nor `uncentered` before. A level's cells are located at the
  population coefficients its margins absorb (`(1 | device)` takes the
  intercept unless a units' block already does), at 0 without one. A level two
  units read is global (`"its draws are not one unit's"`), and Student-t,
  level-correlated (`gr(g; cor=...)`) and known-scale shared blocks are listed
  with their own reasons. A streaming window maps an absent carried unit's
  centred level through that unit's controls. A centred fit of a model with such
  a block differs from its first step on; models without one are unchanged.

## 0.2.248

- Changed: `brm_svi_fit`'s dense units (at most `unit_dense_limit` coordinates)
  estimate their whole curvature on every visit. A visit used to evaluate the
  unit at one antithetic pair of draws, whose Stein estimate is a rank-one
  product, and move the precision a tenth of the way (`unit_step = 0.1`) along
  that one direction; a unit of tens of coordinates then stopped unsettled after
  the visits a minibatch run gives it, and the hyperparameters followed its
  factor. Now a visit evaluates d mutually orthogonal antithetic pairs (2d unit
  gradients for d coordinates; a randomly rotated spherical-radial cubature
  rule), whose Stein estimate is the unit's expected curvature, exact when its
  term is quadratic in its coordinates, and takes the whole natural-gradient
  step (`unit_step` now defaults to 1), inside the same trust regions.
  Structured units (above `unit_dense_limit`) are unchanged.
- Measured on an OU chain cut into window units, with exact oracles. 128 units
  of 66 coordinates (about 500 visits each): the length scale ℓ came out 48.0
  (states as latents) and 65.6 (states from innovations), against 1.4 and 11.4
  before and about 61 at the family's optimum. 2,048 units of 64-65 coordinates
  (about 72 visits each): 18.2 and 20.5 against 0.6 and 18.1 before, 802 and 227
  nats below the family's optimum against 24,756 and 1,440; the rest of that gap
  is a ridge between the hyperparameters and the windows' states, which a unit's
  step does not cross. A pure test with 32 units of 48 coordinates whose prior
  scale is the one global: log σ 0.0345 against the optimum 0.0334, the units'
  factors 0.002 nats from their optimum in all, where the single pair ended at
  0.507 and 100.6 nats.
- Added: `unit_curvature`, `:orthogonal` (default) or `:pair`. `unit_curvature =
  :pair` with `unit_step = 0.1` is the earlier single-pair step exactly, and `unit_step`'s
  default follows `unit_curvature`.
- Changed: a dense unit's visit costs 2d unit gradient evaluations instead of 2,
  and `diagnostics.unit_gradient_calls` counts them. On the toys above a fit took
  4 to 13 times as long. A fit on a fixed seed differs from the single pair's
  from its first dense unit visit. A checkpoint written before this version
  still resumes, with the single pair: it records no `unit_curvature`.
- Unchanged: a sparse global's reader sites keep their precision step 0.1 under
  `:orthogonal` (their curvature still comes from the one global draw; the unit
  step 1 put a window toy's boundaries 88 optimal sds off). With
  `centering=LearnedCentering()`, each of a dense unit's 2d draws is the centered
  term and the controls' gradient and the location and scale adjoints are averaged
  over them.

## 0.2.247

- Fixed: a `LowRankGlobal(r)` fit's rank part steps no faster in settling and
  polishing than in exploration. Both phases step at `polish_lr` (by default five
  times `lr_global`), settling at that size for up to `settle_windows` windows,
  and Β's gradient is minibatch noise in every phase, so its rows random-walked
  again: on a model with 1,094 globals 4 of 10 low-rank fits evaluated non-finite
  draws while settling (three within its first five windows) and a fifth while
  polishing, and with `settle_windows=0` one did in polishing. Β's rows now step `min(1, lr_global / polish_lr)` times as far as μ
  in units of their coordinates' scales. μ, ρ and `MeanFieldGlobal` fits are
  unchanged; a `LowRankGlobal` fit on a fixed seed differs from its first
  settling step on.

## 0.2.246

- New: learned partial centering in minibatch SVI,
  `brm_svi_fit(x; centering=LearnedCentering())` (Gorinova, Moore and Hoffman
  2020's VIP). Each eligible cell gets a control `c ∈ (0, 1)`, optimized with
  the variational factors by the ELBO: `c = 0` is the non-centred coordinate,
  `c = 1` the centred one. Cells are the standardized draws of estimated
  random-effect blocks a unit owns (located at the unit's mu-referenced
  population coefficients; a correlated block is centred margin by margin on
  its conditional mean) and authored Gaussian vector elements a unit owns
  (`z[1:n] .~ Normal(m, tau)`, located at the law's mean). The map is
  ReparametrizableDistributions' `reparam`; scales, correlation factors and
  law arguments stay RK graph values, read once and pulled back once per
  iteration. Each unit's controls take an Adam step (`LearnedCentering(;
  start=0.5, step=0.05)`) whenever the unit is in the batch, while SVI
  explores, and stay fixed while it settles and polishes. Draws stay packed RK points. `fit.diagnostics.centering` reports
  every cell's `c` and what stays uncentred, with the reason: Student-t,
  fixed-covariance and field families (HSGP weights, `cdar`, `car`,
  completions) and shared field weights. Omitting `centering` keeps the
  existing fit.
- New: `brm_svi_centeredness(fit, adaptive_centering_controls(adaptive, sb))`
  returns a centered fit's controls as `adaptive_centering_problem`'s
  `centeredness` vector, a start for WarmupHMC's own adaptation.
- A centered fit's `brm_svi_checkpoint` stores each unit's controls, and a
  streaming update inherits the centering (pass no `centering`). A carried
  unit's prior is its factor through the old controls' map, so relearned
  controls do not reinterpret old factors; `brm_svi_resume` continues a
  centered optimizer checkpoint.
- `brm_svi_combine`, and a factor-level `start=`, refuse a centered fit or
  checkpoint: its unit factors are over the learned coordinates, whose map to
  the packed draws moves with the globals, so they are no site of those draws.
  Fit slices to be combined without centering.
- Compatibility: depends on ReparametrizableDistributions 0.3 (git source).
- Learned centering and reader sites (0.2.240) compose: a sparse global's
  site stands for its reader's whole centered term, so the reader's gradient
  there includes the chain through its cells' locations and scales (the sd of a
  `gr(g, by=b)` stratum few units belong to, say). That chain term reaches every
  other global through the batch's one pullback, as before.

## 0.2.245

- Fixed: a `LowRankGlobal(r)` fit's rank part no longer random-walks during
  exploration. Its rows took constant Adam steps of `lr_global` in absolute
  units, the same for every coordinate, while their gradient is mostly
  minibatch noise, so over a long exploration (a large `min_windows`) they grew
  to several times their coordinates' own sds: the factor came out wider than
  the posterior, its trimmed ELBO fell, and on a model whose log-scale
  hyperparameters overflow a few sds out, typical draws reached a non-finite
  density and the fit stopped with `BRMSVINonfiniteError`. Each Β row now steps
  in units of its coordinate's scale `exp(ρ_k)` in exploration too, as it
  already did in polishing. A ridge a slice's data leave to the prior therefore
  grows over more iterations: fit `LowRankGlobal` slices for `brm_svi_combine`
  with a longer exploration (a larger `min_windows`) than the default 40 windows.
- Fixed: a `LowRankGlobal(r)` fit no longer breaks down when its
  scale-relative steps start (settling, then polishing). They moved μ in units of
  a square root of the whole global covariance, and the restarted Adam's first
  steps, each about its step size in every whitened coordinate, added up along
  the rank part's directions. On a model with 1,094 globals the trimmed ELBO fell
  by two thirds within a few hundred iterations of the switch (measured on 0.2.241's
  schedule, where polishing restarted Adam), and draws went non-finite. μ now moves
  in units of its diagonal scale `exp(ρ_k)` when settling or polishing a low-rank
  factor.
- Unchanged: `MeanFieldGlobal` fits, to the last bit, and μ and ρ in a
  `LowRankGlobal` exploration. A `LowRankGlobal` fit on a fixed seed differs from
  its first exploration step on. A low-rank optimizer checkpoint written by an
  earlier version resumes and continues under the new steps.

## 0.2.244

- Added: the NLME paths take a model whose `@plate` cell declares a latent
  (`x ~ Normal(loc[i], s)`, one draw per cell or per entry) or calls a
  function-style component, whose sampled locals are per cell. `brm_nlme_view`
  reads those coordinates as `:shared_latent`, the reading a top-level vector's
  entries get (`x[1:n] .~ Law`, read at the cell as `x[ids[i]]`): their law stays
  in the population prior, they enter θ (or are fixed by `shared_point`), and
  each unit's conditional likelihood reads its own cells' latents. Before,
  `brm_nlme_view` refused with "coordinate record kind `cell_latent` has no NLME
  role". `brm_svi_fit` gives a cell component's locals, with their prior
  factors, to the unit whose cells read them. A dotted cell latent stays a global
  of the fit for now: ReactiveKernels reports each unit's footprint as reading
  every cell's latent block (ReactiveKernels snag `rkppl-dotted-pla-1629a747`); the
  split is exact either way, and the same model with a top-level vector read as
  `x[ids[i]]` is owned per unit already. A cell latent whose law reads a value the
  model computes per cell (`x ~ Normal(m[i] .+ loc[i], s)`) is refused by name,
  since that law's reads cannot be localized yet; the message names the same
  model written without the read (`d ~ Normal(loc[i], s)`, then `x = m[i] .+ d`)
  (snag `brm-nlme-view-re-ed98959b`).

## 0.2.243

- Fixed: a model reading `hsgp(t; cov=:periodic)` in a plate cell stopped with
  `ArgumentError: Tuple contains 2 elements` when NLMEEstimation's
  `check_protocol` prepared its sparse Laplace, and `prepare_cell_factors` raised
  the same error. The repair is in ReactiveKernels 0.1.96. ReactiveKernelsPPL
  0.1.106 alone, the first with that repair, raises a `BoundsError` in
  `_refresh_cell_shared!` on the second call of a partitioned cell whose shared
  prefix reads only some shared coordinates, which NLMEEstimation's
  `check_protocol` reaches with integrated families; 0.1.107 repairs it. 0.2.239's
  floors (ReactiveKernels 0.1.104, ReactiveKernelsPPL 0.1.114) carry both, and
  `test/runtime_numeric_settings.jl` now checks that model's cell factors against
  its whole likelihood in the test environment.
- **Breaking**, from 0.2.239's ReactiveKernels 0.1.104 floor and in any
  environment on ReactiveKernels 0.1.96 or later: a compact cell's Laplace
  Hessian-vector product (Forward over Reverse) differentiates a matrix product
  that reaches BLAS `gemm!`. On an Enzyme before dogfood `0cdbe188` (libEnzyme
  `3a825a54`) it raises `EnzymeRuntimeException` ("The original primal code hits
  this error condition"), which stops `check_protocol` on a cell that sorts its
  data and on a censored ragged join beside an in-cell observation. Use Enzyme
  dogfood `0cdbe188` or later, as BRM's test environment now does (0.2.239 pinned
  `cecb571a`). That libEnzyme build is for Linux x86_64 only.
- Changed: 0.2.239's floors also guarantee the reuse 0.2.227 left to the
  environment. One lowered program serves every size and support bound
  (ReactiveKernelsPPL 0.1.108), and one gradient query type every value of a
  bound number (0.1.106), so an RK fit no longer recompiles when only such a value
  changes. The `hsgp` docstring and the support-bound comment in
  `src/rk_backend.jl` now say so.

## 0.2.242

- Added: `brm_svi_fit` settles before it polishes. Exploration's stop test finds
  the ELBO flat at exploration's own step, and that step moves a global mean by
  `lr_global` in the coordinate's units whatever its posterior sd, so a tight
  global (sd 7e-4 on a synthetic model) jitters by tens of sds and the ELBO sits on
  that noise floor. Polishing's first windows then climbed from it, and the
  returned Polyak average included the climb. Settling runs the scale-relative step
  `polish_lr` at a constant size, without averaging, until the same trimmed-ELBO
  test passes on its own windows, for at most `settle_windows` windows (new keyword,
  default 100). Polishing then continues that step and its Adam moments. On an
  exactly Gaussian model with 32 tight globals the returned factor came 62 nats
  from the family's optimum instead of 111, at 16% more iterations; with 8, 56
  instead of 75. On 400-unit synthetic models (3 seeds each) the returned
  factor's ELBO rose by 4.4 to 5.6 nats (Gaussian random effects), 0.3 to 1.2
  (Poisson) and 1.4 to 2.3 (a weakly read log-scale field), at 18% more
  iterations. `settle_windows=0` restores the earlier schedule exactly.
- Added: `diagnostics.stages` (and in snapshots): the iterations spent exploring,
  settling and polishing, and `settled`, whether settling ended on its test rather
  than its cap. `converged` with `settled = false` means polishing averaged
  iterates that were still rising. A snapshot's `phase` can read `:settling`.
- Changed: a fit on a fixed seed differs from 0.2.241's from the window at which
  exploration stops. A checkpoint of 0.2.240 or 0.2.241 still resumes, with its
  own schedule (no settling): it records no `settle_windows`.

## 0.2.241

- Changed: `brm_svi_combine` multiplies the slices' global sites jointly. A
  slice's site is the inverse of its global factor's covariance over the
  coordinates its data reaches, `diag(s²) + ΒΒᵀ`, less the prior's projection, so
  a `LowRankGlobal` slice's rank part enters as correlation, where it used to
  enter only through the marginal variances. A ridge in a slice (a direction its
  data leaves to the prior, such as a level two coefficients share) keeps a small
  precision along it, and the product lands where the slices' ridges cross. A
  site's negative part is dropped by eigen-direction, the joint form of the
  dropped coordinate site. `MeanFieldGlobal` slices hold no correlation and give
  the same start as before. Above the new `joint_limit` (default 4096) informed
  globals the product stays coordinate by coordinate.
- Added: a `BRMSVIStart` keeps the product's precision where it is correlated
  (`globals.correlated`: `positions` into `globals.keys` and a dense
  `precision`), and `globals.σ` is its mean-field part `1/√P_kk`. A
  `LowRankGlobal(r)` fit from it starts its rank part on the `r` directions in
  which that precision is widest relative to `σ`, where it used to start at zero.
  A shared level's carried draws keep their level's product precision too; their
  `σ` is now `1/√P_kk` as well, which is unchanged for one-margin blocks. A start
  combined by an earlier BRM still starts a fit, with a zero rank part.
- Added: `receipt.globals.joint` (`:joint`, `:diagonal` when no slice has a rank
  part, `:limit`, `:indefinite` when the joint product was not positive definite
  and the coordinate-by-coordinate one was used) and `receipt.slice_rank`;
  `clamped` counts dropped site directions, each at the coordinate it weighs most.

## 0.2.240

- Fixed: `brm_svi_fit` left globals that few units read far from the family's
  optimum: a time window's boundary states, a read-pair field's states, a
  separator. Such a coordinate has a likelihood gradient only when one of its
  readers is in the batch, then N/B times the reader's own, and Adam gave it a
  noise-driven walk; restarting Adam for polishing widened its sd and pulled its
  mean towards the prior until a reader came round. On window units of OU chains
  whose boundary states two units read (1,024 units, batch 8) the fit ended
  229,317 nats below the family's optimum, the boundaries' sds 4.1 times the
  optimal (median; up to 39) and their means 15 optimal sds off (rms). Each
  reader of such a coordinate now keeps a Gaussian site for it (its curvature and
  pull, refreshed when the reader is visited, its precision bounded per visit as a
  unit's own step is), and the coordinate steps on its prior plus the sum of its readers'
  sites. On that toy the fit ends 941 nats below the optimum, 718 of them the
  units' own widths; the boundaries' sds are 1.01 times the optimal (5-95%:
  0.84-1.24) and their means 1.2 optimal sds off (rms), 0.53 after a later stop
  (`min_windows=120`). On the snag's own case (2,048 window units with learned OU
  and field hyperparameters and a read-pair `gmrf` field whose states 16 units
  read; batch 8) the fit ends 17,645 nats below the family's best factor at its
  own hyperparameters, against 268,852: boundary sds 1.06 times the optimal
  (6.27 before), boundary means 1.14 optimal sds off (11.3), field states 0.98
  and 1.94 (1.47 and 4.26). Most of what remains is the units' own widths. The
  fitted OU length is 17.7 against 13.6 before (family optimum 40.9), but its
  hyperparameters cost 1,595 nats against the optimum's, 856 before. A later
  stop (`min_windows=200`) lowered that fit's ELBO by 88,477 nats; it now costs
  769, from the units' widths alone, while its states and hyperparameters improve.
- A global is stepped this way when its readers are drawn less than once per
  batch on average (`batch_size` times the sum of its readers' sampling
  probabilities below one). Population coefficients and hyperparameters that
  every unit reads keep the minibatch estimate. The objective, the ELBO trace,
  the units' steps and `laplace=GaussianBlockLaplace()` are unchanged.
- Added: `diagnostics.sparse_globals`, the packed positions of the globals
  stepped on reader sites, their reader counts and expected reader draws per
  batch.
- **Breaking:** an optimizer checkpoint (`BRMSVIOptimizerState`) written by an
  earlier version does not load, since the run's fields changed; finish such a
  run on its own version or restart it. A fit with sparse globals on a fixed
  seed differs from an earlier version's from its first iteration.

## 0.2.239

- Added: `brm_svi_fit(...; cell_units)` for connected units. Likelihood cells that
  read no subject's effect used to land in the first unit, with every ordinary
  coordinate only they read, so that unit was evaluated on almost every update
  whatever the batch size. `:components`, the new default under
  `unit_partition=:connected`, gives them inference units of their own, appended
  after the subjects' units: cells reading a common ordinary coordinate no
  subject's cell reads share a unit, and cells with no such coordinate form about
  √m units of √m cells. A coordinate that joins nearly all of them joins them into
  one unit until it is promoted. Scientific roles and the density are unchanged.
  A prepared model's retained SVI target (0.2.226) is also keyed by `cell_units`.
- Changed: under `:components`, unit `work` also counts the rows of a vector-valued
  cell, so `unit_sampling=:work` visits data-rich units more. It is a sampling
  weight, not a time estimate.
- **Breaking:** a fresh connected SVI fit has a different layout, factorization and
  sampling law, so its results differ. `cell_units=:first_unit` restores the earlier
  layout and work. Checkpoints keep the layout they were made with, and earlier
  checkpoints continue on `:first_unit` without being told.
- Changed: the ReactiveKernels and ReactiveKernelsPPL floors rise to 0.1.104 and
  0.1.114. From those versions a response that reads a `@plate`-computed vector
  whole, such as `y ~ Normal(mu, sigma)` with `mu[i]` set in a `@plate`, splits
  into cell units as its indexed form does. Earlier versions prepared its partition
  without the unit's point and refused it under `:components` (ReactiveKernels snag
  `whole-vector-res-64373f39`).
- Measured on a synthetic model, in `research/svi_cell_units/`: 2.5-2.8× the gradient
  variance per update, far less time per update, and a higher ELBO after 20,000
  updates. Its sensors are nearly independent given the globals, so this does not
  show the cost of dropping a real correlation between them.

## 0.2.238

- `brm_svi_fit` polishing averages a dense unit factor's covariance instead
  of its precision. A visit changes a dense unit's precision by up to 2.5-fold
  along one drawn direction, so the mean of those precisions lies far above a
  typical iterate, and the returned factor was narrow, the more so the more
  polishing visits a unit had. On exact Gaussian units under the default
  schedule, with the unit in every step, its sds were 0.86, 0.76 and 0.76 times
  the truth at 13, 40 and 66 coordinates. They are now 0.99 to 1.02, as the raw
  iterate's are; at 66 coordinates the width part of the gap to the optimal
  factor fell from 5.5 to 0.9 nats per unit. A unit with a few hundred visits
  in all has not yet settled from its start and changes little. Units above
  `unit_dense_limit` (diagonal plus low rank) are unchanged. A seeded fit
  explores bit-identically; its `q` and draws differ from polishing on.
- Optimizer checkpoints (`BRMSVIOptimizerState`) are format 2. A checkpoint
  from 0.2.237 or earlier still resumes; one taken while polishing keeps, for
  the iterates it had averaged, their precision average.

## 0.2.237

- `brm_svi_fit(...; field_states=:units)` holds a read-pair field's states
  (`gmrf(...; neighbors=m)`) in units' own Gaussian factors. A state read by one
  unit joins that unit's block, next to its effects; one read by several units
  stays global, and a read's channels stay together. The field's joint prior is
  replaced by one conditional per state, owned by the state's owner, so the
  objective is unchanged and a batch of units estimates it without bias. A step
  costs what its units read: 0.034 to 0.046 s per step from 256 to 16,384
  states on a synthetic station field, where `:global` (the default, unchanged)
  rose to 0.60 s. With ReactiveKernelsPPL 0.1.116 or later, the prepared target
  holds 22.7 KB per unit on that toy beyond the model's own (15.5 KB under
  `:global`), counting each heap object once; 9.6 KB of it is BRM's own
  records, 6.5 KB of those the field's conditionals. Earlier RKPPL copied the
  model's graph into every unit's prior query, about 0.8 MB per unit.
  `:units` is refused for streaming windows, `multiresolution=` fields, and
  fields whose lengths or correlation factor are not global coordinates.
  Checkpoints carry `field_states`; a resume keeps it. A model's held SVI target
  is reused only for the same `field_states`.
- A unit's SVI evaluation writes and reads only its footprint at the plan's
  point and keeps its global gradient as a compact list over the globals it
  touched. Two globals-long buffers per unit per step, and the write and
  restore over every global, are gone, so a step no longer grows with the
  number of globals. Results are unchanged: seeded fits are bit-identical to
  0.2.217's.

## 0.2.236

- Added: a factor-level SVI start from fits of the same declaration on slices of
  the data. `brm_svi_combine(x, slices)` returns a `BRMSVIStart` for the full
  model `x`, and `brm_svi_fit(x; start)` takes it, or the slices' fits or
  `BRMSVIState`s directly. Global coordinates combine as a product of Gaussian
  sites with the prior counted once (`P = P₀ + Σ_s (P_s − P₀)`, one sweep of
  expectation propagation on partitioned data), where `(m₀, P₀)` is the
  diagonal Gaussian projection of the global prior. Only slices whose data
  reaches a coordinate contribute; a negative site precision is dropped and
  counted. Random-effect draws are standardized, so each slice's factor over a
  level's draws is carried from the slice's scales, correlation factor and
  absorbed population coefficients (mu-referenced into the units' block, or
  repeated by a shared grouping's block) to the combined ones by the map that
  keeps the level's effects and totals (`z_s = A_s⁻¹(A_c z_c + δ)`): a unit
  takes its slice's mean and precision carried as a site, with its `N(0, I)`
  prior applied once, so a data-rich unit keeps its totals, and carries the
  shared levels only it reads with it; a global shared level's sites are
  carried before they multiply. A slice unit's diagonal-plus-low-rank factor
  enters through its precision; a full-model unit above `unit_dense_limit`
  starts from the start's marginal sds with its rank part at zero. Connected
  units (`unit_partition=:connected`) are refused. Coordinates match by semantic
  identity: a random-effect draw of the units, a nested or a shared grouping by
  block, margin and level label, any other coordinate by its packed name. A
  slice's preprocessing must equal the full model's except those groupings'
  level tables. `receipt` reports per global coordinate how many slices
  informed it, how many sites were dropped, whether they were carried and the
  slices' standardized spread, and per unit which were carried;
  `diagnostics.start` carries it with the counts applied. The start is not a
  prior; Adam's moments and the global rank part start fresh. `GaussianBlockLaplace`
  refuses a factor-level start. Snag `partitioned-svi-9ced532c`.
- Fixed: on a frozen replay (`RKBRMI(brmi, state)`, `reprocess`, a streaming
  window) whose rows lack some levels of a grouping it does not resample, the
  rows of that grouping's random-effect draws were read from the replayed
  column while the layout keeps the training's level table. `brm_start` then
  wrote a level's effect into another level's draw (`effects=["k3" => v]` into
  the training's `k2`), and coordinate transport, `brm_nlme_fit`'s level names
  and the sparse Laplace level count read the same wrong table. They now read
  the training's table; a fresh binding, or a block that resamples any of its
  columns, is unchanged.
- Changed: `brm_svi_checkpoint` states record, inside `q`, each global's
  semantic key (`keys`) and whether the binding's data reached it (`informed`),
  and each unit's `semantic` keys. The struct and the streaming carry are
  unchanged; a state from an earlier version still carries, but
  `brm_svi_combine` refuses it.

## 0.2.235

- Faster: the whole-model embedded Laplace (`BRMLaplaceProblem`, from
  `brm_laplace_problem`) on connected units (`unit_partition = :connected`) no
  longer costs one η Hessian-vector product and one third-order contraction per
  coordinate of its largest unit. A connected unit joins every subject a
  likelihood cell couples, so where a few effects are read with many subjects
  (one site effect read with each path through the site) the largest unit grew
  with the data, and with it the cost of every evaluation, contrary to "units do
  not change the ratio" in brm-use's Laplace guide. The units' Hessian blocks now
  come from one product per color of a star coloring of the cells' structural
  supports (SparseMatrixColorings.jl, a new dependency), and the gradient's
  third-order sum from two contractions per color; the position coloring stays
  where it needs fewer passes, as for strict subject units. Values and gradients
  are unchanged up to rounding. The first assembly checks the pattern against
  one more product; curvature outside it warns and falls back to positions.

## 0.2.234

- Fixed: `brm_svi_resume` of a `@brm` model read the grouping options before it
  compared the model with the checkpoint. Since 0.2.229 the `@brm` entry of the
  NLME fits chooses which `mi(...)` completions to lower non-centered from the
  groupings, and it built the subject view to choose even for a model without
  a centred completion. So resuming a different model with a wrong grouping
  option failed on the grouping (`brm_nlme_view: subject_group ... is not ...`)
  instead of naming the difference, contrary to resume's contract of comparing
  the model right after lowering. An SVI checkpoint now records which
  completions its model lowered non-centered (`identity.completions`, the
  mode `RKBRMI(brmi; completions)` takes), and resume lowers a `@brm` model
  that way, as `reprocess` keeps a backend's parametrization, so the comparison
  reads no grouping option; preparation then checks the groupings. A model
  without a centred completion is lowered without reading the groupings in
  every fit. A checkpoint written before this records no parametrization and
  is lowered from the grouping options as before. Snag
  `resume-identity-90a2f094`.

## 0.2.233

- Changed: a monotonic term's level count is an input of the program on both
  backends (user principle `0ikeidi`). `mo(x)` and `mo1(x)` wrote their fitted
  levels and the increment simplex's Dirichlet concentration into the program
  (`brm_prepared_indices(x, [1, 2, 3])`, `Dirichlet([1.0, 1.0])`,
  `simplex[2]`), so a column with another number of levels was another program.
  `RKBRMI` now reads them from generated inputs, as a factor's levels are; `SBBRMI`
  reads the concentration as Stan data named after the term (`mo_x_alpha`) and sizes
  the simplex by it. Densities are unchanged.
- Changed: on `SBBRMI`, an ordinal response's threshold count is Stan data,
  `<response>_thresholds_length` (`<response>_cutpoints_length` for
  `OrderedLogistic`), not a literal `ordered[K-1]`.
- Unchanged: `RKBRMI` still writes an ordinal response's threshold count into the
  program. RKPPL's ordinal lowering does not yet take a bound name for that size
  (ReactiveKernels snag `rkppl-threshold-932a2e27`).
- A fit or checkpoint of a model whose program text changes here is refused by a
  frozen replay and needs a fresh fit.

## 0.2.232

- Fixed: on `SBBRMI`, `brm_descriptor` refused a predictor that repeats one
  structured term (`a ~ 0 + mo(g) + mo(g)`, `mu ~ 1 + hsgp(t) + hsgp(t)`) or
  carries the same term both as a summand and inside a random effect
  (`a ~ 0 + mo(g) + (mo(g) | subject)`), with "term `mo_g` on logical predictor
  `a` has 2 logical output owners". Structured carriers (`mo`, `mo1`, `s`,
  `gp`, `hsgp`, `gmrf`) record their predictor and term (since 0.2.103), and
  those two fields did not tell such carriers apart. Each carrier now also records
  whether it is the predictor's own summand or a column inside a random-effect
  block, and which repeat of the term it is, so every formula occurrence owns
  exactly its own carrier. The label of a repeated term still names no single
  carrier (`brm_term_coordinates(d, :a, names; term=:mo_g, ...)` refuses with
  "occurs 2 times"), and beside a random effect `term=:mo_g` names the
  population term's simplex. Emitted programs are unchanged. Snag
  `brm-descriptor-r-2331ddd1`.

## 0.2.231

- Fixed: partition declarations for wholly unit-owned compressed transforms
  use the constrained first-axis label count. A local 2×2 Cholesky factor
  with one packed coordinate no longer fails the BRM ownership adapter.
- Compatibility: require ReactiveKernels 0.1.94 / ReactiveKernelsPPL 0.1.105.
  A tuple plate's statements run only at rows that read their entries, so
  automatic connected units read no other unit's coordinates. Selected-unit
  gradients and shared-row storage stay compact: warm unit-call allocations
  no longer grow with unrelated units in the fixed-footprint scaling checks.
  From ReactiveKernels 0.1.92, native Reverse on a unit problem can raise
  `EnzymeRuntimeActivityError` on Enzyme dogfood before `cecb571a` (libEnzyme
  `7f23cb72`); the test bootstrap pins that commit. Runtime activity is not needed.
- Explicitly connected SVI defaults to bounded automatic separator selection
  and work-weighted unit batches. `separator=:none`, `global_coordinates`,
  `max_unit_coordinates`, `max_unit_work` and `max_promoted_coordinates`
  control promotion. The final partition is rebuilt from structural supports;
  scientific subject identity stays intact. Diagnostics report actual unit
  ownership, promotion reasons and unmet budgets. All authored density factors
  remain exact; the global/local split changes the variational approximation.
- Streaming format 3 retains final inference-unit membership, separator controls
  and promoted global identities alongside dense or structured local factors.
  Later windows retain the saved global set without spending another automatic
  promotion allowance; an unmet structural budget is reported as `:carried`.
  Changed retained membership or global scientific identity is refused.
- `brm_svi_fit` accepts `unit_sampling=:work` for positive work-weighted
  sampling with replacement and inverse-probability objective/global-gradient
  weights. Repeated draws share one local update. Diagnostics distinguish
  draws, visits, selection probabilities and actual gradient calls;
  `:uniform` retains distinct-unit batches and remains the subject-mode default.
- **Breaking:** local SVI factors larger than 64 coordinates now use a
  diagonal-plus-low-rank covariance instead of dense precision matrices.
  `unit_dense_limit` (0:256) and `unit_rank` (0:32, default 8) control the
  approximation. Small dense units retain their natural-gradient updates;
  larger units use posterior-scaled path-gradient Adam updates. For those
  units, diagnostic `q.Λ` and `q.U` are matrix operators and `q.U` is no
  longer triangular: use multiplication rather than triangular indexing or
  solves. Materializing an operator still requires a dense matrix.
- Format-3 SVI checkpoints combine connected membership with structured local factors without optimizer
  moments. Resume preserves their distribution and family under coordinate
  reordering, accepts format-1 checkpoints, and retains carried dense factors.
- SVI targets reuse one global-plus-local workspace per worker and store only local
  ordering metadata per unit. Cached entropy and lazy local polishing avoid
  visiting all local factors on every iteration. The underlying cell plan
  still determines structural-selection storage; full returned draws still
  scale with the complete parameter dimension.
- Weighted batches, visit counts, cached entropy and lazy local polishing now
  participate in optimizer checkpoints, preserving continuation across thread
  counts. Structured local factors retain their Adam state on same-target resume.
- `BRMSVINonfiniteError` (0.2.209) also covers structured local factors and
  work-weighted batches: the iteration is named before it changes `q` or the
  optimizer, whichever local family or sampler the unit uses.
- The 0.2.223 gradient bound covers work-weighted batches and structured local
  factors. Under `unit_sampling=:work` it is `10 √w` times a global's running rms,
  with `w` the batch's largest unit weight (`N / B` for a uniform batch, as
  before), because a unit drawn with chance π carries weight about 1 / π. A local
  factor above `unit_dense_limit` steps with its own Adam only when its unit is
  drawn, so its gradient is limited to `10` times its own running rms.
- A prepared model's retained SVI target (0.2.226) is keyed by the separator policy
  and budgets as well as the requested globals and streaming carry, so a connected
  fit never reuses a partition built under another policy. Snapshots report a
  connected fit's inference units in `levels`, as `diagnostics.unit_levels` does.
  A snapshot's `q.Λ` and `q.U` are plain data: a dense unit's matrix, or a
  structured unit's `(; family=:diagonal_lowrank, ρ, Β)`, so a saved snapshot
  still loads without BRM.
  `diagnostics.unit_gradient_calls` counts actual evaluations: a unit drawn twice
  in one work-weighted batch is evaluated once.
## 0.2.230

- Fixed: on `RKBRMI`, a `gmrf((east, north), time, channel)` field without `cor`
  spelled its identity channel factor as a literal sized by the channel count
  (`hcat(vect(1.0, 0.0), vect(0.0, 1.0))` for two channels), so fields that
  differ only in their number of channels emitted different programs. The
  program now reads the identity from the fitted support
  (`brm_gmrf_identity_factor(support.geometry)`, as per-cell fields already did),
  so one program serves every channel count on the mesh, `neighbors=` and
  `multiresolution=` forms (user principle `0ikeidi`). From ReactiveKernelsPPL
  0.1.113 they also lower to one program. Densities and gradients are
  unchanged; `SBBRMI` already read the identity from its data. Snag
  `gmrf-channel-ide-f3e0758e`.
- Changed: **a model with such a field emits a different program**, a field
  without a `channel` column included (its one-channel factor was
  `hcat(vect(1.0))`), and its SMC and SVI checkpoint tags change. A fitted state
  an earlier version wrote for it is refused at replay ("changes the fitted RK
  program") and needs a fresh fit.

## 0.2.229

- Fixed: on the NLME paths, an `mi(x)` completion whose law reads the units'
  draws (`mi(x1) ~ LogNormal(mu_x, sigma_x)` with `mu_x ~ ... + (1 | region)` and
  `region` the subjects) gave routes that silently disagreed under the default
  centred lowering. The missing entries were shared coordinates in θ, so their
  law was a prior term: each unit's values and one-unit gradient
  (`brm_nlme_loglikelihood(_and_gradient)`) omitted it, the population prior
  evaluated it with every unit draw at zero, and the lockstep gradient
  (`brm_nlme_loglikelihoods_and_gradients`) and the η Hessian carried it. On the
  regional panel the lockstep gradient differed from the gradient of the values
  by up to 4.6, so an NLMEEstimation fit saw a subject's gradient change with its
  batch. Snag `brm-nlme-loglike-7c4b209b`.
- Changed: `brm_nlme_fit`, `brm_two_stage_fit` and `brm_svi_fit` given a `@brm`
  model lower exactly the completions whose law reads the units' draws
  non-centered. Their missing entries are then unit draws, integrated with their
  unit, whose law's density joins the unit's conditional. They also lower the
  grouped entries of `mi(x)::shares(total, rows)` non-centered, whose centred
  reading `brm_nlme_view` has always refused. Every other completion keeps the
  centred reading.
- **Breaking:** `brm_nlme_view` (and so `brm_nlme_model`, `brm_laplace_problem`
  and the fits given an `RKBRMI`) refuses a centred completion whose law reads
  the units' draws, naming `RKBRMI(brmi; completions = :noncentered)` or
  `completions = (:x,)`. Lower the backend that way, or pass the `@brm` model to
  the fit. A submodel's sampled declaration whose law reads the units' draws is
  refused by name too. A fit that lowered such a model from its `@brm` model
  now samples `x.z_mis` where it sampled `x.y_mis`, and θ no longer holds the
  missing entries.
- Added: `RKBRMI(brmi; completions = :noncentered)` takes every continuous
  univariate law. A law without a location and scale keeps its support transform
  on the standard-normal draw (`lower + exp(z)`, `upper - exp(z)`, a scaled
  logistic on an interval, `z` on the real line: the coordinate the centred
  lowering samples) and observes its exact density remainder in `:pointwise`;
  it used to be refused ("needs a location-scale law"). `completions` also takes
  a collection of `mi` sources, `completions = (:x,)`, to non-center those
  completions only. A regression-plan `mi(y)` response, which samples none of its
  missing entries, accepts every spelling instead of refusing `:noncentered`.

## 0.2.228

- Fixed: `brm_svi_resume` refused a checkpoint as "the model differs from the
  checkpoint's in its emitted program" when the same model was loaded from
  another checkout, depot, host or BRM release. A model's program identity
  printed every macro call with the file and line it was written at, and an
  emitted `ReactiveKernels.@kernel` definition is one: BRM's own (with the depot
  slug of this BRM installation and the line in its source) or a source
  provider's (with the consumer's checkout directory). Every identity that
  compares the program now reads it without those places: SVI optimizer
  checkpoints, the SVI and SMC streaming carries, SMC and SMC Laplace
  checkpoints, filter exports, and a fitted state's export and replay. A program
  without such a macro call keeps its identity, so its checkpoints are
  unchanged.
- An SVI optimizer checkpoint written before 0.2.228 holds the digest with
  those places. As before, a checkpoint resumes on the version that wrote it
  (since 0.2.223 a later one does not load an earlier one). Where a later
  version does load it, it resumes when its program has no such call;
  otherwise the refusal says that its digest predates the place-free program
  identity and that only the installation that wrote it reproduces it. An
  older SMC checkpoint, streaming carry or filter export of such a model
  differs from what this version computes, as it already did from any other
  installation. A fitted state exported earlier replays, because it holds the
  text itself.

## 0.2.227

More numbers are inputs of the program instead of text in it (user principle
`0ikeidi`). Densities are unchanged: each form is checked against its literal twin.

- Changed: `multigroup` reads its numbers from data on both backends.
  `multigroup((x, y), lambda; k = kk, c = cc)` accepts data fields for `k` and `c`,
  which used to be refused. `lambda`, and `tau` in `multigroup(Omega, tau)`, were
  accepted from data but written into the program, and are now read by name. A
  literal `k` or `c` also gives one program across values.
- **Breaking:** `SBBRMI` names a coordinate `multigroup` grid after its term, not its
  numbers: `multigroup_x_y_hsgp20_c1p5_center` is now `hsgp_t_by_g_grid_center`,
  and likewise for `_scale`, `_lambda_lower` and the new `_k`. Code that reads these
  from `stan_data` by name must change. Two terms no longer share one grid.
- Changed: on `RKBRMI`, what a prior reads from a data array is an input, and the
  program computes the prior's value from it. A reduction (`Normal(sum(m), 1)`), an
  `MvNormal` mean, covariance or vector scale were written into the program, and a
  size taken from such an array now reads as one program too. `Dirichlet(a)` with a
  data vector and `Normal(m[1], 1)` used to be refused on `RKBRMI`. They now match
  `SBBRMI`.
- Changed: a prior computed from a design column, such as `Normal(mean(x), 1)` or
  `Normal(x[2], 1)`, is a number of the fit, as `standardize(x)` is: the program
  computes it from the column's fit-role copy. A frozen replay on other rows and a
  prediction design keep it, where a replay used to be refused, and a fitted state
  holds the computed mean, not the rows. An array only a prior reads travels with
  the fitted state; a frozen replay refuses another value of it.
- Changed: `effect(m, x) ~ Horseshoe(local_scale = l, global_scale = g)` accepts data
  fields on both backends, read by name. `RKBRMI` shares one global scale per
  predictor, so it refuses by name a predictor whose terms read different global
  scale fields.
- A fit or checkpoint from a model whose program text changes here is refused and
  needs a fresh fit, as a frozen replay refuses any changed program.
- Limits: `mo` and ordinal level counts are still spelled in the program on both
  backends, and a categorical response's count is a true shape change, with one
  predictor per class. One lowered program across sizes and bounds needs RKPPL
  0.1.108 and one query type across values RKPPL 0.1.106. BRM's compat floors do not
  require them: an older RK is correct but recompiles for each value.

## 0.2.226

- Added: `brm_svi_fit` and `brm_svi_resume` hand out snapshots of a running fit
  without returning: `snapshot` (a function, or a file path the run rewrites
  atomically, so it always holds the latest complete snapshot), `snapshot_every`
  (default once per window, or every 50 updates with `GaussianBlockLaplace`) and
  `snapshot_draws` (default none). A snapshot is one named tuple of plain arrays
  that loads without BRM or the model: `q`, what `q` is (`q_kind`:
  `:raw_iterate`, `:polished_average` or `:window_average`), the coordinates,
  units and levels that name it, `iteration`, `phase`, the window ELBOs and, outside
  `GaussianBlockLaplace`, `elbo_windows_trimmed` and `clipped_steps`, `draws`,
  and a `status` that stays `:running` until the boundary where the call ends. An
  observed run is the same to the last bit as an unobserved one, and a snapshot's
  `q` and `draws` equal those of a result stopped at its iteration. BRM keeps none,
  so a long run's memory and files do not grow with snapshots.
- Changed: a prepared `BRMNLMEModel` keeps its minibatch target and worker samplers
  for its next `brm_svi_fit` or `brm_svi_resume` in this process, so a continuation
  on `fit.diagnostics.model` no longer prepares them again. `diagnostics.target`
  reports it: `reused`, `workers_built` and the phase's `seconds`. Results and
  checkpoints are unchanged, and a changed model is still refused by resume's
  identity check. The model retains the target for as long as it lives; a copy or a
  serialized model holds none. `GaussianBlockLaplace` still prepares its target on
  every call.

## 0.2.225

- Fixed: on `RKBRMI`, a data expression can read a numeric scalar data field, as
  on `SBBRMI`. `mu ~ 1 + offset(slope * x)` with `slope` a scalar field was refused
  ("column `slope` must be a plain numeric vector for arithmetic"). It now builds,
  and so do a coefficient's column (`mu ~ 1 + slope * x`), a fitted transform
  (`standardize(slope * x)`), an interaction operand, a random slope, a data call's
  argument, a scalar alone (`offset(slope)`, `offset(2 * slope)`, one value on every
  row) and a model-value offset (`offset(theta * slope * x)`, which brm-use told
  consumers to spell with a literal). A product or sum of three or more,
  `offset(a * b * c)`, was refused as "`*` takes exactly two arguments" and now
  builds on `RKBRMI` too.
- Fixed: on `SBBRMI`, a coefficient's column or a fitted transform that multiplies
  a column by a scalar field (`mu ~ 1 + slope * x`, `center(slope * x)`,
  `(slope * x) & factor(g)`) was refused ("mixes row axes of lengths 1 and 4"): the
  scalar was taken for the row axis. It now builds.
- Changed: the program reads such a scalar by name as a runtime input, so two data
  sets that differ only in its value emit the same program, and the density follows
  the bound value. That includes the data-only assignment `sx = slope * x`, which
  used to fold the number into the program, so **a model of that spelling emits a
  different program** and its SMC and SVI checkpoint tags change. On a frozen
  `reprocess` a scalar read only by a data expression is a design input: omitted,
  it keeps the fitted value; supplied, the design's. The anchors of a fitted
  transform keep the fit's value either way, and a scalar a prior also reads still
  refuses a new value.
- Unchanged: `RKBRMI` refuses a `Bool` field inside a data expression, as it refuses
  a `Bool` column. `SBBRMI` accepts one.

## 0.2.224

- `SBBRMI` builds a `gmrf(...)` field in a `@plate for` cell again. From 0.2.172
  every such model stopped at transpilation with `plate: N do-block params vs M
  positional iterables`, because the plate's iterables left out the inputs the
  field prepares. Model declarations need no changes, and `RKBRMI` was not
  affected. A cell can also read another plate's complete cells beside its field.

## 0.2.223

- Fixed: one extreme draw could stop `brm_svi_fit`'s optimization of the globals
  without any sign of it. A draw that reaches a steep part of the density gives
  a finite ELBO estimate of, say, -1e21, so `nonfinite=:error` never fires, and
  its gradient stays in Adam's second moment for tens of thousands of iterations.
  On one real fit of 7,379 globals, the draws of a single run left 95% of the
  global means with a step more than 10 times too small when polishing ended, 85%
  more than 1,000 times too small and 22% more than a million times too small, so
  they had barely moved during polishing, and the ELBO looked flat as for a
  converged fit. Each coordinate's gradient now enters Adam's moments limited to
  `10 √(N / B)` times that coordinate's running rms. The bound is that wide
  because a coordinate read by few subjects has a zero gradient in most batches;
  a tighter one shrinks its scale without end. `laplace=GaussianBlockLaplace()`
  is unchanged.
- Changed: the stopping rule reads the 10% trimmed mean of each window and its
  standard error (Yuen), not the mean and variance. One draw could decide the
  test: its window's variance swamped every gap, whichever way the ELBO was moving.
  Fits with no extreme draws can stop at a different window than before.
- Added: `diagnostics.elbo_windows_trimmed`, what the stopping rule now reads, and
  `diagnostics.clipped_steps`, how many global steps had a gradient limited.
  `diagnostics.elbo_windows` stays the plain mean of each window, which a single
  draw can dominate, and now includes the last polishing window, which it used to
  omit. The progress message reports the trimmed mean.
- **Breaking:** an optimizer checkpoint (`BRMSVIOptimizerState`) written by an
  earlier version does not load, since the run's fields changed. As before, a
  checkpoint needs the same package version; finish such a run on its own version
  or restart it. A fit on a fixed seed differs from an earlier version's from the
  first clipped step and from the stopping decision on.

## 0.2.222

- Fixed: `brm_svi_fit(...; laplace=GaussianBlockLaplace())` could stop with
  `inconsistent quadrature bounds` on valid input, depending on the seed and the
  host. On a 12-coordinate toy model at 8 updates, 6 of 17 seeds stopped this way.
  The upper and lower bound of each log-determinant probe came from two separate
  `eigen` calls, and near convergence their rounding, about `1e-12` there, exceeded
  the true gap between the bounds. Both bounds now read one Cholesky factorization,
  on which their order holds for the computed numbers. The 11 of those seeds that
  already fitted move by at most `2e-13` in ELBO and take the same Krylov steps.
  A precision too ill-conditioned to factor now always raises `ArgumentError`,
  where it could also raise `nonpositive Radau node`. This fix does not make a
  badly conditioned log determinant more accurate: `GaussianBlockLaplace` now
  documents that rounding, which `logdet_quadrature_width` does not include.

## 0.2.221

- Added: `brm_gmrf_predict(rk, draws, field; east, north, time)` gives a read-pair
  field (`gmrf(...; neighbors=m)`) at positions and times the fit need not have read:
  conditional means, or conditional draws with `rng`. Marginals are the law's; points are
  conditionally independent given the states, so a joint quantity over several new
  points, such as an areal total, is not.
- Added: `brm_nlme_view`, `brm_nlme_model` and `brm_svi_fit` accept a model-scope `gmrf`
  term. Its states and hyperparameters are shared latents. `brm_svi_fit` refuses the mesh
  form by name: one independent factor per mesh state collapses the learned lengths on
  toys. A read-pair field's states are one global factor each under `brm_svi_fit`.

## 0.2.220

- Changed: on `RKBRMI`, a data scalar a prior reads is a runtime input of the
  program instead of a number folded into it (user principle `0ikeidi`). That
  covers `sigma ~ Exponential(s)`, a declared vector `z[1:n] .~ LogNormal.(mu, sd)`,
  an expression over scalars (`LogNormal(log(s), 0.3)`), a scalar an assignment
  reads, population-effect priors, and the hyperparameter priors of `gp`, `hsgp`,
  `car`, `cdar` and `dar` terms. Two data sets that differ only in such a value
  emit the same program, lower to the same program and prepare the same
  `prepare_query` type, and the densities are unchanged. `rk_bound_data` now holds
  these names, and the emitted source spells them instead of their numbers.
- Changed: a stated `cor(:, ID) ~ LKJCholesky(K, eta)` no longer spells `eta` in
  the program on either backend. RKBRMI reads a data field by name and a literal
  through a generated input; SBBRMI reads a generated data field,
  `<block>_lkj_eta`. A frozen RK replay still refuses data holding another value
  of a prior's scalar, `eta` included, and a fresh replay
  (`freeze_constants=false`) takes it. A Stan replay keeps the fit's `eta`.
- Changed: a factor's fitted levels and coefficient count, a known correlation's
  factor (`cor(:, ID) = C`, `gr(g; cor = R)`), a scalar `MvNormal` scale and the
  concentration of `Dirichlet(K, a)` are inputs of the RK program too. A factor
  with other labels but the same count lowers to the same program.
- Changed: a data scalar in a `truncated` or `restricted` bound or a `Uniform` is
  spelled by name in the emitted RK program. ReactiveKernelsPPL still reads such a
  bound into its constrain transform, so the lowered program and query type follow
  the value until ReactiveKernels snag `rkppl-sized-vect-69604d32` lands.
- Changed: what identifies a fit for an SVI source, an SMC frame or carried source
  and a filter now states the numbers its priors read, because the program text no
  longer does. A resume with another prior value is still refused.
- Added: `t2(x, z; k = kk)` takes its basis sizes from a data field on both
  backends. RKBRMI reads them from the fit-role input `<id>_k_fit`, so one emitted
  program serves every size; the coordinates follow the sizes, and RKPPL's layout
  follows them too (`rkppl-sized-vect-69604d32`). A frozen replay refuses another
  size.
- Not changed: a literal an author writes in a prior (`Exponential(1.7)`) stays in
  that author's source. `multigroup` `k` is still a literal, a data vector or matrix
  a prior reads is still spelled in the program, and `Horseshoe` scales still refuse
  data on RKBRMI. ReactiveKernels' `prepare_sampler` still specializes
  on a bound scalar's value (ReactiveKernels snag `prepare-sampler-e69f1b38`), and
  changed extents still change the lowered program
  (`rkppl-sized-vect-69604d32`).

## 0.2.219

- The sparse Laplace marginal gradient is correct at a stiff observation on the
  factor route. Its augmented system lost accuracy as the square of the outer
  curvature, and the gradient keeps that residual on any population coordinate
  aliased with an integrated effect: on a bounded ODL target with outer curvature
  2e7, `odl_offset_Intercept` read 19214 for 395. The system now keeps its outer
  inputs and their images as separate variables and refines each solve against
  the conditional precision itself. Marginal values, definiteness and selected
  covariances are unchanged. Conditional draws use the same solve.
- The Laplace mode search's default Newton budget per continuation stage
  (`maxiter` of `brm_laplace_problem`, `brm_laplace_plan` and the NLME
  integrated families) rises from 50 to 200. Where a followed mode ceases to
  exist between stages, the walk to the next mode can take more than 50 steps:
  on the bounded ODL target a 38-effect component needed 56-59 at stage 4/10
  and up to 122 at stage 5, with exact derivatives, so the default scored a
  point that has a mode `-Inf`. A point that does not converge now costs up to
  four times as many steps before it is `-Inf`. Pass `maxiter=50` for the
  former budget.
- Sparse Laplace supports model-scope helpers shared by ordinary or
  sum-factored observation cells. Sampled-coordinate helper values are
  reused across a cell sweep while their mixed derivatives remain part
  of the marginal gradient. Integrated-coordinate helper reads retain
  strict locality checks.
- Preparation retains local integrated supports without expanding shared
  sampled coordinates into every cell. Retained plans, sampled subsets,
  independent problem copies and matrix-free contractions use the same
  shared-derivative path.
- A cell with no sums takes compact operators when every value it reads is
  computed at its rows. Otherwise it takes its strict factor outer, and is refused
  by name if that refuses it too.
- A factor's Hessian block compiles at the same bounded Enzyme batch width as a
  compact cell's, and takes as many batches as it needs.
- A move of a cell group to reverse over forward on Enzyme's activity error
  (0.2.158) restarts its Newton pass and resumes at the cell that raised, so no
  cell counts twice.
- A `mo()` or `mo1()` term read by a subject's `@plate` cell now plans: the
  subjects' effects are declared partitioned, so its level table, which has no
  row form, is computed once per population point in the shared prefix.
- **Breaking:** under `locality = :error`, a matrix with no row form read as
  `M[p, 2]` is now refused by name rather than computed whole in every cell.
  Shared helpers need the RK repairs of ReactiveKernels 0.1.73 and
  ReactiveKernelsPPL 0.1.90, which the current floors (0.1.82 / 0.1.98)
  include: earlier releases refuse sum-factored cells over shared helpers with
  a locality error, read a producer plate's cell whole under `partitioned`, and
  have no row forms for matrix-row helpers, `vec` of a row value, whole arrays
  read by index or `b[p, 2]` of a computed matrix.

## 0.2.218

- Added: `stan_data`, `stan_model` and `stan_instantiate` accept a
  `GenerativePlan`, as `stan_code` already did. Each reads `plan.model`, so it
  returns what the same call returns on the plan's `SBBRMI`, and `stan_model`
  and `stan_instantiate` pass their keywords on. They raised `MethodError`
  before, although `brm-use` said BRM added these methods. Code that reads
  `plan.model` directly keeps working and gets the same results.
- Added: `stan_data` accepts a `StanModel`, so `stan_data(stan_model(x))` works
  on a plan and on an `SBBRMI`.

## 0.2.217

- Fixed: `gmrf(...; neighbors=m)` depended on the order of the rows wherever a
  step of its maximin position order had a tie, as on a regular lattice.
  Positions were numbered by first appearance in the rows and ties resolved by
  that number, so the same reads in another row order gave another state order,
  other conditioning sets and another implied covariance. The largest difference
  on a 40 × 40 lattice at spacing 1 and spatial lengths 1, 5 and 20 was 0.15 to
  0.30 at `m = 4` and 0.004 to 0.012 at `m = 30`, about the size of the
  approximation's own error at that `m`. Positions are now sorted first, as
  `multiresolution=m` already did, so the footprint is a function of the set of
  read pairs. Build time is unchanged.
- **Breaking:** on a support with ties, a model rebuilt from rows that do not
  first read their positions in sorted `(east, north)` order now has another
  state order and prior law than on an earlier version, under the same coordinate
  names, so no error marks the change. Draws or starts saved from such a fit do
  not transfer to a rebuilt model: refit, or replay the fit's saved
  `brm_fitted_state`, which keeps the footprint it was fitted with. Supports
  without ties, rows already in sorted position order and `multiresolution=m`
  are unchanged, bit for bit.

## 0.2.216

- Fixed: a one-unit NLME gradient copied the whole model to the heap twice per
  call, about 17 of the call's 20 KB on an 80-subject PK model, and most of its
  time. `BRMNLMEModel` now keeps its fields behind a pointer, so passing it costs
  nothing. Wall time of a 40-iteration SAEM fit on that model fell 1.3–2.2× under
  lockstep and 1.7–2.9× under `WarmupHMCNUTS(threads = true)`. Seeded fits keep
  the same bits. The type is a `mutable struct` whose fields are all `const`, so
  its fields still cannot be assigned; `===` between two models now compares
  identity.
- Fixed: the evaluator pool of concurrent one-unit calls prepares at most one
  evaluator per Julia thread, as brm-use states. A contended pool reached
  `Threads.maxthreadid()`, which counts GC threads too: 18 evaluators at 13
  threads, each about 0.35 s and 120 MiB allocated to prepare at that size.

## 0.2.215

- Fixed: `brm_svi_fit` and `brm_svi_resume` could not recover a weak direction
  of a dense unit factor above about 20 coordinates. The single-draw curvature
  estimate is indefinite, and where a direction's precision was far below its
  target the trust-limited step kept widening it, on an exact Gaussian target
  too: at 27 coordinates a direction whose true sd is 1, started at 36.7, ended
  between 14.6 and 2760 after 1100 visits. At 13 and 15 coordinates it recovered;
  at 19 and 21 most runs did; at 23 and above they did not. Draws
  then reach far enough out that the target overflows on both sides. The step
  now uses the positive part of the estimate, which needs no knowledge of the
  prior and equals the old estimate at a matched factor. No option, keyword,
  diagnostic or checkpoint format changes, but dense units' trajectories differ
  from the first visit, so a fit on a fixed seed does not reproduce its earlier
  result. The estimate is biased towards more precision where the factor and
  target differ; on a Gaussian target the stationary sd is about 1.04-1.06 times
  the truth, against 1.11-1.14 before.

## 0.2.214

- Fixed: on SBBRMI, an `@plate for p in groups(g)` whose grouping column has no
  rows. Generating its program failed with `Could not find kernel_<cell>_<x>_ragged`,
  and replaying onto such data with `Do not know how to handle stan_type(…, Any[])`.
  The plate now has no cells: it binds its inputs empty with their columns'
  element types, emits the same program as the same model with rows, and its
  density and gradient are exactly its priors'. `reprocess` moves the program
  between rows and no rows in both directions. A random effect under
  `total_groups=:auto` still picks its parameterization from the data, so its
  program can differ on no rows; a replay keeps the fitted one. Pass the empty
  columns typed (`x = Float64[]`, `day = Int[]`): Stan declares each as int or
  real, and an `Any[]` has no entries to tell which, so SBBRMI still refuses it.
- Fixed: a plate whose cells are named by an empty label column
  (`rows = groups(g => cells)` with `cells = String[]`) was refused on both
  backends as if it were long-format data. It now has no cells.

## 0.2.213

- Fixed: identical models were refused across processes when they call an
  anonymous function or closure. A model's identity hashed such a callable by
  what names it in the process: its generated name, the file and line of its
  methods and Serialization's process-wide object numbers. One more anonymous
  definition earlier in the driver, another line, or serializing another
  anonymous function first changed the digest, so `brm_svi_resume` refused a
  checkpoint on the same science and data, after preparing the whole model. In a
  package module the callable was hashed by generated name alone, so a changed
  body kept its digest. A generated function now enters by its methods' lowered
  code and its captured values, nested ones included, with no line information.
  This covers SVI optimizer checkpoints, the SVI and SMC streaming carries, SMC
  and Laplace SMC identities and filter exports. A model with no generated
  function keeps its exact identity, so its existing checkpoints stay valid.
- Fixed: a fitted state's replay and a streaming SVI window compared the model's
  declaration as text, where a callable outside a plate cell prints by its
  generated name (`Normal(#3(mu), 0.8)`), so the rebuilt model was refused as
  "exported from another model" or "model source changed". The comparison now
  ignores generated names, also in text exported earlier. Like a named function
  or a callable inside a plate cell, the body of such a callable is not compared
  on replay, which reads only this text and the emitted program.
- Changed: `brm_svi_resume` checks the program, callables, data and coordinates
  right after lowering, before it prepares the model and target, and the refusal
  names what differs: the emitted program, the callable's binding, the data
  column, or the packed coordinates. New checkpoints hold these digests in
  `state.identity.binding_parts`, of fixed size per callable and data column.
- Compatibility: an SVI optimizer checkpoint written before this version is still
  accepted when the old digest matches, which for a model with a generated
  callable needs the old driver layout; its refusal cannot name the part. Its
  continuation writes the new digests. An SVI or SMC streaming carry, an SMC
  checkpoint or a filter export of a model with a generated callable, written
  before this version, is not migrated and is refused as a changed model.

## 0.2.212

- Fixed: a population prior that reads data is integrated by exact totals and
  sum-to-zero like the same prior with the numbers spelled out.
  `effect(mu, Intercept) ~ Normal(m, s)` with `m` and `s` in the data was refused
  by an explicit `total_groups` or `s2z_groups` request ("is not Normal, Student-t
  or Flat", "unsupported population prior"). The arguments of a `Normal`,
  `LocationScale(m, s, TDist(nu))` or `TDist(nu)` prior may now be literals, scalar
  data fields, `constant(...)`s, elements `v[k]` of a data vector, data-only
  assignments or calls over these. The values enter the program as data, so models
  differing only in such a value compile one program, whose log density, gradient
  and generated quantities equal those of the literal twin.
- **Changed:** with `total_groups = :auto`, a model whose population priors read
  data used to stay conventional without saying so; it now integrates its blocks,
  as its literal twin always did. The posterior is unchanged, the coordinates are
  not, so draws from such a fit do not transfer to a new build. Pass
  `total_groups = ()` to keep the conventional representation. A frozen replay
  of a fitted model is unaffected.
- A prior that reads a sampled value is still not integrated, and the refusals now
  say why ("arguments that read only numbers and data"). A data-only call that
  fails names the call. The exact-total refusal no longer claims a block's `sd`
  prior needs literal arguments; it already accepted data.

## 0.2.211

- Fixed: on `RKBRMI`, a model-scope `=` assignment that no response reads is
  now a graph value of its authored name, as user decision `0kpugvc` already
  documented. `rk_draw_values(rk, U, (:q,))` and `want=(:q,)` answer for
  `mu ~ 1 + x; q = exp(mu - decay * elapsed); y ~ Normal(mu, 0.4)` instead of
  raising `KeyError: kernel has no port :q`; so do scalars over sampled
  parameters (`sigma ~ Exponential(1); variance = sigma^2`, a literal
  `two = 2.0`, an alias `scale = sigma`). Coordinates and log densities are
  unchanged; a gradient can move by one ulp where the extra statement changes
  the compiled evaluation order. The sampler never runs such a value. An
  assignment that needs a declaration no response reaches (`q = exp(z)` for
  an unread `z ~ Normal(0, 1)`, or an unread formula predictor) stays out as
  before. Still out: a data-only assignment no response reads, and one that
  reads data but no formula predictor in a model none of whose assignments
  reads a formula predictor (`spread = sigma * sum(elapsed)`): RK plans such a
  model on a route that writes the data a statement reads into the program,
  so keeping it would refuse a `reprocess` of other rows. A build of an
  affected model emits one more statement, so its program identity (SMC and
  SVI checkpoint tags) changes.

## 0.2.210

- Fixed: on `RKBRMI`, a formula response whose law reads data, such as
  `y ~ Normal(mu, s)` with `s` a scalar data field, `sd_obs = constant(expr)`
  or a plain alias `sd_obs = s` as the scale, or a per-row column
  `y ~ Normal(mu, sv)`, was refused ("scale cannot be a data column", "assignment
  `sd_obs` aliases data column `s`"). It now builds, and the program reads the
  value by name as a runtime input: two data sets that differ only in the value
  emit the same program, and the density follows the bound value. SBBRMI
  already kept such a value as Stan data. This covers the further arguments of
  every scalar response law, at top level or inside a law argument, for example
  a data shape in Gamma's `mu/alpha`. Binomial trials keep their formula
  spelling, and the cutpoints or category predictors of a leveled response stay
  refused. An invalid value is refused when the backend is built, naming it
  (`per-observation scale s must be finite positive numerics`). An out-of-range
  per-row `zi` or Student-t `nu` is refused that way only from ReactiveKernelsPPL
  0.1.105, which BRM's compat floors do not require yet. Before it, the value
  fails later, as a `-Inf` density or an internal error.
- Changed: scalar-data Student-t degrees of freedom
  (`LocationScale(mu, s, TDist(nu))` with `nu` from data) are read the same
  way instead of being emitted as a numeric literal, so one program serves
  every value. An invalid `nu` is refused when the backend is built
  (`nu data nu must be finite positive numerics`), not by the planner.

## 0.2.209

- Added: `BRMSVINonfiniteError`. When an SVI iteration (`brm_svi_fit`,
  `brm_svi_resume`) evaluates the target to a non-finite value or gradient at
  its Monte Carlo draws, it now throws this error before changing the
  variational factors or the optimizer. The error names the iteration, every
  non-finite term (the global prior term or a unit), what was not finite (value
  or gradient, and at which antithetic draw), the coordinates with non-finite
  gradient entries, and whether the evaluated point itself was finite.
  Previously the non-finite gradient reached a unit's natural-gradient step and
  surfaced as LAPACK's `ArgumentError: matrix contains Infs or NaNs` from
  `eigvals`, after the global step had already changed `q`. Continue from the
  last checkpoint `save` received: the run in memory is not a checkpoint after
  the error.

## 0.2.208

- Added: sizes and numeric settings are data. `gmrf(...; neighbors = nb)`,
  `gmrf(...; multiresolution = m)` and `hsgp(...; k = kk, c = cc)` (and
  `domain`, `period`, `anchor`) take a scalar or per-axis data field, or a
  `name = constant(expr)`, besides a literal, so one declaration serves a
  family of values (user principle `0ikeidi`).
- Added: a random-effect block's correlation prior
  `cor(:, ID) ~ LKJCholesky(K, eta)` reads `K` and `eta` from scalar data
  fields as well as literals, on SBBRMI and RKBRMI (and in `prune_responses`);
  `K` is still checked against the block width. Both backends emit `eta`'s
  number into the program.
- Changed: models that differ only in the `gmrf` and `hsgp` values build one
  Stan program and emit one RK program. RKBRMI reads an `hsgp` term's basis
  counts and settings from the bound fit-role inputs `<id>_modes_fit` and
  `<id>_settings_fit` (as the fitted values `<id>_modes`, `<id>_settings`), and
  a `multiresolution` rank from `<id>_rank_fit`, instead of spelling them in the
  program; the basis size is `prod(modes)`. The densities are unchanged.
  ReactiveKernelsPPL's lowering of the emitted program still specializes on a
  changed basis size and on the fitted length-scale floor, which `c` and
  `domain` move (ReactiveKernels snag `rkppl-sized-vect-69604d32`); a changed
  `multiresolution` rank lowers to the same program.
- Added: a frozen replay refuses data whose field holds another value of such
  a setting than the fit used; a new build takes the new value, and so does a
  fresh replay (`freeze_constants=false`) on RKBRMI and of an SBBRMI `gmrf`.
  An SBBRMI `hsgp` replay refuses in either mode; its new build compiles the
  same Stan program.

## 0.2.207

- Fixed: a data column whose element type is erased (`Any[...]`, as an empty
  comprehension over a DataFrame column or `push!` onto `[]` produces) now
  holds its entries' common type when the model is built, so it is read
  exactly as the same values written typed: `Any[[0.5, -0.2], [1.3]]` is a
  per-cell vector response like `[[0.5, -0.2], [1.3]]`, and `Any[0.3, -0.1]`
  a scalar response like `[0.3, -0.1]`. The entries are not converted and the
  caller's column is not replaced.
- Fixed: on RKBRMI, an in-cell observation `y[i] ~ law(...)` in
  `@plate for i in eachindex(y)` over an empty `y = Any[]` is bound as an
  empty column of the arrays its law observes per cell (a matrix per cell
  for a matrix-variate law, otherwise a vector; `Int` for a discrete law,
  otherwise `Float64`), as `Vector{Float64}[]` would be. It observes zero
  cells, contributes zero likelihood, keeps the priors and differentiates
  under native Reverse. It previously became one whole-vector event and
  failed when queries were prepared (`zero(::Type{Any})`). Other empty
  columns a cell reads have no law to type them: their queries evaluate,
  but native reverse AD needs them constructed with a concrete element type.

## 0.2.206

- Fixed: `brm_start` on RKBRMI writes an ungrouped anisotropic
  `hsgp(...; iso=false)` length scale, as `fitting.md` documents, through
  `term_inits` and `base=:prior_median`. A scalar broadcasts over the axes; a
  vector holds one value per axis, in formula order. Before, both raised
  `type NamedTuple has no field rho`.
- Fixed: an ungrouped one-axis `hsgp(x; iso=false)` on RKBRMI. Its `brm_start`
  raised the same error, and `brm_coordinate_transport` refused it because the
  emission samples `rho_iso` while the record named `rho_1`.

## 0.2.205

- Fixed: `brm_start` on an unconditioned `RKBRMI` (every response omitted,
  the prior spelling) died in a `MethodError` from an internal coordinate
  lookup. Such a program has zero sampled coordinates, so
  `brm_start(rk, (;))`, `base=:prior_median` and the start from the data
  `brm_start(rk)` (`chains = n` too) now return the empty point, and
  `brm_start(sb, model)` the empty Stan point. An initial value
  (`brm_start(rk, (; z=...))`, `term_inits`, `component_inits`) is refused by
  name on both backends, since every declaration is drawn in generated
  quantities; before, SBBRMI refused a population coefficient with an
  unrelated descriptor error. `brm_coordinate_transport` pairs the two empty
  layouts instead of refusing the unconditioned plan, and reads BridgeStan's
  `[""]`, its name list for a model without parameters, as no names.

## 0.2.204

- Added: epidemiological delays with a censored primary event, a censored
  secondary event and truncation, epidist's marginal model, as two response
  families spelled as in CensoredDistributions.jl:
  `y ~ double_interval_censored(LogNormal(mu, sigma); primary_event =
  Uniform(0, pwindow), lower, upper = obs_time, interval = swindow)` and
  `y ~ primary_censored(dist, Uniform(a, b))` (exact secondary times). The
  delay law is `LogNormal`, `Gamma` or `Weibull` (Julia parameterizations,
  linear predictors allowed); windows and bounds are numbers or data
  columns; `weighted(..., fweights(n))` takes aggregated rows. Closed-form
  primary censoring, with upper-tail window probabilities taken from
  survival differences. SBBRMI emits the Stan family `brm_censored_delay`;
  RKBRMI observes the same Julia law with native Enzyme Reverse gradients.
  Both check every observation window against its truncation bounds when the
  model is built. The Julia objects `PrimaryCensored` and
  `DoubleIntervalCensored` (exported) implement `cdf`, `ccdf`, `pdf`,
  `logpdf` and `rand`.

## 0.2.203

- Added: RKBRMI lowers formula-level `weighted(family, fweights(n))` and
  `weighted(family, weights(w))` on a response whose law reads an assigned
  value (`p = reader(...)`, `loc = 2 * mu`), the reader route. The base family
  is observed with each row's log density multiplied by its weight, as on the
  regression route and in Stan; the weight column binds under its own name,
  so `reprocess` and `rk_design_binder` rebind and check it from a new
  design. A mixture with per-row component weights is on that route and
  takes `fweights` too. Before, construction failed with "value-based
  response weights need an authored response". Analytic `aweights`, a
  weighted `mi()` response and a weighted response holding one array per
  subject are refused with an RK-attributed error.

## 0.2.202

- Added: a function-style component called inside a `@plate` cell may return
  named values. The cell reads each as a property of its call,
  `cell ~ bounds(x[i], scale)` then `cell.lo` and `cell.lo + cell.hi`, on SBBRMI
  and RKBRMI. Each cell owns its call's locals. Before, such a call was refused.
- Changed: inside a cell, a bare read of a named-return call or an unknown
  field is refused with the returned field names.
- Compatibility: the StanBlocks floor and test pin advance to 0.2.31, which
  collects a tuple-valued plate cell as an array of tuples.

## 0.2.201

- SMC worker parallelism from ReactiveKernelsPPL 0.1.100 (now the floor).
  Every `brm_smc_tune` method (initial, prepared Laplace, cumulative update,
  carried window) forwards `grid_start`: with `pilot=:adaptive`,
  `grid_start=:independent` runs the grid pilots concurrently on up to
  `options.nthreads` workers, each from its own copy of `init`, after the
  reference pilot. The schedule is the same for every worker count but
  generally differs from the default sequential one (`grid_start=nothing`).
  Every `brm_smc_resume` method (initial/update, Laplace, carried window) takes
  `nthreads=n` to replace a production checkpoint's saved worker cap; the
  resumed cloud does not depend on it.

## 0.2.200

- Fixed: a prior-only program (the response column omitted) with a
  `VonMises` prior builds on SBBRMI and draws the prior with the family's own
  RNG, a sampled mean included. Before, it failed at transpile: family
  `brm_von_mises` requires companion `brm_von_mises_lcdf`. The family's own
  support `[mu - pi, mu + pi]` is now declared through StanBlocks'
  `autokwargs` hook, as `InverseGaussian`'s is since 0.2.198, not spelled as
  bounds whose prior-only re-draw was the truncated draw (todo `0w0ukqc`). A
  fitted program is unchanged: the parameter keeps its moving-support
  declaration and the plain kernel; a bound that narrows the support still
  restricts the prior.
  - `[compat] StanBlocks = "0.2.30"`: StanBlocks reads the qualifier of an
    argument-dependent support from 0.2.30 on (snag
    `autokwargs-suppo-412811f6`).

## 0.2.199

- Direct plate reads of component results use the returned cells' axis on both
  backends. Shared arrays inside a component no longer count as rows of a
  formula predictor. Nested components and plates whose extent comes only
  from the returned value keep the same direct indexing syntax.

## 0.2.198

- Fixed: a prior-only program (the response column omitted) with an
  `InverseGaussian` prior builds on SBBRMI and draws the prior with the
  family's own RNG. Before, it failed at transpile: family
  `brm_inverse_gaussian` requires companion `brm_inverse_gaussian_lcdf`. The
  positive half-line is now the family's declared support (StanBlocks'
  `autokwargs` hook), not a bound whose prior-only re-draw was the truncated
  draw (todo `0w0ukqc`). A fitted program is unchanged: the parameter keeps its
  `real<lower=0>` declaration and the family's plain kernel. A bound that
  narrows the support, `restricted(InverseGaussian(m, l); lower=0.5)`, still
  restricts the prior.
- The prior-program warning at construction no longer says that every
  parameter forward-simulates when the program holds a `restricted` prior. It
  names those priors and says that one whose mass inside its bounds moves with
  a sampled argument or bound stays a sampled parameter, with what its bounds
  and arguments read, so that program needs a sampler (StanBlocks' rule since
  0.2.72).
- Not yet: a `VonMises` prior's prior-only program still asks for
  `brm_von_mises_lcdf`. Declaring its moving support `[mu - pi, mu + pi]` the
  same way needs StanBlocks snag `autokwargs-suppo-412811f6`.

## 0.2.197

- Added: SVI units own ordinary sampled coordinates read only within that
  unit, with their authored priors, Jacobians and global hyperparameter
  gradients counted once. Structural supports close over joint laws and
  transforms; `global_coordinates` explicitly promotes coordinates. Diagnostics
  expose the inference ownership separately from scientific NLME roles.
- Changed: SVI streaming and packed draws preserve ordinary coordinate
  identities through absent, returning and new units. A retained unit's
  Gaussian replaces its complete authored prior bucket.
- Compatibility: authored SVI queries use the batched preparation API from
  ReactiveKernels 0.1.84 / ReactiveKernelsPPL 0.1.96, sharing binding validation
  and packed exemplars while retaining one derivative workspace per bucket.
- Compatibility: a unit's dormant scalar/vector branch differentiates under
  ordinary Reverse from Enzyme dogfood `0fa85df3` (repairs `74abe90` and
  `7ac58eb`), which the test bootstrap's `b2bb4cb6` (libEnzyme `f392eba5`)
  contains.

## 0.2.196

- `brm_model_view` of a bound model lists each component call's own
  declarations in Roles and ranks under the exact call path, such as
  `left › b`, with the same roles and ranks as model-scope declarations. Before,
  only the call's result row appeared. A local observation of a data input marks
  the caller's column as response data. Locals of a call inside a plate cell are
  marked as declared once per iteration. RK and descriptor views resolve a local
  parameter's dimensions by its exact call path. `brm_components` records are
  unchanged.

## 0.2.195

- Changed: one-unit NLME calls from several tasks run in parallel. Each call
  takes its own evaluator from the model's pool: the model's first, or one
  prepared when calls first overlap (its own partition query, compact sampler
  and work points), at most one per Julia thread. Before, every call waited on
  one lock. NLMEEstimation's `WarmupHMCNUTS(threads = true)` therefore runs
  BRM subjects in parallel. A unit's value and gradient do not depend on which
  evaluator computes them. The NLMEEstimation floor is 0.2.5, the first
  release with `threads = true`.
- Changed: overlapping one-unit calls need Enzyme's concurrent-GC repair
  (dogfood from `72d5d25`). Without it, the first call that would prepare a
  second evaluator raises a `BRMNLMEViewError` instead of running beside
  other threads unprotected. `brm_two_stage_fit` with `threads > 1` now
  samples its units through that pool instead of its own worker copies of the
  model, and therefore needs the repair as well.

## 0.2.194

- A function-style component's formula priors read its model-valued inputs on
  `SBBRMI`, as its scalar priors already did: with `center` and `scale` bound
  by the caller to parent parameters, `effect(value, Intercept) ~
  Normal(center, 0.4)` and `sd(value, unit) ~ Exponential(scale)` build. They
  failed with "model-level prior references `center`, a extension", so one
  definition could not be called with different prior locations or scales.
  `RKBRMI` already read such inputs in random-effect SD priors. Its
  population-effect prior arguments remain numeric constants or data, in a
  component as at model scope.

## 0.2.193

- Model views support readable `text/plain` output through HTMXObjects 0.1.65.
  All eager tabs and disclosures retain their labels, source, documentation,
  roles and dimensions without internal node dumps. Component defaults and
  model helpers remain unevaluated. The dependency floor and test pins require
  the shared renderer fix.

## 0.2.192

- `brm_coordinate_transport` pairs random-effect blocks declared inside a
  function-style component call, plain or stratified `gr(g, by=b)`, including
  per-margin scale laws, Student-t mixing weights, repeated calls over
  different groupings and nested calls. Each call's scales, correlation factor
  and standardized draws pair by its own level labels and margins, and their
  addresses carry `scope = <call path>`. Such models previously refused with
  unpaired `<call>.<block>.z` and `<call>_<block>_z_flat` coordinates.

## 0.2.191

- `brm_start`'s `component_inits` takes the top-level `inits` grammar for a
  component call's own linear predictors and `|ID|` blocks on both backends:
  `:left => (; rise=(; Intercept=v), rates=(; sd, cor, effects))` sets the
  call's link-scale coefficients and its shared block's scales, correlation
  (or covariance `rates = Σ`) and per-level effects, keyed by the call's own
  levels. `base=:prior_median` also fills component-local population
  coefficients and `|ID|` scales from their priors. RK still needs no
  compiled Stan model.

## 0.2.190

- Model views render callable and authored Julia documentation through the shared
  HTMXObjects documentation projection. Unresolved Documenter references retain
  their formatted labels without broken links, and doctest examples use Julia
  highlighting in offline HTML. Original source and documentation records remain
  unchanged. Requires HTMXObjects 0.1.62; both test pins select that release.

## 0.2.189

- Function-style components returning one internal plate's collected values
  preserve one value per cell, including vector-valued cells, aliases and
  nested component calls. Parent reads such as `field[row[i]][channel[i]]`
  select the declared row and then its channel on both backends.
  ReactiveKernels keeps selected-row locality instead of reading a flattened
  whole; StanBlocks preserves the cell axis of its matrix storage. Formula
  offsets still require scalar cells.

## 0.2.188

- **Breaking:** BRM now supports only `SBBRMI` (StanBlocks/Stan) and `RKBRMI`
  (ReactiveKernelsPPL). Removed `TuringBRMI`, its Turing/WarmupHMC extensions,
  the `turing_*` helpers, and `VBRMI` with its materialization/parameter-part
  API (`MaterializedColumn`, `getbroadcast`, `Part`, `push_parts!!`, `nparams`,
  `lprior!`, `vbroadcasted`, `vmeta_sampling_rhs`). Build one of the supported
  backends from the same `BRMI`; use that backend's fitting, replay and
  prediction APIs. Custom terms must use the StanBlocks or RK extension hooks.
- Removed the retired backends' dependencies, tests and documentation. Shared
  preparation and mixture-response checks remain on the supported paths. The
  macro explorer now exposes its StanBlocks pipeline without the VBRMI stages.
- Incomplete constrained-coordinate carriers now report their emitted output
  and missing coverage instead of an undefined-variable error.
- Refresh the test and macro-app HTMXObjects pins for the model-view floor.
  The test environment requires DiffEqBase 7.21.6 and OptimizationBase 5.10.1
  so their extensions load with the current Enzyme rule hooks.

## 0.2.187

- Added `gmrf((east, north), time[, channel]; multiresolution=m)`: a bounded
  sparse Matérn-1 spatial basis of shared stationary OU paths. A position uses
  the same normalized weights at every time, preserving exact temporal
  exponential covariance, marginal SD and channel separability on irregular
  reads. Spatial covariance remains approximate. Each innovation retains only
  its readers' times; the representation does not complete a position × time
  grid. It shares the existing hyperparameter addresses and priors.
- The new model-scope form supports `RKBRMI` and `SBBRMI`, native analytic
  derivatives, Stan prior draws, frozen replay and physical coordinate
  transport. Its fitted support construction is part of the native graph.
  Exact reader locality requires ReactiveKernels 0.1.83, now the RK floor.
  Native Reverse needs Enzyme dogfood's literal-dictionary callback repair and,
  from dogfood `7a1f689c`, its repeated-reader repair, without which gradient
  cost grows with support metadata × reads. The test bootstrap's RK
  `b0ffcdc1` and Enzyme `d3189970` pins contain both.
- Clarified the `neighbors=` contract: local Matérn × OU kernel entries are
  exact, but the implied joint temporal covariance is approximate on irregular
  read support. Its numerical behavior is unchanged.

## 0.2.186

- Offset-only predictors on RK retain the declared rows of scalar plate or
  component outputs and fixed data expressions before an explicit gather onto
  observation rows. They no longer need a raw fixed offset or an estimated
  coefficient to establish those rows.

## 0.2.185

- Added: a function-style `@brm` component may return several named values,
  `return (; recovery, decay = exp(log_decay))`. The parent reads each one as a
  property of the call, `kin.recovery`, in formulas, offsets, likelihoods and
  model-scope bindings, on SBBRMI (a Stan tuple) and RKBRMI (the call's native
  local). Each field keeps its own row axis, and the call's priors and
  correlation blocks remain one scope. `brm_output(d, :decay; submodel=:kin)`
  addresses a returned field. A plate cell reads a field through a model-scope
  binding (`recovery = kin.recovery`, then `recovery[i]`).
- Changed: a bare read of a named-return call, an unknown field, a positional
  tuple return, a direct `kin.recovery[i]` plate read and a named-return
  component called inside a plate cell are refused with the supported spelling.
- Fixed: SBBRMI compiles a component whose formulas group by one of its inputs
  (`(1 | p | unit)` with `unit` an input); the raw label column is no longer
  passed to the Stan submodel, which reads its own index data. RKBRMI also
  aligns a field grouped by a label input with the parent's rows, so it can
  enter a formula offset.

## 0.2.184

- `brm_start` accepts top-level matrix parameters on their constrained scale
  on both RK and compiled Stan. Initialize `L ~ LKJCholesky(K, eta)` with
  `(; L=lower_factor)`, preserving its exact matrix shape. RK uses its native
  transform without compiling Stan. Unaddressed base values and caller-owned
  inputs are preserved, as for scalar, vector and component-local starts.

## 0.2.183

- Model views label every authored member of a joint response as response data,
  without exposing the synthetic joint operation as a model quantity.
- Component input tables distinguish positional and keyword arguments, required
  inputs and optional defaults. Defaults are shown as source, never evaluated.
- Unbound builders label potential data inputs without claiming that sampled
  names must be supplied as data; their role depends on binding.
- Callable documentation includes helpers nested in formula containers such as
  mixture-component lists, without executing those helpers.
- Helpers from modules with no documentation registry no longer crash a view.
- Requires HTMXObjects 0.1.60 for code-block PDF tagging without whitespace-only
  PAC Quality warnings, preserving complete source and syntax highlighting.
- Empty input inventories explain their absence. Component-instance and authored
  documentation headings use readable paths while retaining exact tuple paths in
  provenance. Unknown derived shapes explain the limits of inspection.

## 0.2.182

- Fresh-process recovery of adaptive `brm_smc_tune` pilots (`resume=partial`
  or `checkpoint_dir=dir, resume=true`) works for `RKBRMI` models whose laws
  are bound in the generated RK program module, such as `gmrf(...)` fields.
  Before, the checkpoint named that process-local module and a fresh process
  failed with `UndefVarError: ##RKEmittedModels#…`. **Breaking:** the RK backend
  now requires ReactiveKernelsPPL 0.1.98, which normalizes generated-module
  names in SMC checkpoint identities; raise an environment pinning an older
  ReactiveKernels commit to `b0ffcdc1` or a descendant. Checkpoints written
  with ReactiveKernelsPPL 0.1.97 or earlier are not migrated: resume them with
  their original versions or write a new initialization checkpoint.
- At this release, RK `main` from `8e67ee82` broke selective carried-window
  tuning (`coordinates=:auto`) with a `MethodError` in `_cell_joined_rows`, so
  BRM's test pin used `b0ffcdc1`. RK `df3eb8872` repaired the regression; the
  test pin now includes that repair through `db6bca825`.

## 0.2.181

- `brm_start` accepts `component_inits` on both backends, keyed by authored
  component call paths and local parameter names. Repeated and nested calls
  can set independent scalar, vector, simplex and correlation Cholesky starts
  on the constrained scale; RK uses its native layout without compiling Stan.
  `base=:prior_median` now visits component-local scalar priors recursively,
  resolves each call's bound inputs and preserves explicit-start precedence.
  Numeric bases retain every unaddressed value.


## 0.2.180

- `SBBRMI` and `RKBRMI` compose heterogeneous scalar `MixtureModel`
  components with known full-real continuous support, including custom
  function families with different argument counts. Each component retains
  its ordinary parameterization and backend mapping.
- RK preserves native component density graphs, including existing
  `_rk_observation_source!` providers, for dependency-aware scheduling,
  native Reverse gradients, pointwise likelihood and source replay.
  Stan preserves component predictive draws and membership outputs.
  Native RK mixture/custom-law predictive generation remains separate work.
- Requires StanBlocks 0.2.25, ReactiveKernels 0.1.81 and
  ReactiveKernelsPPL 0.1.94. Bounded or mixed-support heterogeneous components
  remain outside the supported mixture contract.

## 0.2.179

- `brm_model_view(model)` with HTMXObjects shows data/parameter roles and value
  ranks from the supplied declaration or prepared model. Unknown dimensions
  stay explicit; descriptor carriers retain their own constrained shapes and
  logical identities. Callable documentation and carrier details are accessible
  by click and keyboard. The view performs no fit or implicit backend preparation.
  The macro app and descriptor mount share this reusable view.
- Source, roles/ranks and callable documentation use eager tabs, retaining all
  content in Markdown/PDF and static HTML exports. Plate response labels reuse
  the compiler's cell observation metadata. Export acceptance exercises the
  standard HTMXObjects PDF and bundled-site exporters.
- Authored declaration docstrings use the shared `brm_documentation` records
  in the view, with formatted prose, exact paths and source provenance.
- Source tabs automatically include reachable BRM component definitions from
  `brm_components`, once per shared definition, with all call-instance paths.
  Complete PDF and static HTML exports include every nested definition.
- Component source and declaration details use the shared Julia source printer,
  preserving readable docstrings without exposing internal doc wrappers.
- The macro app's expression cards read `NamedColumn` parent values for their
  data/parameter styling, matching the current column representation.
- Prepared RK views use ReactiveKernelsPPL 0.1.89's read-only parameter metadata:
  constrained shapes, ragged element shapes and free-coordinate counts remain
  separate. Replays retain the build's binding provenance and show current
  parameter dimensions as unresolved when they share an earlier layout.
- Refresh the macro app's HTMXObjects dependency closure to 0.1.58 and its
  StanBlocks/WarmupHMC pins to versions compatible with the current BRM package.

## 0.2.178

- Added: explicit `unit_partition=:connected` for the NLME view and model,
  NLME fits, SVI, per-unit Laplace and joint starts. Structural likelihood
  supports join scientific subjects into complete inference units, retaining
  their original labels, coordinates and every covariance-family instance.
  The default `:subjects` keeps strict subject independence. Connected starts
  carry unit membership; named estimates restore scientific subject labels.
- Changed: SVI streaming retains the partition policy and refuses changed
  membership of a carried connected unit before updating. Complete unchanged
  units and wholly new units can continue.
- **Breaking:** SVI streaming checkpoints use format 2 with explicit
  `unit_partition`; serialize and reload with matching package versions.

- Changed: NLME unit likelihoods and gradients use one strict RK plate
  partition per unit. Preparation checks structural coordinate ownership,
  including dormant branches, and names nonlocal producers or foreign unit
  effects. Sparse curvature retains individual cell supports without numerical
  probing or gradient-based pruning; repeated observations share one support
  during pattern assembly instead of accumulating duplicate coordinate pairs.
  Conditional subsets evaluate only their requested units; full-cohort and
  population-gradient calls retain batching.
- Changed: a one-unit value or gradient writes θ, σ and the unit's η into a
  dense work point owned by its query or plan, instead of an overlay that RK
  copied into a full packed vector on every call, so one unit's cost and
  allocations no longer grow with the number of units.
  `brm_nlme_loglikelihood_and_gradient`, which returns only the η gradient,
  differentiates only the unit's own effects (RK `shared=false`).
- Changed: NLME cell, partition and gradient queries declare their local
  parameter ownership so RK/PPL can cache shared producers before strict
  lowering. Correlated grouped HSGP values remain shared; local coordinates
  retain their structural checks. Shared-coordinate changes refresh the cache.
- Changed: separated CAR fields expose their shared values through row-local
  graph computations, preserving their law, coordinates, inverse and Jacobian.
- Changed: `cdar(...; cor=C)` emits its fixed correlation factor as a sparse
  matrix, so a subject reads only the innovation groups its factor rows store.
- Compatibility: ReactiveKernels 0.1.77 / ReactiveKernelsPPL 0.1.85 preserve
  sparse factors, stratified group reads, grouped HSGP row forms and shared
  plate computations across several observations in one unit partition.
- Compatibility: the strict partitions need ReactiveKernels 0.1.51 and
  ReactiveKernelsPPL 0.1.54 or later, within the new 0.1.77 / 0.1.85
  floor. Scalar inference preserves ordinary indexed predictor expressions
  under strict partitioning, and nested data-only helpers inside sampled
  expressions, such as a lower-bounded shares completion's
  `lower=lo[x_shares_free]`, keep their shapes through strict cell and
  partition preparation. Strict queries also preserve local scan captures,
  grouped-field row reductions and censored ragged factors. Shared-prefix
  binding includes freshly emitted helpers and preserves local reads from
  custom packed vectors. Unit gradients use ordinary Enzyme Reverse, with the
  test bootstrap's Enzyme dogfood pin, which contains `68ed90fe`'s repair for
  matrix-window partition gradients on Julia 1.10.

## 0.2.177

- Require ReactiveKernels 0.1.82 to restore buffer reuse when batched readers
  evaluate vector-valued plate cells. Emitted and printed RK models retain
  live broadcast fusion after binding data, avoiding a temporary vector per
  cell and position.

## 0.2.176

- Applying an explicitly selected Laplace plan over a per-cell `gmrf` field
  and copying its target per chain preserve the direct target's coordinates,
  value and gradient. The dependency floor now includes ReactiveKernels
  0.1.70's supporting copy machinery; the ReactiveKernelsPPL floor is
  unchanged.
- The block importance diagnostic includes the `gmrf` precision prior in its
  weights. For an exactly Gaussian conditional, every proposed draw retains
  equal weight. This diagnostic does not change automatic eligibility:
  unrecognized plate-cell algebra still requires explicit integration.
- The count-field prior-median diagnostic fixture uses a declared log link,
  retaining its ESS assertions under structural automatic selection.

## 0.2.175

- Selecting several fixed-width vector cells preserves their collection
  semantics in ordinary helpers: `f(cells[indices])` can read each selected
  vector with `rows[t]`, including repeated and reordered indices. This works
  with observed responses and omitted-response prior generation. The
  StanBlocks floor and test pin advance to 0.2.26 for its ordinary view helpers.

## 0.2.174

- `LKJCholesky(K, eta)` accepts data-supplied scalar dimensions and
  concentrations in `SBBRMI` and `RKBRMI`, including reusable and nested
  `@brm function` components. RK source retains named inputs while coordinate
  bookkeeping uses their bound dimension. The RK backend requires
  ReactiveKernelsPPL 0.1.97 for data-supplied LKJ dimensions.

## 0.2.173

- Adaptive `brm_smc_tune` forwards nested Treebars progress, cooperative stop
  callbacks, `checkpoint_dir`, `resume` and `overwrite` for initial, prepared
  Laplace, cumulative-update and carried-window targets. Interrupted pilots
  return RKPPL's `SMCTuningPartial`; completed calls still return a production
  tuning schedule. Resume restores the active pilot and skips completed pilots
  after checking endpoint identities and configuration. Uses the RKPPL recoverable-pilot contract introduced in 0.1.91.
  Stops occur at safe boundaries rather than hard deadlines; active checkpoint
  draws can grow even though outer history and retained checkpoint files are bounded.

## 0.2.172

- `SBBRMI` preserves complete fixed-width vector cells in direct reads from
  another `@plate`. Collected and non-last indexed outputs retain their outer
  cell axis with observed responses and with responses omitted for prior
  draws. Scalar, varying-width vector and matrix cells retain their existing
  iteration behavior; model declarations and priors need no changes.

## 0.2.171

- `@brm function component(inputs...) … return value end` defines a reusable
  component for both StanBlocks and ReactiveKernels. Each `result ~ component(...)`
  call owns its local parameters and reads explicit shared inputs. Components
  support nested calls, array-valued results, prior/output inspection and
  changed-data replay. Existing short-form data-to-model builders retain their
  meaning. Returned row axes remain available to formula offsets and selective
  readers; internal plate cells must prove scalar values before flattening.
  `brm_components` exposes definition identity, source and each logical call path.

## 0.2.170

- **Authored declaration documentation:** ordinary Julia docstrings in BRM
  source survive binding, tuple expansion, model composition, source printing
  and replay. `brm_documentation` returns copied `BRMDocumentation` records
  with logical paths, raw Markdown, authored statements and file/line provenance,
  including declarations inside plates and scans. Documentation stays separate
  from executable operations and numerical model identity.
- `brm_documentation_markdown` and scientific model descriptions include the
  authored prose for HTML/PDF pipelines, without treating it as generated
  scientific coverage. Emitted RK artifacts retain the same records in their
  metadata.
- The RK source retains ordinary declaration and plate-local docstrings using
  the emitter's exact names; reusable definition wrappers remain intact during
  lowering and replay. This requires ReactiveKernelsPPL 0.1.93. Complete BRM
  records retain element-specific prior addresses when several coefficients
  share one emitted array declaration.

## 0.2.169

- `SBBRMI` now accepts an omitted `interval_censored(D; upper=hi)` response
  in the unchanged model. Retained bounds and family arguments determine the
  predictive shape; draws use the uncoarsened base law, with no interval
  likelihood or fabricated lower endpoints. Observed interval densities and
  predictor-side interval terms retain their semantics.
- A censored response may also omit its own endpoint: keep
  `y ~ censored(D; lower=y)` or `upper=y` unchanged and omit `y`. Only that
  endpoint disappears for prediction; independent retained bounds still clamp
  the draw. Observed CDF/CCDF evidence and missing-independent-bound validation
  are unchanged. Requires StanBlocks 0.2.24, now the package and
  test-environment minimum.

## 0.2.168

- `RKBRMI` accepts scalar `@plate` outputs in `offset(field)`, including
  non-last outputs and elementwise assignments over them. The offset reads
  the declared field with coefficient one, preserving selected-row queries
  and native reverse gradients. Vector-valued cells cannot establish row
  alignment merely by having the same flattened length.
- An explicit gather of an opaque helper's output, `field = whole[position]`,
  supplies the index vector's row axis to a model-value offset. The helper
  remains one graph value; its arguments and output length do not implicitly
  declare the consumer's rows.

## 0.2.167

- `RKBRMI` accepts mixture probabilities computed from linked or hierarchical
  predictors, including predictors used only in the weights. The ordinary
  value graph retains their coefficients, group effects and authored priors.
  Probability vectors may be shared or vary by observation; native density,
  Reverse gradients, pointwise likelihood and printed-source replay use the
  same normalized mixture. Requires ReactiveKernelsPPL 0.1.90.

## 0.2.166

- **Breaking: conservative automatic embedded Laplace.** `integrate=:auto`
  now requires recognized conditional log-concavity. Affine Normal rows retain
  exact integration with exact algebra; log-Poisson, logit-binomial/Bernoulli
  and log-NegativeBinomial2 rows with sampled shape retain the local ESS gate.
  Arbitrary plate cells and model-value observations stay sampled even if an
  ESS check at one point would pass. To retain a previously automatic cell
  partition, explicitly name its declarations/groupings in `integrate=(...)`
  after assessing the approximation. Supported explicit selections remain
  available, including effects on scale/shape. Automatic diagnostics now hold
  rejected dispersion terms sampled. All model factors remain in the target;
  no eligible terms gives the full sampled model. Structural concavity and
  passing local ESS do not certify Gaussian approximation accuracy.

## 0.2.165

- **Structure first, priors last:** `SBBRMI` accepts a cell-local prior after
  statements that read it, keeping the draw in its original plate. Prior
  dependencies are emitted before their readers; ordinary cell statements and
  scan steps retain their relative order. The authoring guide and executable
  examples cover scalar, vector, coefficient, scale and correlation priors,
  composed structural/prior builders, and complete printed-source replay.

## 0.2.164

- **Large `gmrf` fields under the sparse Laplace:** a warm value and gradient, at
  the prior medians, with one Normal field and one CPU, took 13 s at 202,500 states
  on 8 probes (was 93 s), and 134 s at 1,698,450 states on 4 probes. A component
  that is exactly one `gmrf` field now has a reference that sees the data, built at
  the prior medians, and the determinant takes one Lanczos recurrence per probe.
- **Lower fixed-probe noise:** `MatrixFreeLaplace(control_variate=true)`, the default,
  rebuilds a field's reference at each point and estimates only the difference from
  it. At 10,000 states and 8 probes the gradient's SD across probe seeds fell from
  about [9, 11, 14] to [2.2, 2.1, 1.6] at the same fixture, for a cost of about twice
  as much per probe. The value and gradient stay one fixed-probe function and its
  exact derivative. Correlated channels read at different places keep the plain
  estimator.
- **Fields read by the same rows** integrate through a reference that couples them
  (two at 20,000 states 10 s against 24 s exact; three at 30,000 states 17 s against
  83 s). Its control variate tracks only the fields' priors: a field's scale keeps
  the plain estimator's noise.
- **`brm_start(...; base=:prior_median)` and the Laplace default start** now read a
  prior with an inner call, such as `LogNormal(log(0.2), 0.5)`. Those coordinates
  used to keep the unit point, so such a model's starts change.
- **Breaking:** `MatrixFreeLaplace` no longer takes `logdet_step` or `logdet_extent`:
  remove them, there is no quadrature. `target.algebra.diagnostics` replaces
  `quadrature_nodes` with `gradient_rank` and adds `ritz`, `tracked_iterations` and
  `tracked_shift`. Matrix-free values change within their probe error, and a problem
  with a `gmrf` field prepares its reference at the prior medians rather than at zero
  coordinates, so an SMC checkpoint from an earlier version is refused.

## 0.2.163

- **Mesh `gmrf` densities at very short spatial lengths:** `RKBRMI` and `SBBRMI`
  no longer produce `NaN` from an overflowing intermediate inverse-squared
  length, such as `exp(-365)`. A rescaled sparse Matérn factor and normalized
  quadratic retain the same proper prior, including its normalizer and
  derivatives, without clipping lengths or changing their declared support.
  Representable tail densities remain finite; a log density below Float64's
  range evaluates to `-Inf`. The numerical precision operations and
  Stan prior RNG use the same scaling.

## 0.2.162

- `RKBRMI` of a model with an omitted response now samples what `SBBRMI` samples,
  so `brm_coordinate_transport` and `brm_nlme_joint_map` work for every partial
  fit of a shared `|ID|` block whose laws they cover.
  - A plate cell no longer keeps a predictor, margin, scale or cell latent that
    no kept likelihood reaches because a local or a reported indexed output
    still reads it (`loc = base[i] + …; z[i] = loc; y[i] ~ Normal(loc, s)`). A
    statement reading only what kept likelihoods reach and data stays, reported
    output included.
  - A shared block `SBBRMI` cannot split stays whole on RK too: Student-t,
    R2D2-derived or known scales, stratified `gr(g, by=b)` and blocks joining
    several grouping columns. This replaces the two refusals of an R2D2 block
    and of a joined block under an omitted response.
  - A long-format response omitted in a cell (`y[rows[i]] ~ …`) now plans, and an
    `effect(lp, …)` prior on a predictor only an omitted response reads no longer
    fails planning.
  - **Breaking:** on such a partial fit `RKBRMI` has fewer coordinates. Saved RK
    draws of the old layout do not fit it, and a cell output computed from what no
    kept likelihood reaches (`z[i]` above) is no longer in the RK program. Read
    it from `SBBRMI`'s generated quantities, or fit with the response bound.
    Fully observed programs and designs that omit a response the fit observed
    (`reprocess`) are unchanged.

## 0.2.161

- **A Laplace mode search no longer stops short of the mode on a stiff
  conditional:** the search stopped at a "rounding floor" whenever a plain
  Newton step's decrement was below 1e-8 and had stopped shrinking fourfold.
  Newton can converge that slowly far from the mode, on an exponential's slope
  or along a stiff curved valley, so the marginal came from a point that was not
  a mode, with no counter or warning. All three problems did this. A decrement
  now counts as at its floor only when the gradient no longer follows the Newton
  model; otherwise the search continues to `tol`, the floor, or `maxiter`, which
  scores `-Inf` and counts in `unconverged`. Log densities and gradients change
  at the points this affected: on one synthetic stiff fixture, values moved by
  0.58 and 0.16 nats and the integrated gradient fell from 7.3 and 1.4 to below
  1e-6. Values at five points of that fixture that had converged are identical.

## 0.2.160

- **`gmrf(...; neighbors=m)`, a space-time field bounded by its reads:** in place
  of `mesh=`, a nearest-neighbour (Vecchia) sparse precision with one state per
  channel at each distinct `(east, north, time)` the rows read. The covariance is
  the exact continuum Matérn × OU at those reads: no mesh boundary inflation, no
  resolution floor, no interpolation. The conditioning set is the approximation;
  channels stay exactly separable, and positions read at the same times keep the
  exact OU correlation across every gap. Same parameters, addresses and priors as
  the mesh form, on SBBRMI and RKBRMI. Frozen replay reads fitted pairs only. One
  model-scope field: not in `@plate` cells and not integrated by embedded Laplace.
  Adds a dependency on Bessels.jl.
- **A `gmrf` term with only default priors transpiles from any `mod`:** the mesh
  form named its submodel in the caller's module and failed with `Could not find
  _sb_gmrf`.
## 0.2.159

- **`RKBRMI`: a model-scope `gmrf` prior can read data.** `sd(field, gmrf(...))
  ~ Exponential(scale)` and a `length_scale` law failed at construction with
  `references unknown name scale`, naming the field's internal `sigma`, while
  `SBBRMI` accepted them. These laws now take what the 0.2.156 priors take: a
  numeric data scalar, an expression or element over numeric data fields, a
  literal assignment or a `name = constant(expr)`. The prior-median and Laplace
  default starts read the value, which stays in the fitted program: a design may
  omit the field or repeat its value, and one that changes it is refused unless
  `freeze_constants=false`.

## 0.2.158

- **Sparse Laplace cells that static Enzyme cannot take forward over reverse:**
  `brm_laplace_problem`, NLMEEstimation's `laplace_marginals_and_gradient` and so
  `check_protocol` no longer raise `EnzymeRuntimeActivityError` on such a `@plate`
  cell. The cell group moves to reverse over forward the first time a product
  raises that error, with a warning naming the group. Values are the same and the
  group allocates up to about twice as much. Any other error still propagates. A
  cell that reduces its data with `unique!(sort!(...))`, as a PK event reader's
  step lengths do, needs this before Enzyme dogfood `0ed41342`, which takes that
  cell forward over reverse itself.

## 0.2.157

- **NLME view of a single-level categorical:** `brm_nlme_view`, and with it
  `brm_nlme_model` and `brm_nlme_fit`, no longer raise ``internal: declaration
  `x` owns no packed coordinate`` on a model with a categorical term that
  observes one level, such as `log(s) ~ 1 + g` with `g` an integer column holding
  a single value. Any declaration the RK layout holds at zero length owns no
  coordinate, so the model has the partition, likelihoods and gradients of the
  same model without the term. The named estimates list the term with no levels
  (`g = []`), and that value starts the model again through `inits`.

## 0.2.156

- **`RKBRMI` random-effect and term priors can read data.** A formula-route
  prior such as `sd(:, p) ~ Exponential(prior_sd_scale)` failed at construction
  with `references unknown name prior_sd_scale`, naming an internal array, while
  `sigma ~ Exponential(prior_sd_scale)` and `SBBRMI` accepted it. Random-effect
  scale priors (including per-margin, truncated and stratified blocks), Student-t
  `ranef(:, id) ~ TDist(nu)` degrees of freedom, `hsgp` priors and an `r2d2`
  `reference_scale` now take a numeric data scalar, an expression or element over
  numeric data fields, a literal assignment (`rate = 1.7`) or a
  `name = constant(expr)`. `SBBRMI` still refuses a data `reference_scale`. Data
  of another type, such as a vector of labels, is refused naming the field.
- `gp`, `car`, `cdar` and `dar` priors, which otherwise take literals only, now
  take a numeric data scalar, an element of a data vector or a
  `name = constant(expr)` as its number. `dar`'s scale prior stays `Normal`.
- These values stay in the fitted program, as a scalar parameter's prior does. A
  design may omit the field or repeat its value; one that changes it fails
  `reprocess`'s program check unless `freeze_constants=false`, and no design binds
  it. Value-route models (`@plate` cells and assignments reading predictors) still
  bind these scalars as before.

## 0.2.155

- **In-cell `MvNormal` on `SBBRMI`:** a `@plate` cell may observe its slice
  jointly, `y[rows[i]] ~ MvNormal(location, covariance)`, with the covariance
  computed in the cell, inline or bound to a local. It lowers to the BRM-owned
  family `brm_mvnormal`, which is Stan's `multi_normal` with log density 0 for
  an empty slice, as in Julia. The pointwise log likelihood is one value per
  cell and the predictive draw is the slice. Only the two-argument constructor
  is accepted in a cell. `RKBRMI` already had this since 0.2.86, and both
  backends now give the same density and gradient.
- **`Diagonal(v)` and `diagm(v)` on `SBBRMI`:** both lower to `diag_matrix(v)`,
  in a cell and in a model-scope expression.

## 0.2.154

- **A replay refuses a changed scalar a prior reads by name:** `reprocess`,
  `RKBRMI(brmi, state)` and `rk_design_binder` accepted a design holding another
  value of a numeric scalar the program reads by name, such as `s` in
  `sd(:, p) ~ Exponential(s)` over a `@plate` model, and evaluated the new prior.
  They now raise "changes the fitted RK program", naming the field, as they
  already did for a scalar the program holds as a literal. This covers every
  sampled parameter's law, a `@plate` cell latent's included, and a scalar read
  through an assignment. A scalar only a cell's values or observations read
  stays a design input. `freeze_constants=false` still rebuilds with the new
  prior.
- **`rk_design_binder` refuses what `reprocess` refuses for held scalars:** after
  the first design of a kind, a design holding another value of a scalar the
  program holds as a literal was accepted and the value ignored. It now replays
  in full, which refuses it. An input read only by `constant(...)` still keeps
  its fitted value.
- **A data-free replay supplies an omitted scalar a `@plate` cell reads:**
  `RKBRMI(builder(design), state)` on a design without such a scalar raised a
  lowering error, because the builder had declared no data field for it. It now
  takes the fitted value, as every other replay of an omitted scalar does.

## 0.2.153

- **The sparse Laplace compiles each `@plate` cell's Hessian block at a bounded
  Enzyme batch width.** A cell's block used to compile one batch over all its
  integrated coordinates, and that compile grows faster than the batch: a cell
  integrating 121 coordinates had not compiled after 2 h 37 min. Cells now
  compile batches of 4 directions and take as many as they need, so preparation
  no longer depends on how many coordinates a cell integrates. Values and
  gradients are unchanged.
- **`brm_laplace_plan` and `brm_laplace_problem` take `progress`:** a Treebars
  node or a backend symbol such as `:state`. It reports the width screening, the
  importance diagnostic and each observation, with a node naming a Hessian
  block's size for the time Enzyme compiles it. `brm_laplace_problem` accepts it
  only with `integrate`, and raises otherwise.

## 0.2.152

- **Fix:** on RKBRMI, `addprop` over a plate output at model scope reads the
  plate's rows again, assigned first (`scales = addprop(reads, a, b)`) or written
  in the law. Since 0.2.125 it received one array per cell, so preparing the model
  worked and evaluating it raised `MethodError: no method matching
  ^(::Vector{Float64}, ::Int64)`. An authored whole-value helper still receives
  one value per cell.

## 0.2.151

- **Periodic `hsgp` under strict type stability:** `NLMEEstimation.check_protocol`
  on a model with `hsgp(x; cov=:periodic, period=...)` no longer raises
  `TypeInstabilityError` in a project that sets ReactiveKernels'
  `type_stability = :error`, at any `k`. The harmonics, and every value computed
  from them, are unchanged.

## 0.2.150

- **`rk_draw_values` reads a `@plate` output per cell:** a draw's value has one
  entry per cell in the cell's own shape (a number, a vector or a matrix), as
  the model reads it, so `value[t][h]` is cell `t`'s entry `h`.
- **`rk_draw_tree(rk, U, names)` labels a `@plate` output by cell:** `:draw`,
  `:cell`, then `:element` for a vector cell or `:row` × `:column` for a matrix
  cell.
- **Breaking:** for vector and matrix cells, `value[i]` was the `i`-th entry of
  the concatenation and is now cell `i`. `reduce(vcat, value)` gives the
  previous vector for vector cells. Number cells hold the same values, and
  their axis in `rk_draw_tree(rk, U, names)` is now `:cell` instead of
  `:element`. `rk_draw_tree(values)` on an `rk_draw_values` result now raises
  for vector and matrix cells, because it sees only shapes; use
  `rk_draw_tree(rk, U, names)`. A raw ReactiveKernels query of the output's name
  and `brm_cut_family((:conc,), …)` keep the concatenated value.

## 0.2.149

- **Prior-median starts of native fields:** `brm_start(rk, inits;
  base=:prior_median)` and `brm_start(sb, model, inits; base=:prior_median)` no
  longer raise on a model-scope `gmrf` term. Its SD and both lengths start at
  their prior medians, the default `LogNormal(0, 1)` included; a known
  hyperparameter stays known, `term_inits` wins, and the field keeps its neutral
  coordinates.
- **Laplace default start of native fields:** the default population point of
  `brm_laplace_plan`, `brm_laplace_mode` and the ladder now takes the same
  medians for those hyperparameters instead of holding them at one. A model with
  a `gmrf` term can therefore start somewhere else than under 0.2.148.

## 0.2.148

- `rk_design_binder` keeps its cached lowering for a design that omits a numeric
  scalar the program reads as data, such as the argument of a correlated block's
  `sd(:, p) ~ Exponential(scale)`. The design takes the fit's value, as on
  `reprocess`. It used to replay every design of that kind in full, on a training
  fit and on a data-free `RKBRMI(builder(design), state)` alike.
- **Fix:** when a design held another column equal by value to such an omitted
  scalar, the binder read the scalar from that column, so a later design that
  changed the column silently changed the scalar. A later design now binds the
  fit's value, the same as `reprocess`.

## 0.2.147

- `brm_laplace_problem`'s conditional modes are found by continuation from
  the prior, in all three problems (sparse, structured and whole-model). Every
  evaluation starts the integrated effects at their prior's mode and follows
  each unit's mode while the likelihood's weight rises from 0 to 1 in 10
  stages. The marginal is evaluated at the mode connected to the prior's mode:
  it still does not depend on earlier evaluations, and as the population
  parameters vary it changes only where that mode ceases to exist. Before, a
  search from zero effects with the full likelihood could reach a different
  mode of a multimodal conditional at points millionths apart, so the marginal
  jumped and an outer gradient sampler saw a discontinuous target.
  Log-concave units and sparse components (closed-form rows with no effect on
  a second parameter, no cells) take one stage; their values and gradients
  are unchanged. `maxiter` now applies per stage.
- `target.folds` counts evaluations in which a unit's followed mode needed a
  modified Newton step after the first stage (the mode ceased to exist on the
  way; the marginal can jump near such points). The sparse problem's
  `target.search` records the last evaluation's stages, Newton steps,
  shifted steps per stage, largest shift, refused full Newton steps
  (`backtracks`), final decrements and `stop` reason (`:tol`, or `:floor` for
  the rounding floor) per component.
- The sparse problem damps each component's Newton step by its own shift
  where its curvature is not positive definite, in exact and matrix-free
  algebra. Before, one shift served all components, and a component damped by
  another's curvature could stop early on its rounding-floor rule, about
  1e-4 nats from its mode, with a gradient error of the same order. The
  rounding-floor stop now applies only to plain Newton steps, in all three
  problems.

## 0.2.146

- **`brm_svi_fit` and `brm_svi_resume` evaluate a batch on several tasks:**
  `threads` (default `Threads.nthreads()`, capped at the batch size) divides an
  iteration's unit terms and the units' factor steps over that many tasks, each
  with its own compact cell samplers and buffers, as `brm_two_stage_fit` does.
  A run does not depend on it: its draws stay on the one `rng`, in batch order,
  and the units' results are summed in batch order, so the ELBO, the gradients,
  the factors, the checkpoints and the draws are the same to the last bit at any
  `threads`, and a resume may use another count. `fit.diagnostics.threads` records
  the tasks a call used. Above one task the fit needs Enzyme's concurrent-GC repair
  (dogfood from `72d5d25`): without it an explicit `threads` above one is refused,
  naming `threads = 1`, and the default stays at one.
  **Behavior:** with that repair, a process started with several threads now uses
  them by default, and prepares a copy of the cell samplers per task before the first
  iteration; pass `threads = 1` to keep one. `laplace=GaussianBlockLaplace()` refuses
  `threads`: its updates evaluate the whole window.
- **The minibatch fit holds BLAS at one thread** while it optimizes and draws, at any
  `threads`, and restores the count afterwards. A process running with more BLAS
  threads may see different last bits than before, and no longer depends on that
  count, including when a run resumes in a new process.

## 0.2.145

- **Per-cell fields in the sparse embedded Laplace:** a `gmrf` formula inside
  `@plate for p in groups(g)` is integrated per cell by
  `brm_laplace_problem(rk; integrate=(declaration,))`, exact for Normal rows, with each cell's strict observation footprint (its
  rows read only its own field). A cell's row reads are emitted as one dotted
  per-row read over the plate's shared mesh locator, which ReactiveKernels
  prepares at the cell's rows.
- **Breaking:** requires ReactiveKernels 0.1.67 and ReactiveKernelsPPL 0.1.76.

## 0.2.144

- **Native space-time fields:** `field ~ 0 + gmrf((east, north), time[, channel];
  mesh=(vertices=xy, cells=triangles)[, cor=Omega])` is a sparse-precision
  Gaussian Markov random field: a mass-lumped finite-element Matérn (ν = 1)
  field on a triangular mesh, stationary Ornstein–Uhlenbeck evolution over the
  fitted times (exact transitions over irregular gaps), and an optional
  channel correlation (`cor=` a known correlation matrix or an
  `LKJCholesky` declaration). Each row reads three mesh vertices at its own
  time and channel. Its normalized prior keeps the determinants that depend
  on the learned SD, spatial length, temporal length and channel factor. The
  term's addresses are `sd(field, gmrf(...))` and
  `length_scale(field, gmrf(...), space|time)`; `= value` holds one fixed.
  The spatial length ℓ_s has κ = √2/ℓ_s, so the practical range is 2ℓ_s.
  Supported on SBBRMI and RKBRMI with density, gradient, coordinate-transport
  and fitted-state parity, frozen replay of the fitted mesh, times and
  channels, and scientific descriptions.
- **Independent fields in plate cells:** a `gmrf` formula inside
  `@plate for p in groups(g)` gives each cell its own field on that cell's
  fitted times and channels, with the cell's own term hyperparameters and
  priors that may read cell-local or model-scope values. On SBBRMI cell
  declarations are `local` (StanBlocks 0.2.20), so caller data of the same
  name never turns a cell prior into an observation. On RKBRMI the cell
  formula is `0 + gmrf(...)` alone, a cell-local `LKJCholesky` correlation is
  not lowered, and replay keeps the fit's cells.
- **Sparse embedded Laplace over a model-scope field:**
  `brm_laplace_problem(rk; integrate=(declaration,))` and
  `brm_laplace_plan(rk; integrate=:auto)` integrate a model-scope `gmrf`
  field (exact for Normal rows), with its precision moving with the lengths
  and channel factor. `brm_laplace_problem(rk; integrate=plan)` applies a
  plan to such a target, and `deepcopy` gives each chain its own copy.
  Integrating per-cell fields waits on a ReactiveKernels fix (snag
  `prepare-strict-o-332ad6b9`). No large-field performance is claimed:
  measured cost grows superlinearly with the state dimension under both exact
  and matrix-free algebra.
- **Breaking:** requires StanBlocks 0.2.20, ReactiveKernels 0.1.65 and
  ReactiveKernelsPPL 0.1.74 (deep copies of prepared queries whose generated
  programs hold modules, as a `gmrf` field's do).

## 0.2.143

- A sparse embedded Laplace target (`brm_laplace_problem(...; integrate)`)
  whose log density is finite and whose conditional modes all converged, but
  whose gradient is not, now reports it: the new counter `nonfinite_gradient`
  counts such evaluations, and a warning (at most three) names the population
  point, how many population coordinates are nonfinite and the first nonfinite
  term, for example "the third derivative of cell 1 of `y`". The gradient
  itself is unchanged, so SMC still raises `SMCNonfiniteError` there and HMC
  still rejects the step. Such a gradient needs a derivative the mode search
  does not: typically a likelihood's third derivative that automatic
  differentiation cannot represent in Float64 although the value and curvature
  are finite. One known source is nested Enzyme Forward derivatives through
  `log` of a value below about 1.5e-154, for example `log(erfc(t))` for `t`
  above about 18.5 (snag `laplace-gradient-db992ae0`).
- Matrix-free components (`solver = :matrixfree`, or `:auto` above 1,024
  coordinates) give a nonfinite solution for a nonfinite right-hand side, as
  exact algebra does, instead of raising "conditional solve failed".

## 0.2.142

- SVI runs can stop after any completed iteration and continue on
  the same model and data exactly as if they had not stopped:
  `brm_svi_resume([save,] model, checkpoint; stop_after, checkpoint_every,
  ndraws, progress)` continues a `BRMSVIOptimizerState` (or a fit result) to
  the same iterates, random stream, ELBO trace, stopping decision, polishing
  average and draws. `brm_svi_optimizer_checkpoint(fit)` copies the state; it
  holds the raw factors, global Adam moments, random stream, iteration,
  stopping and polishing state, controls, a window fit's streaming carry and
  the target's identity, but no observations or compiled model. Resume refuses
  a changed program, callables, bound data, frozen preprocessing, coordinates,
  groupings, subject coordinates or fixed shared values before any step.
- `brm_svi_fit` and window fits `brm_svi_fit(previous, x_new)` take
  `stop_after` (iterations in this call) and an optional first argument
  `save`, called with a checkpoint every `checkpoint_every` iterations
  (default once per window), when the call stops, and before a Treebars
  interrupt propagates. Diagnostics add `finished` and `optimizer`.
- The returned draws (and a GaussianBlockLaplace carry) come from a copy of
  the run's stream: their values are unchanged, but they no longer advance the
  caller's `rng` or `draw_rng`, and a stopped run's intermediate draws do not
  change its continuation.
- `laplace=GaussianBlockLaplace()` runs stop and resume the same way: the
  state holds the global factor, Adam, the averaging window, the random and
  draw streams, and the call counts and numerical records, so a continued
  run's draws, carry and diagnostics equal the uninterrupted run's. A stopped
  run's result averages the iterates of its window so far
  (`stop_reason=:stopped`). `save` defaults to every 50 updates.
- `brm_svi_checkpoint` remains the streaming carry for a new, disjoint window.
- `BRM SMC` checkpoints identify their bindings through a shared helper; the
  identities are unchanged, so existing SMC checkpoints still resume.

## 0.2.141

- Strict local cell queries and sparse Laplace planning support a plate
  selecting rows from another plate's array-valued output, including outputs
  built from shared correlated random effects.
- **Breaking:** the RK backend requires ReactiveKernels 0.1.62 or later.
  Update older pinned RK environments; the test bootstrap selects the
  producer-plate locality repair.

## 0.2.140

- Carried RK SMC windows accept `coordinates=:auto`: particles contain only
  old coordinates required by the new likelihood or fresh priors, plus fresh
  coordinates. Data-driven declared-vector extents can grow while fitted
  preprocessing and existing coordinate meanings stay fixed. RKPPL counts
  fresh priors and their transform Jacobians once and replaces old priors
  with the carried Gaussian.
- `brm_smc_coordinates` describes compact and full-joint coordinate orders;
  `brm_smc_recover` samples omitted coordinates from their carried conditional
  distribution. A completed selective fit carries a flattened bounded-rank
  Gaussian into the next window. Unread coordinates may change marginally
  through correlation; this remains assumed-density inference. Selective
  tuning and checkpoint replay use the same compact coordinate order.
- `brm_transform_constants` reads saved transform anchors on data-free RK
  replay, matching the frozen values applied by the graph.
- Requires ReactiveKernels 0.1.61, ReactiveKernelsPPL 0.1.70 and WarmupHMC
  0.2.11. Existing full-frame windows remain the default.

## 0.2.139

- Adaptive centering and centered prediction replay use the compiler's
  physical random-effect scales when their declarations have a nonzero
  lower bound or a finite upper bound. Fully centered coordinates therefore
  retain the declared covariance, including scales with data-dependent bounds.

## 0.2.138

- `rk_design_binder` reuses its lowered program when HSGP axes, grouping
  keys or validation columns are data-only model assignments. Validation reads
  those definitions from the bound graph, including values folded during
  preparation. Masked HSGPs can read a data-only axis without defining it twice.

## 0.2.137

- `NLMEEstimation.provides_laplace_gradient(::BRMNLMEModel)` returns `false`
  when RK refuses row-local sparse Laplace factors, allowing ordinary protocol
  checks and SAEM to proceed. The structural refusal is cached; explicit
  integration still throws its original diagnostic, and unrelated errors
  continue to propagate.

## 0.2.136

- RK interval-censored Normal responses retain finite log likelihoods and
  native Enzyme Reverse gradients in both extreme tails, verified through
  400 standard deviations against independent high-precision values and
  location/log-scale derivatives.
- **Breaking:** the RK backend requires ReactiveKernelsPPL 0.1.60 or later
  (and its ReactiveKernelsDistributionKernels 0.1.1 floor). Update older
  pinned RK environments; the test bootstrap selects the delivered repair.

## 0.2.135

- Power-scaling sensitivity now uses PowerScaling.jl 0.1.0 and its PSIS.jl
  implementation. `brm_powerscale_inputs` retains BRM's model assembly;
  `brm_powerscale_sensitivity` delegates the summary and accepts `reff`.
  `BRMPowerscaleSensitivity` is an alias for PowerScaling's summary type,
  preserving its fields and numerical table contents; printing uses
  PowerScaling's diagnostic labels and formatting.
- **Breaking:** the generic `brm_psis_weights`, `brm_cjs_dist` and
  `brm_powerscale_weights` exports are removed. Use `PSIS.psis`,
  `PowerScaling.cjs_dist` and `PowerScaling.powerscale_weights` directly.
  PowerScaling is pinned by URL and revision until registered; the test
  bootstrap and Docs installation supply that revision on Julia 1.10.

## 0.2.134

- Data-free RK fitted states retain numeric scalar inputs used by priors and
  `constant(...)`, so prediction designs can omit them. Retained-fit replay
  also supplies omitted numeric scalars. Changed prior arguments still fail
  the fitted-program check; constants keep their fitted values. Re-export,
  serialization and design binders preserve this behavior without retaining
  training-row arrays (snag `rk-fitted-state-56bb407b`).
- **Breaking:** `BRMFittedState` now uses format 2. Re-export persisted states
  from their training fits after upgrading.

## 0.2.133

- `brm_output_tree` preserves observation coordinates through pointwise
  assignments that combine row outputs with scalar model constants or scalar
  data inputs, including aliases and chained transforms. Array constants still
  require their own axis semantics; equal length alone does not label them.

## 0.2.132

- Plans retain their conditional numerical policy when reused or automatically
  replanned. `brm_ladder_fit` accepts the same solver and matrix-free options;
  continuation keeps its existing target's policy.
- Outer SMC identities retain the operator policy, fixed probes and sparse
  reference values. Checkpoint replay rejects a different finite target;
  scratch buffers and evaluation diagnostics do not change its identity.

- `brm_laplace_problem(...; integrate, solver=:auto)` selects exact algebra
  for connected components through 1,024 coordinates and iterative conditional
  algebra for larger components. `solver=:exact` retains the previous sparse-factor
  path; `solver=:matrixfree` forces iterative algebra. `MatrixFreeLaplace`
  sets the component bound, retained probe seed, solve caps and quadrature.
- The large-component determinant is a reproducible fixed-probe rational
  approximation. Its gradient differentiates that same approximation,
  including the nonlinear conditional mode and sum-factored observations.
  A fixed bounded-fill LDL reference conditions the solves and determinant;
  its relative diagonal ridge affects only that numerical reference. No
  unbounded sparse factor or dense covariance is formed. Numerical convergence
  and probe variability remain distinct from Laplace approximation error.
- Component densities use compensated summation on the operator path, so
  accumulated rounding over large fields does not hide the final Newton gain
  from the line search. Acceptance and convergence tolerances are unchanged.
- **Breaking:** large Gaussian components now also have numerical determinant
  approximation error under the default `solver=:auto`. Use `solver=:exact`
  to reproduce the previous determinant algebra. Small components retain
  exact algebra. This does not alter the authored statistical model.

## 0.2.131

- Label ragged response draws in `brm_output_tree` and `brm_predictive_tree`
  with flat observation positions, retaining the existing backing arrays and
  separate chains. Aggregate log likelihoods have one position per group;
  predictive outputs have one per element, with group boundaries in `segments`.

## 0.2.130

- Outer SMC accepts prepared RK-backed generic, structured and sparse embedded
  Laplace problems, including plans with zero integrated terms. Fit, independent
  pilot tuning, comparison, cumulative prepared-target updates and checkpoint
  replay reuse RKPPL 0.1.57's supplied-gradient adapter with independent mutable
  workspaces. Identities include approximation and solver settings, coordinate
  order, frozen preprocessing and exact data.
- Sparse Laplace preparation accepts fixed random-effect scales without
  trying to find sampled coordinates for their known values.
- `brm_laplace_draws(fit::BRMSMCFit; conditional, rng)` completes a finished
  marginal cloud and returns its outer weights alongside full packed draws.
  Conditional importance resampling does not correct the outer approximation.
- **Breaking:** ladder rung 5b is named `population_sampling` (formerly
  `population_hmc`). It accepts a completed SMC fit, explicitly resamples its
  outer weights, and forwards `conditional` to effect completion. Its `method`
  is `:smc` or `:callback`. A Laplace-only callback requires `upto=5`; rung 6
  still needs a sampler for the full-model target.

## 0.2.129

- `brm_output_tree` labels random-effect standardized draws and centered
  effects with fitted levels and coefficient margins, including flattened
  and stratified carriers. Scales, correlation factors and Student-t mixing
  weights carry their corresponding margin, stratum or level coordinates.
  These outputs use fitted coordinate order regardless of the caller's name
  order and retain views into the supplied draws. Unequal per-chain inputs
  work with `TreeActualArray` for parameter-wise convergence diagnostics.
- Per-row block contributions and model-scope assignments using elementary
  pointwise operations on row outputs and scalars carry `:observation`,
  including chained assignments. Equal vector lengths alone do not establish
  observation coordinates.

## 0.2.128

- Sparse embedded Laplace accepts declared independent Gaussian vectors such
  as `field[1:n] .~ Normal.(mean, scale)`, automatically or by declaration name.
  Fixed and outer-dependent means/scales retain their prior normalization,
  diagonal precision and marginal gradients in the authored coordinates.
  Width screening samples whole vectors before derivative preparation;
  nonlinear observations retain the per-coordinate importance diagnostic.
  A vector supplying another Gaussian law's parameters stays sampled.
  Non-Gaussian, constrained and correlated vector priors are not admitted
  (snag `laplace-integrat-61a058f1`).

## 0.2.127

- Carried SMC posteriors now retain their identity across defensive copies
  and serialization round trips, including layouts with dictionary aliases.
  Window updates, pilot tuning, checkpoints and filter calibration still
  reject changed posterior values or frozen preprocessing. Identities hash
  record values rather than dictionary traversal, object sharing or padding.
  Persisted carries and window checkpoints require matching package versions;
  regenerate them from the completed fit when upgrading to this identity format.

## 0.2.126

- Added optional TreeArrays twins for output, random-effect, predictive and
  conditional draws, plus `rk_draw_tree`. They label existing arrays or views
  with descriptor coefficients, group levels and margins, observation positions,
  or actual prediction-grid values. Per-chain matrices retain an outer chain
  axis, including unequal chain lengths and ragged random-effect blocks.
- Stored output and random-effect carriers are viewed without copying their
  draw values; quantities that need computation use the existing extraction
  and simulation paths. Existing array APIs retain owned array results. Axes
  without known coordinates raise an error naming the output.

## 0.2.125

- RKBRMI passes a plate's original cell values to model-scope whole-value
  helpers, including matrix and higher-dimensional array cells. Positional
  and keyword arguments, helpers in likelihood expressions, and non-last
  indexed outputs preserve the inner axes of each cell. Per-row arithmetic
  and direct observation arguments retain their existing flattened layout
  (snag `rkbrmi-model-sco-bcc72190`).
- Function-valued arguments interpolated into `brm` source now emit named
  callable bindings, so higher-order reductions such as `sum($sum, columns)`
  retain ordinary Julia semantics and survive printed-source replay.

## 0.2.124

- Added `brm_smc_carry` and carried-posterior `brm_smc_update` for bounded
  windows of new observations. Frozen preprocessing, source identities and
  the owned Gaussian summary survive checkpoints without old training rows.
  `basis=:fit` delegates bounded correlation and moment fitting to RKPPL and
  WarmupHMC; independent pilots continue to tune fixed mutation kernels,
  with `reference_inflation` forwarded for learned window proposals.
- Added `brm_filter(fit; learn=true, ...)` and `brm_filter_tune` for static
  parameter ensembles with bounded-window replenishment and rejuvenation.
  Selected calibration drops historical latent coordinates. A state-only
  warmed ordinary filter can transfer its finite-state pool and clocks.
- Streaming carries are Gaussian assumed-density approximations. Filter
  boundaries additionally discard parameter correlations and parameter–state
  dependence. Checkpoints support exact continuation of that approximation,
  not exact full-history inference. Requires RKPPL 0.1.55 and WarmupHMC 0.2.10.

## 0.2.123

- Require RK 0.1.52 / RKPPL 0.1.55 for sum recognition through guarded
  gathered observations; tests pin landed RK `5e9d2d03818ba5c8ef7d6d068fef42101f902d0e`.

- Sparse Laplace plans screen structural cell footprints before preparing
  derivatives. `max_cell_width=128` bounds integrated coordinates per ordinary
  cell or local sum-factor block. Overwide cells sample whole unforced latent
  terms in stable order, retaining every likelihood and prior; forced choices
  that exceed the bound raise with the observation, cell and contributing terms.
  Plans report the triggering cells and final integrated/sampled dimensions.
- Sparse patterns deduplicate footprints before pair enumeration, union directly
  into CSC columns, and omit the per-cell quadratic nonzero-index map. Sum
  factors prepare Hessian directions only for integrated coordinates and outer
  sum outputs, while retaining all sampled-coordinate gradients.
- A plan admitting no latent terms now supplies the full model's density,
  gradient, draws and fitting workflow. The ladder reports and skips its dense
  population-curvature diagnostic above 128 coordinates, while retaining HMC.

## 0.2.122

- A `brm_laplace_plan` retains its prepared sparse problem. Building a target
  from that plan on the same backend and automatic-differentiation
  configuration copies the prepared workspaces instead of preparing every
  observation again. Plans that admit only some candidate terms restrict the
  prepared row, plate-cell and sum factors to those terms. Repeated targets
  keep independent mutable state and honor their own solver tolerances.

## 0.2.121

- `brm_smc_tune` forwards `reference_inflation`, added in RKPPL 0.1.47. Its learned
  reference (`pilot=:adaptive, reference=:pilot`) now fits the retained pilot
  draws' spread in the learned basis times that factor (default 2) instead of
  the pilot's local mutation scale, which under-dispersed the population
  coordinates of hierarchies (ReactiveKernels snag learned-smc-refe-cc5677a1).
  Stage summaries carry `reference_sd` and `coordinate_sd`. The test environment
  retains the current RK 0.1.51 / RKPPL 0.1.53 pins. The hierarchy regression
  verifies the reference repair while retaining the known population-intercept
  spread deficit; wider references alone do not establish posterior accuracy.

## 0.2.120

- Require ReactiveKernels 0.1.51 and ReactiveKernelsPPL 0.1.53 for selected-row
  lowering through scalar plate collectors and broadcast observation
  locations. The compiler preserves type information through scalar helpers
  and skips inactive guarded capture indices, so strict cell queries and
  NLME preparation can retain local reads through those collectors.

## 0.2.119

- Sparse embedded Laplace retains model-scope observations of derived values,
  including gathered nonlinear count predictors, using compact observation
  factors, including emitted missing-value completion density corrections.
  Automatic plans gate terms read by these observations with the same
  importance diagnostic as plate cells. Factor selections use structural
  coordinate footprints, including dependencies inactive at the initial point
  (snag sparse-automatic-1de1001a).
- Ragged-row helpers specialize on their axis arity, so factor preparation
  retains type stability for wide predictors with mixed column types.
- An observation's cells share one set of Hessian-vector direction buffers
  that the evaluation sets per cell, instead of `K × m` floats stored per
  cell: a Gemeinde-level observation's cells no longer cost tens of
  gigabytes to prepare.

## 0.2.118

- Plate cells read model-scope computed arrays directly, including transforms
  of formula predictors and assignments that gather or concatenate values.
  SBBRMI and RKBRMI preserve the computed array's own axis and dependencies;
  a derived array can also supply the plate range without a raw-data input.

## 0.2.117

- `brm_svi_fit(...; laplace=GaussianBlockLaplace())` jointly integrates
  Gaussian population coefficients, unit effects and shared coordinate
  multigroup HSGP weights inside a variational target. Structural eligibility
  currently requires affine Gaussian observations. RK/Enzyme derivatives and
  capped Krylov operations avoid dense curvature over the growing latent
  block. Each update evaluates the entire current window; optimization uses
  a fixed update budget and explicitly leaves convergence unassessed.
- `BRMSVILaplaceState` checkpoints retain frozen preprocessing and a fitted
  common Gaussian factor across globals, fields and unit blocks. Continuation
  supports new, absent and returning units with fixed coordinate identities.
  The projection adds Monte Carlo, shrinkage and residual-dependence error;
  checkpoint copying and serialization grow with retained units. Ordinary
  SVI checkpoints cannot be converted to this representation.

## 0.2.116

- A plate may select another plate's complete vector-valued output through
  an index on its own axis, such as `path[which[i]][positions[i]]`. RK and
  Stan preserve the source cell before selecting its entries, for data-only
  and parameter-dependent outputs, including non-last outputs. The plates
  may have different cell counts and several readers may share one cell.
  Captured outputs also retain their producer dependencies when model
  fragments are composed with the reader first.

## 0.2.115

- RK frozen replay accepts changing group labels for coordinate multigroup
  HSGPs (`hsgp(t; by=g, cor=multigroup((x,y), ...))`). Fitted-state export
  freezes the temporal basis, spatial grid and default spatial-scale prior;
  replay maps current level coordinates onto the same shared weights without
  retaining observation rows. Known labels must keep their fitted locations,
  and new locations must lie inside the fitted spatial region. Per-level
  hyperparameter effects and projections still refuse group resampling.

## 0.2.114

- Plate cells retain dependencies on captured model values, including whole
  vectors. Composed models emit those declarations before the cells that read
  them, even when the captured values depend on later parameter declarations.
  Cell-local bindings do not create dependencies on same-named model values.

## 0.2.113

- RK grouped HSGPs accept `iso=false`, retaining one shared length scale per
  axis, declared priors and bounds, and the grouping's sampled or known
  correlation. The same coordinates support Stan transport, term starts,
  shared value readers and frozen replay.

## 0.2.112

- Added RK-only `brm_smc_compare` and `brm_smc_runs`: agreement statistics
  between independent completed SMC fits of the same model and data, per
  unconstrained coordinate (run means and SDs, pooled mean and SD, the largest
  mean difference and the SD-ratio range in pooled-SD units, the worst
  coordinate) and pooled log-evidence estimates with their spread, plus each
  run's surviving lineages, stages, evaluations and divergences. Fits of
  different identities or incomplete fits are refused. `brm_smc_runs` fits
  one bridge per seed with shared settings and returns the fits with their
  comparison. Descriptive, not an acceptance gate.

## 0.2.111

- Plate cells read declared vector parameters using their prior's extent on
  both SBBRMI and RKBRMI. Literal and scalar-data lengths in
  `x[1:n] .~ Family.(...)` work alongside formula predictors and sampled
  hyperparameters; `MvNormal` and `Dirichlet` retain their vector axes too.
  A direct indexed read checks that the vector has one entry per cell.
- RK plate cells and their observations read a preceding plate's complete
  vector at `output[i]`, preserving its inner axis instead of indexing the
  concatenated formula value.

## 0.2.110

- `brm_svi_fit(previous, new_window; ...)` continues on disjoint observations
  with the previous Gaussian factors as both initialization and prior.
  `brm_svi_checkpoint(fit)` returns a data-only `BRMSVIState` for serialization
  and fresh-process resume with the same model and frozen preprocessing.
  Units match by label and coordinate identity: new units start from their
  model prior, absent units retain their factors, and returning units recover
  them. Changed preprocessing, global layouts, or existing unit coordinate
  sets are refused. The history is approximated by the carried Gaussians;
  sequential fits can depend on window order and need not equal a batch fit.
- SVI global precision, entropy and covariance-root operations use
  diagonal-plus-low-rank algebra without a dense global covariance matrix.
  `diagnostics.unit_gradient_calls` reports optimization work on the current
  window. The one-argument calibration entry is unchanged.
- Exporting constants from a data-free RK replay preserves the frozen values
  instead of fitting them again from the replay's new rows.


## 0.2.109

- Added RK-only `brm_smc_tune` for independent pilot tuning of initial and
  cumulative-data SMC bridges. RKPPL's precomputed schedule chooses fixed
  mutation kernels by temperature, preserves its normalized reference and
  exact endpoint identities, and survives checkpoint restoration without
  rerunning pilots. Adaptive pilots also support portable structured scales
  and an independently learned, frozen normalized Gaussian reference.
  Completed-stage history includes selected kernel settings, evaluation
  counts, mean acceptance and unconstrained movement. Requires RKPPL 0.1.42
  and WarmupHMC 0.2.9.

## 0.2.108

- `car(area; edges=(from, to), nodes=all_areas, type=...)`: the optional
  `nodes=` names one data vector holding every area of the graph, declared
  apart from the observed rows, so an area with no rows and no neighbours (an
  island whose every row a hold-out withholds) keeps its place and a frozen
  `reprocess` onto its rows finds it. A row or edge naming an area outside
  `nodes` is refused; the vector may repeat labels. Without `nodes=` the graph
  stays every label in `edges` and in `area`. SBBRMI and RKBRMI alike; a
  frozen replay needs neither the edge nor the node vectors, and
  `freeze_constants=false` rebuilds the graph from all of them.

## 0.2.107

- Added Stan lowering for ordinary `mean` and `var` expressions over a
  lower-truncated LogNormal, plus its separate stable `logcdf`/`logccdf`, in
  plates and model-scope expressions. Constructor parameters and the lower
  bound may depend on model parameters. Numerical evaluation uses the
  Distributions and StanBlocks implementations; normalized moments do not
  implicitly add the support-probability factor to the likelihood. StanBlocks
  0.2.14 supplies the Stan builtins; the test environment pins that release
  and the Distributions fork revision carrying the one-sided moments.
- RK source preserves ordinary `truncated` and `censored` constructors in
  model-value expressions by distinguishing their callable bindings from
  RKPPL's response combinators.

## 0.2.106

- `brm_start(sb, stan_model, inits)` with a block's `effects` holds one
  constrained-plus-transformed-parameter vector at a time while it solves
  for the block's coordinates, locates the block's carriers through one name
  index built once, and measures a level-correlated `gr(g; cor=...)` block
  through `levels + margins - 1` probes instead of `levels × margins`. Memory
  is linear in the model's size, whatever the block's margins and given
  levels; the points are unchanged. Before, the solve held `margins + 1`
  (`levels × margins + 1` for a level-correlated block) full vectors twice
  over, 50 GB on a 623k-parameter model (snag brm-start-explic-aaf34bf9).

## 0.2.105

- RK observation-source providers honor the original constructor's
  `brm_distribution_type`: multivariate and matrixvariate laws receive each
  complete event and its arguments, instead of being broadcast over entries.
  Scalar source providers keep their elementwise behavior.

## 0.2.104

- **Breaking:** SMC numerical failures now propagate RKPPL's
  `SMCNonfiniteError` instead of `DomainError` (or its task wrapper). Catch
  this exported RKPPL exception to inspect the exact unconstrained `position`,
  `value`, optional `gradient`, `quantity`, `endpoint` and `temperature`.
  Rendering is compact, NaN/+Inf still fail fast, and `-Inf` retains its
  outside-support meaning. Requires RKPPL 0.1.37.

## 0.2.103

- `brm_start` accepts `term_inits`, quoted authored term-prior addresses paired
  with natural-scale initial values, on StanBlocks and ReactiveKernels. This
  includes scales, persistence, length scales, spatial mixing, smoothing
  components, latent covariates and simplexes the backend already supports.
  Qualified repeated terms resolve to their own emitted parameters.
- Explicit starts accept `base=:prior_median`, a deterministic prior-informed
  base with explicit values taking precedence. Effective scalar bounds and
  HSGP floors are respected; structured latent blocks keep neutral coordinates.
  Existing zero/numeric bases and no-initial-values data-based starts keep
  their meanings. Initialization never changes the model or its priors.

## 0.2.102

- Both backends accept grouped reads of every indexed output of another
  `@plate`, including outputs before its final assignment. Reads such as
  `location[rows[k]]`, with `rows = groups(group => labels)`, retain their
  model dependencies and gradients and support replay on new groups.

## 0.2.101

- Formula assignments, predictor terms and offsets apply `hypot(x, y)`
  elementwise on the ReactiveKernels backend, including a scalar with a
  vector. This preserves the stable norm without overflowing intermediate
  squares. The formula editor accepts the same spelling. StanBlocks 0.2.12
  supplies the matching Stan builtin; the test environment pins that release.

## 0.2.100

- Fixed: `brm_laplace_plan(...; integrate=:auto)` reports stratified random
  effects as sampled terms instead of failing while reading their declaration.
  Ordinary Gaussian terms over the same grouping can still be integrated,
  preserving the sampled stratified effects and their priors. Partial
  integration over a grouping also skips the dense population-design probe
  when latent terms in that grouping remain sampled.

## 0.2.99

- `RKBRMI` supports exact exponentiated-quadratic `gp(x...)` terms on
  multiple axes, with one shared length scale (`iso=true`) or one length
  scale per axis (`iso=false`). Declared hyperpriors and known scales retain
  their meaning. The covariance is emitted as a ReactiveKernels graph and
  the non-centered latent draw supports native Enzyme density gradients.

## 0.2.98

- Added RK-only finite-state `brm_filter`, observation steps, bounded streaming
  and data-only checkpoints through RKPPL 0.1.36. Calibrated parameter points
  and ensembles retain separate conditional state clouds; explicit authored
  parameter drift is supported. State-only warm-up avoids counting the
  parameter-training likelihood twice. Filtering preserves frozen
  preprocessing and does not retain the training backend or stream history.
- Added `rkppl_model(rk::RKBRMI)` to expose the retained emitted declarations
  as a cached, unbound RKPPL source model, including its submodels and callable
  source providers. Observation roles and frozen preprocessing remain in
  `rk_bound_plan(rk)`; capturing source does not change the executable fit.

## 0.2.97

- Fixed: all three `brm_laplace_problem` engines start each conditional mode
  search at zero standardized effects. Previous evaluations no longer choose
  a basin or change whether a point converges, so the density and gradient at
  a population point are independent of call history. This keeps the existing
  single-mode Laplace approximation; it does not sum multiple modes. Searches
  may take more Newton steps than the former warm starts.

## 0.2.96

- Added: sparse embedded Laplace integration of sum-factored `@plate`
  likelihoods. Summand curvature stays local and outer curvature enters an
  augmented sparse system, including overlapping, repeated and empty sums.
  KLU supplies pivoted solves, SelectedInversion supplies the required inverse
  entries, and MUMPS certifies curvature and log determinants. Conditional
  draws use a bounded Krylov covariance square-root action. Automatic term
  diagnostics reuse baseline sums and evaluate only changed summands.
  Requires the fleet SelectedInversion revision `8b1ee1077cea94c1aff746e182681ae39ce91526`
  (the registered 0.2.2 release does not carry its pivoted-LU API), KLU 0.6,
  MUMPS 1.7 and KrylovKit 0.10; `test/setup_env.jl` records the exact source.
  Requires ReactiveKernels 0.1.39 / ReactiveKernelsPPL 0.1.33 so an indexed
  predictor read beside a sum keeps its local derivative footprint.

## 0.2.95

- Non-centered missing-value completions follow data-indexed gathers of
  their law's predictors when assigning NLME subjects. Shares coordinates,
  free entries and observed entries keep the source rows' subjects even
  when the response is a subset, repetition or permutation of those rows.
- Lower-bounded free completion entries can be NLME unit draws. Their
  support transform and exact density remainder preserve both restricted
  and truncated laws, with variance fixed at one and the same
  unconstrained coordinates as Stan.

## 0.2.94

- Added RK-only `brm_smc_fit` and `brm_smc_update`, using RKPPL's optional
  SequentialMonteCarlo/WarmupHMC integration. Batch fits bridge a normalized
  Gaussian reference to the full posterior; updates bridge to the posterior
  for cumulative data with frozen preprocessing and unchanged coordinates.
- Added bounded-stage continuation, progress reporting, and data-only
  `brm_smc_checkpoint` / `brm_smc_resume` with model/data identity checks.
  Updates replay full histories after parameter moves; they are not
  constant-cost streaming. Requires ReactiveKernelsPPL 0.1.34 and WarmupHMC 0.2.8.
- Raised the optional Treebars floor to 0.1.1 so progress reporting has the
  cooperative interruption API its BRM hooks use.

## 0.2.93

- Fixed: `@plate` can read an `mi()` completion through a grouping that
  lists only the cells it needs, including free missing entries alongside
  `mi(v)::shares(total, rows)` groups. RK no longer treats this read as a
  second observation requiring every completed row to belong to a cell.
- Both backends accept grouped model-scope assignments such as
  `v_part = v[pos_part]`, including gathered formula predictors. The
  grouping column labels the assignment's rows; completed values and
  their derivatives remain live. Coverage is still required for a response
  actually observed inside the cell.

## 0.2.92

- Added: `mi(x)::shares(total, rows)` accepts lower support declared by
  `truncated(Law(...); lower=lo)` or `restricted(Law(...); lower=lo)` on
  SBBRMI and RKBRMI. Scalar, row-specific and model-valued lower bounds
  shift each group's simplex above its missing rows' lower bounds while
  preserving the total. Truncation retains its normalizer; restriction
  retains the base density kernel. The exact-sum density is not divided by
  a marginal density for the total.
- The same laws work for ordinary imputation and `T::shares(total)` cell
  latents, including replay, output descriptors, starts and coordinate
  transport. A singleton keeps its exact residual and density without
  coordinates; a multi-entry group requires strictly positive remaining
  slack. Finite upper bounds remain unsupported by this geometry.
- RK noncentered grouped completions retain their law and Jacobian as
  group weights for NLME unit attribution. Bounded free entries stay
  centered shared latents, with their declared law. Group-local evaluation
  requires ReactiveKernels 0.1.38 / ReactiveKernelsPPL 0.1.32. The test
  environment pins Enzyme dogfood `0e251cfb` and its declared libEnzyme
  `782c9d95` for native HVPs through the lower-support geometry on Julia 1.10.

## 0.2.91

- Added: `brm_ladder_fit(rk; ad_backend, upto = 4, integrate = :auto, …)`, a
  staged fit over the sparse embedded Laplace. Each rung is a complete
  result that starts the next, and each rung's check goes into
  `fit.ladder`:
  1. effect modes at the population start;
  2. the mode of the embedded-Laplace marginal, with the plan measured again
     there under `:auto`;
  3. SAEM over the units the call names, integrating the plan's admitted
     families (NLMEEstimation), with its gap to rung 2;
  4. Laplace draws at the point;
  5. 5a, the marginal's curvature at the point as a labelled diagnostic, and
     5b, HMC over the Laplace marginal;
  6. the exact model from rung 5.

  Rungs 5b and 6 take the caller's `sampler`; a rung a call cannot run is
  skipped and named. `continue_from` keeps the integration choice and runs a
  skipped rung when its units or sampler are supplied. The curvature
  diagnostic keeps the returned effect draws from rung 4.
  Automatic plans are checked again after each SAEM phase; a tuple or vector
  of `saem_options` runs successive phases with fresh averaging and the last
  phase's point as the next start. Rung 5b reports its means and standard
  deviations beside the curvature diagnostic, one entry per coordinate.
- Faster: the sparse Laplace's gradient keeps each term's C rows in one flat
  buffer instead of two new vectors per term. One warm value and gradient,
  same load: a flat Poisson panel (1,126 × 10) 3.32 → 2.85 ms, a nested one
  (12 × 400) 17.2 → 12.6 ms. Both are now below 0.2.75's 3.05 and 12.9 ms.

## 0.2.90

- Fixed: SBBRMI accepts exact `-Inf` and `Inf` response-bound data through
  BridgeStan, including mixed censored and detected Normal and Student-t
  observations. Requires StanBlocks 0.2.10 for non-finite data serialization;
  the existing `censored(...; lower=..., upper=...)` spelling is unchanged.

## 0.2.89

- Added: SBBRMI translates `DirichletMultinomial(N, alpha)` to the native
  StanBlocks 0.2.8 family with its explicit total. Ragged plate observations
  retain one normalized pointwise likelihood per count vector, and predictive
  count vectors sum to their declared totals.
- The test environment explicitly pins Distributions
  `83cec5e3a8a35167d54f19da9dc5afb54f59273e`, whose Dirichlet-multinomial
  density rejects invalid counts and mismatched totals with `-Inf`.
  The RK regression now requires that behavior instead of marking it broken.
  Consumers must select this unreleased Distributions fix explicitly in their
  active environment; upgrading BRM or RKPPL alone does not install it.

## 0.2.88

- Fixed: nested NLME SAEM fits require ReactiveKernels 0.1.29, whose plate
  buffer inference works from older compiled worlds. This repairs the
  population-gradient compilation failure introduced by the test pin in
  0.2.81. The test environment pins the published fix (`5f18f4b84`).

## 0.2.87

- Added: `brm_laplace_plan(rk; ad_backend, integrate = :auto, sample = (),
  ess_floor, draws, at)` and `BRMLaplacePlan`: which latent terms the sparse
  Laplace integrates. Terms only Normal rows read are integrated; terms read
  by counts, binomial rows or `@plate` cells are gated by a per-block
  importance diagnostic, measured after the factorization and reported per
  term; Student-t, stratified and `car_units` terms and terms on a shape or
  dispersion are not integrated unless named. `brm_laplace_problem(rk;
  integrate = :auto)` builds the sparse problem from such a plan, and
  `integrate = plan` takes one as given (`target.plan`).

## 0.2.86

- Fixed: RKBRMI preserves a multivariate distribution's whole count or value
  vector inside a plate cell. `y[k] ~ DirichletMultinomial(n[k], alpha)` now
  uses one normalized likelihood event per vector, including ragged cells and
  model-derived concentration vectors. Constructor bindings survive emitted
  source replay, including families outside RKPPL's built-in vocabulary.
- Added: `DirichletMultinomial` re-exports the Distributions constructor.
  Its supported contract requires a positive integer total and strictly
  positive concentration entries. SBBRMI translation remains pending native
  StanBlocks support (snag `dirichlet-multin-91925143`).

## 0.2.85

- BRM's compat floor is ReactiveKernels 0.1.25. A cell of one observation of
  an `@plate for` loop that computes several observations' arguments (one
  plate returning a tuple) then computes only the values that observation
  reads, not those only its sibling observations read. On ArbeitsplatzAtlas a
  `selb_obs` or `beamt_obs` cell read 147,766 coordinates (1,236 now; warm
  value 18 ms, now 0.02 ms). The test environment already pinned it
  (`a959bef75`, from 0.2.81).

## 0.2.84

- Fixed: the index vector of an in-cell `indexin(idx[rows[i]], target[i])`
  may be a data-only assignment (`idx = f(columns...)` at model scope), as
  `brm-use` documents for every vector a plate cell reads. Before, plain
  `indexin` refused it at construction ("reads `idx` as indices; it must be a
  data vector"); the zero check now reads the assignment's value. The model
  equals the same model given that index as a data column, on SBBRMI and
  RKBRMI.

## 0.2.83

- Fixed: a data-only assignment that indexes a matrix-valued one with a bare
  `:` or a range (`m = mat(x)`, then `z = m[:, 2]` or `u = m[1:length(x), 1]`)
  is data, as `z = second(m)` with `second(m) = m[:, 2]` already was. RKBRMI
  reads it as a population term and both backends read it in a `@plate` cell.
  Before, it stayed a model value: RKBRMI refused the term (`term `z` is not a
  data column`) and both backends refused the cell read (`kernel(...)
  positional arg `z` is not a formula statement`).

## 0.2.82

- Added: NLMEEstimation's integrated families on `brm_nlme_model`'s models.
  `saem(model, …; integrated = [f, …])` integrates those covariance families
  by BRM's sparse Laplace and samples the rest; with every family integrated
  it runs Laplace-EM. The model implements `laplace_conditionals_and_gradients`,
  `laplace_conditional_moments` and
  `laplace_conditionals_and_population_gradient`.
- Added: `provides_laplace_gradient(model)` is true where the sparse Laplace
  integrates every family. `covariance_step` with
  `AdaptiveGaussHermite(nodes = 1)` (the ragged SAEM default) then takes
  `2p + 1` Laplace-gradient calls instead of `2p² + 1` Laplace marginals, and
  `check_protocol` checks that hook.
- Changed: the NLMEEstimation extension needs NLMEEstimation 0.2.4
  (`[compat]`; before, 0.1.3 or 0.2).
- Changed: the sparse Laplace's `@plate` cells take their Newton-step HVPs
  only along their integrated draws, and the gradient's implicit term from
  the third-order products' own HVPs: 23.8 → 20.0 ms per value and gradient on
  a nested panel of 200 subjects.
- Fixed: the sparse Laplace's gradient on a model whose observations are all
  `@plate` cells and which samples a random-effect block (0.2.81 raised a
  `MethodError`).

## 0.2.81

- Added: `brm_laplace_problem(rk; integrate = (:g, …))` takes models whose
  observations sit in `@plate` cells, beside regression rows. Each cell is one
  term over the coordinates it reads; its gradient, Hessian block and third
  derivatives come from Enzyme over that selection (ReactiveKernelsPPL 0.1.22's
  compact-cell AD). Before, such a model was refused. It needs the Enzyme
  dogfood line from `7a886d9`; registered Enzyme raises
  `EnzymeRuntimeActivityError` on the cells.
- Fixed: the Optim extension of 0.2.80 overwrote `brm_laplace_mode`'s
  fallback method, which precompilation refuses, so it was compiled again in
  every session that loaded it. It now adds its own method.

## 0.2.80

- Added: `brm_laplace_mode(target; init, iterations, g_abstol)`, the mode of
  the embedded-Laplace marginal of any `brm_laplace_problem` target, by L-BFGS
  on its own value and gradient. It returns one complete packed point, the
  population mode with every integrated effect at its conditional mode there,
  and diagnostics. It needs Optim, a new weak dependency that `using
  WarmupHMC` loads through Pathfinder.

## 0.2.79

- Added: `brm_start(rk; integrate = (:g, …))` and `brm_start(sb, model;
  integrate)` compute the start from the data over the sparse Laplace of the
  named groupings (`brm_laplace_problem`'s `integrate`), so a model without
  NLME units (crossed groupings, a field across the units) gets a start
  without initial values.

## 0.2.78

- Changed: inside one component of a `BRMSparseLaplaceProblem` (a field
  across the units, crossed groupings), the mode search no longer takes one
  step length for the whole component. It tries the full Newton step; where
  that fails, the component's step blocks (`target.sblocks`, the levels of the
  coarsest integrated grouping its rows read) each halve their own step until
  their own terms gain, and a block Gauss–Seidel sweep follows where the
  component's density still did not rise. `target.blockwise` and
  `target.sweeps` count both. Modes, values and gradients are unchanged where
  the search converged before.
- Faster: each block's probes take derivatives along its own scale
  coordinates only, and an affine population part comes from its Jacobian
  (about 23 % less time per value and gradient on a nested panel of 12 × 400
  units).

## 0.2.77

- A non-centered grouped completion (`mi(x)::shares(total, rows)` on
  `RKBRMI(brmi; completions = :noncentered)`) computes each grouped entry's
  value per entry and each group's term per group, each reading its group's
  coordinates through an index array. A cell reading a completed entry or a
  group's term then reads that group's coordinates only, as recorded on the
  point. Before, both were whole-vector functions (`_brm_grouped_shares_values`,
  `_BRMGroupedSharesWeights`), which every such cell computed whole: on
  ArbeitsplatzAtlas an `osm_pub` cell read 5,232 coordinates (749 now) and a
  `svb1_shares_weights` cell 33,909 (792 now). Values are unchanged. Computing
  the whole model, each entry now recomputes its group's normalizer, so the
  shares' cost grows with the group size.
- A `cdar` walk's products are statements of their own (`steps = eta *
  decay`, and `correlated = L * eta` with a factor), not a product inside a
  dotted term, which ReactiveKernels computed whole in every cell. With
  ReactiveKernels 0.1.22 a cell then reads its own group's innovations (4 of 24
  in `test/rk_cdar.jl`; an ArbeitsplatzAtlas `ba_n` cell no longer reads the
  1,600 of `cdar_log_nl_jahr_nl`); with a correlation factor, the groups it
  couples. Values are unchanged.
- BRM's compat floor is ReactiveKernels 0.1.22. Its 0.1.21 computes
  `view(A, :, k)` (RKPPL's lowering of a column read) at a cell's rows: the
  columns of a joint population effect (`pop[:, j]`, several responses sharing
  one design) and their design are then computed at the rows a cell reads
  (before, on ArbeitsplatzAtlas, a `selb_obs` cell read 148,963 coordinates,
  the whole design and everything upstream of it). Its 0.1.22 does the same
  for a value two statements of one `@plate for` read, and for a dotted value
  reading an untyped numeric array at literal positions (an `amtlich_pub` cell
  read the whole `b_k_gem_kreis`, 7,200 coordinates). The test environment
  pins ReactiveKernels `0d3f92345`.

## 0.2.76

- Added: `brm_laplace_problem(rk; ad_backend, integrate = …)` also integrates
  the `cdar(step; by=g)` walks over a named grouping (their standardized
  innovations) and the `car(area; edges)` fields over a named area column (the
  default lowering: the ICAR values under their graph precision and BYM2's iid
  parts), together with the random-effect blocks over it. A field's
  per-component soft sum-to-zero stays out of the sparse factor (Woodbury).
  Refused: a field lowered with `car_units`, a field component that no row
  reads, a walk correlated across groups (`cor=` with off-diagonal entries).

## 0.2.75

- Added: `brm_laplace_problem(rk; ad_backend, integrate = (:g, …))`
  integrates every level of the named groupings' random-effect blocks by one
  sparse Laplace, and returns a `BRMSparseLaplaceProblem` (exported) with the
  interface and draws of the other two Laplace problems. The groupings may be
  crossed, nested with any number of levels per parent, or cut across the NLME
  units; blocks over other groupings stay sampled coordinates of the target.
  The model needs the row-structured problem's shape (every observation a
  Poisson-log, binomial-logit, `NegativeBinomial2` or Normal regression
  response). The integrated effects' Hessian is one sparse matrix factored by
  CHOLMOD; each set of effects that share likelihood terms takes its own Newton
  steps; the gradient reads the factor's selected inverse.
  `brm_laplace_draws(target, P; conditional = K)` resamples per component.
- Changed: `[compat] SelectedInversion = "0.2.2"` (was `"0.2"`), for
  `selinv_extract`.

## 0.2.74

- **Breaking for saved points:** `RKBRMI(brmi; car_units)` lowers each `car`
  field so that every area reads its mean and `⌈log₂ n⌉` of its unit's draws.
  A unit's interior in one component (a piece of `n` areas, in breadth-first
  order over its edges) is its mean plus an orthonormal Haar basis of its
  zero-sum plane; the shared coordinates `w` are the separator values and piece
  means, each component's last one replaced by its total over `0.001 n_c`. The
  density is the ICAR density with its normalized sums, up to a constant: the
  shared law `_BRMCarShared` holds the components' totals and the edges between
  shared values, and every other edge, with `+½ z²` for each plane draw, is a
  term of its unit, observed as a log weight on a placeholder column and
  attributed to that unit by `brm_nlme_model`. Before, each piece was whitened
  through a sparse Cholesky factor of its precision, so every area depended on
  all of its unit's draws: on ArbeitsplatzAtlas a Gemeinde cell's compact
  gradient touched about 2,230–2,390 effects, a unit's η Hessian was dense in
  each field's draws, and one unit's value and gradient through its compact
  cells was estimated at 452–3,601 s, against 1.56 s measured for one
  whole-model value and gradient over all 12 units (BRM 0.2.63). Now a cell
  reads its own areas' splits (plus the field's shared values `w`, which an
  opaque map computes whole, linear in their count) and a unit's η Hessian is
  its edges' pattern in that basis; on a 16 × 16 grid in four units a cell
  reads at most 47 of the model's 265 coordinates, 35 of them `w`. Field, density and the coordinate
  transport to SBBRMI's field are unchanged up to that constant; points saved
  from an earlier separated lowering do not carry over. The Haar draws are not
  whitened but stay well conditioned under the diagonal metric SAEM's
  `WarmupHMCNUTS` adapts: a unit's diagonally scaled condition number on the
  atlas's graph is 13–133, against 9.4e4–2.7e7 for consecutive differences
  (design decisions `1g11z26` and `1y5br8u`, snags `saem-s-nuts-warm-28e5f1d9`
  and `car-units-local-67dc9162`).
- A completed covariate's observed rows read no drawn entry, and a missing row
  reads its own entry only: RKBRMI completes it as
  `stacked = vcat(observed_component, missing)`, `completed = stacked[order]`,
  `order` each row's position, which ReactiveKernels computes at the rows a
  cell reads (`brm_covariate_geometry` returns the observed component and
  `order`; before, a lookup and a mask, so every observed row looked drawn
  entry 1 up and masked it out, and a cell on it computed that entry and what
  it reads). Values are unchanged (RK snag `on-arbeitsplatza-6065a1ef`).
- BRM's compat floors are ReactiveKernels 0.1.18 and ReactiveKernelsPPL 0.1.18,
  which compute `vcat` operands, multi-index gathers and group totals at the
  rows a cell reads, and evaluate a cell whose number two of its values read
  (0.1.16–0.1.17 raised `MethodError` in `_cell_take_rows` there); the local
  fields and completions rely on them for cells that cost their own rows. The
  test environment pins ReactiveKernels `5e288b439`.

## 0.2.73

- Fixed: `deepcopy` of an `RKBRMI`, a `BRMNLMEModel`, a `BRMLaplaceProblem`
  and a `BRMStructuredLaplaceProblem` works again (it raised `deepcopy of
  Modules not supported`). A copy shares the backend's translated program
  (code, like a compiled function) and copies its data, model and workspaces,
  so WarmupHMC's samplers that copy their density per chain
  (`adaptive_warmup_mcmc(rngs, target)`, `completion_warmup_mcmc`) run on these
  targets.

## 0.2.72

- SBBRMI gives `restricted(D, lower, upper)`, and the deprecated keyword
  bounds that mean it, `D`'s plain kernel, also where a bound or argument is
  sampled. It emits StanBlocks 0.2.7's `restricted(family, args...; lower,
  upper)`: the bounds go on the parameter declaration and the model block keeps
  the family statement, `real<lower=location> x; x ~ normal(0.2, 0.7);`, as
  TuringBRMI already did; RKBRMI still refuses a sampled bound (decision
  `0q18o98`). Before, SBBRMI
  normalized such a prior (the `truncated` law). This covers every prior site:
  scalar, `.~` and vector priors (each margin of a several-family vector prior
  too), `sd(...)`, `effect(...)`, `me()`'s latent, GP term and R2 priors. With
  constant bounds and arguments the emitted Stan is unchanged. `truncated(...)`
  keeps the normalized law.
  - A prior-only program draws a restricted prior in generated quantities only
    when its mass inside the bounds is constant; otherwise the prior stays a
    sampled parameter (StanBlocks' rule).
  - `[compat] StanBlocks = "0.2.7"`.

## 0.2.71

- Fixed: an inline lambda or `do`-block argument of a top-level assignment
  lowers on SBBRMI again (`cannot lift to Stan expression: _BRMInlineBody`
  since `33d9f4aa`).
- Fixed: `brm_prediction_grid` builds for an `mi()` model again: an `mi()`
  column is left out of the grid, as documented, and is refused by name as a
  focal or `fixed=` column. Conditional draws still refuse `mi` models.
- Fixed: `brm_descriptor` and `ranef_blocks` no longer read a structured
  term's group block (`b_<field>_<group>`, for example a consumer term's
  `:correlated_normal` field) as an `|ID|` bucket.
- Fixed: on RKBRMI a Stan-only observation family whose constructor comes from
  a `_rk_callable_source!` provider keeps that constructor route instead of
  being refused for lacking an `_rk_observation_source!` provider.

## 0.2.70

- SBBRMI: `reprocess(sb, rows; resample_groups = [:g])` accepts a grouping
  column `g` of a shared `|ID|` block whose predictors join several columns by
  label (`(1 | k | kreis)` beside `(1 | k | kreis_b)`). Before, it was refused.
  - Naming some of the block's columns redraws the margins their predictors
    read, for every label those columns name in `rows`. A label that is a
    fitted joint level (another column observes it) keeps its fitted margins
    there and draws the redrawn ones from the block's law given them; a label
    the fit never saw draws all of them. The fitted block stays a parameter,
    which the other columns' predictors read as fitted, so the replay has the
    fit's coordinates. Student-t levels keep their fitted weight or draw one.
  - Naming every column redraws the whole block over the labels they name,
    like an ordinary block.
  - It composes with a `cdar(step; by = g, ...)` walk on the same column, whose
    levels `resample_groups` already redraws (0.2.36).
  - Refused: some columns of a joined block with `freeze_constants = false`
    (the redraw is given the fitted block, which a fresh replay rebuilds), and
    any joined column on RKBRMI.
  - Snag `reprocess-resamp-dab66282`.

## 0.2.69

- An NLME model's preparations are counted in its progress tree.
  `brm_nlme_model`'s `Subject cells` step counts its cell queries, one per
  group of cells (`Cell queries (k/n)`). The compact cell gradients a model
  prepares on its first per-subject use (a cell gradient, the η Hessian
  patterns, SVI's unit terms) report under the phase whose estimator needs them,
  as `Subject cell gradients`: its steps `Cell reads`, `Cell reads by gradient`
  and `Cell plan check` count each subject's groups of cells, and `Compact cell
  samplers` (and `Narrowed compact cell samplers`) the groups. That phase is
  `brm_nlme_fit`'s estimator or EBE pass, or `brm_svi_fit`'s target; a first use
  from a direct estimator call reports nothing, as before. On ArbeitsplatzAtlas's
  whole model this first use ran for hours inside SAEM's `NUTS warmup (0/12)`
  with nothing counted. Values are unchanged.
- BRM's compat floor is ReactiveKernelsPPL 0.1.16 (was 0.1.15). Before it,
  every cell query and compact cell sampler BRM prepares regenerated the whole
  program: on ArbeitsplatzAtlas's model (223,941 coordinates, 32 groups of
  cells) about 45 s apiece whatever its cells, 24 of the 39 minutes of
  `brm_nlme_model`'s `Subject cells` step and again for each sampler on first
  use (RK snag `brm-s-nlme-cell-d70abd20`). From 0.1.16 a compiled cell query
  prepares in 0.02 s and a second sampler in 0.1 s; a first preparation costs
  its compilation. The test environment pins ReactiveKernels `452071f1b`.

## 0.2.68

- `brm_descriptor`, `ranef_blocks` and `brm_ranef_draws` read a grouping named
  by a data-only assignment, such as a level-table gather
  `kreis_b = kreis[gem_b]`, as they read a data column. They used to refuse it
  ("grouping factor `kreis_b` (from `kreis_b_idx`) is not a raw data column of
  this model") although SBBRMI and RKBRMI lowered the model (snag
  `brm-descriptor-r-20f52245`). `column_data(brmi, name)` now returns a
  data-only assignment's value, evaluated as the backends evaluate it, and
  `nothing` only for a name that is neither a data column nor such an
  assignment.

## 0.2.67

- RKBRMI binds an `mi()` column (a completed covariate, an `mi(y)` response, a
  completion with group totals) as two concretely typed inputs: its values
  `<x>_raw`, with `one(T)` standing at each missing entry, and its presence
  mask `<x>_present`, a `Vector{Bool}`. It used to bind `<x>_raw` as the
  `Vector{Union{Missing,Float64}}` the data hold, which ReactiveKernels'
  type-stability check cannot type, so the data-only kernel that binds the data
  could not be audited. The observed values, the observed and missing rows and
  a group-totals layout are still computed in the graph, from the pair; values
  and densities are unchanged. `rk_bound_data(rk)` and `binder(design)` carry
  the new `<x>_present` input, so a JAX module's `prepare_data` takes it too.
  A masked `hsgp(x; rows = keep)` axis is still bound as the data hold it.

## 0.2.66

- `restricted(D, lower, upper)` and `restricted(D; lower, upper)`: a prior law,
  `D`'s density on the bounds, not renormalized: Stan's
  `real<lower=lo, upper=hi> x; x ~ D;`. `truncated(D, lower, upper)` stays the
  normalized law. With constant bounds and arguments the two give the same
  posterior; where the mass inside the bounds moves with sampled parameters,
  `restricted` restricts the joint prior and `truncated` draws the arguments,
  then `D` truncated (decision `0q18o98`). Both work on any prior position,
  including `sd(...)`, `effect(...)` and `me()`'s `latent(...)`.
  - TuringBRMI evaluates `restricted` as the plain kernel.
  - RKBRMI emits RKPPL's `restricted` (constant bounds; it refuses a sampled
    bound, as before).
  - SBBRMI currently normalizes when a bound or argument is sampled: its
    StanBlocks pin has no plain-kernel spelling yet (StanBlocks snag
    `in-slic-declare-996328e0`).
- **Deprecated:** `D(args...; lower, upper)` keywords on a prior. They warn once
  per session and keep meaning `restricted`. The warning fires only where the
  statement is surely a prior: an unbound statement over a data vector is an
  omitted response, and its keywords keep bounding its draw.

## 0.2.65

- Labels read from a level table through each part's positions, the data
  holding one table (each area's label, unit and Kreis) and each part only
  its rows' positions: name the gather, `kreis_b = kreis[gem_b]`, and use
  `kreis_b` as a grouping, in a shared `|ID|` block over parts, in
  `gr(g, by = b)`, `(… || g)`, `cdar(…; by = g)`, as a `car(...)` area and as
  a `car_units` / `nested_groups` key. `test/level_table_groupings.jl` checks
  each against the same model given one copy of the labels per part: SBBRMI
  emits the same program and data, RKBRMI has the same coordinates, density
  and gradient, and binds the level table and positions instead of the
  copies.
- A gather written inline in a grouping or term (`(1 | kreis[gem_b])`,
  `gr(g, by = unit[gem_b])`, `car(gem[gem_b]; …)`) is refused with the
  assignment to write (`kreis_gem_b = kreis[gem_b]`): an inline gather has no
  name for a prior, an NLME keyword or the descriptor to address.
- BRM's compat floor is ReactiveKernelsPPL 0.1.15 (was 0.1.14). Before it,
  RKBRMI's density of `gr(g, by = b)` raised `invalid index: "a" of type
  String` (or a `BoundsError` for integer labels other than `1:n`) when `g` or
  `b` was a data-only assignment: RKPPL looked a label up on a `levels(g)`
  axis only for a bound data column (snag `rkppl-submodel-l-00ff0f27`).
  **Breaking** for an environment that pins ReactiveKernels before
  `4f7ef46a`: move that pin with BRM.
- BRM's test environment pins ReactiveKernels `4f7ef46a` (0.1.13).

## 0.2.64

- RKBRMI finds each row's random-effect level (`brm_level_indices`) through one
  lookup table per grouping instead of a search of every level per row, so a
  grouping costs its rows plus its levels. On 174,912 rows over 10,932 levels
  (ArbeitsplatzAtlas's Gemeinde blocks) that takes 4–9 ms, where the search
  needed about 0.3–2.6 s depending on the label type; the indices are the same.

## 0.2.63

- Preparing a model's compact cell gradients on first use costs each cell's own
  reads. Which η a cell reads was measured from compact gradients over its
  subject's whole block, at two probe points and a check, so a unit cost its
  cells times its block. Now each cell's value query runs at the probe points
  through a point that records what it reads; of those reads the plan keeps the
  η its compact gradient touches, the same reads as before, and the check
  compares every cell's value through the plan with its value query. No
  whole-block sampler is prepared, so first use prepares one sampler per group
  of cells (two when a group's width shrinks). On the two-region panel the
  steady cost of the measurement went from 4.9 s to 2.0 s at 21,732 effects per
  unit and from 32.8 s to 7.7 s at 69,473 (one CPU); reads, plans and η Hessian
  patterns are unchanged.

## 0.2.62

- `brm_nlme_loglikelihood` on a nested unit costs time linear in its cells'
  reads. The point it evaluates the unit's cell queries at looked each packed
  coordinate up by a linear search over the subject's block, so a unit's value
  cost its cells times its block, quadratic in the unit's size. NLMEEstimation's
  per-subject `conditional_loglikelihood` and the two-stage unit problem's log
  density take that path. On the two-region panel one unit's value took 1.04 s
  at 21,732 effects and 14.1 s at 69,473; it now takes 0.053 s and 0.164 s
  (one CPU), with values unchanged.
- `test/fitted_constants.jl` expects rank-3 constants and supplied rank-3 data
  to transpile: the pinned StanBlocks has the rank-3 array type, so the two
  `@test_broken` checks had become unexpected passes.

## 0.2.61

- `RKBRMI(brmi; car_units)` takes one entry per `car(...)` field, keyed by its
  area column, for fields on different row sets:
  `car_units = (gem_id = :lgroup, gem_id_etr = :lgroup_etr)`. Each entry is a
  column name or a vector of unit labels, row-aligned with that field's rows.
  A field without an entry keeps the default lowering, and a key that names no
  `car(...)` area column is refused. One column or vector still labels every
  field. Before this, a model whose `car` fields sit on different row sets
  could not use the separator at all ("`car_units` has 43796 labels, but
  `car(gem_id_etr)` has 32847 rows"). A rebuild replay
  (`freeze_constants = false`) passes keyed column names again.

## 0.2.60

- A model whose observations have subject cells prepares its compact cell
  samplers on first use (a cell gradient, a nested model's η Hessian patterns,
  SVI's unit terms or a two-stage fit), two per group of cells, instead of one
  per group and distinct subject block length while `brm_nlme_model` runs.
  Measuring which η each cell reads takes one whole-block sampler per group,
  prepared at the longest block (a shorter block is padded with the longest
  block's η, held at the unit point) and dropped once the per-cell plan is
  checked. On the whole ArbeitsplatzAtlas model (32 cell groups, 12 block
  lengths) that is 64 samplers instead of 416. With ReactiveKernels 0.1.12 one
  sampler's preparation takes about 1 s and 270 MiB at 139k packed
  coordinates, whatever its selection width, after a first one per group that
  compiles. Values and gradients are unchanged.
- Preparing the per-cell plan no longer copies the packed point per subject or
  scans a subject's whole block per cell.
- BRM's compat floor is ReactiveKernels 0.1.12 (was 0.1.7). Before 0.1.12,
  preparing the reverse-over-forward Hessian-vector-product query that dense η
  Hessian blocks and the per-unit Laplace read holds 8n² bytes for the n packed
  coordinates (0.2.57); from 0.1.12 it is linear in n. ReactiveKernelsPPL 0.1.14,
  BRM's floor since 0.2.56, admits ReactiveKernels from 0.1.9, so an environment
  that develops the two packages from different ReactiveKernels checkouts could
  still resolve such a version. **Breaking** for that environment: develop
  ReactiveKernels from `2d415b7e` or later (`3deb61ae`, the ReactiveKernelsPPL
  0.1.14 tip, qualifies).

## 0.2.59

- A `car(...)` field lowered by its unit border separator
  (`RKBRMI(brmi; car_units)`) maps each unit's interior through a sparse
  Cholesky factor of the unit's interior precision, applied by sparse
  triangular solves, instead of dense `n_u × n_u` matrices: memory and every
  gradient are linear in the factors' entries. The shared values (separator
  areas and interior sums) are mapped the same way, through a sparse factor
  of their precision without the sum-to-zero, whose total is its own draw,
  instead of a dense `m × m` factor. On the 10,932 Gemeinden of Germany in 12
  Ländergruppen a field's factors hold 2.2 MiB instead of 121.5 MiB (the
  shared factor 36,970 entries instead of 453² = 205,209) and its separation
  is built in 0.17 s instead of 6.3 s. With three such fields on 43,728 rows
  a gradient takes 13.5 ms instead of 70–101 ms and the separation data
  7.1 MiB instead of 365 MiB (gordito, one CPU). The
  field and its density are unchanged; the unit draws parametrize each
  unit's interior through another orthogonal rotation, so a point saved from
  an earlier separated lowering does not carry over.
- `brm_coordinate_transport` pairs such a field with SBBRMI's field: its
  shared and unit draws pair with Stan's areas as block positions
  (`:car_separated`), and `brm_rk_point_to_stan`, `brm_stan_point_to_rk`,
  `brm_stan_gradient_to_rk` and `brm_stan_logdensity_to_rk` move it through
  the separation's exact linear map (`t.fields`). So `brm_nlme_joint_map`,
  `brm_nlme_joint_start` and `brm_nlme_start(m, map, stan_u)` take a
  separated model, and SAEM points and stage-2 draws reach SBBRMI's readers.
  `brm_check_coordinate_transport` compares the field area by area and
  returns a `fields` count. A Stan-frame `brm_nlme_conditional_problem`
  (`backend = map.sb`) of a separated model is not built (Stan samples the
  areas, none of which is a unit draw) and says so; condition `map.rk`.
- A `car(...)` graph's BYM2 scaling factors are computed exactly from sparse
  factors: the selected inverse of the grounded Laplacian's sparse Cholesky
  factor (new dependency `SelectedInversion = "0.2"`) instead of a dense
  eigendecomposition of each connected component, which was cubic in its
  areas and paid by every `car` field, ICAR included. For the 10,879
  connected Gemeinden: 0.025 s instead of 147 s and 903 MiB per field on one
  CPU; the factors agree to 1.4e-14. Lowering three ICAR fields over them
  (`RKBRMI(brmi)`, compiled) takes 0.5 s instead of 370 s.
- RKBRMI looks each row's area up among the graph's node labels as a data
  column (`<term>_nodes`, one of the fitted graph inputs `rk_design_binder`
  binds once) instead of a literal of every label in the program, and a
  separated field's gathers read their positions without a literal of their
  range: a literal of 10,932 labels cost about 14 s of compilation.

## 0.2.57

- `brm_nlme_model` prepares its sampler Hessian-vector-product query the first
  time it is used (by dense η Hessian blocks and the per-unit Laplace), not
  during the preparation, so a fit whose estimators take the η curvature from
  gradients never compiles or holds it. With ReactiveKernels before 0.1.12 that
  preparation holds 8n² bytes for the n packed coordinates
  (DifferentiationInterface's reverse-over-forward preparation also prepares a
  forward-mode gradient, whose basis is n one-hot vectors of length n): 200 GB
  for a model of 157,808 unit effects, which no `brm_nlme_model` survived.
  ReactiveKernels 0.1.12 prepares it in memory linear in n.

## 0.2.56

- BRM's compat floor is ReactiveKernelsPPL 0.1.14 (was 0.1.12). On 0.1.12, a
  flat three-margin `(1 | pk | subject)` block's compact cell gradient raised
  Enzyme's `EnzymeRuntimeException` ("could not determine how to keep a pointer
  phi … rooted"); `test/nlme_view.jl` now checks its gradient against central
  differences.
  ReactiveKernelsPPL 0.1.14 also reads a compact cell's selected coordinates in
  constant time. **Breaking** for an environment that pins ReactiveKernels
  before `3deb61ae`: move that pin with BRM.
- BRM's test environment pins ReactiveKernels `3deb61ae` (0.1.12).

## 0.2.55

- A nested NLME unit's value and η gradient through its subject cells
  (`brm_nlme_loglikelihood_and_gradient`, `brm_svi_fit`'s unit terms, SAEM's
  per-subject population gradient, `brm_two_stage_fit`'s units) now cost each
  cell's own coordinates: every compact cell call selects the population
  coordinates and only the effects that cell reads, measured once from the
  cell's gradient and checked against the whole-unit selection. Before, every
  call worked over the unit's whole block of effects, so a unit cost the square
  of its size. Unit 1 of the regional panel (`test/nlme_panel_fixture.jl`):
  2,885 effects 21.5 → 18.3 ms; 21,732 effects 363 → 166 ms (7.7 → 6.55 and
  17.1 → 7.81 µs per cell call).
- `brm_nlme_model` no longer evaluates the cells when it prepares a model: the
  cell reads, selections and η Hessian patterns are built on first use. A model
  whose cell gradients are never taken (SAEM's batched path, a whole-model
  fit) is not held up by them. The pattern 0.2.54 stored as `m.eta_sparsity` is
  read through `NLMEEstimation.eta_hessian_sparsity(m, i)`; the field is gone.

## 0.2.54

- A nested NLME model (`brm_nlme_model` with `subject_group` /
  `nested_groups`) whose observations have subject cells declares each
  subject's η Hessian pattern to NLMEEstimation (`eta_hessian_sparsity`):
  which of the subject's effects enter one observation cell together. With it,
  `empirical_bayes` (and so `brm_nlme_fit`'s EBE pass after `saem`),
  `marginal_loglikelihood` and `covariance_step` compute sparse η Hessians
  from batched η gradients, linear in a subject's number of effects, instead
  of dense blocks from one η-HVP per coordinate (5.4 GB for one subject of
  25,871 effects). On the regional panel at 12 regions of 2,856–2,930 effects,
  the patterns take 1.6 s to build, hold 2.1% of each block, and
  `empirical_bayes` from the prior-median start converges for all 12 regions
  in 138 s. Flat models, and nested models without subject cells, declare
  none and keep dense blocks.
- `NLMEEstimation = "0.1.3"` is the compat floor (the version that adds
  `eta_hessian_sparsity`); SparseArrays is a dependency.

## 0.2.53

- BRM's compat floor for its RKBRMI extensions is ReactiveKernelsPPL 0.1.12
  (was 0.1.3). Since 0.2.47 RKBRMI observes a `@plate` totals response as the
  plate's one value per cell. When the cell reads a predictor composed from
  another predictor's values, e.g. `f_off = 0.4 * mu[main]` then
  `log_f ~ 0 + offset(f_off) + (1 | fg | m)`, ReactiveKernelsPPL before 0.1.12
  refuses that value ("applies `_ppl_plate_column` inside a composition"). The
  floor makes Pkg refuse such an environment when it resolves instead.

## 0.2.52

- `RKBRMI(brmi; car_units)` lowers every `car(...)` field by its unit border
  separator. `car_units` gives unit labels row-aligned with the model rows,
  either the name of a column the model reads (`car_units = :region`) or the
  labels themselves; each area takes the unit of its rows. The separator holds
  the areas of no unit or of several units and, on each border between two
  units, the border areas of the side with fewer of them. Given the separator
  values and each unit's interior sum (per connected component), the units'
  interiors are independent Gaussians. The field is then sampled from
  standard-normal draws: shared ones for those values (`<field>.w`) and, per
  unit `k`, ones for its interior (`<field>.z_k`). It is the default
  lowering's field and density, up to a constant Jacobian, the soft
  sum-to-zero included. A component inside one unit is all that unit's draws.
- `brm_nlme_view` reads such a field lowered over the subject labels as unit
  draws: each subject's interior draws, and a BYM2 field's per-area iid part,
  are size-1 families fixed at 1. θ then holds the separator values and the
  sums instead of one coordinate per area, which keeps the unit factorization
  of SAEM, two-stage fits, SVI and per-unit Laplace for a field over
  Gemeinden in Länder. Units that are not the subjects are refused by name.
- The per-unit factors are dense (`n_u²` per unit, `m²` for the shared
  values).
- `reprocess` keeps the fit's separation on a frozen replay, and lowers a
  rebuild (`freeze_constants=false`) with the same `car_units` column (labels
  passed as a vector are refused there, since they label the fit's rows).
  `brm_coordinate_transport` refuses a separated field by name: the linear map
  to Stan's field is not built yet.

## 0.2.51

- BRM's compat floor for its RKBRMI extensions is ReactiveKernels 0.1.7 and
  ReactiveKernelsPPL 0.1.3 (was 0.1). A model-scope `logistic` value that
  RKBRMI emits as `logistic.(...)` (0.2.45) can be a location summand scaled
  by another value, `e = logistic(b) * exp(a)` then
  `y ~ Normal(p + e * exp(a), sigma)`; ReactiveKernelsPPL before 0.1.3 fails
  to bind it ("term references missing column `_rkppl_leaf_1`"). The floor
  makes Pkg refuse such an environment when it resolves instead.
  **Breaking** for an environment that pins ReactiveKernels before
  `2a682926`: move that pin to `2a682926` or later together with BRM.
- BRM's test environment pins ReactiveKernels `2a682926` (0.1.7). The three
  inline count-location parity checks in `test/rk_inline_response_location.jl`
  are ordinary tests now (they needed ReactiveKernelsPPL 0.1.2).

## 0.2.50

- `brm_nlme_model` attributes an observation on rows other than the subject
  column's to the subjects it reads (before, it refused one whose density
  count was not one per subject-column row: "its densities cannot be
  attributed to subjects"). This covers a fragment's rows read through an
  index (`exp(log_f[f_obs])`), a plate over the fragment's own groups
  (`groups(f_hid => f_ky)`) and an observation that reads no random effect.
  Each density's subject is measured on the pointwise query, moving the η
  blocks of halves of the subjects (2⌈log₂ n⌉ evaluations for n subjects),
  so SAEM, SVI and the per-unit Laplace problem take such models. A density
  that reads several subjects' random effects is refused, naming them; one
  that reads none counts toward the first subject. Each measured
  observation's densities are cells of their own subjects, so
  `brm_nlme_loglikelihood` and the compact cell gradients evaluate one
  subject's. Snag `nlme-model-obser-dd665aca`.

## 0.2.49

- `brm_coordinate_transport` pairs the completions of
  `RKBRMI(brmi; completions = :noncentered)`: each standardized draw
  `<column>.z_mis.<i>` (the free entries of `mi(x)::shares(total, rows)`:
  `<column>.z_free.<i>`) pairs with SBBRMI's centred entry
  `<column>_y_mis.<i>` (`<column>_y_free.<i>`) as a `:completion` pair, and
  the part moves through `s = location .+ scale .* z` on Stan's unconstrained
  entries (the values under `Normal`, their logs under `LogNormal`), with the
  law's location and scale RK computes at the point; they may read any
  coordinate, another completion's values included. `brm_rk_point_to_stan`,
  `brm_stan_point_to_rk`, `brm_stan_gradient_to_rk` (the location and scale
  pulled back by reverse-mode Enzyme) and `brm_stan_logdensity_to_rk` (plus
  the log-Jacobian `sum(log, scale)`) move such points, gradients and
  densities, so an NLME fit of the non-centered lowering reaches the same
  BRMI's Stan program. Before, those draws stayed unpaired and the transport
  refused the model. The transport records each part in its new field
  `completions` (the struct gains `completions` and `completion_laws`; the
  5-, 7- and 8-argument constructors are kept); reading
  `logdensity_offset` from a transport with one is an error naming
  `brm_stan_logdensity_to_rk`, and the permutation helpers `brm_rk_to_stan` /
  `brm_stan_to_rk` refuse it. `brm_check_coordinate_transport` compares each
  part's Stan entries with the values RK computes from its draws and returns
  their count as `completions`. Snag `coordinate-trans-23993804`.
- `brm_nlme_joint_map` and `brm_nlme_joint_start` accept non-centered
  completions read as unit draws: their family of fixed unit variance has no
  scale or factor, and the start maps their draws to Stan again once it has
  written the covariance, since the law's location or scale may read it.
- RKBRMI's non-centered completion block also binds its law's location and
  scale to block locals (`<column>.y_mis_location`, `<column>.y_mis_scale`;
  `y_free_location` / `y_free_scale` for free entries), which the transport
  reads by name. No density reads them (the value still reads the law's
  expressions), so the density, its gradient and the coordinates are
  unchanged.

## 0.2.48

- A nested NLME model (`brm_nlme_model` with `subject_group` /
  `nested_groups`) whose observations have subject cells now evaluates one
  unit on its own cells: `brm_nlme_loglikelihood_and_gradient`,
  `brm_nlme_unit_problem` and SAEM's per-subject population gradient take one
  compact cell sampler set per distinct subject block length, as flat models
  do, instead of a whole-model gradient with the other units' effects at zero.
  `brm_two_stage_fit` therefore samples nested units on `threads` tasks (each
  with its own samplers); before, a nested model sampled on one thread. Only
  a nested model without subject cells keeps the whole-model route and one
  thread. A unit's cost is its own cells' only as far as RK computes each
  cell's reads at its rows (ReactiveKernels 0.1.6 with the scalar totals
  cells of 0.2.47); a totals cell's gradient reads its `hcat` design from
  views, which needs the Enzyme `dogfood` line, so on registered Enzyme such a
  model's unit gradient raises `IllegalTypeAnalysisException` where it was
  slow before.
  Each compact cell call costs about 7 µs, so a unit gradient is cheaper than
  the whole-model one only while the unit holds less than about a twelfth of
  the model: 18.4 ms against 26.7 ms for 12 units of 2,795 cells, but 0.23 ms
  against 0.035 ms for 3 small regions (ReactiveKernels snag
  `compact-cell-sam-7b0032c9` asks for one call per unit).

## 0.2.47

- On RKBRMI, an in-cell observation whose response holds one value per cell
  (a `@plate` totals cell's `total[k] ~ Normal(sum(exp.(log_w[rows[k]])), …)`)
  emits its arguments as one scalar per cell and observes them as they are,
  instead of `ones(length(total[k])) .* s` cells flattened by
  `flatten_cells`. RK can then attribute response row `k` to cell `k`, so a
  compact cell gradient (`brm_svi_fit`, the NLME cell queries) costs that
  cell's own rows. On the regional panel of `test/nlme_panel_fixture.jl`, one
  unit's totals cells cost 141,888 B per gradient at both 3 and 9 regions,
  where they grew from 632,016 B to 1,624,032 B before. The emitted program
  no longer reads such a response inside the cell, so its cell input port
  (`<kernel>_input_<y>`) is gone from the RK inputs; densities are unchanged.
  The totals' Reverse gradient differentiates an `hcat` design read from
  views, which needs Enzyme dogfood `d37cb0b4` or later and ReactiveKernels
  0.1.6; registered Enzyme raises `IllegalTypeAnalysisException` there.
- BRM's test environment runs the Enzyme `dogfood` line (`test/setup_env.jl`,
  user decision `0n0kgfh`) and ReactiveKernels 0.1.6, so it builds only on a
  KB fleet host with the dogfood libEnzyme build (gordito). On it,
  `test/rk_enzyme_threads.jl` (concurrent-GC gradients),
  `test/plate_index_into.jl` and `test/rk_structured_consumer.jl` pass.

## 0.2.46

- `brm_coordinate_transport` pairs a stratified `gr(g, by = b)` random-effect
  block: each stratum's scales by margin and its correlation factor, by the
  `b` level, and the standardized draws by (level of `g`, margin). An RK point
  or draw therefore reaches the same BRMI's Stan coordinates
  (`brm_rk_point_to_stan`, `brm_stan_point_to_rk`), and `brm_nlme_joint_map`,
  `brm_nlme_joint_start` and `brm_nlme_conditional_problem` accept such a
  block: an NLME estimate's per-stratum covariance and each level's natural
  effect become that stratum's Stan scale and the level's standardized draw.
  Before, the block's coordinates stayed unpaired and both routes refused the
  model. A stratified pair's address carries `by` and `stratum`, and a
  stratum's factor in `transport.correlations` its `stratum`, `rk_index` and
  `stan_index`. Snag `stratified-block-7a992e61`.
- `brm_nlme_view` reads a stratified block with more than one margin,
  `(1 + x | p | gr(g, by = b))`, as one covariance family per stratum with its
  own correlation factor, as it reads a one-margin block. Before, it refused
  the block (`declaration … .L.2 owns no packed coordinate`): it claimed
  per-stratum declarations `L.<s>`, while RK packs every stratum's factor in
  one `K × K × levels(b)` value with coordinates `L.<m>.<s>`. `brm_nlme_model`,
  `brm_nlme_start` and the NLME fits then take such a block. BRM todo `1srx3ud`.
- `RKBRMI` reads a stratified block's scales at each row's stratum label when
  its margins' SD laws differ (`sd(mu, p, x) ~ …` on
  `(1 + x | p | gr(g, by = b))`). Before, it collected the per-margin scale
  vectors into a matrix with rows `1:S` and indexed that by the `b` label, so
  a String label failed with `ArgumentError: invalid index`. Blocks whose
  margins share one law, and `b` coded `1:S`, were unaffected.

## 0.2.45

- `RKBRMI` evaluates a model-scope `logistic` elementwise, as it does `exp`
  and `log`, and as SBBRMI's `inv_logit` does: `p = logistic(lp + g[rows])`
  over a vector, `e = logistic(b) * exp(r)` over a scalar and a `logistic`
  inside a likelihood argument (`y ~ Normal(p + b * logistic(a), sigma)`) emit
  `logistic.(...)`. Before, RKBRMI emitted an undotted `logistic(...)`, which
  ReactiveKernelsPPL refuses as a value ("calls `logistic`, which only lowers
  in `.~` responses"; from ReactiveKernelsPPL 0.1.2, "`logistic` over a vector
  needs dots").
- A data-only `q = logistic(x)` of a data column is a data field on both
  backends, like `q = exp(x)`; before, it stayed a model value.

## 0.2.44

- `RKBRMI(brmi; completions = :noncentered)` lowers `mi(x)::shares(total, rows)`
  (before, it refused it): the grouped entries' coordinates, the centred ones
  that `brm_coordinate_transport` pairs with Stan's `y_shares`, are
  standard-normal draws `x.z_shares`, and each group observes the rest of
  their density (the law at its entries and the shares' log-Jacobian, less
  the draws' standard-normal density), so the density is the centred one
  under any law. The entries of no group need a location-scale law, as for
  any non-centered completion. `:pointwise` holds the group terms as the
  observation of one placeholder per group (`x_shares_weights`).
- `brm_nlme_view` reads those coordinates as unit draws: each is an instance
  of a size-1 family fixed at 1 in the subject owning its group's rows, and
  the group's term belongs to that subject, so `brm_nlme_model`, SAEM, SVI
  and the per-unit Laplace problem run, and the two-stage fit's unit problems
  differ from the full RK density by their unit alone. A group whose missing
  entries lie on rows of several subjects is refused by name. A centred
  shares completion is refused, naming the non-centered lowering: its
  density is a prior term reading the law, whose arguments may read the
  units' random effects, and the NLME paths evaluate the population prior
  with every random effect at zero. Before, a completion without free
  entries failed with `internal: declaration x.y_free owns no packed
  coordinate`, and one with them was read as shared coordinates in that
  population prior. Snag `nlme-view-shares-f1cfaa71`.

## 0.2.43

- `brm_start(rk)` and `brm_start(sb, stan_model)`, with no initial values,
  compute a start from the data. Pathfinder runs over the population
  coordinates with every unit's random effects integrated out by a Laplace
  approximation (`brm_laplace_problem`), from a point drawn within one unit
  of their prior medians (unconstrained scale); a start is a draw from its
  Gaussian fit, with each unit's effects drawn from its own Laplace Gaussian
  (`brm_laplace_draws`). Each group starts near its own value and each
  block's SD near the groups' spread, with no values written by hand.
  `chains = n` returns `n` independent draws from that fit for a sampler's
  per-chain `init`; a failed Pathfinder run raises. The units are
  `brm_nlme_view`'s (`subject_group`, `nested_groups`, `shared_groups`).
  Its cost is the Laplace problem's: a few model gradients per evaluation for
  regression-shaped models (the row-structured Laplace), many more for the
  whole-model Laplace. Initial
  values go on top of a start through `base`:
  `brm_start(sb, model, inits; base = brm_start(sb, model))`;
  `brm_start(x, inits)` without `base` keeps its zero base. Needs Pathfinder
  loaded (a new weak dependency; WarmupHMC loads it) and, for the default
  `ad_backend`, Enzyme and DifferentiationInterface. A model `brm_nlme_view`
  does not partition into units, and an SBBRMI with exact total-coefficient
  blocks, are refused with the reason.
- `brm_laplace_problem(rk; structured = :auto)` no longer gives the
  row-structured Laplace to a model with observations that are not regression
  responses: a `@plate` cell's, or one whose distribution reads a model value
  (`y ~ Normal(m, sigma)` with `m = …`). The row structure has no terms for
  them, so it was built without them, a marginal that left those observations
  out. Such models take the whole-model Laplace; `structured = true` refuses
  them with the reason.

## 0.2.42

- An omitted response whose law is discrete (`Poisson`, `Binomial`,
  `BetaBinomial2`, `NegativeBinomial2`, `Bernoulli`, `BernoulliLogit`,
  `BinomialLogit`, also under `truncated`) is drawn from the law its fitted
  twin's predictive draw uses: with `c` omitted, `c ~ Poisson(exp(eta) * w)`
  draws `poisson_int_rng(…, exp(eta) .* w)`, as `c_gen` does when `c` is bound.
  Stan has no discrete parameters, so such a statement is always an observation.
  Before, it was lowered like a prior, and the prior program failed to build:
  `logistic(...)` reached StanBlocks untranslated ("`tracetype` not defined for
  logistic(::real)"), vector products stayed `*` and failed stanc, and
  `truncated` over a discrete family was refused ("a sampled truncated prior
  requires a continuous scalar value").
- `BetaBinomial2(trials, mean, precision)` lowers with its response omitted and
  in a `@plate` cell, bound or omitted, as
  `beta_binomial(trials, mean .* precision, (1 - mean) .* precision)`. Before,
  each was refused: "`BetaBinomial2` derives outcome structure from observed
  response values …". Its outcome structure is its trials argument. `Ordinal`,
  `OrderedLogistic`, `CategoricalLogit` and `Multinomial` keep that refusal.
- A prior's arguments, and so an omitted continuous response's, spell Julia
  callables as likelihood arguments do: `logistic` as `inv_logit`, `normcdf` as
  `Phi`, `vcat` as `append_row`, `fill` as `rep_vector`, and so on. Snag
  `prior-draws-refu-aeb3b1e0`.

## 0.2.41

- `brm_nlme_view` (and `brm_nlme_model`, `brm_nlme_fit`, `brm_svi_fit`,
  `brm_laplace_problem` and the other NLME entry points, which pass it on) take
  a nested grouping's parent labels as a vector as well as by column name:
  `nested_groups = (g = data.g_region,)`, row-aligned with `g`. A backend keeps
  only the columns its formulas read, so a parent column no formula reads was
  refused as absent; it is now passed as labels, which are checked as a column
  is. By name, such a column is refused as unread, and the refusal shows the
  labels spelling.

## 0.2.40

- A `@plate` cell may read one vector through several `groups` bindings:
  `eta[rows1[k]]` and `eta[rows2[k]]` read the same formula predictor or data
  column at two row sets, each an independent gather, for example one sum per
  class with a coefficient each. Before, `@brm` refused the second read
  (`` would both bind cell name `eta`; rename one source before the plate ``),
  and that rename was not reachable for a predictor. On SBBRMI each read now
  gathers into its own data name, so a frozen `reprocess` replays both.
  Snag `reading-one-form-4a7413e7`.

## 0.2.39

- `RKBRMI` lowers a response location or scale computed inline from model
  values as it lowers the same value assigned first:
  `y ~ NegativeBinomial2(exp(dev) * (exp(w) * x + z), phi)` and
  `mu = exp(dev) * (exp(w) * x + z); y ~ NegativeBinomial2(mu, phi)` are one
  model on both backends. Before, a model whose other statements kept it on the
  formula route refused it ("mean must be the linear predictor `dev` itself").
  A predictor read bare, or bare inside `logistic(eta)`, `TDist(nu)`, Gamma's
  `mu/alpha` or Beta's `mu*kappa` / `(1-mu)*kappa`, keeps its formula
  spelling. An inline count rate or mean (`Poisson`, `NegativeBinomial2`, …)
  also needs ReactiveKernelsPPL 0.1.2 (RK main `f9a5c9c41`) or later;
  earlier RKPPL refuses it with `broadcast the link`.

## 0.2.38

- `RKBRMI` lowers `cdar(step; by=group, cor=C)`, the per-group stationary
  AR(1) walk, with SBBRMI's law: the marginal SD `sigma` and persistence `rho`
  (their stated priors and known values included), standardized innovations
  on the group × step grid, and the walk read at each row's group and step.
  `brm_coordinate_transport` pairs its coordinates with SBBRMI's. A frozen
  `reprocess` of an RK model with a `cdar` term (forecasts) is still refused.
- `brm_nlme_view` reads a `cdar` walk whose groups are the units, or nest in
  them, as unit draws: each group's innovations are one block of its
  subject, in a family fixed at the identity, and `sigma` and `rho` are
  shared latents. `brm_nlme_model`, `brm_laplace_problem`, `brm_svi_fit` and
  `brm_nlme_fit` (SAEM, empirical Bayes) take such models, and
  `brm_nlme_estimates` reads the walk's values by its term name. A `cor=` that
  couples groups of different units keeps the walk shared as a whole.
- `nested_groups` may also name the `by` grouping of a `cdar(...)` or grouped
  `hsgp(...)` term that no random effect uses, and `shared_groups` may name a
  `cdar` walk's `by` grouping, which keeps the walk shared.

## 0.2.37

- `using BayesianRegressionModels` binds the link and special functions BRM
  lowers: `logit`, `logistic`, `log1pexp`, `logexpm1`, `log1mexp` and
  `logsumexp` (re-exported unchanged from LogExpFunctions.jl) and `normcdf` and
  `norminvcdf` (from StatsFuns.jl). A model module needs no
  `using LogExpFunctions: logit` for a `logit(mu) ~ 1 + x` link any more; before,
  calling the builder raised `UndefVarError: logit not defined`. They are the
  functions SBBRMI spells as Stan builtins and the links whose inverse is one of
  them, kept complete by `test/exported_link_functions.jl`. Importing
  LogExpFunctions.jl or StatsFuns.jl as well stays unambiguous. A module that
  defines its own function of one of these names keeps it, provided it defines
  it before reading the name.

## 0.2.36

- `reprocess(sb, rows; resample_groups=[:g])` on an SBBRMI accepts the `by`
  column of a `cdar(step; by=g, cor=C)` term. It redraws every level of `g` the
  rows name, also levels the fit did not see, so a model with `cdar` can forecast
  whole held-out groups. Each redrawn level's path over the whole step grid (and
  any appended forecast steps) is drawn in generated quantities from the fitted
  `sigma` and `rho`, starting from the stationary law. The draw is conditional
  on the fitted levels the rows do not name, which keep their fitted paths;
  with `C = I` the levels are independent. The fit's unconstrained coordinates
  stay the replay's. A level the fit did not see needs `C` over the sorted union
  of the fitted and the rows' levels, read from the `cor=` data field of the
  replay data; its fitted block must equal the fitted `C`. A plain `reprocess`
  still refuses an unseen level, and its error now names `resample_groups`.
  With `freeze_constants=false` the redraw is refused (the fresh template
  rebuilds the term on the replay's levels alone). Every `cdar` program gains
  one data integer, two data matrices and one empty parameter vector
  (`eta_lev`), so its source and descriptor id change; its density and
  unconstrained coordinates do not. Snag `cdar-replay-new-57df8982`.

## 0.2.35

- `adaptive_centering_problem` centres a random-effect cell on the total of
  every term its level sits in, so its `c = 1` coordinate is that total:
  - the intercept or slope coefficient (as since `303dafff`);
  - for an intercept cell, the level of each `factor(f)` population term the
    level sits in (its cell mean or treatment contrast);
  - the effect of each coarser grouping of the same predictor that holds the
    level, nested or crossed, when that grouping's cell for the term is a
    scalar one.

  For example, `log(w) ~ 0 + factor(cov) + (1 | sec) + (1 | pair)`, with each
  `pair` inside one `cov` level and one `sec` level, centres a pair on
  `beta[cov] + b_sec[sec]`. Before, such a cell was centred on 0 (no intercept,
  and nested), which left the factor level, the section and the pair on a
  ridge only the priors pin.
  - **Behaviour change:** a nested grouping's cells (`(1 | g)` inside
    `(1 | band)`) are no longer centred on 0. They are centred on the intercept
    plus the enclosing level's effect.
  - **Unchanged:** a level inside a correlated grouping's cell keeps location 0.
    So do blocks compiled centred.
  - **Pair order:** containing blocks now come before the blocks they contain.
    A model declaring a nested grouping before its container sees its cells,
    and a per-cell `centeredness=` vector, in that order.
  - `adaptive_centering_controls` names every coordinate a cell's location
    reads in `location`, joined by ` + `.
  - The sampled active margins of a shared `|ID|` block whose other response
    is omitted (`<bucket>_active`) now carry their margin labels. They are
    located like the ordinary block of a model without the omitted response,
    and `adaptive_centering_controls` labels them. Before, they were always
    centred on 0.

  Snag `brm-zone-8-adapt-1b407672`, decision `1wxltaq`; closes owner todo
  `137ptwv`.
## 0.2.34

- A statement may read a name declared below it, in every position a read can
  stand. `mu ~ 1 + c_key` above `c_key = log(1 + x + exp(log_w) * z)` is the
  model with the two lines swapped, so a covariate computed from parameters,
  an `mi()` completion or data alone is a formula term, a grouping or a
  `standardize(...)` input wherever it is declared, and `effect(mu, c_key)`
  addresses its coefficient. Before, such a term failed on SBBRMI
  (`` expected data-backed NamedColumn for `c_key`, got MissingColumn ``) and
  RKBRMI (`` term `c_key` is not a data column ``) unless the declaration came
  first. A read of a column that a later `mi(y) ~ …` completes reads the
  completion, as a read below it does; before, it read the raw column with its
  missing entries and failed. A read of any other data column stays data, as a
  statement above `x = log(x)` reads the raw `x`. Snag
  `formula-covariat-5673c470`.
- `merge(a::BRMI, b::BRMI)` resolves the same way: a fragment that reads a name
  another fragment declares refers to that declaration, in either order. A
  fragment's binding of a data column it only reads (`m ~ 1 + y`) no longer
  replaces another fragment's likelihood of that column: `merge(likelihood,
  reader)` used to drop `y`'s likelihood from the model without an error.

## 0.2.33

- Groupings crossed with the NLME units. `brm_nlme_view`, `brm_nlme_model`,
  `brm_nlme_fit` (and so `brm_two_stage_fit`), `brm_laplace_problem`,
  `brm_svi_fit` and `brm_nlme_joint_map` take `shared_groups = (:g, …)`: a
  random-effect block over a grouping whose levels occur in several subjects
  (`(1 + x || class)` with every class in every subject) is read as shared
  latents, every coordinate of it (standardized draws, scales, correlation
  factor, Student-t mixing; plain, stratified, level-correlated and
  known-value blocks alike) in θ beside the population coefficients, and the
  subjects stay independent given θ. Every grouping except the subject
  grouping is named once, in `nested_groups` or `shared_groups`; the refusal
  now names the groupings named nowhere and the `nested_groups` keys that are
  no grouping (a column of a joined `|ID|` block among them). A `car(...)`
  field is read as shared latents too (its edges couple its areas), where it
  was refused. `brm_nlme_estimates` names a crossed block with `crossed =
  true` and its levels' `effects`. Before, `brm_nlme_view` refused any such
  model (``nested_groups must name every grouping except `lgroup` ``). Snag
  `crossed-grouping-39d39fcc`.

## 0.2.32

- A predictor holding `cdar(step; by = g, cor = C)` may be read through a
  `@plate` grouping, as in `rows = groups(cell => cells)` with
  `tot[k] ~ Normal(sum(exp.(lp[rows[k]])), sigma)`. `C` has one row and column
  per level of `g`, not one entry per row of the predictor. Before, SBBRMI and
  RKBRMI counted its `P × P` entries as rows and refused the predictor with
  "`lp` read through `cell`: `lp` has 160000 entries (data vector `C`) …",
  whether `C` was a data field or computed by `constant(...)`. `cdar`'s `cor`
  is now skipped like `cor` on `gr` and `hsgp` (0.2.5). Snag
  `cdar-cor-matrix-c8278be6`.

## 0.2.31

- `mi(x)::shares(total, rows) ~ Law(…)` with `rows = groups(g => labels)`
  declares a completion whose missing entries have known group sums: the
  completed `x[rows[k]]` sums to `total[k]`, the group's whole total, observed
  entries included. Per group the missing entries are the residual times
  shares (the bijection of a `T::shares(total)` cell latent), with the law's
  density at the entries plus `sum(log.(x)) - log(residual)`: the exact
  conditional of the law given the sum. A group with one missing entry holds
  its residual; missing entries of no listed group stay free. Positive laws
  only. Every formula reads the completed column as before.
  - SBBRMI: coordinates `x_y_free` and `x_y_shares`. The residuals are
    observed (`x_shares_residuals`), so they inform the law even when nothing
    reads the completion, with a pointwise density per group and a predictive
    draw of each group's sum.
  - RKBRMI: the same coordinates (`x.y_free`, `x.y_shares`, the grouped
    entries one joint latent), paired by `brm_coordinate_transport` (relation
    `:grouped_shares`).
  - A frozen `reprocess` keeps the groups and reads new totals.
  - Refused: `resample_covariates` of such a column, RKBRMI
    `completions = :noncentered`, real-valued laws.
  - The scientific description states the constraint.
  - Snag `mi-completion-wi-54222432`, design decision `0guq37f`.

## 0.2.30

- A data column that only a `name = constant(expr)` reads may hold `missing`.
  `expr` takes the column whole (`c = constant(mean(skipmissing(y_data)))`)
  and the model reads only the constant. `SBBRMI` and `RKBRMI` built such a
  model, but `brm_descriptor`, and with it `brm_start` on `SBBRMI`,
  `brm_description`, `prune_responses` and `modeled_covariates`, collected the
  column as model data and refused it ("data column `y_data` contains
  `missing` values"). They now collect the columns the backend does. A column
  the model itself reads still refuses its missing cells. Likewise, a
  `rows=`-masked HSGP axis that a constant also reads keeps its mask on those
  paths, and an `mm(...)` grouping that a constant also reads stays
  Julia-side.

## 0.2.29

- `brm_svi_fit` fits units whose own rows pin them far more tightly than the
  N(0, I) start of their factors, such as a probe's year of daily rows (40
  probes x 366 days with five random-effect margins each). Its per-unit
  natural-gradient step used to multiply such a unit's precision by up to 1e8
  in one visit and then throw `PosDefException`. Each unit in the batch is now
  evaluated at an antithetic pair of draws, `m ± U⁻ᵀε`. Their half difference
  gives the curvature estimate, which cancels the gradient at the mean. The
  step is shortened per unit so that the unit's precision changes at most
  2.5-fold along any direction and its mean moves at most one sd of its factor
  per visit. The precision update is computed in square-root form, so it is
  positive definite in floating point too. An iteration evaluates each batch
  unit twice, so it costs about twice as much as before. Snag
  `brm-svi-fit-posd-0fcb769c`, todo `15id1zs`.

## 0.2.28

- A flat grouped response (`y ~ Normal(m, sigma)` over interleaved rows, no
  `@plate`) is now evaluated one subject at a time: each row is an RK cell of
  its row's subject. `brm_nlme_loglikelihood[_and_gradient]`,
  `brm_nlme_unit_problem`, `brm_two_stage_fit` and `brm_svi_fit` accept such
  models instead of refusing them. When RK computes the row's predictor at
  that row only (a population-PK model, a regression), one subject's work
  does not grow with the number of subjects. Online SAEM keeps the batched
  population gradient for them, since one call per row costs more than a
  lockstep sweep.
- `brm_svi_fit` fits flat-row grouped and nested models. It takes
  `subject_group` and `nested_groups`, as `brm_nlme_model` does; a nested
  model's units are evaluated through one compact cell gradient set per
  distinct block length, and each unit's variational factor has its block's
  own length:
  `diagnostics.q.M`, `.Λ` and `.U` are vectors with one entry per subject
  (they were a `d × N` matrix and `d × d × N` arrays). An iteration costs its
  batch's cells: a cell that computes whole-cohort values (the regional
  panel's totals and completed-covariate rows) costs about one full gradient.

## 0.2.27

- A forecast continues an `rw(time)` or `cdar(step; by = g, cor = C)` walk.
  `reprocess(fit, rows)` (the default `freeze_constants = true`) keeps the
  fitted grid and appends the steps after its last one; their innovations are
  drawn in generated quantities from their `N(0, 1)` prior, so every level
  continues from its fitted state (`delta[g, new] = rho * delta[g, last] +
  sigma * sqrt(1 - rho^2) * (L * eta_new)[g]`). The forecast program keeps the
  fit's unconstrained coordinates, so the fit's draws feed it and `:predict`
  runs. Before, the replay grew the walk's innovations as parameters, and
  `:predict` with the fit's draws failed with a `DimensionMismatch`. A replay
  on the fitted grid is the fit's own program. **Breaking:** a prediction
  replay with a step inside the fitted grid that the fit did not have, or
  before it, is refused (it would have shifted every later innovation onto the
  wrong step); `freeze_constants = false` still samples every step of the
  replay's grid. Every `rw`/`cdar` program gains the data
  `<term>_n_new_steps` (0 in a fit) and an empty `<term>_z_new` /
  `<term>_eta_new` (descriptor component `:auxiliary`), so its Stan source and
  descriptor id change; the density and coordinates do not.
- `adaptive_centering_problem` adapts `cdar` walk cells beside ordinary, R2D2,
  HSGP and car cells in one wrapper, and in the `ranef_order = :learned`
  wrapper, after the car cells. A model without walk cells keeps its pair
  order and checkpoints, and a walk-only model keeps its own plan. Before,
  such a model was refused ("cannot yet mix cdar walk cells"). Walk cells
  beside exact totals or S2Z contrasts are still refused.
- The `cdar` docstrings call `sigma` what the program makes it, the marginal
  (stationary) SD of each deviation; its innovation SD is
  `sigma * sqrt(1 - rho^2)`. Snag `per-group-statio-059ae6f6`.

## 0.2.26

- A `@plate` cell reads a linear predictor none of whose random-effect
  groupings lists one label per row, such as a class over the cells
  (`log(cv) ~ 1 + (1 | cell_class)` read as `cv[k]`), on SBBRMI and RKBRMI.
  Each grouping is an ordinary random effect of its row, as in brms, and the
  predictor rides on the cells the plate's other inputs define (a predictor
  whose grouping labels the rows, a `groups(g => labels)` binding's labels, or
  the pre-grouped data columns), or on its own rows when nothing else defines
  them. Its data vectors, grouping columns included, need one entry per cell.
  Before, beside a binding SBBRMI refused it ("... already labels them by
  `cell_class`"), and elsewhere both backends refused it as long-format data
  ("needs pre-grouped per-subject data"). A binding's cell label vector with
  repeated labels is still refused. Snag `cell-lp-with-a-c-3d6db673`.

## 0.2.25

- `brm_start` sets the scales and correlation of a stratified
  `(… | ID | gr(g, by = b))` block, each stratum (level of `b`) having its own.
  `ID = (; sd)`, `(; cor)` and `ID = Σ` set every stratum; `stratum => value`
  pairs set the listed strata, `sd = ["small" => 0.4, "large" => [0.6, 0.2]]`,
  `cor = ["small" => C]`, `ID = ["small" => Σ]`, and unlisted strata keep their
  values at `base`. Pairs on a block that is not stratified are refused.
  Before, SBBRMI refused `sd` (`` `|ID|` expects Stan value `b_…_log_scale`,
  which the compiled model does not declare ``) and looked up `cor` under the
  block-wide factor name, which a stratified block does not emit; `effects`
  alone already worked. RKBRMI now takes a stratified block's `sd`, `cor` and
  `effects` as SBBRMI does instead of refusing the block, so
  `brm_nlme_start(m, inits)` takes them too for a one-margin block
  (`brm_nlme_view` does not read a multi-margin stratified block yet, todo
  `1srx3ud`). Snag `stratified-block-73231199`.

## 0.2.24

- A gather `x[idx]` lies on the rows of its index. A predictor on its own rows
  that reads another part's values through a model-scope gather
  (`c_b = c[pos_b]`, `z_b = z[pos_b]`) is sized by the gather's index, so a
  `@plate` cell may read it through a grouping over those rows
  (`rows = groups(g_b => cells)`, `log_b[rows[k]]`). Before, when the gathered
  value read data columns of its own rows (`z = w * (v - log(y)) / 2` with
  `mi(y)`), both SBBRMI and RKBRMI took those columns for the predictor's rows
  and refused the grouping (`` `log_b` has 6 entries (data vector `w`) but
  `g_b` has 5 ``). RKBRMI also takes a gathered model value (`c[pos_b]` of an
  `mi()` completion) as a predictor term, as SBBRMI did, instead of refusing it
  as not a data column. Snag `rk-backend-predi-9a0c2121`.

## 0.2.23

- `brm_coordinate_transport` pairs grouped HSGP terms, `hsgp(x...; by = g)`,
  so an RKBRMI fit's draws (`brm_svi_fit`, `brm_two_stage_fit`, a cut fit, …)
  reach SBBRMI's constrained names and every reader that takes them:
  `brm_multigroup_coherence` and `brm_multigroup_field` for
  `cor = multigroup((x, y); k)`, `brm_ranef_draws`, `population_draws`. Paired:
  the length scale and amplitude, or a hyper-predicted one's intercept,
  level-effect scale and level effects (`log(sd(hsgp(x))) ~ 1 + (1 | g)`) by
  level label; the term's own spatial scale under `multigroup((x, y); k)`; and
  the standardized weights by level label and basis function, or, under
  `cor = multigroup((x, y), …)`, by inducing point and basis function. `cor =
  Omega`, `multigroup(Omega, tau)` and declared `lambda` or inducing positions
  pair as their own declarations. `brm_nlme_joint_map` and the Stan subject-draw
  targets still refuse such a model. Snag `read-brm-svi-fit-bb7a9be0`.
- `brm_coordinate_transport` pairs a declared vector, `x[1:n] .~ law` (for
  example the free inducing positions of `multigroup((x, y), lambda; method =
  :inducing, inducing = (px, py))`), element by element, each element under its
  scalar law as an authored scalar pairs. A vector family the transport does
  not pair is refused as `:unsupported_coverage` instead of failing with a
  `KeyError`.

## 0.2.22

- A model description covers the comparisons `==`, `!=`, `<`, `<=`, `>` and
  `>=`, so a model deriving numeric indicators the documented way
  (`bottle = 1.0 * (vessel == 2) + 1.0 * (vessel == 3)`) is complete instead of
  listing each comparison as a coverage gap. A comparison used as a value is
  written as the indicator `\mathbf 1\{\mathrm{vessel} = 2\}`, the notation
  `factor` dummies already use, instead of `\mathrm{==}\left(\mathrm{vessel}, 2\right)`.
  `|` and `&` joining comparisons (in a `@plate` cell) are one indicator of
  their logical or/and, `\mathbf 1\{v_{i} = 2 \lor v_{i} = 3\}`, a nested
  connective in parentheses. A formula's `(1 | g)` and `a & b` are unchanged.
  Snag `description-cove-e12e9953`.

## 0.2.21

- **Breaking:** on RKBRMI, a predictor written with a link is its
  response-scale value under the name you wrote, and its linear predictor
  carries the link: `log(mu) ~ 1 + x + (1 | g)` is `log_mu = …` and
  `mu = exp.(log_mu)`, `logit(p) ~ …` is `p = logistic.(logit_p)`. So
  `want=(:mu,)` and `rk_draw_values(rk, draws, (:mu,))` answer μ, and
  `log_mu` answers log μ. Before, a model of formulas and their responses
  (RK's structural route) named the linear predictor `mu`, which then held
  log μ; a model with computed values already followed this rule. To read the
  linear predictor, ask for `log_mu` (`logit_p`, …). The responses still read
  it through their link-scale families (`NegativeBinomial2Log.(log_mu, …)`),
  so densities and gradients are unchanged (snag `rk-node-of-a-lin-d8216a61`),
  with one exception on registered Enzyme. For a lone linked intercept with
  a literal `BetaBinomial2` precision
  (`logit(mu) ~ 1; c ~ BetaBinomial2(n, mu, 5.0)`), the plain Enzyme Reverse
  gradient now raises `EnzymeMutabilityException` ("cannot be proven
  readonly") on Enzyme 0.13.216. Enzyme's dogfood line from `9d920e1f`
  differentiates it exactly (Enzyme snag `readonly-proof-h-bcbe23d3`).

## 0.2.20

- Long-format data with a plate over each subject's rows,
  `@plate for p in groups(subject; order = t)`, have per-subject cells:
  `brm_nlme_loglikelihood`, `brm_nlme_loglikelihood_and_gradient` and
  `brm_svi_fit` evaluate one subject's cell. The cell's densities (its `Normal`
  entries, or an `AR1Normal` series) belong to the subject owning the cell's
  rows. Before, per-subject cells needed one row of the grouping column per
  subject, and an `AR1Normal` cell was refused as having densities that cannot
  be attributed to subjects. An `AR1Normal` observation is not an independent
  Normal one, so such a model declares no Gaussian observations.
- The refusal for coordinates without an NLME role names the declarations that
  own them instead of pointing at `mm(...)`. Snag `brm-svi-fit-refu-be4efadf`.

## 0.2.19

- On RKBRMI, the graph partitions each ragged join of an `@plate for` cell
  once per evaluation, however many columns the cell reads through it. With
  `ops = groups(op_subject => subject)`, a cell reading `op_time[ops[i]]`,
  `op_amount[ops[i]]` and `log_F[ops[i]]` computed the `op_subject` partition
  three times on every call of a reader whose design is a runtime input; the
  further columns now read the first one's partition through
  `brm_ragged_rows_shared`, which checks only their own length against the
  grouping column, with the same message. A join is its grouping, cell label
  and `order=` columns. Each partition also allocates only its cells, with no
  per-row position buffer beside them. Values, densities and refusals are
  unchanged. A program printed by this version names `brm_ragged_rows_shared`,
  so replaying it needs 0.2.19 or later. Snag `ragged-partition-4ef21f93`.

## 0.2.18

- `adaptive_centering_problem` adapts a grouped HSGP whose levels are
  correlated, `hsgp(x...; by=g, cor=Omega)` with `Omega ~ LKJCholesky(K, eta)`
  or a known correlation matrix, and one whose levels have their own scales
  through a hyper-predictor (`log(sd(hsgp(x))) ~ 1 + (1 | g)`, or the length
  scale's), with or without `cor=`. Before, the first was refused at
  construction and the second failed resolving the term's marginal SD. Basis
  function `b`'s level weights `diag(s_b) * L * z_b` adapt like one group of a
  correlated random-effect block: each level is centred on its conditional mean
  given the levels before it (the grouping's level order), so `c=1` is the
  level weight. The cells follow the HSGP cells in the pair order and
  `adaptive_centering_controls` reports them as `:hsgp` rows with `group`,
  `level` and `rank`. A multigroup `cor=` is still refused. Models without such
  a term keep their pair order and arithmetic. Snag `adaptive-centeri-11062238`.

## 0.2.17

- A printed model (`show(brmi)`, `string(structure_of(m))`,
  `string(priors_of(m))`, `show(builder)`) writes indexing, field access,
  prefix operators, ranges and aliased families as the model wrote them:
  `field[sk]` (was `getindex(field, sk)`), `state.a` (was
  `getproperty(state, :(:a))`), `-k * t` (was `-(k) * t`), `x[1:3]` (was
  `x[Colon(1, 3)]`), `m = a - b` (was `m = (a - b)`), and `LocationScale(...)`
  and `Categorical(...)`, which BRM exports (were Julia's names
  `AffineDistribution(...)` and `DiscreteNonParametric(...)`, which a model
  using only BRM cannot re-parse).
  `(-k)^2` and `(-1)^c` keep their parentheses on the built model and the
  builder; the builder printed `-k ^ 2`, which re-parses as `-(k ^ 2)`. A
  statement whose side is a random-effect bar keeps its parentheses on the
  builder too, `y ~ (1 | g)` (it printed `y ~ 1 | g`). Snag
  `brm-print-author-33b0b41d`.

## 0.2.16

- The model reads every indexed output of an `@plate for` cell by its name, not
  only the cell's last: `tot[s] = sum(x)` followed by `other[s] = x` in one
  cell, then `obs ~ Normal(tot[idx], sigma)` outside it, written after or
  before the cell. Another `@plate for` cell may read it as `tot[i]` too. Before,
  only `other` was a model value, and reading `tot` failed with `Could not find
  tot` (SBBRMI) or as an unknown factor coefficient (RKBRMI). Its per-cell
  reads (`tot[g]`, `sum(tot)`, `sum(tot[idx])`, 0.2.6) mean what they mean for
  the collected output. SBBRMI reads the value StanBlocks already stores for
  the cell, with no second copy; RKBRMI reads its graph value. Density and
  gradient equal the cell with `tot[s]` moved last. A name two cells write is
  refused when the model reads it. Snag `model-scope-read-3649ae8a`.

## 0.2.15

- A term's own parameter takes a known value with `=` on its address, as a
  random-effect SD does: `length_scale(<lp|:>, hsgp(x...)) = value` and
  `sd(<lp|:>, hsgp(x...)) = value` (likewise for `gp`), the smoothing SDs of
  `s(x)` and `t2(x, z)` (a `t2` may hold some of its three), `ar(...)` of
  `dar`/`cdar`, `sd(...)` of `dar`/`rw`/`cdar`/`car` and `mixing(...)` of
  `car`. An anisotropic length scale takes one value per axis (or one for all).
  The value is a numeric constant, a data field or arithmetic over them. SBBRMI
  binds it as the term's own data `<term>_known_<slot>`, a fitted constant
  `reprocess` keeps, and RKBRMI as a constant (`gp`, `hsgp`, `dar`, `car`). A
  known length scale replaces the HSGP's default declaration like an explicit
  prior, so its validity floor does not bind it. `brm_term_coordinates`
  errors for a known role, naming the value; transport, adaptive HSGP
  centering and the HSGP diagnostics read it as a constant, and
  `brm_description` lists it as `Dirac(value)`. Refused with their reasons:
  `latent(...) = value`, `simplex(...) = value`, and a known value beside a
  hyper-predictor of the same parameter. Snag `hsgp-length-scal-200dd332`.
- The parse error for `=` on an address that takes no known value now names
  the term-parameter addresses that do.

## 0.2.14

- On RKBRMI, a plate output whose cells hold vectors is one vector per cell at
  model scope, as on SBBRMI: after `cnt[s] = v` in `@plate for s in …`,
  `sum(cnt)` adds the cells elementwise, `sum(cnt[idx])` adds the cells `idx`,
  `cnt[g]` is cell `g`'s vector and `sel = cnt[idx]` is those cells, for
  vectors of fixed width and for vectors whose length follows the cell's data.
  From 0.2.6 to 0.2.13 these reads raised an error on RKBRMI. Needs
  ReactiveKernels 0.1.2 and ReactiveKernelsPPL 0.1.1 (`e32d442c`,
  ReactiveKernels snag `summing-a-plate-eba2648d`); the test environment pins
  it. With an older ReactiveKernels the reads raise the error as before.

## 0.2.13

- `reprocess(sb, rows; resample_groups = [:g])` on SBBRMI draws new levels of
  a grouped HSGP, `hsgp(x...; by = g)`, as it draws new levels of a
  `(… | g)` block: each new level's standardized basis weights are drawn
  fresh, `z ~ Normal(0, 1)`, in generated quantities and scaled by the
  spectral density at the draw's fitted length scale and amplitude. With a
  per-level hyper-predictor, `log(sd(hsgp(x))) ~ 1 + (1 | g)` or
  `log(length_scale(hsgp(x))) ~ 1 + (1 | g)`, the new level's hyper effect is
  drawn from the fitted SD with it. The same holds for `cv_groups = [:g]`.
  Before, both refused (`no cv-contagious size local`). Still refused, by
  name: a grouped HSGP with `cor = …` (the correlation covers the fitted
  levels only, todo `0lopk07`) and one with `orthogonal_to = …` (each level's
  projection is fitted on that level's training rows, which a new level does
  not have). Snag `resample-groups-6a6b2d58`.
- On RKBRMI, `reprocess(rk, rows; resample_groups = [:g])`,
  `rk_design_binder(rk; resample_groups = [:g])` and
  `RKBRMI(brmi, state; resample_groups = [:g])` refuse a `g` that groups a
  grouped HSGP, naming it: the term has no input for new levels' weights yet
  (todo `10h1myt`). Before, a design that reused the fitted labels replayed
  with the fitted levels' weights as the new levels', and any other design
  failed as a changed program.

## 0.2.12

- `population_draws(…; groups = :g)` and `population_model(sb; groups = :g)`
  treat a grouped HSGP, `hsgp(x...; by = g)`, as one of `g`'s effects: its
  standardized basis weights are zeroed, so the term is exactly zero on every
  row, and so are a per-level hyper-predictor's effects
  (`log(sd(hsgp(x))) ~ 1 + (1 | g)`), so every level sits at the population
  amplitude or length scale. The two routes agree. A grouping whose only
  effects are a grouped HSGP is accepted. The term is selected by its `by=`
  column's own name. Before, both routes kept every group's fitted curve: a
  `groups = :g` that also zeroed a `(1 | g)` block returned a prediction that
  still held each group's HSGP deviation, and a grouping with no other block
  was refused. `ranef_blocks` is unchanged and does not list these
  coordinates.
- `transport_draws` copies a grouped HSGP's weights and per-level hyper
  effects by level label. A level the source lacks (or every level of a
  `resample`d grouping) is refused, naming the
  `reprocess(…; resample_groups = [:g])` route. A target that draws them in
  generated quantities is skipped. Weights on a `cor = multigroup((x, y), …)`
  inducing grid copy row for row. Before, the weights were copied by
  coordinate name: on a target with a level dropped, later levels silently
  read earlier levels' weights.
- An unqualified hyper-predictor, `log(sd(hsgp(t))) ~ …`, on an `hsgp(t)` that
  two predictors carry is still refused as ambiguous. The message now spells
  the statement with its `log(…)` and names each predictor-qualified fix,
  `log(sd(<predictor>, hsgp(t))) ~ …`.
- Snag `brm-use-silent-o-0039d181`, decision `0wlazld`.

## 0.2.11

- `brm_nlme_view` reads a grouped HSGP, `hsgp(x; by = g)`, on RKBRMI. When
  the levels of `g` are the subjects (the subject grouping, a column whose
  labels are subjects such as `ecg_subject`, or a grouping named in
  `nested_groups`), each level's standardized basis weights are unit draws
  of its subject, in one family of size k whose covariance is fixed at the
  identity (`NLMEEstimation.fixed_omega`); the `(1 | g)` effects of a
  hyper-predicted amplitude or length scale are an estimated family of size
  1; the length scale, amplitude and hyper intercepts stay `:shared_latent`.
  Any other grouped HSGP, and one whose `cor=` correlates its levels, is
  `:shared_latent` as a whole, with the declared `Omega ~ LKJCholesky(...)`
  its `cor=` reads. Before, every grouped HSGP left its
  coordinates outside the semantic inventory, and `brm_nlme_view` refused the
  model. Labels that name some subjects and not others are refused. A subject
  without a level of `g` makes the layout ragged. Snag
  `nlme-unit-role-f-c29361b0`.

## 0.2.10

- `cnt[idx]` of a plate output whose cells hold vectors of varying length is
  the cells `idx` on SBBRMI too, as a value of its own (`sel = cnt[k]`, then
  `sel[1]`), and SBBRMI's `sum(cnt)` / `sum(cnt[idx])` of such an output use
  StanBlocks' own `sum` and `rv[idx]` of a ragged value instead of a BRM helper.
  Needs StanBlocks 0.2.2 (`ffbaff09`, snag `selecting-severa-027a8191`); the
  test environment pins it. Correction to 0.2.6: for these cells SBBRMI's
  earlier `sum(cnt)` did not compute the total of every entry; it failed to
  compile. Snag `sum-a-vector-pla-812e04d2`.

## 0.2.9

- A grouped HSGP, `hsgp(x...; by = g)` on SBBRMI, costs one rows x basis pass
  per gradient, within a factor of two of one ungrouped field over the same
  rows. Its spectral weights now scale the levels x basis weights, and each
  level's basis rows are multiplied by its level's weights, the rows sorted by
  level once in transformed data. Before, the program held a rows x basis
  value: the shared-scale form stored `<term>_PHI_scaled` and the per-level
  form (`log(sd(hsgp(…))) ~ 1 + (1 | g)`) built `<term>_S` in one full pass
  per level. The model is unchanged (equal log density and gradient). At 1,622
  zones x 3 levels with `k = 15` per axis, a gradient takes 0.4 ms instead of
  123 ms (per-level amplitude, `cor = Omega`) or 17 ms (shared scale), and a
  draw carries about 23,000 values instead of 1.1 million.
- Draws of such a term no longer carry the intermediates `<term>_PHI_scaled`
  and `<term>_S`; `<term>_beta_scaled` (levels x basis) holds the scaled
  weights, and row `i` of the field is the dot product of basis row `i` with
  row `g[i]` of it. The exported Stan helper `brm_hsgp_by_hyper_S` is replaced
  by `brm_hsgp_by_hyper_sqrt_spd` (the levels x basis spectral weights) and
  `brm_hsgp_by_level_rows`. Snag `grouped-correlat-68e4a63c`.

## 0.2.8

- `brm_start` places a random-effect block's groups at their own values:
  `ID = (; sd, cor, effects = ["a" => 1.2, "b" => -0.4, …])`, one number per
  level for a one-margin `|ID|` block, one vector in `ranefcoefnames` order
  otherwise. An effect is what the level adds to each margin's linear
  predictor on its link scale, the `effects` that `brm_ranef_draws` reads
  back. Before, every group started at the population value: natural effects
  at 0, and every exact total at `A β`. On SBBRMI this covers non-centred,
  centred (`centered_groups`), Student-t (at the base mixing weights),
  stratified (`gr(g, by = b)`) and level-correlated (`gr(g; cor = …)`)
  blocks, an `|ID|` an exact-total block absorbed (each total is then
  `A (W[g, :] .* β)` plus its group's effect) and an `|ID|` a sum-to-zero
  block claimed (its deviations from their mean and `theta = β + B μ`; every
  level must be given). On RKBRMI it covers the plain blocks it addresses.
  Unlisted levels keep their coordinates at `base` (standardized draws, or
  effects in a centred block). Snag `odlplaus-ap-2-st-537291fd`.
## 0.2.7

- `structure_of(builder)` and `priors_of(builder)` split a data-free
  `BRMBuilder` (from `@brm begin … end` or `brm(source; mod)`) into the builder
  of its model statements and the builder of its prior-address statements, as
  they already split a `BRMI`. `string(structure_of(builder))` and
  `string(priors_of(builder))` are the model's structure and prior cards, with
  no data bound; `optional(...)` priors stay as written. Snag
  `show-a-model-s-s-596039d4`.
- A builder prints an elementwise prior or a declared vector parameter as
  written, `effect(mu, x) .~ Normal.(0, 1)`, where it printed
  `.~(effect(mu, x), Normal.(0, 1))`.

## 0.2.6

- A plate output read at model scope is one value per cell, as the Julia loop
  leaves it: after `cnt[s] = v` in `@plate for s in …`, `cnt[g]` is cell `g`'s
  value, `sum(cnt)` adds the cells (elementwise when each holds a vector) and
  `sum(cnt[idx])` adds the cells `idx`. SBBRMI computes these for every cell
  shape; RKBRMI for cells that hold a number (vector cells wait on
  ReactiveKernels snag `summing-a-plate-eba2648d`). Snag
  `sum-a-vector-pla-812e04d2`, decision `09e0gc5`.
  **Breaking:** on SBBRMI, `sum(cnt)` of an output whose cells hold vectors of
  fixed width was the total of every entry and is now the elementwise sum of
  the cells; write `sum(sum(cnt))` for the total. (For cells whose length
  follows their data it did not compile; corrected in 0.2.10.) On RKBRMI,
  `cnt[g]` and `sum(cnt)` of such an output read the cells laid end to end and
  now raise an error until that ReactiveKernels snag lands.

## 0.2.5

- A predictor read through a `@plate` grouping may hold a term whose keyword
  data is not on the predictor's rows: the levels' coordinates in
  `hsgp(t; by = g, cor = multigroup((x, y)))`, a known level correlation
  `cor = C` on `hsgp` or `gr`, and `car(area; edges = (from, to))`. For
  example `st ~ 0 + hsgp(day; by = probe, cor = multigroup((probe_x, probe_y)))`
  read as `st[p]` in `@plate for p in groups(probe; order = day)`. Before,
  RKBRMI and SBBRMI counted those fields as rows of the predictor and refused
  it with "`st` read through `probe`: `st` has 40 entries (data vector
  `probe_x`) …", also when the term sat in a predictor read that way beside
  its other terms. Snag `rkbrmi-on-a-mode-0bd31c04`.

## 0.2.4

- A positive `@plate` cell latent can be given its total:
  `T::shares(total) ~ Law(args…)`, such as
  `T::shares(land[s]) ~ LogNormal(location, scale)`. `T` sums to `total` in
  every draw, with `Law` as its law on that surface. It is the Stan program
  `simplex[K] w; T = total * w; T ~ law(…)`: the sampler moves the K − 1
  coordinates of Stan's simplex per cell, and BRM adds
  `sum(log.(T)) - log(total)`. For a total from data this is the exact
  conditional given the total, which needs no observation of it. The total
  may also be a model value. Positive families only (`LogNormal`, `Gamma`,
  …); SBBRMI and RKBRMI; `brm_start` takes `T`'s values, which must sum to
  each cell's total. A draw of such a latent (prior predictive) is not built.
  Snag `exact-sum-constr-7ba55c76`.

## 0.2.3

- **Breaking:** `SBBRMI`, `generative_plan` and `brm_descriptor` default to
  `total_groups = ()`, the standard (non-centred) parametrization for every
  random-effect block. Before, the default was `:auto`, which emitted exact
  total coefficients for every eligible block. To keep exact totals, pass
  `total_groups = :auto` (or the groups to integrate). For blocks the data
  identify strongly or unevenly, S2Z (`s2z_groups`, `s2z_rho`) with WarmupHMC
  adaptive centring (`adaptive_centering_problem`) is the recommended
  alternative. Exact totals stay available and are not deprecated.
- `s2z_rho` takes one weight vector per predictor,
  `s2z_rho = [:mu => [1, 1, 1, 0], :sigma => 1]`, for predictors with
  different numbers of coefficients on one S2Z grouping. The pairs must name
  every sampled S2Z predictor and no other.
- Under WarmupHMC adaptive centring, each exact total column is centred on its
  own location, the least-squares fit of the column's totals on the
  group-level predictors of the coefficients it absorbs (the mean of the totals
  for an intercept-only block), so there is no location to choose. Other cells
  in a model with totals choose their centring weights when a warm-up window
  restarts. Snag `exact-totals-fun-f116d6b2`.

## 0.2.2

- `brm_start(sb, stan_model, inits)` takes a value for every level of an
  intercept-free predictor's cell-means block (`0 + g`, `0 + factor(g)`),
  `lp = (; g = ["a" => 1.1, "b" => 0.4, ...])`, on the link scale of a linked
  predictor, as `brm_start(rk, inits)` already did. Before, the Stan method
  found no coefficient for any level of such a block. Snag
  `brm-start-sees-n-5b61087d`.
