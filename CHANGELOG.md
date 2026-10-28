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
