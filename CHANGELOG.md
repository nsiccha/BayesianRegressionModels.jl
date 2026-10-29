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

## 0.2.58

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
