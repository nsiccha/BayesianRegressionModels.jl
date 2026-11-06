# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env pins external source dependencies plus the unregistered
# BRM root itself. Each pinned package is materialized at a specific GitHub
# COMMIT in this host's store (`test_package_store()`, outside the checkout);
# there is NO dependence on any shared `~/github/nsiccha/<pkg>` checkout. A
# full-SHA revision is branch-independent, so it does not matter that several of
# these live on a `dev`/`devibe` branch rather than `main`; the commit only has to
# be pushed to GitHub, except fleet revisions and unpublished review candidates,
# which are materialized from their host mirrors (`MIRRORS`).
#
# The store is what lets a fresh worktree or job reuse compiled images: Julia
# rejects an image built from another source path, so a dependency under the
# checkout rebuilt with its whole reverse closure every time. Only BRM itself and
# its extensions rebuild now, whose source really is this checkout's. Develop the
# store's real path: an image built through a symlink in the checkout records that
# path and dies with the checkout. Its checkouts are read-only and never rewritten
# (a fresh mtime rejects an image too), and evicted after two unused weeks.
#
# Enzyme is the ecosystem's `dogfood` line (user pick on decision
# `BayesianRegressionModels/decisions/2026-10-28T15-34-11-008-0n0kgfh`): reviewed
# repairs no registered release carries, which BRM's RK cell gradients need. Its
# commits are materialized from the KB hosts' Enzyme mirror (`MIRRORS`) when
# available, and it runs against its own libEnzyme build
# (`deps/dogfood_libenzyme.toml`). `apply_dogfood_libenzyme()` checks the host
# and Julia version and points `Enzyme_jll` at the matching build
# through the KB resolver's own step, and refuses where there is none: the
# stock library gives wrong Reverse gradients the build fixes (skill
# `enzyme-use`).
#
# ReactiveKernels contributes four developed paths from ONE pinned checkout
# (the monorepo root plus the nested `ReactiveKernelsDistributionKernels`,
# `ReactiveKernelsPPL`, and `ReactiveKernelsPPLExamples` packages, which have
# no standalone repos).
#
# All developed paths enter ONE `Pkg.develop` call on EVERY Julia version we run.
# On 1.11+ the `[sources]` blocks in `test/Project.toml` (which mirror these
# revisions) would also resolve them; on **1.10, which is what this suite runs
# on, `[sources]` is IGNORED**, so a bare `Pkg.resolve()` fails with
#
#     ERROR: expected package `Treebars [e1e568c4]` to be registered
#
# an error that looks like a missing registry rather than a version-gated
# feature. This script is the version-independent answer.
#
# Bumping a pin is a deliberate one-line edit here (+ the matching `[sources]`
# rev in test/Project.toml), reviewed like any other change — not a silent
# consequence of whatever a shared checkout drifted to.
#
# It is idempotent — re-running it validates/reuses the exact cached checkouts —
# so it is safe to run before any suite when unsure of the env's state.
using Pkg

include("dependency_floors.jl")

const REPO = dirname(@__DIR__)
const TESTENV = @__DIR__

# name => (github url, pinned commit)   — comment records the branch the commit
# is on, for humans; the pin itself is the full SHA and needs no branch.
const PINS = [
    ("ReparametrizableDistributions", "https://github.com/nsiccha/ReparametrizableDistributions.jl.git", "8f7d382df700d3780ed31d093bc4983d3734a6c6"),  # 0.3.0: WarmupHMC dependency; BRM no longer depends on this directly
    ("PowerScaling", "https://github.com/nsiccha/PowerScaling.jl.git", "8d27c83f6928781a39c99c31fd3d528b50ef2dad"),  # 0.1.0: generic power-scaling sensitivity on PSIS.jl
    ("HTMX", "https://github.com/nsiccha/HTMX.jl.git", "0b8dc2f94e879bf31237bd28e751cf2ebc8d95f6"),
    ("HTMXObjects", "https://github.com/nsiccha/HTMXObjects.jl.git", "ee46e8d6bd381c8b066f3b26afa7ea3336e23146"), # 0.1.65: readable plain-text views; includes 0.1.62 documentation projection
    ("DynamicObjects", "https://github.com/nsiccha/DynamicObjects.jl.git", "c84047be45cd2bba8a1d02660af9408d4cdf58f4"),
    ("TreeArrays", "https://github.com/nsiccha/TreeArrays.jl.git", "53cfd1b17ee4d1b726dcecfe151458ae41bd20a8"),
    ("AlgebraOfVega", "https://github.com/nsiccha/AlgebraOfVega.jl.git", "73695d5d563074bfd9ab1f98debf2ebe38d7bd00"),
    # Fleet-only pivoted KLU selected inversion; registered 0.2.2 lacks it.
    ("SelectedInversion", "https://github.com/timweiland/SelectedInversion.jl.git", "8b1ee1077cea94c1aff746e182681ae39ce91526"),
    # One-sided LogNormal moments and tail-stable truncation mass, plus the
    # public support correction: reject invalid DirichletMultinomial
    # counts/totals before gamma evaluation. An RK/RKPPL version floor alone
    # cannot select this Distributions revision (snag dirichlet-multin-457e95cf).
    ("Distributions",     "https://github.com/nsiccha/Distributions.jl.git",     "63b968eaae73193f73d784529dc291253d62ee6f"),  # fix/truncated-lognormal-moments (also carries support correction)
    # 359a4fae (StanBlocks 0.2.8) adds the native explicit-total
    # Dirichlet-multinomial density, pointwise likelihood and predictive RNG.
    # StanBlocks 0.2.10 preserves infinite bound data through BridgeStan JSON.
    # 0.2.12 registers stable hypot, including Base.hypot function values.
    # 0.2.14 adds stable lower-truncated LogNormal moments and log-CDF tails.
    # 0.2.18 implements segment in @juliacompat for the native GMRF density.
    # 0.2.19 lowers indexed /= in the same native sparse solve and prior RNG.
    # 0.2.20 supports explicit local declarations shadowing outer cell inputs.
    # 0.2.21 predicts upper-only interval observations with the response omitted.
    # 0.2.24 predicts censored responses bound by the omitted response itself.
    # StanBlocks 0.2.25 preserves component-call keyword labels inside plates.
    # StanBlocks 0.2.26 preserves EachCol/EachRow at ordinary helper boundaries.
    # StanBlocks 0.2.30 (904e839a) makes a number passed to `stan_call` a data
    # literal, so `brm_von_mises`'s argument-dependent `autokwargs` support
    # `[mu - pi, mu + pi]` has a qualifier `stan_descriptor` reads (snag
    # autokwargs-suppo-412811f6).
    # StanBlocks 0.2.31 collects tuple-valued plate cells as array[N] tuple(...),
    # so a component called in a cell may return named values (b28164c).
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "d9396ae6c66c7cb44d641989573b7008154de6c5"),  # devibe: `restricted(family, args...; lower, upper)` declares the bounds and keeps the plain kernel (566bbc0f, StanBlocks 0.2.7, snag in-slic-declare-996328e0), which SBBRMI emits for every prior bound; a `truncated` draw succeeds whatever mass its window holds: after 100 misses it inverts the base CDF on the window (9a4fb3a6, StanBlocks 0.2.6, snag truncated-rng-re-bd555cae), which BRM's omitted truncated responses and `_gen` draws call; `rv[idx]` selects groups of a ragged value and `sum(rv)` adds its groups elementwise (ffbaff09, StanBlocks 0.2.2, snag selecting-severa-027a8191), which BRM's model-scope reads of a ragged plate output pass through; Stan's `sum_to_zero_constrain`/`_unconstrain`/`_jacobian` trace (8e57a83, snag sum-to-zero-cons-90a72024) and `head`/`tail`/`reverse` (359844e, snag stan-vector-buil-5792b9e5), which `T::shares(total)` cell latents call, both in f5e4650; a `Vector{Matrix}` data value ingests as a `RaggedMatrix` and `X[g]` reads the g-th matrix in a plate cell (c9504e99, snag ragged-matrix-da-163255e9); a plate cell may bind a matrix local of fixed or per-cell size (a57951eb, snag binding-a-per-ce-e16efa8d); a plate's Stan names follow from each value's own name, without the `__pl_mem_1` suffix (1d93a65), and `stan_interface` / `stan_data_json` describe the program (dc3f713), both in fd91d0c; tuple descriptor entries carry their components in `fields` (785edf4); ragged sizes are reported derived (9118764); a values-free `maybecv(:y)` marks an observation with no values as predicted given a fit (c4671eb, snag held-out-observa-58a2eb13); shared scan arguments read once per step, linear in series length (b97fc5a6); Stan's vectorised builtins (`fmax`, `fmin`, …) evaluate elementwise in the Julia layer, so a per-axis `fmax(0.0, lo)` bound reads one value per coordinate (c7293b88); descriptor inputs/outputs carry `constraint_values`, their bounds evaluated against the bound data (b233afbc); a `ValueFamily` as the base family of `weighted`/`truncated`/`censored`/`interval_censored` with `lcdf=`/`lccdf=` companions (94e414bb); descriptor outputs carry their value shape as `dims`; non-finite literals render as Stan infinity builtins; `weighted` over censored/truncated/interval_censored; an exchangeable ordered prior re-draws as the sorted iid draw; observed scans and per-cell scan lengths (82a954da), also through a do-cell parameter that shadows its data name (4ee6e0e8); integer step locals in nested scans (2f5b6203); a latent or observed `@scan` inside a called `@slic` submodel (e67bc314), observing a plate cell's group (d3483f11); `interval_censored`'s interval by keyword and unknown combinator keywords refused (897557c); typed `=` in `stan_descriptor` (204e5a7)
    # Cooperative interruption used by BRM's progress hooks.
    # Treebars 0.1.5 satisfies the pinned ReactiveKernelsPPL extension floor.
    # Treebars 0.1.5: nested progress used by recoverable adaptive SMC pilots.
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "44c37e60e22bab9e8a0d3ab88bd47b575dec6bb0"),
    # 0.2.11: public covariance projection used by selective SMC carries.
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "8e1d45a78b38e3e9d82ee75312e0eb7e5203d327"),
    ("SequentialMonteCarlo", "https://github.com/awllee/SequentialMonteCarlo.jl.git", "d3a322394c7a1df07e3b5e941acc2960f4daf656"),
    # Current plain-function/array PPL surface, with data-only declaration inputs
    # retained before predictor inlining (computed membership axes; 606d76d0),
    # plus published live matrices (1t1v8zo) and callable array cells (0mr4zu5).
    # Contains db4f16aa (lower Cholesky + graph allocation + native callback AD
    # repairs), e0930dcd (a declared-array element is a scalar location summand),
    # 712c227b (vector-of-vectors nested-plate observations), 9e425eda
    # (LogDensity laws, link families, per-index arithmetic and derived responses
    # in nested plate cells), 628769da + ea596e50 (built graphs carry no bound
    # row count, so a graph built on one data set evaluates another), 683b8342
    # (one source always lowers to one program) and ed87ceef (a proven scalar
    # response location stays scalar; 0f7d2521 gives other locations their rows
    # once, without a ones vector). 83707d4d adds closure captures in plate
    # cells and scan steps (an enclosing name a do-block reads is shared whole,
    # never zipped), which BRM's emitted do-blocks use instead of Ref operands;
    # bed5d467 deprecates those Ref operands, and ecf99453 packs every hidden
    # bound operand (captures included) into one prepared-AD context, clear
    # of the 32-argument gradient cliff. 1603d87a adds
    # `prepare_sampler(...; retain=(:pointwise,))` and
    # `sampler_value_gradient_and_retained!` (pointwise densities from the
    # gradient's reverse sweep), which BRM's lockstep NLME gradient uses; it
    # contains ecf99453 and bed5d467, plus 3fc1f0c7, 54d4a7b9 and 46d5520c.
    # 5c2b80d5 keeps authored predictor names; 281933141 admits positional
    # level axes over pool definitions (`gg = vcat(g, h)`). f67bd182 adds
    # prepared HVPs, observation queries and pushforwards for Gaussian NLME.
    # 659a6e8b adds joint matrix columns and two-axis priors; cb3a0db7 reads
    # those columns in place during native reverse.
    # 7eafe9ba reads a positional level source as a whole pool, so per-group
    # tables need not have observation length. fbd0aef0 runs authored
    # `@plate for` cells over arrays per index; 2503ede6 supports undotted
    # per-index arithmetic. d702209e evaluates shared live cell outputs once
    # per subject and keeps weights based only on data independent of live
    # parameters. ddb68802 reports Gaussian observation arguments and colored
    # pushforwards for one observed array per plate index. fa23fe7f reads
    # columns of a row-gathered joint monotonic output in place during native
    # reverse.
    # 609c7a9d composes authored plate cells; a2268e47 adds compact
    # selected-coordinate cell gradients used by BRM's NLME adapter.
    # 68a77b03 binds a design without its conditioned response values to the
    # same program (snag rkppl-bind-a-des-e6cfca75), which `reprocess` uses
    # for prediction-only designs; dad59972 extends that to censored,
    # truncated and interval-censored responses (snag
    # rkppl-absent-cen-9af487f3). Both contain f13e68d4 (location-alias
    # elision renames cell parameters). ae731d3f + fe85d9f6 keep a location
    # over a computed design queryable by its authored name, e.g. `mu` over a
    # completed `mi` covariate (snag rkppl-location-a-2ec8954d); cc50dae0 is
    # the RK main containing them. It also observes a carried array that is
    # bound data in `@scan`, at top level and per plate cell (3581f5a8, snag
    # observed-scan-rk-f3f85355), and contains 60ff95f1 (sampled seeds and
    # steps in plate-cell scans). 6c7e18af adds a `@scan` in a per-cell
    # `@rkppl` submodel (c4008d62, snag per-cell-submode-cd87e071), and
    # contains ce48fc77 (`rk_ode_rk45_tol`), 9bef9c04 (per-row provider-law
    # subject cells) and fef26022 (queries skip unread plate outputs).
    # d92d6677 adds ac01d1f1d (`prepare_predictive`) and 2512d6b1c
    # (`predictive_node` / `predictive_targets` for a caller's own vectorized
    # cut), which test/rk_predictive.jl reads through `rk_bound_plan`.
    # 9af9b197 adds 276428589 (kernel construction for another package's
    # module inside a downstream package image, so a whole `RKBRMI` builds
    # in a precompile workload; snag rkppl-build-kern-4607ff90), and contains
    # a89e08381 (the `bind_data` structural cache) and 74e415976 (a computed
    # partition drives a per-cell loop). de89298a adds 7ba63ea6e
    # (`frozen_values`, `bind_data(…; frozen)`, `lower_rkppl(…; frozen)`: a
    # design binds its data-only definitions frozen by value, which
    # test/rk_fitted_state.jl replays without the training data) and
    # 28f8896d8 (a frozen design may omit an input of strings); snag
    # rkppl-bind-froze-da223711. 9f5d1e7de adds 6027d63bd (a position-batched
    # reader keeps each plate cell's result and the `reduce(vcat, cells)`
    # flatten in lane slots; snag position-batched-cc11130b). fd92ced46 adds
    # 207b0e056 (the slot survives a branch on `eltype(cells) <: Number`, the
    # form of BRM's emitted cell flatten; snag type-constant-la-902b10ce).
    # 356142a62 adds `jax_source` (666bb505f) with its host-side
    # `prepare_data` (888bdfd42; RK todo 0rouf39), which test/rk_jax.jl
    # translates RKBRMI backends through. 911df2b96 adds 39185dd24 (a call
    # on static operands, such as the GP jitter/period guards over caller
    # literals, is decided while JAX traces; snag jax-source-liter-a9e6219e).
    # 38265d407 adds 711b2034d (`reshape` with a `Colon` dimension, e.g. the
    # GP pair locations, translates; snag jax-source-resha-959acdfb).
    # 1d5b163fe adds 65bdd443f (a gather by a model-level index reads the
    # gathered value whole, so a model-scope `vcat` of the values `car` terms
    # return lowers; snag lower-a-brm-emit-2136dcc0), and contains 26d24ab55
    # (a `@plate` cell observes one vector jointly with `MvNormal`),
    # 965564845 (a dotted cell latent holds one value per element) and
    # a987c279a (LogExpFunctions 1.0.2 admitted; streamed `logsumexp` keeps
    # tie derivatives). 6ab16f510 adds cf9dde7ca (`prepare_ad_third`,
    # third-order contractions D³f[v, v, ·] for the per-unit Laplace
    # gradient; snag prepared-third-o-1e44dcc9). 8d29971f2 adds d5b22ec53 (a
    # native matrix product with a non-strided operand, such as a compact
    # cell's row of a correlated group block, runs an owned loop instead of
    # LinearAlgebra's tiled generic `mul!`, so each compact-cell gradient of
    # brm_svi_fit allocates ~2.8 KB instead of ~89 KB; snag
    # compact-cell-gra-ebf0a114). 0470e9e13 contains 8d29971f2 and adds
    # fe5c508c2 (a `@plate` cell latent's law reads a model value at the
    # index, `m[i]`; snag rkppl-cell-laten-b18e668a). b5e4760fc contains
    # 0470e9e13 and adds ae5aca6ba (prepared pullbacks and pushforwards over
    # several WANT ports in one sweep, which the structured Laplace's
    # non-affine fallback takes; snag ad-pullback-push-6f576925). bf8040f16
    # contains b5e4760fc and adds 6316fe45d (that pullback over several ports
    # differentiates under plain Enzyme Reverse when the ports read constant
    # design matrices; snag several-port-pul-e6344d44). 1c14fc131 contains
    # bf8040f16 and adds 0e96da576 (one joint `@plate` cell latent per index
    # with a caller-owned geometry, which `T::shares(total)` emits; snag
    # rkppl-per-index-5e9df9fe). e32d442c6 contains 1c14fc131 and adds
    # d35aee6cd (ReactiveKernels 0.1.2, ReactiveKernelsPPL 0.1.1: a `@plate`
    # output whose cells hold vectors is one vector per cell, so model-scope
    # `sum(cnt)`, `sum(cnt[k])` and `cnt[g]` read it; snag
    # summing-a-plate-eba2648d). b31d672fa contains e32d442c6 and is
    # ReactiveKernels 0.1.6: a plate cell computes a value it reads at several
    # rows (an `hcat` design, a column read, a level block) at those rows only
    # (snag compact-cell-sam-d85d96ae), which BRM's nested-NLME totals cells
    # need. 2a6829265 contains b31d672fa and is ReactiveKernels 0.1.7 /
    # ReactiveKernelsPPL 0.1.3: a location summand computed from `logistic`
    # binds (snag rkppl-leaf-colum-59c22941), as RKBRMI emits it from 0.2.45.
    # 06f92402a contains 2a6829265 and is ReactiveKernels 0.1.11 /
    # ReactiveKernelsPPL 0.1.12: a plate column reading a composed predictor
    # is one value, not a composition (e33632cc4, snag
    # rkppl-plate-colu-99114109), so a totals plate over a predictor offset by
    # another's gather lowers as RKBRMI emits it from 0.2.47. 3deb61ae0 contains
    # 06f92402a and is ReactiveKernels 0.1.12 / ReactiveKernelsPPL 0.1.14: a
    # compact cell reads its selected coordinates in constant time (c384a3f3e,
    # snag compact-cell-cal-58569833), and the compact cell gradient of a
    # three-margin `(1 | pk | subject)` block no longer raises Enzyme's
    # EnzymeRuntimeException there. 4f7ef46a4 contains 3deb61ae0 and is
    # ReactiveKernels 0.1.13 / ReactiveKernelsPPL 0.1.15: an array `@plate`
    # cell looks a label up on a `levels(g)` axis when `g` is a data-only
    # definition (06679067f, snag rkppl-submodel-l-00ff0f27), so RKBRMI's
    # stratified `gr(g, by = b)` block reads a grouping gathered from a level
    # table (`test/level_table_groupings.jl`). 452071f1b contains 4f7ef46a4
    # and is ReactiveKernelsPPL 0.1.16: a query preparation checks its binding
    # against a build once and a cell query reads its plate from the build
    # (1672820fb, RK snag brm-s-nlme-cell-d70abd20), so BRM's cell queries and
    # compact cell samplers no longer regenerate the whole program each (about
    # 45 s apiece on ArbeitsplatzAtlas's model). 8009f3a95 contains 452071f1b
    # and is ReactiveKernels 0.1.16 / ReactiveKernelsPPL 0.1.18: a cell query
    # computes `vcat` operands, typed response-location aliases, a source
    # gathered by several index vectors and group totals through an index
    # array at the cell's rows, each value once (RK snag
    # on-arbeitsplatza-6065a1ef), which the local `car_units` fields, grouped
    # shares and completions RKBRMI emits from 0.2.74 rely on for cells that
    # cost their own rows. 5e288b439 contains 8009f3a95 and is ReactiveKernels
    # 0.1.18: a number a cell computes that two of its values read at rows
    # evaluates instead of raising `MethodError` in `_cell_take_rows` (RK snag
    # cell-take-rows-s-e48675d0; ArbeitsplatzAtlas's `beschaeftigte_eg_kg`
    # cells). 51c8a1c4a contains 5e288b439 and is ReactiveKernels 0.1.21: a cell
    # computes `view(A, :, k)`, RKPPL's lowering of a column read `A[:, k]`, at
    # its rows, so the columns of a joint population effect (`pop[:, j]`) and
    # their design are computed at the rows a cell reads (RK snag
    # column-read-lowe-24508269). 0d3f92345 contains 51c8a1c4a and is
    # ReactiveKernels 0.1.22: a cell computes a value two statements of one
    # `@plate for` read (each a plate of its own), and a dotted value reading an
    # untyped numeric array at literal positions, at its rows (RK snag
    # form-value-compu-c1e0965c). a959bef75 contains 0d3f92345 and brings
    # ReactiveKernelsPPL 0.1.22: one cell's Hessian block and third-order
    # products over a compact selection (`prepare_compact_cell_ad` with
    # `prepare_ad_hvp` / `prepare_ad_third`, RK snag build-brm-s-spar-454be53f),
    # which the sparse Laplace's `@plate` cells read. It is ReactiveKernels
    # 0.1.25 (it contains 0d5fb817c): a cell of one value of a shared `@plate
    # for` cell computes nothing only its sibling values read (RK snag
    # sibling-only-pla-2b3df69e), BRM's compat floor from 0.2.85.
    # 5f18f4b84 (ReactiveKernels 0.1.29, ReactiveKernelsPPL 0.1.23) fixes
    # plate buffer inference from older compiled worlds, including the
    # nested SAEM population gradient (RK snag brm-canonical-44-a1653c2a).
    # 9c9434a2 (RK 0.1.32 / PPL 0.1.26) adds public sum-factor values,
    # structural footprints and compact higher derivatives for sparse cells.
    # 30d676c9 (RK 0.1.34 / PPL 0.1.28) preserves vector-valued outer
    # exemplars for a single sum, including empty reductions and selections.
    # 260d9a94 (RK 0.1.35 / RKPPL 0.1.29) preserves shape facts across
    # visible grouped kernels, so strict NLME cells read their owning group.
    # 4925e53d (RK 0.1.36 / RKPPL 0.1.30) includes homogeneous
    # empty/nonempty partition plans and scheduled-loop resets.
    # f851e008 (RK 0.1.37 / RKPPL 0.1.31) also binds data-computed entry
    # axes before shape inference (RK snag prepare-a-strict-13a9e1b8).
    # 81363978 (RK 0.1.38 / RKPPL 0.1.32) binds visible tuple-geometry
    # metadata operands before shape inference (prepare-one-resp-b1f58f91).
    # 23d26ca1 (RK 0.1.39 / RKPPL 0.1.33) keeps indexed direct predictor
    # reads beside reductions local (sum-factor-outer-ca5cc80e).
    # RKPPL 0.1.34: optional SMC adapter and data-only checkpoints.
    # 4df6564e (RK 0.1.40 / RKPPL 0.1.35) preserves natural scalar reads
    # of predictor elements in strict partitions, including separate row domains.
    # RKPPL 0.1.36: finite-state filtering, design rows and grouped scan outputs.
    # RKPPL 0.1.38 (RK 0.1.42): shared-prefix partition sampling and mixed
    # concrete partition-label plans.
    # 171f36b9 (RKPPL 0.1.39) keeps shapes through nested data-only helpers
    # inside sampled expressions, such as a bounded shares completion's
    # `lower=lo[x_shares_free]` (RK snag bounded-shares-g-792cef37);
    # dc06acc8 is the RK main containing it.
    # 8f7832e7 (RK main after 0.1.42 / RKPPL 0.1.39, no version change) bounds
    # partition-plan and metadata compilation at Atlas size: value-only shape
    # reads stay out of plans and bookkeeping helper calls stay separate
    # kernels (RK snag full-atlas-value-740085e1).
    # RKPPL 0.1.37: exact nonfinite SMC diagnostics across mutation tasks.
    # RKPPL 0.1.42 (RK 0.1.44): independent SMC pilot tuning, adaptive pilots and
    # portable correlated references.
    # RKPPL 0.1.47 (RK 0.1.47): the learned reference fits inflated pilot moments
    # (`reference_inflation`, default 2); stage summaries add coordinate and
    # reference SDs (snag learned-smc-refe-cc5677a1).
    # RK 0.1.50 / RKPPL 0.1.52: shared-prefix queries, packed gathered-column
    # shapes, strict row reductions and whole-row gathers,
    # unchanged scan captures, and censored ragged cell-factor projections.
    # RK 0.1.51 / PPL 0.1.53 keeps scalar plate collectors local through
    # helper calls, guarded captures and broadcast observation locations.
    # RKPPL 0.1.54 binds shared-prefix queries in the latest world and preserves
    # custom packed-vector storage and its local reads.
    # 5e9d2d03 contains 8b7caa90: guarded gathered sums retain reductions
    # through redundant typed aliases (RK 0.1.52 / RKPPL 0.1.55).
    # RKPPL 0.1.57: independent prepared-density SMC factories.
    # RK 0.1.57 / PPL 0.1.64: caller-owned joint density KernelSpecs and geometry.
    # RK 0.1.59 / PPL 0.1.66: property-based whole-parameter geometry dimensions.
    # PPL 0.1.60 / DistributionKernels 0.1.1 keeps Normal interval masses
    # and native Reverse gradients finite in both extreme tails.
    # Shared-prefix mixed derivatives, batched values and local supports
    # require RK 0.1.58 / PPL 0.1.65. RK 0.1.59 / PPL 0.1.66 also keeps
    # the broadcast domain through cached factor projection.
    # RK 0.1.61 / PPL 0.1.70: selective growing-window targets, conditional
    # recovery and bounded-rank joint Gaussian carries.
    # RK 0.1.62 preserves inner row selection through array-valued producer
    # plate cells, including BRM's shared correlated random-effect predictors.
    # RK 0.1.63 (57f9a59a; PPL 0.1.73) keeps the shared-prefix cut's frontier
    # types when preparing `partitioned` cell factors, so sparse Laplace
    # sum factors over shared model-scope helpers stay strict
    # (RK snag partitioned-cell-717ab7c7).
    # RKPPL 0.1.72 reports a vector simplex or ordered prior's own block
    # (plus sampled hyperparents) as its authored-factor support.
    # b5740888 (RKPPL 0.1.71) sizes a data-free replay from frozen whole
    # values, such as a gmrf support fitted from an indexed fit-role input.
    # fbf65b46 (RK 0.1.65 / RKPPL 0.1.74) deep-copies prepared queries whose
    # generated programs hold modules, as a Laplace plan over a gmrf does.
    # RK 0.1.66 / RKPPL 0.1.75 computes per-position helpers at a strict
    # cell's rows; f17b6a24 (RK 0.1.67 / RKPPL 0.1.76) also reads joint cell
    # latents and storage fields there. RK 0.1.70 / RKPPL 0.1.80 gives strict
    # cells row forms for matrix-row helpers, `vec` of a row value and whole
    # arrays read by index. 38421748 (RK 0.1.73 / RKPPL 0.1.82) also computes a
    # producer plate's cell and `b[p, 2]` of a computed matrix at their rows.
    # f17b6a24 (RK 0.1.67 / RKPPL 0.1.76) prepares strict observation-cell
    # footprints of a per-cell joint field and its named-tuple row reads.
    # 725059f4 (RKPPL 0.1.89): read-only constrained parameter shapes.
    # 0ee20c70 (RKPPL 0.1.93): authored local and reusable-definition docs.
    # Includes native row-dependent/shared mixture probabilities (0.1.90).
    # Also contains the adaptive pilot lifecycle introduced in RKPPL 0.1.91.
    # c1f175d7 (RK 0.1.84 / PPL 0.1.96): compact authored-query batches
    # share binding validation and exemplars; event inventory avoids full copies.
    # a9a7e9a2 (RKPPL 0.1.97): bound-data LKJ dimensions in reusable definitions.
    # Also contains RK 0.1.82 (b08cb894), which retains live broadcast fusion
    # after data binding, so batched vector-cell readers reuse their result
    # buffers at every position.
    # b0ffcdc1 (RKPPL 0.1.98, on RK main through 6ea1d4ca) normalizes
    # generated-module names in SMC checkpoint identities, so a fresh process
    # resumes an adaptive pilot of an RKBRMI model whose law is an emitted
    # binding (a gmrf field). df3eb8872 also fixes capture footprints through
    # row-computed index lanes in selective carried-window tuning
    # (test/smc_window_selective_api.jl; RK snag selective-carrie-1aa69e67).
    # db6bca825 (RK 0.1.86 / RKPPL 0.1.99) includes both fixes and the
    # follow-up scalar-observation gather-demand repair. Pin by SHA: these
    # versions also label earlier commits without the footprint repair.
    # cb3b33bd (RKPPL 0.1.100) adds opt-in independent adaptive grid pilots
    # (`tune_smc(...; grid_start=:independent)`) and the production worker
    # override `resume_smc(...; nthreads)`, which BRM forwards.
    ("ReactiveKernels",   "https://github.com/nsiccha/ReactiveKernels.jl.git",   "cb3b33bd08da2b13c9105eb75a6c15473b72d467"),
    # ReactiveKernels' own `[sources]` revision of its unregistered
    # OutputSignatures dependency (RK compat 0.1.6); move it with the RK pin.
    ("OutputSignatures",  "https://github.com/nsiccha/OutputSignatures.jl.git",  "f971381f760e91986b3a728a719341e7e7761f55"),  # main
    # d37cb0b4 (Enzyme 0.13.216 dogfood, libEnzyme build 4be29b19): contains
    # 93c452bd, Base concatenation of views, reshapes and transposes of Arrays
    # beside dense or Const arrays in Reverse and Forward (snag
    # native-reverse-t-6d51cae6), which a totals cell's `hcat` design read at
    # its rows needs; EnzymeCore 0.8.27 comes from its `lib/EnzymeCore`.
    # 3e707d79 (libEnzyme build 1797ad22) contains d37cb0b4 and 7a886d9, the
    # static-activity Reverse repair (Enzyme snag static-reverse-s-f9483e9c):
    # a load through a select or phi that merges Const data with active data,
    # which RKPPL's compact-cell AD (`prepare_compact_cell_ad`) runs into under
    # plain Reverse when the sparse Laplace differentiates an `@plate` cell.
    # 3a89b497 includes the compact nested-AD overlay repairs; its declared
    # libEnzyme is 782c9d95, the sum-factor producer's tested native pair.
    # a6458e9c (libEnzyme 782c9d95) repairs ordinary Reverse over
    # constant plan fields and compact grouped overlays.
    # 0e251cfb (libEnzyme build 782c9d95) also fixes second derivatives
    # through dynamic calls on Julia 1.10 (Enzyme snag compute-an-ordin-902b6418),
    # including the lower-bounded completion's ParameterGeometry callbacks.
    # 5325554c keeps that libEnzyme build and repairs erfcx derivatives in
    # positive tails through third order (Enzyme snag erfcx-positive-t-db1c8c84).
    # Embedded Laplace needs those orders for scaled-tail likelihoods.
    # 5325554c also contains the wide-plan tuple-field typing repair used by
    # large compact partitions (32d31ca; same libEnzyme build 782c9d95).
    # 68ed90fe declares libEnzyme 2063db12, repairing the integer division
    # fold that stopped grouped-HSGP partition gradients on Julia 1.10.
    # ae936763 (libEnzyme build 0dd21172) includes 71c203f9: Julia 1.10
    # inference preserves nested reduction types during ordinary Reverse,
    # including sum(sum, columns) after matrix-valued plate helpers.
    # 93e89fc8 / libEnzyme 7f957e67 also supplies the concurrent-GC repair.
    # 474fd636 / libEnzyme df199913 preserves JIT symbols from augmented
    # reverse helpers under Forward, including the digamma partials needed
    # by ordinary negative-binomial compact-cell HVPs.
    # 330b3578 / libEnzyme 2b1f1cfe: the Forward derivative of a division no
    # longer squares its divisor (Enzyme todo 0g6ywzj), so a cell's third order
    # through log(erfc(t)) stays finite in the tail; it also carries the logger
    # repair (Enzyme snag nested-forward-o-206e01ab) that lets a nested
    # derivative first compile under a TestLogger, as @test_logs does here.
    # 07842f2e / libEnzyme fce75191 carries both, and adds the repair for
    # "Illegal cached pointer" from an array element read in a loop, which
    # 330b3578 raised in a nested @plate cell's sparse Laplace HVP and in
    # nlme_integrated.jl (Enzyme snag dogfood-tip-ille-519c9d56).
    # 6053067f / libEnzyme 3425ed9c types floating-point tuple stack slots
    # throughout Reverse, so derived-scale blocks wider than four margins
    # differentiate without changing BRM's block pass (reverse-untyped-a117a92b).
    # de016556 / libEnzyme c1743bf9 retains those fixes and frees constant
    # loop-count caches after Forward over Reverse. Repeated sparse Laplace
    # cell HVPs otherwise retain native memory (forward-over-rev-f93d10bc).
    # d3189970 / libEnzyme 33709d12 contains 6053067f and repairs ordinary
    # Reverse through returned memory (Enzyme snag ordinary-autoenz-6cf1298d):
    # the ordinary AR/DAR and smooth readers in rk_values.jl segfaulted
    # on Julia 1.10 under 6053067f, as did the strict `mo1()` summand
    # partition gradient (RK monotonic-summan-c191e14a). It also keeps
    # 842cbefe: without it, a forced bounded ODL plan's Forward over Reverse
    # cell HVPs stop on "Mismatch codegen type" through a non-inlined
    # `Base.unalias` (julia-1-10-unali-c91b1f15).
    # d3189970 also contains 0fa85df3's 74abe90/7ac58eb: ordinary Reverse of a mixed
    # integer/Float64 select and of a guarded PHI joining scalar stack storage
    # with a boxed broadcast wrapper, which an SVI unit's dormant scalar/vector
    # branch reaches (ReactiveKernels snag brm-retained-poi-a68d4412).
    # b2bb4cb6 / libEnzyme f392eba5 contains d3189970 and caches a value that
    # cannot change inside a loop once per loop entry (Enzyme snag
    # reverse-generato-03b2003d), so a multiresolution gmrf reader's generator
    # sum no longer tapes its captured arrays once per basis term.
    ("Enzyme",            "https://github.com/nsiccha/Enzyme.jl.git",            "b2bb4cb628cf5f5ac4a897d4b7d83fcd2b5df97c"),  # dogfood
]

# Derive every host mirror from the pins so newly added dependencies also
# avoid a public clone when their exact revision is already available locally.
# The resolver falls back to the public origin if the mirror or revision is absent.
const MIRRORS = Dict(
    name => joinpath(homedir(), ".local", "state", "kb-agents", "git-mirrors", "$name.jl.git")
    for (name, _, _) in PINS
)

# The KB resolver's dogfood-libEnzyme step, printed by the public
# `dogfood_libenzyme_program` of Claude's `lib-resolve.sh` (the one
# implementation; skill enzyme-use, "Bringing dogfood into an environment"
# step 4, whose LOCKSTEP note names this consumer): it reads the developed
# Enzyme's `deps/dogfood_libenzyme.toml`, checks this host's build against its
# SHA256SUMS, writes `Enzyme_jll`'s library paths into the ignored
# `test/LocalPreferences.toml`, and errors where the build is missing.
function apply_dogfood_libenzyme()
    lib = get(ENV, "BRM_TEST_KB_RESOLVER",
        joinpath(homedir(), "github", "nsiccha", "Claude", "lib-resolve.sh"))
    isfile(lib) || error("""
        The pinned Enzyme is the dogfood line, which runs against its own
        libEnzyme build; only the KB resolver ($lib, missing here) applies it.
        Set BRM_TEST_KB_RESOLVER to Claude's lib-resolve.sh, or run on a KB
        fleet host that has the build (skill enzyme-use).""")
    step = read(Cmd(["bash", "-c",
        "source \"\$1\" >/dev/null && dogfood_libenzyme_program resolve", "_", lib]), String)
    include_string(Main, step, "dogfood_libenzyme")
    return nothing
end

function main()
    store = test_package_store()
    paths = Dict("BayesianRegressionModels" => REPO)
    for (name, url, rev) in PINS
        paths[name] = resolve_git_revision_checkout(
            name,
            rev;
            cache_root=store,
            mirror=MIRRORS[name],
            origin=url,
        )
    end
    # Before the precompile, which a run's time limit can stop. This run's pins
    # were just marked used, so they are safe.
    evict_unused_checkouts!(store)
    # Enzyme's EnzymeCore is its nested `lib/EnzymeCore` (Enzyme's
    # `[sources]`, which Pkg 1.10 ignores).
    paths["EnzymeCore"] = joinpath(paths["Enzyme"], "lib", "EnzymeCore")
    # Nested monorepo packages develop from the pinned ReactiveKernels
    # checkout (same revision, no separate pins).
    rk_root = paths["ReactiveKernels"]
    paths["ReactiveKernelsDistributionKernels"] =
        joinpath(rk_root, "packages", "ReactiveKernelsDistributionKernels")
    paths["ReactiveKernelsPPL"] =
        joinpath(rk_root, "packages", "ReactiveKernelsPPL")
    # The v1 SB-parity sweep (test/sb_sweep_*.jl) consumes the inventory
    # models' exact data bindings from here — no transcription.
    paths["ReactiveKernelsPPLExamples"] =
        joinpath(rk_root, "packages", "ReactiveKernelsPPLExamples")

    Pkg.activate(TESTENV)
    Pkg.develop(PackageSpec[
        PackageSpec(path=path) for (_name, path) in sort!(collect(paths))
    ])
    Pkg.instantiate(; allow_autoprecomp=false)
    # Before the first precompile, so Enzyme_jll builds against the declared
    # libEnzyme.
    apply_dogfood_libenzyme()
    Pkg.precompile()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
