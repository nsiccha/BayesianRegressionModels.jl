# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env has five external UNREGISTERED dependencies plus the unregistered
# BRM root itself. Each external package is materialized at a specific GitHub
# COMMIT under the ignored `test/.bootstrap/` cache; there is NO dependence on
# any shared `~/github/nsiccha/<pkg>` checkout. A full-SHA revision is
# branch-independent, so it does not matter that several of these live on a
# `dev`/`devibe` branch rather than `main`; the commit only has to be pushed to
# GitHub, which every pin below is except Enzyme.
#
# Enzyme is the ecosystem's `dogfood` line (user pick on decision
# `BayesianRegressionModels/decisions/2026-10-28T15-34-11-008-0n0kgfh`): reviewed
# repairs no registered release carries, which BRM's RK cell gradients need. Its
# commits live only in the KB hosts' Enzyme mirror (`MIRRORS`), and it runs
# against its own libEnzyme build (`deps/dogfood_libenzyme.toml`), which exists
# only on the hosts the Enzyme lane builds it for (gordito, Julia 1.10 and
# 1.12). `apply_dogfood_libenzyme()` points `Enzyme_jll` at this host's build
# through the KB resolver's own step, and refuses where there is none: the
# stock library gives wrong Reverse gradients the build fixes (skill
# `enzyme-use`).
#
# ReactiveKernels contributes four developed paths from ONE pinned checkout
# (the monorepo root plus the nested `ReactiveKernelsDistributionKernels`,
# `ReactiveKernelsPPL`, and `ReactiveKernelsPPLExamples` packages, which have
# no standalone repos).
#
# All nine paths enter ONE `Pkg.develop` call on EVERY Julia version we run.
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
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "9a4fb3a6ba0ec559eddaf5673afed107ac30a254"),  # devibe: a `truncated` draw succeeds whatever mass its window holds: after 100 misses it inverts the base CDF on the window (9a4fb3a6, StanBlocks 0.2.6, snag truncated-rng-re-bd555cae), which BRM's omitted truncated responses and `_gen` draws call; `rv[idx]` selects groups of a ragged value and `sum(rv)` adds its groups elementwise (ffbaff09, StanBlocks 0.2.2, snag selecting-severa-027a8191), which BRM's model-scope reads of a ragged plate output pass through; Stan's `sum_to_zero_constrain`/`_unconstrain`/`_jacobian` trace (8e57a83, snag sum-to-zero-cons-90a72024) and `head`/`tail`/`reverse` (359844e, snag stan-vector-buil-5792b9e5), which `T::shares(total)` cell latents call, both in f5e4650; a `Vector{Matrix}` data value ingests as a `RaggedMatrix` and `X[g]` reads the g-th matrix in a plate cell (c9504e99, snag ragged-matrix-da-163255e9); a plate cell may bind a matrix local of fixed or per-cell size (a57951eb, snag binding-a-per-ce-e16efa8d); a plate's Stan names follow from each value's own name, without the `__pl_mem_1` suffix (1d93a65), and `stan_interface` / `stan_data_json` describe the program (dc3f713), both in fd91d0c; tuple descriptor entries carry their components in `fields` (785edf4); ragged sizes are reported derived (9118764); a values-free `maybecv(:y)` marks an observation with no values as predicted given a fit (c4671eb, snag held-out-observa-58a2eb13); shared scan arguments read once per step, linear in series length (b97fc5a6); Stan's vectorised builtins (`fmax`, `fmin`, …) evaluate elementwise in the Julia layer, so a per-axis `fmax(0.0, lo)` bound reads one value per coordinate (c7293b88); descriptor inputs/outputs carry `constraint_values`, their bounds evaluated against the bound data (b233afbc); a `ValueFamily` as the base family of `weighted`/`truncated`/`censored`/`interval_censored` with `lcdf=`/`lccdf=` companions (94e414bb); descriptor outputs carry their value shape as `dims`; non-finite literals render as Stan infinity builtins; `weighted` over censored/truncated/interval_censored; an exchangeable ordered prior re-draws as the sorted iid draw; observed scans and per-cell scan lengths (82a954da), also through a do-cell parameter that shadows its data name (4ee6e0e8); integer step locals in nested scans (2f5b6203); a latent or observed `@scan` inside a called `@slic` submodel (e67bc314), observing a plate cell's group (d3483f11); `interval_censored`'s interval by keyword and unknown combinator keywords refused (897557c); typed `=` in `stan_descriptor` (204e5a7)
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    # 0194dce (2026-09-27, dev): WindowSelectionPlan (WarmupHMC-held evidence)
    # and the controls interface for custom reparametrizers,
    # which the per-window S2Z Fisher rule needs. Contains 7aed40b (active-state
    # preservation, nonfinite-transport rejection). Matches test/Project.toml.
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "0194dce08e986ff17fd5a788bb315c6431b7858e"),  # dev (contains exact sampling-counter floor 913da79)
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
    # need.
    ("ReactiveKernels",   "https://github.com/nsiccha/ReactiveKernels.jl.git",   "b31d672faa820fe8bda46cf86d7cb2eb43cc8b20"),  # main
    # ReactiveKernels' own `[sources]` revision of its unregistered
    # OutputSignatures dependency (RK compat 0.1.5); move it with the RK pin.
    ("OutputSignatures",  "https://github.com/nsiccha/OutputSignatures.jl.git",  "d16c4efb72b963f45de9d6f632c38d46706c23e1"),  # main
    # d37cb0b4 (Enzyme 0.13.216 dogfood, libEnzyme build 4be29b19): contains
    # 93c452bd, Base concatenation of views, reshapes and transposes of Arrays
    # beside dense or Const arrays in Reverse and Forward (snag
    # native-reverse-t-6d51cae6), which a totals cell's `hcat` design read at
    # its rows needs; EnzymeCore 0.8.27 comes from its `lib/EnzymeCore`.
    ("Enzyme",            "https://github.com/nsiccha/Enzyme.jl.git",            "d37cb0b45df7bad731bdb218d50b8f49c35bc99d"),  # dogfood (KB mirror only)
]

# Pins whose commits are not on GitHub: the KB host mirror they come from.
const MIRRORS = Dict(
    "Enzyme" => joinpath(homedir(), ".local", "state", "kb-agents", "git-mirrors", "Enzyme.jl.git"),
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
    paths = Dict("BayesianRegressionModels" => REPO)
    for (name, url, rev) in PINS
        paths[name] = resolve_git_revision_checkout(
            name,
            rev;
            cache_root=joinpath(TESTENV, ".bootstrap"),
            mirror=get(MIRRORS, name, ""),
            origin=url,
        )
    end
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
    # Older Julia 1.10 runtimes need a serial build of the sibling Turing
    # extensions before the normal parallel pass. Julia 1.10.12 includes the
    # loadable_exts fix and uses ordinary parallel precompilation directly.
    Pkg.instantiate(; allow_autoprecomp=false)
    # Before the first precompile, so Enzyme_jll builds against the declared
    # libEnzyme.
    apply_dogfood_libenzyme()
    if v"1.10" <= VERSION < v"1.10.12"
        withenv("JULIA_NUM_PRECOMPILE_TASKS" => "1") do
            Pkg.precompile(["Pathfinder", "Turing"])
        end
    end
    Pkg.precompile()
    return nothing
end

main()
