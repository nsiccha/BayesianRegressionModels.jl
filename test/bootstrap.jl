# Stand up the test environment described by `test/Project.toml`.
#
#     BRM_TEST_STANBLOCKS=/path/to/StanBlocks.jl \
#     BRM_TEST_TREEBARS=/path/to/Treebars.jl \
#     julia --project=test test/bootstrap.jl
#
# then run any test file with `julia --project=test test/<file>.jl`.
# See `test/README.md` for why each of the constraints below exists.

using Pkg

include("dependency_floors.jl")

const REPO = dirname(@__DIR__)
const TESTENV = @__DIR__

# Four packages must be develop paths rather than registry resolves. The caller
# supplies StanBlocks and Treebars; WarmupHMC may be supplied or reproducibly
# materialized at its enforced floor. Absolute paths are machine-specific and
# deliberately never committed; `test/Manifest.toml` (where `Pkg.develop`
# records them) is covered by the root `.gitignore`.
function dev_path(name, var; required=true)
    raw = get(ENV, var, "")
    if isempty(raw)
        required && error("""
            $var is unset — $name has to be an explicit develop path.
            Point it at a local checkout: $var=/path/to/$name.jl
            """)
        return nothing
    end
    path = abspath(expanduser(raw))
    isdir(path) || error("$var=$raw does not name a directory ($path)")
    isfile(joinpath(path, "Project.toml")) ||
        error("$var=$path is not a Julia package (no Project.toml)")
    path
end

stanblocks = dev_path("StanBlocks", "BRM_TEST_STANBLOCKS")
configured_warmuphmc = dev_path(
    "WarmupHMC", "BRM_TEST_WARMUPHMC"; required=false)
warmuphmc = resolve_git_floor_checkout(
    "WarmupHMC",
    configured_warmuphmc,
    WARMUPHMC_NATIVE_PPL_MINIMUM;
    cache_root=joinpath(TESTENV, ".bootstrap"),
    mirror=get(ENV, "BRM_TEST_WARMUPHMC_MIRROR", joinpath(
        homedir(), ".local", "state", "kb-agents", "git-mirrors",
        "WarmupHMC.jl.git")),
    origin=get(ENV, "BRM_TEST_WARMUPHMC_ORIGIN",
        "https://github.com/nsiccha/WarmupHMC.jl.git"),
    branch="dev",
    reason="Native PPL targets require WarmupHMC's own-gradient Pathfinder initialization.",
)
# Transitive: WarmupHMC depends on the unregistered Treebars and pins it with a
# `[sources]` entry, which Julia 1.10 ignores. `Pkg.develop` can only fix a path
# for a DIRECT dependency, so Treebars is also listed in `test/Project.toml`.
treebars = dev_path("Treebars", "BRM_TEST_TREEBARS")

@info "Developing into $TESTENV" BayesianRegressionModels = REPO StanBlocks = stanblocks WarmupHMC = warmuphmc Treebars = treebars

Pkg.activate(TESTENV)

# ONE `develop` call for all four. Resolution has to satisfy them together:
# developing StanBlocks on its own fails with "expected package
# BayesianRegressionModels to be registered", because BRM is unregistered and
# is not yet in the manifest when StanBlocks' own resolve runs.
Pkg.develop([
    PackageSpec(path=REPO),
    PackageSpec(path=stanblocks),
    PackageSpec(path=warmuphmc),
    PackageSpec(path=treebars),
])

# Everything else comes from the registry, per the committed `test/Project.toml`
# — which is the single source of truth for the dependency list. This script
# only supplies the paths that cannot be committed.
Pkg.instantiate()
Pkg.precompile()

@info "Test environment ready" project = Base.active_project()
