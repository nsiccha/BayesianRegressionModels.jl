# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# Five of the test env's dependencies are UNREGISTERED — MutatingFunctions and
# OutputSignatures (the julianic 0-alloc substrate), TreeArrays (the
# postprocessing containers), plus StanBlocks / Treebars / WarmupHMC. On Julia
# 1.11+ the `[sources]` blocks in `test/Project.toml` resolve them from GitHub;
# on **1.10, which is what this suite runs on, `[sources]` is IGNORED**, so a
# bare `Pkg.resolve()` fails with
#
#     ERROR: expected package `TreeArrays [5daaa025]` to be registered
#
# and the run is lost to an error that looks like a missing registry rather than
# a version-gated feature. This script is the 1.10 answer: it `develop`s each one
# from its local checkout, which is also what pins the julianic substrate to the
# two upstream fixes the surface depends on (`0134a6c` / `121de31`).
#
# It is idempotent — re-running it on an already-correct env is a no-op — so it
# is safe to run before any suite when unsure of the env's state.
using Pkg

const CHECKOUTS = joinpath(homedir(), "github", "nsiccha")

# Ordered by how load-bearing they are, so a missing one is reported before the
# rest of the work happens.
const UNREGISTERED = [
    "MutatingFunctions",    # apply!! — the in-place 0-alloc fills
    "OutputSignatures",     # broadcast_size / output_size — their destination shapes
    "TreeArrays",           # labelled postprocessing draw containers (weak dep)
    "StanBlocks",
    "Treebars",
    "WarmupHMC",
]

function main()
    missing_checkouts = String[]
    specs = PackageSpec[]
    for name in UNREGISTERED
        path = joinpath(CHECKOUTS, name * ".jl")
        if isdir(path)
            push!(specs, PackageSpec(; path))
        else
            push!(missing_checkouts, path)
        end
    end
    if !isempty(missing_checkouts)
        # Fail loudly and completely rather than developing a partial env: a
        # half-built environment fails later, somewhere less obvious.
        error("test/setup_env.jl: missing local checkout(s):\n  " *
              join(missing_checkouts, "\n  ") *
              "\nClone them from https://github.com/nsiccha/<name>.jl and re-run.")
    end
    Pkg.develop(specs)
    Pkg.instantiate()
    return nothing
end

main()
