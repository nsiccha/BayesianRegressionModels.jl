# Rebuild the test environment.
#
#     julia --project=test test/setup_env.jl
#
# The test env has six UNREGISTERED dependencies. Each is pinned to a specific
# GitHub COMMIT and `Pkg.add`ed straight from GitHub by `rev` — there is NO
# dependence on any local/shared `~/github/nsiccha/<pkg>` checkout. A full-SHA
# `rev` is branch-independent, so it does not matter that several of these live
# on a `dev`/`devibe` branch rather than `main`; the commit only has to be
# pushed to GitHub, which every pin below is.
#
# `Pkg.add(...; rev)` is the mechanism on EVERY Julia version we run. On 1.11+
# the `[sources]` blocks in `test/Project.toml` (which mirror these revs) would
# also resolve them; on **1.10, which is what this suite runs on, `[sources]` is
# IGNORED**, so a bare `Pkg.resolve()` fails with
#
#     ERROR: expected package `TreeArrays [5daaa025]` to be registered
#
# an error that looks like a missing registry rather than a version-gated
# feature. This script is the version-independent answer.
#
# Bumping a pin is a deliberate one-line edit here (+ the matching `[sources]`
# rev in test/Project.toml), reviewed like any other change — not a silent
# consequence of whatever a shared checkout drifted to.
#
# It is idempotent — re-running it on an already-correct env is a no-op — so it
# is safe to run before any suite when unsure of the env's state.
using Pkg

# name => (github url, pinned commit)   — comment records the branch the commit
# is on, for humans; the pin itself is the full SHA and needs no branch.
const PINS = [
    # MutatingFunctions carries the SubArray-gather activity fix the julianic
    # 0-alloc surface depends on (main).
    ("MutatingFunctions", "https://github.com/nsiccha/MutatingFunctions.jl.git", "92687c809f60493422b11158ac8abba32b21cdf9"),
    ("OutputSignatures",  "https://github.com/nsiccha/OutputSignatures.jl.git",  "121de3194f02044e00bac0d11019a93458ddb63a"),  # main
    ("TreeArrays",        "https://github.com/nsiccha/TreeArrays.jl.git",        "c317cc003fc41c2d933c27dc80799141eebd434e"),  # main
    ("StanBlocks",        "https://github.com/nsiccha/StanBlocks.jl.git",        "e6f607da17f04f06e7e65d867137940a82ba1392"),  # devibe
    ("Treebars",          "https://github.com/nsiccha/Treebars.jl.git",          "c02aa16ab1b08e4f5283597fe678a88e69555cd1"),  # dev
    ("WarmupHMC",         "https://github.com/nsiccha/WarmupHMC.jl.git",         "8fa829b39d6f519deaf13cd03ee17e8df6b7b9c2"),  # dev
]

function main()
    Pkg.add([PackageSpec(; url, rev) for (_name, url, rev) in PINS])
    Pkg.instantiate()
    return nothing
end

main()
