# Public environment for declaring models and exporting Stan.
# Julia 1.10 ignores [sources], so develop the public source closure together.
using Pkg, TOML

function setup_start()
    root = normpath(joinpath(@__DIR__, "../.."))
    release = TOML.parsefile(joinpath(root, "Project.toml"))["sources"]
    test = TOML.parsefile(joinpath(root, "test", "Project.toml"))["sources"]
    paths = [root]
    for name in ("PowerScaling", "ReparametrizableDistributions", "StanBlocks")
        source = name == "StanBlocks" ? test[name] : release[name]
        revision = source["rev"]
        path = joinpath(@__DIR__, ".sources", "$name-$revision")
        if !isdir(path)
            mkpath(dirname(path))
            run(`git clone --no-checkout $(source["url"]) $path`)
            run(`git -C $path checkout --detach $revision`)
        end
        strip(read(`git -C $path rev-parse HEAD`, String)) == revision ||
            error("$name checkout differs from the declared revision: $path")
        push!(paths, path)
    end
    Pkg.activate(@__DIR__)
    Pkg.develop([PackageSpec(path=path) for path in paths])
    Pkg.instantiate()
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && setup_start()
