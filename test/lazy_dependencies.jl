# test/lazy_dependencies.jl — BRM's fitting-time packages load on first use
# (src/lazy_dependencies.jl; snag `brm-fitting-solv-c9af5617`, user pick on
# decision `0ilan3m`: keep first-use loading and enforce its one rule).
#
# The rule: no BRM code outside package X's own extension names X. Such code
# would compile and fail only when it runs, because BRM no longer imports X;
# it must call into X's extension through `_brm_lazy` instead. This file checks
# that rule on the parsed source (comments and strings do not count; the quoted
# `:X` that `_brm_lazy` takes does), that every package has its extension with
# X as its one trigger and X in `[deps]`, and, in a fresh process, that
# `using BayesianRegressionModels` loads none of them while a first `_brm_lazy`
# call loads one and returns its result.
#
# Run: julia --startup-file=no --project=test test/lazy_dependencies.jl
using Test, BayesianRegressionModels
const BRM = BayesianRegressionModels

const REPO = dirname(@__DIR__)
const LAZY = BRM._BRM_LAZY_EXTENSIONS

# Every unquoted identifier in `ex`, with its line.
function _identifiers!(out, ex, line=0)
    if ex isa Symbol
        push!(out, (ex, line))
    elseif ex isa LineNumberNode
        return ex.line
    elseif ex isa QuoteNode
        return line
    elseif ex isa Expr
        ex.head === :quote && return line
        for a in ex.args
            a isa LineNumberNode ? (line = a.line) : _identifiers!(out, a, line)
        end
    end
    line
end

function _named_lazy_packages(path)
    out = Tuple{Symbol,Int}[]
    _identifiers!(out, Meta.parseall(read(path, String); filename=path))
    [(name, line) for (name, line) in out if haskey(LAZY, name)]
end

@testset "no code outside a package's own extension names it" begin
    own = Dict(joinpath(REPO, "ext", "$(LAZY[name]).jl") => name for name in keys(LAZY))
    files = vcat([joinpath(REPO, "src", f) for f in readdir(joinpath(REPO, "src")) if endswith(f, ".jl")],
        [joinpath(REPO, "ext", f) for f in readdir(joinpath(REPO, "ext")) if endswith(f, ".jl")])
    @test length(files) > 100
    violations = String[]
    for path in files
        endswith(path, joinpath("src", "lazy_dependencies.jl")) && continue
        for (name, line) in _named_lazy_packages(path)
            get(own, path, nothing) === name && continue
            push!(violations, "$(relpath(path, REPO)):$line names `$name`; call its " *
                "extension through `_brm_lazy(:$name, …)` (src/lazy_dependencies.jl)")
        end
    end
    isempty(violations) || foreach(v -> @error(v), violations)
    @test isempty(violations)
    # Positive control: the scan sees a module named in code, not in a comment
    # or a string.
    probe = tempname() * ".jl"
    write(probe, "# KrylovKit in a comment\nf() = \"MUMPS in a string\"\ng(x) = KrylovKit.basis(x)\n")
    @test _named_lazy_packages(probe) == [(:KrylovKit, 3)]
    rm(probe)
end

@testset "each package has its extension, triggered by the package alone, in [deps]" begin
    project = Base.parsed_toml(joinpath(REPO, "Project.toml"))
    for (name, ext) in pairs(LAZY)
        @test haskey(project["deps"], String(name))
        @test !haskey(get(project, "weakdeps", Dict()), String(name))
        @test get(project["extensions"], String(ext), nothing) == String(name)
        @test isfile(joinpath(REPO, "ext", "$ext.jl"))
    end
end

@testset "using BRM loads none of them; a first _brm_lazy call loads one" begin
    names = join(repr.(String.(collect(keys(LAZY)))), ", ")
    script = """
        using BayesianRegressionModels, Random
        const BRM = BayesianRegressionModels
        loaded() = Set(k.name for k in keys(Base.loaded_modules))
        println("EAGER ", join(sort!(collect(intersect(loaded(), [$names]))), ","))
        # One top-level block: the extension's methods are newer than this
        # block's world, which is the case `_brm_lazy`'s `invokelatest` covers.
        let
            stats = BRM._brm_lazy(:MCMCDiagnosticTools, BRM._brm_rank_ess_rhat,
                randn(Xoshiro(1), 100, 1, 2))
            println("CALL ", length(stats.rhat), " ", all(isfinite, stats.ess))
            println("EXT ", Base.get_extension(BRM, :BayesianRegressionModelsMCMCDiagnosticToolsExt) !== nothing)
            println("AFTER ", join(sort!(collect(intersect(loaded(), [$names]))), ","))
        end
        """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    out = read(cmd, String)
    lines = Dict(first(split(l, ' '; limit=2)) => last(split(l, ' '; limit=2)) for l in split(out, '\n')
        if occursin(' ', l))
    @test lines["EAGER"] == ""
    @test lines["CALL"] == "2 true"
    @test lines["EXT"] == "true"
    @test lines["AFTER"] == "MCMCDiagnosticTools"
end
