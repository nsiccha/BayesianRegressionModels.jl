# test/lazy_dependencies.jl — BRM's fitting-time packages are weak dependencies
# (src/fitting_extensions.jl; snag `brm-fitting-solv-c9af5617`, user picks on
# decisions `0ilan3m` and `0mdvrl1`).
#
# The rule: no BRM code outside package X's own extension names X. BRM does not
# depend on X, so such code would compile and fail only when it runs; it calls
# X's extension through the functions src/fitting_extensions.jl declares. This
# file checks that rule on the parsed source (comments and strings do not
# count), that each package is a weak dependency whose extension has it as its
# one trigger and defines every function BRM calls, and, in fresh processes,
# that `using BayesianRegressionModels` loads none of them, that a call before
# its package is loaded names the package, and that a call works once it is,
# including when the package loads during the call.
#
# Run: julia --startup-file=no --project=test test/lazy_dependencies.jl
using Test, BayesianRegressionModels
const BRM = BayesianRegressionModels

const REPO = dirname(@__DIR__)
const EXTENSIONS = BRM._BRM_FITTING_EXTENSIONS

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

function _named_fitting_packages(path)
    out = Tuple{Symbol,Int}[]
    _identifiers!(out, Meta.parseall(read(path, String); filename=path))
    [(name, line) for (name, line) in out if haskey(EXTENSIONS, name)]
end

@testset "no code outside a package's own extension names it" begin
    own = Dict(joinpath(REPO, "ext", "$(EXTENSIONS[name]).jl") => name for name in keys(EXTENSIONS))
    files = vcat([joinpath(REPO, "src", f) for f in readdir(joinpath(REPO, "src")) if endswith(f, ".jl")],
        [joinpath(REPO, "ext", f) for f in readdir(joinpath(REPO, "ext")) if endswith(f, ".jl")])
    @test length(files) > 100
    violations = String[]
    for path in files
        endswith(path, joinpath("src", "fitting_extensions.jl")) && continue
        for (name, line) in _named_fitting_packages(path)
            get(own, path, nothing) === name && continue
            push!(violations, "$(relpath(path, REPO)):$line names `$name`; call its " *
                "extension through a function src/fitting_extensions.jl declares")
        end
    end
    isempty(violations) || foreach(v -> @error(v), violations)
    @test isempty(violations)
    # Positive control: the scan sees a module named in code, not in a comment
    # or a string.
    probe = tempname() * ".jl"
    write(probe, "# KrylovKit in a comment\nf() = \"MUMPS in a string\"\ng(x) = KrylovKit.basis(x)\n")
    @test _named_fitting_packages(probe) == [(:KrylovKit, 3)]
    rm(probe)
end

@testset "each package is a weak dependency whose extension defines what BRM calls" begin
    project = Base.parsed_toml(joinpath(REPO, "Project.toml"))
    for (name, ext) in pairs(EXTENSIONS)
        @test haskey(project["weakdeps"], String(name))
        @test !haskey(project["deps"], String(name))
        @test get(project["extensions"], String(ext), nothing) == String(name)
        source = read(joinpath(REPO, "ext", "$ext.jl"), String)
        for f in BRM._BRM_FITTING_FUNCTIONS[name]
            @test occursin("BRM.$f(", source)
        end
    end
    @test Set(keys(BRM._BRM_FITTING_FUNCTIONS)) == Set(keys(EXTENSIONS))
end

function _fresh(script)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    out = read(pipeline(cmd; stderr=devnull), String)
    Dict(first(split(l, ' '; limit=2)) => last(split(l, ' '; limit=2))
        for l in split(out, '\n') if occursin(' ', l))
end

const _NAMES = join(repr.(String.(collect(keys(EXTENSIONS)))), ", ")

@testset "using BRM loads none of them; a missing one is named" begin
    lines = _fresh("""
        using BayesianRegressionModels, Random
        const BRM = BayesianRegressionModels
        loaded() = Set(k.name for k in keys(Base.loaded_modules))
        println("EAGER ", join(sort!(collect(intersect(loaded(), [$_NAMES]))), ","))
        message = try
            BRM._brm_rank_ess_rhat(randn(Xoshiro(1), 100, 1, 2)); "no error"
        catch err
            sprint(showerror, err)
        end
        println("NAMES ", occursin("needs `MCMCDiagnosticTools`", message) &&
            occursin("`import MCMCDiagnosticTools", message))
        """)
    @test lines["EAGER"] == ""
    @test lines["NAMES"] == "true"
end

@testset "a call works once its package is loaded, also when it loads during the call" begin
    lines = _fresh("""
        using BayesianRegressionModels, Random
        const BRM = BayesianRegressionModels
        # One top-level block: the extension loads during it, so its methods
        # are newer than the block's world and the fallback retries the call.
        let
            Base.require(Base.PkgId(Base.UUID("be115224-59cd-429b-ad48-344e309966f0"),
                "MCMCDiagnosticTools"))
            stats = BRM._brm_rank_ess_rhat(randn(Xoshiro(1), 100, 1, 2))
            println("CALL ", length(stats.rhat), " ", all(isfinite, stats.ess))
            println("EXT ", Base.get_extension(BRM, :BayesianRegressionModelsMCMCDiagnosticToolsExt) !== nothing)
        end
        """)
    @test lines["CALL"] == "2 true"
    @test lines["EXT"] == "true"
end
