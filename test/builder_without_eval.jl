# test/builder_without_eval.jl — `brm(source; mod)`: a model builder from a
# runtime `Expr` that evaluates no code (snag brm-compose-a-mo-035a5e8b).
#
# A model family whose statements vary is built from `Expr` fragments. Before
# `brm`, the only builder for such a body was `Core.eval(mod, _brm(expr))`, and
# `merge(::BRMBuilder...)` evaluated the joined source the same way: a new
# method per model, a compile per model, and an `eval` into the fragments'
# module, which fails while another package's image is generated. `brm` walks
# the body `_brm` generates instead. This file checks that
#
# - it defines nothing: the world counter does not move for `brm`, for calling
#   the builder, or for `merge`;
# - it builds the same model: every `@brm` block of a sweep of this directory
#   gives a structurally equal BRMI through both builders (or the same error),
#   and representative models give byte-identical Stan and data;
# - the Julia a `constant(...)` reads evaluates as compiled code would;
# - definitions and mutation are refused with the offending expression;
# - fragments defined in one package compose in another package's precompile
#   workload, where the evaluating route fails.
#
# Run: julia --startup-file=no --project=test test/builder_without_eval.jl
# BRM_BUILDER_SWEEP=all sweeps every test file instead of the default list
# (BRM_BUILDER_SWEEP=a.jl,b.jl names files).

using Test, BayesianRegressionModels, Distributions, StanBlocks, Random, Statistics
using Pkg, TOML
const BRM = BayesianRegressionModels

const DATA = let rng = Random.MersenneTwister(7)
    n = 24
    (; x=randn(rng, n), c=randn(rng, n), y=randn(rng, n), y2=randn(rng, n),
       subj=repeat(1:6, inner=4), g=repeat(1:3, 8), conc=exp.(randn(rng, n)))
end
const PLATE_DATA = (;
    subject=["b", "a", "c"], weight=[60.0, 75.0, 90.0],
    obs_subject=["a", "c", "b", "a", "b", "c"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0],
    y=[0.81, 0.76, 0.52, 0.49, 0.21, 0.33])

sb(brmi) = SBBRMI(brmi; mod=@__MODULE__)
code_and_data(brmi) = (s = sb(brmi); (BRM.stan_code(s), BRM.stan_data(s)))
compiled(source) = Core.eval(@__MODULE__, BRM._brm(source))

# Two models are the same model when every field agrees. Anonymous functions
# (a constant's evaluator) are compared by what they compute elsewhere, since
# a compiled closure and an interpreted one differ in type by construction.
same(a, b) = a === b || _same(a, b)
_is_anonymous(f) = f isa BRM._BRMClosure || startswith(string(nameof(typeof(f))), '#')
_same(a::Function, b::Function) = typeof(a) === typeof(b) || (_is_anonymous(a) && _is_anonymous(b))
_same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(map(same, a, b))
_same(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(same, a, b))
_same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(map(same, values(a), values(b)))
_same(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(k -> same(a[k], b[k]), keys(a))
_same(a::Union{Expr,Number,AbstractString,Symbol,Module,Type,Nothing,Missing}, b) = isequal(a, b)
function _same(a, b)
    Base.typename(typeof(a)) === Base.typename(typeof(b)) || return false
    all(i -> same(getfield(a, i), getfield(b, i)), 1:nfields(a))
end

# Build with the builder, or the error it throws, as text.
outcome(builder, df) = try
    builder(df)
catch err
    sprint(showerror, err)
end

const FAMILY = quote
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subj)
    log_s = constant(log(mean(conc)) + sum(abs2, [v for v in c if v > 0]) / length(c))
    m = mu + log_s
    y ~ Normal(m, sigma)
    effect(mu, x) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(2)
    cor(:, p) ~ LKJCholesky(2, 2)
end

const PLATE = quote
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs_rows = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        scaled = 2 .* obs_time[obs_rows[i]]
        loc[i] = exp.(-exp(log_CL[i]) .* scaled)
        y[obs_rows[i]] ~ normal(loc[i], sigma)
    end
end

@testset "brm builds without defining anything" begin
    world = Base.get_world_counter()
    builder = brm(FAMILY; mod=@__MODULE__)
    brmi = builder(DATA)
    baked = brm(DATA, FAMILY; mod=@__MODULE__)
    merged = merge(brm(:(mu ~ 1 + x); mod=@__MODULE__),
                   brm(:(begin sigma ~ Exponential(1); y ~ Normal(mu, sigma) end); mod=@__MODULE__))
    merged_brmi = merged(DATA)
    from_text = brm("mu ~ 1 + x\nsigma ~ Exponential(1)\ny ~ Normal(mu, sigma)"; mod=@__MODULE__)
    @test Base.get_world_counter() == world
    @test builder isa BRMBuilder && brmi isa BRMI && baked isa BRMI && merged_brmi isa BRMI
    @test same(brmi, baked)
    @test same(merged_brmi, from_text(DATA))
    # The control: the evaluating route defines a method for every model.
    compiled(FAMILY)
    @test Base.get_world_counter() > world
end

@testset "brm and the compiled builder build the same model" begin
    for (label, source, data) in (("family", FAMILY, DATA), ("plate", PLATE, PLATE_DATA))
        @testset "$label" begin
            interpreted = brm(source; mod=@__MODULE__)
            reference = compiled(source)
            @test brm_columns(interpreted) == brm_columns(reference)
            @test sprint(show, interpreted) == sprint(show, reference)
            @test same(interpreted(data), reference(data))
            @test code_and_data(interpreted(data)) == code_and_data(reference(data))
        end
    end
    # Fragments compose exactly as one block, in the fragments' order, and the
    # grouping and its cell may sit in different fragments.
    grouping = brm(:(obs_rows = groups(obs_subject => subject)); mod=@__MODULE__)
    cell = brm(Expr(:block, filter(s -> !Meta.isexpr(s, :(=)), PLATE.args)...); mod=@__MODULE__)
    @test code_and_data(merge(cell, grouping)(PLATE_DATA)) ==
          code_and_data(compiled(PLATE)(PLATE_DATA))
    # refused: built on its own, a binding no cell reads leaves its vectors
    # ungrouped (snag plate-body-reads-c69f8a88), exactly as `@brm`.
    @test_throws "no cell reads `obs_rows`" grouping(PLATE_DATA)
    # A family member is the same model whichever route built it.
    for covariates in ((:x,), (:x, :c), ())
        rhs = Expr(:call, :+, 1, covariates..., :((1 | p | subj)))
        member = quote
            sigma ~ Exponential(1)
            mu ~ $rhs
            y ~ Normal(mu, sigma)
            optional(effect(mu, c)) ~ Normal(0, 0.1)
        end
        @test code_and_data(brm(member; mod=@__MODULE__)(DATA)) ==
              code_and_data(compiled(member)(DATA))
    end
end

# Names a constant reads resolve as compiled code resolves them: fields of the
# data, then the builder module's globals.
const SCALE = 3.0
square_sum(v) = sum(abs2, v)
@testset "a constant's Julia evaluates as compiled code" begin
    expressions = [
        :(mean(log.(conc))),
        :(SCALE * std(y; corrected=false)),
        :(square_sum(x[begin:end-1]) / length(x)),
        :(sum([2v for v in c if v > 0])),
        :(sum(Iterators.flatten((v, w) for v in x for w in c[1:2]))),
        :(maximum(v -> abs(v - 1), conc)),
        :(first(sort(conc; rev=true))),
        :(length(x) > 3 ? mean(x) : 0.0),
        :((; a=1.5, b=2.5).b + Tuple(x)[2]),
        :(sum(map(c) do v
            v^2 + 1
        end)),
        :(Float64(count(>(0), x) + sum([1 2; 3 4]) + only([1;;]) + sum(Float64[1 2; 3 4]))),
        :(parse(Float64, "$(length(x)).5")),
    ]
    for expr in expressions
        source = quote
            k = constant($expr)
            sigma ~ Exponential(1)
            mu ~ 1 + x
            m = mu + k
            y ~ Normal(m, sigma)
        end
        @test code_and_data(brm(source; mod=@__MODULE__)(DATA)) ==
              code_and_data(compiled(source)(DATA))
    end
end

@testset "definitions and mutation are refused" begin
    # Every name these read is a data field, so each reaches its evaluation.
    for (expr, refused) in ((:(begin global x = 1.0 end), "global x = 1.0"),
                            (:(begin f(x) = x; f(1.0) end), "`f(x) = begin"),
                            (:(begin x[1] = 0.0; sum(x) end), "x[1] = 0.0"),
                            (:(sum(x) + (while false end; 0.0)), "while false"))
        source = quote
            k = constant($(Base.remove_linenums!(expr)))
            sigma ~ Exponential(1)
            y ~ Normal(k, sigma)
        end
        @test_throws refused sb(brm(source; mod=@__MODULE__)(DATA))
    end
    @test_throws "defines a function" brm(:(model(df) = begin y ~ Normal(0, 1) end);
                                          mod=@__MODULE__)
    @test_throws UndefKeywordError brm(:(y ~ Normal(0, 1)))
end

# A generative plan, a descriptor and a per-chain copy `deepcopy` the model. A
# `brm` model holds a constant's evaluator as an interpreted closure over the
# builder's scope, which holds `mod`, and `deepcopy` refuses a module (snag
# brm-source-mod-c-37a8e5ec). A copy holds it as it holds a compiled closure:
# code and module shared, captured variables copied.
interpret(x) = BRM._brm_interpret(BRM._brm_expand(x, @__MODULE__), BRM._BRMScope(@__MODULE__))
@testset "copies hold interpreted closures as compiled ones" begin
    # A closure that captures nothing is its own copy, a module in its body too.
    pure = interpret(:(v -> v + 0 * length(names(@__MODULE__))))
    @test deepcopy(pure) === pure
    # The control: Base's field-by-field copy of the same closure reaches `mod`.
    @test_throws "deepcopy of Modules not supported" invoke(
        Base.deepcopy_internal, Tuple{Any,IdDict}, pure, IdDict())
    # A captured variable is copied, so the copy's is its own; `M` is in scope
    # but unread, so it is not captured (a module, which no copy could hold).
    counter = interpret(:(let M = Base, n = [0], k = 2
        () -> push!(n, k)
    end))
    twin = deepcopy(counter)
    @test counter() == [0, 2]
    @test twin() == [0, 2]
    @test counter() == [0, 2, 2]
    # Closures over one scope still share it after one copy.
    add, current = deepcopy(interpret(:(let n = [0]
        (() -> push!(n, 1), () -> n)
    end)))
    add()
    @test current() == [0, 1]
end

@testset "a brm model is copied, planned, described and replayed" begin
    frame = (; x=DATA.x[1:12], subj=DATA.subj[1:12], y=DATA.y[1:12])
    results = map((brm(FAMILY; mod=@__MODULE__), compiled(FAMILY))) do builder
        brmi = builder(DATA)
        model = sb(brmi)
        @test code_and_data(deepcopy(brmi)) == code_and_data(brmi)
        plan = generative_plan(model)
        descriptor = brm_descriptor(model)
        replay = reprocess(model, frame)
        @test BRM.stan_data(replay)[:log_s] == BRM.stan_data(model)[:log_s]
        (; declarations=[(d.role, d.target) for d in plan.declarations],
           formula=descriptor.formula,
           outputs=[(o.name, o.role, o.size) for o in brm_outputs(descriptor)],
           replay=(BRM.stan_code(replay), BRM.stan_data(replay)))
    end
    @test results[1] == results[2]
end

# Every `@brm` block of this directory: the two builders agree on the columns,
# the printed source and the built model (empty data and placeholder columns),
# or fail with the same error.
module Sweep
using BayesianRegressionModels, Distributions, Statistics, StanBlocks, LinearAlgebra
end
function brm_blocks!(found, x)
    if Meta.isexpr(x, :macrocall) && x.args[1] === Symbol("@brm")
        args = [a for a in x.args[2:end] if !(a isa LineNumberNode)]
        source = last(args)
        Meta.isexpr(source, :block) && push!(found, source)
    end
    x isa Expr && foreach(a -> brm_blocks!(found, a), x.args)
    found
end
const SWEEP_FILES = get(ENV, "BRM_BUILDER_SWEEP", "") == "all" ?
    sort(filter(endswith(".jl"), readdir(@__DIR__))) :
    !isempty(get(ENV, "BRM_BUILDER_SWEEP", "")) ? split(ENV["BRM_BUILDER_SWEEP"], ',') :
    ["model_family_composition.jl", "fixed_ranef_scales.jl", "plate_grouping.jl", "fitted_constants.jl",
     "effect_priors.jl", "ranef_effect_priors.jl", "vector_parameters.jl",
     "correlated_outcomes.jl", "closure_arguments.jl", "lhs_broadcast.jl",
     "level_correlation.jl", "hsgp_options.jl", "dotted_operators.jl"]
@testset "sweep: every @brm block builds the same model" begin
    counts = Dict(:built => 0, :same_error => 0)
    for file in SWEEP_FILES
        path = joinpath(@__DIR__, file)
        isfile(path) || continue
        for source in brm_blocks!(Any[], Meta.parseall(read(path, String)))
            reference = try
                Base.invokelatest(Core.eval, Sweep, BRM._brm(source))
            catch
                continue                        # refused when parsed, by both routes
            end
            interpreted = brm(source; mod=Sweep)
            columns = brm_columns(reference)
            placeholders = NamedTuple{columns}(ntuple(_ -> [1.0, 2.0], length(columns)))
            ok = brm_columns(interpreted) == columns &&
                 sprint(show, interpreted) == sprint(show, reference)
            for df in ((;), placeholders)
                a, b = outcome(interpreted, df), outcome(reference, df)
                ok &= a isa String ? a == b : same(a, b)
                counts[a isa String ? :same_error : :built] += 1
            end
            ok || @info "builders differ" file source
            @test ok
        end
    end
    @info "sweep" SWEEP_FILES counts
    @test counts[:built] > 0
end

# Fragments a package defines compose in another package's precompile
# workload. While an image is generated Julia refuses `eval` into every other
# module, so the evaluating route fails there.
const FRAGMENTS_SOURCE = """
module FamilyFragments
using BayesianRegressionModels, Distributions
const DATA = (; x=[0.1, 0.4, 0.9, 1.3], y=[1.0, 1.2, 1.9, 2.4])
const PREDICTOR = @brm begin mu ~ 1 + x end
const OBSERVATION = @brm begin
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
end
"""
const COMPOSED_SOURCE = """
module FamilyComposed
using BayesianRegressionModels, Distributions
import FamilyFragments
const MERGED = merge(FamilyFragments.PREDICTOR, FamilyFragments.OBSERVATION)(FamilyFragments.DATA)
slope_scale() = 0.5
const BUILT = brm(quote
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
    effect(mu, x) ~ Normal(0, \$(slope_scale()))
end; mod=@__MODULE__)(FamilyFragments.DATA)
const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
end
"""
const EVALUATED_SOURCE = """
module FamilyEvaluated
using BayesianRegressionModels, Distributions
import FamilyFragments
const BUILT = Core.eval(FamilyFragments, BayesianRegressionModels._brm(:(begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end)))(FamilyFragments.DATA)
end
"""
function write_package(root, name, uuid, deps, source)
    dir = joinpath(root, name)
    mkpath(joinpath(dir, "src"))
    write(joinpath(dir, "Project.toml"), sprint(TOML.print, Dict(
        "name" => name, "uuid" => uuid, "version" => "0.1.0", "deps" => deps)))
    write(joinpath(dir, "src", name * ".jl"), source)
    dir
end
function copy_environment(env, project=Base.active_project())
    source = dirname(project)
    cp(project, joinpath(env, "Project.toml"))
    manifest = TOML.parsefile(joinpath(source, "Manifest.toml"))
    for entries in values(manifest["deps"]), entry in entries
        haskey(entry, "path") && (entry["path"] = abspath(source, entry["path"]))
    end
    open(io -> TOML.print(io, manifest), joinpath(env, "Manifest.toml"), "w")
end
function run_julia(env, code)
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env -e $code`,
        "JULIA_PKG_PRECOMPILE_AUTO" => "0")
    out = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    process.exitcode == 0 && process.termsignal == 0, String(take!(out))
end
dep(name) = name => string(Base.identify_package(name).uuid)

@testset "fragments compose in another package's precompile workload" begin
    mktempdir() do root
        env = mkpath(joinpath(root, "env"))
        copy_environment(env)
        common = Dict(dep.(["BayesianRegressionModels", "Distributions"]))
        uuids = Dict("FamilyFragments" => "6b2d8f3c-7e40-4c9b-8d52-3f8e1c7a4b21",
                     "FamilyComposed" => "6b2d8f3c-7e40-4c9b-8d52-3f8e1c7a4b22",
                     "FamilyEvaluated" => "6b2d8f3c-7e40-4c9b-8d52-3f8e1c7a4b23")
        fragments = write_package(root, "FamilyFragments", uuids["FamilyFragments"], common,
                                  FRAGMENTS_SOURCE)
        imports = merge(common, Dict("FamilyFragments" => uuids["FamilyFragments"]))
        composed = write_package(root, "FamilyComposed", uuids["FamilyComposed"], imports,
                                 COMPOSED_SOURCE)
        evaluated = write_package(root, "FamilyEvaluated", uuids["FamilyEvaluated"], imports,
                                  EVALUATED_SOURCE)
        ok, log = run_julia(env, """
            using Pkg
            Pkg.develop([PackageSpec(path=p) for p in $(repr([fragments, composed, evaluated]))])
            """)
        ok || print(log)
        @test ok
        ok, log = run_julia(env, "using Pkg; Pkg.precompile(\"FamilyComposed\")")
        ok || print(log)
        @test ok
        ok, log = run_julia(env, """
            using Test, BayesianRegressionModels, Distributions, StanBlocks
            using FamilyComposed
            @test FamilyComposed.WAS_PRECOMPILED
            data = FamilyComposed.FamilyFragments.DATA
            code(m) = BayesianRegressionModels.stan_code(SBBRMI(m; mod=Main))
            @test code(FamilyComposed.MERGED) == code(@brm data begin
                mu ~ 1 + x
                sigma ~ Exponential(1)
                y ~ Normal(mu, sigma)
            end)
            @test code(FamilyComposed.BUILT) == code(@brm data begin
                mu ~ 1 + x
                sigma ~ Exponential(1)
                y ~ Normal(mu, sigma)
                effect(mu, x) ~ Normal(0, 0.5)
            end)
            """)
        ok || print(log)
        @test ok
        # The control: the evaluating route cannot be precompiled.
        ok, log = run_julia(env, "using Pkg; Pkg.precompile(\"FamilyEvaluated\")")
        @test !ok
        @test occursin("closed module", log)
    end
end
