# bench/svi_target_reuse/image.jl — how much of a fixed model and its optimization
# loop a package image holds, across processes and datasets.
#
#     julia --startup-file=no --project=test bench/svi_target_reuse/image.jl <dir> [out.tsv] [rounds] [tiers]
#
# One throwaway package per tier, in a temporary copy of the test environment:
#
#   none    defines the model, and precompiles nothing of it
#   values  plus a whole RKBRMI and a value read in its @compile_workload
#   target  plus brm_nlme_model and a zero-iteration brm_svi_fit (the target, no gradient)
#   loop    plus a real brm_svi_fit: gradients, Adam, unit steps, polishing, draws
#
# Then fresh processes fit dataset A (the workload's data), B (other values, the
# same sizes) or C (other sizes), with every phase timed and `q` printed so that
# tiers can be compared bit for bit. Tiers and datasets interleave within a round,
# to spread host load over them.
using Pkg, TOML

const ROOT = abspath(get(ARGS, 1, mktempdir()))
const OUT = abspath(get(ARGS, 2, joinpath(ROOT, "image.tsv")))
const ROUNDS = parse(Int, get(ARGS, 3, "2"))
const TIERS = split(get(ARGS, 4, "none,values,target,loop"), ',')
const TESTENV = dirname(Base.active_project())

const MODEL = """
    const BRM = BayesianRegressionModels
    const AD = AutoEnzyme(; mode=Enzyme.Reverse)
    # A toy with a correlated subject block and a per-subject plate.
    model(data) = @brm data begin
        sigma ~ Exponential(1)
        a ~ 1 + (1 | g | subject)
        b ~ 1 + (1 | g | subject)
        @plate for i in eachindex(a)
            y[i] ~ Normal(a[i] .+ b[i] .* x[i], sigma)
        end
    end
    function dataset(seed, N, nobs)
        rng = Xoshiro(seed)
        x = [randn(rng, nobs) for _ in 1:N]
        (; subject=["s\$i" for i in 1:N], x,
            y=[1 + randn(rng) .+ (0.5 + 0.3randn(rng)) .* x[i] .+ 0.5 .* randn(rng, nobs) for i in 1:N])
    end
    const WORKLOAD_DATA = dataset(1, 20, 5)
    const IN_IMAGE = Ref(false)
"""

const WORKLOADS = Dict(
    "none" => "",
    "values" => """
        @compile_workload begin
            rk = RKBRMI(model(WORKLOAD_DATA))
            brm_nlme_view(rk)
        end
        """,
    "target" => """
        @compile_workload begin
            rk = RKBRMI(model(WORKLOAD_DATA))
            m = brm_nlme_model(rk; ad_backend=AD)
            brm_svi_fit(m; rng=Xoshiro(1), max_iterations=0, ndraws=2, threads=1)
        end
        """,
    "loop" => """
        @compile_workload begin
            rk = RKBRMI(model(WORKLOAD_DATA))
            m = brm_nlme_model(rk; ad_backend=AD)
            fit = brm_svi_fit(m; rng=Xoshiro(1), batch_size=4, window=5, min_windows=2, patience=1,
                polish_windows=1, max_iterations=30, ndraws=2, threads=1, stop_after=12)
            brm_svi_resume(m, fit; ndraws=2, threads=1, snapshot=_ -> nothing, snapshot_draws=1)
        end
        """)

package_source(tier) = """
module SVIImage$(uppercasefirst(tier))
using BayesianRegressionModels, PrecompileTools, Random
import ReactiveKernels, ReactiveKernelsPPL, Enzyme    # the extensions' triggers
using DifferentiationInterface: AutoEnzyme
$MODEL
$(WORKLOADS[tier])
end
"""

const CHILD = raw"""
const TIER, DATASET, ROUND = ARGS[1], ARGS[2], ARGS[3]
const OUT = ARGS[4]
const RESULT = NamedTuple[]
Base.cumulative_compile_timing(true)
macro counted(ex)
    quote
        local before = Base.cumulative_compile_time_ns()[1]
        local s = @timed $(esc(ex))
        merge(s, (; compile_time=(Base.cumulative_compile_time_ns()[1] - before) / 1e9))
    end
end
record(name, s) = (push!(RESULT, (; step=name, seconds=s.time, compile=s.compile_time, bytes=s.bytes)); s.value)
const NAME = Symbol("SVIImage", uppercasefirst(TIER))
load = @counted @eval begin
    using Random, Serialization, BayesianRegressionModels
    import ReactiveKernels, ReactiveKernelsPPL, Enzyme
    using $(NAME)
end
record("load", load)
P = getfield(Main, NAME)
const BRM = BayesianRegressionModels
data = (A=() -> P.dataset(1, 20, 5), B=() -> P.dataset(2, 20, 5), C=() -> P.dataset(3, 40, 8))[Symbol(DATASET)]()
rk = record("RKBRMI", @counted RKBRMI(P.model(data)))
m = record("brm_nlme_model", @counted brm_nlme_model(rk; ad_backend=P.AD))
t = record("first target", @counted BRM._brm_svi_target(m))
g, z = copy(m.unit[t.globals]), copy(m.unit[t.units[1]])
record("first prior gradient", @counted BRM._brm_svi_global_term(t, g))
gg, gz = zeros(length(g)), zeros(length(z))
record("first unit gradient", @counted BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z))
record("second unit gradient", @counted BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z))
fit = record("first svi call, 1 iteration", @counted brm_svi_fit(m; rng=Xoshiro(7), batch_size=4, threads=1,
    stop_after=1, ndraws=0))
push!(RESULT, (; step="its target phase", seconds=fit.diagnostics.target.seconds, compile=NaN, bytes=0))
push!(RESULT, (; step="its first iteration", seconds=fit.diagnostics.seconds, compile=NaN, bytes=0))
warm = record("100 warm iterations", @counted brm_svi_resume(m, fit; stop_after=100, ndraws=0))
draws = record("first draws", @counted brm_svi_resume(m, warm; stop_after=0, ndraws=20))
digest = bytes2hex(reinterpret(UInt8, warm.diagnostics.q.μ))[1:32]
# Linked from the image against compiled in this process, on an Enzyme that keeps them.
retained = isdefined(Enzyme.Compiler, :retained_derivative_stats) ?
    Enzyme.Compiler.retained_derivative_stats() : nothing
linked = retained === nothing ? "n/a" : string("compiled=", retained.compiled, " materialized=", retained.materialized,
    " rejected=", retained.rejected, " absent=", retained.absent)
facts = string("image_repair=", isdefined(Enzyme.Compiler, :image_split_thunks), " ", linked, " q=", digest,
    " coordinates=", length(m.unit), " reused=", warm.diagnostics.target.reused)
open(OUT, "a") do io
    for r in RESULT
        println(io, join((TIER, DATASET, ROUND, r.step, round(r.seconds; sigdigits=5),
            round(r.compile; sigdigits=4), r.bytes, facts), '\t'))
    end
end
println("OK ", TIER, " ", DATASET, " total ", round(sum(r.seconds for r in RESULT if !isnan(r.compile)); sigdigits=4), " s ", facts)
"""

function main()
    env = mkpath(joinpath(ROOT, "env"))
    cp(joinpath(TESTENV, "Project.toml"), joinpath(env, "Project.toml"); force=true)
    manifest = TOML.parsefile(joinpath(TESTENV, "Manifest.toml"))
    for entries in values(manifest["deps"]), entry in entries
        haskey(entry, "path") && (entry["path"] = abspath(TESTENV, entry["path"]))
    end
    open(io -> TOML.print(io, manifest), joinpath(env, "Manifest.toml"), "w")
    # Enzyme's libEnzyme build is a preference, part of what its image is keyed on.
    prefs = joinpath(TESTENV, "LocalPreferences.toml")
    isfile(prefs) && cp(prefs, joinpath(env, "LocalPreferences.toml"); force=true)
    uuid(name) = manifest["deps"][name][1]["uuid"]
    deps = Dict(name => uuid(name) for name in ("BayesianRegressionModels", "ReactiveKernels",
        "ReactiveKernelsPPL", "Enzyme", "DifferentiationInterface", "PrecompileTools", "Random"))
    paths = String[]
    for (k, tier) in enumerate(TIERS)
        name = "SVIImage" * uppercasefirst(tier)
        dir = mkpath(joinpath(ROOT, name, "src"))
        write(joinpath(ROOT, name, "Project.toml"), sprint(TOML.print, Dict("name" => name,
            "uuid" => "6b2d9e10-5c1f-4a77-8e21-3f0a7c9d10$(lpad(string(k), 2, '0'))",
            "version" => "0.1.0", "deps" => deps)))
        write(joinpath(dir, name * ".jl"), package_source(tier))
        push!(paths, joinpath(ROOT, name))
    end
    child = joinpath(ROOT, "child.jl")
    write(child, CHILD)
    julia(args...) = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env $(args)`,
        "JULIA_PKG_PRECOMPILE_AUTO" => "0")
    run(julia("-e", "using Pkg; Pkg.develop([PackageSpec(path=p) for p in $(repr(paths))])"))
    open(OUT, "a") do io
        println(io, join(("tier", "dataset", "round", "step", "seconds", "compile", "bytes", "facts"), '\t'))
    end
    for tier in TIERS
        name = "SVIImage" * uppercasefirst(tier)
        started = time()
        ok = success(pipeline(julia("-e", "using Pkg; Pkg.precompile(\"$name\")"); stdout=stdout, stderr=stderr))
        seconds = round(time() - started; sigdigits=4)
        size_bytes = sum(filesize, filter(endswith(".so"), readdir(joinpath(DEPOT_PATH[1], "compiled",
            "v$(VERSION.major).$(VERSION.minor)", name); join=true)); init=0)
        println("PRECOMPILE ", tier, " ok=", ok, " seconds=", seconds, " image_bytes=", size_bytes)
        open(io -> println(io, join((tier, "-", 0, "precompile", seconds, NaN, size_bytes, "ok=$ok"), '\t')), OUT, "a")
        flush(stdout)
    end
    for round in 1:ROUNDS, dataset in ("A", "B", "C"), tier in TIERS
        p = run(pipeline(ignorestatus(julia(child, tier, dataset, string(round), OUT)); stdout=stdout, stderr=stderr))
        p.exitcode == 0 && p.termsignal == 0 ||
            (open(io -> println(io, join((tier, dataset, round, "FAILED", NaN, NaN, 0,
                "exit=$(p.exitcode) signal=$(p.termsignal)"), '\t')), OUT, "a");
             println("FAILED ", tier, " ", dataset, " exit=", p.exitcode, " signal=", p.termsignal))
        flush(stdout)
    end
end
main()
