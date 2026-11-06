# bench/svi_target_reuse/split.jl — where the time of a minibatch SVI fit and its
# continuation goes, on the public toy models of test/nlme_svi*_fixture.jl.
#
#     julia --startup-file=no --project=test bench/svi_target_reuse/split.jl <family> [out.jsonl]
#
# Run one process per family: the first calls are the point, and they depend on
# what the process has already compiled. Each row is one timed step in the order
# it ran: elapsed seconds, Julia's own compile time inside it (`@counted`; Enzyme's
# work in GPUCompiler and LLVM is only partly counted there, so `first - warm` of a
# gradient is the number to read for it), allocated bytes, and how many repeats a
# `warm` row is the median of. Nothing here asserts a speed.
const FAMILY = get(ARGS, 1, "gauss")
# Julia 1.10's `@timed` has no compile_time, so read the counter `@time` prints from.
Base.cumulative_compile_timing(true)
macro counted(ex)
    quote
        local before = Base.cumulative_compile_time_ns()[1]
        local s = @timed $(esc(ex))
        merge(s, (; compile_time=(Base.cumulative_compile_time_ns()[1] - before) / 1e9))
    end
end
const OUT = get(ARGS, 2, "")
const ROWS = NamedTuple[]
function row(step, seconds; compile=NaN, bytes=0, repeats=1, note="")
    r = (; family=FAMILY, step, seconds=round(seconds; sigdigits=5),
        compile=round(compile; sigdigits=4), bytes, repeats, note)
    push!(ROWS, r)
    println(rpad(step, 34), lpad(string(r.seconds), 11), " s  compile ", lpad(string(r.compile), 9),
        "  ", lpad(string(round(bytes / 2^20; digits=1)), 9), " MiB", repeats > 1 ? "  median of $repeats" : "",
        isempty(note) ? "" : "  # $note")
    flush(stdout)
    r
end
macro step(name, ex)
    quote
        local s = @counted $(esc(ex))
        row($(esc(name)), s.time; compile=s.compile_time, bytes=s.bytes)
        s.value
    end
end
median_of(f, n) = (t = [(@elapsed f()) for _ in 1:n]; sort!(t); t[(n + 1) ÷ 2])

load = @counted @eval begin
    using BayesianRegressionModels
    import ReactiveKernels, ReactiveKernelsPPL, Enzyme    # the extensions' triggers
    using DifferentiationInterface: AutoEnzyme
    using Random, Statistics, Serialization
end
row("load", load.time; compile=load.compile_time, bytes=load.bytes)
const BRM = BayesianRegressionModels
const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const TEST = joinpath(@__DIR__, "..", "..", "test")
include(joinpath(TEST, "nlme_svi_fixture.jl"))
include(joinpath(TEST, "nlme_svi_owned_fixture.jl"))

# (builder of the model on a dataset, the datasets: A is what the rows are timed
# on, B the same sizes with other values, C other sizes, prepare's options)
function family(name)
    if name == "gauss"       # one subject draw block per unit, standard normal priors
        return (; build=svi_gauss_model, options=(;),
            A=svi_gauss_data(Xoshiro(1), 200), B=svi_gauss_data(Xoshiro(2), 200),
            C=svi_gauss_data(Xoshiro(3), 300; nobs=8))
    elseif name == "owned"   # authored unit priors: ordinary coordinates join their unit
        return (; build=svi_owned_model, options=(;),
            A=svi_owned_data(collect(1:96)), B=merge(svi_owned_data(collect(1:96)),
                (; y=[[0.3i, 0.1, -0.4] for i in 1:96])),
            C=svi_owned_data(collect(1:128)))
    elseif name == "connected"  # few large units linked by shared rows
        linked(n) = merge(svi_owned_data(collect(1:n)), (; left=collect(1:n),
            right=[i % 12 == 0 ? i : i + 1 for i in 1:n], weight=fill(0.3, n)))
        return (; build=svi_owned_connected_model, options=(; unit_partition=:connected),
            A=linked(120), B=merge(linked(120), (; y=[[0.3i, 0.1, -0.4] for i in 1:120])),
            C=linked(180))
    elseif name == "spacetime"  # grouped HSGP over a day axis, AR(1) cells, per-probe units
        field = :(hsgp(day; k=4, by=probe, cor=multigroup((probe_x, probe_y); k=3)))
        return (; build=svi_spacetime_model(field), options=(;),
            A=svi_spacetime_data(Xoshiro(1); G=8, T=30), B=svi_spacetime_data(Xoshiro(2); G=8, T=30),
            C=svi_spacetime_data(Xoshiro(3); G=12, T=40))
    end
    error("unknown family $name")
end

const F = family(FAMILY)
rk = @step "lower: first RKBRMI" RKBRMI(F.build(F.A))
m = @step "prepare: first brm_nlme_model" brm_nlme_model(rk; ad_backend=AD, F.options...)
first_value(x) = brm_nlme_loglikelihood(x, 1, x.unit[x.theta], x.unit[x.sigma],
    x.unit[x.view.subject_coordinates[1]])
@step "value: first likelihood" first_value(m)
row("value: warm likelihood", median_of(() -> first_value(m), 21); repeats=21)
println("# coordinates=", length(m.unit), " units=", length(m.view.levels))
row("info: coordinates", 0.0; note="$(length(m.unit)) packed, $(length(m.view.levels)) units")

# The first target, piece by piece, then whole.
o = @step "target: first ownership" BRM._brm_svi_ownership(m)
plan = @step "target: first partition plan" BRM._brm_nlme_partition_plan(m.cell_gradient, m.unit,
    o.globals, m.view; units=o.units)
if o.units != m.view.subject_coordinates
    @step "target: first authored prior" BRM._brm_svi_authored_prior(m, o.globals, o.units; factors=o.factors)
else
    row("target: first authored prior", 0.0; note="standard normal units: none")
end
t = @step "target: first whole (pieces warm)" BRM._brm_svi_target(m)
worker = @step "workers: first worker plan" BRM._brm_nlme_worker_plan(t.plan, m.cell_gradient, m.unit)

# First and warm evaluations on this target.
g = copy(m.unit[t.globals])
z = copy(m.unit[t.units[1]])
gg, gz = zeros(length(g)), zeros(length(z))
@step "gradient: first global term" BRM._brm_svi_global_term(t, g)
row("gradient: warm global term", median_of(() -> BRM._brm_svi_global_term(t, g), 21); repeats=21)
@step "gradient: first unit term" BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z)
row("gradient: warm unit term", median_of(() -> BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z), 21); repeats=21)
@step "gradient: first on a worker plan" BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z, worker)

# The same program prepared again in this process: what a continuation repeated
# before this change. Each piece warm, then the whole, then its first gradient.
@step "repeat: ownership" BRM._brm_svi_ownership(m)
@step "repeat: partition plan" BRM._brm_nlme_partition_plan(m.cell_gradient, m.unit, o.globals, m.view; units=o.units)
if o.units != m.view.subject_coordinates
    @step "repeat: authored prior" BRM._brm_svi_authored_prior(m, o.globals, o.units; factors=o.factors)
end
@step "repeat: worker plan" BRM._brm_nlme_worker_plan(t.plan, m.cell_gradient, m.unit)
target_again = @step "repeat: whole target" BRM._brm_svi_target(m)
@step "repeat: first gradient on it" BRM._brm_svi_unit_term!(gg, gz, target_again, 1, g, z, target_again.plan)
row("repeat: whole target", median_of(() -> BRM._brm_svi_target(m), 5); repeats=5)
@step "identity: first" BRM._brm_svi_identity(t)
row("identity: warm", median_of(() -> BRM._brm_svi_identity(t), 5); repeats=5)

# The public calls. `prepared = nothing` before a call reproduces the previous
# behaviour, where every call built its target.
controls = (; batch_size=min(32, length(t.units)), rng=Xoshiro(11), threads=1, ndraws=0)
empty_bytes = Base.summarysize(m)
first_fit = @counted brm_svi_fit(m; controls..., stop_after=1)
row("fit: first call, 1 iteration", first_fit.time; compile=first_fit.compile_time, bytes=first_fit.bytes)
d = first_fit.value.diagnostics
row("fit: its target phase", d.target.seconds; note="reused=$(d.target.reused)")
row("fit: its first iteration", d.seconds)
retained = Base.summarysize(m) - empty_bytes
row("retained: target in the model", 0.0; bytes=retained, note="summarysize with minus without")
warm = brm_svi_resume(m, first_fit.value; stop_after=200, ndraws=0)
row("iteration: warm mean over 200", warm.diagnostics.seconds / 200; repeats=200)
row("resume: its target phase", warm.diagnostics.target.seconds; note="reused=$(warm.diagnostics.target.reused)")
state = brm_svi_optimizer_checkpoint(warm)
reused = [brm_svi_resume(m, state; stop_after=0, ndraws=0).diagnostics.target for _ in 1:5]
@assert all(r -> r.reused, reused)
row("resume: reused target phase", sort!([r.seconds for r in reused])[3]; repeats=5)
rebuilt = map(1:5) do _
    m.svi_target.prepared = nothing
    brm_svi_resume(m, state; stop_after=0, ndraws=0).diagnostics.target
end
@assert !any(r -> r.reused, rebuilt)
row("resume: rebuilt target phase", sort!([r.seconds for r in rebuilt])[3]; repeats=5, note="as before this change")

# One snapshot: factors, and with 50 draws.
observe0 = BRM._brm_svi_observer(_ -> nothing, m.svi_target.prepared.target, state, 0)
observe50 = BRM._brm_svi_observer(_ -> nothing, m.svi_target.prepared.target, state, 50)
observe0(:running); observe50(:running)
row("snapshot: factors only", median_of(() -> observe0(:running), 5); repeats=5)
row("snapshot: with 50 draws", median_of(() -> observe50(:running), 5); repeats=5)
seen = Ref{Any}(nothing)
BRM._brm_svi_observer(s -> (seen[] = s), m.svi_target.prepared.target, state, 50)(:running)
bytes = (io = IOBuffer(); serialize(io, seen[]); position(io))
row("snapshot: serialized size", 0.0; bytes, note="with 50 draws")

# Other datasets in this process, then the first one again: each is a new model of
# the same program with fresh optimizer state. Everything above has been compiled,
# so what remains is what binding this data costs. A retained target serves its own
# model only, so each of these builds its own. Returning to A must reproduce A's
# first step bit for bit, which a stale binding or stale state would not.
reference_q = first_fit.value.diagnostics.q
for (name, data) in (("B same sizes", F.B), ("C other sizes", F.C), ("A again", F.A))
    rk2 = @step "$name: bind RKBRMI" RKBRMI(F.build(data))
    m2 = @step "$name: brm_nlme_model" brm_nlme_model(rk2; ad_backend=AD, F.options...)
    @step "$name: first value" first_value(m2)
    t3 = @step "$name: target" BRM._brm_svi_target(m2)
    g2, z2 = copy(m2.unit[t3.globals]), copy(m2.unit[t3.units[1]])
    gg2, gz2 = zeros(length(g2)), zeros(length(z2))
    @step "$name: first prior gradient" BRM._brm_svi_global_term(t3, g2)
    @step "$name: first unit gradient" BRM._brm_svi_unit_term!(gg2, gz2, t3, 1, g2, z2)
    worker2 = @step "$name: worker plan" BRM._brm_nlme_worker_plan(t3.plan, m2.cell_gradient, m2.unit)
    @step "$name: first worker gradient" BRM._brm_svi_unit_term!(gg2, gz2, t3, 1, g2, z2, worker2)
    row("$name: same types as A", 0.0; note=string("model ", typeof(rk2.model) == typeof(rk.model),
        ", sampler ", typeof(t3.plan.sampler) == typeof(t.plan.sampler),
        ", prior ", typeof(m2.population_prior) == typeof(m.population_prior)))
    fit2 = @counted brm_svi_fit(m2; controls..., rng=Xoshiro(11), stop_after=1)
    d2 = fit2.value.diagnostics
    row("$name: first step, whole call", fit2.time; compile=fit2.compile_time, bytes=fit2.bytes)
    row("$name: first step only", d2.seconds)
    same = length(d2.q.μ) == length(reference_q.μ) && d2.q.μ == reference_q.μ && d2.q.M == reference_q.M
    row("$name: first step equals A's", 0.0; note=string(same))
    warm2 = brm_svi_resume(m2, fit2.value; stop_after=50, ndraws=0)
    row("$name: warm mean over 50", warm2.diagnostics.seconds / 50; repeats=50,
        note="reused=$(warm2.diagnostics.target.reused)")
end

if !isempty(OUT)
    open(OUT, "a") do io
        for r in ROWS
            println(io, join((r.family, r.step, r.seconds, r.compile, r.bytes, r.repeats, r.note), '\t'))
        end
    end
end
