# A running or continuing minibatch SVI fit (src/nlme_svi.jl): a prepared model
# keeps its target and worker samplers for its next call in this process, and a
# fit hands out snapshots of itself without returning. Neither may change the run:
# the reference for every comparison is a run on a model that has never run
# anything, with nothing observing it.
#
# RUN: julia --project=test test/nlme_svi_running.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
using LinearAlgebra, Random, Serialization
import Treebars

# A short exploration (at least six windows of 20 iterations) and polishing
# (four windows), so a run converges in a few hundred iterations.
const SVI_RUN = (; batch_size=4, window=20, min_windows=6, patience=3,
    polish_windows=4, max_iterations=3000)

svi_prepared(data) = brm_nlme_model(RKBRMI(svi_gauss_model(data)); ad_backend=NLME_AD)
svi_held(m) = m.svi_target.prepared

@stestset "a continuation on the prepared model builds no target and is the same run" begin
    data = svi_gauss_data(Xoshiro(101), 12)
    reference = brm_svi_fit(svi_prepared(data); rng=Xoshiro(102), SVI_RUN..., ndraws=30)
    @test reference.diagnostics.converged

    m = svi_prepared(data)
    @test svi_held(m) === nothing
    first_fit = brm_svi_fit(m; rng=Xoshiro(102), SVI_RUN..., stop_after=37, ndraws=5)
    @test !first_fit.diagnostics.target.reused
    held = svi_held(m)
    @test held !== nothing && held.target.m === m
    # Every later call runs on these objects: nothing is constructed.
    second = brm_svi_resume(m, first_fit; stop_after=23, ndraws=5)
    @test second.diagnostics.target.reused && second.diagnostics.target.workers_built == 0
    @test svi_held(m) === held
    third = brm_svi_resume(second.diagnostics.model, second; stop_after=100, ndraws=5)
    @test third.diagnostics.target.reused && svi_held(m) === held
    final = brm_svi_resume(m, third; ndraws=30)
    @test svi_held(m) === held
    @test svi_same_result(final, reference)
    @test svi_same_state(brm_svi_optimizer_checkpoint(final),
        brm_svi_optimizer_checkpoint(reference))

    # A new fit of the model, on another stream or start, shares nothing with the
    # last one but the target: it is the run a fresh model gives.
    other = brm_svi_fit(m; rng=Xoshiro(103), SVI_RUN..., stop_after=50, ndraws=5)
    @test other.diagnostics.target.reused && svi_held(m) === held
    @test svi_same_result(other, brm_svi_fit(svi_prepared(data); rng=Xoshiro(103), SVI_RUN...,
        stop_after=50, ndraws=5))

    # A model in the result of an unprepared input continues the same way, and
    # an RKBRMI or BRMI prepares a model, and so a target, on every call.
    rk = RKBRMI(svi_gauss_model(data))
    started = brm_svi_fit(rk; ad_backend=NLME_AD, rng=Xoshiro(102), SVI_RUN...,
        stop_after=37, ndraws=0)
    @test !started.diagnostics.target.reused
    @test brm_svi_resume(started.diagnostics.model, started; stop_after=1,
        ndraws=0).diagnostics.target.reused
    @test !brm_svi_resume(rk, started; ad_backend=NLME_AD, stop_after=1,
        ndraws=0).diagnostics.target.reused
end

@stestset "a changed model is refused, and a refusal keeps the target" begin
    data = svi_gauss_data(Xoshiro(201), 8)
    m = svi_prepared(data)
    fit = brm_svi_fit(m; rng=Xoshiro(202), SVI_RUN..., stop_after=30, ndraws=0)
    held = svi_held(m)
    state = brm_svi_optimizer_checkpoint(fit)
    calls = Ref(0)
    tally(_) = (calls[] += 1)
    # Other data and another program, each a model that holds a target of its own
    # and has already run.
    other_data = svi_prepared(svi_gauss_data(Xoshiro(203), 8))
    brm_svi_fit(other_data; rng=Xoshiro(204), SVI_RUN..., stop_after=2, ndraws=0)
    @test svi_held(other_data) !== nothing
    @test_throws "bound data" brm_svi_resume(tally, other_data, state)
    @test brm_svi_resume(other_data, brm_svi_fit(other_data; rng=Xoshiro(204), SVI_RUN...,
        stop_after=1, ndraws=0); stop_after=1, ndraws=0).diagnostics.target.reused
    changed = brm_nlme_model(RKBRMI(@brm data begin
        sigma ~ Exponential(2)
        a ~ 1 + (1 | g | subject)
        b ~ 1 + (1 | g | subject)
        @plate for i in eachindex(a)
            y[i] ~ Normal(a[i] .+ b[i] .* x[i], sigma)
        end
    end); ad_backend=NLME_AD)
    @test_throws "emitted program" brm_svi_resume(tally, changed, state)
    # A checkpoint refused on the model it came from costs that model nothing.
    for (part, message) in ((:grouping, "groupings"), (:ownership, "each subject owns"),
            (:fixed, "fixed shared coordinates"))
        forged = BRMSVIOptimizerState(state.format,
            merge(state.identity, NamedTuple{(part,)}(("forged",))), state.previous,
            state.controls, state.run)
        @test_throws message brm_svi_resume(tally, m, forged)
        @test svi_held(m) === held
    end
    short = brm_svi_optimizer_checkpoint(state)
    pop!(short.run.trace)
    @test_throws "inconsistent" brm_svi_resume(tally, m, short)
    @test svi_held(m) === held
    @test calls[] == 0
end

@stestset "other ownership or another carry gets its own target" begin
    data = svi_gauss_data(Xoshiro(301), 8)
    m = svi_prepared(data)
    plain = brm_svi_fit(m; rng=Xoshiro(302), SVI_RUN..., stop_after=5, ndraws=0)
    held = svi_held(m)
    # Promoting a coordinate changes the target; its continuation finds it.
    name = m.view.coordinates[first(m.view.subject_coordinates[1])]
    promoted = brm_svi_fit(m; rng=Xoshiro(302), SVI_RUN..., stop_after=5, ndraws=0,
        global_coordinates=[name])
    @test !promoted.diagnostics.target.reused && svi_held(m) !== held
    @test length(promoted.diagnostics.globals) > length(plain.diagnostics.globals)
    @test brm_svi_resume(m, promoted; stop_after=1,
        ndraws=0).diagnostics.target.reused

    # A window fit's carry is part of its target, and a continuation of the window
    # is the run that never stopped.
    initial = brm_svi_fit(svi_stream_model(svi_stream_data(Xoshiro(303), ["a", "b", "c"], 4));
        rng=Xoshiro(304), max_iterations=25, ndraws=0)
    carry = brm_svi_checkpoint(initial)
    window = svi_stream_data(Xoshiro(305), ["b", "c", "new"], 4)
    controls = (; batch_size=2, window=10, min_windows=4, patience=2,
        polish_windows=3, max_iterations=1000)
    prepare() = brm_nlme_model(RKBRMI(svi_stream_model(window)); ad_backend=NLME_AD)
    full = brm_svi_fit(carry, prepare(); rng=Xoshiro(306), controls..., ndraws=20)
    w = prepare()
    part = brm_svi_fit(carry, w; rng=Xoshiro(306), controls..., stop_after=17, ndraws=0)
    @test !part.diagnostics.target.reused
    held = svi_held(w)
    rest = brm_svi_resume(w, part; ndraws=20)
    @test rest.diagnostics.target.reused && svi_held(w) === held
    @test svi_same_result(rest, full)
    @test svi_stream_same_factors(brm_svi_checkpoint(rest), brm_svi_checkpoint(full))
end

@stestset "a call that fails while optimizing leaves no target, and a copy holds none" begin
    data = svi_gauss_data(Xoshiro(401), 8)
    m = svi_prepared(data)
    @test_throws "boom" brm_svi_fit(_ -> error("boom"), m; rng=Xoshiro(402), SVI_RUN...,
        checkpoint_every=3, ndraws=0)
    @test svi_held(m) === nothing
    # The next call prepares again, and gives what a fresh model gives.
    fit = brm_svi_fit(m; rng=Xoshiro(402), SVI_RUN..., stop_after=9, ndraws=5)
    @test !fit.diagnostics.target.reused
    @test svi_same_result(fit, brm_svi_fit(svi_prepared(data); rng=Xoshiro(402), SVI_RUN...,
        stop_after=9, ndraws=5))
    # Workspace does not travel with the model's value.
    @test svi_held(m) !== nothing
    @test deepcopy(m.svi_target).prepared === nothing
    io = IOBuffer()
    serialize(io, m.svi_target)
    @test position(io) < 200
    @test deserialize(seekstart(io)).prepared === nothing
end

@stestset "an observed run is the run an unobserved one is" begin
    data = svi_gauss_data(Xoshiro(501), 12)
    for controls in ((; SVI_RUN...), (; SVI_RUN..., family=LowRankGlobal(1)))
        reference_states = BRMSVIOptimizerState[]
        reference = brm_svi_fit(s -> push!(reference_states, s), svi_prepared(data);
            checkpoint_every=1, rng=Xoshiro(502), controls..., ndraws=30)
        @test reference.diagnostics.converged
        P = reference.diagnostics.optimizer.run.polish_start
        states = BRMSVIOptimizerState[]
        snapshots = NamedTuple[]
        observed = brm_svi_fit(s -> push!(states, s), svi_prepared(data); checkpoint_every=1,
                rng=Xoshiro(502), controls..., ndraws=30, snapshot=s -> push!(snapshots, s),
                snapshot_every=1, snapshot_draws=7)
        # Iterates, Adam, stream, ELBO, stopping and final draws.
        @test svi_same_result(observed, reference)
        @test length(states) == length(reference_states) &&
            all(svi_same_state.(states, reference_states))

        n = reference.diagnostics.iterations
        @test [s.iteration for s in snapshots] == 1:n
        # No snapshot claims anything about the end of a run until it has ended.
        @test all(s -> !hasproperty(s, :finished) && !hasproperty(s, :converged), snapshots)
        @test all(s -> s.status === :running, snapshots[1:end-1])
        @test last(snapshots).status === :converged
        # What q is, by iteration. Settling starts with the window at which
        # exploration's test passed and holds the raw iterate, as exploring does.
        E = reference.diagnostics.stages.exploring
        @test 0 < E < P && reference.diagnostics.stages.settling == P - E
        @test all(s -> s.phase === :exploring && s.q_kind === :raw_iterate &&
            s.averaged_iterations == 0, snapshots[1:E-1])
        @test all(s -> s.phase === :settling && s.q_kind === :raw_iterate &&
            s.averaged_iterations == 0, snapshots[E:P-1])
        @test all(s -> s.phase === :polishing && s.q_kind === :polished_average &&
            s.averaged_iterations == s.iteration - P, snapshots[P:n])
        # A snapshot is what a result stopped at its iteration reports.
        stopper = svi_prepared(data)
        for k in (1, P - 1, P, P + 13, n)
            stopped = brm_svi_fit(stopper; rng=Xoshiro(502), controls..., stop_after=k, ndraws=7)
            s, d = snapshots[k], stopped.diagnostics
            @test s.draws == stopped.draws
            @test all(name -> isequal(getfield(s.q, name), getfield(d.q, name)),
                (:μ, :ρ, :Β, :M, :Λ, :U))
            @test s.elbo == last(d.elbo) && s.elbo_windows == d.elbo_windows
            @test s.elbo_windows_trimmed == d.elbo_windows_trimmed &&
                s.clipped_steps == d.clipped_steps
            @test s.coordinates == stopped.coordinates && s.globals == d.globals &&
                s.units == d.units && s.levels == d.model.view.levels
        end
        # The raw iterate is not the average, while polishing.
        @test snapshots[P+13].q.μ != states[P+13].run.q.μ
        @test snapshots[P+13].q.μ == states[P+13].run.average.μ
        # Earlier snapshots do not change as the run moves on.
        @test snapshots[1].q.μ != snapshots[2].q.μ
        @test snapshots[10].draws != snapshots[11].draws
    end
    # Scribbling on what it received cannot reach the run.
    reference = brm_svi_fit(svi_prepared(data); rng=Xoshiro(503), SVI_RUN..., ndraws=20)
    scribbled = brm_svi_fit(svi_prepared(data); rng=Xoshiro(503), SVI_RUN..., ndraws=20,
            snapshot_every=1, snapshot_draws=2, snapshot=function (s)
                fill!(s.q.μ, NaN)
                foreach(M -> fill!(M, NaN), s.q.M)
                foreach(U -> fill!(U, NaN), s.q.U)
                foreach(Λ -> fill!(Λ, NaN), s.q.Λ)
                fill!(s.draws, NaN)
                fill!(s.globals, 0)
                foreach(u -> fill!(u, 0), s.units)
                fill!(s.elbo_windows, NaN)
                fill!(s.elbo_windows_trimmed, NaN)
            end)
    @test svi_same_result(scribbled, reference)
end

@stestset "resume, window fits, boundaries, interrupts and refusals" begin
    data = svi_gauss_data(Xoshiro(601), 10)
    m = svi_prepared(data)
    full = brm_svi_fit(svi_prepared(data); rng=Xoshiro(602), SVI_RUN..., ndraws=20)

    # A fit and its continuations observe the same run, on the same target.
    seen = NamedTuple[]
    observe(s) = push!(seen, s)
    part = brm_svi_fit(m; rng=Xoshiro(602), SVI_RUN..., stop_after=25, ndraws=0,
        snapshot=observe, snapshot_every=10)
    @test [(s.iteration, s.status) for s in seen] ==
        [(10, :running), (20, :running), (25, :stopped)]
    empty!(seen)
    rest = brm_svi_resume(m, part; ndraws=20, snapshot=observe, snapshot_every=10)
    @test rest.diagnostics.target.reused
    @test svi_same_result(rest, full)
    @test first(seen).iteration == 30 && all(==(10), diff([s.iteration for s in seen])[1:end-1])
    @test last(seen).status === :converged && last(seen).iteration == full.diagnostics.iterations
    # The default cadence is the window, and with no snapshot_draws no draws.
    empty!(seen)
    brm_svi_fit(m; rng=Xoshiro(602), SVI_RUN..., stop_after=45, ndraws=0, snapshot=observe)
    @test [s.iteration for s in seen] == [20, 40, 45]
    @test all(s -> size(s.draws) == (length(s.coordinates), 0), seen)
    @test length(seen[1].elbo_windows) == 1 && length(seen[2].elbo_windows) == 2
    # A run that reaches its horizon says so, and an empty one has nothing yet.
    empty!(seen)
    brm_svi_fit(m; rng=Xoshiro(602), SVI_RUN..., max_iterations=3, ndraws=0, snapshot=observe)
    @test [(s.iteration, s.status) for s in seen] == [(3, :max_iterations)]
    empty!(seen)
    brm_svi_fit(m; rng=Xoshiro(602), SVI_RUN..., stop_after=0, ndraws=0, snapshot=observe)
    @test [(s.iteration, s.status, s.elbo) for s in seen] == [(0, :stopped, nothing)]

    # A cooperative interrupt reports the window it stops at, after `save`.
    order = Symbol[]
    empty!(seen)
    err = Treebars.with_progress(:state; description="snapshot interrupt") do p
        try
            brm_svi_fit(m; rng=Xoshiro(602), SVI_RUN..., checkpoint_every=30, progress=p,
                    ndraws=0, snapshot_every=30, snapshot=function (s)
                        push!(order, :snapshot)
                        push!(seen, s)
                        s.iteration == 30 && Treebars.request_interrupt!(p)
                    end) do _
                push!(order, :save)
            end
        catch e
            e
        end
    end
    @test err isa Treebars.ProgressInterrupted
    @test [(s.iteration, s.status) for s in seen] == [(30, :running), (40, :interrupted)]
    @test order == [:save, :snapshot, :save, :snapshot]

    # A window fit, whose absent units are still drawn.
    initial = brm_svi_fit(svi_stream_model(svi_stream_data(Xoshiro(603), ["a", "b", "c"], 4));
        rng=Xoshiro(604), max_iterations=25, ndraws=0)
    carry = brm_svi_checkpoint(initial)
    window = svi_stream_model(svi_stream_data(Xoshiro(605), ["b", "c", "new"], 4))
    controls = (; rng=Xoshiro(606), batch_size=2, max_iterations=12, min_windows=1)
    plain = brm_svi_fit(carry, window; controls..., rng=Xoshiro(606), ndraws=6)
    empty!(seen)
    observed = brm_svi_fit(carry, window; controls..., rng=Xoshiro(606), ndraws=6,
        snapshot=observe, snapshot_every=5, snapshot_draws=6)
    @test svi_same_result(observed, plain)
    @test svi_stream_same_factors(brm_svi_checkpoint(observed), brm_svi_checkpoint(plain))
    @test [s.iteration for s in seen] == [5, 10, 12]
    @test last(seen).draws == plain.draws

    # Refused before anything is prepared.
    @test_throws "need a `snapshot`" brm_svi_fit(data; snapshot_every=5)
    @test_throws "need a `snapshot`" brm_svi_fit(data; snapshot_draws=5)
    @test_throws "snapshot_every must be" brm_svi_fit(data; snapshot=identity, snapshot_every=0)
    @test_throws "snapshot_draws must be" brm_svi_fit(data; snapshot=identity, snapshot_draws=-1)
    @test_throws "a function or a file path" brm_svi_fit(data; snapshot=:watch)
    @test_throws "need a `snapshot`" brm_svi_resume(m, part; snapshot_draws=5)
end

@stestset "a path holds the latest snapshot, loads without BRM and stays bounded" begin
    data = svi_gauss_data(Xoshiro(701), 10)
    m = svi_prepared(data)
    mktempdir() do dir
        path = joinpath(dir, "watch", "latest.jls")
        sizes = Int[]
        iterations = Int[]
        latest(_) = (push!(sizes, filesize(path)); push!(iterations, deserialize(path).iteration))
        plain = brm_svi_fit(svi_prepared(data); rng=Xoshiro(702), SVI_RUN..., stop_after=205,
            ndraws=5)
        # `save` is called just before each snapshot is written, so it reads the one
        # before; the file is never partial and never more than one file.
        fit = brm_svi_fit(m; rng=Xoshiro(702), SVI_RUN..., stop_after=205, ndraws=5,
                snapshot=path, snapshot_every=20, snapshot_draws=5, checkpoint_every=20) do s
            s.run.iteration > 20 && latest(s)
            @test readdir(dirname(path)) == (s.run.iteration > 20 ? ["latest.jls"] : String[])
        end
        @test svi_same_result(fit, plain)
        @test iterations == 20:20:200
        @test readdir(dirname(path)) == ["latest.jls"]
        final = deserialize(path)
        @test final.iteration == 205 && final.status === :stopped
        @test final.draws == plain.draws
        # Bounded: a run's file grows by two numbers per window (the plain and the
        # trimmed window ELBO), not by iterations; here one window per snapshot.
        @test maximum(diff(sizes)) <= 32
        @test last(sizes) - first(sizes) <= 16 * length(sizes) + 64

        # Plain data: a process with neither BRM nor the model reads every field.
        script = """
            using Serialization
            s = deserialize(ARGS[1])
            print(s.iteration, " ", s.status, " ", s.q_kind, " ", s.phase, " ",
                length(s.coordinates), " ", repr(s.q.μ == Float64.(s.q.μ)), " ",
                bytes2hex(reinterpret(UInt8, s.q.μ)), " ", size(s.draws, 2), " ",
                isdefined(Main, :BayesianRegressionModels))
            """
        cmd = addenv(`$(Base.julia_cmd()) --startup-file=no -e $script $path`,
            "JULIA_LOAD_PATH" => "@stdlib")
        @test read(cmd, String) == string(205, " stopped ", final.q_kind, " ", final.phase, " ",
            length(final.coordinates), " true ", bytes2hex(reinterpret(UInt8, final.q.μ)),
            " 5 false")

        # A snapshot round-trips with what a consumer needs to name its coordinates.
        @test final.coordinates[final.globals] == fit.coordinates[fit.diagnostics.globals]
        @test final.model_identity == fit.diagnostics.optimizer.identity.binding
        @test final.q.μ == fit.diagnostics.q.μ
    end
end
