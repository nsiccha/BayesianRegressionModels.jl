# Minibatch SVI on several tasks (`threads`, src/nlme_svi.jl): a run must be the
# same to the last bit at any number of tasks, because its draws come from the
# one stream in batch order and its subjects' results are summed in batch order.
# The states after every iteration are compared, and their Adam moments are the
# gradient trajectory. This needs real threads, so the file refuses to run
# without them rather than pass on tasks taking turns on one thread.
#
# RUN: julia --project=test --threads=4 test/nlme_svi_threads.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
using LinearAlgebra, Random

Threads.nthreads() >= 4 || error(
    "test/nlme_svi_threads.jl needs real threads: run it with --threads=4")

const BRM = BayesianRegressionModels

# The legacy schedule keeps these every-iteration trajectories short.
const SVI_THREADS = (; batch_size=6, window=20, min_windows=6, patience=3,
    polish_windows=4, max_iterations=3000, stationarity=false, centering=nothing)

function svi_trajectory(m; kwargs...)
    states = BRMSVIOptimizerState[]
    fit = brm_svi_fit(s -> push!(states, s), m; checkpoint_every=1, kwargs...)
    fit, states
end

svi_same_trajectory(a, b) = length(a) == length(b) && all(svi_same_state.(a, b))

gauss_model_prepared(seed, N) = brm_nlme_model(
    RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(seed), N))); ad_backend=NLME_AD)

@stestset "a run is the same to the last bit at 1 and at several tasks" begin
    m = gauss_model_prepared(201, 16)
    serial, serial_states = svi_trajectory(m; rng=Xoshiro(202), SVI_THREADS..., ndraws=40,
        threads=1)
    @test serial.diagnostics.converged
    @test serial.diagnostics.optimizer.run.polish_start > 0
    @test serial.diagnostics.threads == 1
    for threads in (2, 4)
        fit, states = svi_trajectory(m; rng=Xoshiro(202), SVI_THREADS..., ndraws=40, threads)
        @test fit.diagnostics.threads == threads
        @test svi_same_trajectory(states, serial_states)
        @test svi_same_result(fit, serial)
    end
    # The comparison can tell runs apart: another seed is another run.
    other = brm_svi_fit(m; rng=Xoshiro(203), SVI_THREADS..., ndraws=40, threads=1)
    @test !svi_same_result(other, serial)
    # Tasks past the batch size have nothing to do.
    capped = brm_svi_fit(m; rng=Xoshiro(202), SVI_THREADS..., max_iterations=5, ndraws=0,
        threads=40)
    @test capped.diagnostics.threads == SVI_THREADS.batch_size
end

@stestset "a learned-centering run is the same to the last bit at 1 and at several tasks" begin
    m = gauss_model_prepared(251, 16)
    options = (; SVI_THREADS..., ndraws=40, centering=LearnedCentering())
    serial, serial_states = svi_trajectory(m; rng=Xoshiro(252), options..., threads=1)
    @test serial.diagnostics.centering !== nothing
    @test serial.diagnostics.optimizer.run.polish_start > 0
    controls(states) = [s.run.q.centering.x for s in states]
    for threads in (2, 4)
        fit, states = svi_trajectory(m; rng=Xoshiro(252), options..., threads)
        @test fit.diagnostics.threads == threads
        @test svi_same_trajectory(states, serial_states)
        @test controls(states) == controls(serial_states)
        @test svi_same_result(fit, serial)
    end
end

@stestset "several tasks run on several threads and each unit once" begin
    n = 200
    seen = zeros(Int, n)
    ids = Vector{Int}(undef, n)
    BRM._brm_svi_each_unit(1:4, n) do _, k
        # Long enough for the other tasks to start, without allocating.
        t = time_ns()
        while time_ns() - t < 200_000 end
        seen[k] += 1
        ids[k] = Threads.threadid()
    end
    @test all(==(1), seen)
    @test length(unique(ids)) >= 2
    # One plan runs in order on this task.
    order = Int[]
    BRM._brm_svi_each_unit([:only], 5) do plan, k
        @test plan === :only
        push!(order, k)
    end
    @test order == 1:5
end

@stestset "a failing unit stops the iteration after every task has stopped" begin
    running = Threads.Atomic{Int}(0)
    @test_throws TaskFailedException BRM._brm_svi_each_unit(1:4, 40) do _, k
        Threads.atomic_add!(running, 1)
        k == 3 && error("unit 3 failed")
        sleep(0.01)
        Threads.atomic_sub!(running, 1)
    end
    # Only the failing call is still counted: the others finished before the throw.
    @test running[] == 1
    err = try
        BRM._brm_svi_each_unit((_, k) -> k == 3 && error("unit 3 failed"), 1:4, 8)
    catch e
        e
    end
    @test occursin("unit 3 failed", sprint(showerror, err))
    # On one task the unit's own exception surfaces.
    @test_throws DomainError BRM._brm_svi_each_unit((_, _) -> sqrt(-1.0), [:only], 1)
end

@stestset "the same under a thread that allocates and collects" begin
    # A reverse gradient beside an allocating thread is the condition of Enzyme's
    # concurrent-GC defect (reactivekernels-use §7ao), and of any buffer a
    # collection could move out from under a task.
    m = gauss_model_prepared(211, 16)
    controls = (; SVI_THREADS..., max_iterations=300, ndraws=0)
    serial = brm_svi_fit(m; rng=Xoshiro(212), controls..., threads=1)
    stop = Threads.Atomic{Bool}(false)
    pressure = Threads.@spawn begin
        collections = 0
        while !stop[]
            sum(length, [zeros(1000) for _ in 1:200])
            GC.gc(false)
            collections += 1
        end
        collections
    end
    fit = try
        brm_svi_fit(m; rng=Xoshiro(212), controls..., threads=3)
    finally
        stop[] = true
    end
    @test fetch(pressure) > 0
    @test svi_same_result(fit, serial)
end

@stestset "nested subject blocks, flat rows and a low-rank global factor" begin
    # Blocks of different lengths share one sampler per cell group.
    rng = Xoshiro(221)
    nested = brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    controls = (; batch_size=2, window=10, min_windows=2, patience=1, polish_windows=2,
        max_iterations=120, ndraws=10)
    a, as = svi_trajectory(nested; rng=Xoshiro(222), controls..., threads=1)
    b, bs = svi_trajectory(nested; rng=Xoshiro(222), controls..., threads=2)
    @test length(unique(length.(nested.view.subject_coordinates))) > 1
    @test svi_same_trajectory(as, bs)
    @test svi_same_result(a, b)
    # Each row of a flat grouped response is a cell of its subject.
    flat = brm_nlme_model(RKBRMI(nlme_pk(NLME_DATA)); ad_backend=NLME_AD)
    a = brm_svi_fit(flat; rng=Xoshiro(223), batch_size=3, max_iterations=120, min_windows=1,
        ndraws=10, threads=1)
    b = brm_svi_fit(flat; rng=Xoshiro(223), batch_size=3, max_iterations=120, min_windows=1,
        ndraws=10, threads=3)
    @test svi_same_result(a, b)
    # The global factor's low-rank part goes through BLAS.
    m = gauss_model_prepared(224, 12)
    a = brm_svi_fit(m; rng=Xoshiro(225), family=LowRankGlobal(2), SVI_THREADS...,
        max_iterations=120, ndraws=10, threads=1)
    b = brm_svi_fit(m; rng=Xoshiro(225), family=LowRankGlobal(2), SVI_THREADS...,
        max_iterations=120, ndraws=10, threads=4)
    @test svi_same_result(a, b)
end

@stestset "the default stream: spawning tasks does not move it" begin
    m = gauss_model_prepared(231, 12)
    controls = (; SVI_THREADS..., max_iterations=60, ndraws=10)
    Random.seed!(232)
    a = brm_svi_fit(m; controls..., threads=1)
    after_a = rand()
    Random.seed!(232)
    b = brm_svi_fit(m; controls..., threads=4)
    after_b = rand()
    @test svi_same_result(a, b)
    @test after_a == after_b
end

@stestset "a run continues at another number of tasks, and on a new window" begin
    m = gauss_model_prepared(241, 16)
    controls = (; SVI_THREADS..., ndraws=30)
    full = brm_svi_fit(m; rng=Xoshiro(242), controls..., threads=1)
    first_part = brm_svi_fit(m; rng=Xoshiro(242), controls..., stop_after=37, threads=4)
    @test first_part.diagnostics.threads == 4
    second_part = brm_svi_resume(m, first_part; stop_after=40, ndraws=30, threads=2)
    @test second_part.diagnostics.threads == 2
    final = brm_svi_resume(m, second_part; ndraws=30, threads=1)
    @test final.diagnostics.threads == 1
    @test svi_same_result(final, full)
    # The default is the process's threads.
    @test brm_svi_resume(m, second_part; stop_after=1, ndraws=0).diagnostics.threads ==
        min(Threads.nthreads(), SVI_THREADS.batch_size)
    # A window fit reads the carried prior on every task.
    labels = ["s$i" for i in 1:8]
    data = svi_stream_data(Xoshiro(243), labels, 3)
    next = svi_stream_data(Xoshiro(244), labels, 3)
    state = brm_svi_checkpoint(brm_svi_fit(svi_stream_model(data); rng=Xoshiro(245),
        SVI_STREAM_QUICK..., threads=1))
    window = (; rng=Xoshiro(246), batch_size=4, max_iterations=60, min_windows=1, ndraws=10)
    a = brm_svi_fit(state, svi_stream_model(next); window..., threads=1)
    b = brm_svi_fit(state, svi_stream_model(next); window..., rng=Xoshiro(246), threads=3)
    @test svi_same_result(a, b)
end

@stestset "workers own their samplers and share the cells' read-only tables" begin
    m = gauss_model_prepared(251, 8)
    t = BRM._brm_svi_target(m)
    @test only(t.plans) === t.plan
    w = BRM._brm_svi_with_workers(t, 3)
    @test length(w.plans) == 3 && first(w.plans) === t.plan
    for plan in w.plans[2:end]
        @test plan.sampler !== t.plan.sampler && plan.point !== t.plan.point
        @test all(i -> plan.gradients[i] !== t.plan.gradients[i], eachindex(plan.gradients))
        @test all(i -> plan.local_gradients[i] !== t.plan.local_gradients[i],
            eachindex(plan.local_gradients))
        # Scaling with the units, these must not be copied per worker.
        @test plan.indices === t.plan.indices && plan.footprints === t.plan.footprints &&
            plan.slots === t.plan.slots && plan.local_footprints === t.plan.local_footprints &&
            plan.eta_slots === t.plan.eta_slots
    end
    # A worker's plan gives what the model's own does.
    g, z = m.unit[t.globals], m.unit[t.units[2]] .+ 0.2
    values = map(w.plans) do plan
        gg, gz = zeros(length(g)), zeros(length(z))
        BRM._brm_svi_unit_term!(gg, gz, t, 2, g, z, plan), gg, gz
    end
    @test allequal(values)
end

@stestset "a model keeps its workers for later calls, at any number of tasks" begin
    m = gauss_model_prepared(261, 12)
    # A fit advances the stream it is given, so each fresh run gets its own.
    controls = (; SVI_THREADS..., ndraws=10)
    reference, reference_states = svi_trajectory(gauss_model_prepared(261, 12); controls...,
        rng=Xoshiro(262), threads=1)
    first_fit = brm_svi_fit(m; controls..., rng=Xoshiro(262), stop_after=30, threads=2)
    @test !first_fit.diagnostics.target.reused && first_fit.diagnostics.target.workers_built == 1
    # More tasks build only the ones missing; fewer build none and keep the rest.
    wider = brm_svi_resume(m, first_fit; stop_after=30, ndraws=10, threads=4)
    @test wider.diagnostics.target.reused && wider.diagnostics.target.workers_built == 2
    workers = copy(m.svi_target.prepared.workers)
    @test length(workers) == 3
    narrower = brm_svi_resume(m, wider; stop_after=30, ndraws=10, threads=2)
    @test narrower.diagnostics.threads == 2 && narrower.diagnostics.target.workers_built == 0
    @test all(m.svi_target.prepared.workers .=== workers)
    final = brm_svi_resume(m, narrower; ndraws=10, threads=4)
    @test final.diagnostics.target.workers_built == 0
    @test svi_same_result(final, reference)
    @test svi_same_state(brm_svi_optimizer_checkpoint(final), last(reference_states))
    # Snapshots are the same at any number of tasks, and leave the run alone.
    function snapshots(threads)
        seen = NamedTuple[]
        fit = brm_svi_fit(m; controls..., rng=Xoshiro(263), stop_after=50, threads, snapshot=s -> push!(seen, s),
            snapshot_every=7, snapshot_draws=3)
        fit, seen
    end
    (one, a), (four, b) = snapshots(1), snapshots(4)
    @test svi_same_result(one, four)
    @test all(s -> (s[1].draws, s[1].q) == (s[2].draws, s[2].q), zip(a, b))
    @test all(s -> s.threads == 4, b)
end

@stestset "concurrent fits on one model use their own targets" begin
    m = gauss_model_prepared(271, 12)
    controls = (; SVI_THREADS..., max_iterations=60, ndraws=5, threads=2)
    references = [brm_svi_fit(gauss_model_prepared(271, 12); controls..., rng=Xoshiro(s))
        for s in 272:275]
    brm_svi_fit(m; controls..., rng=Xoshiro(272), stop_after=1)
    @test m.svi_target.prepared !== nothing
    fits = fetch.([Threads.@spawn brm_svi_fit(m; controls..., rng=Xoshiro(s)) for s in 272:275])
    # One found the model's target, and the others found the slot empty.
    @test count(f -> f.diagnostics.target.reused, fits) >= 1
    @test count(f -> !f.diagnostics.target.reused, fits) >= 1
    @test all(svi_same_result.(fits, references))
    # The slot holds one target, whichever finished last.
    @test m.svi_target.prepared.target.m === m
end

@stestset "BLAS stays at one thread during a fit and is given back" begin
    m = gauss_model_prepared(261, 12)
    controls = (; SVI_THREADS..., max_iterations=60, ndraws=10)
    saved = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        reference = brm_svi_fit(m; rng=Xoshiro(262), controls..., threads=1)
        BLAS.set_num_threads(4)
        # A thread count that BLAS itself would use changes nothing in the result.
        fit = brm_svi_fit(m; rng=Xoshiro(262), controls..., threads=1)
        @test BLAS.get_num_threads() == 4
        @test svi_same_result(fit, reference)
        inside = BRM._brm_with_serial_blas() do
            BRM._brm_with_serial_blas(() -> BLAS.get_num_threads()), BLAS.get_num_threads()
        end
        @test inside == (1, 1)
        @test BLAS.get_num_threads() == 4
        @test_throws ErrorException BRM._brm_with_serial_blas(() -> error("fails"))
        @test BLAS.get_num_threads() == 4
    finally
        BLAS.set_num_threads(saved)
    end
end

@stestset "threads: what is refused" begin
    m = gauss_model_prepared(271, 8)
    # refused: the count of tasks is a positive integer.
    for bad in (0, -1, 1.5, true)
        @test_throws "threads must be a positive integer" brm_svi_fit(m; threads=bad,
            max_iterations=1, ndraws=0)
    end
    # refused: a gradient beside an allocating thread can use freed memory without
    # Enzyme's repair (reactivekernels-use §7ao), and one task needs none.
    # The Enzyme extension answers for the default backend, and this Enzyme is repaired.
    @test which(BRM._brm_concurrent_gradients, (Nothing,)).module ===
        Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsEnzymeExt)
    @test BRM._brm_concurrent_gradients(nothing)
    @test_throws "concurrent-GC repair" BRM._brm_svi_threads("fit", 2, nothing; repaired=false)
    @test BRM._brm_svi_threads("fit", 1, nothing; repaired=false) == 1
    # A caller who chose nothing keeps the serial loop on such an Enzyme.
    @test BRM._brm_svi_threads("fit", nothing, nothing; repaired=false) == 1
    @test BRM._brm_svi_threads("fit", nothing, nothing; repaired=true) == Threads.nthreads()
    # refused: GaussianBlockLaplace's updates evaluate the whole window, not a
    # batch of units, so there is nothing `threads` would divide.
    @test_throws "does not use threads" brm_svi_fit(m; laplace=GaussianBlockLaplace(),
        threads=2)
end
