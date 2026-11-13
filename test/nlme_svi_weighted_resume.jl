# Integration of weighted batches and structured local factors with canonical
# task parallelism and same-target checkpoints, including lazy polishing state.
# RUN: julia --project=test --threads=2 test/nlme_svi_weighted_resume.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using Serialization

Threads.nthreads() >= 2 || error("weighted resume checks require --threads=2")

@testset "iterate histories count weighted visits and survive worker changes" begin
    data = svi_gauss_data(Xoshiro(981), 12)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(data)); ad_backend=NLME_AD)
    # Stay in exploration, but cross several history-thinning boundaries.
    options = (; batch_size=4, window=20, min_windows=100, max_iterations=1600,
        unit_sampling=:work, ndraws=3, centering=nothing)
    full = brm_svi_fit(m; rng=Xoshiro(982), options..., threads=1)
    parallel = brm_svi_fit(m; rng=Xoshiro(982), options..., threads=2)
    @test svi_same_result(full, parallel)
    a, b = brm_svi_optimizer_checkpoint(full), brm_svi_optimizer_checkpoint(parallel)
    @test svi_same_state(a, b)
    h = a.run.sampling.stationarity
    @test h.globals.updates == 1600 && h.globals.stride > 1
    @test [u.updates for u in h.units] == a.run.sampling.visits
    @test sum(a.run.sampling.draws) > sum(a.run.sampling.visits)
    part = brm_svi_fit(m; rng=Xoshiro(982), options..., stop_after=137, threads=2)
    io = IOBuffer(); serialize(io, brm_svi_optimizer_checkpoint(part)); seekstart(io)
    resumed = brm_svi_resume(m, deserialize(io); ndraws=3, threads=1)
    @test svi_same_result(resumed, full)
    @test svi_same_state(brm_svi_optimizer_checkpoint(resumed), a)
end

@testset "weighted structured trajectories and same-target continuation" begin
    data = svi_gauss_data(Xoshiro(971), 12)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(data)); ad_backend=NLME_AD)
    controls = (; batch_size=4, window=10, min_windows=3, patience=1,
        polish_windows=3, max_iterations=1000, ndraws=12, stationarity=false,
        centering=nothing)
    fresh = nothing
    for method in (:uniform, :work), dense_limit in (0, 64)
        states = BRMSVIOptimizerState[]
        options = (; controls..., unit_sampling=method, unit_dense_limit=dense_limit)
        serial = brm_svi_fit(s -> push!(states, s), m; rng=Xoshiro(972),
            options..., checkpoint_every=1, threads=1)
        parallel_states = BRMSVIOptimizerState[]
        parallel = brm_svi_fit(s -> push!(parallel_states, s), m; rng=Xoshiro(972),
            options..., checkpoint_every=1, threads=2)
        @test serial.diagnostics.converged
        @test svi_same_result(serial, parallel)
        @test length(states) == length(parallel_states)
        @test all(svi_same_state.(states, parallel_states))
        full_state = brm_svi_optimizer_checkpoint(serial)
        P = full_state.run.polish_start
        @test P >= 30
        # Stop once during exploration and once after averaging has started.
        # Returning draws/averaged q must leave the raw iterate and RNG alone.
        part = brm_svi_fit(m; rng=Xoshiro(972), options..., stop_after=13, threads=2)
        @test svi_same_state(brm_svi_optimizer_checkpoint(part), states[13])
        part = brm_svi_resume(m, part; stop_after=P+7-13, ndraws=5, threads=1)
        saved = brm_svi_optimizer_checkpoint(part)
        @test svi_same_state(saved, states[P+7])
        @test saved.run.average !== nothing
        @test any(saved.run.averaged_through .< 7)
        final = brm_svi_resume(m, saved; ndraws=controls.ndraws, threads=2)
        @test svi_same_result(final, serial)
        @test svi_same_state(brm_svi_optimizer_checkpoint(final), full_state)
        @test svi_same_state(saved, states[P+7])
        @test saved.controls.unit_sampling === method
        # 2d evaluations per visit of a dense unit of d coordinates (the default
        # orthogonal design), 2 per visit of any other, plus one mean-point
        # gradient for each unit with nonempty coupling.
        calls = [U isa Matrix ? 2max(size(U, 1), 1) : 2 for U in full_state.run.q.U]
        calls .+= [!isempty(V) for V in full_state.run.q.coupling.V]
        @test full_state.run.unit_gradient_calls == sum(full_state.run.sampling.visits .* calls)
        @test sum(full_state.run.sampling.draws) == controls.batch_size * full_state.run.iteration
        if method === :work
            @test any(full_state.run.sampling.draws .> full_state.run.sampling.visits)
            dense_limit == 0 && (fresh = (; saved, serial))
        end
    end
    # The existing process helper rebuilds the same native model and resumes a
    # serialized optimizer checkpoint. Its child uses one Julia thread.
    mktempdir() do dir
        checkpoint = joinpath(dir, "checkpoint.jls")
        input = joinpath(dir, "input.jls")
        output = joinpath(dir, "output.jls")
        serialize(checkpoint, fresh.saved)
        serialize(input, (; mode=:finish, data, path=checkpoint, ndraws=controls.ndraws))
        script = joinpath(@__DIR__, "nlme_svi_optimizer_process.jl")
        run(`$(Base.julia_cmd()) --threads=1 --project=$(@__DIR__) $script $input $output`)
        resumed = deserialize(output)
        @test resumed.draws == fresh.serial.draws
        @test svi_same_q(resumed.q, fresh.serial.diagnostics.q)
        @test resumed.elbo == fresh.serial.diagnostics.elbo
        @test svi_same_state(resumed.state, brm_svi_optimizer_checkpoint(fresh.serial))
    end
end
