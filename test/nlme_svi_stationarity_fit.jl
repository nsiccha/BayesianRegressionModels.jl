# End-to-end default stopping and exact continuation with bounded iterate history.
# RUN: julia --project=test test/nlme_svi_stationarity_fit.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using Serialization, Statistics

svi_stationarity_roundtrip(x) = (io = IOBuffer(); serialize(io, x); seekstart(io); deserialize(io))

@stestset "default iterate stopping completes and resumes exactly in every phase" begin
    data = svi_gauss_data(Xoshiro(801), 8; nobs=12)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(data));
        ad_backend=NLME_AD)
    options = (; batch_size=4, max_iterations=50_000, ndraws=30)
    full = brm_svi_fit(m; rng=Xoshiro(802), options...)
    d = full.diagnostics
    @info "stationarity fit" iterations=d.iterations stages=d.stages guard=d.stationarity
    @test d.converged && d.finished
    @test d.stationarity.enabled && d.stationarity.settled
    @test d.stationarity.current.passed
    @test d.stationarity.exploration.iteration == d.stages.exploring
    @test !hasproperty(d.sampling, :stationarity)
    if d.converged
        E, P = d.stages.exploring, d.optimizer.run.polish_start
        part = brm_svi_fit(m; rng=Xoshiro(802), options..., stop_after=113)
        original = brm_svi_optimizer_checkpoint(part)
        state = svi_stationarity_roundtrip(original)
        @test svi_same_state(original, state)
        # Stop inside settling, then inside polishing. Draws and serialization
        # at each stop must not change the optimization stream or diagnostics.
        for finish in (E + 17, P + 13)
            result = brm_svi_resume(m, state; stop_after=finish-state.run.iteration, ndraws=7)
            state = svi_stationarity_roundtrip(brm_svi_optimizer_checkpoint(result))
            @test state.run.iteration == finish
        end
        result = brm_svi_resume(m, state; ndraws=30)
        @test svi_same_result(result, full)
        @test svi_same_state(brm_svi_optimizer_checkpoint(result), d.optimizer)
        @test isequal(result.diagnostics.stationarity, d.stationarity)
        @test svi_same_result(brm_svi_resume(m, result; ndraws=30), full)
        @test original.run.iteration == 113
        # The existing fresh-process driver rebuilds this same synthetic model.
        mktempdir() do dir
            path, input, output = joinpath.(dir, ("state.jls", "input.jls", "output.jls"))
            serialize(path, state)
            serialize(input, (; mode=:finish, data, path, ndraws=30))
            script = joinpath(@__DIR__, "nlme_svi_optimizer_process.jl")
            run(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $script $input $output`)
            fresh = deserialize(output)
            @test fresh.draws == full.draws
            @test svi_same_state(fresh.state, d.optimizer)
        end
    end

    # One settling window cannot provide 200 actual updates to every unit.
    # Finishing the requested polishing still returns useful draws, but does
    # not certify convergence or restart optimization on a resumed checkpoint.
    statuses = Symbol[]
    capped = brm_svi_fit(m; rng=Xoshiro(802), options..., settle_windows=1,
        snapshot=s -> push!(statuses, s.status))
    @test capped.diagnostics.finished && !capped.diagnostics.converged
    @test !capped.diagnostics.stages.settled
    @test last(statuses) == :settle_limit
    @test svi_same_result(brm_svi_resume(m, capped; ndraws=30), capped)

    # The overall budget also reserves final averaging after settling starts.
    # A non-round limit consumes its last partial window, without certifying
    # stationarity or changing the stream when resumed across the transition.
    budget_options = merge(options, (; max_iterations=d.stages.exploring + 2403))
    empty!(statuses)
    budget = brm_svi_fit(m; rng=Xoshiro(802), budget_options...,
        snapshot=s -> push!(statuses, s.status))
    bd = budget.diagnostics
    @test bd.finished && !bd.converged && !bd.stages.settled
    @test bd.iterations == budget_options.max_iterations
    @test bd.stages.polishing == 2003
    @test last(statuses) == :max_iterations
    @test bd.optimizer.run.average !== nothing
    @test svi_same_result(brm_svi_resume(m, budget; ndraws=30), budget)
    part = brm_svi_fit(m; rng=Xoshiro(802), budget_options...,
        stop_after=bd.optimizer.run.polish_start - 17)
    state = svi_stationarity_roundtrip(brm_svi_optimizer_checkpoint(part))
    part = brm_svi_resume(m, state; stop_after=30, ndraws=3)
    @test part.diagnostics.stages.polishing == 13
    state = svi_stationarity_roundtrip(brm_svi_optimizer_checkpoint(part))
    resumed = brm_svi_resume(m, state; ndraws=30)
    @test svi_same_result(resumed, budget)
    @test svi_same_state(brm_svi_optimizer_checkpoint(resumed), bd.optimizer)

    # A pre-guard checkpoint has neither the control nor the extra history.
    # It must retain its own stopping rule after this package is upgraded.
    legacy_options = (; batch_size=4, window=20, min_windows=6, patience=3,
        polish_windows=4, max_iterations=3000, stationarity=false, ndraws=30)
    legacy = brm_svi_fit(m; rng=Xoshiro(803), legacy_options...)
    part = brm_svi_fit(m; rng=Xoshiro(803), legacy_options..., stop_after=113)
    old = brm_svi_optimizer_checkpoint(part)
    controls = Base.structdiff(old.controls, NamedTuple{(:stationarity,)})
    old = BRMSVIOptimizerState(old.format, old.identity, old.previous, controls, old.run)
    resumed = brm_svi_resume(m, svi_stationarity_roundtrip(old); ndraws=30)
    @test svi_same_result(resumed, legacy)
    @test !resumed.diagnostics.stationarity.enabled
    @test !hasproperty(old.run.sampling, :stationarity)
    @test_throws "stationarity must be a Bool" brm_svi_fit(m; stationarity=1)
end
