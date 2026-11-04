# Same-target continuation of minibatch SVI (src/nlme_svi_optimizer.jl): a run
# stopped after any completed iteration and continued from its
# `BRMSVIOptimizerState` must reach exactly the run that never stopped — the
# same iterates, Adam state, random stream, ELBO trace, stopping decision,
# polishing average and draws — in this process and in fresh ones, including
# after an abrupt termination mid-write. A changed target is refused before any
# step. The reference is an uninterrupted run whose `save` collects its state
# after every iteration.
#
# RUN: julia --project=test test/nlme_svi_optimizer.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
using LinearAlgebra, Random, Serialization
import Treebars

# A short exploration (at least six windows of 20 iterations) and polishing
# (four windows), so a run converges in a few hundred iterations.
const SVI_OPT = (; batch_size=4, window=20, min_windows=6, patience=3,
    polish_windows=4, max_iterations=3000)

svi_bytes(x) = (io = IOBuffer(); serialize(io, x); position(io))

# `svi_gauss_model` with another prior: another program on the same data.
svi_gauss_changed_model(data) = @brm data begin
    sigma ~ Exponential(2)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ b[i] .* x[i], sigma)
    end
end

# The uninterrupted reference run and its state after every iteration.
function svi_reference(m; kwargs...)
    states = BRMSVIOptimizerState[]
    fit = brm_svi_fit(s -> push!(states, s), m; checkpoint_every=1, kwargs...)
    @test [s.run.iteration for s in states] == 1:fit.diagnostics.iterations
    fit, states
end

@stestset "an interrupted run continues exactly: exploration, window end, polishing" begin
    m = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(101), 12)));
        ad_backend=NLME_AD)
    full, states = svi_reference(m; rng=Xoshiro(102), SVI_OPT..., ndraws=40)
    @test full.diagnostics.converged && full.diagnostics.finished
    P = full.diagnostics.optimizer.run.polish_start
    @test P >= 120
    # Stops mid-window while exploring, at a window end, and while polishing;
    # each stopped call also draws, which must not move the stream.
    first_fit = brm_svi_fit(m; rng=Xoshiro(102), SVI_OPT..., stop_after=37, ndraws=15)
    @test !first_fit.diagnostics.finished && first_fit.diagnostics.iterations == 37
    s1 = brm_svi_optimizer_checkpoint(first_fit)
    @test svi_same_state(s1, states[37])
    saved = BRMSVIOptimizerState[]
    second = brm_svi_resume(s -> push!(saved, s), m, s1; stop_after=23, ndraws=15)
    @test second.diagnostics.iterations == 60
    @test [s.run.iteration for s in saved] == [40, 60]
    @test svi_same_state(only(filter(s -> s.run.iteration == 60, saved)), states[60])
    third = brm_svi_resume(m, second; stop_after=P + 13 - 60, ndraws=15)
    s3 = brm_svi_optimizer_checkpoint(third)
    @test s3.run.polish_start == P && s3.run.iteration == P + 13
    @test svi_same_state(s3, states[P+13])
    # A stopped polishing run reports its running average, not the raw iterate.
    @test third.diagnostics.q.μ == s3.run.average.μ != s3.run.q.μ
    final = brm_svi_resume(m, s3; ndraws=40)
    @test svi_same_result(final, full)
    @test svi_same_state(brm_svi_optimizer_checkpoint(final),
        brm_svi_optimizer_checkpoint(full))
    # The checkpoints taken along the way are unchanged by their continuation.
    @test svi_same_state(s1, states[37]) && svi_same_state(s3, states[P+13])
    # A finished checkpoint does no iterations and reproduces the draws.
    again = brm_svi_resume(m, full; ndraws=40)
    @test svi_same_result(again, full)
    @test brm_svi_resume(m, s1; stop_after=0, ndraws=0).diagnostics.iterations == 37
    @test sprint(show, s3) ==
        "BRMSVIOptimizerState(iteration $(P + 13) of at most 3000, polishing 13/80)"
end

@stestset "save, interrupts, the task-local stream and low-rank globals" begin
    m = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(201), 10)));
        ad_backend=NLME_AD)
    controls = (; SVI_OPT..., family=LowRankGlobal(1))
    full, states = svi_reference(m; rng=Xoshiro(202), controls..., ndraws=25)
    @test full.diagnostics.converged
    # A cooperative interrupt passes the state of the window it stops at to
    # `save` before it propagates; the run continues from there.
    saved = BRMSVIOptimizerState[]
    interrupted = Treebars.with_progress(:state; description="SVI interrupt") do p
        try
            brm_svi_fit(m; rng=Xoshiro(202), controls..., checkpoint_every=30,
                    progress=p, ndraws=0) do s
                push!(saved, s)
                s.run.iteration == 30 && Treebars.request_interrupt!(p)
            end
        catch err
            err
        end
    end
    @test interrupted isa Treebars.ProgressInterrupted
    @test [s.run.iteration for s in saved] == [30, 40]
    @test svi_same_state(last(saved), states[40])
    @test svi_same_result(brm_svi_resume(m, last(saved); ndraws=25), full)
    # The default stream is the task's: a copy of it continues it.
    Random.seed!(203)
    default = brm_svi_fit(m; controls..., ndraws=25)
    Random.seed!(203)
    stopped = brm_svi_fit(m; controls..., stop_after=50, ndraws=5)
    @test svi_same_result(brm_svi_resume(m, stopped; ndraws=25), default)
end

@stestset "window fits continue with their streaming carry" begin
    initial = brm_svi_fit(svi_stream_model(svi_stream_data(Xoshiro(301), ["a", "b", "c"], 4));
        rng=Xoshiro(302), max_iterations=25, ndraws=0)
    carry = brm_svi_checkpoint(initial)
    data = svi_stream_data(Xoshiro(303), ["b", "c", "new"], 4)
    controls = (; batch_size=2, window=10, min_windows=4, patience=2,
        polish_windows=3, max_iterations=1000)
    full = brm_svi_fit(carry, svi_stream_model(data); rng=Xoshiro(304), controls...,
        ndraws=20)
    @test full.diagnostics.converged
    saved = BRMSVIOptimizerState[]
    part = brm_svi_fit(s -> push!(saved, s), carry, svi_stream_model(data);
        rng=Xoshiro(304), controls..., stop_after=17, ndraws=0)
    @test [s.run.iteration for s in saved] == [10, 17]
    state = brm_svi_optimizer_checkpoint(part)
    @test svi_stream_same_factors(state.previous, carry)
    rest = brm_svi_resume(svi_stream_model(data), state; ndraws=20)
    @test svi_same_result(rest, full)
    @test svi_stream_same_factors(brm_svi_checkpoint(rest), brm_svi_checkpoint(full))
    # refused: another window's data is another target.
    other = svi_stream_data(Xoshiro(305), ["b", "c", "new"], 4)
    @test_throws "bound data" brm_svi_resume(svi_stream_model(other), state)
end

@stestset "a changed target or an inconsistent checkpoint is refused before any step" begin
    data = svi_gauss_data(Xoshiro(401), 8)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(data)); ad_backend=NLME_AD)
    fit = brm_svi_fit(m; rng=Xoshiro(402), SVI_OPT..., stop_after=30, ndraws=0)
    state = brm_svi_optimizer_checkpoint(fit)
    before = brm_svi_optimizer_checkpoint(state)
    calls = Ref(0)
    tally(_) = (calls[] += 1)
    # The same model rebuilt from the same data resumes, through every input form.
    @test brm_svi_resume(svi_gauss_model(data), state; stop_after=0,
        ndraws=0).diagnostics.iterations == 30
    @test brm_svi_resume(RKBRMI(svi_gauss_model(data)), state; stop_after=0,
        ndraws=0).diagnostics.iterations == 30
    # refused: other data, another program.
    @test_throws "bound data" brm_svi_resume(tally,
        svi_gauss_model(svi_gauss_data(Xoshiro(403), 8)), state)
    @test_throws "emitted program" brm_svi_resume(tally, svi_gauss_changed_model(data), state)
    # refused: each identity part, and a factor or trace that does not fit.
    for (part, message) in ((:grouping, "groupings"), (:ownership, "each subject owns"),
            (:fixed, "fixed shared coordinates"))
        forged = BRMSVIOptimizerState(state.format,
            merge(state.identity, NamedTuple{(part,)}(("forged",))), state.previous,
            state.controls, state.run)
        @test_throws message brm_svi_resume(tally, m, forged)
    end
    short = brm_svi_optimizer_checkpoint(state)
    pop!(short.run.trace)
    @test_throws "inconsistent" brm_svi_resume(tally, m, short)
    wrong = brm_svi_optimizer_checkpoint(state)
    push!(wrong.run.q.μ, 0.0)
    @test_throws "inconsistent" brm_svi_resume(tally, m, wrong)
    @test calls[] == 0
    @test svi_same_state(state, before)
    # refused: the checkpoint fixes the controls, the stream and the start.
    @test_throws "unsupported options max_iterations" brm_svi_resume(m, state;
        max_iterations=10)
    @test_throws "unsupported options rng" brm_svi_resume(m, state; rng=Xoshiro(1))
    @test_throws "checkpoint function" brm_svi_fit(m; checkpoint_every=10)
    @test_throws ArgumentError brm_svi_fit(m; stop_after=-1)
    @test_throws "BRMI, an RKBRMI or a BRMNLMEModel" brm_svi_resume(data, state)
    # refused: the two checkpoints are different objects for different targets.
    @test_throws "holds no optimizer state" brm_svi_resume(m, brm_svi_checkpoint(fit))
    @test_throws "brm_svi_resume(model, state)" brm_svi_checkpoint(state)
end

@stestset "fixed shared values are part of the target" begin
    data = svi_spacetime_data(Xoshiro(501))
    rk = RKBRMI(svi_spacetime_model(
        :(hsgp(day; k=4, by=probe, cor=multigroup((probe_x, probe_y); k=3))))(data))
    point = first(BRM._brm_nlme_prior_median_point(brm_nlme_model(rk; ad_backend=NLME_AD)))
    fit = brm_svi_fit(rk; shared_point=point, rng=Xoshiro(502), batch_size=2,
        stop_after=3, ndraws=0)
    state = brm_svi_optimizer_checkpoint(fit)
    @test brm_svi_resume(rk, state; shared_point=point, stop_after=2,
        ndraws=0).diagnostics.iterations == 5
    moved = copy(point)
    shared = findall(==(:shared_latent), fit.diagnostics.model.view.role)
    @test !isempty(shared)
    moved[shared] .+= 0.1
    @test_throws "fixed shared coordinates" brm_svi_resume(rk, state; shared_point=moved)
end

@stestset "checkpoints hold no data and grow only by the ELBO trace" begin
    m = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(601), 12)));
        ad_backend=NLME_AD)
    _, states = svi_reference(m; rng=Xoshiro(602), SVI_OPT..., ndraws=0)
    P = last(states).run.polish_start
    a, b = states[40], states[80]
    @test b.run.polish_start == 0
    @test svi_bytes(b) - svi_bytes(a) <= 8 * 40 + 16 * 2 + 32
    # Polishing adds one copy of the factors: the running average.
    c, d = states[P-1], states[P+1]
    @test svi_bytes(d) - svi_bytes(c) <= svi_bytes(c.run.q) + 8 * 2 + 16 + 64
    # Four times the observations per subject: the same checkpoint size.
    sizes = map((5, 20)) do nobs
        data = svi_gauss_data(Xoshiro(603), 12; nobs)
        fit = brm_svi_fit(svi_gauss_model(data); rng=Xoshiro(604), SVI_OPT...,
            stop_after=10, ndraws=0)
        svi_bytes(brm_svi_optimizer_checkpoint(fit))
    end
    @test sizes[1] == sizes[2]
end

@stestset "fresh processes continue after an abrupt termination mid-write" begin
    data = svi_gauss_data(Xoshiro(701), 10)
    controls = (; SVI_OPT..., family=LowRankGlobal(1))
    m = brm_nlme_model(RKBRMI(svi_gauss_model(data)); ad_backend=NLME_AD)
    full, states = svi_reference(m; rng=Xoshiro(702), controls..., ndraws=30)
    P = full.diagnostics.optimizer.run.polish_start
    @test full.diagnostics.converged && P % 20 == 0
    start = brm_svi_fit(svi_gauss_model(data); rng=Xoshiro(702), controls...,
        stop_after=37, ndraws=0)
    script = joinpath(@__DIR__, "nlme_svi_optimizer_process.jl")
    julia = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $script`
    mktempdir() do dir
        path = joinpath(dir, "svi.jls")
        input, output = joinpath(dir, "input.jls"), joinpath(dir, "output.jls")
        # Exploring from iteration 37, saving every 10, killed while writing
        # the checkpoint of iteration P + 20: the file holds iteration P + 10.
        serialize(input, (; mode=:until_killed, data,
            state=brm_svi_optimizer_checkpoint(start), every=10, kill_at=P + 20, path))
        killed = run(ignorestatus(`$julia $input $output`))
        @test killed.exitcode == 9
        @test isfile(path * ".partial")
        committed = deserialize(path)
        @test committed.run.iteration == P + 10 && committed.run.polish_start == P
        @test svi_same_state(committed, states[P+10])
        serialize(input, (; mode=:finish, data, path, ndraws=30))
        run(`$julia $input $output`)
        resumed = deserialize(output)
        d = full.diagnostics
        @test resumed.draws == full.draws
        @test isequal(resumed.q, d.q)
        @test resumed.elbo == d.elbo && resumed.elbo_windows == d.elbo_windows
        @test resumed.iterations == d.iterations && resumed.converged
        @test resumed.unit_gradient_calls == d.unit_gradient_calls
        @test svi_same_state(resumed.state, brm_svi_optimizer_checkpoint(full))
    end
end
