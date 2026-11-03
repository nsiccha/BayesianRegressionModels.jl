# Same-target continuation of `GaussianBlockLaplace` SVI: a fixed-budget run
# stopped after any completed update and continued from its
# `BRMSVIOptimizerState` must reach exactly the run that never stopped — the
# same Adam iterates, averaging history, random streams, ELBO, call counts,
# numerical records, carry and draws — for an initial fit and a window update.
# The state round-trips through Serialization.
#
# RUN: julia --project=test test/nlme_svi_laplace_optimizer.jl
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, DifferentiationInterface, Random, LinearAlgebra, Serialization
const BRM = BayesianRegressionModels
include("nlme_svi_laplace_streaming_fixture.jl")

const LAPLACE_AD = AutoEnzyme(; mode=Enzyme.Reverse)

laplace_roundtrip(x) = (io = IOBuffer(); serialize(io, x); deserialize(seekstart(io)))

function laplace_same_run(a, b)
    all(n -> isequal(getfield(a.adam, n), getfield(b.adam, n)), fieldnames(typeof(a.adam))) &&
        a.rng == b.rng && isequal(a.draw_rng, b.draw_rng) &&
        all(n -> isequal(getfield(a, n), getfield(b, n)),
            (:q, :iteration, :elbo, :history, :counts, :numerical))
end

# Everything a GaussianBlockLaplace result reports, including its carry.
function laplace_same_result(a, b)
    sa, sb = laplace_window_summary(a), laplace_window_summary(b)
    ok = all(k -> isequal(getfield(sa, k), getfield(sb, k)), keys(sa))
    da, db = a.diagnostics, b.diagnostics
    ok && da.iterations == db.iterations && da.averaging_window == db.averaging_window &&
        da.budget_exhausted == db.budget_exhausted && da.finished == db.finished
end

@testset "GaussianBlockLaplace runs continue exactly, initial and window updates" begin
    builder = laplace_window_builder()
    method = GaussianBlockLaplace(averaging_window=3, carry_draws=12, carry_rank=3)
    data = laplace_window_data([1, 2, 3])
    options = (; laplace=method, ad_backend=LAPLACE_AD, family=LowRankGlobal(2),
        max_iterations=8)
    full = brm_svi_fit(builder(data); options..., rng=Xoshiro(810), ndraws=5)
    @test full.diagnostics.finished && full.diagnostics.iterations == 8
    # Stops before the averaging history is full and after it wraps; each
    # stopped call draws, which must not move the streams or the counts.
    saved = BRMSVIOptimizerState[]
    part = brm_svi_fit(s -> push!(saved, s), builder(data); options..., rng=Xoshiro(810),
        stop_after=2, checkpoint_every=1, ndraws=3)
    @test [s.run.iteration for s in saved] == [1, 2]
    @test !part.diagnostics.finished && part.diagnostics.averaging_window == 1:2
    state = laplace_roundtrip(brm_svi_optimizer_checkpoint(part))
    @test laplace_same_run(state.run, last(saved).run)
    middle = brm_svi_resume(builder(data), state; ad_backend=LAPLACE_AD, stop_after=3,
        ndraws=4)
    @test middle.diagnostics.iterations == 5 && middle.diagnostics.averaging_window == 3:5
    final = brm_svi_resume(builder(data), laplace_roundtrip(brm_svi_optimizer_checkpoint(middle));
        ad_backend=LAPLACE_AD, ndraws=5)
    @test laplace_same_result(final, full)
    @test laplace_same_run(brm_svi_optimizer_checkpoint(final).run,
        brm_svi_optimizer_checkpoint(full).run)
    @test laplace_same_result(brm_svi_resume(builder(data), full; ad_backend=LAPLACE_AD,
        ndraws=5), full)
    @test occursin("GaussianBlockLaplace, iteration 2 of 8", sprint(show, state))

    # A window update keeps the previous carry in its checkpoint; a separate
    # draw stream is kept at its start until the run draws.
    previous = brm_svi_checkpoint(full)
    next = laplace_window_data([3, 4]; shift=0.7)
    update = (; ad_backend=LAPLACE_AD, max_iterations=6)
    whole = brm_svi_fit(previous, builder(next); update..., rng=Xoshiro(811),
        draw_rng=Xoshiro(812), ndraws=4)
    stopped = brm_svi_fit(previous, builder(next); update..., rng=Xoshiro(811),
        draw_rng=Xoshiro(812), stop_after=4, ndraws=2)
    window_state = laplace_roundtrip(brm_svi_optimizer_checkpoint(stopped))
    @test window_state.previous isa BRMSVILaplaceState
    @test window_state.run.draw_rng == Xoshiro(812)
    resumed = brm_svi_resume(builder(next), window_state; ad_backend=LAPLACE_AD, ndraws=4)
    @test laplace_same_result(resumed, whole)
    # refused: other data, and a minibatch-only option.
    @test_throws "bound data" brm_svi_resume(builder(laplace_window_data([3, 4]; shift=0.9)),
        window_state; ad_backend=LAPLACE_AD)
    @test_throws "unsupported options batch_size" brm_svi_resume(builder(next), window_state;
        batch_size=2)
end
