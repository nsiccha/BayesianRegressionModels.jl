# One extreme ELBO draw in minibatch SVI (src/nlme_svi.jl): what it does to the
# global Adam's moments and to the stopping rule.
#
# The estimate's expectation can be infinite while every draw is finite, so
# `nonfinite=:error` never fires and a draw of -1e21 is an ordinary iteration.
# Unclipped, its g² stays in Adam's second moment for about 85,000 iterations and
# the coordinates its term reads stop moving, with a flat ELBO that reads as
# converged. Its window's mean and variance decide the stopping test.
#
# RUN: julia --project=test test/nlme_svi_heavy_tails.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))

# The mean |step| of one Adam coordinate over `recent` iterations, after a stream
# of unit-noise gradients, with one extreme gradient at iteration `spike`.
function svi_adam_steps(; n=3000, spike=2000, size=1e21, clip=Inf, recent=500, seed=1)
    rng = Xoshiro(seed)
    opt = BRM._BRMSVIAdam(1, 0.01)
    x = [0.0]
    moves = Float64[]
    for t in 1:n
        g = t == spike ? [size] : [randn(rng)]
        before = x[1]
        BRM._brm_svi_step!(opt, x, g; clip)
        push!(moves, abs(x[1] - before))
    end
    (; move=mean(moves[end-recent+1:end]), largest=maximum(moves), opt)
end

@stestset "one extreme gradient does not stop a coordinate" begin
    healthy = svi_adam_steps(; spike=0)
    # The old behaviour, as a control for the test: the coordinate freezes.
    frozen = svi_adam_steps()
    @test frozen.move < 1e-15 * healthy.move
    @test frozen.opt.clipped == 0
    bounded = svi_adam_steps(; clip=10.0)
    @test bounded.opt.clipped == 1
    @test 0.8 * healthy.move < bounded.move < 1.2 * healthy.move
    # Every size of spike has the same effect once it is at the limit.
    @test svi_adam_steps(; clip=10.0, size=1e6).move == bounded.move
    # The spike's own step stays of the size of an ordinary one.
    @test bounded.largest < 3 * healthy.largest
    # A gradient inside the limit is untouched, to the last bit.
    opt = BRM._BRMSVIAdam(2, 0.01)
    x = zeros(2)
    BRM._brm_svi_step!(opt, x, [1.0, -1.0]; clip=10.0)
    g = [9.5, -9.5]
    @test BRM._brm_svi_clip(opt, g, 10.0) == g
    @test opt.clipped == 0
    @test BRM._brm_svi_clip(opt, [11.0, -11.0], 10.0) ≈ [10.0, -10.0]
    @test opt.clipped == 1
    # `g` itself is not written.
    h = [50.0, -50.0]
    BRM._brm_svi_clip(opt, h, 10.0)
    @test h == [50.0, -50.0]
end

@stestset "the first gradient, and a coordinate without a gradient yet, have no limit" begin
    opt = BRM._BRMSVIAdam(2, 0.01)
    @test BRM._brm_svi_clip(opt, [1e21, 0.0], 10.0) == [1e21, 0.0]
    x = zeros(2)
    BRM._brm_svi_step!(opt, x, [1.0, 0.0]; clip=10.0)
    @test iszero(opt.v[2])
    @test BRM._brm_svi_clip(opt, [1e3, 1e3], 10.0) == [10.0, 1e3]
    # Adam's other users pass no limit and are unchanged.
    @test BRM._brm_svi_clip(opt, [1e21, 1e21], Inf) == [1e21, 1e21]
end

# A coordinate read by one unit of N, at batch size 1: its gradient is zero in all
# but one batch in `s`.
function svi_sparse_rms(s; clip, n=40_000, seed=2)
    rng = Xoshiro(seed)
    opt = BRM._BRMSVIAdam(1, 0.01)
    x = [0.0]
    for t in 1:n
        g = rand(rng) < 1 / s ? randn(rng) : 0.0
        BRM._brm_svi_step!(opt, x, [g]; clip)
    end
    sqrt(opt.v[1] / (1 - opt.β2^opt.t))
end

@stestset "a coordinate most batches do not read keeps its scale" begin
    for s in (10.0, 100.0, 1000.0)
        unclipped = svi_sparse_rms(s; clip=Inf)
        @test svi_sparse_rms(s; clip=BRM._BRM_SVI_GLOBAL_CLIP * sqrt(s)) ≈ unclipped rtol=0.05
    end
    # The control: a bound tighter than √s shrinks the rms and with it the next
    # limit, which is what the √(N/B) in the bound prevents.
    @test svi_sparse_rms(1000.0; clip=3.0) < 1e-3 * svi_sparse_rms(1000.0; clip=Inf)
end

@stestset "a window's statistics: trimmed mean and its standard error" begin
    rng = Xoshiro(3)
    # Below 10 iterations nothing is trimmed: the mean and the variance.
    w = randn(rng, 9)
    s = BRM._brm_svi_window(w)
    @test s.mean == mean(w)
    @test s.center ≈ mean(w)
    @test s.spread ≈ var(w)
    # `spread / W` is the variance of `center`, on Gaussian and on skewed draws.
    for draw in (n -> randn(rng, n), n -> -randexp(rng, n))
        z = map(1:4000) do _
            a, b = BRM._brm_svi_window(draw(100)), BRM._brm_svi_window(draw(100))
            (b.center - a.center) / sqrt((a.spread + b.spread) / 100)
        end
        @test var(z) ≈ 1 atol=0.15
        @test abs(mean(z)) < 0.1
    end
    # One draw moves neither the center nor the spread far, however extreme.
    w = randn(rng, 100)
    clean = BRM._brm_svi_window(w)
    w[37] = -1e21
    spiked = BRM._brm_svi_window(w)
    @test spiked.mean < -1e18
    @test abs(spiked.center - clean.center) < 0.1
    @test spiked.spread < 1.5 * clean.spread
    # Up to a tenth of the window, the same.
    w = randn(rng, 100)
    w[1:10] .= -1e21
    @test abs(BRM._brm_svi_window(w).center) < 0.5
    # The trace it reads is not reordered.
    trace = randn(rng, 100)
    copied = copy(trace)
    BRM._brm_svi_window(view(trace, 1:100))
    @test trace == copied
end

@stestset "the stopping test: one draw neither stops a rising fit nor hides a stalled one" begin
    rng = Xoshiro(4)
    window(level) = level .+ randn(rng, 100)
    stats = BRM._brm_svi_window
    stalled = BRM._brm_svi_stalled
    # The ELBO rose by 10 standard errors between the windows.
    a, b = window(0.0), window(1.5)
    @test !stalled(stats(a), stats(b), 100)
    # With an extreme draw in the older window, a test of means and variances
    # passes although the rise is plain: its variance swamps every gap.
    spiked = copy(a)
    spiked[50] = -1e6
    mean_test(x, y) = mean(y) - mean(x) < 2 * sqrt((var(x) + var(y)) / 100)
    @test !mean_test(a, b)
    @test mean_test(spiked, b)
    @test !stalled(stats(spiked), stats(b), 100)
    # With one in the newer window, the mean even reads as a fall.
    spiked = copy(b)
    spiked[50] = -1e6
    @test mean_test(a, spiked)
    @test !stalled(stats(a), stats(spiked), 100)
    # A fit that has stopped rising is still seen as stalled.
    a, b = window(0.0), window(0.0)
    @test stalled(stats(a), stats(b), 100)
    b[50] = 1e21
    @test stalled(stats(a), stats(b), 100)
end

# A field on the log scale that tight observations read through a small
# coefficient, so they bound exp(level + field) only from above and q stays near
# the prior below that bound. Its sd sits on the log scale, so E_q log p = -Inf.
function svi_weak_field_data(rng, N; nobs=6, noise=0.05, K=64)
    Φ = map(1:N) do _
        reduce(vcat, [sqrt(2 / K) .* cos.(2π .* rand(rng, K) .+ 3 .* randn(rng, K) .* rand(rng))'
            for _ in 1:nobs])
    end
    y = [0.3 .* randn(rng) .+ noise .* randn(rng, nobs) for _ in 1:N]
    (; subject=["s$i" for i in 1:N], Φ, y)
end

svi_weak_field_model(data) = @brm data begin
    level ~ Normal(3, 2)
    field_sd ~ Exponential(1)
    z[1:64] .~ Normal.(0.0, 1.0)
    a ~ 1 + (1 | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ 0.001 .* exp.(level .+ field_sd .* (Φ[i] * z)), 0.05)
    end
end

@stestset "a fit whose draws reach the steep part of the density keeps moving" begin
    # On this seed a run without the bound ends with every global's step more than
    # ten times too small. A healthy coordinate reads that 8% of the time, by
    # chance, since |m| averages only about ten draws.
    m = brm_nlme_model(RKBRMI(svi_weak_field_model(svi_weak_field_data(Xoshiro(10), 12)));
        ad_backend=NLME_AD)
    controls = (; batch_size=4, window=100, ndraws=0, max_iterations=20_000)
    fit = brm_svi_fit(m; rng=Xoshiro(1000), controls...)
    d = fit.diagnostics
    @test d.converged
    @test d.clipped_steps > 0
    adam = d.optimizer.run.adam
    b1, b2 = 1 - adam.β1^adam.t, 1 - adam.β2^adam.t
    healthy = sqrt((1 - adam.β1) / (1 + adam.β1))
    step = @. (abs(adam.m / b1) / healthy) / sqrt(adam.v / b2)
    @test mean(step[1:length(d.q.μ)] .< 0.1) < 0.5
    # Every window is reported, the last polishing one too, by both statistics.
    @test d.iterations % 100 == 0
    @test length(d.elbo_windows) == length(d.elbo_windows_trimmed) == d.iterations ÷ 100
    @test last(d.elbo_windows) ≈ mean(d.elbo[end-99:end])
    # The count survives the polishing restart of Adam and a continuation, which
    # is the same run to the last bit.
    stopped = brm_svi_fit(m; rng=Xoshiro(1000), controls..., stop_after=d.iterations - 150)
    @test BRM._brm_svi_polishing(stopped.diagnostics.optimizer.run)
    resumed = brm_svi_resume(m, stopped; ndraws=0)
    @test svi_same_state(d.optimizer, resumed.diagnostics.optimizer)
    @test resumed.diagnostics.clipped_steps == d.clipped_steps
    @test resumed.diagnostics.elbo_windows_trimmed == d.elbo_windows_trimmed
end
