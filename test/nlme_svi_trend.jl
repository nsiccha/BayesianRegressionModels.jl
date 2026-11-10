# Exploration's trend test in minibatch SVI (src/nlme_svi.jl, BRM 0.2.262; snag
# brm-svi-fit-rkbr-f41cb255). Exploration used to end at the first window whose
# trimmed ELBO had not risen detectably since `patience` windows before. Two windows
# cannot see a rise below about twice their noise, and a global that meets every
# unit on a ridge climbs that slowly: a non-centred step `m + s z_i` that each unit's
# data pin only in sum lets the spread `s` fall only as fast as every `z_i` grows.
# Exploration now also needs the least-squares slope over the last `trend_windows`
# windows (default `4 patience + 1`) not to exceed twice its standard error.
#
# RUN: julia --project=test test/nlme_svi_trend.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LinearAlgebra, Random, Statistics
import LogDensityProblems

# Windows of a run whose window centres are `c` and whose within-window variance
# (`spread`) is `v`, as `_brm_svi_window` reports them.
svi_trend_windows(c, v) = [(; mean=x, center=x, spread=v) for x in c]

@stestset "the trend test reads a slow rise two windows cannot see" begin
    W, P, L = 100, 10, 41
    c = (; window=W, patience=P, min_windows=40, trend_windows=L)
    # A rise of 0.4 nats per window under window noise of sd 2.5: over 10 windows
    # it rises 4 nats, below the two-window test's 2√(2 · 2.5²) ≈ 7.1, so that test
    # passes at the floor, but 41 windows' slope is about 15 of its standard errors.
    rng = Xoshiro(1)
    centres = -1000 .+ 0.4 .* (1:200) .+ 2.5 .* randn(rng, 200)
    ws = svi_trend_windows(centres, 2.5^2 * W)
    @test BRM._brm_svi_stalled(ws[40-P], ws[40], W)
    @test BRM._brm_svi_rising(ws, 40, L)
    @test !BRM._brm_svi_exploration_stalled(ws, 40, c)
    run = (; windows=ws)
    @test BRM._brm_svi_explored(run, c) == 0
    # Off (`trend_windows = 0`) and in controls written before the keyword, the
    # two-window test alone decides, as in BRM 0.2.261.
    old = Base.structdiff(c, NamedTuple{(:trend_windows,)})
    first_pass = findfirst(w -> BRM._brm_svi_stalled(ws[w-P], ws[w], W), 40:200) + 39
    @test BRM._brm_svi_trend_windows(old) == 0
    @test BRM._brm_svi_explored(run, old) == first_pass
    @test BRM._brm_svi_explored(run, merge(c, (; trend_windows=0))) == first_pass
    # Once the rise stops, the test lets exploration end.
    flat = vcat(centres, fill(centres[end], 100) .+ 2.5 .* randn(rng, 100))
    fw = svi_trend_windows(flat, 2.5^2 * W)
    stop = BRM._brm_svi_explored((; windows=fw), c)
    @test 200 < stop <= 300
    @test !BRM._brm_svi_rising(fw, stop, L)
end

@stestset "a stationary ELBO ends exploration as before" begin
    W, P = 100, 10
    c = (; window=W, patience=P, min_windows=40, trend_windows=41)
    old = merge(c, (; trend_windows=0))
    # On stationary noise the trend test holds exploration past the two-window
    # test's stop only by its chance level: across 200 seeds the median delay is
    # 0 windows.
    delays = map(1:200) do seed
        rng = Xoshiro(seed)
        ws = svi_trend_windows(-1000 .+ 2.5 .* randn(rng, 400), 2.5^2 * W)
        BRM._brm_svi_explored((; windows=ws), c) - BRM._brm_svi_explored((; windows=ws), old)
    end
    @test all(>=(0), delays)
    @test median(delays) == 0
    @test mean(delays .> 0) < 0.1
    # Fewer than three windows, or an exact line, read as they should.
    ws = svi_trend_windows([1.0, 2.0], 1.0)
    @test !BRM._brm_svi_rising(ws, 2, 41)
    line = svi_trend_windows(collect(1.0:10.0), 1.0)
    @test BRM._brm_svi_rising(line, 10, 41)
    @test !BRM._brm_svi_rising(svi_trend_windows(fill(3.0, 10), 1.0), 10, 41)
    @test !BRM._brm_svi_rising(svi_trend_windows(collect(10.0:-1.0:1.0), 1.0), 10, 41)
end

# The ridge: N units whose rows each pin a multiplicative step `exp(m + s z_i)`
# (non-centred, `z_i ~ Normal(0, 1)`) beside a level and noise. The steps share
# one value up to a spread of 0.02, so the posterior spread is small, but the fit
# starts it near its Exponential(1) prior's scale, and `s` falls only as fast as
# every unit's `z_i` grows: the ELBO climbs too slowly for two windows to see.
function svi_ridge_data(rng, N; nobs=100, gm=0.17, gs=0.02, noise=0.1)
    x = [1 .+ rand(rng, nobs) for _ in 1:N]
    lg = gm .+ gs .* randn(rng, N)
    lvl = 0.5 .* randn(rng, N)
    y = [lvl[i] .+ exp(lg[i]) .* x[i] .+ noise .* randn(rng, nobs) for i in 1:N]
    (; unit=["u$i" for i in 1:N], x, y, zero1=[[0.0] for _ in 1:N])
end
svi_ridge_model(data) = @brm data begin
    level ~ 1 + (1 | unit)
    gain_mean ~ Normal(0, 0.5)
    gain_spread ~ Exponential(1)
    noise ~ Exponential(1)
    @plate for i in eachindex(level)
        z ~ Normal(zero1[i], 1)
        steps = gain_mean .+ gain_spread .* z
        y[i] ~ Normal(level[i] .+ exp(steps[1]) .* x[i], noise)
    end
end

# The trimmed ELBO of each fit's factor on one shared set of base draws (brm-use
# nlme.md, "What `q` is").
function svi_ridge_elbos(rk, fits; ndraws=400, rng=Xoshiro(7))
    p = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    ds = [f.diagnostics for f in fits]
    q1 = first(ds).q
    entropy(q) = sum(q.ρ) - sum(sum(log, diag(U)) for U in q.U; init=0.0)
    xs = [copy(first(fits).draws[:, 1]) for _ in fits]
    vals = [Float64[] for _ in fits]
    for _ in 1:ndraws
        ε = randn(rng, length(q1.μ))
        εu = [randn(rng, length(m)) for m in q1.M]
        for (k, d) in enumerate(ds)
            xs[k][d.globals] .= d.q.μ .+ exp.(d.q.ρ) .* ε
            for (i, blk) in enumerate(d.units)
                xs[k][blk] .= d.q.M[i] .+ LowerTriangular(Matrix(d.q.U[i]))' \ εu[i]
            end
            push!(vals[k], LogDensityProblems.logdensity(p, xs[k]) + entropy(d.q))
        end
    end
    trim(v) = (s = sort(v); k = floor(Int, 0.1length(v)); mean(s[k+1:end-k]))
    trim.(vals)
end

const SVI_RIDGE_OPT = (; subject_group=:unit, batch_size=4, ndraws=200)

@stestset "exploration runs on while a ridge still climbs" begin
    rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
    a = brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT...)
    b = brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT..., trend_windows=0)
    long = brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT..., min_windows=400)
    sa, sb = a.diagnostics.stages, b.diagnostics.stages
    W = 100
    @test a.diagnostics.converged && b.diagnostics.converged
    # Both rules explore identically, to the bit, until the two-window test passes;
    # on this ridge it passes at the 40-window floor, where the trend still rises.
    @test sb.exploring == 40W
    @test a.diagnostics.elbo[1:sb.exploring] == b.diagnostics.elbo[1:sb.exploring]
    @test sa.exploring >= 2sb.exploring && sa.exploring % W == 0
    # The two-window stop leaves the step mean and spread far from where a
    # 400-window exploration puts them; the trend stop does not. Measured over
    # 5 seeds each at 8 and 16 units: exp(m) 0.90-0.98 against 1.17-1.19, and
    # the trend stop's factor within 0.5 nats of the long run's, 14.6-36.2 above
    # the two-window stop's.
    gm(f) = median(exp.(rk_restore_draws(rk, f.draws).gain_mean))
    @test gm(b) < 1.0
    @test abs(gm(a) - gm(long)) < 0.01
    ea, eb, el = svi_ridge_elbos(rk, (a, b, long))
    @test ea - eb > 10
    @test abs(ea - el) < 1
end

@stestset "a checkpoint continues on its own rule; an older one on two windows" begin
    rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
    full = brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT...)
    part = brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT..., stop_after=3000)
    state = brm_svi_optimizer_checkpoint(part)
    @test state.controls.trend_windows == 41
    @test svi_same_result(brm_svi_resume(rk, state; subject_group=:unit, ndraws=200), full)
    # A checkpoint of BRM 0.2.261 or earlier records no `trend_windows`: it ends
    # exploration on the two-window test, as it would have.
    controls = Base.structdiff(state.controls, NamedTuple{(:trend_windows,)})
    old = BRMSVIOptimizerState(state.format, state.identity, state.previous, controls, state.run)
    @test svi_same_result(brm_svi_resume(rk, old; subject_group=:unit, ndraws=200),
        brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT..., trend_windows=0))
    # The keyword: nothing (4 patience + 1), 0, or at least 3 windows.
    c = brm_svi_optimizer_checkpoint(brm_svi_fit(rk; rng=Xoshiro(1), SVI_RIDGE_OPT...,
        patience=3, stop_after=1))
    @test c.controls.trend_windows == 13
    for bad in (-1, 1, 2, 2.5)
        @test_throws "trend_windows" brm_svi_fit(rk; SVI_RIDGE_OPT..., trend_windows=bad)
    end
    @test_throws "trend_windows" brm_svi_fit(rk; laplace=GaussianBlockLaplace(),
        trend_windows=41)
end
