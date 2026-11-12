# Settling between exploration and polishing in minibatch SVI (src/nlme_svi.jl,
# BRM 0.2.242; snag svi-exploration-036fd59f). Exploration's stop test finds the
# ELBO flat at exploration's own step, and `lr_global` moves a global mean by an
# absolute amount whatever its posterior sd: a tight global jitters by tens of
# sds, and the ELBO sits on that noise floor. Polishing used to start there and
# average its own climb out of it. Settling runs polishing's scale-relative step
# at a constant size, without averaging, until the same test passes on its own
# windows (or for `settle_windows` windows), and polishing continues from there.
#
# RUN: julia --project=test test/nlme_svi_settling.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LinearAlgebra, Random, Statistics
import LogDensityProblems

# Isolate the legacy ELBO-only phase schedule in these short runs.
const SVI_SETTLE_OPT = (; batch_size=4, window=20, min_windows=6, patience=3,
    polish_windows=4, max_iterations=3000, stationarity=false, centering=nothing)

svi_settle_gauss() = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(101), 12)));
    ad_backend=NLME_AD)

@stestset "settling starts where exploration stopped; 0 windows is the earlier schedule" begin
    m = svi_settle_gauss()
    W, P = SVI_SETTLE_OPT.window, SVI_SETTLE_OPT.patience
    a = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., ndraws=5)
    b = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., settle_windows=0, ndraws=5)
    sa, sb = a.diagnostics.stages, b.diagnostics.stages
    @test a.diagnostics.converged && b.diagnostics.converged
    # The two schedules explore identically, to the bit, until exploration stops.
    @test sa.exploring == sb.exploring >= SVI_SETTLE_OPT.min_windows * W
    @test a.diagnostics.elbo[1:sa.exploring] == b.diagnostics.elbo[1:sb.exploring]
    # The earlier schedule polishes straight away.
    @test sb.settling == 0 && sb.settled && sb.settle_windows == 0
    @test b.diagnostics.optimizer.run.polish_start == sb.exploring
    # Settling runs until its own test passes, on at least patience + 1 windows.
    @test sa.settled && sa.settle_windows == 100
    @test sa.settling >= (P + 1) * W && sa.settling % W == 0
    @test a.diagnostics.optimizer.run.polish_start == sa.exploring + sa.settling
    @test sa.polishing == sb.polishing == SVI_SETTLE_OPT.polish_windows * W
    @test a.diagnostics.iterations == sa.exploring + sa.settling + sa.polishing
    # The cap ends settling, and the stages say that it did.
    c = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., settle_windows=2, ndraws=5)
    sc = c.diagnostics.stages
    @test sc.settling == 2W && !sc.settled && c.diagnostics.converged
    @test c.diagnostics.elbo[1:sc.exploring] == a.diagnostics.elbo[1:sa.exploring]
    @test_throws "settle_windows" brm_svi_fit(m; SVI_SETTLE_OPT..., settle_windows=-1)
    # Mid-run, the stages count what has run so far.
    part = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., stop_after=sa.exploring + 7,
        ndraws=0)
    @test part.diagnostics.stages == (; exploring=sa.exploring, settling=7, polishing=0,
        settled=false, settle_windows=100)
    early = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., stop_after=11, ndraws=0)
    @test early.diagnostics.stages.exploring == 11 && early.diagnostics.stages.settling == 0
end

@stestset "a settling run continues exactly from a checkpoint; an older one does not settle" begin
    m = svi_settle_gauss()
    full = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., ndraws=5)
    E = full.diagnostics.stages.exploring
    part = brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., stop_after=E + 7, ndraws=0)
    state = brm_svi_optimizer_checkpoint(part)
    # Completed windows, as "exploring, window k" counts them.
    @test occursin("settling, window 0", sprint(show, state))
    rest = brm_svi_resume(m, state; ndraws=5)
    @test svi_same_result(rest, full)
    # A checkpoint of BRM 0.2.240 or 0.2.241 records no `settle_windows`: it
    # continues with the schedule it started with.
    early = brm_svi_optimizer_checkpoint(brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT...,
        stop_after=37, ndraws=0))
    controls = Base.structdiff(early.controls, NamedTuple{(:settle_windows,)})
    @test !haskey(controls, :settle_windows)
    old = BRMSVIOptimizerState(early.format, early.identity, early.previous, controls, early.run)
    continued = brm_svi_resume(m, old; ndraws=5)
    @test continued.diagnostics.stages.settling == 0
    @test svi_same_result(continued,
        brm_svi_fit(m; rng=Xoshiro(102), SVI_SETTLE_OPT..., settle_windows=0, ndraws=5))
end

# Globals every unit reads, with posterior sds near 7e-4: N subjects of NOBS rows,
# K population coefficients on row covariates, a per-subject effect u[i] (read by
# its cell alone, so ownership makes it the unit's), known noise sd. A no-op
# `(1 | subject)` with sd ≈ e^-7 defines the units. The target is exactly
# Gaussian but in that sd, which sits at its prior's mode, so q* (exact means,
# the Hessian's diagonal blocks as precisions) is the family's optimum and
# KL(q || p) - KL(q* || p) is exact.
function svi_tight_data(rng; N=1000, nobs=20, K=32, sigma=0.1)
    beta = randn(rng, K)
    xs = [[randn(rng, nobs) for _ in 1:N] for _ in 1:K]
    u = 0.5 .* randn(rng, N)
    y = [sum(xs[k][i] .* beta[k] for k in 1:K) .+ u[i] .+ sigma .* randn(rng, nobs)
         for i in 1:N]
    cols = NamedTuple{Tuple(Symbol("x$k") for k in 1:K)}(Tuple(xs))
    (; subject=["s$i" for i in 1:N], state=collect(1:N), cols..., y, sigma, nsub=N)
end

function svi_tight_model(data, K)
    eta = Expr(:call, :.+, :(u[i] .+ mu[i]),
        (:($(Symbol("x$k"))[i] .* beta[$k]) for k in 1:K)...)
    BayesianRegressionModels.brm(quote
        beta[1:$K] .~ Normal(0.0, 10.0)
        u[1:nsub] .~ Normal(0.0, 0.5)
        sd(mu, subject) ~ LogNormal(-7.0, 0.01)
        mu ~ 0 + (1 | subject)
        @plate for i in eachindex(state)
            y[i] ~ Normal($eta, sigma)
        end
    end; mod=@__MODULE__)(data)
end

function svi_tight_exact(rk)
    p = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    n = LogDensityProblems.dimension(p)
    names = string.(coordinate_names(rk.model.layout))
    scale = findall(c -> occursin(".tau", c), names)
    function hessian(x0)
        _, g0 = LogDensityProblems.logdensity_and_gradient(p, x0)
        H = zeros(n, n)
        for j in 1:n
            δ = j in scale ? 1e-4 : 1.0
            xp = copy(x0); xp[j] += δ
            H[:, j] .= -(LogDensityProblems.logdensity_and_gradient(p, xp)[2] .- g0) ./ δ
        end
        Symmetric((H + H') / 2), g0
    end
    # Newton steps: exact in one but for the no-op sd, whose coupling to the
    # standardized effects moves with them, so the second one recomputes H.
    x = zeros(n); x[scale] .= -7.0
    local H
    for _ in 1:2
        H, g = hessian(x)
        x = x .+ cholesky(H) \ g
    end
    (; H, m=x, n, gradient=maximum(abs, LogDensityProblems.logdensity_and_gradient(p, x)[2]))
end

function svi_tight_gap(ex, q, d)
    H, gl = ex.H, d.globals
    Σg = Diagonal(exp.(2 .* q.ρ)) + q.Β * q.Β'
    Hgg = Matrix(H[gl, gl])
    δ = zeros(ex.n); δ[gl] .= q.μ .- ex.m[gl]
    trace, logdetq, logdetstar = tr(Hgg * Σg), logdet(Σg), logdet(Hgg)
    for (i, blk) in enumerate(d.units)
        δ[blk] .= q.M[i] .- ex.m[blk]
        Hb, Σb = Matrix(H[blk, blk]), inv(Matrix(q.Λ[i]))
        trace += tr(Hb * Σb); logdetq += logdet(Σb); logdetstar += logdet(Hb)
    end
    gap = (trace + dot(δ, H * δ) - ex.n - logdetq - logdetstar) / 2
    z = (q.μ .- ex.m[gl]) .* sqrt.(diag(Hgg))
    (; gap, mean_rms_z=sqrt(mean(abs2, z)))
end

@stestset "settling brings tight globals to the family's optimum" begin
    K = 32
    rk = RKBRMI(svi_tight_model(svi_tight_data(Xoshiro(1); K), K))
    ex = svi_tight_exact(rk)
    @test ex.gradient < 1e-2
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    # The exact block-factorized Gaussian oracle uses the plain coordinates.
    opt = (; batch_size=8, threads=1, ndraws=0, centering=nothing)
    function observe(s)
        r = s.stationarity.current
        if hasproperty(r, :units)
            @info "tight globals stationarity" iteration=s.iteration phase=s.phase status=s.status globals=r.globals failed_units=count(u -> !u.passed, r.units) rhat=r.rhat ess=r.ess
        end
    end
    a = brm_svi_fit(m; opt..., rng=Xoshiro(101), snapshot=observe, snapshot_every=10000)
    # Compare with both earlier schedules on the same target and seed: the
    # pre-settling schedule and the ELBO-only settling that the guard replaces.
    b = brm_svi_fit(m; opt..., rng=Xoshiro(101), settle_windows=0, stationarity=false)
    c = brm_svi_fit(m; opt..., rng=Xoshiro(101), stationarity=false)
    ga, gb, gc = map(f -> svi_tight_gap(ex, f.diagnostics.q, f.diagnostics), (a, b, c))
    @info "tight globals" settle=ga no_settling=gb elbo_only=gc stages=a.diagnostics.stages
    @test a.diagnostics.converged
    @test a.diagnostics.stages.settled
    @test !a.diagnostics.stationarity.exploration.passed
    @test a.diagnostics.stationarity.current.passed
    @test ga.gap <= gc.gap
    # Measured on a Linux fleet host, Julia 1.10.12: 61.6 against 111.3 nats; global means
    # 0.24 against 0.92 optimal sds rms.
    @test ga.gap < 0.75 * gb.gap
    @test ga.mean_rms_z < 0.5 < gb.mean_rms_z
end
