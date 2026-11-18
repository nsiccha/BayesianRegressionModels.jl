# The ridge of test/nlme_svi_trend.jl "exploration runs on while a ridge still climbs"
# at unit_reoptimize = 0 and 1 (todo 12maq21): each fit's trimmed ELBO on shared base
# draws, read two ways: the test's own reading (`svi_ridge_elbos`: the unit factors in
# `diagnostics.q`, the conditionals at the global mean, drawn independently of the
# globals) and the fitted family's (units drawn through their linear coupling,
# z = M + V ζ + U⁻ᵀ ε, ζ the global draw's standard normals at the coupled columns).
#
#     julia --project=test research/svi_reoptimize/trend_ridge_elbos.jl [k...]
const KS = isempty(ARGS) ? [0, 1] : parse.(Int, ARGS)
empty!(ARGS)
const ROOT = get(ENV, "BRM_ROOT", pwd())
include(joinpath(ROOT, "test", "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, Printf
import LogDensityProblems

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
const OPT = (; subject_group=:unit, batch_size=4, ndraws=200, stationarity=false,
    centering=nothing)

function elbos(rk, fits; ndraws=400, rng=Xoshiro(7))
    p = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    ds = [f.diagnostics for f in fits]
    q1 = first(ds).q
    entropy(q) = sum(q.ρ) - sum(sum(log, diag(U)) for U in q.U; init=0.0)
    xs = [copy(first(fits).draws[:, 1]) for _ in fits]
    plain, coupled = [Float64[] for _ in fits], [Float64[] for _ in fits]
    for _ in 1:ndraws
        ε = randn(rng, length(q1.μ))
        εu = [randn(rng, length(m)) for m in q1.M]
        for (k, d) in enumerate(ds), withV in (false, true)
            xs[k][d.globals] .= d.q.μ .+ exp.(d.q.ρ) .* ε
            for (i, blk) in enumerate(d.units)
                z = d.q.M[i] .+ LowerTriangular(Matrix(d.q.U[i]))' \ εu[i]
                if withV && d.coupling !== nothing
                    u = d.coupling.units[i]
                    isempty(u.V) || (z[u.rows] .+= u.V * ε[u.columns])
                end
                xs[k][blk] .= z
            end
            push!(withV ? coupled[k] : plain[k],
                LogDensityProblems.logdensity(p, xs[k]) + entropy(d.q))
        end
    end
    trim(v) = (s = sort(v); j = floor(Int, 0.1length(v)); mean(s[j+1:end-j]))
    trim.(plain), trim.(coupled)
end

rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
for k in KS
    fits = [brm_svi_fit(rk; rng=Xoshiro(1), OPT..., unit_reoptimize=k),
            brm_svi_fit(rk; rng=Xoshiro(1), OPT..., unit_reoptimize=k, trend_windows=0),
            brm_svi_fit(rk; rng=Xoshiro(1), OPT..., unit_reoptimize=k, min_windows=400)]
    plain, coupled = elbos(rk, fits)
    for (name, f, ep, ec) in zip(("trend stop", "two-window stop", "400 windows"), fits,
            plain, coupled)
        d = f.diagnostics
        r = rk_restore_draws(rk, f.draws)
        V = d.coupling === nothing ? 0.0 :
            sqrt(sum(sum(abs2, u.V) for u in d.coupling.units; init=0.0))
        @printf("k=%d %-16s exploring %6d iterations %6d | gain exp(m) median %.3f | spread mean %.4f | ELBO without V %10.2f with V %10.2f | ‖V‖ %.3f\n",
            k, name, d.stages.exploring, d.iterations, median(exp.(r.gain_mean)),
            mean(r.gain_spread), ep, ec, V)
    end
    flush(stdout)
end
