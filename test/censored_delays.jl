# primary_censored / double_interval_censored: the Julia law against an
# independent quadrature of the convolution, its tail accuracy, draws, and
# native Enzyme Reverse gradients (the RKBRMI observation law).
#
#     julia --project=test test/censored_delays.jl
using Test, Random, Statistics
using BayesianRegressionModels
using BayesianRegressionModels: censored_delay_logpdf, _cd_lognormal_parts,
    _cd_gamma_parts, _cd_weibull_parts
using Distributions
import Enzyme
using StanBlocks, BridgeStan, LogDensityProblems, ReactiveKernels, ReactiveKernelsPPL, ADTypes
using DifferentiationInterface: AutoEnzyme

# Tanh-sinh (double-exponential) quadrature, generic in the number type (the
# oracle). It converges geometrically also for the endpoint power behaviour of
# a gamma or Weibull CDF with shape < 1 (F(t) ~ t^k near 0). Nodes near an end
# are placed through exp() so `b - t` and `t - a` keep full precision.
function tanh_sinh(f, a, b; h=1 / 128, smax=4.5)
    T = typeof(float(a))
    h, half = T(h), (b - a)
    total = zero(T)
    for s in -smax:Float64(h):smax
        v = T(pi) / 2 * sinh(T(s))
        e = exp(-2v)
        t = a + half / (1 + e)                        # a + (b - a)(1 + tanh v)/2
        weight = T(pi) / 2 * cosh(T(s)) / cosh(v)^2   # dx/ds of tanh(pi/2 sinh s)
        (t <= a || t >= b) && continue
        total += weight * f(t)
    end
    h * half / 2 * total
end

# The oracle: F_pc(d) = (1/w) ∫_{max(d-w,0)}^{d} F(t) dt, quadrature of the
# delay's CDF only — none of the closed-form partial expectations.
oracle_cdf(dist, d, w) = d <= 0 ? zero(d) :
    tanh_sinh(t -> cdf(dist, t), max(d - w, zero(d)), d) / w
# ∫_l^u of the density (F(t) - F(t - w)) / w: the probability of [l, u].
oracle_mass(dist, l, u, w) =
    tanh_sinh(t -> (cdf(dist, t) - cdf(dist, t - w)) / w, l, u)

@testset "censored delays" begin
    @testset "CDF and survival match the quadrature oracle" begin
        setprecision(BigFloat, 256) do
            cases = [
                (LogNormal(1.8, 0.5), _cd_lognormal_parts, (1.8, 0.5)),
                (LogNormal(0.2, 1.3), _cd_lognormal_parts, (0.2, 1.3)),
                (Weibull(1.7, 6.0), _cd_weibull_parts, (1.7, 6.0)),
                (Weibull(0.6, 2.0), _cd_weibull_parts, (0.6, 2.0)),
            ]
            for (dist, parts, (a, b)) in cases, w in (1.0, 2.5), d in (0.3, 1.0, 2.7, 6.0, 15.0)
                big_dist = typeof(dist).name.wrapper(BigFloat(a), BigFloat(b))
                want = oracle_cdf(big_dist, BigFloat(d), BigFloat(w))
                got = BayesianRegressionModels._cd_cdf(parts, d, w, a, b)
                @test got ≈ Float64(want) rtol = 1e-11 atol = 1e-300
                surv = BayesianRegressionModels._cd_ccdf(parts, d, w, a, b)
                @test surv ≈ Float64(1 - want) rtol = 1e-9 atol = 1e-300
            end
        end
        # Gamma: Float64 oracle (SpecialFunctions has no BigFloat incomplete gamma).
        for (k, θ) in ((2.0, 2.5), (0.7, 4.0)), w in (1.0, 2.5), d in (0.3, 2.7, 6.0, 15.0)
            want = oracle_cdf(Gamma(k, θ), d, w)
            @test BayesianRegressionModels._cd_cdf(_cd_gamma_parts, d, w, k, θ) ≈ want rtol = 1e-9
        end
    end

    @testset "interval probabilities stay accurate in the upper tail" begin
        setprecision(BigFloat, 256) do
            mu, sigma, w = 1.8, 0.5, 1.0
            for y in (5.0, 20.0, 40.0, 60.0)
                want = oracle_mass(LogNormal(BigFloat(mu), BigFloat(sigma)),
                    BigFloat(y), BigFloat(y + 1), BigFloat(w))
                got = censored_delay_logpdf(_cd_lognormal_parts, y, w, 0.0, Inf, 1.0, mu, sigma)
                @test got ≈ Float64(log(want)) rtol = 1e-7
            end
        end
    end

    @testset "double_interval_censored: truncation, windows, shifted primary" begin
        d = double_interval_censored(LogNormal(1.6, 0.5); primary_event=Uniform(0, 1),
            upper=12.0, interval=1.0)
        F(x) = cdf(primary_censored(LogNormal(1.6, 0.5), Uniform(0, 1)), x)
        @test logpdf(d, 4.0) ≈ log((F(5.0) - F(4.0)) / F(12.0)) rtol = 1e-12
        @test logpdf(d, 11.5) == -Inf          # window ends past the truncation bound
        # A primary window [2, 3] is the [0, 1] window shifted by 2.
        shifted = double_interval_censored(LogNormal(1.6, 0.5);
            primary_event=Uniform(2, 3), upper=14.0, interval=1.0)
        @test logpdf(shifted, 6.0) ≈ logpdf(d, 4.0) rtol = 1e-12
        # No secondary window: the truncated density.
        exact = double_interval_censored(LogNormal(1.6, 0.5); primary_event=Uniform(0, 2), upper=10.0)
        pc = primary_censored(LogNormal(1.6, 0.5), Uniform(0, 2))
        @test logpdf(exact, 3.0) ≈ logpdf(pc, 3.0) - log(cdf(pc, 10.0)) rtol = 1e-12
        # The density integrates to one and is the derivative of the CDF.
        # (split at the window width, where the density's derivative jumps)
        @test tanh_sinh(x -> pdf(pc, x), 0.0, 2.0) + tanh_sinh(x -> pdf(pc, x), 2.0, 60.0) +
              ccdf(pc, 60.0) ≈ 1 rtol = 1e-10
        @test (cdf(pc, 3.0 + 1e-6) - cdf(pc, 3.0 - 1e-6)) / 2e-6 ≈ pdf(pc, 3.0) rtol = 1e-6
        @test_throws ArgumentError primary_censored(Normal(0, 1), Uniform(0, 1))
        @test_throws ArgumentError double_interval_censored(LogNormal(0, 1); lower=3, upper=2)
    end

    @testset "draws" begin
        rng = Xoshiro(1)
        pc = primary_censored(LogNormal(1.6, 0.5), Uniform(0, 2))
        x = [rand(rng, pc) for _ in 1:200_000]
        @test mean(x) ≈ mean(LogNormal(1.6, 0.5)) + 1 rtol = 0.01
        @test mean(x .<= 5) ≈ cdf(pc, 5.0) atol = 0.005
        d = double_interval_censored(LogNormal(1.6, 0.5); primary_event=Uniform(0, 1),
            upper=8.0, interval=1.0)
        y = [rand(rng, d) for _ in 1:100_000]
        @test all(v -> isinteger(v) && 0 <= v <= 7, y)
        @test mean(y .== 4) ≈ exp(logpdf(d, 4.0)) atol = 0.005
    end

    @testset "native Enzyme Reverse gradients" begin
        fd(f, x; h=1e-6) = (f(x + h) - f(x - h)) / 2h
        for (parts, a, b) in ((_cd_lognormal_parts, 1.8, 0.5),
                              (_cd_weibull_parts, 1.7, 6.0),
                              (_cd_gamma_parts, 2.0, 2.5))
            f(a, b) = censored_delay_logpdf(parts, 4.0, 1.0, 0.0, 20.0, 1.0, a, b)
            grad = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                Enzyme.Active(a), Enzyme.Active(b))[1]
            @test grad[1] ≈ fd(x -> f(x, b), a) rtol = 1e-6
            @test grad[2] ≈ fd(x -> f(a, x), b) rtol = 1e-6
        end
    end

    @testset "both backends observe the same law" begin
        df = (; delay=[3.0, 4.0, 4.0, 9.0, 7.0, 5.0, 2.0, 0.0, 12.0],
                pwindow=[1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                obs_time=[9.0, 6.0, 9.0, 18.0, 10.0, 12.0, 5.0, 3.0, Inf],
                n=[3, 1, 2, 1, 4, 2, 1, 1, 2], count=[3, 1, 2, 1, 4, 2, 1, 1, 2])
        families = [
            (:LogNormal, (m, s) -> LogNormal(m, s), [(1.6, log(0.5)), (1.2, -0.2), (2.1, -1.1)]),
            (:Gamma, (k, θ) -> Gamma(k, θ), [(log(2.5), log(2.0)), (log(1.3), log(4.0))]),
            (:Weibull, (k, λ) -> Weibull(k, λ), [(log(1.7), log(6.0)), (log(0.9), log(4.0))]),
        ]
        ad = AutoEnzyme(; mode=Enzyme.Reverse)
        for (family, law, points) in families
            builder = if family === :LogNormal
                @brm begin
                    mu ~ 1
                    log(sigma) ~ 1
                    effect(mu, Intercept) ~ Normal(1.0, 1.0)
                    effect(sigma, Intercept) ~ Normal(-0.7, 0.4)
                    delay ~ weighted(double_interval_censored(LogNormal(mu, sigma);
                        primary_event=Uniform(0, pwindow), upper=obs_time, interval=1.0),
                        fweights(n))
                end
            elseif family === :Gamma
                @brm begin
                    log(shape) ~ 1
                    log(scale) ~ 1
                    effect(shape, Intercept) ~ Normal(0.5, 1.0)
                    effect(scale, Intercept) ~ Normal(1.0, 1.0)
                    delay ~ weighted(double_interval_censored(Gamma(shape, scale);
                        primary_event=Uniform(0, pwindow), upper=obs_time, interval=1.0),
                        fweights(n))
                end
            else
                @brm begin
                    log(shape) ~ 1
                    log(scale) ~ 1
                    effect(shape, Intercept) ~ Normal(0.5, 1.0)
                    effect(scale, Intercept) ~ Normal(1.0, 1.0)
                    delay ~ weighted(double_interval_censored(Weibull(shape, scale);
                        primary_event=Uniform(0, pwindow), upper=obs_time, interval=1.0),
                        fweights(n))
                end
            end
            brmi = builder(df)
            prior = family === :LogNormal ? (Normal(1.0, 1.0), Normal(-0.7, 0.4)) :
                (Normal(0.5, 1.0), Normal(1.0, 1.0))
            # The model's log density at (a, log b): `mu`/`log(sigma)` (LogNormal),
            # `log(shape)`/`log(scale)` (Gamma, Weibull), from the Julia law alone.
            function oracle(x)
                a = family === :LogNormal ? x[1] : exp(x[1])
                b = exp(x[2])
                d = [double_interval_censored(law(a, b); primary_event=Uniform(0, w),
                         upper=u, interval=1.0) for (w, u) in zip(df.pwindow, df.obs_time)]
                sum(df.n .* logpdf.(d, df.delay)) + logpdf(prior[1], x[1]) + logpdf(prior[2], x[2])
            end
            fd(x, i; h=1e-6) = (oracle(setindex!(copy(x), x[i] + h, i)) -
                                oracle(setindex!(copy(x), x[i] - h, i))) / 2h
            @testset "$family on SBBRMI" begin
                sb = SBBRMI(brmi; mod=@__MODULE__)
                @test occursin("brm_censored_delay", string(StanBlocks.stan_code(sb)))
                problem = stan_instantiate(sb)
                names = BridgeStan.param_unc_names(problem.model)
                order = occursin("sigma", names[1]) || occursin("scale", names[1]) ? [2, 1] : [1, 2]
                offsets = Float64[]
                for x in points
                    u = zeros(2); u[order] = collect(x)
                    v, g = LogDensityProblems.logdensity_and_gradient(problem, u)
                    push!(offsets, v - oracle(collect(x)))
                    @test g[order] ≈ [fd(collect(x), 1), fd(collect(x), 2)] rtol = 1e-6
                end
                # `~` drops the two normal priors' normalising constants.
                @test maximum(offsets) - minimum(offsets) < 1e-9
            end
            @testset "$family on RKBRMI" begin
                rk = RKBRMI(brmi)
                problem = rk_logdensity_problem(rk; ad_backend=ad)
                names = string.(coordinate_names(build_kernel(rk_bound_plan(rk)).layout))
                order = occursin("sigma", names[1]) || occursin("scale", names[1]) ? [2, 1] : [1, 2]
                for x in points
                    u = zeros(2); u[order] = collect(x)
                    v, g = LogDensityProblems.logdensity_and_gradient(problem, u)
                    @test v ≈ oracle(collect(x)) rtol = 1e-10
                    @test g[order] ≈ [fd(collect(x), 1), fd(collect(x), 2)] rtol = 1e-6
                end
            end
        end
    end

    @testset "exact secondary times and row checks" begin
        df = (; delay=[3.2, 4.7, 1.1], pwindow=[1.0, 2.0, 1.0])
        brmi = @brm df begin
            mu ~ 1
            log(sigma) ~ 1
            delay ~ primary_censored(LogNormal(mu, sigma), Uniform(0, pwindow))
        end
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        u = [1.4, -0.5]
        names = string.(coordinate_names(build_kernel(rk_bound_plan(rk)).layout))
        uu = occursin("sigma", names[1]) ? reverse(u) : u
        prior_part = LogDensityProblems.logdensity(problem, uu) -
            sum(logpdf.(primary_censored.(LogNormal(1.4, exp(-0.5)), Uniform.(0, df.pwindow)), df.delay))
        @test isfinite(prior_part)
        @test occursin("brm_censored_delay", string(StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__))))
        bad = (; delay=[3.0, 8.5], obs_time=[9.0, 9.0])
        builder = @brm begin
            mu ~ 1
            log(sigma) ~ 1
            delay ~ double_interval_censored(LogNormal(mu, sigma); upper=obs_time, interval=1.0)
        end
        @test_throws r"outside the truncation bounds" SBBRMI(builder(bad); mod=@__MODULE__)
        @test_throws r"outside the truncation bounds" RKBRMI(builder(bad))
        @test_throws r"LogNormal`, `Gamma` or `Weibull`" SBBRMI((@brm (; y=[1.0]) begin
            mu ~ 1
            y ~ primary_censored(Normal(mu, 1.0), Uniform(0, 1))
        end); mod=@__MODULE__)
    end
end
