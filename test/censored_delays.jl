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

# Adaptive Simpson quadrature, generic in the number type (the oracle).
function simpson(f, a, b; tol)
    T = typeof(float(a))
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    whole = (b - a) * (fa + 4fc + fb) / 6
    _simpson(f, a, b, fa, fb, fc, whole, T(tol), 60)
end
function _simpson(f, a, b, fa, fb, fc, whole, tol, depth)
    c = (a + b) / 2
    d, e = (a + c) / 2, (c + b) / 2
    fd, fe = f(d), f(e)
    left = (c - a) * (fa + 4fd + fc) / 6
    right = (b - c) * (fc + 4fe + fb) / 6
    if depth <= 0 || abs(left + right - whole) <= 15tol
        return left + right + (left + right - whole) / 15
    end
    _simpson(f, a, c, fa, fc, fd, left, tol / 2, depth - 1) +
        _simpson(f, c, b, fc, fb, fe, right, tol / 2, depth - 1)
end

# The oracle: F_pc(d) = (1/w) ∫_{max(d-w,0)}^{d} F(t) dt, quadrature of the
# delay's CDF only — none of the closed-form partial expectations.
oracle_cdf(dist, d, w; tol=1e-30) = d <= 0 ? zero(d) :
    simpson(t -> cdf(dist, t), max(d - w, zero(d)), d; tol) / w
oracle_mass(dist, l, u, w; tol=1e-30) =
    simpson(t -> (cdf(dist, t) - cdf(dist, t - w)) / w, l, u; tol)  # ∫ density = mass

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
            want = oracle_cdf(Gamma(k, θ), d, w; tol=1e-15)
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
        @test simpson(x -> pdf(pc, x), 0.0, 200.0; tol=1e-12) ≈ 1 rtol = 1e-8
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
end
