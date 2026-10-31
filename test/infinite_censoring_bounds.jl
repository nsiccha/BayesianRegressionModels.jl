# Exact infinite response bounds, including mixed censored and detected rows.
# Run: julia --startup-file=no --project=. test/infinite_censoring_bounds.jl

using Test, BayesianRegressionModels, Distributions, BridgeStan
import StanBlocks

builder = @brm begin
    mu ~ Normal(0.0, 1.0)
    interval_y ~ interval_censored(LogNormal(mu, 0.5); upper=interval_upper)
    normal_left ~ censored(Normal(mu, 0.5); lower=left_limit)
    student_left ~ censored(LocationScale(mu, 0.5, TDist(4.0)); lower=left_limit)
    normal_right ~ censored(Normal(mu, 0.5); upper=right_limit)
    student_right ~ censored(LocationScale(mu, 0.5, TDist(4.0)); upper=right_limit)
    normal_both ~ censored(Normal(mu, 0.5); lower=both_lower, upper=both_upper)
    student_both ~ censored(LocationScale(mu, 0.5, TDist(4.0)); lower=both_lower, upper=both_upper)
end

function evidence(mu)
    n = Normal(mu, 0.5)
    t = LocationScale(mu, 0.5, TDist(4.0))
    ln = LogNormal(mu, 0.5)
    value = logpdf(Normal(0.0, 1.0), mu)
    value += logcdf(ln, 0.2) + log(cdf(ln, 2.5) - cdf(ln, 1.5))
    # These branch choices come from the known evidence categories, independently
    # of BRM's wrapper and bound comparisons.
    for d in (n, t)
        value += logcdf(d, -0.5) + logpdf(d, 0.3)
        value += logccdf(d, 0.7) + logpdf(d, 0.3)
        value += logcdf(d, -0.5) + logpdf(d, 0.3) + logccdf(d, 0.7)
    end
    value
end

@testset "Exact response-bound acceptance" begin
    for (label, lo, hi) in (("finite control", -2.0, 2.0),
                            ("negative infinity", -Inf, 2.0),
                            ("positive infinity", -2.0, Inf),
                            ("both infinities", -Inf, Inf))
        @testset "$label" begin
            data = (; interval_y=[0.0,1.5], interval_upper=[0.2,2.5],
                normal_left=[-0.5,0.3], student_left=[-0.5,0.3], left_limit=[-0.5,lo],
                normal_right=[0.7,0.3], student_right=[0.7,0.3], right_limit=[0.7,hi],
                normal_both=[-0.5,0.3,0.7], student_both=[-0.5,0.3,0.7],
                both_lower=[-0.5,lo,lo], both_upper=[hi,hi,0.7])
            saved = deepcopy(data)
            sb = SBBRMI(builder(data); mod=@__MODULE__)
            stanc = StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false)
            @test stanc.ok
            problem = stan_instantiate(sb; path=joinpath(tempdir(), "infinite_bounds.stan"))
            @test BridgeStan.param_unc_names(problem.model) == ["mu"]
            for mu in (-0.4, 0.0, 0.6)
                value, gradient = BridgeStan.log_density_gradient(problem.model, [mu];
                    propto=false, jacobian=false)
                @test isfinite(value)
                @test all(isfinite, gradient)
                @test value ≈ evidence(mu) atol=1e-10 rtol=1e-12
                h = 1e-5
                numerical = (evidence(mu+h) - evidence(mu-h)) / (2h)
                @test only(gradient) ≈ numerical atol=2e-7 rtol=2e-7
            end
            @test isequal(data, saved)
        end
    end
end
