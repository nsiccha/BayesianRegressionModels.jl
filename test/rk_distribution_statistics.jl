# Consumer acceptance for graph-visible distribution statistics.
# Numerical agreement alone does not establish graph visibility or hoisting;
# research/distribution_graphs audits the public emitted-source placement.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics, SpecialFunctions

module OwnRKStatisticNames
using Distributions: LogNormal
mean(x) = 2x
var(x) = 3x
logcdf(x, y) = x + y
logccdf(x, y) = x - y
end

const StatisticLogNormalAlias = Distributions.LogNormal
const statistic_mean_alias = Statistics.mean

@stestset "RK distribution statistics preserve constructor and callable identities" begin
    function lower(ex, scope=@__MODULE__)
        bindings = Pair{Symbol,Any}[]
        source = BRM._rk_kernel_bind_calls(ex, scope, bindings, Set{Symbol}())
        stats = [last(p) for p in bindings if last(p) isa BRM._RKLogNormalStatistic]
        (; source, stats)
    end
    for (ex, statistic, side, args) in (
        (:(mean(LogNormal())), :mean, :lower, (0.0, 1.0, 0.0)),
        (:(var(LogNormal(mu))), :var, :lower, (:mu, 1.0, 0.0)),
        (:(Statistics.mean(Distributions.LogNormal(mu, sigma))), :mean, :lower, (:mu, :sigma, 0.0)),
        (:(statistic_mean_alias(StatisticLogNormalAlias(mu, sigma))), :mean, :lower, (:mu, :sigma, 0.0)),
        (:(mean(truncated(LogNormal(mu, sigma); lower=lo))), :mean, :lower, (:mu, :sigma, :lo)),
        (:(var(truncated(LogNormal(mu, sigma); upper=hi))), :var, :upper, (:mu, :sigma, :hi)),
        (:(mean(truncated(LogNormal(mu, sigma), nothing, hi))), :mean, :upper, (:mu, :sigma, :hi)),
        (:(var(truncated(LogNormal(mu, sigma), lo, nothing))), :var, :lower, (:mu, :sigma, :lo)),
        (:(logcdf(LogNormal(mu, sigma), bound)), :logcdf, :base, (:mu, :sigma, :bound)),
        (:(logccdf(LogNormal(mu, sigma), bound)), :logccdf, :base, (:mu, :sigma, :bound)))
        result = lower(ex)
        @test only(result.stats) isa BRM._RKLogNormalStatistic{statistic,side}
        @test Tuple(result.source.args[2:end]) == args
    end
    for ex in (:(mean(x)), :(var(x)),
               :(mean(truncated(LogNormal(mu, sigma); lower=lo, upper=hi))),
               :(mean(truncated(LogNormal(mu, sigma); bounds...))),
               :(logcdf(truncated(LogNormal(mu, sigma); lower=lo), x)))
        @test isempty(lower(ex).stats)
    end
    for ex in (:(mean(LogNormal(mu, sigma))), :(var(LogNormal(mu, sigma))),
               :(logcdf(LogNormal(mu, sigma), bound)), :(logccdf(LogNormal(mu, sigma), bound)))
        @test isempty(lower(ex, OwnRKStatisticNames).stats)
    end
    # Model-scope values have already resolved identities and keep nested
    # prepared expression records rather than the plate's Julia AST.
    prepared = BRM._BRMPreparedExpr
    law = prepared(LogNormal, (:mu, :sigma), (;))
    bounded = prepared(truncated, (law,), (; upper=:hi))
    bindings = Pair{Symbol,Any}[]
    source = BRM._rk_value_expr!(bindings, prepared(var, (bounded,), (;)), Set{Symbol}())
    @test only(last(p) for p in bindings if last(p) isa BRM._RKLogNormalStatistic) isa
        BRM._RKLogNormalStatistic{:var,:upper}
    @test Tuple(source.args[2:end]) == (:mu, :sigma, :hi)
end

function ordinary_moment_model(data, in_plate)
    if in_plate
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 0.5)
            @plate for i in eachindex(y)
                m[i] = mean(LogNormal(mu + offset[i], exp(log_sigma)))
                v[i] = var(LogNormal(mu + offset[i], exp(log_sigma)))
                logmass[i] = logccdf(LogNormal(mu + offset[i], exp(log_sigma)), bound[i])
                logcomplement[i] = logcdf(LogNormal(mu + offset[i], exp(log_sigma)), bound[i])
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.1))
                survived[i] ~ BernoulliLogit(logmass[i] - logcomplement[i])
            end
        end
    else
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 0.5)
            m = mean(LogNormal(mu, exp(log_sigma)))
            v = var(LogNormal(mu, exp(log_sigma)))
            logmass = logccdf(LogNormal(mu, exp(log_sigma)), 0.8)
            logcomplement = logcdf(LogNormal(mu, exp(log_sigma)), 0.8)
            y ~ Normal(m, sqrt(v + 0.1))
            survived ~ BernoulliLogit(logmass - logcomplement)
        end
    end
end

@stestset "RK ordinary LogNormal moments preserve Reverse and source replay" begin
    @test Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    data = (; y=[1.4, 2.1, 0.6], offset=[-0.3, 0.2, 0.4],
        bound=[0.8, 1.1, 0.7], survived=ones(Int, 3))
    original = deepcopy(data)
    for in_plate in (false, true)
        backend, problem = consumer_problem(ordinary_moment_model(data, in_plate))
        names = coordinate_names(backend.model.layout)
        oracle = u -> begin
            mu, ls = u[coordinate(names, :mu)], u[coordinate(names, :log_sigma)]
            value = logpdf(Normal(0, 1), mu) + logpdf(Normal(0, 0.5), ls)
            for i in eachindex(data.y)
                law = LogNormal(mu + (in_plate ? data.offset[i] : 0.0), exp(ls))
                value += logpdf(Normal(mean(law), sqrt(var(law) + 0.1)), data.y[i]) +
                    logccdf(law, in_plate ? data.bound[i] : 0.8)
            end
            value
        end
        for u in (zeros(2), [0.1, -0.2], [-0.2, 0.1])
            check_consumer_point(problem, u, oracle)
            check_consumer_point(problem, u, oracle)
        end
    end
    @test isequal(data, original)
end

function native_moment_reference(mu, sigma, bound, upper)
    setprecision(BigFloat, 512) do
        m, s, b = BigFloat(mu), BigFloat(sigma), BigFloat(bound)
        z = (log(b) - m) / s
        logccdf_normal(t) = t >= 0 ? log(erfc(t / sqrt(big(2))) / 2) :
            log1p(-erfc(-t / sqrt(big(2))) / 2)
        logmass_at(t) = logccdf_normal(upper ? -t : t)
        mass = logmass_at(z)
        logmoment(k) = k*m + k^2*s^2/2 + logmass_at(z-k*s) - mass
        first, second = logmoment(1), logmoment(2)
        (; mean=exp(first), var=exp(2first)*expm1(second-2first),
           logmass=mass, logcomplement=logccdf_normal(upper ? z : -z))
    end
end

function native_moment_plate(data, upper)
    if upper
        @brm data begin
            shift ~ Normal(0, 1)
            log_scale ~ Normal(0, 0.5)
            log_bound ~ Normal(0, 0.3)
            @plate for i in eachindex(y)
                location = baseline[i] + shift
                scale = sigma[i] * exp(log_scale)
                limit = bound[i] * exp(log_bound)
                m[i] = mean(truncated(LogNormal(location, scale); upper=limit))
                v[i] = var(truncated(LogNormal(location, scale); upper=limit))
                logmass[i] = logcdf(LogNormal(location, scale), limit)
                logcomplement[i] = logccdf(LogNormal(location, scale), limit)
                y[i] ~ Normal(m[i], sqrt(v[i] + noise[i]^2))
                survived[i] ~ BernoulliLogit(logmass[i] - logcomplement[i])
            end
        end
    else
        @brm data begin
            shift ~ Normal(0, 1)
            log_scale ~ Normal(0, 0.5)
            log_bound ~ Normal(0, 0.3)
            @plate for i in eachindex(y)
                location = baseline[i] + shift
                scale = sigma[i] * exp(log_scale)
                limit = bound[i] * exp(log_bound)
                m[i] = mean(truncated(LogNormal(location, scale); lower=limit))
                v[i] = var(truncated(LogNormal(location, scale); lower=limit))
                logmass[i] = logccdf(LogNormal(location, scale), limit)
                logcomplement[i] = logcdf(LogNormal(location, scale), limit)
                y[i] ~ Normal(m[i], sqrt(v[i] + noise[i]^2))
                survived[i] ~ BernoulliLogit(logmass[i] - logcomplement[i])
            end
        end
    end
end

@stestset "RK one-sided LogNormal moments retain extreme-tail precision" begin
    cuts = (-40., -10., -3., -2., 0., 1., 2., 3., 10., 15., 40., 100., 1000.)
    scales = (1e-8, 1e-5, 1e-3, 0.2, 1., 3., 5.)
    points = [(z, s) for z in cuts for s in scales]
    data = (; baseline=[-z*s for (z,s) in points],
        sigma=[s for (_,s) in points], bound=ones(length(points)),
        y=ones(length(points)), noise=ones(length(points)), survived=ones(Int,length(points)))
    original = deepcopy(data)
    for upper in (false, true)
        rk = RKBRMI(native_moment_plate(data, upper))
        n = length(coordinate_names(rk.model.layout))
        values = rk_draw_values(rk, zeros(n, 1), (:m, :v, :logmass, :logcomplement))
        actual = map(only, values)
        for i in eachindex(points)
            mu, s, b = data.baseline[i], data.sigma[i], data.bound[i]
            expected = native_moment_reference(mu, s, b, upper)
            law = upper ? truncated(LogNormal(mu,s); upper=b) :
                truncated(LogNormal(mu,s); lower=b)
            # Zero atol makes underflow an explicit Float64 rounding check;
            # it cannot conceal a relative error in a small nonzero variance.
            for (got, reference) in ((actual.m[i], expected.mean),
                                      (actual.v[i], expected.var))
                @test isapprox(got, Float64(reference); rtol=1e-13, atol=0)
            end
            @test isapprox(actual.m[i], mean(law); rtol=1e-13, atol=0)
            @test isapprox(actual.v[i], var(law); rtol=1e-13, atol=0)
            @test isapprox(actual.logmass[i], Float64(expected.logmass); rtol=1e-13, atol=0)
            @test isapprox(actual.logcomplement[i], Float64(expected.logcomplement); rtol=1e-13, atol=0)
        end
        @test isequal(data, original)
    end
end

function native_moment_density(data, names, upper, u)
    shift = u[coordinate(names, :shift)]
    ls = u[coordinate(names, :log_scale)]
    lb = u[coordinate(names, :log_bound)]
    value = logpdf(Normal(0,1), shift) + logpdf(Normal(0,0.5), ls) +
        logpdf(Normal(0,0.3), lb)
    for i in eachindex(data.y)
        ref = native_moment_reference(data.baseline[i] + shift,
            data.sigma[i]*exp(ls), data.bound[i]*exp(lb), upper)
        value += logpdf(Normal(Float64(ref.mean), sqrt(Float64(ref.var)+data.noise[i]^2)),
            data.y[i]) + Float64(ref.logmass)
    end
    value
end

@stestset "RK lower and upper moments preserve full density, Reverse and source replay" begin
    data = (; baseline=[-3., 0.2, 1.7], sigma=[0.2, 0.7, 1.3],
        bound=[1., 0.8, 2.], y=[1.1, 0.7, 1.8], noise=[0.1, 0.2, 0.3],
        survived=ones(Int,3))
    for upper in (false, true)
        backend, problem = consumer_problem(native_moment_plate(data, upper))
        names = coordinate_names(backend.model.layout)
        @test Set(names) == Set((:shift, :log_scale, :log_bound))
        oracle(u) = native_moment_density(data, names, upper, u)
        for u in (zeros(3), [0.15,-0.1,0.08], [-0.2,0.1,-0.15])
            check_consumer_point(problem, u, oracle)
            # Repeated evaluation must retain the same target and inputs.
            check_consumer_point(problem, u, oracle)
        end
    end
end
