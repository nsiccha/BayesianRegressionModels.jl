# Ordinary distribution statistics in a plate, including the separate
# probability of the event X >= lower. No application-specific density.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics, SpecialFunctions, LinearAlgebra

module OwnMomentFunctions
using Distributions: LogNormal
mean(x) = 2x
var(x) = 3x
end

@stestset "distribution statistics preserve callable bindings" begin
    spell(ex, scope=@__MODULE__) = BRM._sb_cell_value(ex, scope)
    # Array statistics and unrelated callables keep their established meaning.
    @test spell(:(mean(x))) == :(mean(x))
    @test spell(:(var(x))) == :(var(x))
    @test spell(:(mean(LogNormal(mu, sigma))), OwnMomentFunctions) ==
        :(mean(LogNormal(mu, sigma)))
    @test spell(:(var(LogNormal(mu, sigma))), OwnMomentFunctions) ==
        :(var(LogNormal(mu, sigma)))
    @test spell(:(Statistics.mean(Distributions.LogNormal(mu, sigma)))) ==
        spell(:(mean(LogNormal(mu, sigma))))
    @test spell(:(mean(LogNormal()))) == spell(:(mean(LogNormal(0.0, 1.0))))
    @test spell(:(var(LogNormal(mu)))) == spell(:(var(LogNormal(mu, 1.0))))
    @test spell(:(mean(truncated(LogNormal(mu, sigma), lo, nothing)))) ==
        spell(:(mean(truncated(LogNormal(mu, sigma); lower=lo))))
    @test spell(:(mean(truncated(LogNormal(mu, sigma); bounds...)))) ==
        :(mean(truncated(LogNormal(mu, sigma); bounds...)))
    @test_throws "lower truncation only" spell(
        :(mean(truncated(LogNormal(mu, sigma); lower=lo, upper=hi))))
    @test_throws "upper=nothing" spell(
        :(mean(truncated(LogNormal(mu, sigma); lower=lo, upper=Inf))))
end

# Independent tilted-normal identity evaluated at 256-bit precision. The
# production implementation must work at Float64 precision, including tails.
function bounded_reference(mu, sigma, lower)
    setprecision(BigFloat, 256) do
        m, s, l = BigFloat(mu), BigFloat(sigma), BigFloat(lower)
        a = (log(l) - m) / s
        logtail(z) = log(erfc(z / sqrt(big(2))) / 2)
        mass = logtail(a)
        logmoment(k) = k * m + k^2 * s^2 / 2 + logtail(a - k * s) - mass
        first, second = logmoment(1), logmoment(2)
        (Float64(exp(first)), Float64(exp(2first) * expm1(second - 2first)),
            Float64(mass))
    end
end

function scalar_bounded_model(data)
    @brm data begin
        mu ~ Normal(0, 1)
        log_sigma ~ Normal(0, 0.5)
        m = mean(truncated(LogNormal(mu, exp(log_sigma)); lower=0.8))
        v = var(truncated(LogNormal(mu, exp(log_sigma)); lower=0.8))
        logmass = logccdf(LogNormal(mu, exp(log_sigma)), 0.8)
        logcomplement = logcdf(LogNormal(mu, exp(log_sigma)), 0.8)
        unbounded = mean(LogNormal(mu, exp(log_sigma)))
        y ~ Normal(m + 0.01 * unbounded, sqrt(v + 0.1))
        survived ~ BernoulliLogit(logmass - logcomplement)
    end
end

@stestset "model-scope distribution statistics share plate semantics" begin
    data = (; y=[1.4, 2.1], survived=ones(Int, 2))
    brmi = scalar_bounded_model(data)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    oracle = u -> begin
        mu, ls = u[coordinate(names, :mu)], u[coordinate(names, :log_sigma)]
        m, v, mass = bounded_reference(mu, exp(ls), 0.8)
        unbounded = exp(mu + exp(2ls) / 2)
        logpdf(Normal(0, 1), mu) + logpdf(Normal(0, 0.5), ls) +
            sum(logpdf.(Normal(m + 0.01 * unbounded, sqrt(v + 0.1)), data.y)) +
            length(data.survived) * mass
    end
    stan = consumer_stan(brmi, "scalar-bounded-lognormal-moments")
    for u in (zeros(2), [0.1, -0.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan,
            [:mu => "mu", :log_sigma => "log_sigma"], backend, u)
    end
end

const MOMENT_DATA = (; baseline=[-3.0, 0.0, 3.0, -40.0],
    scale=[0.2, 0.2, 0.8, 1.0], bound=[1.0, 0.7, 2.0, 1.2],
    y=[1.02, 1.05, 20.0, 1.24], noise=[0.03, 0.05, 0.5, 0.04],
    survived=ones(Int, 4))

function bounded_plate(data)
    @brm data begin
        shift ~ Normal(0, 1)
        log_scale ~ Normal(0, 0.5)
        log_bound ~ Normal(0, 0.3)
        @plate for i in eachindex(y)
            mu = baseline[i] + shift
            sigma = scale[i] * exp(log_scale)
            lo = bound[i] * exp(log_bound)
            m[i] = mean(truncated(LogNormal(mu, sigma); lower=lo))
            v[i] = var(truncated(LogNormal(mu, sigma); lower=lo))
            logmass[i] = logccdf(LogNormal(mu, sigma), lo)
            logcomplement[i] = logcdf(LogNormal(mu, sigma), lo)
            y[i] ~ Normal(m[i], sqrt(v[i] + noise[i]^2))
            # The observed event contributes logmass, independently of the
            # normalized distribution's moments, even when exp(logmass) is 0
            # or 1. Compute both log tails directly to retain the complement.
            survived[i] ~ BernoulliLogit(logmass[i] - logcomplement[i])
        end
    end
end

function moment_oracle(data, names)
    u -> begin
        shift = u[coordinate(names, :shift)]
        log_scale = u[coordinate(names, :log_scale)]
        log_bound = u[coordinate(names, :log_bound)]
        value = logpdf(Normal(0, 1), shift) + logpdf(Normal(0, 0.5), log_scale) +
            logpdf(Normal(0, 0.3), log_bound)
        for i in eachindex(data.y)
            m, v, mass = bounded_reference(data.baseline[i] + shift,
                data.scale[i] * exp(log_scale), data.bound[i] * exp(log_bound))
            value += logpdf(Normal(m, sqrt(v + data.noise[i]^2)), data.y[i]) + mass
        end
        value
    end
end

@stestset "bounded moments and separate support mass on Stan and RK" begin
    brmi = bounded_plate(MOMENT_DATA)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    oracle = moment_oracle(MOMENT_DATA, names)
    stan = consumer_stan(brmi, "bounded-lognormal-moments")
    mapping = [:shift => "shift", :log_scale => "log_scale", :log_bound => "log_bound"]
    near_full = zeros(3)
    near_full[coordinate(names, :shift)] = 40.0
    for u in (zeros(3), [0.15, -0.1, 0.08], [-0.2, 0.1, -0.15], near_full)
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
end

function grouped_bounded_plate(data)
    @brm data begin
        mu ~ 1 + (1 | subject)
        log_scale ~ Normal(0, 0.5)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(mu)
            sigma = scale[i] * exp(log_scale)
            m = mean(truncated(LogNormal(mu[i], sigma); lower=bound[i]))
            v = var(truncated(LogNormal(mu[i], sigma); lower=bound[i]))
            logmass = logccdf(LogNormal(mu[i], sigma), bound[i])
            logcomplement = logcdf(LogNormal(mu[i], sigma), bound[i])
            y[obs[i]] ~ Normal(m, sqrt(v + noise[i]^2))
            survived[obs[i]] ~ BernoulliLogit(logmass - logcomplement)
        end
    end
end

@stestset "bounded moments and event evidence retain NLME unit attribution" begin
    # One cell per subject, with repeated, interleaved observations joined to
    # that cell. Both the conditional response and the event density repeat.
    data = (; subject=["b", "a", "c"], obs_subject=["b", "a", "b", "c"],
        scale=[0.2, 0.8, 1.0], bound=[1.0, 0.7, 1.2], noise=[0.03, 0.05, 0.04],
        y=[1.02, 1.05, 1.04, 1.24], survived=ones(Int, 4))
    rk = RKBRMI(grouped_bounded_plate(data))
    ad = AutoEnzyme(; mode=Enzyme.Reverse)
    model = brm_nlme_model(rk; ad_backend=ad, subject_group=:subject)
    view = model.view
    @test all(!=(:unclaimed), view.role)
    expected = [findfirst(==(s), string.(view.levels)) for s in data.subject]
    for response in (:y, :survived)
        @test only(last(a) for a in model.attribution if first(a) === response) == expected
    end
    theta = fill(0.1, length(model.theta))
    sigma = fill(-0.1, length(model.sigma))
    n_effects = length(view.block_margins)
    omega = Matrix{Float64}(I, n_effects, n_effects)
    full = rk_logdensity_problem(rk; ad_backend=ad)
    for (i, block) in enumerate(view.subject_coordinates)
        unit = brm_nlme_unit_problem(model, i, theta, sigma, omega)
        z1, z2 = fill(-0.25, length(block)), fill(0.4, length(block))
        function full_point(z)
            u = copy(model.unit)
            u[model.theta] = theta
            u[model.sigma] = sigma
            for other in view.subject_coordinates
                u[other] .= 0.15
            end
            u[block] = z
            u
        end
        @test LogDensityProblems.logdensity(unit, z1) - LogDensityProblems.logdensity(unit, z2) ≈
            LogDensityProblems.logdensity(full, full_point(z1)) -
            LogDensityProblems.logdensity(full, full_point(z2)) atol=1e-9 rtol=1e-10
        _, gradient = LogDensityProblems.logdensity_and_gradient(unit, z1)
        _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, full_point(z1))
        @test gradient ≈ full_gradient[block] atol=1e-9 rtol=1e-10
    end
end
