# Formula-level typed observation weights on the RK reader (value) route: a
# response whose law reads an assigned value, `p = reader(...)` or
# `loc = 2 * mu`, observed as `weighted(family, fweights(n))` or
# `weighted(family, weights(w))`. Each row's log density is multiplied by its
# weight, exactly as on the regression route and in Stan.
#
#   julia --startup-file=no --project=test test/rk_value_weighted_observations.jl [testset substring ...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using StatsFuns: normcdf

module ValueWeights
using BayesianRegressionModels, StanBlocks
using StatsFuns: normcdf

# A lognormal delay's probability of falling in [l, u]: a whole-array Julia
# reader with a same-named Stan function, as an interval-censored delay model
# writes its row probabilities.
StanBlocks.@deffun begin
    interval_prob(l::vector[n], u::vector[n], mu::vector[n], sigma::vector[n])::vector[n] = begin
        p::vector[n]
        for i in 1:n
            p[i] = Phi((log(u[i]) - mu[i]) / sigma[i]) - Phi((log(l[i]) - mu[i]) / sigma[i])
        end
        return p
    end
    interval_prob(l::vector[n], u::vector[n], mu::real, sigma::real)::vector[n] = begin
        p::vector[n]
        for i in 1:n
            p[i] = Phi((log(u[i]) - mu) / sigma) - Phi((log(l[i]) - mu) / sigma)
        end
        return p
    end
end
interval_prob(l::AbstractVector, u::AbstractVector, mu, sigma) =
    normcdf.((log.(u) .- mu) ./ sigma) .- normcdf.((log.(l) .- mu) ./ sigma)

# The reported spelling and the Binomial(n, p) equivalent with `count = n`:
# log Binomial(n | n, p) = n log p = n * log Bernoulli(1 | p).
delay(data, form) = form === :weighted ? (@brm data begin
        mu ~ 1
        log(sigma) ~ 1
        effect(mu, Intercept) ~ Normal(1.0, 0.5)
        effect(sigma, Intercept) ~ Normal(-0.5, 0.3)
        p = interval_prob(lwr, upr, mu, sigma)
        observed ~ weighted(Bernoulli(p), fweights(n))
    end) : (@brm data begin
        mu ~ 1
        log(sigma) ~ 1
        effect(mu, Intercept) ~ Normal(1.0, 0.5)
        effect(sigma, Intercept) ~ Normal(-0.5, 0.3)
        p = interval_prob(lwr, upr, mu, sigma)
        count ~ Binomial(n, p)
    end)

scalar(data, kind) = kind === :power ? (@brm data begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        loc = a .+ b .* x
        sigma = exp(log_s)
        y ~ weighted(Normal(loc, sigma), weights(w))
    end) : kind === :frequency ? (@brm data begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        loc = a .+ b .* x
        sigma = exp(log_s)
        y ~ weighted(Normal(loc, sigma), fweights(n))
    end) : kind === :censored ? (@brm data begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        loc = a .+ b .* x
        sigma = exp(log_s)
        y ~ weighted(censored(Normal(loc, sigma); lower=lo), fweights(n))
    end) : (@brm data begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        loc = a .+ b .* x
        sigma = exp(log_s)
        y ~ weighted(Normal(loc, sigma), aweights(k))
    end)

# A formula predictor read by an assignment selects the reader route too.
predictor(data) = @brm data begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    loc = 2 * mu
    y ~ weighted(Normal(loc, sigma), fweights(n))
end

# Per-row mixture weights from a linked predictor select the reader route.
mixture(data, form) = form === :weighted ? (@brm data begin
        mu ~ 1
        logit(q) ~ 1 + x
        y ~ weighted(MixtureModel([Normal(mu, 1.0), Normal(0.0, 3.0)], [1 - q, q]), fweights(n))
    end) : (@brm data begin
        mu ~ 1
        logit(q) ~ 1 + x
        y ~ MixtureModel([Normal(mu, 1.0), Normal(0.0, 3.0)], [1 - q, q])
    end)

missing_weighted(data) = @brm data begin
    a ~ Normal(0, 1)
    loc = a .+ x
    mi(y) ~ weighted(Normal(loc, 1.0), fweights(n))
end
end

const DELAY = (; lwr=[0.5, 1.0, 2.0, 3.5, 0.2], upr=[1.0, 2.0, 3.0, 5.0, 0.6],
    n=[3, 1, 4, 2, 7], observed=ones(Int, 5), count=[3, 1, 4, 2, 7])

# The first two rows sit at the censoring limit `lo`.
const ROWS = (; x=[-1.0, -0.2, 0.4, 1.1, 2.0], y=[0.15, 0.15, 0.2, 1.4, 1.7],
    w=[1.0, 0.0, 2.5, 0.5, 1.0], n=[2, 1, 0, 3, 1], k=[1.0, 2.0, 3.0, 4.0, 5.0],
    lo=fill(0.15, 5))

points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

@stestset "the reported spelling: a reader's probability under fweights" begin
    weighted_backend, weighted_problem = consumer_problem(ValueWeights.delay(DELAY, :weighted))
    binomial_backend, binomial_problem = consumer_problem(ValueWeights.delay(DELAY, :binomial))
    names = coordinate_names(weighted_backend.model.layout)
    @test names == coordinate_names(binomial_backend.model.layout)
    source = string(rkppl_model(weighted_backend))
    @test occursin("weighted.(Bernoulli.(", source)
    imu, isg = coordinate(names, :mu_Intercept), coordinate(names, :sigma_Intercept)
    oracle = u -> begin
        mu, sigma = u[imu], exp(u[isg])
        p = ValueWeights.interval_prob(DELAY.lwr, DELAY.upr, mu, sigma)
        logpdf(Normal(1.0, 0.5), u[imu]) + logpdf(Normal(-0.5, 0.3), u[isg]) +
            sum(DELAY.n .* log.(p))
    end
    for u in points(length(names))
        value, gradient = check_consumer_point(weighted_problem, u, oracle)
        workaround = LogDensityProblems.logdensity_and_gradient(binomial_problem, u)
        @test value ≈ workaround[1] atol=1e-12
        @test gradient ≈ workaround[2] atol=1e-12
    end
    # The same BRMI on SBBRMI emits Stan's `weighted` and has the same density.
    stan = consumer_stan(ValueWeights.delay(DELAY, :weighted), "rk-value-weighted-delay")
    stan_names = BridgeStan.param_unc_names(stan.model)
    mapping = [names[imu] => only(filter(s -> occursin("mu", s), stan_names)),
        names[isg] => only(filter(s -> occursin("sigma", s), stan_names))]
    for u in points(length(names))
        check_consumer_stan(weighted_problem, stan, mapping, weighted_backend, u)
    end
end

@stestset "power, frequency and bounded weights on the reader route" begin
    for kind in (:power, :frequency, :censored)
        backend, problem = consumer_problem(ValueWeights.scalar(ROWS, kind))
        names = coordinate_names(backend.model.layout)
        ia, ib, is = coordinate(names, :a), coordinate(names, :b), coordinate(names, :log_s)
        oracle = u -> begin
            a, b, s = u[ia], u[ib], exp(u[is])
            loc = a .+ b .* ROWS.x
            rows = if kind === :censored
                [y <= l ? logcdf(Normal(m, s), l) : logpdf(Normal(m, s), y)
                    for (m, y, l) in zip(loc, ROWS.y, ROWS.lo)]
            else
                logpdf.(Normal.(loc, s), ROWS.y)
            end
            weight = kind === :power ? ROWS.w : ROWS.n
            logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
                logpdf(Normal(0, 0.5), u[is]) + sum(weight .* rows)
        end
        stan = consumer_stan(ValueWeights.scalar(ROWS, kind), "rk-value-weighted-$kind")
        mapping = [:a => "a", :b => "b", :log_s => "log_s"]
        for u in points(length(names))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
    end
end

@stestset "a predictor read by an assignment, weighted" begin
    backend, problem = consumer_problem(ValueWeights.predictor(ROWS))
    names = coordinate_names(backend.model.layout)
    @test length(names) == 3
    isg = coordinate(names, :sigma)
    i1, i2 = coordinate(names, Symbol("pop_mu.beta_pop.1")),
        coordinate(names, Symbol("pop_mu.beta_pop.2"))
    # Default priors are read off the backend, not restated: compare against
    # the unweighted twin plus the weighted rows' extra (n - 1) copies.
    twin = RKBRMI(@brm ROWS begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        loc = 2 * mu
        y ~ Normal(loc, sigma)
    end)
    @test coordinate_names(twin.model.layout) == names
    twin_problem = rk_logdensity_problem(twin;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in points(length(names))
        mu = u[i1] .+ u[i2] .* ROWS.x
        sigma = exp(u[isg])
        rows = logpdf.(Normal.(2 .* mu, sigma), ROWS.y)
        @test LogDensityProblems.logdensity(problem, u) ≈
            LogDensityProblems.logdensity(twin_problem, u) + sum((ROWS.n .- 1) .* rows) atol=1e-11
    end
end

@stestset "a mixture with per-row weights, frequency-weighted" begin
    backend, problem = consumer_problem(ValueWeights.mixture(ROWS, :weighted))
    _, plain = consumer_problem(ValueWeights.mixture(ROWS, :plain))
    names = coordinate_names(backend.model.layout)
    @test occursin("weighted.(MixtureModel.(", string(rkppl_model(backend)))
    im = coordinate(names, :mu_Intercept)
    i1 = coordinate(names, Symbol("pop_logit_q.beta_pop.1"))
    i2 = coordinate(names, Symbol("pop_logit_q.beta_pop.2"))
    # The weighted density is the unweighted one plus (n - 1) more copies of
    # each row's independent mixture log density.
    extra = u -> begin
        w = LogExpFunctions.logistic.(u[i1] .+ u[i2] .* ROWS.x)
        rows = [logpdf(MixtureModel([Normal(u[im], 1.0), Normal(0.0, 3.0)], [1 - wi, wi]), y)
            for (wi, y) in zip(w, ROWS.y)]
        sum((ROWS.n .- 1) .* rows)
    end
    for u in points(length(names))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        value0, gradient0 = LogDensityProblems.logdensity_and_gradient(plain, u)
        @test value ≈ value0 + extra(u) atol=1e-11
        step = 1e-5
        extra_gradient = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += step; minus[j] -= step
            (extra(plus) - extra(minus)) / (2step)
        end
        @test gradient ≈ gradient0 .+ extra_gradient atol=2e-8 rtol=2e-8
    end
end

@stestset "a new design rebinds and checks the weights" begin
    rk = RKBRMI(ValueWeights.scalar(ROWS, :frequency))
    design = merge(ROWS, (; n=[1, 4, 2, 0, 3]))
    replay = reprocess(rk, design)
    @test rk_bound_data(replay).n == design.n
    fresh = RKBRMI(ValueWeights.scalar(design, :frequency))
    replayed, rebuilt = map(b -> rk_logdensity_problem(b;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), (replay, fresh))
    for u in points(3)
        @test LogDensityProblems.logdensity(replayed, u) ≈
            LogDensityProblems.logdensity(rebuilt, u) atol=1e-12
    end
    # The first design of a kind replays in full; later ones bind checked raw
    # columns without planning. A fractional count would bind as a power
    # weight, so only BRM's check refuses it.
    binder = rk_design_binder(rk)
    counts = merge(design, (; n=Float64.(design.n)))
    @test binder(counts).n == counts.n
    @test binder(merge(counts, (; n=fill(2.0, 5)))).n == fill(2.0, 5)
    @test_throws "nonnegative integer-valued counts" binder(
        merge(counts, (; n=[2.0, 1.5, 2.0, 2.0, 2.0])))
    @test_throws "nonnegative integer-valued counts" binder(
        merge(design, (; n=[2, -1, 2, 2, 2])))
    @test_throws "nonnegative integer-valued counts" reprocess(rk,
        merge(design, (; n=[2.0, 1.5, 2.0, 2.0, 2.0])))
end

@stestset "weights the reader route cannot align or lower are refused" begin
    @test_throws "weights are analytic; RKBRMI admits frequency (`fweights`) and power (`weights`) weights" RKBRMI(
        ValueWeights.scalar(ROWS, :analytic))
    missing_rows = merge(ROWS, (; y=[0.15, missing, 0.2, 1.4, missing]))
    @test_throws "combines `mi()` with `weighted(...)`" RKBRMI(
        ValueWeights.missing_weighted(missing_rows))
    # Weight values are checked as on every route.
    @test_throws "nonnegative integer-valued counts" RKBRMI(
        ValueWeights.scalar(merge(ROWS, (; n=[2, 1, -1, 3, 1])), :frequency))
end
