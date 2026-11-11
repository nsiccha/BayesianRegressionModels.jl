#
# The law is a filter's log likelihood run along a station's hours: an
# ordinary Julia function the consumer writes, bound as a leaf of the
# whole-event observation provider (`brm_distribution_type` +
# `_rk_observation_source!`, brm-use reactivekernels.md "Whole-vector and
# matrix events"). The time-local states are integrated out by the filter, so
# each station's SVI factor holds its statics only.
#
# Toy, public: a stationary AR(1) state per station, x_t = a x_{t-1} +
# sqrt(1 - a^2) e_t with x_1 ~ N(m0, P0), read as y_t = offset + gain x_t +
# noise n_t. The exact Kalman filter's log likelihood is checked against the
# dense MvNormal of the readings, the RK target's native Reverse gradient
# against central differences, and the SVI fit against long NUTS runs of the
# same target.
#
# RUN: julia --project=test test/nlme_svi_series_law.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics, WarmupHMC
import BayesianRegressionModels: _rk_observation_source!, brm_distribution_type

# ---- the consumer's filter: ordinary Julia, no BRM or RK types --------------
function series_kf_loglik(y, offset, gain, noise, a, m0, P0)
    m, P = m0, P0
    ll = zero(offset * gain * noise * a)
    q = 1 - a * a
    for t in eachindex(y)
        if t > firstindex(y)
            m = a * m
            P = a * a * P + q
        end
        S = gain * gain * P + noise * noise
        r = y[t] - offset - gain * m
        ll += -0.5 * (log(2pi) + log(S) + r * r / S)
        K = P * gain / S
        m += K * r
        P -= K * gain * P
    end
    ll
end

# The filter's state after the last reading (what the next window carries).
function series_kf_end(y, offset, gain, noise, a, m0, P0)
    m, P = m0, P0
    for t in eachindex(y)
        if t > firstindex(y)
            m = a * m
            P = a * a * P + (1 - a * a)
        end
        S = gain * gain * P + noise * noise
        K = P * gain / S
        m += K * (y[t] - offset - gain * m)
        P -= K * gain * P
    end
    (m, P)
end

# The family token and its whole-event law.
function series_kf end
brm_distribution_type(::typeof(series_kf)) = ContinuousMultivariateDistribution
function _rk_observation_source!(definitions, bindings, entry, ::typeof(series_kf))
    filter = Symbol(entry, :_filter)
    push!(bindings, filter => series_kf_loglik)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, offset, gain, noise, a, m0, P0) = begin
        ll = $filter(y, offset, gain, noise, a, m0, P0)
        return ll
    end))
    :done
end

function series_dense_loglik(y, offset, gain, noise, a)
    n = length(y)
    C = [gain^2 * a^abs(i - j) + (i == j) * noise^2 for i in 1:n, j in 1:n]
    logpdf(MvNormal(fill(offset, n), Symmetric(C)), y)
end

function series_data(rng, S; T=60, a=0.8, mu=(0.5, log(1.2), log(0.4)), sd=(0.5, 0.3, 0.3))
    eff = [mu[k] .+ sd[k] .* randn(rng, S) for k in 1:3]
    y = map(1:S) do s
        n = T + rand(rng, -10:10)
        x = zeros(n)
        x[1] = randn(rng)
        for t in 2:n
            x[t] = a * x[t-1] + sqrt(1 - a^2) * randn(rng)
        end
        eff[1][s] .+ exp(eff[2][s]) .* x .+ exp(eff[3][s]) .* randn(rng, n)
    end
    (; station=["st$s" for s in 1:S], y, m0=zeros(S), P0=ones(S))
end

series_model(data) = @brm data begin
    a ~ Uniform(0, 1)
    level ~ 1 + (1 | st | station)
    log_gain ~ 1 + (1 | st | station)
    log_noise ~ 1 + (1 | st | station)
    @plate for s in eachindex(level)
        y[s] ~ series_kf(level[s], exp(log_gain[s]), exp(log_noise[s]), a, m0[s], P0[s])
    end
end

# One station's statics under a population law held at known values: known
# means through `offset(...)`, known SDs and correlation through `=`.
const SERIES_R = [1.0 0.2 0.0; 0.2 1.0 0.1; 0.0 0.1 1.0]
known_series_model(data) = @brm data begin
    a ~ Uniform(0, 1)
    level ~ 0 + offset(mu_level) + (1 | st | station)
    log_gain ~ 0 + offset(mu_gain) + (1 | st | station)
    log_noise ~ 0 + offset(mu_noise) + (1 | st | station)
    sd(level, st) = 0.5
    sd(log_gain, st) = 0.3
    sd(log_noise, st) = 0.3
    cor(:, st) = [1.0 0.2 0.0; 0.2 1.0 0.1; 0.0 0.1 1.0]
    @plate for s in eachindex(level)
        y[s] ~ series_kf(level[s], exp(log_gain[s]), exp(log_noise[s]), a, m0[s], P0[s])
    end
end

# Four NUTS chains of the full RK target, one packed draw per column.
function series_nuts(rk, m; seed, n_draws=1000, chains=4)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    reduce(hcat, map(1:chains) do c
        run = adaptive_warmup_mcmc(Xoshiro(seed + c), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws)
        convert(Matrix{Float64}, run.posterior_position)
    end)
end

function series_moments(draws, reference, idx)
    sd = vec(std(reference[idx, :]; dims=2))
    (; mean_error=abs.(vec(mean(draws[idx, :]; dims=2)) .-
        vec(mean(reference[idx, :]; dims=2))) ./ sd,
       sd_ratio=vec(std(draws[idx, :]; dims=2)) ./ sd)
end

# Each station's statics and the shared persistence at one packed point.
function series_statics(rk, u)
    U = reshape(collect(u), :, 1)
    v = rk_draw_values(rk, U, (:level, :log_gain, :log_noise))
    (; level=only(v.level), log_gain=only(v.log_gain), log_noise=only(v.log_noise),
       a=only(rk_restore_draws(rk, U).a))
end

function series_fd_check(problem, u; h=1e-6)
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
    fd = map(eachindex(u)) do j
        up, um = copy(u), copy(u)
        up[j] += h
        um[j] -= h
        (LogDensityProblems.logdensity(problem, up) -
            LogDensityProblems.logdensity(problem, um)) / 2h
    end
    value, gradient, fd
end

@stestset "a filter likelihood is one whole-series observation per station" begin
    rng = Xoshiro(7)
    y = randn(rng, 25)
    for (o, g, n, a) in ((0.3, 1.1, 0.4, 0.7), (-1.0, 0.5, 1.3, 0.2), (0.0, 2.0, 0.1, 0.95))
        @test series_kf_loglik(y, o, g, n, a, 0.0, 1.0) ≈ series_dense_loglik(y, o, g, n, a) rtol = 1e-10
    end
    data = series_data(Xoshiro(1), 6; T=20)
    rk = RKBRMI(series_model(data))
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    pointwise = prepare_query(rk.model, rk_bound_plan(rk), :pointwise)
    for seed in 1:3
        u = 0.3 .* randn(Xoshiro(seed), LogDensityProblems.dimension(problem))
        value, gradient, fd = series_fd_check(problem, u)
        @test all(isfinite, gradient)
        @test gradient ≈ fd rtol = 1e-5 atol = 1e-5
        # One pointwise term per station: the filter over its whole series.
        terms = Base.invokelatest(pointwise, u).y
        @test length(terms) == length(data.y)
        st = series_statics(rk, u)
        oracle = [series_dense_loglik(data.y[s], st.level[s], exp(st.log_gain[s]),
            exp(st.log_noise[s]), st.a) for s in eachindex(data.y)]
        @test terms ≈ oracle rtol = 1e-9
    end
end

@stestset "the stations are SVI units whose factors hold their statics" begin
    data = series_data(Xoshiro(1), 30)
    rk = RKBRMI(series_model(data))
    names = coordinate_names(rk.model.layout)
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    view = m.view
    @test length(view.levels) == 30
    @test all(c -> length(c) == 3, view.subject_coordinates)
    fit = brm_svi_fit(m; batch_size=8, rng=Xoshiro(11), ndraws=4000, threads=1)
    @test fit.diagnostics.converged
    @test all(f -> f.dimension == 3, fit.diagnostics.unit_families)
    ref = series_nuts(rk, m; seed=100)
    globals = [i for (i, n) in enumerate(names) if !occursin(".z", string(n))]
    units = [i for (i, n) in enumerate(names) if occursin(".z", string(n))]
    g, u = series_moments(fit.draws, ref, globals), series_moments(fit.draws, ref, units)
    println("series SVI globals ", names[globals], ": mean error / NUTS sd ",
        round.(g.mean_error; digits=2), ", sd ratio ", round.(g.sd_ratio; digits=2))
    println("series SVI units: mean error / NUTS sd median ",
        round(median(u.mean_error); digits=2), " max ", round(maximum(u.mean_error); digits=2),
        ", sd ratio median ", round(median(u.sd_ratio); digits=2))
    # What this file checks is the law: the units' statics. The globals carry
    # minibatch SVI's own hyperparameter error, which nlme_svi.jl measures;
    # here the three block SDs from 30 stations are weakly identified, and on
    # seed 11 one of them ends more than one NUTS sd off (1.18 at BRM 0.2.247,
    # 1.11 at 0.2.249).
    @test maximum(g.mean_error) < 1.5
    @test median(u.mean_error) < 0.25
    @test 0.7 < median(u.sd_ratio) < 1.2
end

@stestset "a station under a known population law" begin
    full = series_data(Xoshiro(1), 30)
    one = (; station=["st1"], y=[full.y[1]], m0=[0.0], P0=[1.0],
        mu_level=[0.5], mu_gain=[log(1.2)], mu_noise=[log(0.4)])
    rk = RKBRMI(known_series_model(one))
    names = coordinate_names(rk.model.layout)
    # No scale or correlation coordinate: the block's values are constants.
    println("known-law station coordinates: ", names)
    @test length(names) == 4
    @test count(n -> occursin(".z", string(n)), names) == 3
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    view = m.view
    @test length(view.levels) == 1
    family = only(view.families)
    @test family.fixed == Matrix{Float64}(I, 3, 3)
    @test isempty(family.scales) && isempty(family.factor)
    @test count(==(:subject_effect), view.role) == 3
    @test isempty(view.mu_references)
    # The unit's density: its standardized draws, mapped by the known factor,
    # are the statics; the prior is the population law.
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    L = cholesky(Symmetric(SERIES_R)).L
    D = Diagonal([0.5, 0.3, 0.3])
    mu = [0.5, log(1.2), log(0.4)]
    ia = findfirst(==(:a), names)
    iz = [i for (i, n) in enumerate(names) if occursin(".z", string(n))]
    for seed in 1:3
        u = 0.4 .* randn(Xoshiro(seed), length(names))
        z, ua = u[iz], u[ia]
        a = 1 / (1 + exp(-ua))
        θ = mu .+ D * L * z
        oracle = sum(logpdf.(Normal(), z)) + log(a) + log1p(-a) +
            series_dense_loglik(one.y[1], θ[1], exp(θ[2]), exp(θ[3]), a)
        value, gradient, fd = series_fd_check(problem, u)
        @test value ≈ oracle rtol = 1e-9
        @test gradient ≈ fd rtol = 1e-5 atol = 1e-5
    end
    fit = brm_svi_fit(m; batch_size=1, rng=Xoshiro(3), ndraws=4000, threads=1)
    @test fit.diagnostics.converged
    ref = series_nuts(rk, m; seed=200)
    mo = series_moments(fit.draws, ref, eachindex(names))
    println("known-law station SVI ", names, ": mean error / NUTS sd ",
        round.(mo.mean_error; digits=2), ", sd ratio ", round.(mo.sd_ratio; digits=2))
    @test maximum(mo.mean_error) < 0.3
    @test all(r -> 0.6 < r < 1.3, mo.sd_ratio)
    # A block that mixes known and sampled values holds the known margin's SD and
    # estimates the rest (test/nlme_partly_known_blocks.jl fits one).
    mixed = @brm one begin
        a ~ Uniform(0, 1)
        level ~ 0 + offset(mu_level) + (1 | st | station)
        log_gain ~ 0 + offset(mu_gain) + (1 | st | station)
        sd(level, st) = 0.5
        @plate for s in eachindex(level)
            y[s] ~ series_kf(level[s], exp(log_gain[s]), 0.4, a, m0[s], P0[s])
        end
    end
    held = only(brm_nlme_view(RKBRMI(mixed)).families)
    @test held.fixed === nothing
    @test isequal(BRM._brm_nlme_held_sd(held.record), [1.0, nothing])
end

@stestset "chained windows carry each station's filter state as data" begin
    data = series_data(Xoshiro(5), 12; T=80)
    cut = [div(length(y), 2) for y in data.y]
    first_window = (; data..., y=[y[1:c] for (y, c) in zip(data.y, cut)])
    builder = @brm begin
        a ~ Uniform(0, 1)
        level ~ 1 + (1 | st | station)
        log_gain ~ 1 + (1 | st | station)
        log_noise ~ 1 + (1 | st | station)
        @plate for s in eachindex(level)
            y[s] ~ series_kf(level[s], exp(log_gain[s]), exp(log_noise[s]), a, m0[s], P0[s])
        end
    end
    fit1 = brm_svi_fit(builder(first_window); batch_size=4, rng=Xoshiro(1), ndraws=1000,
        threads=1)
    @test fit1.diagnostics.converged
    # The next window starts each station's filter where the first left it,
    # under the first fit's mean statics: the lag's cut, as data.
    st = series_statics(RKBRMI(builder(first_window)), vec(mean(fit1.draws; dims=2)))
    ends = [series_kf_end(first_window.y[s], st.level[s], exp(st.log_gain[s]),
        exp(st.log_noise[s]), st.a, 0.0, 1.0) for s in eachindex(data.y)]
    second_window = (; data..., y=[y[(c + 1):end] for (y, c) in zip(data.y, cut)],
        m0=first.(ends), P0=last.(ends))
    fit2 = brm_svi_fit(brm_svi_checkpoint(fit1), builder(second_window); batch_size=4,
        rng=Xoshiro(2), ndraws=1000, threads=1)
    @test fit2.diagnostics.converged
    @test size(fit2.draws, 1) == size(fit1.draws, 1)
    @test all(isfinite, fit2.draws)
end
