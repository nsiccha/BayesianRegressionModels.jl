# Fitted grouped AR1Normal replay without response values (StanBlocks 0.2.68+).
# Run: julia --startup-file=no --project=test test/ar1_forecast.jl
using Test, BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using Statistics: mean, var

const BRM = BayesianRegressionModels
include("testset_filter.jl")

const AR1_FIT = (;
    group=["b", "a", "c", "b", "a", "c", "b", "a", "c"],
    time=[3, 2, 1, 1, 1, 4, 6, 5, 7],
    x=[0.3, -0.2, 0.5, 0.1, -0.4, 0.0, 0.8, -0.6, 0.2],
    y=[0.41, -0.12, 0.66, 0.05, -0.31, 0.27, 0.92, -0.48, 0.18])
# Different, unequal group lengths, a singleton, gaps, and interleaved rows.
const AR1_FUTURE = (;
    group=["b", "a", "c", "b", "a", "b", "a", "b"],
    time=[12, 11, 9, 8, 8, 9, 14, 15],
    x=[-0.1, 0.3, -0.2, 0.6, 0.1, -0.5, 0.4, 0.2])
const AR1_SHARED = @brm begin
    mu ~ 1 + x + (1 | group)
    sigma ~ Exponential(1)
    phi ~ Uniform(-1, 1)
    @plate for p in groups(group; order=time)
        y[p] ~ AR1Normal(mu[p], sigma, phi; time=time[p])
    end
end
const AR1_VARYING = @brm begin
    mu ~ 1 + x + (1 | group)
    log(sigma) ~ 1 + x
    atanh(phi) ~ 1 + x
    @plate for p in groups(group; order=time)
        y[p] ~ AR1Normal(mu[p], sigma[p], phi[p]; time=time[p])
    end
end
const AR1_STEPPED = @brm begin
    mu ~ 1 + x + (1 | group)
    sigma ~ Exponential(1)
    phi ~ Uniform(-1, 1)
    @plate for p in groups(group; order=time)
        y[p] ~ AR1Normal(mu[p], sigma, phi)
    end
end

ar1_without_y(data) = (; group=data.group, time=data.time, x=data.x)
ar1_columns(model, q) = Dict(zip(
    BridgeStan.param_names(model; include_tp=true, include_gq=false),
    BridgeStan.param_constrain(model, q; include_tp=true, include_gq=false)))

function ar1_standardized_innovations(draw, mu, sigma, phi, rows, times; stepped)
    z = similar(draw)
    position = 0
    for group in rows
        firstrow = first(group)
        position += 1
        z[position] = (draw[position] - mu[firstrow]) *
            sqrt(1 - phi[firstrow]^2) / sigma[firstrow]
        for (a, b) in zip(group[1:end-1], group[2:end])
            position += 1
            gap = stepped ? 1 : times[b] - times[a]
            persistence = phi[b]^gap
            scale = sigma[b] * sqrt((1 - persistence^2) / (1 - phi[b]^2))
            z[position] = (draw[position] - mu[b] -
                persistence * (draw[position-1] - mu[a])) / scale
        end
    end
    z
end

for (label, builder, varying, stepped) in (
        ("shared", AR1_SHARED, false, false),
        ("varying", AR1_VARYING, true, false),
        ("stepped", AR1_STEPPED, false, true))
    @stestset "AR1Normal fitted forecast: $label" begin
        fit = SBBRMI(builder(AR1_FIT); mod=@__MODULE__)
        fit_data = deepcopy(fit.data)
        future_data = deepcopy(AR1_FUTURE)
        forecast = reprocess(fit, AR1_FUTURE)
        code = BRM.stan_code(forecast)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test forecast.held_out == Set((:kernel_y_cell_y_ragged,))
        @test all(!occursin("_y_ragged", String(key)) for key in keys(BRM.stan_data(forecast)))
        @test !occursin("_likelihood", code)
        descriptor = brm_descriptor(forecast)
        @test :predict in [op.name for op in descriptor.operations]
        @test :fit ∉ [op.name for op in descriptor.operations]
        predictive = brm_output(descriptor, :y; role=:posterior_predictive)
        fitted_predictive = brm_output(brm_descriptor(fit), :y; role=:posterior_predictive)
        @test predictive.name == fitted_predictive.name

        fit_problem = BRM.stan_instantiate(fit)
        forecast_problem = BRM.stan_instantiate(forecast)
        @test BridgeStan.param_unc_names(fit_problem.model) ==
            BridgeStan.param_unc_names(forecast_problem.model)
        n = LogDensityProblems.dimension(fit_problem)
        q = collect(range(0.35, 0.75; length=n))
        for problem in (fit_problem, forecast_problem)
            density, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
            @test isfinite(density)
            @test all(isfinite, gradient)
        end

        # On the original rows, the forecast equals the fit's predictive twin
        # at the same fitted point and seed. It does not use observed lagged y.
        same_rows = reprocess(fit, ar1_without_y(AR1_FIT))
        same_problem = BRM.stan_instantiate(same_rows)
        fitted_draw = brm_execute(brm_descriptor(fit), :predict;
            problem=fit_problem, draws=q, seed=2481)
        replay_draw = brm_execute(brm_descriptor(same_rows), :predict;
            problem=same_problem, draws=q, seed=2481)
        @test getproperty(fitted_draw, predictive.name) == getproperty(replay_draw, predictive.name)

        # A fit on the forecast rows is a density control, reusing the fit's
        # preparation. Its generated trajectory also equals the forecast's.
        labelled = merge(AR1_FUTURE, (; y=collect(range(-2., 2.; length=length(AR1_FUTURE.x)))))
        control = reprocess(fit, labelled)
        control_problem = BRM.stan_instantiate(control)
        draw = brm_execute(descriptor, :predict; problem=forecast_problem, draws=q, seed=913)
        twin = brm_execute(brm_descriptor(control), :predict;
            problem=control_problem, draws=q, seed=913)
        @test getproperty(draw, predictive.name) == getproperty(twin, predictive.name)
        @test length(getproperty(draw, predictive.name)) == length(AR1_FUTURE.x)

        constrained = ar1_columns(forecast_problem.model, q)
        count = length(AR1_FUTURE.x)
        mu = [constrained["mu.$i"] for i in 1:count]
        sigma = varying ? [constrained["sigma.$i"] for i in 1:count] : fill(constrained["sigma"], count)
        phi = varying ? [constrained["phi.$i"] for i in 1:count] : fill(constrained["phi"], count)
        rows = groups(AR1_FUTURE.group; order=AR1_FUTURE.time)
        ordered = reduce(vcat, rows)
        @test predictive.segments == cumsum(length.(rows))
        obs_z = ar1_standardized_innovations(labelled.y[ordered], mu, sigma, phi,
            rows, AR1_FUTURE.time; stepped)
        # Independent Normal conditional likelihood, including all scale terms.
        loglik = 0.0
        position = 0
        for group in rows, (j, b) in enumerate(group)
            position += 1
            gap = j == 1 ? 0 : (stepped ? 1 : AR1_FUTURE.time[b] - AR1_FUTURE.time[group[j-1]])
            sd = j == 1 ? sigma[b] / sqrt(1 - phi[b]^2) :
                sigma[b] * sqrt((1 - phi[b]^(2gap)) / (1 - phi[b]^2))
            loglik += -0.5 * (log(2pi) + obs_z[position]^2) - log(sd)
        end
        @test BridgeStan.log_density(control_problem.model, q; propto=false) -
            BridgeStan.log_density(forecast_problem.model, q; propto=false) ≈ loglik atol=2e-10

        # These are generated draws at fixed fitted coordinates, not an MCMC
        # fit. Standardizing by the law must give independent N(0,1) errors,
        # including the stationary first row and gaps using the later row's
        # sigma/phi. A one-step draw from observed lags fails this check.
        names = BridgeStan.param_names(forecast_problem.model; include_tp=true, include_gq=true)
        indices = [only(findall(==(string(predictive.name, ".", i)), names)) for i in 1:count]
        rng = BridgeStan.StanRNG(forecast_problem.model, 732)
        errors = Matrix{Float64}(undef, count, 6000)
        for sample in axes(errors, 2)
            values = BridgeStan.param_constrain(forecast_problem.model, q;
                include_tp=true, include_gq=true, rng)
            errors[:, sample] = ar1_standardized_innovations(values[indices], mu, sigma, phi,
                rows, AR1_FUTURE.time; stepped)
        end
        @test maximum(abs, mean(errors; dims=2)) < 0.06
        @test maximum(abs, var(errors; dims=2) .- 1) < 0.09
        @test maximum(abs, [mean(errors[a, :] .* errors[b, :])
            for a in 1:count for b in 1:a-1]) < 0.07

        # Fresh omission has no fitted coordinates; the replay retains them.
        prior = SBBRMI(builder(AR1_FUTURE); mod=@__MODULE__)
        @test isempty(StanBlocks.stan_unc_names(prior.model))
        @test StanBlocks.stanc_check(BRM.stan_code(prior); warn_pedantic=false).ok
        @test n > 0
        @test isequal(fit.data, fit_data)
        @test isequal(AR1_FUTURE, future_data)
    end
end
