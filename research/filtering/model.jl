using BayesianRegressionModels, Distributions, Random

"""Independent synthetic count series, with all model settings supplied as data."""
function filtering_fixture(; seed=20261110, units=3, days=60)
    rng = Xoshiro(seed)
    level, rho, tau = 2.0, 0.8, 0.3
    offset = collect(range(-0.4, 0.4; length=units))
    state = zeros(units, days)
    y = [zeros(Int, days) for _ in 1:units]
    for i in 1:units
        state[i, 1] = randn(rng)
        for t in 2:days
            state[i, t] = rho * state[i, t - 1] + tau * randn(rng)
        end
        for t in 1:days
            y[i][t] = rand(rng, Poisson(exp(level + offset[i] + state[i, t])))
        end
    end
    settings = (; U=units, T=days, exposure=[ones(days) for _ in 1:units],
        level_location=2.0, level_scale=1.0, offset_location=0.0, offset_scale=0.5,
        rho_alpha=8.0, rho_beta=2.0, tau_scale=0.5,
        state_location=0.0, state_scale=1.0)
    data = (; settings..., y)
    truth = (; level, rho, tau, offset, state)
    (; data, truth)
end

function growing_model(data=(; U=2, unit=[1, 2], y=[6, 14],
        level_location=2.0, level_scale=1.0, offset_scale=0.5))
    @brm data begin
        level ~ Normal(level_location, level_scale)
        log_rate[1:U] .~ Normal.(level, offset_scale)
        rate = exp(log_rate[unit])
        y ~ Poisson(rate)
    end
end

"""Use a disjoint window or a cumulative prefix without passing future counts."""
function filtering_window(data, first_day, last_day; horizon=data.T)
    exposure = [[first_day <= t <= last_day ? 1.0 : 0.0 for t in 1:horizon]
        for _ in 1:data.U]
    y = [[first_day <= t <= last_day ? ys[t] : 0 for t in 1:horizon]
        for ys in data.y]
    (; data..., T=horizon, exposure, y)
end

function filtering_model(data=filtering_fixture(; units=2, days=6).data)
    @brm data begin
        level ~ Normal(level_location, level_scale)
        offset[1:U] .~ Normal.(offset_location, offset_scale)
        rho ~ Beta(rho_alpha, rho_beta)
        tau ~ Exponential(tau_scale)
        @plate for i in eachindex(y)
            @scan begin
                state[1] ~ Normal(state_location, state_scale)
                for t in 2:T
                    state[t] ~ Normal(rho * state[t - 1], tau)
                end
            end
            y[i] ~ Poisson(exposure[i] .* exp.(level + offset[i] .+ state))
        end
    end
end
