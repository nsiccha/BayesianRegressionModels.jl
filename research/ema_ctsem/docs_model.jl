# An irregularly observed Ornstein-Uhlenbeck process with its exact transition.
# This is the executable model of docs/src/state-space.md.
using BayesianRegressionModels, StanBlocks, Distributions, Random

StanBlocks.@deffun begin
    ou_path(dt::vector[n], z::vector[n], mu::real, rate::real, stationary_sd::real)::vector[n] = begin
        x::vector[n]
        x[1] = mu + stationary_sd * z[1]
        for j in 2:n
            persistence = exp(-rate * dt[j])
            innovation_sd = stationary_sd * sqrt(-expm1(-2.0 * rate * dt[j]))
            x[j] = mu + persistence * (x[j - 1] - mu) + innovation_sd * z[j]
        end
        x
    end
end

"A public synthetic series, generated from the same exact continuous-time transition."
function state_space_data(; n=40, seed=144)
    rng = Xoshiro(seed)
    dt = vcat(0.0, 0.3 .+ rand(rng, n - 1))
    time = cumsum(dt)
    mu, rate, stationary_sd, sigma = 0.5, 0.8, 0.6, 0.15
    truth = zeros(n)
    truth[1] = mu + stationary_sd * randn(rng)
    for j in 2:n
        a = exp(-rate * dt[j])
        truth[j] = mu + a * (truth[j - 1] - mu) + stationary_sd * sqrt(1 - a^2) * randn(rng)
    end
    y = truth + sigma * randn(rng, n)
    (; n, time, dt, y, truth)
end

function state_space_model(data=state_space_data())
    @brm data begin
        mu ~ Normal(0.0, 1.0)
        rate ~ LogNormal(log(0.8), 0.5)
        stationary_sd ~ LogNormal(log(0.6), 0.5)
        sigma ~ LogNormal(log(0.15), 0.4)
        z[1:n] .~ Normal.(0.0, 1.0)
        state = ou_path(dt, z, mu, rate, stationary_sd)
        y ~ Normal(state, sigma)
    end
end
