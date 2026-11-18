# test/nlme_svi_trend.jl's ridge, shared with test/nlme_svi_reoptimize.jl and
# research/svi_reoptimize/trend_ridge_*.jl. Include it after
# `using BayesianRegressionModels`.
#
# The ridge: N units whose rows each pin a multiplicative step `exp(m + s z_i)`
# (non-centred, `z_i ~ Normal(0, 1)`) beside a level and noise. The steps share
# one value up to a spread of 0.02, so the posterior spread is small, but the fit
# starts it near its Exponential(1) prior's scale, and `s` falls only as fast as
# every unit's `z_i` grows: the ELBO climbs too slowly for two windows to see.
function svi_ridge_data(rng, N; nobs=100, gm=0.17, gs=0.02, noise=0.1)
    x = [1 .+ rand(rng, nobs) for _ in 1:N]
    lg = gm .+ gs .* randn(rng, N)
    lvl = 0.5 .* randn(rng, N)
    y = [lvl[i] .+ exp(lg[i]) .* x[i] .+ noise .* randn(rng, nobs) for i in 1:N]
    (; unit=["u$i" for i in 1:N], x, y, zero1=[[0.0] for _ in 1:N])
end
svi_ridge_model(data) = @brm data begin
    level ~ 1 + (1 | unit)
    gain_mean ~ Normal(0, 0.5)
    gain_spread ~ Exponential(1)
    noise ~ Exponential(1)
    @plate for i in eachindex(level)
        z ~ Normal(zero1[i], 1)
        steps = gain_mean .+ gain_spread .* z
        y[i] ~ Normal(level[i] .+ exp(steps[1]) .* x[i], noise)
    end
end
