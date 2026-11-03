module SMCSelectiveFixture
using BayesianRegressionModels

hierarchical(data) = @brm data begin
    m ~ Normal(0, 4)
    z[1:nb] .~ Normal.(m, 2)
    mu = z[block]
    y ~ Normal(mu, 1)
end

scaled(data) = @brm data begin
    mu ~ 0 + zscale(x)
    effect(mu, :) ~ Normal(0, 1)
    z[1:nb] .~ Normal.(0, 1)
    eta = mu .+ z[block]
    y ~ Normal(eta, 0.7)
end

plated(data) = @brm data begin
    z[1:nb] .~ Normal.(0, 2)
    @plate for i in eachindex(block)
        y[i] ~ Normal(z[block[i]], 1)
    end
end
end
