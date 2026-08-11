module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: Exponential, Normal, product_distribution
using Turing

const BRM = BayesianRegressionModels

Turing.@model function _brm_population_gaussian(
    X, y, beta_location, beta_scale, sigma_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    mu = X * beta_pop
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
    (; mu)
end

function BRM.TuringBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_turing_gaussian_plan(brmi)
    model = _brm_population_gaussian(
        plan.design.matrix,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.sigma_scale,
    )
    BRM.TuringBRMI(brmi, plan, model)
end

end # module
