module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: Exponential, Normal, Poisson, product_distribution
using Turing

const BRM = BayesianRegressionModels

Turing.@model function _brm_population_gaussian(
    X, fixed, y, beta_location, beta_scale, sigma_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    mu = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
    (; mu)
end

Turing.@model function _brm_population_bernoulli_logit(
    X, fixed, y, beta_location, beta_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    eta = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ BRM.BernoulliLogit(eta[i])
    end
    (; eta)
end

Turing.@model function _brm_population_poisson_log(
    X, fixed, y, beta_location, beta_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_rate = X * beta_pop + fixed
    rate = exp.(log_rate)
    for i in eachindex(y)
        y[i] ~ Poisson(rate[i])
    end
    (; log_rate, rate)
end

function BRM._brm_turing_model(
    plan::BRM._TuringPopulationPlan{Val{:normal_identity}})
    _brm_population_gaussian(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.scale_prior,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringPopulationPlan{Val{:bernoulli_logit}})
    _brm_population_bernoulli_logit(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringPopulationPlan{Val{:poisson_log}})
    _brm_population_poisson_log(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
    )
end

function BRM.TuringBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_turing_plan(brmi)
    model = BRM._brm_turing_model(plan)
    BRM.TuringBRMI(brmi, plan, model)
end

end # module
