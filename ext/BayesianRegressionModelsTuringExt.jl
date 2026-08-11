module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: Exponential, Normal, Poisson, product_distribution
using LogExpFunctions: logistic
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

Turing.@model function _brm_population_binomial_logit(
    X, fixed, trials, y, beta_location, beta_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    eta = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ BRM.BinomialLogit(trials[i], eta[i])
    end
    (; eta, trials)
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

Turing.@model function _brm_population_negative_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    log_mu = X_mean * beta_mean + fixed_mean
    log_phi = X_precision * beta_precision + fixed_precision
    mu = exp.(log_mu)
    phi = exp.(log_phi)
    for i in eachindex(y)
        y[i] ~ BRM.NegativeBinomial2(mu[i], phi[i])
    end
    (; log_mu, log_phi, mu, phi)
end

Turing.@model function _brm_population_beta_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, trials, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    logit_mean = X_mean * beta_mean + fixed_mean
    log_precision = X_precision * beta_precision + fixed_precision
    mean = logistic.(logit_mean)
    precision = exp.(log_precision)
    for i in eachindex(y)
        y[i] ~ BRM.BetaBinomial2(trials[i], mean[i], precision[i])
    end
    (; logit_mean, log_precision, mean, precision, trials)
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
    plan::BRM._TuringPopulationPlan{Val{:binomial_logit}})
    _brm_population_binomial_logit(
        plan.design.matrix,
        plan.design.fixed,
        plan.family_args.trials,
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


function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:negative_binomial2}})
    _brm_population_negative_binomial2(
        plan.mean.design.matrix,
        plan.mean.design.fixed,
        plan.precision.design.matrix,
        plan.precision.design.fixed,
        plan.response,
        plan.mean.beta_location,
        plan.mean.beta_scale,
        plan.precision.beta_location,
        plan.precision.beta_scale,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:beta_binomial2}})
    _brm_population_beta_binomial2(
        plan.mean.design.matrix,
        plan.mean.design.fixed,
        plan.precision.design.matrix,
        plan.precision.design.fixed,
        plan.family_args.trials,
        plan.response,
        plan.mean.beta_location,
        plan.mean.beta_scale,
        plan.precision.beta_location,
        plan.precision.beta_scale,
    )
end

function BRM.TuringBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_turing_plan(brmi)
    model = BRM._brm_turing_model(plan)
    BRM.TuringBRMI(brmi, plan, model)
end

end # module
