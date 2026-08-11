module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: ContinuousUnivariateDistribution,
                     DiscreteUnivariateDistribution, Exponential, LKJCholesky,
                     Normal, Poisson, censored, logcdf,
                     product_distribution, truncated
using LinearAlgebra: Diagonal
using LogExpFunctions: logistic, log1mexp
import Distributions: logpdf
import Random: AbstractRNG, rand
using Turing

const BRM = BayesianRegressionModels

struct _BRMContinuousIntervalEvidence{D,U} <:
       ContinuousUnivariateDistribution
    base::D
    upper::U
end

struct _BRMDiscreteIntervalEvidence{D,U} <:
       DiscreteUnivariateDistribution
    base::D
    upper::U
end


function _brm_interval_logmass(base, lower, upper)
    upper_logcdf = logcdf(base, upper)
    lower_logcdf = logcdf(base, lower)
    lower_logcdf < upper_logcdf || return oftype(upper_logcdf, -Inf)
    upper_logcdf + log1mexp(lower_logcdf - upper_logcdf)
end

logpdf(d::_BRMContinuousIntervalEvidence, lower::Real) =
    _brm_interval_logmass(d.base, lower, d.upper)
logpdf(d::_BRMDiscreteIntervalEvidence, lower::Real) =
    _brm_interval_logmass(d.base, lower, d.upper)
rand(rng::AbstractRNG, d::_BRMContinuousIntervalEvidence) = rand(rng, d.base)
rand(rng::AbstractRNG, d::_BRMDiscreteIntervalEvidence) = rand(rng, d.base)

_brm_interval_evidence(base::ContinuousUnivariateDistribution, upper) =
    _BRMContinuousIntervalEvidence(base, upper)
_brm_interval_evidence(base::DiscreteUnivariateDistribution, upper) =
    _BRMDiscreteIntervalEvidence(base, upper)

_brm_normal_observation(::Nothing, mu, sigma, _i) = Normal(mu, sigma)
function _brm_normal_observation(
        modifier::BRM._BRMResponseModifierPlan, mu, sigma, i)
    lower = isnothing(modifier.lower) ? nothing :
        BRM._brm_response_bound_at(modifier.lower, i)
    upper = isnothing(modifier.upper) ? nothing :
        BRM._brm_response_bound_at(modifier.upper, i)
    base = Normal(mu, sigma)
    modifier.kind === :truncated && return truncated(base; lower, upper)
    modifier.kind === :censored && return censored(base; lower, upper)
    modifier.kind === :interval_censored &&
        return _brm_interval_evidence(base, upper)
    error("Turing backend: internal unsupported Normal response modifier " *
          "`$(modifier.kind)`")
end

_brm_poisson_observation(::Nothing, rate, _i) = Poisson(rate)
function _brm_poisson_observation(
        modifier::BRM._BRMResponseModifierPlan, rate, i)
    lower = isnothing(modifier.lower) ? nothing :
        BRM._brm_response_bound_at(modifier.lower, i)
    upper = isnothing(modifier.upper) ? nothing :
        BRM._brm_response_bound_at(modifier.upper, i)
    base = Poisson(rate)
    modifier.kind === :truncated && return truncated(base; lower, upper)
    modifier.kind === :censored && return censored(base; lower, upper)
    modifier.kind === :interval_censored &&
        return _brm_interval_evidence(base, upper)
    error("Turing backend: internal unsupported Poisson response modifier " *
          "`$(modifier.kind)`")
end

_random_effect_args(component) = isempty(component.random_effects) ?
    (0, zeros(0, 0), Int[], 0) : let block = only(component.random_effects)
        (block.intercept_only ? 1 : 2, block.matrix, block.indices,
         length(block.levels))
    end

Turing.@model function _brm_population_gaussian(
    X, fixed, y, beta_location, beta_scale, sigma_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    mu = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(response_modifier, mu[i], sigma, i)
    end
    (; mu)
end

Turing.@model function _brm_population_gaussian_random_intercept(
    X, fixed, group_idx, n_groups, y,
    beta_location, beta_scale, sigma_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    mu = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(response_modifier, mu[i], sigma, i)
    end
    (; mu, group_scale, group_effect)
end

Turing.@model function _brm_population_gaussian_correlated_group(
    X, fixed, Z, group_idx, n_groups, y,
    beta_location, beta_scale, sigma_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    n_terms = size(Z, 2)
    L_group ~ LKJCholesky(n_terms, 1.0)
    tau_group ~ product_distribution(
        fill(truncated(Normal(), 0.0, Inf), n_terms))
    z_group_flat ~ product_distribution(
        fill(Normal(), n_terms * n_groups))
    z_group = reshape(z_group_flat, n_terms, n_groups)
    b_group = transpose(Diagonal(tau_group) * Matrix(L_group.L) * z_group)
    group_effect = vec(sum(Z .* b_group[group_idx, :]; dims=2))
    mu = X * beta_pop + fixed + group_effect
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(response_modifier, mu[i], sigma, i)
    end
    (; mu, L_group, tau_group, b_group, group_effect)
end

Turing.@model function _brm_population_gaussian_zero_correlation_group(
    X, fixed, Z, group_idx, n_groups, intercept_index, y,
    beta_location, beta_scale, sigma_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    n_terms = size(Z, 2)
    n_slopes = n_terms - (intercept_index > 0)
    group_intercept_scale = nothing
    if intercept_index > 0
        log_group_intercept_scale ~ Normal()
        group_intercept_scale = exp(log_group_intercept_scale)
    end
    tau_group_slopes ~ product_distribution(
        fill(truncated(Normal(), 0.0, Inf), n_slopes))
    scales = if intercept_index == 0
        tau_group_slopes
    else
        vcat(tau_group_slopes[1:(intercept_index - 1)],
             [group_intercept_scale], tau_group_slopes[intercept_index:end])
    end
    z_group_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    z_group = reshape(z_group_flat, n_terms, n_groups)
    b_group = transpose(reshape(scales, :, 1) .* z_group)
    group_effect = vec(sum(Z .* b_group[group_idx, :]; dims=2))
    mu = X * beta_pop + fixed + group_effect
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(response_modifier, mu[i], sigma, i)
    end
    (; mu, group_intercept_scale, tau_group_slopes, scales,
       b_group, group_effect)
end

Turing.@model function _brm_population_glm_correlated_group(
    family, X, fixed, Z, group_idx, n_groups, trials, y,
    beta_location, beta_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    n_terms = size(Z, 2)
    L_group ~ LKJCholesky(n_terms, 1.0)
    tau_group ~ product_distribution(
        fill(truncated(Normal(), 0.0, Inf), n_terms))
    z_group_flat ~ product_distribution(
        fill(Normal(), n_terms * n_groups))
    z_group = reshape(z_group_flat, n_terms, n_groups)
    b_group = transpose(Diagonal(tau_group) * Matrix(L_group.L) * z_group)
    group_effect = vec(sum(Z .* b_group[group_idx, :]; dims=2))
    eta = X * beta_pop + fixed + group_effect
    rate = nothing
    if family isa Val{:bernoulli_logit}
        for i in eachindex(y)
            y[i] ~ BRM.BernoulliLogit(eta[i])
        end
    elseif family isa Val{:binomial_logit}
        for i in eachindex(y)
            y[i] ~ BRM.BinomialLogit(trials[i], eta[i])
        end
    elseif family isa Val{:poisson_log}
        rate = exp.(eta)
        for i in eachindex(y)
            y[i] ~ _brm_poisson_observation(response_modifier, rate[i], i)
        end
    else
        error("Turing backend: internal unsupported grouped GLM family $family")
    end
    (; eta, rate, L_group, tau_group, b_group, group_effect)
end

Turing.@model function _brm_population_glm_zero_correlation_group(
    family, X, fixed, Z, group_idx, n_groups, intercept_index, trials, y,
    beta_location, beta_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    n_terms = size(Z, 2)
    n_slopes = n_terms - (intercept_index > 0)
    group_intercept_scale = nothing
    if intercept_index > 0
        log_group_intercept_scale ~ Normal()
        group_intercept_scale = exp(log_group_intercept_scale)
    end
    tau_group_slopes ~ product_distribution(
        fill(truncated(Normal(), 0.0, Inf), n_slopes))
    scales = if intercept_index == 0
        tau_group_slopes
    else
        vcat(tau_group_slopes[1:(intercept_index - 1)],
             [group_intercept_scale], tau_group_slopes[intercept_index:end])
    end
    z_group_flat ~ product_distribution(fill(Normal(), n_terms * n_groups))
    z_group = reshape(z_group_flat, n_terms, n_groups)
    b_group = transpose(reshape(scales, :, 1) .* z_group)
    group_effect = vec(sum(Z .* b_group[group_idx, :]; dims=2))
    eta = X * beta_pop + fixed + group_effect
    rate = nothing
    if family isa Val{:bernoulli_logit}
        for i in eachindex(y)
            y[i] ~ BRM.BernoulliLogit(eta[i])
        end
    elseif family isa Val{:binomial_logit}
        for i in eachindex(y)
            y[i] ~ BRM.BinomialLogit(trials[i], eta[i])
        end
    elseif family isa Val{:poisson_log}
        rate = exp.(eta)
        for i in eachindex(y)
            y[i] ~ _brm_poisson_observation(response_modifier, rate[i], i)
        end
    else
        error("Turing backend: internal unsupported zero-correlation GLM family $family")
    end
    (; eta, rate, group_intercept_scale, tau_group_slopes, scales,
       b_group, group_effect)
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

Turing.@model function _brm_population_bernoulli_logit_random_intercept(
    X, fixed, group_idx, n_groups, y, beta_location, beta_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    eta = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ BRM.BernoulliLogit(eta[i])
    end
    (; eta, group_scale, group_effect)
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

Turing.@model function _brm_population_binomial_logit_random_intercept(
    X, fixed, group_idx, n_groups, trials, y, beta_location, beta_scale)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    eta = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ BRM.BinomialLogit(trials[i], eta[i])
    end
    (; eta, trials, group_scale, group_effect)
end

Turing.@model function _brm_population_poisson_log(
    X, fixed, y, beta_location, beta_scale, response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_rate = X * beta_pop + fixed
    rate = exp.(log_rate)
    for i in eachindex(y)
        y[i] ~ _brm_poisson_observation(response_modifier, rate[i], i)
    end
    (; log_rate, rate)
end

Turing.@model function _brm_population_poisson_log_random_intercept(
    X, fixed, group_idx, n_groups, y, beta_location, beta_scale,
    response_modifier)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    log_rate = X * beta_pop + fixed + group_effect[group_idx]
    rate = exp.(log_rate)
    for i in eachindex(y)
        y[i] ~ _brm_poisson_observation(response_modifier, rate[i], i)
    end
    (; log_rate, rate, group_scale, group_effect)
end

Turing.@model function _brm_population_negative_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale,
    mean_group_kind, mean_Z,
    mean_group_idx, n_mean_groups,
    precision_group_kind, precision_Z,
    precision_group_idx, n_precision_groups)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    mean_group_scale = nothing
    L_mean_group = nothing
    tau_mean_group = nothing
    b_mean_group = nothing
    mean_group_effect = zeros(length(y))
    if mean_group_kind == 1
        log_mean_group_scale ~ Normal()
        z_mean_group ~ product_distribution(fill(Normal(), n_mean_groups))
        mean_group_scale = exp(log_mean_group_scale)
        mean_group_values = mean_group_scale .* z_mean_group
        mean_group_effect = mean_group_values[mean_group_idx]
    elseif mean_group_kind == 2
        n_mean_terms = size(mean_Z, 2)
        L_mean_group ~ LKJCholesky(n_mean_terms, 1.0)
        tau_mean_group ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_mean_terms))
        z_mean_group_flat ~ product_distribution(
            fill(Normal(), n_mean_terms * n_mean_groups))
        z_mean_group = reshape(
            z_mean_group_flat, n_mean_terms, n_mean_groups)
        b_mean_group = transpose(
            Diagonal(tau_mean_group) * Matrix(L_mean_group.L) * z_mean_group)
        mean_group_effect = vec(sum(
            mean_Z .* b_mean_group[mean_group_idx, :]; dims=2))
    end
    precision_group_scale = nothing
    L_precision_group = nothing
    tau_precision_group = nothing
    b_precision_group = nothing
    precision_group_effect = zeros(length(y))
    if precision_group_kind == 1
        log_precision_group_scale ~ Normal()
        z_precision_group ~ product_distribution(
            fill(Normal(), n_precision_groups))
        precision_group_scale = exp(log_precision_group_scale)
        precision_group_values = precision_group_scale .* z_precision_group
        precision_group_effect = precision_group_values[precision_group_idx]
    elseif precision_group_kind == 2
        n_precision_terms = size(precision_Z, 2)
        L_precision_group ~ LKJCholesky(n_precision_terms, 1.0)
        tau_precision_group ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_precision_terms))
        z_precision_group_flat ~ product_distribution(
            fill(Normal(), n_precision_terms * n_precision_groups))
        z_precision_group = reshape(
            z_precision_group_flat, n_precision_terms, n_precision_groups)
        b_precision_group = transpose(
            Diagonal(tau_precision_group) * Matrix(L_precision_group.L) *
            z_precision_group)
        precision_group_effect = vec(sum(
            precision_Z .* b_precision_group[precision_group_idx, :]; dims=2))
    end
    log_mu = X_mean * beta_mean + fixed_mean + mean_group_effect
    log_phi = X_precision * beta_precision + fixed_precision +
              precision_group_effect
    mu = exp.(log_mu)
    phi = exp.(log_phi)
    for i in eachindex(y)
        y[i] ~ BRM.NegativeBinomial2(mu[i], phi[i])
    end
    (; log_mu, log_phi, mu, phi,
       mean_group_scale, mean_group_effect,
       L_mean_group, tau_mean_group, b_mean_group,
       precision_group_scale, precision_group_effect,
       L_precision_group, tau_precision_group, b_precision_group)
end

Turing.@model function _brm_population_beta_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, trials, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale,
    mean_group_kind, mean_Z,
    mean_group_idx, n_mean_groups,
    precision_group_kind, precision_Z,
    precision_group_idx, n_precision_groups)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    mean_group_scale = nothing
    L_mean_group = nothing
    tau_mean_group = nothing
    b_mean_group = nothing
    mean_group_effect = zeros(length(y))
    if mean_group_kind == 1
        log_mean_group_scale ~ Normal()
        z_mean_group ~ product_distribution(fill(Normal(), n_mean_groups))
        mean_group_scale = exp(log_mean_group_scale)
        mean_group_values = mean_group_scale .* z_mean_group
        mean_group_effect = mean_group_values[mean_group_idx]
    elseif mean_group_kind == 2
        n_mean_terms = size(mean_Z, 2)
        L_mean_group ~ LKJCholesky(n_mean_terms, 1.0)
        tau_mean_group ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_mean_terms))
        z_mean_group_flat ~ product_distribution(
            fill(Normal(), n_mean_terms * n_mean_groups))
        z_mean_group = reshape(
            z_mean_group_flat, n_mean_terms, n_mean_groups)
        b_mean_group = transpose(
            Diagonal(tau_mean_group) * Matrix(L_mean_group.L) * z_mean_group)
        mean_group_effect = vec(sum(
            mean_Z .* b_mean_group[mean_group_idx, :]; dims=2))
    end
    precision_group_scale = nothing
    L_precision_group = nothing
    tau_precision_group = nothing
    b_precision_group = nothing
    precision_group_effect = zeros(length(y))
    if precision_group_kind == 1
        log_precision_group_scale ~ Normal()
        z_precision_group ~ product_distribution(
            fill(Normal(), n_precision_groups))
        precision_group_scale = exp(log_precision_group_scale)
        precision_group_values = precision_group_scale .* z_precision_group
        precision_group_effect = precision_group_values[precision_group_idx]
    elseif precision_group_kind == 2
        n_precision_terms = size(precision_Z, 2)
        L_precision_group ~ LKJCholesky(n_precision_terms, 1.0)
        tau_precision_group ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_precision_terms))
        z_precision_group_flat ~ product_distribution(
            fill(Normal(), n_precision_terms * n_precision_groups))
        z_precision_group = reshape(
            z_precision_group_flat, n_precision_terms, n_precision_groups)
        b_precision_group = transpose(
            Diagonal(tau_precision_group) * Matrix(L_precision_group.L) *
            z_precision_group)
        precision_group_effect = vec(sum(
            precision_Z .* b_precision_group[precision_group_idx, :]; dims=2))
    end
    logit_mean = X_mean * beta_mean + fixed_mean + mean_group_effect
    log_precision = X_precision * beta_precision + fixed_precision +
                    precision_group_effect
    mean = logistic.(logit_mean)
    precision = exp.(log_precision)
    for i in eachindex(y)
        y[i] ~ BRM.BetaBinomial2(trials[i], mean[i], precision[i])
    end
    (; logit_mean, log_precision, mean, precision, trials,
       mean_group_scale, mean_group_effect,
       L_mean_group, tau_mean_group, b_mean_group,
       precision_group_scale, precision_group_effect,
       L_precision_group, tau_precision_group, b_precision_group)
end

function BRM._brm_turing_model(
    plan::BRM._TuringPopulationPlan{Val{:normal_identity}})
    if !isempty(plan.random_effects)
        block = only(plan.random_effects)
        if block.zero_correlation && !block.intercept_only
            intercept_index = something(
                findfirst(column -> column.label === :Intercept, block.columns), 0)
            return _brm_population_gaussian_zero_correlation_group(
                plan.design.matrix,
                plan.design.fixed,
                block.matrix,
                block.indices,
                length(block.levels),
                intercept_index,
                plan.response,
                plan.beta_location,
                plan.beta_scale,
                plan.scale_prior,
                plan.response_modifier,
            )
        end
        if !block.intercept_only
            return _brm_population_gaussian_correlated_group(
                plan.design.matrix,
                plan.design.fixed,
                block.matrix,
                block.indices,
                length(block.levels),
                plan.response,
                plan.beta_location,
                plan.beta_scale,
                plan.scale_prior,
                plan.response_modifier,
            )
        end
        return _brm_population_gaussian_random_intercept(
            plan.design.matrix,
            plan.design.fixed,
            block.indices,
            length(block.levels),
            plan.response,
            plan.beta_location,
            plan.beta_scale,
            plan.scale_prior,
            plan.response_modifier,
        )
    end
    _brm_population_gaussian(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.scale_prior,
        plan.response_modifier,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringPopulationPlan{Val{:bernoulli_logit}})
    if !isempty(plan.random_effects)
        block = only(plan.random_effects)
        if block.zero_correlation && !block.intercept_only
            intercept_index = something(
                findfirst(column -> column.label === :Intercept, block.columns), 0)
            return _brm_population_glm_zero_correlation_group(
                Val(:bernoulli_logit), plan.design.matrix, plan.design.fixed,
                block.matrix, block.indices, length(block.levels),
                intercept_index, Int[], plan.response,
                plan.beta_location, plan.beta_scale, nothing)
        end
        if !block.intercept_only
            return _brm_population_glm_correlated_group(
                Val(:bernoulli_logit),
                plan.design.matrix,
                plan.design.fixed,
                block.matrix,
                block.indices,
                length(block.levels),
                Int[],
                plan.response,
                plan.beta_location,
                plan.beta_scale,
                nothing,
            )
        end
        return _brm_population_bernoulli_logit_random_intercept(
            plan.design.matrix,
            plan.design.fixed,
            block.indices,
            length(block.levels),
            plan.response,
            plan.beta_location,
            plan.beta_scale,
        )
    end
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
    if !isempty(plan.random_effects)
        block = only(plan.random_effects)
        if block.zero_correlation && !block.intercept_only
            intercept_index = something(
                findfirst(column -> column.label === :Intercept, block.columns), 0)
            return _brm_population_glm_zero_correlation_group(
                Val(:binomial_logit), plan.design.matrix, plan.design.fixed,
                block.matrix, block.indices, length(block.levels),
                intercept_index, plan.family_args.trials, plan.response,
                plan.beta_location, plan.beta_scale, nothing)
        end
        if !block.intercept_only
            return _brm_population_glm_correlated_group(
                Val(:binomial_logit),
                plan.design.matrix,
                plan.design.fixed,
                block.matrix,
                block.indices,
                length(block.levels),
                plan.family_args.trials,
                plan.response,
                plan.beta_location,
                plan.beta_scale,
                nothing,
            )
        end
        return _brm_population_binomial_logit_random_intercept(
            plan.design.matrix,
            plan.design.fixed,
            block.indices,
            length(block.levels),
            plan.family_args.trials,
            plan.response,
            plan.beta_location,
            plan.beta_scale,
        )
    end
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
    if !isempty(plan.random_effects)
        block = only(plan.random_effects)
        if block.zero_correlation && !block.intercept_only
            intercept_index = something(
                findfirst(column -> column.label === :Intercept, block.columns), 0)
            return _brm_population_glm_zero_correlation_group(
                Val(:poisson_log), plan.design.matrix, plan.design.fixed,
                block.matrix, block.indices, length(block.levels),
                intercept_index, Int[], plan.response,
                plan.beta_location, plan.beta_scale, plan.response_modifier)
        end
        if !block.intercept_only
            return _brm_population_glm_correlated_group(
                Val(:poisson_log),
                plan.design.matrix,
                plan.design.fixed,
                block.matrix,
                block.indices,
                length(block.levels),
                Int[],
                plan.response,
                plan.beta_location,
                plan.beta_scale,
                plan.response_modifier,
            )
        end
        return _brm_population_poisson_log_random_intercept(
            plan.design.matrix,
            plan.design.fixed,
            block.indices,
            length(block.levels),
            plan.response,
            plan.beta_location,
            plan.beta_scale,
            plan.response_modifier,
        )
    end
    _brm_population_poisson_log(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.response_modifier,
    )
end


function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:negative_binomial2}})
    mean_group_kind, mean_Z, mean_group_idx, n_mean_groups =
        _random_effect_args(plan.mean)
    precision_group_kind, precision_Z, precision_group_idx,
        n_precision_groups = _random_effect_args(plan.precision)
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
        mean_group_kind,
        mean_Z,
        mean_group_idx,
        n_mean_groups,
        precision_group_kind,
        precision_Z,
        precision_group_idx,
        n_precision_groups,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:beta_binomial2}})
    mean_group_kind, mean_Z, mean_group_idx, n_mean_groups =
        _random_effect_args(plan.mean)
    precision_group_kind, precision_Z, precision_group_idx,
        n_precision_groups = _random_effect_args(plan.precision)
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
        mean_group_kind,
        mean_Z,
        mean_group_idx,
        n_mean_groups,
        precision_group_kind,
        precision_Z,
        precision_group_idx,
        n_precision_groups,
    )
end

function BRM.TuringBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_turing_plan(brmi)
    model = BRM._brm_turing_model(plan)
    BRM.TuringBRMI(brmi, plan, model)
end

end # module
