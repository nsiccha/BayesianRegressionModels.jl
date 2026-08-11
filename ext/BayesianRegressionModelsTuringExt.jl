module BayesianRegressionModelsTuringExt

using BayesianRegressionModels
using Distributions: ContinuousUnivariateDistribution,
                     DiscreteUnivariateDistribution, Exponential, LKJCholesky,
                     Normal, Poisson, censored, logcdf,
                     product_distribution, truncated
using LinearAlgebra: Diagonal
using LogExpFunctions: logistic, log1mexp
import Distributions: logpdf
import Random: AbstractRNG, default_rng, rand
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

struct _BRMContinuousObjectiveWeight{D,W} <:
       ContinuousUnivariateDistribution
    base::D
    weight::W
end

struct _BRMDiscreteObjectiveWeight{D,W} <:
       DiscreteUnivariateDistribution
    base::D
    weight::W
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

logpdf(d::_BRMContinuousObjectiveWeight, x::Real) =
    d.weight * logpdf(d.base, x)
logpdf(d::_BRMDiscreteObjectiveWeight, x::Real) =
    d.weight * logpdf(d.base, x)
rand(rng::AbstractRNG, d::_BRMContinuousObjectiveWeight) = rand(rng, d.base)
rand(rng::AbstractRNG, d::_BRMDiscreteObjectiveWeight) = rand(rng, d.base)

_brm_interval_evidence(base::ContinuousUnivariateDistribution, upper) =
    _BRMContinuousIntervalEvidence(base, upper)
_brm_interval_evidence(base::DiscreteUnivariateDistribution, upper) =
    _BRMDiscreteIntervalEvidence(base, upper)

_brm_objective_observation(base, ::Nothing, _i) = base
function _brm_objective_observation(
        base::ContinuousUnivariateDistribution,
        weight::BRM._BRMObservationWeightPlan, i)
    weight.kind === :analytic && return base
    _BRMContinuousObjectiveWeight(base, weight.values[i])
end
function _brm_objective_observation(
        base::DiscreteUnivariateDistribution,
        weight::BRM._BRMObservationWeightPlan, i)
    weight.kind === :analytic && return base
    _BRMDiscreteObjectiveWeight(base, weight.values[i])
end

_brm_analytic_scale(::Nothing, sigma, _i) = sigma
_brm_analytic_scale(weight::BRM._BRMObservationWeightPlan, sigma, i) =
    weight.kind === :analytic ? sigma / sqrt(weight.values[i]) : sigma

_brm_normal_observation(::Nothing, weight, mu, sigma, i) =
    _brm_objective_observation(
        Normal(mu, _brm_analytic_scale(weight, sigma, i)), weight, i)
function _brm_normal_observation(
        modifier::BRM._BRMResponseModifierPlan, weight, mu, sigma, i)
    lower = isnothing(modifier.lower) ? nothing :
        BRM._brm_response_bound_at(modifier.lower, i)
    upper = isnothing(modifier.upper) ? nothing :
        BRM._brm_response_bound_at(modifier.upper, i)
    base = Normal(mu, _brm_analytic_scale(weight, sigma, i))
    observation = modifier.kind === :truncated ? truncated(base; lower, upper) :
                  modifier.kind === :censored ? censored(base; lower, upper) :
                  modifier.kind === :interval_censored ?
                      _brm_interval_evidence(base, upper) :
                  error("Turing backend: internal unsupported Normal response " *
                        "modifier `$(modifier.kind)`")
    _brm_objective_observation(observation, weight, i)
end

_brm_poisson_observation(::Nothing, weight, rate, i) =
    _brm_objective_observation(Poisson(rate), weight, i)
function _brm_poisson_observation(
        modifier::BRM._BRMResponseModifierPlan, weight, rate, i)
    lower = isnothing(modifier.lower) ? nothing :
        BRM._brm_response_bound_at(modifier.lower, i)
    upper = isnothing(modifier.upper) ? nothing :
        BRM._brm_response_bound_at(modifier.upper, i)
    base = Poisson(rate)
    observation = modifier.kind === :truncated ? truncated(base; lower, upper) :
                  modifier.kind === :censored ? censored(base; lower, upper) :
                  modifier.kind === :interval_censored ?
                      _brm_interval_evidence(base, upper) :
                  error("Turing backend: internal unsupported Poisson response " *
                        "modifier `$(modifier.kind)`")
    _brm_objective_observation(observation, weight, i)
end

_random_effect_args(component) = isempty(component.random_effects) ?
    (0, zeros(0, 0), Int[], 0, 0) : let block = only(component.random_effects)
        intercept_index = something(
            findfirst(column -> column.label === :Intercept, block.columns), 0)
        kind = block.intercept_only ? 1 : block.zero_correlation ? 3 : 2
        (kind, block.matrix, block.indices, length(block.levels),
         intercept_index)
    end

function _zero_correlation_scales(intercept_index, intercept_scale,
                                  slope_scales)
    intercept_index == 0 && return slope_scales
    vcat(slope_scales[1:(intercept_index - 1)], [intercept_scale],
         slope_scales[intercept_index:end])
end

function _noncentered_group_coefficients(scales, z_flat, n_groups)
    n_terms = length(scales)
    transpose(reshape(scales, n_terms, 1) .*
              reshape(z_flat, n_terms, n_groups))
end

Turing.@model function _brm_population_gaussian(
    X, fixed, y, beta_location, beta_scale, sigma_scale, response_modifier,
    observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    mu = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(
            response_modifier, observation_weight, mu[i], sigma, i)
    end
    (; mu, sigma, response=y)
end

Turing.@model function _brm_population_gaussian_random_intercept(
    X, fixed, group_idx, n_groups, y,
    beta_location, beta_scale, sigma_scale, response_modifier,
    observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    sigma ~ Exponential(sigma_scale)
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    mu = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(
            response_modifier, observation_weight, mu[i], sigma, i)
    end
    (; mu, sigma, response=y, group_scale, group_effect)
end

Turing.@model function _brm_population_gaussian_correlated_group(
    X, fixed, Z, group_idx, n_groups, y,
    beta_location, beta_scale, sigma_scale, response_modifier,
    observation_weight)
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
        y[i] ~ _brm_normal_observation(
            response_modifier, observation_weight, mu[i], sigma, i)
    end
    (; mu, sigma, response=y, L_group, tau_group, b_group, group_effect)
end

Turing.@model function _brm_population_gaussian_zero_correlation_group(
    X, fixed, Z, group_idx, n_groups, intercept_index, y,
    beta_location, beta_scale, sigma_scale, response_modifier,
    observation_weight)
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
        y[i] ~ _brm_normal_observation(
            response_modifier, observation_weight, mu[i], sigma, i)
    end
    (; mu, sigma, response=y, group_intercept_scale, tau_group_slopes, scales,
       b_group, group_effect)
end

Turing.@model function _brm_population_glm_correlated_group(
    family, X, fixed, Z, group_idx, n_groups, trials, y,
    beta_location, beta_scale, response_modifier, observation_weight)
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
            y[i] ~ _brm_objective_observation(
                BRM.BernoulliLogit(eta[i]), observation_weight, i)
        end
    elseif family isa Val{:binomial_logit}
        for i in eachindex(y)
            y[i] ~ _brm_objective_observation(
                BRM.BinomialLogit(trials[i], eta[i]), observation_weight, i)
        end
    elseif family isa Val{:poisson_log}
        rate = exp.(eta)
        for i in eachindex(y)
            y[i] ~ _brm_poisson_observation(
                response_modifier, observation_weight, rate[i], i)
        end
    else
        error("Turing backend: internal unsupported grouped GLM family $family")
    end
    (; eta, rate, response=y, L_group, tau_group, b_group, group_effect)
end

Turing.@model function _brm_population_glm_zero_correlation_group(
    family, X, fixed, Z, group_idx, n_groups, intercept_index, trials, y,
    beta_location, beta_scale, response_modifier, observation_weight)
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
            y[i] ~ _brm_objective_observation(
                BRM.BernoulliLogit(eta[i]), observation_weight, i)
        end
    elseif family isa Val{:binomial_logit}
        for i in eachindex(y)
            y[i] ~ _brm_objective_observation(
                BRM.BinomialLogit(trials[i], eta[i]), observation_weight, i)
        end
    elseif family isa Val{:poisson_log}
        rate = exp.(eta)
        for i in eachindex(y)
            y[i] ~ _brm_poisson_observation(
                response_modifier, observation_weight, rate[i], i)
        end
    else
        error("Turing backend: internal unsupported zero-correlation GLM family $family")
    end
    (; eta, rate, response=y, group_intercept_scale, tau_group_slopes, scales,
       b_group, group_effect)
end

Turing.@model function _brm_population_bernoulli_logit(
    X, fixed, y, beta_location, beta_scale, observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    eta = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BernoulliLogit(eta[i]), observation_weight, i)
    end
    (; eta, response=y)
end

Turing.@model function _brm_population_bernoulli_logit_random_intercept(
    X, fixed, group_idx, n_groups, y, beta_location, beta_scale,
    observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    eta = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BernoulliLogit(eta[i]), observation_weight, i)
    end
    (; eta, response=y, group_scale, group_effect)
end

Turing.@model function _brm_population_binomial_logit(
    X, fixed, trials, y, beta_location, beta_scale, observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    eta = X * beta_pop + fixed
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BinomialLogit(trials[i], eta[i]), observation_weight, i)
    end
    (; eta, trials, response=y)
end

Turing.@model function _brm_population_binomial_logit_random_intercept(
    X, fixed, group_idx, n_groups, trials, y, beta_location, beta_scale,
    observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    eta = X * beta_pop + fixed + group_effect[group_idx]
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BinomialLogit(trials[i], eta[i]), observation_weight, i)
    end
    (; eta, trials, response=y, group_scale, group_effect)
end

Turing.@model function _brm_population_poisson_log(
    X, fixed, y, beta_location, beta_scale, response_modifier,
    observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_rate = X * beta_pop + fixed
    rate = exp.(log_rate)
    for i in eachindex(y)
        y[i] ~ _brm_poisson_observation(
            response_modifier, observation_weight, rate[i], i)
    end
    (; log_rate, rate, response=y)
end

Turing.@model function _brm_population_poisson_log_random_intercept(
    X, fixed, group_idx, n_groups, y, beta_location, beta_scale,
    response_modifier, observation_weight)
    beta_pop ~ product_distribution(Normal.(beta_location, beta_scale))
    log_group_scale ~ Normal()
    z_group ~ product_distribution(fill(Normal(), n_groups))
    group_scale = exp(log_group_scale)
    group_effect = group_scale .* z_group
    log_rate = X * beta_pop + fixed + group_effect[group_idx]
    rate = exp.(log_rate)
    for i in eachindex(y)
        y[i] ~ _brm_poisson_observation(
            response_modifier, observation_weight, rate[i], i)
    end
    (; log_rate, rate, response=y, group_scale, group_effect)
end

Turing.@model function _brm_population_negative_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale,
    mean_group_kind, mean_Z,
    mean_group_idx, n_mean_groups, mean_intercept_index,
    precision_group_kind, precision_Z,
    precision_group_idx, n_precision_groups, precision_intercept_index,
    observation_weight)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    mean_group_scale = nothing
    L_mean_group = nothing
    tau_mean_group = nothing
    mean_group_intercept_scale = nothing
    tau_mean_group_slopes = nothing
    mean_group_scales = nothing
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
    elseif mean_group_kind == 3
        n_mean_terms = size(mean_Z, 2)
        n_mean_slopes = n_mean_terms - (mean_intercept_index > 0)
        if mean_intercept_index > 0
            log_mean_group_intercept_scale ~ Normal()
            mean_group_intercept_scale = exp(log_mean_group_intercept_scale)
        end
        tau_mean_group_slopes ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_mean_slopes))
        mean_group_scales = _zero_correlation_scales(
            mean_intercept_index, mean_group_intercept_scale,
            tau_mean_group_slopes)
        z_mean_group_flat ~ product_distribution(
            fill(Normal(), n_mean_terms * n_mean_groups))
        b_mean_group = _noncentered_group_coefficients(
            mean_group_scales, z_mean_group_flat, n_mean_groups)
        mean_group_effect = vec(sum(
            mean_Z .* b_mean_group[mean_group_idx, :]; dims=2))
    end
    precision_group_scale = nothing
    L_precision_group = nothing
    tau_precision_group = nothing
    precision_group_intercept_scale = nothing
    tau_precision_group_slopes = nothing
    precision_group_scales = nothing
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
    elseif precision_group_kind == 3
        n_precision_terms = size(precision_Z, 2)
        n_precision_slopes = n_precision_terms - (precision_intercept_index > 0)
        if precision_intercept_index > 0
            log_precision_group_intercept_scale ~ Normal()
            precision_group_intercept_scale = exp(
                log_precision_group_intercept_scale)
        end
        tau_precision_group_slopes ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_precision_slopes))
        precision_group_scales = _zero_correlation_scales(
            precision_intercept_index, precision_group_intercept_scale,
            tau_precision_group_slopes)
        z_precision_group_flat ~ product_distribution(
            fill(Normal(), n_precision_terms * n_precision_groups))
        b_precision_group = _noncentered_group_coefficients(
            precision_group_scales, z_precision_group_flat,
            n_precision_groups)
        precision_group_effect = vec(sum(
            precision_Z .* b_precision_group[precision_group_idx, :]; dims=2))
    end
    log_mu = X_mean * beta_mean + fixed_mean + mean_group_effect
    log_phi = X_precision * beta_precision + fixed_precision +
              precision_group_effect
    mu = exp.(log_mu)
    phi = exp.(log_phi)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.NegativeBinomial2(mu[i], phi[i]), observation_weight, i)
    end
    (; log_mu, log_phi, mu, phi, response=y,
       mean_group_scale, mean_group_effect,
       L_mean_group, tau_mean_group, mean_group_intercept_scale,
       tau_mean_group_slopes, mean_group_scales, b_mean_group,
       precision_group_scale, precision_group_effect,
       L_precision_group, tau_precision_group,
       precision_group_intercept_scale, tau_precision_group_slopes,
       precision_group_scales, b_precision_group)
end

Turing.@model function _brm_population_beta_binomial2(
    X_mean, fixed_mean, X_precision, fixed_precision, trials, y,
    mean_beta_location, mean_beta_scale,
    precision_beta_location, precision_beta_scale,
    mean_group_kind, mean_Z,
    mean_group_idx, n_mean_groups, mean_intercept_index,
    precision_group_kind, precision_Z,
    precision_group_idx, n_precision_groups, precision_intercept_index,
    observation_weight)
    beta_mean ~ product_distribution(
        Normal.(mean_beta_location, mean_beta_scale))
    beta_precision ~ product_distribution(
        Normal.(precision_beta_location, precision_beta_scale))
    mean_group_scale = nothing
    L_mean_group = nothing
    tau_mean_group = nothing
    mean_group_intercept_scale = nothing
    tau_mean_group_slopes = nothing
    mean_group_scales = nothing
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
    elseif mean_group_kind == 3
        n_mean_terms = size(mean_Z, 2)
        n_mean_slopes = n_mean_terms - (mean_intercept_index > 0)
        if mean_intercept_index > 0
            log_mean_group_intercept_scale ~ Normal()
            mean_group_intercept_scale = exp(log_mean_group_intercept_scale)
        end
        tau_mean_group_slopes ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_mean_slopes))
        mean_group_scales = _zero_correlation_scales(
            mean_intercept_index, mean_group_intercept_scale,
            tau_mean_group_slopes)
        z_mean_group_flat ~ product_distribution(
            fill(Normal(), n_mean_terms * n_mean_groups))
        b_mean_group = _noncentered_group_coefficients(
            mean_group_scales, z_mean_group_flat, n_mean_groups)
        mean_group_effect = vec(sum(
            mean_Z .* b_mean_group[mean_group_idx, :]; dims=2))
    end
    precision_group_scale = nothing
    L_precision_group = nothing
    tau_precision_group = nothing
    precision_group_intercept_scale = nothing
    tau_precision_group_slopes = nothing
    precision_group_scales = nothing
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
    elseif precision_group_kind == 3
        n_precision_terms = size(precision_Z, 2)
        n_precision_slopes = n_precision_terms - (precision_intercept_index > 0)
        if precision_intercept_index > 0
            log_precision_group_intercept_scale ~ Normal()
            precision_group_intercept_scale = exp(
                log_precision_group_intercept_scale)
        end
        tau_precision_group_slopes ~ product_distribution(
            fill(truncated(Normal(), 0.0, Inf), n_precision_slopes))
        precision_group_scales = _zero_correlation_scales(
            precision_intercept_index, precision_group_intercept_scale,
            tau_precision_group_slopes)
        z_precision_group_flat ~ product_distribution(
            fill(Normal(), n_precision_terms * n_precision_groups))
        b_precision_group = _noncentered_group_coefficients(
            precision_group_scales, z_precision_group_flat,
            n_precision_groups)
        precision_group_effect = vec(sum(
            precision_Z .* b_precision_group[precision_group_idx, :]; dims=2))
    end
    logit_mean = X_mean * beta_mean + fixed_mean + mean_group_effect
    log_precision = X_precision * beta_precision + fixed_precision +
                    precision_group_effect
    mean = logistic.(logit_mean)
    precision = exp.(log_precision)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BetaBinomial2(trials[i], mean[i], precision[i]),
            observation_weight, i)
    end
    (; logit_mean, log_precision, mean, precision, trials, response=y,
       mean_group_scale, mean_group_effect,
       L_mean_group, tau_mean_group, mean_group_intercept_scale,
       tau_mean_group_slopes, mean_group_scales, b_mean_group,
       precision_group_scale, precision_group_effect,
       L_precision_group, tau_precision_group,
       precision_group_intercept_scale, tau_precision_group_slopes,
       precision_group_scales, b_precision_group)
end

Turing.@model function _brm_shared_normal_response(
    mu, sigma, y, response_modifier, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_normal_observation(
            response_modifier, observation_weight, mu[i], sigma, i)
    end
    (; mu, sigma, response=y)
end

Turing.@model function _brm_shared_bernoulli_response(
    eta, y, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BernoulliLogit(eta[i]), observation_weight, i)
    end
    (; eta, response=y)
end

Turing.@model function _brm_shared_binomial_response(
    eta, trials, y, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BinomialLogit(trials[i], eta[i]), observation_weight, i)
    end
    (; eta, trials, response=y)
end

Turing.@model function _brm_shared_poisson_response(
    rate, y, response_modifier, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_poisson_observation(
            response_modifier, observation_weight, rate[i], i)
    end
    (; log_rate=log.(rate), rate, response=y)
end

Turing.@model function _brm_shared_negative_binomial2_response(
    mu, phi, y, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.NegativeBinomial2(mu[i], phi[i]), observation_weight, i)
    end
    (; mu, phi, response=y)
end

Turing.@model function _brm_shared_beta_binomial2_response(
    mean, precision, trials, y, observation_weight)
    for i in eachindex(y)
        y[i] ~ _brm_objective_observation(
            BRM.BetaBinomial2(trials[i], mean[i], precision[i]),
            observation_weight, i)
    end
    (; mean, precision, trials, response=y)
end


_brm_shared_response_model(
    plan::BRM._TuringPopulationPlan{Val{:normal_identity}}, owner) =
    _brm_shared_normal_response(
        owner.mu, owner.sigma, plan.response, plan.response_modifier,
        plan.observation_weight)
_brm_shared_response_model(
    plan::BRM._TuringPopulationPlan{Val{:bernoulli_logit}}, owner) =
    _brm_shared_bernoulli_response(
        owner.eta, plan.response, plan.observation_weight)
_brm_shared_response_model(
    plan::BRM._TuringPopulationPlan{Val{:binomial_logit}}, owner) =
    _brm_shared_binomial_response(
        owner.eta, plan.family_args.trials, plan.response,
        plan.observation_weight)
_brm_shared_response_model(
    plan::BRM._TuringPopulationPlan{Val{:poisson_log}}, owner) =
    _brm_shared_poisson_response(
        owner.rate, plan.response, plan.response_modifier,
        plan.observation_weight)
_brm_shared_response_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:negative_binomial2}}, owner) =
    _brm_shared_negative_binomial2_response(
        owner.mu, owner.phi, plan.response, plan.observation_weight)
_brm_shared_response_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:beta_binomial2}}, owner) =
    _brm_shared_beta_binomial2_response(
        owner.mean, owner.precision, plan.family_args.trials, plan.response,
        plan.observation_weight)


Turing.@model function _brm_multi_response(models, plans, owners)
    responses = Vector{Any}(undef, length(models))
    for i in eachindex(models)
        if owners[i] == i
            responses[i] ~ to_submodel(models[i])
        else
            responses[i] ~ to_submodel(
                _brm_shared_response_model(plans[i], responses[owners[i]]))
        end
    end
    (; responses)
end

function BRM._brm_turing_model(plan::BRM._TuringMultiResponsePlan)
    models = Tuple(BRM._brm_turing_model(child) for child in plan.plans)
    _brm_multi_response(models, plan.plans, plan.owners)
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
                plan.observation_weight,
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
                plan.observation_weight,
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
            plan.observation_weight,
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
        plan.observation_weight,
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
                plan.beta_location, plan.beta_scale, nothing,
                plan.observation_weight)
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
                plan.observation_weight,
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
            plan.observation_weight,
        )
    end
    _brm_population_bernoulli_logit(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.observation_weight,
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
                plan.beta_location, plan.beta_scale, nothing,
                plan.observation_weight)
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
                plan.observation_weight,
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
            plan.observation_weight,
        )
    end
    _brm_population_binomial_logit(
        plan.design.matrix,
        plan.design.fixed,
        plan.family_args.trials,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.observation_weight,
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
                plan.beta_location, plan.beta_scale, plan.response_modifier,
                plan.observation_weight)
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
                plan.observation_weight,
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
            plan.observation_weight,
        )
    end
    _brm_population_poisson_log(
        plan.design.matrix,
        plan.design.fixed,
        plan.response,
        plan.beta_location,
        plan.beta_scale,
        plan.response_modifier,
        plan.observation_weight,
    )
end


function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:negative_binomial2}})
    mean_group_kind, mean_Z, mean_group_idx, n_mean_groups,
        mean_intercept_index =
        _random_effect_args(plan.mean)
    precision_group_kind, precision_Z, precision_group_idx,
        n_precision_groups, precision_intercept_index =
        _random_effect_args(plan.precision)
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
        mean_intercept_index,
        precision_group_kind,
        precision_Z,
        precision_group_idx,
        n_precision_groups,
        precision_intercept_index,
        plan.observation_weight,
    )
end

function BRM._brm_turing_model(
    plan::BRM._TuringMeanPrecisionPlan{Val{:beta_binomial2}})
    mean_group_kind, mean_Z, mean_group_idx, n_mean_groups,
        mean_intercept_index =
        _random_effect_args(plan.mean)
    precision_group_kind, precision_Z, precision_group_idx,
        n_precision_groups, precision_intercept_index =
        _random_effect_args(plan.precision)
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
        mean_intercept_index,
        precision_group_kind,
        precision_Z,
        precision_group_idx,
        n_precision_groups,
        precision_intercept_index,
        plan.observation_weight,
    )
end

function BRM.TuringBRMI(brmi::BRM.BRMI)
    plan = BRM._brm_turing_plan(brmi)
    model = BRM._brm_turing_model(plan)
    BRM.TuringBRMI(brmi, plan, model)
end

"""
    reprocess(backend::TuringBRMI, new_data;
              freeze_constants=true, resample_groups=())

Rebuild a direct-BRMI Turing backend on `new_data`. Frozen replay retains the
training transform constants, categorical coordinates, effect priors, and
ordinary group coordinates; unseen levels fail loudly. Groups named in
`resample_groups` instead take their levels from `new_data`. At posterior
prediction time [`turing_posterior_predictive`](@ref) retains fitted group
scales/correlation factors and redraws only those groups' standardized effects.

`freeze_constants=false` has fresh-fit semantics and should be refitted rather
than evaluated with the old posterior draw.
"""
function BRM.reprocess(
        backend::BRM.TuringBRMI, new_data;
        freeze_constants::Bool=true, resample_groups=())
    groups = resample_groups === nothing ? () :
             resample_groups isa Symbol ? (resample_groups,) :
             Tuple(resample_groups)
    all(group -> group isa Symbol, groups) || error(
        "Turing backend: `resample_groups` expects a Symbol or collection of " *
        "Symbols")
    length(unique(groups)) == length(groups) || error(
        "Turing backend: `resample_groups` contains duplicate group names")
    available_groups = BRM._turing_group_names(backend.plan)
    unknown_groups = setdiff(Set(groups), available_groups)
    isempty(unknown_groups) || error(
        "Turing backend: `resample_groups` names no fitted random-effect " *
        "block for $(sort!(collect(unknown_groups)))")
    prepared_data = freeze_constants ?
                    BRM._turing_replay_input(backend.plan, new_data) : new_data
    rebound = BRM._brm_rebind_brmi(backend.parent, prepared_data)
    fresh = BRM._brm_turing_plan(rebound)
    plan = freeze_constants ? BRM._turing_replay_plan(
        backend.plan, fresh, Set{Symbol}(groups)) : fresh
    model = BRM._brm_turing_model(plan)
    replay = BRM._TuringReplayState(Tuple(groups))
    BRM.TuringBRMI(rebound, plan, model, replay)
end

_brm_pointwise_indices(plan::BRM._TuringPopulationPlan) =
    isnothing(plan.missing_response) ? eachindex(plan.response) :
    plan.missing_response.observed_indices
_brm_pointwise_indices(plan::BRM._TuringMeanPrecisionPlan) =
    eachindex(plan.response)

function _brm_typed_loglikelihoods(values)
    isempty(values) && return Float64[]
    T = promote_type(map(typeof, values)...)
    T[values...]
end

function _brm_pointwise_response(plan, raw_values, first_index)
    indices = _brm_pointwise_indices(plan)
    last_index = first_index + length(indices) - 1
    segment = _brm_typed_loglikelihoods(raw_values[first_index:last_index])
    isnothing(plan.missing_response) && return segment, last_index + 1

    T = eltype(segment)
    aligned = Vector{Union{Missing,T}}(undef, length(plan.response))
    fill!(aligned, missing)
    aligned[indices] = segment
    aligned, last_index + 1
end

function _brm_named_pointwise(plan, raw_values)
    result, next_index = _brm_pointwise_response(plan, raw_values, 1)
    next_index == length(raw_values) + 1 || error(
        "Turing backend: DynamicPPL returned an unexpected number of " *
        "pointwise likelihood terms")
    response_name = BRM._turing_direct_observation(plan.context.parent).key
    NamedTuple{(response_name,)}((result,))
end

function _brm_named_pointwise(
        plan::BRM._TuringMultiResponsePlan, raw_values)
    results = Any[]
    next_index = 1
    for child in plan.plans
        result, next_index = _brm_pointwise_response(
            child, raw_values, next_index)
        push!(results, result)
    end
    next_index == length(raw_values) + 1 || error(
        "Turing backend: DynamicPPL returned an unexpected number of " *
        "pointwise likelihood terms")
    NamedTuple{plan.responses}(Tuple(results))
end

function BRM.turing_pointwise_loglikelihoods(
        backend::BRM.TuringBRMI, parameters)
    pointwise = Turing.DynamicPPL.pointwise_loglikelihoods(
        backend.model, Turing.DynamicPPL.InitFromParams(parameters))
    raw_values = values(pointwise)
    all(value -> value isa Number, raw_values) || error(
        "Turing backend: DynamicPPL returned a non-scalar pointwise " *
        "likelihood term for a rowwise BRM observation")
    _brm_named_pointwise(backend.plan, raw_values)
end

function BRM.turing_predictive_model(backend::BRM.TuringBRMI)
    BRM._brm_turing_model(BRM._turing_predictive_plan(backend.plan))
end

BRM.turing_generated_quantities(backend::BRM.TuringBRMI, parameters) =
    Turing.DynamicPPL.returned(backend.model, parameters)

function _brm_complete_predictive_response(response)
    any(ismissing, response) && error(
        "Turing backend: posterior-predictive execution left an ungenerated " *
        "response value")
    collect(nonmissingtype(eltype(response)), response)
end

function _brm_named_predictive(plan, returned)
    response_name = BRM._turing_direct_observation(plan.context.parent).key
    response = _brm_complete_predictive_response(returned.response)
    NamedTuple{(response_name,)}((response,))
end

function _brm_named_predictive(
        plan::BRM._TuringMultiResponsePlan, returned)
    responses = Tuple(
        _brm_complete_predictive_response(child.response)
        for child in returned.responses)
    NamedTuple{plan.responses}(responses)
end

function _brm_resampled_latents(
        plan::BRM._TuringPopulationPlan, groups)
    names = Symbol[]
    for block in plan.random_effects
        block.group in groups || continue
        push!(names, block.intercept_only ? :z_group : :z_group_flat)
    end
    Set(names)
end

function _brm_component_resampled_latents!(
        names, component, groups, prefix::Symbol)
    for block in component.random_effects
        block.group in groups || continue
        suffix = block.intercept_only ? :group : :group_flat
        push!(names, Symbol(:z_, prefix, :_, suffix))
    end
    names
end

function _brm_resampled_latents(
        plan::BRM._TuringMeanPrecisionPlan, groups)
    names = Set{Symbol}()
    _brm_component_resampled_latents!(names, plan.mean, groups, :mean)
    _brm_component_resampled_latents!(
        names, plan.precision, groups, :precision)
end

function _brm_without_fields(parameters::NamedTuple, removed)
    kept = Tuple(name for name in keys(parameters) if name ∉ removed)
    NamedTuple{kept}(Tuple(getproperty(parameters, name) for name in kept))
end

function _brm_resampled_parameters(backend::BRM.TuringBRMI, parameters)
    groups = Set{Symbol}(backend.replay.resample_groups)
    isempty(groups) && return parameters
    if backend.plan isa BRM._TuringMultiResponsePlan
        point_parameters = Turing.DynamicPPL.VarNamedTuple(parameters)
        removed = Set{String}()
        for i in eachindex(backend.plan.plans)
            backend.plan.owners[i] == i || continue
            for name in _brm_resampled_latents(backend.plan.plans[i], groups)
                push!(removed, "responses[$i].$name")
            end
        end
        kept = Dict{Turing.DynamicPPL.VarName,Any}()
        for variable in keys(point_parameters)
            string(variable) in removed && continue
            kept[variable] = point_parameters[variable]
        end
        return kept
    end
    parameters isa NamedTuple || error(
        "Turing backend: `resample_groups` posterior prediction currently " *
        "requires one constrained parameter draw as a NamedTuple")
    removed = _brm_resampled_latents(backend.plan, groups)
    _brm_without_fields(parameters, removed)
end

function BRM.turing_posterior_predictive(
        rng::AbstractRNG, backend::BRM.TuringBRMI, parameters)
    predictive = BRM.turing_predictive_model(backend)
    fixed_parameters = _brm_resampled_parameters(backend, parameters)
    fixed = Turing.fix(predictive, fixed_parameters)
    draw = rand(rng, fixed)
    returned = Turing.DynamicPPL.returned(fixed, draw.data)
    _brm_named_predictive(backend.plan, returned)
end

BRM.turing_posterior_predictive(
    backend::BRM.TuringBRMI, parameters; rng=default_rng()) =
    BRM.turing_posterior_predictive(rng, backend, parameters)

end # module
