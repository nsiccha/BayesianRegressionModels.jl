using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, LKJCholesky, Normal, Poisson, logpdf
using Enzyme
using LogDensityProblems
using Random: MersenneTwister, rand, randexp, randn

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL

function direct_native_model(family::Symbol, transform::Symbol;
                             coefficient_keys=(:Intercept, :x))
    location = family === :gaussian ? :mu :
        family === :bernoulli ? :eta : :log_rate
    coefficient_name = Symbol(:beta_, location)
    coefficients = NP.parameter(
        NP.RealSupport(), coefficient_keys;
        transform=NP.Identity(), prior=NP.StandardNormal())
    parameters = if family === :gaussian
        scale = NP.parameter(
            NP.PositiveSupport(), (:sigma,);
            transform=NP.Exp(), prior=NP.Exponential(2.0))
        NamedTuple{(coefficient_name, :sigma)}((coefficients, scale))
    else
        NamedTuple{(coefficient_name,)}((coefficients,))
    end

    transform_name = transform === :identity ? nothing :
        Symbol(transform, :_x_for_, location)
    transform_declaration = transform === :center ? NP.center(:x) :
        transform === :zscale ? NP.zscale(:x) : nothing
    affine_input = transform_name === nothing ? :x : transform_name
    affine_declaration = NP.affine(affine_input, coefficient_name)
    nodes = transform_name === nothing ?
        NamedTuple{(location,)}((affine_declaration,)) :
        NamedTuple{(transform_name, location)}(
            (transform_declaration, affine_declaration))
    observation = if family === :gaussian
        NP.normal(:y, location, :sigma)
    elseif family === :bernoulli
        NP.bernoulli_logit(:y, location)
    else
        rate_name = Symbol(:exp_, location)
        nodes = merge(nodes, NamedTuple{(rate_name,)}((NP.exp_link(location),)))
        NP.poisson(:y, rate_name)
    end
    NP.model(
        inputs=(; x=NP.input()),
        parameters=parameters,
        nodes=nodes,
        observations=(; y=NP.broadcasted(observation)))
end

conditioned(model, data) =
    NP.condition(NP.substitute(model; x=data.x); y=data.y)

function direct_multi_model(family::Symbol;
                            coefficient_keys=(:Intercept, :x, :w))
    coefficients = NP.parameter(
        NP.RealSupport(), coefficient_keys;
        transform=NP.Identity(), prior=NP.StandardNormal())
    location = family === :gaussian ? :mu :
        family === :bernoulli ? :eta : :log_rate
    parameters = NamedTuple{(Symbol(:beta_, location),)}((coefficients,))
    if family === :gaussian
        sigma = NP.parameter(
            NP.PositiveSupport(), (:sigma,);
            transform=NP.Exp(), prior=NP.Exponential(2.0))
        parameters = merge(parameters, (; sigma))
    end
    transform_names = if location === :mu
        (:zscale_x_for_mu, :center_w_for_mu)
    else
        (Symbol(:zscale_x_for_, location), Symbol(:center_w_for_, location))
    end
    nodes = NamedTuple{transform_names}((NP.zscale(:x), NP.center(:w)))
    nodes = merge(nodes, NamedTuple{(location,)}((NP.affine(
        transform_names, Symbol(:beta_, location)),)))
    observation = if family === :gaussian
        NP.normal(:y, location, :sigma)
    elseif family === :bernoulli
        NP.bernoulli_logit(:y, location)
    else
        rate = Symbol(:exp_, location)
        nodes = merge(nodes, NamedTuple{(rate,)}((NP.exp_link(location),)))
        NP.poisson(:y, rate)
    end
    NP.model(
        inputs=(; x=NP.input(), w=NP.input()),
        parameters=parameters, nodes=nodes,
        observations=(; y=NP.broadcasted(observation)))
end

direct_multi_gaussian_model() = direct_multi_model(
    :gaussian; coefficient_keys=(:intercept, :beta_x, :beta_w))

function native_brmi(family::Symbol, transform::Symbol, data)
    if family === :gaussian
        transform === :identity && return @brm data begin
            sigma ~ Exponential(2.0)
            mu ~ 1 + x
            y ~ Normal(mu, sigma)
        end
        transform === :center && return @brm data begin
            sigma ~ Exponential(2.0)
            mu ~ 1 + center(x)
            y ~ Normal(mu, sigma)
        end
        return @brm data begin
            sigma ~ Exponential(2.0)
            mu ~ 1 + zscale(x)
            y ~ Normal(mu, sigma)
        end
    elseif family === :bernoulli
        transform === :identity && return @brm data begin
            eta ~ 1 + x
            y ~ BernoulliLogit(eta)
        end
        transform === :center && return @brm data begin
            eta ~ 1 + center(x)
            y ~ BernoulliLogit(eta)
        end
        return @brm data begin
            eta ~ 1 + zscale(x)
            y ~ BernoulliLogit(eta)
        end
    end
    transform === :identity && return @brm data begin
        log_rate ~ 1 + x
        y ~ Poisson(exp(log_rate))
    end
    transform === :center && return @brm data begin
        log_rate ~ 1 + center(x)
        y ~ Poisson(exp(log_rate))
    end
    @brm data begin
        log_rate ~ 1 + zscale(x)
        y ~ Poisson(exp(log_rate))
    end
end

function multi_native_brmi(family::Symbol, data)
    family === :gaussian && return @brm data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + zscale(x) + center(w)
        y ~ Normal(mu, sigma)
    end
    family === :bernoulli && return @brm data begin
        eta ~ 1 + zscale(x) + center(w)
        y ~ BernoulliLogit(eta)
    end
    @brm data begin
        log_rate ~ 1 + zscale(x) + center(w)
        y ~ Poisson(exp(log_rate))
    end
end

function deterministic_preprocessing_composition(
    family::Symbol, data; concise::Bool=true)
    preprocessing_instance = if concise
        concise_zscale_component(data.x)
    else
        preprocessing_model = NP.model(
            inputs=(; raw=NP.input()),
            nodes=(; scaled=NP.zscale(:raw)),
            observations=(;),
            outputs=(; scaled=:scaled))
        NP.substitute(preprocessing_model; raw=data.x)
    end
    preprocessing = NP.component(
        :preprocessing,
        preprocessing_instance)
    regression_model = direct_native_model(family, :identity)
    regression = NP.component(
        :regression,
        NP.condition(
            NP.substitute(
                regression_model;
                x=NP.output(preprocessing, :scaled));
            y=data.y))
    NP.compose(preprocessing, regression)
end

function check_plan_structure(left, right)
    @test typeof(left) === typeof(right)
    @test keys(left.axes) == keys(right.axes)
    @test keys(left.inputs) == keys(right.inputs)
    @test keys(left.parameters) == keys(right.parameters)
    @test keys(left.nodes) == keys(right.nodes)
    @test keys(left.factors) == keys(right.factors)
    @test keys(left.queries) == keys(right.queries)
    @test left.axes.observation.keys == right.axes.observation.keys
    @test left.axes.coefficient.keys == right.axes.coefficient.keys
    @test map(typeof, values(left.nodes)) == map(typeof, values(right.nodes))
    @test map(typeof, values(left.factors)) == map(typeof, values(right.factors))
    @test map(typeof, values(left.queries)) == map(typeof, values(right.queries))
    @test sprint(show, left) == sprint(show, right)
    if !isempty(left.nodes.transforms)
        left_transforms = values(left.nodes.transforms)
        right_transforms = values(right.nodes.transforms)
        @test length(left_transforms) == length(right_transforms)
        for (left_transform, right_transform) in
            zip(left_transforms, right_transforms)
            @test typeof(left_transform) === typeof(right_transform)
            @test left_transform.mean == right_transform.mean
            if left_transform isa BRM.NativePPLZScaleNode
                @test left_transform.scale == right_transform.scale
            end
        end
    end
    if hasproperty(left.factors, :scale_prior)
        @test left.factors.scale_prior.scale == right.factors.scale_prior.scale
    end
    nothing
end

NP.@model function macro_gaussian_identity(
    x::AbstractVector{<:Real})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    sigma ~ Exponential(2.0)
    mu = intercept + slope * x
    @. y ~ Normal(mu, sigma)
end

NP.@model function macro_gaussian_zscale(
    x::AbstractVector{<:Real})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    sigma ~ Exponential(2.0)
    mu = intercept + slope * standardize(x)
    @. y ~ Normal(mu, sigma)
end

NP.@model function macro_scalar_gaussian(x::AbstractVector{<:Real})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    sigma ~ Exponential(2.0)
    mu = intercept + slope * x
    y ~ Normal(mu, sigma)
end

NP.@model function macro_multi_gaussian(
    x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    intercept ~ Normal()
    beta_x ~ Normal()
    beta_w ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept + beta_x * zscale(x) + beta_w * center(w)
    @. y ~ Normal(mu, sigma)
end

NP.@model function macro_multi_bernoulli(
    x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    intercept ~ Normal()
    beta_x ~ Normal()
    beta_w ~ Normal()
    eta = intercept + beta_x * zscale(x) + beta_w * center(w)
    @. y ~ BernoulliLogit(eta)
end

NP.@model function macro_multi_poisson(
    x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    intercept ~ Normal()
    beta_x ~ Normal()
    beta_w ~ Normal()
    log_rate = intercept + beta_x * zscale(x) + beta_w * center(w)
    @. y ~ Poisson(exp(log_rate))
end

NP.@model function macro_multi_gaussian_dotted(
    x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    intercept ~ Normal()
    beta_x ~ Normal()
    beta_w ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ zscale(x) .* beta_x .+ beta_w .* center(w)
    @. y ~ Normal(mu, sigma)
end

NP.@model function composable_gaussian(x)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept + slope * x
    @. y ~ Normal(mu, sigma)
    return y
end

NP.@model function concise_zscale_component(raw)
    scaled = zscale(raw)
    return scaled
end

NP.@model function concise_named_preprocessing(raw)
    centered = center(raw)
    scaled = zscale(raw)
    return (centered=centered, standardized=scaled)
end

NP.@model function unknown_output_component(raw)
    scaled = zscale(raw)
    return missing
end

NP.@model function concise_passthrough(raw)
    return raw
end

NP.@model function concise_aliased_zscale(raw)
    scaled = zscale(raw)
    return (standardized=scaled,)
end

NP.@model function scalar_normal_prior()
    theta ~ Normal()
    return theta
end

NP.@model function scalar_normal_site(mu, tau)
    z ~ Normal(mu, tau)
    return z
end

NP.@model function aliased_scalar_normal_prior()
    theta ~ Normal(0, 1)
    return (coefficient=theta,)
end

NP.@model function named_scalar_normal_priors()
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    return (; intercept, slope)
end

NP.@model function scalar_normal_likelihood(mu)
    sigma ~ Exponential(2.0)
    @. y ~ Normal(mu, sigma)
    return y
end

NP.@model function natural_latent_normal(prior_mu, prior_tau)
    z ~ scalar_normal_site(prior_mu, prior_tau)
    y ~ scalar_normal_likelihood(z)
    return y
end

module QualifiedStagedModels
using BayesianRegressionModels
using Distributions: Normal

const NP = BayesianRegressionModels.NativePPL

NP.@model function qualified_nested(prior_mu, prior_tau)
    z ~ Normal(prior_mu, prior_tau)
    return z
end
end

NP.@model function qualified_nested(prior_mu, prior_tau)
    z ~ QualifiedStagedModels.qualified_nested(prior_mu, prior_tau)
    return z
end

NP.@model function natural_preprocessed_normal(raw)
    scaled = concise_zscale_component(raw)
    y ~ composable_gaussian(scaled)
    return y
end

NP.@model function assigned_stochastic_submodel()
    z = scalar_normal_site(0.0, 1.0)
    return z
end

NP.@model function sampled_deterministic_submodel(raw)
    scaled ~ concise_zscale_component(raw)
    return scaled
end

NP.@model function ambiguous_multioutput_submodel()
    values ~ named_scalar_normal_priors()
    return values
end

NP.@model function hierarchical_latent_graph()
    population ~ Normal()
    population_scale ~ Exponential(1.0)
    individual ~ Normal(population, population_scale)
    observation_scale ~ Exponential(2.0)
    @. y ~ Normal(individual, observation_scale)
    return y
end

NP.@model function hierarchy_population_source()
    value ~ Normal()
    return value
end

NP.@model function hierarchy_scale_source(scale)
    value ~ Exponential(scale)
    return value
end

NP.@model function hierarchy_individual_source(population, population_scale)
    value ~ Normal(population, population_scale)
    return value
end

NP.@model function hierarchy_observation_source(individual)
    observation_scale ~ Exponential(2.0)
    @. y ~ Normal(individual, observation_scale)
    return y
end

NP.@model function naturally_composed_hierarchy()
    population ~ hierarchy_population_source()
    population_scale ~ hierarchy_scale_source(1.0)
    individual ~ hierarchy_individual_source(
        population, population_scale)
    y ~ hierarchy_observation_source(individual)
    return y
end

NP.@model function deterministic_scale_factor_graph(unit_scale)
    population ~ Normal()
    log_observation_scale ~ Normal(population, unit_scale)
    observation_scale = exp(log_observation_scale)
    @. y ~ Normal(population, observation_scale)
    return y
end

function distributional_gaussian_factor_graph()
    NP.model(
        inputs=(; x=NP.input(), z=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:Intercept, :x);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            beta_log_sigma=NP.parameter(
                NP.RealSupport(), (:Intercept, :z);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        nodes=(;
            mu=NP.affine(:x, :beta_mu),
            log_sigma=NP.affine(:z, :beta_log_sigma),
            sigma=NP.exp_link(:log_sigma)),
        observations=(;
            y=NP.broadcasted(NP.normal(:y, :mu, :sigma))))
end

NP.@model function natural_distributional_gaussian(x, z)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    beta_log_sigma[(:Intercept, :z)] ~ StandardNormal()
    mu = dot(beta_mu, (1, x))
    log_sigma = dot(beta_log_sigma, (1, z))
    sigma = exp(log_sigma)
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_sampled_offset_regression(x)
    latent ~ Normal()
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + offset(latent)
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_exposure_poisson(x, exposure)
    beta_log_rate[(:Intercept, :x)] ~ StandardNormal()
    log_rate = dot(beta_log_rate, (1, x)) + offset(log(exposure))
    @. y ~ Poisson(exp(log_rate))
end

NP.@model function natural_exposure_only_poisson(exposure)
    beta_log_rate[(:Intercept,)] ~ StandardNormal()
    log_rate = dot(beta_log_rate, (1,)) + offset(log(exposure))
    @. y ~ Poisson(exp(log_rate))
end

NP.@model function natural_raw_offset_poisson(x, exposure)
    beta_log_rate[(:Intercept, :x)] ~ StandardNormal()
    log_rate = dot(beta_log_rate, (1, x)) + offset(exposure)
    @. y ~ Poisson(exp(log_rate))
end

NP.@model function natural_varying_intercept(x, group)
    tau_p_group ~ Exponential(1)
    b_p_group[group] ~ Normal(0.0, tau_p_group)
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + b_p_group[group]
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_weighted_varying_intercept(
        x, group, replicates)
    tau_g_group ~ Exponential(1)
    b_g_group[group] ~ Normal(0.0, tau_g_group)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (1, x)) + b_g_group[group]
    @. y ~ weighted(Normal(mu, sigma), aweights(replicates))
end

NP.@model function natural_varying_slope(x, group)
    tau_p_group ~ Exponential(1)
    b_p_group[group] ~ Normal(0.0, tau_p_group)
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + b_p_group[group] * x
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_correlated_varying_intercept_slope(x, group)
    tau_p_group[(:Intercept, :x)] ~ Exponential(1)
    L_p_group[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + dot(b_p_group[group], (1, x))
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_transformed_correlated_varying_slope(x, group)
    tau_p_group[(:Intercept, :x)] ~ Exponential(1)
    L_p_group[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu[(:x,)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (zscale(x),)) +
        dot(b_p_group[group], (1, zscale(x)))
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_correlated_varying_three(x, w, group)
    tau_p_group[(:Intercept, :x, :w)] ~ Exponential(1)
    L_p_group[(:Intercept, :x, :w)] ~ LKJCholesky(3, 2)
    b_p_group[group, (:Intercept, :x, :w)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu[(:x, :w)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (x, w)) +
        dot(b_p_group[group], (1, x, w))
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_crossed_group_regression(x, subject, item)
    tau_p_subject[(:Intercept, :x)] ~ Exponential(1)
    L_p_subject[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_subject[subject, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_subject, L_p_subject)
    tau_q_item ~ Exponential(1)
    b_q_item[item] ~ Normal(0.0, tau_q_item)
    beta_mu[(:x,)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (x,)) +
        dot(b_p_subject[subject], (1, x)) + b_q_item[item]
    @. y ~ Normal(mu, sigma)
end

NP.@model function natural_correlated_bernoulli_logit(x, group)
    tau_p_group[(:Intercept, :x)] ~ Exponential(1)
    L_p_group[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta ~ Normal()
    mu = beta * x + dot(b_p_group[group], (1, x))
    @. y ~ BernoulliLogit(mu)
end

NP.@model function natural_correlated_poisson_log(x, group)
    tau_p_group[(:Intercept, :x)] ~ Exponential(1)
    L_p_group[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta ~ Normal()
    log_rate = beta * x + dot(b_p_group[group], (1, x))
    @. y ~ Poisson(exp(log_rate))
end

NP.@model function monolithic_scalar_normal()
    theta ~ Normal()
    sigma ~ Exponential(2.0)
    @. y ~ Normal(theta, sigma)
end

NP.@model function monolithic_latent_normal(prior_mu, prior_tau)
    z ~ Normal(prior_mu, prior_tau)
    sigma ~ Exponential(2.0)
    @. y ~ Normal(z, sigma)
end

NP.@model function monolithic_scale_named_site(prior_mu, prior_tau)
    scale ~ Normal(prior_mu, prior_tau)
    sigma ~ Exponential(2.0)
    @. y ~ Normal(scale, sigma)
end

NP.@model function macro_bernoulli_center(
    x::AbstractVector{<:Real})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    eta = intercept .+ slope .* center(x)
    y .~ BernoulliLogit.(eta)
end

NP.@model function macro_poisson_zscale(
    x::AbstractVector{<:Real})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    log_rate = intercept .+ slope .* zscale(x)
    y .~ Poisson.(exp(log_rate))
end

module NativePPLMacroHygiene
import BayesianRegressionModels
const NP = BayesianRegressionModels.NativePPL
const model = :shadow
const input = :shadow
const parameter = :shadow
const Normal = :shadow
const Exponential = :shadow
const zscale = :shadow

NP.@model function hygienic(x::Vector{Float64})
    intercept ~ Normal()
    slope ~ Normal(0, 1)
    sigma ~ Exponential(2.0)
    mu = intercept + slope * zscale(x)
    @. y ~ Normal(mu, sigma)
end
end

function capability_error(f)
    try
        f()
    catch err
        err isa BRM.NativePPLCapabilityError || rethrow()
        return err
    end
    error("expected NativePPLCapabilityError")
end

function bundle_execution_allocated(
    outputs, workspace, prepared, positions, queries)
    @allocated BRM.NativePPL.execute_draws!(
        outputs, workspace, prepared, positions, queries)
end

function check_transformed_execution(transformed_plan, explicit_plan, position)
    transformed = BRM.NativePPL.prepare(transformed_plan)
    explicit = BRM.NativePPL.prepare(explicit_plan)
    transformed_work = BRM.NativePPL.workspace(
        transformed, Float64, DI.AutoEnzyme())
    explicit_work = BRM.NativePPL.workspace(
        explicit, Float64, DI.AutoEnzyme())
    linear = BRM.NativePPL.LinearPredictor()
    pointwise = BRM.NativePPL.PointwiseLogLikelihood()
    predictive = BRM.NativePPL.PosteriorPredictive()

    @test BRM.NativePPL.logdensity!(transformed_work, transformed, position) ≈
          BRM.NativePPL.logdensity!(explicit_work, explicit, position)
    transformed_density, transformed_gradient =
        BRM.NativePPL.logdensity_and_gradient!(
            transformed_work, transformed, position)
    explicit_density, explicit_gradient =
        BRM.NativePPL.logdensity_and_gradient!(
            explicit_work, explicit, position)
    @test transformed_density ≈ explicit_density
    @test transformed_gradient ≈ explicit_gradient
    @test BRM.NativePPL.evaluate(
        transformed_work, transformed, position, linear) ≈
          BRM.NativePPL.evaluate(explicit_work, explicit, position, linear)
    @test BRM.NativePPL.evaluate(
        transformed_work, transformed, position, pointwise) ≈
          BRM.NativePPL.evaluate(explicit_work, explicit, position, pointwise)
    @test BRM.NativePPL.simulate(
        MersenneTwister(901), transformed_work, transformed, position, predictive) ≈
          BRM.NativePPL.simulate(
              MersenneTwister(901), explicit_work, explicit, position, predictive)

    positions = permutedims(hcat(position, position ./ 2))
    queries = (; location=linear, pointwise, prediction=predictive)
    transformed_signatures = BRM.NativePPL.batch_output_signature(
        transformed, positions, queries)
    explicit_signatures = BRM.NativePPL.batch_output_signature(
        explicit, positions, queries)
    @test map(BRM.NativePPL.output_axes, transformed_signatures) ==
          map(BRM.NativePPL.output_axes, explicit_signatures)
    transformed_outputs = BRM.NativePPL.allocate_output(
        transformed_signatures, transformed)
    explicit_outputs = BRM.NativePPL.allocate_output(
        explicit_signatures, explicit)
    BRM.NativePPL.execute_draws!(
        MersenneTwister(902), transformed_outputs, transformed_work, transformed,
        positions, queries)
    BRM.NativePPL.execute_draws!(
        MersenneTwister(902), explicit_outputs, explicit_work, explicit,
        positions, queries)
    @test transformed_outputs.location ≈ explicit_outputs.location
    @test transformed_outputs.pointwise ≈ explicit_outputs.pointwise
    @test transformed_outputs.prediction ≈ explicit_outputs.prediction
    @test bundle_execution_allocated(
        MersenneTwister(903), transformed_outputs, transformed_work, transformed,
        positions, queries) == 0
    transformed
end

replace_plan_nodes(plan, nodes) = BRM.NativePPLPlan(
    plan.axes, plan.inputs, plan.parameters, nodes, plan.factors,
    plan.queries, plan.bindings)
replace_plan_transform(plan, transform) = replace_plan_nodes(
    plan, merge(plan.nodes, (; transforms=NamedTuple{
        keys(plan.nodes.transforms)}((transform,)))))
replace_plan_factors(plan, factors) = BRM.NativePPLPlan(
    plan.axes, plan.inputs, plan.parameters, plan.nodes, factors,
    plan.queries, plan.bindings)
replace_plan_queries(plan, queries) = BRM.NativePPLPlan(
    plan.axes, plan.inputs, plan.parameters, plan.nodes, plan.factors,
    queries, plan.bindings)

function bundle_execution_allocated(
    rng, outputs, workspace, prepared, positions, queries)
    @allocated BRM.NativePPL.execute_draws!(
        rng, outputs, workspace, prepared, positions, queries)
end

function argument_error(f)
    try
        f()
    catch err
        err isa ArgumentError || rethrow()
        return err
    end
    error("expected ArgumentError")
end

function steady_state_allocations(workspace, prepared, position)
    BRM._native_ppl_logdensity!(workspace, prepared, position)
    BRM._native_ppl_logdensity_and_gradient!(workspace, prepared, position)
    primal = @allocated(BRM._native_ppl_logdensity!(workspace, prepared, position))
    gradient = @allocated(
        BRM._native_ppl_logdensity_and_gradient!(workspace, prepared, position))
    (; primal, gradient)
end

function factor_steady_state_allocations(workspace::NP.FactorWorkspace,
                                         prepared::NP.FactorPrepared,
                                         position)
    NP.logdensity!(workspace, prepared, position)
    NP.logdensity_and_gradient!(workspace, prepared, position)
    primal = @allocated NP.logdensity!(workspace, prepared, position)
    gradient = @allocated NP.logdensity_and_gradient!(
        workspace, prepared, position)
    (; primal, gradient)
end

function factor_query_allocations(workspace::NP.FactorWorkspace,
                                  prepared::NP.FactorPrepared,
                                  position, linear, pointwise, predictive,
                                  prior_position)
    query = NP.LinearPredictor()
    likelihood = NP.PointwiseLogLikelihood()
    rng = MersenneTwister(921)
    prior_rng = MersenneTwister(922)
    NP.evaluate!(linear, workspace, prepared, position, query)
    NP.evaluate!(pointwise, workspace, prepared, position, likelihood)
    NP.simulate!(rng, predictive, workspace, prepared, position)
    NP.simulate_prior!(
        prior_rng, prior_position, predictive, workspace, prepared)
    linear_bytes = @allocated NP.evaluate!(
        linear, workspace, prepared, position, query)
    pointwise_bytes = @allocated NP.evaluate!(
        pointwise, workspace, prepared, position, likelihood)
    predictive_bytes = @allocated NP.simulate!(
        rng, predictive, workspace, prepared, position)
    prior_bytes = @allocated NP.simulate_prior!(
        prior_rng, prior_position, predictive, workspace, prepared)
    (; linear=linear_bytes, pointwise=pointwise_bytes,
       predictive=predictive_bytes, prior=prior_bytes)
end

function factor_node_query_allocations(
    workspace::NP.FactorWorkspace, prepared::NP.FactorPrepared,
    position, mu, sigma, positions, mu_draws, sigma_draws,
    bundle_outputs, bundle_queries)
    mu_query = NP.NodeOutput(:mu)
    sigma_query = NP.NodeOutput(:sigma)
    NP.evaluate!(mu, workspace, prepared, position, mu_query)
    NP.evaluate!(sigma, workspace, prepared, position, sigma_query)
    NP.evaluate_draws!(
        mu_draws, workspace, prepared, positions, mu_query)
    NP.evaluate_draws!(
        sigma_draws, workspace, prepared, positions, sigma_query)
    NP.execute_draws!(
        bundle_outputs, workspace, prepared, positions, bundle_queries)
    scalar_mu = @allocated NP.evaluate!(
        mu, workspace, prepared, position, mu_query)
    scalar_sigma = @allocated NP.evaluate!(
        sigma, workspace, prepared, position, sigma_query)
    batch_mu = @allocated NP.evaluate_draws!(
        mu_draws, workspace, prepared, positions, mu_query)
    batch_sigma = @allocated NP.evaluate_draws!(
        sigma_draws, workspace, prepared, positions, sigma_query)
    bundle = @allocated NP.execute_draws!(
        bundle_outputs, workspace, prepared, positions, bundle_queries)
    (; scalar_mu, scalar_sigma, batch_mu, batch_sigma, bundle)
end

function factor_predictive_allocations(
    rng, output, workspace::NP.FactorWorkspace,
    prepared::NP.FactorPrepared, position)
    NP.simulate!(rng, output, workspace, prepared, position)
    @allocated NP.simulate!(
        rng, output, workspace, prepared, position)
end

function factor_generated_draw_allocations(
    predictive_rng, linear_rng, bundle_rng, predictive, linear, bundle,
    workspace::NP.FactorWorkspace, prepared::NP.FactorPrepared,
    positions, queries)
    NP.simulate_draws!(
        predictive_rng, predictive, workspace, prepared, positions)
    NP.evaluate_draws!(
        linear_rng, linear, workspace, prepared, positions,
        NP.LinearPredictor())
    NP.execute_draws!(
        bundle_rng, bundle, workspace, prepared, positions, queries)
    predictive_bytes = @allocated NP.simulate_draws!(
        predictive_rng, predictive, workspace, prepared, positions)
    linear_bytes = @allocated NP.evaluate_draws!(
        linear_rng, linear, workspace, prepared, positions,
        NP.LinearPredictor())
    bundle_bytes = @allocated NP.execute_draws!(
        bundle_rng, bundle, workspace, prepared, positions, queries)
    (; predictive=predictive_bytes, linear=linear_bytes,
       bundle=bundle_bytes)
end

function factor_batch_allocations(workspace::NP.FactorWorkspace,
                                  prepared::NP.FactorPrepared,
                                  positions, linear, pointwise, predictive,
                                  bundle)
    linear_query = NP.LinearPredictor()
    pointwise_query = NP.PointwiseLogLikelihood()
    queries = (;
        linear=linear_query,
        pointwise=pointwise_query,
        predictive=NP.PosteriorPredictive())
    rng = MersenneTwister(924)
    bundle_rng = MersenneTwister(925)
    NP.evaluate_draws!(
        linear, workspace, prepared, positions, linear_query)
    NP.evaluate_draws!(
        pointwise, workspace, prepared, positions, pointwise_query)
    NP.simulate_draws!(
        rng, predictive, workspace, prepared, positions)
    NP.execute_draws!(
        bundle_rng, bundle, workspace, prepared, positions, queries)
    linear_bytes = @allocated NP.evaluate_draws!(
        linear, workspace, prepared, positions, linear_query)
    pointwise_bytes = @allocated NP.evaluate_draws!(
        pointwise, workspace, prepared, positions, pointwise_query)
    predictive_bytes = @allocated NP.simulate_draws!(
        rng, predictive, workspace, prepared, positions)
    bundle_bytes = @allocated NP.execute_draws!(
        bundle_rng, bundle, workspace, prepared, positions, queries)
    (; linear=linear_bytes, pointwise=pointwise_bytes,
       predictive=predictive_bytes, bundle=bundle_bytes)
end

function allocating_query_bytes(workspace, prepared, position)
    linear = BRM.NativePPL.LinearPredictor()
    predictive = BRM.NativePPL.PosteriorPredictive()
    BRM.NativePPL.evaluate(workspace, prepared, position, linear)
    BRM.NativePPL.simulate(
        MersenneTwister(9), workspace, prepared, position, predictive)
    allocation = @allocated(BRM.NativePPL.allocate_output(prepared, linear))
    evaluation = @allocated(
        BRM.NativePPL.evaluate(workspace, prepared, position, linear))
    rng = MersenneTwister(9)
    simulation = @allocated(
        BRM.NativePPL.simulate(rng, workspace, prepared, position, predictive))
    (; allocation, evaluation, simulation)
end

@testset "typed native Gaussian plan" begin
    data = (;
        dose=[-1.0, 0.0, 2.0],
        response=Float32[0.5, 1.0, 2.5],
    )
    brmi = @brm data begin
        residual ~ Exponential(2.5)
        location ~ 1 + dose
        response ~ Normal(location, residual)
    end

    plan = BRM._native_ppl_plan(brmi)

    @test plan isa BRM.NativePPLPlan
    @test LogDensityProblems.dimension(plan) == 3

    @test BRM.native_axis_name(plan.axes.observation) == :observation
    @test plan.axes.observation.keys == Base.OneTo(3)
    @test BRM.native_axis_name(plan.axes.coefficient) == :location_coefficient
    @test plan.axes.coefficient.keys == (:Intercept, :dose)
    @test BRM.native_axis_name(plan.axes.scale) == :residual_scalar
    @test plan.axes.scale.keys == (:residual,)

    @test BRM.native_input_name(plan.inputs.predictors.dose) == :dose
    @test BRM.native_input_role(plan.inputs.predictors.dose) == :predictor
    @test eltype(plan.inputs.predictors.dose) == Float64
    @test BRM.native_input_name(plan.inputs.response) == :response
    @test BRM.native_input_role(plan.inputs.response) == :response
    @test eltype(plan.inputs.response) == Float32

    @test BRM.native_parameter_name(plan.parameters.coefficients) == :beta_location
    @test plan.parameters.coefficients.support isa BRM.NativePPLRealSupport
    @test plan.parameters.coefficients.transform isa BRM.NativePPLIdentityTransform
    @test plan.parameters.coefficients.axis === plan.axes.coefficient
    @test plan.parameters.coefficients.unconstrained == 1:2
    @test BRM.native_parameter_name(plan.parameters.scale) == :residual
    @test plan.parameters.scale.support isa BRM.NativePPLPositiveSupport
    @test plan.parameters.scale.transform isa BRM.NativePPLExpTransform
    @test plan.parameters.scale.axis === plan.axes.scale
    @test plan.parameters.scale.unconstrained == 3:3

    position = [0.25, -0.5, -2.0]
    @test BRM.native_parameter_value(
        plan.parameters.coefficients, position, 1) == position[1]
    @test BRM.native_parameter_logabsdetjac(
        plan.parameters.coefficients, position, 2) == 0.0
    @test BRM.native_parameter_value(plan.parameters.scale, position) == exp(-2.0)
    @test BRM.native_parameter_logvalue(plan.parameters.scale, position) == -2.0
    @test BRM.native_parameter_logabsdetjac(plan.parameters.scale, position) == -2.0
    @test_throws TypeError BRM.NativePPLParameter(
        :invalid, BRM.NativePPLPositiveSupport(), BRM.NativePPLIdentityTransform(),
        plan.axes.scale, 3:3)

    @test BRM.native_node_name(plan.nodes.location) == :location
    @test BRM.native_affine_input(plan.nodes.location) == :dose
    @test plan.nodes.location.axis === plan.axes.observation
    @test plan.nodes.location.intercept_index == 1
    @test plan.nodes.location.slope_indices == (2,)
    @test_throws ArgumentError BRM.NativePPLAffineNode(
        :invalid, (), plan.axes.observation, 1, ())
    @test_throws ArgumentError BRM.NativePPLAffineNode(
        :invalid, (:dose, :dose), plan.axes.observation, 1, (2, 3))

    @test plan.factors.coefficient_prior isa BRM.NativePPLStandardNormalFactor
    @test plan.factors.coefficient_prior.unconstrained == 1:2
    @test plan.factors.scale_prior isa BRM.NativePPLExponentialFactor
    @test plan.factors.scale_prior.unconstrained_index == 3
    @test plan.factors.scale_prior.scale == 2.5
    @test plan.factors.likelihood isa BRM.NativePPLNormalFactor
    @test plan.factors.likelihood.axis === plan.axes.observation

    coefficient_prior = BRM._native_ppl_factor_logdensity(
        plan.factors.coefficient_prior, plan.parameters.coefficients, position)
    @test coefficient_prior ≈
          logpdf(Normal(), position[1]) + logpdf(Normal(), position[2])
    scale_prior = BRM._native_ppl_factor_logdensity(
        plan.factors.scale_prior, plan.parameters.scale, position)
    @test scale_prior ≈
          logpdf(Exponential(2.5), exp(position[3])) + position[3]
    mismatched_coefficient_factor =
        BRM.NativePPLStandardNormalFactor(:wrong_parameter, 1:2)
    @test_throws MethodError BRM._native_ppl_factor_logdensity(
        mismatched_coefficient_factor, plan.parameters.coefficients, position)
    mismatched_scale_factor =
        BRM.NativePPLExponentialFactor(:wrong_parameter, 3, 2.5)
    @test_throws MethodError BRM._native_ppl_factor_logdensity(
        mismatched_scale_factor, plan.parameters.scale, position)

    @test plan.bindings.dose === data.dose
    @test plan.bindings.response === data.response
    @test sprint(show, plan) ==
        "NativePPLPlan(3 unconstrained parameters, 3 observations)\n" *
        "  inputs: dose, response\n" *
        "  parameters: beta_location, residual\n" *
        "  nodes: location\n" *
        "  factors: NativePPLStandardNormalFactor, " *
        "NativePPLExponentialFactor, NativePPLNormalFactor\n" *
        "  queries: linear_predictor, pointwise_loglikelihood, posterior_predictive"

    @test BRM.native_query_name(plan.queries.linear_predictor) == :linear_predictor
    @test plan.queries.linear_predictor isa
        BRM.NativePPLQuerySpec{
            :linear_predictor, :per_draw, :workspace, :until_next_evaluation,
        }
    @test BRM.native_query_name(plan.queries.pointwise_loglikelihood) ==
          :pointwise_loglikelihood
    @test plan.queries.pointwise_loglikelihood isa
        BRM.NativePPLQuerySpec{
            :pointwise_loglikelihood, :per_draw, :workspace,
            :until_next_evaluation,
        }
    @test BRM.native_query_name(plan.queries.posterior_predictive) ==
          :posterior_predictive
    @test plan.queries.posterior_predictive isa
        BRM.NativePPLQuerySpec{
            :posterior_predictive, :per_draw, :rng, :caller_owned,
        }
    @test all(
        BRM.native_output_axis(BRM.native_query_output(query)) ===
            plan.axes.observation
        for query in plan.queries)
    @test all(
        BRM.native_output_layout(BRM.native_query_output(query)) isa
            BRM.NativePPLDenseVectorLayout
        for query in plan.queries)
end


@testset "prepared native Gaussian execution" begin
    data = (;
        dose=[-1.0, 0.0, 2.0],
        response=Float32[0.5, 1.0, 2.5],
    )
    brmi = @brm data begin
        residual ~ Exponential(2.5)
        location ~ 1 + dose
        response ~ Normal(location, residual)
    end
    plan = BRM._native_ppl_plan(brmi)
    prepared = BRM._native_ppl_prepare(plan)
    workspace = BRM.NativePPL.workspace(prepared, Float64, DI.AutoEnzyme())

    @test Base.get_extension(
        BRM, :BayesianRegressionModelsDifferentiationInterfaceExt) !== nothing
    @test prepared isa BRM.NativePPLPrepared
    @test eltype(prepared) == Float64
    @test LogDensityProblems.dimension(prepared) == 3
    @test hasproperty(prepared, :predictor)
    @test prepared.predictor == data.dose
    @test prepared.predictor !== data.dose
    @test prepared.response == data.response
    @test prepared.response !== data.response
    @test prepared.workspace_spec.observation_axis === plan.axes.observation
    @test prepared.workspace_spec.gradient_length == 3
    @test sprint(show, prepared) == "NativePPLPrepared(3 observations, eltype=Float64)"

    @test workspace isa BRM.NativePPLWorkspace
    @test eltype(workspace) == Float64
    @test workspace.derivative !== nothing
    @test length(workspace.gradient) == 3
    @test sprint(show, workspace) ==
        "NativePPLWorkspace(eltype=Float64, location=3, " *
        "pointwise_loglikelihood=3, gradient=3)"

    position = [0.1, -0.2, log(1.3)]
    location = position[1] .+ position[2] .* prepared.predictor
    scale = exp(position[3])
    pointwise = logpdf.(Normal.(location, scale), prepared.response)
    expected_density =
        logpdf(Normal(), position[1]) +
        logpdf(Normal(), position[2]) +
        logpdf(Exponential(2.5), scale) + position[3] + sum(pointwise)
    expected_gradient = [
        -position[1] + sum((prepared.response .- location) ./ scale^2),
        -position[2] +
            sum(prepared.predictor .* (prepared.response .- location) ./ scale^2),
        1 - scale / 2.5 +
            sum(-1 .+ ((prepared.response .- location) ./ scale) .^ 2),
    ]

    density = BRM._native_ppl_logdensity!(workspace, prepared, position)
    @test density ≈ expected_density
    @test workspace.primal.location ≈ location
    @test workspace.primal.pointwise_loglikelihood ≈ pointwise

    likelihood = BRM._native_ppl_factor_logdensity!(
        plan.factors.likelihood, plan.inputs.response, plan.nodes.location,
        plan.parameters.scale, position, prepared, workspace.primal)
    @test likelihood ≈ sum(pointwise)
    mismatched_likelihood = BRM.NativePPLNormalFactor(
        :wrong_response, :location, :residual, plan.axes.observation)
    @test_throws MethodError BRM._native_ppl_factor_logdensity!(
        mismatched_likelihood, plan.inputs.response, plan.nodes.location,
        plan.parameters.scale, position, prepared, workspace.primal)

    gradient_density, gradient =
        BRM._native_ppl_logdensity_and_gradient!(workspace, prepared, position)
    @test gradient_density ≈ expected_density
    @test gradient === workspace.gradient
    @test gradient ≈ expected_gradient
    @test workspace.primal.location ≈ location
    @test workspace.primal.pointwise_loglikelihood ≈ pointwise

    allocations = steady_state_allocations(workspace, prepared, position)
    @test allocations == (; primal=0, gradient=0)

    underflow_density = BRM._native_ppl_logdensity!(
        workspace, prepared, [0.0, 0.0, -1000.0])
    @test underflow_density == -Inf
    @test all(==(-Inf), workspace.primal.pointwise_loglikelihood)

    prepared32 = BRM._native_ppl_prepare(plan; T=Float32)
    workspace32 = BRM.NativePPL.workspace(prepared32, Float32, DI.AutoEnzyme())
    density32, gradient32 = BRM._native_ppl_logdensity_and_gradient!(
        workspace32, prepared32, Float32.(position))
    @test density32 ≈ Float32(expected_density)
    @test gradient32 ≈ Float32.(expected_gradient)
end


@testset "native Bernoulli-logit workflow" begin
    data = (;
        x=[-2.0, 0.0, 1.5, 3.0],
        y=Bool[false, true, true, false],
    )
    brmi = @brm data begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end
    plan = BRM.NativePPL.compile(brmi)

    @test LogDensityProblems.dimension(plan) == 2
    @test keys(plan.axes) == (:observation, :coefficient)
    @test keys(plan.parameters) == (:coefficients,)
    @test keys(plan.factors) == (:coefficient_prior, :likelihood)
    @test BRM.native_parameter_name(plan.parameters.coefficients) == :beta_eta
    @test plan.parameters.coefficients.transform isa BRM.NativePPLIdentityTransform
    @test plan.parameters.coefficients.unconstrained == 1:2
    @test plan.factors.likelihood isa BRM.NativePPLBernoulliLogitFactor
    @test plan.factors.likelihood.axis === plan.axes.observation
    @test BRM.native_node_name(plan.nodes.location) == :eta
    @test BRM.native_affine_input(plan.nodes.location) == :x

    linear_query = BRM.NativePPL.LinearPredictor()
    pointwise_query = BRM.NativePPL.PointwiseLogLikelihood()
    predictive_query = BRM.NativePPL.PosteriorPredictive()
    linear_signature = BRM.NativePPL.output_signature(plan, linear_query)
    pointwise_signature = BRM.NativePPL.output_signature(plan, pointwise_query)
    predictive_signature = BRM.NativePPL.output_signature(plan, predictive_query)
    @test BRM.NativePPL.output_eltype(linear_signature, Float32) === Float32
    @test BRM.NativePPL.output_eltype(pointwise_signature, Float32) === Float32
    @test BRM.NativePPL.output_eltype(predictive_signature, Float32) === Bool
    @test BRM.NativePPL.output_axis(predictive_signature) === plan.axes.observation

    prepared = BRM.NativePPL.prepare(plan; T=Float64)
    @test prepared.response == Float64.(data.y)
    @test prepared.response !== data.y
    @test eltype(prepared) === Float64
    workspace = BRM.NativePPL.workspace(
        prepared, Float64, DI.AutoEnzyme())
    position = [0.3, -0.7]
    location = position[1] .+ position[2] .* data.x
    probability = 1.0 ./ (1.0 .+ exp.(-location))
    pointwise = logpdf.(BRM.BernoulliLogit.(location), data.y)
    expected_density =
        logpdf(Normal(), position[1]) +
        logpdf(Normal(), position[2]) + sum(pointwise)
    expected_gradient = [
        -position[1] + sum(data.y .- probability),
        -position[2] + sum(data.x .* (data.y .- probability)),
    ]

    density = BRM.NativePPL.logdensity!(workspace, prepared, position)
    @test density ≈ expected_density
    @test workspace.primal.location ≈ location
    @test workspace.primal.pointwise_loglikelihood ≈ pointwise
    gradient_density, gradient = BRM.NativePPL.logdensity_and_gradient!(
        workspace, prepared, position)
    @test gradient_density ≈ expected_density
    @test gradient ≈ expected_gradient
    @test steady_state_allocations(workspace, prepared, position) ==
          (; primal=0, gradient=0)

    likelihood = BRM._native_ppl_factor_logdensity!(
        plan.factors.likelihood, plan.inputs.response, plan.nodes.location,
        position, prepared, workspace.primal)
    @test likelihood ≈ sum(pointwise)
    mismatched_likelihood = BRM.NativePPLBernoulliLogitFactor(
        :wrong_response, :eta, plan.axes.observation)
    @test_throws MethodError BRM._native_ppl_factor_logdensity!(
        mismatched_likelihood, plan.inputs.response, plan.nodes.location,
        position, prepared, workspace.primal)

    location_output = BRM.NativePPL.allocate_output(prepared, linear_query)
    pointwise_output = BRM.NativePPL.allocate_output(prepared, pointwise_query)
    prediction_output = BRM.NativePPL.allocate_output(prepared, predictive_query)
    @test location_output isa Vector{Float64}
    @test pointwise_output isa Vector{Float64}
    @test prediction_output isa Vector{Bool}
    @test BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query) ===
          location_output
    @test location_output ≈ location
    @test BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query) ===
          pointwise_output
    @test pointwise_output ≈ pointwise
    prediction_a = BRM.NativePPL.simulate(
        MersenneTwister(71), workspace, prepared, position)
    prediction_b = BRM.NativePPL.simulate(
        MersenneTwister(71), workspace, prepared, position)
    @test prediction_a == prediction_b
    @test prediction_a isa Vector{Bool}
    @test_throws ArgumentError BRM.NativePPL.simulate!(
        MersenneTwister(71), zeros(length(data.y)), workspace, prepared,
        position)

    BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query)
    BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query)
    rng = MersenneTwister(72)
    BRM.NativePPL.simulate!(
        rng, prediction_output, workspace, prepared, position)
    @test @allocated(BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query)) == 0
    @test @allocated(BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query)) == 0
    rng = MersenneTwister(72)
    @test @allocated(BRM.NativePPL.simulate!(
        rng, prediction_output, workspace, prepared, position)) == 0
    predictive_allocation = @allocated(
        BRM.NativePPL.allocate_output(prepared, predictive_query))
    rng = MersenneTwister(72)
    predictive_wrapper_allocation = @allocated(
        BRM.NativePPL.simulate(rng, workspace, prepared, position))
    @test predictive_wrapper_allocation == predictive_allocation

    extreme_data = (; x=[-1.0, 1.0], y=Bool[false, true])
    extreme_plan = BRM.NativePPL.compile(@brm extreme_data begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end)
    extreme_prepared = BRM.NativePPL.prepare(extreme_plan)
    extreme_workspace = BRM.NativePPL.workspace(extreme_prepared)
    extreme_position = [0.0, 1000.0]
    extreme_pointwise = BRM.NativePPL.evaluate(
        extreme_workspace, extreme_prepared, extreme_position,
        pointwise_query)
    @test all(isfinite, extreme_pointwise)
    @test all(iszero, extreme_pointwise)
    @test BRM._native_ppl_softplus(1000.0) == 1000.0
    @test BRM._native_ppl_softplus(-1000.0) == 0.0
    @test BRM._native_ppl_logistic(1000.0) == 1.0
    @test BRM._native_ppl_logistic(-1000.0) == 0.0

    rebound = BRM.NativePPL.rebind(
        prepared, (; x=Float32[-1, 2], y=Int[1, 0]); T=Float32)
    @test BRM.NativePPL.has_response(rebound)
    @test eltype(rebound) === Float32
    @test rebound.response == Float32[1, 0]
    @test BRM.NativePPL.output_eltype(
        BRM.NativePPL.output_signature(rebound, predictive_query), rebound) === Bool
    @test BRM.NativePPL.logdensity!(
        BRM.NativePPL.workspace(rebound), rebound, Float32.(position)) isa Float32

    prediction_only = BRM.NativePPL.rebind(
        prepared, (; x=Float32[-1, 2]); T=Float32)
    @test !BRM.NativePPL.has_response(prediction_only)
    prediction_only_workspace = BRM.NativePPL.workspace(prediction_only)
    @test BRM.NativePPL.evaluate(
        prediction_only_workspace, prediction_only, Float32.(position),
        linear_query) isa Vector{Float32}
    @test BRM.NativePPL.simulate(
        MersenneTwister(73), prediction_only_workspace, prediction_only,
        Float32.(position)) isa Vector{Bool}
    err = argument_error(() -> BRM.NativePPL.logdensity!(
        prediction_only_workspace, prediction_only, Float32.(position)))
    @test occursin("requires an observed response binding", err.msg)

    @test_throws ArgumentError BRM.NativePPL.rebind(
        prepared, (; x=[-1.0, 2.0], y=[0, 2]))
    invalid_data = (; x=[-1.0, 2.0], y=[0, 2])
    invalid_brmi = @brm invalid_data begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end
    @test capability_error(() -> BRM.NativePPL.compile(invalid_brmi)).capability ==
          :response_support

    float_data = (; x=[-1.0, 2.0], y=[0.0, 1.0])
    @test BRM.NativePPL.compile(@brm float_data begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end) isa BRM.NativePPL.Plan
end


@testset "native Poisson-log workflow" begin
    data = (;
        x=[-1.0, 0.0, 1.0, 2.0],
        y=[0, 1, 4, 10],
    )
    brmi = @brm data begin
        eta ~ 1 + x
        y ~ Poisson(exp(eta))
    end
    plan = BRM.NativePPL.compile(brmi)

    @test LogDensityProblems.dimension(plan) == 2
    @test keys(plan.axes) == (:observation, :coefficient)
    @test keys(plan.parameters) == (:coefficients,)
    @test keys(plan.nodes) == (:transforms, :location, :rate)
    @test isempty(plan.nodes.transforms)
    @test keys(plan.factors) == (:coefficient_prior, :likelihood)
    @test plan.factors.likelihood isa BRM.NativePPLPoissonFactor
    @test plan.factors.likelihood.axis === plan.axes.observation
    @test plan.nodes.rate isa BRM.NativePPLExpNode
    @test BRM.native_node_name(plan.nodes.rate) == :exp_eta
    @test BRM.native_exp_input(plan.nodes.rate) == :eta
    @test plan.nodes.rate.axis === plan.axes.observation

    linear_query = BRM.NativePPL.LinearPredictor()
    pointwise_query = BRM.NativePPL.PointwiseLogLikelihood()
    predictive_query = BRM.NativePPL.PosteriorPredictive()
    predictive_signature = BRM.NativePPL.output_signature(plan, predictive_query)
    @test BRM.NativePPL.output_eltype(predictive_signature, Float32) === Int
    @test BRM.NativePPL.output_axis(predictive_signature) === plan.axes.observation

    prepared = BRM.NativePPL.prepare(plan)
    workspace = BRM.NativePPL.workspace(
        prepared, Float64, DI.AutoEnzyme())
    position = [0.2, 0.6]
    log_rate = position[1] .+ position[2] .* data.x
    rate = exp.(log_rate)
    pointwise = logpdf.(Poisson.(rate), data.y)
    expected_density =
        logpdf(Normal(), position[1]) +
        logpdf(Normal(), position[2]) + sum(pointwise)
    expected_gradient = [
        -position[1] + sum(data.y .- rate),
        -position[2] + sum(data.x .* (data.y .- rate)),
    ]

    density = BRM.NativePPL.logdensity!(workspace, prepared, position)
    @test density ≈ expected_density
    @test workspace.primal.location ≈ log_rate
    @test workspace.primal.pointwise_loglikelihood ≈ pointwise
    gradient_density, gradient = BRM.NativePPL.logdensity_and_gradient!(
        workspace, prepared, position)
    @test gradient_density ≈ expected_density
    @test gradient ≈ expected_gradient
    @test steady_state_allocations(workspace, prepared, position) ==
          (; primal=0, gradient=0)

    likelihood = BRM._native_ppl_factor_logdensity!(
        plan.factors.likelihood, plan.inputs.response, plan.nodes.rate,
        plan.nodes.location, position, prepared, workspace.primal)
    @test likelihood ≈ sum(pointwise)
    mismatched_likelihood = BRM.NativePPLPoissonFactor(
        :y, :wrong_rate, plan.axes.observation)
    @test_throws MethodError BRM._native_ppl_factor_logdensity!(
        mismatched_likelihood, plan.inputs.response, plan.nodes.rate,
        plan.nodes.location, position, prepared, workspace.primal)

    location_output = BRM.NativePPL.allocate_output(prepared, linear_query)
    pointwise_output = BRM.NativePPL.allocate_output(prepared, pointwise_query)
    prediction_output = BRM.NativePPL.allocate_output(prepared, predictive_query)
    @test location_output isa Vector{Float64}
    @test pointwise_output isa Vector{Float64}
    @test prediction_output isa Vector{Int}
    @test BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query) ===
          location_output
    @test location_output ≈ log_rate
    @test BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query) ===
          pointwise_output
    @test pointwise_output ≈ pointwise
    prediction_a = BRM.NativePPL.simulate(
        MersenneTwister(81), workspace, prepared, position)
    prediction_b = BRM.NativePPL.simulate(
        MersenneTwister(81), workspace, prepared, position)
    @test prediction_a == prediction_b
    @test prediction_a isa Vector{Int}
    @test all(>=(0), prediction_a)
    @test_throws ArgumentError BRM.NativePPL.simulate!(
        MersenneTwister(81), zeros(length(data.y)), workspace, prepared,
        position)

    BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query)
    BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query)
    rng = MersenneTwister(82)
    BRM.NativePPL.simulate!(
        rng, prediction_output, workspace, prepared, position)
    @test @allocated(BRM.NativePPL.evaluate!(
        location_output, workspace, prepared, position, linear_query)) == 0
    @test @allocated(BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position, pointwise_query)) == 0
    rng = MersenneTwister(82)
    @test @allocated(BRM.NativePPL.simulate!(
        rng, prediction_output, workspace, prepared, position)) == 0
    predictive_allocation = @allocated(
        BRM.NativePPL.allocate_output(prepared, predictive_query))
    rng = MersenneTwister(82)
    predictive_wrapper_allocation = @allocated(
        BRM.NativePPL.simulate(rng, workspace, prepared, position))
    @test predictive_wrapper_allocation == predictive_allocation

    large_count = Float32(1_000_000_000)
    large_log_rate = log(large_count)
    large_pointwise = BRM._native_ppl_poisson_logdensity(
        large_count, large_log_rate)
    large_reference =
        -Float32(0.5) * log(Float32(2π) * large_count) -
        BRM._native_ppl_stirling_correction(large_count)
    @test large_pointwise ≈ large_reference rtol=4eps(Float32)
    @test -20 < large_pointwise < -10
    @test BRM._native_ppl_poisson_logdensity(0.0, 1000.0) == -Inf
    @test BRM._native_ppl_poisson_logdensity(1.0, -Inf) == -Inf

    @test BRM._native_ppl_rand_poisson(
        MersenneTwister(83), Float64, -Inf) == 0
    @test_throws DomainError BRM._native_ppl_rand_poisson(
        MersenneTwister(83), Float64, NaN)
    @test_throws DomainError BRM._native_ppl_rand_poisson(
        MersenneTwister(83), Float64, Inf)
    @test_throws DomainError BRM._native_ppl_rand_poisson(
        MersenneTwister(83), Float32,
        log(maxintfloat(Float32) / Float32(2)))

    rng = MersenneTwister(84)
    draw_sum = 0
    draw_square_sum = 0
    draws = 20_000
    for _ in 1:draws
        draw = BRM._native_ppl_rand_poisson(rng, Float64, log(20.0))
        draw_sum += draw
        draw_square_sum += draw * draw
    end
    draw_mean = draw_sum / draws
    draw_variance =
        (draw_square_sum - draws * draw_mean * draw_mean) / (draws - 1)
    @test draw_mean ≈ 20 atol=0.2
    @test draw_variance ≈ 20 atol=1.0

    rebound = BRM.NativePPL.rebind(
        prepared, (; x=Float32[-1, 2], y=Float32[2, 3]); T=Float32)
    @test BRM.NativePPL.has_response(rebound)
    @test rebound.response == Float32[2, 3]
    @test BRM.NativePPL.output_eltype(
        BRM.NativePPL.output_signature(rebound, predictive_query), rebound) === Int
    @test BRM.NativePPL.logdensity!(
        BRM.NativePPL.workspace(rebound), rebound, Float32.(position)) isa Float32

    prediction_only = BRM.NativePPL.rebind(
        prepared, (; x=Float32[-1, 2]); T=Float32)
    @test !BRM.NativePPL.has_response(prediction_only)
    prediction_only_workspace = BRM.NativePPL.workspace(prediction_only)
    @test BRM.NativePPL.evaluate(
        prediction_only_workspace, prediction_only, Float32.(position),
        linear_query) isa Vector{Float32}
    @test BRM.NativePPL.simulate(
        MersenneTwister(85), prediction_only_workspace, prediction_only,
        Float32.(position)) isa Vector{Int}
    @test_throws ArgumentError BRM.NativePPL.logdensity!(
        prediction_only_workspace, prediction_only, Float32.(position))

    @test_throws ArgumentError BRM.NativePPL.rebind(
        prepared, (; x=[-1.0, 2.0], y=[0, -1]))
    imprecise_data = (; x=Float32[0.5], y=[16_777_217])
    imprecise_plan = BRM.NativePPL.compile(@brm imprecise_data begin
        eta ~ 1 + x
        y ~ Poisson(exp(eta))
    end)
    err = argument_error(() -> BRM.NativePPL.prepare(
        imprecise_plan; T=Float32))
    @test occursin("cannot be represented exactly as Float32", err.msg)

    invalid_data = (; x=[-1.0, 2.0], y=[0.0, 1.5])
    invalid_brmi = @brm invalid_data begin
        eta ~ 1 + x
        y ~ Poisson(exp(eta))
    end
    @test capability_error(() -> BRM.NativePPL.compile(invalid_brmi)).capability ==
          :response_support
    unlinked_data = (; x=[-1.0, 2.0], y=[0, 1])
    unlinked_brmi = @brm unlinked_data begin
        eta ~ 1 + x
        y ~ Poisson(eta)
    end
    @test capability_error(() -> BRM.NativePPL.compile(unlinked_brmi)).capability ==
          :likelihood_link
end


@testset "native PPL batched draw queries" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.5, 1.0, 2.5])
    plan = BRM.NativePPL.compile(@brm data begin
        sigma ~ Exponential(2.5)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    prepared = BRM.NativePPL.prepare(plan)
    workspace = BRM.NativePPL.workspace(prepared)
    positions = [
        0.1 -0.2 log(1.3)
        -0.4 0.7 log(0.8)
        1.0 0.0 log(2.0)
    ]
    linear_query = BRM.NativePPL.LinearPredictor()
    pointwise_query = BRM.NativePPL.PointwiseLogLikelihood()
    predictive_query = BRM.NativePPL.PosteriorPredictive()

    linear_signature = BRM.NativePPL.batch_output_signature(
        prepared, positions, linear_query)
    @test linear_signature isa BRM.NativePPL.BatchOutputSignature
    @test BRM.native_axis_name(
        BRM.NativePPL.output_draw_axis(linear_signature)) == :draw
    @test BRM.NativePPL.output_draw_axis(linear_signature).keys == Base.OneTo(3)
    @test BRM.NativePPL.output_axis(linear_signature) === plan.axes.observation
    @test BRM.NativePPL.output_axes(linear_signature) ==
          (BRM.NativePPL.output_draw_axis(linear_signature), plan.axes.observation)
    @test BRM.NativePPL.output_eltype(linear_signature, prepared) === Float64
    @test BRM.NativePPL.output_layout(linear_signature) isa
          BRM.NativePPL.DenseMatrixLayout

    expected_location = Matrix{Float64}(undef, 3, 3)
    expected_pointwise = similar(expected_location)
    for draw in axes(positions, 1)
        expected_location[draw, :] .=
            positions[draw, 1] .+ positions[draw, 2] .* prepared.predictor
        expected_pointwise[draw, :] .= logpdf.(
            Normal.(expected_location[draw, :], exp(positions[draw, 3])),
            prepared.response)
    end

    location_output = BRM.NativePPL.allocate_output(linear_signature, prepared)
    pointwise_signature = BRM.NativePPL.batch_output_signature(
        prepared, positions, pointwise_query)
    pointwise_output = BRM.NativePPL.allocate_output(
        pointwise_signature, prepared)
    @test location_output isa Matrix{Float64}
    @test size(location_output) == (3, 3)
    @test BRM.NativePPL.evaluate_draws!(
        location_output, workspace, prepared, positions, linear_query) ===
          location_output
    @test location_output ≈ expected_location
    @test BRM.NativePPL.evaluate_draws!(
        pointwise_output, workspace, prepared, positions, pointwise_query) ===
          pointwise_output
    @test pointwise_output ≈ expected_pointwise

    BRM.NativePPL.evaluate_draws!(
        location_output, workspace, prepared, positions, linear_query)
    BRM.NativePPL.evaluate_draws!(
        pointwise_output, workspace, prepared, positions, pointwise_query)
    @test @allocated(BRM.NativePPL.evaluate_draws!(
        location_output, workspace, prepared, positions, linear_query)) == 0
    @test @allocated(BRM.NativePPL.evaluate_draws!(
        pointwise_output, workspace, prepared, positions, pointwise_query)) == 0
    allocation = @allocated(
        BRM.NativePPL.allocate_output(linear_signature, prepared))
    BRM.NativePPL.evaluate_draws(workspace, prepared, positions, linear_query)
    wrapper_allocation = @allocated(
        BRM.NativePPL.evaluate_draws(
            workspace, prepared, positions, linear_query))
    @test wrapper_allocation == allocation

    predictive_signature = BRM.NativePPL.batch_output_signature(
        prepared, positions, predictive_query)
    prediction_output = BRM.NativePPL.allocate_output(
        predictive_signature, prepared)
    @test prediction_output isa Matrix{Float64}
    prediction_a = BRM.NativePPL.simulate_draws(
        MersenneTwister(91), workspace, prepared, positions)
    prediction_b = BRM.NativePPL.simulate_draws(
        MersenneTwister(91), workspace, prepared, positions)
    @test prediction_a == prediction_b
    manual_prediction = similar(prediction_a)
    manual_rng = MersenneTwister(91)
    for draw in axes(positions, 1)
        manual_prediction[draw, :] .= BRM.NativePPL.simulate(
            manual_rng, workspace, prepared, collect(positions[draw, :]))
    end
    @test prediction_a == manual_prediction
    rng = MersenneTwister(92)
    BRM.NativePPL.simulate_draws!(
        rng, prediction_output, workspace, prepared, positions)
    rng = MersenneTwister(92)
    @test @allocated(BRM.NativePPL.simulate_draws!(
        rng, prediction_output, workspace, prepared, positions)) == 0
    predictive_allocation = @allocated(
        BRM.NativePPL.allocate_output(predictive_signature, prepared))
    rng = MersenneTwister(92)
    BRM.NativePPL.simulate_draws(
        rng, workspace, prepared, positions)
    rng = MersenneTwister(92)
    predictive_wrapper_allocation = @allocated(
        BRM.NativePPL.simulate_draws(
            rng, workspace, prepared, positions))
    @test predictive_wrapper_allocation == predictive_allocation

    bernoulli_data = (; x=[-1.0, 0.5, 2.0], y=Bool[false, true, true])
    bernoulli_prepared = BRM.NativePPL.prepare(
        BRM.NativePPL.compile(@brm bernoulli_data begin
            eta ~ 1 + x
            y ~ BernoulliLogit(eta)
        end))
    bernoulli_workspace = BRM.NativePPL.workspace(bernoulli_prepared)
    binary_positions = [0.2 -0.4; -1.0 0.7]
    binary_signature = BRM.NativePPL.batch_output_signature(
        bernoulli_prepared, binary_positions, predictive_query)
    @test BRM.NativePPL.output_eltype(binary_signature, bernoulli_prepared) === Bool
    binary_prediction = BRM.NativePPL.simulate_draws(
        MersenneTwister(93), bernoulli_workspace, bernoulli_prepared,
        binary_positions)
    @test binary_prediction isa Matrix{Bool}
    @test binary_prediction == BRM.NativePPL.simulate_draws(
        MersenneTwister(93), bernoulli_workspace, bernoulli_prepared,
        binary_positions)

    poisson_data = (; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])
    poisson_prepared = BRM.NativePPL.prepare(
        BRM.NativePPL.compile(@brm poisson_data begin
            eta ~ 1 + x
            y ~ Poisson(exp(eta))
        end))
    poisson_workspace = BRM.NativePPL.workspace(poisson_prepared)
    poisson_signature = BRM.NativePPL.batch_output_signature(
        poisson_prepared, binary_positions, predictive_query)
    @test BRM.NativePPL.output_eltype(poisson_signature, poisson_prepared) === Int
    poisson_prediction = BRM.NativePPL.simulate_draws(
        MersenneTwister(94), poisson_workspace, poisson_prepared,
        binary_positions)
    @test poisson_prediction isa Matrix{Int}
    @test all(>=(0), poisson_prediction)
    @test poisson_prediction == BRM.NativePPL.simulate_draws(
        MersenneTwister(94), poisson_workspace, poisson_prepared,
        binary_positions)

    prediction_only = BRM.NativePPL.rebind(prepared, (; x=data.x))
    prediction_only_workspace = BRM.NativePPL.workspace(prediction_only)
    @test BRM.NativePPL.evaluate_draws(
        prediction_only_workspace, prediction_only, positions,
        linear_query) ≈ expected_location
    @test BRM.NativePPL.simulate_draws(
        MersenneTwister(95), prediction_only_workspace, prediction_only,
        positions) isa Matrix{Float64}
    unavailable = fill(123.0, size(expected_pointwise))
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        unavailable, prediction_only_workspace, prediction_only, positions,
        pointwise_query)
    @test all(==(123.0), unavailable)

    @test_throws DimensionMismatch BRM.NativePPL.batch_output_signature(
        prepared, zeros(2, 2), linear_query)
    @test_throws DimensionMismatch BRM.NativePPL.evaluate_draws!(
        zeros(2, 3), workspace, prepared, positions, linear_query)
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        zeros(Float32, 3, 3), workspace, prepared, positions, linear_query)
    output_view = @view zeros(3, 4)[:, 1:3]
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        output_view, workspace, prepared, positions, linear_query)
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        positions, workspace, prepared, positions, linear_query)
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        zeros(3, 3), workspace, prepared, Float32.(positions), linear_query)

    alias_workspace = BRM.NativePPL.workspace(prepared)
    aliased_positions = reshape(alias_workspace.primal.location, 1, :)
    @test_throws ArgumentError BRM.NativePPL.evaluate_draws!(
        zeros(1, 3), alias_workspace, prepared, aliased_positions,
        linear_query)

    sliced_workspace = BRM.NativePPL.workspace(prepared)
    pop!(sliced_workspace.gradient)
    @test BRM.NativePPL.evaluate_draws!(
        zeros(3, 3), sliced_workspace, prepared, positions, linear_query) isa
          Matrix{Float64}
    @test_throws DimensionMismatch BRM.NativePPL.evaluate_draws!(
        zeros(3, 3), sliced_workspace, prepared, positions, pointwise_query)

    empty_positions = zeros(0, 3)
    empty_signature = BRM.NativePPL.batch_output_signature(
        prepared, empty_positions, linear_query)
    @test BRM.NativePPL.output_draw_axis(empty_signature).keys == Base.OneTo(0)
    @test size(BRM.NativePPL.evaluate_draws(
        workspace, prepared, empty_positions, linear_query)) == (0, 3)
end


@testset "native PPL fused query bundles" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.5, 1.0, 2.5])
    plan = BRM.NativePPL.compile(@brm data begin
        sigma ~ Exponential(2.5)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)
    prepared = BRM.NativePPL.prepare(plan)
    workspace = BRM.NativePPL.workspace(prepared)
    positions = [0.1 -0.2 log(1.3); -0.4 0.7 log(0.8)]
    linear_query = BRM.NativePPL.LinearPredictor()
    pointwise_query = BRM.NativePPL.PointwiseLogLikelihood()
    predictive_query = BRM.NativePPL.PosteriorPredictive()

    deterministic_queries = (;
        location=linear_query,
        pointwise=pointwise_query,
    )
    signatures = BRM.NativePPL.batch_output_signature(
        prepared, positions, deterministic_queries)
    @test keys(signatures) == keys(deterministic_queries)
    @test all(
        signature isa BRM.NativePPL.BatchOutputSignature
        for signature in signatures)
    @test all(
        BRM.NativePPL.output_axes(signature) ==
            (BRM.NativePPL.output_draw_axis(signature), plan.axes.observation)
        for signature in signatures)
    outputs = BRM.NativePPL.allocate_output(signatures, prepared)
    @test keys(outputs) == keys(deterministic_queries)
    @test outputs.location isa Matrix{Float64}
    @test outputs.pointwise isa Matrix{Float64}
    @test !Base.mightalias(outputs.location, outputs.pointwise)

    expected_location = BRM.NativePPL.evaluate_draws(
        workspace, prepared, positions, linear_query)
    expected_pointwise = BRM.NativePPL.evaluate_draws(
        workspace, prepared, positions, pointwise_query)
    @test BRM.NativePPL.execute_draws!(
        outputs, workspace, prepared, positions, deterministic_queries) === outputs
    @test outputs.location ≈ expected_location
    @test outputs.pointwise ≈ expected_pointwise
    BRM.NativePPL.execute_draws!(
        outputs, workspace, prepared, positions, deterministic_queries)
    @test bundle_execution_allocated(
        outputs, workspace, prepared, positions, deterministic_queries) == 0

    allocation = @allocated(BRM.NativePPL.allocate_output(signatures, prepared))
    BRM.NativePPL.execute_draws(
        workspace, prepared, positions, deterministic_queries)
    wrapper_allocation = @allocated(BRM.NativePPL.execute_draws(
        workspace, prepared, positions, deterministic_queries))
    @test wrapper_allocation == allocation

    mixed_queries = (;
        location=linear_query,
        prediction=predictive_query,
        pointwise=pointwise_query,
    )
    mixed_signatures = BRM.NativePPL.batch_output_signature(
        prepared, positions, mixed_queries)
    mixed_outputs = BRM.NativePPL.allocate_output(mixed_signatures, prepared)
    fill!(mixed_outputs.location, -999)
    fill!(mixed_outputs.prediction, -999)
    fill!(mixed_outputs.pointwise, -999)
    @test_throws ArgumentError BRM.NativePPL.execute_draws!(
        mixed_outputs, workspace, prepared, positions, mixed_queries)
    @test all(==(-999), mixed_outputs.location)
    @test all(==(-999), mixed_outputs.prediction)
    @test all(==(-999), mixed_outputs.pointwise)

    mixed_rng = MersenneTwister(101)
    @test BRM.NativePPL.execute_draws!(
        mixed_rng, mixed_outputs, workspace, prepared, positions,
        mixed_queries) === mixed_outputs
    @test mixed_outputs.location ≈ expected_location
    @test mixed_outputs.pointwise ≈ expected_pointwise
    @test mixed_outputs.prediction == BRM.NativePPL.simulate_draws(
        MersenneTwister(101), workspace, prepared, positions)
    mixed_rng = MersenneTwister(102)
    BRM.NativePPL.execute_draws!(
        mixed_rng, mixed_outputs, workspace, prepared, positions,
        mixed_queries)
    mixed_rng = MersenneTwister(102)
    @test bundle_execution_allocated(
        mixed_rng, mixed_outputs, workspace, prepared, positions,
        mixed_queries) == 0

    mixed_allocation = @allocated(
        BRM.NativePPL.allocate_output(mixed_signatures, prepared))
    mixed_rng = MersenneTwister(102)
    BRM.NativePPL.execute_draws(
        mixed_rng, workspace, prepared, positions, mixed_queries)
    mixed_rng = MersenneTwister(102)
    mixed_wrapper_allocation = @allocated(BRM.NativePPL.execute_draws(
        mixed_rng, workspace, prepared, positions, mixed_queries))
    @test mixed_wrapper_allocation == mixed_allocation

    deterministic_rng = MersenneTwister(103)
    untouched_rng = MersenneTwister(103)
    BRM.NativePPL.execute_draws!(
        deterministic_rng, outputs, workspace, prepared, positions,
        deterministic_queries)
    @test rand(deterministic_rng) == rand(untouched_rng)

    prediction_only = BRM.NativePPL.rebind(prepared, (; x=data.x))
    prediction_only_workspace = BRM.NativePPL.workspace(prediction_only)
    prediction_queries = (; location=linear_query, prediction=predictive_query)
    prediction_signatures = BRM.NativePPL.batch_output_signature(
        prediction_only, positions, prediction_queries)
    prediction_outputs = BRM.NativePPL.allocate_output(
        prediction_signatures, prediction_only)
    @test BRM.NativePPL.execute_draws!(
        MersenneTwister(104), prediction_outputs,
        prediction_only_workspace, prediction_only, positions,
        prediction_queries) === prediction_outputs
    @test prediction_outputs.location ≈ expected_location

    unavailable_queries = (;
        location=linear_query,
        pointwise=pointwise_query,
        prediction=predictive_query,
    )
    unavailable_signatures = BRM.NativePPL.batch_output_signature(
        prediction_only, positions, unavailable_queries)
    unavailable_outputs = BRM.NativePPL.allocate_output(
        unavailable_signatures, prediction_only)
    for output in unavailable_outputs
        fill!(output, 7)
    end
    @test_throws ArgumentError BRM.NativePPL.execute_draws!(
        MersenneTwister(105), unavailable_outputs,
        prediction_only_workspace, prediction_only, positions,
        unavailable_queries)
    @test all(output -> all(==(7), output), unavailable_outputs)

    wrong_keys = (; other=zeros(2, 3), pointwise=zeros(2, 3))
    @test_throws ArgumentError BRM.NativePPL.execute_draws!(
        wrong_keys, workspace, prepared, positions, deterministic_queries)
    malformed_output = (; location=zeros(6), pointwise=zeros(2, 3))
    @test_throws ArgumentError BRM.NativePPL.execute_draws!(
        malformed_output, workspace, prepared, positions,
        deterministic_queries)
    malformed_queries = (; location=linear_query, pointwise=1)
    @test_throws ArgumentError BRM.NativePPL.batch_output_signature(
        prepared, positions, malformed_queries)
    aliased = zeros(2, 3)
    aliased_outputs = (; first=aliased, second=aliased)
    aliased_queries = (; first=linear_query, second=linear_query)
    @test_throws ArgumentError BRM.NativePPL.execute_draws!(
        aliased_outputs, workspace, prepared, positions, aliased_queries)

    sliced_workspace = BRM.NativePPL.workspace(prepared)
    empty!(sliced_workspace.primal.pointwise_loglikelihood)
    location_only = (; location=linear_query,)
    location_signature = BRM.NativePPL.batch_output_signature(
        prepared, positions, location_only)
    location_only_output = BRM.NativePPL.allocate_output(
        location_signature, prepared)
    @test BRM.NativePPL.execute_draws!(
        location_only_output, sliced_workspace, prepared, positions,
        location_only) === location_only_output
    @test_throws DimensionMismatch BRM.NativePPL.execute_draws!(
        outputs, sliced_workspace, prepared, positions, deterministic_queries)

    empty_queries = NamedTuple()
    empty_outputs = NamedTuple()
    @test BRM.NativePPL.batch_output_signature(
        prepared, positions, empty_queries) == empty_queries
    @test BRM.NativePPL.execute_draws!(
        empty_outputs, workspace, prepared, positions, empty_queries) ==
          empty_outputs
    @test_throws DimensionMismatch BRM.NativePPL.batch_output_signature(
        prepared, zeros(2, 2), empty_queries)
    @test_throws DimensionMismatch BRM.NativePPL.execute_draws!(
        empty_outputs, workspace, prepared, zeros(2, 2), empty_queries)

    bernoulli_data = (; x=[-1.0, 1.0], y=Bool[false, true])
    bernoulli_prepared = BRM.NativePPL.prepare(
        BRM.NativePPL.compile(@brm bernoulli_data begin
            eta ~ 1 + x
            y ~ BernoulliLogit(eta)
        end))
    poisson_data = (; x=[-1.0, 1.0], y=[0, 3])
    poisson_prepared = BRM.NativePPL.prepare(
        BRM.NativePPL.compile(@brm poisson_data begin
            eta ~ 1 + x
            y ~ Poisson(exp(eta))
        end))
    family_positions = [0.2 -0.3; 0.5 0.1]
    family_queries = (; location=linear_query, prediction=predictive_query)
    bernoulli_signatures = BRM.NativePPL.batch_output_signature(
        bernoulli_prepared, family_positions, family_queries)
    poisson_signatures = BRM.NativePPL.batch_output_signature(
        poisson_prepared, family_positions, family_queries)
    @test BRM.NativePPL.output_eltype(
        bernoulli_signatures.prediction, bernoulli_prepared) === Bool
    @test BRM.NativePPL.output_eltype(
        poisson_signatures.prediction, poisson_prepared) === Int
    @test BRM.NativePPL.allocate_output(
        bernoulli_signatures, bernoulli_prepared).prediction isa Matrix{Bool}
    @test BRM.NativePPL.allocate_output(
        poisson_signatures, poisson_prepared).prediction isa Matrix{Int}
end


@testset "native PPL fitted centering and replay" begin
    gaussian_data = (;
        x=[1.0, 2.0, 3.0, 4.0], y=[0.2, -0.1, 1.1, 0.7])
    gaussian_centered = BRM.NativePPL.compile(@brm gaussian_data begin
        sigma ~ Exponential(2.0)
        center_x ~ 1 + center(x)
        y ~ Normal(center_x, sigma)
    end)
    gaussian_explicit_data = (;
        xc=[-1.5, -0.5, 0.5, 1.5], y=gaussian_data.y)
    gaussian_explicit = BRM.NativePPL.compile(@brm gaussian_explicit_data begin
        sigma ~ Exponential(2.0)
        center_x ~ 1 + xc
        y ~ Normal(center_x, sigma)
    end)
    gaussian = check_transformed_execution(
        gaussian_centered, gaussian_explicit, [0.2, -0.4, log(0.8)])

    @test length(gaussian_centered.nodes.transforms) == 1
    transform = only(values(gaussian_centered.nodes.transforms))
    @test transform isa BRM.NativePPLCenterNode
    @test BRM.native_center_input(transform) === :x
    @test transform.mean == 2.5
    @test transform.axis === gaussian_centered.axes.observation
    @test BRM.native_node_name(transform) !==
          BRM.native_node_name(gaussian_centered.nodes.location)
    @test BRM.native_affine_input(gaussian_centered.nodes.location) ===
          BRM.native_node_name(transform)
    @test gaussian_centered.bindings.x === gaussian_data.x
    @test gaussian.predictor == gaussian_explicit_data.xc
    @test gaussian.predictor !== gaussian_data.x

    bernoulli_data = (;
        x=[1.0, 2.0, 3.0, 4.0], y=Bool[false, false, true, true])
    bernoulli_centered = BRM.NativePPL.compile(@brm bernoulli_data begin
        eta ~ 1 + center(x)
        y ~ BernoulliLogit(eta)
    end)
    bernoulli_explicit_data = (;
        xc=[-1.5, -0.5, 0.5, 1.5], y=bernoulli_data.y)
    bernoulli_explicit = BRM.NativePPL.compile(@brm bernoulli_explicit_data begin
        eta ~ 1 + xc
        y ~ BernoulliLogit(eta)
    end)
    bernoulli = check_transformed_execution(
        bernoulli_centered, bernoulli_explicit, [-0.3, 0.6])

    poisson_data = (;
        x=[1.0, 2.0, 3.0, 4.0], y=[0, 1, 3, 6])
    poisson_centered = BRM.NativePPL.compile(@brm poisson_data begin
        log_rate ~ 1 + center(x)
        y ~ Poisson(exp(log_rate))
    end)
    poisson_explicit_data = (;
        xc=[-1.5, -0.5, 0.5, 1.5], y=poisson_data.y)
    poisson_explicit = BRM.NativePPL.compile(@brm poisson_explicit_data begin
        log_rate ~ 1 + xc
        y ~ Poisson(exp(log_rate))
    end)
    poisson = check_transformed_execution(
        poisson_centered, poisson_explicit, [0.1, 0.2])

    new_x = [10.0, 14.0]
    frozen = BRM.NativePPL.rebind(
        gaussian, (; x=new_x, y=[0.4, 0.9]))
    refitted = BRM.NativePPL.rebind(
        gaussian, (; x=new_x, y=[0.4, 0.9]); freeze_constants=false)
    @test only(values(frozen.plan.nodes.transforms)).mean == 2.5
    @test frozen.predictor == [7.5, 11.5]
    @test only(values(refitted.plan.nodes.transforms)).mean == 12.0
    @test refitted.predictor == [-2.0, 2.0]
    @test frozen.plan.bindings.x === new_x
    @test only(values(frozen.plan.nodes.transforms)).axis ===
          frozen.plan.axes.observation
    @test frozen.plan.nodes.location.axis === frozen.plan.axes.observation
    @test frozen.plan.factors.likelihood.axis === frozen.plan.axes.observation

    position = [0.2, -0.4, log(0.8)]
    linear = BRM.NativePPL.LinearPredictor()
    @test BRM.NativePPL.evaluate(
        BRM.NativePPL.workspace(frozen), frozen, position, linear) ≈
          0.2 .- 0.4 .* [7.5, 11.5]
    @test BRM.NativePPL.evaluate(
        BRM.NativePPL.workspace(refitted), refitted, position, linear) ≈
          0.2 .- 0.4 .* [-2.0, 2.0]

    for (prepared, family_position) in
        ((gaussian, [0.2, -0.4, log(0.8)]),
         (bernoulli, [-0.3, 0.6]),
         (poisson, [0.1, 0.2]))
        prediction_only = BRM.NativePPL.rebind(prepared, (; x=new_x))
        refit_prediction_only = BRM.NativePPL.rebind(
            prepared, (; x=new_x); freeze_constants=false)
        @test !BRM.NativePPL.has_response(prediction_only)
        @test prediction_only.predictor == [7.5, 11.5]
        @test refit_prediction_only.predictor == [-2.0, 2.0]
        @test length(BRM.NativePPL.simulate(
            MersenneTwister(904), BRM.NativePPL.workspace(prediction_only),
            prediction_only, family_position)) == 2
        @test_throws ArgumentError BRM.NativePPL.evaluate(
            BRM.NativePPL.workspace(prediction_only), prediction_only,
            family_position, BRM.NativePPL.PointwiseLogLikelihood())
    end

    extreme = floatmax(Float64)
    extreme_data = (; x=[extreme, extreme], y=[0.0, 1.0])
    extreme_plan = BRM.NativePPL.compile(@brm extreme_data begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + center(x)
        y ~ Normal(mu, sigma)
    end)
    @test only(values(extreme_plan.nodes.transforms)).mean == extreme
    @test BRM.NativePPL.prepare(extreme_plan).predictor == [0.0, 0.0]

    @test_throws ArgumentError BRM.NativePPL.rebind(
        gaussian, (; x=[1, 2], y=[0.0, 1.0]))
    @test_throws BRM.NativePPLCapabilityError BRM.NativePPL.rebind(
        gaussian, (; x=[1.0, NaN], y=[0.0, 1.0]);
        freeze_constants=false)
end


@testset "native PPL fitted sample-SD scaling and replay" begin
    raw_x = [1.0, 2.0, 3.0, 4.0]
    fitted_mean = 2.5
    fitted_scale = sqrt(5 / 3)
    standardized_x = (raw_x .- fitted_mean) ./ fitted_scale

    gaussian_data = (; x=raw_x, y=[0.2, -0.1, 1.1, 0.7])
    gaussian_scaled = BRM.NativePPL.compile(@brm gaussian_data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + zscale(x)
        y ~ Normal(mu, sigma)
    end)
    gaussian_explicit_data = (; xz=standardized_x, y=gaussian_data.y)
    gaussian_explicit = BRM.NativePPL.compile(@brm gaussian_explicit_data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + xz
        y ~ Normal(mu, sigma)
    end)
    gaussian = check_transformed_execution(
        gaussian_scaled, gaussian_explicit, [0.2, -0.4, log(0.8)])

    gaussian_standardized = BRM.NativePPL.compile(@brm gaussian_data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + standardize(x)
        y ~ Normal(mu, sigma)
    end)
    transform = only(values(gaussian_scaled.nodes.transforms))
    alias_transform = only(values(gaussian_standardized.nodes.transforms))
    @test transform isa BRM.NativePPLZScaleNode
    @test typeof(alias_transform) === typeof(transform)
    @test BRM.native_zscale_input(transform) === :x
    @test transform.mean == fitted_mean
    @test transform.scale ≈ fitted_scale
    @test alias_transform.mean == transform.mean
    @test alias_transform.scale == transform.scale
    @test BRM.native_node_name(alias_transform) === BRM.native_node_name(transform)
    @test transform.axis === gaussian_scaled.axes.observation
    @test BRM.native_affine_input(gaussian_scaled.nodes.location) ===
          BRM.native_node_name(transform)
    @test gaussian.predictor ≈ standardized_x
    @test gaussian.predictor !== gaussian_data.x

    bernoulli_data = (; x=raw_x, y=Bool[false, false, true, true])
    bernoulli_scaled = BRM.NativePPL.compile(@brm bernoulli_data begin
        eta ~ 1 + zscale(x)
        y ~ BernoulliLogit(eta)
    end)
    bernoulli_explicit_data = (; xz=standardized_x, y=bernoulli_data.y)
    bernoulli_explicit = BRM.NativePPL.compile(@brm bernoulli_explicit_data begin
        eta ~ 1 + xz
        y ~ BernoulliLogit(eta)
    end)
    bernoulli = check_transformed_execution(
        bernoulli_scaled, bernoulli_explicit, [-0.3, 0.6])

    poisson_data = (; x=raw_x, y=[0, 1, 3, 6])
    poisson_scaled = BRM.NativePPL.compile(@brm poisson_data begin
        log_rate ~ 1 + standardize(x)
        y ~ Poisson(exp(log_rate))
    end)
    poisson_explicit_data = (; xz=standardized_x, y=poisson_data.y)
    poisson_explicit = BRM.NativePPL.compile(@brm poisson_explicit_data begin
        log_rate ~ 1 + xz
        y ~ Poisson(exp(log_rate))
    end)
    poisson = check_transformed_execution(
        poisson_scaled, poisson_explicit, [0.1, 0.2])

    new_x = [10.0, 14.0]
    frozen_x = (new_x .- fitted_mean) ./ fitted_scale
    refitted_x = [-inv(sqrt(2.0)), inv(sqrt(2.0))]
    for (prepared, response, position) in
        ((gaussian, [0.4, 0.9], [0.2, -0.4, log(0.8)]),
         (bernoulli, Bool[false, true], [-0.3, 0.6]),
         (poisson, [2, 4], [0.1, 0.2]))
        frozen = BRM.NativePPL.rebind(prepared, (; x=new_x, y=response))
        refitted = BRM.NativePPL.rebind(
            prepared, (; x=new_x, y=response); freeze_constants=false)
        @test frozen.predictor ≈ frozen_x
        @test refitted.predictor ≈ refitted_x
        @test frozen.plan.parameters === prepared.plan.parameters
        frozen_transform = only(values(frozen.plan.nodes.transforms))
        refitted_transform = only(values(refitted.plan.nodes.transforms))
        @test frozen_transform.mean == fitted_mean
        @test frozen_transform.scale ≈ fitted_scale
        @test refitted_transform.mean == 12.0
        @test refitted_transform.scale ≈ sqrt(8.0)
        @test frozen_transform.axis === frozen.plan.axes.observation
        @test frozen.plan.nodes.location.axis === frozen.plan.axes.observation
        @test frozen.plan.factors.likelihood.axis === frozen.plan.axes.observation

        prediction_only = BRM.NativePPL.rebind(prepared, (; x=new_x))
        refit_prediction_only = BRM.NativePPL.rebind(
            prepared, (; x=new_x); freeze_constants=false)
        @test !BRM.NativePPL.has_response(prediction_only)
        @test prediction_only.predictor ≈ frozen_x
        @test refit_prediction_only.predictor ≈ refitted_x
        @test length(BRM.NativePPL.simulate(
            MersenneTwister(905), BRM.NativePPL.workspace(prediction_only),
            prediction_only, position)) == 2
        @test_throws ArgumentError BRM.NativePPL.evaluate(
            BRM.NativePPL.workspace(prediction_only), prediction_only,
            position, BRM.NativePPL.PointwiseLogLikelihood())

        single_prediction = BRM.NativePPL.rebind(prepared, (; x=[10.0]))
        @test length(single_prediction.predictor) == 1
        @test_throws BRM.NativePPLCapabilityError BRM.NativePPL.rebind(
            prepared, (; x=[10.0]); freeze_constants=false)
        constant_prediction = BRM.NativePPL.rebind(prepared, (; x=[10.0, 10.0]))
        @test length(constant_prediction.predictor) == 2
        @test_throws BRM.NativePPLCapabilityError BRM.NativePPL.rebind(
            prepared, (; x=[10.0, 10.0]); freeze_constants=false)
    end

    gaussian_work = BRM.NativePPL.workspace(
        gaussian, Float64, DI.AutoEnzyme())
    @test steady_state_allocations(
        gaussian_work, gaussian, [0.2, -0.4, log(0.8)]) ==
          (; primal=0, gradient=0)

    function scaled_gaussian(values)
        data = (; x=values, y=zeros(length(values)))
        BRM.NativePPL.compile(@brm data begin
            sigma ~ Exponential(1.0)
            mu ~ 1 + zscale(x)
            y ~ Normal(mu, sigma)
        end)
    end
    @test capability_error(() -> scaled_gaussian([1.0])).capability ==
          :predictor_transform
    @test capability_error(() -> scaled_gaussian([1.0, 1.0])).capability ==
          :predictor_transform
    @test capability_error(() -> scaled_gaussian([1.0, NaN])).capability ==
          :predictor_transform
    @test capability_error(() -> scaled_gaussian([1.0, Inf])).capability ==
          :predictor_transform

    extreme = floatmax(Float64)
    extreme_plan = scaled_gaussian([-extreme, extreme, extreme, extreme])
    @test only(values(extreme_plan.nodes.transforms)).mean ≈ extreme / 2
    @test only(values(extreme_plan.nodes.transforms)).scale ≈ extreme
    @test BRM.NativePPL.prepare(extreme_plan).predictor ≈ [-1.5, 0.5, 0.5, 0.5]
    @test capability_error(() -> scaled_gaussian([-extreme, extreme])).capability ==
          :predictor_transform
    tiny_plan = scaled_gaussian([-1.0e-50, 1.0e-50])
    @test_throws ArgumentError BRM.NativePPL.prepare(tiny_plan; T=Float32)
    @test_throws ArgumentError BRM.NativePPL.rebind(
        gaussian, (; x=[1.0, NaN]))
    @test_throws BRM.NativePPLCapabilityError BRM.NativePPL.rebind(
        gaussian, (; x=[1.0, NaN]); freeze_constants=false)

    location = gaussian_scaled.nodes.location
    wrong_location = BRM.NativePPLAffineNode(
        BRM.native_node_name(location), :x, location.axis,
        location.intercept_index, only(location.slope_indices))
    bad_plan = replace_plan_nodes(
        gaussian_scaled, merge(
            gaussian_scaled.nodes, (; location=wrong_location)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    wrong_transform = BRM.NativePPLZScaleNode(
        BRM.native_node_name(transform), :wrong, transform.axis,
        transform.mean, transform.scale)
    bad_plan = replace_plan_transform(gaussian_scaled, wrong_transform)
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    equal_axis = BRM.NativePPLAxis(:observation, collect(eachindex(raw_x)))
    wrong_axis_transform = BRM.NativePPLZScaleNode(
        BRM.native_node_name(transform), :x, equal_axis,
        transform.mean, transform.scale)
    bad_plan = replace_plan_transform(gaussian_scaled, wrong_axis_transform)
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    rate = poisson_scaled.nodes.rate
    wrong_rate = BRM.NativePPLExpNode(
        BRM.native_node_name(rate), :wrong, rate.axis)
    bad_plan = replace_plan_nodes(
        poisson_scaled,
        merge(poisson_scaled.nodes, (; rate=wrong_rate)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    wrong_likelihood = BRM.NativePPLNormalFactor(
        :wrong_response, :mu, :sigma, gaussian_scaled.axes.observation)
    bad_plan = replace_plan_factors(
        gaussian_scaled,
        merge(gaussian_scaled.factors, (; likelihood=wrong_likelihood)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    wrong_prior = BRM.NativePPLStandardNormalFactor(:wrong_parameter, 1:2)
    bad_plan = replace_plan_factors(
        gaussian_scaled,
        merge(gaussian_scaled.factors, (; coefficient_prior=wrong_prior)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    bad_plan = replace_plan_queries(
        gaussian_scaled,
        merge(gaussian_scaled.queries, (; linear_predictor=1)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity

    wrong_query_axis = BRM.NativePPLAxis(
        :observation, collect(eachindex(raw_x)))
    wrong_output = BRM.NativePPLOutputSignature(
        wrong_query_axis, BRM.NativePPLPreparedElementType(),
        BRM.NativePPLDenseVectorLayout())
    wrong_query = BRM.NativePPLQuerySpec(
        :linear_predictor, :per_draw, :workspace,
        :until_next_evaluation, wrong_output)
    bad_plan = replace_plan_queries(
        gaussian_scaled,
        merge(gaussian_scaled.queries, (; linear_predictor=wrong_query)))
    @test capability_error(() -> BRM.NativePPL.prepare(bad_plan)).capability ==
          :graph_identity
end


@testset "public native PPL Model lowering and binding" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
    transforms = (:identity, :center, :zscale)
    families = (:gaussian, :bernoulli, :poisson)
    for family in families, transform in transforms
        response = family === :gaussian ? [0.2, -0.1, 1.1, 0.7] :
            family === :bernoulli ? Bool[false, false, true, true] :
            [0, 1, 3, 6]
        data = (; x=raw_x, y=response)
        position = family === :gaussian ? [0.2, -0.4, log(0.8)] :
            family === :bernoulli ? [-0.3, 0.6] : [0.1, 0.2]
        brmi = native_brmi(family, transform, data)
        direct_model = direct_native_model(family, transform)
        lowered_model = NP.lower(brmi)
        @test direct_model isa NP.Model
        @test typeof(direct_model) === typeof(lowered_model)
        @test sprint(show, direct_model) == sprint(show, lowered_model)

        direct_plan = NP.bind(conditioned(direct_model, data))
        keyword_plan = NP.bind(
            direct_model, (; x=data.x); conditions=(; y=data.y))
        compiled_keyword_plan = NP.compile(
            direct_model, (; x=data.x); conditions=(; y=data.y))
        compiled_direct_plan = NP.compile(conditioned(direct_model, data))
        lowered_plan = NP.bind(conditioned(lowered_model, data))
        brm_plan = NP.compile(brmi)
        compatibility_plan = BRM._native_ppl_plan(brmi)
        for candidate in (
            keyword_plan, compiled_keyword_plan, compiled_direct_plan,
            lowered_plan, brm_plan, compatibility_plan)
            check_plan_structure(direct_plan, candidate)
        end
        @test direct_plan.bindings.x === data.x
        @test direct_plan.bindings.y === data.y
        prepared = check_transformed_execution(
            direct_plan, brm_plan, position)
        @test NP.prepare(conditioned(direct_model, data)).predictor ==
              prepared.predictor
        @test NP.prepare(
            direct_model, (; x=data.x); conditions=(; y=data.y)).predictor ==
              prepared.predictor
        @test steady_state_allocations(
            NP.workspace(prepared, Float64, DI.AutoEnzyme()),
            prepared, position) == (; primal=0, gradient=0)

        unconditioned = NP.prepare(NP.bind(direct_model, (; x=data.x)))
        @test !NP.has_response(unconditioned)
        @test length(NP.simulate(
            MersenneTwister(904), NP.workspace(unconditioned),
            unconditioned, position)) == length(data.x)

        new_x = [10.0, 14.0, 20.0]
        new_y = family === :gaussian ? [0.4, 0.9, 1.3] :
            family === :bernoulli ? Bool[false, true, true] : [2, 4, 7]
        direct_prepared = NP.prepare(direct_plan)
        brm_prepared = NP.prepare(brm_plan)
        for freeze_constants in (true, false)
            direct_rebound = NP.rebind(
                direct_prepared, (; x=new_x, y=new_y); freeze_constants)
            brm_rebound = NP.rebind(
                brm_prepared, (; x=new_x, y=new_y); freeze_constants)
            check_plan_structure(direct_rebound.plan, brm_rebound.plan)
            @test direct_rebound.predictor == brm_rebound.predictor
            direct_work = NP.workspace(
                direct_rebound, Float64, DI.AutoEnzyme())
            brm_work = NP.workspace(
                brm_rebound, Float64, DI.AutoEnzyme())
            @test NP.logdensity!(direct_work, direct_rebound, position) ≈
                  NP.logdensity!(brm_work, brm_rebound, position)
            direct_density, direct_gradient = NP.logdensity_and_gradient!(
                direct_work, direct_rebound, position)
            brm_density, brm_gradient = NP.logdensity_and_gradient!(
                brm_work, brm_rebound, position)
            @test direct_density ≈ brm_density
            @test direct_gradient ≈ brm_gradient

            direct_prediction = NP.rebind(
                direct_prepared, (; x=new_x); freeze_constants)
            brm_prediction = NP.rebind(
                brm_prepared, (; x=new_x); freeze_constants)
            @test !NP.has_response(direct_prediction)
            @test direct_prediction.predictor == brm_prediction.predictor
            @test NP.simulate(
                MersenneTwister(906), NP.workspace(direct_prediction),
                direct_prediction, position) ==
                  NP.simulate(
                      MersenneTwister(906), NP.workspace(brm_prediction),
                      brm_prediction, position)
        end
    end

    gaussian_model = direct_native_model(:gaussian, :identity)
    gaussian_data = (; x=[1.0, 2.0], y=[0.0, 1.0])
    open_gaussian = gaussian_model
    @test NP.input() isa NP.Input{:value}
    open_instance = NP.substitute(open_gaussian; x=gaussian_data.x)
    @test keys(open_instance.bindings) == (:x,)
    @test isempty(open_instance.conditions)
    conditioned_instance = NP.condition(open_instance; y=gaussian_data.y)
    @test conditioned_instance.bindings.x === gaussian_data.x
    @test conditioned_instance.conditions.y === gaussian_data.y
    @test occursin("conditions=(:y,)", sprint(show, conditioned_instance))
    @test NP.substitute(conditioned_instance; x=[3.0, 4.0]).bindings.x ==
          [3.0, 4.0]
    @test_throws ArgumentError NP.substitute(open_gaussian; unknown=[1.0])
    @test_throws ArgumentError NP.condition(open_gaussian; unknown=[1.0])
    @test_throws ArgumentError NP.substitute(
        open_gaussian, Dict(:x => gaussian_data.x))
    @test_throws ArgumentError NP.condition(
        open_gaussian, Dict(:y => gaussian_data.y))
    malformed_instance = NP.ModelInstance(
        open_gaussian, (; x=gaussian_data.x), Dict(:y => gaussian_data.y))
    @test_throws ArgumentError NP.substitute(
        malformed_instance; x=gaussian_data.x)
    @test_throws ArgumentError NP.condition(
        malformed_instance; y=gaussian_data.y)
    @test_throws ArgumentError NP.bind(gaussian_model, gaussian_data.x)
    @test !NP.has_response(NP.prepare(NP.bind(
        gaussian_model, (; x=gaussian_data.x))))
    @test_throws ArgumentError NP.bind(
        gaussian_model, merge(gaussian_data, (; extra=[1.0, 2.0])))
    @test NP.instantiate(
        gaussian_model, (; x=gaussian_data.x)) isa NP.ModelInstance
    bad_instance = NP.ModelInstance(
        gaussian_model, (;))
    @test_throws ArgumentError NP.bind(bad_instance)

    malformed_model = NP.Model(
        [NP.input(:predictor)], (;), (;), (;))
    @test_throws ArgumentError NP.bind(malformed_model, (;))
    @test_throws ArgumentError NP.input(:unknown)
    @test_throws ArgumentError NP.Exponential(0.0)
    @test_throws ArgumentError NP.parameter(
        NP.RealSupport(), ();
        transform=NP.Identity(), prior=NP.StandardNormal())
    @test_throws ArgumentError NP.parameter(
        NP.RealSupport(), (:x, :x);
        transform=NP.Identity(), prior=NP.StandardNormal())

    extra_nodes = merge(
        gaussian_model.nodes, (; exp_mu=NP.exp_link(:mu)))
    extra_model = NP.model(
        inputs=gaussian_model.inputs,
        parameters=gaussian_model.parameters,
        nodes=extra_nodes,
        observations=gaussian_model.observations)
    @test capability_error(
        () -> NP.bind(conditioned(extra_model, gaussian_data))).capability ==
          :additional_nodes

    bad_coefficients = NP.parameter(
        NP.RealSupport(), (:intercept,);
        transform=NP.Identity(), prior=NP.StandardNormal())
    bad_coefficient_model = NP.model(
        inputs=gaussian_model.inputs,
        parameters=(; beta_mu=bad_coefficients,
                    sigma=gaussian_model.parameters.sigma),
        nodes=gaussian_model.nodes,
        observations=gaussian_model.observations)
    @test capability_error(
        () -> NP.bind(conditioned(bad_coefficient_model, gaussian_data))).capability ==
          :parameter_axis

    bad_scale = NP.parameter(
        NP.PositiveSupport(), (:residual,);
        transform=NP.Exp(), prior=NP.Exponential(2.0))
    bad_scale_model = NP.model(
        inputs=gaussian_model.inputs,
        parameters=(; beta_mu=gaussian_model.parameters.beta_mu,
                    sigma=bad_scale),
        nodes=gaussian_model.nodes,
        observations=gaussian_model.observations)
    @test capability_error(
        () -> NP.bind(conditioned(bad_scale_model, gaussian_data))).capability ==
          :parameter_axis
    @test capability_error(
        () -> NP.bind(NP.condition(
            NP.substitute(gaussian_model; x=[1, 2]); y=[0.0, 1.0]))).capability ==
          :predictor_type
    @test capability_error(
        () -> NP.bind(
            NP.condition(NP.substitute(
                direct_native_model(:bernoulli, :identity);
                x=[1.0, 2.0]); y=[0, 2]))).capability ==
          :response_support
    @test capability_error(
        () -> NP.bind(
            NP.condition(NP.substitute(
                direct_native_model(:poisson, :identity);
                x=[1.0, 2.0]); y=[0, 1.5]))).capability ==
          :response_support
end


@testset "public multi-predictor affine Model and executor" begin
    data = (
        x=[-2.0, 0.0, 1.0, 5.0],
        w=[1.0, 2.0, 4.0, 8.0],
        y=[0.2, -0.1, 1.1, 0.7],
    )
    declaration = direct_multi_gaussian_model()
    instance = NP.condition(
        NP.substitute(declaration; x=data.x, w=data.w); y=data.y)
    plan = NP.compile(instance)
    @test keys(plan.inputs.predictors) == (:x, :w)
    @test keys(plan.nodes.transforms) ==
          (:zscale_x_for_mu, :center_w_for_mu)
    @test BRM.native_affine_inputs(plan.nodes.location) ==
          (:zscale_x_for_mu, :center_w_for_mu)
    @test plan.nodes.location.slope_indices == (2, 3)
    @test plan.parameters.coefficients.unconstrained == 1:3
    @test plan.parameters.scale.unconstrained == 4:4

    prepared = NP.prepare(plan)
    @test keys(prepared.predictors) ==
          (:zscale_x_for_mu, :center_w_for_mu)
    expected_x = (data.x .- 1.0) ./ sqrt(26 / 3)
    expected_w = data.w .- sum(data.w) / length(data.w)
    @test prepared.predictors.zscale_x_for_mu ≈ expected_x
    @test prepared.predictors.center_w_for_mu ≈ expected_w
    @test !hasproperty(prepared, :predictor)
    @test_throws ArgumentError prepared.predictor

    position = [0.3, -0.4, 0.2, log(0.8)]
    expected_mu = position[1] .+ position[2] .* expected_x .+
        position[3] .* expected_w
    work = NP.workspace(prepared)
    @test NP.evaluate(work, prepared, position, NP.LinearPredictor()) ≈
          expected_mu
    expected_density = sum(logpdf.(Normal(), position[1:3])) +
        logpdf(Exponential(2.0), exp(position[4])) + position[4] +
        sum(logpdf.(Normal.(expected_mu, exp(position[4])), data.y))
    @test NP.logdensity!(work, prepared, position) ≈ expected_density

    prediction = NP.prepare(NP.substitute(
        declaration; x=data.x, w=data.w))
    @test !NP.has_response(prediction)
    @test length(NP.simulate(
        MersenneTwister(904), NP.workspace(prediction), prediction,
        position)) == length(data.x)

    rebound = NP.rebind(
        prepared, (; x=[10.0, 14.0], w=[3.0, 9.0]);
        freeze_constants=false)
    @test rebound.predictors.zscale_x_for_mu ≈
          [-1 / sqrt(2), 1 / sqrt(2)]
    @test rebound.predictors.center_w_for_mu == [-3.0, 3.0]

    macro_instance = NP.condition(
        macro_multi_gaussian(data.x, data.w); y=data.y)
    @test typeof(macro_instance.declaration) === typeof(declaration)
    macro_plan = NP.compile(macro_instance)
    check_plan_structure(plan, macro_plan)
    macro_prepared = NP.prepare(macro_plan)
    macro_work = NP.workspace(macro_prepared)
    @test NP.logdensity!(macro_work, macro_prepared, position) ≈
          expected_density
    brmi = @brm data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + zscale(x) + center(w)
        y ~ Normal(mu, sigma)
    end
    lowered = NP.lower(brmi)
    @test keys(lowered.inputs) == (:x, :w)
    @test keys(lowered.parameters) == (:beta_mu, :sigma)
    @test lowered.parameters.beta_mu.axis_keys == (:Intercept, :x, :w)
    @test keys(lowered.nodes) ==
          (:zscale_x_for_mu, :center_w_for_mu, :mu)
    @test lowered.nodes.mu ==
          NP.affine((:zscale_x_for_mu, :center_w_for_mu), :beta_mu)
    @test lowered.observations.y isa NP.BroadcastObservation

    brm_plan = NP.compile(brmi)
    @test keys(brm_plan.inputs.predictors) == (:x, :w)
    @test brm_plan.axes.coefficient.keys == (:Intercept, :x, :w)
    @test keys(brm_plan.nodes.transforms) ==
          (:zscale_x_for_mu, :center_w_for_mu)
    brm_prepared = NP.prepare(brm_plan)
    @test brm_prepared.predictors == macro_prepared.predictors
    @test NP.logdensity!(
        NP.workspace(brm_prepared), brm_prepared, position) ≈ expected_density

    reordered_brmi = @brm data begin
        sigma ~ Exponential(2.0)
        mu ~ 1 + center(w) + x
        y ~ Normal(mu, sigma)
    end
    reordered_lowered = NP.lower(reordered_brmi)
    @test reordered_lowered.parameters.beta_mu.axis_keys ==
          (:Intercept, :w, :x)
    @test reordered_lowered.nodes.mu ==
          NP.affine((:center_w_for_mu, :x), :beta_mu)
    reordered_brmi_prediction = NP.prepare(NP.substitute(
        reordered_lowered; x=data.x, w=data.w))
    @test !NP.has_response(reordered_brmi_prediction)
    @test keys(reordered_brmi_prediction.predictors) ==
          (:center_w_for_mu, :x)
    @test NP.evaluate(
        NP.workspace(reordered_brmi_prediction),
        reordered_brmi_prediction, [0.3, 0.2, -0.4, log(0.8)],
        NP.LinearPredictor()) ≈
          0.3 .+ 0.2 .* expected_w .- 0.4 .* data.x

    dotted_instance = NP.condition(
        macro_multi_gaussian_dotted(data.x, data.w); y=data.y)
    dotted_plan = NP.compile(dotted_instance)
    check_plan_structure(plan, dotted_plan)
    dotted_prepared = NP.prepare(dotted_plan)
    @test NP.evaluate(
        NP.workspace(dotted_prepared), dotted_prepared, position,
        NP.LinearPredictor()) ≈ expected_mu
    @test_throws ArgumentError NP.rebind(prepared, (; x=data.x))
    @test_throws ArgumentError NP.rebind(
        prepared, (; x=data.x, w=data.w, extra=data.x))
    @test_throws DimensionMismatch NP.rebind(
        prepared, (; x=data.x, w=data.w[1:3]))

    reordered_coefficients = NP.parameter(
        NP.RealSupport(), (:intercept, :beta_w, :beta_x);
        transform=NP.Identity(), prior=NP.StandardNormal())
    reordered_declaration = NP.model(
        inputs=declaration.inputs,
        parameters=(; beta_mu=reordered_coefficients,
                    sigma=declaration.parameters.sigma),
        nodes=(; w_centered=NP.center(:w),
               mu=NP.affine((:w_centered, :x), :beta_mu)),
        observations=declaration.observations)
    reordered = NP.prepare(NP.condition(
        NP.substitute(reordered_declaration; x=data.x, w=data.w); y=data.y))
    @test keys(reordered.predictors) == (:w_centered, :x)
    reordered_position = [0.3, 0.2, -0.4, log(0.8)]
    @test NP.evaluate(
        NP.workspace(reordered), reordered, reordered_position,
        NP.LinearPredictor()) ≈
          0.3 .+ 0.2 .* expected_w .- 0.4 .* data.x

    short_coefficients = NP.parameter(
        NP.RealSupport(), (:intercept, :beta_x);
        transform=NP.Identity(), prior=NP.StandardNormal())
    short_declaration = NP.model(
        inputs=declaration.inputs,
        parameters=(; beta_mu=short_coefficients,
                    sigma=declaration.parameters.sigma),
        nodes=declaration.nodes,
        observations=declaration.observations)
    @test capability_error(() -> NP.compile(NP.condition(
        NP.substitute(short_declaration; x=data.x, w=data.w); y=data.y))).capability ==
          :parameter_axis
    @test capability_error(() -> NP.compile(NP.condition(
        NP.substitute(declaration; x=data.x, w=data.w[1:3]); y=data.y))).capability ==
          :observation_axis

    location = plan.nodes.location
    coefficient_axis = BRM.NativePPLAxis(
        BRM.native_axis_name(plan.axes.coefficient),
        (:intercept, :beta_x, :beta_w, :duplicate))
    coefficients = BRM.NativePPLParameter(
        BRM.native_parameter_name(plan.parameters.coefficients),
        plan.parameters.coefficients.support,
        plan.parameters.coefficients.transform,
        coefficient_axis, 1:4)
    scale = BRM.NativePPLParameter(
        BRM.native_parameter_name(plan.parameters.scale),
        plan.parameters.scale.support,
        plan.parameters.scale.transform,
        plan.parameters.scale.axis, 5:5)
    duplicated_inputs = (
        :zscale_x_for_mu, :center_w_for_mu, :center_w_for_mu)
    duplicated_location = BRM.NativePPLAffineNode{
        BRM.native_node_name(location),duplicated_inputs,
        typeof(location.axis),Tuple{Int,Int,Int}}(
            location.axis, 1, (2, 3, 4))
    malformed_plan = BRM.NativePPLPlan(
        merge(plan.axes, (; coefficient=coefficient_axis)),
        plan.inputs, (; coefficients, scale),
        merge(plan.nodes, (; location=duplicated_location)),
        merge(plan.factors, (;
            coefficient_prior=BRM.NativePPLStandardNormalFactor(:beta_mu, 1:4),
            scale_prior=BRM.NativePPLExponentialFactor(:sigma, 5, 2.0))),
        plan.queries, plan.bindings)
    @test capability_error(() -> NP.prepare(malformed_plan)).capability ==
          :graph_identity
end


@testset "multi-predictor family workflows and replay" begin
    raw_x = [-2.0, 0.0, 1.0, 5.0]
    raw_w = [1.0, 2.0, 4.0, 8.0]
    for family in (:gaussian, :bernoulli, :poisson)
        response = family === :gaussian ? [0.2, -0.1, 1.1, 0.7] :
            family === :bernoulli ? Bool[false, false, true, true] :
            [0, 1, 3, 6]
        position = family === :gaussian ?
            [0.3, -0.4, 0.2, log(0.8)] : [0.3, -0.4, 0.2]
        data = (; x=raw_x, w=raw_w, y=response)
        direct_model = direct_multi_model(family)
        brmi = multi_native_brmi(family, data)
        lowered_model = NP.lower(brmi)
        @test typeof(direct_model) === typeof(lowered_model)
        @test sprint(show, direct_model) == sprint(show, lowered_model)

        direct_plan = NP.bind(NP.condition(
            NP.substitute(direct_model; x=data.x, w=data.w); y=data.y))
        brm_plan = NP.compile(brmi)
        check_plan_structure(direct_plan, brm_plan)
        prepared = check_transformed_execution(
            direct_plan, brm_plan, position)
        @test steady_state_allocations(
            NP.workspace(prepared, Float64, DI.AutoEnzyme()),
            prepared, position) == (; primal=0, gradient=0)

        macro_instance = family === :gaussian ?
            NP.condition(macro_multi_gaussian(data.x, data.w); y=data.y) :
            family === :bernoulli ?
            NP.condition(macro_multi_bernoulli(data.x, data.w); y=data.y) :
            NP.condition(macro_multi_poisson(data.x, data.w); y=data.y)
        macro_plan = NP.compile(macro_instance)
        macro_prepared = check_transformed_execution(
            macro_plan, brm_plan, position)
        @test steady_state_allocations(
            NP.workspace(macro_prepared, Float64, DI.AutoEnzyme()),
            macro_prepared, position) == (; primal=0, gradient=0)

        new_x = [10.0, 14.0, 20.0]
        new_w = [3.0, 9.0, 15.0]
        new_y = family === :gaussian ? [0.4, 0.9, 1.3] :
            family === :bernoulli ? Bool[false, true, true] : [2, 4, 7]
        for freeze_constants in (true, false)
            direct_rebound = NP.rebind(
                prepared, (; x=new_x, w=new_w, y=new_y); freeze_constants)
            macro_rebound = NP.rebind(
                macro_prepared, (; x=new_x, w=new_w, y=new_y);
                freeze_constants)
            @test direct_rebound.predictors == macro_rebound.predictors
            direct_work = NP.workspace(
                direct_rebound, Float64, DI.AutoEnzyme())
            macro_work = NP.workspace(
                macro_rebound, Float64, DI.AutoEnzyme())
            direct_density, direct_gradient = NP.logdensity_and_gradient!(
                direct_work, direct_rebound, position)
            macro_density, macro_gradient = NP.logdensity_and_gradient!(
                macro_work, macro_rebound, position)
            @test direct_density ≈ macro_density
            @test direct_gradient ≈ macro_gradient

            direct_prediction = NP.rebind(
                prepared, (; x=new_x, w=new_w); freeze_constants)
            macro_prediction = NP.rebind(
                macro_prepared, (; x=new_x, w=new_w); freeze_constants)
            @test !NP.has_response(direct_prediction)
            @test direct_prediction.predictors == macro_prediction.predictors
            @test NP.simulate(
                MersenneTwister(908), NP.workspace(direct_prediction),
                direct_prediction, position) == NP.simulate(
                    MersenneTwister(908), NP.workspace(macro_prediction),
                    macro_prediction, position)
        end

        @test_throws ArgumentError NP.rebind(prepared, (; x=new_x, y=new_y))
        @test_throws ArgumentError NP.rebind(
            prepared, (; x=new_x, w=new_w, y=new_y, extra=new_x))
        @test_throws DimensionMismatch NP.rebind(
            prepared, (; x=new_x, w=new_w[1:2], y=new_y))
        @test_throws ArgumentError NP.rebind(
            prepared, (; x=new_x, w=Int.(new_w), y=new_y))
    end
end


@testset "public namespaced component composition" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
    source = NP.component(:source, macro_gaussian_identity(raw_x))
    @test NP.component_namespace(source) === :source

    bound_input = NP.output(source, :x)
    parameter = NP.output(source, :beta_mu)
    deterministic = NP.output(source, :mu)
    stochastic = NP.output(source, :y)
    @test (NP.graph_namespace(bound_input), NP.graph_name(bound_input),
           NP.graph_kind(bound_input)) == (:source, :x, :binding)
    @test NP.graph_kind(parameter) === :parameter
    @test NP.graph_kind(deterministic) === :node
    @test NP.graph_kind(stochastic) === :site
    @test_throws ArgumentError NP.GraphRef{1,:mu,:node}()
    @test_throws ArgumentError NP.GraphRef{:source,1,:node}()
    @test_throws ArgumentError NP.GraphRef{:source,:mu,:unknown}()
    @test occursin(
        "source.mu, kind=node", sprint(show, deterministic))

    sink_instance = composable_gaussian(deterministic)
    @test sink_instance.bindings.x === deterministic
    sink = NP.component(:sink, sink_instance)
    composition = NP.compose(source, sink)
    @test keys(composition.components) == (:source, :sink)
    @test composition.components.source === source
    @test composition.components.sink === sink
    @test NP.Composition((; source, sink)).components ==
          composition.components
    @test occursin(
        "components=(:source, :sink)", sprint(show, composition))

    # A constant and a graph reference use the same substitution map. The
    # difference is the connected value, not a distinct data/pinning role.
    @test source.instance.bindings.x === raw_x
    @test sink.instance.bindings.x === deterministic

    parameter_sink = NP.component(
        :parameter_sink, composable_gaussian(parameter))
    stochastic_sink = NP.component(
        :stochastic_sink, composable_gaussian(stochastic))
    parameter_composition = NP.compose(source, parameter_sink)
    stochastic_composition = NP.compose(source, stochastic_sink)
    @test parameter_composition.components.parameter_sink.instance.
          bindings.x === parameter
    @test stochastic_composition.components.stochastic_sink.instance.
          bindings.x === stochastic
    site_error = capability_error(() -> NP.lower(stochastic_composition))
    @test site_error.capability == :active_site_connection
    @test occursin("source.y", site_error.detail)

    active_lowered = NP.lower(composition)
    sink_location_name = NP.qualified_name(:sink, :mu)
    @test !hasproperty(
        active_lowered.declaration.inputs,
        NP.qualified_name(:sink, :x))
    @test NP.node_inputs(
        getproperty(active_lowered.declaration.nodes, sink_location_name)) ==
          (NP.qualified_name(:source, :mu),)
    @test capability_error(() -> NP.compile(composition)).capability == :outcomes

    @test_throws ArgumentError NP.compose(sink, source)
    @test_throws ArgumentError NP.compose()
    @test_throws ArgumentError NP.compose(
        source, NP.component(:source, composable_gaussian(raw_x)))
    @test_throws ArgumentError NP.component(
        Symbol(""), composable_gaussian(raw_x))
    @test_throws ArgumentError NP.Component{1,typeof(source.instance)}(
        source.instance)
    @test_throws ArgumentError NP.Composition((source, sink))
    @test_throws ArgumentError NP.Composition((;))
    @test_throws ArgumentError NP.Composition((; sink, source))
    @test_throws ArgumentError NP.Composition((; not_a_component=1))
    @test_throws ArgumentError NP.output(source, :missing)
    open_component = NP.component(:open, direct_native_model(
        :gaussian, :identity))
    @test_throws ArgumentError NP.output(open_component, :x)

    bad_reference = NP.GraphRef{:source,:mu,:site}()
    bad_sink = NP.component(:bad_sink, composable_gaussian(bad_reference))
    @test_throws ArgumentError NP.compose(source, bad_sink)

    site_conditioned = NP.component(
        :site_conditioned,
        NP.condition(composable_gaussian(raw_x); y=stochastic))
    conditioned_composition = NP.compose(source, site_conditioned)
    @test conditioned_composition.components.site_conditioned.instance.
          conditions.y === stochastic

    preprocessing_instance = concise_zscale_component(raw_x)
    @test isempty(preprocessing_instance.declaration.observations)
    @test preprocessing_instance.declaration.outputs == (; scaled=:scaled)
    preprocessing = NP.component(:preprocessing, preprocessing_instance)
    scaled = NP.output(preprocessing, :scaled)
    response = [0.2, -0.1, 1.1, 0.7]
    regression = NP.component(
        :regression,
        NP.condition(composable_gaussian(scaled); y=response))
    executable = NP.compose(preprocessing, regression)
    lowered = NP.lower(executable)

    raw_name = NP.qualified_name(:preprocessing, :raw)
    scaled_name = NP.qualified_name(:preprocessing, :scaled)
    coefficient_name = NP.qualified_name(:regression, :beta_mu)
    location_name = NP.qualified_name(:regression, :mu)
    response_name = NP.qualified_name(:regression, :y)
    @test keys(lowered.declaration.inputs) == (raw_name,)
    @test keys(lowered.declaration.parameters) ==
          (coefficient_name, NP.qualified_name(:regression, :sigma))
    @test keys(lowered.declaration.nodes) == (scaled_name, location_name)
    @test keys(lowered.declaration.observations) == (response_name,)
    @test keys(lowered.bindings) == (raw_name,)
    @test keys(lowered.conditions) == (response_name,)

    prepared = NP.prepare(executable)
    direct = NP.prepare(NP.condition(
        macro_gaussian_zscale(raw_x); y=response))
    @test prepared.predictor == direct.predictor
    position = [0.3, -0.4, log(0.8)]
    composed_work = NP.workspace(prepared, Float64, DI.AutoEnzyme())
    direct_work = NP.workspace(direct, Float64, DI.AutoEnzyme())
    composed_density, composed_gradient = NP.logdensity_and_gradient!(
        composed_work, prepared, position)
    direct_density, direct_gradient = NP.logdensity_and_gradient!(
        direct_work, direct, position)
    @test composed_density ≈ direct_density
    @test composed_gradient ≈ direct_gradient
    @test NP.evaluate(
        composed_work, prepared, position, NP.LinearPredictor()) ≈
          NP.evaluate(direct_work, direct, position, NP.LinearPredictor())
    @test NP.simulate(
        MersenneTwister(909), composed_work, prepared, position) ==
          NP.simulate(MersenneTwister(909), direct_work, direct, position)

    named_preprocessing = NP.component(
        :named_preprocessing, concise_named_preprocessing(raw_x))
    standardized = NP.output(named_preprocessing, :standardized)
    @test NP.graph_kind(standardized) === :node
    @test NP.graph_name(standardized) === :standardized
    @test_throws ArgumentError NP.output(named_preprocessing, :scaled)

    aliased_preprocessing = NP.component(
        :aliased_preprocessing, concise_aliased_zscale(raw_x))
    aliased_standardized = NP.output(
        aliased_preprocessing, :standardized)
    @test NP.graph_name(aliased_standardized) === :standardized
    @test NP.graph_kind(aliased_standardized) === :node
    @test_throws ArgumentError NP.output(aliased_preprocessing, :scaled)
    aliased_regression = NP.component(
        :aliased_regression,
        NP.condition(composable_gaussian(aliased_standardized); y=response))
    aliased_composed = NP.prepare(NP.compose(
        aliased_preprocessing, aliased_regression))
    @test only(values(aliased_composed.predictors)) == direct.predictor

    scalar_prior_component = NP.component(:prior, scalar_normal_prior())
    theta = NP.output(scalar_prior_component, :theta)
    scalar_likelihood_component = NP.component(
        :likelihood,
        NP.condition(scalar_normal_likelihood(theta); y=response))
    active_scalar = NP.compose(
        scalar_prior_component, scalar_likelihood_component)
    active_scalar_lowered = NP.lower(active_scalar)
    theta_name = NP.qualified_name(:prior, :theta)
    sigma_name = NP.qualified_name(:likelihood, :sigma)
    scalar_response_name = NP.qualified_name(:likelihood, :y)
    @test isempty(active_scalar_lowered.declaration.inputs)
    @test keys(active_scalar_lowered.declaration.parameters) ==
          (theta_name, sigma_name)
    @test NP.observation_dependencies(getproperty(
        active_scalar_lowered.declaration.observations,
        scalar_response_name)) == (theta_name, sigma_name)
    @test isempty(active_scalar_lowered.bindings)
    @test keys(active_scalar_lowered.conditions) == (scalar_response_name,)
    scalar_plan = NP.compile(active_scalar)
    @test isempty(scalar_plan.inputs.predictors)
    @test scalar_plan.nodes.location isa BRM.NativePPLScalarBroadcastNode
    @test BRM.native_node_name(scalar_plan.nodes.location) === theta_name
    @test BRM.native_scalar_parameter(scalar_plan.nodes.location) === theta_name
    @test LogDensityProblems.dimension(scalar_plan) == 2

    aliased_active_prior = NP.component(
        :aliased_active_prior, aliased_scalar_normal_prior())
    aliased_theta = NP.output(aliased_active_prior, :coefficient)
    aliased_active = NP.compose(
        aliased_active_prior,
        NP.component(
            :aliased_active_likelihood,
            NP.condition(scalar_normal_likelihood(aliased_theta); y=response)))
    aliased_active_plan = NP.compile(aliased_active)
    @test BRM.native_node_name(aliased_active_plan.nodes.location) ===
          NP.qualified_name(:aliased_active_prior, :theta)
    @test LogDensityProblems.dimension(aliased_active_plan) == 2

    scalar_prepared = NP.prepare(scalar_plan)
    scalar_position = [0.3, log(0.8)]
    scalar_workspace = NP.workspace(
        scalar_prepared, Float64, DI.AutoEnzyme())
    expected_location = fill(scalar_position[1], length(response))
    expected_density = logpdf(Normal(), scalar_position[1]) +
        logpdf(Exponential(2.0), exp(scalar_position[2])) +
        scalar_position[2] +
        sum(logpdf.(Normal.(expected_location, exp(scalar_position[2])), response))
    residuals = response .- scalar_position[1]
    expected_gradient = [
        -scalar_position[1] + sum(residuals) / exp(2 * scalar_position[2]),
        1 - exp(scalar_position[2]) / 2 - length(response) +
            sum(abs2, residuals) / exp(2 * scalar_position[2]),
    ]
    scalar_density, scalar_gradient = NP.logdensity_and_gradient!(
        scalar_workspace, scalar_prepared, scalar_position)
    @test scalar_density ≈ expected_density
    @test scalar_gradient ≈ expected_gradient
    @test NP.evaluate(
        scalar_workspace, scalar_prepared, scalar_position,
        NP.LinearPredictor()) == expected_location
    @test length(NP.simulate(
        MersenneTwister(911), scalar_workspace, scalar_prepared,
        scalar_position)) == length(response)
    @test steady_state_allocations(
        scalar_workspace, scalar_prepared, scalar_position) ==
          (; primal=0, gradient=0)

    rebound_response = [0.1, 0.4, 0.8]
    scalar_rebound = NP.rebind(
        scalar_prepared,
        NamedTuple{(scalar_response_name,)}((rebound_response,)))
    @test isempty(scalar_rebound.predictors)
    @test scalar_rebound.response == rebound_response
    @test length(scalar_rebound.plan.axes.observation) == 3
    scalar_prediction = NP.rebind(scalar_prepared, (;))
    @test !NP.has_response(scalar_prediction)
    @test eltype(scalar_prediction) === Float64
    @test length(scalar_prediction.plan.axes.observation) == length(response)
    scalar_prediction_workspace = NP.workspace(scalar_prediction)
    @test length(NP.simulate(
        MersenneTwister(912), scalar_prediction_workspace,
        scalar_prediction, scalar_position)) == length(response)
    @test_throws ArgumentError NP.evaluate(
        scalar_prediction_workspace, scalar_prediction, scalar_position,
        NP.PointwiseLogLikelihood())

    scalar_prepared32 = NP.prepare(scalar_plan; T=Float32)
    scalar_prediction32 = NP.rebind(scalar_prepared32, (;))
    @test eltype(scalar_prediction32) === Float32
    @test_throws DimensionMismatch NP.rebind(
        scalar_prepared,
        NamedTuple{(scalar_response_name,)}((Float64[],)))

    integer_likelihood = NP.component(
        :integer_likelihood,
        NP.condition(scalar_normal_likelihood(theta); y=[0, 1, 2, 3]))
    integer_prepared = NP.prepare(NP.compose(
        scalar_prior_component, integer_likelihood))
    @test eltype(integer_prepared.response) === Float64
    @test eltype(integer_prepared) === Float64
    @test NP.workspace(integer_prepared) isa NP.Workspace
    @test NP.LogDensityProblem(
        integer_prepared, DI.AutoEnzyme()) isa NP.LogDensityProblem

    monolithic_plan = NP.compile(NP.condition(
        monolithic_scalar_normal(); y=response))
    scalar_data = (; y=response)
    brm_scalar = @brm scalar_data begin
        sigma ~ Exponential(2.0)
        mu ~ 1
        y ~ Normal(mu, sigma)
    end
    lowered_brm_scalar = NP.lower(brm_scalar)
    @test isempty(lowered_brm_scalar.inputs)
    @test keys(lowered_brm_scalar.parameters) == (:mu, :sigma)
    @test isempty(lowered_brm_scalar.nodes)
    brm_scalar_plan = NP.compile(brm_scalar)
    for candidate_plan in (monolithic_plan, brm_scalar_plan)
        candidate = NP.prepare(candidate_plan)
        candidate_work = NP.workspace(
            candidate, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient = NP.logdensity_and_gradient!(
            candidate_work, candidate, scalar_position)
        @test candidate_density ≈ scalar_density
        @test candidate_gradient ≈ scalar_gradient
        @test NP.evaluate(
            candidate_work, candidate, scalar_position,
            NP.LinearPredictor()) == expected_location
        @test NP.simulate(
            MersenneTwister(912), candidate_work, candidate,
            scalar_position) == NP.simulate(
                MersenneTwister(912), scalar_workspace, scalar_prepared,
                scalar_position)
    end

    scalar_positions = [0.3 log(0.8); -0.2 log(1.1)]
    scalar_queries = (;
        location=NP.LinearPredictor(),
        pointwise=NP.PointwiseLogLikelihood(),
    )
    scalar_signatures = NP.batch_output_signature(
        scalar_prepared, scalar_positions, scalar_queries)
    scalar_outputs = NP.allocate_output(
        scalar_signatures, scalar_prepared)
    NP.execute_draws!(
        scalar_outputs, scalar_workspace, scalar_prepared,
        scalar_positions, scalar_queries)
    @test scalar_outputs.location[1, :] == expected_location
    @test scalar_outputs.pointwise[1, :] ≈
          logpdf.(Normal.(expected_location, 0.8), response)
    @test bundle_execution_allocated(
        scalar_outputs, scalar_workspace, scalar_prepared,
        scalar_positions, scalar_queries) == 0

    unconditioned_scalar = NP.compose(
        scalar_prior_component,
        NP.component(:unconditioned, scalar_normal_likelihood(theta)))
    @test capability_error(
        () -> NP.compile(unconditioned_scalar)).capability == :observation_axis
end


@testset "composed preprocessing family workflows" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
    raw_name = NP.qualified_name(:preprocessing, :raw)
    response_name = NP.qualified_name(:regression, :y)
    @test NP.qualified_name(:a, :b_c) != NP.qualified_name(:a_b, :c)
    @test NP.qualified_name(:a, Symbol("b#c")) !=
          NP.qualified_name(Symbol("a#b"), :c)
    @test_throws ArgumentError NP.qualified_name(Symbol(""), :raw)

    for family in (:gaussian, :bernoulli, :poisson)
        response = family === :gaussian ? [0.2, -0.1, 1.1, 0.7] :
            family === :bernoulli ? Bool[false, false, true, true] :
            [0, 1, 3, 6]
        position = family === :gaussian ? [0.2, -0.4, log(0.8)] :
            family === :bernoulli ? [-0.3, 0.6] : [0.1, 0.2]
        data = (; x=raw_x, y=response)
        composition = deterministic_preprocessing_composition(family, data)
        composed_plan = NP.compile(composition)
        builder_plan = NP.compile(deterministic_preprocessing_composition(
            family, data; concise=false))
        direct_plan = NP.bind(conditioned(
            direct_native_model(family, :zscale), data))
        check_plan_structure(composed_plan, builder_plan)
        composed = check_transformed_execution(
            composed_plan, direct_plan, position)
        direct = NP.prepare(direct_plan)
        @test composed.predictor == direct.predictor
        @test steady_state_allocations(
            NP.workspace(composed, Float64, DI.AutoEnzyme()),
            composed, position) == (; primal=0, gradient=0)

        new_x = [10.0, 14.0, 20.0]
        new_y = family === :gaussian ? [0.4, 0.9, 1.3] :
            family === :bernoulli ? Bool[false, true, true] : [2, 4, 7]
        composed_bindings = NamedTuple{(raw_name, response_name)}(
            (new_x, new_y))
        for freeze_constants in (true, false)
            composed_rebound = NP.rebind(
                composed, composed_bindings; freeze_constants)
            direct_rebound = NP.rebind(
                direct, (; x=new_x, y=new_y); freeze_constants)
            @test composed_rebound.predictor == direct_rebound.predictor
            composed_work = NP.workspace(
                composed_rebound, Float64, DI.AutoEnzyme())
            direct_work = NP.workspace(
                direct_rebound, Float64, DI.AutoEnzyme())
            composed_density, composed_gradient =
                NP.logdensity_and_gradient!(
                    composed_work, composed_rebound, position)
            direct_density, direct_gradient = NP.logdensity_and_gradient!(
                direct_work, direct_rebound, position)
            @test composed_density ≈ direct_density
            @test composed_gradient ≈ direct_gradient

            composed_prediction = NP.rebind(
                composed,
                NamedTuple{(raw_name,)}((new_x,)); freeze_constants)
            direct_prediction = NP.rebind(
                direct, (; x=new_x); freeze_constants)
            @test !NP.has_response(composed_prediction)
            @test composed_prediction.predictor == direct_prediction.predictor
            @test NP.simulate(
                MersenneTwister(910), NP.workspace(composed_prediction),
                composed_prediction, position) == NP.simulate(
                    MersenneTwister(910), NP.workspace(direct_prediction),
                    direct_prediction, position)
        end

        @test_throws ArgumentError NP.rebind(
            composed, NamedTuple{(response_name,)}((new_y,)))
        @test_throws DimensionMismatch NP.rebind(
            composed,
            NamedTuple{(raw_name, response_name)}((new_x[1:2], new_y)))
    end

    source = NP.component(:source, macro_gaussian_identity(raw_x))
    active_condition = NP.component(
        :active_condition,
        NP.condition(composable_gaussian(raw_x); y=NP.output(source, :y)))
    @test capability_error(
        () -> NP.lower(NP.compose(source, active_condition))).capability ==
          :active_condition

    empty_component = NP.model(
        inputs=(; x=NP.input()), observations=(;))
    @test capability_error(
        () -> NP.bind(empty_component, (; x=raw_x))).capability == :outcomes
end


@testset "composed constant and multi-input preprocessing" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
    raw_w = [4.0, 2.0, 3.0, 9.0]
    response = [0.2, -0.1, 1.1, 0.7]

    constant_source = NP.component(
        :constant_source, concise_passthrough(raw_x))
    constant_regression = NP.component(
        :constant_regression,
        NP.condition(
            composable_gaussian(NP.output(constant_source, :raw));
            y=response))
    constant_composed = NP.prepare(NP.compose(
        constant_source, constant_regression))
    constant_direct = NP.prepare(NP.condition(
        macro_gaussian_identity(raw_x); y=response))
    @test only(values(constant_composed.predictors)) ==
          constant_direct.predictor
    constant_position = [0.3, -0.4, log(0.8)]
    constant_work = NP.workspace(
        constant_composed, Float64, DI.AutoEnzyme())
    direct_work = NP.workspace(constant_direct, Float64, DI.AutoEnzyme())
    constant_density, constant_gradient = NP.logdensity_and_gradient!(
        constant_work, constant_composed, constant_position)
    direct_density, direct_gradient = NP.logdensity_and_gradient!(
        direct_work, constant_direct, constant_position)
    @test constant_density ≈ direct_density
    @test constant_gradient ≈ direct_gradient
    @test steady_state_allocations(
        constant_work, constant_composed, constant_position) ==
          (; primal=0, gradient=0)

    x_preprocessing_model = NP.model(
        inputs=(; raw=NP.input()),
        nodes=(; scaled=NP.zscale(:raw)), observations=(;))
    w_preprocessing_model = NP.model(
        inputs=(; raw=NP.input()),
        nodes=(; centered=NP.center(:raw)), observations=(;))
    x_preprocessing = NP.component(
        :x_preprocessing,
        NP.substitute(x_preprocessing_model; raw=raw_x))
    w_preprocessing = NP.component(
        :w_preprocessing,
        NP.substitute(w_preprocessing_model; raw=raw_w))
    stacked_regression = NP.component(
        :stacked_regression,
        NP.condition(
            NP.substitute(
                direct_multi_gaussian_model();
                x=NP.output(x_preprocessing, :scaled),
                w=NP.output(w_preprocessing, :centered));
            y=response))
    @test capability_error(
        () -> NP.prepare(NP.compose(
            x_preprocessing, w_preprocessing, stacked_regression))).capability ==
          :graph_identity
    coefficients = NP.parameter(
        NP.RealSupport(), (:intercept, :beta_x, :beta_w);
        transform=NP.Identity(), prior=NP.StandardNormal())
    sigma = NP.parameter(
        NP.PositiveSupport(), (:sigma,);
        transform=NP.Exp(), prior=NP.Exponential(2.0))
    regression_model = NP.model(
        inputs=(; x=NP.input(), w=NP.input()),
        parameters=(; beta_mu=coefficients, sigma),
        nodes=(; mu=NP.affine((:x, :w), :beta_mu)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :mu, :sigma))))
    regression = NP.component(
        :multi_regression,
        NP.condition(
            NP.substitute(
                regression_model;
                x=NP.output(x_preprocessing, :scaled),
                w=NP.output(w_preprocessing, :centered));
            y=response))
    composed = NP.prepare(NP.compose(
        x_preprocessing, w_preprocessing, regression))
    direct = NP.prepare(NP.condition(
        macro_multi_gaussian(raw_x, raw_w); y=response))
    @test Tuple(values(composed.predictors)) ==
          Tuple(values(direct.predictors))
    @test composed.plan.axes.coefficient.keys == Tuple(
        NP.qualified_name(:multi_regression, key)
        for key in (:intercept, :beta_x, :beta_w))
    position = [0.3, -0.4, 0.2, log(0.8)]
    composed_work = NP.workspace(composed, Float64, DI.AutoEnzyme())
    direct_work = NP.workspace(direct, Float64, DI.AutoEnzyme())
    composed_density, composed_gradient = NP.logdensity_and_gradient!(
        composed_work, composed, position)
    direct_density, direct_gradient = NP.logdensity_and_gradient!(
        direct_work, direct, position)
    @test composed_density ≈ direct_density
    @test composed_gradient ≈ direct_gradient
    @test NP.simulate(
        MersenneTwister(911), composed_work, composed, position) ==
          NP.simulate(MersenneTwister(911), direct_work, direct, position)
    @test steady_state_allocations(
        composed_work, composed, position) == (; primal=0, gradient=0)

    new_x = [10.0, 14.0, 20.0]
    new_w = [3.0, 9.0, 15.0]
    new_y = [0.4, 0.9, 1.3]
    x_name = NP.qualified_name(:x_preprocessing, :raw)
    w_name = NP.qualified_name(:w_preprocessing, :raw)
    y_name = NP.qualified_name(:multi_regression, :y)
    rebound = NP.rebind(
        composed,
        NamedTuple{(x_name, w_name, y_name)}((new_x, new_w, new_y));
        freeze_constants=false)
    direct_rebound = NP.rebind(
        direct, (; x=new_x, w=new_w, y=new_y);
        freeze_constants=false)
    @test Tuple(values(rebound.predictors)) ==
          Tuple(values(direct_rebound.predictors))
    rebound_work = NP.workspace(rebound, Float64, DI.AutoEnzyme())
    direct_rebound_work = NP.workspace(
        direct_rebound, Float64, DI.AutoEnzyme())
    rebound_density, rebound_gradient = NP.logdensity_and_gradient!(
        rebound_work, rebound, position)
    direct_rebound_density, direct_rebound_gradient =
        NP.logdensity_and_gradient!(
            direct_rebound_work, direct_rebound, position)
    @test rebound_density ≈ direct_rebound_density
    @test rebound_gradient ≈ direct_rebound_gradient
end


@testset "public native PPL @model semantics" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
    response = [0.2, -0.1, 1.1, 0.7]
    preprocessing = concise_zscale_component(raw_x)
    @test keys(preprocessing.declaration.inputs) == (:raw,)
    @test keys(preprocessing.declaration.nodes) == (:scaled,)
    @test isempty(preprocessing.declaration.parameters)
    @test isempty(preprocessing.declaration.observations)
    @test preprocessing.declaration.outputs == (; scaled=:scaled)
    @test occursin("outputs=(:scaled,)", sprint(show, preprocessing.declaration))

    named_preprocessing = concise_named_preprocessing(raw_x)
    @test named_preprocessing.declaration.outputs ==
          (; centered=:centered, standardized=:scaled)
    @test_throws ArgumentError unknown_output_component(raw_x)
    @test capability_error(() -> NP.bind(preprocessing)).capability == :outcomes
    passthrough = concise_passthrough(raw_x)
    @test passthrough.declaration.outputs == (; raw=:raw)
    @test NP.graph_kind(NP.output(
        NP.component(:passthrough, passthrough), :raw)) === :binding

    scalar_prior = scalar_normal_prior()
    @test isempty(scalar_prior.declaration.inputs)
    @test keys(scalar_prior.declaration.parameters) == (:theta,)
    @test scalar_prior.declaration.parameters.theta.axis_keys == (:theta,)
    @test scalar_prior.declaration.parameters.theta.prior isa NP.StandardNormal
    @test scalar_prior.declaration.outputs == (; theta=:theta)
    prior_component = NP.component(:prior, scalar_prior)
    @test NP.graph_kind(NP.output(prior_component, :theta)) === :parameter
    @test capability_error(() -> NP.bind(scalar_prior)).capability == :outcomes
    @test_throws ArgumentError NP.NormalPrior(NaN, 1.0)
    @test_throws ArgumentError NP.NormalPrior(0.0, 0.0)
    @test_throws ArgumentError NP.ExponentialPrior(0.0)

    latent_site = scalar_normal_site(0.25, 0.8)
    @test keys(latent_site.declaration.inputs) == (:mu, :tau)
    @test isempty(latent_site.declaration.parameters)
    @test isempty(latent_site.declaration.nodes)
    @test keys(latent_site.declaration.observations) == (:z,)
    @test latent_site.declaration.observations.z isa NP.NormalObservation
    @test !NP.is_broadcast_observation(
        latent_site.declaration.observations.z)
    @test NP.observation_dependencies(
        latent_site.declaration.observations.z) == (:mu, :tau)
    @test latent_site.declaration.outputs == (; z=:z)
    @test latent_site.bindings == (; mu=0.25, tau=0.8)
    @test isempty(latent_site.conditions)
    latent_component = NP.component(:latent, latent_site)
    latent_output = NP.output(latent_component, :z)
    @test NP.graph_kind(latent_output) === :site
    @test (NP.graph_namespace(latent_output), NP.graph_name(latent_output)) ==
          (:latent, :z)
    latent_sink = NP.component(
        :sink,
        NP.condition(
            scalar_normal_likelihood(latent_output);
            y=[0.2, -0.1, 1.1, 0.7]))
    latent_composition = NP.compose(latent_component, latent_sink)
    @test latent_composition.components.sink.instance.bindings.mu ===
          latent_output
    latent_lowered = NP.lower(latent_composition)
    latent_name = NP.qualified_name(:latent, :z)
    latent_response_name = NP.qualified_name(:sink, :y)
    latent_scale_name = NP.qualified_name(:sink, :sigma)
    @test keys(latent_lowered.declaration.observations) ==
          (latent_name, latent_response_name)
    @test NP.observation_dependencies(getproperty(
        latent_lowered.declaration.observations,
        latent_response_name)) == (latent_name, latent_scale_name)
    @test keys(latent_lowered.conditions) == (latent_response_name,)
    latent_plan = NP.compile(latent_composition)
    @test keys(latent_plan.parameters) == (:site, :scale)
    @test BRM.native_parameter_name(latent_plan.parameters.site) ===
          latent_name
    @test keys(latent_plan.factors) ==
          (:site_prior, :scale_prior, :likelihood)
    @test latent_plan.factors.site_prior isa
          BRM.NativePPLScalarNormalFactor
    @test BRM.native_scalar_parameter(latent_plan.nodes.location) ===
          latent_name
    @test LogDensityProblems.dimension(latent_plan) == 2

    latent_prepared = NP.prepare(latent_plan)
    latent_position = [0.3, log(0.8)]
    latent_workspace = NP.workspace(
        latent_prepared, Float64, DI.AutoEnzyme())
    latent_location = fill(latent_position[1], length(response))
    latent_residuals = response .- latent_position[1]
    latent_expected_density =
        logpdf(Normal(0.25, 0.8), latent_position[1]) +
        logpdf(Exponential(2.0), exp(latent_position[2])) +
        latent_position[2] +
        sum(logpdf.(
            Normal.(latent_location, exp(latent_position[2])), response))
    latent_expected_gradient = [
        -(latent_position[1] - 0.25) / 0.8^2 +
            sum(latent_residuals) / exp(2 * latent_position[2]),
        1 - exp(latent_position[2]) / 2 - length(response) +
            sum(abs2, latent_residuals) / exp(2 * latent_position[2]),
    ]
    latent_density, latent_gradient = NP.logdensity_and_gradient!(
        latent_workspace, latent_prepared, latent_position)
    @test latent_density ≈ latent_expected_density
    @test latent_gradient ≈ latent_expected_gradient
    @test NP.evaluate(
        latent_workspace, latent_prepared, latent_position,
        NP.LinearPredictor()) == latent_location
    latent_pointwise = NP.evaluate(
        latent_workspace, latent_prepared, latent_position,
        NP.PointwiseLogLikelihood())
    @test latent_density ≈
          logpdf(Normal(0.25, 0.8), latent_position[1]) +
          logpdf(Exponential(2.0), exp(latent_position[2])) +
          latent_position[2] + sum(latent_pointwise)
    @test steady_state_allocations(
        latent_workspace, latent_prepared, latent_position) ==
          (; primal=0, gradient=0)
    @test NP.LogDensityProblem(
        latent_prepared, DI.AutoEnzyme()) isa NP.LogDensityProblem

    monolithic_latent = NP.prepare(NP.condition(
        monolithic_latent_normal(0.25, 0.8); y=response))
    monolithic_latent_workspace = NP.workspace(
        monolithic_latent, Float64, DI.AutoEnzyme())
    monolithic_density, monolithic_gradient = NP.logdensity_and_gradient!(
        monolithic_latent_workspace, monolithic_latent, latent_position)
    @test monolithic_density ≈ latent_density
    @test monolithic_gradient ≈ latent_gradient
    @test NP.evaluate(
        monolithic_latent_workspace, monolithic_latent, latent_position,
        NP.LinearPredictor()) == latent_location
    @test NP.simulate(
        MersenneTwister(913), monolithic_latent_workspace,
        monolithic_latent, latent_position) == NP.simulate(
            MersenneTwister(913), latent_workspace,
            latent_prepared, latent_position)

    natural_latent = natural_latent_normal(0.25, 0.8)
    @test natural_latent isa NP.ModelInstance
    @test natural_latent.declaration.outputs == (; y=:y)
    @test keys(natural_latent.declaration.observations) ==
          (NP.qualified_name(:z, :z), :y)
    @test isempty(natural_latent.conditions)
    conditioned_natural_latent = NP.condition(natural_latent; y=response)
    @test keys(conditioned_natural_latent.conditions) == (:y,)
    natural_latent_prepared = NP.prepare(conditioned_natural_latent)
    natural_latent_workspace = NP.workspace(
        natural_latent_prepared, Float64, DI.AutoEnzyme())
    natural_density, natural_gradient = NP.logdensity_and_gradient!(
        natural_latent_workspace, natural_latent_prepared, latent_position)
    @test natural_density ≈ latent_density
    @test natural_gradient ≈ latent_gradient
    @test NP.evaluate(
        natural_latent_workspace, natural_latent_prepared, latent_position,
        NP.LinearPredictor()) == latent_location
    @test NP.evaluate(
        natural_latent_workspace, natural_latent_prepared, latent_position,
        NP.PointwiseLogLikelihood()) == latent_pointwise
    @test NP.simulate(
        MersenneTwister(913), natural_latent_workspace,
        natural_latent_prepared, latent_position) == NP.simulate(
            MersenneTwister(913), latent_workspace,
            latent_prepared, latent_position)
    natural_rebound_response = [0.1, 0.4, 0.8]
    natural_rebound = NP.rebind(
        natural_latent_prepared, (; y=natural_rebound_response))
    @test natural_rebound.response == natural_rebound_response
    @test length(natural_rebound.plan.axes.observation) == 3
    natural_prediction = NP.rebind(natural_latent_prepared, (;))
    @test !NP.has_response(natural_prediction)
    @test length(NP.simulate(
        MersenneTwister(914), NP.workspace(natural_prediction),
        natural_prediction, latent_position)) == length(response)

    qualified = qualified_nested(0.25, 0.8)
    @test qualified isa NP.ModelInstance
    @test qualified.declaration.outputs == (; z=:z)
    @test keys(NP.condition(qualified; z=0.4).conditions) == (:z,)

    natural_preprocessed = NP.condition(
        natural_preprocessed_normal(raw_x); y=response)
    explicit_preprocessing_component = NP.component(
        :scaled, concise_zscale_component(raw_x))
    explicit_preprocessing_output = NP.output(
        explicit_preprocessing_component, :scaled)
    explicit_preprocessed = NP.compose(
        explicit_preprocessing_component,
        NP.component(
            :y,
            NP.condition(
                composable_gaussian(explicit_preprocessing_output);
                y=response)))
    natural_preprocessed_plan = NP.compile(natural_preprocessed)
    explicit_preprocessed_plan = NP.compile(explicit_preprocessed)
    natural_preprocessed_prepared = NP.prepare(natural_preprocessed_plan)
    explicit_preprocessed_prepared = NP.prepare(explicit_preprocessed_plan)
    natural_preprocessed_workspace = NP.workspace(
        natural_preprocessed_prepared, Float64, DI.AutoEnzyme())
    explicit_preprocessed_workspace = NP.workspace(
        explicit_preprocessed_prepared, Float64, DI.AutoEnzyme())
    preprocessing_position = [0.3, -0.4, log(0.8)]
    natural_preprocessed_density, natural_preprocessed_gradient =
        NP.logdensity_and_gradient!(
            natural_preprocessed_workspace, natural_preprocessed_prepared,
            preprocessing_position)
    explicit_preprocessed_density, explicit_preprocessed_gradient =
        NP.logdensity_and_gradient!(
            explicit_preprocessed_workspace, explicit_preprocessed_prepared,
            preprocessing_position)
    @test natural_preprocessed_density ≈ explicit_preprocessed_density
    @test natural_preprocessed_gradient ≈ explicit_preprocessed_gradient
    @test NP.evaluate(
        natural_preprocessed_workspace, natural_preprocessed_prepared,
        preprocessing_position, NP.LinearPredictor()) == NP.evaluate(
            explicit_preprocessed_workspace, explicit_preprocessed_prepared,
            preprocessing_position, NP.LinearPredictor())
    @test steady_state_allocations(
        natural_preprocessed_workspace, natural_preprocessed_prepared,
        preprocessing_position) == (; primal=0, gradient=0)
    @test_throws ArgumentError assigned_stochastic_submodel()
    @test_throws ArgumentError sampled_deterministic_submodel(raw_x)
    @test_throws ArgumentError ambiguous_multioutput_submodel()

    hierarchy = hierarchical_latent_graph()
    @test hierarchy.declaration.site_order ==
          (:population, :population_scale, :individual,
           :observation_scale, :y)
    hierarchy_graph = NP.factor_graph(NP.condition(
        hierarchy; y=response))
    @test keys(hierarchy_graph.sites) == hierarchy.declaration.site_order
    @test hierarchy_graph.dimension == 4
    @test keys(hierarchy_graph.coordinates) ==
          (:population, :population_scale, :individual, :observation_scale)
    @test hierarchy_graph.coordinates.population.indices == 1:1
    @test hierarchy_graph.coordinates.population_scale.indices == 2:2
    @test hierarchy_graph.coordinates.individual.indices == 3:3
    @test hierarchy_graph.coordinates.observation_scale.indices == 4:4
    @test hierarchy_graph.sites.population.activity isa NP.FreeSite
    @test hierarchy_graph.sites.population.factor isa
          NP.StandardNormalSiteFactor
    @test hierarchy_graph.sites.population_scale.factor isa
          NP.ExponentialSiteFactor
    @test hierarchy_graph.sites.individual.factor isa NP.NormalSiteFactor
    @test NP.site_factor_dependencies(
        hierarchy_graph.sites.individual.factor) ==
          (:population, :population_scale)
    @test hierarchy_graph.sites.y.activity isa NP.ConditionedSite
    @test hierarchy_graph.sites.y.shape isa NP.BroadcastSiteShape
    @test NP.site_factor_dependencies(hierarchy_graph.sites.y.factor) ==
          (:individual, :observation_scale)

    deterministic_scale_instance = NP.condition(
        deterministic_scale_factor_graph(1.0); y=response)
    deterministic_scale_graph = NP.factor_graph(
        deterministic_scale_instance)
    @test keys(deterministic_scale_graph.sites) ==
          (:population, :log_observation_scale, :y)
    @test keys(deterministic_scale_graph.nodes) == (:observation_scale,)
    @test deterministic_scale_graph.schedule ==
          (:population, :log_observation_scale, :observation_scale, :y)
    @test deterministic_scale_graph.nodes.observation_scale isa
          NP.ExpFactorNode{NP.SiteValue{:log_observation_scale}}
    @test NP.factor_node_dependencies(
        deterministic_scale_graph.nodes.observation_scale) ==
          (:log_observation_scale,)
    @test deterministic_scale_graph.sites.y.factor.location isa
          NP.SiteValue{:population}
    @test deterministic_scale_graph.sites.y.factor.scale isa
          NP.NodeValue{:observation_scale}
    @test NP.site_factor_dependencies(
        deterministic_scale_graph.sites.y.factor) ==
          (:population, :observation_scale)
    @test deterministic_scale_graph.dimension == 2

    cyclic_factor_declaration = NP.model(
        inputs=(;),
        parameters=(;
            population=NP.parameter(
                NP.RealSupport(), (:population,);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        nodes=(; scale=NP.exp_link(:latent)),
        observations=(;
            latent=NP.normal(:latent, :population, :scale)),
        outputs=(; latent=:latent),
        site_order=(:population, :latent))
    @test capability_error(
        () -> NP.factor_graph(cyclic_factor_declaration)).capability ==
          :factor_schedule

    deterministic_scale_plan = NP.compile(deterministic_scale_instance)
    @test deterministic_scale_plan isa NP.FactorPlan
    @test deterministic_scale_plan.node_indices ==
          (; observation_scale=1)
    deterministic_scale_prepared = NP.prepare(deterministic_scale_plan)
    deterministic_scale_workspace = NP.workspace(
        deterministic_scale_prepared, Float64, DI.AutoEnzyme())
    deterministic_scale_position = [0.2, log(0.5)]
    deterministic_population = deterministic_scale_position[1]
    deterministic_log_scale = deterministic_scale_position[2]
    deterministic_observation_scale = exp(deterministic_log_scale)
    deterministic_scale_latent_residual =
        deterministic_log_scale - deterministic_population
    deterministic_scale_observation_residuals =
        response .- deterministic_population
    deterministic_scale_expected_density =
        logpdf(Normal(), deterministic_population) +
        logpdf(Normal(deterministic_population, 1.0),
               deterministic_log_scale) +
        sum(logpdf.(Normal(
            deterministic_population, deterministic_observation_scale),
            response))
    deterministic_scale_expected_gradient = [
        -deterministic_population + deterministic_scale_latent_residual +
            sum(deterministic_scale_observation_residuals) /
                deterministic_observation_scale^2,
        -deterministic_scale_latent_residual - length(response) +
            sum(abs2, deterministic_scale_observation_residuals) /
                deterministic_observation_scale^2,
    ]
    deterministic_scale_density, deterministic_scale_gradient =
        NP.logdensity_and_gradient!(
            deterministic_scale_workspace, deterministic_scale_prepared,
            deterministic_scale_position)
    @test deterministic_scale_density ≈
          deterministic_scale_expected_density
    @test deterministic_scale_gradient ≈
          deterministic_scale_expected_gradient
    @test deterministic_scale_workspace.primal.node_values ==
          [deterministic_observation_scale]
    deterministic_scale_linear = NP.evaluate(
        deterministic_scale_workspace, deterministic_scale_prepared,
        deterministic_scale_position, NP.LinearPredictor())
    @test deterministic_scale_linear ==
          fill(deterministic_population, length(response))
    deterministic_scale_pointwise = NP.evaluate(
        deterministic_scale_workspace, deterministic_scale_prepared,
        deterministic_scale_position, NP.PointwiseLogLikelihood())
    @test deterministic_scale_pointwise ≈ logpdf.(Normal(
        deterministic_population, deterministic_observation_scale), response)
    deterministic_scale_predictive_rng = MersenneTwister(931)
    deterministic_scale_expected_predictive_rng = MersenneTwister(931)
    @test NP.simulate(
        deterministic_scale_predictive_rng,
        deterministic_scale_workspace, deterministic_scale_prepared,
        deterministic_scale_position) == [
            deterministic_population + deterministic_observation_scale *
                randn(deterministic_scale_expected_predictive_rng)
            for _ in response
        ]
    deterministic_scale_prior_rng = MersenneTwister(932)
    deterministic_scale_expected_prior_rng = MersenneTwister(932)
    deterministic_prior_population = randn(
        deterministic_scale_expected_prior_rng)
    deterministic_prior_log_scale = deterministic_prior_population +
        randn(deterministic_scale_expected_prior_rng)
    deterministic_prior_scale = exp(deterministic_prior_log_scale)
    deterministic_prior_response = [
        deterministic_prior_population + deterministic_prior_scale *
            randn(deterministic_scale_expected_prior_rng)
        for _ in response
    ]
    deterministic_scale_prior = NP.simulate_prior(
        deterministic_scale_prior_rng, deterministic_scale_workspace,
        deterministic_scale_prepared)
    @test deterministic_scale_prior.position ==
          [deterministic_prior_population, deterministic_prior_log_scale]
    @test deterministic_scale_prior.response == deterministic_prior_response
    @test factor_steady_state_allocations(
        deterministic_scale_workspace, deterministic_scale_prepared,
        deterministic_scale_position) == (; primal=0, gradient=0)
    @test factor_query_allocations(
        deterministic_scale_workspace, deterministic_scale_prepared,
        deterministic_scale_position, similar(response), similar(response),
        similar(response), similar(deterministic_scale_position)) ==
          (; linear=0, pointwise=0, predictive=0, prior=0)
    deterministic_scale_rebound = NP.rebind(
        deterministic_scale_prepared, (; y=[0.1, 0.4, 0.8]))
    @test deterministic_scale_rebound.plan.graph.schedule ==
          deterministic_scale_graph.schedule
    @test deterministic_scale_rebound.plan.bindings ==
          deterministic_scale_plan.bindings

    distributional_z = [0.5, -1.0, 1.5, 0.25]
    distributional_declaration = distributional_gaussian_factor_graph()
    distributional_data = (;
        x=raw_x, z=distributional_z, y=response)
    distributional_brmi = @brm distributional_data begin
        mu ~ 1 + x
        log_sigma ~ 1 + z
        y ~ Normal(mu, exp(log_sigma))
    end
    @test popcoefnames(distributional_brmi, :mu) == [:Intercept, :x]
    @test popcoefnames(distributional_brmi, :log_sigma) ==
          [:Intercept, :z]
    @test SBBRMI(distributional_brmi; mod=@__MODULE__) isa SBBRMI
    distributional_lowered = NP.lower(distributional_brmi)
    @test distributional_lowered == distributional_declaration
    natural_distributional = natural_distributional_gaussian(
        raw_x, distributional_z)
    @test natural_distributional.declaration == distributional_declaration
    @test NP.condition(natural_distributional; y=response) ==
          NP.instantiate(
              distributional_declaration,
              (; x=raw_x, z=distributional_z);
              conditions=(; y=response))
    distributional_plan = NP.compile(
        distributional_declaration, (; x=raw_x, z=distributional_z);
        conditions=(; y=response))
    distributional_brm_plan = NP.compile(distributional_brmi)
    @test distributional_brm_plan.declaration ==
          distributional_plan.declaration
    @test distributional_brm_plan.bindings == distributional_plan.bindings
    @test distributional_brm_plan.conditions ==
          distributional_plan.conditions
    @test distributional_brm_plan.graph.schedule ==
          distributional_plan.graph.schedule
    @test distributional_plan isa NP.FactorPlan
    @test distributional_plan.graph.schedule ==
          (:beta_mu, :beta_log_sigma, :mu, :log_sigma, :sigma, :y)
    @test distributional_plan.graph.dimension == 4
    @test distributional_plan.graph.nodes.sigma isa NP.ExpFactorNode
    @test distributional_plan.graph.sites.y.factor.scale isa
          NP.NodeValue{:sigma}
    distributional_prepared = NP.prepare(distributional_plan)
    distributional_workspace = NP.workspace(
        distributional_prepared, Float64, DI.AutoEnzyme())
    distributional_position = [0.3, -0.4, -0.2, 0.25]
    distributional_mu =
        distributional_position[1] .+
        distributional_position[2] .* raw_x
    distributional_log_sigma =
        distributional_position[3] .+
        distributional_position[4] .* distributional_z
    distributional_sigma = exp.(distributional_log_sigma)
    distributional_residuals = response .- distributional_mu
    distributional_scaled_residuals =
        distributional_residuals ./ distributional_sigma
    distributional_expected_density =
        sum(logpdf.(Normal(), distributional_position)) +
        sum(logpdf.(Normal.(distributional_mu, distributional_sigma),
                    response))
    distributional_scale_score =
        abs2.(distributional_scaled_residuals) .- 1
    distributional_expected_gradient = [
        -distributional_position[1] +
            sum(distributional_residuals ./ distributional_sigma .^ 2),
        -distributional_position[2] +
            sum(raw_x .* distributional_residuals ./
                distributional_sigma .^ 2),
        -distributional_position[3] + sum(distributional_scale_score),
        -distributional_position[4] +
            sum(distributional_z .* distributional_scale_score),
    ]
    distributional_density, distributional_gradient =
        NP.logdensity_and_gradient!(
            distributional_workspace, distributional_prepared,
            distributional_position)
    @test distributional_density ≈ distributional_expected_density
    @test distributional_gradient ≈ distributional_expected_gradient
    @test distributional_workspace.primal.node_rows[1, :] ≈
          distributional_mu
    @test distributional_workspace.primal.node_rows[2, :] ≈
          distributional_log_sigma
    @test distributional_workspace.primal.node_rows[3, :] ≈
          distributional_sigma
    @test NP.evaluate(
        distributional_workspace, distributional_prepared,
        distributional_position, NP.LinearPredictor()) ≈ distributional_mu
    @test NP.evaluate(
        distributional_workspace, distributional_prepared,
        distributional_position, NP.PointwiseLogLikelihood()) ≈
          logpdf.(Normal.(distributional_mu, distributional_sigma), response)
    distributional_mu_query = NP.NodeOutput(:mu)
    distributional_log_sigma_query = NP.NodeOutput(:log_sigma)
    distributional_sigma_query = NP.NodeOutput(:sigma)
    @test distributional_mu_query isa
          BRM.NativePPLNodeOutput{:mu}
    for query in (
        distributional_mu_query,
        distributional_log_sigma_query,
        distributional_sigma_query,
    )
        signature = NP.output_signature(distributional_prepared, query)
        @test NP.output_axis(signature).keys == eachindex(response)
        @test NP.output_eltype(signature, distributional_prepared) ===
              Float64
        @test NP.allocate_output(
            distributional_prepared, query) isa Vector{Float64}
    end
    @test NP.evaluate(
        distributional_workspace, distributional_prepared,
        distributional_position, distributional_mu_query) ≈
          distributional_mu
    @test NP.evaluate(
        distributional_workspace, distributional_prepared,
        distributional_position, distributional_log_sigma_query) ≈
          distributional_log_sigma
    @test NP.evaluate(
        distributional_workspace, distributional_prepared,
        distributional_position, distributional_sigma_query) ≈
          distributional_sigma
    @test capability_error(() -> NP.output_signature(
        distributional_prepared, NP.NodeOutput(:missing))).capability == :query
    @test capability_error(() -> NP.output_signature(
        distributional_prepared, NP.NodeOutput(:y))).capability == :query

    distributional_positions = [
        distributional_position';
        [0.1, 0.2, -0.4, -0.15]'
    ]
    distributional_mu_draws = NP.evaluate_draws(
        distributional_workspace, distributional_prepared,
        distributional_positions, distributional_mu_query)
    distributional_sigma_draws = NP.evaluate_draws(
        distributional_workspace, distributional_prepared,
        distributional_positions, distributional_sigma_query)
    @test distributional_mu_draws[1, :] ≈ distributional_mu
    @test distributional_sigma_draws[1, :] ≈ distributional_sigma
    for draw in axes(distributional_positions, 1)
        @test distributional_mu_draws[draw, :] ≈ NP.evaluate(
            distributional_workspace, distributional_prepared,
            @view(distributional_positions[draw, :]),
            distributional_mu_query)
        @test distributional_sigma_draws[draw, :] ≈ NP.evaluate(
            distributional_workspace, distributional_prepared,
            @view(distributional_positions[draw, :]),
            distributional_sigma_query)
    end
    distributional_bundle_queries = (;
        mu=distributional_mu_query,
        log_sigma=distributional_log_sigma_query,
        sigma=distributional_sigma_query,
        pointwise=NP.PointwiseLogLikelihood())
    distributional_bundle = NP.execute_draws(
        distributional_workspace, distributional_prepared,
        distributional_positions, distributional_bundle_queries)
    @test distributional_bundle.mu == distributional_mu_draws
    @test distributional_bundle.sigma == distributional_sigma_draws
    @test distributional_bundle.log_sigma[1, :] ≈
          distributional_log_sigma
    @test distributional_bundle.pointwise[1, :] ≈
          logpdf.(Normal.(distributional_mu, distributional_sigma), response)
    @test factor_node_query_allocations(
        distributional_workspace, distributional_prepared,
        distributional_position, similar(response), similar(response),
        distributional_positions, similar(distributional_mu_draws),
        similar(distributional_sigma_draws),
        map(similar, distributional_bundle),
        distributional_bundle_queries) == (;
            scalar_mu=0, scalar_sigma=0, batch_mu=0, batch_sigma=0,
            bundle=0)

    distributional_replay_x = [-0.5, 0.25, 1.5]
    distributional_replay_z = [1.0, -0.5, 0.75]
    distributional_prediction_only = NP.rebind(
        distributional_prepared, (;);
        bindings=(;
            x=distributional_replay_x, z=distributional_replay_z))
    @test !NP.has_response(distributional_prediction_only)
    distributional_prediction_workspace = NP.workspace(
        distributional_prediction_only, Float64, DI.AutoEnzyme())
    @test NP.evaluate(
        distributional_prediction_workspace,
        distributional_prediction_only, distributional_position,
        distributional_mu_query) ≈
          distributional_position[1] .+
          distributional_position[2] .* distributional_replay_x
    @test NP.evaluate(
        distributional_prediction_workspace,
        distributional_prediction_only, distributional_position,
        distributional_sigma_query) ≈ exp.(
            distributional_position[3] .+
            distributional_position[4] .* distributional_replay_z)
    @test_throws ArgumentError NP.evaluate(
        distributional_prediction_workspace,
        distributional_prediction_only, distributional_position,
        NP.PointwiseLogLikelihood())

    distributional_float32 = NP.rebind(
        distributional_prepared, (; y=response);
        bindings=(; x=raw_x, z=distributional_z), T=Float32)
    distributional_float32_workspace = NP.workspace(
        distributional_float32, Float32, DI.AutoEnzyme())
    @test NP.evaluate(
        distributional_float32_workspace, distributional_float32,
        Float32.(distributional_position),
        distributional_sigma_query) isa Vector{Float32}
    distributional_predictive_rng = MersenneTwister(933)
    distributional_expected_predictive_rng = MersenneTwister(933)
    @test NP.simulate(
        distributional_predictive_rng, distributional_workspace,
        distributional_prepared, distributional_position) ≈
          distributional_mu .+
          distributional_sigma .* [
              randn(distributional_expected_predictive_rng)
              for _ in response
          ]
    @test factor_steady_state_allocations(
        distributional_workspace, distributional_prepared,
        distributional_position) == (; primal=0, gradient=0)
    distributional_underflow_position =
        [0.3, -0.4, -1000.0, 0.0]
    @test NP.logdensity!(
        distributional_workspace, distributional_prepared,
        distributional_underflow_position) == -Inf

    invalid_scale_declaration = NP.model(
        inputs=(; x=NP.input()),
        parameters=(;
            beta=NP.parameter(
                NP.RealSupport(), (:Intercept, :x);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            beta_scale=NP.parameter(
                NP.RealSupport(), (:Intercept, :x);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        nodes=(;
            mu=NP.affine(:x, :beta),
            invalid_scale=NP.affine(:x, :beta_scale)),
        observations=(;
            y=NP.broadcasted(NP.normal(:y, :mu, :invalid_scale))))
    @test capability_error(() -> NP.compile(
        invalid_scale_declaration, (; x=raw_x);
        conditions=(; y=response))).capability == :factor_scale

    block_exp_scale_declaration = NP.model(
        inputs=(; location=NP.input()),
        parameters=(;
            beta=NP.parameter(
                NP.RealSupport(), (:a, :b);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        nodes=(; scale=NP.exp_link(:beta)),
        observations=(;
            y=NP.broadcasted(NP.normal(:y, :location, :scale))))
    @test capability_error(() -> NP.compile(
        block_exp_scale_declaration, (; location=0.0);
        conditions=(; y=response))).capability == :factor_shape

    sampled_offset_data = (;
        x=[-1.0, 0.0, 1.0, 2.0],
        y=[0.1, 0.4, 0.8, 1.0])
    sampled_offset_brmi = @brm sampled_offset_data begin
        latent ~ Normal(0, 1)
        sigma ~ Exponential(2)
        mu ~ 0 + x + offset(latent)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(sampled_offset_brmi, :mu) == [:x]
    @test dependencies(sampled_offset_brmi, :mu).intermediates == [:latent]
    sampled_offset_model = NP.lower(sampled_offset_brmi)
    direct_sampled_offset_model = NP.model(
        inputs=(; x=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:x,); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2)),
            latent=NP.parameter(
                NP.RealSupport(), (:latent,); transform=NP.Identity(),
                prior=NP.StandardNormal())),
        nodes=(; mu=NP.affine(
            :x, :beta_mu; offsets=(:latent,), intercept=false)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :sigma))),
        site_order=(:latent, :beta_mu, :sigma, :y))
    @test typeof(sampled_offset_model) === typeof(direct_sampled_offset_model)
    @test sprint(show, sampled_offset_model) ==
          sprint(show, direct_sampled_offset_model)
    natural_sampled_offset = NP.condition(
        natural_sampled_offset_regression(sampled_offset_data.x);
        y=sampled_offset_data.y)
    @test typeof(natural_sampled_offset.declaration) ===
          typeof(sampled_offset_model)
    @test sprint(show, natural_sampled_offset.declaration) ==
          sprint(show, sampled_offset_model)
    @test SBBRMI(sampled_offset_brmi; mod=@__MODULE__) isa SBBRMI
    sampled_offset_graph = NP.factor_graph(
        sampled_offset_model; conditions=(; y=sampled_offset_data.y))
    @test sampled_offset_graph.schedule ==
          (:latent, :beta_mu, :sigma, :mu, :y)
    @test sampled_offset_graph.dimension == 3
    @test sampled_offset_graph.nodes.mu isa NP.AffineFactorNode
    @test !NP.affine_has_intercept(sampled_offset_graph.nodes.mu)
    @test sampled_offset_graph.nodes.mu.offsets ==
          (NP.SiteValue{:latent}(),)
    @test NP.factor_node_dependencies(sampled_offset_graph.nodes.mu) ==
          (:beta_mu, :latent)

    row_node_to_scalar_model = NP.model(
        inputs=direct_sampled_offset_model.inputs,
        parameters=direct_sampled_offset_model.parameters,
        nodes=direct_sampled_offset_model.nodes,
        observations=(;
            z=NP.normal(:z, :mu, :sigma),
            y=direct_sampled_offset_model.observations.y),
        site_order=(:latent, :beta_mu, :sigma, :z, :y))
    @test capability_error(() -> NP.compile(
        row_node_to_scalar_model, (; x=sampled_offset_data.x);
        conditions=(; y=sampled_offset_data.y))).capability == :factor_shape
    affine_scale_model = NP.model(
        inputs=direct_sampled_offset_model.inputs,
        parameters=direct_sampled_offset_model.parameters,
        nodes=direct_sampled_offset_model.nodes,
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :latent, :mu))),
        site_order=(:latent, :beta_mu, :sigma, :y))
    @test capability_error(() -> NP.compile(
        affine_scale_model, (; x=sampled_offset_data.x);
        conditions=(; y=sampled_offset_data.y))).capability == :factor_scale
    row_exp_model = NP.model(
        inputs=direct_sampled_offset_model.inputs,
        parameters=direct_sampled_offset_model.parameters,
        nodes=(; mu=direct_sampled_offset_model.nodes.mu,
                 row_scale=NP.exp_link(:mu)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :latent, :row_scale))),
        site_order=(:latent, :beta_mu, :sigma, :y))
    row_exp_plan = NP.compile(
        row_exp_model, (; x=sampled_offset_data.x);
        conditions=(; y=sampled_offset_data.y))
    @test row_exp_plan isa NP.FactorPlan
    @test row_exp_plan.graph.nodes.row_scale isa NP.ExpFactorNode
    exposure_data = (;
        x=[-1.0, 0.0, 0.5, 1.0, 2.0],
        exposure=[0.5, 1.0, 2.0, 4.0, 1.5],
        y=[0, 1, 2, 3, 1])
    exposure_brm = @brm exposure_data begin
        log_rate ~ 1 + x + offset(log(exposure))
        y ~ Poisson(exp(log_rate))
    end
    @test popcoefnames(exposure_brm, :log_rate) == [:Intercept, :x]
    @test SBBRMI(exposure_brm; mod=@__MODULE__) isa SBBRMI
    direct_exposure_model = NP.model(
        inputs=(; x=NP.input(), exposure=NP.input()),
        parameters=(;
            beta_log_rate=NP.parameter(
                NP.RealSupport(), (:Intercept, :x);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        nodes=(;
            log_exposure_for_log_rate=NP.log_link(:exposure),
            log_rate=NP.affine(
                :x, :beta_log_rate;
                offsets=(:log_exposure_for_log_rate,), intercept=true),
            exp_log_rate=NP.exp_link(:log_rate)),
        observations=(;
            y=NP.broadcasted(NP.poisson(:y, :exp_log_rate))))
    exposure_model = NP.lower(exposure_brm)
    @test exposure_model == direct_exposure_model
    natural_exposure = NP.condition(
        natural_exposure_poisson(
            exposure_data.x, exposure_data.exposure);
        y=exposure_data.y)
    @test natural_exposure.declaration == exposure_model
    exposure_only_brm = @brm exposure_data begin
        log_rate ~ 1 + offset(log(exposure))
        y ~ Poisson(exp(log_rate))
    end
    @test SBBRMI(exposure_only_brm; mod=@__MODULE__) isa SBBRMI
    natural_exposure_only = NP.condition(
        natural_exposure_only_poisson(exposure_data.exposure);
        y=exposure_data.y)
    @test natural_exposure_only.declaration == NP.lower(exposure_only_brm)
    exposure_only_prepared = NP.prepare(exposure_only_brm)
    @test exposure_only_prepared.plan.graph.dimension == 1
    @test NP.evaluate(
        NP.workspace(exposure_only_prepared), exposure_only_prepared,
        [0.2], NP.LinearPredictor()) ≈
          0.2 .+ log.(exposure_data.exposure)
    exposure_plan = NP.compile(exposure_brm)
    @test exposure_plan.graph.schedule == (
        :beta_log_rate, :log_exposure_for_log_rate,
        :log_rate, :exp_log_rate, :y)
    @test exposure_plan.graph.dimension == 2
    @test exposure_plan.graph.nodes.log_exposure_for_log_rate isa
          NP.LogFactorNode
    @test NP.factor_node_dependencies(
        exposure_plan.graph.nodes.log_exposure_for_log_rate) == ()
    @test exposure_plan.graph.nodes.log_rate.offsets ==
          (NP.NodeValue{:log_exposure_for_log_rate}(),)
    @test NP.factor_node_dependencies(exposure_plan.graph.nodes.log_rate) ==
          (:beta_log_rate, :log_exposure_for_log_rate)
    exposure_prepared = NP.prepare(exposure_plan)
    exposure_workspace = NP.workspace(
        exposure_prepared, Float64, DI.AutoEnzyme())
    exposure_position = [0.2, -0.3]
    exposure_log_rate = exposure_position[1] .+
        exposure_position[2] .* exposure_data.x .+
        log.(exposure_data.exposure)
    exposure_rate = exp.(exposure_log_rate)
    exposure_expected_density =
        sum(logpdf.(Normal(), exposure_position)) +
        sum(BRM._native_ppl_poisson_logdensity.(
            Float64.(exposure_data.y), exposure_log_rate))
    exposure_expected_gradient = [
        -exposure_position[1] +
            sum(exposure_data.y .- exposure_rate),
        -exposure_position[2] +
            sum(exposure_data.x .* (exposure_data.y .- exposure_rate))]
    exposure_density, exposure_gradient = NP.logdensity_and_gradient!(
        exposure_workspace, exposure_prepared, exposure_position)
    @test exposure_density ≈ exposure_expected_density
    @test exposure_gradient ≈ exposure_expected_gradient
    @test NP.evaluate(
        exposure_workspace, exposure_prepared, exposure_position,
        NP.LinearPredictor()) ≈ exposure_log_rate
    @test NP.evaluate(
        exposure_workspace, exposure_prepared, exposure_position,
        NP.NodeOutput(:exp_log_rate)) ≈ exposure_rate
    @test NP.evaluate(
        exposure_workspace, exposure_prepared, exposure_position,
        NP.PointwiseLogLikelihood()) ≈
          BRM._native_ppl_poisson_logdensity.(
              Float64.(exposure_data.y), exposure_log_rate)
    @test factor_steady_state_allocations(
        exposure_workspace, exposure_prepared,
        exposure_position) == (; primal=0, gradient=0)
    for invalid_exposure in (
            [0.5, 0.0, 2.0, 4.0, 1.5],
            [0.5, -1.0, 2.0, 4.0, 1.5],
            [0.5, Inf, 2.0, 4.0, 1.5],
            [0.5, NaN, 2.0, 4.0, 1.5])
        @test_throws ArgumentError NP.prepare(NP.condition(
            natural_exposure_poisson(exposure_data.x, invalid_exposure);
            y=exposure_data.y))
    end
    raw_offset_prepared = NP.prepare(NP.condition(
        natural_raw_offset_poisson(
            exposure_data.x, log.(exposure_data.exposure));
        y=exposure_data.y))
    @test NP.evaluate(
        NP.workspace(raw_offset_prepared), raw_offset_prepared,
        exposure_position, NP.LinearPredictor()) ≈ exposure_log_rate
    for invalid_offset in (
            [log(0.5), Inf, log(2.0), log(4.0), log(1.5)],
            [log(0.5), NaN, log(2.0), log(4.0), log(1.5)])
        @test_throws ArgumentError NP.prepare(NP.condition(
            natural_raw_offset_poisson(exposure_data.x, invalid_offset);
            y=exposure_data.y))
    end
    exposure_predictive_rng = MersenneTwister(990)
    exposure_expected_rng = MersenneTwister(990)
    exposure_predictive = NP.allocate_output(
        exposure_prepared, NP.PosteriorPredictive())
    NP.simulate!(
        exposure_predictive_rng, exposure_predictive,
        exposure_workspace, exposure_prepared, exposure_position)
    @test exposure_predictive == [
        BRM._native_ppl_rand_poisson(
            exposure_expected_rng, Float64, log_rate)
        for log_rate in exposure_log_rate]
    @test factor_query_allocations(
        exposure_workspace, exposure_prepared, exposure_position,
        zeros(5), zeros(5), zeros(Int, 5), zeros(2)) ==
          (; linear=0, pointwise=0, predictive=0, prior=0)

    exposure_replay_bindings = (;
        x=Float32[1.5, -0.5, 0.25],
        exposure=Float32[2.0, 0.75, 3.0])
    exposure_prediction_only = NP.rebind(
        exposure_prepared, (;); bindings=exposure_replay_bindings)
    @test !NP.has_response(exposure_prediction_only)
    @test eltype(exposure_prediction_only) === Float64
    exposure_replay_position = [0.1, 0.4]
    exposure_replay_log_rate = exposure_replay_position[1] .+
        exposure_replay_position[2] .*
            Float64.(exposure_replay_bindings.x) .+
        log.(Float64.(exposure_replay_bindings.exposure))
    exposure_replay_workspace = NP.workspace(
        exposure_prediction_only, Float64, DI.AutoEnzyme())
    @test NP.evaluate(
        exposure_replay_workspace, exposure_prediction_only,
        exposure_replay_position, NP.LinearPredictor()) ≈
          exposure_replay_log_rate
    exposure_replay_rng = MersenneTwister(991)
    exposure_replay_expected_rng = MersenneTwister(991)
    @test NP.simulate(
        exposure_replay_rng, exposure_replay_workspace,
        exposure_prediction_only, exposure_replay_position) == [
            BRM._native_ppl_rand_poisson(
                exposure_replay_expected_rng, Float64, log_rate)
            for log_rate in exposure_replay_log_rate]
    @test_throws ArgumentError NP.rebind(
        exposure_prepared, (;);
        bindings=(; x=Float32[0, 1], exposure=Float32[1, 0]))
    exposure_rebound_response = Int[2, 0, 1]
    exposure_rebound = NP.rebind(
        exposure_prepared, (; y=exposure_rebound_response);
        bindings=exposure_replay_bindings)
    exposure_rebound_workspace = NP.workspace(
        exposure_rebound, Float64, DI.AutoEnzyme())
    exposure_rebound_density, exposure_rebound_gradient =
        NP.logdensity_and_gradient!(
            exposure_rebound_workspace, exposure_rebound,
            exposure_replay_position)
    @test isfinite(exposure_rebound_density)
    @test all(isfinite, exposure_rebound_gradient)
    @test NP.evaluate(
        exposure_rebound_workspace, exposure_rebound,
        exposure_replay_position, NP.PointwiseLogLikelihood()) ≈
          BRM._native_ppl_poisson_logdensity.(
              Float64.(exposure_rebound_response),
              exposure_replay_log_rate)

    exposure_float32 = NP.condition(
        natural_exposure_poisson(
            Float32.(exposure_data.x), Float32.(exposure_data.exposure));
        y=exposure_data.y)
    exposure_float32_prepared = NP.prepare(exposure_float32; T=Float32)
    @test eltype(exposure_float32_prepared) === Float32
    exposure_float32_workspace = NP.workspace(
        exposure_float32_prepared, Float32, DI.AutoEnzyme())
    exposure_float32_position = Float32.(exposure_position)
    exposure_float32_density, exposure_float32_gradient =
        NP.logdensity_and_gradient!(
            exposure_float32_workspace, exposure_float32_prepared,
            exposure_float32_position)
    @test exposure_float32_density ≈ Float32(exposure_density) rtol=1f-5
    @test exposure_float32_gradient ≈
          Float32.(exposure_gradient) rtol=1f-5

    exposure_draw_positions = [
        exposure_position';
        0.1 0.25]
    exposure_draw_linear = zeros(2, 5)
    exposure_draw_pointwise = zeros(2, 5)
    exposure_draw_predictive = zeros(Int, 2, 5)
    NP.evaluate_draws!(
        exposure_draw_linear, exposure_workspace, exposure_prepared,
        exposure_draw_positions, NP.LinearPredictor())
    NP.evaluate_draws!(
        exposure_draw_pointwise, exposure_workspace, exposure_prepared,
        exposure_draw_positions, NP.PointwiseLogLikelihood())
    exposure_draw_rng = MersenneTwister(992)
    exposure_manual_rng = MersenneTwister(992)
    exposure_manual_predictive = similar(exposure_draw_predictive)
    for draw in axes(exposure_draw_positions, 1)
        NP.simulate!(
            exposure_manual_rng,
            @view(exposure_manual_predictive[draw, :]),
            exposure_workspace, exposure_prepared,
            @view(exposure_draw_positions[draw, :]))
    end
    NP.simulate_draws!(
        exposure_draw_rng, exposure_draw_predictive,
        exposure_workspace, exposure_prepared,
        exposure_draw_positions)
    @test exposure_draw_predictive == exposure_manual_predictive
    exposure_queries = (;
        linear=NP.LinearPredictor(),
        pointwise=NP.PointwiseLogLikelihood(),
        predictive=NP.PosteriorPredictive())
    exposure_bundle = (;
        linear=zeros(2, 5),
        pointwise=zeros(2, 5),
        predictive=zeros(Int, 2, 5))
    NP.execute_draws!(
        MersenneTwister(992), exposure_bundle,
        exposure_workspace, exposure_prepared,
        exposure_draw_positions, exposure_queries)
    @test exposure_bundle.linear == exposure_draw_linear
    @test exposure_bundle.pointwise == exposure_draw_pointwise
    @test exposure_bundle.predictive == exposure_manual_predictive
    @test factor_batch_allocations(
        exposure_workspace, exposure_prepared,
        exposure_draw_positions, exposure_draw_linear,
        exposure_draw_pointwise, exposure_draw_predictive,
        exposure_bundle) ==
          (; linear=0, pointwise=0, predictive=0, bundle=0)

    grouped_declaration = NP.model(
        inputs=(; group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:tau,); transform=NP.Exp(),
                prior=NP.Exponential(1)),
            varying=NP.grouped_normal(:group, 0.0, :tau),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(; varying_by_row=NP.group_gather(:varying, :group)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :varying_by_row, :sigma))),
        site_order=(:tau, :varying, :sigma, :y))
    grouped_bindings = (; group=[:a, :b, :a, :c])
    grouped_graph = NP.factor_graph(
        grouped_declaration; bindings=grouped_bindings,
        conditions=(; y=sampled_offset_data.y))
    @test grouped_graph.schedule ==
          (:tau, :varying, :sigma, :varying_by_row, :y)
    @test grouped_graph.dimension == 5
    @test keys(grouped_graph.coordinates) == (:tau, :varying, :sigma)
    @test grouped_graph.coordinates.varying.keys == (
        NP.GroupCoordinateKey(:varying, :a),
        NP.GroupCoordinateKey(:varying, :b),
        NP.GroupCoordinateKey(:varying, :c))
    @test grouped_graph.coordinates.varying.indices == 2:4
    @test grouped_graph.sites.varying.shape isa NP.BlockSiteShape
    @test grouped_graph.sites.varying.factor isa NP.NormalSiteFactor
    @test NP.site_factor_dependencies(
        grouped_graph.sites.varying.factor) == (:tau,)
    @test grouped_graph.nodes.varying_by_row isa NP.GroupGatherFactorNode
    @test NP.factor_node_dependencies(
        grouped_graph.nodes.varying_by_row) == (:varying,)
    @test_throws ArgumentError NP.factor_graph(
        grouped_declaration; conditions=(; y=sampled_offset_data.y))
    grouped_plan = NP.compile(
        grouped_declaration, grouped_bindings;
        conditions=(; y=sampled_offset_data.y))
    @test grouped_plan isa NP.FactorPlan
    @test grouped_plan.group_indices ==
          (; varying_by_row=(1, 2, 1, 3))
    grouped_prepared = NP.prepare(grouped_plan)
    @test grouped_prepared.plan.bindings.group !== grouped_bindings.group
    grouped_source = copy(grouped_bindings.group)
    grouped_owned = NP.prepare(NP.compile(
        grouped_declaration, (; group=grouped_source);
        conditions=(; y=sampled_offset_data.y)))
    grouped_source[1] = :changed_after_prepare
    @test grouped_owned.plan.bindings.group == grouped_bindings.group
    @test grouped_owned.plan.group_indices == grouped_plan.group_indices
    grouped_workspace = NP.workspace(
        grouped_prepared, Float64, DI.AutoEnzyme())
    grouped_position = [log(0.7), 0.2, -0.1, 0.4, log(0.5)]
    grouped_tau = exp(grouped_position[1])
    grouped_effects = grouped_position[2:4]
    grouped_sigma = exp(grouped_position[5])
    grouped_mu = grouped_effects[[1, 2, 1, 3]]
    grouped_residuals = sampled_offset_data.y .- grouped_mu
    grouped_expected_density =
        logpdf(Exponential(1), grouped_tau) + grouped_position[1] +
        sum(logpdf.(Normal(0, grouped_tau), grouped_effects)) +
        logpdf(Exponential(2), grouped_sigma) + grouped_position[5] +
        sum(logpdf.(Normal.(grouped_mu, grouped_sigma),
                    sampled_offset_data.y))
    grouped_expected_gradient = [
        1 - grouped_tau - length(grouped_effects) +
            sum(abs2, grouped_effects) / grouped_tau^2,
        -grouped_effects[1] / grouped_tau^2 +
            (grouped_residuals[1] + grouped_residuals[3]) /
                grouped_sigma^2,
        -grouped_effects[2] / grouped_tau^2 +
            grouped_residuals[2] / grouped_sigma^2,
        -grouped_effects[3] / grouped_tau^2 +
            grouped_residuals[4] / grouped_sigma^2,
        1 - grouped_sigma / 2 - length(sampled_offset_data.y) +
            sum(abs2, grouped_residuals) / grouped_sigma^2,
    ]
    grouped_density, grouped_gradient = NP.logdensity_and_gradient!(
        grouped_workspace, grouped_prepared, grouped_position)
    @test grouped_density ≈ grouped_expected_density
    @test grouped_gradient ≈ grouped_expected_gradient
    @test NP.evaluate(
        grouped_workspace, grouped_prepared, grouped_position,
        NP.LinearPredictor()) == grouped_mu
    @test NP.evaluate(
        grouped_workspace, grouped_prepared, grouped_position,
        NP.PointwiseLogLikelihood()) ≈
          logpdf.(Normal.(grouped_mu, grouped_sigma), sampled_offset_data.y)
    @test factor_steady_state_allocations(
        grouped_workspace, grouped_prepared, grouped_position) ==
          (; primal=0, gradient=0)
    grouped_prediction_only = NP.rebind(grouped_prepared, (;))
    @test !NP.has_response(grouped_prediction_only)
    @test length(NP.simulate(
        MersenneTwister(934), NP.workspace(grouped_prediction_only),
        grouped_prediction_only, grouped_position)) ==
          length(grouped_bindings.group)

    row_product_declaration = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:tau,); transform=NP.Exp(),
                prior=NP.Exponential(1)),
            varying=NP.grouped_normal(:group, 0.0, :tau),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(;
            varying_by_row=NP.group_gather(:varying, :group),
            varying_slope=NP.row_product(:varying_by_row, :x)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :varying_slope, :sigma))),
        site_order=(:tau, :varying, :sigma, :y))
    @test NP.row_product_inputs(
        row_product_declaration.nodes.varying_slope) ==
          (:varying_by_row, :x)
    row_product_bindings = (;
        x=[0.5, -1.0, 1.5, 2.0], group=grouped_bindings.group)
    row_product_graph = NP.factor_graph(
        row_product_declaration; bindings=row_product_bindings,
        conditions=(; y=sampled_offset_data.y))
    @test row_product_graph.schedule == (
        :tau, :varying, :sigma, :varying_by_row, :varying_slope, :y)
    @test row_product_graph.nodes.varying_slope isa NP.RowProductFactorNode
    @test NP.factor_node_dependencies(
        row_product_graph.nodes.varying_slope) == (:varying_by_row,)
    row_product_prepared = NP.prepare(NP.compile(
        row_product_declaration, row_product_bindings;
        conditions=(; y=sampled_offset_data.y)))
    row_product_workspace = NP.workspace(
        row_product_prepared, Float64, DI.AutoEnzyme())
    row_product_mu = grouped_mu .* row_product_bindings.x
    row_product_residuals = sampled_offset_data.y .- row_product_mu
    row_product_expected_density =
        logpdf(Exponential(1), grouped_tau) + grouped_position[1] +
        sum(logpdf.(Normal(0, grouped_tau), grouped_effects)) +
        logpdf(Exponential(2), grouped_sigma) + grouped_position[5] +
        sum(logpdf.(Normal.(row_product_mu, grouped_sigma),
                    sampled_offset_data.y))
    row_product_expected_gradient = [
        grouped_expected_gradient[1],
        -grouped_effects[1] / grouped_tau^2 +
            (row_product_residuals[1] * row_product_bindings.x[1] +
             row_product_residuals[3] * row_product_bindings.x[3]) /
                grouped_sigma^2,
        -grouped_effects[2] / grouped_tau^2 +
            row_product_residuals[2] * row_product_bindings.x[2] /
                grouped_sigma^2,
        -grouped_effects[3] / grouped_tau^2 +
            row_product_residuals[4] * row_product_bindings.x[4] /
                grouped_sigma^2,
        1 - grouped_sigma / 2 - length(sampled_offset_data.y) +
            sum(abs2, row_product_residuals) / grouped_sigma^2,
    ]
    row_product_density, row_product_gradient = NP.logdensity_and_gradient!(
        row_product_workspace, row_product_prepared, grouped_position)
    @test row_product_density ≈ row_product_expected_density
    @test row_product_gradient ≈ row_product_expected_gradient
    @test NP.evaluate(
        row_product_workspace, row_product_prepared, grouped_position,
        NP.LinearPredictor()) == row_product_mu
    @test factor_steady_state_allocations(
        row_product_workspace, row_product_prepared, grouped_position) ==
          (; primal=0, gradient=0)

    direct_block_product = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=row_product_declaration.parameters,
        nodes=(;
            routed=NP.group_gather(:varying, :group),
            invalid=NP.row_product(:varying, :x)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :invalid, :sigma))),
        site_order=row_product_declaration.site_order)
    direct_block_product_error = capability_error(() -> NP.compile(
        direct_block_product, row_product_bindings;
        conditions=(; y=sampled_offset_data.y)))
    @test direct_block_product_error.capability == :factor_shape
    @test occursin(
        "cannot consume block site `varying` directly",
        sprint(showerror, direct_block_product_error))

    structured_group_declaration = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:Intercept, :x);
                transform=NP.Exp(), prior=NP.Exponential(1)),
            correlation=NP.cholesky_correlation((:Intercept, :x), 2.0),
            z=NP.grouped_standard_normal(:group, (:Intercept, :x)),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        observations=(; y=NP.broadcasted(NP.normal(:y, :x, :sigma))),
        site_order=(:tau, :correlation, :z, :sigma, :y))
    @test NP.group_input(structured_group_declaration.parameters.z) === :group
    @test NP.group_coefficients(
        structured_group_declaration.parameters.z) == (:Intercept, :x)
    @test NP.correlation_coefficients(
        structured_group_declaration.parameters.correlation) ==
          (:Intercept, :x)
    structured_group_graph = NP.factor_graph(
        structured_group_declaration;
        bindings=(; x=row_product_bindings.x,
                  group=row_product_bindings.group),
        conditions=(; y=sampled_offset_data.y))
    @test structured_group_graph.schedule ==
          (:tau, :correlation, :z, :sigma, :y)
    @test structured_group_graph.dimension == 10
    @test structured_group_graph.sites.tau.shape isa NP.BlockSiteShape
    @test structured_group_graph.sites.tau.factor isa NP.ExponentialSiteFactor
    @test structured_group_graph.sites.correlation.support isa
          NP.CholeskyCorrelationSupport{2}
    @test structured_group_graph.sites.correlation.transform isa
          NP.CholeskyCorrelationTransform{2}
    @test structured_group_graph.sites.correlation.factor isa
          NP.LKJCholeskySiteFactor
    @test structured_group_graph.sites.correlation.factor.eta == 2.0
    @test structured_group_graph.coordinates.correlation.keys ==
          (NP.CorrelationCoordinateKey(:correlation, 2, 1),)
    @test structured_group_graph.coordinates.z.keys == (
        NP.GroupCoefficientKey(:z, :a, :Intercept),
        NP.GroupCoefficientKey(:z, :a, :x),
        NP.GroupCoefficientKey(:z, :b, :Intercept),
        NP.GroupCoefficientKey(:z, :b, :x),
        NP.GroupCoefficientKey(:z, :c, :Intercept),
        NP.GroupCoefficientKey(:z, :c, :x))
    @test_throws ArgumentError NP.grouped_standard_normal(:group, ())
    @test_throws ArgumentError NP.grouped_standard_normal(
        :group, (:x, :x))
    @test_throws ArgumentError NP.cholesky_correlation((:x,), 2.0)
    @test_throws ArgumentError NP.cholesky_correlation(
        (:Intercept, :x), 0.0)
    @test_throws ArgumentError NP.factor_graph(
        structured_group_declaration;
        bindings=(; x=row_product_bindings.x,
                  group=row_product_bindings.group),
        conditions=(; y=sampled_offset_data.y),
        group_levels=(; z=()))
    @test_throws ArgumentError NP.factor_graph(
        structured_group_declaration;
        bindings=(; x=row_product_bindings.x,
                  group=row_product_bindings.group),
        conditions=(; y=sampled_offset_data.y),
        group_levels=(; z=(:a, :a)))

    correlated_group_declaration = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:Intercept, :x);
                transform=NP.Exp(), prior=NP.Exponential(1)),
            correlation=NP.cholesky_correlation((:Intercept, :x), 2.0),
            z=NP.grouped_standard_normal(:group, (:Intercept, :x)),
            beta=NP.parameter(
                NP.RealSupport(), (:x,); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(;
            correlated_by_row=NP.grouped_affine(
                :z, :tau, :correlation, :group, (nothing, :x)),
            mu=NP.affine(
                :x, :beta; offsets=(:correlated_by_row,),
                intercept=false)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :sigma))),
        site_order=(:tau, :correlation, :z, :beta, :sigma, :y))
    @test_throws ArgumentError NP.grouped_affine(
        :z, :tau, :correlation, :group, ())
    @test_throws ArgumentError NP.grouped_affine(
        :z, :tau, :correlation, :group, (nothing, nothing))
    correlated_group_node = correlated_group_declaration.nodes.correlated_by_row
    @test NP.grouped_standardized(correlated_group_node) === :z
    @test NP.grouped_scales(correlated_group_node) === :tau
    @test NP.grouped_correlation(correlated_group_node) === :correlation
    @test NP.group_input(correlated_group_node) === :group
    @test NP.grouped_predictors(correlated_group_node) == (nothing, :x)
    correlated_group_plan = NP.compile(
        correlated_group_declaration,
        (; x=sampled_offset_data.x, group=grouped_bindings.group);
        conditions=(; y=sampled_offset_data.y))
    @test correlated_group_plan isa NP.FactorPlan
    @test correlated_group_plan.graph.schedule == (
        :tau, :correlation, :z, :beta, :sigma,
        :correlated_by_row, :mu, :y)
    @test correlated_group_plan.graph.dimension == 11
    @test correlated_group_plan.group_indices ==
          (; correlated_by_row=(1, 2, 1, 3))
    @test correlated_group_plan.graph.nodes.correlated_by_row isa
          NP.GroupedAffineFactorNode
    @test NP.factor_node_dependencies(
        correlated_group_plan.graph.nodes.correlated_by_row) ==
          (:z, :tau, :correlation)
    correlated_scalar_consumer = NP.model(
        inputs=correlated_group_declaration.inputs,
        parameters=correlated_group_declaration.parameters,
        nodes=correlated_group_declaration.nodes,
        observations=(;
            scalar=NP.normal(:scalar, :correlated_by_row, :sigma),
            y=correlated_group_declaration.observations.y),
        site_order=(
            :tau, :correlation, :z, :beta, :sigma, :scalar, :y))
    @test capability_error(() -> NP.compile(
        correlated_scalar_consumer,
        (; x=sampled_offset_data.x, group=grouped_bindings.group);
        conditions=(; y=sampled_offset_data.y))).capability == :factor_shape
    correlated_block_predictor = NP.model(
        inputs=correlated_group_declaration.inputs,
        parameters=correlated_group_declaration.parameters,
        nodes=(;
            invalid=NP.grouped_affine(
                :z, :tau, :correlation, :group, (:tau, :x)),),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :invalid, :sigma))),
        site_order=correlated_group_declaration.site_order)
    @test capability_error(() -> NP.compile(
        correlated_block_predictor,
        (; x=sampled_offset_data.x, group=grouped_bindings.group);
        conditions=(; y=sampled_offset_data.y))).capability == :factor_shape

    correlated_glmm_parameters = (;
        tau=correlated_group_declaration.parameters.tau,
        correlation=correlated_group_declaration.parameters.correlation,
        z=correlated_group_declaration.parameters.z,
        beta=correlated_group_declaration.parameters.beta)
    correlated_glmm_nodes = correlated_group_declaration.nodes
    correlated_bernoulli_declaration = NP.model(
        inputs=correlated_group_declaration.inputs,
        parameters=correlated_glmm_parameters,
        nodes=correlated_glmm_nodes,
        observations=(;
            y=NP.broadcasted(NP.bernoulli_logit(:y, :mu))),
        site_order=(:tau, :correlation, :z, :beta, :y))
    correlated_bernoulli_plan = NP.compile(
        correlated_bernoulli_declaration,
        (; x=sampled_offset_data.x, group=grouped_bindings.group);
        conditions=(; y=Bool[true, false, true, true]))
    @test correlated_bernoulli_plan isa NP.FactorPlan
    @test correlated_bernoulli_plan.graph.sites.y.factor isa
          NP.BernoulliLogitSiteFactor
    correlated_bernoulli_prepared = NP.prepare(correlated_bernoulli_plan)
    correlated_bernoulli_predictive_signature = NP.output_signature(
        correlated_bernoulli_plan, NP.PosteriorPredictive())
    @test NP.output_eltype(
        correlated_bernoulli_predictive_signature, Float64) === Bool
    @test NP.allocate_output(
        correlated_bernoulli_prepared,
        NP.PosteriorPredictive()) isa Vector{Bool}
    @test NP.allocate_output(
        correlated_bernoulli_prepared,
        NP.PointwiseLogLikelihood()) isa Vector{Float64}

    correlated_poisson_declaration = NP.model(
        inputs=correlated_group_declaration.inputs,
        parameters=correlated_glmm_parameters,
        nodes=merge(correlated_glmm_nodes,
                    (; rate=NP.exp_link(:mu))),
        observations=(; y=NP.broadcasted(NP.poisson(:y, :rate))),
        site_order=(:tau, :correlation, :z, :beta, :y))
    correlated_poisson_plan = NP.compile(
        correlated_poisson_declaration,
        (; x=sampled_offset_data.x, group=grouped_bindings.group);
        conditions=(; y=Int[2, 0, 1, 3]))
    @test correlated_poisson_plan isa NP.FactorPlan
    @test correlated_poisson_plan.graph.sites.y.factor isa
          NP.PoissonSiteFactor
    correlated_poisson_prepared = NP.prepare(correlated_poisson_plan)
    correlated_poisson_predictive_signature = NP.output_signature(
        correlated_poisson_plan, NP.PosteriorPredictive())
    @test NP.output_eltype(
        correlated_poisson_predictive_signature, Float64) === Int
    @test NP.allocate_output(
        correlated_poisson_prepared,
        NP.PosteriorPredictive()) isa Vector{Int}
    @test NP.allocate_output(
        correlated_poisson_prepared,
        NP.LinearPredictor()) isa Vector{Float64}

    grouped_discrete_parameters = (;
        tau=grouped_declaration.parameters.tau,
        varying=grouped_declaration.parameters.varying)
    grouped_bernoulli_declaration = NP.model(
        inputs=grouped_declaration.inputs,
        parameters=grouped_discrete_parameters,
        nodes=grouped_declaration.nodes,
        observations=(; y=NP.broadcasted(
            NP.bernoulli_logit(:y, :varying_by_row))),
        site_order=(:tau, :varying, :y))
    grouped_bernoulli_prediction = NP.prepare(NP.compile(
        grouped_bernoulli_declaration, grouped_bindings))
    grouped_bernoulli_prior_rng = MersenneTwister(947)
    grouped_bernoulli_expected_rng = MersenneTwister(947)
    grouped_bernoulli_tau = randexp(grouped_bernoulli_expected_rng)
    grouped_bernoulli_effects = grouped_bernoulli_tau .* [
        randn(grouped_bernoulli_expected_rng) for _ in 1:3]
    grouped_bernoulli_expected = [
        rand(grouped_bernoulli_expected_rng) <
            BRM._native_ppl_logistic(value)
        for value in grouped_bernoulli_effects[[1, 2, 1, 3]]]
    grouped_bernoulli_prior = NP.simulate_prior(
        grouped_bernoulli_prior_rng,
        NP.workspace(grouped_bernoulli_prediction),
        grouped_bernoulli_prediction)
    @test grouped_bernoulli_prior.position ≈ [
        log(grouped_bernoulli_tau), grouped_bernoulli_effects...]
    @test grouped_bernoulli_prior.response == grouped_bernoulli_expected
    @test grouped_bernoulli_prior.response isa Vector{Bool}
    grouped_bernoulli_prior_position = zeros(4)
    grouped_bernoulli_prior_response = falses(4)
    grouped_bernoulli_prior_work = NP.workspace(
        grouped_bernoulli_prediction)
    NP.simulate_prior!(
        MersenneTwister(948), grouped_bernoulli_prior_position,
        grouped_bernoulli_prior_response, grouped_bernoulli_prior_work,
        grouped_bernoulli_prediction)
    grouped_bernoulli_allocation_rng = MersenneTwister(949)
    @test @allocated(NP.simulate_prior!(
        grouped_bernoulli_allocation_rng, grouped_bernoulli_prior_position,
        grouped_bernoulli_prior_response, grouped_bernoulli_prior_work,
        grouped_bernoulli_prediction)) == 0

    grouped_poisson_declaration = NP.model(
        inputs=grouped_declaration.inputs,
        parameters=grouped_discrete_parameters,
        nodes=merge(grouped_declaration.nodes,
                    (; rate=NP.exp_link(:varying_by_row))),
        observations=(; y=NP.broadcasted(NP.poisson(:y, :rate))),
        site_order=(:tau, :varying, :y))
    grouped_poisson_prediction = NP.prepare(NP.compile(
        grouped_poisson_declaration, grouped_bindings))
    grouped_poisson_prior_rng = MersenneTwister(950)
    grouped_poisson_expected_rng = MersenneTwister(950)
    grouped_poisson_tau = randexp(grouped_poisson_expected_rng)
    grouped_poisson_effects = grouped_poisson_tau .* [
        randn(grouped_poisson_expected_rng) for _ in 1:3]
    grouped_poisson_expected = [
        BRM._native_ppl_rand_poisson(
            grouped_poisson_expected_rng, Float64, value)
        for value in grouped_poisson_effects[[1, 2, 1, 3]]]
    grouped_poisson_prior = NP.simulate_prior(
        grouped_poisson_prior_rng, NP.workspace(grouped_poisson_prediction),
        grouped_poisson_prediction)
    @test grouped_poisson_prior.position ≈ [
        log(grouped_poisson_tau), grouped_poisson_effects...]
    @test grouped_poisson_prior.response == grouped_poisson_expected
    @test grouped_poisson_prior.response isa Vector{Int}
    grouped_poisson_prior_position = zeros(4)
    grouped_poisson_prior_response = zeros(Int, 4)
    grouped_poisson_prior_work = NP.workspace(grouped_poisson_prediction)
    NP.simulate_prior!(
        MersenneTwister(951), grouped_poisson_prior_position,
        grouped_poisson_prior_response, grouped_poisson_prior_work,
        grouped_poisson_prediction)
    grouped_poisson_allocation_rng = MersenneTwister(952)
    @test @allocated(NP.simulate_prior!(
        grouped_poisson_allocation_rng, grouped_poisson_prior_position,
        grouped_poisson_prior_response, grouped_poisson_prior_work,
        grouped_poisson_prediction)) == 0

    invalid_poisson_rate_node = NP.model(
        inputs=grouped_declaration.inputs,
        parameters=grouped_discrete_parameters,
        nodes=grouped_declaration.nodes,
        observations=(; y=NP.broadcasted(
            NP.poisson(:y, :varying_by_row))),
        site_order=(:tau, :varying, :y))
    @test capability_error(() -> NP.compile(
        invalid_poisson_rate_node, grouped_bindings)).capability == :factor_rate
    invalid_poisson_rate_input = NP.model(
        inputs=(; group=NP.input(), rate=NP.input()),
        parameters=grouped_discrete_parameters,
        nodes=grouped_declaration.nodes,
        observations=(; y=NP.broadcasted(NP.poisson(:y, :rate))),
        site_order=(:tau, :varying, :y))
    invalid_poisson_rate_plan = NP.compile(
        invalid_poisson_rate_input,
        (; group=grouped_bindings.group, rate=[1.0, -1.0, 2.0, 3.0]))
    @test_throws ArgumentError NP.prepare(invalid_poisson_rate_plan)
    three_coefficient_group_declaration = NP.model(
        inputs=(; x=NP.input(), w=NP.input(), group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:Intercept, :x, :w);
                transform=NP.Exp(), prior=NP.Exponential(1)),
            correlation=NP.cholesky_correlation(
                (:Intercept, :x, :w), 2.0),
            z=NP.grouped_standard_normal(
                :group, (:Intercept, :x, :w)),
            beta=NP.parameter(
                NP.RealSupport(), (:x, :w); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(;
            correlated_by_row=NP.grouped_affine(
                :z, :tau, :correlation, :group,
                (nothing, :x, :w)),
            mu=NP.affine(
                (:x, :w), :beta; offsets=(:correlated_by_row,),
                intercept=false)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :sigma))),
        site_order=(:tau, :correlation, :z, :beta, :sigma, :y))
    three_coefficient_group_plan = NP.compile(
        three_coefficient_group_declaration,
        (; x=sampled_offset_data.x,
           w=[0.5, -1.0, 1.5, 2.0],
           group=grouped_bindings.group);
        conditions=(; y=sampled_offset_data.y))
    @test three_coefficient_group_plan isa NP.FactorPlan
    @test three_coefficient_group_plan.graph.dimension == 18
    @test three_coefficient_group_plan.graph.coordinates.correlation.keys == (
        NP.CorrelationCoordinateKey(:correlation, 2, 1),
        NP.CorrelationCoordinateKey(:correlation, 3, 1),
        NP.CorrelationCoordinateKey(:correlation, 3, 2))
    @test three_coefficient_group_plan.graph.coordinates.z.keys == (
        NP.GroupCoefficientKey(:z, :a, :Intercept),
        NP.GroupCoefficientKey(:z, :a, :x),
        NP.GroupCoefficientKey(:z, :a, :w),
        NP.GroupCoefficientKey(:z, :b, :Intercept),
        NP.GroupCoefficientKey(:z, :b, :x),
        NP.GroupCoefficientKey(:z, :b, :w),
        NP.GroupCoefficientKey(:z, :c, :Intercept),
        NP.GroupCoefficientKey(:z, :c, :x),
        NP.GroupCoefficientKey(:z, :c, :w))
    @test NP.factor_node_dependencies(
        three_coefficient_group_plan.graph.nodes.correlated_by_row) ==
          (:z, :tau, :correlation)
    mismatched_three_correlation = NP.model(
        inputs=three_coefficient_group_declaration.inputs,
        parameters=merge(
            three_coefficient_group_declaration.parameters,
            (; correlation=NP.cholesky_correlation(
                (:Intercept, :x), 2.0))),
        nodes=three_coefficient_group_declaration.nodes,
        observations=three_coefficient_group_declaration.observations,
        site_order=three_coefficient_group_declaration.site_order)
    @test capability_error(() -> NP.compile(
        mismatched_three_correlation,
        (; x=sampled_offset_data.x,
           w=[0.5, -1.0, 1.5, 2.0],
           group=grouped_bindings.group);
        conditions=(; y=sampled_offset_data.y))).capability == :factor_nodes

    correlated_group_prepared = NP.prepare(correlated_group_plan)
    correlated_group_workspace = NP.workspace(
        correlated_group_prepared, Float64, DI.AutoEnzyme())
    correlated_prior_rng = MersenneTwister(949)
    correlated_prior_expected_rng = MersenneTwister(949)
    @test capability_error(() -> NP.simulate_prior(
        correlated_prior_rng, correlated_group_workspace,
        correlated_group_prepared)).capability == :prior_simulation
    @test randn(correlated_prior_rng) == randn(correlated_prior_expected_rng)
    correlated_group_position = [
        log(0.6), log(0.4), 0.25,
        0.2, -0.3, -0.1, 0.5, 0.4, -0.2,
        0.7, log(0.5)]
    correlated_tau = exp.(correlated_group_position[1:2])
    correlated_raw = correlated_group_position[3]
    correlated_rho = tanh(correlated_raw)
    correlated_sech = 1 / cosh(correlated_raw)
    correlated_z = reshape(correlated_group_position[4:9], 2, 3)
    correlated_effects = [
        (correlated_tau[1] * correlated_z[1, group],
         correlated_tau[2] *
            (correlated_rho * correlated_z[1, group] +
             correlated_sech * correlated_z[2, group]))
        for group in 1:3]
    correlated_beta = correlated_group_position[10]
    correlated_sigma = exp(correlated_group_position[11])
    correlated_mu = [
        correlated_beta * sampled_offset_data.x[row] +
        correlated_effects[[1, 2, 1, 3][row]][1] +
        correlated_effects[[1, 2, 1, 3][row]][2] *
            sampled_offset_data.x[row]
        for row in eachindex(sampled_offset_data.x)]
    correlated_residuals = sampled_offset_data.y .- correlated_mu
    correlated_eta = 2.0
    correlated_log_constant =
        BRM.loggamma(correlated_eta + 0.5) -
        BRM.loggamma(correlated_eta) - 0.5 * log(pi)
    correlated_expected_density =
        sum(logpdf.(Exponential(1), correlated_tau)) +
        sum(correlated_group_position[1:2]) +
        correlated_log_constant -
        2 * correlated_eta * log(cosh(correlated_raw)) +
        sum(logpdf.(Normal(), correlated_z)) +
        logpdf(Normal(), correlated_beta) +
        logpdf(Exponential(2), correlated_sigma) +
        correlated_group_position[11] +
        sum(logpdf.(Normal.(correlated_mu, correlated_sigma),
                    sampled_offset_data.y))
    correlated_scores = correlated_residuals ./ correlated_sigma^2
    correlated_intercept_scores = [
        sum(correlated_scores[[1, 3]]),
        correlated_scores[2], correlated_scores[4]]
    correlated_slope_scores = [
        sum(correlated_scores[[1, 3]] .*
            sampled_offset_data.x[[1, 3]]),
        correlated_scores[2] * sampled_offset_data.x[2],
        correlated_scores[4] * sampled_offset_data.x[4]]
    correlated_tau0_gradient =
        1 - correlated_tau[1] + sum(
            correlated_intercept_scores[group] *
            correlated_effects[group][1] for group in 1:3)
    correlated_tau1_gradient =
        1 - correlated_tau[2] + sum(
            correlated_slope_scores[group] *
            correlated_effects[group][2] for group in 1:3)
    correlated_raw_gradient =
        -2 * correlated_eta * correlated_rho + sum(
            correlated_slope_scores[group] * correlated_tau[2] *
            (correlated_sech^2 * correlated_z[1, group] -
             correlated_rho * correlated_sech *
                correlated_z[2, group]) for group in 1:3)
    correlated_z1_gradients = [
        -correlated_z[1, group] +
        correlated_tau[1] * correlated_intercept_scores[group] +
        correlated_tau[2] * correlated_rho *
            correlated_slope_scores[group]
        for group in 1:3]
    correlated_z2_gradients = [
        -correlated_z[2, group] + correlated_tau[2] *
            correlated_sech * correlated_slope_scores[group]
        for group in 1:3]
    correlated_interleaved_gradients = collect(Iterators.flatten(
        (correlated_z1_gradients[group],
         correlated_z2_gradients[group]) for group in 1:3))
    correlated_expected_gradient = [
        correlated_tau0_gradient,
        correlated_tau1_gradient,
        correlated_raw_gradient,
        correlated_interleaved_gradients...,
        -correlated_beta +
            sum(correlated_scores .* sampled_offset_data.x),
        1 - correlated_sigma / 2 - length(sampled_offset_data.y) +
            sum(abs2, correlated_residuals) / correlated_sigma^2]

    correlated_glmm_position = correlated_group_position[1:10]
    correlated_glmm_prior_density =
        sum(logpdf.(Exponential(1), correlated_tau)) +
        sum(correlated_group_position[1:2]) +
        correlated_log_constant -
        2 * correlated_eta * log(cosh(correlated_raw)) +
        sum(logpdf.(Normal(), correlated_z)) +
        logpdf(Normal(), correlated_beta)
    correlated_glmm_gradient = scores -> begin
        intercept_scores = [
            sum(scores[[1, 3]]), scores[2], scores[4]]
        slope_scores = [
            sum(scores[[1, 3]] .* sampled_offset_data.x[[1, 3]]),
            scores[2] * sampled_offset_data.x[2],
            scores[4] * sampled_offset_data.x[4]]
        z1_gradients = [
            -correlated_z[1, group] +
            correlated_tau[1] * intercept_scores[group] +
            correlated_tau[2] * correlated_rho * slope_scores[group]
            for group in 1:3]
        z2_gradients = [
            -correlated_z[2, group] + correlated_tau[2] *
            correlated_sech * slope_scores[group]
            for group in 1:3]
        [
            1 - correlated_tau[1] + sum(
                intercept_scores[group] * correlated_effects[group][1]
                for group in 1:3),
            1 - correlated_tau[2] + sum(
                slope_scores[group] * correlated_effects[group][2]
                for group in 1:3),
            -2 * correlated_eta * correlated_rho + sum(
                slope_scores[group] * correlated_tau[2] *
                (correlated_sech^2 * correlated_z[1, group] -
                 correlated_rho * correlated_sech *
                    correlated_z[2, group]) for group in 1:3),
            collect(Iterators.flatten(
                (z1_gradients[group], z2_gradients[group])
                for group in 1:3))...,
            -correlated_beta + sum(scores .* sampled_offset_data.x)]
    end

    correlated_bernoulli_response = Bool[true, false, true, true]
    correlated_bernoulli_probabilities =
        BRM._native_ppl_logistic.(correlated_mu)
    correlated_bernoulli_pointwise = [
        response ? -BRM._native_ppl_softplus(-logit) :
            -BRM._native_ppl_softplus(logit)
        for (response, logit) in
            zip(correlated_bernoulli_response, correlated_mu)]
    correlated_bernoulli_scores = Float64.(correlated_bernoulli_response) .-
        correlated_bernoulli_probabilities
    correlated_bernoulli_workspace = NP.workspace(
        correlated_bernoulli_prepared, Float64, DI.AutoEnzyme())
    correlated_bernoulli_density, correlated_bernoulli_gradient =
        NP.logdensity_and_gradient!(
            correlated_bernoulli_workspace,
            correlated_bernoulli_prepared, correlated_glmm_position)
    @test correlated_bernoulli_density ≈
          correlated_glmm_prior_density + sum(correlated_bernoulli_pointwise)
    @test correlated_bernoulli_gradient ≈
          correlated_glmm_gradient(correlated_bernoulli_scores)
    @test NP.evaluate(
        correlated_bernoulli_workspace, correlated_bernoulli_prepared,
        correlated_glmm_position, NP.LinearPredictor()) ≈ correlated_mu
    @test NP.evaluate(
        correlated_bernoulli_workspace, correlated_bernoulli_prepared,
        correlated_glmm_position, NP.PointwiseLogLikelihood()) ≈
          correlated_bernoulli_pointwise
    correlated_bernoulli_rng = MersenneTwister(953)
    correlated_bernoulli_expected_rng = MersenneTwister(953)
    correlated_bernoulli_predictive = NP.simulate(
        correlated_bernoulli_rng, correlated_bernoulli_workspace,
        correlated_bernoulli_prepared, correlated_glmm_position)
    @test correlated_bernoulli_predictive == [
        rand(correlated_bernoulli_expected_rng) < probability
        for probability in correlated_bernoulli_probabilities]
    @test factor_steady_state_allocations(
        correlated_bernoulli_workspace, correlated_bernoulli_prepared,
        correlated_glmm_position) == (; primal=0, gradient=0)
    @test_throws ArgumentError NP.rebind(
        correlated_bernoulli_prepared,
        (; y=[true, false, 2, true]))

    correlated_poisson_response = Int[2, 0, 1, 3]
    correlated_poisson_pointwise = [
        BRM._native_ppl_poisson_logdensity(Float64(count), log_rate)
        for (count, log_rate) in
            zip(correlated_poisson_response, correlated_mu)]
    correlated_poisson_scores = correlated_poisson_response .-
        exp.(correlated_mu)
    correlated_poisson_workspace = NP.workspace(
        correlated_poisson_prepared, Float64, DI.AutoEnzyme())
    correlated_poisson_density, correlated_poisson_gradient =
        NP.logdensity_and_gradient!(
            correlated_poisson_workspace, correlated_poisson_prepared,
            correlated_glmm_position)
    @test correlated_poisson_density ≈
          correlated_glmm_prior_density + sum(correlated_poisson_pointwise)
    @test correlated_poisson_gradient ≈
          correlated_glmm_gradient(correlated_poisson_scores)
    @test NP.evaluate(
        correlated_poisson_workspace, correlated_poisson_prepared,
        correlated_glmm_position, NP.LinearPredictor()) ≈ correlated_mu
    @test NP.evaluate(
        correlated_poisson_workspace, correlated_poisson_prepared,
        correlated_glmm_position, NP.PointwiseLogLikelihood()) ≈
          correlated_poisson_pointwise
    correlated_poisson_rng = MersenneTwister(954)
    correlated_poisson_expected_rng = MersenneTwister(954)
    correlated_poisson_predictive = NP.simulate(
        correlated_poisson_rng, correlated_poisson_workspace,
        correlated_poisson_prepared, correlated_glmm_position)
    @test correlated_poisson_predictive == [
        BRM._native_ppl_rand_poisson(
            correlated_poisson_expected_rng, Float64, log_rate)
        for log_rate in correlated_mu]
    @test factor_steady_state_allocations(
        correlated_poisson_workspace, correlated_poisson_prepared,
        correlated_glmm_position) == (; primal=0, gradient=0)
    @test_throws ArgumentError NP.rebind(
        correlated_poisson_prepared, (; y=[2, -1, 1, 3]))
    @test_throws ArgumentError NP.rebind(
        correlated_poisson_prepared, (; y=[2, 0.5, 1, 3]))
    @test_throws ArgumentError NP.rebind(
        correlated_poisson_prepared,
        (; y=[16_777_217, 0, 1, 3]); T=Float32)
    correlated_poisson_extreme = copy(correlated_glmm_position)
    correlated_poisson_extreme[10] = 1000.0
    correlated_poisson_extreme_density = NP.logdensity!(
        correlated_poisson_workspace, correlated_poisson_prepared,
        correlated_poisson_extreme)
    @test correlated_poisson_extreme_density == -Inf

    correlated_density, correlated_gradient = NP.logdensity_and_gradient!(
        correlated_group_workspace, correlated_group_prepared,
        correlated_group_position)
    correlated_gradient = copy(correlated_gradient)
    @test correlated_density ≈ correlated_expected_density
    @test correlated_gradient ≈ correlated_expected_gradient
    @test NP.evaluate(
        correlated_group_workspace, correlated_group_prepared,
        correlated_group_position, NP.LinearPredictor()) ≈ correlated_mu
    @test factor_steady_state_allocations(
        correlated_group_workspace, correlated_group_prepared,
        correlated_group_position) == (; primal=0, gradient=0)
    for raw_correlation in (-1_000.0, 1_000.0)
        extreme_position = copy(correlated_group_position)
        extreme_position[3] = raw_correlation
        extreme_density, extreme_gradient = NP.logdensity_and_gradient!(
            correlated_group_workspace, correlated_group_prepared,
            extreme_position)
        @test isfinite(extreme_density)
        @test all(isfinite, extreme_gradient)
        @test all(isfinite, NP.evaluate(
            correlated_group_workspace, correlated_group_prepared,
            extreme_position, NP.LinearPredictor()))
    end

    three_coefficient_group_prepared = NP.prepare(
        three_coefficient_group_plan)
    three_coefficient_group_workspace = NP.workspace(
        three_coefficient_group_prepared, Float64, DI.AutoEnzyme())
    three_coefficient_group_position = [
        log(0.6), log(0.4), log(0.3),
        0.2, -0.25, 0.35,
        0.1, -0.3, 0.2,
        -0.2, 0.4, -0.1,
        0.5, -0.4, 0.3,
        0.7, -0.4, log(0.5)]
    three_tau = exp.(three_coefficient_group_position[1:3])
    three_raw = three_coefficient_group_position[4:6]
    three_z = reshape(three_coefficient_group_position[7:15], 3, 3)
    three_L31 = tanh(three_raw[2])
    three_L32 = (1 / cosh(three_raw[2])) * tanh(three_raw[3])
    three_L33 = (1 / cosh(three_raw[2])) * (1 / cosh(three_raw[3]))
    three_L = [
        1.0 0.0 0.0;
        tanh(three_raw[1]) 1 / cosh(three_raw[1]) 0.0;
        three_L31 three_L32 three_L33]
    three_effects = [
        three_tau .* (three_L * three_z[:, group])
        for group in 1:3]
    three_w = three_coefficient_group_plan.bindings.w
    three_beta = three_coefficient_group_position[16:17]
    three_sigma = exp(three_coefficient_group_position[18])
    three_mu = [
        three_beta[1] * sampled_offset_data.x[row] +
        three_beta[2] * three_w[row] +
        three_effects[[1, 2, 1, 3][row]][1] +
        three_effects[[1, 2, 1, 3][row]][2] *
            sampled_offset_data.x[row] +
        three_effects[[1, 2, 1, 3][row]][3] * three_w[row]
        for row in eachindex(sampled_offset_data.x)]
    three_eta = 2.0
    three_alpha1 = three_eta + 0.5
    three_alpha2 = three_eta
    three_log_normalizer(alpha) =
        BRM.loggamma(alpha + 0.5) - BRM.loggamma(alpha) - 0.5 * log(pi)
    three_lkj_density =
        2 * three_log_normalizer(three_alpha1) +
        three_log_normalizer(three_alpha2) +
        three_alpha1 * (
            NP._factor_logsech2(three_raw[1]) +
            NP._factor_logsech2(three_raw[2])) +
        three_alpha2 * NP._factor_logsech2(three_raw[3])
    three_expected_density =
        sum(logpdf.(Exponential(1), three_tau)) +
        sum(three_coefficient_group_position[1:3]) +
        three_lkj_density +
        sum(logpdf.(Normal(), three_z)) +
        sum(logpdf.(Normal(), three_beta)) +
        logpdf(Exponential(2), three_sigma) +
        three_coefficient_group_position[18] +
        sum(logpdf.(Normal.(three_mu, three_sigma), sampled_offset_data.y))
    three_density, three_gradient = NP.logdensity_and_gradient!(
        three_coefficient_group_workspace,
        three_coefficient_group_prepared,
        three_coefficient_group_position)
    @test three_density ≈ three_expected_density
    @test NP.evaluate(
        three_coefficient_group_workspace,
        three_coefficient_group_prepared,
        three_coefficient_group_position, NP.LinearPredictor()) ≈ three_mu
    three_finite_difference = similar(three_gradient)
    three_plus = copy(three_coefficient_group_position)
    three_minus = copy(three_coefficient_group_position)
    three_step = 1e-6
    for coordinate in eachindex(three_finite_difference)
        three_plus[coordinate] += three_step
        three_minus[coordinate] -= three_step
        three_finite_difference[coordinate] = (
            NP.logdensity!(
                three_coefficient_group_workspace,
                three_coefficient_group_prepared, three_plus) -
            NP.logdensity!(
                three_coefficient_group_workspace,
                three_coefficient_group_prepared, three_minus)) /
            (2 * three_step)
        three_plus[coordinate] =
            three_coefficient_group_position[coordinate]
        three_minus[coordinate] =
            three_coefficient_group_position[coordinate]
    end
    @test three_gradient ≈ three_finite_difference rtol=2e-5 atol=2e-6
    @test factor_steady_state_allocations(
        three_coefficient_group_workspace,
        three_coefficient_group_prepared,
        three_coefficient_group_position) == (; primal=0, gradient=0)

    # Exercise a dimension beyond the public K=3 example so the static-K
    # executor cannot accidentally be a three-coefficient specialization.
    four_declaration = NP.model(
        inputs=(; x=NP.input(), w=NP.input(), v=NP.input(),
                group=NP.input()),
        parameters=(;
            tau=NP.parameter(
                NP.PositiveSupport(), (:Intercept, :x, :w, :v);
                transform=NP.Exp(), prior=NP.Exponential(1)),
            correlation=NP.cholesky_correlation(
                (:Intercept, :x, :w, :v), 2.0),
            z=NP.grouped_standard_normal(
                :group, (:Intercept, :x, :w, :v)),
            beta=NP.parameter(
                NP.RealSupport(), (:x, :w, :v);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,);
                transform=NP.Exp(), prior=NP.Exponential(2))),
        nodes=(;
            by_row=NP.grouped_affine(
                :z, :tau, :correlation, :group,
                (nothing, :x, :w, :v)),
            mu=NP.affine(
                (:x, :w, :v), :beta;
                offsets=(:by_row,), intercept=false)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :sigma))),
        site_order=(:tau, :correlation, :z, :beta, :sigma, :y))
    four_bindings = (;
        x=[0.2, -0.3], w=[0.5, 0.1], v=[-0.1, 0.4],
        group=[:a, :a])
    four_plan = NP.compile(
        four_declaration, four_bindings; conditions=(; y=[0.1, -0.2]))
    @test four_plan.graph.dimension == 18
    @test length(four_plan.graph.coordinates.correlation.keys) == 6
    four_prepared = NP.prepare(four_plan)
    four_workspace = NP.workspace(
        four_prepared, Float64, DI.AutoEnzyme())
    four_position = [
        log(0.5), log(0.6), log(0.7), log(0.8),
        0.1, -0.2, 0.3, -0.15, 0.25, -0.35,
        0.1, -0.2, 0.3, -0.4,
        0.2, -0.1, 0.3, log(0.5)]
    four_tau = exp.(four_position[1:4])
    four_raw = four_position[5:10]
    four_L = zeros(4, 4)
    four_raw_index = 1
    for column in 1:3
        residuals = ones(4)
        for previous in 1:(column - 1), row in (column + 1):4
            previous_raw_index =
                sum((4 - prior for prior in 1:(previous - 1)); init=0) +
                row - previous
            residuals[row] *= 1 / cosh(four_raw[previous_raw_index])
        end
        for row in (column + 1):4
            four_L[row, column] =
                residuals[row] * tanh(four_raw[four_raw_index])
            four_raw_index += 1
        end
    end
    for row in 1:4
        four_L[row, row] = prod(
            1 / cosh(four_raw[
                sum((4 - prior for prior in 1:(column - 1)); init=0) +
                row - column])
            for column in 1:(row - 1);
            init=1.0)
    end
    four_effect = four_tau .* (four_L * four_position[11:14])
    four_beta = four_position[15:17]
    four_mu = [
        four_beta[1] * four_bindings.x[row] +
        four_beta[2] * four_bindings.w[row] +
        four_beta[3] * four_bindings.v[row] +
        four_effect[1] +
        four_effect[2] * four_bindings.x[row] +
        four_effect[3] * four_bindings.w[row] +
        four_effect[4] * four_bindings.v[row]
        for row in eachindex(four_bindings.x)]
    four_lkj_density = 0.0
    four_raw_index = 1
    for column in 1:3
        alpha = 2.0 + (4 - column - 1) / 2
        for _ in (column + 1):4
            four_lkj_density += three_log_normalizer(alpha) +
                alpha * NP._factor_logsech2(four_raw[four_raw_index])
            four_raw_index += 1
        end
    end
    four_expected_density =
        sum(logpdf.(Exponential(1), four_tau)) +
        sum(four_position[1:4]) + four_lkj_density +
        sum(logpdf.(Normal(), four_position[11:14])) +
        sum(logpdf.(Normal(), four_beta)) +
        logpdf(Exponential(2), exp(four_position[18])) +
        four_position[18] +
        sum(logpdf.(Normal.(four_mu, exp(four_position[18])), [0.1, -0.2]))
    four_density, four_gradient = NP.logdensity_and_gradient!(
        four_workspace, four_prepared, four_position)
    @test four_density ≈ four_expected_density
    @test NP.evaluate(
        four_workspace, four_prepared, four_position,
        NP.LinearPredictor()) ≈ four_mu
    four_finite_difference = similar(four_gradient)
    four_plus = copy(four_position)
    four_minus = copy(four_position)
    for coordinate in eachindex(four_finite_difference)
        four_plus[coordinate] += three_step
        four_minus[coordinate] -= three_step
        four_finite_difference[coordinate] = (
            NP.logdensity!(
                four_workspace, four_prepared, four_plus) -
            NP.logdensity!(
                four_workspace, four_prepared, four_minus)) /
            (2 * three_step)
        four_plus[coordinate] = four_position[coordinate]
        four_minus[coordinate] = four_position[coordinate]
    end
    @test four_gradient ≈ four_finite_difference rtol=2e-5 atol=2e-6
    @test factor_steady_state_allocations(
        four_workspace, four_prepared, four_position) ==
          (; primal=0, gradient=0)

    three_group_data = (;
        x=sampled_offset_data.x,
        w=three_w,
        group=grouped_bindings.group,
        y=sampled_offset_data.y)
    three_group_brm = @brm three_group_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + w + (1 + x + w | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(3, 2)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(three_group_brm, :mu) == [:x, :w]
    @test ranefcoefnames(three_group_brm, :p) == [
        (; predictor=:mu, coefficient=:Intercept),
        (; predictor=:mu, coefficient=:x),
        (; predictor=:mu, coefficient=:w)]
    @test SBBRMI(three_group_brm; mod=@__MODULE__) isa SBBRMI
    natural_three_group = NP.condition(
        natural_correlated_varying_three(
            three_group_data.x, three_group_data.w,
            three_group_data.group);
        y=three_group_data.y)
    three_group_model = NP.lower(three_group_brm)
    @test three_group_model == natural_three_group.declaration
    @test three_group_model.parameters.beta_mu.axis_keys == (:x, :w)
    @test three_group_model.parameters.tau_p_group.axis_keys ==
          (:Intercept, :x, :w)
    for candidate in (natural_three_group, three_group_brm)
        candidate_prepared = NP.prepare(candidate)
        candidate_workspace = NP.workspace(
            candidate_prepared, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient =
            NP.logdensity_and_gradient!(
                candidate_workspace, candidate_prepared,
                three_coefficient_group_position)
        @test candidate_density ≈ three_density
        @test candidate_gradient ≈ three_gradient
        @test NP.evaluate(
            candidate_workspace, candidate_prepared,
            three_coefficient_group_position,
            NP.LinearPredictor()) ≈ three_mu
        @test factor_steady_state_allocations(
            candidate_workspace, candidate_prepared,
            three_coefficient_group_position) ==
              (; primal=0, gradient=0)
    end

    three_group_prepared = NP.prepare(three_group_brm)
    three_known_bindings = (;
        x=[2.5, -0.5, 1.0],
        w=[-0.25, 1.5, 0.75],
        group=[:c, :a, :b])
    three_known_replay = NP.rebind(
        three_group_prepared, (;); bindings=three_known_bindings)
    @test three_known_replay.plan.graph.dimension == 18
    @test three_known_replay.plan.group_indices ==
          (; b_p_group_by_group_for_mu=(3, 1, 2))
    three_known_mu = [
        three_beta[1] * three_known_bindings.x[row] +
        three_beta[2] * three_known_bindings.w[row] +
        three_effects[[3, 1, 2][row]][1] +
        three_effects[[3, 1, 2][row]][2] *
            three_known_bindings.x[row] +
        three_effects[[3, 1, 2][row]][3] *
            three_known_bindings.w[row]
        for row in eachindex(three_known_bindings.x)]
    @test NP.evaluate(
        NP.workspace(three_known_replay), three_known_replay,
        three_coefficient_group_position,
        NP.LinearPredictor()) ≈ three_known_mu

    three_new_bindings = (;
        x=[-1.0, 0.5, 1.5, 2.0],
        w=[0.25, -0.75, 1.25, 0.5],
        group=[:a, :d, :d, :c])
    three_new_replay = NP.rebind(
        three_group_prepared, (;); bindings=three_new_bindings,
        new_groups=:resample)
    @test three_new_replay.plan.graph.dimension == 18
    @test three_new_replay.plan.generated_group_levels ==
          (; b_p_group=(:d,))
    @test three_new_replay.plan.generated_group_indices ==
          (; b_p_group=1:3)
    @test three_new_replay.plan.group_indices ==
          (; b_p_group_by_group_for_mu=(1, -1, -1, 3))
    three_new_workspace = NP.workspace(three_new_replay)
    three_new_rng = MersenneTwister(970)
    three_new_expected_rng = MersenneTwister(970)
    three_new_z = randn(three_new_expected_rng, 3)
    three_new_effect = three_tau .* (three_L * three_new_z)
    three_new_effects = [
        three_effects[1], three_new_effect,
        three_new_effect, three_effects[3]]
    three_new_mu = [
        three_beta[1] * three_new_bindings.x[row] +
        three_beta[2] * three_new_bindings.w[row] +
        three_new_effects[row][1] +
        three_new_effects[row][2] * three_new_bindings.x[row] +
        three_new_effects[row][3] * three_new_bindings.w[row]
        for row in eachindex(three_new_bindings.x)]
    three_new_linear = zeros(4)
    NP.evaluate!(
        three_new_rng, three_new_linear,
        three_new_workspace, three_new_replay,
        three_coefficient_group_position,
        NP.LinearPredictor())
    @test three_new_linear ≈ three_new_mu
    @test three_new_workspace.primal.generated_group_values ≈ three_new_z
    @test_throws NP.CapabilityError NP.logdensity!(
        three_new_workspace, three_new_replay,
        three_coefficient_group_position)

    three_draw_positions = [
        three_coefficient_group_position';
        (three_coefficient_group_position .+
         [0.02, -0.03, 0.01, 0.04, -0.02, 0.03,
          0.01, -0.01, 0.02, -0.02, 0.01, -0.01,
          0.03, -0.02, 0.01, 0.04, -0.03, 0.02])']
    three_draw_predictive = zeros(2, 4)
    three_draw_linear = zeros(2, 4)
    three_manual_predictive = zeros(2, 4)
    three_manual_linear = zeros(2, 4)
    three_manual_fused_linear = zeros(2, 4)
    three_draw_rng = MersenneTwister(971)
    three_manual_rng = MersenneTwister(971)
    for draw in axes(three_draw_positions, 1)
        NP.simulate!(
            three_manual_rng,
            @view(three_manual_predictive[draw, :]),
            three_new_workspace, three_new_replay,
            @view(three_draw_positions[draw, :]))
        three_manual_fused_linear[draw, :] .=
            @view three_new_workspace.primal.node_rows[2, :]
    end
    NP.simulate_draws!(
        three_draw_rng, three_draw_predictive,
        three_new_workspace, three_new_replay,
        three_draw_positions)
    @test three_draw_predictive == three_manual_predictive
    NP.evaluate_draws!(
        MersenneTwister(972), three_draw_linear,
        three_new_workspace, three_new_replay,
        three_draw_positions, NP.LinearPredictor())
    three_linear_rng = MersenneTwister(972)
    for draw in axes(three_draw_positions, 1)
        NP.evaluate!(
            three_linear_rng,
            @view(three_manual_linear[draw, :]),
            three_new_workspace, three_new_replay,
            @view(three_draw_positions[draw, :]),
            NP.LinearPredictor())
    end
    @test three_draw_linear == three_manual_linear
    three_queries = (;
        linear=NP.LinearPredictor(),
        predictive=NP.PosteriorPredictive())
    three_bundle = (;
        linear=zeros(2, 4), predictive=zeros(2, 4))
    NP.execute_draws!(
        MersenneTwister(971), three_bundle,
        three_new_workspace, three_new_replay,
        three_draw_positions, three_queries)
    @test three_bundle.linear == three_manual_fused_linear
    @test three_bundle.predictive == three_manual_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(973), MersenneTwister(974),
        MersenneTwister(975),
        three_draw_predictive, three_draw_linear,
        three_bundle, three_new_workspace,
        three_new_replay, three_draw_positions,
        three_queries) ==
          (; predictive=0, linear=0, bundle=0)

    crossed_group_data = (;
        x=[-1.0, 0.0, 1.0, 2.0, -0.5, 0.5],
        subject=[:s1, :s2, :s1, :s3, :s2, :s3],
        item=[:i1, :i1, :i2, :i2, :i3, :i3],
        y=[0.1, 0.4, 0.8, 1.0, 0.2, 0.7])
    crossed_group_brm = @brm crossed_group_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (1 + x | p | subject) + (1 | q | item)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        sd(:, q) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test SBBRMI(crossed_group_brm; mod=@__MODULE__) isa SBBRMI
    crossed_group_model = NP.lower(crossed_group_brm)
    @test keys(crossed_group_model.inputs) == (:x, :subject, :item)
    @test keys(crossed_group_model.parameters) == (
        :beta_mu, :tau_p_subject, :L_p_subject, :b_p_subject,
        :tau_q_item, :b_q_item, :sigma)
    @test keys(crossed_group_model.nodes) == (
        :b_p_subject_by_subject_for_mu, :r_mu_q_item, :mu)
    @test crossed_group_model.site_order == (
        :tau_p_subject, :L_p_subject, :b_p_subject,
        :tau_q_item, :b_q_item, :beta_mu, :sigma, :y)
    crossed_group_plan = NP.compile(crossed_group_brm)
    @test crossed_group_plan.graph.dimension == 15
    @test crossed_group_plan.group_indices == (
        b_p_subject_by_subject_for_mu=(1, 2, 1, 3, 2, 3),
        r_mu_q_item=(1, 1, 2, 2, 3, 3))
    @test crossed_group_plan.graph.schedule == (
        :tau_p_subject, :L_p_subject, :b_p_subject,
        :tau_q_item, :b_q_item, :beta_mu, :sigma,
        :b_p_subject_by_subject_for_mu, :r_mu_q_item, :mu, :y)
    crossed_group_prepared = NP.prepare(crossed_group_plan)
    crossed_group_workspace = NP.workspace(
        crossed_group_prepared, Float64, DI.AutoEnzyme())
    crossed_group_position = [
        log(0.6), log(0.4), 0.25,
        0.2, -0.3, -0.1, 0.5, 0.4, -0.2,
        log(0.3), 0.1, -0.2, 0.25,
        0.5, log(0.45)]
    crossed_subject_tau = exp.(crossed_group_position[1:2])
    crossed_subject_raw = crossed_group_position[3]
    crossed_subject_rho = tanh(crossed_subject_raw)
    crossed_subject_sech = 1 / cosh(crossed_subject_raw)
    crossed_subject_z = reshape(crossed_group_position[4:9], 2, 3)
    crossed_subject_effects = [
        (crossed_subject_tau[1] * crossed_subject_z[1, group],
         crossed_subject_tau[2] *
            (crossed_subject_rho * crossed_subject_z[1, group] +
             crossed_subject_sech * crossed_subject_z[2, group]))
        for group in 1:3]
    crossed_item_tau = exp(crossed_group_position[10])
    crossed_item_effects = crossed_group_position[11:13]
    crossed_beta = crossed_group_position[14]
    crossed_sigma = exp(crossed_group_position[15])
    crossed_mu = [
        crossed_beta * crossed_group_data.x[row] +
        crossed_subject_effects[[1, 2, 1, 3, 2, 3][row]][1] +
        crossed_subject_effects[[1, 2, 1, 3, 2, 3][row]][2] *
            crossed_group_data.x[row] +
        crossed_item_effects[[1, 1, 2, 2, 3, 3][row]]
        for row in eachindex(crossed_group_data.x)]
    crossed_lkj_constant = BRM.loggamma(2.5) - BRM.loggamma(2.0) -
        0.5 * log(pi)
    crossed_expected_density =
        sum(logpdf.(Exponential(1), crossed_subject_tau)) +
        sum(crossed_group_position[1:2]) +
        crossed_lkj_constant +
        2 * NP._factor_logsech2(crossed_subject_raw) +
        sum(logpdf.(Normal(), crossed_subject_z)) +
        logpdf(Exponential(1), crossed_item_tau) +
        crossed_group_position[10] +
        sum(logpdf.(Normal(0, crossed_item_tau), crossed_item_effects)) +
        logpdf(Normal(), crossed_beta) +
        logpdf(Exponential(2), crossed_sigma) +
        crossed_group_position[15] +
        sum(logpdf.(Normal.(crossed_mu, crossed_sigma),
                    crossed_group_data.y))
    crossed_density, crossed_gradient = NP.logdensity_and_gradient!(
        crossed_group_workspace, crossed_group_prepared,
        crossed_group_position)
    @test crossed_density ≈ crossed_expected_density
    @test NP.evaluate(
        crossed_group_workspace, crossed_group_prepared,
        crossed_group_position, NP.LinearPredictor()) ≈ crossed_mu
    @test NP.evaluate(
        crossed_group_workspace, crossed_group_prepared,
        crossed_group_position, NP.PointwiseLogLikelihood()) ≈
          logpdf.(Normal.(crossed_mu, crossed_sigma), crossed_group_data.y)
    crossed_finite_difference = similar(crossed_gradient)
    crossed_plus = copy(crossed_group_position)
    crossed_minus = copy(crossed_group_position)
    crossed_step = 1e-6
    for coordinate in eachindex(crossed_finite_difference)
        crossed_plus[coordinate] += crossed_step
        crossed_minus[coordinate] -= crossed_step
        crossed_finite_difference[coordinate] = (
            NP.logdensity!(
                crossed_group_workspace, crossed_group_prepared,
                crossed_plus) -
            NP.logdensity!(
                crossed_group_workspace, crossed_group_prepared,
                crossed_minus)) / (2 * crossed_step)
        crossed_plus[coordinate] = crossed_group_position[coordinate]
        crossed_minus[coordinate] = crossed_group_position[coordinate]
    end
    @test crossed_gradient ≈ crossed_finite_difference rtol=2e-5 atol=2e-6
    @test factor_steady_state_allocations(
        crossed_group_workspace, crossed_group_prepared,
        crossed_group_position) == (; primal=0, gradient=0)
    natural_crossed_group = NP.condition(
        natural_crossed_group_regression(
            crossed_group_data.x, crossed_group_data.subject,
            crossed_group_data.item);
        y=crossed_group_data.y)
    @test natural_crossed_group.declaration == crossed_group_model
    natural_crossed_prepared = NP.prepare(natural_crossed_group)
    natural_crossed_workspace = NP.workspace(
        natural_crossed_prepared, Float64, DI.AutoEnzyme())
    natural_crossed_density, natural_crossed_gradient =
        NP.logdensity_and_gradient!(
            natural_crossed_workspace, natural_crossed_prepared,
            crossed_group_position)
    @test natural_crossed_density ≈ crossed_density
    @test natural_crossed_gradient ≈ crossed_gradient
    @test NP.evaluate(
        natural_crossed_workspace, natural_crossed_prepared,
        crossed_group_position, NP.LinearPredictor()) ≈ crossed_mu
    @test factor_steady_state_allocations(
        natural_crossed_workspace, natural_crossed_prepared,
        crossed_group_position) == (; primal=0, gradient=0)

    crossed_known_bindings = (;
        x=[1.5, -0.5, 0.75, 2.0],
        subject=[:s3, :s1, :s2, :s3],
        item=[:i2, :i3, :i1, :i2])
    crossed_known_replay = NP.rebind(
        crossed_group_prepared, (;); bindings=crossed_known_bindings)
    @test crossed_known_replay.plan.graph.dimension ==
          crossed_group_plan.graph.dimension
    @test crossed_known_replay.plan.group_indices == (
        b_p_subject_by_subject_for_mu=(3, 1, 2, 3),
        r_mu_q_item=(2, 3, 1, 2))
    crossed_known_mu = [
        crossed_beta * crossed_known_bindings.x[row] +
        crossed_subject_effects[[3, 1, 2, 3][row]][1] +
        crossed_subject_effects[[3, 1, 2, 3][row]][2] *
            crossed_known_bindings.x[row] +
        crossed_item_effects[[2, 3, 1, 2][row]]
        for row in eachindex(crossed_known_bindings.x)]
    @test NP.evaluate(
        NP.workspace(crossed_known_replay), crossed_known_replay,
        crossed_group_position, NP.LinearPredictor()) ≈ crossed_known_mu

    crossed_subject_only_bindings = (;
        x=[0.0, 1.0], subject=[:s4, :s1], item=[:i1, :i2])
    crossed_subject_only = NP.rebind(
        crossed_group_prepared, (;);
        bindings=crossed_subject_only_bindings, new_groups=:resample)
    @test crossed_subject_only.plan.generated_group_levels ==
          (; b_p_subject=(:s4,), b_q_item=())
    @test crossed_subject_only.plan.generated_group_indices.b_p_subject == 1:2
    @test isempty(
        crossed_subject_only.plan.generated_group_indices.b_q_item)
    @test crossed_subject_only.plan.group_indices == (
        b_p_subject_by_subject_for_mu=(-1, 1),
        r_mu_q_item=(1, 2))

    crossed_item_only_bindings = (;
        x=[0.0, 1.0], subject=[:s1, :s2], item=[:i4, :i2])
    crossed_item_only = NP.rebind(
        crossed_group_prepared, (;);
        bindings=crossed_item_only_bindings, new_groups=:resample)
    @test crossed_item_only.plan.generated_group_levels ==
          (; b_p_subject=(), b_q_item=(:i4,))
    @test isempty(
        crossed_item_only.plan.generated_group_indices.b_p_subject)
    @test crossed_item_only.plan.generated_group_indices.b_q_item == 1:1
    @test crossed_item_only.plan.group_indices == (
        b_p_subject_by_subject_for_mu=(1, 2),
        r_mu_q_item=(-1, 2))

    crossed_new_bindings = (;
        x=[-1.0, 0.5, 1.5, 2.0, -0.25],
        subject=[:s1, :s4, :s4, :s3, :s2],
        item=[:i4, :i2, :i4, :i3, :i2])
    @test_throws NP.CapabilityError NP.rebind(
        crossed_group_prepared, (;); bindings=crossed_new_bindings)
    crossed_new_replay = NP.rebind(
        crossed_group_prepared, (;); bindings=crossed_new_bindings,
        new_groups=:resample)
    @test crossed_new_replay.plan.graph.dimension ==
          crossed_group_plan.graph.dimension
    @test crossed_new_replay.plan.generated_group_levels ==
          (; b_p_subject=(:s4,), b_q_item=(:i4,))
    @test crossed_new_replay.plan.generated_group_indices ==
          (; b_p_subject=1:2, b_q_item=3:3)
    @test crossed_new_replay.plan.group_indices == (
        b_p_subject_by_subject_for_mu=(1, -1, -1, 3, 2),
        r_mu_q_item=(-1, 2, -1, 3, 2))
    @test_throws NP.CapabilityError NP.rebind(
        crossed_group_prepared, (; y=zeros(5));
        bindings=crossed_new_bindings, new_groups=:resample)
    crossed_new_workspace = NP.workspace(crossed_new_replay)
    crossed_new_rng = MersenneTwister(980)
    crossed_new_expected_rng = MersenneTwister(980)
    crossed_new_subject_z = (
        randn(crossed_new_expected_rng),
        randn(crossed_new_expected_rng))
    crossed_new_subject_effect = (
        crossed_subject_tau[1] * crossed_new_subject_z[1],
        crossed_subject_tau[2] *
            (crossed_subject_rho * crossed_new_subject_z[1] +
             crossed_subject_sech * crossed_new_subject_z[2]))
    crossed_new_item_effect =
        crossed_item_tau * randn(crossed_new_expected_rng)
    crossed_new_subject_effects = [
        crossed_subject_effects[1], crossed_new_subject_effect,
        crossed_new_subject_effect, crossed_subject_effects[3],
        crossed_subject_effects[2]]
    crossed_new_item_effects = [
        crossed_new_item_effect, crossed_item_effects[2],
        crossed_new_item_effect, crossed_item_effects[3],
        crossed_item_effects[2]]
    crossed_new_mu = [
        crossed_beta * crossed_new_bindings.x[row] +
        crossed_new_subject_effects[row][1] +
        crossed_new_subject_effects[row][2] *
            crossed_new_bindings.x[row] +
        crossed_new_item_effects[row]
        for row in eachindex(crossed_new_bindings.x)]
    crossed_new_expected = [
        location + crossed_sigma * randn(crossed_new_expected_rng)
        for location in crossed_new_mu]
    crossed_new_output = zeros(5)
    NP.simulate!(
        crossed_new_rng, crossed_new_output,
        crossed_new_workspace, crossed_new_replay,
        crossed_group_position)
    @test crossed_new_output ≈ crossed_new_expected
    @test crossed_new_workspace.primal.generated_group_values ≈
          [crossed_new_subject_z..., crossed_new_item_effect]
    @test vec(crossed_new_workspace.primal.node_rows[3, :]) ≈
          crossed_new_mu
    @test factor_predictive_allocations(
        MersenneTwister(981), crossed_new_output,
        crossed_new_workspace, crossed_new_replay,
        crossed_group_position) == 0
    @test_throws NP.CapabilityError NP.logdensity!(
        crossed_new_workspace, crossed_new_replay,
        crossed_group_position)
    @test_throws NP.CapabilityError NP.evaluate!(
        similar(crossed_new_output), crossed_new_workspace,
        crossed_new_replay, crossed_group_position,
        NP.LinearPredictor())
    crossed_new_linear = zeros(5)
    NP.evaluate!(
        MersenneTwister(982), crossed_new_linear,
        crossed_new_workspace, crossed_new_replay,
        crossed_group_position, NP.LinearPredictor())
    crossed_new_linear_expected_rng = MersenneTwister(982)
    crossed_linear_subject_z = (
        randn(crossed_new_linear_expected_rng),
        randn(crossed_new_linear_expected_rng))
    crossed_linear_subject_effect = (
        crossed_subject_tau[1] * crossed_linear_subject_z[1],
        crossed_subject_tau[2] *
            (crossed_subject_rho * crossed_linear_subject_z[1] +
             crossed_subject_sech * crossed_linear_subject_z[2]))
    crossed_linear_item_effect =
        crossed_item_tau * randn(crossed_new_linear_expected_rng)
    crossed_linear_subject_effects = [
        crossed_subject_effects[1], crossed_linear_subject_effect,
        crossed_linear_subject_effect, crossed_subject_effects[3],
        crossed_subject_effects[2]]
    crossed_linear_item_effects = [
        crossed_linear_item_effect, crossed_item_effects[2],
        crossed_linear_item_effect, crossed_item_effects[3],
        crossed_item_effects[2]]
    @test crossed_new_linear ≈ [
        crossed_beta * crossed_new_bindings.x[row] +
        crossed_linear_subject_effects[row][1] +
        crossed_linear_subject_effects[row][2] *
            crossed_new_bindings.x[row] +
        crossed_linear_item_effects[row]
        for row in eachindex(crossed_new_bindings.x)]

    crossed_draw_positions = [
        crossed_group_position';
        (crossed_group_position .+
         [0.02, -0.03, 0.01, 0.04, -0.02,
          0.03, -0.01, 0.02, -0.04, 0.01,
          0.03, -0.02, 0.01, 0.05, -0.03])']
    crossed_draw_predictive = zeros(2, 5)
    crossed_draw_linear = zeros(2, 5)
    crossed_manual_predictive = zeros(2, 5)
    crossed_manual_linear = zeros(2, 5)
    crossed_manual_fused_linear = zeros(2, 5)
    crossed_draw_rng = MersenneTwister(983)
    crossed_manual_rng = MersenneTwister(983)
    for draw in axes(crossed_draw_positions, 1)
        NP.simulate!(
            crossed_manual_rng,
            @view(crossed_manual_predictive[draw, :]),
            crossed_new_workspace, crossed_new_replay,
            @view(crossed_draw_positions[draw, :]))
        crossed_manual_fused_linear[draw, :] .=
            @view crossed_new_workspace.primal.node_rows[3, :]
    end
    NP.simulate_draws!(
        crossed_draw_rng, crossed_draw_predictive,
        crossed_new_workspace, crossed_new_replay,
        crossed_draw_positions)
    @test crossed_draw_predictive == crossed_manual_predictive
    NP.evaluate_draws!(
        MersenneTwister(984), crossed_draw_linear,
        crossed_new_workspace, crossed_new_replay,
        crossed_draw_positions, NP.LinearPredictor())
    crossed_linear_rng = MersenneTwister(984)
    for draw in axes(crossed_draw_positions, 1)
        NP.evaluate!(
            crossed_linear_rng,
            @view(crossed_manual_linear[draw, :]),
            crossed_new_workspace, crossed_new_replay,
            @view(crossed_draw_positions[draw, :]),
            NP.LinearPredictor())
    end
    @test crossed_draw_linear == crossed_manual_linear
    crossed_queries = (;
        linear=NP.LinearPredictor(),
        predictive=NP.PosteriorPredictive())
    crossed_bundle = (;
        linear=zeros(2, 5), predictive=zeros(2, 5))
    NP.execute_draws!(
        MersenneTwister(983), crossed_bundle,
        crossed_new_workspace, crossed_new_replay,
        crossed_draw_positions, crossed_queries)
    @test crossed_bundle.linear == crossed_manual_fused_linear
    @test crossed_bundle.predictive == crossed_manual_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(985), MersenneTwister(986),
        MersenneTwister(987),
        crossed_draw_predictive, crossed_draw_linear,
        crossed_bundle, crossed_new_workspace,
        crossed_new_replay, crossed_draw_positions,
        crossed_queries) ==
          (; predictive=0, linear=0, bundle=0)

    varying_brm_data = (;
        x=sampled_offset_data.x,
        group=grouped_bindings.group,
        y=sampled_offset_data.y)
    correlated_varying_brm = @brm varying_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (1 + x | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(correlated_varying_brm, :mu) == [:x]
    @test ranefcoefnames(correlated_varying_brm, :p) == [
        (; predictor=:mu, coefficient=:Intercept),
        (; predictor=:mu, coefficient=:x)]
    @test SBBRMI(correlated_varying_brm; mod=@__MODULE__) isa SBBRMI
    correlated_varying_model = NP.lower(correlated_varying_brm)
    natural_correlated_varying = NP.condition(
        natural_correlated_varying_intercept_slope(
            varying_brm_data.x, varying_brm_data.group);
        y=varying_brm_data.y)
    @test typeof(correlated_varying_model) ===
          typeof(natural_correlated_varying.declaration)
    @test sprint(show, correlated_varying_model) ==
          sprint(show, natural_correlated_varying.declaration)
    @test keys(correlated_varying_model.parameters) == (
        :beta_mu, :tau_p_group, :L_p_group, :b_p_group, :sigma)
    @test keys(correlated_varying_model.nodes) ==
          (:b_p_group_by_group_for_mu, :mu)
    correlated_varying_plan = NP.compile(correlated_varying_brm)
    @test correlated_varying_plan.graph.schedule == (
        :tau_p_group, :L_p_group, :b_p_group, :beta_mu, :sigma,
        :b_p_group_by_group_for_mu, :mu, :y)
    @test correlated_varying_plan.graph.dimension == 11
    @test correlated_varying_plan.graph.coordinates.b_p_group.keys == (
        NP.GroupCoefficientKey(:b_p_group, :a, :Intercept),
        NP.GroupCoefficientKey(:b_p_group, :a, :x),
        NP.GroupCoefficientKey(:b_p_group, :b, :Intercept),
        NP.GroupCoefficientKey(:b_p_group, :b, :x),
        NP.GroupCoefficientKey(:b_p_group, :c, :Intercept),
        NP.GroupCoefficientKey(:b_p_group, :c, :x))
    correlated_varying_prepared = NP.prepare(correlated_varying_plan)
    correlated_varying_workspace = NP.workspace(
        correlated_varying_prepared, Float64, DI.AutoEnzyme())
    correlated_varying_density, correlated_varying_gradient =
        NP.logdensity_and_gradient!(
            correlated_varying_workspace, correlated_varying_prepared,
            correlated_group_position)
    @test correlated_varying_density ≈ correlated_density
    @test correlated_varying_gradient ≈ correlated_gradient
    @test NP.evaluate(
        correlated_varying_workspace, correlated_varying_prepared,
        correlated_group_position, NP.LinearPredictor()) ≈ correlated_mu
    @test factor_steady_state_allocations(
        correlated_varying_workspace, correlated_varying_prepared,
        correlated_group_position) == (; primal=0, gradient=0)

    correlated_bernoulli_data = (;
        x=sampled_offset_data.x,
        group=grouped_bindings.group,
        y=correlated_bernoulli_response)
    correlated_bernoulli_brm = @brm correlated_bernoulli_data begin
        mu ~ 0 + x + (1 + x | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ BernoulliLogit(mu)
    end
    @test SBBRMI(correlated_bernoulli_brm; mod=@__MODULE__) isa SBBRMI
    correlated_bernoulli_model = NP.lower(correlated_bernoulli_brm)
    natural_correlated_bernoulli = NP.condition(
        natural_correlated_bernoulli_logit(
            correlated_bernoulli_data.x,
            correlated_bernoulli_data.group);
        y=correlated_bernoulli_data.y)
    @test typeof(correlated_bernoulli_model) ===
          typeof(natural_correlated_bernoulli.declaration)
    @test sprint(show, correlated_bernoulli_model) ==
          sprint(show, natural_correlated_bernoulli.declaration)
    @test keys(correlated_bernoulli_model.parameters) == (
        :beta_mu, :tau_p_group, :L_p_group, :b_p_group)
    @test keys(correlated_bernoulli_model.nodes) ==
          (:b_p_group_by_group_for_mu, :mu)
    correlated_bernoulli_brm_plan = NP.compile(correlated_bernoulli_brm)
    @test correlated_bernoulli_brm_plan.graph.schedule == (
        :tau_p_group, :L_p_group, :b_p_group, :beta_mu,
        :b_p_group_by_group_for_mu, :mu, :y)
    correlated_bernoulli_brm_prepared = NP.prepare(
        correlated_bernoulli_brm_plan)
    correlated_bernoulli_brm_workspace = NP.workspace(
        correlated_bernoulli_brm_prepared, Float64, DI.AutoEnzyme())
    correlated_bernoulli_brm_density,
    correlated_bernoulli_brm_gradient = NP.logdensity_and_gradient!(
        correlated_bernoulli_brm_workspace,
        correlated_bernoulli_brm_prepared, correlated_glmm_position)
    @test correlated_bernoulli_brm_density ≈ correlated_bernoulli_density
    @test correlated_bernoulli_brm_gradient ≈ correlated_bernoulli_gradient
    @test NP.evaluate(
        correlated_bernoulli_brm_workspace,
        correlated_bernoulli_brm_prepared, correlated_glmm_position,
        NP.LinearPredictor()) ≈ correlated_mu

    correlated_poisson_data = (;
        x=sampled_offset_data.x,
        group=grouped_bindings.group,
        y=correlated_poisson_response)
    correlated_poisson_brm = @brm correlated_poisson_data begin
        log_rate ~ 0 + x + (1 + x | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Poisson(exp(log_rate))
    end
    @test SBBRMI(correlated_poisson_brm; mod=@__MODULE__) isa SBBRMI
    correlated_poisson_model = NP.lower(correlated_poisson_brm)
    natural_correlated_poisson = NP.condition(
        natural_correlated_poisson_log(
            correlated_poisson_data.x, correlated_poisson_data.group);
        y=correlated_poisson_data.y)
    @test typeof(correlated_poisson_model) ===
          typeof(natural_correlated_poisson.declaration)
    @test sprint(show, correlated_poisson_model) ==
          sprint(show, natural_correlated_poisson.declaration)
    @test keys(correlated_poisson_model.parameters) == (
        :beta_log_rate, :tau_p_group, :L_p_group, :b_p_group)
    @test keys(correlated_poisson_model.nodes) == (
        :b_p_group_by_group_for_log_rate, :log_rate, :exp_log_rate)
    correlated_poisson_brm_plan = NP.compile(correlated_poisson_brm)
    @test correlated_poisson_brm_plan.graph.schedule == (
        :tau_p_group, :L_p_group, :b_p_group, :beta_log_rate,
        :b_p_group_by_group_for_log_rate, :log_rate, :exp_log_rate, :y)
    correlated_poisson_brm_prepared = NP.prepare(correlated_poisson_brm_plan)
    correlated_poisson_brm_workspace = NP.workspace(
        correlated_poisson_brm_prepared, Float64, DI.AutoEnzyme())
    correlated_poisson_brm_density, correlated_poisson_brm_gradient =
        NP.logdensity_and_gradient!(
            correlated_poisson_brm_workspace,
            correlated_poisson_brm_prepared, correlated_glmm_position)
    @test correlated_poisson_brm_density ≈ correlated_poisson_density
    @test correlated_poisson_brm_gradient ≈ correlated_poisson_gradient
    @test NP.evaluate(
        correlated_poisson_brm_workspace, correlated_poisson_brm_prepared,
        correlated_glmm_position, NP.LinearPredictor()) ≈ correlated_mu

    correlated_known_bindings = (;
        x=[2.5, -0.5, 1.0], group=[:c, :a, :b])
    correlated_known_replay = NP.rebind(
        correlated_varying_prepared, (;);
        bindings=correlated_known_bindings)
    @test correlated_known_replay.plan.graph.dimension == 11
    @test correlated_known_replay.plan.group_indices ==
          (; b_p_group_by_group_for_mu=(3, 1, 2))
    correlated_known_mu = [
        correlated_beta * correlated_known_bindings.x[row] +
        correlated_effects[[3, 1, 2][row]][1] +
        correlated_effects[[3, 1, 2][row]][2] *
            correlated_known_bindings.x[row]
        for row in eachindex(correlated_known_bindings.x)]
    @test NP.evaluate(
        NP.workspace(correlated_known_replay), correlated_known_replay,
        correlated_group_position, NP.LinearPredictor()) ≈
          correlated_known_mu

    correlated_new_bindings = (;
        x=[-1.0, 0.5, 1.5, 2.0], group=[:a, :d, :d, :c])
    correlated_new_replay = NP.rebind(
        correlated_varying_prepared, (;);
        bindings=correlated_new_bindings, new_groups=:resample)
    @test correlated_new_replay.plan.graph.dimension == 11
    @test correlated_new_replay.plan.generated_group_levels ==
          (; b_p_group=(:d,))
    @test correlated_new_replay.plan.generated_group_indices ==
          (; b_p_group=1:2)
    @test correlated_new_replay.plan.group_indices ==
          (; b_p_group_by_group_for_mu=(1, -1, -1, 3))
    correlated_new_workspace = NP.workspace(correlated_new_replay)
    @test length(correlated_new_workspace.primal.generated_group_values) == 2
    correlated_new_rng = MersenneTwister(951)
    correlated_new_expected_rng = MersenneTwister(951)
    correlated_new_z = (
        randn(correlated_new_expected_rng),
        randn(correlated_new_expected_rng))
    correlated_new_effect = (
        correlated_tau[1] * correlated_new_z[1],
        correlated_tau[2] *
            (correlated_rho * correlated_new_z[1] +
             correlated_sech * correlated_new_z[2]))
    correlated_new_effects = [
        correlated_effects[1], correlated_new_effect,
        correlated_new_effect, correlated_effects[3]]
    correlated_new_mu = [
        correlated_beta * correlated_new_bindings.x[row] +
        correlated_new_effects[row][1] +
        correlated_new_effects[row][2] * correlated_new_bindings.x[row]
        for row in eachindex(correlated_new_bindings.x)]
    correlated_new_expected = [
        location + correlated_sigma * randn(correlated_new_expected_rng)
        for location in correlated_new_mu]
    correlated_new_output = zeros(4)
    NP.simulate!(
        correlated_new_rng, correlated_new_output,
        correlated_new_workspace, correlated_new_replay,
        correlated_group_position)
    @test correlated_new_output ≈ correlated_new_expected
    @test correlated_new_workspace.primal.generated_group_values ≈
          collect(correlated_new_z)
    @test vec(correlated_new_workspace.primal.node_rows[1, :]) ≈
          [effect[1] + effect[2] * x for (effect, x) in
           zip(correlated_new_effects, correlated_new_bindings.x)]
    @test_throws NP.CapabilityError NP.logdensity!(
        correlated_new_workspace, correlated_new_replay,
        correlated_group_position)
    @test_throws NP.CapabilityError NP.evaluate!(
        similar(correlated_new_output), correlated_new_workspace,
        correlated_new_replay, correlated_group_position,
        NP.LinearPredictor())
    correlated_new_linear_rng = MersenneTwister(952)
    correlated_new_linear_expected_rng = MersenneTwister(952)
    correlated_linear_z = (
        randn(correlated_new_linear_expected_rng),
        randn(correlated_new_linear_expected_rng))
    correlated_linear_effect = (
        correlated_tau[1] * correlated_linear_z[1],
        correlated_tau[2] *
            (correlated_rho * correlated_linear_z[1] +
             correlated_sech * correlated_linear_z[2]))
    correlated_linear_effects = [
        correlated_effects[1], correlated_linear_effect,
        correlated_linear_effect, correlated_effects[3]]
    correlated_new_linear_expected = [
        correlated_beta * correlated_new_bindings.x[row] +
        correlated_linear_effects[row][1] +
        correlated_linear_effects[row][2] *
            correlated_new_bindings.x[row]
        for row in eachindex(correlated_new_bindings.x)]
    correlated_new_linear = zeros(4)
    NP.evaluate!(
        correlated_new_linear_rng, correlated_new_linear,
        correlated_new_workspace, correlated_new_replay,
        correlated_group_position, NP.LinearPredictor())
    @test correlated_new_linear ≈ correlated_new_linear_expected
    @test factor_predictive_allocations(
        MersenneTwister(953), correlated_new_output,
        correlated_new_workspace, correlated_new_replay,
        correlated_group_position) == 0

    correlated_draw_positions = [
        correlated_group_position';
        (correlated_group_position .+
         [0.05, -0.03, 0.02, 0.01, -0.02, 0.03, -0.01,
          0.02, -0.04, 0.06, -0.02])']
    correlated_draw_predictive = zeros(2, 4)
    correlated_draw_linear = zeros(2, 4)
    correlated_manual_predictive = zeros(2, 4)
    correlated_manual_linear = zeros(2, 4)
    correlated_manual_fused_linear = zeros(2, 4)
    correlated_draw_rng = MersenneTwister(954)
    correlated_manual_rng = MersenneTwister(954)
    for draw in axes(correlated_draw_positions, 1)
        NP.simulate!(
            correlated_manual_rng,
            @view(correlated_manual_predictive[draw, :]),
            correlated_new_workspace, correlated_new_replay,
            @view(correlated_draw_positions[draw, :]))
        correlated_manual_fused_linear[draw, :] .=
            @view correlated_new_workspace.primal.node_rows[2, :]
    end
    NP.simulate_draws!(
        correlated_draw_rng, correlated_draw_predictive,
        correlated_new_workspace, correlated_new_replay,
        correlated_draw_positions)
    @test correlated_draw_predictive == correlated_manual_predictive
    correlated_draw_linear_rng = MersenneTwister(955)
    correlated_manual_linear_rng = MersenneTwister(955)
    for draw in axes(correlated_draw_positions, 1)
        NP.evaluate!(
            correlated_manual_linear_rng,
            @view(correlated_manual_linear[draw, :]),
            correlated_new_workspace, correlated_new_replay,
            @view(correlated_draw_positions[draw, :]),
            NP.LinearPredictor())
    end
    NP.evaluate_draws!(
        correlated_draw_linear_rng, correlated_draw_linear,
        correlated_new_workspace, correlated_new_replay,
        correlated_draw_positions, NP.LinearPredictor())
    @test correlated_draw_linear == correlated_manual_linear
    correlated_queries = (;
        linear=NP.LinearPredictor(),
        predictive=NP.PosteriorPredictive())
    correlated_bundle = (;
        linear=zeros(2, 4), predictive=zeros(2, 4))
    NP.execute_draws!(
        MersenneTwister(954), correlated_bundle,
        correlated_new_workspace, correlated_new_replay,
        correlated_draw_positions, correlated_queries)
    @test correlated_bundle.linear == correlated_manual_fused_linear
    @test correlated_bundle.predictive == correlated_manual_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(956), MersenneTwister(957), MersenneTwister(958),
        correlated_draw_predictive, correlated_draw_linear,
        correlated_bundle, correlated_new_workspace,
        correlated_new_replay, correlated_draw_positions,
        correlated_queries) ==
          (; predictive=0, linear=0, bundle=0)

    correlated_glmm_known_expected = [
        correlated_beta * correlated_known_bindings.x[row] +
        correlated_effects[[3, 1, 2][row]][1] +
        correlated_effects[[3, 1, 2][row]][2] *
            correlated_known_bindings.x[row]
        for row in eachindex(correlated_known_bindings.x)]
    for prepared in (
            correlated_bernoulli_brm_prepared,
            correlated_poisson_brm_prepared)
        known_replay = NP.rebind(
            prepared, (;); bindings=correlated_known_bindings)
        @test known_replay.plan.graph.dimension == 10
        @test NP.evaluate(
            NP.workspace(known_replay), known_replay,
            correlated_glmm_position, NP.LinearPredictor()) ≈
              correlated_glmm_known_expected
    end

    correlated_glmm_new_expected = function(rng, position)
        tau = exp.(position[1:2])
        raw = position[3]
        rho = tanh(raw)
        sech = 1 / cosh(raw)
        z = reshape(position[4:9], 2, 3)
        fitted = [
            (tau[1] * z[1, group],
             tau[2] * (rho * z[1, group] + sech * z[2, group]))
            for group in 1:3]
        generated_z = (randn(rng), randn(rng))
        generated = (
            tau[1] * generated_z[1],
            tau[2] *
                (rho * generated_z[1] + sech * generated_z[2]))
        effects = [fitted[1], generated, generated, fitted[3]]
        beta = position[10]
        linear = [
            beta * correlated_new_bindings.x[row] + effects[row][1] +
            effects[row][2] * correlated_new_bindings.x[row]
            for row in eachindex(correlated_new_bindings.x)]
        (; generated_z, linear)
    end

    correlated_glmm_draw_positions = [
        correlated_glmm_position';
        (correlated_glmm_position .+
         [0.05, -0.03, 0.02, 0.01, -0.02,
          0.03, -0.01, 0.02, -0.04, 0.06])']
    correlated_glmm_cases = (;
        bernoulli=(;
            prepared=correlated_bernoulli_brm_prepared,
            seed=959,
            sample=(rng, linear) -> [
                rand(rng) < BRM._native_ppl_logistic(value)
                for value in linear]),
        poisson=(;
            prepared=correlated_poisson_brm_prepared,
            seed=963,
            sample=(rng, linear) -> [
                BRM._native_ppl_rand_poisson(rng, Float64, value)
                for value in linear]))
    for (family, case) in pairs(correlated_glmm_cases)
        replay = NP.rebind(
            case.prepared, (;); bindings=correlated_new_bindings,
            new_groups=:resample)
        @test replay.plan.graph.dimension == 10
        @test replay.plan.generated_group_levels == (; b_p_group=(:d,))
        @test replay.plan.generated_group_indices == (; b_p_group=1:2)
        group_node = family === :bernoulli ?
            :b_p_group_by_group_for_mu :
            :b_p_group_by_group_for_log_rate
        @test getproperty(replay.plan.group_indices, group_node) ==
              (1, -1, -1, 3)
        work = NP.workspace(replay)
        rng = MersenneTwister(case.seed)
        expected_rng = MersenneTwister(case.seed)
        expected = correlated_glmm_new_expected(
            expected_rng, correlated_glmm_position)
        expected_predictive = case.sample(expected_rng, expected.linear)
        predictive = NP.allocate_output(
            replay, NP.PosteriorPredictive())
        NP.simulate!(
            rng, predictive, work, replay, correlated_glmm_position)
        @test predictive == expected_predictive
        @test work.primal.generated_group_values ≈
              collect(expected.generated_z)
        @test factor_predictive_allocations(
            MersenneTwister(case.seed + 1), predictive, work, replay,
            correlated_glmm_position) == 0
        @test_throws NP.CapabilityError NP.logdensity!(
            work, replay, correlated_glmm_position)
        @test_throws ArgumentError NP.evaluate!(
            zeros(4), work, replay, correlated_glmm_position,
            NP.PointwiseLogLikelihood())

        positions = correlated_glmm_draw_positions
        predictive_signature = NP.batch_output_signature(
            replay, positions, NP.PosteriorPredictive())
        draw_predictive = NP.allocate_output(
            predictive_signature, replay)
        draw_linear = zeros(2, 4)
        manual_predictive = similar(draw_predictive)
        manual_linear = zeros(2, 4)
        manual_fused_linear = zeros(2, 4)
        draw_rng = MersenneTwister(case.seed + 2)
        manual_rng = MersenneTwister(case.seed + 2)
        for draw in axes(positions, 1)
            NP.simulate!(
                manual_rng, @view(manual_predictive[draw, :]),
                work, replay, @view(positions[draw, :]))
            for row in axes(manual_fused_linear, 2)
                manual_fused_linear[draw, row] =
                    NP._factor_terminal_linear(replay, work.primal, row)
            end
        end
        NP.simulate_draws!(
            draw_rng, draw_predictive, work, replay, positions)
        @test draw_predictive == manual_predictive
        linear_rng = MersenneTwister(case.seed + 3)
        manual_linear_rng = MersenneTwister(case.seed + 3)
        for draw in axes(positions, 1)
            NP.evaluate!(
                manual_linear_rng, @view(manual_linear[draw, :]),
                work, replay, @view(positions[draw, :]),
                NP.LinearPredictor())
        end
        NP.evaluate_draws!(
            linear_rng, draw_linear, work, replay, positions,
            NP.LinearPredictor())
        @test draw_linear == manual_linear
        queries = (;
            linear=NP.LinearPredictor(),
            predictive=NP.PosteriorPredictive())
        bundle = (;
            linear=zeros(2, 4),
            predictive=similar(draw_predictive))
        NP.execute_draws!(
            MersenneTwister(case.seed + 2), bundle,
            work, replay, positions, queries)
        @test bundle.linear == manual_fused_linear
        @test bundle.predictive == manual_predictive
        @test factor_generated_draw_allocations(
            MersenneTwister(case.seed + 4),
            MersenneTwister(case.seed + 5),
            MersenneTwister(case.seed + 6),
            draw_predictive, draw_linear, bundle,
            work, replay, positions, queries) ==
              (; predictive=0, linear=0, bundle=0)
    end

    varying_brm = @brm varying_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (1 | p | group)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(varying_brm, :mu) == [:x]
    @test ranefcoefnames(varying_brm, :p) == [
        (; predictor=:mu, coefficient=:Intercept)]
    @test SBBRMI(varying_brm; mod=@__MODULE__) isa SBBRMI
    varying_model = NP.lower(varying_brm)
    direct_varying_model = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:x,); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            tau_p_group=NP.parameter(
                NP.PositiveSupport(), (:tau_p_group,); transform=NP.Exp(),
                prior=NP.Exponential(1)),
            b_p_group=NP.grouped_normal(
                :group, 0.0, :tau_p_group),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(;
            r_mu_p_group=NP.group_gather(:b_p_group, :group),
            mu=NP.affine(
                :x, :beta_mu; offsets=(:r_mu_p_group,),
                intercept=false)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :mu, :sigma))),
        site_order=(:tau_p_group, :b_p_group, :beta_mu, :sigma, :y))
    @test typeof(varying_model) === typeof(direct_varying_model)
    @test sprint(show, varying_model) == sprint(show, direct_varying_model)
    natural_varying = NP.condition(
        natural_varying_intercept(
            varying_brm_data.x, varying_brm_data.group);
        y=varying_brm_data.y)
    @test keys(natural_varying.declaration.parameters) ==
          keys(varying_model.parameters)
    @test keys(natural_varying.declaration.nodes) ==
          (:r_mu_p_group, :mu)
    @test natural_varying.declaration.site_order == varying_model.site_order
    @test natural_varying.declaration.parameters.beta_mu.axis_keys == (:beta,)
    @test varying_model.parameters.beta_mu.axis_keys == (:x,)
    @test NP.group_input(
        natural_varying.declaration.parameters.b_p_group) === :group
    @test NP.group_values(
        natural_varying.declaration.nodes.r_mu_p_group) ===
          :b_p_group
    @test NP.group_input(
        natural_varying.declaration.nodes.r_mu_p_group) ===
          :group
    varying_plan = NP.compile(varying_brm)
    @test varying_plan isa NP.FactorPlan
    @test varying_plan.graph.schedule == (
        :tau_p_group, :b_p_group, :beta_mu, :sigma,
        :r_mu_p_group, :mu, :y)
    @test varying_plan.group_indices ==
          (; r_mu_p_group=(1, 2, 1, 3))
    @test varying_plan.graph.dimension == 6
    @test varying_plan.graph.coordinates.b_p_group.keys == (
        NP.GroupCoordinateKey(:b_p_group, :a),
        NP.GroupCoordinateKey(:b_p_group, :b),
        NP.GroupCoordinateKey(:b_p_group, :c))
    varying_prepared = NP.prepare(varying_plan)
    varying_workspace = NP.workspace(
        varying_prepared, Float64, DI.AutoEnzyme())
    varying_position = [log(0.7), 0.2, -0.1, 0.4, -0.3, log(0.5)]
    varying_tau = exp(varying_position[1])
    varying_effects = varying_position[2:4]
    varying_beta = varying_position[5]
    varying_sigma = exp(varying_position[6])
    varying_mu = varying_beta .* varying_brm_data.x .+
        varying_effects[[1, 2, 1, 3]]
    varying_residuals = varying_brm_data.y .- varying_mu
    varying_expected_density =
        logpdf(Exponential(1), varying_tau) + varying_position[1] +
        sum(logpdf.(Normal(0, varying_tau), varying_effects)) +
        logpdf(Normal(), varying_beta) +
        logpdf(Exponential(2), varying_sigma) + varying_position[6] +
        sum(logpdf.(Normal.(varying_mu, varying_sigma), varying_brm_data.y))
    varying_expected_gradient = [
        1 - varying_tau - length(varying_effects) +
            sum(abs2, varying_effects) / varying_tau^2,
        -varying_effects[1] / varying_tau^2 +
            (varying_residuals[1] + varying_residuals[3]) /
                varying_sigma^2,
        -varying_effects[2] / varying_tau^2 +
            varying_residuals[2] / varying_sigma^2,
        -varying_effects[3] / varying_tau^2 +
            varying_residuals[4] / varying_sigma^2,
        -varying_beta +
            sum(varying_residuals .* varying_brm_data.x) /
                varying_sigma^2,
        1 - varying_sigma / 2 - length(varying_brm_data.y) +
            sum(abs2, varying_residuals) / varying_sigma^2,
    ]
    varying_density, varying_gradient = NP.logdensity_and_gradient!(
        varying_workspace, varying_prepared, varying_position)
    @test varying_density ≈ varying_expected_density
    @test varying_gradient ≈ varying_expected_gradient
    @test NP.evaluate(
        varying_workspace, varying_prepared, varying_position,
        NP.LinearPredictor()) ≈ varying_mu
    @test factor_steady_state_allocations(
        varying_workspace, varying_prepared, varying_position) ==
          (; primal=0, gradient=0)
    varying_margin_prior = @brm varying_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (1 | p | group)
        sd(:, p, x) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test capability_error(
        () -> NP.lower(varying_margin_prior)).capability == :group_prior
    varying_no_population_data = (;
        group=varying_brm_data.group, y=varying_brm_data.y)
    varying_without_population = @brm varying_no_population_data begin
        sigma ~ Exponential(2)
        mu ~ 1 + (1 | p | group)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test capability_error(
        () -> NP.lower(varying_without_population)).capability ==
          :predictor_terms
    natural_varying_plan = NP.compile(natural_varying)
    direct_varying_plan = NP.compile(
        direct_varying_model,
        (; x=varying_brm_data.x, group=varying_brm_data.group);
        conditions=(; y=varying_brm_data.y))
    for candidate_plan in (natural_varying_plan, direct_varying_plan)
        canonical_schedule = map(candidate_plan.graph.schedule) do name
            name === :b_p_group_by_group_for_mu ? :r_mu_p_group : name
        end
        @test canonical_schedule == varying_plan.graph.schedule
        @test only(values(candidate_plan.group_indices)) ==
              only(values(varying_plan.group_indices))
        @test candidate_plan.graph.dimension == varying_plan.graph.dimension
        candidate_prepared = NP.prepare(candidate_plan)
        candidate_workspace = NP.workspace(
            candidate_prepared, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient = NP.logdensity_and_gradient!(
            candidate_workspace, candidate_prepared, varying_position)
        @test candidate_density ≈ varying_density
        @test candidate_gradient ≈ varying_gradient
        @test NP.evaluate(
            candidate_workspace, candidate_prepared, varying_position,
            NP.LinearPredictor()) ≈ varying_mu
        @test factor_steady_state_allocations(
            candidate_workspace, candidate_prepared, varying_position) ==
              (; primal=0, gradient=0)
    end
    varying_replay_bindings = (;
        x=[3.0, -2.0, 0.5], group=[:c, :a, :c])
    varying_replay_response = [0.2, -0.3, 0.7]
    varying_replay = NP.rebind(
        varying_prepared, (; y=varying_replay_response);
        bindings=varying_replay_bindings)
    @test varying_replay.plan.group_indices ==
          (; r_mu_p_group=(3, 1, 3))
    @test varying_replay.plan.graph.coordinates.b_p_group.keys ==
          varying_plan.graph.coordinates.b_p_group.keys
    @test varying_replay.plan.graph.dimension == varying_plan.graph.dimension
    varying_replay_workspace = NP.workspace(
        varying_replay, Float64, DI.AutoEnzyme())
    @test NP.evaluate(
        varying_replay_workspace, varying_replay, varying_position,
        NP.LinearPredictor()) ≈
          varying_beta .* varying_replay_bindings.x .+
          varying_effects[[3, 1, 3]]
    varying_prediction_only = NP.rebind(
        varying_prepared, (;); bindings=varying_replay_bindings)
    @test !NP.has_response(varying_prediction_only)
    @test length(NP.simulate(
        MersenneTwister(935), NP.workspace(varying_prediction_only),
        varying_prediction_only, varying_position)) == 3
    natural_varying_replay = NP.rebind(
        NP.prepare(natural_varying_plan), (;);
        bindings=varying_replay_bindings)
    @test only(values(natural_varying_replay.plan.group_indices)) ==
          only(values(varying_replay.plan.group_indices))
    @test NP.evaluate(
        NP.workspace(natural_varying_replay), natural_varying_replay,
        varying_position, NP.LinearPredictor()) ≈
          varying_beta .* varying_replay_bindings.x .+
          varying_effects[[3, 1, 3]]
    new_group_error = capability_error(() -> NP.rebind(
        varying_prepared, (;);
        bindings=(; x=[0.0, 1.0], group=[:a, :new_group])))
    @test new_group_error.capability == :new_group
    @test occursin("new_groups=:resample", new_group_error.detail)
    varying_new_group_bindings = (;
        x=[0.0, 1.0, 2.0, 3.0],
        group=[:a, :new_group, :newer_group, :new_group])
    varying_new_group_replay = NP.rebind(
        varying_prepared, (;); bindings=varying_new_group_bindings,
        new_groups=:resample)
    @test varying_new_group_replay.plan.graph.dimension ==
          varying_plan.graph.dimension
    @test varying_new_group_replay.plan.generated_group_levels ==
          (; b_p_group=(:new_group, :newer_group))
    @test varying_new_group_replay.plan.group_indices ==
          (; r_mu_p_group=(1, -1, -2, -1))
    @test occursin(
        "2 generated groups", sprint(show, varying_new_group_replay.plan))
    conditioned_new_group_error = capability_error(() -> NP.rebind(
        varying_prepared, (; y=zeros(4));
        bindings=varying_new_group_bindings, new_groups=:resample))
    @test conditioned_new_group_error.capability == :new_group_activity
    @test occursin("prediction-only", conditioned_new_group_error.detail)
    @test_throws ArgumentError NP.rebind(
        varying_prepared, (;); bindings=varying_new_group_bindings,
        new_groups=:invent)
    varying_new_group_workspace = NP.workspace(varying_new_group_replay)
    varying_new_group_output = zeros(4)
    varying_new_group_rng = MersenneTwister(936)
    varying_new_group_expected_rng = MersenneTwister(936)
    expected_new_group_effects = [
        varying_tau * randn(varying_new_group_expected_rng),
        varying_tau * randn(varying_new_group_expected_rng),
    ]
    varying_new_group_expected_mu = [
        varying_beta * varying_new_group_bindings.x[1] + varying_effects[1],
        varying_beta * varying_new_group_bindings.x[2] +
            expected_new_group_effects[1],
        varying_beta * varying_new_group_bindings.x[3] +
            expected_new_group_effects[2],
        varying_beta * varying_new_group_bindings.x[4] +
            expected_new_group_effects[1],
    ]
    varying_new_group_expected_response = map(
        varying_new_group_expected_mu) do location
        location + varying_sigma * randn(varying_new_group_expected_rng)
    end
    NP.simulate!(
        varying_new_group_rng, varying_new_group_output,
        varying_new_group_workspace, varying_new_group_replay,
        varying_position)
    @test varying_new_group_output == varying_new_group_expected_response
    @test varying_new_group_workspace.primal.generated_group_values ==
          expected_new_group_effects
    @test vec(varying_new_group_workspace.primal.node_rows[2, :]) ==
          varying_new_group_expected_mu
    @test factor_predictive_allocations(
        varying_new_group_rng, varying_new_group_output,
        varying_new_group_workspace, varying_new_group_replay,
        varying_position) == 0
    multi_varying_model = NP.model(
        inputs=(; x=NP.input(), w=NP.input(), group=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:x, :w); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            tau_p_group=direct_varying_model.parameters.tau_p_group,
            b_p_group=direct_varying_model.parameters.b_p_group,
            sigma=direct_varying_model.parameters.sigma),
        nodes=(;
            r_mu_p_group=NP.group_gather(:b_p_group, :group),
            mu=NP.affine(
                (:x, :w), :beta_mu; offsets=(:r_mu_p_group,),
                intercept=false)),
        observations=direct_varying_model.observations,
        site_order=direct_varying_model.site_order)
    multi_varying_prepared = NP.prepare(NP.compile(
        multi_varying_model,
        (; x=varying_brm_data.x, w=reverse(varying_brm_data.x),
           group=varying_brm_data.group);
        conditions=(; y=varying_brm_data.y)))
    multi_varying_replay = NP.rebind(
        multi_varying_prepared, (;);
        bindings=(; x=varying_new_group_bindings.x,
                    w=reverse(varying_new_group_bindings.x),
                    group=varying_new_group_bindings.group),
        new_groups=:resample)
    multi_varying_output = zeros(4)
    multi_varying_workspace = NP.workspace(multi_varying_replay)
    multi_varying_rng = MersenneTwister(938)
    NP.simulate!(
        multi_varying_rng, multi_varying_output,
        multi_varying_workspace, multi_varying_replay,
        [log(0.7), varying_effects..., -0.3, 0.1, log(0.5)])
    @test all(isfinite, multi_varying_output)
    @test factor_predictive_allocations(
        multi_varying_rng, multi_varying_output,
        multi_varying_workspace, multi_varying_replay,
        [log(0.7), varying_effects..., -0.3, 0.1, log(0.5)]) == 0
    new_group_density_error = capability_error(() -> NP.logdensity!(
        varying_new_group_workspace, varying_new_group_replay,
        varying_position))
    @test new_group_density_error.capability == :new_group_activity
    new_group_linear_error = capability_error(() -> NP.evaluate!(
        similar(varying_new_group_output), varying_new_group_workspace,
        varying_new_group_replay, varying_position, NP.LinearPredictor()))
    @test new_group_linear_error.capability == :new_group_activity
    new_group_prior_error = capability_error(() -> NP.simulate_prior!(
        MersenneTwister(937), similar(varying_position),
        similar(varying_new_group_output), varying_new_group_workspace,
        varying_new_group_replay))
    @test new_group_prior_error.capability == :new_group_activity
    varying_new_group_positions = [
        varying_position';
        log(0.9) -0.2 0.3 -0.4 0.15 log(0.6);
    ]
    varying_new_group_predictive = zeros(2, 4)
    varying_new_group_manual_predictive = similar(
        varying_new_group_predictive)
    varying_new_group_manual_linear = similar(
        varying_new_group_predictive)
    varying_new_group_draw_rng = MersenneTwister(938)
    varying_new_group_manual_rng = MersenneTwister(938)
    for draw in axes(varying_new_group_positions, 1)
        NP.simulate!(
            varying_new_group_manual_rng,
            @view(varying_new_group_manual_predictive[draw, :]),
            varying_new_group_workspace, varying_new_group_replay,
            @view(varying_new_group_positions[draw, :]))
        varying_new_group_manual_linear[draw, :] .=
            @view varying_new_group_workspace.primal.node_rows[2, :]
    end
    NP.simulate_draws!(
        varying_new_group_draw_rng, varying_new_group_predictive,
        varying_new_group_workspace, varying_new_group_replay,
        varying_new_group_positions)
    @test varying_new_group_predictive ==
          varying_new_group_manual_predictive
    varying_new_group_linear = zeros(2, 4)
    varying_new_group_linear_manual = similar(varying_new_group_linear)
    varying_new_group_linear_rng = MersenneTwister(939)
    varying_new_group_linear_manual_rng = MersenneTwister(939)
    for draw in axes(varying_new_group_positions, 1)
        NP.evaluate!(
            varying_new_group_linear_manual_rng,
            @view(varying_new_group_linear_manual[draw, :]),
            varying_new_group_workspace, varying_new_group_replay,
            @view(varying_new_group_positions[draw, :]),
            NP.LinearPredictor())
    end
    NP.evaluate_draws!(
        varying_new_group_linear_rng, varying_new_group_linear,
        varying_new_group_workspace, varying_new_group_replay,
        varying_new_group_positions, NP.LinearPredictor())
    @test varying_new_group_linear == varying_new_group_linear_manual
    varying_new_group_queries = (;
        linear=NP.LinearPredictor(),
        predictive=NP.PosteriorPredictive())
    varying_new_group_bundle = (;
        linear=zeros(2, 4), predictive=zeros(2, 4))
    varying_new_group_bundle_rng = MersenneTwister(938)
    NP.execute_draws!(
        varying_new_group_bundle_rng, varying_new_group_bundle,
        varying_new_group_workspace, varying_new_group_replay,
        varying_new_group_positions, varying_new_group_queries)
    @test varying_new_group_bundle.linear == varying_new_group_manual_linear
    @test varying_new_group_bundle.predictive ==
          varying_new_group_manual_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(940), MersenneTwister(941), MersenneTwister(942),
        varying_new_group_predictive, varying_new_group_linear,
        varying_new_group_bundle, varying_new_group_workspace,
        varying_new_group_replay, varying_new_group_positions,
        varying_new_group_queries) ==
          (; predictive=0, linear=0, bundle=0)
    @test_throws ArgumentError NP.execute_draws!(
        varying_new_group_bundle, varying_new_group_workspace,
        varying_new_group_replay, varying_new_group_positions,
        varying_new_group_queries)
    node_rows_alias_rng = MersenneTwister(943)
    node_rows_alias_expected_rng = MersenneTwister(943)
    @test_throws ArgumentError NP.simulate_draws!(
        node_rows_alias_rng,
        varying_new_group_workspace.primal.node_rows,
        varying_new_group_workspace, varying_new_group_replay,
        varying_new_group_positions)
    @test rand(node_rows_alias_rng) == rand(node_rows_alias_expected_rng)
    pairwise_alias_output = zeros(2, 4)
    pairwise_alias_rng = MersenneTwister(944)
    pairwise_alias_expected_rng = MersenneTwister(944)
    @test_throws ArgumentError NP.execute_draws!(
        pairwise_alias_rng,
        (; linear=pairwise_alias_output,
           predictive=pairwise_alias_output),
        varying_new_group_workspace, varying_new_group_replay,
        varying_new_group_positions, varying_new_group_queries)
    @test rand(pairwise_alias_rng) == rand(pairwise_alias_expected_rng)
    varying_new_group_signature = NP.batch_output_signature(
        varying_new_group_replay, varying_new_group_positions,
        NP.LinearPredictor())
    @test NP.output_axes(varying_new_group_signature) == (
        BRM.NativePPLAxis(:draw, Base.OneTo(2)),
        BRM.NativePPLAxis(:observation, Base.OneTo(4)))

    varying_slope_brm = @brm varying_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (0 + x | p | group)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(varying_slope_brm, :mu) == [:x]
    @test ranefcoefnames(varying_slope_brm, :p) == [
        (; predictor=:mu, coefficient=:x)]
    @test SBBRMI(varying_slope_brm; mod=@__MODULE__) isa SBBRMI
    varying_slope_model = NP.lower(varying_slope_brm)
    direct_varying_slope_model = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:x,); transform=NP.Identity(),
                prior=NP.StandardNormal()),
            tau_p_group=NP.parameter(
                NP.PositiveSupport(), (:tau_p_group,); transform=NP.Exp(),
                prior=NP.Exponential(1)),
            b_p_group=NP.grouped_normal(:group, 0.0, :tau_p_group),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,); transform=NP.Exp(),
                prior=NP.Exponential(2))),
        nodes=(;
            r_mu_p_group=NP.group_gather(:b_p_group, :group),
            r_mu_p_group_times_x=NP.row_product(:r_mu_p_group, :x),
            mu=NP.affine(
                :x, :beta_mu; offsets=(:r_mu_p_group_times_x,),
                intercept=false)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :mu, :sigma))),
        site_order=(:tau_p_group, :b_p_group, :beta_mu, :sigma, :y))
    @test typeof(varying_slope_model) ===
          typeof(direct_varying_slope_model)
    @test sprint(show, varying_slope_model) ==
          sprint(show, direct_varying_slope_model)
    @test keys(varying_slope_model.nodes) ==
          (:r_mu_p_group, :r_mu_p_group_times_x, :mu)
    @test varying_slope_model.nodes.r_mu_p_group_times_x isa NP.RowProduct
    natural_varying_slope_instance = NP.condition(
        natural_varying_slope(
            varying_brm_data.x, varying_brm_data.group);
        y=varying_brm_data.y)
    @test keys(natural_varying_slope_instance.declaration.parameters) ==
          keys(varying_slope_model.parameters)
    @test keys(natural_varying_slope_instance.declaration.nodes) == (
        :r_mu_p_group,
        :r_mu_p_group_times_x,
        :mu)
    @test natural_varying_slope_instance.declaration.site_order ==
          varying_slope_model.site_order
    varying_slope_plan = NP.compile(varying_slope_brm)
    @test varying_slope_plan.graph.schedule == (
        :tau_p_group, :b_p_group, :beta_mu, :sigma,
        :r_mu_p_group, :r_mu_p_group_times_x, :mu, :y)
    @test varying_slope_plan.graph.dimension == varying_plan.graph.dimension
    @test varying_slope_plan.group_indices == varying_plan.group_indices
    @test varying_slope_plan.graph.nodes.r_mu_p_group_times_x isa
          NP.RowProductFactorNode
    @test NP.factor_node_dependencies(
        varying_slope_plan.graph.nodes.r_mu_p_group_times_x) ==
          (:r_mu_p_group,)
    varying_slope_prepared = NP.prepare(varying_slope_plan)
    varying_slope_workspace = NP.workspace(
        varying_slope_prepared, Float64, DI.AutoEnzyme())
    varying_slope_mu = (
        varying_beta .+ varying_effects[[1, 2, 1, 3]]) .*
        varying_brm_data.x
    varying_slope_residuals = varying_brm_data.y .- varying_slope_mu
    varying_slope_expected_density =
        logpdf(Exponential(1), varying_tau) + varying_position[1] +
        sum(logpdf.(Normal(0, varying_tau), varying_effects)) +
        logpdf(Normal(), varying_beta) +
        logpdf(Exponential(2), varying_sigma) + varying_position[6] +
        sum(logpdf.(Normal.(varying_slope_mu, varying_sigma),
                    varying_brm_data.y))
    varying_slope_expected_gradient = [
        varying_expected_gradient[1],
        -varying_effects[1] / varying_tau^2 +
            (varying_slope_residuals[1] * varying_brm_data.x[1] +
             varying_slope_residuals[3] * varying_brm_data.x[3]) /
                varying_sigma^2,
        -varying_effects[2] / varying_tau^2 +
            varying_slope_residuals[2] * varying_brm_data.x[2] /
                varying_sigma^2,
        -varying_effects[3] / varying_tau^2 +
            varying_slope_residuals[4] * varying_brm_data.x[4] /
                varying_sigma^2,
        -varying_beta +
            sum(varying_slope_residuals .* varying_brm_data.x) /
                varying_sigma^2,
        1 - varying_sigma / 2 - length(varying_brm_data.y) +
            sum(abs2, varying_slope_residuals) / varying_sigma^2,
    ]
    varying_slope_density, varying_slope_gradient =
        NP.logdensity_and_gradient!(
            varying_slope_workspace, varying_slope_prepared,
            varying_position)
    @test varying_slope_density ≈ varying_slope_expected_density
    @test varying_slope_gradient ≈ varying_slope_expected_gradient
    @test NP.evaluate(
        varying_slope_workspace, varying_slope_prepared,
        varying_position, NP.LinearPredictor()) ≈ varying_slope_mu
    @test factor_steady_state_allocations(
        varying_slope_workspace, varying_slope_prepared,
        varying_position) == (; primal=0, gradient=0)
    for candidate_plan in (
        NP.compile(natural_varying_slope_instance),
        NP.compile(
            direct_varying_slope_model,
            (; x=varying_brm_data.x, group=varying_brm_data.group);
            conditions=(; y=varying_brm_data.y)))
        candidate_prepared = NP.prepare(candidate_plan)
        candidate_workspace = NP.workspace(
            candidate_prepared, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient = NP.logdensity_and_gradient!(
            candidate_workspace, candidate_prepared, varying_position)
        @test candidate_density ≈ varying_slope_density
        @test candidate_gradient ≈ varying_slope_gradient
        @test NP.evaluate(
            candidate_workspace, candidate_prepared, varying_position,
            NP.LinearPredictor()) ≈ varying_slope_mu
        @test factor_steady_state_allocations(
            candidate_workspace, candidate_prepared, varying_position) ==
              (; primal=0, gradient=0)
    end
    varying_slope_known_bindings = (;
        x=[2.0, -1.5, 0.25], group=[:c, :a, :c])
    varying_slope_known_replay = NP.rebind(
        varying_slope_prepared, (;);
        bindings=varying_slope_known_bindings)
    @test varying_slope_known_replay.plan.graph.dimension ==
          varying_slope_plan.graph.dimension
    @test varying_slope_known_replay.plan.group_indices ==
          (; r_mu_p_group=(3, 1, 3))
    @test NP.evaluate(
        NP.workspace(varying_slope_known_replay),
        varying_slope_known_replay, varying_position,
        NP.LinearPredictor()) ≈
          (varying_beta .+ varying_effects[[3, 1, 3]]) .*
          varying_slope_known_bindings.x
    varying_slope_new_bindings = (;
        x=[0.5, -1.0, 2.0, 1.5],
        group=[:a, :new_slope_group, :new_slope_group, :c])
    varying_slope_new_replay = NP.rebind(
        varying_slope_prepared, (;);
        bindings=varying_slope_new_bindings, new_groups=:resample)
    @test varying_slope_new_replay.plan.generated_group_levels ==
          (; b_p_group=(:new_slope_group,))
    @test varying_slope_new_replay.plan.group_indices ==
          (; r_mu_p_group=(1, -1, -1, 3))
    varying_slope_new_workspace = NP.workspace(varying_slope_new_replay)
    varying_slope_new_output = zeros(4)
    varying_slope_new_rng = MersenneTwister(945)
    varying_slope_new_expected_rng = MersenneTwister(945)
    varying_slope_generated_effect =
        varying_tau * randn(varying_slope_new_expected_rng)
    varying_slope_new_effects = [
        varying_effects[1], varying_slope_generated_effect,
        varying_slope_generated_effect, varying_effects[3]]
    varying_slope_new_mu = (
        varying_beta .+ varying_slope_new_effects) .*
        varying_slope_new_bindings.x
    varying_slope_new_expected = map(varying_slope_new_mu) do location
        location + varying_sigma * randn(varying_slope_new_expected_rng)
    end
    NP.simulate!(
        varying_slope_new_rng, varying_slope_new_output,
        varying_slope_new_workspace, varying_slope_new_replay,
        varying_position)
    @test varying_slope_new_output ≈ varying_slope_new_expected
    @test only(varying_slope_new_workspace.primal.generated_group_values) ==
          varying_slope_generated_effect
    @test vec(varying_slope_new_workspace.primal.node_rows[3, :]) ≈
          varying_slope_new_mu
    @test factor_predictive_allocations(
        varying_slope_new_rng, varying_slope_new_output,
        varying_slope_new_workspace, varying_slope_new_replay,
        varying_position) == 0
    varying_slope_draw_positions = [
        varying_position';
        log(0.9) -0.2 0.3 -0.4 0.15 log(0.6);
    ]
    varying_slope_draw_predictive = zeros(2, 4)
    varying_slope_draw_linear = zeros(2, 4)
    varying_slope_manual_predictive = zeros(2, 4)
    varying_slope_manual_linear = zeros(2, 4)
    varying_slope_manual_fused_linear = zeros(2, 4)
    varying_slope_draw_rng = MersenneTwister(946)
    varying_slope_manual_rng = MersenneTwister(946)
    for draw in axes(varying_slope_draw_positions, 1)
        NP.simulate!(
            varying_slope_manual_rng,
            @view(varying_slope_manual_predictive[draw, :]),
            varying_slope_new_workspace, varying_slope_new_replay,
            @view(varying_slope_draw_positions[draw, :]))
        varying_slope_manual_fused_linear[draw, :] .=
            @view varying_slope_new_workspace.primal.node_rows[3, :]
    end
    NP.simulate_draws!(
        varying_slope_draw_rng, varying_slope_draw_predictive,
        varying_slope_new_workspace, varying_slope_new_replay,
        varying_slope_draw_positions)
    @test varying_slope_draw_predictive == varying_slope_manual_predictive
    varying_slope_linear_rng = MersenneTwister(947)
    varying_slope_linear_manual_rng = MersenneTwister(947)
    for draw in axes(varying_slope_draw_positions, 1)
        NP.evaluate!(
            varying_slope_linear_manual_rng,
            @view(varying_slope_manual_linear[draw, :]),
            varying_slope_new_workspace, varying_slope_new_replay,
            @view(varying_slope_draw_positions[draw, :]),
            NP.LinearPredictor())
    end
    NP.evaluate_draws!(
        varying_slope_linear_rng, varying_slope_draw_linear,
        varying_slope_new_workspace, varying_slope_new_replay,
        varying_slope_draw_positions, NP.LinearPredictor())
    @test varying_slope_draw_linear == varying_slope_manual_linear
    varying_slope_queries = (;
        linear=NP.LinearPredictor(),
        predictive=NP.PosteriorPredictive())
    varying_slope_bundle = (;
        linear=zeros(2, 4), predictive=zeros(2, 4))
    NP.execute_draws!(
        MersenneTwister(946), varying_slope_bundle,
        varying_slope_new_workspace, varying_slope_new_replay,
        varying_slope_draw_positions, varying_slope_queries)
    @test varying_slope_bundle.linear == varying_slope_manual_fused_linear
    @test varying_slope_bundle.predictive == varying_slope_manual_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(948), MersenneTwister(949), MersenneTwister(950),
        varying_slope_draw_predictive, varying_slope_draw_linear,
        varying_slope_bundle, varying_slope_new_workspace,
        varying_slope_new_replay, varying_slope_draw_positions,
        varying_slope_queries) ==
          (; predictive=0, linear=0, bundle=0)
    varying_slope_signature = NP.batch_output_signature(
        varying_slope_new_replay, varying_slope_draw_positions,
        NP.LinearPredictor())
    @test NP.output_axes(varying_slope_signature) == (
        BRM.NativePPLAxis(:draw, Base.OneTo(2)),
        BRM.NativePPLAxis(:observation, Base.OneTo(4)))

    transformed_group_brm = @brm varying_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 0 + zscale(x) + (1 + zscale(x) | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, sigma)
    end
    @test popcoefnames(transformed_group_brm, :mu) == [:zscale_x]
    @test ranefcoefnames(transformed_group_brm, :p) == [
        (; predictor=:mu, coefficient=:Intercept),
        (; predictor=:mu, coefficient=:zscale_x)]
    transformed_group_sb = SBBRMI(
        transformed_group_brm; mod=@__MODULE__)
    @test transformed_group_sb isa SBBRMI
    @test transformed_group_sb.data[:zscale_x] ≈
          (varying_brm_data.x .- 0.5) ./ sqrt(5 / 3)
    transformed_group_model = NP.lower(transformed_group_brm)
    direct_transformed_group_model = NP.model(
        inputs=(; x=NP.input(), group=NP.input()),
        parameters=transformed_group_model.parameters,
        nodes=(;
            zscale_x_for_mu=NP.zscale(:x),
            b_p_group_by_group_for_mu=NP.grouped_affine(
                :b_p_group, :tau_p_group, :L_p_group, :group,
                (nothing, :zscale_x_for_mu)),
            mu=NP.affine(
                :zscale_x_for_mu, :beta_mu;
                offsets=(:b_p_group_by_group_for_mu,), intercept=false)),
        observations=(; y=NP.broadcasted(
            NP.normal(:y, :mu, :sigma))),
        site_order=transformed_group_model.site_order)
    @test direct_transformed_group_model == transformed_group_model
    natural_transformed_group = NP.condition(
        natural_transformed_correlated_varying_slope(
            varying_brm_data.x, varying_brm_data.group);
        y=varying_brm_data.y)
    @test transformed_group_model ==
          natural_transformed_group.declaration
    @test keys(transformed_group_model.nodes) == (
        :zscale_x_for_mu,
        :b_p_group_by_group_for_mu,
        :mu)
    @test transformed_group_model.nodes.zscale_x_for_mu isa NP.ZScale
    @test transformed_group_model.nodes.b_p_group_by_group_for_mu isa
          NP.GroupedAffine
    @test NP.grouped_predictors(
        transformed_group_model.nodes.b_p_group_by_group_for_mu) ==
          (nothing, :zscale_x_for_mu)
    transformed_group_plan = NP.compile(transformed_group_brm)
    @test transformed_group_plan.graph.schedule == (
        :tau_p_group, :L_p_group, :b_p_group, :beta_mu, :sigma,
        :zscale_x_for_mu, :b_p_group_by_group_for_mu, :mu, :y)
    @test transformed_group_plan.graph.nodes.zscale_x_for_mu isa
          NP.ZScaleFactorNode
    @test transformed_group_plan.fitted_nodes.zscale_x_for_mu.mean == 0.5
    @test transformed_group_plan.fitted_nodes.zscale_x_for_mu.scale ≈
          sqrt(5 / 3)
    transformed_group_node =
        transformed_group_plan.graph.nodes.b_p_group_by_group_for_mu
    @test transformed_group_node isa NP.GroupedAffineFactorNode
    @test transformed_group_node.predictors ==
          (nothing, NP.NodeValue{:zscale_x_for_mu}())
    @test transformed_group_plan.graph.dimension == 11
    invalid_transformed_scalar = NP.model(
        inputs=direct_transformed_group_model.inputs,
        parameters=direct_transformed_group_model.parameters,
        nodes=direct_transformed_group_model.nodes,
        observations=(;
            scalar=NP.normal(:scalar, :zscale_x_for_mu, :sigma),
            y=direct_transformed_group_model.observations.y),
        site_order=(
            transformed_group_model.site_order[1:(end - 1)]...,
            :scalar, :y))
    @test capability_error(() -> NP.compile(
        invalid_transformed_scalar,
        (; x=varying_brm_data.x, group=varying_brm_data.group);
        conditions=(; y=varying_brm_data.y))).capability == :factor_shape
    nested_fitted_transform = NP.model(
        inputs=direct_transformed_group_model.inputs,
        parameters=direct_transformed_group_model.parameters,
        nodes=(;
            zscale_x_for_mu=NP.zscale(:x),
            nested_zscale=NP.zscale(:zscale_x_for_mu),
            b_p_group_by_group_for_mu=NP.grouped_affine(
                :b_p_group, :tau_p_group, :L_p_group, :group,
                (nothing, :nested_zscale)),
            mu=NP.affine(
                :nested_zscale, :beta_mu;
                offsets=(:b_p_group_by_group_for_mu,), intercept=false)),
        observations=direct_transformed_group_model.observations,
        site_order=direct_transformed_group_model.site_order)
    @test capability_error(() -> NP.compile(
        nested_fitted_transform,
        (; x=varying_brm_data.x, group=varying_brm_data.group);
        conditions=(; y=varying_brm_data.y))).capability == :factor_nodes
    transformed_group_prepared = NP.prepare(transformed_group_plan)
    transformed_group_workspace = NP.workspace(
        transformed_group_prepared, Float64, DI.AutoEnzyme())
    transformed_x = (varying_brm_data.x .- 0.5) ./ sqrt(5 / 3)
    transformed_group_mu = [
        correlated_beta * transformed_x[row] +
        correlated_effects[[1, 2, 1, 3][row]][1] +
        correlated_effects[[1, 2, 1, 3][row]][2] * transformed_x[row]
        for row in eachindex(transformed_x)]
    transformed_group_expected_density =
        correlated_expected_density -
        sum(logpdf.(Normal.(correlated_mu, correlated_sigma),
                    varying_brm_data.y)) +
        sum(logpdf.(Normal.(transformed_group_mu, correlated_sigma),
                    varying_brm_data.y))
    transformed_group_density, transformed_group_gradient =
        NP.logdensity_and_gradient!(
            transformed_group_workspace, transformed_group_prepared,
            correlated_group_position)
    @test transformed_group_density ≈ transformed_group_expected_density
    @test NP.evaluate(
        transformed_group_workspace, transformed_group_prepared,
        correlated_group_position, NP.LinearPredictor()) ≈
          transformed_group_mu
    @test NP.evaluate(
        transformed_group_workspace, transformed_group_prepared,
        correlated_group_position,
        NP.NodeOutput(:zscale_x_for_mu)) ≈ transformed_x
    transformed_node_output = similar(transformed_x)
    transformed_node_query = NP.NodeOutput(:zscale_x_for_mu)
    NP.evaluate!(
        transformed_node_output, transformed_group_workspace,
        transformed_group_prepared, correlated_group_position,
        transformed_node_query)
    @test transformed_node_output ≈ transformed_x
    @test @allocated(NP.evaluate!(
        transformed_node_output, transformed_group_workspace,
        transformed_group_prepared, correlated_group_position,
        transformed_node_query)) == 0
    transformed_group_finite_difference = similar(
        transformed_group_gradient)
    transformed_group_plus = copy(correlated_group_position)
    transformed_group_minus = copy(correlated_group_position)
    transformed_group_step = 1e-6
    for coordinate in eachindex(transformed_group_finite_difference)
        transformed_group_plus[coordinate] += transformed_group_step
        transformed_group_minus[coordinate] -= transformed_group_step
        transformed_group_finite_difference[coordinate] = (
            NP.logdensity!(
                transformed_group_workspace, transformed_group_prepared,
                transformed_group_plus) -
            NP.logdensity!(
                transformed_group_workspace, transformed_group_prepared,
                transformed_group_minus)) / (2 * transformed_group_step)
        transformed_group_plus[coordinate] = correlated_group_position[coordinate]
        transformed_group_minus[coordinate] = correlated_group_position[coordinate]
    end
    @test transformed_group_gradient ≈ transformed_group_finite_difference rtol=2e-5 atol=2e-6
    @test factor_steady_state_allocations(
        transformed_group_workspace, transformed_group_prepared,
        correlated_group_position) == (; primal=0, gradient=0)
    transformed_replay_bindings = (;
        x=[10.0, 20.0, 30.0], group=[:c, :a, :b])
    transformed_frozen = NP.rebind(
        transformed_group_prepared, (;);
        bindings=transformed_replay_bindings)
    transformed_refitted = NP.rebind(
        transformed_group_prepared, (;);
        bindings=transformed_replay_bindings, freeze_constants=false)
    @test transformed_frozen.plan.fitted_nodes ==
          transformed_group_plan.fitted_nodes
    @test transformed_refitted.plan.fitted_nodes.zscale_x_for_mu.mean == 20.0
    @test transformed_refitted.plan.fitted_nodes.zscale_x_for_mu.scale == 10.0
    frozen_x = (transformed_replay_bindings.x .- 0.5) ./ sqrt(5 / 3)
    refitted_x = [-1.0, 0.0, 1.0]
    for (replay, transformed_values) in (
            (transformed_frozen, frozen_x),
            (transformed_refitted, refitted_x))
        expected = [
            correlated_effects[group][1] +
            (correlated_beta + correlated_effects[group][2]) *
                transformed_values[row]
            for (row, group) in enumerate((3, 1, 2))]
        replay_workspace = NP.workspace(replay)
        @test NP.evaluate(
            replay_workspace, replay, correlated_group_position,
            NP.LinearPredictor()) ≈ expected
        @test NP.evaluate(
            replay_workspace, replay, correlated_group_position,
            NP.NodeOutput(:zscale_x_for_mu)) ≈ transformed_values
    end
    transformed_positions = [
        correlated_group_position';
        log(0.8) log(0.5) -0.15 -0.1 0.3 0.2 -0.4 0.1 0.35 -0.2 log(0.7);
    ]
    transformed_mu_draws = zeros(2, 4)
    transformed_scaled_draws = zeros(2, 4)
    NP.evaluate_draws!(
        transformed_mu_draws, transformed_group_workspace,
        transformed_group_prepared, transformed_positions,
        NP.NodeOutput(:mu))
    NP.evaluate_draws!(
        transformed_scaled_draws, transformed_group_workspace,
        transformed_group_prepared, transformed_positions,
        transformed_node_query)
    @test transformed_mu_draws[1, :] ≈ transformed_group_mu
    @test all(row -> row ≈ transformed_x,
              eachrow(transformed_scaled_draws))
    transformed_bundle_queries = (;
        mu=NP.NodeOutput(:mu),
        scaled=transformed_node_query,
        pointwise=NP.PointwiseLogLikelihood(),
        predictive=NP.PosteriorPredictive())
    transformed_bundle = (;
        mu=zeros(2, 4), scaled=zeros(2, 4),
        pointwise=zeros(2, 4), predictive=zeros(2, 4))
    transformed_draw_predictive = zeros(2, 4)
    transformed_bundle_rng = MersenneTwister(993)
    transformed_predictive_rng = MersenneTwister(993)
    NP.execute_draws!(
        transformed_bundle_rng, transformed_bundle,
        transformed_group_workspace, transformed_group_prepared,
        transformed_positions, transformed_bundle_queries)
    NP.simulate_draws!(
        transformed_predictive_rng, transformed_draw_predictive,
        transformed_group_workspace, transformed_group_prepared,
        transformed_positions)
    @test transformed_bundle.mu == transformed_mu_draws
    @test transformed_bundle.scaled == transformed_scaled_draws
    @test transformed_bundle.predictive == transformed_draw_predictive
    @test factor_generated_draw_allocations(
        MersenneTwister(994), MersenneTwister(995), MersenneTwister(996),
        transformed_draw_predictive, similar(transformed_mu_draws),
        transformed_bundle, transformed_group_workspace,
        transformed_group_prepared, transformed_positions,
        transformed_bundle_queries) ==
          (; predictive=0, linear=0, bundle=0)

    transformed_new_bindings = (;
        x=[-2.0, 0.5, 1.5, 3.0],
        group=[:a, :new_transformed_group, :new_transformed_group, :c])
    transformed_new = NP.rebind(
        transformed_group_prepared, (;);
        bindings=transformed_new_bindings, new_groups=:resample)
    @test transformed_new.plan.generated_group_levels ==
          (; b_p_group=(:new_transformed_group,))
    @test transformed_new.plan.group_indices ==
          (; b_p_group_by_group_for_mu=(1, -1, -1, 3))
    @test transformed_new.plan.fitted_nodes ==
          transformed_group_plan.fitted_nodes
    transformed_new_x = (transformed_new_bindings.x .- 0.5) ./ sqrt(5 / 3)
    transformed_new_workspace = NP.workspace(transformed_new)
    @test capability_error(() -> NP.evaluate(
        transformed_new_workspace, transformed_new,
        correlated_group_position,
        NP.NodeOutput(:mu))).capability == :new_group_activity
    @test NP.evaluate(
        transformed_new_workspace, transformed_new,
        correlated_group_position,
        transformed_node_query) ≈ transformed_new_x
    transformed_new_rng = MersenneTwister(997)
    transformed_new_expected_rng = MersenneTwister(997)
    transformed_new_z = randn(transformed_new_expected_rng, 2)
    transformed_new_effect = (
        correlated_tau[1] * transformed_new_z[1],
        correlated_tau[2] * (
            correlated_rho * transformed_new_z[1] +
            correlated_sech * transformed_new_z[2]))
    transformed_new_effects = (
        correlated_effects[1], transformed_new_effect,
        transformed_new_effect, correlated_effects[3])
    transformed_new_mu = [
        effect[1] + (correlated_beta + effect[2]) * transformed_new_x[row]
        for (row, effect) in enumerate(transformed_new_effects)]
    transformed_new_output = zeros(4)
    NP.evaluate!(
        transformed_new_rng, transformed_new_output,
        transformed_new_workspace, transformed_new,
        correlated_group_position, NP.LinearPredictor())
    @test transformed_new_output ≈ transformed_new_mu
    @test transformed_new_workspace.primal.generated_group_values ≈
          transformed_new_z

    transformed_input_scale_model = NP.model(
        inputs=(; x=NP.input(), log_scale=NP.input(), group=NP.input()),
        parameters=(;
            tau_p_group=transformed_group_model.parameters.tau_p_group,
            L_p_group=transformed_group_model.parameters.L_p_group,
            b_p_group=transformed_group_model.parameters.b_p_group,
            beta_mu=transformed_group_model.parameters.beta_mu),
        nodes=(;
            zscale_x_for_mu=NP.zscale(:x),
            zscale_log_scale_for_y=NP.zscale(:log_scale),
            b_p_group_by_group_for_mu=NP.grouped_affine(
                :b_p_group, :tau_p_group, :L_p_group, :group,
                (nothing, :zscale_x_for_mu)),
            mu=NP.affine(
                :zscale_x_for_mu, :beta_mu;
                offsets=(:b_p_group_by_group_for_mu,), intercept=false),
            scale=NP.exp_link(:zscale_log_scale_for_y)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :scale))),
        site_order=(
            :tau_p_group, :L_p_group, :b_p_group, :beta_mu, :y))
    transformed_input_scale_plan = NP.compile(
        transformed_input_scale_model,
        (; x=varying_brm_data.x, log_scale=[-1.0, 0.0, 1.0, 2.0],
           group=varying_brm_data.group);
        conditions=(; y=varying_brm_data.y))
    transformed_input_scale_prepared = NP.prepare(
        transformed_input_scale_plan)
    transformed_input_scale_new = NP.rebind(
        transformed_input_scale_prepared, (;);
        bindings=(;
            x=transformed_new_bindings.x,
            log_scale=[2.0, 3.0, 4.0, 5.0],
            group=transformed_new_bindings.group),
        new_groups=:resample)
    transformed_input_scale_workspace = NP.workspace(
        transformed_input_scale_new)
    transformed_input_scale_position = correlated_group_position[1:10]
    transformed_input_scale_output = zeros(4)
    transformed_input_scale_query = NP.NodeOutput(:scale)
    NP.evaluate!(
        transformed_input_scale_output,
        transformed_input_scale_workspace, transformed_input_scale_new,
        transformed_input_scale_position, transformed_input_scale_query)
    @test transformed_input_scale_output ≈
          exp.(([2.0, 3.0, 4.0, 5.0] .- 0.5) ./ sqrt(5 / 3))
    @test @allocated(NP.evaluate!(
        transformed_input_scale_output,
        transformed_input_scale_workspace, transformed_input_scale_new,
        transformed_input_scale_position, transformed_input_scale_query)) == 0

    transformed_large = @brm (
            x=[-1e300, 0.0, 1e300, 5e299],
            group=varying_brm_data.group, y=varying_brm_data.y) begin
        sigma ~ Exponential(2)
        mu ~ 0 + zscale(x) + (1 + zscale(x) | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, sigma)
    end
    @test_throws ArgumentError NP.prepare(
        NP.compile(transformed_large); T=Float32)

    sampled_offset_plan = NP.compile(sampled_offset_brmi)
    @test sampled_offset_plan isa NP.FactorPlan
    @test sampled_offset_plan.bindings.x == sampled_offset_data.x
    natural_sampled_offset_plan = NP.compile(natural_sampled_offset)
    direct_sampled_offset_plan = NP.compile(
        direct_sampled_offset_model, (; x=sampled_offset_data.x);
        conditions=(; y=sampled_offset_data.y))
    @test natural_sampled_offset_plan.graph.schedule ==
          sampled_offset_plan.graph.schedule
    @test direct_sampled_offset_plan.graph.schedule ==
          sampled_offset_plan.graph.schedule
    sampled_offset_prepared = NP.prepare(sampled_offset_plan)
    sampled_offset_workspace = NP.workspace(
        sampled_offset_prepared, Float64, DI.AutoEnzyme())
    sampled_offset_position = [0.2, -0.3, log(0.7)]
    sampled_offset_latent = sampled_offset_position[1]
    sampled_offset_beta = sampled_offset_position[2]
    sampled_offset_sigma = exp(sampled_offset_position[3])
    sampled_offset_mu = sampled_offset_latent .+
        sampled_offset_beta .* sampled_offset_data.x
    sampled_offset_residuals = sampled_offset_data.y .- sampled_offset_mu
    sampled_offset_expected_density =
        logpdf(Normal(), sampled_offset_latent) +
        logpdf(Normal(), sampled_offset_beta) +
        logpdf(Exponential(2), sampled_offset_sigma) +
        sampled_offset_position[3] +
        sum(logpdf.(Normal.(sampled_offset_mu, sampled_offset_sigma),
                    sampled_offset_data.y))
    sampled_offset_expected_gradient = [
        -sampled_offset_latent +
            sum(sampled_offset_residuals) / sampled_offset_sigma^2,
        -sampled_offset_beta +
            sum(sampled_offset_residuals .* sampled_offset_data.x) /
                sampled_offset_sigma^2,
        1 - sampled_offset_sigma / 2 - length(sampled_offset_data.y) +
            sum(abs2, sampled_offset_residuals) / sampled_offset_sigma^2,
    ]
    sampled_offset_density, sampled_offset_gradient =
        NP.logdensity_and_gradient!(
            sampled_offset_workspace, sampled_offset_prepared,
            sampled_offset_position)
    @test sampled_offset_density ≈ sampled_offset_expected_density
    @test sampled_offset_gradient ≈ sampled_offset_expected_gradient
    sampled_offset_underflow_position = copy(sampled_offset_position)
    sampled_offset_underflow_position[3] = -1000.0
    @test NP.logdensity!(
        sampled_offset_workspace, sampled_offset_prepared,
        sampled_offset_underflow_position) == -Inf
    @test vec(sampled_offset_workspace.primal.node_rows[1, :]) ≈
          sampled_offset_mu
    @test NP.evaluate(
        sampled_offset_workspace, sampled_offset_prepared,
        sampled_offset_position, NP.LinearPredictor()) ≈ sampled_offset_mu
    @test NP.evaluate(
        sampled_offset_workspace, sampled_offset_prepared,
        sampled_offset_position, NP.PointwiseLogLikelihood()) ≈
          logpdf.(Normal.(sampled_offset_mu, sampled_offset_sigma),
                  sampled_offset_data.y)
    @test factor_steady_state_allocations(
        sampled_offset_workspace, sampled_offset_prepared,
        sampled_offset_position) == (; primal=0, gradient=0)
    sampled_offset_source_x = copy(sampled_offset_data.x)
    sampled_offset_source_y = copy(sampled_offset_data.y)
    sampled_offset_owned = NP.prepare(NP.bind(
        sampled_offset_model, (; x=sampled_offset_source_x);
        conditions=(; y=sampled_offset_source_y)))
    sampled_offset_owned_workspace = NP.workspace(sampled_offset_owned)
    sampled_offset_owned_density = NP.logdensity!(
        sampled_offset_owned_workspace, sampled_offset_owned,
        sampled_offset_position)
    @test sampled_offset_owned.plan.bindings.x !== sampled_offset_source_x
    @test sampled_offset_owned.conditions.y !== sampled_offset_source_y
    sampled_offset_source_x[1] = 100.0
    sampled_offset_source_y[1] = 100.0
    @test NP.logdensity!(
        sampled_offset_owned_workspace, sampled_offset_owned,
        sampled_offset_position) == sampled_offset_owned_density
    for candidate_plan in (
        natural_sampled_offset_plan, direct_sampled_offset_plan)
        candidate_prepared = NP.prepare(candidate_plan)
        candidate_workspace = NP.workspace(
            candidate_prepared, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient = NP.logdensity_and_gradient!(
            candidate_workspace, candidate_prepared,
            sampled_offset_position)
        @test candidate_density ≈ sampled_offset_density
        @test candidate_gradient ≈ sampled_offset_gradient
        @test NP.evaluate(
            candidate_workspace, candidate_prepared,
            sampled_offset_position, NP.LinearPredictor()) ≈
              sampled_offset_mu
        @test factor_steady_state_allocations(
            candidate_workspace, candidate_prepared,
            sampled_offset_position) == (; primal=0, gradient=0)
    end
    sampled_offset_prediction_only = NP.rebind(
        sampled_offset_prepared, (;))
    @test !NP.has_response(sampled_offset_prediction_only)
    @test length(NP.simulate(
        MersenneTwister(933), NP.workspace(sampled_offset_prediction_only),
        sampled_offset_prediction_only, sampled_offset_position)) ==
          length(sampled_offset_data.x)

    hierarchy_plan = NP.compile(NP.condition(hierarchy; y=response))
    @test hierarchy_plan isa NP.FactorPlan
    @test LogDensityProblems.dimension(hierarchy_plan) == 4
    @test hierarchy_plan.output_site === :y
    hierarchy_prepared = NP.prepare(hierarchy_plan)
    hierarchy_workspace = NP.workspace(
        hierarchy_prepared, Float64, DI.AutoEnzyme())
    hierarchy_position = [0.2, log(0.7), -0.1, log(0.5)]
    hierarchy_population = hierarchy_position[1]
    hierarchy_population_scale = exp(hierarchy_position[2])
    hierarchy_individual = hierarchy_position[3]
    hierarchy_observation_scale = exp(hierarchy_position[4])
    hierarchy_residual = hierarchy_individual - hierarchy_population
    hierarchy_observation_residuals = response .- hierarchy_individual
    hierarchy_expected_density =
        logpdf(Normal(), hierarchy_population) +
        logpdf(Exponential(1.0), hierarchy_population_scale) +
        hierarchy_position[2] +
        logpdf(Normal(
            hierarchy_population, hierarchy_population_scale),
            hierarchy_individual) +
        logpdf(Exponential(2.0), hierarchy_observation_scale) +
        hierarchy_position[4] +
        sum(logpdf.(Normal(
            hierarchy_individual, hierarchy_observation_scale), response))
    hierarchy_expected_gradient = [
        -hierarchy_population +
            hierarchy_residual / hierarchy_population_scale^2,
        1 - hierarchy_population_scale - 1 +
            hierarchy_residual^2 / hierarchy_population_scale^2,
        -hierarchy_residual / hierarchy_population_scale^2 +
            sum(hierarchy_observation_residuals) /
                hierarchy_observation_scale^2,
        1 - hierarchy_observation_scale / 2 - length(response) +
            sum(abs2, hierarchy_observation_residuals) /
                hierarchy_observation_scale^2,
    ]
    hierarchy_density, hierarchy_gradient = NP.logdensity_and_gradient!(
        hierarchy_workspace, hierarchy_prepared, hierarchy_position)
    @test hierarchy_density ≈ hierarchy_expected_density
    @test hierarchy_gradient ≈ hierarchy_expected_gradient
    hierarchy_prepared32 = NP.prepare(hierarchy_plan; T=Float32)
    hierarchy_position32 = Float32.(hierarchy_position)
    hierarchy_density32, hierarchy_gradient32 = NP.logdensity_and_gradient!(
        NP.workspace(hierarchy_prepared32, Float32, DI.AutoEnzyme()),
        hierarchy_prepared32, hierarchy_position32)
    @test hierarchy_density32 ≈ Float32(hierarchy_expected_density) rtol=1f-5
    @test hierarchy_gradient32 ≈
          Float32.(hierarchy_expected_gradient) rtol=1f-5
    @test_throws ArgumentError NP.prepare(
        hierarchy_plan; T=AbstractFloat)
    hierarchy_underflow_position = copy(hierarchy_position)
    hierarchy_underflow_position[2] = -1000.0
    @test NP.logdensity!(
        NP.workspace(hierarchy_prepared), hierarchy_prepared,
        hierarchy_underflow_position) == -Inf
    @test hierarchy_workspace.primal.pointwise_loglikelihood ≈
        logpdf.(Normal(
            hierarchy_individual, hierarchy_observation_scale), response)
    hierarchy_linear = NP.evaluate(
        hierarchy_workspace, hierarchy_prepared, hierarchy_position,
        NP.LinearPredictor())
    @test hierarchy_linear == fill(
        hierarchy_individual, length(response))
    hierarchy_pointwise = NP.evaluate(
        hierarchy_workspace, hierarchy_prepared, hierarchy_position,
        NP.PointwiseLogLikelihood())
    @test hierarchy_pointwise ≈ logpdf.(Normal(
        hierarchy_individual, hierarchy_observation_scale), response)
    hierarchy_predictive_rng = MersenneTwister(919)
    hierarchy_expected_predictive_rng = MersenneTwister(919)
    hierarchy_expected_predictive = [
        hierarchy_individual + hierarchy_observation_scale *
            randn(hierarchy_expected_predictive_rng)
        for _ in response
    ]
    hierarchy_predictive = NP.simulate(
        hierarchy_predictive_rng, hierarchy_workspace,
        hierarchy_prepared, hierarchy_position)
    @test hierarchy_predictive == hierarchy_expected_predictive

    hierarchy_prior_rng = MersenneTwister(920)
    hierarchy_expected_prior_rng = MersenneTwister(920)
    expected_population = randn(hierarchy_expected_prior_rng)
    expected_population_scale = randexp(hierarchy_expected_prior_rng)
    expected_individual = expected_population + expected_population_scale *
        randn(hierarchy_expected_prior_rng)
    expected_observation_scale = 2 * randexp(hierarchy_expected_prior_rng)
    expected_prior_response = [
        expected_individual + expected_observation_scale *
            randn(hierarchy_expected_prior_rng)
        for _ in response
    ]
    hierarchy_prior = NP.simulate_prior(
        hierarchy_prior_rng, hierarchy_workspace, hierarchy_prepared)
    @test hierarchy_prior.position ≈ [
        expected_population,
        log(expected_population_scale),
        expected_individual,
        log(expected_observation_scale),
    ]
    @test hierarchy_prior.response == expected_prior_response

    hierarchy_signature = NP.output_signature(
        hierarchy_prepared, NP.PosteriorPredictive())
    @test NP.output_axis(hierarchy_signature).keys ==
          Base.OneTo(length(response))
    @test NP.output_eltype(hierarchy_signature, hierarchy_prepared) === Float64
    hierarchy_rebound_response = [0.1, 0.4, 0.8]
    hierarchy_rebound = NP.rebind(
        hierarchy_prepared, (; y=hierarchy_rebound_response))
    @test NP.has_response(hierarchy_rebound)
    @test hierarchy_rebound.conditions.y == hierarchy_rebound_response
    @test length(hierarchy_rebound.plan.observation_axis) == 3
    hierarchy_prediction_only = NP.rebind(hierarchy_prepared, (;))
    @test !NP.has_response(hierarchy_prediction_only)
    @test length(hierarchy_prediction_only.plan.observation_axis) ==
          length(response)
    @test_throws ArgumentError NP.evaluate(
        NP.workspace(hierarchy_prediction_only), hierarchy_prediction_only,
        hierarchy_position, NP.PointwiseLogLikelihood())
    @test length(NP.simulate(
        MersenneTwister(923), NP.workspace(hierarchy_prediction_only),
        hierarchy_prediction_only, hierarchy_position)) == length(response)

    hierarchy_linear_buffer = similar(response)
    hierarchy_pointwise_buffer = similar(response)
    hierarchy_predictive_buffer = similar(response)
    hierarchy_prior_position = similar(hierarchy_position)
    @test factor_steady_state_allocations(
        hierarchy_workspace, hierarchy_prepared, hierarchy_position) ==
          (; primal=0, gradient=0)
    @test factor_query_allocations(
        hierarchy_workspace, hierarchy_prepared, hierarchy_position,
        hierarchy_linear_buffer, hierarchy_pointwise_buffer,
        hierarchy_predictive_buffer, hierarchy_prior_position) ==
          (; linear=0, pointwise=0, predictive=0, prior=0)

    hierarchy_positions = [
        hierarchy_position';
        0.1 log(0.9) 0.2 log(0.7);
        -0.3 log(1.1) 0.4 log(0.6);
    ]
    hierarchy_linear_draws = NP.evaluate_draws(
        hierarchy_workspace, hierarchy_prepared, hierarchy_positions,
        NP.LinearPredictor())
    hierarchy_pointwise_draws = NP.evaluate_draws(
        hierarchy_workspace, hierarchy_prepared, hierarchy_positions,
        NP.PointwiseLogLikelihood())
    for draw in axes(hierarchy_positions, 1)
        draw_position = collect(@view hierarchy_positions[draw, :])
        @test hierarchy_linear_draws[draw, :] == NP.evaluate(
            hierarchy_workspace, hierarchy_prepared, draw_position,
            NP.LinearPredictor())
        @test hierarchy_pointwise_draws[draw, :] == NP.evaluate(
            hierarchy_workspace, hierarchy_prepared, draw_position,
            NP.PointwiseLogLikelihood())
    end
    hierarchy_batch_signature = NP.batch_output_signature(
        hierarchy_prepared, hierarchy_positions, NP.LinearPredictor())
    @test NP.output_axes(hierarchy_batch_signature) == (
        BRM.NativePPLAxis(:draw, Base.OneTo(3)),
        hierarchy_prepared.plan.observation_axis)

    hierarchy_predictive_draws_rng = MersenneTwister(926)
    hierarchy_predictive_scalar_rng = MersenneTwister(926)
    hierarchy_predictive_draws = NP.simulate_draws(
        hierarchy_predictive_draws_rng, hierarchy_workspace,
        hierarchy_prepared, hierarchy_positions)
    hierarchy_predictive_scalar = similar(hierarchy_predictive_draws)
    for draw in axes(hierarchy_positions, 1)
        NP.simulate!(
            hierarchy_predictive_scalar_rng,
            @view(hierarchy_predictive_scalar[draw, :]),
            hierarchy_workspace, hierarchy_prepared,
            @view(hierarchy_positions[draw, :]))
    end
    @test hierarchy_predictive_draws == hierarchy_predictive_scalar

    hierarchy_bundle_queries = (;
        linear=NP.LinearPredictor(),
        pointwise=NP.PointwiseLogLikelihood(),
        predictive=NP.PosteriorPredictive())
    hierarchy_bundle_rng = MersenneTwister(927)
    hierarchy_bundle_predictive_rng = MersenneTwister(927)
    hierarchy_bundle = NP.execute_draws(
        hierarchy_bundle_rng, hierarchy_workspace, hierarchy_prepared,
        hierarchy_positions, hierarchy_bundle_queries)
    @test hierarchy_bundle.linear == hierarchy_linear_draws
    @test hierarchy_bundle.pointwise == hierarchy_pointwise_draws
    @test hierarchy_bundle.predictive == NP.simulate_draws(
        hierarchy_bundle_predictive_rng, hierarchy_workspace,
        hierarchy_prepared, hierarchy_positions)

    hierarchy_batch_linear = similar(hierarchy_linear_draws)
    hierarchy_batch_pointwise = similar(hierarchy_pointwise_draws)
    hierarchy_batch_predictive = similar(hierarchy_predictive_draws)
    hierarchy_batch_bundle = (;
        linear=similar(hierarchy_linear_draws),
        pointwise=similar(hierarchy_pointwise_draws),
        predictive=similar(hierarchy_predictive_draws))
    @test factor_batch_allocations(
        hierarchy_workspace, hierarchy_prepared, hierarchy_positions,
        hierarchy_batch_linear, hierarchy_batch_pointwise,
        hierarchy_batch_predictive, hierarchy_batch_bundle) ==
          (; linear=0, pointwise=0, predictive=0, bundle=0)
    hierarchy_aliased_bundle = similar(hierarchy_linear_draws)
    hierarchy_aliased_bundle_rng = MersenneTwister(928)
    hierarchy_aliased_bundle_control_rng = MersenneTwister(928)
    @test_throws ArgumentError NP.execute_draws!(
        hierarchy_aliased_bundle_rng,
        (; linear=hierarchy_aliased_bundle,
           predictive=hierarchy_aliased_bundle),
        hierarchy_workspace, hierarchy_prepared, hierarchy_positions,
        (; linear=NP.LinearPredictor(),
           predictive=NP.PosteriorPredictive()))
    @test rand(hierarchy_aliased_bundle_rng) ==
          rand(hierarchy_aliased_bundle_control_rng)

    hierarchy_empty_positions = zeros(Float32, 0, 4)
    hierarchy_empty_output = zeros(Float64, 0, length(response))
    @test_throws ArgumentError NP.evaluate_draws!(
        hierarchy_empty_output,
        NP.workspace(hierarchy_prepared, Float32), hierarchy_prepared,
        hierarchy_empty_positions, NP.LinearPredictor())

    natural_hierarchy = NP.condition(
        naturally_composed_hierarchy(); y=response)
    natural_hierarchy_graph = NP.factor_graph(natural_hierarchy)
    natural_population = NP.qualified_name(:population, :value)
    natural_population_scale = NP.qualified_name(
        :population_scale, :value)
    natural_individual = NP.qualified_name(:individual, :value)
    natural_observation_scale = NP.qualified_name(
        :y, :observation_scale)
    @test keys(natural_hierarchy_graph.sites) == (
        natural_population, natural_population_scale, natural_individual,
        natural_observation_scale, :y)
    @test NP.site_factor_dependencies(
        natural_hierarchy_graph.sites[natural_individual].factor) ==
          (natural_population, natural_population_scale)
    @test NP.site_factor_dependencies(
        natural_hierarchy_graph.sites.y.factor) ==
          (natural_individual, natural_observation_scale)

    explicit_population_component = NP.component(
        :population, hierarchy_population_source())
    explicit_population = NP.output(
        explicit_population_component, :value)
    explicit_population_scale_component = NP.component(
        :population_scale, hierarchy_scale_source(1.0))
    explicit_population_scale = NP.output(
        explicit_population_scale_component, :value)
    explicit_individual_component = NP.component(
        :individual,
        hierarchy_individual_source(
            explicit_population, explicit_population_scale))
    explicit_individual = NP.output(
        explicit_individual_component, :value)
    explicit_response_component = NP.component(
        :y,
        NP.condition(
            hierarchy_observation_source(explicit_individual);
            y=response))
    explicit_hierarchy = NP.compose(
        explicit_population_component,
        explicit_population_scale_component,
        explicit_individual_component,
        explicit_response_component)
    explicit_hierarchy_graph = NP.factor_graph(
        NP.lower(explicit_hierarchy))
    explicit_response = NP.qualified_name(:y, :y)
    @test keys(explicit_hierarchy_graph.sites) == (
        natural_population, natural_population_scale, natural_individual,
        natural_observation_scale, explicit_response)
    @test NP.site_factor_dependencies(
        explicit_hierarchy_graph.sites[natural_individual].factor) ==
          (natural_population, natural_population_scale)
    @test NP.site_factor_dependencies(
        explicit_hierarchy_graph.sites[explicit_response].factor) ==
          (natural_individual, natural_observation_scale)

    hierarchy_brm_data = (; y=response)
    hierarchy_brm = @brm hierarchy_brm_data begin
        population ~ Normal()
        population_scale ~ Exponential(1.0)
        individual ~ Normal(population, population_scale)
        observation_scale ~ Exponential(2.0)
        y ~ Normal(individual, observation_scale)
    end
    hierarchy_brm_model = NP.lower(hierarchy_brm)
    @test typeof(hierarchy_brm_model) === typeof(hierarchy.declaration)
    @test sprint(show, hierarchy_brm_model) ==
          sprint(show, hierarchy.declaration)
    hierarchy_brm_plan = NP.compile(hierarchy_brm)
    @test hierarchy_brm_plan isa NP.FactorPlan
    @test hierarchy_brm_plan.declaration.site_order ==
          hierarchy_plan.declaration.site_order

    natural_hierarchy_prepared = NP.prepare(natural_hierarchy)
    explicit_hierarchy_prepared = NP.prepare(NP.compile(explicit_hierarchy))
    hierarchy_brm_prepared = NP.prepare(hierarchy_brm)
    for candidate_prepared in (
        natural_hierarchy_prepared,
        explicit_hierarchy_prepared,
        hierarchy_brm_prepared)
        candidate_workspace = NP.workspace(
            candidate_prepared, Float64, DI.AutoEnzyme())
        candidate_density, candidate_gradient = NP.logdensity_and_gradient!(
            candidate_workspace, candidate_prepared, hierarchy_position)
        @test candidate_density ≈ hierarchy_density
        @test candidate_gradient ≈ hierarchy_gradient
        @test NP.evaluate(
            candidate_workspace, candidate_prepared, hierarchy_position,
            NP.LinearPredictor()) == hierarchy_linear
        @test NP.evaluate(
            candidate_workspace, candidate_prepared, hierarchy_position,
            NP.PointwiseLogLikelihood()) ≈ hierarchy_pointwise
        @test NP.simulate(
            MersenneTwister(928), candidate_workspace,
            candidate_prepared, hierarchy_position) == NP.simulate(
                MersenneTwister(928), hierarchy_workspace,
                hierarchy_prepared, hierarchy_position)
        candidate_prior = NP.simulate_prior(
            MersenneTwister(929), candidate_workspace, candidate_prepared)
        reference_prior = NP.simulate_prior(
            MersenneTwister(929), hierarchy_workspace, hierarchy_prepared)
        @test candidate_prior.position == reference_prior.position
        @test candidate_prior.response == reference_prior.response
        @test NP.execute_draws(
            MersenneTwister(930), candidate_workspace, candidate_prepared,
            hierarchy_positions, hierarchy_bundle_queries) ==
              NP.execute_draws(
                  MersenneTwister(930), hierarchy_workspace,
                  hierarchy_prepared, hierarchy_positions,
                  hierarchy_bundle_queries)
        @test factor_steady_state_allocations(
            candidate_workspace, candidate_prepared, hierarchy_position) ==
              (; primal=0, gradient=0)
    end

    bound_factor_declaration = NP.model(
        inputs=(; latent_scale=NP.input(), observation_scale=NP.input()),
        parameters=(;
            population=NP.parameter(
                NP.RealSupport(), (:population,);
                transform=NP.Identity(), prior=NP.StandardNormal())),
        observations=(;
            individual=NP.normal(
                :individual, :population, :latent_scale),
            y=NP.broadcasted(NP.normal(
                :y, :individual, :observation_scale))),
        outputs=(; y=:y),
        site_order=(:population, :individual, :y))
    bound_factor_plan = NP.compile(
        bound_factor_declaration,
        (; latent_scale=0.7, observation_scale=0.5);
        conditions=(; y=response))
    @test bound_factor_plan isa NP.FactorPlan
    @test bound_factor_plan.bindings ==
          (; latent_scale=0.7, observation_scale=0.5)
    @test bound_factor_plan.graph.sites.individual.factor.scale isa
          NP.InputValue{:latent_scale}
    @test bound_factor_plan.graph.sites.y.factor.scale isa
          NP.InputValue{:observation_scale}
    bound_factor_prepared = NP.prepare(bound_factor_plan)
    bound_factor_workspace = NP.workspace(
        bound_factor_prepared, Float64, DI.AutoEnzyme())
    bound_factor_position = [0.2, -0.1]
    bound_factor_density, bound_factor_gradient =
        NP.logdensity_and_gradient!(
            bound_factor_workspace, bound_factor_prepared,
            bound_factor_position)
    bound_latent_residual =
        bound_factor_position[2] - bound_factor_position[1]
    bound_observation_residuals = response .- bound_factor_position[2]
    @test bound_factor_density ≈
          logpdf(Normal(), bound_factor_position[1]) +
          logpdf(Normal(
              bound_factor_position[1], 0.7), bound_factor_position[2]) +
          sum(logpdf.(Normal(bound_factor_position[2], 0.5), response))
    @test bound_factor_gradient ≈ [
        -bound_factor_position[1] + bound_latent_residual / 0.7^2,
        -bound_latent_residual / 0.7^2 +
            sum(bound_observation_residuals) / 0.5^2,
    ]
    @test factor_steady_state_allocations(
        bound_factor_workspace, bound_factor_prepared,
        bound_factor_position) == (; primal=0, gradient=0)
    @test NP.rebind(bound_factor_prepared, (;)).plan.bindings ==
          bound_factor_plan.bindings
    @test_throws ArgumentError NP.prepare(NP.compile(
        bound_factor_declaration,
        (; latent_scale=Inf, observation_scale=0.5);
        conditions=(; y=response)))
    for (latent_scale, observation_scale) in ((0.0, 0.5), (0.7, -0.5))
        @test_throws ArgumentError NP.prepare(NP.compile(
            bound_factor_declaration,
            (; latent_scale, observation_scale);
            conditions=(; y=response)))
    end
    @test NP.LogDensityProblem(
        hierarchy_prepared, DI.AutoEnzyme()) isa NP.FactorLogDensityProblem
    conditioned_individual_plan = NP.compile(NP.condition(
        hierarchy; individual=0.1, y=response))
    @test conditioned_individual_plan isa NP.FactorPlan
    @test LogDensityProblems.dimension(conditioned_individual_plan) == 3
    conditioned_individual_prepared = NP.prepare(conditioned_individual_plan)
    @test isfinite(NP.logdensity!(
        NP.workspace(conditioned_individual_prepared),
        conditioned_individual_prepared,
        [0.2, log(0.7), log(0.5)]))
    @test_throws ArgumentError NP.compile(NP.condition(
        hierarchy; individual=[0.1], y=response))

    bad_scale_graph_model = NP.model(
        inputs=(;),
        parameters=(;
            bad_scale=NP.parameter(
                NP.RealSupport(), (:bad_scale,);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            likelihood_scale=NP.parameter(
                NP.PositiveSupport(), (:likelihood_scale,);
                transform=NP.Exp(), prior=NP.Exponential(1.0))),
        observations=(;
            child=NP.normal(:child, :bad_scale, :bad_scale),
            observed=NP.broadcasted(
                NP.normal(:observed, :child, :likelihood_scale))),
        site_order=(:bad_scale, :child, :likelihood_scale, :observed))
    @test capability_error(() -> NP.compile(NP.condition(
        bad_scale_graph_model; observed=response))).capability == :factor_scale

    broadcast_parent_graph_model = NP.model(
        inputs=(;),
        parameters=(;
            root=NP.parameter(
                NP.RealSupport(), (:root,);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            downstream_scale=NP.parameter(
                NP.PositiveSupport(), (:downstream_scale,);
                transform=NP.Exp(), prior=NP.Exponential(1.0))),
        observations=(;
            observed=NP.broadcasted(
                NP.normal(:observed, :root, :downstream_scale)),
            child=NP.normal(:child, :observed, :downstream_scale)),
        site_order=(:root, :downstream_scale, :observed, :child))
    @test capability_error(() -> NP.compile(NP.condition(
        broadcast_parent_graph_model; observed=response))).capability ==
          :factor_dependencies

    broadcast_node_graph_model = NP.model(
        inputs=(;),
        parameters=(;
            root=NP.parameter(
                NP.RealSupport(), (:root,);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            likelihood_scale=NP.parameter(
                NP.PositiveSupport(), (:likelihood_scale,);
                transform=NP.Exp(), prior=NP.Exponential(1.0))),
        nodes=(; derived_scale=NP.exp_link(:observed)),
        observations=(;
            observed=NP.broadcasted(
                NP.normal(:observed, :root, :likelihood_scale)),
            child=NP.normal(:child, :root, :derived_scale)),
        site_order=(:root, :likelihood_scale, :observed, :child))
    @test capability_error(() -> NP.compile(NP.condition(
        broadcast_node_graph_model; observed=response))).capability ==
          :factor_dependencies
    @test NP.factor_graph(hierarchy).sites.y.activity isa NP.GeneratedSite
    conditioned_population_graph = NP.factor_graph(
        hierarchy.declaration;
        conditions=(; population=0.1, y=response))
    @test conditioned_population_graph.sites.population.activity isa
          NP.ConditionedSite
    @test conditioned_population_graph.dimension == 3
    @test keys(conditioned_population_graph.coordinates) ==
          (:population_scale, :individual, :observation_scale)

    natural_factor_graph = NP.factor_graph(conditioned_natural_latent)
    @test keys(natural_factor_graph.sites) ==
          (NP.qualified_name(:z, :z), NP.qualified_name(:y, :sigma), :y)
    @test natural_factor_graph.dimension == 2
    @test keys(natural_factor_graph.coordinates) ==
          (NP.qualified_name(:z, :z), NP.qualified_name(:y, :sigma))

    unordered_graph_model = NP.model(
        inputs=(; literal_mu=NP.input(), literal_tau=NP.input()),
        observations=(;
            upstream=NP.normal(:upstream, :literal_mu, :literal_tau),
            downstream=NP.normal(:downstream, :upstream, :literal_tau)),
        site_order=(:downstream, :upstream))
    @test capability_error(
        () -> NP.factor_graph(unordered_graph_model)).capability == :site_order

    unordered_storage_graph_model = NP.model(
        inputs=(; literal_mu=NP.input(), literal_tau=NP.input()),
        observations=(;
            downstream=NP.normal(:downstream, :upstream, :literal_tau),
            upstream=NP.normal(:upstream, :literal_mu, :literal_tau)),
        outputs=(; public_downstream=:downstream),
        site_order=(:upstream, :downstream))
    unordered_storage_graph = NP.factor_graph(
        unordered_storage_graph_model;
        conditions=(; public_downstream=0.3))
    @test keys(unordered_storage_graph.sites) == (:upstream, :downstream)
    @test NP.site_factor_dependencies(
        unordered_storage_graph.sites.downstream.factor) ==
          (:upstream,)
    @test unordered_storage_graph.sites.downstream.factor.scale isa
          NP.InputValue{:literal_tau}
    @test unordered_storage_graph.sites.downstream.activity isa
          NP.ConditionedSite
    @test_throws ArgumentError NP.factor_graph(
        unordered_storage_graph_model;
        conditions=(; downstream=0.3, public_downstream=0.3))

    cyclic_graph_model = NP.model(
        inputs=(; literal_tau=NP.input(),),
        observations=(;
            first_site=NP.normal(:first_site, :second_site, :literal_tau),
            second_site=NP.normal(:second_site, :first_site, :literal_tau)),
        site_order=(:first_site, :second_site))
    @test capability_error(
        () -> NP.factor_graph(cyclic_graph_model)).capability == :site_order

    legacy_response_graph_model = NP.model(
        inputs=(;
            literal_mu=NP.input(), literal_tau=NP.input(),
            legacy_y=NP.input(:response)),
        observations=(;
            legacy_y=NP.broadcasted(
                NP.normal(:legacy_y, :literal_mu, :literal_tau))))
    legacy_response_graph = NP.factor_graph(NP.instantiate(
        legacy_response_graph_model,
        (; literal_mu=0.0, literal_tau=1.0, legacy_y=response)))
    @test legacy_response_graph.sites.legacy_y.activity isa NP.ConditionedSite
    @test legacy_response_graph.dimension == 0

    latent_brm_data = (; y=response)
    latent_brm = @brm latent_brm_data begin
        sigma ~ Exponential(2.0)
        mu ~ 1
        effect(mu, Intercept) ~ Normal(0.25, 0.8)
        y ~ Normal(mu, sigma)
    end
    latent_brm_model = NP.lower(latent_brm)
    @test isempty(latent_brm_model.inputs)
    @test latent_brm_model.parameters.mu.prior isa NP.NormalPrior
    @test latent_brm_model.parameters.mu.prior.location == 0.25
    @test latent_brm_model.parameters.mu.prior.scale == 0.8
    latent_brm_plan = NP.compile(latent_brm)
    @test latent_brm_plan.factors.coefficient_prior isa
          BRM.NativePPLScalarNormalFactor
    latent_brm_prepared = NP.prepare(latent_brm_plan)
    latent_brm_workspace = NP.workspace(
        latent_brm_prepared, Float64, DI.AutoEnzyme())
    latent_brm_density, latent_brm_gradient = NP.logdensity_and_gradient!(
        latent_brm_workspace, latent_brm_prepared, latent_position)
    @test latent_brm_density ≈ latent_density
    @test latent_brm_gradient ≈ latent_gradient
    @test NP.simulate(
        MersenneTwister(913), latent_brm_workspace,
        latent_brm_prepared, latent_position) == NP.simulate(
            MersenneTwister(913), latent_workspace,
            latent_prepared, latent_position)

    scale_named_plan = NP.compile(NP.condition(
        monolithic_scale_named_site(0.25, 0.8);
        y=response))
    @test keys(scale_named_plan.parameters) == (:site, :scale)
    @test BRM.native_parameter_name(
        scale_named_plan.parameters.site) === :scale
    scale_named_prepared = NP.prepare(scale_named_plan)
    scale_named_density, scale_named_gradient = NP.logdensity_and_gradient!(
        NP.workspace(scale_named_prepared, Float64, DI.AutoEnzyme()),
        scale_named_prepared, latent_position)
    @test scale_named_density ≈ latent_density
    @test scale_named_gradient ≈ latent_gradient

    latent_rebound_response = [0.1, 0.4, 0.8]
    latent_rebound = NP.rebind(
        latent_prepared,
        NamedTuple{(latent_response_name,)}((latent_rebound_response,)))
    @test latent_rebound.response == latent_rebound_response
    @test length(latent_rebound.plan.axes.observation) == 3
    latent_prediction = NP.rebind(latent_prepared, (;))
    @test !NP.has_response(latent_prediction)
    @test length(latent_prediction.plan.axes.observation) == length(response)
    @test length(NP.simulate(
        MersenneTwister(914), NP.workspace(latent_prediction),
        latent_prediction, latent_position)) == length(response)
    @test_throws ArgumentError NP.evaluate(
        NP.workspace(latent_prediction), latent_prediction, latent_position,
        NP.PointwiseLogLikelihood())

    prior_rng = MersenneTwister(915)
    expected_site = 0.25 + 0.8 * randn(prior_rng)
    expected_scale = 2.0 * randexp(prior_rng)
    expected_prior_response = [
        expected_site + expected_scale * randn(prior_rng)
        for _ in eachindex(response)
    ]
    prior_position = zeros(2)
    prior_response = similar(response)
    @test NP.simulate_prior!(
        MersenneTwister(915), prior_position, prior_response,
        latent_workspace, latent_prepared) === prior_response
    @test prior_position ≈ [expected_site, log(expected_scale)]
    @test prior_response == expected_prior_response
    allocated_prior = NP.simulate_prior(
        MersenneTwister(915), latent_workspace, latent_prepared)
    @test allocated_prior.position == prior_position
    @test allocated_prior.response == prior_response
    monolithic_prior = NP.simulate_prior(
        MersenneTwister(915), monolithic_latent_workspace,
        monolithic_latent)
    @test monolithic_prior == allocated_prior
    @test NP.simulate_prior(
        MersenneTwister(915), latent_brm_workspace,
        latent_brm_prepared) == allocated_prior
    @test NP.simulate_prior(
        MersenneTwister(915), natural_latent_workspace,
        natural_latent_prepared) == allocated_prior
    prediction_prior = NP.simulate_prior(
        MersenneTwister(915), NP.workspace(latent_prediction),
        latent_prediction)
    @test prediction_prior == allocated_prior
    allocation_rng = MersenneTwister(916)
    NP.simulate_prior!(
        allocation_rng, prior_position, prior_response,
        latent_workspace, latent_prepared)
    @test @allocated(NP.simulate_prior!(
        allocation_rng, prior_position, prior_response,
        latent_workspace, latent_prepared)) == 0

    two_row_latent = NP.rebind(
        latent_prepared,
        NamedTuple{(latent_response_name,)}(([0.1, 0.4],)))
    two_row_workspace = NP.workspace(two_row_latent)
    for (position_buffer, output_buffer) in (
        (two_row_workspace.gradient, zeros(2)),
        (two_row_latent.response, zeros(2)),
        (zeros(2), two_row_latent.response),
    )
        rejected_rng = MersenneTwister(917)
        control_rng = MersenneTwister(917)
        @test_throws ArgumentError NP.simulate_prior!(
            rejected_rng, position_buffer, output_buffer,
            two_row_workspace, two_row_latent)
        @test rand(rejected_rng) == rand(control_rng)
    end

    latent_prepared32 = NP.prepare(latent_plan; T=Float32)
    latent_density32, latent_gradient32 = NP.logdensity_and_gradient!(
        NP.workspace(latent_prepared32, Float32, DI.AutoEnzyme()),
        latent_prepared32, Float32.(latent_position))
    @test latent_density32 ≈ Float32(latent_expected_density) rtol=1f-5
    @test latent_gradient32 ≈ Float32.(latent_expected_gradient) rtol=1f-5

    aliased_prior = aliased_scalar_normal_prior()
    @test keys(aliased_prior.declaration.parameters) == (:theta,)
    @test aliased_prior.declaration.outputs == (; coefficient=:theta)
    aliased_prior_component = NP.component(:aliased_prior, aliased_prior)
    @test NP.graph_kind(
        NP.output(aliased_prior_component, :coefficient)) === :parameter
    @test_throws ArgumentError NP.output(aliased_prior_component, :theta)

    named_priors = named_scalar_normal_priors()
    @test keys(named_priors.declaration.parameters) == (:intercept, :slope)
    @test named_priors.declaration.outputs ==
          (; intercept=:intercept, slope=:slope)
    @test_throws ArgumentError NP.model(
        inputs=(; raw=NP.input()),
        nodes=(; scaled=NP.zscale(:raw)),
        observations=(;), outputs=(; missing=:unknown))
    @test_throws ArgumentError NP.model(
        inputs=(; raw=NP.input()),
        nodes=(; scaled=NP.zscale(:raw)),
        observations=(;), outputs=(; first=:scaled, second=:scaled))
    @test_throws ArgumentError NP.model(
        inputs=(; literal_mu=NP.input(), literal_tau=NP.input()),
        observations=(;
            first_site=NP.normal(:first_site, :literal_mu, :literal_tau),
            second_site=NP.normal(
                :second_site, :literal_mu, :literal_tau)),
        outputs=(; first_site=:second_site))
    @test_throws ArgumentError NP.model(
        inputs=(; raw=NP.input()), observations=(;), outputs=(;))

    unconditioned = macro_gaussian_identity(raw_x)
    @test keys(unconditioned.declaration.inputs) == (:x,)
    @test unconditioned.declaration.inputs.x isa NP.Input{:value}
    @test unconditioned.declaration.observations.y isa NP.BroadcastObservation
    @test isempty(unconditioned.conditions)
    unconditioned_prepared = NP.prepare(unconditioned)
    @test !NP.has_response(unconditioned_prepared)
    @test eltype(NP.simulate(
        MersenneTwister(905), NP.workspace(unconditioned_prepared),
        unconditioned_prepared, [0.2, -0.4, log(0.8)])) === Float64
    scalar_instance = NP.condition(
        macro_scalar_gaussian(raw_x); y=[0.2, -0.1, 1.1, 0.7])
    @test !(
        scalar_instance.declaration.observations.y isa NP.BroadcastObservation)
    @test capability_error(() -> NP.bind(scalar_instance)).capability ==
          :broadcast_lifting
    cases = (
        (instance=NP.condition(
             macro_gaussian_identity(raw_x);
             y=[0.2, -0.1, 1.1, 0.7]),
         builder=direct_native_model(
             :gaussian, :identity;
             coefficient_keys=(:intercept, :slope)),
         position=[0.2, -0.4, log(0.8)]),
        (instance=NP.condition(
             macro_gaussian_zscale(raw_x);
             y=[0.2, -0.1, 1.1, 0.7]),
         builder=direct_native_model(
             :gaussian, :zscale;
             coefficient_keys=(:intercept, :slope)),
         position=[0.2, -0.4, log(0.8)]),
        (instance=NP.condition(
             macro_bernoulli_center(raw_x);
             y=Bool[false, false, true, true]),
         builder=direct_native_model(
             :bernoulli, :center;
             coefficient_keys=(:intercept, :slope)),
         position=[-0.3, 0.6]),
        (instance=NP.condition(
             macro_poisson_zscale(raw_x); y=[0, 1, 3, 6]),
         builder=direct_native_model(
             :poisson, :zscale;
             coefficient_keys=(:intercept, :slope)),
         position=[0.1, 0.2]),
    )
    for (; instance, builder, position) in cases
        @test instance isa NP.ModelInstance
        @test typeof(instance.declaration) === typeof(builder)
        @test keys(instance.bindings) == keys(builder.inputs)
        @test occursin("NativePPL.ModelInstance", sprint(show, instance))
        macro_plan = NP.bind(instance)
        @test typeof(NP.compile(instance)) === typeof(macro_plan)
        builder_plan = NP.bind(NP.condition(
            NP.substitute(builder, instance.bindings), instance.conditions))
        check_plan_structure(macro_plan, builder_plan)
        prepared = check_transformed_execution(
            macro_plan, builder_plan, position)
        @test NP.prepare(instance).predictor == prepared.predictor
        @test steady_state_allocations(
            NP.workspace(prepared, Float64, DI.AutoEnzyme()),
            prepared, position) == (; primal=0, gradient=0)

        prediction = NP.rebind(prepared, (; x=[10.0, 14.0]))
        builder_prediction = NP.rebind(
            NP.prepare(builder_plan), (; x=[10.0, 14.0]))
        @test !NP.has_response(prediction)
        @test prediction.predictor == builder_prediction.predictor
        @test NP.simulate(
            MersenneTwister(907), NP.workspace(prediction),
            prediction, position) ==
              NP.simulate(
                  MersenneTwister(907), NP.workspace(builder_prediction),
                  builder_prediction, position)
    end

    hygienic = NP.condition(
        NativePPLMacroHygiene.hygienic(raw_x);
        y=[0.2, -0.1, 1.1, 0.7])
    @test hygienic isa NP.ModelInstance
    @test NP.compile(hygienic) isa NP.Plan
    @test NP.prepare(hygienic) isa NP.Prepared
    @test_throws MethodError macro_gaussian_identity(1.0, [1.0])

    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model bad_model = nothing)))
    @test occursin("must wrap a function definition", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function argument_observation(x, y)
            @. y ~ Normal(x, 1)
        end)))
    @test occursin("cannot also be a function argument", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function bad_keyword(x, y; scale=1)
            y ~ Normal(x, scale)
        end)))
    @test occursin("plain or typed positional arguments", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function bad_statement(x)
            println(x)
            @. y ~ Normal(x, x)
        end)))
    @test occursin("unsupported statement", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function bad_prior(x)
            sigma ~ StandardNormal(1)
            @. y ~ Normal(x, sigma)
        end)))
    @test occursin("StandardNormal() takes no arguments", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function missing_observation(x)
            centered = center(x)
        end)))
    @test occursin("at least one stochastic site", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function nonfinal_return(x)
            centered = center(x)
            return centered
            scaled = zscale(x)
        end)))
    @test occursin("return must be the final statement", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function duplicate_return(x)
            centered = center(x)
            return (first=centered, second=centered)
        end)))
    @test occursin("returned graph values must be distinct", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function empty_return(x)
            return (;)
        end)))
    @test occursin("at least one named graph value", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function mixed_scalar_prior_component(x)
            intercept ~ Normal()
            slope ~ Normal()
            theta ~ Normal()
            mu = intercept + slope * x
            return theta
        end)))
    @test occursin("cannot yet mix explicitly returned scalar priors", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function bad_link(x)
            intercept ~ Normal()
            slope ~ Normal()
            log_rate = intercept + slope * x
            @. y ~ Poisson(log_rate + x)
        end)))
    @test occursin("named rate or `exp(named_log_rate)`", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function repeated_coefficient(x, w)
            intercept ~ Normal()
            beta ~ Normal()
            sigma ~ Exponential(1.0)
            mu = intercept + beta * x + beta * w
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("coefficients must be used once each", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function repeated_path(x, w)
            intercept ~ Normal()
            beta_x ~ Normal()
            beta_w ~ Normal()
            sigma ~ Exponential(1.0)
            mu = intercept + beta_x * x + beta_w * x
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("predictor paths must be unique", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function repeated_raw_input(x, w)
            intercept ~ Normal()
            beta_x ~ Normal()
            beta_centered_x ~ Normal()
            sigma ~ Exponential(1.0)
            mu = intercept + beta_x * x + beta_centered_x * center(x)
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("features must use distinct raw inputs", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function unknown_data_offset(x)
            beta ~ Normal()
            sigma ~ Exponential(1)
            mu = beta * x + offset(expsoure)
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin(
        "must name a preceding scalar site or function argument", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function unknown_log_offset(x)
            beta ~ Normal()
            sigma ~ Exponential(1)
            mu = beta * x + offset(log(expsoure))
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("must name a function argument", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function mismatched_parameter_dot(x, w)
            beta[(:x, :w)] ~ StandardNormal()
            extra ~ Normal()
            sigma ~ Exponential(2)
            mu = dot(beta, (w, x)) + extra * x
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("must match its declared coefficient keys", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function mixed_parameter_dot(x, w)
            beta[(:x, :w)] ~ StandardNormal()
            extra ~ Normal()
            sigma ~ Exponential(2)
            mu = dot(beta, (x, w)) + extra * x
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin(
        "cannot mix a parameter dot with scalar population coefficients",
        err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function unsupported_grouped_transform(
                x, group)
            tau[(:Intercept, :x)] ~ Exponential(1)
            L[(:Intercept, :x)] ~ LKJCholesky(2, 2)
            b[group, (:Intercept, :x)] ~ MvNormalCholesky(tau, L)
            beta[(:x,)] ~ StandardNormal()
            sigma ~ Exponential(2)
            mu = dot(beta, (log(x),)) + dot(b[group], (1, log(x)))
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("support only center(input) or zscale(input)", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function expression_grouped_transform(
                x, group)
            tau[(:Intercept, :x)] ~ Exponential(1)
            L[(:Intercept, :x)] ~ LKJCholesky(2, 2)
            b[group, (:Intercept, :x)] ~ MvNormalCholesky(tau, L)
            beta[(:x,)] ~ StandardNormal()
            sigma ~ Exponential(2)
            mu = dot(beta, (zscale(x + 1),)) +
                dot(b[group], (1, zscale(x + 1)))
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("requires one named input", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function grouped_nonargument(x)
            tau ~ Exponential(1)
            b[group] ~ Normal(0, tau)
            beta ~ Normal()
            sigma ~ Exponential(2)
            mu = beta * x + b[group]
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("must index one function argument", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function grouped_wrong_family(x, group)
            tau ~ Exponential(1)
            b[group] ~ Exponential(tau)
            beta ~ Normal()
            sigma ~ Exponential(2)
            mu = beta * x + b[group]
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("requires Normal(location, scale)", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function grouped_wrong_gather(x, group, other)
            tau ~ Exponential(1)
            b[group] ~ Normal(0, tau)
            beta ~ Normal()
            sigma ~ Exponential(2)
            mu = beta * x + b[other]
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("gathered with its declared group input", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function repeated_grouped_offset(x, group)
            tau ~ Exponential(1)
            b[group] ~ Normal(0, tau)
            beta ~ Normal()
            sigma ~ Exponential(2)
            mu = beta * x + b[group] + b[group]
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("grouped offsets must be used once each", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function repeated_correlated_group(
                x, group)
            tau[(:Intercept, :x)] ~ Exponential(1)
            L[(:Intercept, :x)] ~ LKJCholesky(2, 2)
            b[group, (:Intercept, :x)] ~ MvNormalCholesky(tau, L)
            beta ~ Normal()
            sigma ~ Exponential(2)
            mu = beta * x + dot(b[group], (1, x)) +
                dot(b[group], (1, x))
            @. y ~ Normal(mu, sigma)
        end)))
    @test occursin("distinct public node identities", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function recursive_model(x)
            z ~ recursive_model(x)
            return z
        end)))
    @test occursin("cannot recursively stage itself", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function mixed_submodel(x)
            z ~ scalar_normal_site(0.0, 1.0)
            sigma ~ Exponential(1.0)
            return z
        end)))
    @test occursin("cannot yet mix staged submodel connections", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function missing_submodel_return(x)
            z ~ scalar_normal_site(0.0, 1.0)
        end)))
    @test occursin("requires an explicit returned graph value", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function returned_upstream_site(x)
            z ~ scalar_normal_site(0.0, 1.0)
            y ~ scalar_normal_likelihood(z)
            return z
        end)))
    @test occursin("public site aliases currently require a terminal", err.msg)
end


@testset "typed native PPL observation weights" begin
    weight = NP.observation_weight(:analytic, :replicates)
    @test NP.observation_weight_kind(weight) === :analytic
    @test NP.observation_weight_source(weight) === :replicates
    @test_throws ArgumentError NP.observation_weight(:probability, :weights)

    weighted = NP.weighted_observation(
        NP.normal(:y, :mu, :sigma), weight)
    declaration = NP.model(
        inputs=(; x=NP.input(), replicates=NP.input()),
        parameters=(;
            beta_mu=NP.parameter(
                NP.RealSupport(), (:Intercept, :x);
                transform=NP.Identity(), prior=NP.StandardNormal()),
            sigma=NP.parameter(
                NP.PositiveSupport(), (:sigma,);
                transform=NP.Exp(), prior=NP.Exponential(2.0))),
        nodes=(; mu=NP.affine(:x, :beta_mu)),
        observations=(; y=NP.broadcasted(weighted)))
    graph = NP.factor_graph(
        declaration;
        bindings=(; x=[-1.0, 0.5], replicates=[1, 4]),
        conditions=(; y=[-0.2, 0.8]))
    factor = graph.sites.y.factor
    @test factor isa NP.WeightedSiteFactor
    @test factor.factor isa NP.NormalSiteFactor
    @test factor.values isa NP.InputValue{:replicates}
    @test NP.observation_weight_kind(factor.weight) === :analytic
    @test NP.site_factor_dependencies(factor) == (:mu, :sigma)

    err = capability_error(() -> NP.model(
        inputs=(; x=NP.input()),
        parameters=declaration.parameters,
        nodes=declaration.nodes,
        observations=declaration.observations))
    @test err.capability == :graph_identity
    @test occursin("replicates", err.detail)

    function weighted_prepared(kind, values)
        weight = NP.observation_weight(kind, :observation_weights)
        observation = NP.weighted_observation(
            NP.normal(:y, :mu, :sigma), weight)
        declaration = NP.model(
            inputs=(; x=NP.input(), observation_weights=NP.input()),
            parameters=(;
                beta_mu=NP.parameter(
                    NP.RealSupport(), (:Intercept, :x);
                    transform=NP.Identity(), prior=NP.StandardNormal()),
                sigma=NP.parameter(
                    NP.PositiveSupport(), (:sigma,);
                    transform=NP.Exp(), prior=NP.Exponential(2.0))),
            nodes=(; mu=NP.affine(:x, :beta_mu)),
            observations=(; y=NP.broadcasted(observation)))
        NP.prepare(NP.bind(
            declaration,
            (; x=[-1.0, 0.5, 2.0], observation_weights=values);
            conditions=(; y=[-0.4, 0.3, 1.2])))
    end

    position = [0.2, 0.4, log(1.1)]
    response = [-0.4, 0.3, 1.2]
    location = 0.2 .+ 0.4 .* [-1.0, 0.5, 2.0]
    scale = 1.1
    unit_prepared = weighted_prepared(:unit, ones(3))
    unit_work = NP.workspace(unit_prepared)
    unit_density = NP.logdensity!(unit_work, unit_prepared, position)
    base_pointwise = logpdf.(Normal.(location, scale), response)
    prior_density = unit_density - sum(base_pointwise)
    @test NP.evaluate(
        unit_work, unit_prepared, position,
        NP.PointwiseLogLikelihood()) ≈ base_pointwise

    analytic_weights = [1.0, 4.0, 2.5]
    analytic_prepared = weighted_prepared(:analytic, analytic_weights)
    analytic_work = NP.workspace(analytic_prepared)
    analytic_pointwise = logpdf.(
        Normal.(location, scale ./ sqrt.(analytic_weights)), response)
    @test NP.logdensity!(
        analytic_work, analytic_prepared, position) ≈
        prior_density + sum(analytic_pointwise)
    @test NP.evaluate(
        analytic_work, analytic_prepared, position,
        NP.PointwiseLogLikelihood()) ≈ analytic_pointwise
    analytic_rng = MersenneTwister(707)
    analytic_expected_rng = MersenneTwister(707)
    @test NP.simulate(
        analytic_rng, analytic_work, analytic_prepared, position) ==
        location .+ scale ./ sqrt.(analytic_weights) .*
            [randn(analytic_expected_rng) for _ in response]

    for (kind, values) in (
            (:frequency, [1, 3, 0]),
            (:power, [0.25, 1.0, 2.5]))
        prepared = weighted_prepared(kind, values)
        work = NP.workspace(prepared)
        expected_pointwise = values .* base_pointwise
        @test NP.logdensity!(work, prepared, position) ≈
              prior_density + sum(expected_pointwise)
        @test NP.evaluate(
            work, prepared, position,
            NP.PointwiseLogLikelihood()) ≈ expected_pointwise
        rng = MersenneTwister(708)
        expected_rng = MersenneTwister(708)
        @test NP.simulate(rng, work, prepared, position) ==
              location .+ scale .*
                [randn(expected_rng) for _ in response]
    end

    weighted_brm_data = (;
        x=[-1.0, 0.5, 2.0],
        group=[1, 1, 2],
        replicates=[1, 4, 2],
        y=response)
    weighted_brmi = @brm weighted_brm_data begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | g | group)
        sd(:, g) ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), aweights(replicates))
    end
    natural_weighted = NP.condition(
        natural_weighted_varying_intercept(
            weighted_brm_data.x, weighted_brm_data.group,
            weighted_brm_data.replicates);
        y=weighted_brm_data.y)
    lowered_weighted = NP.lower(weighted_brmi)
    @test typeof(lowered_weighted) ===
          typeof(natural_weighted.declaration)
    @test sprint(show, lowered_weighted) ==
          sprint(show, natural_weighted.declaration)
    @test keys(lowered_weighted.inputs) == (:x, :group, :replicates)
    @test lowered_weighted.observations.y.scalar isa NP.WeightedObservation
    @test NP.observation_weight_kind(
        lowered_weighted.observations.y.scalar.weight) === :analytic
    @test NP.observation_weight_source(
        lowered_weighted.observations.y.scalar.weight) === :replicates
    weighted_plan = NP.compile(weighted_brmi)
    @test weighted_plan isa NP.FactorPlan
    @test weighted_plan.graph.sites.y.factor isa NP.WeightedSiteFactor
    @test weighted_plan.graph.schedule == (
        :tau_g_group, :b_g_group, :beta_mu, :sigma,
        :r_mu_g_group, :mu, :y)
    @test SBBRMI(weighted_brmi; mod=@__MODULE__) isa SBBRMI
end

@testset "native PPL workflow queries and replay" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.5, 1.0, 2.5])
    brmi = @brm data begin
        sigma ~ Exponential(2.5)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    plan = BRM.NativePPL.compile(brmi)
    prepared = BRM.NativePPL.prepare(plan)
    workspace = BRM.NativePPL.workspace(prepared)
    position = [0.1, -0.2, log(1.3)]
    location = position[1] .+ position[2] .* prepared.predictor
    pointwise = logpdf.(Normal.(location, exp(position[3])), prepared.response)

    @test plan isa BRM.NativePPL.Plan
    @test prepared isa BRM.NativePPL.Prepared
    @test workspace isa BRM.NativePPL.Workspace

    linear_query = BRM.NativePPL.LinearPredictor()
    linear_signature = BRM.NativePPL.output_signature(plan, linear_query)
    @test linear_signature isa BRM.NativePPL.OutputSignature
    @test BRM.NativePPL.output_signature(prepared, linear_query) === linear_signature
    @test BRM.NativePPL.output_axis(linear_signature) === plan.axes.observation
    @test BRM.NativePPL.output_eltype(linear_signature, prepared) == Float64
    @test BRM.NativePPL.output_eltype(linear_signature, Float32) == Float32
    @test BRM.NativePPL.output_layout(linear_signature) isa
          BRM.NativePPL.DenseVectorLayout
    @test_throws MethodError BRM.NativePPL.output_eltype(linear_signature, Int)

    @test BRM.NativePPL.output_signature(
        plan, BRM.NativePPL.PointwiseLogLikelihood()) === linear_signature
    @test BRM.NativePPL.output_signature(
        plan, BRM.NativePPL.PosteriorPredictive()) === linear_signature

    allocated_output = BRM.NativePPL.allocate_output(prepared, linear_query)
    @test allocated_output isa Vector{Float64}
    @test axes(allocated_output, 1) == plan.axes.observation.keys
    @test length(allocated_output) == length(plan.axes.observation)
    @test BRM.NativePPL.allocate_output(linear_signature, prepared) isa
          Vector{Float64}

    bad_signature = BRM.NativePPL.OutputSignature(
        BRM.NativePPLAxis(:observation, (:a, :b)),
        linear_signature.element_type, linear_signature.layout)
    err = capability_error(
        () -> BRM.NativePPL.allocate_output(bad_signature, prepared))
    @test err.capability == :output_layout
    @test BRM.NativePPL.logdensity!(workspace, prepared, position) ≈
          BRM._native_ppl_logdensity!(workspace, prepared, position)
    @test_throws ArgumentError BRM.NativePPL.logdensity_and_gradient!(
        workspace, prepared, position)

    linear_output = similar(prepared.response)
    @test BRM.NativePPL.evaluate!(
        linear_output, workspace, prepared, position,
        linear_query) === linear_output
    @test linear_output ≈ location

    allocated_linear = BRM.NativePPL.evaluate(
        workspace, prepared, position, linear_query)
    @test allocated_linear isa Vector{Float64}
    @test allocated_linear ≈ location
    @test allocated_linear !== linear_output

    pointwise_output = similar(prepared.response)
    @test BRM.NativePPL.evaluate!(
        pointwise_output, workspace, prepared, position,
        BRM.NativePPL.PointwiseLogLikelihood()) === pointwise_output
    @test pointwise_output ≈ pointwise

    predictive = similar(prepared.response)
    expected_predictive = copy(location)
    expected_rng = MersenneTwister(41)
    for i in eachindex(expected_predictive)
        expected_predictive[i] += exp(position[3]) * randn(expected_rng, Float64)
    end
    @test BRM.NativePPL.simulate!(
        MersenneTwister(41), predictive, workspace, prepared, position) === predictive
    @test predictive ≈ expected_predictive

    allocated_predictive = BRM.NativePPL.simulate(
        MersenneTwister(41), workspace, prepared, position)
    @test allocated_predictive isa Vector{Float64}
    @test allocated_predictive ≈ expected_predictive
    @test allocated_predictive !== predictive

    allocating_bytes = allocating_query_bytes(workspace, prepared, position)
    @test allocating_bytes.evaluation == allocating_bytes.allocation
    @test allocating_bytes.simulation == allocating_bytes.allocation

    BRM.NativePPL.evaluate!(
        linear_output, workspace, prepared, position,
        linear_query)
    query_allocations = @allocated(BRM.NativePPL.evaluate!(
        linear_output, workspace, prepared, position,
        linear_query))
    BRM.NativePPL.simulate!(
        MersenneTwister(7), predictive, workspace, prepared, position)
    rng = MersenneTwister(7)
    simulation_allocations = @allocated(BRM.NativePPL.simulate!(
        rng, predictive, workspace, prepared, position))
    @test (; query_allocations, simulation_allocations) ==
          (; query_allocations=0, simulation_allocations=0)

    rebound = BRM.NativePPL.rebind(
        prepared, (; x=Float32[3, 4], y=Float32[2, 3]); T=Float32)
    @test rebound isa BRM.NativePPL.Prepared
    @test eltype(rebound) == Float32
    @test rebound.predictor == Float32[3, 4]
    @test rebound.response == Float32[2, 3]
    @test rebound.plan.parameters === prepared.plan.parameters
    @test rebound.plan.axes.observation.keys == Base.OneTo(2)
    @test rebound.plan.nodes.location.axis === rebound.plan.axes.observation
    @test rebound.plan.factors.likelihood.axis === rebound.plan.axes.observation
    @test all(
        BRM.NativePPL.output_axis(BRM.NativePPL.output_signature(rebound, query)) ===
            rebound.plan.axes.observation
        for query in (
            BRM.NativePPL.LinearPredictor(),
            BRM.NativePPL.PointwiseLogLikelihood(),
            BRM.NativePPL.PosteriorPredictive(),
        ))
    @test prepared.plan.axes.observation.keys == Base.OneTo(3)

    rebound_output = BRM.NativePPL.allocate_output(
        rebound, BRM.NativePPL.LinearPredictor())
    @test rebound_output isa Vector{Float32}
    @test length(rebound_output) == 2

    prediction_only = BRM.NativePPL.rebind(
        prepared, (; x=Float32[3, 4]); T=Float32)
    @test !BRM.NativePPL.has_response(prediction_only)
    @test prediction_only.response isa BRM.NativePPLNoResponse
    @test eltype(prediction_only) == Float32
    @test prediction_only.plan.axes.observation.keys == Base.OneTo(2)
    @test keys(prediction_only.plan.bindings) == (:x,)
    @test sprint(show, prediction_only) ==
          "NativePPLPrepared(2 observations, eltype=Float32, prediction-only)"
    prediction_only_workspace = BRM.NativePPL.workspace(prediction_only)
    prediction_only_position = Float32.(position)
    prediction_only_location = BRM.NativePPL.evaluate(
        prediction_only_workspace, prediction_only, prediction_only_position,
        BRM.NativePPL.LinearPredictor())
    @test prediction_only_location ≈
          prediction_only_position[1] .+
          prediction_only_position[2] .* prediction_only.predictor
    prediction_a = BRM.NativePPL.simulate(
        MersenneTwister(52), prediction_only_workspace, prediction_only,
        prediction_only_position)
    prediction_b = BRM.NativePPL.simulate(
        MersenneTwister(52), prediction_only_workspace, prediction_only,
        prediction_only_position)
    @test prediction_a == prediction_b
    @test prediction_a isa Vector{Float32}

    err = argument_error(() -> BRM.NativePPL.logdensity!(
        prediction_only_workspace, prediction_only, Float32[]))
    @test occursin("requires an observed response binding", err.msg)
    err = argument_error(() -> BRM.NativePPL.logdensity_and_gradient!(
        prediction_only_workspace, prediction_only, prediction_only_position))
    @test occursin("gradient requires an observed response binding", err.msg)
    err = argument_error(() -> BRM.NativePPL.evaluate!(
        BRM.NativePPL.allocate_output(
            prediction_only, BRM.NativePPL.PointwiseLogLikelihood()),
        prediction_only_workspace, prediction_only, prediction_only_position,
        BRM.NativePPL.PointwiseLogLikelihood()))
    @test occursin("pointwise log likelihood requires an observed response", err.msg)
    err = argument_error(() -> BRM.NativePPL.workspace(
        prediction_only, Float32, DI.AutoEnzyme()))
    @test occursin(
        "DifferentiationInterface gradient preparation requires an observed response",
        err.msg)

    restored = BRM.NativePPL.rebind(
        prediction_only, (; x=Float32[3, 4], y=Float32[2, 3]))
    @test BRM.NativePPL.has_response(restored)
    @test BRM.NativePPL.logdensity!(
        BRM.NativePPL.workspace(restored), restored, prediction_only_position) isa
          Float32

    sliced = BRM.NativePPL.prepare(plan)
    pop!(sliced.response)
    sliced_workspace = BRM.NativePPL.workspace(sliced)
    pop!(sliced_workspace.primal.pointwise_loglikelihood)
    sliced_location = BRM.NativePPL.allocate_output(sliced, linear_query)
    location_only_position = [position[1], position[2], NaN]
    @test BRM.NativePPL.evaluate!(
        sliced_location, sliced_workspace, sliced, location_only_position,
        linear_query) === sliced_location
    @test sliced_location ≈ location
    sliced_prediction = BRM.NativePPL.simulate(
        MersenneTwister(41), sliced_workspace, sliced, position)
    @test sliced_prediction ≈ expected_predictive
    @test_throws DimensionMismatch BRM.NativePPL.evaluate!(
        similar(sliced_location), sliced_workspace, sliced, position,
        BRM.NativePPL.PointwiseLogLikelihood())

    gradient_independent_workspace = BRM.NativePPL.workspace(prepared)
    empty!(gradient_independent_workspace.gradient)
    @test BRM.NativePPL.evaluate!(
        similar(location), gradient_independent_workspace, prepared,
        location_only_position, linear_query) ≈ location
    @test_throws DimensionMismatch BRM.NativePPL.logdensity!(
        gradient_independent_workspace, prepared, position)

    @test_throws ArgumentError BRM.NativePPL.rebind(prepared, (; y=data.y))
    @test_throws ArgumentError BRM.NativePPL.rebind(
        prepared, (; x=1.0, y=data.y))
    @test_throws DimensionMismatch BRM.NativePPL.rebind(
        prepared, (; x=[1.0, 2.0], y=data.y))
    @test_throws DimensionMismatch BRM.NativePPL.rebind(
        prepared, (; x=Float64[], y=Float64[]))
    @test_throws ArgumentError BRM.NativePPL.rebind(
        prepared, (; x=[1, 2, 3], y=data.y))

    @test_throws DimensionMismatch BRM.NativePPL.evaluate!(
        zeros(2), workspace, prepared, position,
        BRM.NativePPL.LinearPredictor())
    @test_throws ArgumentError BRM.NativePPL.evaluate!(
        zeros(Float32, 3), workspace, prepared, position,
        BRM.NativePPL.LinearPredictor())
    @test_throws ArgumentError BRM.NativePPL.evaluate!(
        @view(zeros(3)[:]), workspace, prepared, position,
        BRM.NativePPL.LinearPredictor())
end


@testset "prepared native execution fails closed" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.5, 1.0, 2.5])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    plan = BRM._native_ppl_plan(brmi)
    prepared = BRM._native_ppl_prepare(plan)
    workspace = BRM.NativePPL.workspace(prepared, Float64, DI.AutoEnzyme())
    position = zeros(3)

    @test_throws DimensionMismatch BRM._native_ppl_logdensity!(
        workspace, prepared, zeros(2))
    @test_throws ArgumentError BRM._native_ppl_logdensity!(
        workspace, prepared, zeros(Float32, 3))
    @test_throws ArgumentError BRM._native_ppl_logdensity_and_gradient!(
        workspace, prepared, @view(position[:]))
    @test_throws ArgumentError BRM._native_ppl_logdensity_and_gradient!(
        workspace, prepared, zeros(Float32, 3))
    @test_throws ArgumentError BRM._native_ppl_logdensity!(
        workspace, prepared, workspace.gradient)
    @test_throws ArgumentError BRM._native_ppl_logdensity!(
        workspace, prepared, workspace.primal.location)
    @test_throws ArgumentError BRM._native_ppl_prepare(plan; T=AbstractFloat)
    @test_throws ArgumentError BRM.NativePPLWorkspace(prepared, AbstractFloat)

    bad_gradient = BRM.NativePPLWorkspace(prepared)
    pop!(bad_gradient.gradient)
    @test_throws DimensionMismatch BRM._native_ppl_logdensity!(
        bad_gradient, prepared, position)

    bad_pointwise = BRM.NativePPLWorkspace(prepared)
    pop!(bad_pointwise.primal.pointwise_loglikelihood)
    @test_throws DimensionMismatch BRM._native_ppl_logdensity!(
        bad_pointwise, prepared, position)

    bad_location = BRM.NativePPL.workspace(prepared, Float64, DI.AutoEnzyme())
    pop!(bad_location.primal.location)
    @test_throws DimensionMismatch BRM._native_ppl_logdensity_and_gradient!(
        bad_location, prepared, position)

    bad_prepared = BRM._native_ppl_prepare(plan)
    pop!(bad_prepared.predictor)
    @test_throws DimensionMismatch BRM._native_ppl_logdensity!(
        BRM.NativePPLWorkspace(prepared), bad_prepared, position)

    nonfinite = @brm (x=[-1.0, NaN, 2.0], y=data.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test_throws ArgumentError BRM._native_ppl_prepare(BRM._native_ppl_plan(nonfinite))

    overflow = @brm (x=BigFloat[big"1e1000", 0, 2], y=BigFloat.(data.y)) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test_throws ArgumentError BRM._native_ppl_prepare(BRM._native_ppl_plan(overflow))
end

@testset "native Gaussian lowering fails closed" begin
    data = (; x=[-1.0, 0.0, 2.0], y=[0.5, 1.0, 2.5])

    constant_scale = @brm data begin
        mu ~ 1 + x
        y ~ Normal(mu, 1.0)
    end
    err = capability_error(() -> BRM._native_ppl_plan(constant_scale))
    @test err.capability == :likelihood_scale
    @test occursin("named scalar parameter", err.detail)

    integer_predictor = @brm (x=[1, 2, 3], y=data.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(integer_predictor)).capability ==
          :predictor_type

    mismatched_rows = @brm (x=[1.0, 2.0], y=data.y) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(mismatched_rows)).capability ==
          :observation_axis

    empty_rows = @brm (x=Float64[], y=Float64[]) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(empty_rows)).capability ==
          :observation_axis

    extra_operation = @brm data begin
        sigma ~ Exponential(1)
        nuisance ~ Normal(0, 1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(extra_operation)).capability ==
          :additional_operations

    duplicate_term = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(duplicate_term)).capability ==
          :predictor_terms

    duplicate_raw_term = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + center(x)
        y ~ Normal(mu, sigma)
    end
    @test capability_error(
        () -> BRM._native_ppl_plan(duplicate_raw_term)).capability ==
          :predictor_terms

    response_as_predictor = @brm (y=data.y,) begin
        sigma ~ Exponential(1)
        mu ~ 1 + y
        y ~ Normal(mu, sigma)
    end
    err = capability_error(() -> BRM._native_ppl_plan(response_as_predictor))
    @test err.capability == :input_roles
    @test occursin("also the observed response", err.detail)

    bad_scale_family = @brm data begin
        sigma ~ Normal(0, 1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(bad_scale_family)).capability ==
          :scale_prior

    bad_scale_value = @brm data begin
        sigma ~ Exponential(0.0)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(bad_scale_value)).capability ==
          :scale_prior

    wrong_likelihood = @brm data begin
        mu ~ 1 + x
        y ~ Exponential(exp(mu))
    end
    @test capability_error(() -> BRM._native_ppl_plan(wrong_likelihood)).capability ==
          :likelihood

    multiple_outcomes = @brm merge(data, (; z=[1.0, 2.0, 3.0])) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        z ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(multiple_outcomes)).capability ==
          :outcomes
end
