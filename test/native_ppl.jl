using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, Normal, Poisson, logpdf
using Enzyme
using LogDensityProblems
using Random: MersenneTwister, rand, randn

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

function direct_multi_gaussian_model()
    coefficients = NP.parameter(
        NP.RealSupport(), (:intercept, :beta_x, :beta_w);
        transform=NP.Identity(), prior=NP.StandardNormal())
    sigma = NP.parameter(
        NP.PositiveSupport(), (:sigma,);
        transform=NP.Exp(), prior=NP.Exponential(2.0))
    NP.model(
        inputs=(; x=NP.input(), w=NP.input()),
        parameters=(; beta_mu=coefficients, sigma),
        nodes=(; zscale_x_for_mu=NP.zscale(:x),
               center_w_for_mu=NP.center(:w),
               mu=NP.affine(
                   (:zscale_x_for_mu, :center_w_for_mu), :beta_mu)),
        observations=(; y=NP.broadcasted(NP.normal(:y, :mu, :sigma))))
end

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
end


@testset "public native PPL @model semantics" begin
    raw_x = [-1.0, 0.0, 2.0, 4.0]
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
    @test occursin("exactly one observation", err.msg)
    err = argument_error(() -> macroexpand(
        @__MODULE__, :(NP.@model function bad_link(x)
            intercept ~ Normal()
            slope ~ Normal()
            log_rate = intercept + slope * x
            @. y ~ Poisson(log_rate + x)
        end)))
    @test occursin("named rate or `exp(named_log_rate)`", err.msg)
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

    multiple_terms = @brm merge(data, (; z=[2.0, 1.0, 0.0])) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + z
        y ~ Normal(mu, sigma)
    end
    @test capability_error(() -> BRM._native_ppl_plan(multiple_terms)).capability ==
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
