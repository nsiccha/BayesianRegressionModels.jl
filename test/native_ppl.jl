using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Exponential, Normal, Poisson, logpdf
using Enzyme
using LogDensityProblems
using Random: MersenneTwister, rand, randn

const BRM = BayesianRegressionModels

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

    @test BRM.native_input_name(plan.inputs.predictor) == :dose
    @test BRM.native_input_role(plan.inputs.predictor) == :predictor
    @test eltype(plan.inputs.predictor) == Float64
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
    @test plan.nodes.location.slope_index == 2

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
    @test keys(plan.nodes) == (:location, :rate)
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
