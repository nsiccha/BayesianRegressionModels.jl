using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal, Poisson
using LogDensityProblems

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
    @test plan.parameters.coefficients.axis === plan.axes.coefficient
    @test plan.parameters.coefficients.unconstrained == 1:2
    @test BRM.native_parameter_name(plan.parameters.scale) == :residual
    @test plan.parameters.scale.support isa BRM.NativePPLPositiveSupport
    @test plan.parameters.scale.axis === plan.axes.scale
    @test plan.parameters.scale.unconstrained == 3:3

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

    @test plan.bindings.dose === data.dose
    @test plan.bindings.response === data.response
    @test sprint(show, plan) ==
        "NativePPLPlan(3 unconstrained parameters, 3 observations)\n" *
        "  inputs: dose, response\n" *
        "  parameters: beta_location, residual\n" *
        "  nodes: location\n" *
        "  factors: NativePPLStandardNormalFactor, " *
        "NativePPLExponentialFactor, NativePPLNormalFactor"
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
        y ~ Poisson(exp(mu))
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
