# Run: julia --project=test test/dotted_value_calls.jl
using Test, Distributions, BayesianRegressionModels, StanBlocks
using DifferentiationInterface: AutoEnzyme
import BridgeStan, Enzyme, LogDensityProblems, ReactiveKernelsPPL
include(joinpath(@__DIR__, "testset_filter.jl"))

module DottedValueFunctions
using StanBlocks
@deffun @juliacompat shift(x::real, delta::real)::real = x + delta
# A whole-vector call has a different, valid Julia meaning.
shift(x::AbstractVector, delta::Real) = fill(delta, length(x))
# A Julia-only function deliberately shares a builtin's name.
exp(x::Real) = x + 3
end

const exp_alias = Base.exp
const dotted_builder = @brm begin
    b[1:K] .~ Normal.(0, 1)
    eta = exp_alias.(b)
    y ~ Normal(sin.(eta), 1)
    z .~ Normal.(DottedValueFunctions.shift.(Base.exp.(b), delta), 1)
end

@stestset "dotted data assignments apply the scalar method on replay" begin
    data = (; x=[-0.8, 0.3], delta=0.7, y=[0.3, 1.2])
    changed = (; x=[-0.2, 0.6, 0.9], delta=-0.3, y=[0.4, -0.1, 1.3])
    model = @brm data begin
        a ~ Normal(0, 1)
        shifted = DottedValueFunctions.shift.(x, delta)
        y ~ Normal(a + shifted, 1)
    end
    sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
    for (s, r, rows) in ((sb, rk, data), (reprocess(sb, changed), reprocess(rk, changed), changed))
        predicted = rows.x .+ rows.delta
        @test s.data[:shifted] ≈ predicted
        problem = rk_logdensity_problem(r; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, grad = LogDensityProblems.logdensity_and_gradient(problem, [0.2])
        residual = rows.y .- predicted .- 0.2
        @test value ≈ logpdf(Normal(), 0.2) + sum(logpdf.(Normal(), residual))
        @test only(grad) ≈ sum(residual) - 0.2 atol=1e-10
    end
end

@stestset "a local alias keeps the caller's scalar method" begin
    caller_exp = DottedValueFunctions.exp
    builder = @brm begin
        b[1:K] .~ Normal.(0, 1)
        eta = caller_exp.(b)
        y ~ Normal(eta, 1)
    end
    b = [-0.4, 0.2]
    data = (; K=2, y=[0.1, 0.5], caller_exp=23)
    model = builder(data)
    @test occursin("caller_exp.(b)", string(model))
    rk = RKBRMI(model)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, b)
    residuals = data.y - caller_exp.(b)
    @test value ≈ sum(logpdf.(Normal(), b)) + sum(logpdf.(Normal(), residuals))
    @test gradient ≈ residuals - b
    err = try
        stan_code(SBBRMI(model; mod=@__MODULE__))
        nothing
    catch e
        e
    end
    @test !isnothing(err)
    if !isnothing(err)
        message = sprint(showerror, err)
        @test occursin("DottedValueFunctions.exp", message)
        @test occursin("Stan", message)
    end
end

@stestset "dotted model calls preserve their binding, source and target" begin
    data = (; K=2, y=[0.7, -0.4], z=[-0.5, 0.2], delta=0.3, exp_alias=99)
    changed = (; K=2, y=[0.1, 1.2], z=[0.8, -0.2], delta=-0.25, exp_alias=0)
    function check_target(sb, rk, rows, path)
        b = collect(range(-0.3, 0.25; length=rows.K))
        v = exp.(b)
        ry, rz = rows.y .- sin.(v), rows.z .- (v .+ rows.delta)
        value = sum(logpdf.(Normal(), b)) + sum(logpdf.(Normal(), ry)) +
            sum(logpdf.(Normal(), rz))
        gradient = -b + v .* (cos.(v) .* ry + rz)
        stan = stan_instantiate(sb; path).model
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        names = ["b.$j" for j in eachindex(b)]
        @test BridgeStan.param_unc_names(stan) == names
        @test ReactiveKernelsPPL.coordinate_names(rk.model.layout) == Symbol.(names)
        for (lp, grad) in (BridgeStan.log_density_gradient(stan, b; propto=false),
                LogDensityProblems.logdensity_and_gradient(problem, b))
            @test lp ≈ value atol=1e-10
            @test grad ≈ gradient atol=1e-10
        end
    end
    mktempdir() do dir
        source = string(dotted_builder(data))
        @test Set(brm_columns(dotted_builder)) == Set((:K, :y, :z, :delta))
        @test occursin("eta = exp_alias.(b)", source)
        @test occursin("y ~ Normal(sin.(eta), 1)", source)
        @test occursin("z .~ Normal.(DottedValueFunctions.shift.(Base.exp.(b), delta), 1)", source)
        for builder in (dotted_builder, brm(dotted_builder.source; mod=@__MODULE__),
                brm(source; mod=@__MODULE__))
            model = builder(data)
            @test string(model) == source
            sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
            path = joinpath(dir, "dotted.stan")
            check_target(sb, rk, data, path)
            # A frozen Stan replay keeps scalar data at the fitted value.
            observations = merge(data, (; changed.y, changed.z))
            check_target(reprocess(sb, observations), reprocess(rk, observations), observations, path)
            check_target(reprocess(sb, changed; freeze_constants=false),
                reprocess(rk, changed; freeze_constants=false), changed, path)
        end
    end
end
