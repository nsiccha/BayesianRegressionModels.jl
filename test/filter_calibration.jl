using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using SequentialMonteCarlo, Distributions, Random, Statistics
using Enzyme, ADTypes, DifferentiationInterface, WarmupHMC
include("filter_models.jl")
const FM = FilterModels

drifting_intercept(d) = @brm d begin
    @plate for i in eachindex(y)
        @scan begin
            coefficient[1] ~ Normal(0, 1)
            for t in 2:T
                coefficient[t] ~ Normal(coefficient[t-1], 0.35)
            end
        end
        y[i] ~ Normal(coefficient, 0.2)
    end
end

@testset "BRM explicit parameter drift follows its authored transition" begin
    rk = RKBRMI(drifting_intercept((; y=[zeros(3)], T=3)))
    filter = brm_filter(rk; groups=(; i=1:1),
        drifting=(Symbol("i.coefficient"),), particles=4000, rng=Xoshiro(81))
    mean, variance = 0.0, 1.0
    ys = [0.1, 0.4, 0.8, 1.2, 1.7, 2.0]
    for (index, y) in enumerate(ys)
        index > 1 && (variance += 0.35^2)
        score = logpdf(Normal(mean, sqrt(variance+0.2^2)), y)
        gain = variance/(variance+0.2^2)
        mean += gain*(y-mean)
        variance *= 1-gain
        filter = brm_filter_step(filter, (; y=[y]))
        samples = [only(values(carry))[end, 1] for carry in filter.state.carries]
        @test filter.state.predictive_logscore ≈ score atol=0.12
        @test sum(samples .* filter.state.weights) ≈ mean atol=0.06
    end
    @test mean > 1.8
    @test_throws ArgumentError brm_filter(rk; groups=(; i=1:1), drifting=(:missing_parameter,))
end

@testset "BRM concentrated calibrated parameters preserve predictive uncertainty" begin
    training = (; x=[0.5, 0.7, 0.3], y=[0.4, 0.8, 0.9])
    rk = RKBRMI(FM.provider(training))
    mean, sd = 1.0, 0.02
    members = [(; gain=mean+delta*sd) for delta in (-sqrt(3), 0, sqrt(3))]
    filter = brm_filter(rk; ensemble=members, weights=[1/6, 2/3, 1/6],
        inputs=(:x,), particles=8, rng=Xoshiro(91))
    history = [(; observation=(; y=training.y[i]), inputs=(; x=training.x[i]))
        for i in eachindex(training.x)]
    warm = brm_filter_stream(filter, history; condition_state_only=true)
    next_x, next_y = 1.2, 1.8
    coefficient = last(FM.accumulated_input([training.x; next_x], 1.0))
    # Three-node Gaussian quadrature preserves the narrow parameter posterior's
    # moments through degree five; this checks its marginalized predictive law.
    expected = logpdf(Normal(coefficient*mean, sqrt(1+(coefficient*sd)^2)), next_y)
    next = brm_filter_step(warm, (; y=next_y); inputs=(; x=next_x))
    @test next.state.predictive_logscore ≈ expected atol=1e-7
    @test warm.state.log_normalizer == 0
    @test warm.state.parameter_mass ≈ [1/6, 2/3, 1/6]
end

@testset "BRM posterior SMC particles initialize a calibrated filter" begin
    data = (; x=[0.5, 0.7, 0.3], y=[0.4, 0.8, 0.9])
    rk = RKBRMI(FM.provider(data))
    initial = brm_smc_fit(rk; rng=Xoshiro(29), particles=32, stages=0,
        options=SMCOptions(; moves=1, max_tree_depth=3, stepsize=0.35))
    @test_throws ArgumentError brm_filter(initial; inputs=(:x,), particles=4)
    fitted = brm_smc_fit(initial)
    filter = brm_filter(fitted; inputs=(:x,), particles=4, rng=Xoshiro(39))
    @test filter.state.parameter_mass ≈ fitted.state.weights
    @test filter.configuration.parameters.gain == fitted.state.positions[1, 1]
    @test length(filter.state.carries) == 4length(fitted.state.weights)
end
