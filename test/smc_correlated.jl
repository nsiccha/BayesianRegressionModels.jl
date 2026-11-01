using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Serialization
include("smc_tuning_fixture.jl")

correlated_model(data) = @brm data begin
    a ~ Normal(0, 1)
    b ~ Normal(a, 0.2)
    y ~ Normal(b, 1)
end

@testset "BRM explicit correlated reference and mutation metric" begin
    data = (; y=[0.4, 1.1, -0.2])
    rk = RKBRMI(correlated_model(data))
    order = Int.(indexin([:a, :b], coordinate_names(rk.model.layout)))
    # Public scale-data recipe: L = H*Diagonal(sqrt.([2, .02])). A positive
    # second reflection component gives covariance [1.01 .99; .99 1.01].
    scale = (; diagonal=[sqrt(2.0), sqrt(0.02)], indices=[[1, 2]],
        reflections=[[cos(pi/8), sin(pi/8)]])
    reference = SMCReference(zeros(2), scale)
    options = SMCOptions(; metric=WarmupHMC.scale_data(WarmupHMC.scale_from_data(scale)),
        stepsize=0.3, moves=3, max_tree_depth=6, history_limit=3)
    # Constructors own their data; editing the caller's input must not change
    # the reference, the mutation metric, or subsequent checkpoint replay.
    scale.diagonal[1] = 100
    scale.indices[1][1] = 2
    scale.reflections[1][2] = -1
    initial = brm_smc_fit(rk; rng=Xoshiro(781), particles=1536,
        reference, options, stages=0)
    initial_summary = SMCTuningFixture.direction_summary(
        initial.state.positions, initial.state.weights, order)
    @test all(abs.(initial_summary.means) .< 0.12 .* sqrt.([2.0, 0.02]))
    @test all(abs.(initial_summary.variances ./ [2.0, 0.02] .- 1) .< 0.15)
    @test initial.state.log_normalizer == 0

    fit = brm_smc_fit(initial)
    @test fit.state.temperature == 1
    n, s = length(data.y), sum(data.y)
    determinant = 25 + 26n
    means = [51s, -s] ./ (sqrt(2)*determinant)
    variances = [101 + n, 1 + n] ./ (2determinant)
    exact_logz = -n*log(2pi)/2 - log1p(1.04n)/2 -
        (sum(abs2, data.y) - 1.04s*s/(1 + 1.04n))/2
    measured = SMCTuningFixture.direction_summary(fit.state.positions, fit.state.weights, order)
    mean_error = (measured.means .- means) ./ sqrt.(variances)
    variance_ratio = measured.variances ./ variances
    logz_error = fit.state.log_normalizer - exact_logz
    println("CORRELATED_REFERENCE standardized_mean_error=", mean_error,
        " variance_ratio=", variance_ratio, " logz_error=", logz_error)
    @test all(abs.(mean_error) .< 0.2)
    @test all(abs.(variance_ratio .- 1) .< 0.25)
    @test abs(logz_error) < 0.25
    @test all(stage -> 0 <= stage.mean_acceptance <= 1, fit.state.history)
    checkpoint = brm_smc_checkpoint(initial)
    io = IOBuffer()
    serialize(io, checkpoint)
    seekstart(io)
    restored = brm_smc_fit(brm_smc_resume(rk, deserialize(io)))
    @test SMCTuningFixture.checkpoint_equal(brm_smc_checkpoint(restored),
        brm_smc_checkpoint(fit))
end
