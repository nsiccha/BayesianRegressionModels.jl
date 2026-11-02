using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
include("filter_smc_fixture.jl")
include("streaming_oracles.jl")
const F = FilterSMCFixture
const W = SMCWindowFixture
const O = StreamingOracles

scalar_root(reference) = reference.scale isa AbstractVector ? only(reference.scale) :
    only(WarmupHMC.scale_from_data(reference.scale)*[1.0])

@testset "BRM learning filter static posterior agrees with Kalman recursion" begin
    fit, _ = F.calibration()
    # In one dimension the supplied basis has no orientation error. Its
    # deliberately inflated magnitude must be recalibrated from the cloud.
    calibration = brm_smc_carry(fit; parameters=:filter, basis=[7.0])
    diagonal = brm_smc_carry(fit; parameters=:filter)
    reference = calibration.posterior.reference
    root = scalar_root(reference)
    diagonal_root = scalar_root(diagonal.posterior.reference)
    @test root^2 ≈ diagonal_root^2
    @test reference.location ≈ diagonal.posterior.reference.location
    fitted = brm_smc_carry(fit; parameters=:filter, basis=:fit,
        max_reflections=2, maxiter=100)
    @test scalar_root(fitted.posterior.reference)^2 ≈ diagonal_root^2
    @test fitted.posterior.reference.location ≈ reference.location
    initial_oracle = O.kalman_prior(; level_mean=only(reference.location),
        level_variance=root^2)
    ys = Union{Missing,Float64}[1.4, 2.3, missing, 1.9, 2.0, 1.7, 2.4, 1.5, 2.1]
    exact = foldl(O.kalman_step, ys; init=initial_oracle)
    rng, init = Xoshiro(401), zeros(1)
    rng_before = copy(rng)
    saved = W.roundtrip(calibration)
    schedule = brm_filter_tune(calibration, F.stub(); rng, init,
        groups=(; i=1:1), temperatures=(0.0, 0.5, 1.0), pilot_steps=24,
        options=SMCOptions(; moves=2, stepsize=0.3, max_tree_depth=4))
    @test rng == rng_before
    @test init == zeros(1)
    @test W.equal(calibration, saved)
    # Stay within a single window: this oracle checks particle parameter
    # learning, without conflating it with the documented boundary approximation.
    for seed in (402, 403, 404)
        current = brm_filter(calibration, F.stub(); rng=Xoshiro(seed),
            groups=(; i=1:1), members=512, particles=128, window_length=32,
            ess_fraction=0.99, options=SMCOptions(; tuning=schedule), history_limit=3)
        replenishments = proposals = likelihood_evaluations = 0
        for y in ys
            observation = (; y=[y])
            before = deepcopy(observation)
            current = brm_filter_step(current, observation)
            @test isequal(observation, before)
            @test length(current.state.records) <= current.state.window_length
            @test length(current.state.history) <= 3
            diagnostic = last(current.state.history)
            replenishments += diagnostic.replenished
            proposals += diagnostic.proposals
            likelihood_evaluations += diagnostic.likelihood_evaluations
        end
        state = current.state
        mean = only(state.positions*state.parameter_mass)
        variance = only((state.positions .- mean).^2*state.parameter_mass)
        mean_error = (mean-exact.mean[1])/sqrt(exact.covariance[1])
        variance_ratio = variance/exact.covariance[1]
        evidence_error = state.log_normalizer-exact.log_evidence
        println("BRM_FILTER_KALMAN seed=", seed, " mean=", mean,
            " variance=", variance, " exact_mean=", exact.mean[1],
            " exact_variance=", exact.covariance[1],
            " standardized_mean_error=", mean_error, " variance_ratio=", variance_ratio,
            " evidence_error=", evidence_error, " replenishments=", replenishments,
            " proposals=", proposals, " likelihood_evaluations=", likelihood_evaluations)
        @test abs(mean_error) < 0.3
        @test abs(variance_ratio-1) < 0.3
        @test abs(evidence_error) < 0.6
        @test replenishments > 0
        @test proposals > 0
        @test likelihood_evaluations > 0
        @test state.time == length(ys)
        @test sum(state.parameter_mass) ≈ 1
        @test W.equal(calibration, saved)
    end
end
