using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
include("filter_smc_fixture.jl")
const F = FilterSMCFixture
const W = SMCWindowFixture

@testset "BRM selected-static filter transfer and replay" begin
    fit, training = F.calibration()
    @test size(fit.state.positions, 1) > 1
    calibration = W.roundtrip(brm_smc_carry(fit; parameters=:filter,
        basis=:fit, max_reflections=2, maxiter=100))
    @test length(calibration.posterior.reference.location) == 1
    @test calibration.posterior.fit_diagnostics !== nothing
    parameters = smc_parameters(calibration.posterior,
        calibration.posterior.reference.location)
    @test keys(parameters) == (:theta,)
    saved = W.roundtrip(calibration)
    rng = Xoshiro(185)
    before = copy(rng)
    initial = brm_filter(calibration, F.stub(); rng, members=16, particles=32,
        groups=(; i=1:1), window_length=3, ess_fraction=0.95,
        options=F.options(), history_limit=3)
    @test initial isa BRMFilterSMC
    @test rng == before
    @test W.equal(calibration, saved)
    @test_throws r"selected filter calibration" brm_smc_update(calibration, F.stub())
    @test_throws r"warm an ordinary filter" brm_filter_step(initial,
        (; y=[1.0]); condition_state_only=true)
    cp = W.roundtrip(brm_filter_checkpoint(initial))
    @test cp.format == 2
    # After transfer, neither the old fit nor its training rows are needed.
    fill!(training.y[1], NaN)
    stream = F.records([0.7, 1.2, 0.4, 1.8, 1.5, 1.1, 0.8])
    first = brm_filter_stream(initial, stream[1:2])
    continued = brm_filter_stream(first, stream[3:end])
    restored = brm_filter_resume(F.stub(), W.roundtrip(brm_filter_checkpoint(first)))
    replayed = brm_filter_stream(restored, stream[3:end])
    @test W.equal(brm_filter_checkpoint(continued), brm_filter_checkpoint(replayed))
    @test W.equal(brm_filter_checkpoint(initial), cp)
    @test W.equal(calibration, saved)
    @test_throws r"authored source" brm_filter_resume(
        F.changed((; y=[zeros(1)], T=1)), cp)
    corrupted = W.roundtrip(cp)
    corrupted.calibration.posterior.reference.location[1] += 1
    @test_throws r"carried posterior" brm_filter_resume(F.stub(), corrupted)
    # Missing observations and active groups use the ordinary filter contract.
    missing_step = brm_filter_step(first, (; y=[missing]); groups=(; i=[1]))
    @test isfinite(missing_step.state.log_normalizer)
    @test W.equal(brm_filter_checkpoint(first), brm_filter_checkpoint(restored))
end

@testset "BRM warm state-only boundary preserves identity" begin
    fit, training = F.calibration(; particles=64)
    ordinary = brm_filter(fit; groups=(; i=1:1), particles=16, rng=Xoshiro(187))
    warm = brm_filter_stream(ordinary, F.records(training.y[1]); condition_state_only=true)
    @test warm.state.log_normalizer == 0
    checkpoint = W.roundtrip(brm_filter_checkpoint(warm))
    learning = brm_filter(fit; learn=true, boundary=warm, groups=(; i=1:1),
        members=12, particles=16, window_length=3, options=F.options(), rng=Xoshiro(188))
    @test learning.model_id == warm.model_id
    @test learning.state.log_normalizer == 0
    @test W.equal(brm_filter_checkpoint(warm), checkpoint)
    @test_throws r"boundary source" brm_filter(fit; learn=true, boundary=warm,
        groups=(; i=1:2), members=12, particles=16, options=F.options())
    next = brm_filter_step(learning, (; y=[1.2]))
    restored = brm_filter_resume(F.stub(), W.roundtrip(brm_filter_checkpoint(learning)))
    @test W.equal(brm_filter_checkpoint(next),
        brm_filter_checkpoint(brm_filter_step(restored, (; y=[1.2]))))
end
