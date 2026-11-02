using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
include("filter_smc_fixture.jl")
const F = FilterSMCFixture
const W = SMCWindowFixture
mode, path = ARGS
if mode == "write"
    fit, _ = F.calibration(; particles=64)
    initial = brm_filter(fit; learn=true, basis=:fit, groups=(; i=1:1), rng=Xoshiro(193),
        members=16, particles=32, window_length=3, ess_fraction=0.95,
        options=F.options(), history_limit=3)
    partial = brm_filter_stream(initial, F.records([0.7, 1.2]))
    expected = brm_filter_stream(partial, F.records([0.4, 1.8, 1.5, 1.1]))
    serialize(path, (; checkpoint=brm_filter_checkpoint(partial),
        expected=brm_filter_checkpoint(expected)))
elseif mode == "read"
    # This process constructs a one-slice declaration, never the calibration
    # rows, their historical latent layout, an old fit or an old target.
    saved = deserialize(path)
    restored = brm_filter_resume(F.stub(), saved.checkpoint)
    actual = brm_filter_stream(restored, F.records([0.4, 1.8, 1.5, 1.1]))
    @test W.equal(brm_filter_checkpoint(actual), saved.expected)
else
    error("expected write or read")
end
