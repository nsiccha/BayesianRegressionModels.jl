using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, Statistics
include("smc_window_fixture.jl")
include("filter_smc_fixture.jl")
const F = FilterSMCFixture
const W = SMCWindowFixture

@testset "BRM learning filter has bounded retained history" begin
    fit, _ = F.calibration(; particles=64)
    current = brm_filter(fit; learn=true, basis=:fit, groups=(; i=1:1), rng=Xoshiro(195),
        members=16, particles=32, window_length=3, ess_fraction=0.95,
        options=F.options(), history_limit=3)
    bytes = Int[]
    observation = (; y=[0.8])
    for elapsed in 1:120
        current = brm_filter_step(current, observation)
        elapsed in (12, 120) || continue
        push!(bytes, W.serialized_size(brm_filter_checkpoint(current)))
        # Identical phase, dimensions and workload: the next first-window step.
        # Repetitions begin from the same immutable state and copied RNG.
        brm_filter_step(current, observation)
        measurements = map(1:7) do _
            @timed brm_filter_step(current, observation)
        end
        println("BRM_FILTER_SCALING elapsed=", elapsed,
            " checkpoint_bytes=", last(bytes),
            " median_seconds=", median(m.time for m in measurements),
            " median_allocated_bytes=", median(m.bytes for m in measurements))
    end
    @test maximum(bytes)-minimum(bytes) <= 512
end
