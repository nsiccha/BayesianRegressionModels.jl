using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Statistics, Serialization
include("smc_window_fixture.jl")
const F = SMCWindowFixture

@testset "BRM carried state stays bounded over elapsed windows" begin
    options = SMCOptions(; moves=1, stepsize=0.1, max_tree_depth=2, history_limit=2)
    training = (; y=[0.2, 0.4, 0.6])
    initial = brm_smc_fit(F.normal(training); rng=Xoshiro(901), particles=64, options)
    carried = brm_smc_carry(initial; basis=:fit)
    window = (; y=[0.3, 0.5, 0.4])
    sizes, checkpoint_sizes = Int[], Int[]
    timings = Float64[]
    for step in 1:128
        fit = brm_smc_update(carried, F.normal(window);
            rng=Xoshiro(1000+step), particles=64, options)
        carried = brm_smc_carry(fit; basis=:fit)
        step in (8, 32, 128) || continue
        @test fit.state.temperature == 1
        @test length(rk_bound_data(fit.backend).y) == length(window.y)
        push!(sizes, F.serialized_size(carried))
        push!(checkpoint_sizes, F.serialized_size(brm_smc_checkpoint(fit)))
        # Time identical work at each history length: target preparation and
        # initialization of a fixed cloud for this same three-row window.
        # Warm first; report rather than assert wall time on a shared host.
        initialize() = brm_smc_update(carried, F.normal(window);
            rng=Xoshiro(999), particles=64, options, stages=0)
        initialize()
        samples = map(1:7) do _
            @elapsed initialize()
        end
        push!(timings, median(samples))
        println("BRM_WINDOW_SCALING elapsed_windows=", step,
            " carried_bytes=", last(sizes), " checkpoint_bytes=", last(checkpoint_sizes),
            " median_seconds=", median(samples), " minimum_seconds=", minimum(samples))
    end
    @test maximum(sizes)-minimum(sizes) <= 64
    @test maximum(checkpoint_sizes)-minimum(checkpoint_sizes) <= 512
    @test all(isfinite, timings)
end
