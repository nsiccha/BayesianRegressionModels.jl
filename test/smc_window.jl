using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, Statistics
include("smc_window_fixture.jl")
const F = SMCWindowFixture
const options = SMCOptions(; moves=3, stepsize=0.3, max_tree_depth=4, history_limit=8)

@testset "Carried BRM posterior uses only the new window" begin
    training = (; y=[0.9, 1.6, 0.2, 1.3, 0.6, 1.1])
    all_y = copy(training.y)
    initial = brm_smc_fit(F.normal(training); rng=Xoshiro(51), particles=512, options)
    carried = F.roundtrip(brm_smc_carry(initial))
    initial_mean = only(F.weighted_mean(initial.state))
    initial_variance = only(F.weighted_variance(initial.state))
    fill!(training.y, NaN)
    sizes = Int[]
    for step in 1:4
        window = (; y=[0.5+0.1step, 1.2-0.05step, 0.8])
        append!(all_y, window.y)
        saved = F.roundtrip(carried)
        rng = Xoshiro(71+step)
        untouched = copy(rng)
        partial = brm_smc_update(carried, F.normal(window);
            rng, particles=512, options, stages=0)
        @test rng == untouched
        @test F.equal(carried, saved)
        @test partial.state.temperature == 0
        @test_throws r"complete the posterior bridge" brm_smc_carry(partial)
        completed = brm_smc_fit(partial)
        @test partial.state.temperature == 0
        @test completed.state.temperature == 1
        expected_variance = inv(inv(initial_variance)+length(window.y))
        expected_mean = expected_variance*(initial_mean/initial_variance+sum(window.y))
        @test only(F.weighted_mean(completed.state)) ≈ expected_mean atol=0.07
        @test only(F.weighted_variance(completed.state)) ≈ expected_variance atol=0.04
        checkpoint = F.roundtrip(brm_smc_checkpoint(partial))
        restored = brm_smc_fit(brm_smc_resume(F.normal(window), checkpoint))
        @test F.equal(brm_smc_checkpoint(restored), brm_smc_checkpoint(completed))
        @test_throws r"window data has changed" brm_smc_resume(
            F.normal((; y=window.y .+ 1)), checkpoint)
        @test_throws r"another model" brm_smc_update(carried, F.changed(window); stages=0)
        changed = deepcopy(carried)
        changed.preprocessing.constants[:invalid_anchor] = 0.0
        @test_throws r"frozen preprocessing has changed" brm_smc_update(
            changed, F.normal(window); stages=0)
        carried = F.roundtrip(brm_smc_carry(completed))
        push!(sizes, F.serialized_size(carried))
        initial_mean = only(F.weighted_mean(completed.state))
        initial_variance = only(F.weighted_variance(completed.state))
    end
    @test maximum(sizes)-minimum(sizes) <= 32
    @test initial_mean ≈ sum(all_y)/(1+length(all_y)) atol=0.08
    @test initial_variance ≈ inv(1+length(all_y)) atol=0.025
    batch = brm_smc_fit(F.normal((; y=all_y)); rng=Xoshiro(123), particles=512, options)
    @test initial_mean ≈ only(F.weighted_mean(batch.state)) atol=0.1
    @test initial_variance ≈ only(F.weighted_variance(batch.state)) atol=0.03
end

@testset "Window checkpoints replay in a fresh process without training rows" begin
    mktempdir() do dir
        checkpoint = joinpath(dir, "window.bin")
        script = joinpath(@__DIR__, "smc_window_process.jl")
        project = dirname(Base.active_project())
        command = `$(Base.julia_cmd()) --startup-file=no --project=$project $script`
        @test success(`$command write $checkpoint`)
        @test success(`$command read $checkpoint`)
    end
end
