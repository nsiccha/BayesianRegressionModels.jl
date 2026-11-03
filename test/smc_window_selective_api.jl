using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
include("smc_window_selective_fixture.jl")
const F = SMCWindowFixture
const SF = SMCSelectiveFixture

@testset "Selective coordinate descriptors and tuning use the compact frame" begin
    training = (; nb=2, block=[1, 2], y=[0.0, 0.0])
    rk = RKBRMI(SF.hierarchical(training))
    carried = brm_smc_carry(rk; reference=SMCReference(zeros(3), ones(3)))
    window = (; nb=3, block=[3], y=[0.8])
    options = SMCOptions(; moves=1, stepsize=0.2, max_tree_depth=2)
    partial = brm_smc_update(carried, SF.hierarchical(window);
        coordinates=:auto, rng=Xoshiro(2401), particles=32, options, stages=0)
    saved = F.roundtrip(brm_smc_checkpoint(partial))
    mapping = brm_smc_coordinates(partial)
    current = coordinate_names(partial.backend.model.layout)
    old = coordinate_names(rk.model.layout)
    @test Set(current[mapping.current_indices]) == Set([:m, Symbol("z.3")])
    @test old[mapping.active_old] == current[mapping.current_indices[mapping.active_slots]]
    @test current[mapping.current_indices[mapping.fresh_slots]] == [Symbol("z.3")]
    @test sort(vcat(mapping.active_slots, mapping.fresh_slots)) == [1, 2]
    point = [0.3, -0.7]
    recovered = brm_smc_recover(partial, point; rng=Xoshiro(2402))
    @test recovered[mapping.joint_indices] == point
    mapping.current_indices[1] = -1
    @test brm_smc_coordinates(partial).current_indices[1] > 0
    @test F.equal(brm_smc_checkpoint(partial), saved)

    rng, init = Xoshiro(2403), zeros(2)
    saved_rng, saved_carry = copy(rng), F.roundtrip(carried)
    schedule = brm_smc_tune(carried, SF.hierarchical(window);
        coordinates=:auto, rng, init, temperatures=(0.0, 1.0),
        pilot_steps=8, options)
    @test rng == saved_rng
    @test init == zeros(2)
    @test F.equal(carried, saved_carry)
    @test length(smc_reference(schedule).location) == 2
    tuned = brm_smc_update(carried, SF.hierarchical(window);
        coordinates=:auto, rng=Xoshiro(2404), particles=64,
        options=SMCOptions(; tuning=schedule), stages=0)
    completed = brm_smc_fit(tuned)
    @test size(completed.state.positions) == (2, 64)
    @test all(stage -> stage.kernel == schedule.kernels[stage.tuning_index],
        completed.state.history)
    restored = brm_smc_fit(brm_smc_resume(SF.hierarchical(window),
        F.roundtrip(brm_smc_checkpoint(tuned))))
    @test F.equal(brm_smc_checkpoint(completed), brm_smc_checkpoint(restored))
    @test_throws ArgumentError brm_smc_update(carried,
        SF.hierarchical(merge(window, (; y=[1.2]))); coordinates=:auto,
        options=SMCOptions(; tuning=schedule), stages=0)
end
