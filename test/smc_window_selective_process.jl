using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
include("smc_window_selective_fixture.jl")
const F = SMCWindowFixture
const SF = SMCSelectiveFixture

function transformed_x(rk)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(:zscale_x,), bound)
    value = Base.invokelatest(query, zeros(rk.model.layout.total))
    value isa Tuple ? only(value) : value
end

mode, path = ARGS
window = (; nb=3, block=[3, 3], x=[2.0, 8.0], y=[0.4, 0.9])
if mode == "write"
    training = (; nb=2, block=[1, 2, 1], x=[-3.0, 1.0, 5.0], y=[0.1, 0.2, 0.3])
    rk = RKBRMI(SF.scaled(training))
    d = length(coordinate_names(rk.model.layout))
    carried = brm_smc_carry(rk; reference=SMCReference(zeros(d), ones(d)))
    options = SMCOptions(; moves=2, stepsize=0.25, max_tree_depth=3)
    partial = brm_smc_update(carried, SF.scaled(window);
        coordinates=:auto, rng=Xoshiro(2301), particles=128, options, stages=0)
    @test size(partial.state.positions, 1) == 2
    @test transformed_x(partial.backend) ≈ (window.x .- 1.0) ./ 4.0
    @test brm_transform_constants(partial.backend) == brm_transform_constants(rk)
    @test brm_transform_constants(partial.backend) != brm_transform_constants(RKBRMI(SF.scaled(window)))
    completed = brm_smc_fit(partial)
    compact = copy(completed.state.positions[:, 1])
    serialize(path, (; partial=brm_smc_checkpoint(partial),
        completed=brm_smc_checkpoint(completed), compact,
        coordinates=brm_smc_coordinates(completed),
        recovered=brm_smc_recover(completed, compact; rng=Xoshiro(2302)),
        carried=brm_smc_carry(completed)))
elseif mode == "read"
    saved = deserialize(path)
    # This process never receives the old training arrays or backend.
    completed = brm_smc_fit(brm_smc_resume(SF.scaled(window), saved.partial))
    @test transformed_x(completed.backend) ≈ (window.x .- 1.0) ./ 4.0
    @test brm_transform_constants(completed.backend, :zscale_x).center == 1.0
    @test brm_transform_constants(completed.backend, :zscale_x).scale == 4.0
    @test F.equal(brm_smc_checkpoint(completed), saved.completed)
    @test F.equal(brm_smc_coordinates(completed), saved.coordinates)
    @test brm_smc_recover(completed, saved.compact; rng=Xoshiro(2302)) == saved.recovered
    @test F.equal(brm_smc_carry(completed), saved.carried)
else
    error("expected write or read")
end
