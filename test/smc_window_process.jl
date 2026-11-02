using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
const F = SMCWindowFixture
mode, path = ARGS
window = (; x=[-1.5, 0.5, 1.2, 3.0], y=[-0.1, 0.5, 0.7, 1.0])
if mode == "write"
    # The reading process never constructs or receives these training arrays.
    training = (; x=[-3.0, 1.0, 5.0], y=[-0.2, 0.1, 0.8])
    options = SMCOptions(; moves=2, stepsize=0.3, max_tree_depth=3)
    calibration = brm_smc_fit(F.scaled(training); rng=Xoshiro(52), particles=64, options)
    carried = brm_smc_carry(calibration; basis=:fit)
    partial = brm_smc_update(carried, F.scaled(window);
        rng=Xoshiro(53), particles=64, options, stages=0)
    completed = brm_smc_fit(partial)
    serialize(path, (; partial=brm_smc_checkpoint(partial),
        completed=brm_smc_checkpoint(completed)))
elseif mode == "read"
    saved = deserialize(path)
    completed = brm_smc_fit(brm_smc_resume(F.scaled(window), saved.partial))
    @test F.equal(brm_smc_checkpoint(completed), saved.completed)
else
    error("expected write or read")
end
