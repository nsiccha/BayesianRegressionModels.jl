using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, LogDensityProblems
include("smc_tuning_fixture.jl")

data = SMCTuningFixture.data()
first_data = SMCTuningFixture.prefix(data, 16)
rk = RKBRMI(SMCTuningFixture.model(first_data))
mode, path = ARGS
if mode == "write"
    temperatures = (0.0, 0.01, 0.1, 1.0)
    pilot_options = SMCOptions(; moves=2, stepsize=0.1, max_tree_depth=5)
    schedule = brm_smc_tune(rk; rng=Xoshiro(741), init=zeros(2),
        temperatures, pilot_steps=24, options=pilot_options)
    initial = brm_smc_fit(rk; rng=Xoshiro(742), particles=64,
        options=SMCOptions(; tuning=schedule), stages=1)
    fitted = brm_smc_fit(initial)
    update_schedule = brm_smc_tune(fitted, data; rng=Xoshiro(743), init=zeros(2),
        temperatures, pilot_steps=24, options=pilot_options)
    partial = brm_smc_update(fitted, data;
        options=SMCOptions(; tuning=update_schedule), stages=1)
    updated = brm_smc_fit(partial)
    serialize(path, (; initial=brm_smc_checkpoint(initial), fitted=brm_smc_checkpoint(fitted),
        partial=brm_smc_checkpoint(partial), updated=brm_smc_checkpoint(updated)))
elseif mode == "read"
    saved = deserialize(path)
    fitted = brm_smc_fit(brm_smc_resume(rk, saved.initial))
    updated = brm_smc_fit(brm_smc_resume(rk, saved.partial; data))
    @test SMCTuningFixture.checkpoint_equal(brm_smc_checkpoint(fitted), saved.fitted)
    @test SMCTuningFixture.checkpoint_equal(brm_smc_checkpoint(updated), saved.updated)
else
    error("expected write or read")
end
