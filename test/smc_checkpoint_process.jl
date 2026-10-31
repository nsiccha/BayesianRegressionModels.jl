# Invoked by smc.jl in two separate Julia processes.
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization

build(d) = @brm d begin
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, 1)
end
training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
cumulative = (; x=[-2.0, 0.0, 2.0, 4.0], y=[-0.3, 0.4, 0.9, 1.1])
rk = RKBRMI(build(training))
mode, path = ARGS
if mode == "write"
    fit = brm_smc_fit(rk; rng=Xoshiro(52), particles=64,
        options=SMCOptions(; moves=1, stepsize=0.3, max_tree_depth=3))
    partial = brm_smc_update(fit, cumulative; stages=0)
    completed = brm_smc_fit(partial)
    serialize(path, (; checkpoint=brm_smc_checkpoint(partial),
        positions=completed.state.positions, log_normalizer=completed.state.log_normalizer))
elseif mode == "read"
    saved = deserialize(path)
    restored = brm_smc_resume(rk, saved.checkpoint; data=cumulative)
    completed = brm_smc_fit(restored)
    @test completed.state.positions == saved.positions
    @test completed.state.log_normalizer == saved.log_normalizer
else
    error("expected write or read")
end
