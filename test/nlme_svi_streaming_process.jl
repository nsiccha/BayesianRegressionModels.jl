using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Serialization, Random
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
input = deserialize(ARGS[1])
fit = brm_svi_fit(input.state, svi_stream_model(input.data); rng=Xoshiro(54),
    max_iterations=25, ndraws=0)
serialize(ARGS[2], brm_svi_checkpoint(fit))
