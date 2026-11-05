using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Serialization
include("nlme_svi_owned_fixture.jl")
input = deserialize(ARGS[1])
fit = brm_svi_resume(svi_owned_model(input.data), input.state; ndraws=4, threads=1)
serialize(ARGS[2], (; fit.draws, fit.diagnostics.q))
