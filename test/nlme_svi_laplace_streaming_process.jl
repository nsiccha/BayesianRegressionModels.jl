using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, DifferentiationInterface, Random, Serialization
include("nlme_svi_laplace_streaming_fixture.jl")
input=deserialize(ARGS[1])
fit=brm_svi_fit(input.state,laplace_window_builder()(input.data);
    ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(731),
    max_iterations=4,ndraws=12)
serialize(ARGS[2],laplace_window_summary(fit))
