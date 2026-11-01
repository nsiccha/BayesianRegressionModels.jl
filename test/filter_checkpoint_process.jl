# Rebuild the model and frozen summaries independently; never serialize its code.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using SequentialMonteCarlo, Random, Serialization
include("filter_models.jl")
const FM = FilterModels
records = [(; observation=(; y=1.2), inputs=(; x=8.0)),
           (; observation=(; y=1.3), inputs=(; x=9.0))]
mode, path = ARGS
if mode == "write"
    training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
    rk = RKBRMI(FM.scaled(training))
    initial = brm_filter(rk; parameters=fill(0.2, rk.model.layout.total),
        inputs=(:x,), particles=16, rng=Xoshiro(28))
    partial = brm_filter_stream(initial, records; steps=1)
    completed = brm_filter_stream(partial, records[2:end])
    serialize(path, (; checkpoint=brm_filter_checkpoint(partial),
        fitted=brm_fitted_state(rk), expected=brm_filter_checkpoint(completed),
        expected_rng=rand(copy(completed.state.rng), UInt64, 8)))
elseif mode == "read"
    saved = deserialize(path)
    rebuilt = RKBRMI(FM.scaled((; x=[8.0], y=[1.2])), saved.fitted)
    resumed = brm_filter_resume(rebuilt, saved.checkpoint)
    completed = brm_filter_stream(resumed, records[2:end])
    @test isequal(brm_filter_checkpoint(completed), saved.expected)
    @test rand(copy(completed.state.rng), UInt64, 8) == saved.expected_rng
else
    error("expected write or read")
end
