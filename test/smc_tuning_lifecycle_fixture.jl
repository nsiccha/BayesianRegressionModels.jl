module SMCTuningLifecycleFixture

using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, Treebars, LogDensityProblems
include("smc_tuning_fixture.jl")
const equal = SMCTuningFixture.checkpoint_equal

const DATA = (; x=[-1.0, -0.2, 0.5, 1.4], y=[-0.4, 0.1, 0.2, 0.8])
const CONTROLS = (; pilot=:adaptive, pilot_steps=8, max_reflections=0,
    recording_target=16, pilot_evaluations=32, temperatures=(0.0, 1.0),
    options=SMCOptions(; moves=1, max_tree_depth=3, history_limit=2))

model(data=DATA) = @brm data begin
    mu ~ 0 + zscale(x)
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

function roundtrip(value)
    io = IOBuffer()
    serialize(io, value)
    seekstart(io)
    deserialize(io)
end

function targets(kind)
    rk = RKBRMI(model())
    if kind === :initial
        return (rk,), (; init=zeros(1), reference=:pilot)
    elseif kind === :cumulative
        fit = brm_smc_fit(rk; rng=Xoshiro(501), particles=32,
            options=SMCOptions(; moves=1, max_tree_depth=3))
        data = (; x=vcat(DATA.x, [2.0]), y=vcat(DATA.y, [0.6]))
        return (fit, data), (; init=zeros(1))
    elseif kind === :window
        carry = brm_smc_carry(rk; reference=SMCReference([0.2], [0.8]))
        data = (; x=[0.8, 1.7], y=[0.3, 0.9])
        return (carry, model(data)), (; init=zeros(1), reference=:pilot,
            coordinates=:auto)
    elseif kind === :laplace
        data = (; g=["a", "a", "b", "b"], y=[0.2, 0.4, -0.1, 0.3])
        brmi = @brm data begin
            mu ~ 1 + (1 | g)
            effect(mu, Intercept) ~ Normal(0, 1)
            sd(mu, g) = 0.7
            y ~ Normal(mu, 1)
        end
        problem = brm_laplace_problem(RKBRMI(brmi);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), integrate=(:g,))
        return (problem,), (; init=zeros(1), reference=:pilot)
    end
    error("unknown lifecycle fixture: $kind")
end

tune(args, settings; rng=Xoshiro(502), kwargs...) =
    brm_smc_tune(args...; rng, CONTROLS..., settings..., kwargs...)

end
