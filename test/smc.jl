# Synthetic consumer integration: no Stan compilation or application data.
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Statistics, Serialization, LogDensityProblems, Treebars
const BRM = BayesianRegressionModels
const PPL = ReactiveKernelsPPL

normal_model(d) = @brm d begin
    mu ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

scaled_model(d) = @brm d begin
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, 1)
end
grouped_model(d) = @brm d begin
    mu ~ 1 + (1 | subject)
    y ~ Normal(mu, 1)
end

# A synthetic deterministic recurrence, exercising history dependence without
# importing or reproducing any private ODL model.
function accumulated_input(x, gain)
    carry = 0.0
    map(x) do value
        carry = 0.7carry + gain * value
    end
end
function BRM._rk_callable_source!(definitions, bindings, entry, ::typeof(accumulated_input))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = ReactiveKernels.scan(x; init=0.0) do carry, value
            next = 0.7carry + gain * value
            (next, next)
        end
        return values
    end))
    :done
end
scan_model(d) = @brm d begin
    gain ~ Normal(0, 1)
    loc = accumulated_input(x, gain)
    y ~ Normal(loc, 1)
end

options = SMCOptions(; moves=2, stepsize=0.35, max_tree_depth=4, history_limit=8)
function roundtrip(x)
    io = IOBuffer()
    serialize(io, x)
    seekstart(io)
    deserialize(io)
end
weighted_mean(s) = vec(s.positions * s.weights)
weighted_variance(s) = vec((s.positions .- weighted_mean(s)).^2 * s.weights)

@testset "BRM SMC batch, data update and continuation" begin
    data = (; y=[0.9, 1.6, 0.2, 1.3, 0.6, 1.1])
    first_data = (; y=data.y[1:2])
    rk = RKBRMI(normal_model(first_data))
    rng = Xoshiro(201)
    untouched = copy(rng)
    initial = brm_smc_fit(rk; rng, particles=512, options, stages=0)
    @test rand(rng) == rand(untouched)
    @test initial.state.temperature == 0
    @test initial.state.positions isa Matrix{Float64}
    @test_throws ArgumentError brm_smc_update(initial, data)
    @test_throws ArgumentError brm_smc_fit(initial; stages=-1)
    fit = brm_smc_fit(initial; progress=:state)
    @test fit.state.temperature == 1
    @test initial.state.temperature == 0
    @test weighted_mean(fit.state)[1] ≈ sum(first_data.y)/3 atol=0.09
    @test weighted_variance(fit.state)[1] ≈ 1/3 atol=0.09
    cp = roundtrip(brm_smc_checkpoint(initial))
    resumed = brm_smc_fit(brm_smc_resume(RKBRMI(normal_model(first_data)), cp))
    @test resumed.state.positions == fit.state.positions
    @test resumed.state.log_normalizer == fit.state.log_normalizer
    stopped = Treebars.with_progress(:state; description="SMC interruption") do node
        Treebars.request_interrupt!(node)
        brm_smc_fit(initial; progress=node)
    end
    @test stopped.state.temperature == 0
    @test stopped.state.positions == initial.state.positions
    @test brm_smc_fit(stopped).state.positions == fit.state.positions

    partial = brm_smc_update(fit, data; stages=0)
    updated = brm_smc_fit(partial)
    restored = brm_smc_fit(brm_smc_resume(rk,
        roundtrip(brm_smc_checkpoint(partial)); data))
    @test restored.state.positions == updated.state.positions
    @test restored.state.log_normalizer == updated.state.log_normalizer
    @test updated.state.temperature == 1
    @test fit.state.temperature == 1
    @test sum(updated.state.weights) ≈ 1
    @test weighted_mean(updated.state)[1] ≈ sum(data.y)/7 atol=0.07
    @test weighted_variance(updated.state)[1] ≈ 1/7 atol=0.05
    batch = brm_smc_fit(normal_model(data); rng=Xoshiro(303), particles=512, options)
    @test weighted_mean(batch.state)[1] ≈ sum(data.y)/7 atol=0.07
    @test weighted_variance(batch.state)[1] ≈ 1/7 atol=0.05
    exact_logz = -length(data.y)*log(2pi)/2 - log(7)/2 -
        (sum(abs2, data.y) - sum(data.y)^2/7)/2
    @test updated.state.log_normalizer ≈ exact_logz atol=0.2
    @test batch.state.log_normalizer ≈ exact_logz atol=0.2
    @test length(updated.state.history) <= options.history_limit
    bad = (; y=data.y .+ 1)
    @test_throws ArgumentError brm_smc_resume(rk, brm_smc_checkpoint(partial); data=bad)

    later = (; y=vcat(data.y, 0.7))
    next = brm_smc_update(updated, later; stages=0)
    next_cp = roundtrip(brm_smc_checkpoint(next))
    @test_throws ArgumentError brm_smc_resume(rk, next_cp; data=later)
    replay = brm_smc_resume(rk, next_cp; data=later, previous_data=data)
    @test brm_smc_fit(replay).state.positions == brm_smc_fit(next).state.positions
    first_data.y[1] += 1
    @test_throws ArgumentError brm_smc_fit(initial)
end

@testset "BRM SMC freezes training preprocessing and group coordinates" begin
    training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
    cumulative = (; x=[-2.0, 0.0, 2.0, 10.0], y=[-0.3, 0.4, 0.9, 1.1])
    fit = brm_smc_fit(scaled_model(training); rng=Xoshiro(7), particles=128, options)
    update = brm_smc_update(fit, cumulative; stages=0)
    expected = reprocess(fit.template, cumulative)
    fresh = RKBRMI(scaled_model(cumulative))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    u = fill(0.3, size(update.state.positions, 1))
    density(rk) = LogDensityProblems.logdensity(rk_logdensity_problem(rk; ad_backend=backend), u)
    @test density(update.backend) ≈ density(expected)
    @test !isapprox(density(update.backend), density(fresh); atol=1e-6)
    g = (; subject=["a", "b", "a", "b"], y=[0.2, 0.3, 0.1, 0.5])
    gf = brm_smc_fit(grouped_model(g); rng=Xoshiro(17), particles=64, options)
    error = try
        brm_smc_update(gf, (; subject=["a", "b", "c"], y=[0.2, 0.3, 0.1]); stages=0)
        nothing
    catch exception
        exception
    end
    @test error !== nothing
    @test occursin(r"level|group|subject", sprint(showerror, error))
end

@testset "BRM SMC deterministic scan history" begin
    data = (; x=[1.0, 0.2, -0.3, 0.9], y=[0.7, 0.5, 0.2, 0.8])
    first_data = (; x=data.x[1:2], y=data.y[1:2])
    fit = brm_smc_fit(scan_model(first_data); rng=Xoshiro(41), particles=512, options)
    update = brm_smc_update(fit, data)
    design = accumulated_input(data.x, 1.0)
    precision = 1 + sum(abs2, design)
    @test weighted_mean(update.state)[1] ≈ sum(design .* data.y)/precision atol=0.08
    @test weighted_variance(update.state)[1] ≈ 1/precision atol=0.06
    fresh = brm_smc_fit(scan_model(data); rng=Xoshiro(42), particles=512, options)
    @test weighted_mean(fresh.state)[1] ≈ sum(design .* data.y)/precision atol=0.08
    @test weighted_variance(fresh.state)[1] ≈ 1/precision atol=0.06
end

@testset "BRM SMC checkpoint across fresh processes" begin
    mktempdir() do directory
        path = joinpath(directory, "checkpoint.jls")
        script = joinpath(@__DIR__, "smc_checkpoint_process.jl")
        project = @__DIR__
        for mode in ("write", "read")
            @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script $mode $path`)
        end
    end
end
