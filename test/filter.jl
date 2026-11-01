using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using SequentialMonteCarlo, Distributions, Random, Serialization, Statistics, Treebars
const BRM = BayesianRegressionModels
const PPL = ReactiveKernelsPPL
include("filter_models.jl")
const FM = FilterModels

function checkpoint_bytes(filter)
    io = IOBuffer()
    serialize(io, brm_filter_checkpoint(filter))
    take!(io)
end
records(x, y) = [(; observation=(; y=y[i]), inputs=(; x=x[i])) for i in eachindex(x)]
const TRAIN = (; x=[-1.0, 0.0, 1.0, 2.0], y=[0.3, -0.1, 0.7, 0.4])

@testset "BRM deterministic streaming and cooperative continuation" begin
    rk = RKBRMI(FM.provider(TRAIN))
    rng = Xoshiro(16)
    original_rng = copy(rng)
    initial = brm_filter(rk; parameters=(; gain=1.2), inputs=(:x,), rng,
        particles=16, history_limit=4)
    @test rand(rng) == rand(original_rng)
    @test !hasproperty(initial, :backend)
    @test !haskey(initial.target.data, :x) && !haskey(initial.target.data, :y)
    x = [0.4, -0.5, 1.3, 0.8, -0.2, 0.9]
    y = [0.0, 0.5, 0.4, -0.1, 1.0, 0.6]
    all_records = records(x, y)
    filtered = initial
    carry = 0.0
    total = 0.0
    for record in all_records
        carry = 0.7carry + 1.2record.inputs.x
        score = logpdf(Normal(carry, 1), record.observation.y)
        filtered = brm_filter_step(filtered, record.observation; inputs=record.inputs)
        @test filtered.state.predictive_logscore ≈ score atol=1e-12
        total += score
    end
    @test filtered.state.log_normalizer ≈ total atol=1e-12
    streamed = brm_filter_stream(initial, all_records)
    @test checkpoint_bytes(streamed) == checkpoint_bytes(filtered)
    live = Iterators.Stateful(all_records)
    stopped = brm_filter_stream(initial, live; stop=_ -> true)
    @test checkpoint_bytes(stopped) == checkpoint_bytes(initial)
    @test length(live) == length(all_records)
    partial = brm_filter_stream(initial, live; steps=2)
    @test length(live) == length(all_records)-2
    @test checkpoint_bytes(brm_filter_stream(partial, live)) == checkpoint_bytes(filtered)
    interrupted = Treebars.with_progress(:state; description="filter interruption") do node
        Treebars.request_interrupt!(node)
        brm_filter_stream(initial, all_records; progress=node)
    end
    @test checkpoint_bytes(interrupted) == checkpoint_bytes(initial)
    restored = brm_filter_resume(RKBRMI(FM.provider(TRAIN)),
        deserialize(IOBuffer(checkpoint_bytes(partial))))
    @test checkpoint_bytes(brm_filter_stream(restored, all_records[3:end])) == checkpoint_bytes(filtered)
    @test_throws ArgumentError brm_filter_stream(initial, all_records; steps=-1)
    @test_throws ArgumentError brm_filter(rk; ensemble=NamedTuple[])
    old = checkpoint_bytes(filtered)
    @test_throws Exception brm_filter_step(filtered, (; y=Inf); inputs=(; x=0.1))
    @test checkpoint_bytes(filtered) == old
    longer = brm_filter_stream(filtered, Iterators.repeated(first(all_records), 100))
    @test length(longer.state.history) <= 4
    @test Base.summarysize(longer) <= Base.summarysize(filtered) + 1024
    @test length(checkpoint_bytes(longer)) <= length(old) + 1024
    large_training = (; x=fill(1.0, 10_000), y=fill(0.0, 10_000))
    large = brm_filter(RKBRMI(FM.provider(large_training)); parameters=(; gain=1.2),
        inputs=(:x,), rng=Xoshiro(16), particles=16, history_limit=4)
    @test Base.summarysize(large) <= Base.summarysize(initial) + 1024
end

@testset "BRM calibrated ensemble warm-up counts training only in states" begin
    rk = RKBRMI(FM.provider(TRAIN))
    initial = brm_filter(rk; ensemble=[(; gain=0.5), (; gain=1.5)],
        weights=[0.2, 0.8], inputs=(:x,), rng=Xoshiro(19), particles=8)
    warm = brm_filter_stream(initial, records(TRAIN.x, TRAIN.y); condition_state_only=true)
    @test warm.state.parameter_mass ≈ [0.2, 0.8]
    @test warm.state.log_normalizer == 0
    x, y = 0.8, 1.1
    carries = [last(FM.accumulated_input([TRAIN.x; x], gain)) for gain in (0.5, 1.5)]
    likelihoods = pdf.(Normal.(carries, 1), y)
    masses = [0.2, 0.8] .* likelihoods
    updated = brm_filter_step(warm, (; y); inputs=(; x))
    @test updated.state.predictive_logscore ≈ log(sum(masses)) atol=1e-12
    @test updated.state.parameter_mass ≈ masses ./ sum(masses) atol=1e-12
    @test warm.state.parameter_mass ≈ [0.2, 0.8]
end

@testset "BRM authored AR filtering agrees with the Kalman recursion" begin
    data = (; t=collect(1.0:4.0), y=zeros(4))
    rk = RKBRMI(FM.ar_model(data))
    point = (; ar_mu_t=(; phi_raw=atanh(0.6)), mu_Intercept=0.0, mu_ar_mu_t=1.0)
    filter = brm_filter(rk; parameters=point, particles=6000, rng=Xoshiro(41))
    ys = [0.4, -0.3, 0.7, 1.0, 0.2]
    mean, variance = 0.0, 1.0
    for (i, y) in enumerate(ys)
        if i > 1
            mean, variance = 0.6mean, 0.36variance + 1
        end
        score = logpdf(Normal(mean, sqrt(variance+1)), y)
        gain = variance/(variance+1)
        mean += gain*(y-mean)
        variance *= 1-gain
        filter = brm_filter_step(filter, (; y))
        samples = [only(values(carry))[end, 1] for carry in filter.state.carries]
        estimated = sum(samples .* filter.state.weights)
        @test filter.state.predictive_logscore ≈ score atol=0.07
        @test estimated ≈ mean atol=0.08
        @test sum((samples .- estimated).^2 .* filter.state.weights) ≈ variance atol=0.08
    end
end

@testset "BRM observed groups advance independently" begin
    rk = RKBRMI(FM.observed_groups((; y=[[0.0, 0.0], [0.0, 0.0]], T=2)))
    initial = brm_filter(rk; groups=(; i=1:2), particles=8, rng=Xoshiro(38))
    first_step = brm_filter_step(initial, (; y=[1.0, 2.0]))
    @test first_step.state.predictive_logscore ≈ sum(logpdf.(Normal(), [1.0, 2.0]))
    second = brm_filter_step(first_step, (; y=[missing, 3.0]); groups=(; i=[2]))
    @test second.state.predictive_logscore ≈ logpdf(Normal(1, 1), 3)
    third = brm_filter_step(second, (; y=[1.5, missing]); groups=(; i=[1]))
    @test third.state.predictive_logscore ≈ logpdf(Normal(0.5, 1), 1.5)
end

@testset "BRM streaming keeps fitted preprocessing" begin
    training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
    rk = RKBRMI(FM.scaled(training))
    point = fill(0.2, rk.model.layout.total)
    filtered = brm_filter(rk; parameters=point, inputs=(:x,), particles=8, rng=Xoshiro(44))
    @test isempty(intersect(keys(filtered.target.data), (:x, :x_fit, :X_mu, :y)))
    for (x, y) in ((8.0, 1.2), (9.0, 1.3))
        replay = reprocess(rk, (; x=[x], y=[y]))
        score = PPL.prepare_query(replay.model, rk_bound_plan(replay), :likelihood)(point)
        filtered = brm_filter_step(filtered, (; y); inputs=(; x))
        @test filtered.state.predictive_logscore ≈ score atol=1e-12
    end
    checkpoint = brm_filter_checkpoint(filtered)
    changed = RKBRMI(FM.scaled((; x=[-4.0, 0.0, 4.0], y=training.y)))
    @test_throws ArgumentError brm_filter_resume(changed, checkpoint)
    saved = brm_fitted_state(rk)
    datafree = RKBRMI(FM.scaled((; x=[8.0], y=[1.2])), saved)
    restored = brm_filter_resume(datafree, checkpoint)
    @test checkpoint_bytes(restored) == checkpoint_bytes(filtered)
end

@testset "BRM filter checkpoints across independent Julia processes" begin
    mktempdir() do folder
        path = joinpath(folder, "filter.jls")
        script = joinpath(@__DIR__, "filter_checkpoint_process.jl")
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script write $path`)
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script read $path`)
    end
end
