using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Serialization
using LinearAlgebra
include("smc_tuning_fixture.jl")
include("smc_window_fixture.jl")
include("streaming_oracles.jl")
const G = SMCTuningFixture
const W = SMCWindowFixture

# A two-dimensional test oracle only. The public scale action reconstructs
# this Gaussian's covariance; no dense geometry is supplied to any sampler.
function reference_oracle(reference, order)
    scale = reference.scale isa AbstractVector ? Diagonal(reference.scale) :
        WarmupHMC.scale_from_data(reference.scale)
    root = hcat(scale*[1.0, 0.0], scale*[0.0, 1.0])
    covariance = (root*root')[order, order]
    precision = inv(covariance)
    center = reference.location[order]
    information = precision*center
    StreamingOracles.gaussian_summary(precision[1, 1], precision[1, 2],
        precision[2, 2], information[1], information[2], 0.0, 0)
end
function oracle_directions(oracle)
    m1, m2 = oracle.mean
    v1, cov, v2 = oracle.covariance
    (; means=[m1+m2, m1-m2]./sqrt(2),
       variances=[v1+v2+2cov, v1+v2-2cov]./2)
end

@testset "Correlated window posteriors with independent frozen pilots" begin
    all_data = G.data(; rows=32, separation=0.08)
    take_rows(rows) = map(column -> column[rows], all_data)
    training = take_rows(1:16)
    rk = RKBRMI(G.model(training))
    pilot_options = SMCOptions(; moves=3, max_tree_depth=7)
    controls = (; pilot=:adaptive, pilot_steps=256, max_reflections=4,
        recording_target=256, pilot_evaluations=256,
        temperatures=(0.0, 0.01, 0.1, 1.0), options=pilot_options)
    initial_schedule = brm_smc_tune(rk; rng=Xoshiro(861), init=zeros(2),
        reference=:pilot, controls...)
    initial = brm_smc_fit(rk; rng=Xoshiro(862), particles=1024,
        options=SMCOptions(; tuning=initial_schedule))
    basis = smc_reference(initial_schedule).scale
    basis_before = W.roundtrip(basis)
    supplied = brm_smc_carry(initial; basis)
    @test W.equal(basis, basis_before)
    @test supplied.posterior.reference.location ≈ W.weighted_mean(initial.state)
    initial_before = W.roundtrip(smc_checkpoint(initial.state))
    carried = brm_smc_carry(initial; basis=:fit, max_reflections=4, maxiter=100)
    @test W.equal(smc_checkpoint(initial.state), initial_before)
    @test carried.posterior.fit_diagnostics !== nothing
    initial_order = Int.(indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")],
        coordinate_names(rk.model.layout)))
    initial_exact = G.posterior(training)
    initial_carried = oracle_directions(reference_oracle(carried.posterior.reference, initial_order))
    println("FITTED_INITIAL_REFERENCE variance_ratio=",
        initial_carried.variances ./ initial_exact.variances,
        " diagnostics=", carried.posterior.fit_diagnostics)
    @test carried.posterior.reference.location ≈ W.weighted_mean(initial.state)
    @test_throws ArgumentError brm_smc_carry(initial; basis, scale=basis)
    # The explicit-reference form remains useful for externally calibrated
    # posterior summaries, and owns the supplied reference.
    external = deepcopy(carried.posterior.reference)
    direct = brm_smc_carry(rk; reference=external)
    external.location[1] += 100
    @test W.equal(direct.posterior.reference, carried.posterior.reference)
    sizes = Int[]
    completed = nothing
    for (index, rows) in enumerate((17:24, 25:32))
        data = take_rows(rows)
        before = W.roundtrip(carried)
        rng, init = Xoshiro(870+index), zeros(2)
        rng_before = copy(rng)
        schedule = brm_smc_tune(carried, G.model(data);
            rng, init, reference=:pilot, controls...)
        @test rng == rng_before
        @test init == zeros(2)
        @test W.equal(carried, before)
        if index == 1
            uninflated = brm_smc_tune(carried, G.model(data);
                rng, init, reference=:pilot, reference_inflation=1.0, controls...)
            @test rng == rng_before
            @test init == zeros(2)
            @test W.equal(carried, before)
            @test schedule.pilot.reference_work.reference_inflation == 2.0
            @test uninflated.pilot.reference_work.reference_inflation == 1.0
            @test uninflated.pilot.reference_work.draw_sd ==
                schedule.pilot.reference_work.draw_sd
            reference = smc_reference(schedule)
            @test reference.location == smc_reference(uninflated).location
            # Identical pilot moments and production RNG: only proposal SDs
            # change. The target keeps the same carried posterior.
            normal = brm_smc_update(carried, G.model(data);
                rng=Xoshiro(879), particles=64, stages=0,
                options=SMCOptions(; tuning=uninflated))
            inflated = brm_smc_update(carried, G.model(data);
                rng=Xoshiro(879), particles=64, stages=0,
                options=SMCOptions(; tuning=schedule))
            @test inflated.state.positions .- reference.location ≈
                2 .* (normal.state.positions .- reference.location)
            @test W.equal(inflated.previous, normal.previous)
        end
        options = SMCOptions(; tuning=schedule, history_limit=3)
        partial = brm_smc_update(carried, G.model(data);
            rng=Xoshiro(880+index), particles=1024, options, stages=0)
        completed = brm_smc_fit(partial)
        @test W.equal(smc_reference(completed.state), smc_reference(schedule))
        @test all(stage -> stage.kernel == schedule.kernels[stage.tuning_index],
            completed.state.history)
        restored = brm_smc_fit(brm_smc_resume(G.model(data),
            W.roundtrip(brm_smc_checkpoint(partial))))
        @test W.equal(brm_smc_checkpoint(completed), brm_smc_checkpoint(restored))
        @test_throws ArgumentError brm_smc_update(carried,
            G.model(merge(data, (; y=data.y .+ 0.1))); options, stages=0)
        exact = G.posterior(take_rows(1:last(rows)))
        order = Int.(indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")],
            coordinate_names(completed.backend.model.layout)))
        measured = G.direction_summary(completed.state.positions, completed.state.weights, order)
        actual = StreamingOracles.gaussian_window(
            reference_oracle(carried.posterior.reference, order), data)
        actual_directions = oracle_directions(actual)
        actual_mean_error = (measured.means .- actual_directions.means) ./
            sqrt.(actual_directions.variances)
        actual_variance_ratio = measured.variances ./ actual_directions.variances
        println("ACTUAL_CARRIED_GAUSSIAN rows=", last(rows),
            " standardized_mean_error=", actual_mean_error,
            " variance_ratio=", actual_variance_ratio,
            " logz_error=", completed.state.log_normalizer-actual.log_evidence)
        @test all(abs.(actual_mean_error) .< 0.25)
        @test all(abs.(actual_variance_ratio .- 1) .< 0.3)
        @test abs(completed.state.log_normalizer-actual.log_evidence) < 0.3
        if haskey(ENV, "BRM_WINDOW_DIAGNOSTICS")
            path = ENV["BRM_WINDOW_DIAGNOSTICS"]
            mkpath(path)
            serialize(joinpath(path, "window-$index.bin"),
                (; reference=carried.posterior.reference,
                   proposal=smc_reference(schedule), data, order,
                   actual, actual_directions, measured,
                   positions=completed.state.positions, weights=completed.state.weights,
                   log_normalizer=completed.state.log_normalizer))
        end
        mean_error = (measured.means .- exact.means) ./ sqrt.(exact.variances)
        variance_ratio = measured.variances ./ exact.variances
        println("FITTED_CARRY rows=", last(rows), " standardized_mean_error=", mean_error,
            " variance_ratio=", variance_ratio)
        @test all(abs.(mean_error) .< 0.4)
        @test all(abs.(variance_ratio .- 1) .< 0.4)
        carried = brm_smc_carry(completed; basis=:fit, max_reflections=4, maxiter=100)
        push!(sizes, W.serialized_size(carried))
    end
    batch_schedule = brm_smc_tune(G.model(all_data); rng=Xoshiro(890), init=zeros(2),
        reference=:pilot, controls...)
    batch = brm_smc_fit(G.model(all_data); rng=Xoshiro(891), particles=1024,
        options=SMCOptions(; tuning=batch_schedule))
    order = Int.(indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")],
        coordinate_names(batch.backend.model.layout)))
    a = G.direction_summary(completed.state.positions, completed.state.weights, order)
    b = G.direction_summary(batch.state.positions, batch.state.weights, order)
    @test all(abs.(a.means .- b.means) .< 0.4 .* sqrt.(b.variances))
    @test all(abs.(a.variances ./ b.variances .- 1) .< 0.4)
    # The carried fit retains a bounded number of reflection directions.
    @test maximum(sizes) < 32_768
end
