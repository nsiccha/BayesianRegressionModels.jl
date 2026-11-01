using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Serialization
include("smc_tuning_fixture.jl")

function check_adaptive_posterior(fit, data)
    @test fit.state.temperature == 1
    order = Int.(indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")],
        coordinate_names(fit.backend.model.layout)))
    exact = SMCTuningFixture.posterior(data)
    measured = SMCTuningFixture.direction_summary(fit.state.positions, fit.state.weights, order)
    mean_error = (measured.means .- exact.means) ./ sqrt.(exact.variances)
    variance_ratio = measured.variances ./ exact.variances
    logz_error = fit.state.log_normalizer - exact.log_normalizer
    println("ADAPTIVE_GAUSSIAN standardized_mean_error=", mean_error,
        " variance_ratio=", variance_ratio, " logz_error=", logz_error)
    @test all(abs.(mean_error) .< 0.2)
    @test all(abs.(variance_ratio .- 1) .< 0.3)
    @test abs(logz_error) < 0.4
end

@testset "BRM adaptive SMC learns and freezes its normalized reference" begin
    data = SMCTuningFixture.data()
    first_data = SMCTuningFixture.prefix(data, 16)
    rk = RKBRMI(SMCTuningFixture.model(first_data))
    init = zeros(2)
    rng = Xoshiro(761)
    before_rng = copy(rng)
    pilot_options = SMCOptions(; moves=3, max_tree_depth=7)
    controls = (; pilot=:adaptive, pilot_steps=64, max_reflections=4,
        recording_target=64, pilot_evaluations=256,
        temperatures=(0.0, 0.001, 0.01, 0.1, 1.0), options=pilot_options)
    schedule = brm_smc_tune(rk; rng, init, reference=:pilot, controls...)
    @test rng == before_rng
    @test init == zeros(2)
    reference = smc_reference(schedule)
    @test all(isfinite, reference.location)
    owned_reference = smc_reference(schedule)
    owned_reference.location[1] += 100
    @test SMCTuningFixture.checkpoint_equal(reference, smc_reference(schedule))

    options = SMCOptions(; tuning=schedule, history_limit=2)
    initial = brm_smc_fit(rk; rng=Xoshiro(762), particles=1024, options, stages=0)
    @test SMCTuningFixture.checkpoint_equal(reference, smc_reference(initial.state))
    before = brm_smc_checkpoint(initial)
    fitted = brm_smc_fit(initial)
    check_adaptive_posterior(fitted, first_data)
    @test SMCTuningFixture.checkpoint_equal(before, brm_smc_checkpoint(initial))
    @test length(fitted.state.history) <= 2
    for stage in fitted.state.history
        @test 0 <= stage.mean_acceptance <= 1
        @test stage.evaluations > 0
        @test stage.kernel == schedule.kernels[stage.tuning_index]
    end
    @test_throws ArgumentError brm_smc_fit(rk; rng=Xoshiro(763), particles=10,
        options, reference=SMCReference(zeros(2), ones(2)))

    # A second independent production cloud uses the same frozen pilot result.
    second = brm_smc_fit(rk; rng=Xoshiro(764), particles=1024, options)
    check_adaptive_posterior(second, first_data)
    update_schedule = brm_smc_tune(fitted, data;
        rng=Xoshiro(765), init, controls...)
    @test smc_reference(update_schedule) === nothing
    update_options = SMCOptions(; tuning=update_schedule, history_limit=2)
    updated = brm_smc_update(fitted, data; options=update_options)
    check_adaptive_posterior(updated, data)
    @test_throws ArgumentError brm_smc_update(fitted,
        (; data.x1, data.x2, y=data.y .+ 0.1); options=update_options)
end

@testset "adaptive SMC checkpoints across Julia processes" begin
    mktempdir() do directory
        path = joinpath(directory, "adaptive-smc.bin")
        worker = joinpath(@__DIR__, "smc_adaptive_checkpoint_process.jl")
        for mode in ("write", "read")
            process = run(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(@__DIR__) $worker $mode $path`))
            @test success(process)
            success(process) || break
        end
    end
end
