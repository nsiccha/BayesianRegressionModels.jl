using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Serialization
include("smc_tuning_fixture.jl")
using .SMCTuningFixture

@testset "SMC tuning: authored correlated Gaussian oracle" begin
    complete = SMCTuningFixture.data()
    for data in (SMCTuningFixture.prefix(complete, 16), complete)
        rk = RKBRMI(SMCTuningFixture.model(data))
        names = coordinate_names(rk.model.layout)
        coefficient_names = [Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")]
        @test Set(names) == Set(coefficient_names)
        order = Int.(indexin(coefficient_names, names))
        problem = rk_logdensity_problem(rk;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        p = SMCTuningFixture.posterior(data)
        @test p.variances[2] / p.variances[1] > 300
        for beta in ([0.0, 0.0], [0.7, -0.3], p.mean)
            u = zeros(2)
            u[order] = beta
            joint = SMCTuningFixture.logjoint(data, beta)
            @test LogDensityProblems.logdensity(problem, u) ≈ joint atol=2e-10
            @test joint ≈ SMCTuningFixture.logposterior(p, beta) + p.log_normalizer atol=2e-10
        end
    end
end

function check_posterior(fit, data)
    @test fit.state.temperature == 1
    order = Int.(indexin([Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")],
        coordinate_names(fit.backend.model.layout)))
    exact = SMCTuningFixture.posterior(data)
    measured = SMCTuningFixture.direction_summary(fit.state.positions, fit.state.weights, order)
    println("TUNED_GAUSSIAN standardized_mean_error=",
        (measured.means .- exact.means) ./ sqrt.(exact.variances),
        " variance_ratio=", measured.variances ./ exact.variances,
        " logz_error=", fit.state.log_normalizer - exact.log_normalizer)
    @test all(abs.(measured.means .- exact.means) .< 0.2 .* sqrt.(exact.variances))
    @test all(abs.(measured.variances ./ exact.variances .- 1) .< 0.30)
    @test fit.state.log_normalizer ≈ exact.log_normalizer atol=0.4
end

roundtrip_tuning(x) = deserialize(IOBuffer(let io=IOBuffer(); serialize(io, x); take!(io); end))

@testset "BRM independent tuning, bridge identities and exact continuation" begin
    data = SMCTuningFixture.data()
    first_data = SMCTuningFixture.prefix(data, 16)
    rk = RKBRMI(SMCTuningFixture.model(first_data))
    pilot_rng = Xoshiro(731)
    pilot_before = copy(pilot_rng)
    init = zeros(2)
    reference = SMCReference([-0.2, 0.3], [1.4, 0.8])
    temperatures = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0)
    pilot_options = SMCOptions(; moves=3, stepsize=0.15, max_tree_depth=7)
    schedule = brm_smc_tune(rk; rng=pilot_rng, init, reference,
        temperatures, pilot_steps=128, options=pilot_options)
    @test pilot_rng == pilot_before
    @test init == zeros(2)
    @test smc_reference(schedule).location == reference.location
    @test smc_reference(schedule).scale == reference.scale
    copy_reference = smc_reference(schedule)
    copy_reference.location[1] = 100
    @test smc_reference(schedule).location == reference.location
    options = SMCOptions(; tuning=schedule, history_limit=3)
    production_rng = Xoshiro(732)
    production_before = copy(production_rng)
    initial = brm_smc_fit(rk; rng=production_rng, particles=768, options, stages=0)
    @test production_rng == production_before
    @test smc_reference(initial.state).location == reference.location
    @test smc_reference(initial.state).scale == reference.scale
    @test_throws ArgumentError brm_smc_tune(initial, data; rng=Xoshiro(733), init)
    @test_throws ArgumentError brm_smc_fit(rk; rng=Xoshiro(734), particles=10, options,
        reference=SMCReference(zeros(2), ones(2)))
    initial_cp = brm_smc_checkpoint(initial)
    fitted = brm_smc_fit(initial)
    check_posterior(fitted, first_data)
    @test SMCTuningFixture.checkpoint_equal(initial_cp, brm_smc_checkpoint(initial))
    restored = brm_smc_fit(brm_smc_resume(rk, roundtrip_tuning(initial_cp)))
    @test SMCTuningFixture.checkpoint_equal(brm_smc_checkpoint(restored), brm_smc_checkpoint(fitted))
    @test length(fitted.state.history) <= 3
    for stage in fitted.state.history
        @test stage.tuning_index > 0
        @test stage.kernel == schedule.kernels[stage.tuning_index]
        @test stage.evaluations > 0
        @test isfinite(stage.mean_squared_movement) && stage.mean_squared_movement >= 0
    end

    before = brm_smc_checkpoint(fitted)
    @test_throws ArgumentError brm_smc_update(fitted, data)
    @test_throws ArgumentError brm_smc_fit(SMCTuningFixture.model(data);
        rng=Xoshiro(735), particles=10, options)
    @test SMCTuningFixture.checkpoint_equal(before, brm_smc_checkpoint(fitted))
    update_schedule = brm_smc_tune(fitted, data; rng=Xoshiro(736), init,
        temperatures, pilot_steps=128, options=pilot_options)
    @test smc_reference(update_schedule) === nothing
    update_options = SMCOptions(; tuning=update_schedule, history_limit=3)
    partial = brm_smc_update(fitted, data; options=update_options, stages=0)
    @test smc_reference(partial.state) === nothing
    updated = brm_smc_fit(partial)
    check_posterior(updated, data)
    replay = brm_smc_fit(brm_smc_resume(rk,
        roundtrip_tuning(brm_smc_checkpoint(partial)); data))
    @test SMCTuningFixture.checkpoint_equal(brm_smc_checkpoint(replay), brm_smc_checkpoint(updated))
    changed = (; data.x1, data.x2, y=data.y .+ 0.1)
    @test_throws ArgumentError brm_smc_update(fitted, changed; options=update_options)
    @test SMCTuningFixture.checkpoint_equal(before, brm_smc_checkpoint(fitted))

    batch_model = RKBRMI(SMCTuningFixture.model(data))
    batch_schedule = brm_smc_tune(batch_model; rng=Xoshiro(737), init,
        temperatures, pilot_steps=128, options=pilot_options)
    batch_options = SMCOptions(; tuning=batch_schedule)
    for seed in (738, 739)
        batch = brm_smc_fit(batch_model; rng=Xoshiro(seed), particles=768, options=batch_options)
        check_posterior(batch, data)
    end
end

@testset "tuned SMC checkpoints across Julia processes" begin
    mktempdir() do directory
        path = joinpath(directory, "tuned-smc.bin")
        worker = joinpath(@__DIR__, "smc_tuning_checkpoint_process.jl")
        for mode in ("write", "read")
            process = run(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(@__DIR__) $worker $mode $path`))
            @test success(process)
            success(process) || break
        end
    end
end
