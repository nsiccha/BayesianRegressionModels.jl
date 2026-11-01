# Public BRM integration of RKPPL's structured numerical failures.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, LogDensityProblems, Treebars

diagnostic_model(d) = @brm d begin
    mu ~ Normal(0, 1)
    y ~ Normal(mu, 1)
end

# Deliberately unstable sigmoid arithmetic overflows at distant proposals,
# even though every initial particle has a finite density and gradient.
mutation_failure_model(d) = @brm d begin
    mu ~ Normal(0, 1)
    exploding = exp(1e4 * mu^2)
    location = exploding / (1 + exploding)
    y ~ Normal(location, 1)
end

function capture_smc_failure(f)
    try
        f()
        nothing
    catch exception
        (; exception, backtrace=catch_backtrace())
    end
end

function checkpoint_bytes(fit)
    io = IOBuffer()
    serialize(io, brm_smc_checkpoint(fit))
    take!(io)
end

@testset "BRM exposes exact SMC numerical diagnostics" begin
    good = RKBRMI(diagnostic_model((; y=[0.3])))
    bad = RKBRMI(diagnostic_model((; y=[NaN])))
    rng = Xoshiro(901)
    untouched = copy(rng)
    initial = brm_smc_fit(good; rng, particles=16, stages=0)
    expected_position = copy(initial.state.positions[:, 1])

    # Both public progress modes must expose the exception itself, not a
    # task wrapper whose rendering expands the model's parametric types.
    for progress in (nothing, :state)
        failure = capture_smc_failure() do
            brm_smc_fit(bad; rng, particles=16, stages=1, progress)
        end
        @test failure !== nothing
        if failure !== nothing
            err = failure.exception
            @test err isa SMCNonfiniteError
            if err isa SMCNonfiniteError
                @test err.position == expected_position
                @test isnan(err.value)
                @test err.quantity == :density
                @test err.endpoint == :target
                @test err.temperature == 0
                @test err.gradient === nothing
                @test isnan(LogDensityProblems.logdensity(
                    rk_logdensity_problem(bad; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)),
                    err.position))
                @test ncodeunits(sprint(showerror, err)) < 2048
                @test ncodeunits(sprint(showerror, err, failure.backtrace)) < 16384
                restored = let io = IOBuffer()
                    serialize(io, err)
                    seekstart(io)
                    deserialize(io)
                end
                @test restored.position == expected_position
                @test restored.position !== err.position
                @test restored.endpoint == err.endpoint
            end
        end
    end
    @test rand(copy(rng)) == rand(copy(untouched))

    fit = brm_smc_fit(initial)
    @test fit.state.temperature == 1
    before = checkpoint_bytes(fit)
    for progress in (nothing, :state)
        failure = capture_smc_failure() do
            brm_smc_update(fit, (; y=[0.3, NaN]); stages=1, progress)
        end
        @test failure !== nothing
        if failure !== nothing
            err = failure.exception
            @test err isa SMCNonfiniteError
            if err isa SMCNonfiniteError
                @test err.position == fit.state.positions[:, 1]
                @test isnan(err.value)
                @test err.endpoint == :target
                @test err.temperature == 0
            end
        end
        # Covers the complete public checkpoint, including the copied RNG.
        @test checkpoint_bytes(fit) == before
    end
end

@testset "BRM preserves the failing mutation proposal and completed stage" begin
    rk = RKBRMI(mutation_failure_model((; y=[0.3])))
    initial = brm_smc_fit(rk; rng=Xoshiro(902), particles=16, stages=0,
        reference=SMCReference([0.0], [1e-4]),
        options=SMCOptions(; moves=1, stepsize=1e4, max_tree_depth=2))
    target = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test all(abs.(initial.state.positions) .< 0.1)
    @test all(p -> isfinite(LogDensityProblems.logdensity(target, p)),
        eachcol(initial.state.positions))
    before = checkpoint_bytes(initial)
    for progress in (nothing, :state)
        failure = capture_smc_failure() do
            brm_smc_fit(initial; stages=1, progress)
        end
        @test failure !== nothing
        if failure !== nothing
            err = failure.exception
            @test err isa SMCNonfiniteError
            if err isa SMCNonfiniteError
                @test abs(only(err.position)) > 0.1
                @test all(p -> p != err.position, eachcol(initial.state.positions))
                @test isnan(err.value)
                @test isnan(LogDensityProblems.logdensity(target, err.position))
                @test err.quantity == :density
                @test err.endpoint == :target
                @test 0 < err.temperature <= 1
                @test ncodeunits(sprint(showerror, err, failure.backtrace)) < 16384
                saved_position = copy(err.position)
                # Advancing the same retained stage again cannot mutate the
                # diagnostic through an aliased sampler workspace.
                again = capture_smc_failure(() -> brm_smc_fit(initial; stages=1))
                @test again.exception isa SMCNonfiniteError
                @test again.exception.position == saved_position
                @test again.exception.position !== err.position
                @test err.position == saved_position
            end
        end
        @test checkpoint_bytes(initial) == before
    end
end
