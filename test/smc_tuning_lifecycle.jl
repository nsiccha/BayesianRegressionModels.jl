using Test
include("smc_tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture
using BayesianRegressionModels, ReactiveKernelsPPL, Random, Serialization, Treebars
const F = SMCTuningLifecycleFixture

function lifecycle_tests(progress)
    mktempdir() do directory
        for kind in (:initial, :cumulative, :window, :laplace)
            @testset "BRM adaptive pilot lifecycle: $kind" begin
                args, settings = F.targets(kind)
                rng = Xoshiro(502)
                rng_before = copy(rng)
                init_before = copy(settings.init)
                schedule = F.tune(args, settings; rng, progress)
                @test schedule isa SMCTuningSchedule
                @test rng == rng_before
                @test settings.init == init_before

                # Even a zero-draw stop must be recoverable, never a schedule.
                early = F.tune(args, settings; callback=(s,b) -> b === :init)
                @test early isa SMCTuningPartial
                @test early.progress.retained_draws == 0
                @test_throws ArgumentError SMCOptions(; tuning=early)
                before = F.roundtrip(early)
                @test F.equal(F.tune(args, settings; resume=early), schedule)
                @test F.equal(early, before)

                if kind === :initial
                    reference_stop = F.tune(args, settings;
                        callback=(s,b) -> s.phase === :reference &&
                            b === :window && s.retained_draws > 0)
                    @test reference_stop isa SMCTuningPartial
                    @test reference_stop.progress.retained_draws > 0
                    @test F.equal(F.tune(args, settings; resume=reference_stop), schedule)
                end

                checkpoint_dir = joinpath(directory, String(kind))
                tree_seen = Ref("")
                partial = F.tune(args, settings; progress, checkpoint_dir,
                    callback=(s,b) -> begin
                        tree_seen[] = render_text(progress)
                        s.phase === :grid && s.knot == 2 && b === :window &&
                            s.retained_draws > 0
                    end)
                @test partial isa SMCTuningPartial
                @test partial.progress.completed_knots == 1
                @test partial.progress.retained_draws > 0
                @test count(==('\n'), tree_seen[]) >= 3
                @test occursin("pilot", lowercase(tree_seen[]))
                @test readdir(checkpoint_dir) == ["tuning_latest.jls"]
                serialize(joinpath(directory, "$kind.jls"), (; partial, schedule))
            end
        end

        @testset "BRM identity, progress interruption and checkpoint ownership" begin
            args, settings = F.targets(:initial)
            dir = joinpath(directory, "identity")
            early = F.tune(args, settings; checkpoint_dir=dir,
                callback=(s,b) -> b === :init)
            changed_data = F.model(merge(F.DATA, (; y=F.DATA.y .+ 0.1)))
            @test_throws ArgumentError F.tune((changed_data,), settings; resume=early)
            changed_transform = F.model(merge(F.DATA, (; x=F.DATA.x .* 2)))
            @test_throws ArgumentError F.tune((changed_transform,), settings; resume=early)
            changed_frame = @brm F.DATA begin
                mu ~ 1 + zscale(x)
                effect(mu, :) ~ Normal(0, 1)
                y ~ Normal(mu, 1)
            end
            @test_throws ArgumentError F.tune((changed_frame,),
                merge(settings, (; init=zeros(2))); resume=early)
            @test_throws ArgumentError F.tune(args, settings;
                resume=early, pilot_evaluations=33)
            @test_throws ArgumentError F.tune(args, settings; checkpoint_dir=dir)
            write(joinpath(dir, "keep.txt"), "unrelated")
            replacement = F.tune(args, settings; checkpoint_dir=dir, overwrite=true,
                callback=(s,b) -> b === :init)
            @test replacement isa SMCTuningPartial
            @test read(joinpath(dir, "keep.txt"), String) == "unrelated"
            stopped = with_progress(:state; description="stopped pilot") do node
                request_interrupt!(node)
                F.tune(args, settings; progress=node)
            end
            @test stopped isa SMCTuningPartial
            @test F.tune(args, settings; resume=stopped) isa SMCTuningSchedule

            # Production consumes completed schedules and keeps the exact reference.
            schedule = F.tune(args, settings)
            fit = brm_smc_fit(args...; rng=Xoshiro(503), particles=32,
                options=SMCOptions(; tuning=schedule))
            @test fit.state.temperature == 1
            @test F.equal(smc_reference(fit.state), smc_reference(schedule))
        end

        @testset "fresh Julia process reads partials and disk checkpoints" begin
            worker = joinpath(@__DIR__, "smc_tuning_lifecycle_process.jl")
            process = run(ignorestatus(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$(@__DIR__) $worker $directory`))
            @test success(process)
        end
    end
end

with_progress(:state; description="BRM pilot recovery verification") do progress
    if haskey(ENV, "KB_PROGRESS_FILE")
        snapshot_progress(progress, ENV["KB_PROGRESS_FILE"]) do
            lifecycle_tests(progress)
        end
    else
        lifecycle_tests(progress)
    end
end
