using Test
include("smc_tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture
using BayesianRegressionModels, ReactiveKernelsPPL, Random, Serialization
const F = SMCTuningLifecycleFixture

directory = only(ARGS)
for kind in (:initial, :cumulative, :window, :laplace)
    @testset "fresh-process pilot recovery: $kind" begin
        args, settings = F.targets(kind)
        saved = deserialize(joinpath(directory, "$kind.jls"))
        events = []
        callback = (s, b) -> (push!(events, (s.phase, s.knot)); false)
        resumed = F.tune(args, settings; rng=Xoshiro(999),
            resume=saved.partial, callback)
        @test F.equal(resumed, saved.schedule)
        @test all(e -> e[1] === :grid && e[2] == 2, events)
        disk = F.tune(args, settings; rng=Xoshiro(998),
            checkpoint_dir=joinpath(directory, String(kind)), resume=true)
        @test F.equal(disk, saved.schedule)
        # A complete disk checkpoint returns without executing a callback.
        again = F.tune(args, settings;
            checkpoint_dir=joinpath(directory, String(kind)), resume=true,
            callback=(s,b) -> error("completed tuning must not run a pilot"))
        @test F.equal(again, saved.schedule)
    end
end
