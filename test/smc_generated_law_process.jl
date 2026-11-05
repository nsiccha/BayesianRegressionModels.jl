# Producer/consumer worker for smc_generated_law_recovery.jl.
#
# A gmrf field's law is a binding of the RK program's fresh generated module
# (`##RKEmittedModels#N`), so its raw prior expression names a module that a
# fresh process does not have. Checkpoints and tuning identities must not.
using Test
include("smc_tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture
using BayesianRegressionModels, ReactiveKernelsPPL, Random, Serialization
const F = SMCTuningLifecycleFixture

const DATA = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
    time=[0.0, 0.3, 0.3, 1.7], y=[0.2, -0.1, 0.4, 0.0],
    xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5])

model() = RKBRMI(@brm DATA begin
    field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
    sd(field, gmrf((east, north), time)) = 0.8
    length_scale(field, gmrf((east, north), time), space) = 0.5
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.0, 0.5)
    y ~ Normal(field, 0.3)
end)

settings(rk) = (; init=zeros(rk.model.layout.total), reference=:pilot)
stop_at(boundary) = (s, b) -> b === boundary

mode, directory = ARGS
checkpoint_dir = joinpath(directory, "checkpoint")
if mode == "write"
    @testset "generated-module law: producer" begin
        rk = model()
        laws = [e.sampling for e in rk.model.layout.entries if e.sampling !== nothing]
        @test !isempty(laws)
        # Positive control: the raw law expression is what a fresh process
        # cannot read back (the consumer asserts the UndefVarError).
        serialize(joinpath(directory, "raw-law.jls"), first(laws).rhs)
        schedule = F.tune((rk,), settings(rk))
        @test schedule isa SMCTuningSchedule
        partial = F.tune((rk,), settings(rk); checkpoint_dir, callback=stop_at(:init))
        @test partial isa SMCTuningPartial
        @test partial.progress.retained_draws == 0
        # A rebuilt model owns another generated module; it is the same target.
        rebuilt = model()
        next = F.tune((rebuilt,), settings(rebuilt); resume=partial,
            callback=stop_at(:window))
        @test next isa SMCTuningPartial
        @test next.progress.boundary === :window
        @test next.progress.evaluations > 0
        @test F.equal(F.tune((rebuilt,), settings(rebuilt); resume=next), schedule)
        serialize(joinpath(directory, "saved.jls"), (; partial, next, schedule))
    end
elseif mode == "read"
    @testset "generated-module law: fresh-process consumer" begin
        # Read everything before this process has built any model module.
        saved = deserialize(joinpath(directory, "saved.jls"))
        @test_throws UndefVarError deserialize(joinpath(directory, "raw-law.jls"))
        rk = model()
        before = F.roundtrip(saved.partial)
        next = F.tune((rk,), settings(rk); rng=Xoshiro(999), resume=saved.partial,
            callback=stop_at(:window))
        @test F.equal(next, saved.next)
        @test F.equal(saved.partial, before)
        @test F.equal(F.tune((rk,), settings(rk); resume=next), saved.schedule)
        disk = F.tune((rk,), settings(rk); rng=Xoshiro(998), checkpoint_dir,
            resume=true)
        @test F.equal(disk, saved.schedule)
        # Changed settings are still refused against the restored identity.
        @test_throws ArgumentError F.tune((rk,), settings(rk);
            resume=saved.partial, pilot_evaluations=33)
    end
else
    error("expected write or read")
end
