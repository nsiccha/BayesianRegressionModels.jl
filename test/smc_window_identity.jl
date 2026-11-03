using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization
include("smc_window_fixture.jl")
const W = SMCWindowFixture
const E = Base.get_extension(BayesianRegressionModels, :BayesianRegressionModelsSMCExt)

struct PaddedWindowRecord
    active::Bool
    index::Int64
end

function serialized_bytes(value)
    io = IOBuffer()
    serialize(io, value)
    take!(io)
end

@testset "Streaming identities describe values, not storage" begin
    a = [PaddedWindowRecord(true, 7)]
    b = deepcopy(a)
    # Set only unused padding, making the old byte hash fail deterministically
    # without depending on which bytes the allocator happened to leave there.
    @test fieldoffset(PaddedWindowRecord, 2) > sizeof(Bool)
    reinterpret(UInt8, a)[2] = 0x31
    reinterpret(UInt8, b)[2] = 0x32
    @test a == b
    @test serialized_bytes(a) != serialized_bytes(b)
    @test E.streaming_identity(a) == E.streaming_identity(b)
    @test E.streaming_identity(a) == E.streaming_identity(W.roundtrip(a))
    b[1] = PaddedWindowRecord(false, 7)
    @test E.streaming_identity(a) != E.streaming_identity(b)

    values = [0.25, -0.75]
    shared = (; first=values, second=values)
    independent = (; first=copy(values), second=copy(values))
    @test serialized_bytes(shared) != serialized_bytes(independent)
    @test E.streaming_identity(shared) == E.streaming_identity(independent)
    left = Dict(:a => shared, :b => independent)
    right = Dict(:b => independent, :a => shared)
    @test E.streaming_identity(left) == E.streaming_identity(right)
    @test E.streaming_identity(left) == E.streaming_identity(deepcopy(left))
    # Layout aliases exposed the failure: deepcopy rebuilds a Dict,
    # changing its table capacity/order without changing any key or value.
    aliases = Dict{Symbol,Tuple{Vararg{Symbol}}}()
    sizehint!(aliases, 1024)
    for i in 1:32
        aliases[Symbol("coordinate_$i")] = (:group, :level)
    end
    rebuilt = deepcopy(aliases)
    @test aliases == rebuilt
    @test serialized_bytes(aliases) != serialized_bytes(rebuilt)
    @test E.streaming_identity(aliases) == E.streaming_identity(rebuilt)
    rebuilt[:coordinate_1] = (:group, :different_level)
    @test E.streaming_identity(aliases) != E.streaming_identity(rebuilt)
    @test E.streaming_identity(values) != E.streaming_identity(reshape(values, 1, 2))
    @test E.streaming_identity(values) != E.streaming_identity(Float32.(values))
    @test E.streaming_identity(0.0) != E.streaming_identity(-0.0)
    independent.first[1] += 1
    @test E.streaming_identity(shared) != E.streaming_identity(independent)
end

@testset "Copied carries preserve update, tuning and checkpoint validation" begin
    training = (; x=[-2.0, 0.0, 3.0], y=[-0.2, 0.1, 0.8])
    window = (; x=[-1.5, 0.5, 1.2, 3.0], y=[-0.1, 0.5, 0.7, 1.0])
    options = SMCOptions(; moves=1, stepsize=0.3, max_tree_depth=2)
    fit = brm_smc_fit(W.scaled(training); rng=Xoshiro(141), particles=32, options)
    for basis in (nothing, :fit)
        carried = brm_smc_carry(fit; basis)
        saved = W.roundtrip(carried)
        for copy in (deepcopy(carried), W.roundtrip(carried))
            partial = brm_smc_update(copy, W.scaled(window);
                rng=Xoshiro(142), particles=16, options, stages=0)
            @test partial.state.temperature == 0
            @test W.equal(carried, saved)
            checkpoint = W.roundtrip(brm_smc_checkpoint(partial))
            restored = brm_smc_resume(W.scaled(window), checkpoint)
            @test W.equal(brm_smc_checkpoint(restored), checkpoint)
            rng, init = Xoshiro(143), zeros(1)
            rng_before = Base.copy(rng)
            schedule = brm_smc_tune(copy, W.scaled(window); rng, init,
                temperatures=(0.0, 1.0), pilot_steps=2, options)
            @test schedule !== nothing
            @test rng == rng_before
            @test init == zeros(1)
            @test W.equal(carried, saved)
            # A fit owns its carry; a later caller edit cannot change it.
            copy.posterior.reference.location[1] += 1
            @test W.equal(brm_smc_checkpoint(partial), checkpoint)
            @test_throws r"carried posterior" brm_smc_update(copy, W.scaled(window); stages=0)
            @test_throws r"carried posterior" brm_smc_tune(copy, W.scaled(window); rng, init)
            checkpoint.previous.posterior.reference.location[1] += 1
            @test_throws r"carried posterior" brm_smc_resume(W.scaled(window), checkpoint)
        end
        changed = deepcopy(carried)
        changed.preprocessing.constants[:invalid_anchor] = 0.0
        @test_throws r"frozen preprocessing" brm_smc_update(changed, W.scaled(window); stages=0)
    end
end
