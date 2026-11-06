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

# Rebuilds of one model around an anonymous callable: each evaluation of the
# same source has another generated name.
window_shift = nothing
function window_callable_backend(source)
    global window_shift = eval(source)
    RKBRMI(BayesianRegressionModels.brm(quote
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(mu, g) ~ LogNormal(log(0.7), 0.2)
        y ~ Normal(window_shift(mu), 0.8)
    end; mod=@__MODULE__)((; g=repeat(["a", "b", "c"], inner=2),
        y=[0.3, 0.9, -0.5, 0.1, 1.1, 0.7])))
end
# In a cell, where a Laplace problem's partition can evaluate the callable by row.
function window_callable_cell_backend(source)
    global window_shift = eval(source)
    RKBRMI(BayesianRegressionModels.brm(quote
        mu ~ 1 + (1 | subject)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.2)
        @plate for i in eachindex(mu)
            y[i] ~ Normal(window_shift(mu[i]), 0.8)
        end
    end; mod=@__MODULE__)((; subject=["a", "b", "c"],
        y=[[0.3, 0.9], [-0.5, 0.1], [1.1, 0.7]])))
end

@testset "Identities read what a generated callable computes, not what names it" begin
    B = BayesianRegressionModels
    F = Base.get_extension(B, :BayesianRegressionModelsFilterExt)
    a = window_callable_backend(:(eta -> eta .+ 0.1))
    b = window_callable_backend(:(eta -> eta .+ 0.1))
    other = window_callable_backend(:(eta -> eta .+ 0.2))
    # Positive controls: what names it tells the rebuilds apart, also as source text.
    @test B._brm_rk_binding_identity(a, "t"; stable=false) !=
        B._brm_rk_binding_identity(b, "t"; stable=false)
    @test string(brm_source(a)) != string(brm_source(b))
    for identity in (E.binding_identity, rk -> E.streaming_source(rk, nothing),
            rk -> F.source_identity(rk, nothing, nothing), B._brm_svi_source_id)
        @test identity(a) == identity(b)
        @test identity(a) != identity(other)
    end
    frame(rk) = E.laplace_identity(E.laplace_frame(brm_laplace_problem(rk;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), structured=false)))
    cell_a = window_callable_cell_backend(:(eta -> eta .+ 0.1))
    cell_b = window_callable_cell_backend(:(eta -> eta .+ 0.1))
    cell_other = window_callable_cell_backend(:(eta -> eta .+ 0.2))
    @test frame(cell_a) == frame(cell_b)
    @test frame(cell_a) != frame(cell_other)
    # A fitted state replays on the rebuilt declaration, also one exported as old
    # text that holds its generated name.
    state = brm_fitted_state(a)
    @test !occursin("#", replace(state.declaration, "#generated" => ""))
    @test RKBRMI(brm_source(b), state) isa RKBRMI
    @test B._brm_stable_declaration(string(brm_source(a))) == state.declaration
    @test_throws "exported from another model" RKBRMI(
        brm_source(RKBRMI(W.normal((; y=[0.2, -0.1, 0.4])))), state)
    # Without a generated callable each keeps the digest and the text it always had.
    plain = RKBRMI(W.normal((; y=[0.2, -0.1, 0.4])))
    @test B._brm_rk_binding_identity(plain, "t") ==
        B._brm_rk_binding_identity(plain, "t"; stable=false)
    @test B._brm_declaration_text(brm_source(plain)) ==
        sprint(show, brm_source(plain); context=:brm_documentation => false)
    @test B._brm_stable_declaration("y ~ Normal(#3(mu), f(#3#4, #inner#12), 0.8)") ==
        "y ~ Normal(#generated(mu), f(#generated, #generated), 0.8)"
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
