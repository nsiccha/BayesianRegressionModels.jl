# A read-pair prior split by owner: the owners' pieces are the whole density and gradient,
# and each piece touches only its conditionals. Plain Julia: no backend.
using Test, LinearAlgebra, Random

module VecchiaLocalFixture
using LinearAlgebra, Random, SparseArrays
import SpecialFunctions
besselk0(x) = SpecialFunctions.besselk(0, x)
besselk1(x) = SpecialFunctions.besselk(1, x)
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "gmrf_precision.jl"))
include(joinpath(SRC, "gmrf_vecchia.jl"))
include(joinpath(SRC, "gmrf_vecchia_local.jl"))
end
const VL = VecchiaLocalFixture

function split_field(; npos=14, times=[0.0, 1.0, 3.0], channels=1, neighbors=4, units=4, global_share=0.1, seed=1)
    rng = Xoshiro(seed)
    east, north = 10 .* rand(rng, npos), 10 .* rand(rng, npos)
    f = VL._brm_vecchia_footprint(repeat(east; outer=length(times)), repeat(north; outer=length(times)),
        repeat(times; inner=npos), channels; neighbors)
    n = length(f.position)
    # A station owns its position's states, as a unit owns its reads.
    position_owner = rand(rng, 1:units, length(f.east))
    owner = [rand(rng) < global_share ? 0 : position_owner[f.position[s]] for s in 1:n]
    L = channels == 1 ? ones(1, 1) : Matrix(cholesky(0.4 .* ones(channels, channels) + 0.6I).L)
    (; f, owner, L, x=randn(rng, f.ncoordinates), units)
end

coordinates(f, states) = [c + f.nchannels * (s - 1) for s in states for c in 1:f.nchannels]

@testset "the footprint's own gradient is unchanged by the shared contraction" begin
    fx = split_field(; channels=2)
    whole = VL._brm_vecchia_gradient(fx.x, 2.5, 3.0, fx.L, fx.f)
    @test whole.value ≈ VL._brm_vecchia_logdensity(fx.x, 2.5, 3.0, fx.L, fx.f) rtol=1e-12
    h = 1e-6
    numeric = (VL._brm_vecchia_logdensity(fx.x, 2.5 + h, 3.0, fx.L, fx.f) -
        VL._brm_vecchia_logdensity(fx.x, 2.5 - h, 3.0, fx.L, fx.f)) / 2h
    @test whole.spatial_length ≈ numeric rtol=1e-5
    numeric = (VL._brm_vecchia_logdensity(fx.x, 2.5, 3.0 + h, fx.L, fx.f) -
        VL._brm_vecchia_logdensity(fx.x, 2.5, 3.0 - h, fx.L, fx.f)) / 2h
    @test whole.temporal_length ≈ numeric rtol=1e-5
end

@testset "owners' pieces are the whole, channels $channels" for channels in (1, 3)
    fx = split_field(; channels)
    f = fx.f
    ℓs, ℓt = 2.5, 3.0
    whole = VL._brm_vecchia_gradient(fx.x, ℓs, ℓt, fx.L, f)
    plans = VL._brm_vecchia_local_plans(f, fx.owner, fx.units)
    work = VL._BRMVecchiaLocalWork(f)
    value, dℓs, dℓt = 0.0, 0.0, 0.0
    dL = zeros(channels, channels)
    summed = zeros(f.ncoordinates)
    n = length(f.position)
    for (i, plan) in enumerate(plans)
        part = VL._brm_vecchia_local!(work, plan, f, fx.x, ℓs, ℓt, fx.L)
        value += part.value
        dℓs += part.spatial_length
        dℓt += part.temporal_length
        dL .+= part.L
        for j in coordinates(f, plan.states)
            summed[j] += work.gown[j]
        end
        # On a unit's own states the piece's whole gradient is the density's.
        if i > 1
            for j in coordinates(f, plan.owned)
                @test work.gall[j] ≈ whole.field[j] atol=1e-9 rtol=1e-9
            end
        end
        VL._brm_vecchia_local_clear!(work, plan, f)
    end
    @test value ≈ whole.value rtol=1e-11
    @test dℓs ≈ whole.spatial_length rtol=1e-9
    @test dℓt ≈ whole.temporal_length rtol=1e-9
    @test dL ≈ whole.L rtol=1e-9
    # Own-conditional gradients add up to the whole on every state, global ones included.
    @test summed ≈ whole.field rtol=1e-9
    # Clearing leaves the scratch as it was: a second owner never sees the first's values.
    @test all(iszero, work.own) && all(iszero, work.all) && all(iszero, work.gown) && all(iszero, work.gall)
end

@testset "a piece reads only what its plan names" begin
    fx = split_field(; units=6, npos=30, neighbors=3)
    f = fx.f
    plans = VL._brm_vecchia_local_plans(f, fx.owner, fx.units)
    work = VL._BRMVecchiaLocalWork(f)
    plan = plans[2]
    @test length(plan.states) < length(f.position)
    # NaN everywhere else: touching an unnamed state would poison the result.
    poisoned = fill(NaN, f.ncoordinates)
    for j in coordinates(f, plan.states)
        poisoned[j] = fx.x[j]
    end
    a = VL._brm_vecchia_local!(work, plan, f, poisoned, 2.0, 3.0, fx.L)
    VL._brm_vecchia_local_clear!(work, plan, f)
    b = VL._brm_vecchia_local!(work, plan, f, fx.x, 2.0, 3.0, fx.L)
    @test isfinite(a.value) && a.value == b.value
    @test a.spatial_length == b.spatial_length && a.temporal_length == b.temporal_length
    # Each state's conditional is owned exactly once, and each plan lists no state twice.
    @test sort!(reduce(vcat, [p.owned for p in plans])) == collect(1:length(f.position))
    @test all(p -> allunique(p.states) && allunique(p.other) && isempty(intersect(p.owned, p.other)), plans)
end

@testset "storage follows the conditionals, not units × states" begin
    small = split_field(; units=4, npos=20, global_share=0.0)
    large = split_field(; units=40, npos=20, global_share=0.0)
    entries(plans) = sum(p -> length(p.states) + length(p.own) + length(p.other), plans)
    @test entries(VL._brm_vecchia_local_plans(large.f, large.owner, large.units)) <
        3 * entries(VL._brm_vecchia_local_plans(small.f, small.owner, small.units))
    @test_throws DimensionMismatch VL._brm_vecchia_local_plans(small.f, [1, 2], 4)
    @test_throws ArgumentError VL._brm_vecchia_local_plans(small.f, fill(9, length(small.f.position)), 4)
end
