# A read-pair field at unfitted points, against the exact Gaussian conditional from an
# independent Bessel implementation. Plain Julia: no backend.
using Test, LinearAlgebra, Random, Statistics
using SpecialFunctions: besselk

module VecchiaPredictFixture
using LinearAlgebra, Random, SparseArrays
import SpecialFunctions
besselk0(x) = SpecialFunctions.besselk(0, x)
besselk1(x) = SpecialFunctions.besselk(1, x)
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "gmrf_precision.jl"))
include(joinpath(SRC, "gmrf_vecchia.jl"))
include(joinpath(SRC, "gmrf_vecchia_predict.jl"))
end
const VP = VecchiaPredictFixture

exact_matern(r, ℓ) = r == 0 ? 1.0 : (sqrt(2) / ℓ * r) * besselk(1, sqrt(2) / ℓ * r)
exact_cov(x1, y1, t1, x2, y2, t2, ℓs, ℓt) = [exact_matern(hypot(x1[i] - x2[j], y1[i] - y2[j]), ℓs) *
    exp(-abs(t1[i] - t2[j]) / ℓt) for i in eachindex(x1), j in eachindex(x2)]

state_times(f) = [VP._brm_vecchia_state_time(f, s) for s in eachindex(f.position)]

@testset "with every state in the set, a new point follows the exact conditional" begin
    rng = Xoshiro(1)
    n = 30
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    f = VP._brm_vecchia_footprint(east, north, zeros(n), 1; neighbors=100)
    ℓs, ℓt = 3.0, 5.0
    x, y, t = f.east[f.position], f.north[f.position], state_times(f)
    K = exact_cov(x, y, t, x, y, t, ℓs, ℓt)
    state = cholesky(Symmetric(K + 1e-10I)).L * randn(rng, n)
    px, py, pt = 10 .* rand(rng, 12), 10 .* rand(rng, 12), zeros(12)
    p = VP._brm_vecchia_predictor(f, px, py, pt; neighbors=n)
    mean_out = zeros(12)
    VP._brm_vecchia_predict!(mean_out, p, state, ℓs, ℓt, ones(1, 1), f)
    k = exact_cov(px, py, pt, x, y, t, ℓs, ℓt)
    oracle = k * (cholesky(Symmetric(K + 1e-10I)) \ state)
    @test mean_out ≈ oracle rtol=1e-6
    # The draw's spread is the exact conditional variance, by Monte Carlo at one point.
    variance = 1 .- diag(k * (cholesky(Symmetric(K + 1e-10I)) \ k'))
    draws = zeros(4000, 12)
    for d in 1:4000
        out = zeros(12)
        VP._brm_vecchia_predict!(out, p, state, ℓs, ℓt, ones(1, 1), f; rng)
        draws[d, :] .= out
    end
    @test vec(mean(draws; dims=1)) ≈ oracle atol=0.05
    @test vec(var(draws; dims=1)) ≈ variance rtol=0.15
end

@testset "a point on a read state returns that state exactly" begin
    rng = Xoshiro(2)
    east, north = 10 .* rand(rng, 9), 10 .* rand(rng, 9)
    time = repeat([0.0, 2.0]; inner=9)
    f = VP._brm_vecchia_footprint(repeat(east, 2), repeat(north, 2), time, 1; neighbors=4)
    state = randn(rng, f.ncoordinates)
    s = 7
    px, py, pt = [f.east[f.position[s]]], [f.north[f.position[s]]], [VP._brm_vecchia_state_time(f, s)]
    p = VP._brm_vecchia_predictor(f, px, py, pt)
    @test p.exact == [s]
    out = zeros(1)
    VP._brm_vecchia_predict!(out, p, state, 2.0, 3.0, ones(1, 1), f; rng)
    @test out == [state[s]]
end

@testset "a point between read times uses both, with the exact space-time conditional" begin
    rng = Xoshiro(3)
    npos = 8
    east, north = 10 .* rand(rng, npos), 10 .* rand(rng, npos)
    time = repeat([0.0, 4.0]; inner=npos)
    f = VP._brm_vecchia_footprint(repeat(east, 2), repeat(north, 2), time, 1; neighbors=100)
    ℓs, ℓt = 4.0, 6.0
    x, y, t = f.east[f.position], f.north[f.position], state_times(f)
    K = exact_cov(x, y, t, x, y, t, ℓs, ℓt)
    state = cholesky(Symmetric(K + 1e-10I)).L * randn(rng, length(x))
    px, py, pt = [5.0, 1.0], [5.0, 9.0], [1.5, 4.0]
    p = VP._brm_vecchia_predictor(f, px, py, pt; neighbors=npos)
    @test p.ptr[2] - p.ptr[1] == 2npos       # both read times around t = 1.5
    @test all(s -> state_times(f)[s] in (0.0, 4.0), p.states)
    out = zeros(2)
    VP._brm_vecchia_predict!(out, p, state, ℓs, ℓt, ones(1, 1), f)
    # Point 1 sees every state. Point 2 is at a read time and also conditions on both
    # times here, because it sits at the last read time and nothing lies after it.
    k = exact_cov(px, py, pt, x, y, t, ℓs, ℓt)
    @test out[1] ≈ dot(k[1, :], cholesky(Symmetric(K + 1e-10I)) \ state) rtol=1e-5
    # Before the first and after the last read time stay well defined.
    q = VP._brm_vecchia_predictor(f, [5.0, 5.0], [5.0, 5.0], [-3.0, 9.0]; neighbors=3)
    far = zeros(2)
    VP._brm_vecchia_predict!(far, q, state, ℓs, ℓt, ones(1, 1), f)
    @test all(isfinite, far)
end

@testset "channels follow their factor, and an unknown channel is refused" begin
    rng = Xoshiro(4)
    n = 12
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    f = VP._brm_vecchia_footprint(east, north, zeros(n), 2; neighbors=100)
    L = [1.0 0.0; 0.6 0.8]
    white = randn(rng, 2n)
    state = copy(white)
    for s in 1:n
        state[2 + 2(s - 1)] = L[2, 1] * white[1 + 2(s - 1)] + L[2, 2] * white[2 + 2(s - 1)]
    end
    p = VP._brm_vecchia_predictor(f, [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]; neighbors=n)
    out = zeros(2)
    VP._brm_vecchia_predict!(out, p, state, 3.0, 5.0, L, f; channels=[1, 2])
    @test out[2] ≈ L[2, 1] * out[1] + L[2, 2] * begin
        one_channel = zeros(2)
        VP._brm_vecchia_predict!(one_channel, p, white[2:2:end] |> w -> begin
            x = zeros(2n); x[1:2:end] .= w; x
        end, 3.0, 5.0, Matrix{Float64}(I, 2, 2), f; channels=[1, 1])
        one_channel[1]
    end rtol=1e-8
    # The draw's channel covariance is F L Lᵀ.
    draws = zeros(4000, 2)
    q = VP._brm_vecchia_predictor(f, [5.0], [5.0], [0.0]; neighbors=n)
    for d in 1:4000, c in 1:2
        o = zeros(1)
        # One conditional per channel call, so only each channel's own marginal is checked.
        VP._brm_vecchia_predict!(o, q, state, 3.0, 5.0, L, f; channels=[c], rng)
        draws[d, c] = o[1]
    end
    @test abs(var(draws[:, 1]) - var(draws[:, 2])) / var(draws[:, 1]) < 0.2
    @test_throws ArgumentError VP._brm_vecchia_predict!(zeros(1), q, state, 3.0, 5.0, L, f; channels=[3])
end

@testset "a footprint equals a copy of itself and differs from another" begin
    east, north = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0]
    f = VP._brm_vecchia_footprint(east, north, zeros(4), 1; neighbors=2)
    g = deepcopy(f)
    @test isequal(f, g) && f == g && hash(f) == hash(g)
    h = VP._brm_vecchia_footprint(east .+ 1, north, zeros(4), 1; neighbors=2)
    @test !isequal(f, h) && f != h
end

@testset "refusals name what is wrong" begin
    f = VP._brm_vecchia_footprint([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], zeros(3), 1; neighbors=2)
    @test_throws DimensionMismatch VP._brm_vecchia_predictor(f, [0.0], [0.0, 1.0], [0.0])
    @test_throws ArgumentError VP._brm_vecchia_predictor(f, [NaN], [0.0], [0.0])
    @test_throws ArgumentError VP._brm_vecchia_predictor(f, [0.0], [0.0], [0.0]; neighbors=0)
    p = VP._brm_vecchia_predictor(f, [0.5], [0.5], [0.0])
    @test_throws DomainError VP._brm_vecchia_predict!(zeros(1), p, [0.0, 0.0, 0.0], -1.0, 1.0, ones(1, 1), f)
    @test_throws DimensionMismatch VP._brm_vecchia_predict!(zeros(2), p, [0.0, 0.0, 0.0], 1.0, 1.0, ones(1, 1), f)
end
