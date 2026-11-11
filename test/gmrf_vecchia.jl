# The footprint (nearest-neighbour) representation of a `gmrf` field: exact-Gaussian
# oracles for its numerical core and its data-only structure. Plain Julia: no backend.
using Test, BayesianRegressionModels, LinearAlgebra, Random
using SpecialFunctions: besselk
const BRM = BayesianRegressionModels

vecchia_state_time(f) = reduce(vcat,
    [fill(f.times[t], f.time_ptr[t + 1] - f.time_ptr[t]) for t in eachindex(f.times)])

# The specified covariance, from an independent Bessel implementation.
vecchia_exact(x, y, s, t, spatial_length, temporal_length) = [
    (r = hypot(x[i] - x[j], y[i] - y[j]);
     (r == 0 ? 1.0 : (sqrt(2) / spatial_length * r) * besselk(1, sqrt(2) / spatial_length * r)) *
        exp(-abs(s[i] - s[j]) / temporal_length))
    for i in eachindex(x), j in eachindex(x)]

function vecchia_exact(f, spatial_length, temporal_length)
    vecchia_exact(f.east[f.position], f.north[f.position], vecchia_state_time(f), nothing,
        spatial_length, temporal_length)
end
vecchia_exact(x, y, s, ::Nothing, spatial_length, temporal_length) =
    vecchia_exact(x, y, s, s, spatial_length, temporal_length)

function vecchia_implied(f, spatial_length, temporal_length)
    n = length(f.position)
    factors = BRM._brm_vecchia_factors(f, spatial_length, temporal_length)
    C, column, work = zeros(n, n), zeros(n), zeros(n)
    for j in 1:n
        BRM._brm_vecchia_covariance_column!(column, work, factors, f, j)
        C[:, j] .= column
    end
    C
end

gaussian_logdensity(x, K) = -(dot(x, K \ x) + logdet(Symmetric(K)) + length(x) * log(2pi)) / 2

@testset "One time with every neighbour is the exact spatial Gaussian" begin
    rng = MersenneTwister(1)
    east, north = 10 .* rand(rng, 25), 10 .* rand(rng, 25)
    f = BRM._brm_vecchia_footprint(east, north, zeros(25), 1; neighbors=100)
    K = vecchia_exact(f, 2.3, 1.7)
    x = cholesky(Symmetric(K)).L * randn(rng, 25)
    @test BRM._brm_vecchia_logdensity(x, 2.3, 1.7, ones(1, 1), f) ≈ gaussian_logdensity(x, K) atol=1e-10
    @test vecchia_implied(f, 2.3, 1.7) ≈ K atol=1e-10
end

@testset "A full position × time grid with irregular gaps is exact at every neighbour count" begin
    rng = MersenneTwister(2)
    n = 18
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    times = [0.0, 1.0, 2.0, 5.0, 6.0, 78.0]
    E, N, T = repeat(east; outer=6), repeat(north; outer=6), repeat(times; inner=n)
    exact = BRM._brm_vecchia_footprint(E, N, T, 1; neighbors=100)
    K = vecchia_exact(exact, 2.3, 1.7)
    x = cholesky(Symmetric(K)).L * randn(rng, length(E))
    @test BRM._brm_vecchia_logdensity(x, 2.3, 1.7, ones(1, 1), exact) ≈ gaussian_logdensity(x, K) atol=1e-9
    @test vecchia_implied(exact, 2.3, 1.7) ≈ K atol=1e-10
    gradient = BRM._brm_vecchia_gradient(x, 2.3, 1.7, ones(1, 1), exact)
    @test gradient.field ≈ -(K \ x) atol=1e-8
    # The conditioning set approximates only space: the time factor stays exact.
    few = BRM._brm_vecchia_footprint(E, N, T, 1; neighbors=3)
    implied = vecchia_implied(few, 2.3, 1.7)
    s = vecchia_state_time(few)
    p = few.position
    for j in eachindex(s), i in eachindex(s)
        reference = findfirst(k -> p[k] == p[i] && s[k] == s[j], eachindex(s))
        @test implied[i, j] ≈ implied[reference, j] * exp(-abs(s[i] - s[j]) / 1.7) atol=1e-12
    end
    # Its marginal variance at a read stays close to one.
    @test maximum(abs, diag(implied) .- 1) < 0.05
end

@testset "Repeated geometry shares one solve" begin
    rng = MersenneTwister(3)
    n = 12
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    times = collect(0.0:9.0)
    f = BRM._brm_vecchia_footprint(repeat(east; outer=10), repeat(north; outer=10),
        repeat(times; inner=n), 1; neighbors=4)
    # The first time has no previous read; every later time repeats one geometry per position.
    @test length(f.position) == 120
    @test length(f.pattern_ptr) - 1 == 2n
    @test sort(f.pattern_states) == 1:120
    @test all(issorted(view(f.neighbor, f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1))) for s in 1:120)
    @test all(f.neighbor[q] < s for s in 1:120 for q in f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1))
end

@testset "A long gap never cuts time: a late state conditions on the previous read" begin
    east, north = [0.0, 1.0, 40.0], [0.0, 0.0, 0.0]
    time = [0.0, 5000.0, 5100.0]
    f = BRM._brm_vecchia_footprint(east, north, time, 1; neighbors=3)
    # With few reads per time the look-back reaches all the way: nothing is dropped.
    @test f.neighbor_ptr[3] - f.neighbor_ptr[2] == 1
    @test f.neighbor_ptr[4] - f.neighbor_ptr[3] == 2
    implied = vecchia_implied(f, 30.0, 400.0)
    expected = vecchia_exact(f, 30.0, 400.0)
    @test implied ≈ expected atol=1e-12
    # With one neighbour a chain still links every read through the previous one.
    chain = BRM._brm_vecchia_footprint(east, north, time, 1; neighbors=1)
    @test all(chain.neighbor_ptr[s + 1] - chain.neighbor_ptr[s] == 1 for s in 2:3)
    @test vecchia_implied(chain, 30.0, 400.0)[1, 3] != 0
end

@testset "Reads map to states and channels, and unread pairs are refused" begin
    east, north = [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]
    time, channel = [0.0, 0.0, 1.0, 1.0], ["b", "b", "a", "a"]
    f = BRM._brm_vecchia_footprint(east, north, time, 2; neighbors=2)
    @test f.ncoordinates == 8
    index = BRM._brm_vecchia_field_index(f, ["a", "b"], east, north, time, channel)
    @test allunique(index)
    for (i, j) in enumerate(index)
        coordinate = BRM._brm_vecchia_coordinate(f, ["a", "b"], j)
        @test (coordinate.east, coordinate.north, coordinate.time, coordinate.channel) ==
            (east[i], north[i], time[i], channel[i])
    end
    @test BRM._brm_vecchia_field_index(f, ["a", "b"], east[[3, 1]], north[[3, 1]], time[[3, 1]],
        channel[[3, 1]]) == index[[3, 1]]
    # Rows repeating a read share its state.
    repeated = BRM._brm_vecchia_footprint(vcat(east, east[1]), vcat(north, north[1]), vcat(time, time[1]), 2; neighbors=2)
    @test repeated.ncoordinates == f.ncoordinates
    @test BRM._brm_vecchia_field_index(repeated, ["a", "b"], east[[1, 1]], north[[1, 1]], time[[1, 1]],
        channel[[1, 1]]) == fill(index[1], 2)
    @test_throws ArgumentError BRM._brm_vecchia_field_index(f, ["a", "b"], [0.5], [0.0], [0.0], ["a"])
    @test_throws ArgumentError BRM._brm_vecchia_field_index(f, ["a", "b"], [0.0], [0.0], [2.0], ["a"])
    @test_throws ArgumentError BRM._brm_vecchia_field_index(f, ["a", "b"], [0.0], [0.0], [0.0], ["z"])
end

@testset "Reverse derivative against finite differences, channels included" begin
    rng = MersenneTwister(4)
    n = 40
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    time = Float64.(rand(rng, 1:5, n))
    east[2], north[2], time[2] = east[1], north[1], time[1] + 7.0
    f = BRM._brm_vecchia_footprint(east, north, time, 2; neighbors=4)
    x = 0.3 .* randn(rng, f.ncoordinates)
    L = [1.0 0.0; 0.6 0.8]
    density(x, s, t, L) = BRM._brm_vecchia_logdensity(x, s, t, L, f)
    g = BRM._brm_vecchia_gradient(x, 2.3, 1.7, L, f)
    h = 1e-6
    @test g.value ≈ density(x, 2.3, 1.7, L) atol=1e-10
    @test g.spatial_length ≈ (density(x, 2.3 + h, 1.7, L) - density(x, 2.3 - h, 1.7, L)) / 2h atol=1e-4
    @test g.temporal_length ≈ (density(x, 2.3, 1.7 + h, L) - density(x, 2.3, 1.7 - h, L)) / 2h atol=1e-4
    for k in rand(rng, 1:f.ncoordinates, 12)
        e = zeros(f.ncoordinates)
        e[k] = h
        @test g.field[k] ≈ (density(x .+ e, 2.3, 1.7, L) - density(x .- e, 2.3, 1.7, L)) / 2h atol=1e-5
    end
    for (c, k) in ((1, 1), (2, 1), (2, 2))
        d = zeros(2, 2)
        d[c, k] = h
        @test g.L[c, k] ≈ (density(x, 2.3, 1.7, L .+ d) - density(x, 2.3, 1.7, L .- d)) / 2h atol=1e-5
    end
    @test g.L[1, 2] == 0
end

@testset "Channels are exactly separable" begin
    rng = MersenneTwister(5)
    n = 14
    east, north = 10 .* rand(rng, n), 10 .* rand(rng, n)
    f = BRM._brm_vecchia_footprint(east, north, zeros(n), 2; neighbors=100)
    K = vecchia_exact(f, 2.3, 1.7)
    Omega = [1.0 0.6; 0.6 1.0]
    L = Matrix(cholesky(Omega).L)
    x = 0.4 .* randn(rng, f.ncoordinates)
    # Coordinates hold the channels of a state together, so the joint is K ⊗ Ω.
    @test BRM._brm_vecchia_logdensity(x, 2.3, 1.7, L, f) ≈ gaussian_logdensity(x, kron(K, Omega)) atol=1e-9
end

@testset "The same reads in another row order give the same footprint" begin
    same_footprint(a, b) = all(name -> getfield(a, name) == getfield(b, name),
        fieldnames(BRM._BRMGMRFFootprint))
    # A regular lattice: four positions are nearest the centroid and most maximin
    # steps have several farthest positions.
    grid = [(Float64(i), Float64(j)) for i in 1:12 for j in 1:12]
    east, north = first.(grid), last.(grid)
    p = randperm(MersenneTwister(1), length(east))
    physical_order(x, y) = (o = BRM._brm_vecchia_position_order(x, y); collect(zip(x[o], y[o])))
    # These ties do resolve by list index, so the lattice tests what it claims to.
    @test physical_order(east, north) != physical_order(east[p], north[p])
    @test same_footprint(BRM._brm_vecchia_footprint(east, north, zeros(144); neighbors=4),
        BRM._brm_vecchia_footprint(east[p], north[p], zeros(144); neighbors=4))
    # Intermittent reads, two channels, and a read pair several rows repeat.
    rng = MersenneTwister(2)
    rows = [(i, j, t) for i in 1:10 for j in 1:10 for t in (0.0, 1.0, 2.0, 80.0) if rand(rng) < 0.6]
    rows = vcat(rows, rows[1:20])
    x, y, t = Float64[r[1] for r in rows], Float64[r[2] for r in rows], Float64[r[3] for r in rows]
    f = BRM._brm_vecchia_footprint(x, y, t, 2; neighbors=6)
    for seed in 1:3
        q = randperm(MersenneTwister(seed), length(x))
        @test same_footprint(f, BRM._brm_vecchia_footprint(x[q], y[q], t[q], 2; neighbors=6))
    end
    # Without ties the order is the positions' own, however the rows list them.
    rng = MersenneTwister(3)
    ux, uy = 100 .* rand(rng, 60), 100 .* rand(rng, 60)
    q = randperm(rng, 60)
    @test physical_order(ux, uy) == physical_order(ux[q], uy[q])
end

@testset "Outside the support the density is -Inf, never an error" begin
    # An optimizer's line search proposed these lengths from a prior-median start
    # (snag gmrf-neighbors-d-952fd2fb): the conditioning correlations round to one.
    rng = Xoshiro(20261110)
    xy = [(x + 5rand(rng), y + 5rand(rng)) for y in range(10, 85; length=4) for x in range(10, 85; length=6)]
    rows = [(x, y, t) for (x, y) in xy for t in (1.0, 3.0, 6.0) for _ in 1:2]
    f = BRM._brm_vecchia_footprint(getindex.(rows, 1), getindex.(rows, 2), getindex.(rows, 3), 1;
        neighbors=8)
    x = randn(rng, f.ncoordinates)
    outside(s, t) = (BRM._brm_vecchia_logdensity(x, s, t, ones(1, 1), f),
        BRM._brm_vecchia_gradient(x, s, t, ones(1, 1), f))
    for (s, t) in ((8.877173681259308e64, 8.260477390538313e19), (8.9e64, 4.0), (25.0, 1e300),
            (Inf, 4.0), (25.0, Inf), (0.0, 4.0), (25.0, 0.0))
        value, gradient = outside(s, t)
        @test value == -Inf
        @test gradient.value == -Inf
        @test length(gradient.field) == f.ncoordinates && all(iszero, gradient.field)
        @test gradient.spatial_length == 0 && gradient.temporal_length == 0 && all(iszero, gradient.L)
    end
    # Prediction's strict solve still refuses the singular system.
    @test_throws DomainError BRM._brm_vecchia_factors(f, 8.877173681259308e64, 8.260477390538313e19)
    # A negative or NaN length is no transform's output: still an error.
    for (s, t) in ((-1.0, 4.0), (25.0, -1.0), (NaN, 4.0), (25.0, NaN))
        @test_throws DomainError BRM._brm_vecchia_logdensity(x, s, t, ones(1, 1), f)
        @test_throws DomainError BRM._brm_vecchia_gradient(x, s, t, ones(1, 1), f)
    end
    # Long lengths the system still resolves keep their finite density and derivative.
    for (s, t) in ((1e3, 4.0), (1e5, 4.0), (25.0, 1e6))
        value, gradient = outside(s, t)
        @test isfinite(value) && gradient.value ≈ value
        @test all(isfinite, gradient.field) && isfinite(gradient.spatial_length)
    end
end

@testset "Unsupported inputs are refused with their reason" begin
    @test_throws ArgumentError BRM._brm_vecchia_footprint(Float64[], Float64[], Float64[], 1; neighbors=2)
    @test_throws ArgumentError BRM._brm_vecchia_footprint([1.0], [1.0], [1.0], 1; neighbors=0)
    @test_throws ArgumentError BRM._brm_vecchia_footprint([1.0], [1.0], [1.0], 1; neighbors=101)
    @test_throws ArgumentError BRM._brm_vecchia_footprint([NaN], [1.0], [1.0], 1; neighbors=2)
    @test_throws DimensionMismatch BRM._brm_vecchia_footprint([1.0, 2.0], [1.0], [1.0], 1; neighbors=2)
    f = BRM._brm_vecchia_footprint([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], 1; neighbors=2)
    @test_throws DimensionMismatch BRM._brm_vecchia_logdensity(zeros(3), 1.0, 1.0, ones(1, 1), f)
    @test_throws DomainError BRM._brm_vecchia_logdensity(zeros(2), -1.0, 1.0, ones(1, 1), f)
end
