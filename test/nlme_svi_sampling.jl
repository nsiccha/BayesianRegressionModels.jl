# Exact finite-population controls for the production batch sampler. Only
# stdlibs: no model preparation, AD, posterior simulation or environment setup.
using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "nlme_svi_sampling.jl"))

@testset "SVI work probabilities and alias distribution" begin
    for work in ([0.0, 0, 0], [0.0, 1, 1000], [floatmax(Float64), 1, 2], [4.0])
        sampler = _brm_svi_unit_sampler(:work, work)
        p, N = sampler.probabilities, length(work)
        @test sum(p) ≈ 1
        @test all(>=(1 / (2N)), p)
        # Integrate the alias table exactly over its uniform columns and coin.
        actual = zeros(N)
        for i in 1:N
            @test 0 <= sampler.threshold[i] <= 1
            actual[i] += sampler.threshold[i] / N
            actual[sampler.alias[i]] += (1 - sampler.threshold[i]) / N
        end
        @test actual ≈ p
    end
    for work in (Float64[], [1.0, -1], [NaN], [Inf])
        @test_throws ArgumentError _brm_svi_unit_sampler(:work, work)
    end
    @test_throws ArgumentError _brm_svi_unit_sampler(:unknown, [1.0])
end

@testset "SVI weighted sum and gradient expectation, including repeats" begin
    # A different quadratic contribution from each unit, with deliberately
    # unequal costs. Density and gradient have independently known totals.
    centers, precision, g = [-2.0, 0.5, 3.0], [0.2, 4.0, 9.0], 0.7
    values = @. -precision * (g - centers)^2 / 2
    gradients = @. -precision * (g - centers)
    p = _brm_svi_unit_sampler(:work, [1.0, 50, 4]).probabilities
    expected_value, expected_gradient, mass = 0.0, 0.0, 0.0
    for draws in Iterators.product(1:3, 1:3, 1:3)
        probability = prod(p[i] for i in draws)
        batch = _brm_svi_weighted_batch(draws, p)
        @test length(unique(batch.indices)) == length(batch.indices)
        @test sum(batch.multiplicities) == 3
        expected_value += probability * dot(batch.weights, values[batch.indices])
        expected_gradient += probability * dot(batch.weights, gradients[batch.indices])
        mass += probability
    end
    @test mass ≈ 1
    @test expected_value ≈ sum(values)
    @test expected_gradient ≈ sum(gradients)
    repeated = _brm_svi_weighted_batch([2, 2, 2], p)
    @test repeated.indices == [2]
    @test repeated.multiplicities == [3]
    @test repeated.weights ≈ [1 / p[2]]

    # All equally likely size-two subsets, including the old N/B weighting.
    subsets = ([1, 2], [1, 3], [2, 3])
    @test sum(1.5sum(values[i]) for i in subsets) / 3 ≈ sum(values)
    @test sum(1.5sum(gradients[i]) for i in subsets) / 3 ≈ sum(gradients)
end

@testset "SVI sampler execution and bounded batch storage" begin
    uniform = _brm_svi_unit_sampler(:uniform, ones(9))
    for B in (1, 4, 9)
        batch = _brm_svi_batch(Xoshiro(9), uniform, B)
        @test length(unique(batch.indices)) == B
        @test all(i -> 1 <= i <= 9, batch.indices)
        @test batch.weights == fill(9 / B, B)
        @test batch.multiplicities == ones(Int, B)
    end
    sampler = _brm_svi_unit_sampler(:work, [0.0, 1, 20])
    batch = _brm_svi_batch(Xoshiro(14), sampler, 100_000)
    @test batch.indices == [1, 2, 3]
    @test batch.multiplicities ./ 100_000 ≈ sampler.probabilities atol=0.006
    @test _brm_svi_batch(Xoshiro(14), sampler, 32) ==
        _brm_svi_batch(Xoshiro(14), sampler, 32)
    small = _brm_svi_unit_sampler(:work, ones(1000))
    large = _brm_svi_unit_sampler(:work, ones(100_000))
    @test Base.summarysize(large) < 110Base.summarysize(small)
    _brm_svi_batch(Xoshiro(1), small, 16)
    _brm_svi_batch(Xoshiro(1), large, 16)
    a = @allocated _brm_svi_batch(Xoshiro(1), small, 16)
    b = @allocated _brm_svi_batch(Xoshiro(1), large, 16)
    @test b <= a + 2048
end
