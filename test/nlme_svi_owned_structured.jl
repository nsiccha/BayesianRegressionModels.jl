using Test, Random, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, LinearAlgebra
include("nlme_svi_owned_fixture.jl")

@testset "absent ordinary coordinates retain structured carried uncertainty" begin
    initial = brm_svi_fit(svi_owned_model(svi_owned_data()); rng=Xoshiro(104),
        unit_dense_limit=0, unit_rank=1, max_iterations=0, ndraws=0, threads=1)
    state = brm_svi_checkpoint(initial)
    for label in ("s1", "s3", "s5")
        factor = state.units[label]
        @test factor.family === :diagonal_lowrank
        factor.M .= [0.3, 1.7, -0.8]
        factor.ρ .= log(0.2)
        factor.Β .= [0.0, 0.4, -0.3]
    end
    next_data = merge(svi_owned_data([6, 2, 4]), (; ncoeff=12))
    fit = brm_svi_fit(state, svi_owned_model(next_data); rng=Xoshiro(105),
        max_iterations=0, ndraws=4000, threads=1)
    positions = Dict(name => i for (i, name) in enumerate(fit.coordinates))
    for label in ("s1", "s3", "s5")
        old = state.units[label]
        source = findall(key -> hasproperty(key, :coordinate), old.keys)
        destination = [positions[old.keys[k].coordinate] for k in source]
        samples = fit.draws[destination, :]
        mean = vec(sum(samples; dims=2)) / size(samples, 2)
        centered = samples .- mean
        covariance = centered * centered' / (size(samples, 2)-1)
        expected = Matrix(Diagonal(exp.(2old.ρ[source]))) + old.Β[source, :] * old.Β[source, :]'
        @test mean ≈ old.M[source] atol=0.03
        @test covariance ≈ expected atol=0.025
        @test brm_svi_checkpoint(fit).units[label] == old
    end
end
