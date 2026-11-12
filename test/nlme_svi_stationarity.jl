# Statistical and storage controls for the iterate gate; no model compilation.
# RUN: julia --project=test test/nlme_svi_stationarity.jl [testset-filter...]
using Test, BayesianRegressionModels, Random, Statistics, Serialization, LinearAlgebra
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels

function svi_iterate_history(xs)
    h = BRM._BRMSVIIterates(size(xs, 1))
    for x in eachcol(xs)
        slot = BRM._brm_svi_iterate_slot!(h)
        slot === nothing || (slot .= x)
    end
    h
end

@stestset "stationarity history is bounded and has an exact update clock" begin
    h = svi_iterate_history(reshape(collect(1.0:10000), 1, :))
    @test h.updates == 10000
    @test h.count <= size(h.values, 2) == 512
    @test h.values[1, 1:h.count] == collect(Float64(h.stride):h.stride:10000)
    io = IOBuffer()
    serialize(io, h); seekstart(io)
    copy = deserialize(io)
    @test isequal(h, copy)
    for i in 10001:10500
        a, b = BRM._brm_svi_iterate_slot!(h), BRM._brm_svi_iterate_slot!(copy)
        @test (a === nothing) == (b === nothing)
        a === nothing || (a .= i; b .= i)
    end
    @test isequal(h, copy)
end

@stestset "stationarity diagnostics handle constants, missing history, and bad moments" begin
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(zeros(2, 99))).passed
    @test BRM._brm_svi_stationarity_check(svi_iterate_history(zeros(2, 100))).passed
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(fill(Inf, 1, 200))).passed
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(fill(NaN, 1, 200))).passed
    @test BRM._brm_svi_stationarity_check(BRM._BRMSVIIterates(0)).passed
end

function svi_stationary_ar(rng, d, n, rho)
    x = randn(rng, d, n)
    for j in 2:n
        x[:, j] .= rho .* view(x, :, j-1) .+ sqrt(1-rho^2) .* view(x, :, j)
    end
    x
end

@stestset "stationarity calibration covers autocorrelation and many coordinates" begin
    # Treat entire independent realizations as trials, not correlated time points.
    # The familywise maximum must be usable even at thousands of coordinates.
    for (d, n, rho) in ((1, 4096, 0.95), (32, 16384, 0.95), (1000, 512, 0.0))
        passed = map(1:20) do seed
            x = svi_stationary_ar(Xoshiro(seed), d, n, rho)
            BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
        end
        @info "stationarity calibration" d n rho successes=sum(passed) trials=length(passed)
        @test mean(passed) >= 0.9
    end
    # A single moving coordinate cannot hide in a high quantile over stable ones.
    x = randn(Xoshiro(11), 1000, 512)
    x[end, :] .+= 0.05 .* (1:512)
    @test !BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
    for seed in 1:20
        x = svi_stationary_ar(Xoshiro(seed), 3, 4096, 0.9)
        x[2, :] .+= 0.003 .* (1:4096)
        @test !BRM._brm_svi_stationarity_check(svi_iterate_history(x)).passed
    end
end

@stestset "stationarity reads marginal scales without dense low-rank covariance" begin
    rng = Xoshiro(4)
    q = (; μ=[0.2, -0.3], ρ=[-0.5, 400.0], Β=reshape([0.1, 0.0], 2, 1))
    h = BRM._BRMSVIIterates(4)
    BRM._brm_svi_record_global!(h, q)
    @test h.values[1:2, 1] == q.μ
    @test h.values[3, 1] ≈ log(sqrt(exp(-1) + 0.01))
    @test h.values[4, 1] == 400
    U = [2.0 0; 0.7 1.0]
    @test BRM._brm_svi_marginal_logsd(U*U') ≈ log.(diag(inv(U*U'))) ./ 2
    f = BRM._BRMSVILowRankUnit(randn(rng, 10000), randn(rng, 10000, 2))
    s = BRM._brm_svi_marginal_logsd(BRM._BRMSVIUnitPrecision(f))
    @test s ≈ log.(exp.(2f.ρ) .+ vec(sum(abs2, f.Β; dims=2))) ./ 2
end

@stestset "stationarity cannot certify an unvisited or drifting unit" begin
    c = (; stationarity=true, window=1, min_windows=1, patience=1, settle_windows=10)
    q = (; μ=[0.0], ρ=[0.0], Β=zeros(1,0), M=[[0.0]], Λ=[ones(1,1)])
    h = BRM._brm_svi_stationarity_state(q, c)
    h.globals = svi_iterate_history(randn(Xoshiro(3), 2, 512))
    run = (; iteration=512, sampling=(; stationarity=h))
    @test !BRM._brm_svi_stationary!(run, c)
    @test h.report.units[1].reason == :insufficient_history
    h.units[1] = svi_iterate_history(vcat(reshape(collect(1.0:512),1,:), zeros(1,512)))
    h.next_check = 512
    @test !BRM._brm_svi_stationary!(run, c)
    h.units[1] = svi_iterate_history(randn(Xoshiro(5), 2, 512))
    h.next_check = 512
    @test BRM._brm_svi_stationary!(run, c)
    @test h.report.passed
    # New observations are required before another check can pass.
    @test !BRM._brm_svi_stationary!(run, c)
end
