include(joinpath(@__DIR__, "fitting_packages.jl"))
using Test, BayesianRegressionModels, LinearAlgebra, Random
const BRM = BayesianRegressionModels

@testset "SVI diagonal plus low-rank geometry" begin
    rng = Xoshiro(302)
    for r in (0, 1, 3), zero_rank in (false, true)
        D = 11
        q = (; ρ=0.3randn(rng, D), Β=zero_rank ? zeros(D, r) : 0.4randn(rng, D, r))
        h = BRM._brm_svi_global_geometry(q)
        Σ = Matrix(BRM._brm_svi_global_covariance(q))
        x = randn(rng, D)
        @test BRM._brm_svi_global_precision(h, x) ≈ Σ \ x
        @test BRM._brm_svi_global_logdet(h) ≈ logdet(Σ)
        c = BRM._brm_svi_global_sqrt(h)
        C = reduce(hcat, [BRM._brm_svi_global_sqrt_mul(c, e)
            for e in eachcol(Matrix{Float64}(I, D, D))])
        @test C * C' ≈ Σ
        @test BRM._brm_svi_global_sqrt_tmul(c, x) ≈ C' * x
        w, ε, η = randn(rng, D), randn(rng, D), randn(rng, r)
        P = inv(Σ)
        expected = vcat(w, w .* ε .* exp.(q.ρ) + diag(P) .* exp.(2q.ρ),
            vec(w * η' + P * q.Β))
        @test BRM._brm_svi_global_gradient(q, w, ε, η) ≈ expected
    end
end

@testset "a narrow diagonal with finite rank loadings" begin
    rng = Xoshiro(17)
    for (D, r) in ((19, 4), (3, 8))
        s = ones(D)
        s[end] = 1e-10
        Β = 0.2randn(rng, D, r)
        Β[end, :] .= 0.15
        q = (; ρ=log.(s), Β)
        # Σ is well conditioned. The unit contribution disappears if I + V'V
        # is formed first, since the last row of V is about 1e9.
        Σ = cholesky(Symmetric(Diagonal(s.^2) + Β * Β'))
        P = inv(Σ)
        h = BRM._brm_svi_global_geometry(q)
        x = randn(rng, D, 3)
        @test BRM._brm_svi_global_precision(h, x) ≈ Σ \ x
        @test BRM._brm_svi_global_precision(h, x[:, 1]) ≈ Σ \ x[:, 1]
        @test BRM._brm_svi_global_logdet(h) ≈ logdet(Σ)
        gradient = BRM._brm_svi_global_gradient(q, zeros(D), zeros(D), zeros(r))
        @test gradient[D+1:2D] ./ s.^2 ≈ diag(P)
        @test reshape(gradient[2D+1:end], D, r) ≈ Σ \ Β
        c = BRM._brm_svi_global_sqrt(h)
        C = BRM._brm_svi_global_sqrt_mul(c, Matrix{Float64}(I, D, D))
        @test C * C' ≈ Matrix(Σ)
        @test BRM._brm_svi_global_sqrt_tmul(c, x) ≈ C' * x
    end
end

function geometry_bytes(D, r; narrow=false)
    q = (; ρ=zeros(D), Β=fill(0.01, D, r))
    narrow && (q.ρ[end] = log(1e-10))
    x = ones(D)
    h = BRM._brm_svi_global_geometry(q)
    BRM._brm_svi_global_precision(h, x)
    BRM._brm_svi_global_sqrt(h)
    @allocated begin
        h = BRM._brm_svi_global_geometry(q)
        BRM._brm_svi_global_precision(h, x)
        c = BRM._brm_svi_global_sqrt(h)
        BRM._brm_svi_global_sqrt_mul(c, x)
    end
end

@testset "SVI global geometry allocates linearly in field size" begin
    for (r, narrow) in ((0, false), (3, false), (3, true))
        geometry_bytes(100, r; narrow)
        small = geometry_bytes(10_000, r; narrow)
        large = geometry_bytes(20_000, r; narrow)
        @test large < 2.2small
        @test large < 20_000_000
    end
end
