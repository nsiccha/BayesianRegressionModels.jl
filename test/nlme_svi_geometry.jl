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

function geometry_bytes(D, r)
    q = (; ρ=zeros(D), Β=fill(0.01, D, r))
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
    for r in (0, 3)
        geometry_bytes(100, r)
        small = geometry_bytes(10_000, r)
        large = geometry_bytes(20_000, r)
        @test large < 2.2small
        @test large < 20_000_000
    end
end
