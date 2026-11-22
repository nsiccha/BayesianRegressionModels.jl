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
        # The shared kernels hold the Cholesky form for ordinary scales. The
        # entropy and a structured unit's whole step keep this geometry here.
        @test BRM._svi_narrow_geometry(q.ρ, Β)
        @test !BRM._svi_narrow_geometry(zeros(D), Β)
        @test BRM._brm_svi_factor_logdet(q) ≈ logdet(Σ)
        f = BRM._BRMSVILowRankUnit(copy(q.ρ), copy(Β))
        @test BRM._brm_svi_unit_logdet_root(BRM._BRMSVIUnitRoot(f)) ≈ -logdet(Σ) / 2
        gz, gd, ε = randn(rng, D), randn(rng, D), randn(rng, D + r)
        e, η = ε[1:D], ε[D+1:end]
        score = gd + Σ \ (s .* e + Β * η)
        gradient = vcat(gz, score .* s .* e, vec(score * η'))
        @test BRM._brm_svi_lowrank_gradient(f, gz, gd, ε) ≈ gradient
        # Adam's first step is lr * g / (|g| + 1e-8) in the whitened
        # coordinates. At this rate the trust bound shortens the whole move.
        mean = randn(rng, D)
        old = vcat(mean, f.ρ, vec(f.Β))
        scale = vcat(ones(D), 0.7 .* repeat(s, r))
        g = vcat(C' * gradient[1:D], gradient[D+1:end] .* scale)
        step = 2.0 .* g ./ (abs.(g) .+ 1e-8)
        move = vcat(C * step[1:D], step[D+1:end] .* scale)
        distance = sqrt(dot(move[1:D], Σ \ move[1:D]))
        relative = max(maximum(abs, move[D+1:2D]), maximum(abs, move[2D+1:end] ./ repeat(s, r)))
        fraction = min(1 / distance, 0.5 / relative)
        @test fraction < 1
        expected = old + fraction * move
        opt = BRM._BRMSVIAdam(length(old), 0.1)
        BRM._brm_svi_lowrank_step!(mean, f, opt, gz, gd, ε, 2.0; rank=0.7)
        @test vcat(mean, f.ρ, vec(f.Β)) ≈ expected
        @test opt.t == 1
    end
end

@testset "a GPU fit stops at a narrow factor" begin
    Β = reshape([0.2, 0.15], 2, 1)
    ordinary = BRM._BRMSVIUnitPrecision(BRM._BRMSVILowRankUnit(zeros(2), copy(Β)))
    narrow = BRM._BRMSVIUnitPrecision(BRM._BRMSVILowRankUnit([0.0, log(1e-10)], copy(Β)))
    dense = Matrix{Float64}(I, 2, 2)
    q = (; ρ=zeros(2), Β=copy(Β), Λ=[dense, ordinary])
    @test BRM._brm_svi_require_device_geometry(q, 40) === nothing
    @test BRM._brm_svi_require_device_geometry((; ρ=zeros(2), Β=zeros(2, 0), Λ=[dense]), 40) === nothing
    @test_throws "iteration 40. A diagonal scale of the factor of unit 3" BRM._brm_svi_require_device_geometry(
        (; q..., Λ=[dense, ordinary, narrow]), 40)
    @test_throws "the global factor" BRM._brm_svi_require_device_geometry(
        (; q..., ρ=[0.0, log(1e-10)]), 40)
    @test_throws "Continue from the last checkpoint with device=:cpu" BRM._brm_svi_require_device_geometry(
        (; q..., ρ=[0.0, log(1e-10)]), 40)
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
