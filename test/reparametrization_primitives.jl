using Test, BayesianRegressionModels, LinearAlgebra, Distributions
import ReparametrizableDistributions as RD
const BRM = BayesianRegressionModels
include("reparametrization_reference.jl")

@testset "BRM consumes RD primitives" begin
    @test BRM.advance!! === RD.advance!!

    @testset "shared and stratified Cholesky parity" begin
        for (; x, eta, lp, L) in CHOLESKY_REFERENCE
            n = size(L, 1)
            p = BRM.Part(BRM.chol, (; L=fill(91.0, n, n)))
            @test BRM.nparams(p) == length(x)
            @test BRM.lprior!(p, x; eta) ≈ lp atol=1e-12 rtol=1e-12
            @test Matrix(LowerTriangular(p.data.L)) ≈ L atol=1e-12 rtol=1e-12
            @test all(p.data.L[i, j] == 91.0 for j in 1:n for i in 1:j-1)
        end

        # Distinct strata expose position, ordering and slice-write mistakes.
        a, b = CHOLESKY_REFERENCE[8], CHOLESKY_REFERENCE[11]
        Ls = fill(91.0, 3, 3, 2)
        p = BRM.Part(BRM.chol_by, (; Ls))
        x = vcat(a.x, b.x)
        original = copy(x)
        @test BRM.nparams(p) == length(x)
        @test BRM.lprior!(p, x; eta=a.eta) ≈ a.lp + b.lp atol=1e-12
        @test Matrix(LowerTriangular(Ls[:, :, 1])) ≈ a.L atol=1e-12
        @test Matrix(LowerTriangular(Ls[:, :, 2])) ≈ b.L atol=1e-12
        @test x == original

        values = zeros(3, 3)
        grouped = BRM.Part(BRM.grouped_normal_by, (; values, Ls, stratum_idx=[2, 1, 2]))
        z = collect(range(-0.8, 1.1; length=9))
        @test BRM.lprior!(grouped, z) ≈ sum(logpdf.(Normal(), z)) atol=1e-12
        @test values[1, :] ≈ b.L * z[1:3] atol=1e-12
        @test values[2, :] ≈ a.L * z[4:6] atol=1e-12
        @test values[3, :] ≈ b.L * z[7:9] atol=1e-12
    end

    @testset "simplex parity" begin
        for (; x, alpha, lp, values) in SIMPLEX_REFERENCE
            p = BRM.Part(BRM.simplex, (; values=zeros(length(values))))
            @test BRM.nparams(p) == length(x)
            @test BRM.lprior!(p, x; alpha) ≈ lp atol=1e-12 rtol=1e-12
            @test p.data.values ≈ values atol=1e-12 rtol=1e-12
        end
    end

    @testset "nested block cursor and grouped values" begin
        c, s = CHOLESKY_REFERENCE[8], SIMPLEX_REFERENCE[8]
        L = zeros(3, 3)
        population = BRM.Part(BRM.normal, (; values=zeros(1, 2)))
        factor = BRM.Part(BRM.chol, (; L))
        effects = BRM.Part(BRM.grouped_normal, (; values=zeros(2, 3), L))
        simplex = BRM.Part(BRM.simplex, (; values=zeros(4)))
        blocks = (; population=(population,), group=(factor, effects), monotonic=(simplex,))
        beta, z = [0.25, -0.5], [-0.7, 0.1, 0.8, 0.2, -0.3, 0.4]
        x = vcat(beta, c.x, z, s.x)
        original = copy(x)
        @test BRM.nparams(blocks) == length(x)
        @test BRM.lprior!(blocks, x) ≈ sum(logpdf.(Normal(), vcat(beta, z))) + c.lp + s.lp atol=1e-12
        @test vec(population.data.values) == beta
        @test effects.data.values[1, :] ≈ c.L * z[1:3] atol=1e-12
        @test effects.data.values[2, :] ≈ c.L * z[4:6] atol=1e-12
        @test simplex.data.values ≈ s.values atol=1e-12
        @test x == original
    end
end
