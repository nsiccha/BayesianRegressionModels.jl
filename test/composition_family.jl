# test/composition_family.jl — gate for the Multinomial composition-observation
# family on the @brm surface: a per-row K-category count-vector response (`int[n,K]`)
# observed via `Multinomial(N, probs)` with probs shared across rows (StanBlocks'
# multi-row `multinomial_lpmf(obs::int[n,K], probs::vector[K], row_N::int[n])`).
# Transpile + stanc.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Multinomial
import StanBlocks.stan: transpiles

cf_stanc_ok(brmi) = begin
    m = SBBRMI(brmi; mod = @__MODULE__).model
    transpiles(m) && StanBlocks.stanc_check(StanBlocks.stan_code(m); warn_pedantic = false).ok
end

@testset "Multinomial composition obs — int[n,K] response, shared probs" begin
    K = 3; n = 12
    pr = [0.3, 0.5, 0.2]
    comps = [[6, 10, 4] for _ in 1:n]                 # per-row K-category counts
    compmat = permutedims(reduce(hcat, comps))        # n x K Int matrix
    Nrow = [sum(c) for c in comps]
    df = (; year = collect(1:n), comp = compmat, samp = Nrow, probs = pr)
    m(df) = @brm df begin
        comp ~ Multinomial(samp, probs)
    end
    @test cf_stanc_ok(m(df))
end

@testset "Multinomial composition obs — int[n,K] response, PER-ROW probs (matrix)" begin
    K = 3; n = 12
    probsmat = permutedims(reduce(hcat, [[0.3, 0.5, 0.2] .+ 0.05 * sin(t) for t in 1:n]))  # n x K per-row
    probsmat = probsmat ./ sum(probsmat; dims = 2)                                          # renormalize rows
    comps = [[6, 10, 4] for _ in 1:n]
    compmat = permutedims(reduce(hcat, comps))
    Nrow = [sum(c) for c in comps]
    df = (; year = collect(1:n), comp = compmat, samp = Nrow, rowprobs = probsmat)
    m(df) = @brm df begin
        comp ~ Multinomial(samp, rowprobs)      # per-row simplex -> brm_multinomial matrix method
    end
    @test cf_stanc_ok(m(df))
end
