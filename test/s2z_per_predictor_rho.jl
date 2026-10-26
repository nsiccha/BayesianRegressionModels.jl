# Per-predictor S2Z weights: `s2z_rho = [:mu => w_mu, :sigma => w_sigma]` gives
# every predictor on an S2Z grouping its own centering weights, so a location
# predictor with several coefficients and a one-coefficient scale predictor on
# the same grouping can differ (snag `exact-totals-fun-f116d6b2`, user pick
# `0rz9l5s`). The weights are data of the compiled program: the Stan source is
# the scalar build's, and only each block's `s2z_rho_*` matrix changes.
#
# Run: julia --project=test test/s2z_per_predictor_rho.jl
using Test, Random, Statistics
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels

const RHO_BUILDER = @brm begin
    mu ~ 1 + x + (1 | a | g) + (0 + x | b | g)
    log(sigma) ~ 1 + (1 | c | g)
    effect(mu, Intercept) ~ Normal(0.0, 2.0)
    effect(mu, x) ~ Normal(0.0, 1.0)
    effect(sigma, Intercept) ~ Normal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end

const RHO_DATA = let rng = Xoshiro(20261026), J = 6, n = 8
    g = repeat(1:J; inner=n)
    x = randn(rng, J * n)
    (; g, x, y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(rng, J)[g] .+ 0.2 .* randn(rng, J * n))
end

rho_build(rho) = SBBRMI(RHO_BUILDER(RHO_DATA); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=rho)
rho_of(sb, lp) = only(b for b in s2z_effect_blocks(sb) if b.predictor === lp).rho
rho_data(sb) = Dict(k => v for (k, v) in sb.data if startswith(String(k), "s2z_rho_"))

@testset "per-predictor weights reach each block" begin
    sb = rho_build([:mu => [1.0, 0.0], :sigma => 0.5])
    J = length(unique(RHO_DATA.g))
    @test rho_of(sb, :mu) == repeat([1.0 0.0], J, 1)
    @test rho_of(sb, :sigma) == fill(0.5, J, 1)
    # a J-by-K matrix and a tuple of pairs are accepted too
    m = [range(0, 1; length=J) range(1, 0; length=J)]
    sbm = rho_build((:mu => m, :sigma => 1.0))
    @test rho_of(sbm, :mu) == m
    @test rho_of(sbm, :sigma) == fill(1.0, J, 1)
end

@testset "same program as the scalar build; only the weights differ" begin
    scalar = rho_build(1.0)
    same = rho_build([:mu => 1.0, :sigma => 1.0])
    mixed = rho_build([:mu => [1.0, 0.0], :sigma => 0.5])
    @test StanBlocks.stan_code(same.model) == StanBlocks.stan_code(scalar.model)
    @test StanBlocks.stan_code(mixed.model) == StanBlocks.stan_code(scalar.model)
    @test rho_data(same) == rho_data(scalar)
    @test rho_data(mixed) != rho_data(scalar)
    @test Set(keys(mixed.data)) == Set(keys(scalar.data))
    for k in keys(scalar.data)
        startswith(String(k), "s2z_rho_") && continue
        @test isequal(mixed.data[k], scalar.data[k])
    end
end

@testset "compiled per-predictor weights run" begin
    sb = rho_build([:mu => [1.0, 0.0], :sigma => 0.5])
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    prob = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "s2z_per_predictor.stan"))
    x = 0.1 .* randn(Xoshiro(1), LogDensityProblems.dimension(prob))
    ld, g = LogDensityProblems.logdensity_and_gradient(prob, x)
    @test isfinite(ld)
    @test all(isfinite, g)
end

@testset "refused per-predictor weights" begin
    # a predictor without a block on the S2Z grouping
    @test_throws r"`s2z_rho` names `nu`" rho_build([:mu => 1.0, :sigma => 1.0, :nu => 1.0])
    # an S2Z predictor without weights
    @test_throws r"`s2z_rho` gives no weights for `sigma`" rho_build([:mu => 1.0])
    @test_throws r"more than once" rho_build([:mu => 1.0, :sigma => 1.0, :mu => 0.0])
    @test_throws r"Symbol predictor names" rho_build(["mu" => 1.0, "sigma" => 1.0])
    # one predictor's weights still match its own coefficient count
    @test_throws r"K = 2 coefficients" rho_build([:mu => [1.0, 0.0, 1.0], :sigma => 1.0])
end

@testset "a single pair when one predictor is S2Z" begin
    one = @brm begin
        mu ~ 1 + x + (1 | a | g)
        sigma ~ Exponential(1.0)
        effect(mu, Intercept) ~ Normal(0.0, 2.0)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(one(RHO_DATA); mod=@__MODULE__, s2z_groups=[:g], s2z_rho=:mu => 0.25)
    @test only(s2z_effect_blocks(sb)).rho == fill(0.25, length(unique(RHO_DATA.g)), 1)
end
