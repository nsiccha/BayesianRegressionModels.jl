# test/ar_term.jl — gate for the `ar(time; p=1)` AR(1) formula term, including the
# regression that TWO `ar(...)` terms over the same time axis on different responses
# must not collide (the AR(1) column is namespaced by the response, not just the time
# column). Transpile + stanc only — no BridgeStan needed for this structural gate.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Normal
import StanBlocks.stan: transpiles

ar_stanc_ok(brmi) = begin
    m = SBBRMI(brmi; mod = @__MODULE__).model
    transpiles(m) && StanBlocks.stanc_check(StanBlocks.stan_code(m); warn_pedantic = false).ok
end

@testset "ar(time; p=1) — single AR(1) residual term" begin
    df = (; t = collect(1.0:20.0), y = randn(20))
    brmi = @brm df begin
        mu ~ 1 + ar(t; p = 1)
        y ~ Normal(mu, 1.0)
    end
    @test ar_stanc_ok(brmi)
end

# Regression: two `ar(t; p=1)` terms over the SAME time column on DIFFERENT responses
# used to both emit `ar_t ~ _sb_ar1(...)` and collide (`name ∉ keys(info)`). The AR(1)
# column is now namespaced by the response (`ar_<resp>_t`).
@testset "ar(time; p=1) — two AR(1) terms share a time axis (no collision)" begin
    df = (; t = collect(1.0:20.0), y = randn(20))
    brmi = @brm df begin
        a ~ 1 + ar(t; p = 1)
        b ~ 1 + ar(t; p = 1)
        y ~ Normal(a + b, 1.0)
    end
    @test ar_stanc_ok(brmi)
end
