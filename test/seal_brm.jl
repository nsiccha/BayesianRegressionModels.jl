# test/seal_brm.jl — gate for the faithful-structure grey-seal IPM on the @brm formula
# surface (research/seal/grey_seal_brm.jl). Gates the MODEL ARTIFACT: transpile + stanc
# + finite BridgeStan density/gradient at the unconstrained origin. Not a fit.
#
# Demonstrates: birth-rate covariate regression + per-year (1|year) RE + a mechanistic
# @deffun scan returning a NamedTuple consumed via FIELD ACCESS + three observation
# streams (NegativeBinomial2 pup counts, Binomial pregnancy, per-row Multinomial age
# composition).

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_brm.jl"))

const SEAL_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const SEAL_CACHE = joinpath(tempdir(), "brm-seal-brm")

function seal_bridgestan_finite(model)
    isdir(SEAL_CACHE) || mkpath(SEAL_CACHE)
    code = StanBlocks.stan_code(model)
    prob = StanBlocks.stan_instantiate(model; path = joinpath(SEAL_CACHE, string(hash(code)) * ".stan"))
    dim = LogDensityProblems.dimension(prob)
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, zeros(dim))
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

seal_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "grey-seal IPM — @brm formula surface (regression + (1|year) RE + scan + 3 streams)" begin
    m = SBBRMI(grey_seal_brm_model()).model
    @test transpiles(m)
    @test compiles(m)
    @test seal_stanc_ok(m)
    SEAL_RUN_BRIDGESTAN && @test seal_bridgestan_finite(m)
end

@testset "grey-seal IPM — @brm form re-binds new data (different horizon)" begin
    m = SBBRMI(grey_seal_brm_model(grey_seal_brm_fixture(; T = 60))).model
    @test transpiles(m)
    @test seal_stanc_ok(m)
end
