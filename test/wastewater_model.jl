# test/wastewater_model.jl — compile gate for the faithful EpiSewer-style
# wastewater→Rt model in research/wastewater/renewal.jl (decision
# `2026-08-26T11-06-02-112-12s1pu8`). Gates the MODEL ARTIFACT — transpile +
# stanc + finite BridgeStan density/gradient — not a fit (no sampling required).

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "wastewater", "renewal.jl"))

const WW_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const WW_CACHE = joinpath(tempdir(), "brm-wastewater-model")

function ww_bridgestan_finite(model)
    isdir(WW_CACHE) || mkpath(WW_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(WW_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

ww_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "wastewater EpiSewer-style model — core (lognormal WW obs)" begin
    m = wastewater_model()
    @test transpiles(m)
    @test compiles(m)
    @test ww_stanc_ok(m)
    WW_RUN_BRIDGESTAN && @test ww_bridgestan_finite(m)
end

@testset "wastewater EpiSewer-style model — below-LOD censored obs" begin
    m = wastewater_censored_model()
    @test transpiles(m)
    @test compiles(m)
    @test ww_stanc_ok(m)
    WW_RUN_BRIDGESTAN && @test ww_bridgestan_finite(m)
end

# Re-binding new data (a different site/timepoint count) must trace clean too.
@testset "wastewater EpiSewer-style model — re-bind new data" begin
    m = wastewater_model(wastewater_fixture(; nsites = 2, nt = 20))
    @test transpiles(m)
    @test ww_stanc_ok(m)
end
