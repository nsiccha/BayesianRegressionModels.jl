# test/cdc_ww_inference.jl — gate for the full CDC ww-inference model port in
# research/wastewater/cdc_ww_inference.jl (decision `2026-08-26T16-44-33-638-1yz7ybe`,
# option B). Gates the MODEL ARTIFACT: transpile + stanc + finite BridgeStan
# density/gradient (at the unconstrained origin, i.e. stable Rt≈1). Not a fit.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "wastewater", "cdc_ww_inference.jl"))

const CDC_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const CDC_CACHE = joinpath(tempdir(), "brm-cdc-ww-inference")

function cdc_bridgestan_finite(model)
    isdir(CDC_CACHE) || mkpath(CDC_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(CDC_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, zeros(dim))
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

cdc_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "CDC ww-inference — full model (multi-subpop + WW + joint hosp)" begin
    m = cdc_ww_inference_model()
    @test transpiles(m)
    @test compiles(m)
    @test cdc_stanc_ok(m)
    CDC_RUN_BRIDGESTAN && @test cdc_bridgestan_finite(m)
end

@testset "CDC ww-inference — re-bind new data (different K / horizon)" begin
    m = cdc_ww_inference_model(cdc_ww_inference_fixture(; K = 2, nt = 56, n_weeks = 8))
    @test transpiles(m)
    @test cdc_stanc_ok(m)
end
