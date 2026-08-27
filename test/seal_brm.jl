# test/seal_brm.jl — gate for the FULL grey-seal IPM on the @brm formula surface
# (research/seal/grey_seal_brm.jl). Gates the MODEL ARTIFACT at the same level as the
# @slic reference `test/seal_ipm.jl`: transpile + `stanc` + `compiles`. Like the
# reference it is NOT sampled — the mechanistic scan's simplex constraints make a naive
# init degenerate for both the @slic and @brm forms (verified: the @slic reference is
# also non-finite at the origin), so the deliverable is the model, not a fit.
#
# Demonstrates the full hoist: brms-formula LPs (herring→birth regression, per-year /
# per-demo / per-noise random effects with estimated group SDs) feeding the VERBATIM
# run_state_process scan (Leslie + hunting ODE + multinomial-allocation) consumed via
# field access, with all EIGHT observation streams as @brm family likelihoods
# (NegativeBinomial2 / Normal / Multinomial / Binomial) on their own year-subset frames.

using Test
using BayesianRegressionModels
using StanBlocks
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_brm.jl"))

seal_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

@testset "grey-seal IPM — full @brm formula-surface port (verbatim scan + 8 streams)" begin
    m = SBBRMI(grey_seal_brm_model()).model
    @test transpiles(m)
    @test compiles(m)
    @test seal_stanc_ok(m)
end
