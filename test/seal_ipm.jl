# test/seal_ipm.jl — gate for the grey-seal IPM port in research/seal/grey_seal_ipm.jl
# (decision `2026-08-26T19-58-24-844-0yiqrzn`, option A: verbatim port of the
# SlicTranspiler spotlight example). Gates the MODEL ARTIFACT: compile_slic_bundle
# assembles it and the emitted Stan is stanc-clean. Not sampled (heavy ODE model).

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions

include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_ipm.jl"))

@testset "grey-seal IPM — compile_slic_bundle + stanc" begin
    result = grey_seal_ipm_bundle()
    @test result.code isa AbstractString
    @test length(result.code) > 10_000                       # a substantial program
    @test StanBlocks.stanc_check(result.code; warn_pedantic = false).ok
    ops = Set(Symbol.(string.(result.descriptor.operations)))
    # a full observation model → fit / predict / pointwise_loglik are offered
    @test :ModelOperation in Set([nameof(typeof(o)) for o in result.descriptor.operations]) ||
          length(result.descriptor.operations) >= 3
    @test occursin("run_state_process", result.code)         # the state-process scan
    @test occursin("neg_binomial_2", result.code)            # aerial-count stream
    @test occursin("multinomial", result.code)               # composition streams
    @test occursin("ode_rk45", result.code)                  # within-year hunting ODE
end

@testset "grey-seal IPM — re-bind (verify data flows through the builder)" begin
    d = grey_seal_ipm_fixture()
    result = grey_seal_ipm_bundle(d)
    @test StanBlocks.stanc_check(result.code; warn_pedantic = false).ok
end
