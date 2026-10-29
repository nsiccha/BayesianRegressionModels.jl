# test/nlme_offaxis_observations_fit.jl — NLMEEstimation on the panel of
# test/nlme_offaxis_fixture.jl, whose fragment's observations lie on rows of
# their own and are attributed to their regions by measurement: the protocol
# checks (which include the forward-over-reverse η Hessian blocks) and a short
# SAEM fit (snag nlme-model-obser-dd665aca).
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_offaxis_observations_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_offaxis_fixture.jl"))
using NLMEEstimation

function offaxis_nlme()
    data = offaxis_panel()
    brm_nlme_model(RKBRMI(offaxis_model(data)); ad_backend=NLME_AD, offaxis_groups(data)...)
end

@stestset "the protocol holds with the fragment's densities" begin
    m = offaxis_nlme()
    start = brm_nlme_start(m)
    @test isfinite(start.population_logprior)
    @test NLMEEstimation.check_protocol(m, start.theta, start.sigma, start.omega)
end

@stestset "SAEM fits the panel" begin
    m = offaxis_nlme()
    f = brm_nlme_fit(m, saem; rng=Xoshiro(1),
        options=SAEMOptions(iterations=60, burnin=30), ebes=false)
    @test !f.result.interrupted && f.result.iterations == 60
    @test all(isfinite, f.result.theta) && all(isfinite, f.result.sigma)
end

println("nlme_offaxis_observations_fit.jl: all testsets passed")
