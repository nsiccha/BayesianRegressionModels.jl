# Run in the NLMEEstimation scratch environment from test/README.md.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_completion_fixture.jl"))
using NLMEEstimation, Random
const NE = NLMEEstimation

@stestset "gathered bounded completions retain fixed families through the NLME protocol and SAEM" begin
    rk = RKBRMI(bounded_gathered_completion(COMPLETION_AXES_DATA); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject)
    fixed = findall(f -> get(f.block, :completion, false), m.view.families)
    @test length(fixed) == 2
    @test sort(first.(NE.fixed_omega(m))) == fixed
    start = brm_nlme_start(m)
    @test all(k -> start.omega_matrix[k] == fill(1., 1, 1), fixed)
    @test NE.check_protocol(m, start.theta, start.sigma, start.omega)
    fit = brm_nlme_fit(m, saem; rng=Xoshiro(48),
        options=SAEMOptions(iterations=30, burnin=15), ebes=false)
    @test !fit.result.interrupted && fit.result.iterations == 30
    @test all(isfinite, fit.result.theta) && all(isfinite, fit.result.sigma)
    @test all(k -> fit.result.omega_matrix[k] == fill(1., 1, 1), fixed)
end
