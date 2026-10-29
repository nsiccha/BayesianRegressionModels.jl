# test/nlme_shares_units_fit.jl — NLMEEstimation on the shares panel of
# test/nlme_shares_fixture.jl: the grouped coordinates of a non-centered
# `mi(x)::shares(total, rows)` completion held at unit variance
# (`NLMEEstimation.fixed_omega`), the protocol checks (which include the
# forward-over-reverse η Hessian blocks) and a short SAEM fit
# (snag nlme-view-shares-f1cfaa71).
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_shares_units_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_shares_fixture.jl"))
using NLMEEstimation
const NE = NLMEEstimation

shares_nlme() = brm_nlme_model(
    RKBRMI(shares_model(shares_panel()); completions=:noncentered);
    ad_backend=NLME_AD, SHARES_NESTED...)

@stestset "the grouped coordinates are a family fixed at unit variance" begin
    m = shares_nlme()
    k = only(f for (f, family) in enumerate(m.view.families)
        if get(family.block, :shares, false))
    @test first(only(NE.fixed_omega(m))) == k
    start = brm_nlme_start(m)
    @test start.omega_matrix[k] == fill(1.0, 1, 1)
    @test isfinite(start.population_logprior)
    @test NE.check_protocol(m, start.theta, start.sigma, start.omega)
end

@stestset "SAEM fits the panel" begin
    m = shares_nlme()
    f = brm_nlme_fit(m, saem; rng=Xoshiro(1),
        options=SAEMOptions(iterations=60, burnin=30), ebes=false)
    @test !f.result.interrupted && f.result.iterations == 60
    @test all(isfinite, f.result.theta) && all(isfinite, f.result.sigma)
    @test f.estimates.point.sigma > 0
end

println("nlme_shares_units_fit.jl: all testsets passed")
