# test/rk_cdar_fit.jl — NLMEEstimation on a model whose `cdar(year; by=kreis)`
# walk nests in the region units (see test/rk_cdar.jl): the innovation family
# held at the identity (`NLMEEstimation.fixed_omega`), the protocol checks on
# the ragged layout, and short SAEM and empirical-Bayes fits read back by name.
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/rk_cdar_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using NLMEEstimation, LinearAlgebra, Random
const NE = NLMEEstimation
include(joinpath(@__DIR__, "rk_cdar_fixture.jl"))

nested_model() = brm_nlme_model(RKBRMI(walk_model()); ad_backend=NLME_AD, NESTED...)
innovations(v) = findall(b -> hasproperty(b.block, :cdar), v.block_margins)

@stestset "the innovation family is fixed at the identity on a ragged layout" begin
    m = nested_model()
    @test NE.nlme_layout(m) isa NE.RaggedNLMELayout
    k = cdar_family(m.view)
    fixed = NE.fixed_omega(m)
    @test length(fixed) == 1 && first(only(fixed)) == k
    @test last(only(fixed)).sd == ones(4)
    @test last(only(fixed)).correlation == Matrix{Float64}(I, 4, 4)
    start = brm_nlme_start(m)
    @test start.omega_matrix[k] == Matrix{Float64}(I, 4, 4)
    @test isfinite(start.population_logprior)
    rng = Xoshiro(1)
    @test check_protocol(m, start.theta, start.sigma, start.omega;
        η=[0.3 .* randn(rng, length(e)) for e in start.eta])
end

@stestset "SAEM and empirical Bayes fits read back by name" begin
    m = nested_model()
    f = brm_nlme_fit(m, saem; rng=Xoshiro(2), options=SAEMOptions(iterations=6, burnin=3))
    @test f.result isa RaggedSAEMResult
    @test all(isfinite, f.result.theta) && all(isfinite, f.result.sigma)
    k = cdar_family(m.view)
    @test f.result.omega_matrix[k] == Matrix{Float64}(I, 4, 4)
    # Each region's effects: its intercept, then four innovations per district.
    eta = f.estimates.eta
    @test eta.subjects == ["r1", "r2", "r3"]
    @test length.(eta.values) == [9, 9, 9]
    @test eta.names[1][2:5] == ["kreis[k1]:mu:cdar(year)[$t]" for t in 1:4]
    # The walk's marginal SD and persistence are its shared values.
    walk = only(v for (key, v) in pairs(f.estimates.point.other)
        if startswith(String(key), "cdar_mu_year"))
    @test walk.sigma > 0 && 0 < walk.rho < 1

    eb = brm_nlme_fit(m, empirical_bayes; start=f.start)
    @test length.(eb.estimates.eta.values) == [9, 9, 9]
end

println("rk_cdar_fit.jl: all testsets passed")
