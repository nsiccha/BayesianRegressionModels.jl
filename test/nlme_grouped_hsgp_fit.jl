# test/nlme_grouped_hsgp_fit.jl — NLMEEstimation on models whose grouped HSGP
# `hsgp(ecg_time; by=ecg_subject)` has the subjects as its levels (see
# test/nlme_grouped_hsgp.jl): the weight family held at the identity
# (`NLMEEstimation.fixed_omega`), the protocol checks on flat and ragged
# layouts, the population prior's ω gradient with a hyper-predicted amplitude's
# estimated family beside the fixed one, and short SAEM, FOCE(I) and
# empirical-Bayes fits read back by name.
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_grouped_hsgp_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_grouped_hsgp_fixture.jl"))
using NLMEEstimation, LinearAlgebra, Random
const NE = NLMEEstimation

gh_model(brmi) = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)

@stestset "the weight family is fixed at the identity" begin
    m = gh_model(plate_hsgp(GH_PLATE))
    k = weight_family(m.view)
    fixed = NE.fixed_omega(m)
    @test length(fixed) == 1 && first(only(fixed)) == k
    @test last(only(fixed)).sd == ones(4)
    @test last(only(fixed)).correlation == Matrix{Float64}(I, 4, 4)
    start = brm_nlme_start(m)
    indices = findall(b -> hasproperty(b.block, :hsgp), m.view.block_margins)
    @test start.omega_matrix[indices, indices] == Matrix{Float64}(I, 4, 4)
    @test isfinite(start.population_logprior)
    @test check_protocol(m, start.theta, start.sigma, start.omega;
        η=0.3 .* randn(Xoshiro(1), size(start.eta)...))
end

@stestset "a ragged layout when a subject has no level" begin
    m = gh_model(plate_hsgp(GH_PLATE_GAP))
    @test NE.nlme_layout(m) isa NE.RaggedNLMELayout
    start = brm_nlme_start(m)
    @test start.omega_matrix[weight_family(m.view)] == Matrix{Float64}(I, 4, 4)
    rng = Xoshiro(2)
    @test check_protocol(m, start.theta, start.sigma, start.omega;
        η=[0.3 .* randn(rng, length(e)) for e in start.eta])
end

@stestset "the population prior's ω gradient with a hyper-predicted amplitude" begin
    m = gh_model(plate_hyper(GH_PLATE))
    start = brm_nlme_start(m)
    θ, σ, ω = start.theta, start.sigma, copy(start.omega)
    probe = NE.apply_fixed_omega(m, ω .+ 0.05 .* randn(Xoshiro(3), length(ω)))
    value, _, _, gω = NE.population_logprior_and_gradient(m, θ, σ, probe)
    @test value ≈ NE.population_logprior(m, θ, σ, probe)
    step = 1e-6
    free = 0
    for j in eachindex(probe)
        e = zeros(length(probe)); e[j] = step
        up, down = NE.apply_fixed_omega(m, probe .+ e), NE.apply_fixed_omega(m, probe .- e)
        up == down && continue # a fixed entry
        free += 1
        fd = (NE.population_logprior(m, θ, σ, up) - NE.population_logprior(m, θ, σ, down)) / 2step
        @test gω[j] ≈ fd atol=1e-5 rtol=1e-5
    end
    @test free == 2 # the random intercept's and the amplitude effects' scales
end

@stestset "SAEM, FOCE(I) and empirical Bayes fits read back by name" begin
    m = gh_model(plate_hsgp(GH_PLATE))
    f = brm_nlme_fit(m, saem; rng=Xoshiro(4), options=SAEMOptions(iterations=6, burnin=3))
    @test all(isfinite, f.result.theta) && all(isfinite, f.result.sigma)
    k = weight_family(m.view)
    indices = findall(b -> hasproperty(b.block, :hsgp), m.view.block_margins)
    @test f.result.omega_matrix[indices, indices] == Matrix{Float64}(I, 4, 4)
    names = f.estimates.eta.names
    @test names[indices] == ["qt_time:hsgp(ecg_time)[$j]" for j in 1:4]
    @test size(f.estimates.eta.values) == (5, 4)
    # The length scale and amplitude are the term's shared values.
    hsgp_values = only(v for (key, v) in pairs(f.estimates.point.other)
        if startswith(String(key), "hsgp_ecg_time"))
    @test hsgp_values.rho > 0 && hsgp_values.sigma > 0

    fo = brm_nlme_fit(m, focei; options=FOCEOptions(maxeval=20))
    @test isfinite(fo.result.ofv)
    @test fo.result.omega_matrix[indices, indices] == Matrix{Float64}(I, 4, 4)

    eb = brm_nlme_fit(m, empirical_bayes; start=f.start)
    @test size(eb.estimates.eta.values) == (5, 4)

    hyper = gh_model(plate_hyper(GH_PLATE))
    g = brm_nlme_fit(hyper, saem; rng=Xoshiro(5), options=SAEMOptions(iterations=6, burnin=3))
    key = Symbol("log(sd(qt_time, hsgp(ecg_time)))|ecg_subject")
    block = getproperty(g.estimates.point, key)
    @test block.group === :ecg_subject && only(block.sd) > 0
end
