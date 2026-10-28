# test/nlme_crossed_fit.jl — NLMEEstimation on models with a grouping crossed
# with the units (`shared_groups`, test/nlme_crossed.jl): the population prior
# carries the crossed block's authored priors, prior-median starts set its
# scales, `brm_nlme_fit` estimates it in θ and names it, and the two-stage fit
# holds it at the stage-1 estimate while it samples each unit.
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_crossed_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_crossed_fixture.jl"))
using NLMEEstimation, LogDensityProblems
const NE = NLMEEstimation

const CR_SHARED = (; shared_groups=(:class,))
const CR_SAEM = NE.SAEMOptions(iterations=4, burnin=2, mcmc_burnin=0, adapt_iterations=0,
    numeric_maxiter=2, omega_floor=0.05)

@stestset "the population prior carries the crossed block's authored priors" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, CR_SHARED...)
    rng = Xoshiro(11)
    θ, σ = 0.3 .* randn(rng, length(m.theta)), 0.3 .* randn(rng, length(m.sigma))
    τ = 0.7
    # The RK prior and log-Jacobian at the same point: the subjects' draws at
    # zero (their N(0, 1) terms belong to NLME), their scale at τ.
    u = copy(m.unit)
    u[m.theta] = θ
    u[m.sigma] = σ
    u[only(findall(==(:subject_scale), m.view.role))] = log(τ)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    draws = findall(==(:subject_effect), m.view.role)
    @test all(iszero, u[draws])
    oracle = Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        length(draws) * logpdf(Normal(), 0.0)
    @test NE.population_logprior(m, θ, σ, [log(τ)]) ≈ oracle atol=1e-10
    value, gθ, _, _ = NE.population_logprior_and_gradient(m, θ, σ, [log(τ)])
    @test value ≈ oracle atol=1e-10
    class = owned(m.view, BRM._brm_nlme_ranef_declaration(class_record(rk)))
    # The class block's standardized draws have their N(0, 1) prior in θ.
    for c in class
        occursin(".z.", String(m.view.coordinates[c])) || continue
        k = findfirst(==(c), m.theta)
        @test gθ[k] ≈ -θ[k] atol=1e-9
    end
end

@stestset "prior-median starts set the crossed block's scales" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, CR_SHARED...)
    start = brm_nlme_start(m)
    @test isfinite(start.population_logprior)
    record = class_record(rk)
    scales = owned(m.view, Symbol(BRM._brm_nlme_ranef_declaration(record), ".tau"))
    @test !isempty(scales) && m.view.coordinates[scales] ⊆ start.start_source.prior_median
end

@stestset "brm_nlme_fit estimates the crossed block in θ and names it" begin
    fit = brm_nlme_fit(RKBRMI(crossed_flat(CR_FLAT)), NE.saem; ebes=false,
        rng=Xoshiro(12), options=CR_SAEM, CR_SHARED...)
    @test fit.model.view.shared_groups == (:class,)
    block = fit.estimates.point.var"mu|class"
    @test block.crossed && length(block.effects) == 3
    @test all(isfinite, block.sd) && all(e -> all(isfinite, last(e)), block.effects)
    # A prepared model already fixes its groupings.
    @test_throws ArgumentError brm_nlme_fit(fit.model, NE.saem; options=CR_SAEM,
        CR_SHARED...)
end

@stestset "the two-stage fit holds the crossed block at its stage-1 estimate" begin
    rk = RKBRMI(crossed_plate(CR_PLATE))
    sampler(unit) = 0.1 .* randn(Xoshiro(unit.subject), LogDensityProblems.dimension(unit), 4)
    fit = brm_two_stage_fit(sampler, rk; threads=1, rng=Xoshiro(13), options=CR_SAEM,
        CR_SHARED...)
    m = fit.stage1.model
    @test size(fit.draws) == (length(m.view.coordinates), 4)
    @test all(isfinite, fit.draws)
    class = class_coordinates(m.view, rk)
    θ = fit.stage1.result.theta
    for c in class
        @test all(==(θ[findfirst(==(c), m.theta)]), fit.draws[c, :])
    end
end
