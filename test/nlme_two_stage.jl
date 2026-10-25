# test/nlme_two_stage.jl — one unit's conditional posterior
# (`brm_nlme_unit_problem`), full RK packed draws from unit draws
# (`brm_nlme_draws`) and the two-stage fit (`brm_two_stage_fit`: SAEM for the
# population, then every unit's conditional posterior sampled on its own).
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_two_stage.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using NLMEEstimation, LogDensityProblems, LinearAlgebra, Random, Statistics
import WarmupHMC
const NE = NLMEEstimation

nlme(rk) = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
rk_density(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

@stestset "a unit problem is the RK density in BRM's subject coordinates" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    m = nlme(rk)
    start = brm_nlme_start(m)
    θ, σ, ω, Ω = start.theta, start.sigma, start.omega, start.omega_matrix
    n = length(m.view.levels)
    full = rk_density(rk)
    rng = Random.MersenneTwister(4)
    others = [randn(rng, 1, length(b)) for b in m.view.subject_coordinates]
    for i in 1:n
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(m.view.subject_coordinates[i])
        @test brm_nlme_unit_eta(p, ones(d)) ≈ cholesky(Symmetric(Ω)).L * ones(d)
        z1, z2 = 0.3 .* randn(rng, d), 0.3 .* randn(rng, d)
        point(z) = vec(brm_nlme_draws(m, θ, σ, ω,
            [j == i ? permutedims(z) : others[j] for j in 1:n]))
        # Only unit i changes: the difference of the full RK density is the
        # difference of the unit problem's.
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(z1)) -
              LogDensityProblems.logdensity(full, point(z2)) atol=1e-9
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        @test value ≈ LogDensityProblems.logdensity(p, z1)
        step = 1e-6
        fd = [(LogDensityProblems.logdensity(p, z1 .+ step .* (1:d .== k)) -
               LogDensityProblems.logdensity(p, z1 .- step .* (1:d .== k))) / 2step
              for k in 1:d]
        @test gradient ≈ fd atol=1e-5 rtol=1e-5
    end
    # The population coordinates of an assembled draw are the fixed point's.
    draws = brm_nlme_draws(m, θ, σ, ω, others)
    @test size(draws) == (1, length(m.view.coordinates))
    @test draws[1, m.theta] ≈ θ
    @test draws[1, m.sigma] ≈ σ
end

@stestset "wrong inputs are refused" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    m = nlme(rk)
    start = brm_nlme_start(m)
    θ, σ, ω, Ω = start.theta, start.sigma, start.omega, start.omega_matrix
    @test_throws BoundsError brm_nlme_unit_problem(m, 99, θ, σ, Ω)
    n = length(m.view.levels)
    ragged = [zeros(j == 1 ? 2 : 3, length(b)) for (j, b) in enumerate(m.view.subject_coordinates)]
    @test_throws "the same draw count for every unit" brm_nlme_draws(m, θ, σ, ω, ragged)
    @test_throws "not both" brm_two_stage_fit(identity, rk; minibatch_size=2,
        options=NE.SAEMOptions())
end

function warmup_sampler(unit)
    run = WarmupHMC.adaptive_warmup_mcmc(Random.Xoshiro(unit.subject), unit)
    permutedims(Matrix(run.posterior_position))
end

@stestset "two-stage fit: SAEM population, sampled units, RK packed draws" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    for minibatch_size in (nothing, 2)
        fit = brm_two_stage_fit(warmup_sampler, rk; minibatch_size)
        m = fit.stage1.model
        @test fit.coordinates == coordinate_names(rk.model.layout)
        @test length(fit.units) == length(m.view.levels)
        S = size(first(fit.unit_draws), 1)
        @test S > 10
        @test size(fit.draws) == (S, length(fit.coordinates))
        @test all(isfinite, fit.draws)
        # Population coordinates are fixed at the SAEM estimates; units vary.
        @test all(fit.draws[:, m.theta] .== permutedims(fit.stage1.result.theta))
        for (i, block) in enumerate(m.view.subject_coordinates)
            @test fit.draws[:, block] == fit.unit_draws[i]
            @test all(>(0), vec(std(fit.unit_draws[i]; dims=1)))
        end
        # Every draw is a finite point of the full RK model.
        full = rk_density(rk)
        @test all(isfinite(LogDensityProblems.logdensity(full, fit.draws[k, :])) for k in 1:S)
    end
end
