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
    others = [randn(rng, length(b), 1) for b in m.view.subject_coordinates]
    for i in 1:n
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(m.view.subject_coordinates[i])
        @test brm_nlme_unit_eta(p, ones(d)) ≈ cholesky(Symmetric(Ω)).L * ones(d)
        z1, z2 = 0.3 .* randn(rng, d), 0.3 .* randn(rng, d)
        point(z) = vec(brm_nlme_draws(m, θ, σ, ω,
            [j == i ? reshape(z, :, 1) : others[j] for j in 1:n]))
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
    @test size(draws) == (length(m.view.coordinates), 1)
    @test draws[m.theta, 1] ≈ θ
    @test draws[m.sigma, 1] ≈ σ
    # The readers take this orientation directly: one restored value per draw.
    @test length(rk_restore_draws(rk, draws).sigma) == 1
end

@stestset "wrong inputs are refused" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    m = nlme(rk)
    start = brm_nlme_start(m)
    θ, σ, ω, Ω = start.theta, start.sigma, start.omega, start.omega_matrix
    @test_throws BoundsError brm_nlme_unit_problem(m, 99, θ, σ, Ω)
    n = length(m.view.levels)
    ragged = [zeros(length(b), j == 1 ? 2 : 3) for (j, b) in enumerate(m.view.subject_coordinates)]
    @test_throws "the same draw count for every unit" brm_nlme_draws(m, θ, σ, ω, ragged)
    @test_throws "not both" brm_two_stage_fit(identity, rk; minibatch_size=2,
        options=NE.SAEMOptions())
end

function warmup_sampler(unit)
    run = WarmupHMC.adaptive_warmup_mcmc(Random.Xoshiro(unit.subject), unit)
    Matrix(run.posterior_position)
end

@stestset "two-stage fit: SAEM population, sampled units, RK packed draws" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    for minibatch_size in (nothing, 2)
        # SAEM draws from `rng`; NLMEEstimation defaults to the global RNG.
        fit = brm_two_stage_fit(warmup_sampler, rk; minibatch_size,
            rng=Random.Xoshiro(2026))
        m = fit.stage1.model
        @test fit.coordinates == coordinate_names(rk.model.layout)
        @test length(fit.units) == length(m.view.levels)
        S = size(first(fit.unit_draws), 2)
        @test S > 10
        @test size(fit.draws) == (length(fit.coordinates), S)
        @test all(isfinite, fit.draws)
        # Population coordinates are fixed at the SAEM estimates; units vary.
        @test all(fit.draws[m.theta, :] .== fit.stage1.result.theta)
        for (i, block) in enumerate(m.view.subject_coordinates)
            @test fit.draws[block, :] == fit.unit_draws[i]
            @test all(>(0), vec(std(fit.unit_draws[i]; dims=2)))
        end
        # Every draw is a finite point of the full RK model.
        full = rk_density(rk)
        @test all(isfinite(LogDensityProblems.logdensity(full, fit.draws[:, k])) for k in 1:S)
        # Readers take the draws as they come: one restored value per draw,
        # and σ fixed at the SAEM estimate.
        restored = rk_restore_draws(rk, fit.draws)
        @test length(restored.sigma) == S
        @test all(==(first(restored.sigma)), restored.sigma)
    end
end

@stestset "units sampled on several threads give the same draws" begin
    # Each unit's sampler is seeded by the unit, so the worker copies of the
    # model's cells must reproduce the single-threaded draws exactly, given
    # one shared stage-1 SAEM fit.
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    stage1 = brm_nlme_fit(rk, NE.saem; ebes=false, rng=Random.Xoshiro(2026))
    serial = brm_two_stage_fit(warmup_sampler, rk; stage1, threads=1)
    parallel = brm_two_stage_fit(warmup_sampler, rk; stage1, threads=3)
    @test serial.stage1 === stage1
    @test parallel.draws == serial.draws
    @test_throws "must be at least 1" brm_two_stage_fit(warmup_sampler, rk; stage1, threads=0)
    @test_throws "not both" brm_two_stage_fit(warmup_sampler, rk; stage1, minibatch_size=2)
end

@stestset "nested units: block-diagonal Ω over the unit's families" begin
    rk = RKBRMI(nested_model(NESTED_DATA))
    groups = (; subject_group=:subject, nested_groups=(unit=:obs_subject,))
    m = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), groups...)
    layout = NE.nlme_layout(m)
    θ, σ = [0.15], [log(0.7)]
    Ω = [reshape([0.49], 1, 1), reshape([0.36], 1, 1)]
    ω = NE.omega_parameters(layout, Ω)
    full = rk_density(rk)
    rng = Random.MersenneTwister(7)
    others = [randn(rng, length(b), 1) for b in m.view.subject_coordinates]
    n = length(m.view.levels)
    for i in 1:n
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(m.view.subject_coordinates[i])
        # One subject effect, then one effect per owned nested unit.
        @test diag(p.factor) ≈ sqrt.([0.49; fill(0.36, d - 1)])
        z1, z2 = 0.3 .* randn(rng, d), 0.3 .* randn(rng, d)
        point(z) = vec(brm_nlme_draws(m, θ, σ, ω,
            [j == i ? reshape(z, :, 1) : others[j] for j in 1:n]))
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(z1)) -
              LogDensityProblems.logdensity(full, point(z2)) atol=1e-9
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        step = 1e-6
        fd = [(LogDensityProblems.logdensity(p, z1 .+ step .* (1:d .== k)) -
               LogDensityProblems.logdensity(p, z1 .- step .* (1:d .== k))) / 2step
              for k in 1:d]
        @test gradient ≈ fd atol=1e-5 rtol=1e-5
    end
    # Nested units evaluate on their own cells, so each worker gets its own
    # compact samplers, planned from the reads `m` measured, and the draws
    # match one thread.
    @test m.cell_gradient !== nothing
    stage1 = brm_nlme_fit(rk, NE.saem; ebes=false, groups..., rng=Random.Xoshiro(2026),
        options=NE.SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=3, omega_floor=0.05))
    serial = brm_two_stage_fit(warmup_sampler, rk; stage1, threads=1)
    fit = brm_two_stage_fit(warmup_sampler, rk; stage1, threads=3)
    @test fit.draws == serial.draws
    @test size(fit.draws, 1) == length(m.view.coordinates)
    @test all(isfinite(LogDensityProblems.logdensity(full, fit.draws[:, k]))
              for k in 1:size(fit.draws, 2))
end
