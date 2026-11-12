# Public synthetic random-intercept regression from brm-svi-fit-defa-7beb59f3.
# Population uncertainty is checked against independently sampled NUTS, in
# weak, strong and unbalanced groups. The fixed-scale Gaussian width oracle
# lives in nlme_svi_centering_algebra.jl.
# RUN: julia --project=test test/nlme_svi_default_centering.jl [filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using Random, Statistics, WarmupHMC, TreeArrays, MCMCDiagnosticTools

function default_centering_data(counts; seed=1)
    rng = Xoshiro(seed)
    u = randn(rng, length(counts))
    g = reduce(vcat, [fill(i, n) for (i, n) in enumerate(counts)])
    (; g, y=0.5 .+ u[g] .+ randn(rng, length(g)), prior_sd=5.0)
end

default_centering_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, prior_sd)
    y ~ Normal(mu, sigma)
end

@stestset "default centering: disjoint slices and factor-level starts" begin
    data = default_centering_data(fill(4, 8))
    full = RKBRMI(default_centering_model(data))
    frozen = brm_fitted_state(full)
    slices = map((1:4, 5:8)) do labels
        rows = in.(data.g, Ref(labels))
        subset = (; g=data.g[rows], y=data.y[rows], data.prior_sd)
        rk = RKBRMI(default_centering_model(subset), frozen; resample_groups=(:g,))
        brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(4), max_iterations=200, ndraws=0)
    end
    start = brm_svi_combine(full, collect(slices); subject_group=:g)
    @test start.receipt.centering.slices == [true, true]
    @test length(start.units) == 8
    for centering in (LearnedCentering(), nothing)
        from_start = brm_svi_fit(full; subject_group=:g, start, centering,
            max_iterations=0, ndraws=10, rng=Xoshiro(5))
        from_slices = brm_svi_fit(full; subject_group=:g, start=collect(slices), centering,
            max_iterations=0, ndraws=10, rng=Xoshiro(5))
        @test from_start.draws == from_slices.draws
        @test all(isfinite, from_start.draws)
        @test (from_start.diagnostics.centering === nothing) == (centering === nothing)
    end
end

@stestset "default centering: no eligible cells remains a valid fit" begin
    data = merge(default_centering_data(fill(3, 4)), (; known_sd=1.0, known_sigma=1.0))
    model = @brm data begin
        mu ~ 1 + (1 | g)
        sd(mu, g) = known_sd
        y ~ Normal(mu, known_sigma)
    end
    fit = brm_svi_fit(model; subject_group=:g, max_iterations=2, ndraws=5)
    @test isempty(fit.diagnostics.centering.cells)
    @test !isempty(fit.diagnostics.centering.uncentered)
    @test all(isfinite, fit.draws)
end

# Keep chain and draw axes through the named reductions.
function population_summary(draws)
    tree = ndims(draws) == 1 ? TreeData(draws, :draw) : TreeData(draws, :draw, :chain)
    dims = ndims(draws) == 1 ? :draw : (:draw, :chain)
    (; mean=parent(mean(tree; dims)), sd=parent(std(tree; dims)))
end

function check_default_population(label, counts)
        rk = RKBRMI(default_centering_model(default_centering_data(counts)))
        fit = brm_svi_fit(rk; subject_group=:g, batch_size=8, rng=Xoshiro(2), ndraws=4000)
        @test fit.diagnostics.centering !== nothing
        @test length(fit.diagnostics.centering.cells) == length(counts)
        idx = only(findall(==(:mu_Intercept), fit.coordinates))
        problems = [rk_logdensity_problem(rk; ad_backend=NLME_AD) for _ in 1:4]
        init = brm_start(rk; chains=4, rng=Xoshiro(3))
        reference = completion_warmup_mcmc([Xoshiro(10 + c) for c in 1:4], problems;
            min_completed=4, n_draws=1000, blas_threads=1, init)
        @test reference.completion.n_completed == 4
        @test reference.completion.n_omitted == 0
        chains = hcat([Matrix{Float64}(r.posterior_position)[idx, :]
            for r in reference.results]...)
        diagnostic = ess_rhat(reshape(chains, size(chains)..., 1))
        rhat, ess = only(diagnostic.rhat), only(diagnostic.ess)
        divergent = sum(r.n_divergent_samples for r in reference.results)
        @test rhat < 1.05
        @test ess > 100
        @test divergent / length(chains) < 0.01
        nuts, svi = population_summary(chains), population_summary(fit.draws[idx, :])
        ratio, error = svi.sd / nuts.sd, abs(svi.mean - nuts.mean) / nuts.sd
        println("DEFAULT_CENTERING case=$label mean_error=$error sd_ratio=$ratio " *
            "svi_sd=$(svi.sd) nuts_sd=$(nuts.sd) iterations=$(fit.diagnostics.iterations) " *
            "converged=$(fit.diagnostics.converged) nuts_rhat=$rhat nuts_ess=$ess " *
            "nuts_divergent=$divergent")
        @test error < 0.6
        @test 0.6 < ratio < 1.4
        if label == "strong"
            plain = brm_svi_fit(rk; subject_group=:g, batch_size=8, rng=Xoshiro(2),
                ndraws=4000, centering=nothing)
            ps = population_summary(plain.draws[idx, :])
            println("DEFAULT_CENTERING plain_sd_ratio=$(ps.sd / nuts.sd) " *
                "plain_sd=$(ps.sd) converged=$(plain.diagnostics.converged)")
            @test plain.diagnostics.centering === nothing
            @test ps.sd / nuts.sd < 0.25
            @test svi.sd > 3ps.sd
        end
end

@stestset "default centering: strong population intervals against NUTS" begin
    check_default_population("strong", fill(50, 32))
end
@stestset "default centering: weak population intervals against NUTS" begin
    check_default_population("weak", fill(1, 32))
end
@stestset "default centering: unbalanced population intervals against NUTS" begin
    check_default_population("unbalanced", repeat([1, 50], 16))
end
