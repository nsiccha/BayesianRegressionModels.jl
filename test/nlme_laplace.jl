# brm_laplace_problem: every unit's random effects integrated out by a Laplace
# approximation, the population coordinates left for HMC.
#
# The marginal is checked against independent oracles written from the model's
# equations: for Gaussian likelihoods the exact marginal y_i ~ N(X_i β,
# Z_i Ω Z_iᵀ + σ² I), which the Laplace approximation must reproduce, and for a
# Poisson random intercept a scalar Newton search and second derivative per
# unit. The model's priors and log-Jacobians of the population coordinates come
# from RK's `:prior` and `:log_jacobian` queries, less the standardized draws'
# N(0, 1) terms.
#
# RUN: julia --project=test test/nlme_laplace.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics

function laplace_population_logprior(rk, p, u)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    draws = findall(==(:subject_effect), p.model.view.role)
    Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        sum(logpdf(Normal(), u[j]) for j in draws)
end

laplace_point(p, φ) = (u = zeros(length(p.model.unit)); u[p.population] .= φ; u)

# Unit labels in the view's order, and each unit's data rows.
laplace_rows(p, g) = [findall(==(string(level)), g) for level in p.model.view.levels]

const LAPLACE_RNG = Xoshiro(20261025)
const LAPLACE_G = vcat([fill("g$i", 2 + i % 3) for i in 1:6]...)
const LAPLACE_X = randn(LAPLACE_RNG, length(LAPLACE_G))
const LAPLACE_Y = let effects = Dict(g => 0.7 * randn(LAPLACE_RNG) for g in unique(LAPLACE_G))
    [0.5 + 0.8 * x + effects[g] + 0.4 * randn(LAPLACE_RNG)
        for (g, x) in zip(LAPLACE_G, LAPLACE_X)]
end

laplace_intercept_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    y ~ Normal(mu, sigma)
end

laplace_slope_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | g)
    y ~ Normal(mu, sigma)
end

# The unit design Z_i and random-effect covariance Ω of either Gaussian model,
# from the constrained values: η_i = diag(τ) L z_i.
function laplace_gaussian_parts(rk, u)
    values = BRM._rk_constrained_values(rk, u)
    τ = values.b_g.tau
    L = hasproperty(values.b_g, :L) ? values.b_g.L : Matrix(1.0I, 1, 1)
    D = Diagonal(τ)
    (; β=values.pop_mu.beta_pop, σ=values.sigma, Ω=D * L * L' * D,
       design=(x -> length(τ) == 1 ? [1.0] : [1.0, x]))
end

function laplace_gaussian_marginal(p, rk, data, φ)
    u = laplace_point(p, φ)
    (; β, σ, Ω, design) = laplace_gaussian_parts(rk, u)
    marginal = sum(laplace_rows(p, data.g)) do rows
        Z = reduce(vcat, (design(x)' for x in data.x[rows]))
        Σ = Symmetric(Z * Ω * Z' + σ^2 * I)
        logpdf(MvNormal(β[1] .+ β[2] .* data.x[rows], Σ), data.y[rows])
    end
    laplace_population_logprior(rk, p, u) + marginal
end

@stestset "a Gaussian model's Laplace marginal is its analytic marginal" begin
    data = (; g=LAPLACE_G, x=LAPLACE_X, y=LAPLACE_Y)
    for build in (laplace_intercept_model, laplace_slope_model)
        rk = RKBRMI(build(data))
        p = brm_laplace_problem(rk; ad_backend=NLME_AD)
        nη = length(first(p.model.view.subject_coordinates))
        @test length(p.population) == length(p.model.view.coordinates) - 6nη
        @test LogDensityProblems.dimension(p) == length(p.names)
        # The oracle's reading of the coordinates is the model's: its
        # conditional density at a random point equals the hand-written one.
        u = randn(LAPLACE_RNG, length(p.model.unit))
        (; β, σ, Ω, design) = laplace_gaussian_parts(rk, u)
        values = BRM._rk_constrained_values(rk, u)
        L = hasproperty(values.b_g, :L) ? values.b_g.L : Matrix(1.0I, 1, 1)
        g = similar(u)
        _, _, pointwise = BRM._rk_value_gradient_and_pointwise!(p.model.sampler, g, u)
        model = BRM._brm_nlme_sum_by_subject(p.model, pointwise)
        hand = map(enumerate(laplace_rows(p, data.g))) do (i, rows)
            η = Diagonal(values.b_g.tau) * L * u[p.model.view.subject_coordinates[i]]
            sum(logpdf(Normal(β[1] + β[2] * data.x[k] + dot(design(data.x[k]), η), σ),
                data.y[k]) for k in rows)
        end
        @test model ≈ hand rtol=1e-12
        for _ in 1:4
            φ = 0.5 .* randn(LAPLACE_RNG, length(p.population))
            @test LogDensityProblems.logdensity(p, φ) ≈
                laplace_gaussian_marginal(p, rk, data, φ) atol=1e-10
        end
    end
end

@stestset "a Poisson random intercept's marginal is the Laplace formula per unit" begin
    rng = Xoshiro(7)
    y = [rand(rng, Poisson(exp(0.3 + 0.5 * x))) for x in LAPLACE_X]
    data = (; g=LAPLACE_G, x=LAPLACE_X, y)
    rk = RKBRMI(@brm data begin
        log(mu) ~ 1 + x + (1 | g)
        y ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD)
    for _ in 1:3
        φ = 0.4 .* randn(rng, length(p.population))
        u = laplace_point(p, φ)
        values = BRM._rk_constrained_values(rk, u)
        β, τ = values.pop_log_mu.beta_pop, only(values.b_g.tau)
        # Per unit, ℓ(z) = Σ_k [y_k (a_k + τ z) − exp(a_k + τ z) − log y_k!] − z²/2,
        # maximized by scalar Newton; −ℓ''(z) = τ² Σ_k exp(a_k + τ z) + 1.
        units = sum(laplace_rows(p, data.g)) do rows
            a = β[1] .+ β[2] .* data.x[rows]
            ℓ(z) = sum(logpdf.(Poisson.(exp.(a .+ τ * z)), y[rows])) + logpdf(Normal(), z)
            z = 0.0
            for _ in 1:100
                rate = exp.(a .+ τ * z)
                z += (τ * sum(y[rows] .- rate) - z) / (τ^2 * sum(rate) + 1)
            end
            ℓ(z) + log(2π) / 2 - log(τ^2 * sum(exp.(a .+ τ * z)) + 1) / 2
        end
        @test LogDensityProblems.logdensity(p, φ) ≈
            laplace_population_logprior(rk, p, u) + units atol=1e-10
    end
    # Conditional draws by importance resampling follow each unit's exact
    # conditional posterior, here its mean by quadrature on a fine grid.
    φ = 0.4 .* randn(rng, length(p.population))
    u = laplace_point(p, φ)
    values = BRM._rk_constrained_values(rk, u)
    β, τ = values.pop_log_mu.beta_pop, only(values.b_g.tau)
    n = 2000
    out = brm_laplace_draws(p, repeat(φ, 1, n); rng=Xoshiro(9), conditional=64)
    @test size(out.diagnostics.importance_ess) == (length(p.model.view.levels), n)
    @test all(e -> 1 <= e <= 64, out.diagnostics.importance_ess)
    grid = range(-8, 8; length=4001)
    for (i, rows) in enumerate(laplace_rows(p, data.g))
        a = β[1] .+ β[2] .* data.x[rows]
        logpost = [sum(logpdf.(Poisson.(exp.(a .+ τ * z)), y[rows])) + logpdf(Normal(), z)
            for z in grid]
        w = exp.(logpost .- maximum(logpost))
        mean_z = sum(w .* grid) / sum(w)
        sd_z = sqrt(sum(w .* (grid .- mean_z) .^ 2) / sum(w))
        z = out.draws[only(p.model.view.subject_coordinates[i]), :]
        @test abs(mean(z) - mean_z) < 4 * sd_z / sqrt(n)
        @test std(z) ≈ sd_z rtol=0.1
    end
end

@stestset "Laplace draws keep the population draws and draw each unit from its Gaussian" begin
    data = (; g=LAPLACE_G, x=LAPLACE_X, y=LAPLACE_Y)
    rk = RKBRMI(laplace_intercept_model(data))
    p = brm_laplace_problem(rk; ad_backend=NLME_AD)
    φ = 0.3 .* randn(LAPLACE_RNG, length(p.population))
    n = 4000
    out = brm_laplace_draws(p, repeat(φ, 1, n); rng=Xoshiro(3))
    @test size(out.draws) == (length(p.model.unit), n)
    @test all(out.draws[p.population, :] .== φ)
    # Later draws start at the earlier draws' modes, which are this point's.
    @test all(==(0), out.diagnostics.newton_steps[2:end])
    @test rk_restore_draws(rk, out.draws) isa NamedTuple
    # A Gaussian unit's conditional posterior is exact: η_i | y_i has precision
    # n_i/σ² + 1/τ² and mean (Σ_k r_k / σ²) / precision, with z_i = η_i / τ.
    (; β, σ) = laplace_gaussian_parts(rk, laplace_point(p, φ))
    τ = only(BRM._rk_constrained_values(rk, laplace_point(p, φ)).b_g.tau)
    for (i, rows) in enumerate(laplace_rows(p, data.g))
        precision = length(rows) / σ^2 + 1 / τ^2
        mean_z = sum(data.y[rows] .- β[1] .- β[2] .* data.x[rows]) / σ^2 / precision / τ
        sd_z = 1 / sqrt(precision) / τ
        z = out.draws[only(p.model.view.subject_coordinates[i]), :]
        @test abs(mean(z) - mean_z) < 4 * sd_z / sqrt(n)
        @test std(z) ≈ sd_z rtol=0.1
    end
    # The Laplace Gaussian is a Gaussian unit's exact conditional, so every
    # importance weight is equal.
    conditional = brm_laplace_draws(p, repeat(φ, 1, 20); rng=Xoshiro(4), conditional=50)
    @test all(isapprox.(conditional.diagnostics.importance_ess, 50; rtol=1e-8))
    @test_throws ArgumentError brm_laplace_draws(p, repeat(φ, 1, 2); conditional=-1)
    @test_throws DimensionMismatch brm_laplace_draws(p, zeros(1, 2))
    @test_throws DimensionMismatch LogDensityProblems.logdensity(p, zeros(1))
end
