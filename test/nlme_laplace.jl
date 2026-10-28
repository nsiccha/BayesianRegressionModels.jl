# brm_laplace_problem: every unit's random effects integrated out by a Laplace
# approximation, the population coordinates left for HMC.
#
# The marginal is checked against independent oracles written from the model's
# equations: for Gaussian likelihoods the exact marginal y_i ~ N(X_i β,
# Z_i Ω Z_iᵀ + σ² I), which the Laplace approximation must reproduce, and for a
# Poisson random intercept a scalar Newton search and second derivative per
# unit. The model's priors and log-Jacobians of the population coordinates come
# from RK's `:prior` and `:log_jacobian` queries, less the standardized draws'
# N(0, 1) terms. Every oracle runs against both implementations: the
# row-structured one (`BRMStructuredLaplaceProblem`, which these regression
# models get by default) and the whole-model one (`structured=false`).
#
# RUN: julia --project=test test/nlme_laplace.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics
using LogExpFunctions: logit

const LAPLACE_STRUCTURED = (true, false)

laplace_view(p::BRMLaplaceProblem) = p.model.view
laplace_view(p::BRMStructuredLaplaceProblem) = p.view
laplace_problem(rk, structured) =
    brm_laplace_problem(rk; ad_backend=NLME_AD, structured)

function laplace_population_logprior(rk, p, u)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    draws = findall(==(:subject_effect), laplace_view(p).role)
    Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        sum(logpdf(Normal(), u[j]) for j in draws)
end

laplace_point(p, φ) = (u = zeros(length(p.u)); u[p.population] .= φ; u)

# Unit labels in the view's order, and each unit's data rows.
laplace_rows(p, g) = [findall(==(string(level)), g) for level in laplace_view(p).levels]

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
    for build in (laplace_intercept_model, laplace_slope_model),
            structured in LAPLACE_STRUCTURED
        rk = RKBRMI(build(data))
        p = laplace_problem(rk, structured)
        @test p isa (structured ? BRMStructuredLaplaceProblem : BRMLaplaceProblem)
        nη = length(first(laplace_view(p).subject_coordinates))
        @test length(p.population) == length(laplace_view(p).coordinates) - 6nη
        @test LogDensityProblems.dimension(p) == length(p.names)
        for _ in 1:4
            φ = 0.5 .* randn(LAPLACE_RNG, length(p.population))
            @test LogDensityProblems.logdensity(p, φ) ≈
                laplace_gaussian_marginal(p, rk, data, φ) atol=1e-10
        end
        structured && continue
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
    for structured in LAPLACE_STRUCTURED
        p = laplace_problem(rk, structured)
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
        # A population point where the rates overflow scores -Inf, as the full
        # model's density there does, and is counted. Without previous modes (no
        # predictor) the whole-model search starts at zero, where they overflow;
        # the structured one sees the overflow in the rows' parameters.
        overflow = zeros(length(p.population))
        overflow[findfirst(==(Symbol("pop_log_mu.beta_pop.1")), p.names)] = 800.0
        if !structured
            p.u .= p.model.unit
            p.factors[] = nothing
        end
        before = p.nonfinite
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, overflow)
        @test value == -Inf && all(isnan, gradient)
        @test p.nonfinite == before + 1
        @test !isfinite(LogDensityProblems.logdensity(
            rk_logdensity_problem(rk; ad_backend=NLME_AD), laplace_point(p, overflow)))
        if !structured
            # From modes at an ordinary point the predictor starts far up the
            # rates' exponential, where Newton gains about one unit per step: the
            # search does not converge, scores -Inf, is counted and warns.
            LogDensityProblems.logdensity(p, zeros(length(p.population)))
            before = p.unconverged
            value = @test_logs (:warn, r"did not converge") match_mode=:any LogDensityProblems.logdensity(p, overflow)
            @test value == -Inf
            @test p.unconverged == before + 1
        end
        # Conditional draws by importance resampling follow each unit's exact
        # conditional posterior, here its mean by quadrature on a fine grid.
        φ = 0.4 .* randn(rng, length(p.population))
        u = laplace_point(p, φ)
        values = BRM._rk_constrained_values(rk, u)
        β, τ = values.pop_log_mu.beta_pop, only(values.b_g.tau)
        n = 2000
        out = brm_laplace_draws(p, repeat(φ, 1, n); rng=Xoshiro(9), conditional=64)
        @test size(out.diagnostics.importance_ess) == (length(laplace_view(p).levels), n)
        @test all(e -> 1 <= e <= 64, out.diagnostics.importance_ess)
        grid = range(-8, 8; length=4001)
        for (i, rows) in enumerate(laplace_rows(p, data.g))
            a = β[1] .+ β[2] .* data.x[rows]
            logpost = [sum(logpdf.(Poisson.(exp.(a .+ τ * z)), y[rows])) + logpdf(Normal(), z)
                for z in grid]
            w = exp.(logpost .- maximum(logpost))
            mean_z = sum(w .* grid) / sum(w)
            sd_z = sqrt(sum(w .* (grid .- mean_z) .^ 2) / sum(w))
            z = out.draws[only(laplace_view(p).subject_coordinates[i]), :]
            @test abs(mean(z) - mean_z) < 4 * sd_z / sqrt(n)
            @test std(z) ≈ sd_z rtol=0.1
        end
    end
end

@stestset "Laplace draws keep the population draws and draw each unit from its Gaussian" begin
    data = (; g=LAPLACE_G, x=LAPLACE_X, y=LAPLACE_Y)
    rk = RKBRMI(laplace_intercept_model(data))
    for structured in LAPLACE_STRUCTURED
        p = laplace_problem(rk, structured)
        φ = 0.3 .* randn(LAPLACE_RNG, length(p.population))
        n = 4000
        out = brm_laplace_draws(p, repeat(φ, 1, n); rng=Xoshiro(3))
        @test size(out.draws) == (length(p.u), n)
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
            z = out.draws[only(laplace_view(p).subject_coordinates[i]), :]
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
end

# Fourth-order central differences of `f` at `φ`.
laplace_fd(f, φ; h=1e-3) = map(eachindex(φ)) do j
    e = (1:length(φ)) .== j
    (-f(φ .+ 2h .* e) + 8f(φ .+ h .* e) - 8f(φ .- h .* e) + f(φ .- 2h .* e)) / 12h
end

@stestset "the Laplace gradient follows the implicit-function theorem" begin
    # Gaussian: the gradient of the analytic marginal.
    data = (; g=LAPLACE_G, x=LAPLACE_X, y=LAPLACE_Y)
    rk = RKBRMI(laplace_slope_model(data))
    for structured in LAPLACE_STRUCTURED, _ in 1:2
        p = laplace_problem(rk, structured)
        φ = 0.4 .* randn(LAPLACE_RNG, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test value ≈ laplace_gaussian_marginal(p, rk, data, φ) atol=1e-10
        @test gradient ≈ laplace_fd(ψ -> laplace_gaussian_marginal(p, rk, data, ψ), φ) rtol=1e-7 atol=1e-7
    end
    # Count likelihoods, where the unit Hessians move with the modes: the
    # gradient of the Laplace marginal itself.
    rng = Xoshiro(13)
    η = [0.3 + 0.5 * x + 0.5 * randn(rng) for x in LAPLACE_X]
    counts = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + exp(e)))) for e in η]
    successes = [rand(rng, Binomial(12, 1 / (1 + exp(0.5 - e)))) for e in η]
    models = (
        @brm((; g=LAPLACE_G, x=LAPLACE_X, y=counts), begin
            phi ~ Exponential(1)
            log(mu) ~ 1 + x + (1 + x | g)
            y ~ NegativeBinomial2(mu, phi)
        end),
        @brm((; g=LAPLACE_G, x=LAPLACE_X, H=successes, n=fill(12, length(successes))), begin
            logit(p) ~ 1 + x + (1 | g)
            H ~ Binomial(n, p)
        end),
    )
    for brmi in models, structured in LAPLACE_STRUCTURED
        p = laplace_problem(RKBRMI(brmi), structured)
        φ = 0.3 .* randn(rng, length(p.population))
        _, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test gradient ≈ laplace_fd(ψ -> LogDensityProblems.logdensity(p, ψ), φ) rtol=1e-7 atol=1e-7
    end
end

@stestset "a unit density that is not concave everywhere still finds its mode" begin
    # A random effect on a negative-binomial shape makes a unit's density
    # non-concave away from its mode; the mode search then steps along −H
    # with reflected, floored eigenvalues.
    rng = Xoshiro(17)
    η = [0.8 + 0.4 * x + 0.5 * randn(rng) for x in LAPLACE_X]
    counts = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(e)))) for e in η]
    rk = RKBRMI(@brm (; g=LAPLACE_G, x=LAPLACE_X, y=counts) begin
        log(mu) ~ 1 + x + (1 | um | g)
        log(shape) ~ 1 + (1 | us | g)
        y ~ NegativeBinomial2(mu, shape)
    end)
    generic = laplace_problem(rk, false)
    scales = findall(n -> occursin(".tau.", String(n)), generic.names)
    @test length(scales) == 2
    indefinite(φ) = any(B -> !isposdef(Symmetric(-B)),
        BRM._brm_nlme_sampler_hessian_blocks(generic.model, laplace_point(generic, φ)))
    candidates = [(φ = zeros(length(generic.population)); φ[scales] .= s; φ)
        for s in (1.0, 2.0, 3.0, 4.0)]
    φ = candidates[findfirst(indefinite, candidates)]
    @test indefinite(φ)
    for structured in LAPLACE_STRUCTURED
        p = laplace_problem(rk, structured)
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test isfinite(value) && all(isfinite, gradient)
        @test p.nonfinite == 0
        @test gradient ≈ laplace_fd(ψ -> LogDensityProblems.logdensity(p, ψ), φ; h=1e-4) rtol=1e-6 atol=1e-6
    end
end

@stestset "the row-structured Laplace is the whole-model one, and who gets it" begin
    # Two count outcomes on one panel, one with a per-row shape predictor and
    # its own random effect, a binomial share with trials, and correlated
    # effects on another: every family slot of the structured path at once.
    rng = Xoshiro(19)
    e = Dict(g => 0.4 .* randn(rng, 3) for g in unique(LAPLACE_G))
    works = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + exp(0.8 + 0.3x + e[g][1]))))
        for (g, x) in zip(LAPLACE_G, LAPLACE_X)]
    cites = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(1.0 + e[g][2]))))
        for g in LAPLACE_G]
    shares = [rand(rng, Binomial(w, 1 / (1 + exp(0.3 - e[g][3]))))
        for (g, w) in zip(LAPLACE_G, works)]
    data = (; g=LAPLACE_G, x=LAPLACE_X, works, cites, shares, trials=copy(works))
    panel = @brm data begin
        phi_w ~ Exponential(1)
        log(mu_w) ~ 1 + x + (1 + x | w | g)
        works ~ NegativeBinomial2(mu_w, phi_w)
        log(mu_c) ~ 1 + (1 | c | g)
        log(shape_c) ~ 1 + (1 | s | g)
        cites ~ NegativeBinomial2(mu_c, shape_c)
        logit(p_s) ~ 1 + x + (1 | sh | g)
        shares ~ Binomial(trials, p_s)
    end
    rk = RKBRMI(panel)
    structured = laplace_problem(rk, :auto)
    @test structured isa BRMStructuredLaplaceProblem
    generic = laplace_problem(rk, false)
    # The whole-model gradient's third-order contractions do not differentiate
    # this panel's correlated block yet (Enzyme snag `nested-forward-o-9434176c`:
    # the forward `gemm` rule's `dlascl` under an outer reverse sweep), so the
    # structured gradient is checked against differences of its value, which
    # equals the whole-model value.
    for _ in 1:3
        φ = 0.3 .* randn(rng, length(generic.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(structured, φ)
        @test value ≈ LogDensityProblems.logdensity(generic, φ) atol=1e-9
        @test gradient ≈ laplace_fd(ψ -> LogDensityProblems.logdensity(structured, ψ), φ) rtol=1e-6 atol=1e-6
    end
    # Families outside the row structure keep the whole-model path, and
    # requiring the structured one names why.
    lognormal = RKBRMI(@brm (; g=LAPLACE_G, x=LAPLACE_X, y=exp.(LAPLACE_Y)) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ LogNormal(mu, sigma)
    end)
    @test laplace_problem(lognormal, :auto) isa BRMLaplaceProblem
    @test_throws BRM.BRMNLMEViewError laplace_problem(lognormal, true)
    @test_throws ArgumentError laplace_problem(lognormal, :yes)
    # So do observations that are not regression responses: a `@plate` cell's,
    # or one whose distribution reads a model value. The row structure has no
    # terms for them; before, :auto built it without them, a marginal that left
    # those observations out.
    for brmi in (nlme_plate_pk(NLME_PLATE_DATA), nlme_pk(NLME_DATA))
        rk = RKBRMI(brmi)
        @test laplace_problem(rk, :auto) isa BRMLaplaceProblem
        @test_throws "is not a regression response" laplace_problem(rk, true)
    end
end

@stestset "parameters not affine in the population coordinates are pulled back through RK" begin
    # A monotonic effect's coefficient times its simplex is not affine in the
    # packed coordinates, so the structured gradient pulls both NB2
    # parameters back through RK in one sweep instead of a constant Jacobian.
    rng = Xoshiro(23)
    rank = [1 + j % 3 for j in eachindex(LAPLACE_G)]
    e = Dict(g => 0.5 * randn(rng) for g in unique(LAPLACE_G))
    y = [rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + exp(0.6 + 0.4r + e[g]))))
        for (g, r) in zip(LAPLACE_G, rank)]
    rk = RKBRMI(@brm (; g=LAPLACE_G, x=LAPLACE_X, rank, y) begin
        log(mu) ~ 1 + mo(rank) + (1 | g)
        log(shape) ~ 1 + x
        y ~ NegativeBinomial2(mu, shape)
    end)
    structured = laplace_problem(rk, true)
    @test !structured.affine
    generic = laplace_problem(rk, false)
    for _ in 1:3
        φ = 0.3 .* randn(rng, length(generic.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(structured, φ)
        @test value ≈ LogDensityProblems.logdensity(generic, φ) atol=1e-9
        @test gradient ≈ last(LogDensityProblems.logdensity_and_gradient(generic, φ)) rtol=1e-7 atol=1e-7
        @test gradient ≈ laplace_fd(ψ -> LogDensityProblems.logdensity(structured, ψ), φ) rtol=1e-6 atol=1e-6
    end
end
