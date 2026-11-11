#
# The linear Gaussian model gives exact oracles: each subject's conditional
# log likelihood is a Normal density, and the empirical Bayes mode of η is a
# linear solve.
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_known_blocks_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using NLMEEstimation, LinearAlgebra, Random, Statistics
const NE = NLMEEstimation

const KNOWN_SD = [0.6, 0.25]
const KNOWN_R = [1.0 0.3; 0.3 1.0]

function known_data(rng; N=8, n=5)
    subject = repeat(1:N, inner=n)
    x = round.(randn(rng, N * n); digits=3)
    A = Diagonal(KNOWN_SD) * cholesky(KNOWN_R).L
    b = [A * randn(rng, 2) for _ in 1:N]
    y = [1.0 + b[s][1] + (0.5 + b[s][2]) * x[i] + 0.3 * randn(rng)
        for (i, s) in enumerate(subject)]
    (; subject, x, y)
end

known_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, Intercept) = 0.6
    sd(mu, p, x) = 0.25
    cor(:, p) = [1.0 0.3; 0.3 1.0]
    y ~ Normal(mu, sigma)
end

const KNOWN_DATA = known_data(Xoshiro(20261108))
known_nlme() = brm_nlme_model(RKBRMI(known_model(KNOWN_DATA)); ad_backend=NLME_AD)

# The population coefficients and sigma at an NLME point, by name.
function known_population(m, θ, σ)
    named = brm_nlme_estimates(m, θ, σ, Matrix{Float64}(I, 2, 2))
    (; beta=[named.point.mu.Intercept, named.point.mu.x], sigma=named.point.sigma, named)
end

@stestset "the known block is a family fixed at the identity" begin
    m = known_nlme()
    family = only(m.view.families)
    @test family.fixed == Matrix{Float64}(I, 2, 2)
    fixed = NE.fixed_omega(m)
    @test first(only(fixed)) == 1
    @test last(only(fixed)).sd == [1.0, 1.0]
    @test last(only(fixed)).correlation == Matrix{Float64}(I, 2, 2)
    # Its η reach the predictor through the known `diag(sd) L`, so no
    # population coefficient is mu-referenced through them.
    @test isempty(m.view.mu_references)
    start = brm_nlme_start(m)
    @test start.omega_matrix == Matrix{Float64}(I, 2, 2)
    @test isfinite(start.population_logprior)
    # The estimates name the block with its known values.
    pop = known_population(m, start.theta, start.sigma)
    block = pop.named.point.p
    @test block.known === true
    @test block.sd ≈ KNOWN_SD
    @test block.cor ≈ KNOWN_R
    @test block.covariance ≈ Diagonal(KNOWN_SD) * KNOWN_R * Diagonal(KNOWN_SD)
end

@stestset "conditional likelihoods and empirical Bayes modes match the Normal oracle" begin
    m = known_nlme()
    start = brm_nlme_start(m)
    θ, σ, ω = start.theta, start.sigma, start.omega
    pop = known_population(m, θ, σ)
    A = Diagonal(KNOWN_SD) * cholesky(KNOWN_R).L
    rng = Xoshiro(4)
    H = 0.5 .* randn(rng, 2, length(m.view.levels))
    ll = brm_nlme_loglikelihoods(m, θ, σ, H)
    for (j, level) in enumerate(m.view.levels)
        rows = findall(==(level), KNOWN_DATA.subject)
        Z = hcat(ones(length(rows)), KNOWN_DATA.x[rows])
        effect = A * H[:, j]
        location = Z * (pop.beta .+ effect)
        oracle = sum(logpdf.(Normal.(location, pop.sigma), KNOWN_DATA.y[rows]))
        @test ll[j] ≈ oracle rtol = 1e-10
    end
    @test check_protocol(m, θ, σ, ω; η=H)
    ebe = empirical_bayes(m, θ, σ, ω)
    @test all(ebe.converged)
    for (k, j) in enumerate(ebe.subjects)
        rows = findall(==(m.view.levels[j]), KNOWN_DATA.subject)
        Z = hcat(ones(length(rows)), KNOWN_DATA.x[rows])
        r = KNOWN_DATA.y[rows] .- Z * pop.beta
        B = Z * A
        mode = (I + B' * B / pop.sigma^2) \ (B' * r / pop.sigma^2)
        @test ebe.modes[:, k] ≈ mode rtol = 1e-6 atol = 1e-8
    end
end

@stestset "FOCE and SAEM estimate the population with the block held" begin
    m = known_nlme()
    for estimator in (NE.focei, NE.saem)
        kwargs = estimator === NE.saem ? (; rng=Xoshiro(9)) : (;)
        fit = brm_nlme_fit(m, estimator; kwargs...)
        @test isfinite(fit.estimates.point.sigma)
        block = fit.estimates.point.p
        @test block.known === true
        @test block.covariance ≈ Diagonal(KNOWN_SD) * KNOWN_R * Diagonal(KNOWN_SD)
        println(nameof(estimator), ": mu = ", fit.estimates.point.mu, ", sigma = ",
            fit.estimates.point.sigma)
        @test abs(fit.estimates.point.mu.Intercept - 1.0) < 0.6
        @test abs(fit.estimates.point.mu.x - 0.5) < 0.4
    end
end
