# RUN: julia --project=test test/cdar_laplace_coordinates.jl [testset-filter...]
# Independent Gaussian marginal and full-density coordinate oracles, nonlinear
# marginal derivatives, and sparse support as paths and group counts grow.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems, Random

const CDAR_STATE_DATA = let cells = [(g, t) for g in 1:3 for t in 1:6
        if (g, t) ∉ ((1, 2), (3, 1), (3, 6))]
    (; g=string.(first.(cells)), t=last.(cells),
       profile=string.(mod1.(last.(cells), 3)), y=0.4 .* sin.(eachindex(cells)))
end
const CDAR_STATE_NORMAL = @brm begin
    mu ~ 1 + cdar(t; by=g) + (1 | profile)
    y ~ Normal(mu, 0.7)
end
const CDAR_STATE_COUNT = @brm begin
    log(mu) ~ 1 + cdar(t; by=g) + (1 | profile)
    shape ~ LogNormal(0, 1)
    y ~ NegativeBinomial2(mu, shape)
end

state_fd(f, x; h=1e-5) = [(f(x + h * e) - f(x - h * e)) / (2h)
    for e in eachcol(Matrix{Float64}(I, length(x), length(x)))]
state_problem(rk; kwargs...) = brm_laplace_problem(rk; ad_backend=NLME_AD,
    integrate=(:g, :profile), solver=:exact, kwargs...)

@stestset "cdar state marginal, public coordinates, and chain copies" begin
    data = CDAR_STATE_DATA
    rk = RKBRMI(CDAR_STATE_NORMAL(data))
    p = state_problem(rk)
    public = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    field = rk_logdensity_problem(p.walk_frame.backend; ad_backend=NLME_AD)
    ports = BRM._rk_value_reader(rk, (Symbol("cdar_mu_t.sigma"),
        Symbol("cdar_mu_t.rho"), Symbol("b_profile.tau")))
    function oracle(x)
        u = zeros(length(p.u)); u[p.population] .= x
        sigma, rho, tau = ports(u)
        mu = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
        means = mu[p.entry]
        covariance = [
            (data.g[i] == data.g[j] ? sigma^2 * rho^abs(data.t[i] - data.t[j]) : 0.0) +
            (data.profile[i] == data.profile[j] ? only(tau)^2 : 0.0)
            for i in eachindex(data.y), j in eachindex(data.y)]
        covariance += 0.7^2 * I
        population = LogDensityProblems.logdensity(public, u) -
            sum(logpdf.(Normal.(means, 0.7), data.y)) + length(p.integrated) * log(2pi) / 2
        population + logpdf(MvNormal(means, Symmetric(covariance)), data.y)
    end
    rng = Xoshiro(20261127)
    for j in 1:3
        x = 0.3 .* randn(rng, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test value ≈ oracle(x) atol=1e-9
        @test gradient ≈ state_fd(oracle, x) atol=2e-6 rtol=2e-6
        u = 0.4 .* randn(rng, length(p.u)); u[p.population] .= x
        rho = ports(u)[2]
        public_u = BRM._brm_laplace_public_point!(p, copy(u))
        # det(d innovations / d states) = (1-rho²)^(-G(T-1)/2).
        logjac = -3 * (6 - 1) * log1p(-rho^2) / 2
        @test LogDensityProblems.logdensity(field, u) ≈
            LogDensityProblems.logdensity(public, public_u) + logjac atol=1e-9
        q = deepcopy(p)
        @test q.walk_frame.read[] === nothing
        @test q.walk_frame.pullback[] === nothing
        copy_value, copy_gradient = LogDensityProblems.logdensity_and_gradient(q, x)
        @test copy_value ≈ value
        @test copy_gradient ≈ gradient
        LogDensityProblems.logdensity_and_gradient(q, x .+ 0.2)
        repeated_value, repeated_gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test repeated_value ≈ value
        @test repeated_gradient ≈ gradient
    end
end

@stestset "cdar state precision moves in nonlinear exact and operator gradients" begin
    data = merge(CDAR_STATE_DATA, (; y=mod.(eachindex(CDAR_STATE_DATA.y), 5)))
    rk = RKBRMI(CDAR_STATE_COUNT(data))
    exact = state_problem(rk)
    operator = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:g, :profile),
        solver=:matrixfree)
    for p in (exact, operator)
        x = 0.15 .* randn(Xoshiro(119), length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test isfinite(value)
        @test gradient ≈ state_fd(x -> LogDensityProblems.logdensity(p, x), x) atol=3e-5 rtol=3e-5
    end
end

@stestset "cdar field supports fixed and derived persistence and a single step" begin
    fixed = @brm begin
        mu ~ 1 + cdar(t; by=g)
        ar(mu, cdar(t)) = 0.6
        y ~ Normal(mu, 0.7)
    end
    derived = @brm begin
        r ~ Beta(2, 3)
        mu ~ 1 + cdar(t; by=g)
        ar(mu, cdar(t)) = r
        y ~ Normal(mu, 0.7)
    end
    for model in (fixed, derived), data in (CDAR_STATE_DATA,
            merge(CDAR_STATE_DATA, (; t=ones(Int, length(CDAR_STATE_DATA.t)))))
        p = brm_laplace_problem(RKBRMI(model(data)); ad_backend=NLME_AD,
            integrate=:g, solver=:exact)
        x = 0.2 .* randn(Xoshiro(11), length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test isfinite(value)
        @test gradient ≈ state_fd(x -> LogDensityProblems.logdensity(p, x), x) atol=2e-6 rtol=2e-6
    end
end

@stestset "cdar state support grows linearly in groups and steps" begin
    # No timing assertion: the stored row reads and prior graph directly
    # detect a return to an all-step innovation clique.
    for (groups, steps) in ((2, 5), (2, 10), (4, 10))
        data = (; g=string.(repeat(1:groups; inner=steps)),
            t=repeat(1:steps; outer=groups), y=zeros(groups * steps))
        rk = RKBRMI(@brm data begin
            mu ~ 1 + cdar(t; by=g)
            y ~ Normal(mu, 0.7)
        end)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g, solver=:exact)
        @test length(p.integrated) == groups * steps
        @test maximum(diff(p.tptr)) == 1
        @test nnz(p.pattern) == groups * (3steps - 2)
        @test length(p.components) == groups
    end
end

@stestset "cdar field plan restriction and release retain public outer coordinates" begin
    data = merge(CDAR_STATE_DATA, (; y=mod.(eachindex(CDAR_STATE_DATA.y), 5)))
    rk = RKBRMI(CDAR_STATE_COUNT(data))
    for forced in (:g, :profile, (:g, :profile))
        plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=forced,
            draws=12, ess_floor=1.1, solver=:exact)
        reused = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        fresh = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=forced, solver=:exact)
        @test reused.names == fresh.names
        @test (reused.walk_frame === nothing) == (forced === :profile)
        x = 0.2 .* randn(Xoshiro(12), length(fresh.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(reused, x)
        fresh_value, fresh_gradient = LogDensityProblems.logdensity_and_gradient(fresh, x)
        @test value ≈ fresh_value atol=1e-9
        @test gradient ≈ fresh_gradient atol=1e-8
        @test gradient ≈ state_fd(x -> LogDensityProblems.logdensity(reused, x), x) atol=3e-6 rtol=3e-6
    end
end

@stestset "cdar latent summaries retain innovation means and variances" begin
    data = CDAR_STATE_DATA
    rk = RKBRMI(CDAR_STATE_NORMAL(data))
    exact = state_problem(rk)
    n = length(exact.integrated)
    operator = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:g, :profile),
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=n))
    # A complete orthogonal set of white probes removes Monte Carlo error
    # from the covariance-root variance estimate, leaving a dense oracle.
    @test length(operator.components) == 1
    operator.algebra.probes[1] .= sqrt(n) .* Matrix{Float64}(I, n, n)
    read_mu = BRM._rk_value_reader(rk, (:mu,))
    x = 0.15 .* randn(Xoshiro(28), length(exact.population))
    u = zeros(length(exact.u)); u[exact.population] .= x
    mu0 = copy(read_mu(u).mu)
    C = reduce(hcat, map(exact.integrated) do i
        shifted = copy(u); shifted[i] = 1.0
        read_mu(shifted).mu - mu0
    end)
    covariance = inv(Symmetric(I + C' * C / 0.7^2))
    expected_mean = covariance * C' * (data.y - mu0) / 0.7^2
    for p in (exact, operator), method in (:gaussian, :mean_corrected)
        LogDensityProblems.logdensity(p, x)
        summary = BRM._brm_lint_conditional(p, method)
        @test summary.mean ≈ expected_mean atol=1e-8
        @test summary.sd ≈ sqrt.(diag(covariance)) atol=2e-7
        @test summary.packed[p.integrated] ≈ summary.mean atol=1e-10
        latent = BRM._brm_lint_latent_marginals(p, reshape(x, :, 1), [1.0],
            Dict(x => summary), method)
        @test latent.names == BRM._rk_layout_coordinate_names(rk)[p.integrated]
    end
end
