# Model-scope observation factors keep every authored likelihood, including
# gathered nonlinear predictors and observations on different row axes.
# RUN: julia --project=test test/nlme_laplace_values.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

function values_fd(f, x; h=1e-5)
    map(eachindex(x)) do j
        e = zeros(length(x)); e[j] = h
        (f(x + e) - f(x - e)) / (2h)
    end
end

@stestset "non-centered shares retain completion density corrections" begin
    d = (; g=["a", "a", "b", "b"], ids=["a", "b"], total=[5.0, 4.0],
        x=Union{Missing,Float64}[missing, missing, missing, 1.0],
        y=[0.2, -0.4, 0.8, 0.3])
    rk = RKBRMI(@brm(d, begin
        eta ~ 1 + (1 | g)
        rows = groups(g => ids)
        mi(x)::shares(total, rows) ~ LogNormal(eta, 0.6)
        y ~ Normal(eta, 0.4)
    end); completions=:noncentered)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g)
    @test :x_shares_weights in (c.name for c in p.cells)
    @test length(only(c for c in p.cells if c.name === :x_shares_weights).cells) == 2
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    phi = 0.1 .* randn(Xoshiro(44), length(p.population))
    u = copy(p.u); u[p.population] .= phi; u[p.integrated] .= 0
    f(z) = (v = copy(u); v[p.integrated] .= z;
        LogDensityProblems.logdensity(full, v))
    n = length(p.integrated)
    z = zeros(n); f0 = f(z)
    grad = values_fd(f, z)
    H = zeros(n, n)
    for i in 1:n, j in 1:n
        ei = zeros(n); ei[i] = 0.1
        ej = zeros(n); ej[j] = 0.1
        H[i,j] = -(f(ei+ej) - f(ei-ej) - f(-ei+ej) + f(-ei-ej)) / 0.04
    end
    expected = f0 + dot(grad, H \ grad)/2 + n*log(2pi)/2 - logdet(H)/2
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
    @test value ≈ expected rtol=1e-8
    @test gradient ≈ values_fd(x -> LogDensityProblems.logdensity(p, x), phi) rtol=2e-6 atol=2e-6
end

@stestset "gathered model-value Gaussian observations preserve the exact marginal" begin
    d = (; g=["a", "b", "c"], idx=[1, 2, 1, 3],
        y=[0.2, -0.4, 0.8, 0.3], x=[0.3, -0.1, 0.5])
    rk = RKBRMI(@brm d begin
        eta ~ 1 + (1 | g)
        sigma ~ Exponential(1)
        mu = eta[idx]
        nu = 2 * eta
        y ~ Normal(mu, sigma)
        x ~ Normal(nu, sigma)
    end)
    @test BRM._brm_slap_response_refusal(rk) !== nothing
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=:g)
    @test Set(c.name for c in p.cells) == Set((:x, :y))
    @test sum(length(c.cells) for c in p.cells) == 7
    @test all(length(c.lz) == 1 for group in p.cells for c in group.cells)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    phi = 0.15 .* randn(Xoshiro(42), length(p.population))
    u = copy(p.u); u[p.population] .= phi; u[p.integrated] .= 0
    # The full conditional is a quadratic: recover its coefficients at zero
    # and integrate it independently of the sparse factor implementation.
    f(z) = (v = copy(u); v[p.integrated] .= z;
        LogDensityProblems.logdensity(full, v))
    n = length(p.integrated)
    z = zeros(n); f0 = f(z)
    grad = values_fd(f, z)
    H = zeros(n, n)
    for i in 1:n, j in 1:n
        ei = zeros(n); ei[i] = 0.1
        ej = zeros(n); ej[j] = 0.1
        H[i,j] = -(f(ei+ej) - f(ei-ej) - f(-ei+ej) + f(-ei-ej)) / 0.04
    end
    expected = f0 + dot(grad, H \ grad)/2 + n*log(2pi)/2 - logdet(H)/2
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
    @test value ≈ expected rtol=1e-8
    @test gradient ≈ values_fd(x -> LogDensityProblems.logdensity(p, x), phi) rtol=2e-6 atol=2e-6
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, draws=30)
    @test only(plan.terms).admitted
    @test only(plan.terms).importance_ess.min ≈ 30 rtol=1e-8
end

@stestset "derived count observations agree with the same likelihood in plate cells" begin
    d = (; g=["a", "b", "c"], idx=[1, 2, 1, 3], y=[2, 1, 3, 1])
    derived = RKBRMI(@brm d begin
        eta ~ 1 + (1 | g)
        shape ~ Exponential(1)
        mu = exp(eta[idx])
        y ~ NegativeBinomial2(mu, shape)
    end)
    plated = RKBRMI(@brm d begin
        eta ~ 1 + (1 | g)
        shape ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] ~ NegativeBinomial2(exp(eta[idx[i]]), shape)
        end
    end)
    p = brm_laplace_problem(derived; ad_backend=NLME_AD, integrate=:g)
    q = brm_laplace_problem(plated; ad_backend=NLME_AD, integrate=:g)
    @test p.names == q.names
    @test length(only(p.cells).cells) == length(d.y)
    phi = 0.1 .* randn(Xoshiro(43), length(p.population))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
    ref, refgradient = LogDensityProblems.logdensity_and_gradient(q, phi)
    @test value ≈ ref rtol=1e-10
    @test gradient ≈ refgradient rtol=1e-8 atol=1e-8
    @test gradient ≈ values_fd(x -> LogDensityProblems.logdensity(p, x), phi) rtol=2e-6 atol=2e-6
    plan = brm_laplace_plan(derived; ad_backend=NLME_AD, draws=30, ess_floor=0.01)
    @test only(plan.terms).class === :gated
    @test only(plan.terms).admitted
    @test occursin("model-value observations", only(plan.terms).reason)
    sampled = brm_laplace_plan(derived; ad_backend=NLME_AD, sample=:g)
    @test only(sampled.terms).class === :sampled
    @test !only(sampled.terms).admitted
end
