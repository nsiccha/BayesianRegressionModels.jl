# Model-scope observation factors keep every authored likelihood, including
# gathered nonlinear predictors and observations on different row axes.
# RUN: julia --project=test test/nlme_laplace_values.jl
include(joinpath(@__DIR__, "fitting_packages.jl"))
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

function values_fd(f, x; h=1e-5)
    map(eachindex(x)) do j
        e = zeros(length(x)); e[j] = h
        (f(x + e) - f(x - e)) / (2h)
    end
end

@stestset "computed NB2 shape uses regression rows with a fixed conditional dispersion" begin
    d = (; g=repeat(["a", "b", "c"], 3),
        exposure=[1, 4, 2, 3, 2, 6, 1, 2, 3], y=[0, 4, 2, 1, 3, 3, 0, 2, 2])
    rk = RKBRMI(@brm d begin
        log(mu) ~ 1 + (1 | g)
        sd(mu, g) = 0.8
        a ~ LogNormal(0, 1)
        b ~ LogNormal(0, 1)
        y ~ NegativeBinomial2(mu, 1 ./ (1 ./ (a .* exposure) .+ 1 ./ b))
    end)
    p = brm_laplace_problem(rk; integrate=:g, ad_backend=NLME_AD)
    named = RKBRMI(@brm d begin
        log(mu) ~ 1 + (1 | g)
        sd(mu, g) = 0.8
        a ~ LogNormal(0, 1)
        b ~ LogNormal(0, 1)
        shape = 1 ./ (1 ./ (a .* exposure) .+ 1 ./ b)
        y ~ NegativeBinomial2(mu, shape)
    end)
    q = brm_laplace_problem(named; integrate=:g, ad_backend=NLME_AD)
    @test p.backend === rk
    @test length(p.integrated) == 3
    @test length(p.family) == length(q.family) == length(d.y)
    @test isempty(p.cells) && isempty(q.cells)
    @test p.names == q.names && p.integrated == q.integrated
    @test BRM._brm_spl_concave(p)

    # Each group has a single standard-normal draw. Solve its score equation
    # and integrate the curvature against the authored joint density.
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    params = BRM._rk_value_reader(rk, (:mu, :a, :b))
    function oracle(x)
        u = zeros(length(p.u))
        u[p.population] .= x
        law = params(u)
        shape = @. 1 / (1 / (law.a * d.exposure) + 1 / law.b)
        correction = 0.0
        for (j, group) in enumerate(sort(unique(d.g)))
            rows = findall(==(group), d.g)
            z = 0.0
            for iteration in 1:20
                mu = law.mu[rows] .* exp(0.8z)
                probability = mu ./ (shape[rows] .+ mu)
                count_shape = d.y[rows] .+ shape[rows]
                score = 0.8 * sum(d.y[rows] .- count_shape .* probability) - z
                precision = 1 + 0.8^2 * sum(count_shape .* probability .* (1 .- probability))
                if abs(score) < 1e-12
                    correction += 0.5 * log(2pi / precision)
                    break
                end
                iteration == 20 && error("scalar NB2 reference mode did not converge")
                z += score / precision
            end
            u[p.integrated[j]] = z
        end
        LogDensityProblems.logdensity(full, u) + correction
    end
    rng = Xoshiro(47)
    for x in (zeros(length(p.names)), 0.3 .* randn(rng, length(p.names)))
        v, g = LogDensityProblems.logdensity_and_gradient(p, x)
        @test v ≈ oracle(x) atol=1e-8
        @test g ≈ values_fd(oracle, x) rtol=2e-7 atol=2e-8
        w, h = LogDensityProblems.logdensity_and_gradient(q, x)
        @test w ≈ v atol=1e-9
        @test h ≈ g atol=1e-9
    end
    latent_shape = RKBRMI(@brm d begin
        log(mu) ~ 1 + (1 | g)
        a ~ LogNormal(0, 1)
        y ~ NegativeBinomial2(mu, a .* mu)
    end)
    records = BRM._brm_spl_records(latent_shape, [:g])
    @test isempty(last(BRM._brm_spl_fixed_scale_rows(latent_shape, records)))
end

@stestset "computed NB2 promotion distinguishes arithmetic and kernel shapes in one model" begin
    d = (; g=["a", "b", "c"], exposure=[2, 1, 3], y=[1, 0, 2], other=[2, 1, 0])
    rk = RKBRMI(@brm d begin
        log(mu) ~ 1 + (1 | g)
        a ~ LogNormal(0, 1)
        shape = a .* exposure
        kernel_shape ~ kernel(mu) do m
            return 1 + m
        end
        y ~ NegativeBinomial2(mu, shape)
        other ~ NegativeBinomial2(mu, kernel_shape)
    end)
    records = BRM._brm_spl_records(rk, [:g])
    _, responses, promoted = BRM._brm_spl_fixed_scale_rows(rk, records)
    @test promoted == [:y]
    @test only(responses).scale === :shape
end

@stestset "affine predictor offsets and noisy Gaussian measurements share sparse rows" begin
    d = (; site=["a", "b", "a", "c"], reading=[0.4, -0.8, 0.6, 0.1],
        y=[0, 3, 2, 1], variance=[0.2, 0.3, 0.1, 0.4], flag=[0, 1, 0, 1],
        exposure=[1, 2, 1, 3])
    rk = RKBRMI(@brm d begin
        level ~ 1 + (1 | site)
        sd(level, site) = 0.7
        log(mu) ~ 1 + offset(level)
        tau ~ Exponential(1)
        a ~ LogNormal(0, 1)
        noise = sqrt.(variance .+ flag .* tau .* tau)
        shape = a .* exposure
        reading .~ Normal.(level, noise)
        y ~ NegativeBinomial2(mu, shape)
    end)
    p = brm_laplace_problem(rk; integrate=:site, ad_backend=NLME_AD)
    @test isempty(p.cells)
    @test count(==(:reading), p.response) == count(==(:y), p.response) == length(d.y)
    @test BRM._brm_spl_concave(p)

    # Evaluate the authored joint at independent scalar modes. Both the
    # Gaussian measurement and the count contribute to each site's precision.
    joint = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    reader = BRM._rk_value_reader(rk, (:level, :mu, :noise, :shape))
    function oracle(x)
        u = zeros(length(p.u))
        u[p.population] .= x
        law = reader(u)
        correction = 0.0
        for (j, site) in enumerate(sort(unique(d.site)))
            rows = findall(==(site), d.site)
            z = 0.0
            for iter in 1:30
                mu = law.mu[rows] .* exp(0.7z)
                prob = mu ./ (law.shape[rows] .+ mu)
                dispersion = law.shape[rows] .+ d.y[rows]
                residual = d.reading[rows] .- law.level[rows] .- 0.7z
                prec = 1 ./ abs2.(law.noise[rows])
                score = 0.7 * sum(d.y[rows] .- dispersion .* prob .+ residual .* prec) - z
                h = 1 + 0.7^2 * sum(dispersion .* prob .* (1 .- prob) .+ prec)
                if abs(score) < 1e-12
                    correction += 0.5 * log(2pi / h)
                    break
                end
                iter == 30 && error("reference conditional mode did not converge")
                z += score / h
            end
            u[p.integrated[j]] = z
        end
        LogDensityProblems.logdensity(joint, u) + correction
    end
    for x in (zeros(length(p.names)), 0.3 .* randn(Xoshiro(87), length(p.names)))
        v, g = LogDensityProblems.logdensity_and_gradient(p, x)
        @test v ≈ oracle(x) atol=1e-8
        @test g ≈ values_fd(oracle, x) rtol=2e-7 atol=2e-8
    end

    nonlinear = RKBRMI(@brm d begin
        log(level) ~ 1 + (1 | site)
        log(mu) ~ 1 + offset(level)
        a ~ LogNormal(0, 1)
        y ~ NegativeBinomial2(mu, a .* exposure)
    end)
    records = BRM._brm_spl_records(nonlinear, [:site])
    @test isempty(last(BRM._brm_spl_fixed_scale_rows(nonlinear, records)))
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
    @test !only(plan.terms).admitted
    @test only(plan.terms).importance_ess === nothing
    forced = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=:g)
    @test only(forced.terms).admitted
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
    @test only(plan.terms).class === :sampled
    @test !only(plan.terms).admitted
    @test occursin("model-value observations", only(plan.terms).reason)
    sampled = brm_laplace_plan(derived; ad_backend=NLME_AD, sample=:g)
    @test only(sampled.terms).class === :sampled
    @test !only(sampled.terms).admitted
end
