# brm_laplace_problem(rk; integrate=(...)): every level of the named
# random-effect groupings integrated by one sparse Laplace, wherever they sit.
#
# Checked against: the per-unit row-structured Laplace where both apply (one
# grouping, which is the units); the exact marginal of Gaussian models with
# crossed and with ragged nested groupings, integrated whole or in part (the
# Laplace is exact there); central differences of its own value on count
# models, where the curvature moves with the modes; and the exact conditional
# for its draws.
#
# RUN: julia --project=test test/nlme_laplace_sparse.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics

const SPL_RNG = Xoshiro(20261030)

spl_problem(rk, integrate) = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate)
spl_point(p, φ) = (u = copy(p.u); u[p.population] .= φ; u[p.integrated] .= 0.0; u)

# RK's population prior and log-Jacobians at a point, less the integrated
# draws' N(0, 1) terms.
function spl_population_logprior(rk, p, u)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        sum(logpdf(Normal(), u[j]) for j in p.integrated)
end

function spl_fd(f, φ; h=1e-5)
    map(eachindex(φ)) do k
        e = zeros(length(φ))
        e[k] = h
        (f(φ .+ e) - f(φ .- e)) / 2h
    end
end
spl_fd(p::BRMSparseLaplaceProblem, φ; h=1e-5) =
    spl_fd(ψ -> LogDensityProblems.logdensity(p, ψ), φ; h)

indicator(labels) = (levels = unique(labels); [l == level for l in labels, level in levels])

# The exact marginal of a Normal model whose integrated groupings enter its
# mean as random intercepts: y ~ N(μ₀, Σ_g τ_g² Z_g Z_gᵀ + σ² I), with μ₀ the
# model's mean at the integrated draws' zero (it carries the sampled draws).
function spl_gaussian_marginal(p, rk, data, φ, scales)
    u = spl_point(p, φ)
    ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
    μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
    σ = [exp(ψ[p.entry[t] + 1]) for t in eachindex(p.family)]
    values = BRM._rk_constrained_values(rk, u)
    Σ = Diagonal(σ .^ 2) + sum(only(getproperty(values, Symbol(:b_, g)).tau)^2 *
        indicator(data[g]) * indicator(data[g])' for g in scales)
    spl_population_logprior(rk, p, u) + logpdf(MvNormal(μ0, Symmetric(Σ)), data.y)
end

# Crossed groupings: 6 levels of `a` crossed with 4 of `b`, 2 rows per cell.
const SPL_CROSSED = let rng = Xoshiro(11)
    a = repeat(["a$i" for i in 1:6], inner=8)
    b = repeat(["b$i" for i in 1:4], outer=12)
    x = randn(rng, length(a))
    ea = Dict(l => 0.7randn(rng) for l in unique(a))
    eb = Dict(l => 0.5randn(rng) for l in unique(b))
    η = [0.4 + 0.6x[i] + ea[a[i]] + eb[b[i]] for i in eachindex(a)]
    (; a, b, x, η, y=η .+ 0.3 .* randn(rng, length(a)))
end

# Ragged nested groupings: 4 regions, each with 1 to 4 districts of 2 to 4 rows.
const SPL_NESTED = let rng = Xoshiro(12)
    g = String[]; h = String[]
    for (i, nh) in enumerate((1, 3, 2, 4)), j in 1:nh, _ in 1:(2 + (i + j) % 3)
        push!(g, "g$i"); push!(h, "g$(i)h$j")
    end
    x = randn(rng, length(g))
    eg = Dict(l => 0.6randn(rng) for l in unique(g))
    eh = Dict(l => 0.4randn(rng) for l in unique(h))
    η = [0.2 + 0.5x[i] + eg[g[i]] + eh[h[i]] for i in eachindex(g)]
    (; g, h, x, η, y=η .+ 0.3 .* randn(rng, length(g)))
end

@stestset "one grouping that is the units: the sparse Laplace is the row-structured one" begin
    rng = Xoshiro(21)
    g = vcat([fill("g$i", 2 + i % 3) for i in 1:6]...)
    x = randn(rng, length(g))
    y = 0.5 .+ 0.8 .* x .+ 0.4 .* randn(rng, length(g))
    counts = [rand(rng, Poisson(exp(0.3 + 0.5xi))) for xi in x]
    for brmi in (
            @brm((; g, x, y), begin
                sigma ~ Exponential(1)
                mu ~ 1 + x + (1 + x | g)
                y ~ Normal(mu, sigma)
            end),
            @brm((; g, x, y=counts), begin
                log(mu) ~ 1 + x + (1 | g)
                y ~ Poisson(mu)
            end))
        rk = RKBRMI(brmi)
        sparse = spl_problem(rk, :g)
        structured = brm_laplace_problem(rk; ad_backend=NLME_AD)
        @test sparse isa BRMSparseLaplaceProblem
        @test structured isa BRMStructuredLaplaceProblem
        @test sparse.names == structured.names
        @test length(sparse.components) == 6
        for _ in 1:3
            φ = 0.4 .* randn(rng, length(sparse.population))
            v1, g1 = LogDensityProblems.logdensity_and_gradient(sparse, φ)
            v2, g2 = LogDensityProblems.logdensity_and_gradient(structured, φ)
            @test v1 ≈ v2 rtol=1e-10 atol=1e-10
            @test g1 ≈ g2 rtol=1e-8 atol=1e-8
        end
    end
end

@stestset "crossed groupings: one component, the exact Gaussian marginal" begin
    data = SPL_CROSSED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    for integrate in ((:a, :b), :a)
        p = spl_problem(rk, integrate)
        scales = integrate isa Symbol ? (integrate,) : integrate
        @test length(p.integrated) == sum(g -> length(unique(data[g])), scales)
        @test length(p.components) == (integrate isa Symbol ? 6 : 1)
        @test !any(n -> startswith(String(n), "b_a.z"), p.names)
        for _ in 1:3
            φ = 0.4 .* randn(SPL_RNG, length(p.population))
            value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
            @test value ≈ spl_gaussian_marginal(p, rk, data, φ, scales) atol=1e-9
            @test gradient ≈ spl_fd(ψ -> spl_gaussian_marginal(p, rk, data, ψ, scales), φ) rtol=1e-6 atol=1e-6
        end
    end
end

@stestset "ragged nested groupings: one component per region, the exact Gaussian marginal" begin
    data = SPL_NESTED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g) + (1 | h)
        y ~ Normal(mu, sigma)
    end)
    p = spl_problem(rk, (:g, :h))
    @test length(p.components) == 4
    @test length(p.integrated) == 4 + 10
    for _ in 1:3
        φ = 0.4 .* randn(SPL_RNG, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test value ≈ spl_gaussian_marginal(p, rk, data, φ, (:g, :h)) atol=1e-9
        @test gradient ≈ spl_fd(ψ -> spl_gaussian_marginal(p, rk, data, ψ, (:g, :h)), φ) rtol=1e-6 atol=1e-6
    end
end

@stestset "count likelihoods: the gradient of the Laplace marginal itself" begin
    rng = Xoshiro(23)
    c = SPL_CROSSED
    n = SPL_NESTED
    counts_c = [rand(rng, Poisson(exp(0.2 + 0.4e))) for e in c.η]
    counts_n = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + exp(e)))) for e in n.η]
    shares_n = [rand(rng, Binomial(10, 1 / (1 + exp(-e)))) for e in n.η]
    cases = (
        (@brm((; c.a, c.b, c.x, y=counts_c), begin
            log(mu) ~ 1 + x + (1 + x | a) + (1 | b)
            y ~ Poisson(mu)
        end), ((:a, :b), :b)),
        (@brm((; n.g, n.h, n.x, y=counts_n), begin
            phi ~ Exponential(1)
            log(mu) ~ 1 + x + (1 | g) + (1 | h)
            y ~ NegativeBinomial2(mu, phi)
        end), ((:g, :h), :h)),
        (@brm((; n.g, n.h, n.x, H=shares_n, trials=fill(10, length(shares_n))), begin
            logit(p) ~ 1 + x + (1 | g) + (1 | h)
            H ~ Binomial(trials, p)
        end), ((:g, :h),)),
    )
    for (brmi, integrates) in cases, integrate in integrates
        p = spl_problem(RKBRMI(brmi), integrate)
        φ = 0.3 .* randn(rng, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test isfinite(value)
        @test value ≈ LogDensityProblems.logdensity(p, φ) rtol=1e-12
        @test gradient ≈ spl_fd(p, φ) rtol=1e-6 atol=1e-6
    end
end

@stestset "a density that is not concave everywhere still finds its mode" begin
    # A random effect on a negative-binomial shape makes the density
    # non-concave away from its mode, where h need not be positive definite
    # (the step then takes a shifted factor); the search still reaches the
    # mode and the gradient there is the marginal's.
    rng = Xoshiro(29)
    n = SPL_NESTED
    counts = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(0.8 + e)))) for e in n.η]
    rk = RKBRMI(@brm (; n.g, n.h, n.x, y=counts) begin
        log(mu) ~ 1 + x + (1 | um | g) + (1 | h)
        log(shape) ~ 1 + (1 | us | g)
        y ~ NegativeBinomial2(mu, shape)
    end)
    p = spl_problem(rk, (:g, :h))
    scales = findall(m -> occursin(".tau.", String(m)), p.names)
    for s in (1.0, 2.0, 3.0)
        φ = zeros(length(p.population))
        φ[scales] .= s
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test isfinite(value) && all(isfinite, gradient)
        @test gradient ≈ spl_fd(p, φ; h=1e-4) rtol=1e-5 atol=1e-5
    end
    @test p.nonfinite == 0
end

@stestset "draws: the exact Gaussian conditional, and importance resampling per component" begin
    data = SPL_CROSSED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    p = spl_problem(rk, (:a, :b))
    φ = 0.3 .* randn(SPL_RNG, length(p.population))
    ndraws = 4000
    out = brm_laplace_draws(p, repeat(φ, 1, ndraws); rng=Xoshiro(31))
    @test size(out.draws) == (length(p.u), ndraws)
    @test out.draws[p.population, :] == repeat(φ, 1, ndraws)
    # The exact conditional of the standardized draws: h = I + Cᵀ C / σ², mean
    # h⁻¹ Cᵀ (y − μ₀) / σ².
    u = spl_point(p, φ)
    ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
    μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
    σ = exp(ψ[p.entry[1] + 1])
    values = BRM._rk_constrained_values(rk, u)
    C = zeros(length(data.y), length(p.integrated))
    for (g, block) in zip((:a, :b), p.blocks)
        τ = only(getproperty(values, Symbol(:b_, g)).tau)
        levels = BRM._rk_grouping_levels(rk.plan.columns[g])
        for (i, label) in enumerate(data[g])
            C[i, block.z[findfirst(==(label), levels), 1]] = τ
        end
    end
    h = I + C' * C / σ^2
    mean_z = h \ (C' * (data.y .- μ0) / σ^2)
    sd_z = sqrt.(diag(inv(h)))
    z = out.draws[p.integrated, :]
    @test all(abs.(vec(mean(z; dims=2)) .- mean_z) .< 4 .* sd_z ./ sqrt(ndraws))
    @test vec(std(z; dims=2)) ≈ sd_z rtol=0.06
    # Importance resampling: one ESS per component and draw.
    q = spl_problem(rk, :a)
    φq = 0.3 .* randn(SPL_RNG, length(q.population))
    resampled = brm_laplace_draws(q, repeat(φq, 1, 50); rng=Xoshiro(32), conditional=16)
    @test size(resampled.diagnostics.importance_ess) == (length(q.components), 50)
    @test all(e -> 1 <= e <= 16 + 1e-9, resampled.diagnostics.importance_ess)
end

@stestset "a sparse Laplace problem copies per chain" begin
    data = SPL_CROSSED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    p = spl_problem(rk, (:a, :b))
    φ = 0.3 .* randn(SPL_RNG, length(p.population))
    expected = LogDensityProblems.logdensity_and_gradient(p, φ)
    @test p.factor[] !== nothing
    q = deepcopy(p)
    @test q.factor[] === nothing
    GC.gc()
    for problem in (q, p, q)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, φ .+ 0.01)
        @test isfinite(value)
    end
    @test LogDensityProblems.logdensity_and_gradient(q, φ)[1] ≈ expected[1] rtol=1e-12
    @test q.factor[] !== p.factor[]
end

@stestset "what integrate refuses" begin
    data = SPL_CROSSED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        y ~ Normal(mu, sigma)
    end)
    @test_throws r"groups no random-effect block" spl_problem(rk, :b)
    @test_throws ArgumentError spl_problem(rk, ())
end
