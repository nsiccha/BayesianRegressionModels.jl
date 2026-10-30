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
using SparseArrays: rowvals, nzrange, nonzeros
include(joinpath(@__DIR__, "rk_cdar_fixture.jl"))

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

# Areas "a".."h" (test/car_term.jl's graph): component {a, b, c, d, h} with
# a–b, b–c, c–d, a–c, d–h; component {e, f}; "g" an island with rows; "h" has
# no rows. Three rows per area with rows, a covariate x.
const SPL_EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
const SPL_CAR = let rng = Xoshiro(41)
    area = repeat(["a", "b", "c", "d", "e", "f", "g"], inner=3)
    x = randn(rng, length(area))
    effect = Dict("a" => 0.4, "b" => 0.2, "c" => -0.1, "d" => -0.5, "e" => 0.3,
        "f" => -0.3, "g" => 0.0)
    η = [0.3 + 0.5x[i] + effect[area[i]] for i in eachindex(area)]
    (; area, x, η, y=η .+ 0.3 .* randn(rng, length(area)), from=SPL_EDGES[1], to=SPL_EDGES[2])
end

# The field's covariance over the rows, P⁻¹ on the structured areas with P the
# graph Laplacian plus 11ᵀ/(0.001 n)² per component, built from the labels.
function spl_car_row_covariance(area, from, to)
    nodes = sort(unique(vcat(from, to)))
    k = Dict(l => i for (i, l) in enumerate(nodes))
    L = zeros(length(nodes), length(nodes))
    for (a, b) in zip(from, to)
        i, j = k[a], k[b]
        L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    end
    for comp in (["a", "b", "c", "d", "h"], ["e", "f"])
        u = [l in comp ? 1.0 : 0.0 for l in nodes]
        L .+= u * u' / (0.001 * length(comp))^2
    end
    A = [haskey(k, a) && n == a ? 1.0 : 0.0 for a in area, n in nodes]
    A * inv(L) * A'
end

spl_block_values(rk, p, u, j) =
    getproperty(BRM._rk_constrained_values(rk, u), p.blocks[j].declaration)

function spl_icar_oracle(p, rk, data, φ)
    u = spl_point(p, φ)
    ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
    μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
    σ = [exp(ψ[p.entry[t] + 1]) for t in eachindex(p.family)]
    τ = only(spl_block_values(rk, p, u, 1).sigma)
    Σ = Diagonal(σ .^ 2) + τ^2 * spl_car_row_covariance(data.area, data.from, data.to)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = Base.invokelatest(prepare_query(rk.model, plan, :prior), u) +
        Base.invokelatest(prepare_query(rk.model, plan, :log_jacobian), u)
    prior + logpdf(MvNormal(μ0, Symmetric(Σ)), data.y)
end

@stestset "car fields: the ICAR field's exact Gaussian marginal, BYM2 and counts" begin
    data = SPL_CAR
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + car(area; edges=(from, to), type=:icar)
        y ~ Normal(mu, sigma)
    end)
    p = spl_problem(rk, :area)
    @test length(p.integrated) == 7
    @test length(p.wcols) == 2
    @test length(p.components) == 2
    φs = [0.4 .* randn(SPL_RNG, length(p.population)) for _ in 1:3]
    values = [LogDensityProblems.logdensity_and_gradient(p, φ) for φ in φs]
    oracle = [spl_icar_oracle(p, rk, data, φ) for φ in φs]
    # equal up to the field prior's normalizing constant, a constant of the graph
    @test values[2][1] - values[1][1] ≈ oracle[2] - oracle[1] atol=1e-8
    @test values[3][1] - values[1][1] ≈ oracle[3] - oracle[1] atol=1e-8
    for (φ, (_, gradient)) in zip(φs, values)
        @test gradient ≈ spl_fd(ψ -> spl_icar_oracle(p, rk, data, ψ), φ) rtol=1e-6 atol=1e-6
    end
    # Draws: the exact Gaussian conditional of the field's values (the soft
    # sums' rank-one part enters through the kriging correction).
    φ = φs[1]
    u = spl_point(p, φ)
    ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
    μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
    σ = exp(ψ[p.entry[1] + 1])
    τ = only(spl_block_values(rk, p, u, 1).sigma)
    C, _, _, _ = BRM._brm_spl_cmaps(p, BRM._brm_slap_zero_integrated!(copy(u), p))
    Cm = zeros(length(data.y), length(p.integrated))
    for e in eachindex(p.port), r in p.rptr[e]:(p.rptr[e + 1] - 1)
        Cm[findfirst(==(e), p.entry), p.blocks[p.rblock[r]].z[p.rlevel[r], 1]] = C[p.coff[r] + 1]
    end
    @test all(c -> c == 0 || c ≈ τ, Cm)
    h = zeros(length(p.integrated), length(p.integrated))
    rows, vals = rowvals(p.pattern), zeros(length(nonzeros(p.pattern)))
    for (k, v) in zip(p.prior_nz, p.prior_val)
        vals[k] += v
    end
    for j in axes(h, 2), k in nzrange(p.pattern, j)
        h[rows[k], j] += vals[k]
    end
    for (c, cols) in enumerate(p.wcols)
        h[cols, cols] .+= p.wλ[c]
    end
    h .+= Cm' * Cm / σ^2
    mean_z = h \ (Cm' * (data.y .- μ0) / σ^2)
    sd_z = sqrt.(diag(inv(h)))
    ndraws = 4000
    z = brm_laplace_draws(p, repeat(φ, 1, ndraws); rng=Xoshiro(43)).draws[p.integrated, :]
    @test all(abs.(vec(mean(z; dims=2)) .- mean_z) .< 4 .* sd_z ./ sqrt(ndraws))
    @test vec(std(z; dims=2)) ≈ sd_z rtol=0.06
    # BYM2 (structured and iid parts) and Poisson counts: the gradient of the
    # Laplace marginal itself.
    rng = Xoshiro(44)
    counts = [rand(rng, Poisson(exp(e))) for e in data.η]
    for (brmi, nint) in (
            (@brm((; data.area, data.x, data.y, data.from, data.to), begin
                sigma ~ Exponential(1)
                mu ~ 1 + x + car(area; edges=(from, to), type=:bym2)
                y ~ Normal(mu, sigma)
            end), 7 + 8),
            (@brm((; data.area, data.x, y=counts, data.from, data.to), begin
                log(mu) ~ 1 + x + car(area; edges=(from, to), type=:icar)
                y ~ Poisson(mu)
            end), 7))
        q = spl_problem(RKBRMI(brmi), :area)
        @test length(q.integrated) == nint
        φ = 0.3 .* randn(rng, length(q.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(q, φ)
        @test isfinite(value)
        @test gradient ≈ spl_fd(q, φ) rtol=1e-6 atol=1e-6
    end
end

@stestset "cdar walks: the exact Gaussian marginal of a stationary AR(1), and counts" begin
    data = WALK_DF
    rk = RKBRMI(walk_model(data))
    p = spl_problem(rk, :kreis)
    @test length(p.integrated) == 6 * 4
    @test length(p.components) == 6
    function oracle(φ)
        u = spl_point(p, φ)
        ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
        μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
        σ = [exp(ψ[p.entry[t] + 1]) for t in eachindex(p.family)]
        walk = spl_block_values(rk, p, u, 1)
        sw, ρ = only(walk.sigma), only(walk.rho)
        Σ = Diagonal(σ .^ 2) + [k1 == k2 ? sw^2 * ρ^abs(t1 - t2) : 0.0
            for (k1, t1) in zip(data.kreis, data.year), (k2, t2) in zip(data.kreis, data.year)]
        plan = RKEXT._rk_translated_plan(rk.plan)
        prior = Base.invokelatest(prepare_query(rk.model, plan, :prior), u) +
            Base.invokelatest(prepare_query(rk.model, plan, :log_jacobian), u)
        prior - sum(logpdf(Normal(), u[j]) for j in p.integrated; init=0.0) +
            logpdf(MvNormal(μ0, Symmetric(Σ)), data.y)
    end
    for _ in 1:2
        φ = 0.3 .* randn(SPL_RNG, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
        @test value ≈ oracle(φ) atol=1e-8
        @test gradient ≈ spl_fd(oracle, φ) rtol=1e-6 atol=1e-6
    end
    rng = Xoshiro(45)
    counts = [rand(rng, Poisson(exp(0.3 + 0.4x))) for x in data.x]
    q = spl_problem(RKBRMI(@brm((; data.kreis, data.year, data.region, data.x, y=counts,
            data.C), begin
        log(mu) ~ 1 + x + (1 | region) + cdar(year; by=kreis, cor=C)
        y ~ Poisson(mu)
    end)), (:kreis, :region))
    @test length(q.components) == 3
    φ = 0.3 .* randn(rng, length(q.population))
    _, gradient = LogDensityProblems.logdensity_and_gradient(q, φ)
    @test gradient ≈ spl_fd(q, φ) rtol=1e-6 atol=1e-6
    # A walk correlated across groups reads other groups' innovations.
    @test_throws r"several levels" spl_problem(RKBRMI(across_model(data)), :kreis)
end

@stestset "step blocks: the coarsest grouping's levels inside a component" begin
    # Crossed groupings form one component; its steps go by the coarser
    # grouping's levels (b, 4 levels).
    rk = RKBRMI(@brm SPL_CROSSED begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a) + (1 | b)
        y ~ Normal(mu, sigma)
    end)
    p = spl_problem(rk, (:a, :b))
    @test length(p.components) == 1
    @test 1 < length(p.sblocks) <= 4
    # Nested units: one block per unit, as before.
    rk = RKBRMI(@brm SPL_NESTED begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g) + (1 | h)
        y ~ Normal(mu, sigma)
    end)
    q = spl_problem(rk, (:g, :h))
    @test length(q.sblocks) == length(q.components) == 4
    # An ICAR field alone: its areas step on their own; the area without rows
    # ("h") joins its neighbour's block.
    c = spl_problem(RKBRMI(@brm SPL_CAR begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + car(area; edges=(from, to), type=:icar)
        y ~ Normal(mu, sigma)
    end), :area)
    @test length(c.sblocks) == 6
    # A non-concave density over crossed groupings (a random effect on the
    # negative-binomial shape): the blocks step on their own, and where the
    # component's density does not rise a block Gauss–Seidel sweep follows;
    # the search converges and the gradient is the marginal's.
    rng = Xoshiro(47)
    counts = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + exp(0.8 + e)))) for e in SPL_CROSSED.η]
    r = spl_problem(RKBRMI(@brm (; SPL_CROSSED.a, SPL_CROSSED.b, SPL_CROSSED.x, y=counts) begin
        log(mu) ~ 1 + x + (1 | a) + (1 | b)
        log(shape) ~ 1 + (1 | b)
        y ~ NegativeBinomial2(mu, shape)
    end), (:a, :b))
    scales = findall(m -> occursin(".tau.", String(m)), r.names)
    for scale in (0.5, 1.5, 2.5)
        φ = zeros(length(r.population))
        φ[scales] .= scale
        value, gradient = LogDensityProblems.logdensity_and_gradient(r, φ)
        @test isfinite(value) && all(isfinite, gradient)
        @test gradient ≈ spl_fd(r, φ; h=1e-4) rtol=1e-5 atol=1e-5
    end
    @test r.nonfinite == 0 && r.unconverged == 0
    # Where the full step fails (here a Newton direction scaled by 100), the
    # blocks step on their own and, where the component's density still does
    # not rise, a Gauss–Seidel sweep follows; either way the density rises.
    φ = zeros(length(r.population))
    φ[scales] .= 1.0
    pop = BRM._brm_spl_population(r, φ)
    nz = length(r.integrated)
    z = 0.3 .* randn(Xoshiro(48), nz)
    fc = zeros(length(r.components)); g = zeros(nz); h = copy(r.pattern)
    BRM._brm_spl_assemble!(fc, g, h, r, pop.A, pop.C, z; derivatives=true)
    δ = 100 .* BRM._brm_spl_direction(r, h, g)
    before = (r.blockwise, r.sweeps)
    z1 = copy(z)
    @test BRM._brm_spl_step!(r, pop, z1, δ, g, h, fc, falses(length(r.components)))
    @test r.blockwise == before[1] + 1
    f1 = zeros(length(r.components))
    BRM._brm_spl_assemble!(f1, nothing, nothing, r, pop.A, pop.C, z1; derivatives=false)
    @test sum(f1) > sum(fc)
end

# `@plate` cells: subjects whose cells read their own effect and their rows'
# nested unit effects (test/nlme_fixtures.jl's `nested_model`), with a
# Gaussian cell and one whose mean is `exp.(pred)` (non-Gaussian in the
# effects, so the cells' third derivatives matter).
spl_plate_exp(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | p | subject)
    unit_shift ~ 0 + (1 | u | unit)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(mu)
        pred[i] = mu[i] .+ unit_shift[obs[i]]
        y[obs[i]] ~ Normal(exp.(pred[i]), sigma)
    end
end

@stestset "@plate cells: the sparse Laplace is the whole-model per-unit Laplace" begin
    rng = Xoshiro(51)
    for build in (nested_model, spl_plate_exp)
        rk = RKBRMI(build(NESTED_DATA))
        sparse = spl_problem(rk, (:subject, :unit))
        whole = brm_laplace_problem(rk; ad_backend=NLME_AD, subject_group=:subject,
            nested_groups=(unit=:obs_subject,))
        @test whole isa BRMLaplaceProblem
        @test sparse.names == whole.names
        @test length(sparse.cells) == 1 && length(only(sparse.cells).cells) == 3
        @test length(sparse.components) == 3
        for _ in 1:3
            φ = 0.3 .* randn(rng, length(sparse.population))
            v1, g1 = LogDensityProblems.logdensity_and_gradient(sparse, φ)
            v2, g2 = LogDensityProblems.logdensity_and_gradient(whole, φ)
            @test v1 ≈ v2 rtol=1e-8 atol=1e-8
            @test g1 ≈ g2 rtol=1e-6 atol=1e-6
        end
        φ = 0.3 .* randn(rng, length(sparse.population))
        @test last(LogDensityProblems.logdensity_and_gradient(sparse, φ)) ≈
            spl_fd(sparse, φ) rtol=1e-5 atol=1e-5
        # a copy per chain carries its own cells' AD
        expected = LogDensityProblems.logdensity_and_gradient(sparse, φ)
        q = deepcopy(sparse)
        @test LogDensityProblems.logdensity_and_gradient(q, φ .+ 0.01)[1] ≈
            LogDensityProblems.logdensity_and_gradient(sparse, φ .+ 0.01)[1] rtol=1e-12
        @test all(LogDensityProblems.logdensity_and_gradient(q, φ) .≈ expected)
        # the units integrated, the subjects' draws sampled (population
        # coordinates the rows' pullback does not reach: there are no rows)
        partial = spl_problem(rk, :unit)
        ψ = 0.3 .* randn(rng, length(partial.population))
        @test last(LogDensityProblems.logdensity_and_gradient(partial, ψ)) ≈
            spl_fd(partial, ψ) rtol=1e-5 atol=1e-5
    end
end

# Per-unit values and the gradient of a subset of units (the NLME protocol's
# integrated families, ext/BayesianRegressionModelsNLMEEstimationExt.jl):
# levels of `a` as the units, each with its rows and its integrated draws.
@stestset "unit values and a subset of units' gradient" begin
    c = SPL_CROSSED
    rng = Xoshiro(41)
    data = (; c.a, c.x, y=[rand(rng, Poisson(exp(0.2 + 0.4e))) for e in c.η])
    rk = RKBRMI(@brm data begin
        log(mu) ~ 1 + x + (1 + x | a)
        y ~ Poisson(mu)
    end)
    p = spl_problem(rk, :a)
    level = Dict(l => k for (k, l) in enumerate(unique(data.a)))
    owners = BRM._brm_spl_term_owners((name, r) -> level[data.a[r]], p)
    nunits = length(level)
    @test length(owners) == length(p.family)
    φ = 0.3 .* randn(rng, length(p.population))
    state = BRM._brm_spl_mode!(p, φ)
    units = BRM._brm_spl_unit_values(p, state, owners, nunits)
    value, gradient = BRM._brm_spl_value_and_gradient(p, state)
    @test sum(units) ≈ value - BRM._rk_slap_prior(p.kernels, state.uA) -
        length(p.integrated) / 2 * log(2π) rtol=1e-10
    # without the prior, the whole gradient is the units' sum's
    _, g0 = BRM._brm_spl_value_and_gradient(p, state; prior=false)
    unitsum(S) = ψ -> (st = BRM._brm_spl_mode!(p, ψ);
        sum(BRM._brm_spl_unit_values(p, st, owners, nunits)[S]))
    @test g0 ≈ spl_fd(unitsum(1:nunits), φ) rtol=1e-5 atol=1e-6
    # a subset's gradient: its terms only
    S = [2, 5]
    _, gS = BRM._brm_spl_value_and_gradient(p, BRM._brm_spl_mode!(p, φ); prior=false,
        active=owners .∈ Ref(S))
    @test gS ≈ spl_fd(unitsum(S), φ) rtol=1e-5 atol=1e-6
    @test !(gS ≈ g0)
    # the blocks of h⁻¹ on each level's two margins
    state = BRM._brm_spl_mode!(p, φ)
    sets = [only(p.blocks).z[l, :] for l in axes(only(p.blocks).z, 1)]
    blocks = BRM._brm_spl_inverse_blocks(p, state, sets)
    hinv = inv(Matrix(Symmetric(state.h, :L)))
    @test all(b ≈ hinv[set, set] for (b, set) in zip(blocks, sets))
end

@stestset "what integrate refuses" begin
    data = SPL_CROSSED
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a)
        y ~ Normal(mu, sigma)
    end)
    @test_throws r"groups no random-effect block" spl_problem(rk, :b)
    @test_throws r"groups no random-effect block, `cdar` walk or `car` field" spl_problem(rk, :x)
    @test_throws ArgumentError spl_problem(rk, ())
end
