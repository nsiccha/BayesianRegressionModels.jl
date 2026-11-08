# test/nlme_partly_known_blocks.jl — the NLME reading of a subject block that holds
# some of its values known (snag `brm-nlme-view-re-2fd0380d`, todo `0840h34`):
#
# - known scales under a sampled correlation: one family that holds those margins'
#   SDs (NLMEEstimation's `FixedOmega(sd = ...)`), a known margin's η being its
#   effect over its known SD;
# - a known correlation next to sampled scales: one family held at the identity,
#   its sampled scales population coordinates.
#
# The linear Gaussian model gives exact oracles: each subject's conditional and
# marginal log likelihood is a Normal density, the empirical Bayes mode a linear
# solve, and the population prior a sum of named densities. The free coordinates of
# a family that holds some SDs are NLMEEstimation's (log sd, m), m_ij = L_ij / L_jj.
#
# NLMEEstimation is not in test/Project.toml: run this file in the NLMEEstimation
# scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_partly_known_blocks.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using NLMEEstimation, LinearAlgebra, LogDensityProblems, Random, Statistics
using BridgeStan, WarmupHMC
const NE = NLMEEstimation

const TRUE_SD = [0.6, 0.25]
const TRUE_COR = 0.3
const TRUE_R = [1.0 TRUE_COR; TRUE_COR 1.0]
const TRUE_BETA = [1.0, 0.5]
const TRUE_SIGMA = 0.3

function block_data(rng; N=30, n=6)
    subject = repeat(1:N, inner=n)
    x = round.(randn(rng, N * n); digits=3)
    A = Diagonal(TRUE_SD) * cholesky(TRUE_R).L
    b = [A * randn(rng, 2) for _ in 1:N]
    y = [TRUE_BETA[1] + b[s][1] + (TRUE_BETA[2] + b[s][2]) * x[i] + TRUE_SIGMA * randn(rng)
        for (i, s) in enumerate(subject)]
    (; subject, x, y, sd_known=TRUE_SD[1], sd_slope=TRUE_SD[2], R_known=TRUE_R)
end

# The same block declared in a function-style component, its held value an input of
# the call.
@brm function held_component(subject, x, scale)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) = scale
    sd(mu, p, x) ~ Exponential(2)
    cor(:, p) ~ LKJCholesky(2, 2)
    return mu
end
component_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ held_component(subject, x, sd_known)
    y ~ Normal(mu, sigma)
end

# The reported shape: one scale held at a data input, the others estimated, with
# the block's correlation estimated too.
held_first(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) = sd_known
    sd(mu, p, x) ~ Exponential(2)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(mu, sigma)
end

# The held margin is not the first: a free row precedes a held one.
held_second(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) ~ Exponential(2)
    sd(mu, p, x) = sd_slope
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(mu, sigma)
end

# Every scale known, the correlation estimated.
held_both(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) = sd_known
    sd(mu, p, x) = sd_slope
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(mu, sigma)
end

# The correlation known, the scales estimated.
known_correlation(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) ~ Exponential(2)
    sd(mu, p, x) ~ Exponential(3)
    cor(:, p) = R_known
    y ~ Normal(mu, sigma)
end

# The correlation and one scale known.
known_correlation_one_scale(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) = sd_known
    sd(mu, p, x) ~ Exponential(3)
    cor(:, p) = R_known
    y ~ Normal(mu, sigma)
end

const DATA = block_data(Xoshiro(20261108))
rows_of(j) = findall(==(j), DATA.subject)
design_of(j) = hcat(ones(length(rows_of(j))), DATA.x[rows_of(j)])

nlme(model, data=DATA) = brm_nlme_model(RKBRMI(model(data)); ad_backend=NLME_AD)

# The exact log marginal likelihood of every subject at (β, sigma, Cov).
function exact_marginal(β, sigma, Cov)
    sum(unique(DATA.subject)) do j
        Z = design_of(j)
        C = Z * Cov * Z' + sigma^2 * I
        logpdf(MvNormal(Z * β, Symmetric(C)), DATA.y[rows_of(j)])
    end
end

# A block's natural covariance from named scales and the (2,1) entry of its
# correlation's Cholesky factor.
natural_cov(sd, L21) = Diagonal(sd) * [1 0; L21 sqrt(1 - L21^2)] * [1 L21; 0 sqrt(1 - L21^2)] *
    Diagonal(sd)

@stestset "known scales under a sampled correlation: the reading" begin
    m = nlme(held_first)
    family = only(m.view.families)
    @test family.fixed === nothing
    @test length(family.scales) == 1 && length(family.factor) == 1
    @test count(==(:subject_scale), m.view.role) == 1
    @test count(==(:subject_correlation), m.view.role) == 1
    @test !any(==(:shared_latent), m.view.role)
    constraint = last(only(NE.fixed_omega(m)))
    @test isequal(constraint.sd, [1.0, nothing])
    @test constraint.correlation === nothing
    # Both population coefficients stay mu-referenced: η absorbs a known margin's
    # coefficient over its known SD.
    refs = sort(m.view.mu_references; by=r -> r.subject)
    @test [r.coefficient for r in refs] == [:Intercept, :x]
    @test [r.eta_scale for r in refs] == [TRUE_SD[1], 1.0]
    design = NE.mu_referencing(m).design
    @test design[1][1, 1] ≈ 1 / TRUE_SD[1]
    @test design[1][2, 2] == 1.0
    # Every scale known: only the correlation is estimated.
    both = nlme(held_both)
    family = only(both.view.families)
    @test isempty(family.scales) && length(family.factor) == 1
    @test isequal(last(only(NE.fixed_omega(both))).sd, [1.0, 1.0])
end

@stestset "a block in a component reads the same, its held value an input" begin
    m = nlme(component_model)
    family = only(m.view.families)
    @test family.fixed === nothing && family.size == 2
    @test length(family.scales) == 1 && length(family.factor) == 1
    @test isequal(last(only(NE.fixed_omega(m))).sd, [1.0, nothing])
    # At the unit point the conditional is the Normal with the held SD on its margin.
    start = brm_nlme_start(m)
    H = 0.5 .* randn(Xoshiro(4), 2, length(m.view.levels))
    twin = nlme(held_first)
    twin_start = brm_nlme_start(twin)
    named = brm_nlme_estimates(twin, twin_start.theta, twin_start.sigma, twin_start.omega_matrix)
    β, sigma = [named.point.mu.Intercept, named.point.mu.x], named.point.sigma
    # The component's population coefficients are its call's; set them by name.
    names = m.view.coordinates[m.theta]
    θ = copy(start.theta)
    for (i, n) in enumerate(names)
        s = string(n)
        occursin("Intercept", s) && (θ[i] = β[1])
        occursin("Intercept", s) || (θ[i] = β[2])
    end
    ll = brm_nlme_loglikelihoods(m, θ, twin_start.sigma, H)
    @test ll ≈ brm_nlme_loglikelihoods(twin, twin_start.theta, twin_start.sigma, H) rtol = 1e-10
    @test check_protocol(m, θ, twin_start.sigma, start.omega; η=H)
    # Another value is the same program, and reaches the density.
    other = RKBRMI(component_model(merge(DATA, (; sd_known=0.9))))
    @test string(other.model.program) == string(m.backend.model.program)
    @test brm_nlme_loglikelihoods(brm_nlme_model(other; ad_backend=NLME_AD), θ,
        twin_start.sigma, H) != ll
end

@stestset "the conditional likelihoods and modes match their oracles" begin
    for (model, known) in ((held_first, [TRUE_SD[1], 1.0]), (held_second, [1.0, TRUE_SD[2]]),
            (held_both, TRUE_SD))
        m = nlme(model)
        start = brm_nlme_start(m)
        θ, σ, ω = start.theta, start.sigma, start.omega
        @test isfinite(start.population_logprior)
        S = Diagonal(known)
        named = brm_nlme_estimates(m, θ, σ, start.omega_matrix)
        β = [named.point.mu.Intercept, named.point.mu.x]
        sigma = named.point.sigma
        H = 0.5 .* randn(Xoshiro(4), 2, length(m.view.levels))
        ll = brm_nlme_loglikelihoods(m, θ, σ, H)
        for j in eachindex(m.view.levels)
            Z = design_of(m.view.levels[j])
            @test ll[j] ≈ sum(logpdf.(Normal.(Z * (β .+ S * H[:, j]), sigma),
                DATA.y[rows_of(m.view.levels[j])])) rtol = 1e-10
        end
        @test check_protocol(m, θ, σ, ω; η=H)
        ebe = empirical_bayes(m, θ, σ, ω)
        @test all(ebe.converged)
        Ω = start.omega_matrix
        for (k, j) in enumerate(ebe.subjects)
            Z = design_of(m.view.levels[j])
            B = Z * S
            r = DATA.y[rows_of(m.view.levels[j])] .- Z * β
            mode = (inv(Ω) + B' * B / sigma^2) \ (B' * r / sigma^2)
            @test ebe.modes[:, k] ≈ mode rtol = 1e-6 atol = 1e-8
        end
    end
end

# At a whole packed point the model's effects are diag(τ) L z, whichever values are
# known; the start's η and Ω are that point on the estimators' scale.
@stestset "a packed point reads back on the estimators' scale" begin
    for (model, known) in ((held_first, [TRUE_SD[1], nothing]),
            (held_second, [nothing, TRUE_SD[2]]))
        m = nlme(model)
        rng = Xoshiro(7)
        u = 0.4 .* randn(rng, length(m.view.coordinates))
        start = brm_nlme_start(m, u)
        values = BRM._rk_constrained_values(m.backend, u)
        record = only(m.view.families).record
        L = Matrix{Float64}(BRM._brm_rk_declaration_value(values, record.L))
        τ = [isnothing(k) ? only(BRM._brm_rk_declaration_value(values, record.scales[i])) : k
            for (i, k) in enumerate(known)]
        @test all(i -> isnothing(known[i]) || isapprox(sqrt(start.omega_matrix[i, i]), 1; atol=1e-12), 1:2)
        S = Diagonal([isnothing(k) ? 1.0 : k for k in known])
        @test S * start.omega_matrix * S ≈ Diagonal(τ) * L * L' * Diagonal(τ) rtol = 1e-10
        @test NE.apply_fixed_omega(m, start.omega) ≈ start.omega rtol = 1e-10
        # The conditional at η is the likelihood at the packed point.
        named = brm_nlme_estimates(m, start.theta, start.sigma, start.omega_matrix)
        for (i, level) in enumerate(m.view.levels)
            Z = design_of(level)
            z = u[m.view.subject_coordinates[i]]
            β = [named.point.mu.Intercept, named.point.mu.x]
            effect = Diagonal(τ) * L * z
            @test start.loglikelihoods[i] ≈ sum(logpdf.(Normal.(Z * (β .+ effect),
                named.point.sigma), DATA.y[rows_of(level)])) rtol = 1e-9
        end
        # The named block is in natural units, with what it holds.
        block = named.point.p
        @test isequal(block.known_sd, known)
        @test block.covariance ≈ Diagonal(τ) * L * L' * Diagonal(τ) rtol = 1e-10
        @test all(i -> isnothing(known[i]) || block.sd[i] ≈ known[i], 1:2)
        # `brm_start` takes it back: the known SD agrees with the model's.
        restart = brm_nlme_start(m, (; p=(; sd=block.sd, cor=block.cor)); base=u)
        @test restart.omega_matrix ≈ start.omega_matrix rtol = 1e-8
    end
end

# The density NLMEEstimation's search moves in for a family that holds some SDs:
# the prior of (τ_free, L) over the free coordinates (log sd, m).
function held_prior_oracle(; s, m)
    L21 = m / sqrt(1 + m^2)
    L22 = 1 / sqrt(1 + m^2)
    logpdf(Exponential(2), exp(s)) +
        logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular([1.0 0.0; L21 L22]))) +
        s + 3 * log(L22)
end

@stestset "the population prior is the authored prior over the free coordinates" begin
    for (model, free, known) in ((held_first, 2, [1.0, nothing]), (held_second, 1, [nothing, 1.0]))
        m = nlme(model)
        layout = NE.nlme_layout(m)
        start = brm_nlme_start(m)
        θ, σ = start.theta, start.sigma
        point(s, q) = begin
            sd = free == 2 ? [0.0, s] : [s, 0.0]
            NE._logcholesky_coordinates(sd, [q], 2)
        end
        rng = Xoshiro(11)
        base = NE.population_logprior(m, θ, σ, point(0.0, 0.0))
        oracle0 = held_prior_oracle(; s=0.0, m=0.0)
        for _ in 1:5
            s, q = 0.8 * randn(rng), 0.8 * randn(rng)
            @test NE.population_logprior(m, θ, σ, point(s, q)) - base ≈
                held_prior_oracle(; s, m=q) - oracle0 rtol = 1e-9 atol = 1e-9
            # Its gradient, in the coordinates the estimators chain through.
            ω = point(s, q)
            _, gθ, gσ, gω = NE.population_logprior_and_gradient(m, θ, σ, ω)
            fd = map(eachindex(ω)) do c
                h = 1e-5
                plus, minus = copy(ω), copy(ω)
                plus[c] += h
                minus[c] -= h
                (NE.population_logprior(m, θ, σ, plus) -
                    NE.population_logprior(m, θ, σ, minus)) / 2h
            end
            @test gω ≈ fd rtol = 1e-5 atol = 1e-6
            fθ = map(eachindex(θ)) do c
                h = 1e-5
                plus, minus = copy(θ), copy(θ)
                plus[c] += h
                minus[c] -= h
                (NE.population_logprior(m, plus, σ, ω) - NE.population_logprior(m, minus, σ, ω)) / 2h
            end
            @test gθ ≈ fθ rtol = 1e-5 atol = 1e-6
        end
        # A proper density over the free coordinates: it integrates to one.
        grid = range(-9.0, 4.0; length=401)
        qrid = range(-30.0, 30.0; length=601)
        total = sum(exp(NE.population_logprior(m, θ, σ, point(s, q)) - base + oracle0)
            for s in grid, q in qrid) * step(grid) * step(qrid)
        println("population prior over (s, m), held=", known, ": integrates to ", total)
        @test total ≈ 1 atol = 2e-3
    end
    # With every SD held only the correlation coordinate moves.
    m = nlme(held_both)
    start = brm_nlme_start(m)
    point(q) = NE._logcholesky_coordinates([0.0, 0.0], [q], 2)
    base = NE.population_logprior(m, start.theta, start.sigma, point(0.0))
    oracle(q) = logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(
        [1.0 0.0; q / sqrt(1 + q^2) 1 / sqrt(1 + q^2)]))) - 3 * log(1 + q^2) / 2
    for q in (-1.3, 0.4, 2.1)
        @test NE.population_logprior(m, start.theta, start.sigma, point(q)) - base ≈
            oracle(q) - oracle(0.0) rtol = 1e-9 atol = 1e-9
    end
end

# The exact log posterior in the search's coordinates, and its gradient norm in
# them by central differences, for a family holding the first margin's SD.
function held_first_objective(m, θ, σ, ω)
    β = θ    # (Intercept, x): this model's only population coordinates
    sigma = exp(only(σ))
    Ω = NE.omega_matrix(NE.nlme_layout(m), ω)
    S = Diagonal([TRUE_SD[1], 1.0])
    exact_marginal(β, sigma, S * Ω * S) + NE.population_logprior(m, θ, σ, ω)
end

@stestset "FOCE(I) and SAEM estimate only the free parts" begin
    m = nlme(held_first)
    focei = brm_nlme_fit(m, NE.focei; options=NE.FOCEOptions(; maxeval=2000))
    saem = brm_nlme_fit(m, NE.saem; rng=Xoshiro(9))
    objective(r) = held_first_objective(m, r.theta, r.sigma, r.omega)
    # FOCE(I) is exact on a linear Gaussian model, so its estimate is the exact
    # maximum: the independent objective's gradient vanishes there.
    y = vcat(focei.result.theta, focei.result.sigma,
        NE._free_omega(NE._omega_constraints(m), focei.result.omega))
    constraints = NE._omega_constraints(m)
    nθ, nσ = length(focei.result.theta), length(focei.result.sigma)
    at(y) = held_first_objective(m, y[1:nθ], y[(nθ + 1):(nθ + nσ)],
        NE._constrained_omega(constraints, y[(nθ + nσ + 1):end]))
    gradient = map(eachindex(y)) do c
        h = 1e-4
        plus, minus = copy(y), copy(y)
        plus[c] += h
        minus[c] -= h
        (at(plus) - at(minus)) / 2h
    end
    println("FOCE(I) exact-objective gradient: ", round.(gradient; digits=4))
    @test maximum(abs, gradient) < 0.02
    println("objective at FOCE(I) ", objective(focei.result), ", SAEM ", objective(saem.result))
    @test objective(saem.result) > objective(focei.result) - 0.1
    for fit in (focei, saem)
        block = fit.estimates.point.p
        @test isequal(block.known_sd, [TRUE_SD[1], nothing])
        @test block.sd[1] ≈ TRUE_SD[1] rtol = 1e-7
        println("mu = ", fit.estimates.point.mu, ", sigma = ", fit.estimates.point.sigma,
            ", sd = ", block.sd, ", cor = ", block.cor[2, 1])
        @test abs(block.sd[2] - TRUE_SD[2]) < 0.15
        @test abs(block.cor[2, 1] - TRUE_COR) < 0.5
        @test abs(fit.estimates.point.mu.Intercept - TRUE_BETA[1]) < 0.4
        @test abs(fit.estimates.point.mu.x - TRUE_BETA[2]) < 0.25
    end
    # Standard errors cover the free coordinates only.
    covariance = NE.covariance_step(m, focei.result)
    @test length(covariance.standard_errors) == nθ + nσ + 2
    @test all(isfinite, covariance.standard_errors)
end

@stestset "a known correlation: draws under a fixed identity, scales in θ" begin
    for (model, known) in ((known_correlation, [nothing, nothing]),
            (known_correlation_one_scale, [TRUE_SD[1], nothing]))
        m = nlme(model)
        family = only(m.view.families)
        @test family.fixed == Matrix{Float64}(I, 2, 2)
        @test isempty(family.scales) && isempty(family.factor)
        sampled = count(isnothing, known)
        @test count(==(:shared_latent), m.view.role) == sampled
        @test !any(in((:subject_scale, :subject_correlation)), m.view.role)
        @test isempty(m.view.mu_references)
        @test last(only(NE.fixed_omega(m))).correlation == Matrix{Float64}(I, 2, 2)
        start = brm_nlme_start(m)
        θ, σ, ω = start.theta, start.sigma, start.omega
        @test start.omega_matrix == Matrix{Float64}(I, 2, 2)
        H = 0.5 .* randn(Xoshiro(4), 2, length(m.view.levels))
        @test check_protocol(m, θ, σ, ω; η=H)
        named = brm_nlme_estimates(m, θ, σ, start.omega_matrix)
        block = named.point.p
        @test block.known_cor === true
        @test block.cor ≈ TRUE_R rtol = 1e-10
        sampled == 1 && @test isequal(block.known_sd, known)
        A = Diagonal(block.sd) * cholesky(TRUE_R).L
        β = [named.point.mu.Intercept, named.point.mu.x]
        ll = brm_nlme_loglikelihoods(m, θ, σ, H)
        for (j, level) in enumerate(m.view.levels)
            Z = design_of(level)
            @test ll[j] ≈ sum(logpdf.(Normal.(Z * (β .+ A * H[:, j]), named.point.sigma),
                DATA.y[rows_of(level)])) rtol = 1e-10
        end
        # The exact maximum of the marginal, reached through the population
        # coordinates; SAEM's numerical M-step moves the scales.
        exact(point) = begin
            sd, cov = point.p.sd, point.p.covariance
            exact_marginal([point.mu.Intercept, point.mu.x], point.sigma, cov)
        end
        focei = brm_nlme_fit(m, NE.focei; options=NE.FOCEOptions(; maxeval=2000))
        saem = brm_nlme_fit(m, NE.saem; rng=Xoshiro(9))
        pf = focei.estimates.point
        ps = saem.estimates.point
        println("known correlation, known=", known, ": FOCE(I) sd ", pf.p.sd, " marginal ", exact(pf),
            "; SAEM sd ", ps.p.sd, " marginal ", exact(ps))
        @test ps.p.cor ≈ TRUE_R rtol = 1e-10
        @test pf.p.cor ≈ TRUE_R rtol = 1e-10
        sampled == 1 && @test pf.p.sd[1] ≈ TRUE_SD[1] rtol = 1e-10
        @test abs(pf.p.sd[2] - TRUE_SD[2]) < 0.15
        # SAEM moves these scales by its numerical M-step alone, so it is noisier than
        # on a family whose Ω has a closed-form update: over seeds 1, 2, 3 and 9 at
        # the default options, known_correlation ended 0.10 to 0.97 below the maximum.
        @test exact(ps) > exact(pf) - 1.0
    end
end

# The Stan twin takes the same reading: a known scale is data on both sides.
@stestset "brm_nlme_joint_start: a checked full Stan start" begin
    dir = joinpath(tempdir(), "brm-nlme-partly-known")
    mkpath(dir)
    for (label, model, Ω, sd_index) in (("held_first", held_first, [1.0 0.25; 0.25 0.09], 2),
            ("held_second", held_second, [0.36 0.15; 0.15 1.0], 1),
            ("known_correlation", known_correlation, Matrix{Float64}(I, 2, 2), nothing))
        brmi = model(DATA)
        rk, sb = RKBRMI(brmi), SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        map = brm_nlme_joint_map(rk, sb; path=joinpath(dir, "$label.stan"))
        view = map.view
        rng = Xoshiro(3)
        p, s, shared = findall(==(:population), view.role), findall(==(:scalar), view.role),
            findall(==(:shared_latent), view.role)
        estimated = sort!(vcat(p, shared))
        θ = 0.3 .* randn(rng, length(estimated))
        σ = [-0.8]
        H = 0.4 .* randn(rng, 2, length(view.levels))
        start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω, eta=H,
            subjects=collect(view.levels))
        @test start.checked.max_error <= 1e-9
        back = brm_nlme_joint_values(map, start.stan)
        @test back.omega ≈ Ω atol = 1e-10
        @test back.eta ≈ H atol = 1e-10
        # The RK and Stan models agree at the transported point.
        u = brm_stan_point_to_rk(map.transport, start.stan)
        m = brm_nlme_model(rk; ad_backend=NLME_AD)
        @test brm_nlme_start(m, map, start.stan).omega_matrix ≈ Ω atol = 1e-9
        # A sampled scale's constrained value reaches Stan.
        if sd_index !== nothing
            constrained = BridgeStan.param_constrain(map.stan_model, start.stan)
            names = BridgeStan.param_names(map.stan_model)
            carrier = only(i for (i, n) in enumerate(names) if occursin("tau_free", n))
            @test constrained[carrier] ≈ sqrt(Ω[sd_index, sd_index]) rtol = 1e-10
        end
    end
end

# The reported call: SVI over the subjects, a partly known subject block next to a
# shared grouping whose scale is known too, both through data inputs.
const SVI_DATA = let rng = Xoshiro(5), N = 24, n = 5
    subject = repeat(1:N, inner=n)
    device = rand(rng, 1:3, N * n)
    x = round.(randn(rng, N * n); digits=3)
    A = Diagonal(TRUE_SD) * cholesky(TRUE_R).L
    b = [A * randn(rng, 2) for _ in 1:N]
    c = 0.2 .* randn(rng, 3)
    y = [TRUE_BETA[1] + b[s][1] + c[device[i]] + (TRUE_BETA[2] + b[s][2]) * x[i] +
        TRUE_SIGMA * randn(rng) for (i, s) in enumerate(subject)]
    (; subject, device, x, y, sd_known=TRUE_SD[1], sd_device=0.2, rate=2.0, eta=2.0)
end

svi_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject) + (1 | cal | device)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 5)
    sd(mu, p, Intercept) = sd_known
    sd(mu, p, x) ~ Exponential(rate)
    cor(:, p) ~ LKJCholesky(2, eta)
    sd(mu, cal) = sd_device
    y ~ Normal(mu, sigma)
end

@stestset "brm_svi_fit reads the block and matches NUTS" begin
    rk = RKBRMI(svi_model(SVI_DATA))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:subject, shared_groups=(:device,))
    family = only(m.view.families)
    @test family.fixed === nothing
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    reference = reduce(hcat, map(1:4) do c
        run = adaptive_warmup_mcmc(Xoshiro(200 + c), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws=1000)
        convert(Matrix{Float64}, run.posterior_position)
    end)
    names = m.view.coordinates
    sd = vec(std(reference; dims=2))
    units = [i for (i, n) in enumerate(names) if m.view.role[i] === :subject_effect]
    globals = setdiff(eachindex(names), units)
    for (label, kwargs) in (("uncentred", (;)), ("learned centering",
            (; centering=LearnedCentering())))
        fit = brm_svi_fit(m; batch_size=8, rng=Xoshiro(3), ndraws=4000, threads=1, kwargs...)
        @test fit.diagnostics.converged
        mean_error = abs.(vec(mean(fit.draws; dims=2)) .- vec(mean(reference; dims=2))) ./ sd
        sd_ratio = vec(std(fit.draws; dims=2)) ./ sd
        println("SVI ", label, " globals ", names[globals], ": mean error / NUTS sd ",
            round.(mean_error[globals]; digits=2), ", sd ratio ",
            round.(sd_ratio[globals]; digits=2))
        println("SVI ", label, " units: mean error median ",
            round(median(mean_error[units]); digits=2), " max ",
            round(maximum(mean_error[units]); digits=2), ", sd ratio median ",
            round(median(sd_ratio[units]); digits=2))
        @test median(mean_error[units]) < 0.3
        @test 0.7 < median(sd_ratio[units]) < 1.25
        @test maximum(mean_error[globals]) < 1.0
        if haskey(kwargs, :centering)
            # The partly known block is centred like an estimated one; the known
            # shared block stays out, by name.
            report = fit.diagnostics.centering
            @test !any(u -> hasproperty(u, :block) && isequal(u.block, family.block),
                report.uncentered)
            @test any(c -> c.kind === :ranef && isequal(c.block, family.block), report.cells)
        end
    end
end

# Another value of the input is the same program: only the input differs.
@stestset "the held value is an input of the conditional density" begin
    a = RKBRMI(held_first(DATA))
    b = RKBRMI(held_first(merge(DATA, (; sd_known=0.9))))
    @test string(a.model.program) == string(b.model.program)
    ma, mb = brm_nlme_model(a; ad_backend=NLME_AD), brm_nlme_model(b; ad_backend=NLME_AD)
    start = brm_nlme_start(ma)
    H = 0.5 .* randn(Xoshiro(4), 2, length(ma.view.levels))
    la = brm_nlme_loglikelihoods(ma, start.theta, start.sigma, H)
    lb = brm_nlme_loglikelihoods(mb, start.theta, start.sigma, H)
    @test la != lb
    named = brm_nlme_estimates(mb, start.theta, start.sigma, start.omega_matrix)
    β = [named.point.mu.Intercept, named.point.mu.x]
    for (j, level) in enumerate(mb.view.levels)
        Z = design_of(level)
        @test lb[j] ≈ sum(logpdf.(Normal.(Z * (β .+ Diagonal([0.9, 1.0]) * H[:, j]),
            named.point.sigma), DATA.y[rows_of(level)])) rtol = 1e-10
    end
end
