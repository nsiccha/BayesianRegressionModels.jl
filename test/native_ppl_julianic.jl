# Julianic "run-the-body" NativePPL surface (`NP.@jmodel`).
#
# The load-bearing invariant: every julianic model must match the declarative
# `@model` executor as a DIFFERENTIAL ORACLE — log density AND Enzyme gradient
# agreeing over random unconstrained positions — for as long as the declarative
# executor exists. Each model class below is authored twice, once per surface,
# and pinned against the other here rather than in a throwaway script.
#
# The second half pins the fail-closed boundaries. They matter more than usual
# on this surface because latent-vs-observation is decided SYNTACTICALLY: a
# scalar `~` on a conditioned name is a well-formed program that scores the
# wrong model, so it has to be an error, not a result.

using Test
using BayesianRegressionModels
import DifferentiationInterface as DI
using Distributions: Bernoulli, Dirichlet, Exponential, LKJCholesky, Normal,
                     Poisson, censored, logpdf, product_distribution
using Enzyme
using LogExpFunctions: logistic
using LogDensityProblems
using Random: MersenneTwister, randn

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL

# The SAME backend the declarative executor uses. This is load-bearing, not
# incidental: the run-the-body primal briefly required
# `Enzyme.set_runtime_activity` because its mutable context mixed the constant
# conditioned data with the active accumulator. The data is now threaded through
# as a separate `DI.Constant`, so static activity analysis suffices — and pinning
# the plain backend here is what keeps a regression from silently reintroducing
# the workaround.
const JULIANIC_BACKEND = DI.AutoEnzyme()

# Charter tolerance. The observed gaps are ~1e-14, i.e. float roundoff on
# identical arithmetic; 1e-9 is the contract, not the measurement.
const ORACLE_TOLERANCE = 1e-9

"""
Compare a julianic model against the declarative executor over random
unconstrained positions. Returns the worst density/gradient gaps so a caller
can report them, and asserts both the dimension match and the tolerance.
"""
function oracle_parity(declarative, julianic, observations; positions=6, seed=20260804)
    oracle = NP.prepare(NP.compile(
        NP.condition(NP.substitute(declarative); observations...)))
    prepared = NP.jprepare(NP.jcondition(julianic; observations...))

    @test NP.dimension(prepared) == LogDensityProblems.dimension(oracle)

    oracle_work = NP.workspace(oracle, Float64, DI.AutoEnzyme())
    julianic_work = NP.jworkspace(prepared, Float64, JULIANIC_BACKEND)

    rng = MersenneTwister(seed)
    density_gap = 0.0
    gradient_gap = 0.0
    for _ in 1:positions
        theta = randn(rng, NP.dimension(prepared))
        oracle_density = NP.logdensity!(oracle_work, oracle, theta)
        julianic_density = NP.jlogdensity(prepared, theta)
        _, oracle_gradient =
            NP.logdensity_and_gradient!(oracle_work, oracle, theta)
        _, julianic_gradient =
            NP.jlogdensity_and_gradient!(julianic_work, prepared, theta)
        density_gap = max(density_gap, abs(oracle_density - julianic_density))
        gradient_gap = max(
            gradient_gap, maximum(abs, oracle_gradient .- julianic_gradient))
    end

    @test density_gap <= ORACLE_TOLERANCE
    @test gradient_gap <= ORACLE_TOLERANCE
    return (; density_gap, gradient_gap)
end

argument_error(f) = try
    f()
    error("expected an ArgumentError")
catch err
    err isa ArgumentError ? err : rethrow()
end

# ---------------------------------------------------------------------------
# Model pairs. Where the two surfaces accept the SAME source text, it is written
# the same way on purpose — that identity is the point of the julianic surface.
# ---------------------------------------------------------------------------

NP.@model function declarative_affine_gaussian(x)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_affine_gaussian(x)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ Normal(mu, sigma)
end

# The latent-depends-on-latents edge. Under run-the-body it needs no special
# machinery: by the time the `individual` line runs, `population` and
# `population_scale` are ordinary bound Julia variables.
NP.@model function declarative_hierarchical()
    population ~ Normal()
    population_scale ~ Exponential(1.0)
    individual ~ Normal(population, population_scale)
    observation_scale ~ Exponential(2.0)
    @. y ~ Normal(individual, observation_scale)
end

NP.@jmodel function julianic_hierarchical()
    population ~ Normal()
    population_scale ~ Exponential(1.0)
    individual ~ Normal(population, population_scale)
    observation_scale ~ Exponential(2.0)
    @. y ~ Normal(individual, observation_scale)
end

# Distributional / heteroskedastic: mean AND log-scale regressions. The
# declarative side's coefficient BLOCK becomes plain scalar priors plus an
# explicit affine; the θ layout is declaration order either way.
NP.@model function declarative_distributional(x, z)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    beta_log_sigma[(:Intercept, :z)] ~ StandardNormal()
    mu = dot(beta_mu, (1, x))
    log_sigma = dot(beta_log_sigma, (1, z))
    sigma = exp(log_sigma)
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_distributional(x, z)
    beta_mu_intercept ~ Normal()
    beta_mu_x ~ Normal()
    beta_log_sigma_intercept ~ Normal()
    beta_log_sigma_z ~ Normal()
    mu = beta_mu_intercept .+ beta_mu_x .* x
    log_sigma = beta_log_sigma_intercept .+ beta_log_sigma_z .* z
    sigma = exp.(log_sigma)
    @. y ~ Normal(mu, sigma)
end

NP.@model function declarative_poisson(x)
    intercept ~ Normal()
    slope ~ Normal()
    log_rate = intercept .+ slope .* x
    @. y ~ Poisson(exp(log_rate))
end

NP.@jmodel function julianic_poisson(x)
    intercept ~ Normal()
    slope ~ Normal()
    log_rate = intercept .+ slope .* x
    @. y ~ Poisson(exp(log_rate))
end

# The declarative side has a bespoke `BernoulliLogit` factor; the julianic side
# needs no such thing — `Bernoulli(logistic(eta))` is ordinary Julia and must
# reproduce it exactly.
NP.@model function declarative_bernoulli(x)
    intercept ~ Normal()
    slope ~ Normal()
    eta = intercept .+ slope .* x
    @. y ~ BernoulliLogit(eta)
end

NP.@jmodel function julianic_bernoulli(x)
    intercept ~ Normal()
    slope ~ Normal()
    eta = intercept .+ slope .* x
    @. y ~ Bernoulli(logistic(eta))
end

# Grouped / blocked latents: one multivariate `~` consuming a contiguous block,
# indexed afterwards in ordinary Julia.
NP.@model function declarative_varying_intercept(x, group)
    tau_p_group ~ Exponential(1)
    b_p_group[group] ~ Normal(0.0, tau_p_group)
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + b_p_group[group]
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_varying_intercept(x, group)
    tau_p_group ~ Exponential(1.0)
    b_p_group ~ product_distribution(
        fill(Normal(0.0, tau_p_group), maximum(group)))
    beta ~ Normal()
    sigma ~ Exponential(2.0)
    mu = beta .* x .+ b_p_group[group]
    @. y ~ Normal(mu, sigma)
end

# Correlated grouped intercept+slope — the model class LKJ latents exist for,
# and the sharpest oracle test on this surface: it pins the LKJ manifold
# transform, its fused log-Jacobian, the positive-support VECTOR of scales, and
# the executor's NON-CENTERED storage all at once.
#
# The declarative executor stores `b_p_group ~ MvNormalCholesky(tau, L)`
# non-centered: the coordinates are standard-normal `z`, and the effect is the
# deterministic `diag(tau) * L * z`. Run-the-body has no such storage rule, so
# the julianic author writes that reparameterization out by hand — which is
# exactly why it needs pinning. The `z` block is GROUP-major / coefficient-minor
# (`graph.coordinates.b_p_group.keys`, test/native_ppl.jl:5620), so `z[2g-1]` is
# group g's intercept coordinate and `z[2g]` its slope.
NP.@model function declarative_correlated_varying(x, group)
    tau_p_group[(:Intercept, :x)] ~ Exponential(1)
    L_p_group[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta ~ Normal()
    sigma ~ Exponential(2)
    mu = beta * x + dot(b_p_group[group], (1, x))
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_correlated_varying(x, group)
    groups = maximum(group)
    tau_p_group ~ product_distribution(fill(Exponential(1.0), 2))
    L_p_group ~ LKJCholesky(2, 2)
    z_p_group ~ product_distribution(fill(Normal(0.0, 1.0), 2 * groups))
    beta ~ Normal()
    sigma ~ Exponential(2.0)
    # b = diag(tau) * L * z, written out per coefficient. `L` is lower
    # triangular with a unit first diagonal, so the intercept row is just z.
    z_intercept = z_p_group[1:2:end]
    z_slope = z_p_group[2:2:end]
    b_intercept = tau_p_group[1] .* (L_p_group[1, 1] .* z_intercept)
    b_slope = tau_p_group[2] .*
        (L_p_group[2, 1] .* z_intercept .+ L_p_group[2, 2] .* z_slope)
    mu = beta .* x .+ b_intercept[group] .+ b_slope[group] .* x
    @. y ~ Normal(mu, sigma)
end

# K=3 is not redundant with K=2 above: at K=2 the correlation factor has ONE raw
# coordinate, so `_julianic_corr_cholesky`'s address arithmetic
# (`(s-1)*K - (s-1)*s÷2 + c - s`) and `_julianic_lkj_logdensity`'s per-column
# `alpha` both degenerate to a single term. K=3 is the first case that actually
# exercises the column walk — and the first that would catch a transposed or
# off-by-one coordinate mapping, which at K=2 is invisible.
NP.@model function declarative_correlated_varying_three(x, w, group)
    tau_p_group[(:Intercept, :x, :w)] ~ Exponential(1)
    L_p_group[(:Intercept, :x, :w)] ~ LKJCholesky(3, 2)
    b_p_group[group, (:Intercept, :x, :w)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu[(:x, :w)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (x, w)) + dot(b_p_group[group], (1, x, w))
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_correlated_varying_three(x, w, group)
    groups = maximum(group)
    tau_p_group ~ product_distribution(fill(Exponential(1.0), 3))
    L_p_group ~ LKJCholesky(3, 2)
    z_p_group ~ product_distribution(fill(Normal(0.0, 1.0), 3 * groups))
    beta_mu ~ product_distribution(fill(Normal(0.0, 1.0), 2))
    sigma ~ Exponential(2.0)
    # Column g of `z` is group g's (Intercept, x, w) block — the group-major
    # layout the executor uses. `b = diag(tau) * L * z`, one column per group.
    z = reshape(z_p_group, 3, groups)
    b = tau_p_group .* (L_p_group * z)
    mu = beta_mu[1] .* x .+ beta_mu[2] .* w .+
        b[1, group] .+ b[2, group] .* x .+ b[3, group] .* w
    @. y ~ Normal(mu, sigma)
end

# Distributional-GROUPED: one correlated group effect loading on BOTH the mean
# and the log-scale. This is the hardest composition in the gallery — it stacks
# the LKJ manifold, the positive-support scale vector, the non-centered block and
# two separate regressions reading different rows of the same effect.
NP.@model function declarative_grouped_distributional(x, z, group)
    tau_p_group[(:mu, :log_sigma)] ~ Exponential(1)
    L_p_group[(:mu, :log_sigma)] ~ LKJCholesky(2, 2)
    b_p_group[group, (:mu, :log_sigma)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    beta_log_sigma[(:Intercept, :z)] ~ StandardNormal()
    mu = dot(beta_mu, (1, x)) + dot(b_p_group[group, (:mu,)], (1,))
    log_sigma = dot(beta_log_sigma, (1, z)) +
        dot(b_p_group[group, (:log_sigma,)], (1,))
    sigma = exp(log_sigma)
    @. y ~ Normal(mu, sigma)
end

NP.@jmodel function julianic_grouped_distributional(x, z, group)
    groups = maximum(group)
    tau_p_group ~ product_distribution(fill(Exponential(1.0), 2))
    L_p_group ~ LKJCholesky(2, 2)
    z_p_group ~ product_distribution(fill(Normal(0.0, 1.0), 2 * groups))
    beta_mu ~ product_distribution(fill(Normal(0.0, 1.0), 2))
    beta_log_sigma ~ product_distribution(fill(Normal(0.0, 1.0), 2))
    b = tau_p_group .* (L_p_group * reshape(z_p_group, 2, groups))
    mu = beta_mu[1] .+ beta_mu[2] .* x .+ b[1, group]
    log_sigma = beta_log_sigma[1] .+ beta_log_sigma[2] .* z .+ b[2, group]
    sigma = exp.(log_sigma)
    @. y ~ Normal(mu, sigma)
end

# Censored observations need NOTHING from the `~` runtime: `censored(...)` is an
# ordinary `Distributions` object with a `logpdf`, so the julianic body writes
# the same call the declarative macro pattern-matches. That is the general claim
# the surface makes about likelihoods — worth one pinned instance.
NP.@model function declarative_censored(x, group)
    tau_g_group ~ Exponential(1)
    b_g_group[group] ~ Normal(0.0, tau_g_group)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (1, x)) + b_g_group[group]
    @. y ~ censored(Normal(mu, sigma); lower=-0.5, upper=1.0)
end

NP.@jmodel function julianic_censored(x, group)
    tau_g_group ~ Exponential(1.0)
    b_g_group ~ product_distribution(
        fill(Normal(0.0, tau_g_group), maximum(group)))
    beta_mu ~ product_distribution(fill(Normal(0.0, 1.0), 2))
    sigma ~ Exponential(2.0)
    mu = beta_mu[1] .+ beta_mu[2] .* x .+ b_g_group[group]
    @. y ~ censored(Normal(mu, sigma); lower=-0.5, upper=1.0)
end

# Observation weights. The declarative surface carries the weight KIND in the
# StatsBase constructor's NAME, because a declaration cannot say `sigma/sqrt(w)`
# — the weight type has to say it for the author. A julianic body is Julia, so it
# says it directly, and the two kinds land in different places:
#
#   :frequency / :power  MULTIPLY the log density   -> `weighted(dist, w)`, the
#                                                      one thing plain Julia has
#                                                      no spelling for
#   :analytic            scale PRECISION            -> written out, no wrapper
#
# Both are pinned against the declarative oracle here, since the second is a
# claim about arithmetic identity (`Normal(mu, sigma/sqrt(w))` IS the analytic
# weighting) rather than about the `~` runtime. Decision `06le2au`.
NP.@model function declarative_frequency_weighted(x, replicates)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ weighted(Normal(mu, sigma), fweights(replicates))
end

NP.@jmodel function julianic_frequency_weighted(x, replicates)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ weighted(Normal(mu, sigma), replicates)
end

NP.@model function declarative_analytic_weighted(x, replicates)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ weighted(Normal(mu, sigma), aweights(replicates))
end

NP.@jmodel function julianic_analytic_weighted(x, replicates)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    @. y ~ Normal(mu, sigma / sqrt(replicates))
end

# `y .~ D.(...)` — the spelling the declarative surface uses, where the author
# has already dotted the RHS. Must be the same model as the `@.` form.
NP.@jmodel function julianic_affine_gaussian_dotted(x)
    intercept ~ Normal()
    slope ~ Normal()
    sigma ~ Exponential(2.0)
    mu = intercept .+ slope .* x
    y .~ Normal.(mu, sigma)
end

const PREDICTOR = [-1.2, 0.3, 0.9, 2.1, -0.4, 1.5]
const SECONDARY = [0.4, -0.8, 1.7, 0.2, -1.1, 0.6]
const GROUP = [1, 2, 1, 3, 2, 3]
const CONTINUOUS = (; y=[0.1, -0.5, 1.3, 2.2, 0.0, 0.8])
const COUNTS = (; y=[0, 2, 1, 4, 1, 3])
const BINARY = (; y=[0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
# Censoring puts point mass at the bounds, so every observation must lie INSIDE
# `[lower, upper]` — an out-of-range value scores `-Inf` on both surfaces and
# would make the comparison vacuous rather than wrong.
const CENSORED = (; y=[0.1, -0.5, 1.0, 1.0, 0.0, 0.8])
# Frequency weights are counts; analytic weights are arbitrary positive reals.
# Neither is all-ones — an all-ones weight makes every kind agree and would pass
# whatever the wrapper did.
const FREQUENCIES = [1, 3, 2, 1, 4, 2]
const ANALYTIC = [0.5, 2.0, 1.0, 3.0, 0.25, 1.5]

# Everything below is nested inside ONE outer set on purpose. A top-level
# `@testset` throws as it finishes if anything inside it failed, which aborts the
# rest of the FILE — so a single broken model class in the parity sweep takes every
# later set down with it, unrun and unreported. Nested sets do not throw
# individually: they report up, the outer set throws once at the very end, and a
# failing run still tells you the state of the whole file.
@testset "julianic run-the-body surface" begin

@testset "julianic oracle parity" begin
    @testset "flat affine gaussian" begin
        oracle_parity(
            declarative_affine_gaussian(PREDICTOR),
            julianic_affine_gaussian(PREDICTOR),
            CONTINUOUS)
    end

    @testset "centered hierarchical" begin
        oracle_parity(
            declarative_hierarchical(), julianic_hierarchical(), CONTINUOUS)
    end

    @testset "distributional gaussian" begin
        oracle_parity(
            declarative_distributional(PREDICTOR, SECONDARY),
            julianic_distributional(PREDICTOR, SECONDARY),
            CONTINUOUS)
    end

    @testset "poisson log link" begin
        oracle_parity(
            declarative_poisson(PREDICTOR), julianic_poisson(PREDICTOR), COUNTS)
    end

    @testset "bernoulli logit link" begin
        oracle_parity(
            declarative_bernoulli(PREDICTOR),
            julianic_bernoulli(PREDICTOR),
            BINARY)
    end

    @testset "grouped varying intercept" begin
        oracle_parity(
            declarative_varying_intercept(PREDICTOR, GROUP),
            julianic_varying_intercept(PREDICTOR, GROUP),
            CONTINUOUS)
    end

    @testset "correlated varying intercept and slope (LKJ)" begin
        oracle_parity(
            declarative_correlated_varying(PREDICTOR, GROUP),
            julianic_correlated_varying(PREDICTOR, GROUP),
            CONTINUOUS)
    end

    @testset "correlated K=3 block (LKJ column walk)" begin
        oracle_parity(
            declarative_correlated_varying_three(PREDICTOR, SECONDARY, GROUP),
            julianic_correlated_varying_three(PREDICTOR, SECONDARY, GROUP),
            CONTINUOUS)
    end

    @testset "distributional-grouped LKJ (effect on mean AND log-scale)" begin
        oracle_parity(
            declarative_grouped_distributional(PREDICTOR, SECONDARY, GROUP),
            julianic_grouped_distributional(PREDICTOR, SECONDARY, GROUP),
            CONTINUOUS)
    end

    @testset "censored observation" begin
        oracle_parity(
            declarative_censored(PREDICTOR, GROUP),
            julianic_censored(PREDICTOR, GROUP),
            CENSORED)
    end

    @testset "frequency-weighted observation" begin
        oracle_parity(
            declarative_frequency_weighted(PREDICTOR, FREQUENCIES),
            julianic_frequency_weighted(PREDICTOR, FREQUENCIES),
            CONTINUOUS)
    end

    @testset "analytic-weighted observation (precision scaling)" begin
        oracle_parity(
            declarative_analytic_weighted(PREDICTOR, ANALYTIC),
            julianic_analytic_weighted(PREDICTOR, ANALYTIC),
            CONTINUOUS)
    end
end

# The `weighted` wrapper is a real `Distributions.Distribution`, not a macro
# device — that is the whole point of the julianic surface, so it has to hold
# OUTSIDE a model body too.
@testset "weighted is an ordinary distribution" begin
    base = Normal(0.5, 2.0)
    @test logpdf(weighted(base, 3.0), 0.3) ≈ 3.0 * logpdf(base, 0.3)
    @test logpdf(weighted(base, 1.0), 0.3) == logpdf(base, 0.3)
    # Elementwise under broadcast — the form a model body actually produces.
    @test logpdf.(weighted.(Normal.([0.1, 0.2], 1.0), [2.0, 5.0]), [0.0, 1.0]) ≈
        [2.0, 5.0] .* logpdf.(Normal.([0.1, 0.2], 1.0), [0.0, 1.0])
    # The un-dotted StatsBase spellings, kept identical to the declarative
    # surface. Frequency/power apply; the two kinds this wrapper cannot express
    # fail closed instead of silently multiplying.
    @test logpdf.(weighted(base, fweights([1, 2])), [0.0, 1.0]) ≈
        [1, 2] .* logpdf.(base, [0.0, 1.0])
    @test occursin("analytic weights scale PRECISION",
                   argument_error(() -> weighted(base, aweights([1.0, 2.0]))).msg)
    @test occursin("ProbabilityWeights semantics are not implemented",
                   argument_error(() -> weighted(base, pweights([1.0, 2.0]))).msg)
end

# Steady-state allocations. The declarative executor pins
# `(primal=0, gradient=0)` and the julianic kernel is held to the SAME bar
# (decision `08w0buk`): every array-valued intermediate is filled in place into a
# preallocated pool threaded as a `DI.Cache`, so a gradient eval allocates
# nothing no matter how many data rows there are — the 32-bytes-per-row cost this
# surface used to carry is gone, not merely reduced.
#
# The measurement MUST happen inside a function with typed locals: at global
# scope `@allocated` boxes the untyped globals and reports a meaningless nonzero
# number, which is exactly how a false "DI has an allocation floor" conclusion
# gets reached.
# Both entry points take the WORKSPACE — `jlogdensity(prepared, theta)` without
# one allocates its pool per call by construction, so it is the convenience form,
# not the one this bar is about.
function steady_state_allocations(prepared, workspace, theta)
    NP.jlogdensity!(workspace, prepared, theta)
    NP.jlogdensity_and_gradient!(workspace, prepared, theta)
    primal = @allocated NP.jlogdensity!(workspace, prepared, theta)
    gradient = @allocated NP.jlogdensity_and_gradient!(workspace, prepared, theta)
    return (; primal, gradient)
end

@testset "julianic steady-state allocations" begin
    # Scaled 10x past the pinned gallery size, because a per-row leak is what
    # this is guarding against — it would be invisible at n=6.
    rng = MersenneTwister(20260804)
    x = randn(rng, 60)
    weights = rand(rng, 60) .+ 0.5
    counts = rand(rng, 1:4, 60)
    observations = (; y=randn(rng, 60))

    for (name, model) in (
            "affine gaussian" => julianic_affine_gaussian(x),
            "frequency-weighted" => julianic_frequency_weighted(x, counts),
            "analytic-weighted" => julianic_analytic_weighted(x, weights))
        prepared = NP.jprepare(NP.jcondition(model; observations...))
        workspace = NP.jworkspace(prepared, Float64, JULIANIC_BACKEND)
        theta = randn(rng, NP.dimension(prepared))
        # The pooled primal must be the SAME number as the allocating one — a
        # buffer reused across evals without a proper rewind would show up here
        # and nowhere else.
        @test NP.jlogdensity!(workspace, prepared, theta) ==
            NP.jlogdensity(prepared, theta)
        allocations = steady_state_allocations(prepared, workspace, theta)
        @test allocations.primal == 0
        @test allocations.gradient == 0
        allocations == (primal=0, gradient=0) ||
            @info "julianic allocations" model=name allocations
    end
end

# Downstream consumers (WarmupHMC, the samplers) take a `LogDensityProblem`, not
# a `Prepared` — so julianic cannot stand in for the declarative surface without
# this, and it has to be the SAME problem, not merely a similar one.
@testset "julianic LogDensityProblems interface" begin
    prepared = NP.jprepare(NP.jcondition(
        julianic_affine_gaussian(PREDICTOR); CONTINUOUS...))
    problem = NP.JulianicLogDensityProblem(prepared, JULIANIC_BACKEND)
    oracle = NP.LogDensityProblem(
        NP.prepare(NP.compile(NP.condition(
            NP.substitute(declarative_affine_gaussian(PREDICTOR)); CONTINUOUS...))),
        DI.AutoEnzyme())

    @test LogDensityProblems.capabilities(typeof(problem)) ==
        LogDensityProblems.LogDensityOrder{1}()
    @test LogDensityProblems.dimension(problem) ==
        LogDensityProblems.dimension(oracle)
    @test eltype(problem) == Float64

    rng = MersenneTwister(20260805)
    for _ in 1:4
        theta = randn(rng, LogDensityProblems.dimension(problem))
        @test LogDensityProblems.logdensity(problem, theta) ≈
            LogDensityProblems.logdensity(oracle, theta) atol=ORACLE_TOLERANCE
        density, gradient = LogDensityProblems.logdensity_and_gradient(problem, theta)
        oracle_density, oracle_gradient =
            LogDensityProblems.logdensity_and_gradient(oracle, theta)
        @test density ≈ oracle_density atol=ORACLE_TOLERANCE
        @test maximum(abs, gradient .- oracle_gradient) <= ORACLE_TOLERANCE
    end

    # The gradient must be a COPY — the workspace buffer is reused, so handing
    # out an alias would let the next call silently rewrite a caller's result.
    theta = zeros(LogDensityProblems.dimension(problem))
    _, first_gradient = LogDensityProblems.logdensity_and_gradient(problem, theta)
    saved = copy(first_gradient)
    LogDensityProblems.logdensity_and_gradient(problem, theta .+ 1.0)
    @test first_gradient == saved
end

@testset "julianic observation spellings agree" begin
    dotted = NP.jprepare(NP.jcondition(
        julianic_affine_gaussian_dotted(PREDICTOR); CONTINUOUS...))
    macroed = NP.jprepare(NP.jcondition(
        julianic_affine_gaussian(PREDICTOR); CONTINUOUS...))
    @test NP.dimension(dotted) == NP.dimension(macroed)
    for theta in ([0.0, 0.0, 0.0], [0.4, -1.1, 0.7], [-2.0, 1.3, -0.6])
        @test NP.jlogdensity(dotted, theta) == NP.jlogdensity(macroed, theta)
    end
end

# ---------------------------------------------------------------------------
# Fail-closed boundaries.
# ---------------------------------------------------------------------------

NP.@jmodel function julianic_scalar_site_on_data(x)
    mu ~ Normal()
    y ~ Normal(mu, 1.0)
end

NP.@jmodel function julianic_unconditioned_observation(x)
    mu ~ Normal()
    @. absent ~ Normal(mu, 1.0)
end

# A positive-support vector IS supported (its own `_sample!`, exp transform per
# coordinate) — this is the model the guard below must NOT catch.
NP.@jmodel function julianic_positive_block(x)
    scales ~ product_distribution(fill(Exponential(1.0), 3))
    # `@.` dots the WHOLE statement, so the reduction has to be hoisted out —
    # inline it and `sum(scales)` becomes the elementwise `sum.(scales)`.
    total = sum(scales)
    @. y ~ Normal(total, 1.0)
end

# A `Dirichlet` block has no `_sample!` of its own, so it would fall through to
# the real-support multivariate method — identity transform, zero log-Jacobian,
# and simplex coordinates read straight off an unconstrained vector.
NP.@jmodel function julianic_constrained_block(x)
    weights ~ Dirichlet([1.0, 1.0, 1.0])
    @. y ~ Normal(sum(weights), 1.0)
end

NP.@jmodel function julianic_upper_lkj(x)
    corr ~ LKJCholesky(2, 2.0, :U)
    @. y ~ Normal(corr[2, 1], 1.0)
end

# ---------------------------------------------------------------------------
# `end` / `begin` in index position.
#
# The 0-alloc lowering hoists an index expression out of `A[...]` and into a
# `_cached_gather!(ctx, A, idx)` argument. `end`/`begin` only mean anything
# INSIDE the bracket, so the hoist has to resolve them against the array first.
# When it did not, ordinary Julia like `z[1:2:end]` expanded cleanly and then
# died at `jprepare` with `UndefVarError: end not defined` — no macro-time
# complaint, and a whole model class (any non-centered reparameterization that
# strides a latent block) unusable. The spelling must not change the answer, so
# each model below is checked against its literal-index twin.
# ---------------------------------------------------------------------------

NP.@jmodel function julianic_index_end_range(x)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[1:2:end])
    @. y ~ Normal(total, sigma)
end

NP.@jmodel function julianic_index_literal_range(x)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[1:2:3])
    @. y ~ Normal(total, sigma)
end

NP.@jmodel function julianic_index_end_scalar(x)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[begin:end])
    @. y ~ Normal(total, sigma)
end

NP.@jmodel function julianic_index_literal_scalar(x)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[1:4])
    @. y ~ Normal(total, sigma)
end

# A nested `:ref` carries its OWN `end`: the inner one belongs to `selector`,
# the outer to `z`. Resolving both against the outer array would silently gather
# the wrong coordinates whenever the two lengths differ — as they do here (2 vs 4).
NP.@jmodel function julianic_index_end_nested(x, selector)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[selector[begin:end]])
    @. y ~ Normal(total, sigma)
end

NP.@jmodel function julianic_index_literal_nested(x, selector)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[selector[1:2]])
    @. y ~ Normal(total, sigma)
end

@testset "julianic `end`/`begin` index resolution" begin
    selector = [2, 4]
    # `gradient` is false for the nested case ONLY because differentiating it is
    # blocked upstream, not because `end` resolves differently there: gathering a
    # `Vector{Int}` index out of a view into θ trips `EnzymeRuntimeActivityError`
    # (snag `gather-subarray-11c41349`). Range indices — the other two cases —
    # differentiate fine. Restore the gradient check when that snag lands.
    cases = (
        ("strided range `z[1:2:end]`", true,
         julianic_index_end_range(PREDICTOR),
         julianic_index_literal_range(PREDICTOR)),
        ("whole span `z[begin:end]`", true,
         julianic_index_end_scalar(PREDICTOR),
         julianic_index_literal_scalar(PREDICTOR)),
        ("nested `z[selector[begin:end]]`", false,
         julianic_index_end_nested(PREDICTOR, selector),
         julianic_index_literal_nested(PREDICTOR, selector)),
    )
    for (name, gradient, with_end, with_literal) in cases
        @testset "$name" begin
            # Regression: this `jprepare` is where `UndefVarError: end` struck.
            bounded = NP.jprepare(NP.jcondition(with_end; CONTINUOUS...))
            literal = NP.jprepare(NP.jcondition(with_literal; CONTINUOUS...))
            @test NP.dimension(bounded) == NP.dimension(literal)

            rng = MersenneTwister(20260805)
            thetas = [randn(rng, NP.dimension(bounded)) for _ in 1:4]
            for theta in thetas
                # Same arithmetic in the same order — an exact match, not a
                # tolerance.
                @test NP.jlogdensity(bounded, theta) ==
                      NP.jlogdensity(literal, theta)
            end
            gradient || continue

            bounded_work = NP.jworkspace(bounded, Float64, JULIANIC_BACKEND)
            literal_work = NP.jworkspace(literal, Float64, JULIANIC_BACKEND)
            for theta in thetas
                _, bounded_gradient =
                    NP.jlogdensity_and_gradient!(bounded_work, bounded, theta)
                _, literal_gradient =
                    NP.jlogdensity_and_gradient!(literal_work, literal, theta)
                @test bounded_gradient == literal_gradient
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Non-float intermediates must not be routed through the buffer pool.
#
# The pool is a flat `Vector{T}` at the AD eltype. Sending an integer or boolean
# intermediate through it retypes the values: an index vector `[2, 4]` comes back
# `[2.0, 4.0]` and stops being an index, a `Bool` mask comes back `[1.0, 0.0, …]`
# and stops being a mask. Both passed `jprepare` — the TRACE pass returns the real
# `getindex`, so the model looked fine right up to the first density evaluation.
# Each model is checked against a twin that receives the same intermediate
# precomputed as data, so the in-body computation cannot change the answer.
# ---------------------------------------------------------------------------

NP.@jmodel function julianic_integer_intermediate(x, selector, group)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    picked = selector[group]
    effect = z[picked]
    @. y ~ Normal(effect, sigma)
end

NP.@jmodel function julianic_integer_precomputed(x, picked)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    effect = z[picked]
    @. y ~ Normal(effect, sigma)
end

NP.@jmodel function julianic_boolean_mask(x, mask_source)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    mask = mask_source[1:4]
    total = sum(z[mask])
    @. y ~ Normal(total, sigma)
end

NP.@jmodel function julianic_boolean_precomputed(x, mask)
    z ~ product_distribution(fill(Normal(0.0, 1.0), 4))
    sigma ~ Exponential(1.0)
    total = sum(z[mask])
    @. y ~ Normal(total, sigma)
end

@testset "julianic non-float intermediates keep their type" begin
    selector = [2, 4]
    group = [1, 2, 1, 2, 1, 2]
    mask = [true, false, true, false]
    pairs = (
        ("integer gather `selector[group]` reused as an index",
         julianic_integer_intermediate(PREDICTOR, selector, group),
         julianic_integer_precomputed(PREDICTOR, selector[group])),
        ("boolean gather `mask_source[1:4]` reused as a mask",
         julianic_boolean_mask(PREDICTOR, mask),
         julianic_boolean_precomputed(PREDICTOR, mask)),
    )
    # Density only. Both models index θ with a computed integer/boolean VECTOR,
    # and gathering one of those out of a view into θ is blocked upstream by
    # `EnzymeRuntimeActivityError` (snag `gather-subarray-11c41349`) — a separate
    # defect from the retyping under test here, which is a primal-side bug and
    # shows up in the density alone. Add the gradient leg when that snag lands.
    for (name, in_body, precomputed) in pairs
        @testset "$name" begin
            computed = NP.jprepare(NP.jcondition(in_body; CONTINUOUS...))
            given = NP.jprepare(NP.jcondition(precomputed; CONTINUOUS...))
            @test NP.dimension(computed) == NP.dimension(given)

            rng = MersenneTwister(20260805)
            for _ in 1:4
                theta = randn(rng, NP.dimension(computed))
                # Regression: the primal used to throw `invalid index: 2.0 of
                # type Float64` (integers) / `BoundsError … eltype Float64`
                # (booleans) right here, while `jprepare` had reported success.
                @test NP.jlogdensity(computed, theta) ==
                      NP.jlogdensity(given, theta)
            end
        end
    end
end

@testset "julianic fails closed" begin
    @testset "scalar site on conditioned data" begin
        # Without this check the site consumes a θ coordinate, scores the PRIOR
        # at θ, rebinds `y`, and never reads the observation — a well-formed
        # program computing the wrong density. The declarative `@model` accepts
        # this exact spelling as an observation, so silence is not an option.
        err = argument_error(() -> NP.jprepare(NP.jcondition(
            julianic_scalar_site_on_data(PREDICTOR); y=3.0)))
        @test occursin("conditioned data", err.msg)
        @test occursin("y", err.msg)

        # …and the same body IS a valid latent when `y` is not conditioned.
        latent_only = NP.jprepare(NP.jcondition(
            julianic_scalar_site_on_data(PREDICTOR)))
        @test NP.dimension(latent_only) == 2
        @test NP.jlogdensity(latent_only, [0.5, 0.7]) ≈
              logpdf(Normal(), 0.5) + logpdf(Normal(0.5, 1.0), 0.7)
    end

    @testset "observation with no conditioned data" begin
        err = argument_error(() -> NP.jprepare(NP.jcondition(
            julianic_unconditioned_observation(PREDICTOR); CONTINUOUS...)))
        @test occursin("absent", err.msg)
        @test occursin("conditioned sites: y", err.msg)
    end

    @testset "constrained multivariate block" begin
        # The multivariate `_sample!` dispatches on the whole
        # `MultivariateDistribution` type but implements only the real-support
        # case (identity transform, zero log-Jacobian). A family with no method
        # of its own must be refused rather than scored under the wrong measure
        # — the scalar `Exponential` site gets an exp transform, so silently
        # skipping the constraint here would make the two spellings disagree.
        err = argument_error(() -> NP.jprepare(NP.jcondition(
            julianic_constrained_block(PREDICTOR); CONTINUOUS...)))
        @test occursin("support is not all of", err.msg)
        @test occursin("weights", err.msg)

        # …but the guard must not fire on a family that DOES have a method. A
        # positive-support vector takes the exp path, so it agrees coordinate
        # for coordinate with three scalar `Exponential` sites.
        positive = NP.jprepare(NP.jcondition(
            julianic_positive_block(PREDICTOR); CONTINUOUS...))
        @test NP.dimension(positive) == 3
        raw = [0.3, -0.7, 1.1]
        expected = sum(logpdf.(Exponential(1.0), exp.(raw))) + sum(raw) +
            sum(logpdf.(Normal(sum(exp.(raw)), 1.0), CONTINUOUS.y))
        @test NP.jlogdensity(positive, raw) ≈ expected
    end

    @testset "LKJ site declared upper-triangular" begin
        # The density is `uplo`-invariant, so an unguarded `uplo=:U` site scores
        # correctly and hands the BODY a transposed factor — a wrong answer that
        # only ever surfaces downstream. Refuse at prepare instead.
        err = argument_error(() -> NP.jprepare(NP.jcondition(
            julianic_upper_lkj(PREDICTOR); CONTINUOUS...)))
        @test occursin("uplo", err.msg)
        @test occursin("corr", err.msg)
    end

    @testset "position of the wrong length" begin
        # The cursor reads are deliberately NOT `@inbounds`: a short position
        # must be a `DimensionMismatch`, never an out-of-bounds read feeding
        # garbage into Enzyme.
        prepared = NP.jprepare(NP.jcondition(
            julianic_affine_gaussian(PREDICTOR); CONTINUOUS...))
        @test NP.dimension(prepared) == 3
        @test_throws DimensionMismatch NP.jlogdensity(prepared, [0.1, 0.2])
        @test_throws DimensionMismatch NP.jlogdensity(
            prepared, [0.1, 0.2, 0.3, 0.4])
        workspace = NP.jworkspace(prepared, Float64, JULIANIC_BACKEND)
        @test_throws DimensionMismatch NP.jlogdensity_and_gradient!(
            workspace, prepared, [0.1, 0.2])
    end

    @testset "sampling nested in control flow" begin
        # Milestone 1 lowers TOP-LEVEL statements only. A nested `~` used to
        # survive verbatim and fail at run time with an `UndefVarError` naming
        # the site — a message that points nowhere near the cause.
        nested_loop = :(function nested(x)
            s ~ Exponential(1.0)
            for i in 1:2
                z ~ Normal()
            end
            @. y ~ Normal(0.0, s)
        end)
        err = argument_error(() -> NP._julianic_model_syntax(nested_loop))
        @test occursin("TOP LEVEL", err.msg)

        nested_branch = :(function nested(x)
            s ~ Exponential(1.0)
            if length(x) > 1
                @. y ~ Normal(0.0, s)
            end
        end)
        @test occursin(
            "TOP LEVEL",
            argument_error(() -> NP._julianic_model_syntax(nested_branch)).msg)

        # Unary `~` is ordinary Julia (bitwise not) and must be left alone.
        bitwise = :(function fine(x)
            s ~ Exponential(1.0)
            mask = ~UInt8(3)
            @. y ~ Normal(Float64(mask), s)
        end)
        @test NP._julianic_model_syntax(bitwise) isa Expr
    end

    @testset "unsupported syntax" begin
        @test_throws ArgumentError NP._julianic_model_syntax(:(f(x) = x))
        @test_throws ArgumentError NP._julianic_model_syntax(
            :(function f(x); x[1] ~ Normal(); end))
    end
end

end # @testset "julianic run-the-body surface"
