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
                     Poisson, logpdf, product_distribution
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
