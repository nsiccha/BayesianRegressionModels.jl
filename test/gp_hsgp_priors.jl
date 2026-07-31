# test/gp_hsgp_priors.jl — priors on a gp/hsgp term's own scale parameters.
#
# Run: julia --project=. test/gp_hsgp_priors.jl
# Set BRM_GP_PRIOR_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# `test/term_priors.jl` covers the term-internal address shape for `s`, `t2`,
# `mo` and `me`. A Gaussian process owns two more private parameters, and until
# this file they had no address at all:
#
#     length_scale(<lp|:>, gp(x...))    the GP length scale `rho`
#     length_scale(<lp|:>, hsgp(x...))  ditto, for the Hilbert-space basis
#     sd(<lp|:>, gp(x...))              the marginal amplitude `sigma`
#     sd(<lp|:>, hsgp(x...))            ditto
#
# Three invariants the tests below pin down:
#
#   * The base density is REPLACED, not doubled. Two priors on one parameter is
#     a valid Stan program that transpiles, passes stanc and gives a finite
#     density, so every override test asserts the default `lognormal` is GONE —
#     the presence assertion alone would not catch it.
#   * The typed declaration SURVIVES. Three of the six gp/hsgp submodels declare
#     `rho` per-axis (`vector[n_axes]`) and three declare it as a plain scalar;
#     `Base.merge` replaces a matching-named statement WHOLESALE, so an override
#     that does not reproduce the declaration silently drops the type and the
#     constraint (StanBlocks snag `merge-plain-over-f228c5b2`).
#   * Declaration bounds and density support AGREE. A `Uniform(a, b)` length
#     scale emits `<lower=a, upper=b>` on the parameter, so the sampler cannot
#     propose outside the support — the shape a hand-written Stan port uses to
#     hold an HSGP length scale above its own approximation-validity threshold.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Gamma, InverseGamma, LogNormal, Normal, Uniform, Beta

const GP_PRIOR_CACHE = joinpath(tempdir(), "brm-gp-priors")
const GP_PRIOR_RUNTIME = get(ENV, "BRM_GP_PRIOR_RUNTIME", "1") != "0"

const GP_PRIOR_N = 24
gp_prior_df() = (;
    y  = [sin(3i) for i in 1:GP_PRIOR_N],
    x  = collect(range(-1.0, 1.0; length=GP_PRIOR_N)),
    x2 = collect(range(-1.0, 1.0; length=GP_PRIOR_N)) .^ 2,
    g  = repeat(["a", "b"], inner=GP_PRIOR_N ÷ 2),
)

code_of(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
transpiles_and_stanc(brmi) = begin
    sb = SBBRMI(brmi; mod=@__MODULE__)
    StanBlocks.stan.transpiles(sb.model) &&
        StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end

# --------------------------------------------------------------------- default

@testset "unconfigured gp/hsgp keep their historical density" begin
    df = gp_prior_df()
    unconfigured = [
        :gp => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
        end),
        :gp_aniso => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x, x2; iso=false)
        end),
        :hsgp => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
        end),
        :hsgp_aniso => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
        end),
        :hsgp_by => (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=4, by=g)
        end),
    ]
    for (label, m) in unconfigured
        code = code_of(m)
        @test occursin("lognormal(0.0, 1.0)", code)
        @test !occursin("uniform(", code)
        @test transpiles_and_stanc(m)
        @test isempty(term_priors(m))
        label === :gp_aniso || label === :hsgp_aniso ||
            @test occursin("<lower=0.0>", code)
    end
end

# --------------------------------------------- length_scale, isotropic (scalar)

@testset "length_scale on an isotropic term keeps its scalar declaration" begin
    df = gp_prior_df()

    m_gp = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + gp(x)
        length_scale(:, gp(x)) ~ InverseGamma(5, 5)
    end
    code = code_of(m_gp)
    @test occursin("real<lower=0.0> gp_x_rho;", code)
    @test occursin("gp_x_rho ~ inv_gamma(5.0, 5.0);", code)
    # `sigma` was not addressed, so it keeps the default -- and the ONE
    # remaining `lognormal` in the program is its.
    @test occursin("gp_x_sigma ~ lognormal(0.0, 1.0);", code)
    @test count(l -> occursin("lognormal", l), split(code, '\n')) == 1
    @test transpiles_and_stanc(m_gp)

    # The isotropic HSGP samples ONE length scale named `rho_iso` and broadcasts
    # it with `rep_vector`; the override has to name that parameter, not `rho`.
    m_hsgp = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ InverseGamma(5, 5)
    end
    code = code_of(m_hsgp)
    @test occursin("real<lower=0.0> hsgp_x_rho_iso;", code)
    @test occursin("hsgp_x_rho_iso ~ inv_gamma(5.0, 5.0);", code)
    @test occursin("rep_vector(hsgp_x_rho_iso", code)
    @test !occursin("hsgp_x_rho_iso ~ lognormal", code)
    @test transpiles_and_stanc(m_hsgp)
end

# ------------------------------------------ length_scale, anisotropic (vector)

@testset "length_scale on an anisotropic term keeps its per-axis declaration" begin
    df = gp_prior_df()
    for (label, m, prefix) in (
        (:gp, (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x, x2; iso=false)
            length_scale(:, gp(x, x2)) ~ LogNormal(1.0, 0.3)
        end), "gp_x_x2"),
        (:hsgp, (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
            length_scale(:, hsgp(x, x2)) ~ LogNormal(1.0, 0.3)
        end), "hsgp_x_x2"),
    )
        code = code_of(m)
        # The declaration a plain-LHS override would silently drop: BOTH the
        # per-axis container and the positivity constraint have to survive.
        @test occursin("vector<lower=0.0>[$(prefix)_n_axes] $(prefix)_rho;", code)
        @test occursin("$(prefix)_rho ~ lognormal(1.0, 0.3);", code)
        @test !occursin("$(prefix)_rho ~ lognormal(0.0, 1.0);", code)
        @test transpiles_and_stanc(m)
    end
end

# ---------------------------------------------------- bounded uniform + amplitude

@testset "a Uniform length scale bounds the declaration to its support" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
        sd(:, hsgp(x)) ~ Normal(0, 0.5)
    end
    code = code_of(m)
    @test occursin("real<lower=0.8366, upper=2.0> hsgp_x_rho_iso;", code)
    @test occursin("hsgp_x_rho_iso ~ uniform(0.8366, 2.0);", code)
    # Amplitude addressed through `sd`, so it is a half-normal, not lognormal.
    @test occursin("real<lower=0.0> hsgp_x_sigma;", code)
    @test occursin("hsgp_x_sigma ~ normal(0.0, 0.5);", code)
    # Both parameters were configured, so NO default survives anywhere.
    @test !occursin("lognormal", code)
    @test transpiles_and_stanc(m)
end

@testset "a per-group hsgp shares one configured length scale" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=4, by=g)
        length_scale(:, hsgp(x)) ~ Uniform(0.5, 2.0)
        sd(:, hsgp(x)) ~ Exponential(2.0)
    end
    code = code_of(m)
    @test occursin("real<lower=0.5, upper=2.0> hsgp_x_by_g_rho_iso;", code)
    @test occursin("hsgp_x_by_g_rho_iso ~ uniform(0.5, 2.0);", code)
    # Distributions.Exponential takes a SCALE; Stan's exponential takes a rate.
    @test occursin("hsgp_x_by_g_sigma ~ exponential(0.5);", code)
    @test !occursin("lognormal", code)
    # Only the tensor-basis weights vary per group (decision `7p44fo`); the
    # configured hyperparameters stay shared.
    @test occursin("rep_vector(hsgp_x_by_g_rho_iso", code)
    @test transpiles_and_stanc(m)
end

# ----------------------------------------------------- parameterisation mapping

@testset "Julia constructors map onto Stan's parameterisation" begin
    df = gp_prior_df()
    cases = [
        (:(Gamma(2.0, 0.5)), "gamma(2.0, 2.0)"),         # scale -> rate
        (:(Exponential(4.0)), "exponential(0.25)"),      # scale -> rate
        (:(InverseGamma(5, 5)), "inv_gamma(5.0, 5.0)"),  # direct
        (:(LogNormal(0.0, 0.5)), "lognormal(0.0, 0.5)"), # direct
        (:(Normal(0, 2.0)), "normal(0.0, 2.0)"),         # half-normal at lower=0
    ]
    for (call, expected) in cases
        m = @eval @brm $df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
            length_scale(:, gp(x)) ~ $call
        end
        code = code_of(m)
        @test occursin("gp_x_rho ~ $expected;", code)
        @test occursin("real<lower=0.0> gp_x_rho;", code)
        @test transpiles_and_stanc(m)
    end
end

# ------------------------------------------------------------------ predictors

@testset "the predictor slot resolves like every other term address" begin
    df = gp_prior_df()
    # A named linear predictor beats the `:` default layer.
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ InverseGamma(5, 5)
        length_scale(mu, hsgp(x)) ~ Uniform(0.5, 2.0)
    end
    code = code_of(m)
    @test occursin("hsgp_x_rho_iso ~ uniform(0.5, 2.0);", code)
    @test !occursin("inv_gamma", code)
    @test transpiles_and_stanc(m)
end

# ------------------------------------------------------------------ inspection

@testset "term_priors reports the new class" begin
    df = gp_prior_df()
    m = @brm df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
        sd(mu, hsgp(x)) ~ Normal(0, 0.5)
    end
    specs = term_priors(m)
    @test length(specs) == 2
    ls = only(p for p in specs if p.class === :term_length_scale)
    @test ls.term === Symbol("hsgp(x)")
    @test isnothing(ls.predictor)
    @test isnothing(ls.component)
    @test ls.family === Uniform
    @test ls.arguments == (0.8366, 2.0)
    amp = only(p for p in specs if p.class === :term_sd)
    @test amp.term === Symbol("hsgp(x)")
    @test amp.predictor === :mu
end

# -------------------------------------------------------------------- refusals

@testset "mis-addressed gp/hsgp scale priors are refused by name" begin
    df = gp_prior_df()
    # Parse-time refusals: the target slot must be a term, not a bare symbol.
    @test_throws "must be a term as the formula spells it" (@eval @brm $df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + hsgp(x; k=5)
        length_scale(:, g) ~ Uniform(0.5, 2.0)
    end)

    # Resolve-time refusals: the class reaches a term with no such parameter, or
    # the family/bounds are not usable for a positive scale.
    cases = [
        ("has no length scale to configure", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + s(x)
            length_scale(:, s(x)) ~ Uniform(0.5, 2.0)
        end)),
        ("supports `LogNormal`", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Beta(2, 2)
        end)),
        ("0 <= lower < upper", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Uniform(-1.0, 2.0)
        end)),
        ("matches no `hsgp(x)` term", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + gp(x)
            length_scale(:, hsgp(x)) ~ Uniform(0.5, 2.0)
        end)),
        ("does not accept keywords", (@brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Uniform(0.5, 2.0; lower=0.1)
        end)),
    ]
    for (fragment, m) in cases
        err = try
            code_of(m)
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin(fragment, sprint(showerror, err))
    end
end

# -------------------------------------------------------------------- runtime

@testset "BridgeStan finite density and gradient" begin
    if GP_PRIOR_RUNTIME
        df = gp_prior_df()
        m = @brm df begin
            y ~ Normal(mu, 1.)
            mu ~ 1 + hsgp(x; k=5)
            length_scale(:, hsgp(x)) ~ Uniform(0.8366, 2.0)
            sd(:, hsgp(x)) ~ Normal(0, 0.5)
        end
        sb = SBBRMI(m; mod=@__MODULE__)
        isdir(GP_PRIOR_CACHE) || mkpath(GP_PRIOR_CACHE)
        code = StanBlocks.stan_code(sb.model)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(GP_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping BridgeStan gp/hsgp prior runtime gate (BRM_GP_PRIOR_RUNTIME=0)"
        @test true
    end
end
