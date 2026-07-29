# test/effect_priors.jl — named population-coefficient prior overrides.
#
# Run: julia --project=. test/effect_priors.jl
# Set BRM_EFFECT_RUNTIME=0 to skip the BridgeStan density/gradient probe.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Exponential, Normal

const EFFECT_PRIOR_CACHE = joinpath(tempdir(), "brm-effect-priors")
const EFFECT_PRIOR_RUNTIME = get(ENV, "BRM_EFFECT_RUNTIME", "1") != "0"

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

builder = @brm begin
    log_ka ~ 1 + x + (1 | pk | subject)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    effect(log_ka, x) ~ Normal(0, 0.1)
    y ~ Normal(log_ka, 0.2)
end

@testset "effect prior capture and lowering" begin
    brmi = builder(df)
    @test popcoefnames(brmi, :log_ka) == [:Intercept, :x]
    @test length(linear_predictors(brmi)) == 1

    priors = effect_priors(brmi)
    @test length(priors) == 2
    @test [(p.predictor, p.coefficient) for p in priors] ==
          [(:log_ka, :Intercept), (:log_ka, :x)]
    @test all(p -> p.family <: Normal, priors)
    @test all(isempty(p.keywords) for p in priors)
    @test occursin("effect(log_ka, Intercept) ~ Normal", sprint(show, brmi))

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("vector[pop_log_ka_n_covariates] pop_log_ka_beta_pop;", code)
    @test occursin("pop_log_ka_beta_pop ~ normal(", code)
    @test occursin(string(log(1 / 8)), code)
    @test occursin("[0.8, 0.1]'", code)

    plan = generative_plan(sb)
    pop_decl = only(d for d in plan.declarations if d.target === :pop_log_ka)
    @test pop_decl.family === :_popefs_normal
    @test pop_decl.role === :prior

    descriptor = brm_descriptor(sb)
    byname = Dict(o.name => o for o in descriptor.outputs)
    @test byname[:pop_log_ka].role === :population_effect
    @test byname[:pop_log_ka_beta_pop].role === :population_effect
    @test byname[:pop_log_ka_beta_pop].labels == [:Intercept, :x]

    if EFFECT_PRIOR_RUNTIME
        isdir(EFFECT_PRIOR_CACHE) || mkpath(EFFECT_PRIOR_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(EFFECT_PRIOR_CACHE, string(hash(code)) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping effect-prior BridgeStan gate (BRM_EFFECT_RUNTIME=0)"
    end
end

@testset "defaults, concise address, and validation" begin
    default = @brm df begin
        mu ~ 1 + x
        y ~ Normal(mu, 1)
    end
    @test occursin("pop_mu_beta_pop ~ std_normal();",
                   BayesianRegressionModels.stan_code(SBBRMI(default; mod=@__MODULE__)))

    concise = @brm df begin
        mu ~ 1 + x
        effect(x) ~ Normal(0, 0.25)
        y ~ Normal(mu, 1)
    end
    @test occursin("[1.0, 0.25]'",
                   BayesianRegressionModels.stan_code(SBBRMI(concise; mod=@__MODULE__)))

    ambiguous = @brm df begin
        a ~ 1
        b ~ 1
        effect(Intercept) ~ Normal(0, 2)
        y ~ Normal(a + b, 1)
    end
    @test_throws "ambiguous" SBBRMI(ambiguous; mod=@__MODULE__)

    excluded = @brm df begin
        mu ~ 1 + x + (1 | subject)
        effect(mu, subject) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1)
    end
    @test_throws "not a population coefficient" SBBRMI(excluded; mod=@__MODULE__)

    unsupported = @brm df begin
        mu ~ 1 + x
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1)
    end
    @test_throws "support only `Normal" SBBRMI(unsupported; mod=@__MODULE__)
end

# `linear_predictors` reports EVERY non-data `~` binding, not just population
# formulas: a scalar parameter prior (`log_F_bottle ~ Normal(0, 0.5)`) and a
# submodel binding (`pred ~ kernel(...) do … end`) are both in it, and neither
# has `beta_pop` columns for `popcoefnames` to name. Resolving labels for every
# entry the moment ANY effect prior existed therefore made one unrelated
# statement fatal for the whole model — `effect(...)` overrides and ordinary
# scalar priors could not coexist. Labels are now resolved lazily, per address.
@testset "unrelated statements do not block effect-prior resolution" begin
    explicit = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test [lp.name for lp in linear_predictors(explicit)] == [:log_F_bottle, :mu]
    @test popcoefnames(explicit, :mu) == [:Intercept, :x]
    # The scalar prior is exactly the entry `popcoefnames` cannot name.
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(explicit, :log_F_bottle)

    sb = SBBRMI(explicit; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    # The scalar prior stays an ordinary parameter…
    @test occursin("real log_F_bottle;", code)
    @test occursin("log_F_bottle ~ normal(0, 0.5);", code)
    # …and the override still reaches only the addressed population column.
    @test occursin("pop_mu_beta_pop ~ normal([0.0, 0]', [1.0, 0.25]');", code)

    # The concise form is the path that enumerates candidates, so it must skip
    # the unnameable entries rather than fail on them.
    concise = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test occursin("pop_mu_beta_pop ~ normal([0.0, 0]', [1.0, 0.25]');",
                   BayesianRegressionModels.stan_code(SBBRMI(concise; mod=@__MODULE__)))

    # Addressing the scalar prior itself is still an error, and names what IS
    # addressable instead of leaking `popcoefnames`' internal failure.
    at_prior = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(log_F_bottle, x) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test_throws "names no linear predictor with population coefficients" SBBRMI(
        at_prior; mod=@__MODULE__)

    # An unknown concise label is still rejected — tolerance must not mute this.
    unknown = @brm df begin
        log_F_bottle ~ Normal(0, 0.5)
        mu ~ 1 + x
        effect(nope) ~ Normal(0, 0.25)
        y ~ Normal(mu + log_F_bottle, 1)
    end
    @test_throws "matches no population coefficient" SBBRMI(unknown; mod=@__MODULE__)
end

# The PMX shape the snag was reported against: a population PK model carries a
# scalar prior (`sigma ~ Exponential(1)`) AND a `kernel(...)` submodel binding
# whose columns `popcoefnames` cannot name — two distinct unnameable entries —
# alongside `effect(...)` overrides on the per-subject linear predictors.
@testset "effect priors coexist with a kernel(...) PMX model" begin
    n = 6
    pk_df = (; t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:n],
               dose    = fill(100.0, n),
               dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:n],
               subject = collect(1:n))

    pk = @brm pk_df begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        log_V  ~ 1 + (1 | p | subject)
        log_Ka ~ 1 + (1 | p | subject)
        effect(log_CL, Intercept) ~ Normal(log(1 / 8), 0.8)
        pred ~ kernel(t, dose, dv, log_CL, log_V, log_Ka) do ts, d, yy, lCL, lV, lKa
            CL = exp(lCL); Vc = exp(lV); Ka = exp(lKa)
            ke = CL / Vc
            mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
            yy ~ normal(mu, sigma)
            mu
        end
    end
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(pk, :sigma)
    @test_throws "cannot resolve the `beta_pop` column(s)" popcoefnames(pk, :pred)
    @test popcoefnames(pk, :log_CL) == [:Intercept]

    sb = SBBRMI(pk; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("pop_log_CL_beta_pop ~ normal([$(log(1 / 8))]', [0.8]');", code)
    # Unaddressed predictors keep the default, and the scalar prior is untouched.
    @test occursin("pop_log_V_beta_pop ~ std_normal();", code)
    @test occursin("sigma ~ exponential(", code)
end
