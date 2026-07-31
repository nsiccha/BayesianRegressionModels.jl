# test/r2d2_priors.jl — R2D2 variance-decomposition priors (`effect(lp, :) ~ r2d2(...)`).
#
# Run: julia --project=. test/r2d2_priors.jl
# Set BRM_R2D2_RUNTIME=0 to skip the BridgeStan density/gradient probe.
#
# The inciting case is the population-PK shape: a latent per-subject parameter
# whose subject random effect PLAYS the residual role, so `R2` splits its total
# between-subject variance `tau_bsv^2` into a covariate-explained part (split
# further over the population columns by a Dirichlet simplex) and an
# unexplained part that becomes the random-effect SD.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Beta, Exponential, Normal

const R2D2_CACHE = joinpath(tempdir(), "brm-r2d2-priors")
const R2D2_RUNTIME = get(ENV, "BRM_R2D2_RUNTIME", "1") != "0"

df = (;
    conc    = [2.4, 2.2, 2.0, 1.8, 1.7, 1.5, 1.6, 1.9],
    time    = [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
    wt      = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.25, 0.75],
    age     = [0.2, -0.4, 1.1, -0.9, 0.3, 0.7, -1.2, 0.1],
    subject = [1, 1, 1, 1, 2, 2, 2, 2],
)

builder = @brm begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    log_V  ~ 1 + wt + (1 | p | subject)

    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_V, :)  ~ r2d2(R2=Beta(2, 3), tau_bsv=0.25, alpha=0.5)

    conc ~ Normal(exp(log_CL - log_V) * time, 1)
end

@testset "r2d2 capture, introspection, and lowering" begin
    brmi = builder(df)

    specs = r2d2_priors(brmi)
    @test [s.predictor for s in specs] == [:log_CL, :log_V]
    @test all(s -> s.family === r2d2, specs)
    @test all(s -> isempty(s.arguments), specs)
    # Keyword values stay as unevaluated column carriers -- the backend turns
    # them into Stan constants, so the formula surface never materializes a
    # Distributions object.
    @test getf(specs[1].keywords.R2) === Beta
    @test getargs(specs[1].keywords.R2) == (1, 1)
    @test specs[1].keywords.tau_bsv == 0.5
    @test getargs(specs[2].keywords.R2) == (2, 3)
    @test specs[2].keywords.alpha == 0.5

    # A whole-predictor `Colon` address belongs to `r2d2_priors` alone -- it
    # names every population coefficient at once, so it must not leak into the
    # per-column population API.
    @test isempty(effect_priors(brmi))
    @test isempty(ranef_effect_priors(brmi))

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # `R2` is the constrained unit-interval parameter; `phi` the simplex over
    # the NON-intercept population columns only (the intercept design column is
    # constant, so `Var(x_k) = 0` there and it carries no explained variance).
    @test occursin("real<lower=0, upper=1> r2d2_log_CL_R2;", code)
    @test occursin("simplex[r2d2_log_CL_alpha_n] r2d2_log_CL_phi;", code)
    @test occursin("r2d2_log_CL_R2 ~ beta(1.0, 1.0);", code)
    @test occursin("r2d2_log_V_R2 ~ beta(2.0, 3.0);", code)
    @test occursin("r2d2_log_CL_phi ~ dirichlet(r2d2_log_CL_alpha);", code)
    @test sb.data[:r2d2_log_CL_alpha] == [1.0, 1.0]
    @test sb.data[:r2d2_log_CL_share_idx] == [0, 1, 2]
    @test sb.data[:r2d2_log_CL_tau_bsv] == 0.5

    # A single share needs no simplex parameter at all -- `phi == [1]` is a
    # data constant, and `log_V` has exactly one non-intercept column.
    @test sb.data[:r2d2_log_V_phi] == [1.0]
    @test !occursin("simplex[r2d2_log_V_alpha_n]", code)
    @test !haskey(sb.data, :r2d2_log_V_alpha)

    # Column variances are data-only, so they hoist to transformed data.
    @test occursin("vector[(2 + 1)] r2d2_log_CL_varx = brm_col_variances(", code)
    # The derived scale is a transformed parameter feeding the shipped
    # `_popefs_normal` seam.
    @test occursin("r2d2_log_CL_beta_scale = brm_r2d2_scale(", code)
    @test occursin(
        "pop_log_CL_beta_pop ~ normal(r2d2_log_CL_beta_loc, r2d2_log_CL_beta_scale);",
        code)

    # The shared `|p|` block's marginal scales are the derived residual scales,
    # one per predictor, and LKJ eta stays free at 1.
    @test occursin("sqrt(((1.0 - r2d2_log_CL_R2) * (r2d2_log_CL_tau_bsv ^ 2)))", code)
    @test occursin("sqrt(((1.0 - r2d2_log_V_R2) * (r2d2_log_V_tau_bsv ^ 2)))", code)
    @test occursin("vector[2] b_p_subject_r2d2_tau = [", code)
    @test occursin("b_p_subject_L ~ lkj_corr_cholesky(1.0);", code)
    # No sampled SD prior survives -- the decomposition derives the scales.
    @test !occursin("brm_ranef_sd", code)

    if R2D2_RUNTIME
        isdir(R2D2_CACHE) || mkpath(R2D2_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model;
            path=joinpath(R2D2_CACHE, string(hash(code)) * ".stan"),
        )
        dimension = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:dimension]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)
    else
        @info "Skipping r2d2 BridgeStan gate (BRM_R2D2_RUNTIME=0)"
    end
end

@testset "plain (un-ID'd) random effect and the sampled-tau default" begin
    # No `tau_bsv=` kwarg: the between-subject scale becomes a sampled
    # half-standard-normal. A latent per-subject parameter has no observed
    # response to derive a scale from, so there is nothing else to default to.
    plain_builder = @brm begin
        eta ~ 1 + wt + age + (1 | subject)
        effect(:, :) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta, 1)
    end
    plain = SBBRMI(plain_builder(df); mod=@__MODULE__)
    plain_code = BayesianRegressionModels.stan_code(plain)
    @test StanBlocks.stan.transpiles(plain.model)
    @test StanBlocks.stanc_check(plain_code; warn_pedantic=false).ok
    @test occursin("real<lower=0.0> r2d2_eta_tau_bsv;", plain_code)
    @test occursin("r2d2_eta_tau_bsv ~ std_normal();", plain_code)
    @test occursin("r_eta_subject_r2d2_scale = sqrt(((1.0 - r2d2_eta_R2) * (r2d2_eta_tau_bsv ^ 2)))",
                   plain_code)
    @test !occursin("ranef_intercept(", plain_code)
end

@testset "byte-identical emission when no r2d2 statement is present" begin
    control_builder = @brm begin
        eta ~ 1 + wt + age + (1 | subject)
        conc ~ Normal(eta, 1)
    end
    control = SBBRMI(control_builder(df); mod=@__MODULE__)
    control_code = BayesianRegressionModels.stan_code(control)
    @test !occursin("r2d2", control_code)
    @test occursin("pop_eta_beta_pop ~ std_normal();", control_code)
    @test occursin("r_eta_subject", control_code)
end

@testset "rejected shapes error loudly" begin
    # `:` is positional in every slot now, so `effect(:, eta)` parses — but an
    # r2d2 decomposition is whole-predictor by construction, so its address
    # still has to END in `:`. The refusal moved from the parser to the walker.
    not_whole = @brm begin
        eta ~ 1 + wt
        effect(:, eta) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta, 1)
    end
    @test_throws "must end in" SBBRMI(not_whole(df); mod=@__MODULE__)

    # `effect(:, :) ~ r2d2(...)` only resolves when there is exactly one
    # population predictor to decompose.
    ambiguous = @brm begin
        eta_a ~ 1 + wt
        eta_b ~ 1 + age
        effect(:, :) ~ r2d2(R2=Beta(1, 1))
        conc ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "linear predictor" SBBRMI(ambiguous(df); mod=@__MODULE__)

    unknown_kwarg = @brm begin
        eta ~ 1 + wt
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), nope=1)
        conc ~ Normal(eta, 1)
    end
    @test_throws "nope" SBBRMI(unknown_kwarg(df); mod=@__MODULE__)

    # All-or-nothing per shared `|ID|` bucket (decision `1db6zkr`): a bucket
    # whose margins are only partly r2d2-scoped has no consistent tau vector.
    partial_bucket = @brm begin
        eta_a ~ 1 + wt + (1 | p | subject)
        eta_b ~ 1 + age + (1 | p | subject)
        effect(eta_a, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta_a + eta_b, 1)
    end
    @test_throws "all-or-nothing" SBBRMI(partial_bucket(df); mod=@__MODULE__)

    # An `sd(:, ...)` prior on an r2d2 block has nothing to apply to.
    sd_conflict = @brm begin
        eta ~ 1 + wt + (1 | p | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        sd(:, p) ~ Exponential(1)
        conc ~ Normal(eta, 1)
    end
    @test_throws "DERIVES" SBBRMI(sd_conflict(df); mod=@__MODULE__)

    # Splitting the derived residual variance across several RE margins needs a
    # second simplex the flat decomposition does not build.
    multi_margin = @brm begin
        eta ~ 1 + wt + (1 + wt | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "single intercept" SBBRMI(multi_margin(df); mod=@__MODULE__)

    multi_margin_id = @brm begin
        eta ~ 1 + wt + (1 + wt | p | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "second simplex" SBBRMI(multi_margin_id(df); mod=@__MODULE__)

    # Adaptive centering and cv-contagious sizing both interact with a derived
    # scale in ways nobody has designed.
    centered = @brm begin
        eta ~ 1 + wt + (1 | subject)
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "centered" SBBRMI(centered(df); mod=@__MODULE__,
                                   centered_groups=[:subject])
    @test_throws "cv_groups" SBBRMI(centered(df); mod=@__MODULE__,
                                    cv_groups=[:subject])

    stratified_df = merge(df, (; stratum=[1, 1, 1, 1, 2, 2, 2, 2]))
    stratified = @brm begin
        eta ~ 1 + wt + (1 | p | gr(subject, by=stratum))
        effect(eta, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        conc ~ Normal(eta, 1)
    end
    @test_throws "stratified" SBBRMI(stratified(stratified_df); mod=@__MODULE__)
end
