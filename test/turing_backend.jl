using Test
using Random: Xoshiro
using BayesianRegressionModels
using Distributions: Binomial, Cauchy, Exponential, Normal, Poisson, logpdf
using LogExpFunctions: logit
using Turing

const BRM = BayesianRegressionModels

@testset "Turing extension — executable population Gaussian" begin
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    @test !isnothing(ext)
    source = read(pathof(ext), String)
    @test !occursin("StanBlocks", source)
    @test !occursin("SBBRMI", source)
    @test !occursin("GenerativePlan", source)

    df = (; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4])
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)

    @test parent(backend) === brmi
    @test backend.model isa Turing.DynamicPPL.Model
    @test backend.plan.design.matrix == [1.0 -1.0; 1.0 0.5; 1.0 2.0]
    @test sprint(show, backend) ==
          "TuringBRMI with 2 population coefficients and 3 observations"

    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = backend.plan.design.matrix * params.beta_pop
    expected = sum(logpdf.(Normal(), params.beta_pop)) +
               logpdf(Exponential(2), params.sigma) +
               sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈ expected atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈
          sum(logpdf.(Normal(), params.beta_pop)) +
          logpdf(Exponential(2), params.sigma) atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          sum(logpdf.(Normal.(mu, params.sigma), df.y)) atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).mu == mu

    prior_draw = rand(Xoshiro(42), backend.model)
    @test prior_draw.data.beta_pop isa Vector{Float64}
    @test prior_draw.data.sigma isa Float64
end

@testset "Turing extension — fitted numeric population transforms" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        w=[-2.0, 1.0, 5.0],
        y=[0.2, 1.1, -0.4],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) + center(w)
        effect(mu, zscale_x) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)

    x_mean = sum(df.x) / length(df.x)
    x_scale = sqrt(sum((df.x .- x_mean) .^ 2) / (length(df.x) - 1))
    w_mean = sum(df.w) / length(df.w)
    expected_X = hcat(
        ones(length(df.y)),
        (df.x .- x_mean) ./ x_scale,
        df.w .- w_mean,
    )
    @test backend.plan.design.matrix ≈ expected_X
    @test Tuple(c.label for c in backend.plan.design.columns) ==
          (:Intercept, :zscale_x, :center_w)
    @test backend.plan.beta_location == [0.0, 0.5, 0.0]
    @test backend.plan.beta_scale == [1.0, 0.25, 1.0]

    params = (; beta_pop=[0.25, -0.5, 0.4], sigma=0.8)
    mu = expected_X * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).mu ≈ mu
end

@testset "Turing extension — continuous transformed interaction" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        w=[-2.0, 1.0, 5.0],
        y=[0.2, 1.1, -0.4],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) & w
        effect(mu, int_zscale_x_x_w) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)

    fitted = BRM._brm_fit_zscale(df.x)
    interaction = BRM._brm_apply_zscale(fitted, df.x) .* df.w
    expected_X = hcat(ones(length(df.y)), interaction)
    @test backend.plan.design.matrix ≈ expected_X
    @test Tuple(c.label for c in backend.plan.design.columns) ==
          (:Intercept, :int_zscale_x_x_w)
    @test backend.plan.beta_location == [0.0, 0.5]
    @test backend.plan.beta_scale == [1.0, 0.25]

    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = expected_X * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).mu ≈ mu
end

@testset "Turing extension — categorical treatment contrasts" begin
    df = (;
        g=[1, 2, 3, 1, 2, 3],
        x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + g + x
        effect(mu, g) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)

    expected_X = hcat(
        ones(6),
        Float64.(df.g .== 2),
        Float64.(df.g .== 3),
        df.x,
    )
    @test backend.plan.design.matrix == expected_X
    @test Tuple(c.label for c in backend.plan.design.columns) ==
          (:Intercept, :g_lvl_2, :g_lvl_3, :x)
    @test backend.plan.beta_location == [0.0, 0.5, 0.5, 0.0]
    @test backend.plan.beta_scale == [1.0, 0.25, 0.25, 1.0]

    params = (; beta_pop=[0.25, -0.5, 0.4, 0.2], sigma=0.8)
    mu = expected_X * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).mu ≈ mu

    declared = BRM.CA.categorical(
        [1, 2, 3, 1, 2, 3]; levels=[3, 1, 2])
    declared_brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + g
        effect(mu, g) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)((; g=declared, y=df.y))
    declared_backend = TuringBRMI(declared_brmi)
    @test declared_backend.plan.design.matrix == hcat(
        ones(6), Float64.(declared .== 1), Float64.(declared .== 2))
    @test declared_backend.plan.design.columns[2].preprocess.const_.levels ==
          [3, 1, 2]
    @test declared_backend.plan.beta_location == [0.0, 0.5, 0.5]
    @test declared_backend.plan.beta_scale == [1.0, 0.25, 0.25]
end

@testset "Turing extension — categorical reference level" begin
    df = (;
        g=[1, 2, 3, 1, 2, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + factor(g; ref=3)
        effect(mu, g) ~ Normal(-0.5, 0.2)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)
    expected_X = hcat(
        ones(6), Float64.(df.g .== 2), Float64.(df.g .== 1))
    @test backend.plan.design.matrix == expected_X
    @test Tuple(c.label for c in backend.plan.design.columns) ==
          (:Intercept, :g__ref_3_lvl_2, :g__ref_3_lvl_3)
    @test backend.plan.beta_location == [0.0, -0.5, -0.5]
    @test backend.plan.beta_scale == [1.0, 0.2, 0.2]

    params = (; beta_pop=[0.25, -0.5, 0.4], sigma=0.8)
    mu = expected_X * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12

    ambiguous = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + factor(g; ref=2) + factor(g; ref=3)
        effect(mu, g) ~ Normal(0, 0.5)
        y ~ Normal(mu, sigma)
    end)(df)
    @test_throws "ambiguously names categorical contrast blocks" begin
        TuringBRMI(ambiguous)
    end
end

@testset "Turing extension — categorical interactions" begin
    df = (;
        x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        g=[1, 2, 3, 1, 2, 3],
        h=[1, 1, 2, 2, 1, 2],
        y=[0.2, 1.1, -0.4, 0.7, 1.4, -0.8],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x & g + g & h
        effect(mu, int_x_x_g_lvl_2) ~ Normal(0.5, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)
    expected_X = hcat(
        ones(6),
        df.x .* (df.g .== 2),
        df.x .* (df.g .== 3),
        (df.g .== 2) .* (df.h .== 2),
        (df.g .== 3) .* (df.h .== 2),
    )
    @test backend.plan.design.matrix == expected_X
    @test Tuple(c.label for c in backend.plan.design.columns) == (
        :Intercept,
        :int_x_x_g_lvl_2,
        :int_x_x_g_lvl_3,
        :int_g_lvl_2_x_h_lvl_2,
        :int_g_lvl_3_x_h_lvl_2,
    )
    @test backend.plan.beta_location == [0.0, 0.5, 0.0, 0.0, 0.0]
    @test backend.plan.beta_scale == [1.0, 0.25, 1.0, 1.0, 1.0]

    params = (; beta_pop=[0.25, -0.5, 0.4, 0.2, -0.1], sigma=0.8)
    mu = expected_X * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
end

@testset "Turing extension — pure expression and exposure offset" begin
    df = (;
        x=[1.0, 2.0, 4.0],
        exposure=[2.0, 4.0, 8.0],
        y=[0, 2, 5],
    )
    brmi = (@brm begin
        log_rate ~ 1 + log(x) + offset(log(exposure))
        y ~ Poisson(exp(log_rate))
    end)(df)
    backend = TuringBRMI(brmi)
    expected_X = hcat(ones(3), log.(df.x))
    @test backend.plan.design.matrix == expected_X
    @test backend.plan.design.fixed == log.(df.exposure)
    @test only(backend.plan.design.fixed_terms).source === :exposure

    params = (; beta_pop=[0.25, -0.5])
    log_rate = expected_X * params.beta_pop + log.(df.exposure)
    rate = exp.(log_rate)
    prior = sum(logpdf.(Normal(), params.beta_pop))
    likelihood = sum(logpdf.(Poisson.(rate), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test returned.log_rate ≈ log_rate
    @test returned.rate ≈ rate
end

@testset "Turing extension — population effect-prior overrides" begin
    df = (; x=[-1.0, 0.5, 2.0], y=[0.2, 1.1, -0.4])
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        effect(:, :) ~ Normal(-1, 3)
        effect(mu, Intercept) ~ Normal(log(2), 0.5)
        effect(mu, x) ~ Normal(0, 0.25)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)

    @test backend.plan.beta_location == [log(2), 0.0]
    @test backend.plan.beta_scale == [0.5, 0.25]
    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = backend.plan.design.matrix * params.beta_pop
    prior = sum(logpdf.(Normal.(backend.plan.beta_location,
                               backend.plan.beta_scale), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12

    wrong_family = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Cauchy(0, 1)
        y ~ Normal(mu, sigma)
    end)(df)
    @test_throws "support only `Normal" TuringBRMI(wrong_family)

    unknown = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, nope) ~ Normal(0, 1)
        y ~ Normal(mu, sigma)
    end)(df)
    @test_throws "not a population coefficient" TuringBRMI(unknown)
end

@testset "Turing extension — Bernoulli-logit population GLM" begin
    df = (; x=[-1.0, 0.5, 2.0, 0.25], y=[0, 1, 1, 0])
    brmi = (@brm begin
        eta ~ 1 + x
        y ~ BernoulliLogit(eta)
    end)(df)
    backend = TuringBRMI(brmi)
    params = (; beta_pop=[-0.2, 0.7])
    eta = backend.plan.design.matrix * params.beta_pop
    prior = sum(logpdf.(Normal(), params.beta_pop))
    likelihood = sum(logpdf.(BRM.BernoulliLogit.(eta), df.y))

    @test backend.plan.family isa Val{:bernoulli_logit}
    @test backend.plan.response == df.y
    @test Turing.logjoint(backend.model, params) ≈ prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈ likelihood atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(backend.model, params).eta == eta
    @test length(rand(Xoshiro(43), backend.model).data.beta_pop) == 2

    canonical = (@brm begin
        logit(p) ~ 1 + x
        y ~ Bernoulli(p)
    end)(df)
    canonical_backend = TuringBRMI(canonical)
    @test canonical_backend.plan.predictor.link_lhs_fn === logit
    @test Turing.logjoint(canonical_backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
end

@testset "Turing extension — Binomial-logit population GLM" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        trials=[2, 4, 6, 3],
        y=[0, 2, 5, 1],
    )
    brmi = (@brm begin
        eta ~ 1 + x
        y ~ BRM.BinomialLogit(trials, eta)
    end)(df)
    sb = SBBRMI(brmi)
    backend = TuringBRMI(brmi)
    params = (; beta_pop=[-0.2, 0.7])
    eta = backend.plan.design.matrix * params.beta_pop
    prior = sum(logpdf.(Normal(), params.beta_pop))
    likelihood = sum(logpdf.(BRM.BinomialLogit.(df.trials, eta), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test backend.plan.family isa Val{:binomial_logit}
    @test backend.plan.family_args.trials == df.trials
    @test sb.data[:trials] == df.trials
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.eta == eta
    @test returned.trials == df.trials

    canonical = (@brm begin
        logit(p) ~ 1 + x
        y ~ Binomial(trials, p)
    end)(df)
    canonical_backend = TuringBRMI(canonical)
    @test canonical_backend.plan.predictor.link_lhs_fn === logit
    @test canonical_backend.plan.family_args.trials == df.trials
    @test Turing.logjoint(canonical_backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12

    constant_trials = (@brm begin
        eta ~ 1 + x
        y ~ BRM.BinomialLogit(6, eta)
    end)(df)
    @test TuringBRMI(constant_trials).plan.family_args.trials == fill(6, 4)

    invalid_trials = (@brm begin
        eta ~ 1 + x
        y ~ BRM.BinomialLogit(trials, eta)
    end)((; x=df.x, trials=[2, 4.5, 6, 3], y=df.y))
    @test_throws "must contain only nonnegative integers" SBBRMI(invalid_trials)
    @test_throws "must contain only nonnegative integers" TuringBRMI(invalid_trials)

    invalid_response = (@brm begin
        eta ~ 1 + x
        y ~ BRM.BinomialLogit(trials, eta)
    end)((; x=df.x, trials=df.trials, y=[0, 5, 5, 1]))
    @test_throws "between zero and its row's trial count" begin
        SBBRMI(invalid_response)
    end
    @test_throws "between zero and its row's trial count" begin
        TuringBRMI(invalid_response)
    end
end

@testset "Turing extension — Poisson-log population GLM" begin
    df = (; x=[-1.0, 0.5, 2.0], y=[0, 2, 5])
    brmi = (@brm begin
        log_rate ~ 1 + x
        y ~ Poisson(exp(log_rate))
    end)(df)
    backend = TuringBRMI(brmi)
    params = (; beta_pop=[0.3, 0.4])
    log_rate = backend.plan.design.matrix * params.beta_pop
    rate = exp.(log_rate)
    prior = sum(logpdf.(Normal(), params.beta_pop))
    likelihood = sum(logpdf.(Poisson.(rate), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test backend.plan.family isa Val{:poisson_log}
    @test backend.plan.predictor.link_lhs_fn === identity
    @test backend.plan.response == df.y
    @test Turing.logjoint(backend.model, params) ≈ prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈ likelihood atol=1e-12 rtol=1e-12
    @test returned.log_rate == log_rate
    @test returned.rate == rate
    @test length(rand(Xoshiro(44), backend.model).data.beta_pop) == 2

    canonical = (@brm begin
        log(lambda) ~ 1 + x
        y ~ Poisson(lambda)
    end)(df)
    canonical_sb = SBBRMI(canonical)
    canonical_backend = TuringBRMI(canonical)
    canonical_returned = Turing.DynamicPPL.returned(
        canonical_backend.model, params)
    @test canonical_backend.plan.predictor.link_lhs_fn === log
    @test canonical_backend.plan.predictor.emitted_name === :log_lambda
    @test canonical_backend.plan.design.matrix == backend.plan.design.matrix
    @test Turing.logjoint(canonical_backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test canonical_returned.log_rate == log_rate
    @test canonical_returned.rate == rate
    @test haskey(canonical_sb.data, :x)

    incompatible = (@brm begin
        lambda ~ 1 + x
        y ~ Poisson(lambda)
    end)(df)
    @test_throws "requires predictor `lambda` to use `log(...)`" begin
        TuringBRMI(incompatible)
    end
end

@testset "Turing extension — distributional NegativeBinomial2" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        y=[0, 2, 5, 1],
    )
    brmi = (@brm begin
        log(mu) ~ 1 + x
        log(phi) ~ 1 + z
        effect(mu, x) ~ Normal(0.25, 0.5)
        effect(phi, Intercept) ~ Normal(-0.2, 0.3)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)(df)
    sb = SBBRMI(brmi)
    backend = TuringBRMI(brmi)
    params = (; beta_mean=[0.1, 0.2], beta_precision=[-0.4, 0.15])
    log_mu = backend.plan.mean.design.matrix * params.beta_mean
    log_phi = backend.plan.precision.design.matrix * params.beta_precision
    mu = exp.(log_mu)
    phi = exp.(log_phi)
    prior = sum(logpdf.(
        Normal.(backend.plan.mean.beta_location, backend.plan.mean.beta_scale),
        params.beta_mean)) + sum(logpdf.(
        Normal.(backend.plan.precision.beta_location,
                backend.plan.precision.beta_scale), params.beta_precision))
    likelihood = sum(logpdf.(BRM.NegativeBinomial2.(mu, phi), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test backend.plan.family isa Val{:negative_binomial2}
    @test backend.plan.mean.predictor.link_lhs_fn === log
    @test backend.plan.precision.predictor.link_lhs_fn === log
    @test backend.plan.mean.beta_location == [0.0, 0.25]
    @test backend.plan.precision.beta_location == [-0.2, 0.0]
    @test sprint(show, backend) ==
          "TuringBRMI with 4 population coefficients and 4 observations"
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.log_mu == log_mu
    @test returned.log_phi == log_phi
    @test returned.mu == mu
    @test returned.phi == phi
    @test haskey(sb.data, :x) && haskey(sb.data, :z)
end

@testset "Turing extension — unsupported shapes fail loudly" begin
    unsupported_string_column = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + group
        y ~ Normal(mu, sigma)
    end)((; group=["a", "b", "a"], y=zeros(3)))
    @test_throws "supports `1`, continuous raw-data columns" begin
        TuringBRMI(unsupported_string_column)
    end

    poisson_identity = (@brm begin
        mu ~ 1 + x
        y ~ Poisson(mu)
    end)((; x=[0.0, 1.0], y=[0, 1]))
    @test_throws "requires predictor `mu` to use `log(...)`" begin
        TuringBRMI(poisson_identity)
    end
end
