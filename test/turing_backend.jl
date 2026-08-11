using Test
using Random: Xoshiro
using BayesianRegressionModels
using Distributions: Binomial, Cauchy, Exponential, LKJCholesky, Normal,
                     Poisson, censored, cdf, logpdf, truncated
using LinearAlgebra: Diagonal, Symmetric, cholesky
using LogExpFunctions: logistic, logit
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

    renamed_df = (; x=df.x, outcome=df.y)
    renamed = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        outcome ~ Normal(mu, sigma)
    end)(renamed_df))
    @test renamed.plan.response == renamed_df.outcome
    @test Turing.logjoint(renamed.model, params) ≈ expected atol=1e-12 rtol=1e-12
end


@testset "Turing extension — partially missing Gaussian response" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        y=Union{Missing,Float64}[0.2, missing, -0.4, missing],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end)(df))
    missing_plan = backend.plan.missing_response

    @test missing_plan.source === :y
    @test missing_plan.observed_indices == [1, 3]
    @test missing_plan.missing_indices == [2, 4]
    @test missing_plan.observed_values == [0.2, -0.4]
    @test isequal(backend.plan.response, df.y)

    draw = rand(Xoshiro(73), backend.model)
    returned = Turing.DynamicPPL.returned(backend.model, draw.data)
    params = draw.data
    mu = backend.plan.design.matrix * params.beta_pop
    complete_y = returned.response
    coefficient_prior = sum(logpdf.(Normal(), params.beta_pop))
    scale_prior = logpdf(Exponential(2), params.sigma)
    observed_likelihood = sum(missing_plan.observed_indices) do i
        logpdf(Normal(mu[i], params.sigma), complete_y[i])
    end
    imputation_prior = sum(missing_plan.missing_indices) do i
        logpdf(Normal(mu[i], params.sigma), complete_y[i])
    end

    @test !any(ismissing, complete_y)
    @test complete_y[missing_plan.observed_indices] ==
          missing_plan.observed_values
    @test Turing.logjoint(backend.model, params) ≈
          coefficient_prior + scale_prior + observed_likelihood +
          imputation_prior atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈
          coefficient_prior + scale_prior + imputation_prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          observed_likelihood atol=1e-12 rtol=1e-12

    complete = merge(df, (; y=Union{Missing,Float64}[0.2, 0.1, -0.4, 0.3]))
    @test_throws "found no missing values" begin
        TuringBRMI((@brm begin
            sigma ~ Exponential(2)
            mu ~ 1 + x
            mi(y) ~ Normal(mu, sigma)
        end)(complete))
    end

    binary = (; x=df.x, y=Union{Missing,Int}[1, missing, 0, 1])
    @test_throws "currently supports only `Normal" begin
        TuringBRMI((@brm begin
            eta ~ 1 + x
            mi(y) ~ BernoulliLogit(eta)
        end)(binary))
    end
end


@testset "Turing extension — truncated Gaussian response" begin
    df = (;
        x=[-1.0, 0.5, 2.0],
        lower=[-0.7, -0.2, -0.5],
        outcome=[-0.4, 0.6, 0.9],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        outcome ~ truncated(
            Normal(mu, sigma); lower=lower, upper=1.25)
    end)(df)
    backend = TuringBRMI(brmi)
    modifier = backend.plan.response_modifier

    @test modifier.kind === :truncated
    @test modifier.lower == df.lower
    @test modifier.upper == 1.25

    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = backend.plan.design.matrix * params.beta_pop
    likelihood = sum(eachindex(df.outcome)) do i
        logpdf(truncated(
            Normal(mu[i], params.sigma);
            lower=df.lower[i], upper=1.25), df.outcome[i])
    end
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12

    outside = merge(df, (; outcome=[-0.8, 0.6, 0.9]))
    @test_throws "contains values outside its bounds" begin
        TuringBRMI((@brm begin
            sigma ~ Exponential(2)
            mu ~ 1 + x
            outcome ~ truncated(
                Normal(mu, sigma); lower=lower, upper=1.25)
        end)(outside))
    end
end


@testset "Turing extension — censored Gaussian response" begin
    df = (;
        x=[-1.0, 0.5, 2.0],
        lower=[-0.5, -0.2, -0.5],
        outcome=[-0.5, 0.6, 1.0],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        outcome ~ censored(
            Normal(mu, sigma); lower=lower, upper=1.0)
    end)(df))
    modifier = backend.plan.response_modifier

    @test modifier.kind === :censored
    @test modifier.lower == df.lower
    @test modifier.upper == 1.0

    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = backend.plan.design.matrix * params.beta_pop
    likelihood = sum(eachindex(df.outcome)) do i
        logpdf(censored(
            Normal(mu[i], params.sigma);
            lower=df.lower[i], upper=1.0), df.outcome[i])
    end
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
end


@testset "Turing extension — analytic Normal weights" begin
    df = (;
        x=[-1.0, 0.5, 2.0],
        precision=[1.0, 4.0, 2.25],
        y=[-0.4, 0.6, 0.9],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ weighted(Normal(mu, sigma), aweights(precision))
    end)(df))
    weight = backend.plan.observation_weight
    @test weight.kind === :analytic
    @test weight.source === :precision
    @test weight.values == df.precision

    params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = backend.plan.design.matrix * params.beta_pop
    likelihood = sum(logpdf.(
        Normal.(mu, params.sigma ./ sqrt.(df.precision)), df.y))
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12

    invalid = merge(df, (; precision=[1.0, 0.0, 2.25]))
    @test_throws "must be strictly positive" begin
        TuringBRMI((@brm begin
            sigma ~ Exponential(2)
            mu ~ 1 + x
            y ~ weighted(Normal(mu, sigma), aweights(precision))
        end)(invalid))
    end
end


@testset "Turing extension — frequency/power objective weights" begin
    normal_data = (;
        x=[-1.0, 0.5, 2.0],
        repeats=[1, 3, 2],
        y=[-0.4, 0.6, 0.9],
    )
    normal_backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ weighted(Normal(mu, sigma), fweights(repeats))
    end)(normal_data))
    normal_weight = normal_backend.plan.observation_weight
    normal_params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = normal_backend.plan.design.matrix * normal_params.beta_pop
    normal_likelihood = sum(eachindex(normal_data.y)) do i
        normal_data.repeats[i] *
        logpdf(Normal(mu[i], normal_params.sigma), normal_data.y[i])
    end
    normal_prior = sum(logpdf.(Normal(), normal_params.beta_pop)) +
                   logpdf(Exponential(2), normal_params.sigma)
    @test normal_weight.kind === :frequency
    @test Turing.logjoint(normal_backend.model, normal_params) ≈
          normal_prior + normal_likelihood atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(normal_backend.model, normal_params) ≈
          normal_likelihood atol=1e-12 rtol=1e-12

    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    objective_normal = ext._brm_normal_observation(
        nothing, normal_weight, mu[2], normal_params.sigma, 2)
    @test rand(Xoshiro(41), objective_normal) ==
          rand(Xoshiro(41), Normal(mu[2], normal_params.sigma))

    poisson_data = (;
        x=[-1.0, 0.5, 2.0],
        power=[0.5, 2.0, 1.25],
        count=[0, 2, 4],
    )
    poisson_backend = TuringBRMI((@brm begin
        log_rate ~ 1 + x
        count ~ weighted(
            censored(Poisson(exp(log_rate)); lower=0, upper=4),
            weights(power))
    end)(poisson_data))
    poisson_weight = poisson_backend.plan.observation_weight
    poisson_modifier = poisson_backend.plan.response_modifier
    poisson_params = (; beta_pop=[0.3, 0.4])
    rate = exp.(poisson_backend.plan.design.matrix * poisson_params.beta_pop)
    poisson_likelihood = sum(eachindex(poisson_data.count)) do i
        poisson_data.power[i] * logpdf(
            censored(Poisson(rate[i]); lower=0, upper=4),
            poisson_data.count[i])
    end
    poisson_prior = sum(logpdf.(Normal(), poisson_params.beta_pop))
    @test poisson_weight.kind === :power
    @test poisson_modifier.kind === :censored
    @test Turing.logjoint(poisson_backend.model, poisson_params) ≈
          poisson_prior + poisson_likelihood atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(poisson_backend.model, poisson_params) ≈
          poisson_likelihood atol=1e-12 rtol=1e-12

    objective_poisson = ext._brm_poisson_observation(
        poisson_modifier, poisson_weight, rate[2], 2)
    base_poisson = censored(Poisson(rate[2]); lower=0, upper=4)
    @test rand(Xoshiro(52), objective_poisson) ==
          rand(Xoshiro(52), base_poisson)

    @test_throws "analytic/precision weights cannot yet be composed" begin
        TuringBRMI((@brm begin
            sigma ~ Exponential(2)
            mu ~ 1 + x
            y ~ weighted(
                censored(Normal(mu, sigma); upper=1.0),
                aweights(repeats))
        end)(normal_data))
    end
end


@testset "Turing extension — Gaussian plain random intercept" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 | subject)
        y ~ Normal(mu, sigma)
    end)(df)
    backend = TuringBRMI(brmi)
    block = only(backend.plan.random_effects)
    params = (;
        beta_pop=[0.25, -0.5], sigma=0.8,
        log_group_scale=log(0.6), z_group=[-0.2, 0.4, 1.1])
    group_effect = exp(params.log_group_scale) .* params.z_group
    mu = backend.plan.design.matrix * params.beta_pop +
         group_effect[block.indices]
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma) +
            logpdf(Normal(), params.log_group_scale) +
            sum(logpdf.(Normal(), params.z_group))
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test block.levels == ["a", "b", "c"]
    @test block.indices == [2, 1, 2, 3]
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.mu == mu
    @test returned.group_scale == 0.6
    @test returned.group_effect == group_effect
end


@testset "Turing extension — Gaussian correlated random slope" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x | subject)
        y ~ Normal(mu, sigma)
    end)(df))
    block = only(backend.plan.random_effects)
    L_matrix = [1.0 0.0; 0.3 sqrt(1 - 0.3^2)]
    L_group = cholesky(Symmetric(L_matrix * transpose(L_matrix)))
    params = (;
        beta_pop=[0.25, -0.5], sigma=0.8,
        L_group,
        tau_group=[0.4, 0.7],
        z_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8],
    )
    z_group = reshape(params.z_group_flat, 2, 3)
    b_group = transpose(
        Diagonal(params.tau_group) * Matrix(params.L_group.L) * z_group)
    group_effect = vec(sum(
        block.matrix .* b_group[block.indices, :]; dims=2))
    mu = backend.plan.design.matrix * params.beta_pop + group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma) +
            logpdf(LKJCholesky(2, 1.0), params.L_group) +
            sum(logpdf.(half_normal, params.tau_group)) +
            sum(logpdf.(Normal(), params.z_group_flat))
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test block.matrix == hcat(ones(4), df.x)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.mu == mu
    @test returned.tau_group == params.tau_group
    @test returned.b_group == b_group
    @test returned.group_effect == group_effect
end


@testset "Turing extension — Gaussian zero-correlation random slope" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x || subject)
        y ~ Normal(mu, sigma)
    end)(df))
    block = only(backend.plan.random_effects)
    params = (;
        beta_pop=[0.25, -0.5], sigma=0.8,
        log_group_intercept_scale=log(0.6),
        tau_group_slopes=[0.7],
        z_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8],
    )
    scales = [0.6, 0.7]
    b_group = transpose(reshape(scales, :, 1) .*
        reshape(params.z_group_flat, 2, 3))
    group_effect = vec(sum(
        block.matrix .* b_group[block.indices, :]; dims=2))
    mu = backend.plan.design.matrix * params.beta_pop + group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma) +
            logpdf(Normal(), params.log_group_intercept_scale) +
            sum(logpdf.(half_normal, params.tau_group_slopes)) +
            sum(logpdf.(Normal(), params.z_group_flat))
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test block.zero_correlation
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.scales == scales
    @test returned.b_group == b_group
    @test returned.group_effect == group_effect
end


@testset "Turing extension — GLM plain random intercepts" begin
    base = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        trials=[2, 4, 6, 3])
    bernoulli = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 | subject)
        y ~ Bernoulli(p)
    end)((; base..., y=[0, 1, 1, 0])))
    binomial = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 | subject)
        y ~ Binomial(trials, p)
    end)((; base..., y=[0, 2, 5, 1])))
    poisson = TuringBRMI((@brm begin
        log(lambda) ~ 1 + x + (1 | subject)
        y ~ Poisson(lambda)
    end)((; base..., y=[0, 2, 5, 1])))

    params = (;
        beta_pop=[0.25, -0.5],
        log_group_scale=log(0.6), z_group=[-0.2, 0.4, 1.1])
    group_effect = exp(params.log_group_scale) .* params.z_group
    indices = only(bernoulli.plan.random_effects).indices
    eta = bernoulli.plan.design.matrix * params.beta_pop + group_effect[indices]
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Normal(), params.log_group_scale) +
            sum(logpdf.(Normal(), params.z_group))

    bern_lik = sum(logpdf.(BRM.BernoulliLogit.(eta), bernoulli.plan.response))
    @test Turing.logjoint(bernoulli.model, params) ≈
          prior + bern_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(bernoulli.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(bernoulli.model, params).eta == eta

    bin_lik = sum(logpdf.(
        BRM.BinomialLogit.(base.trials, eta), binomial.plan.response))
    @test Turing.logjoint(binomial.model, params) ≈
          prior + bin_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(binomial.model, params) ≈
          bin_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(binomial.model, params).eta == eta

    rate = exp.(eta)
    poisson_lik = sum(logpdf.(Poisson.(rate), poisson.plan.response))
    @test Turing.logjoint(poisson.model, params) ≈
          prior + poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(poisson.model, params) ≈
          poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(poisson.model, params).rate == rate
end


@testset "Turing extension — GLM correlated random slopes" begin
    base = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        trials=[2, 4, 6, 3])
    bernoulli = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 + x | subject)
        y ~ Bernoulli(p)
    end)((; base..., y=[0, 1, 1, 0])))
    binomial = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 + x | subject)
        y ~ Binomial(trials, p)
    end)((; base..., y=[0, 2, 5, 1])))
    poisson = TuringBRMI((@brm begin
        log(lambda) ~ 1 + x + (1 + x | subject)
        y ~ Poisson(lambda)
    end)((; base..., y=[0, 2, 5, 1])))

    L_matrix = [1.0 0.0; -0.25 sqrt(1 - 0.25^2)]
    L_group = cholesky(Symmetric(L_matrix * transpose(L_matrix)))
    params = (;
        beta_pop=[0.25, -0.5], L_group,
        tau_group=[0.4, 0.7],
        z_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8])
    block = only(bernoulli.plan.random_effects)
    z_group = reshape(params.z_group_flat, 2, 3)
    b_group = transpose(
        Diagonal(params.tau_group) * Matrix(params.L_group.L) * z_group)
    group_effect = vec(sum(
        block.matrix .* b_group[block.indices, :]; dims=2))
    eta = bernoulli.plan.design.matrix * params.beta_pop + group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(LKJCholesky(2, 1.0), params.L_group) +
            sum(logpdf.(half_normal, params.tau_group)) +
            sum(logpdf.(Normal(), params.z_group_flat))

    bern_lik = sum(logpdf.(BRM.BernoulliLogit.(eta), bernoulli.plan.response))
    @test Turing.logjoint(bernoulli.model, params) ≈
          prior + bern_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(bernoulli.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(bernoulli.model, params).eta == eta

    bin_lik = sum(logpdf.(BRM.BinomialLogit.(
        base.trials, eta), binomial.plan.response))
    @test Turing.logjoint(binomial.model, params) ≈
          prior + bin_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(binomial.model, params) ≈
          bin_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(binomial.model, params).group_effect ==
          group_effect

    rate = exp.(eta)
    poisson_lik = sum(logpdf.(Poisson.(rate), poisson.plan.response))
    @test Turing.logjoint(poisson.model, params) ≈
          prior + poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(poisson.model, params) ≈
          poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(poisson.model, params).rate == rate
end


@testset "Turing extension — GLM zero-correlation random slopes" begin
    base = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        trials=[2, 4, 6, 3])
    bernoulli = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 + x || subject)
        y ~ Bernoulli(p)
    end)((; base..., y=[0, 1, 1, 0])))
    binomial = TuringBRMI((@brm begin
        logit(p) ~ 1 + x + (1 + x || subject)
        y ~ Binomial(trials, p)
    end)((; base..., y=[0, 2, 5, 1])))
    poisson = TuringBRMI((@brm begin
        log(lambda) ~ 1 + x + (1 + x || subject)
        y ~ Poisson(lambda)
    end)((; base..., y=[0, 2, 5, 1])))

    params = (;
        beta_pop=[0.25, -0.5],
        log_group_intercept_scale=log(0.6),
        tau_group_slopes=[0.7],
        z_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8])
    block = only(bernoulli.plan.random_effects)
    scales = [0.6, 0.7]
    b_group = transpose(reshape(scales, :, 1) .*
        reshape(params.z_group_flat, 2, 3))
    group_effect = vec(sum(
        block.matrix .* b_group[block.indices, :]; dims=2))
    eta = bernoulli.plan.design.matrix * params.beta_pop + group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Normal(), params.log_group_intercept_scale) +
            sum(logpdf.(half_normal, params.tau_group_slopes)) +
            sum(logpdf.(Normal(), params.z_group_flat))

    bern_lik = sum(logpdf.(BRM.BernoulliLogit.(eta), bernoulli.plan.response))
    @test Turing.logjoint(bernoulli.model, params) ≈
          prior + bern_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(bernoulli.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(bernoulli.model, params).scales == scales

    bin_lik = sum(logpdf.(BRM.BinomialLogit.(
        base.trials, eta), binomial.plan.response))
    @test Turing.logjoint(binomial.model, params) ≈
          prior + bin_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(binomial.model, params) ≈
          bin_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(binomial.model, params).b_group == b_group

    rate = exp.(eta)
    poisson_lik = sum(logpdf.(Poisson.(rate), poisson.plan.response))
    @test Turing.logjoint(poisson.model, params) ≈
          prior + poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(poisson.model, params) ≈
          poisson_lik atol=1e-12 rtol=1e-12
    @test Turing.DynamicPPL.returned(poisson.model, params).rate == rate
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

@testset "Turing extension — bounded Poisson responses" begin
    base = (;
        x=[-1.0, 0.5, 2.0],
        lower=[1, 1, 2],
        upper=[3, 4, 5],
    )
    truncated_backend = TuringBRMI((@brm begin
        log_rate ~ 1 + x
        count ~ truncated(
            Poisson(exp(log_rate)); lower=lower, upper=upper)
    end)((; base..., count=[1, 2, 4])))
    censored_data = (;
        x=base.x, lower=[0, 1, 2], upper=[3, 4, 4], count=[0, 2, 4])
    censored_backend = TuringBRMI((@brm begin
        log_rate ~ 1 + x
        count ~ censored(
            Poisson(exp(log_rate)); lower=lower, upper=upper)
    end)(censored_data))

    params = (; beta_pop=[0.3, 0.4])
    prior = sum(logpdf.(Normal(), params.beta_pop))
    for (backend, data, wrapper) in (
            (truncated_backend, merge(base, (; count=[1, 2, 4])), truncated),
            (censored_backend, censored_data, censored))
        rate = exp.(backend.plan.design.matrix * params.beta_pop)
        likelihood = sum(eachindex(data.count)) do i
            logpdf(wrapper(
                Poisson(rate[i]); lower=data.lower[i], upper=data.upper[i]),
                data.count[i])
        end
        @test backend.plan.response_modifier.lower == data.lower
        @test backend.plan.response_modifier.upper == data.upper
        @test Turing.logjoint(backend.model, params) ≈
              prior + likelihood atol=1e-12 rtol=1e-12
        @test Turing.logprior(backend.model, params) ≈
              prior atol=1e-12 rtol=1e-12
        @test Turing.loglikelihood(backend.model, params) ≈
              likelihood atol=1e-12 rtol=1e-12
    end

    fractional_bound = (;
        x=base.x, lower=[0.5, 1.0, 2.0], count=[1, 2, 4])
    @test_throws "discrete lower bounds must be integers" begin
        TuringBRMI((@brm begin
            log_rate ~ 1 + x
            count ~ truncated(Poisson(exp(log_rate)); lower=lower)
        end)(fractional_bound))
    end
end

@testset "Turing extension — interval-censored Normal and Poisson" begin
    normal_data = (;
        x=[-1.0, 0.5, 2.0],
        outcome=[-0.5, 0.2, 0.8],
        upper=[0.0, 0.7, 1.2],
    )
    normal_backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        outcome ~ interval_censored(Normal(mu, sigma); upper=upper)
    end)(normal_data))
    normal_params = (; beta_pop=[0.25, -0.5], sigma=0.8)
    mu = normal_backend.plan.design.matrix * normal_params.beta_pop
    normal_likelihood = sum(eachindex(normal_data.outcome)) do i
        log(cdf(Normal(mu[i], normal_params.sigma), normal_data.upper[i]) -
            cdf(Normal(mu[i], normal_params.sigma), normal_data.outcome[i]))
    end
    normal_prior = sum(logpdf.(Normal(), normal_params.beta_pop)) +
                   logpdf(Exponential(2), normal_params.sigma)

    @test normal_backend.plan.response_modifier.kind === :interval_censored
    @test normal_backend.plan.response_modifier.upper == normal_data.upper
    @test Turing.logjoint(normal_backend.model, normal_params) ≈
          normal_prior + normal_likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(normal_backend.model, normal_params) ≈
          normal_prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(normal_backend.model, normal_params) ≈
          normal_likelihood atol=1e-12 rtol=1e-12

    poisson_data = (;
        x=[-1.0, 0.5, 2.0], count=[0, 1, 2], upper=[1, 3, 5])
    poisson_backend = TuringBRMI((@brm begin
        log_rate ~ 1 + x
        count ~ interval_censored(Poisson(exp(log_rate)); upper=upper)
    end)(poisson_data))
    poisson_params = (; beta_pop=[0.3, 0.4])
    rate = exp.(poisson_backend.plan.design.matrix * poisson_params.beta_pop)
    poisson_likelihood = sum(eachindex(poisson_data.count)) do i
        log(cdf(Poisson(rate[i]), poisson_data.upper[i]) -
            cdf(Poisson(rate[i]), poisson_data.count[i]))
    end
    poisson_prior = sum(logpdf.(Normal(), poisson_params.beta_pop))
    @test poisson_backend.plan.response_modifier.kind === :interval_censored
    @test Turing.logjoint(poisson_backend.model, poisson_params) ≈
          poisson_prior + poisson_likelihood atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(poisson_backend.model, poisson_params) ≈
          poisson_likelihood atol=1e-12 rtol=1e-12

    invalid = merge(normal_data, (; upper=[0.0, 0.2, 1.2]))
    @test_throws "strictly below upper endpoints" begin
        TuringBRMI((@brm begin
            sigma ~ Exponential(2)
            mu ~ 1 + x
            outcome ~ interval_censored(Normal(mu, sigma); upper=upper)
        end)(invalid))
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


@testset "Turing extension — distributional BetaBinomial2" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        trials=[4, 6, 8, 5],
        y=[1, 4, 5, 0],
    )
    brmi = (@brm begin
        logit(mean) ~ 1 + x
        log(precision) ~ 1 + z
        effect(mean, x) ~ Normal(0.25, 0.5)
        effect(precision, Intercept) ~ Normal(-0.2, 0.3)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)(df)
    backend = TuringBRMI(brmi)
    params = (; beta_mean=[0.1, 0.2], beta_precision=[-0.4, 0.15])
    logit_mean = backend.plan.mean.design.matrix * params.beta_mean
    log_precision = backend.plan.precision.design.matrix * params.beta_precision
    mean = logistic.(logit_mean)
    precision = exp.(log_precision)
    prior = sum(logpdf.(
        Normal.(backend.plan.mean.beta_location, backend.plan.mean.beta_scale),
        params.beta_mean)) + sum(logpdf.(
        Normal.(backend.plan.precision.beta_location,
                backend.plan.precision.beta_scale), params.beta_precision))
    likelihood = sum(logpdf.(
        BRM.BetaBinomial2.(df.trials, mean, precision), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)

    @test backend.plan.family isa Val{:beta_binomial2}
    @test backend.plan.family_args.trials == df.trials
    @test backend.plan.mean.beta_location == [0.0, 0.25]
    @test backend.plan.precision.beta_location == [-0.2, 0.0]
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, params) ≈
          likelihood atol=1e-12 rtol=1e-12
    @test returned.logit_mean == logit_mean
    @test returned.log_precision == log_precision
    @test returned.mean == mean
    @test returned.precision == precision
    @test returned.trials == df.trials
end


@testset "Turing extension — grouped mean/precision families" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
        batch=[2, 1, 1, 2],
        trials=[4, 6, 8, 5],
    )
    negative_binomial = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 | subject)
        log(phi) ~ 1 + z + (1 | batch)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    beta_binomial = TuringBRMI((@brm begin
        logit(mean) ~ 1 + x + (1 | subject)
        log(precision) ~ 1 + z + (1 | batch)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)((; dist_data..., y=[1, 4, 5, 0])))

    params = (;
        beta_mean=[0.1, 0.2], beta_precision=[-0.4, 0.15],
        log_mean_group_scale=log(0.6),
        z_mean_group=[-0.2, 0.4, 1.1],
        log_precision_group_scale=log(0.3),
        z_precision_group=[0.5, -0.7],
    )
    mean_block = only(negative_binomial.plan.mean.random_effects)
    precision_block = only(
        negative_binomial.plan.precision.random_effects)
    mean_group_values = exp(params.log_mean_group_scale) .* params.z_mean_group
    precision_group_values = exp(params.log_precision_group_scale) .*
                             params.z_precision_group
    linear_mean = negative_binomial.plan.mean.design.matrix * params.beta_mean +
                  mean_group_values[mean_block.indices]
    linear_precision = negative_binomial.plan.precision.design.matrix *
                       params.beta_precision +
                       precision_group_values[precision_block.indices]
    prior = sum(logpdf.(Normal(), params.beta_mean)) +
            sum(logpdf.(Normal(), params.beta_precision)) +
            logpdf(Normal(), params.log_mean_group_scale) +
            sum(logpdf.(Normal(), params.z_mean_group)) +
            logpdf(Normal(), params.log_precision_group_scale) +
            sum(logpdf.(Normal(), params.z_precision_group))

    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    nb_lik = sum(logpdf.(
        BRM.NegativeBinomial2.(mu, phi), negative_binomial.plan.response))
    nb_returned = Turing.DynamicPPL.returned(
        negative_binomial.model, params)
    @test Turing.logjoint(negative_binomial.model, params) ≈
          prior + nb_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(negative_binomial.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test nb_returned.mu == mu
    @test nb_returned.phi == phi

    mean_prob = logistic.(linear_mean)
    precision = exp.(linear_precision)
    bb_lik = sum(logpdf.(BRM.BetaBinomial2.(
        dist_data.trials, mean_prob, precision), beta_binomial.plan.response))
    bb_returned = Turing.DynamicPPL.returned(beta_binomial.model, params)
    @test Turing.logjoint(beta_binomial.model, params) ≈
          prior + bb_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(beta_binomial.model, params) ≈
          bb_lik atol=1e-12 rtol=1e-12
    @test bb_returned.mean == mean_prob
    @test bb_returned.precision == precision
    @test bb_returned.mean_group_effect ==
          mean_group_values[mean_block.indices]
    @test bb_returned.precision_group_effect ==
          precision_group_values[precision_block.indices]
end


@testset "Turing extension — mean/precision correlated random slopes" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
        batch=[2, 1, 1, 2],
        trials=[4, 6, 8, 5],
    )
    negative_binomial = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 + x | subject)
        log(phi) ~ 1 + z + (1 + z | batch)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    beta_binomial = TuringBRMI((@brm begin
        logit(mean) ~ 1 + x + (1 + x | subject)
        log(precision) ~ 1 + z + (1 + z | batch)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)((; dist_data..., y=[1, 4, 5, 0])))

    mean_L_matrix = [1.0 0.0; 0.3 sqrt(1 - 0.3^2)]
    precision_L_matrix = [1.0 0.0; -0.2 sqrt(1 - 0.2^2)]
    L_mean_group = cholesky(Symmetric(
        mean_L_matrix * transpose(mean_L_matrix)))
    L_precision_group = cholesky(Symmetric(
        precision_L_matrix * transpose(precision_L_matrix)))
    params = (;
        beta_mean=[0.1, 0.2], beta_precision=[-0.4, 0.15],
        L_mean_group, tau_mean_group=[0.4, 0.7],
        z_mean_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8],
        L_precision_group, tau_precision_group=[0.25, 0.45],
        z_precision_group_flat=[0.5, -0.7, 0.2, 0.6],
    )
    mean_block = only(negative_binomial.plan.mean.random_effects)
    precision_block = only(negative_binomial.plan.precision.random_effects)
    mean_b = transpose(Diagonal(params.tau_mean_group) *
        Matrix(params.L_mean_group.L) * reshape(params.z_mean_group_flat, 2, 3))
    precision_b = transpose(Diagonal(params.tau_precision_group) *
        Matrix(params.L_precision_group.L) *
        reshape(params.z_precision_group_flat, 2, 2))
    mean_group_effect = vec(sum(
        mean_block.matrix .* mean_b[mean_block.indices, :]; dims=2))
    precision_group_effect = vec(sum(
        precision_block.matrix .* precision_b[precision_block.indices, :];
        dims=2))
    linear_mean = negative_binomial.plan.mean.design.matrix * params.beta_mean +
                  mean_group_effect
    linear_precision = negative_binomial.plan.precision.design.matrix *
                       params.beta_precision + precision_group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_mean)) +
            sum(logpdf.(Normal(), params.beta_precision)) +
            logpdf(LKJCholesky(2, 1.0), params.L_mean_group) +
            sum(logpdf.(half_normal, params.tau_mean_group)) +
            sum(logpdf.(Normal(), params.z_mean_group_flat)) +
            logpdf(LKJCholesky(2, 1.0), params.L_precision_group) +
            sum(logpdf.(half_normal, params.tau_precision_group)) +
            sum(logpdf.(Normal(), params.z_precision_group_flat))

    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    nb_lik = sum(logpdf.(BRM.NegativeBinomial2.(
        mu, phi), negative_binomial.plan.response))
    nb_returned = Turing.DynamicPPL.returned(negative_binomial.model, params)
    @test Turing.logjoint(negative_binomial.model, params) ≈
          prior + nb_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(negative_binomial.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test nb_returned.b_mean_group == mean_b
    @test nb_returned.b_precision_group == precision_b

    mean_prob = logistic.(linear_mean)
    precision = exp.(linear_precision)
    bb_lik = sum(logpdf.(BRM.BetaBinomial2.(
        dist_data.trials, mean_prob, precision), beta_binomial.plan.response))
    bb_returned = Turing.DynamicPPL.returned(beta_binomial.model, params)
    @test Turing.logjoint(beta_binomial.model, params) ≈
          prior + bb_lik atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(beta_binomial.model, params) ≈
          bb_lik atol=1e-12 rtol=1e-12
    @test bb_returned.mean_group_effect == mean_group_effect
    @test bb_returned.precision_group_effect == precision_group_effect
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

    random_slope = (@brm begin
        log(mu) ~ 1 + x + (1 + x || subject)
        log(phi) ~ 1
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((;
        x=[0.0, 1.0, 2.0], subject=["a", "a", "b"], y=[0, 1, 2]))
    @test_throws "zero-correlation `||` random effects for predictor `mu`" begin
        TuringBRMI(random_slope)
    end
end
