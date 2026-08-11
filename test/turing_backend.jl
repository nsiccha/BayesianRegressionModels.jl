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
    generated = turing_generated_quantities(backend, params)
    @test generated.mu == mu
    @test generated.response == df.y
    pointwise = turing_pointwise_loglikelihoods(backend, params)
    @test propertynames(pointwise) == (:y,)
    @test pointwise.y ≈ logpdf.(Normal.(mu, params.sigma), df.y)
    @test sum(pointwise.y) ≈ Turing.loglikelihood(backend.model, params)

    predictive_model = turing_predictive_model(backend)
    @test predictive_model isa Turing.DynamicPPL.Model
    predictive = turing_posterior_predictive(
        Xoshiro(61), backend, params)
    @test propertynames(predictive) == (:y,)
    @test predictive == turing_posterior_predictive(
        Xoshiro(61), backend, params)
    @test length(predictive.y) == length(df.y)
    @test all(isfinite, predictive.y)

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


@testset "Turing extension — frozen same-group replay" begin
    training = (;
        x=[1.0, 2.0, 4.0, 5.0, 7.0, 8.0],
        w=[-2.0, 1.0, 5.0, 0.0, 3.0, 6.0],
        g=[1, 2, 3, 1, 2, 3],
        subject=["b", "a", "b", "c", "a", "c"],
        exposure=[2.0, 4.0, 8.0, 3.0, 6.0, 9.0],
        y=[0.2, 1.1, -0.4, 0.7, -0.2, 0.5],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) + center(w) + g +
             zscale(x) & w + offset(log(exposure)) +
             (1 + x | subject)
        y ~ Normal(mu, sigma)
    end)(training))
    future = (;
        x=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        w=[4.0, 2.0, 8.0, 6.0, 10.0, 12.0],
        g=[3, 2, 1, 3, 1, 2],
        subject=["c", "a", "b", "c", "b", "a"],
        exposure=[5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
        y=[-0.1, 0.3, 0.9, -0.4, 0.6, 0.2],
    )
    replayed = reprocess(backend, future)
    fresh = reprocess(backend, future; freeze_constants=false)
    fitted_x = BRM._brm_fit_zscale(training.x)
    fitted_w = BRM._brm_fit_center(training.w)
    scaled_x = BRM._brm_apply_zscale(fitted_x, future.x)
    centered_w = BRM._brm_apply_center(fitted_w, future.w)
    labels = Tuple(column.label for column in replayed.plan.design.columns)

    @test labels == Tuple(column.label for column in backend.plan.design.columns)
    @test replayed.plan.design.matrix[:, 2] ≈ scaled_x
    @test replayed.plan.design.matrix[:, 3] ≈ centered_w
    @test replayed.plan.design.matrix[:, 4] == Float64.(future.g .== 2)
    @test replayed.plan.design.matrix[:, 5] == Float64.(future.g .== 3)
    interaction_index = findfirst(==(Symbol(:int_zscale_x_x_w)), labels)
    @test replayed.plan.design.matrix[:, interaction_index] ≈
          scaled_x .* future.w
    @test replayed.plan.design.fixed ≈ log.(future.exposure)
    @test replayed.plan.response == future.y
    @test fresh.plan.design.matrix[:, 2] != replayed.plan.design.matrix[:, 2]

    block = only(replayed.plan.random_effects)
    @test block.levels == only(backend.plan.random_effects).levels
    @test block.indices == [3, 1, 2, 3, 2, 1]
    @test block.matrix == hcat(ones(6), future.x)

    draw = rand(Xoshiro(62), backend.model)
    @test isfinite(Turing.logjoint(replayed.model, draw.data))
    predictive = turing_posterior_predictive(
        Xoshiro(63), replayed, draw.data)
    @test length(predictive.y) == length(future.y)

    subset = (;
        x=future.x[1:3], w=future.w[1:3], g=fill(2, 3),
        subject=fill("a", 3), exposure=future.exposure[1:3],
        y=future.y[1:3])
    subset_replay = reprocess(backend, subset)
    @test Tuple(column.label for column in subset_replay.plan.design.columns) ==
          labels
    @test subset_replay.plan.design.matrix[:, 4] == ones(3)
    @test subset_replay.plan.design.matrix[:, 5] == zeros(3)
    @test only(subset_replay.plan.random_effects).levels == block.levels
    @test only(subset_replay.plan.random_effects).indices == ones(Int, 3)

    unseen_group = merge(future, (; subject=["new", future.subject[2:end]...]))
    @test_throws "contains unseen level" reprocess(backend, unseen_group)
    new_population = merge(future, (;
        subject=["new_b", "new_a", "new_b", "new_c", "new_a", "new_c"]))
    resampled = reprocess(
        backend, new_population; resample_groups=:subject)
    resampled_block = only(resampled.plan.random_effects)
    @test resampled.replay.resample_groups == (:subject,)
    @test resampled_block.levels == ["new_a", "new_b", "new_c"]
    @test resampled_block.indices == [2, 1, 2, 3, 1, 3]
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    fixed_parameters = ext._brm_resampled_parameters(resampled, draw.data)
    @test !hasproperty(fixed_parameters, :z_group_flat)
    @test fixed_parameters.L_group == draw.data.L_group
    @test fixed_parameters.tau_group == draw.data.tau_group
    resampled_predictive = turing_posterior_predictive(
        Xoshiro(64), resampled, draw.data)
    @test resampled_predictive == turing_posterior_predictive(
        Xoshiro(64), resampled, draw.data)
    @test length(resampled_predictive.y) == length(new_population.y)
    @test_throws "names no fitted random-effect block" reprocess(
        backend, new_population; resample_groups=:unknown)

    unseen_category = merge(future, (; g=[4, future.g[2:end]...]))
    @test_throws "contains unseen level" reprocess(backend, unseen_category)
end


@testset "Turing extension — replay composes responses and evidence" begin
    training = (;
        x=[-2.0, -0.5, 1.0, 3.0],
        upper=[0.5, 0.8, 1.1, 1.4],
        repeats=[1, 2, 3, 4],
        outcome=[0.2, 0.8, -0.1, 1.4],
    )
    evidence = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + center(x)
        outcome ~ weighted(
            censored(Normal(mu, sigma); upper=upper),
            fweights(repeats))
    end)(training))
    future = (;
        x=[10.0, 12.0, 14.0],
        upper=[0.4, 0.9, 1.3],
        repeats=[0, 5, 2],
        outcome=[0.1, 0.9, 0.7],
    )
    replayed = reprocess(evidence, future)
    params = (; beta_pop=[0.3, -0.2], sigma=0.7)
    fitted_center = BRM._brm_fit_center(training.x)
    centered_x = BRM._brm_apply_center(fitted_center, future.x)
    mu = params.beta_pop[1] .+ params.beta_pop[2] .* centered_x
    expected = sum(eachindex(future.outcome)) do i
        future.repeats[i] * logpdf(censored(
            Normal(mu[i], params.sigma); upper=future.upper[i]),
            future.outcome[i])
    end

    @test replayed.plan.design.matrix[:, 2] ≈ centered_x
    @test replayed.plan.response_modifier.upper == future.upper
    @test replayed.plan.observation_weight.values == future.repeats
    @test Turing.loglikelihood(replayed.model, params) ≈
          expected atol=1e-12 rtol=1e-12
    @test turing_pointwise_loglikelihoods(replayed, params).outcome ≈ [
        future.repeats[i] * logpdf(censored(
            Normal(mu[i], params.sigma); upper=future.upper[i]),
            future.outcome[i]) for i in eachindex(future.outcome)
    ]

    shared_training = (;
        x=training.x,
        y=training.outcome,
        y2=reverse(training.outcome),
    )
    shared = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(mu, sigma)
    end)(shared_training))
    shared_future = (;
        x=[4.0, 7.0], y=[0.3, -0.2], y2=[-0.1, 0.6])
    shared_replay = reprocess(shared, shared_future)
    shared_draw = rand(Xoshiro(64), shared.model)
    shared_predictive = turing_posterior_predictive(
        Xoshiro(65), shared_replay, shared_draw.data)

    @test shared_replay.plan.owners == (1, 1)
    @test shared_replay.plan.responses == (:y, :y2)
    @test isfinite(Turing.logjoint(shared_replay.model, shared_draw.data))
    @test propertynames(shared_predictive) == (:y, :y2)
    @test length(shared_predictive.y) == length(shared_future.y)
    @test length(shared_predictive.y2) == length(shared_future.y2)
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
    pointwise = turing_pointwise_loglikelihoods(backend, params)
    @test propertynames(pointwise) == (:y,)
    @test ismissing(pointwise.y[2])
    @test ismissing(pointwise.y[4])
    @test pointwise.y[missing_plan.observed_indices] ≈ [
        logpdf(Normal(mu[i], params.sigma), complete_y[i])
        for i in missing_plan.observed_indices
    ]
    @test sum(skipmissing(pointwise.y)) ≈ observed_likelihood
    predictive = turing_posterior_predictive(Xoshiro(74), backend, params)
    @test propertynames(predictive) == (:y,)
    @test !any(ismissing, predictive.y)
    @test length(predictive.y) == length(df.y)
    @test predictive.y[missing_plan.missing_indices] !=
          complete_y[missing_plan.missing_indices]

    # A fitted chain contains latent `y[i]` parameters for the missing rows.
    # BRM's chain-level predict entry point must remove those before delegating
    # to DynamicPPL, otherwise the old imputed values are silently reused.
    fitted = sample(Xoshiro(75), backend.model, Prior(), 2; progress=false)
    response_root = Turing.DynamicPPL.@varname(y)
    response_keys = filter(collect(keys(fitted))) do key
        hasfield(typeof(key), :name) || return false
        name = getfield(key, :name)
        name isa Turing.DynamicPPL.VarName &&
            Turing.DynamicPPL.subsumes(response_root, name)
    end
    @test length(response_keys) == length(missing_plan.missing_indices)
    fitted_latents = Dict(
        string(getfield(key, :name)) => fitted[key] for key in response_keys)
    chain_predictive = Turing.predict(
        Xoshiro(76), backend, fitted; include_all=false)
    predicted_key = only(filter(collect(keys(chain_predictive))) do key
        hasfield(typeof(key), :name) || return false
        getfield(key, :name) == response_root
    end)
    predicted_rows = chain_predictive[predicted_key]
    for missing_index in missing_plan.missing_indices
        latent = fitted_latents["y[$missing_index]"]
        @test all(CartesianIndices(predicted_rows)) do sample
            predicted_rows[sample][missing_index] != latent[sample]
        end
    end

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


@testset "Turing extension — independent mixed-family responses" begin
    df = (;
        x=[-1.0, 0.5, 2.0],
        y=[0.2, 1.1, -0.4],
        count=[0, 2, 4],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        log_rate ~ 1 + x
        count ~ Poisson(exp(log_rate))
    end)(df))

    @test backend.plan.responses == (:y, :count)
    @test length(backend.plan.plans) == 2
    @test sprint(show, backend) ==
          "TuringBRMI with 4 population coefficients across 2 responses and 6 observations"

    draw = rand(Xoshiro(91), backend.model)
    returned = Turing.DynamicPPL.returned(backend.model, draw.data)
    normal_params = draw.data.responses[1].data
    poisson_params = draw.data.responses[2].data
    normal_plan, poisson_plan = backend.plan.plans
    mu = normal_plan.design.matrix * normal_params.beta_pop
    rate = exp.(poisson_plan.design.matrix * poisson_params.beta_pop)
    prior = sum(logpdf.(Normal(), normal_params.beta_pop)) +
            logpdf(Exponential(2), normal_params.sigma) +
            sum(logpdf.(Normal(), poisson_params.beta_pop))
    likelihood = sum(logpdf.(Normal.(mu, normal_params.sigma), df.y)) +
                 sum(logpdf.(Poisson.(rate), df.count))

    @test returned.responses[1].mu == mu
    @test returned.responses[2].rate == rate
    @test Turing.logjoint(backend.model, draw.data) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, draw.data) ≈
          prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(backend.model, draw.data) ≈
          likelihood atol=1e-12 rtol=1e-12
    pointwise = turing_pointwise_loglikelihoods(backend, draw.data)
    @test propertynames(pointwise) == (:y, :count)
    @test pointwise.y ≈ logpdf.(Normal.(mu, normal_params.sigma), df.y)
    @test pointwise.count ≈ logpdf.(Poisson.(rate), df.count)
    @test sum(pointwise.y) + sum(pointwise.count) ≈ likelihood
    generated = turing_generated_quantities(backend, draw.data)
    @test generated.responses[1].response == df.y
    @test generated.responses[2].response == df.count
    predictive = turing_posterior_predictive(
        Xoshiro(92), backend, draw.data)
    @test propertynames(predictive) == (:y, :count)
    @test length(predictive.y) == length(df.y)
    @test length(predictive.count) == length(df.count)
    @test eltype(predictive.count) <: Integer

    composed = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        mi(y_missing) ~ Normal(mu, sigma)
        log(rate_weighted) ~ 1 + x
        count_weighted ~ weighted(
            Poisson(rate_weighted), fweights(repeats))
    end)((;
        x=df.x,
        y_missing=Union{Missing,Float64}[0.2, missing, -0.4],
        count_weighted=df.count,
        repeats=[0, 3, 2],
    )))
    composed_draw = rand(Xoshiro(93), composed.model)
    composed_returned = Turing.DynamicPPL.returned(
        composed.model, composed_draw.data)
    gaussian_params = composed_draw.data.responses[1].data
    poisson_params = composed_draw.data.responses[2].data
    gaussian_plan, weighted_plan = composed.plan.plans
    completed_y = composed_returned.responses[1].response
    mu_missing = gaussian_plan.design.matrix * gaussian_params.beta_pop
    rate_weighted = exp.(
        weighted_plan.design.matrix * poisson_params.beta_pop)
    gaussian_observed = sum(gaussian_plan.missing_response.observed_indices) do i
        logpdf(Normal(mu_missing[i], gaussian_params.sigma), completed_y[i])
    end
    gaussian_imputed = sum(gaussian_plan.missing_response.missing_indices) do i
        logpdf(Normal(mu_missing[i], gaussian_params.sigma), completed_y[i])
    end
    weighted_poisson = sum(eachindex(df.count)) do i
        [0, 3, 2][i] * logpdf(Poisson(rate_weighted[i]), df.count[i])
    end
    composed_prior = sum(logpdf.(Normal(), gaussian_params.beta_pop)) +
                     logpdf(Exponential(2), gaussian_params.sigma) +
                     gaussian_imputed +
                     sum(logpdf.(Normal(), poisson_params.beta_pop))

    @test completed_y[gaussian_plan.missing_response.observed_indices] ==
          gaussian_plan.missing_response.observed_values
    @test Turing.logprior(composed.model, composed_draw.data) ≈
          composed_prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(composed.model, composed_draw.data) ≈
          gaussian_observed + weighted_poisson atol=1e-12 rtol=1e-12
    composed_pointwise = turing_pointwise_loglikelihoods(
        composed, composed_draw.data)
    @test propertynames(composed_pointwise) == (:y_missing, :count_weighted)
    @test ismissing(composed_pointwise.y_missing[2])
    @test composed_pointwise.y_missing[[1, 3]] ≈ [
        logpdf(Normal(mu_missing[i], gaussian_params.sigma), completed_y[i])
        for i in (1, 3)
    ]
    @test composed_pointwise.count_weighted ≈ [
        [0, 3, 2][i] * logpdf(Poisson(rate_weighted[i]), df.count[i])
        for i in eachindex(df.count)
    ]
    @test composed_pointwise.count_weighted[1] == 0.0
    composed_predictive = turing_posterior_predictive(
        Xoshiro(94), composed, composed_draw.data)
    @test composed_predictive.y_missing[2] != completed_y[2]
end


@testset "Turing extension — shared multi-response parameter blocks" begin
    gaussian_data = (;
        x=[-1.0, 0.5, 2.0],
        y=[0.2, 1.1, -0.4],
        y2=[-0.1, 0.3, 0.7],
    )
    gaussian = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        y2 ~ Normal(mu, sigma)
    end)(gaussian_data))
    @test gaussian.plan.owners == (1, 1)

    gaussian_draw = rand(Xoshiro(101), gaussian.model)
    gaussian_returned = Turing.DynamicPPL.returned(
        gaussian.model, gaussian_draw.data)
    gaussian_params = gaussian_draw.data.responses[1].data
    mu = gaussian.plan.plans[1].design.matrix * gaussian_params.beta_pop
    gaussian_prior = sum(logpdf.(Normal(), gaussian_params.beta_pop)) +
                     logpdf(Exponential(2), gaussian_params.sigma)
    gaussian_likelihood =
        sum(logpdf.(Normal.(mu, gaussian_params.sigma), gaussian_data.y)) +
        sum(logpdf.(Normal.(mu, gaussian_params.sigma), gaussian_data.y2))
    @test gaussian_returned.responses[1].mu == mu
    @test gaussian_returned.responses[2].mu == mu
    @test Turing.logjoint(gaussian.model, gaussian_draw.data) ≈
          gaussian_prior + gaussian_likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(gaussian.model, gaussian_draw.data) ≈
          gaussian_prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(gaussian.model, gaussian_draw.data) ≈
          gaussian_likelihood atol=1e-12 rtol=1e-12

    grouped_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        group=["b", "a", "b", "c"],
        binary=[1, 0, 1, 1],
        successes=[1, 2, 4, 1],
        trials=[3, 4, 5, 2],
    )
    grouped = TuringBRMI((@brm begin
        eta ~ 1 + x + (1 | group)
        binary ~ BernoulliLogit(eta)
        successes ~ BRM.BinomialLogit(trials, eta)
    end)(grouped_data))
    @test grouped.plan.owners == (1, 1)

    grouped_draw = rand(Xoshiro(102), grouped.model)
    grouped_returned = Turing.DynamicPPL.returned(grouped.model, grouped_draw.data)
    grouped_params = grouped_draw.data.responses[1].data
    block = only(grouped.plan.plans[1].random_effects)
    group_effect = exp(grouped_params.log_group_scale) .* grouped_params.z_group
    eta = grouped.plan.plans[1].design.matrix * grouped_params.beta_pop +
          group_effect[block.indices]
    grouped_prior = sum(logpdf.(Normal(), grouped_params.beta_pop)) +
                    logpdf(Normal(), grouped_params.log_group_scale) +
                    sum(logpdf.(Normal(), grouped_params.z_group))
    grouped_likelihood =
        sum(logpdf.(BRM.BernoulliLogit.(eta), grouped_data.binary)) +
        sum(logpdf.(BRM.BinomialLogit.(grouped_data.trials, eta),
                    grouped_data.successes))
    @test grouped_returned.responses[1].eta == eta
    @test grouped_returned.responses[2].eta == eta
    @test Turing.logjoint(grouped.model, grouped_draw.data) ≈
          grouped_prior + grouped_likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(grouped.model, grouped_draw.data) ≈
          grouped_prior atol=1e-12 rtol=1e-12
    @test Turing.loglikelihood(grouped.model, grouped_draw.data) ≈
          grouped_likelihood atol=1e-12 rtol=1e-12

    new_grouped_data = merge(grouped_data, (;
        group=["new_b", "new_a", "new_b", "new_c"]))
    grouped_replay = reprocess(
        grouped, new_grouped_data; resample_groups=:group)
    @test grouped_replay.plan.owners == (1, 1)
    @test only(grouped_replay.plan.plans[1].random_effects).levels ==
          ["new_a", "new_b", "new_c"]
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    grouped_fixed = ext._brm_resampled_parameters(
        grouped_replay, grouped_draw.data)
    @test all(!endswith(string(variable), ".z_group")
              for variable in keys(grouped_fixed))
    @test any(endswith(string(variable), ".log_group_scale")
              for variable in keys(grouped_fixed))
    grouped_predictive = turing_posterior_predictive(
        Xoshiro(103), grouped_replay, grouped_draw.data)
    @test propertynames(grouped_predictive) == (:binary, :successes)
    @test length(grouped_predictive.binary) == length(grouped_data.binary)
    @test length(grouped_predictive.successes) ==
          length(grouped_data.successes)

    fitted_chain = sample(
        Xoshiro(104), grouped.model, Prior(), 2; progress=false)
    chain_predictive = Turing.predict(
        Xoshiro(105), grouped_replay, fitted_chain; include_all=false)
    old_latent_key = only(filter(collect(keys(fitted_chain))) do key
        hasfield(typeof(key), :name) &&
            string(getfield(key, :name)) == "responses[1].z_group"
    end)
    new_latent_key = only(filter(collect(keys(chain_predictive))) do key
        hasfield(typeof(key), :name) &&
            string(getfield(key, :name)) == "responses[1].z_group"
    end)
    old_latents = fitted_chain[old_latent_key]
    new_latents = chain_predictive[new_latent_key]
    @test all(CartesianIndices(new_latents)) do sample
        new_latents[sample] != old_latents[sample]
    end

    @test_throws "duplicate group names" reprocess(
        grouped, new_grouped_data; resample_groups=(:group, :group))
    @test_throws "names no fitted random-effect block" reprocess(
        grouped, new_grouped_data; resample_groups=:unknown)
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
    @test turing_pointwise_loglikelihoods(backend, params).y ≈
          logpdf.(Normal.(mu, params.sigma ./ sqrt.(df.precision)), df.y)

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
    pointwise_poisson = turing_pointwise_loglikelihoods(
        poisson_backend, poisson_params)
    @test pointwise_poisson.count ≈ [
        poisson_data.power[i] * logpdf(
            censored(Poisson(rate[i]); lower=0, upper=4),
            poisson_data.count[i])
        for i in eachindex(poisson_data.count)
    ]

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


@testset "Turing extension — categorical and transformed random slopes" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        w=[1.0, 2.0, 4.0, 8.0],
        category=[1, 2, 3, 2],
        subject=["b", "a", "b", "c"],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x +
             (1 + zscale(x) + factor(category) + log(w) | subject)
        y ~ Normal(mu, sigma)
    end)(df))
    block = only(backend.plan.random_effects)
    labels = Tuple(column.label for column in block.columns)
    @test labels[1:4] ==
          (:Intercept, :zscale_x, :category_dummy_2, :category_dummy_3)
    @test startswith(String(labels[5]), "log_expr_")
    @test block.columns[5].preprocess.kind === :protect
    @test size(block.matrix) == (4, 5)
    @test block.matrix[:, 3] == Float64.(df.category .== 2)
    @test block.matrix[:, 4] == Float64.(df.category .== 3)
    @test block.matrix[:, 5] == log.(df.w)

    L_group = cholesky(Symmetric(Matrix{Float64}(I, 5, 5)))
    params = (;
        beta_pop=[0.25, -0.5], sigma=0.8, L_group,
        tau_group=[0.4, 0.7, 0.3, 0.5, 0.6],
        z_group_flat=collect(range(-0.7, 0.7; length=15)),
    )
    coefficients = transpose(Diagonal(params.tau_group) *
        Matrix(L_group.L) * reshape(params.z_group_flat, 5, 3))
    group_effect = vec(sum(
        block.matrix .* coefficients[block.indices, :]; dims=2))
    mu = backend.plan.design.matrix * params.beta_pop + group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params.beta_pop)) +
            logpdf(Exponential(2), params.sigma) +
            logpdf(LKJCholesky(5, 1.0), params.L_group) +
            sum(logpdf.(half_normal, params.tau_group)) +
            sum(logpdf.(Normal(), params.z_group_flat))
    likelihood = sum(logpdf.(Normal.(mu, params.sigma), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test returned.b_group == coefficients
    @test returned.group_effect == group_effect

    new_df = (;
        x=[2.5, -0.5, 1.0, 0.0],
        w=[3.0, 6.0, 12.0, 24.0],
        category=[3, 1, 2, 1],
        subject=["c", "a", "b", "a"],
        y=zeros(4),
    )
    replayed = reprocess(backend, new_df)
    replay_block = only(replayed.plan.random_effects)
    transform = block.columns[2].preprocess.const_
    @test replay_block.matrix[:, 2] ≈
          (new_df.x .- transform[1]) ./ transform[2]
    @test replay_block.matrix[:, 3] == Float64.(new_df.category .== 2)
    @test replay_block.matrix[:, 4] == Float64.(new_df.category .== 3)
    @test replay_block.matrix[:, 5] == log.(new_df.w)
    @test_throws "unseen level" reprocess(
        backend, (; new_df..., category=[1, 2, 4, 1]))
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


@testset "Turing extension — multiple crossed random-effect blocks" begin
    df = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        subject=["b", "a", "b", "c"],
        item=[2, 1, 1, 2],
        y=[0.2, 1.1, -0.4, 0.7],
    )
    backend = TuringBRMI((@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x + (1 + x | subject) + (1 | item)
        y ~ Normal(mu, sigma)
    end)(df))
    @test Tuple(block.group for block in backend.plan.random_effects) ==
          (:subject, :item)

    L_matrix = [1.0 0.0; 0.25 sqrt(1 - 0.25^2)]
    L_subject = cholesky(Symmetric(L_matrix * transpose(L_matrix)))
    params = Dict(
        Turing.@varname(beta_pop) => [0.25, -0.5],
        Turing.@varname(sigma) => 0.8,
        Turing.@varname(groups[1].L) => L_subject,
        Turing.@varname(groups[1].tau) => [0.4, 0.7],
        Turing.@varname(groups[1].z_flat) =>
            [-0.2, 0.4, 1.1, 0.3, -0.5, 0.8],
        Turing.@varname(groups[2].log_scale) => log(0.6),
        Turing.@varname(groups[2].z) => [0.2, -0.4],
    )
    subject_block, item_block = backend.plan.random_effects
    subject_coefficients = transpose(Diagonal(params[
        Turing.@varname(groups[1].tau)]) *
        Matrix(L_subject.L) * reshape(params[
            Turing.@varname(groups[1].z_flat)], 2, 3))
    subject_effect = vec(sum(subject_block.matrix .*
        subject_coefficients[subject_block.indices, :]; dims=2))
    item_values = 0.6 .* params[Turing.@varname(groups[2].z)]
    item_effect = item_values[item_block.indices]
    group_effect = subject_effect + item_effect
    mu = backend.plan.design.matrix * params[Turing.@varname(beta_pop)] +
         group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params[Turing.@varname(beta_pop)])) +
            logpdf(Exponential(2), params[Turing.@varname(sigma)]) +
            logpdf(LKJCholesky(2, 1.0), L_subject) +
            sum(logpdf.(half_normal,
                params[Turing.@varname(groups[1].tau)])) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(groups[1].z_flat)])) +
            logpdf(Normal(), params[Turing.@varname(groups[2].log_scale)]) +
            sum(logpdf.(Normal(), params[Turing.@varname(groups[2].z)]))
    likelihood = sum(logpdf.(Normal.(mu, params[Turing.@varname(sigma)]), df.y))
    returned = Turing.DynamicPPL.returned(backend.model, params)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test returned.groups[1].coefficients == subject_coefficients
    @test returned.groups[2].values == item_values
    @test returned.group_effect == group_effect
    @test returned.mu == mu

    replayed = reprocess(
        backend,
        (; df..., item=[20, 10, 10, 20], y=zeros(length(df.y)));
        resample_groups=:item)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    replay_parameters = ext._brm_resampled_parameters(replayed, params)
    replay_names = Set(string(variable) for variable in keys(replay_parameters))
    @test "groups[1].z_flat" in replay_names
    @test "groups[2].z" ∉ replay_names
    @test "groups[2].log_scale" in replay_names
    predictive = turing_posterior_predictive(Xoshiro(113), replayed, params)
    @test length(predictive.y) == length(df.y)
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

    new_dist_data = merge(dist_data, (;
        subject=["new_b", "new_a", "new_b", "new_c"],
        batch=[20, 10, 10, 20]))
    replayed = reprocess(
        beta_binomial, (; new_dist_data..., y=[1, 4, 5, 0]);
        resample_groups=[:subject, :batch])
    @test replayed.replay.resample_groups == (:subject, :batch)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    replay_parameters = ext._brm_resampled_parameters(replayed, params)
    @test !hasproperty(replay_parameters, :z_mean_group)
    @test !hasproperty(replay_parameters, :z_precision_group)
    @test replay_parameters.log_mean_group_scale == params.log_mean_group_scale
    @test replay_parameters.log_precision_group_scale ==
          params.log_precision_group_scale
    predictive = turing_posterior_predictive(
        Xoshiro(111), replayed, params)
    @test length(predictive.y) == length(dist_data.x)
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

@testset "Turing extension — mean/precision zero-correlation random slopes" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
        batch=[2, 1, 1, 2],
        trials=[4, 6, 8, 5],
    )
    negative_binomial = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 + x || subject)
        log(phi) ~ 1 + z + (0 + z || batch)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    beta_binomial = TuringBRMI((@brm begin
        logit(mean) ~ 1 + x + (1 + x || subject)
        log(precision) ~ 1 + z + (0 + z || batch)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)((; dist_data..., y=[1, 4, 5, 0])))

    params = (;
        beta_mean=[0.1, 0.2], beta_precision=[-0.4, 0.15],
        log_mean_group_intercept_scale=log(0.4),
        tau_mean_group_slopes=[0.7],
        z_mean_group_flat=[-0.2, 0.4, 1.1, 0.3, -0.5, 0.8],
        tau_precision_group_slopes=[0.45],
        z_precision_group_flat=[0.5, -0.7],
    )
    mean_block = only(negative_binomial.plan.mean.random_effects)
    precision_block = only(negative_binomial.plan.precision.random_effects)
    mean_scales = [0.4, 0.7]
    precision_scales = [0.45]
    mean_b = transpose(reshape(mean_scales, :, 1) .*
        reshape(params.z_mean_group_flat, 2, 3))
    precision_b = transpose(reshape(precision_scales, :, 1) .*
        reshape(params.z_precision_group_flat, 1, 2))
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
            logpdf(Normal(), params.log_mean_group_intercept_scale) +
            sum(logpdf.(half_normal, params.tau_mean_group_slopes)) +
            sum(logpdf.(Normal(), params.z_mean_group_flat)) +
            sum(logpdf.(half_normal, params.tau_precision_group_slopes)) +
            sum(logpdf.(Normal(), params.z_precision_group_flat))

    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    nb_lik = sum(logpdf.(BRM.NegativeBinomial2.(
        mu, phi), negative_binomial.plan.response))
    nb_returned = Turing.DynamicPPL.returned(negative_binomial.model, params)
    @test mean_block.zero_correlation
    @test precision_block.zero_correlation
    @test Turing.logjoint(negative_binomial.model, params) ≈
          prior + nb_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(negative_binomial.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test nb_returned.mean_group_scales == mean_scales
    @test nb_returned.precision_group_scales == precision_scales
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

    replayed = reprocess(
        beta_binomial,
        (; dist_data...,
         subject=["new_b", "new_a", "new_b", "new_c"],
         batch=[20, 10, 10, 20], y=[1, 4, 5, 0]);
        resample_groups=[:subject, :batch])
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    replay_parameters = ext._brm_resampled_parameters(replayed, params)
    @test !hasproperty(replay_parameters, :z_mean_group_flat)
    @test !hasproperty(replay_parameters, :z_precision_group_flat)
    @test replay_parameters.log_mean_group_intercept_scale ==
          params.log_mean_group_intercept_scale
    @test replay_parameters.tau_mean_group_slopes ==
          params.tau_mean_group_slopes
    @test replay_parameters.tau_precision_group_slopes ==
          params.tau_precision_group_slopes
    predictive = turing_posterior_predictive(Xoshiro(112), replayed, params)
    @test length(predictive.y) == length(dist_data.x)
end


@testset "Turing extension — distributional multiple group blocks" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
        item=[2, 1, 1, 2],
        batch=[20, 10, 10, 20],
        trials=[4, 6, 8, 5],
    )
    negative_binomial = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 | subject) + (1 | item)
        log(phi) ~ 1 + z + (1 | batch)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    beta_binomial = TuringBRMI((@brm begin
        logit(mean) ~ 1 + x + (1 | subject) + (1 | item)
        log(precision) ~ 1 + z + (1 | batch)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)((; dist_data..., y=[1, 4, 5, 0])))

    params = Dict(
        Turing.@varname(beta_mean) => [0.1, 0.2],
        Turing.@varname(beta_precision) => [-0.4, 0.15],
        Turing.@varname(mean_groups[1].log_scale) => log(0.4),
        Turing.@varname(mean_groups[1].z) => [-0.2, 0.4, 1.1],
        Turing.@varname(mean_groups[2].log_scale) => log(0.7),
        Turing.@varname(mean_groups[2].z) => [0.3, -0.5],
        Turing.@varname(precision_groups[1].log_scale) => log(0.45),
        Turing.@varname(precision_groups[1].z) => [0.5, -0.7],
    )
    mean_subject, mean_item = negative_binomial.plan.mean.random_effects
    precision_batch = only(negative_binomial.plan.precision.random_effects)
    subject_values = 0.4 .* params[Turing.@varname(mean_groups[1].z)]
    item_values = 0.7 .* params[Turing.@varname(mean_groups[2].z)]
    batch_values = 0.45 .* params[Turing.@varname(precision_groups[1].z)]
    mean_group_effect = subject_values[mean_subject.indices] +
                        item_values[mean_item.indices]
    precision_group_effect = batch_values[precision_batch.indices]
    linear_mean = negative_binomial.plan.mean.design.matrix *
                  params[Turing.@varname(beta_mean)] + mean_group_effect
    linear_precision = negative_binomial.plan.precision.design.matrix *
                       params[Turing.@varname(beta_precision)] +
                       precision_group_effect
    prior = sum(logpdf.(Normal(), params[Turing.@varname(beta_mean)])) +
            sum(logpdf.(Normal(), params[Turing.@varname(beta_precision)])) +
            logpdf(Normal(),
                params[Turing.@varname(mean_groups[1].log_scale)]) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(mean_groups[1].z)])) +
            logpdf(Normal(),
                params[Turing.@varname(mean_groups[2].log_scale)]) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(mean_groups[2].z)])) +
            logpdf(Normal(),
                params[Turing.@varname(precision_groups[1].log_scale)]) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(precision_groups[1].z)]))

    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    nb_lik = sum(logpdf.(BRM.NegativeBinomial2.(
        mu, phi), negative_binomial.plan.response))
    nb_returned = Turing.DynamicPPL.returned(negative_binomial.model, params)
    @test Turing.logjoint(negative_binomial.model, params) ≈
          prior + nb_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(negative_binomial.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test nb_returned.mean_group_effect == mean_group_effect
    @test nb_returned.precision_group_effect == precision_group_effect

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
end


@testset "Turing extension — shared distributional |ID| group block" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
        trials=[4, 6, 8, 5],
    )
    negative_binomial = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 + x | joint | subject)
        log(phi) ~ 1 + z + (1 + z | joint | subject)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    beta_binomial = TuringBRMI((@brm begin
        logit(mean) ~ 1 + x + (1 + x | joint | subject)
        log(precision) ~ 1 + z + (1 + z | joint | subject)
        y ~ BRM.BetaBinomial2(trials, mean, precision)
    end)((; dist_data..., y=[1, 4, 5, 0])))
    mean_block = only(negative_binomial.plan.mean.random_effects)
    precision_block = only(negative_binomial.plan.precision.random_effects)
    @test mean_block.id === :joint
    @test precision_block.id === :joint

    L_shared = cholesky(Symmetric(Matrix{Float64}(I, 4, 4)))
    params = Dict(
        Turing.@varname(beta_mean) => [0.1, 0.2],
        Turing.@varname(beta_precision) => [-0.4, 0.15],
        Turing.@varname(shared_groups[1].L) => L_shared,
        Turing.@varname(shared_groups[1].tau) => [0.4, 0.7, 0.25, 0.45],
        Turing.@varname(shared_groups[1].z_flat) =>
            [-0.2, 0.4, 0.5, -0.7,
             1.1, 0.3, 0.2, 0.6,
             -0.5, 0.8, 0.4, -0.1],
    )
    shared_coefficients = transpose(Diagonal(params[
        Turing.@varname(shared_groups[1].tau)]) * Matrix(L_shared.L) *
        reshape(params[Turing.@varname(shared_groups[1].z_flat)], 4, 3))
    mean_coefficients = @view shared_coefficients[:, 1:2]
    precision_coefficients = @view shared_coefficients[:, 3:4]
    mean_group_effect = vec(sum(mean_block.matrix .*
        mean_coefficients[mean_block.indices, :]; dims=2))
    precision_group_effect = vec(sum(precision_block.matrix .*
        precision_coefficients[precision_block.indices, :]; dims=2))
    linear_mean = negative_binomial.plan.mean.design.matrix *
                  params[Turing.@varname(beta_mean)] + mean_group_effect
    linear_precision = negative_binomial.plan.precision.design.matrix *
                       params[Turing.@varname(beta_precision)] +
                       precision_group_effect
    half_normal = truncated(Normal(), 0.0, Inf)
    prior = sum(logpdf.(Normal(), params[Turing.@varname(beta_mean)])) +
            sum(logpdf.(Normal(), params[Turing.@varname(beta_precision)])) +
            logpdf(LKJCholesky(4, 1.0), L_shared) +
            sum(logpdf.(half_normal,
                params[Turing.@varname(shared_groups[1].tau)])) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(shared_groups[1].z_flat)]))

    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    nb_lik = sum(logpdf.(BRM.NegativeBinomial2.(
        mu, phi), negative_binomial.plan.response))
    nb_returned = Turing.DynamicPPL.returned(negative_binomial.model, params)
    @test Turing.logjoint(negative_binomial.model, params) ≈
          prior + nb_lik atol=1e-12 rtol=1e-12
    @test Turing.logprior(negative_binomial.model, params) ≈
          prior atol=1e-12 rtol=1e-12
    @test nb_returned.shared_groups[1].coefficients == shared_coefficients
    @test nb_returned.mean_group_effect == mean_group_effect
    @test nb_returned.precision_group_effect == precision_group_effect

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

    replayed = reprocess(
        beta_binomial,
        (; dist_data..., subject=["new_b", "new_a", "new_b", "new_c"],
         y=[1, 4, 5, 0]); resample_groups=:subject)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    replay_parameters = ext._brm_resampled_parameters(replayed, params)
    replay_names = Set(string(variable) for variable in keys(replay_parameters))
    @test "shared_groups[1].z_flat" ∉ replay_names
    @test "shared_groups[1].L" in replay_names
    @test "shared_groups[1].tau" in replay_names
    predictive = turing_posterior_predictive(Xoshiro(114), replayed, params)
    @test length(predictive.y) == length(dist_data.x)
end


@testset "Turing extension — random-effect sd/cor prior overrides" begin
    dist_data = (;
        x=[-1.0, 0.5, 2.0, 0.25],
        z=[0.0, 1.0, -0.5, 0.75],
        subject=["b", "a", "b", "c"],
    )
    backend = TuringBRMI((@brm begin
        log(mu) ~ 1 + x + (1 + x | joint | subject)
        log(phi) ~ 1 + z + (1 + z | joint | subject)
        sd(:, joint) ~ Exponential(2)
        sd(mu, joint, x) ~ Exponential(0.25)
        sd(phi, joint, z) ~ Exponential(0.5)
        cor(:, joint) ~ LKJCholesky(4, 2.5)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1])))
    mean_block = only(backend.plan.mean.random_effects)
    precision_block = only(backend.plan.precision.random_effects)
    @test mean_block.sd_family == [1, 1]
    @test mean_block.sd_rate == [0.5, 4.0]
    @test precision_block.sd_family == [1, 1]
    @test precision_block.sd_rate == [0.5, 2.0]
    @test mean_block.lkj_eta == precision_block.lkj_eta == 2.5

    L_shared = cholesky(Symmetric(Matrix{Float64}(I, 4, 4)))
    params = Dict(
        Turing.@varname(beta_mean) => [0.1, 0.2],
        Turing.@varname(beta_precision) => [-0.4, 0.15],
        Turing.@varname(shared_groups[1].L) => L_shared,
        Turing.@varname(shared_groups[1].tau) => [0.4, 0.7, 0.25, 0.45],
        Turing.@varname(shared_groups[1].z_flat) =>
            [-0.2, 0.4, 0.5, -0.7,
             1.1, 0.3, 0.2, 0.6,
             -0.5, 0.8, 0.4, -0.1],
    )
    shared_coefficients = transpose(Diagonal(params[
        Turing.@varname(shared_groups[1].tau)]) * Matrix(L_shared.L) *
        reshape(params[Turing.@varname(shared_groups[1].z_flat)], 4, 3))
    mean_group_effect = vec(sum(mean_block.matrix .*
        shared_coefficients[mean_block.indices, 1:2]; dims=2))
    precision_group_effect = vec(sum(precision_block.matrix .*
        shared_coefficients[precision_block.indices, 3:4]; dims=2))
    linear_mean = backend.plan.mean.design.matrix *
                  params[Turing.@varname(beta_mean)] + mean_group_effect
    linear_precision = backend.plan.precision.design.matrix *
                       params[Turing.@varname(beta_precision)] +
                       precision_group_effect
    mu = exp.(linear_mean)
    phi = exp.(linear_precision)
    prior = sum(logpdf.(Normal(), params[Turing.@varname(beta_mean)])) +
            sum(logpdf.(Normal(), params[Turing.@varname(beta_precision)])) +
            logpdf(LKJCholesky(4, 2.5), L_shared) +
            sum(logpdf.([Exponential(2.0), Exponential(0.25),
                         Exponential(2.0), Exponential(0.5)],
                        params[Turing.@varname(shared_groups[1].tau)])) +
            sum(logpdf.(Normal(),
                params[Turing.@varname(shared_groups[1].z_flat)]))
    likelihood = sum(logpdf.(BRM.NegativeBinomial2.(
        mu, phi), backend.plan.response))
    returned = Turing.DynamicPPL.returned(backend.model, params)
    @test Turing.logjoint(backend.model, params) ≈
          prior + likelihood atol=1e-12 rtol=1e-12
    @test Turing.logprior(backend.model, params) ≈ prior atol=1e-12 rtol=1e-12
    @test returned.shared_groups[1].coefficients == shared_coefficients

    wrong_dimension = (@brm begin
        log(mu) ~ 1 + x + (1 + x | joint | subject)
        log(phi) ~ 1 + z + (1 + z | joint | subject)
        cor(:, joint) ~ LKJCholesky(3, 2)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1]))
    @test_throws "does not match" TuringBRMI(wrong_dimension)

    unknown_margin = (@brm begin
        log(mu) ~ 1 + x + (1 + x | joint | subject)
        log(phi) ~ 1 + z + (1 + z | joint | subject)
        sd(mu, joint, nope) ~ Exponential(1)
        y ~ BRM.NegativeBinomial2(mu, phi)
    end)((; dist_data..., y=[0, 2, 5, 1]))
    @test_throws "matches no random-effect margin" TuringBRMI(unknown_margin)
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
