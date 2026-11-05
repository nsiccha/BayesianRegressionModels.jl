# Run: julia --startup-file=no --project=test test/omitted_interval_response.jl
using Test, BayesianRegressionModels, StanBlocks, BridgeStan
using LogDensityProblems, Distributions
using Statistics: mean, var

const BRM = BayesianRegressionModels

const interval_builder = @brm begin
    location ~ Normal(0.3, 0.7)
    sigma ~ LogNormal(-0.2, 0.4)
    mu = location + x
    y ~ interval_censored(Normal(mu[row], sigma); upper=hi)
end
const interval_data = (; x=[-0.5, 0.2, 1.1], row=[3, 1, 3, 2],
    y=[-0.1, -0.7, 0.4, 0.1], hi=[0.7, 0.2, 1.3, 0.8])
const interval_prior_data = Base.structdiff(interval_data, NamedTuple{(:y,)})

@testset "bound intervals retain their probability mass and validation" begin
    bound = SBBRMI(interval_builder(interval_data); mod=@__MODULE__, total_groups=())
    problem = BRM.stan_instantiate(bound)
    @test BridgeStan.param_unc_names(problem.model) == ["location", "sigma"]
    q = [0.2, log(0.8)]
    function expected(q)
        location, log_sigma = q
        sigma = exp(log_sigma)
        lp = logpdf(Normal(0.3, 0.7), location) +
             logpdf(LogNormal(-0.2, 0.4), sigma) + log_sigma
        for i in eachindex(interval_data.y)
            law = Normal(location + interval_data.x[interval_data.row[i]], sigma)
            lp += log(cdf(law, interval_data.hi[i]) - cdf(law, interval_data.y[i]))
        end
        lp
    end
    gradient = similar(q)
    lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
        propto=false, jacobian=true)
    @test lp ≈ expected(q) atol=1e-10
    h = 1e-5
    finite_difference = [(expected(q + h * [i == j for i in eachindex(q)]) -
                          expected(q - h * [i == j for i in eachindex(q)])) / (2h)
                         for j in eachindex(q)]
    @test gradient ≈ finite_difference atol=1e-7
    @test_throws "strictly below" SBBRMI(interval_builder(
        merge(interval_data, (; hi=copy(interval_data.y)))); mod=@__MODULE__)
    @test_throws "rows but response" SBBRMI(interval_builder(
        merge(interval_data, (; hi=[1.0, 2.0]))); mod=@__MODULE__)
end

@testset "omitted interval evidence draws the unchanged joint prior" begin
    before = deepcopy(interval_prior_data)
    prior = SBBRMI(interval_builder(interval_prior_data); mod=@__MODULE__, total_groups=())
    @test isequal(interval_prior_data, before)
    code = BRM.stan_code(prior)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test :y in prior.model.observations
    declaration = only(d for d in generative_plan(prior).declarations if d.target === :y)
    @test declaration.role === :observation
    @test declaration.family === :interval_censored
    @test isnothing(declaration.data_source)
    descriptor = brm_descriptor(prior)
    @test brm_output(descriptor, :y; role=:posterior_predictive).name === :y_gen
    @test [o.name for o in descriptor.outputs if o.logical === :y] == [:y_gen]
    problem = BRM.stan_instantiate(prior)
    @test LogDensityProblems.dimension(problem) == 0
    @test BridgeStan.param_unc_num(problem.model) == 0
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    @test all(n -> n in names, ["y_gen.$i" for i in eachindex(interval_data.hi)])
    rng = BridgeStan.StanRNG(problem.model, 47)
    draws = [BridgeStan.param_constrain(problem.model, Float64[];
        include_tp=true, include_gq=true, rng) for _ in 1:512]
    @test all(v -> all(isfinite, v), draws)
    locations = [d[findfirst(==("location"), names)] for d in draws]
    @test length(unique(locations)) == length(draws)
    @test mean(locations) ≈ 0.3 atol=0.1
    @test var(locations) ≈ 0.7^2 atol=0.1
    residuals = Float64[]
    for (d, location) in zip(draws, locations), i in eachindex(interval_data.hi)
        y = d[findfirst(==("y_gen.$i"), names)]
        @test y == d[findfirst(==("y.$i"), names)]
        sigma = d[findfirst(==("sigma"), names)]
        push!(residuals, (y - location - interval_data.x[interval_data.row[i]]) / sigma)
    end
    @test mean(residuals) ≈ 0.0 atol=0.1
    @test var(residuals) ≈ 1.0 atol=0.12
    @test any(d[findfirst(==("y_gen.1"), names)] > interval_data.hi[1] for d in draws)
    @test any(d[findfirst(==("y_gen.1"), names)] < interval_data.y[1] for d in draws)
end

@testset "retained upper endpoints size scalar continuous and discrete laws" begin
    scalar = @brm begin
        location ~ Normal(0.0, 1.0)
        y ~ interval_censored(Normal(location, 1.0); upper=hi)
    end
    discrete = @brm begin
        rate ~ LogNormal(1.5, 0.2)
        y ~ interval_censored(Poisson(rate); upper=hi)
    end
    for (builder, hi) in ((scalar, fill(-100.0, 3)), (discrete, [1, 2, 3]))
        sb = SBBRMI(builder((; hi)); mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
        problem = BRM.stan_instantiate(sb)
        @test LogDensityProblems.dimension(problem) == 0
        names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        coordinates = brm_output_coordinates(
            brm_output(brm_descriptor(sb), :y; role=:posterior_predictive), names)
        @test length(coordinates) == length(hi)
        rng = BridgeStan.StanRNG(problem.model, 82)
        draws = [BridgeStan.param_constrain(problem.model, Float64[];
            include_tp=true, include_gq=true, rng)[coordinates] for _ in 1:64]
        @test all(d -> all(isfinite, d), draws)
        @test any(d -> any(d .> hi), draws)
        if builder === discrete
            @test all(d -> all(x -> isinteger(x) && x >= 0, d), draws)
        end
    end
    @test_throws "discrete upper bounds must be integers" SBBRMI(
        discrete((; hi=[1.5, 2.5])); mod=@__MODULE__)
end

@testset "an omitted interval leaves the retained response density intact" begin
    builder = @brm begin
        location ~ Normal(0.0, 1.0)
        mu = location + x
        y ~ interval_censored(Normal(mu, 1.0); upper=hi)
        z ~ Normal(location, 1.0)
    end
    data = (; x=[-0.5, 0.5, 1.0], hi=fill(-100.0, 3), z=[-0.1, 0.2, 0.4])
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb)
    @test BridgeStan.param_unc_names(problem.model) == ["location"]
    @test LogDensityProblems.dimension(problem) == 1
    gradient = zeros(1)
    lp, _ = BridgeStan.log_density_gradient!(problem.model, [0.15], gradient;
        propto=false, jacobian=true)
    @test lp ≈ logpdf(Normal(), 0.15) + sum(logpdf.(Normal(0.15, 1.0), data.z)) atol=1e-10
    @test only(gradient) ≈ -0.15 + sum(data.z .- 0.15) atol=1e-10
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    coordinates = brm_output_coordinates(
        brm_output(brm_descriptor(sb), :y; role=:posterior_predictive), names)
    @test length(coordinates) == length(data.hi)
    rng = BridgeStan.StanRNG(problem.model, 92)
    draw = BridgeStan.param_constrain(problem.model, [0.15];
        include_tp=true, include_gq=true, rng)
    @test all(draw[coordinates] .> data.hi)
    @test all(n -> !startswith(n, "y_likelihood"), names)
end

@testset "upper-only ragged inputs retain empty groups" begin
    builder = @brm begin
        location ~ Normal(0.0, 1.0)
        y ~ interval_censored(Normal(location, 1.0); upper=hi)
    end
    sb = SBBRMI(builder((; hi=[[0.1, 0.2], Float64[], [0.3]])); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    output = brm_output(brm_descriptor(sb), :y; role=:posterior_predictive)
    @test output.segments == [2, 2, 3]
    problem = BRM.stan_instantiate(sb)
    @test LogDensityProblems.dimension(problem) == 0
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    coordinates = brm_output_coordinates(output, names)
    @test length(coordinates) == 3
    rng = BridgeStan.StanRNG(problem.model, 93)
    draw = BridgeStan.param_constrain(problem.model, Float64[];
        include_tp=true, include_gq=true, rng)
    @test all(isfinite, draw[coordinates])
end

@testset "a sole interval predictor remains a predictor" begin
    builder = @brm begin
        mu ~ interval_censored(conc; upper=lloq)
        y ~ Normal(mu, 1.0)
    end
    brmi = builder((; conc=[0.7, 0.25, 1.2], lloq=fill(0.25, 3),
        y=[0.4, 0.2, 1.1]))
    @test popcoefnames(brmi, :mu) == [:conc]
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    @test :mu ∉ sb.model.observations
end

@testset "fixed family arguments and a scalar upper still declare an observation" begin
    builder = @brm begin
        y ~ interval_censored(Normal(0.0, 1.0); upper=hi)
    end
    sb = SBBRMI(builder((; hi=0.5)); mod=@__MODULE__)
    @test :y in sb.model.observations
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    @test brm_output(brm_descriptor(sb), :y; role=:posterior_predictive).name === :y_gen
end
