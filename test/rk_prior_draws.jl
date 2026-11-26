using Test, BayesianRegressionModels, Distributions, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL
using Statistics: mean, var
import BridgeStan

const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

@stestset "seeded native prior draws preserve conditional laws across designs" begin
    model = @brm begin
        scale ~ Exponential(1.0)
        slope ~ Normal(0.0, scale)
        mu = slope * dose
        y ~ Normal(mu, 1.0)
    end
    prior = RKBRMI(model((; dose=[1.0, 2.0])))
    draws = rk_prior_draws(prior, 2048; seed=839)
    sample = rk_draw_values(prior, draws, (:scale, :slope))
    @test length(draws) == length(sample.scale) == 2048
    @test all(>(0), sample.scale)
    @test mean(sample.scale) ≈ 1 atol=0.08
    z = sample.slope ./ sample.scale
    for selected in (sample.scale .< log(2), sample.scale .>= log(2))
        @test abs(mean(z[selected])) < 0.12
        @test abs(var(z[selected]) - 1) < 0.20
    end
    first_dose, second_dose = [2.0], [3.0, 7.0]
    a = reprocess(prior, (; dose=first_dose))
    b = reprocess(prior, (; dose=second_dose, y=[-12.0, 18.0]))
    read_mu(rk, batch) = rk_draw_values(rk, batch, (:mu,)).mu
    read_a, read_b = read_mu(a, draws), read_mu(b, draws)
    again = rk_prior_draws(prior, length(draws); seed=839)
    @test read_mu(b, again) == read_b
    @test read_mu(a, again) == read_a
    @test read_a == [s .* first_dose for s in sample.slope]
    @test read_b == [s .* second_dose for s in sample.slope]
    @test first(read_mu(a, rk_prior_draws(prior, 1; seed=840))) != first(read_a)
    @test_throws ArgumentError rk_prior_draws(prior, 0; seed=1)
    @test_throws ArgumentError rk_prior_draws(RKBRMI(model((; dose=[1.0], y=[2.0]))), 1; seed=1)
    @test isempty(coordinate_names(prior.model.layout))
end

@stestset "native prior draws compared to Stan at the same physical parameters" begin
    model = @brm begin
        sigma ~ Exponential(1.0)
        slope ~ Normal(0.0, sigma)
        mu = slope * dose
        y ~ Normal(mu, sigma)
    end
    prior = RKBRMI(model((; dose=[1.0, 2.0])))
    batch = rk_prior_draws(prior, 4; seed=723)
    design = (; dose=[0.3, 3.0, 7.1], y=zeros(3))
    native = rk_draw_values(reprocess(prior, design), batch, (:mu, :slope, :sigma))
    sb = SBBRMI(model(design); mod=@__MODULE__)
    stan = BRM.stan_instantiate(sb).model
    names = BridgeStan.param_names(stan; include_tp=false, include_gq=false)
    outputs = BridgeStan.param_names(stan; include_tp=true, include_gq=false)
    descriptor = brm_descriptor(sb)
    mu_indices = brm_output_coordinates(descriptor, :mu, outputs)
    @test Set(names) == Set(("slope", "sigma"))
    for i in 1:length(batch)
        theta = [getproperty(native, Symbol(name))[i] for name in names]
        u = BridgeStan.param_unconstrain(stan, theta)
        quantities = BridgeStan.param_constrain(stan, u; include_tp=true, include_gq=false)
        @test quantities[mu_indices] ≈ native.mu[i]
    end
end

@stestset "native prior draws refuse parameter-dependent restricted mass" begin
    model = @brm begin
        center ~ Normal(0, 1)
        bounded ~ restricted(Normal(0.2, 0.7), center, center + 2)
        y ~ Normal(bounded, 0.8)
    end
    rk = RKBRMI(model((;)))
    @test_throws "keeps sampled parameters" rk_prior_draws(rk, 2; seed=23)
end

@stestset "prior value graph replays the same hierarchical parameter points" begin
    model = @brm begin
        scale ~ Exponential(1.0)
        slope ~ Normal(0.0, scale)
        mu = slope * dose
        y ~ Normal(mu, 1.0)
    end
    prior = RKBRMI(model((; dose=[1.0, 2.0])))
    fit = BRM._rk_prior_values(prior).backend
    @test isempty(coordinate_names(prior.model.layout))
    @test BRM._rk_prior_response_names(prior.plan) == (:y,)
    names = coordinate_names(fit.model.layout)
    @test Set(names) == Set((:scale, :slope))
    U = zeros(length(names), 2)
    U[only(findall(==(:scale), names)), :] = log.([0.5, 1.25])
    U[only(findall(==(:slope), names)), :] = [-1.0, 3.0]
    function read_design(dose)
        # Measured outcomes on a reporting design never change the prior fit.
        rk = reprocess(prior, (; dose, y=fill(11.0, length(dose))))
        @test isempty(coordinate_names(rk.model.layout))
        values = BRM._rk_prior_values(rk).backend
        @test values.model === fit.model
        rk_draw_values(values, U, (:scale, :slope, :mu))
    end
    a, b = [3.0, 4.0], [5.0]
    first_a, first_b = read_design(a), read_design(b)
    @test read_design(b) == first_b
    @test read_design(a) == first_a
    for (dose, values) in ((a, first_a), (b, first_b))
        @test values.scale ≈ [0.5, 1.25]
        @test values.slope == [-1.0, 3.0]
        @test values.mu == [-dose, 3 .* dose]
    end
end

@stestset "prior value graph retains shared group effects and cell readouts" begin
    model = @brm begin
        eta ~ 1 + (1 | p | subject)
        offset ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        curve ~ kernel(dose, eta, offset, y) do d, a, b, obs
            location = b + d * exp(a)
            obs ~ normal(location, 1.0)
            location
        end
    end
    data = (; subject=["s2", "s1"], dose=[1.0, 2.0])
    prior = RKBRMI(model(data))
    fit = BRM._rk_prior_values(prior).backend
    U = reshape(collect(range(-0.4, 0.4; length=fit.model.layout.total)), :, 1)
    @test size(U, 1) > 4
    function values_on(dose)
        target = reprocess(prior, merge(data, (; dose, y=[0.5, 0.2])))
        values = BRM._rk_prior_values(target).backend
        map(only, rk_draw_values(values, U, (:eta, :offset, :curve)))
    end
    a, b = values_on(data.dose), values_on(2 .* data.dose)
    @test a.eta == b.eta
    @test a.offset == b.offset
    @test a.curve ≈ a.offset .+ data.dose .* exp.(a.eta)
    @test b.curve ≈ b.offset .+ 2 .* data.dose .* exp.(b.eta)
    draws = rk_prior_draws(prior, 4; seed=174)
    first_design = rk_draw_values(prior, draws, (:eta, :offset, :curve))
    doubled = 2 .* data.dose
    replay = reprocess(prior, merge(data, (; dose=doubled)))
    second_design = rk_draw_values(replay, draws, (:eta, :offset, :curve))
    @test first_design.eta == second_design.eta
    @test first_design.offset == second_design.offset
    for i in 1:length(draws)
        @test second_design.curve[i] ≈ first_design.offset[i] .+ doubled .* exp.(first_design.eta[i])
    end
end
