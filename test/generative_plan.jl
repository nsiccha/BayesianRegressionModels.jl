# test/generative_plan.jl — declaration-driven generative-plan acceptance

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

stanc_ok(model) = StanBlocks.stanc_check(
    StanBlocks.stan_code(model); warn_pedantic=false).ok

@testset "generative plan — ordinary and multiple outputs" begin
    df = (; x=[0.0, 1.0, 2.0], y=[1.0, 1.5, 2.0], z=[2.0, 2.5, 3.0])
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
        z ~ Normal(mu, 2 * sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    plan = generative_plan(sb)

    @test BayesianRegressionModels.stan_code(plan) ==
          BayesianRegressionModels.stan_code(sb)
    @test stanc_ok(plan.model)
    @test count(d -> d.role === :observation, plan.declarations) == 2
    @test [(d.target, d.data_source, d.draw) for d in plan.declarations
           if d.role === :observation] ==
          [(:y, :y, :y_gen), (:z, :z, :z_gen)]
    @test occursin("y_gen", BayesianRegressionModels.stan_code(plan))
    @test occursin("z_gen", BayesianRegressionModels.stan_code(plan))
    @test any(d -> d.target === :sigma && d.family === :exponential,
              plan.declarations)

    replay = reprocess(plan, (; x=[3.0, 4.0], y=[0.0, 0.0], z=[0.0, 0.0]))
    @test replay.data[:x] == [3.0, 4.0]
    @test length(filter(d -> d.role === :observation, replay.declarations)) == 2

    reusable = generative_plan(builder, df; mod=@__MODULE__)
    rebuilt = generative_plan(reusable,
        (; x=[5.0, 6.0, 7.0, 8.0], y=zeros(4), z=zeros(4)))
    @test rebuilt.data[:x] == [5.0, 6.0, 7.0, 8.0]
end

StanBlocks.@deffun begin
    generative_kernel_cell(ts::vector[nt], d::real, eta::vector[ne])::vector[nt] = begin
        CL = 1.0 * exp(eta[1])
        V = 10.0 * exp(eta[2])
        d / V * exp(-(CL / V) * ts)
    end
end

kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    pred ~ kernel(t, dose; by=subject, model=generative_kernel_cell, n_eta=2,
                  obs=CombinedError(dv, sigma_a, sigma_p))
end

kernel_doblock_builder = @brm begin
    sigma ~ Exponential(1)
    pred ~ kernel(t, dose, dv; by=subject, n_eta=2) do ts, d, yy, eta
        CL = 1.0 * exp(eta[1])
        V = 10.0 * exp(eta[2])
        mu = d / V * exp(-(CL / V) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

kernel_schedule(n) = (;
    t=[collect(1.0:3.0) for _ in 1:n],
    dose=fill(100.0, n),
    dv=[zeros(3) for _ in 1:n],
    subject=collect(1:n),
)

@testset "generative plan — kernel eta block and new groups" begin
    plan = generative_plan(kernel_builder, kernel_schedule(2); mod=@__MODULE__)
    obs = only(filter(d -> d.role === :observation, plan.declarations))

    @test obs.target === :kernel_y
    @test obs.family === :normal
    @test obs.data_source === :dv
    @test obs.context == (:pred,)
    @test obs.draw === :pred_kernel_y_gen
    # StanBlocks currently emits no generated quantity for an observation
    # nested inside a plate. The plan still assigns it a stable executor-facing
    # draw name instead of pretending the posterior model already produces it.
    @test !occursin(string(obs.draw), BayesianRegressionModels.stan_code(plan))
    @test any(d -> d.target === :kernel_L_pred && d.family === :lkj_corr_cholesky,
              plan.declarations)
    @test any(d -> d.target === :kernel_om_pred && d.family === :std_normal,
              plan.declarations)
    @test any(d -> d.target === :kernel_z && d.context == (:pred,),
              plan.declarations)
    @test stanc_ok(plan.model)

    rebuilt = generative_plan(plan, kernel_schedule(4))
    @test rebuilt.data[:kernel_nsub_pred] == 4
    @test only(filter(d -> d.role === :observation, rebuilt.declarations)).data_source === :dv

    doblock = generative_plan(kernel_doblock_builder, kernel_schedule(2);
                              mod=@__MODULE__)
    doblock_obs = only(filter(d -> d.role === :observation, doblock.declarations))
    @test doblock_obs.target === :yy
    @test doblock_obs.data_source === :dv
    @test doblock_obs.context == (:pred,)
end
