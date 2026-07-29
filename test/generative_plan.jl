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

kernel_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV
        CL = exp(lCL)
        V = exp(lV)
        mu = d / V * exp(-(CL / V) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

kernel_schedule(n; subject=collect(1:n)) = (;
    t=[collect(1.0:3.0) for _ in 1:n],
    dose=fill(100.0, n),
    dv=[zeros(3) for _ in 1:n],
    subject,
)

@testset "generative plan — kernel formula LPs and new groups" begin
    plan = generative_plan(kernel_builder, kernel_schedule(2); mod=@__MODULE__)
    obs = only(filter(d -> d.role === :observation, plan.declarations))

    @test obs.target === :yy
    @test obs.family === :normal
    @test obs.data_source === :dv
    @test obs.context == (:pred,)
    @test obs.draw === :pred_yy_gen
    # StanBlocks currently emits no generated quantity for an observation
    # nested inside a plate. The plan still assigns it a stable executor-facing
    # draw name instead of pretending the posterior model already produces it.
    @test !occursin(string(obs.draw), BayesianRegressionModels.stan_code(plan))
    @test count(d -> d.family === :popefs, plan.declarations) == 2
    @test any(d -> d.family === :ranef_correlated_draws, plan.declarations)
    @test !any(d -> startswith(string(d.target), "kernel_L_") ||
                    startswith(string(d.target), "kernel_om_") ||
                    startswith(string(d.target), "kernel_z"),
              plan.declarations)
    @test stanc_ok(plan.model)

    rebuilt = generative_plan(plan,
        kernel_schedule(4; subject=["s4", "s2", "s3", "s1"]))
    @test rebuilt.data[:kernel_nsub_pred] == 4
    @test only(filter(d -> d.role === :observation, rebuilt.declarations)).data_source === :dv
    @test !haskey(rebuilt.data, :subject)

    repeated = kernel_schedule(4; subject=[10_001, 10_001, 10_002, 10_002])
    @test_throws "pre-grouped per-subject data" generative_plan(plan, repeated)
end

@testset "generative plan — structured RHS parameters, dimension, constraints" begin
    plan = generative_plan(kernel_builder, kernel_schedule(2); mod=@__MODULE__)
    decl(t) = only(filter(d -> d.target === t, plan.declarations))

    # Positional arguments describe the backend expression BRM actually emits.
    # Julia `Exponential(theta)` uses scale while Stan `exponential(beta)` uses
    # rate, so the declaration exposes the exact translated expression.
    @test decl(:sigma).arguments == (:(1.0 ./ 1),)
    @test decl(:sigma).keywords == NamedTuple()

    # declarations that spell no size of their own report `()` rather than
    # guessing: scalars, and extents owned by data or by a submodel.
    @test decl(:sigma).dimension == ()
    @test decl(:pred).dimension == ()
    @test decl(:yy).dimension == ()
    @test decl(:yy).arguments[1] === :mu

    # a `do`-block RHS decomposes on its underlying call, matching `family`
    @test decl(:pred).family === :plate
    @test decl(:pred).arguments == (:t, :dose, :dv, :log_CL, :log_V)
    @test haskey(decl(:pred).keywords, :outer)

    # structured latent submodels keep their keywords verbatim; a submodel's
    # own sizing kwargs are NOT mistaken for a declared dimension.
    df = (; x=randn(20), y=randn(20), subject=repeat(1:5, 4))
    ranef_plan = generative_plan((@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | subject)
        y ~ Normal(mu, sigma)
    end), df; mod=@__MODULE__)
    r = only(filter(d -> d.family === :ranef_correlated, ranef_plan.declarations))
    @test keys(r.keywords) == (:Z, :group_idx, :n_groups, :n_terms)
    @test r.dimension == ()
    @test r.constraints == NamedTuple()
    @test stanc_ok(ranef_plan.model)
end
