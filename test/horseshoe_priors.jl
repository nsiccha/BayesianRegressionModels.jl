# test/horseshoe_priors.jl — configurable scalar Horseshoe scale contract.
#
# Run:
#   julia --project=test test/horseshoe_priors.jl

using Test
using BayesianRegressionModels
using Distributions: Normal
using StanBlocks

const BRM = BayesianRegressionModels
const HS_DF = (; y=[-0.4, 0.1, 0.7])

default_builder = @brm begin
    beta ~ Horseshoe()
    y ~ Normal(beta, 1)
end

scaled_builder = @brm begin
    beta ~ Horseshoe(local_scale=1 / 4, global_scale=0.1)
    y ~ Normal(beta, 1)
end

@testset "scalar Horseshoe scale keywords" begin
    default = SBBRMI(default_builder(HS_DF); mod=@__MODULE__)
    scaled = SBBRMI(scaled_builder(HS_DF); mod=@__MODULE__)
    default_code = BRM.stan_code(default)
    scaled_code = BRM.stan_code(scaled)

    # No keywords keep the historical sibling and unit half-Cauchy scales.
    default_decl = only(d for d in generative_plan(default).declarations
                        if d.target === :beta)
    @test default_decl.family === :_sb_horseshoe
    @test occursin("beta_lambda ~ cauchy(0.0, 1.0);", default_code)
    @test occursin("beta_tau ~ cauchy(0.0, 1.0);", default_code)
    @test occursin("beta_raw ~ std_normal();", default_code)

    # Configured calls take the sibling seam. Literal arithmetic is evaluated
    # with Julia semantics before becoming a Stan literal.
    scaled_decl = only(d for d in generative_plan(scaled).declarations
                       if d.target === :beta)
    @test scaled_decl.family === :_sb_horseshoe_scaled
    @test scaled_decl.keywords.local_scale == 0.25
    @test scaled_decl.keywords.global_scale == 0.1
    @test occursin("beta_lambda ~ cauchy(0.0, 0.25);", scaled_code)
    @test occursin("beta_tau ~ cauchy(0.0, 0.1);", scaled_code)
    @test occursin("beta_raw ~ std_normal();", scaled_code)

    @test StanBlocks.stan.transpiles(default.model)
    @test StanBlocks.stan.transpiles(scaled.model)
    @test StanBlocks.stanc_check(default_code; warn_pedantic=false).ok
    @test StanBlocks.stanc_check(scaled_code; warn_pedantic=false).ok
end

@testset "one scale defaults and invalid configurations refuse loudly" begin
    local_only = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=0.3)
        y ~ Normal(beta, 1)
    end
    local_code = BRM.stan_code(SBBRMI(local_only; mod=@__MODULE__))
    @test occursin("beta_lambda ~ cauchy(0.0, 0.3);", local_code)
    @test occursin("beta_tau ~ cauchy(0.0, 1.0);", local_code)

    positional = @brm HS_DF begin
        beta ~ Horseshoe(0.5, 0.1)
        y ~ Normal(beta, 1)
    end
    @test_throws "accepts no positional arguments" SBBRMI(
        positional; mod=@__MODULE__)

    unknown = @brm HS_DF begin
        beta ~ Horseshoe(scale=0.5)
        y ~ Normal(beta, 1)
    end
    @test_throws "accepts only `local_scale` and `global_scale`" SBBRMI(
        unknown; mod=@__MODULE__)

    zero = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=0.0)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(zero; mod=@__MODULE__)

    negative = @brm HS_DF begin
        beta ~ Horseshoe(global_scale=-0.1)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(
        negative; mod=@__MODULE__)

    nonfinite = @brm HS_DF begin
        beta ~ Horseshoe(global_scale=1.0 / 0.0)
        y ~ Normal(beta, 1)
    end
    @test_throws "finite and strictly positive" SBBRMI(
        nonfinite; mod=@__MODULE__)

    boolean = @brm HS_DF begin
        beta ~ Horseshoe(local_scale=true)
        y ~ Normal(beta, 1)
    end
    @test_throws "numeric formula constant" SBBRMI(boolean; mod=@__MODULE__)
end
