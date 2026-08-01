# test/ragged_formula_lhs.jl — group a flat observation axis at the formula LHS.
#
# Run: julia --project=test test/ragged_formula_lhs.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal
using LogDensityProblems

flat_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(obs_y, obs_subject) ~ Normal(pred, sigma)
end

grouped_builder = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    obs_y ~ Normal(pred, sigma)
end

no_kernel_builder = @brm begin
    mu ~ 1 + x
    ragged(y, g) ~ Normal(mu, 1.0)
end

flat_data = (;
    subject=["s2", "s1", "s3"],
    obs_subject=["s1", "s2", "s3", "s2", "s3", "s3"],
    obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    obs_y=[0.11, 0.21, 0.31, 0.22, 0.32, 0.33],
)
expected = [[0.21, 0.22], [0.11], [0.31, 0.32, 0.33]]
grouped_data = merge(flat_data, (; obs_y=expected))

@testset "ragged observation LHS — flat formula boundary" begin
    brmi = flat_builder(flat_data)
    @test :obs_y in keys(brmi.operations)
    @test only(outcomes(brmi)).response === :obs_y

    flat = SBBRMI(brmi; mod=@__MODULE__)
    grouped = SBBRMI(grouped_builder(grouped_data); mod=@__MODULE__)

    @test flat.data[:obs_y] == expected
    @test flat.data[:obs_y] == grouped.data[:obs_y]
    @test BayesianRegressionModels.stan_code(flat) ==
          BayesianRegressionModels.stan_code(grouped)

    @test StanBlocks.stan.transpiles(flat.model)
    code = BayesianRegressionModels.stan_code(flat)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    d = brm_descriptor(flat_builder, flat_data; mod=@__MODULE__)
    draw = brm_output(d, :obs_y; role=:posterior_predictive)
    loglik = brm_output(d, :obs_y; role=:pointwise_loglik)
    @test draw.source === :obs_y && loglik.source === :obs_y
    @test draw.logical === :obs_y && loglik.logical === :obs_y
    @test draw.segments == [2, 3, 6]
    @test loglik.segments == draw.segments

    flat_problem = brm_execute(d, :fit)
    grouped_problem = brm_execute(
        brm_descriptor(grouped_builder, grouped_data; mod=@__MODULE__), :fit)
    dim = LogDensityProblems.dimension(flat_problem)
    @test LogDensityProblems.dimension(grouped_problem) == dim
    q = [0.05 * ((i % 7) - 3) for i in 1:dim]
    flat_lp, flat_grad = LogDensityProblems.logdensity_and_gradient(flat_problem, q)
    grouped_lp, grouped_grad =
        LogDensityProblems.logdensity_and_gradient(grouped_problem, q)
    @test flat_lp == grouped_lp
    @test flat_grad == grouped_grad

    @testset "invalid axes fail before Stan" begin
        mismatched = merge(flat_data, (; obs_subject=flat_data.obs_subject[1:end-1]))
        @test_throws "grouping column must name" SBBRMI(
            flat_builder(mismatched); mod=@__MODULE__)

        unknown = merge(flat_data, (;
            obs_subject=["s1", "s2", "s3", "s2", "s3", "not-a-subject"],
        ))
        @test_throws "name no subject" SBBRMI(flat_builder(unknown); mod=@__MODULE__)

        already_grouped = merge(flat_data, (; obs_y=expected))
        @test_throws "ALREADY-ragged response" SBBRMI(
            flat_builder(already_grouped); mod=@__MODULE__)

        @test_throws "needs a `kernel(...)` result" SBBRMI(
            no_kernel_builder((; x=[1.0, 2.0], y=[0.1, 0.2], g=[1, 1]));
            mod=@__MODULE__)
    end
end
