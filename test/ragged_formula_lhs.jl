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

censored_flat_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
end

# The pre-fix workaround: callers had to group both the response and its bound
# themselves, even though the kernel still consumed the original flat frame.
censored_grouped_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    pk_conc ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=pk_lloq)
end

censored_scalar_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=0.10)
end

truncated_two_bound_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ truncated(
        Normal(loc, addprop(loc, sigma_add, sigma_prop));
        lower=pk_lloq, upper=pk_uloq)
end

interval_flat_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ interval_censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop)); upper=pk_uloq)
end

explicit_bound_grouping_builder = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    loc ~ kernel(ragged(obs_idx, obs_subject), log_CL) do idxs, lCL
        exp(lCL) .+ 0.0 .* idxs
    end
    ragged(pk_conc, obs_subject) ~ censored(
        Normal(loc, addprop(loc, sigma_add, sigma_prop));
        lower=ragged(pk_lloq, obs_subject))
end

flat_data = (;
    subject=["s2", "s1", "s3"],
    obs_subject=["s1", "s2", "s3", "s2", "s3", "s3"],
    obs_idx=[1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
    obs_y=[0.11, 0.21, 0.31, 0.22, 0.32, 0.33],
)
expected = [[0.21, 0.22], [0.11], [0.31, 0.32, 0.33]]
grouped_data = merge(flat_data, (; obs_y=expected))

censored_flat_data = (;
    subject=["s2", "s1"],
    obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
    obs_idx=collect(1.0:7.0),
    pk_conc=[0.15, 0.20, 0.10, 0.25, 0.10, 0.30, 0.18],
    pk_lloq=fill(0.10, 7),
    pk_uloq=fill(0.40, 7),
)
censored_rows = [[2, 4, 5, 7], [1, 3, 6]]
censored_grouped_data = merge(censored_flat_data, (;
    pk_conc=[censored_flat_data.pk_conc[r] for r in censored_rows],
    pk_lloq=[censored_flat_data.pk_lloq[r] for r in censored_rows],
    pk_uloq=[censored_flat_data.pk_uloq[r] for r in censored_rows],
))

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

@testset "ragged observation LHS — observed censoring bound" begin
    flat = SBBRMI(censored_flat_builder(censored_flat_data); mod=@__MODULE__)
    grouped = SBBRMI(
        censored_grouped_builder(censored_grouped_data); mod=@__MODULE__)

    expected_response = censored_grouped_data.pk_conc
    expected_lower = censored_grouped_data.pk_lloq
    @test flat.data[:pk_conc] == expected_response == grouped.data[:pk_conc]
    @test flat.data[:pk_conc_lower_pk_lloq_ragged] ==
          expected_lower == grouped.data[:pk_lloq]
    @test flat.data[:pk_lloq] == censored_flat_data.pk_lloq

    # The exact reported formula, including BRM's add+proportional residual
    # helper over the ragged kernel output, must survive both compiler gates.
    @test StanBlocks.stan.transpiles(flat.model)
    @test StanBlocks.stan.transpiles(grouped.model)
    scalar_bound = SBBRMI(
        censored_scalar_bound_builder(censored_flat_data); mod=@__MODULE__)
    @test StanBlocks.stan.transpiles(scalar_bound.model)
    two_bound = SBBRMI(
        truncated_two_bound_builder(censored_flat_data); mod=@__MODULE__)
    interval = SBBRMI(interval_flat_builder(censored_flat_data); mod=@__MODULE__)
    @test two_bound.data[:pk_conc_lower_pk_lloq_ragged] == expected_lower
    @test two_bound.data[:pk_conc_upper_pk_uloq_ragged] ==
          censored_grouped_data.pk_uloq
    @test interval.data[:pk_conc_upper_pk_uloq_ragged] ==
          censored_grouped_data.pk_uloq
    for brmi in (flat, grouped)
        code = BayesianRegressionModels.stan_code(brmi)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    end

    # Diff the fix against the old manual-grouping workaround at the executable
    # boundary. The derived bound has a different Stan data name, but the model
    # density and gradient must be exactly the same at the same unconstrained q.
    flat_problem = brm_execute(
        brm_descriptor(censored_flat_builder, censored_flat_data; mod=@__MODULE__),
        :fit,
    )
    grouped_problem = brm_execute(
        brm_descriptor(
            censored_grouped_builder, censored_grouped_data; mod=@__MODULE__),
        :fit,
    )
    dim = LogDensityProblems.dimension(flat_problem)
    @test LogDensityProblems.dimension(grouped_problem) == dim
    q = fill(-0.2, dim)
    flat_lp_grad = LogDensityProblems.logdensity_and_gradient(flat_problem, q)
    grouped_lp_grad =
        LogDensityProblems.logdensity_and_gradient(grouped_problem, q)
    @test flat_lp_grad == grouped_lp_grad
    @test isfinite(first(flat_lp_grad))
    @test all(isfinite, last(flat_lp_grad))

    # The authoritative grouping lives at the formula LHS. Asking a bound to
    # perform a second, independent grouping remains unsupported and loud.
    @test_throws "bounds must be numeric literals or observed data columns" SBBRMI(
        explicit_bound_grouping_builder(censored_flat_data); mod=@__MODULE__)

    short_bound = merge(censored_flat_data, (;
        pk_lloq=censored_flat_data.pk_lloq[1:end-1],
    ))
    @test_throws "has 6 rows but the flat response has 7" SBBRMI(
        censored_flat_builder(short_bound); mod=@__MODULE__)

    wrong_segments = merge(censored_flat_data, (;
        pk_lloq=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
    ))
    @test_throws "group lengths [3, 4]; expected [4, 3]" SBBRMI(
        censored_flat_builder(wrong_segments); mod=@__MODULE__)
end
