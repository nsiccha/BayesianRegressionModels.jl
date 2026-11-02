# BRM power-scaling assembly and integration acceptance.
# Generic CJS, PSIS and sensitivity tests live in PowerScaling.jl.
# Run: julia --project=test test/powerscale.jl

using Test
using BayesianRegressionModels
using BridgeStan
using StanBlocks
using Distributions
using PowerScaling
using Random
using Statistics
using LinearAlgebra
using LogDensityProblems

const BRM = BayesianRegressionModels
const BS = BridgeStan

# The gate runs before any draw is touched, so rejected shapes are provoked
# with dummy draws — no compile, no sampling.
function gate_message(builder, df)
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    try
        brm_powerscale_inputs(sb, zeros(2, 2), ["a", "b"])
    catch err
        return sprint(showerror, err)
    end
    error("gate test: expected brm_powerscale_inputs to throw")
end

@testset "coverage gate names everything outside" begin
    df = (; x=[0.0, 1.0, 2.0, 3.0], g=[1, 1, 2, 2],
           y=[0.2, 0.4, 0.6, 0.8], n=[10, 10, 10, 10])

    ranef = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end
    @test occursin("`|`", gate_message(ranef, df))

    hsgp_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(x)
        y ~ Normal(mu, sigma)
    end
    @test occursin("hsgp", gate_message(hsgp_builder, df))

    r2d2_builder = @brm begin
        effect(mu, :) ~ r2d2(R2=Beta(1, 1))
        mu ~ 1 + x
        y ~ Normal(mu, 1.0)
    end
    @test occursin("r2d2", gate_message(r2d2_builder, df))

    shared = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | g)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    @test occursin("sd", gate_message(shared, df))

    simplex = @brm begin
        s ~ Dirichlet(3, 1.0)
        mu ~ 1 + x
        y ~ Normal(mu, s[1])
    end
    @test occursin("Dirichlet", gate_message(simplex, df))

    gamma_prior = @brm begin
        tau ~ Gamma(2.0, 1.0)
        mu ~ 1 + x
        y ~ Normal(mu, tau)
    end
    @test occursin("Gamma", gate_message(gamma_prior, df))

    # Poisson is a LIKELIHOOD rejection (v1 admits Normal/BernoulliLogit):
    # the boundary is deliberate and this pins it until it grows.
    poisson = @brm begin
        mu ~ 1 + x
        n ~ Poisson(mu)
    end
    @test occursin("Poisson", gate_message(poisson, df))

    hierarchical = @brm begin
        tau ~ Exponential(1)
        sigma ~ Exponential(tau)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test occursin("non-constant", gate_message(hierarchical, df))

    gp_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + gp(x)
        y ~ Normal(mu, sigma)
    end
    @test occursin("gp", gate_message(gp_builder, df))
end

# ---- BridgeStan identity battery -------------------------------------------
#
# For each admitted shape: the Julia-side joint prior plus the Stan twins
# must reconstruct the BridgeStan target, up to ONE additive constant (draw
# independent by construction — normalisation constants cancel in importance
# weights, but a MISSING prior term would show up as draw-varying residual).
# Hand unconstrained draws keep this deterministic; no MCMC is involved.
function battery_identity(sb, df; sigma_unc::Union{Symbol,Nothing})
    d = brm_descriptor(sb)
    @test :pointwise_loglik in Symbol[op.name for op in d.operations]
    prob = brm_execute(d, :fit)
    n = LogDensityProblems.dimension(prob)
    S = 25
    q = 0.3 .* randn(Xoshiro(61), S, n)
    full_names = BS.param_names(
        prob.model; include_tp=true, include_gq=true)
    constrained = permutedims(reduce(hcat, [
        BS.param_constrain(prob.model, collect(row);
                           include_tp=true, include_gq=true,
                           rng=BS.new_rng(prob.model, 10_000 + i))
        for (i, row) in enumerate(eachrow(q))]))
    inputs = brm_powerscale_inputs(sb, constrained, full_names)
    @test all(isfinite, inputs.log_prior)
    @test all(isfinite, inputs.log_lik)
    target = [LogDensityProblems.logdensity(prob, collect(row))
              for row in eachrow(q)]
    jac = if isnothing(sigma_unc)
        zeros(S)
    else
        unc_names = BS.param_unc_names(prob.model)
        j = findfirst(==(string(sigma_unc)), unc_names)
        @test !isnothing(j)
        # `<lower=0>` sigma: q = log(sigma), so log|J| = q.
        vec(q[:, j])
    end
    julia = inputs.log_prior .+ inputs.log_lik
    stan = target .- jac
    @test maximum(abs, (julia .- mean(julia)) .- (stan .- mean(stan))) < 1e-6
    return d, constrained, full_names, inputs
end

@testset "identity battery — gaussian with overrides and cell means" begin
    df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25],
            g=[1, 1, 2, 2, 3, 3, 1, 2],
            h=[1, 2, 1, 2, 1, 2, 1, 2],
            y=[0.2, 0.1, -0.3, 0.4, 0.2, -0.2, 0.1, 0.0],
            z=[1.2, 1.1, 0.7, 1.4, 1.2, 0.8, 1.1, 1.0])
    builder = @brm begin
        sigma ~ Exponential(1)
        effect(mu, x) ~ Normal(0, 2)
        effect(mu, g) ~ Normal(0.5, 1.5)
        effect(mu2, h_lvl_2) ~ Normal(1.0, 0.5)
        mu ~ 1 + x + g
        mu2 ~ 0 + h
        y ~ Normal(mu, sigma)
        z ~ Normal(mu2, sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    d, constrained, full_names, inputs =
        battery_identity(sb, df; sigma_unc=:sigma)
    # The h block really is cell-mean coded — otherwise this test would prove
    # the wrong coding.
    h_res = brm_population_effect_coordinates(d, :mu2, full_names; coefficient=:h)
    @test h_res.coding === :cellmeans
    @test length(h_res.coordinates) == 2
    # Default variables: parameters in, twins and sampler diagnostics out.
    @test :sigma in inputs.variables
    @test !any(v -> endswith(string(v), "__"), inputs.variables)
    @test !any(v -> occursin("likelihood", string(v)), inputs.variables)
    @test !any(v -> occursin("_gen", string(v)), inputs.variables)
    # The summary runs end to end on the assembled triple.
    sens = brm_powerscale_sensitivity(constrained, inputs.log_prior,
                                   inputs.log_lik, full_names;
                                   variables=inputs.variables)
    @test BRMPowerscaleSensitivity === PowerScaling.PowerscaleSensitivity
    @test sens isa BRMPowerscaleSensitivity
    @test sens.variables == inputs.variables
    @test all(isfinite, sens.prior) && all(isfinite, sens.likelihood)
    @test all(isfinite, (sens.pareto_k.prior_lower, sens.pareto_k.prior_upper,
                         sens.pareto_k.likelihood_lower,
                         sens.pareto_k.likelihood_upper))
    rendered = sprint(show, MIME("text/plain"), sens)
    @test occursin("Power-scaling sensitivity", rendered)
    @test occursin("Pareto-k", rendered)
    @test inputs.descriptor isa BRMDescriptor

    # Original BRM implementation at 078a2324e6db85c889a9fcead76464c2e4179dae,
    # evaluated on these exact draws. Generic numerical tests live upstream;
    # this pins the assembled model's seven sampled-parameter rows.
    variables = [:sigma, Symbol("pop_mu_beta_pop.1"), Symbol("pop_mu_beta_pop.2"),
        Symbol("cat_mu_g_beta.1"), Symbol("cat_mu_g_beta.2"),
        Symbol("cat_mu2_h_beta.1"), Symbol("cat_mu2_h_beta.2")]
    selected = brm_powerscale_sensitivity(constrained, inputs.log_prior,
        inputs.log_lik, full_names; variables)
    @test selected.variables == variables
    @test selected.prior ≈ [0.13789838674866176, 0.10152085703156137,
        0.15786421648447513, 0.09308084960904192, 0.10227357588013433,
        0.087747645770414, 0.2630639281739019] atol=1e-8 rtol=1e-8
    @test selected.likelihood ≈ [0.22601292785881805, 0.26565718181789977,
        0.5806870907723355, 0.3175449811138312, 0.14218735930757082,
        0.24014589224273009, 0.40081559804927136] atol=1e-8 rtol=1e-8
    @test collect(selected.pareto_k) ≈ [-0.10368410781725323,
        1.6166814108773797, 1.1248250990003723, 0.4829095130520477] atol=1e-8 rtol=1e-8
    @test selected.diagnosis == fill(:prior_data_conflict, length(variables))
    # The deliberately dispersed supplied draws are a density-assembly test,
    # not a posterior fit: both old and new diagnostics flag unreliable IS.
    @test selected.reliable == (prior=false, likelihood=false)
end

@testset "identity battery — bernoullilogit, jacobian-free" begin
    df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25],
            y=[0, 1, 0, 1, 1, 0, 1, 0])
    builder = @brm begin
        effect(mu, Intercept) ~ Normal(0, 5)
        mu ~ 1 + x
        y ~ BernoulliLogit(mu)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    battery_identity(sb, df; sigma_unc=nothing)
end

@testset "missing generated quantities error loudly" begin
    df = (; x=[0.0, 1.0, 2.0, 3.0], y=[0.2, 0.4, 0.6, 0.8])
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    # This is exactly the Tier-1 GLM-fused shape (pure-population Gaussian):
    # fusion must preserve the pointwise twins power-scaling assembles.
    @test :pointwise_loglik in Symbol[op.name for op in d.operations]
    prob = brm_execute(d, :fit)
    n = LogDensityProblems.dimension(prob)
    q = 0.3 .* randn(Xoshiro(71), 5, n)
    bare_names = BS.param_names(prob.model; include_tp=true, include_gq=false)
    bare = permutedims(reduce(hcat, [
        BS.param_constrain(prob.model, collect(row);
                           include_tp=true, include_gq=false)
        for row in eachrow(q)]))
    msg = try
        brm_powerscale_inputs(sb, bare, bare_names)
        "NO-THROW"
    catch err
        sprint(showerror, err)
    end
    @test occursin("generated quantities", msg)
end
