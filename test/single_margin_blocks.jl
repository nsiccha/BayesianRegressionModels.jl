# test/single_margin_blocks.jl — a random-effect block with one margin has no
# correlation, so SBBRMI declares no correlation factor and no LKJ density for
# it, whichever family emits it: shared `|ID|` blocks (default, configured
# `sd(...)`, known `sd(...) = value`, Student-t, R2D2, cv, centered),
# per-predictor centered, `mm(...)` and stratified `gr(g, by=b)` blocks, and an
# `|ID|` block whose omitted response leaves one active margin. RKBRMI already
# emits none at K = 1 (snag brm-translation-cac32d2f).
#
# Run: julia --startup-file=no --project=test test/single_margin_blocks.jl
# Set BRM_SINGLE_MARGIN_RUNTIME=0 to skip the BridgeStan density probes.

using Test
using Random
using StanBlocks
using BayesianRegressionModels
import BridgeStan
import Distributions
using Distributions: ContinuousUnivariateDistribution, InverseGamma
const BRM = BayesianRegressionModels

# A custom scale-prior family. SBBRMI lowers its configured `sd(...)` prior as a
# value (a StanBlocks `ValueUDF`, which references a module), so a one-margin
# block's form must be derived without copying the family body: docs
# `monster.md`'s `sd(:, VPR) ~ ScaledInvChiScale(...)` failed with "deepcopy of
# Modules not supported" when it was.
struct SMScaledInvChiScale <: ContinuousUnivariateDistribution
    nu::Float64
    s::Float64
end
Distributions.logpdf(d::SMScaledInvChiScale, tau::Real) = tau > 0 ?
    Distributions.logpdf(InverseGamma(d.nu / 2, d.nu * d.s^2 / 2), tau^2) + log(2 * tau) : -Inf
Base.minimum(::SMScaledInvChiScale) = 0.0
Base.maximum(::SMScaledInvChiScale) = Inf
BRM._sb_stan_dist_name(::Type{<:SMScaledInvChiScale}) = :sm_scaled_inv_chi_scale
StanBlocks.@deffun begin
    @lpxf sm_scaled_inv_chi_scale_lpdf(tau::real, nu::real, s::real)::real =
        scaled_inv_chi_square_lpdf(square(tau), nu, s) + log(2.0 * tau)
    sm_scaled_inv_chi_scale_lpdfs(tau::real, nu::real, s::real)::real =
        sm_scaled_inv_chi_scale_lpdf(tau, nu, s)
    sm_scaled_inv_chi_scale_rng(nu::real, s::real)::real =
        sqrt(scaled_inv_chi_square_rng(nu, s))
end

const SM_RUNTIME = get(ENV, "BRM_SINGLE_MARGIN_RUNTIME", "1") != "0"
const SM_N = 60
const SM_DATA = let rng = MersenneTwister(1)
    g = repeat(["a", "b", "c", "d", "e"], SM_N ÷ 5)
    (; y=randn(rng, SM_N), y2=randn(rng, SM_N), x1=Float64.(rand(rng, 0:1, SM_N)),
       x2=Float64.(rand(rng, 0:1, SM_N)), g,
       h=[level in ("a", "b") ? "u" : "v" for level in g],   # strata nest the groups
       m1=rand(rng, ["a", "b", "c"], SM_N), m2=rand(rng, ["a", "b", "c"], SM_N))
end

sm_build(brmi; kwargs...) = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), kwargs...)
sm_factor_lines(sb) = filter(l -> occursin(r"cholesky_factor_corr|lkj_corr", l),
                             split(stan_code(sb), '\n'))
sm_stanc_ok(sb) = StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false).ok
sm_cor_outputs(sb) = [o.name for o in brm_descriptor(sb; name=:m).outputs
                      if o.component === :correlation_cholesky]

const SM_CASES = [
    "zerocorr || with sd = value" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + x2 + (0 + x1 + x2 || g)
        sd(mu, g) = 0.3
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "(1 | g) with sd = value" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | g)
        sd(mu, g) = 0.3
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| default" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| default, cv" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; cv_groups=[:g])),
    "|ID| default, centered" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; centered_groups=[:g])),
    "|ID| sd ~ prior" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sd(:, p) ~ Exponential(2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| sd ~ custom-family prior" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sd(:, p) ~ SMScaledInvChiScale(4.0, log(1.3))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| sd ~ prior, centered" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sd(:, p) ~ Exponential(2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; centered_groups=[:g])),
    "|ID| sd = value, centered" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        sd(:, p) = 0.3
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; centered_groups=[:g])),
    "|ID| Student-t" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        ranef(:, p) ~ TDist(4)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| Student-t, sd ~ prior" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        ranef(:, p) ~ TDist(4)
        sd(:, p) ~ Exponential(2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "|ID| Student-t, sd = value" => ((@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        ranef(:, p) ~ TDist(4)
        sd(:, p) = 0.3
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "centered slope (0 + x1 | g)" => ((@brm SM_DATA begin
        mu ~ 1 + (0 + x1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; centered_groups=[:g])),
    "stratified slope gr(g, by=h)" => ((@brm SM_DATA begin
        mu ~ 1 + (0 + x1 | gr(g, by=h))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "stratified |ID| slope" => ((@brm SM_DATA begin
        mu ~ 1 + (0 + x1 | p | gr(g, by=h))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
    "mm(...) slope" => ((@brm SM_DATA begin
        mu ~ 1 + (0 + x1 | mm(m1, m2))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (;)),
]

@testset "a one-margin block declares no correlation factor" begin
    for (label, (brmi, kwargs)) in SM_CASES
        @testset "$label" begin
            sb = sm_build(brmi; kwargs...)
            @test isempty(sm_factor_lines(sb))
            @test isempty(sm_cor_outputs(sb))
            @test sm_stanc_ok(sb)
            for block in ranef_blocks(sb)
                @test block.n_terms == 1
            end
        end
    end
    # Control: two margins keep their correlation factor and its LKJ density.
    two = sm_build(@brm SM_DATA begin
        mu ~ 1 + x1 + (1 + x1 | p | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test length(sm_factor_lines(two)) == 2
    @test sm_cor_outputs(two) == [:b_p_g_L]
end

@testset "R2D2 over a one-margin |ID| block" begin
    sb = sm_build(@brm SM_DATA begin
        mu ~ 1 + x1 + (1 | p | g)
        effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)
    @test any(b -> b.family === :ranef_correlated_draws_r2d2, ranef_blocks(sb))
    @test isempty(sm_factor_lines(sb))
    @test sm_stanc_ok(sb)
end

@testset "one active margin of a split |ID| block" begin
    builder = @brm begin
        b_loc ~ 1 + (1 | p | g)
        c_loc ~ 1 + (1 | p | g)
        y ~ Normal(b_loc, 0.5)
        y2 ~ Normal(c_loc, 0.5)
    end
    # `y2` omitted: the `c_loc` margin is inactive, `b_loc` the one active margin.
    sb = sm_build(builder((; SM_DATA.y, SM_DATA.g)))
    @test any(b -> b.family === :ranef_correlated_draws_marginal, ranef_blocks(sb))
    @test isempty(filter(l -> occursin("lkj_corr", l), split(stan_code(sb), '\n')))
    @test BRM.stan_data(sb)[:b_p_g_active_L] == ones(1, 1)
    @test sm_stanc_ok(sb)
end

SM_RUNTIME && @testset "BridgeStan: a one-margin |ID| block is the plain block's law" begin
    # `(0 + x1 | p | g)` emits the shared-block family, `(0 + x1 | g)` the
    # one-slope `ranef_slope`: the same half-normal scale and standard draws.
    shared = @brm SM_DATA begin
        mu ~ 1 + (0 + x1 | p | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    plain = @brm SM_DATA begin
        mu ~ 1 + (0 + x1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    folder = mktempdir()
    model(brmi, name) = BRM.stan_instantiate(sm_build(brmi);
        path=joinpath(folder, name * ".stan")).model
    shared_model, plain_model = model(shared, "shared"), model(plain, "plain")
    rename = Dict("r_mu_g_tau.1" => "b_p_g_tau.1",
                  ("r_mu_g_xi.$j" => "b_p_g_z_flat.$j" for j in 1:5)...)
    plain_names = BridgeStan.param_unc_names(plain_model)
    shared_names = BridgeStan.param_unc_names(shared_model)
    @test Set(get(rename, n, n) for n in plain_names) == Set(shared_names)
    rng = MersenneTwister(5)
    for _ in 1:3
        u = Dict(zip(plain_names, randn(rng, length(plain_names))))
        u_plain = [u[n] for n in plain_names]
        renamed = Dict(get(rename, n, n) => v for (n, v) in u)
        u_shared = [renamed[n] for n in shared_names]
        @test BridgeStan.log_density(shared_model, u_shared; propto=false, jacobian=true) ≈
              BridgeStan.log_density(plain_model, u_plain; propto=false, jacobian=true) atol=1e-10
    end
end
