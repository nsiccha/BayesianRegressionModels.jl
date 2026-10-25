# test/level_correlation.jl — correlation across the LEVELS of a grouping
# factor (the intrinsic coregionalisation model): `gr(g; cor=Omega)` and
# `hsgp(x...; by=g, cor=Omega)`, with `Omega ~ LKJCholesky(K, eta)` declared at
# model scope or a known correlation matrix supplied as data, on SBBRMI and
# RKBRMI (src/level_correlation.jl).
#
#     julia --project=test test/level_correlation.jl
#
# Set `BRM_LEVEL_CORRELATION_RUNTIME=0` to skip the compiled-model sets.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using Random
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
import JSON

const BRM = BayesianRegressionModels
const BS = BridgeStan
const RUNTIME = get(ENV, "BRM_LEVEL_CORRELATION_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-level-correlation")
const R = 4

function level_df(; N=36, seed=11)
    rng = Random.MersenneTwister(seed)
    (; t=collect(range(0.0, 10.0; length=N)), g=repeat(1:R, N ÷ R),
       y=randn(rng, N), z=randn(rng, N),
       C=[1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0])
end

const slope = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + t + (0 + t | gr(g; cor=Omega))
    y ~ Normal(mu, 1.0)
end
# The reporter's shape: regional slopes and regional time curves share Omega.
const icm = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + t + (0 + t | gr(g; cor=Omega)) + hsgp(t; k=5, by=g, cor=Omega)
    y ~ Normal(mu, 1.0)
end
const known = @brm begin
    mu ~ 1 + t + (1 + t | gr(g; cor=C))
    y ~ Normal(mu, 1.0)
end
const shared = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + (1 | r | gr(g; cor=Omega))
    nu ~ 1 + (0 + t | r | gr(g; cor=Omega))
    y ~ Normal(mu, 1.0)
    z ~ Normal(nu, 1.0)
end
const addressed = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + (1 | gr(g; cor=Omega))
    sd(:, g) ~ Exponential(2.0)
    y ~ Normal(mu, 1.0)
end

stan(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))
builds(f) = try
    f(); true
catch
    false
end

@testset "SBBRMI emission" begin
    df = level_df()
    code = stan(icm(df))
    @test occursin("cholesky_factor_corr[Omega_n] Omega;", code)
    @test occursin("Omega ~ lkj_corr_cholesky(2.0);", code)
    # the slope block's levels x coefficients effects, left-multiplied; one
    # margin, so no correlation factor of its own (test/single_margin_blocks.jl)
    @test occursin("b_mu__g_g = (Omega * ((diag_matrix(b_mu__g_g_tau) * b_mu__g_g_z)'));", code)
    # the grouped HSGP's weights, left-multiplied by the same factor
    @test occursin("hsgp_t_by_g_beta_level = (Omega * b_hsgpw_t_g);", code)
    @test occursin("rows_dot_product(hsgp_t_by_g_PHI_scaled, hsgp_t_by_g_beta_level[g_idx, :])", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A known correlation is its Cholesky factor as data.
    sb = SBBRMI(known(df); mod=@__MODULE__)
    @test sb.data[:L_levelcor_C] ≈ Matrix(cholesky(Symmetric(df.C)).L)
    @test occursin("b_mu__g_g = (L_levelcor_C * ", BRM.stan_code(sb))
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok

    # One Omega correlates a shared `|ID|` block across its predictors.
    code = stan(shared(df))
    @test count("cholesky_factor_corr[Omega_n] Omega;", code) == 1
    @test occursin("b_r_g = (Omega * ", code)
    @test occursin("r_nu_r_g = (t .* b_r_g[g_idx, 2]);", code)

    # A grouping-column scale prior reaches the level-correlated block.
    code = stan(addressed(df))
    @test occursin("b_mu__g_g = (Omega * ", code)
    @test occursin("b_mu__g_g_tau ~ exponential((1.0 ./ 2.0));", code)
end

@testset "descriptor, plan and description" begin
    df = level_df()
    sb = SBBRMI(icm(df); mod=@__MODULE__)
    d = brm_descriptor(sb)
    omega = brm_output(d, :Omega)
    @test omega.role === :parameter
    @test omega.dims == (R, R)
    @test [b.family for b in BRM.ranef_blocks(d)] == [:ranef_correlated_draws_levelcor]
    targets = [x.target for x in generative_plan(sb).declarations]
    @test first(targets) === :Omega
    md = BRM.brm_description_markdown(BRM.brm_description(d))
    @test occursin("Its levels are correlated with each other by the level correlation `Omega`", md)
    @test occursin("basis weights are correlated across groups by the level correlation `Omega`", md)
    de = BRM.brm_description_markdown(BRM.brm_description(d; texts=BRM.description_texts(:de)))
    @test occursin("Stufenkorrelation `Omega`", de)
end

@testset "wrong shapes are refused" begin
    df = level_df()
    # refused: the correlation needs one row and column per level (src/level_correlation.jl)
    small = @brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "has 4 levels" SBBRMI(small(df); mod=@__MODULE__)
    @test_throws "has 4 levels" RKBRMI(small(df))
    # refused: `cor=` names the correlation of the `by=` grouping
    @test_throws "give `hsgp(x...; by=g, cor=Omega)` a `by=`" @brm(df, begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + hsgp(t; k=5, cor=Omega)
        y ~ Normal(mu, 1.0)
    end)
    # refused: a level correlation is an LKJCholesky factor or a data matrix
    scalar = @brm begin
        Omega ~ Normal(0.0, 1.0)
        mu ~ 1 + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "neither a `Omega ~ LKJCholesky(K, eta)` declaration nor a data matrix" SBBRMI(scalar(df); mod=@__MODULE__)
    # refused: a known correlation must be positive definite
    bad = merge(df, (; C=[1.0 2.0 0.0 0.0; 2.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]))
    @test_throws "must be positive definite" SBBRMI(known(bad); mod=@__MODULE__)
    # refused: terms that coalesce into one block must agree on its level correlation
    mixed = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + (1 | g) + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "name different level correlations" SBBRMI(mixed(df); mod=@__MODULE__)
    # A level correlation is not an independent-level block: VBRMI refuses it.
    @test_throws "VBRMI does not implement" VBRMI(slope(df))
end

@testset "combinations not implemented yet" begin
    df = level_df()
    m = slope(df)
    # Capability gaps, each refused by name today; tracked by BRM todo
    # `2026-10-25T05-58-32-890-0lopk07`.
    @test_broken builds(() -> SBBRMI(m; mod=@__MODULE__, cv_groups=[:g]))
    @test_broken builds(() -> SBBRMI(m; mod=@__MODULE__, centered_groups=[:g]))
    @test_broken builds(() -> reprocess(SBBRMI(m; mod=@__MODULE__), df; resample_groups=[:g]))
    @test_broken builds(() -> reprocess(RKBRMI(m), df; resample_groups=[:g]))
    @test_broken builds(() -> brm_nlme_view(RKBRMI(m)))
    stratified = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + (0 + t | gr(g; by=b, cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    @test_broken builds(() -> SBBRMI(stratified(merge(df, (; b=repeat([1, 2], 18)))); mod=@__MODULE__))
end

# A raw-Stan intrinsic coregionalisation model written by hand: slopes
# `e = Omega * (tau * zb)` and per-basis region weights `Omega * Z`, as the
# reporter's StanBlocks model writes them. Its density must equal the
# BRM-emitted program's at every point.
const ICM_REFERENCE = """
data { int N; int R; int M; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower; }
parameters {
  cholesky_factor_corr[R] Omega;
  vector<lower=0>[1] tau;
  vector[R] zb;
  vector[2] beta;
  vector[M * R] zw;
  real<lower=rho_lower> rho;
  real<lower=0> sf;
}
model {
  vector[R] e = Omega * (tau[1] * zb);
  matrix[R, M] W = Omega * to_matrix(zw, M, R)';
  vector[M] spd = sf * sqrt(rho * sqrt(2 * pi())) * exp(-0.25 * square(rho) * omega2);
  vector[N] f;
  for (i in 1:N) f[i] = dot_product(PHI[i], W[g[i]]' .* spd);
  Omega ~ lkj_corr_cholesky(2.0);
  tau ~ std_normal();
  zb ~ std_normal();
  beta ~ std_normal();
  zw ~ std_normal();
  rho ~ lognormal(0, 1);
  sf ~ lognormal(0, 1);
  y ~ normal(beta[1] + beta[2] * t + t .* e[g] + f, 1);
}
"""

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

# BRM's ICM coordinates under the reference's names (M basis functions).
function icm_reference_name(n, M)
    startswith(n, "Omega.") && return n
    n == "hsgp_t_by_g_rho_iso" && return "rho"
    n == "hsgp_t_by_g_sigma" && return "sf"
    for (a, r) in ("b_mu__g_g_tau." => "tau.", "b_mu__g_g_z_flat." => "zb.",
                   "pop_mu_beta_pop." => "beta.", "zflat_hsgpw_t_g." => "zw.")
        startswith(n, a) && return r * n[length(a)+1:end]
    end
    error("unmapped BRM coordinate $n")
end

@testset "SBBRMI density equals a hand-written ICM program" begin
    if RUNTIME
        df = level_df()
        sb = SBBRMI(icm(df); mod=@__MODULE__)
        problem = compiled(sb, "icm")
        names_brm = String.(BS.param_unc_names(problem.model))
        PHI = sb.data[:PHI_hsgp_t_by_g]
        M = size(PHI, 2)
        mkpath(CACHE)
        path = joinpath(CACHE, "icm_reference_" * string(hash(ICM_REFERENCE)) * ".stan")
        isfile(path) || write(path, ICM_REFERENCE)
        reference = BS.StanModel(path, JSON.json(Dict("N"=>length(df.t), "R"=>R, "M"=>M,
            "t"=>df.t, "y"=>df.y, "g"=>df.g, "PHI"=>[PHI[i, :] for i in axes(PHI, 1)],
            "omega2"=>vec(sb.data[:omega2_hsgp_t_by_g]),
            "rho_lower"=>sb.data[:rho_lower_hsgp_t_by_g])))
        names_ref = String.(BS.param_unc_names(reference))
        @test length(names_ref) == length(names_brm)
        perm = [only(findall(==(icm_reference_name(n, M)), names_ref)) for n in names_brm]
        rng = Random.MersenneTwister(7)
        for _ in 1:4
            u = 0.7 .* randn(rng, length(names_brm))
            v = zeros(length(names_ref)); v[perm] .= u
            gu, gv = similar(u), similar(v)
            lu, _ = BS.log_density_gradient!(problem.model, u, gu; propto=false, jacobian=true)
            lv, _ = BS.log_density_gradient!(reference, v, gv; propto=false, jacobian=true)
            @test lu ≈ lv atol=1e-10 rtol=1e-12
            @test gu ≈ gv[perm] atol=1e-10 rtol=1e-10
        end
    else
        @info "Skipping the compiled ICM reference (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

function check_rk_sb(rk, stan, perm, points)
    problem = rk_problem(rk)
    for u in points
        stan_u = zeros(length(u)); stan_u[perm] .= u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        g = similar(stan_u)
        stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                propto=false, jacobian=true)
        @test value ≈ stan_value atol=1e-10 rtol=1e-12
        @test gradient ≈ g[perm] atol=1e-10 rtol=1e-10
    end
end

points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

@testset "RKBRMI equals SBBRMI" begin
    if RUNTIME
        df = level_df()
        # Level-correlated blocks pair through the semantic coordinate transport.
        for (tag, model) in ("slope" => slope, "known" => known)
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
            stan = compiled(sb, tag)
            transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
            @test length(transport) == length(coordinate_names(rk.model.layout))
            n = length(transport)
            problem = rk_problem(rk)
            for u in points(n)
                stan_u = brm_rk_point_to_stan(transport, u)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                g = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                        propto=false, jacobian=true)
                @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=1e-10 rtol=1e-12
                @test gradient ≈ brm_stan_gradient_to_rk(transport, u, g) atol=1e-10 rtol=1e-10
                checked = brm_check_coordinate_transport(transport, rk, stan.model, u)
                @test checked.max_error <= 1e-12
            end
        end
        # The grouped HSGP is not paired by the transport yet; map its names.
        m = icm(df)
        rk = RKBRMI(m)
        sb = SBBRMI(m; mod=@__MODULE__)
        stan = compiled(sb, "icm")
        rk_names = String.(coordinate_names(rk.model.layout))
        stan_names = BS.param_unc_names(stan.model)
        M = size(sb.data[:PHI_hsgp_t_by_g], 2)
        function stan_name(n)
            startswith(n, "Omega.") && return n
            n == "hsgp_t.rho" && return "hsgp_t_by_g_rho_iso"
            n == "hsgp_t.sigma" && return "hsgp_t_by_g_sigma"
            n == "b_mu__g_g.tau.1" && return "b_mu__g_g_tau.1"
            x = match(r"^b_mu__g_g\.z\.(\d+)\.1$", n)
            x === nothing || return "b_mu__g_g_z_flat.$(x[1])"
            x = match(r"^hsgp_t\.z\.(\d+)\.(\d+)$", n)
            x === nothing || return "zflat_hsgpw_t_g.$((parse(Int, x[1]) - 1) * M + parse(Int, x[2]))"
            x = match(r"^pop_mu\.beta_pop\.(\d+)$", n)
            x === nothing || return "pop_mu_beta_pop.$(x[1])"
            error("unmapped RK coordinate $n")
        end
        perm = [only(findall(==(stan_name(n)), stan_names)) for n in rk_names]
        @test sort(perm) == collect(eachindex(stan_names))
        check_rk_sb(rk, stan, perm, points(length(rk_names)))
    else
        @info "Skipping RK/SB parity (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

@testset "brm_ranef_draws reports the level correlation" begin
    if RUNTIME
        df = level_df()
        for (tag, model) in ("slope" => slope, "known" => known)
            sb = SBBRMI(model(df); mod=@__MODULE__, total_groups=())
            stan = compiled(sb, tag)
            names = BS.param_names(stan.model; include_tp=true)
            rng = Random.MersenneTwister(3)
            unc = [0.4 .* randn(rng, BS.param_unc_num(stan.model)) for _ in 1:3]
            draws = reduce(vcat, permutedims(BS.param_constrain(stan.model, u;
                include_tp=true)) for u in unc)
            block = only(brm_ranef_draws(brm_descriptor(sb), draws, names))
            @test size(block.level_cor) == (3, R, R)
            # Capability gap: per-level centering of correlated levels
            # (BRM todo `2026-10-25T05-58-32-890-0lopk07`).
            @test_broken builds(() -> BRM.adaptive_centering_blocks(sb,
                BS.param_unc_names(stan.model)))
            for i in 1:3
                expected = if tag == "slope"
                    L = reshape(draws[i, findall(startswith("Omega."), names)], R, R)
                    L * L'
                else
                    df.C
                end
                @test block.level_cor[i, :, :] ≈ expected
            end
        end
    else
        @test true
    end
end

@testset "replay keeps the fitted levels and their correlation" begin
    df = level_df()
    sb = SBBRMI(icm(df); mod=@__MODULE__)
    @test BRM.stan_code(reprocess(sb, df)) == BRM.stan_code(sb)
    subset = (; t=df.t[1:8], g=df.g[1:8], y=df.y[1:8])
    @test occursin("hsgp_t_by_g_beta_level = (Omega * ", BRM.stan_code(reprocess(sb, subset)))
    @test RKBRMI(icm(df)) isa RKBRMI
    @test reprocess(RKBRMI(icm(df)), df) isa RKBRMI
end

println("level_correlation.jl: all testsets passed")
