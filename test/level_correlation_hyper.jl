# test/level_correlation_hyper.jl — per-level amplitudes and length scales of a
# grouped HSGP whose levels are correlated: the hyper-predictors
# `log(sd(hsgp(x))) ~ 1 + (1 | g)` and `log(length_scale(hsgp(x))) ~ 1 + (1 | g)`
# on `hsgp(x...; by=g, cor=Omega)`, `cor=multigroup(Omega, tau)` and
# `cor=multigroup((x, y), lambda; k)`, on SBBRMI and RKBRMI (src/level_correlation.jl, `_brm_check_hyper_level_correlation`).
# Basis function `j` draws the level weights from
# `MVN(0, diag(s_j) C_j diag(s_j))`, `s_j[g] = sqrt(S_g(omega_j))`.
#
#     julia --project=test test/level_correlation_hyper.jl
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
const CACHE = joinpath(tempdir(), "brm-level-correlation-hyper")
const R = 4

function level_df(; N=36, seed=11)
    rng = Random.MersenneTwister(seed)
    (; t=collect(range(0.0, 10.0; length=N)), u=collect(range(-1.0, 1.0; length=N)),
       g=repeat(1:R, N ÷ R), y=randn(rng, N),
       # the levels' coordinates, for the multigroup coherence over coordinates
       sx=[10.0, 40.0, 70.0, 95.0], sy=[20.0, 130.0, 60.0, 100.0],
       C=[1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0])
end

# Per-level amplitudes, one length scale and the level correlation shared.
const icm_sd = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=Omega)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
# Per-level amplitudes and length scales under a constant level correlation.
const icm_both = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=Omega)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    log(length_scale(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
const icm_known = @brm begin
    mu ~ 1 + hsgp(t; k=5, by=g, cor=C)
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
# The reporter's shape: one two-dimensional field per level.
const icm_2d = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 0 + hsgp(t, u; k=4, by=g, cor=Omega)
    log(sd(hsgp(t, u))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
const mg_sd = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup(Omega, tau))
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
const mg_2d = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    mu ~ 0 + hsgp(t, u; k=4, by=g, cor=multigroup(Omega, 0.7))
    log(sd(hsgp(t, u))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end

# The multigroup coherence over the levels' coordinates, with a declared
# spatial scale and with the term's own.
const mgc_sd = @brm begin
    lambda ~ LogNormal(5.7, 0.3; lower=230.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), lambda; k=6))
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end
const mgd_sd = @brm begin
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup((sx, sy); k=6))
    log(sd(hsgp(t))) ~ 1 + (1 | g)
    y ~ Normal(mu, 1.0)
end

stan(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))

@testset "SBBRMI emission" begin
    df = level_df()
    for model in (icm_sd, icm_both, icm_known, icm_2d, mg_sd, mg_2d, mgc_sd, mgd_sd)
        @test StanBlocks.stanc_check(stan(model(df)); warn_pedantic=false).ok
    end
    code = stan(icm_sd(df))
    @test occursin("hsgp_t_by_g_beta_level = (Omega * b_hsgpw_t_g);", code)
    @test occursin("rows_dot_product(hsgp_t_by_g_S, hsgp_t_by_g_beta_level[g_idx, :])", code)
    code = stan(mg_sd(df))
    @test occursin("hsgp_t_by_g_beta_level = brm_hsgp_multigroup_weights(b_hsgpw_t_g, " *
                   "omega2_hsgp_t_by_g, hsgp_t_by_g_rho, Omega, tau);", code)
    # The emitted call spans lines; compare without whitespace.
    code = replace(stan(mgc_sd(df)), r"\s+" => "")
    @test occursin("hsgp_t_by_g_beta_level=brm_hsgp_multigroup_coordinate_weights(" *
                   "b_hsgpw_t_g,omega2_hsgp_t_by_g,hsgp_t_by_g_rho,", code)
    @test occursin("rows_dot_product(hsgp_t_by_g_S,hsgp_t_by_g_beta_level[g_idx,:])", code)
    @test occursin("hsgp_t_by_g_lambda~lognormal(", replace(stan(mgd_sd(df)), r"\s+" => ""))
    sb = SBBRMI(icm_known(df); mod=@__MODULE__)
    @test sb.data[:L_levelcor_C] ≈ Matrix(cholesky(Symmetric(df.C)).L)
    @test occursin("L_levelcor_C * b_hsgpw_t_g", BRM.stan_code(sb))
end

@testset "a per-level length scale under the multigroup coherence is refused" begin
    df = level_df()
    # refused: the multigroup coherence exp(-(rho omega)^2 / 2 * D2) is a
    # function of the one length scale every level shares (the kernel adopted
    # on decision `multigroup_gp:bench/decisions/2026-10-25T18-22-49-687-1v8g79k`);
    # a per-level length scale leaves it undefined.
    mg_rho = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        tau ~ LogNormal(0.0, 1.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, tau))
        log(length_scale(hsgp(t))) ~ 1 + (1 | g)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "with a hyper-predicted length scale is refused" SBBRMI(mg_rho(df); mod=@__MODULE__)
    @test_throws "with a hyper-predicted length scale is refused" RKBRMI(mg_rho(df))
    # refused: over known coordinates the range lambda / (rho omega) reads the
    # one shared length scale the same way.
    mgc_rho = @brm begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy); k=6))
        log(length_scale(hsgp(t))) ~ 1 + (1 | g)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "with a hyper-predicted length scale is refused" SBBRMI(mgc_rho(df); mod=@__MODULE__)
    @test_throws "with a hyper-predicted length scale is refused" RKBRMI(mgc_rho(df))
end

@testset "descriptor, description and replay" begin
    df = level_df()
    for model in (icm_both, mg_sd, mgd_sd)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        d = brm_descriptor(sb)
        @test brm_output(d, :Omega).dims == (R, R)
        md = BRM.brm_description_markdown(BRM.brm_description(d))
        @test occursin("`Omega`", md)
        @test BRM.stan_code(reprocess(sb, df)) == BRM.stan_code(sb)
        @test reprocess(RKBRMI(model(df)), df) isa RKBRMI
    end
end

# Raw-Stan programs written by hand. Level weights `W = Omega * Z` (ICM) or
# `W[:, j] = chol(C_j + jitter I) * Z[:, j]` (multigroup); level `k` scales
# basis function `j` by its own spectral density, from
# `sf_k = exp(b0_sigma + sd_sigma * z_sigma[k])` and
# `rho_k = fmax(exp(b0_rho + sd_rho * z_rho[k]), rho_lower)` (or one shared rho).
const ICM_HYPER_REFERENCE = """
data { int N; int R; int M; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower; }
parameters {
  cholesky_factor_corr[R] Omega;
  vector[2] beta;
  vector[M * R] zw;
  real b0_sigma; real<lower=0> sd_sigma; vector[R] z_sigma;
  real b0_rho; real<lower=0> sd_rho; vector[R] z_rho;
}
model {
  matrix[R, M] W = Omega * to_matrix(zw, M, R)';
  vector[R] sf = exp(b0_sigma + sd_sigma * z_sigma);
  vector[R] rho;
  for (k in 1:R) rho[k] = fmax(exp(b0_rho + sd_rho * z_rho[k]), rho_lower);
  vector[N] f;
  for (i in 1:N) {
    int k = g[i];
    vector[M] spd = sf[k] * sqrt(rho[k] * sqrt(2 * pi())) * exp(-0.25 * square(rho[k]) * omega2);
    f[i] = dot_product(PHI[i], W[k]' .* spd);
  }
  Omega ~ lkj_corr_cholesky(2.0);
  beta ~ std_normal();
  zw ~ std_normal();
  b0_sigma ~ std_normal(); sd_sigma ~ std_normal(); z_sigma ~ std_normal();
  b0_rho ~ std_normal(); sd_rho ~ std_normal(); z_rho ~ std_normal();
  y ~ normal(beta[1] + beta[2] * t + f, 1);
}
"""
const MG_HYPER_REFERENCE = """
data { int N; int R; int M; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower; }
parameters {
  cholesky_factor_corr[R] Omega;
  real<lower=0> tau;
  vector[2] beta;
  vector[M * R] zw;
  real<lower=rho_lower> rho;
  real b0_sigma; real<lower=0> sd_sigma; vector[R] z_sigma;
}
model {
  matrix[R, R] Om = multiply_lower_tri_self_transpose(Omega);
  matrix[R, M] Z = to_matrix(zw, M, R)';
  matrix[R, M] W;
  for (j in 1:M) {
    matrix[R, R] C;
    for (a in 1:R) for (b in 1:R)
      C[a, b] = a == b ? 1 + 1e-6 : exp(-square(rho) * omega2[j] * tau * (1 - Om[a, b]));
    W[:, j] = cholesky_decompose(C) * Z[:, j];
  }
  vector[R] sf = exp(b0_sigma + sd_sigma * z_sigma);
  vector[M] spd = sqrt(rho * sqrt(2 * pi())) * exp(-0.25 * square(rho) * omega2);
  vector[N] f;
  for (i in 1:N) f[i] = dot_product(PHI[i], W[g[i]]' .* (sf[g[i]] * spd));
  Omega ~ lkj_corr_cholesky(2.0);
  tau ~ lognormal(0, 1);
  beta ~ std_normal();
  zw ~ std_normal();
  rho ~ lognormal(0, 1);
  b0_sigma ~ std_normal(); sd_sigma ~ std_normal(); z_sigma ~ std_normal();
  y ~ normal(beta[1] + beta[2] * t + f, 1);
}
"""

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

# BRM's SB coordinates under the references' names.
function reference_name(n)
    (startswith(n, "Omega.") || n == "tau") && return n
    n == "hsgp_t_by_g_rho_iso" && return "rho"
    for (a, r) in ("pop_mu_beta_pop." => "beta.", "zflat_hsgpw_t_g." => "zw.",
                   "hsgp_t_by_g_z_sigma." => "z_sigma.", "hsgp_t_by_g_z_rho." => "z_rho.")
        startswith(n, a) && return r * n[length(a)+1:end]
    end
    for hyper in ("beta0_sigma" => "b0_sigma", "sd_sigma" => "sd_sigma",
                  "beta0_rho" => "b0_rho", "sd_rho" => "sd_rho")
        n == "hsgp_t_by_g_" * first(hyper) && return last(hyper)
    end
    error("unmapped BRM coordinate $n")
end

@testset "SBBRMI density equals a hand-written program" begin
    if RUNTIME
        df = level_df()
        for (tag, model, source) in (("icm_both", icm_both, ICM_HYPER_REFERENCE),
                                     ("mg_sd", mg_sd, MG_HYPER_REFERENCE))
            sb = SBBRMI(model(df); mod=@__MODULE__)
            problem = compiled(sb, tag)
            names_brm = String.(BS.param_unc_names(problem.model))
            PHI = sb.data[:PHI_hsgp_t_by_g]
            M = size(PHI, 2)
            path = joinpath(CACHE, tag * "_reference_" * string(hash(source)) * ".stan")
            isfile(path) || write(path, source)
            reference = BS.StanModel(path, JSON.json(Dict("N"=>length(df.t), "R"=>R, "M"=>M,
                "t"=>df.t, "y"=>df.y, "g"=>df.g, "PHI"=>[PHI[i, :] for i in axes(PHI, 1)],
                "omega2"=>vec(sb.data[:omega2_hsgp_t_by_g]),
                "rho_lower"=>sb.data[:rho_lower_hsgp_t_by_g])))
            names_ref = String.(BS.param_unc_names(reference))
            @test length(names_ref) == length(names_brm)
            perm = [only(findall(==(reference_name(n)), names_ref)) for n in names_brm]
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
        end
    else
        @info "Skipping the compiled references (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

# RK's coordinate `n` under SB's name, for a term over the axes `axes`
# ("t" or "t_u") with `M` basis functions.
function stan_name(n, axes, M)
    (startswith(n, "Omega.") || n in ("tau", "lambda")) && return n
    rk, sb = "hsgp_" * axes * ".", "hsgp_" * axes * "_by_g_"
    n == rk * "rho" && return sb * "rho_iso"
    n == rk * "lambda" && return sb * "lambda"
    n == "mu_Intercept" && return "pop_mu_beta_pop.1"
    for hyper in ("rho", "sigma")
        n == rk * hyper * "_Intercept" && return sb * "beta0_" * hyper
        n == rk * hyper * "_sd" && return sb * "sd_" * hyper
        x = match(Regex("^" * replace(rk, "." => "\\.") * hyper * "_z\\.(\\d+)\$"), n)
        x === nothing || return sb * "z_" * hyper * "." * x[1]
    end
    x = match(Regex("^" * replace(rk, "." => "\\.") * "z\\.(\\d+)\\.(\\d+)\$"), n)
    x === nothing || return "zflat_hsgpw_" * axes * "_g.$((parse(Int, x[1]) - 1) * M + parse(Int, x[2]))"
    x = match(r"^pop_mu\.beta_pop\.(\d+)$", n)
    x === nothing || return "pop_mu_beta_pop.$(x[1])"
    error("unmapped RK coordinate $n")
end

@testset "RKBRMI equals SBBRMI" begin
    if RUNTIME
        df = level_df()
        for (tag, model, axes) in (("icm_sd", icm_sd, "t"), ("icm_both", icm_both, "t"),
                                   ("icm_known", icm_known, "t"), ("icm_2d", icm_2d, "t_u"),
                                   ("mg_sd", mg_sd, "t"), ("mg_2d", mg_2d, "t_u"),
                                   ("mgc_sd", mgc_sd, "t"), ("mgd_sd", mgd_sd, "t"))
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__)
            stan = compiled(sb, tag)
            rk_names = String.(coordinate_names(rk.model.layout))
            stan_names = BS.param_unc_names(stan.model)
            M = size(sb.data[Symbol(:PHI_hsgp_, axes, :_by_g)], 2)
            perm = [only(findall(==(stan_name(n, axes, M)), stan_names)) for n in rk_names]
            @test sort(perm) == collect(eachindex(stan_names))
            problem = rk_problem(rk)
            for u in points(length(rk_names))
                stan_u = zeros(length(u)); stan_u[perm] .= u
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                g = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                        propto=false, jacobian=true)
                @test value ≈ stan_value atol=1e-10 rtol=1e-12
                @test gradient ≈ g[perm] atol=1e-10 rtol=1e-10
            end
        end
    else
        @info "Skipping RK/SB parity (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

println("level_correlation_hyper.jl: all testsets passed")
