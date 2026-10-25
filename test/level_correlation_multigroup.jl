# test/level_correlation_multigroup.jl — the multigroup coherence of a grouped
# HSGP, `hsgp(x...; by=g, cor=multigroup(Omega, tau))` (src/level_correlation.jl):
# basis frequency `omega` correlates the levels of `g` by
# `C = exp(-(rho omega)^2 / 2 * D2)`, `D2 = 2 tau (1 - Omega)`, on SBBRMI and
# RKBRMI. Spelling: decision `0w54lkq`.
#
#     julia --project=test test/level_correlation_multigroup.jl
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
const CACHE = joinpath(tempdir(), "brm-level-correlation-multigroup")
const R = 4
const JITTER = 1e-6

function level_df(; N=36, seed=11)
    rng = Random.MersenneTwister(seed)
    (; t=collect(range(0.0, 10.0; length=N)), g=repeat(1:R, N ÷ R),
       u=collect(range(-1.0, 1.0; length=N)), y=randn(rng, N),
       C=[1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0])
end

const mg = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup(Omega, tau))
    y ~ Normal(mu, 1.0)
end
# A known correlation and a constant Gram scale.
const known = @brm begin
    mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(C, 0.7))
    y ~ Normal(mu, 1.0)
end
const aniso = @brm begin
    Omega ~ LKJCholesky(4, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    mu ~ 1 + hsgp(t, u; k=3, by=g, iso=false, cor=multigroup(Omega, tau))
    y ~ Normal(mu, 1.0)
end

stan(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))

function random_factor(rng, n)
    A = randn(rng, n, n)
    S = A * A' + n * I
    D = Diagonal(1 ./ sqrt.(diag(S)))
    Matrix(cholesky(Symmetric(D * S * D)).L)
end

# Basis column `b`'s factor, read off unit standardized weights.
function weight_factor(omega2, rho, L, tau, b)
    M = size(omega2, 1)
    hcat((brm_hsgp_multigroup_weights(repeat(Matrix(1.0I, R, R)[:, k], 1, M),
                                      omega2, rho, L, tau)[:, b] for k in 1:R)...)
end

@testset "the coherence is exp(-(rho omega)^2 / 2 * D2)" begin
    rng = Random.MersenneTwister(5)
    L = random_factor(rng, R)
    Omega = L * L'
    tau = 0.8
    D2 = 2tau .* (1 .- Omega)
    # one axis (a frequency per mode) and two axes, isotropic and not
    for (omega2, rho, exponent) in (
            ([0.1, 0.9, 2.5], 1.3, (o, b) -> 1.3^2 * o[b]),
            ([0.1 0.4; 0.9 0.2; 2.5 1.0], 0.7, (o, b) -> 0.7^2 * sum(o[b, :])),
            ([0.1 0.4; 0.9 0.2; 2.5 1.0], [0.7, 1.4], (o, b) -> 0.7^2 * o[b, 1] + 1.4^2 * o[b, 2]))
        for b in axes(omega2, 1)
            F = weight_factor(omega2, rho, L, tau, b)
            @test istril(F)
            C = exp.(-exponent(omega2, b) / 2 .* D2) + JITTER * I
            @test F * F' ≈ C atol=1e-12
        end
        z = randn(rng, R, size(omega2, 1))
        W = brm_hsgp_multigroup_weights(z, omega2, rho, L, tau)
        @test all(W[:, b] ≈ weight_factor(omega2, rho, L, tau, b) * z[:, b] for b in axes(z, 2))
    end
    # Zero frequency shares everything: the coherence is all ones.
    F = weight_factor([0.0], 1.0, L, tau, 1)
    @test F * F' ≈ ones(R, R) + JITTER * I atol=1e-12
end

@testset "lag-0 and lagged correlations match the exact kernel" begin
    # The HSGP covariance of the group functions approaches the multigroup
    # kernel sf² (1 + D2)^(-1/2) exp(-dx² / (2 ell² (1 + D2))) as k grows.
    df = level_df(N=40)
    dense = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        tau ~ LogNormal(0.0, 1.0)
        mu ~ 1 + hsgp(t; k=40, by=g, cor=multigroup(Omega, tau))
        y ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(dense(df); mod=@__MODULE__)
    PHI = sb.data[:PHI_hsgp_t_by_g]
    omega2 = sb.data[:omega2_hsgp_t_by_g]
    rng = Random.MersenneTwister(9)
    L = random_factor(rng, R)
    tau, ell = 0.5, 1.0
    D2 = 2tau .* (1 .- L * L')
    S = brm_hsgp_sqrt_spd(omega2, 1.0, [ell]) .^ 2
    factors = [weight_factor(omega2, [ell], L, tau, b) for b in axes(omega2, 1)]
    covariance(i, j) = sum(PHI[i, b] * PHI[j, b] * S[b] .* (factors[b] * factors[b]')
                           for b in axes(omega2, 1))
    exact(dx) = (1 .+ D2) .^ -0.5 .* exp.(-dx^2 ./ (2ell^2 .* (1 .+ D2)))
    i = 20                                   # interior rows, t near 5
    for j in (20, 21, 23)
        @test covariance(i, j) ≈ exact(df.t[j] - df.t[i]) atol=1e-3
    end
end

@testset "SBBRMI emission" begin
    df = level_df()
    code = stan(mg(df))
    @test occursin("cholesky_factor_corr[Omega_n] Omega;", code)
    @test occursin("real<lower=0.0> tau;", code)
    @test occursin("hsgp_t_by_g_beta_level = brm_hsgp_multigroup_weights(b_hsgpw_t_g, " *
                   "omega2_hsgp_t_by_g, hsgp_t_by_g_rho, Omega, tau);", code)
    @test occursin("rows_dot_product(hsgp_t_by_g_PHI_scaled, hsgp_t_by_g_beta_level[g_idx, :])", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # A known correlation is its Cholesky factor as data; a constant scale is inline.
    sb = SBBRMI(known(df); mod=@__MODULE__)
    @test sb.data[:L_levelcor_C] ≈ Matrix(cholesky(Symmetric(df.C)).L)
    @test occursin("brm_hsgp_multigroup_weights(b_hsgpw_t_g, omega2_hsgp_t_by_g, " *
                   "hsgp_t_by_g_rho, L_levelcor_C, 0.7)", BRM.stan_code(sb))
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    # One length scale per axis.
    code = stan(aniso(df))
    @test occursin("brm_hsgp_multigroup_weights(b_hsgpw_t_u_g, omega2_hsgp_t_u_by_g, " *
                   "hsgp_t_u_by_g_rho, Omega, tau)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "descriptor and description" begin
    df = level_df()
    d = brm_descriptor(SBBRMI(mg(df); mod=@__MODULE__))
    @test brm_output(d, :Omega).dims == (R, R)
    @test brm_output(d, :tau).role === :parameter
    md = BRM.brm_description_markdown(BRM.brm_description(d))
    @test occursin("multigroup coherence of the level correlation `Omega` and the Gram scale `tau`", md)
    @test !occursin("the same at every frequency", md)
    de = BRM.brm_description_markdown(BRM.brm_description(d; texts=BRM.description_texts(:de)))
    @test occursin("Multigroup-Kohärenz der Stufenkorrelation `Omega`", de)
end

@testset "wrong shapes are refused" begin
    df = level_df()
    # refused: a random-effect block has no frequencies for the coherence to
    # depend on (decision `0w54lkq`, scope)
    block = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        tau ~ LogNormal(0.0, 1.0)
        mu ~ 1 + (0 + t | gr(g; cor=multigroup(Omega, tau)))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "only a grouped `hsgp(...; by=..., cor=multigroup(...))` has" SBBRMI(block(df); mod=@__MODULE__)
    @test_throws "only a grouped `hsgp(...; by=..., cor=multigroup(...))` has" RKBRMI(block(df))
    # refused: the Gram distance 2 tau (1 - Omega) needs tau > 0
    negative = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, -0.5))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "needs a positive finite Gram scale" SBBRMI(negative(df); mod=@__MODULE__)
    # refused: `multigroup` takes exactly a correlation and a scale
    three = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, 0.5, 0.5))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "takes `multigroup(Omega, tau)`" SBBRMI(three(df); mod=@__MODULE__)
    # refused: a data column is not one Gram scale
    column = @brm begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, u))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "must be one positive number" SBBRMI(column(df); mod=@__MODULE__)
end

# A raw-Stan multigroup HSGP written by hand: per-basis weights
# `chol(C_j + jitter I) * Z[:, j]` with `C_j = exp(-rho^2 omega2_j tau (1 - Omega))`.
const MG_REFERENCE = """
data { int N; int R; int M; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower; }
parameters {
  cholesky_factor_corr[R] Omega;
  real<lower=0> tau;
  vector[2] beta;
  vector[M * R] zw;
  real<lower=rho_lower> rho;
  real<lower=0> sf;
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
  vector[M] spd = sf * sqrt(rho * sqrt(2 * pi())) * exp(-0.25 * square(rho) * omega2);
  vector[N] f;
  for (i in 1:N) f[i] = dot_product(PHI[i], W[g[i]]' .* spd);
  Omega ~ lkj_corr_cholesky(2.0);
  tau ~ lognormal(0, 1);
  beta ~ std_normal();
  zw ~ std_normal();
  rho ~ lognormal(0, 1);
  sf ~ lognormal(0, 1);
  y ~ normal(beta[1] + beta[2] * t + f, 1);
}
"""

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

function reference_name(n)
    (startswith(n, "Omega.") || n == "tau") && return n
    n == "hsgp_t_by_g_rho_iso" && return "rho"
    n == "hsgp_t_by_g_sigma" && return "sf"
    for (a, r) in ("pop_mu_beta_pop." => "beta.", "zflat_hsgpw_t_g." => "zw.")
        startswith(n, a) && return r * n[length(a)+1:end]
    end
    error("unmapped BRM coordinate $n")
end

@testset "SBBRMI density equals a hand-written multigroup program" begin
    if RUNTIME
        df = level_df()
        sb = SBBRMI(mg(df); mod=@__MODULE__)
        problem = compiled(sb, "mg")
        names_brm = String.(BS.param_unc_names(problem.model))
        PHI = sb.data[:PHI_hsgp_t_by_g]
        M = size(PHI, 2)
        mkpath(CACHE)
        path = joinpath(CACHE, "mg_reference_" * string(hash(MG_REFERENCE)) * ".stan")
        isfile(path) || write(path, MG_REFERENCE)
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
    else
        @info "Skipping the compiled multigroup reference (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

@testset "RKBRMI equals SBBRMI" begin
    if RUNTIME
        df = level_df()
        for (tag, model) in ("mg" => mg, "known" => known)
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__)
            stan = compiled(sb, tag)
            rk_names = String.(coordinate_names(rk.model.layout))
            stan_names = BS.param_unc_names(stan.model)
            M = size(sb.data[:PHI_hsgp_t_by_g], 2)
            function stan_name(n)
                (startswith(n, "Omega.") || n == "tau") && return n
                n == "hsgp_t.rho" && return "hsgp_t_by_g_rho_iso"
                n == "hsgp_t.sigma" && return "hsgp_t_by_g_sigma"
                n == "mu_Intercept" && return "pop_mu_beta_pop.1"
                x = match(r"^hsgp_t\.z\.(\d+)\.(\d+)$", n)
                x === nothing || return "zflat_hsgpw_t_g.$((parse(Int, x[1]) - 1) * M + parse(Int, x[2]))"
                x = match(r"^pop_mu\.beta_pop\.(\d+)$", n)
                x === nothing || return "pop_mu_beta_pop.$(x[1])"
                error("unmapped RK coordinate $n")
            end
            perm = [only(findall(==(stan_name(n)), stan_names)) for n in rk_names]
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
