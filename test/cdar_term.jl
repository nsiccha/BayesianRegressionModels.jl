# test/cdar_term.jl — semantic and executable contract for the grouped,
# correlated, damped random-walk deviation term `cdar(step; by=group, cor=C)`.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra: LinearAlgebra, I, cholesky, Symmetric
using Distributions: Normal
import Random
import StanBlocks.stan: transpiles

const CDAR_RUNTIME = get(ENV, "BRM_CDAR_RUNTIME", "1") != "0"
const CDAR_CACHE = joinpath(tempdir(), "brm-cdar-term")
const BS = StanBlocks.BridgeStan

# two groups, two steps, one row per (group, step); `C` as a data field
cdar_df(; C=[1.0 0.0; 0.0 1.0], steps=2) = (;
    week=repeat(1:steps; inner=2), patch=repeat(["a", "b"]; outer=steps),
    y=zeros(2 * steps), C=C)

function cdar_model(df=cdar_df())
    @brm df begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 1.0)
    end
end

stan(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)

@testset "cdar is a direct grouped deviation term: one intercept, sigma, rho, P·W innovations" begin
    df = cdar_df()
    walk = cdar_model(df)
    @test popcoefnames(walk, :mu) == [:Intercept]
    sb = SBBRMI(walk; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("correlated_damped_walk", code)
    @test occursin("cdar_mu_week_sigma", code)
    @test occursin("cdar_mu_week_rho", code)
    @test occursin("cdar_mu_week_eta", code)
    @test sb.data[:cdar_mu_week_n_groups] == 2
    @test sb.data[:cdar_mu_week_n_steps] == 2
    @test sb.data[:cdar_mu_week_L] == [1.0 0.0; 0.0 1.0]
    @test sb.data[:cdar_mu_week_group_idx] == [1, 2, 1, 2]
    @test sb.data[:cdar_mu_week_step_idx] == [1, 1, 2, 2]
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "cdar ships the Cholesky factor of cor, from a data field or a literal" begin
    C = [1.0 0.5; 0.5 1.0]
    sb = SBBRMI(cdar_model(cdar_df(; C)); mod=@__MODULE__)
    @test sb.data[:cdar_mu_week_L] ≈ Matrix(cholesky(Symmetric(C)).L)
    df = cdar_df()
    literal = @brm df begin
        mu ~ 1 + cdar(week; by=patch, cor=[1.0 0.5; 0.5 1.0])
        y ~ Normal(mu, 1.0)
    end
    @test SBBRMI(literal; mod=@__MODULE__).data[:cdar_mu_week_L] ≈ Matrix(cholesky(Symmetric(C)).L)
    @test_throws "positive definite" SBBRMI(cdar_model(cdar_df(; C=[1.0 2.0; 2.0 1.0])); mod=@__MODULE__)
    @test_throws "levels" SBBRMI(cdar_model(cdar_df(; C=[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])); mod=@__MODULE__)
end

@testset "cdar scale and persistence are addressable" begin
    df = cdar_df()
    configured = @brm df begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        sd(:, cdar(week)) ~ Normal(0.0, 0.2)
        ar(:, cdar(week)) ~ Normal(0.8, 0.1)
        y ~ Normal(mu, 1.0)
    end
    specs = term_priors(configured)
    @test Set(s.class for s in specs) == Set((:term_sd, :term_ar))
    @test all(s.term === Symbol("cdar(week)") for s in specs)
    code = stan(configured)
    @test occursin("real<lower=0.0> cdar_mu_week_sigma", code)
    @test occursin("cdar_mu_week_sigma ~ normal(0.0, 0.2)", code)
    @test occursin("real<lower=0.0, upper=1.0> cdar_mu_week_rho", code)
    @test occursin("cdar_mu_week_rho ~ normal(0.8, 0.1)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # a Beta persistence prior keeps the [0, 1] declaration (brm-use formulas.md)
    beta_prior = @brm df begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        ar(:, cdar(week)) ~ Beta(2, 2)
        y ~ Normal(mu, 1.0)
    end
    code = stan(beta_prior)
    @test occursin("real<lower=0.0, upper=1.0> cdar_mu_week_rho", code)
    @test occursin("cdar_mu_week_rho ~ beta(", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "cdar replay: frozen groups and factor, a growing step grid" begin
    sb = SBBRMI(cdar_model(); mod=@__MODULE__)
    # an appended step is a forecast: the fitted grid keeps its innovations and the
    # new step's are drawn in generated quantities (test/walk_forecast.jl)
    longer = reprocess(sb, cdar_df(; steps=3))
    @test longer.data[:cdar_mu_week_n_steps] == 2
    @test StanBlocks.getvalue(longer.data[:cdar_mu_week_n_new_steps]) == 1
    @test longer.data[:cdar_mu_week_step_idx] == [1, 1, 2, 2, 3, 3]
    @test longer.data[:cdar_mu_week_L] == sb.data[:cdar_mu_week_L]
    same = reprocess(sb, cdar_df())
    @test same.data[:cdar_mu_week_n_new_steps] == 0
    @test StanBlocks.stan_code(same.model) == StanBlocks.stan_code(sb.model)
    unseen = (; week=[1, 1], patch=["a", "c"], y=zeros(2), C=[1.0 0.0; 0.0 1.0])
    @test_throws "not a fitted group level" reprocess(sb, unseen)
    d = brm_descriptor(sb)
    @test any(o -> o.logical === :cdar_mu_week, d.outputs)
end

@testset "cdar BridgeStan path, density, gradient, and coordinates" begin
    if CDAR_RUNTIME
        sb = SBBRMI(cdar_model(); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        isdir(CDAR_CACHE) || mkpath(CDAR_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(CDAR_CACHE, string(hash(code)) * ".stan"))
        sm = problem.model
        string_names = String.(BS.param_names(sm))
        q = zeros(LogDensityProblems.dimension(problem))
        @test LogDensityProblems.dimension(problem) == 1 + 1 + 1 + 4   # intercept, sigma, rho, eta[1:4]
        sigma_i = only(findall(==("cdar_mu_week_sigma"), string_names))
        rho_i = only(findall(==("cdar_mu_week_rho"), string_names))
        eta_i = findall(startswith("cdar_mu_week_eta."), string_names)
        @test length(eta_i) == 4
        # sigma = 0.1, rho = 0.5 (unconstrained 0 on [0, 1]), eta column-major [1, -1 | 2, 0.5], L = I:
        #   delta[:, 1] = 0.1 * [1, -1] = [0.1, -0.1]
        #   delta[:, 2] = 0.5 * delta[:, 1] + 0.1 * sqrt(0.75) * [2, 0.5]
        q[sigma_i] = log(0.1); q[rho_i] = 0.0; q[eta_i] .= [1.0, -1.0, 2.0, 0.5]
        s = 0.1 * sqrt(0.75)
        expected = [0.1, -0.1, 0.05 + 2s, -0.05 + 0.5s]
        constrained_names = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        mu = [v for (nm, v) in zip(constrained_names, constrained) if startswith(String(nm), "mu.")]
        @test mu ≈ expected
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
        descriptor = brm_descriptor(sb)
        @test length(brm_term_coordinates(descriptor, :mu, constrained_names;
            term=:cdar_mu_week, parameter=:sd).coordinates) == 1
        @test length(brm_term_coordinates(descriptor, :mu, constrained_names;
            term=:cdar_mu_week, parameter=:ar).coordinates) == 1
        @test length(brm_term_coordinates(descriptor, :mu, constrained_names;
            term=:cdar_mu_week, parameter=:innovations).coordinates) == 4
        # With the intercept at zero the predictor IS the term's value.
        value = brm_term_coordinates(descriptor, :mu, constrained_names;
            term=:cdar_mu_week, parameter=:value)
        @test constrained[value.coordinates] ≈ mu
    else
        @info "Skipping BridgeStan cdar runtime gate (BRM_CDAR_RUNTIME=0)"
        @test true
    end
end

# Independent levels (snag independent-per-e13d14f0): no `cor=`, or `cor=I` as a
# `LinearAlgebra.I` data field, is the `C = I` walk without any `P × P` matrix:
# `_sb_cdar_iid` ships no `L` / `A_new` / `L_new` and runs no factor product.
cdar_iid_model(df=cdar_df()) = @brm df begin
    mu ~ 1 + cdar(week; by=patch)
    y ~ Normal(mu, 1.0)
end

@testset "cdar independent levels: no cor, or cor = I, forms no P × P matrix" begin
    sb = SBBRMI(cdar_iid_model(); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("independent_damped_walk", code)
    @test !occursin("cdar_mu_week_L", code) && !occursin("cdar_mu_week_A_new", code)
    for key in (:cdar_mu_week_L, :cdar_mu_week_A_new, :cdar_mu_week_L_new)
        @test !haskey(sb.data, key)
    end
    @test sb.data[:cdar_mu_week_n_groups] == 2
    @test sb.data[:cdar_mu_week_n_new_groups] == 0
    @test sb.data[:cdar_mu_week_group_idx] == [1, 2, 1, 2]
    @test sb.data[:cdar_mu_week_step_idx] == [1, 1, 2, 2]
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # `cor = C` with `C = I` (a `UniformScaling` field) is the same program
    field = SBBRMI(cdar_model(cdar_df(; C=I)); mod=@__MODULE__)
    @test StanBlocks.stan_code(field.model) == code
    @test !haskey(field.data, :cdar_mu_week_L)
    @test !any(v -> v isa LinearAlgebra.UniformScaling, values(field.data))
    # a scaled identity is not a correlation
    @test_throws "unit diagonal" SBBRMI(cdar_model(cdar_df(; C=2.0I)); mod=@__MODULE__)
    # the scale and persistence stay addressable
    configured = @brm cdar_df() begin
        mu ~ 1 + cdar(week; by=patch)
        sd(:, cdar(week)) ~ Normal(0.0, 0.3)
        ar(:, cdar(week)) ~ Beta(2, 2)
        y ~ Normal(mu, 1.0)
    end
    ccode = stan(configured)
    @test occursin("independent_damped_walk", ccode)
    @test occursin("cdar_mu_week_sigma ~ normal(0.0, 0.3)", ccode)
    @test occursin("cdar_mu_week_rho ~ beta(", ccode)
    @test StanBlocks.stanc_check(ccode; warn_pedantic=false).ok
    # replay: a forecast appends steps, frozen levels, still no factor
    longer = reprocess(sb, cdar_df(; steps=3))
    @test StanBlocks.getvalue(longer.data[:cdar_mu_week_n_new_steps]) == 1
    @test !haskey(longer.data, :cdar_mu_week_L)
    @test StanBlocks.stan_code(reprocess(sb, cdar_df()).model) == code
    # many levels: the data carries no matrix (a dense `L` alone would be 32 MB),
    # and every row finds its level by lookup, linear in rows plus levels
    P = 2000
    many = (; week=repeat(1:3; inner=P), patch=repeat(string.("g", 1:P); outer=3),
              y=zeros(3P))
    big = SBBRMI(cdar_iid_model(many); mod=@__MODULE__)
    @test big.data[:cdar_mu_week_n_groups] == P
    @test !haskey(big.data, :cdar_mu_week_L)
    @test Base.summarysize(big.data) < 1_000_000
    @test big.data[:cdar_mu_week_group_idx] == repeat(sortperm(string.("g", 1:P)) |>
        invperm; outer=3)
    # the model description states independent levels and no factor `L`
    md = brm_description_markdown(brm_description(sb))
    @test occursin("independent levels", md)
    @test !occursin("correlation factor L", md)
    @test occursin("correlation factor L", brm_description_markdown(brm_description(
        SBBRMI(cdar_model(); mod=@__MODULE__))))
    @test occursin("unabhängige Stufen", brm_description_markdown(brm_description(sb;
        texts=description_texts(:de))))
end

@testset "cdar independent levels: BridgeStan density and gradient equal the dense C = I walk" begin
    if CDAR_RUNTIME
        isdir(CDAR_CACHE) || mkpath(CDAR_CACHE)
        inst(sb) = StanBlocks.stan_instantiate(sb.model;
            path=joinpath(CDAR_CACHE, string(hash(StanBlocks.stan_code(sb.model))) * ".stan"))
        dense = inst(SBBRMI(cdar_model(cdar_df(; steps=3)); mod=@__MODULE__))
        iid = inst(SBBRMI(cdar_iid_model(cdar_df(; steps=3)); mod=@__MODULE__))
        field = inst(SBBRMI(cdar_model(cdar_df(; C=I, steps=3)); mod=@__MODULE__))
        @test String.(BS.param_unc_names(iid.model)) == String.(BS.param_unc_names(dense.model))
        D = LogDensityProblems.dimension(dense)
        @test D == 3 + 6
        for seed in 1:5
            q = 0.7 .* randn(Random.Xoshiro(seed), D)
            lp, g = LogDensityProblems.logdensity_and_gradient(dense, q)
            for problem in (iid, field)
                lp2, g2 = LogDensityProblems.logdensity_and_gradient(problem, q)
                @test lp2 ≈ lp rtol=1e-12
                @test g2 ≈ g rtol=1e-10
            end
        end
    else
        @info "Skipping BridgeStan independent cdar gate (BRM_CDAR_RUNTIME=0)"
        @test true
    end
end

println("cdar_term.jl: all testsets passed")
