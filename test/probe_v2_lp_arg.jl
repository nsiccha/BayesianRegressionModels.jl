# test/probe_v2_lp_arg.jl — kernel(...) v2 blocker: a LATENT per-subject linear
# predictor as a positional kernel arg (todo `0h5cs3w`, GO `0dnesv9`).
#
# `log_CL` / `log_V` are NOT dataframe columns — they are formula statements with
# their own `|p|` ranef bucket. Before this, `_sb_kernel_doblock!` rejected them
# with "positional args (after the do-block) must be data columns", so a cell
# could only ever receive raw data and the typical values had to be hard-coded
# inside the cell body. Lifting that is what the whole v2 surface stands on.
#
# WHY THE `sb.data` ASSERTION IS THE LOAD-BEARING ONE: the tempting fix is to
# register the LP like any other column. That transpiles, passes stanc, and
# samples — while silently shadowing the sampled parameter with a constant. Text
# checks cannot see it. Absence from `sb.data` plus a finite BridgeStan GRADIENT
# is what actually pins the parameter as live.
#
# Grouping is derived from the LPs' shared ranef bucket (`0xuaz0k`): `by=` is
# gone, disagreeing LP buckets fail loudly, and a kernel with NO per-subject LP
# fails loudly (decision `1kg5340`).

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential

const V2_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const V2_N = 6

# Pre-grouped per-subject frame (the kernel contract): one row per subject, so an
# ordinary LP over it is ALREADY length n_subjects — no second LP shape needed.
v2_df() = (;
    t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:V2_N],
    dose    = fill(100.0, V2_N),
    dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:V2_N],
    weight  = collect(range(60.0, 90.0; length = V2_N)),
    subject = collect(1:V2_N),
    site    = collect(V2_N:-1:1),
)

v2_lp_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    pred   ~ kernel(t, dose, dv, log_CL, log_V; n_eta = 1) do ts, d, yy, lCL, lV, eta
        CL = exp(lCL); Vc = exp(lV); Ka = 1.5 * exp(eta[1])
        ke = CL / Vc
        mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_no_lp_model(df) = @brm df begin
    sigma ~ Exponential(1)
    pred  ~ kernel(t, dose, dv; n_eta = 1) do ts, d, yy, eta
        mu = d * exp(-exp(eta[1]) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_by_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t, dose, dv, log_CL; by = subject, n_eta = 1) do ts, d, yy, lCL, eta
        mu = d * exp(-exp(lCL + eta[1]) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_disagreeing_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 + (1 | q | site)
    pred   ~ kernel(t, dose, dv, log_CL, log_V; n_eta = 1) do ts, d, yy, lCL, lV, eta
        mu = d * exp(-exp(lCL - lV + eta[1]) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

@testset "kernel(...) v2 — latent per-subject LP as a positional arg" begin
    df = v2_df()
    sb = SBBRMI(v2_lp_model(df); mod = @__MODULE__)

    @testset "grouping is derived from one shared LP bucket" begin
        @test_throws "needs at least one per-subject linear-predictor" SBBRMI(
            v2_no_lp_model(df); mod = @__MODULE__)
        @test_throws "no longer accepts `by=`" SBBRMI(
            v2_by_model(df); mod = @__MODULE__)
        @test_throws "disagree on their grouping" SBBRMI(
            v2_disagreeing_model(df); mod = @__MODULE__)
    end

    @testset "LPs are parameters, not data" begin
        for nm in (:log_CL, :log_V)
            @test !haskey(sb.data, nm)
        end
    end

    @testset "transpile + stanc" begin
        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        for nm in ("log_CL", "log_V")
            @test occursin(nm, code)
        end
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        @test !occursin("vector[\"", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    end

    @testset "BridgeStan runtime — the LP is a live parameter" begin
        if V2_RUN_BRIDGESTAN
            using BridgeStan, LogDensityProblems
            cache = joinpath(tempdir(), "brm-v2-lp-arg")
            isdir(cache) || mkpath(cache)
            code = StanBlocks.stan_code(sb.model)
            prob = StanBlocks.stan_instantiate(sb.model;
                                               path = joinpath(cache, string(hash(code)) * ".stan"))
            dim = LogDensityProblems.dimension(prob)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end
