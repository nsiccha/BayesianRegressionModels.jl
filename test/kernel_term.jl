# test/kernel_term.jl — acceptance suite for the kernel(...) do-block term.
#
# Formula-declared per-subject LPs own population effects and correlated BSV;
# kernel derives their grouping and broadcasts the inline cell. A full
# depot_1cmt population PMX model written as ONE @brm formula must transpile,
# pass stanc, and produce a finite BridgeStan log-density + gradient.
#
# The commit only checked `transpiles` (text emission), which the StanBlocks
# README warns "proves SLIC can emit text, not that stanc accepts it". stanc in
# fact REJECTED the emitted code — the auto-introduced top-level names had a
# leading underscore, which Stan forbids (`_kernel_nsub_pred`, ...); fixed in
# `cf8b218` (rename to `kernel_*`). The stanc + BridgeStan gates below, plus the
# binary-free leading-underscore regex, pin that fix so it cannot regress.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Normal

const KERNEL_CACHE = joinpath(tempdir(), "brm-kernel-term")
const RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"

function stanc_accepts(model)
    result = StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected kernel-term model" output=result.output
    result.ok
end

function bridgestan_accepts(model)
    code = StanBlocks.stan_code(model)
    isdir(KERNEL_CACHE) || mkpath(KERNEL_CACHE)
    path = joinpath(KERNEL_CACHE, string(hash(code)) * ".stan")
    problem = StanBlocks.stan_instantiate(model; path)
    dimension = LogDensityProblems.dimension(problem)
    q = [0.1 * ((i % 5) - 2) for i in 1:dimension]
    lp = LogDensityProblems.logdensity(problem, q)
    lp_grad, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    ok = isfinite(lp) && isfinite(lp_grad) && length(gradient) == dimension &&
         all(isfinite, gradient)
    ok || @error "non-finite kernel BridgeStan probe" dimension lp lp_grad gradient_length=length(gradient) finite_gradient=all(isfinite, gradient)
    ok
end

# Log-density + gradient at a deterministic fixed point, for the cleaner-cut ≡
# obs-in-cell equivalence check. Reproducible q so the two forms compare exactly.
function kernel_lp_grad(model)
    isdir(KERNEL_CACHE) || mkpath(KERNEL_CACHE)
    path = joinpath(KERNEL_CACHE, string(hash(StanBlocks.stan_code(model))) * ".stan")
    problem = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(problem)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, g = LogDensityProblems.logdensity_and_gradient(problem, q)
    (dim, lp, g)
end

# Consumer-side deterministic structural cell: 1-compartment depot analytical
# solution. eta -> per-subject prediction vector over the subject's time grid.
StanBlocks.@deffun begin
    kernel_depot_cell(ts::vector[nt], d::real, eta::vector[ne])::vector[nt] = begin
        CL = 1.0 * exp(eta[1]); Vc = 10.0 * exp(eta[2]); Ka = 1.5 * exp(eta[3])
        ke = CL / Vc
        d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
    end
end

# Pre-grouped per-subject PMX data (v1 contract): one ragged time/dv value per
# subject, `subject` = 1:n in order. Deterministic values keep the BridgeStan
# gate reproducible.
const KERNEL_N = 6
kernel_df() = (;
    t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:KERNEL_N],
    dose    = fill(100.0, KERNEL_N),
    dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:KERNEL_N],
    subject = collect(1:KERNEL_N),
)

# The do-block surface (commit `cdf60a5`): the per-subject cell is written INLINE
# as a plate-style do-block, obs as `~` statements in the body — no `model=`/`obs=`
# DSL, no separately-declared @deffun cell. Whole-vector locals need NO size
# annotation since the infer-vector-loc fix (StanBlocks `853a321`): a bare
# `mu = <broadcast>` emits valid Stan with an UNQUOTED size (before the fix it
# emitted `vector["dims(ts)[1]"]`, which stanc rejects). This is the concise
# surface the do-block design targeted.
kernel_doblock_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    log_Ka ~ 1 + (1 | p | subject)
    pred  ~ kernel(t, dose, dv, log_CL, log_V, log_Ka) do ts, d, yy, lCL, lV, lKa
        CL = exp(lCL); Vc = exp(lV); Ka = exp(lKa)
        ke = CL / Vc
        mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
        yy ~ normal(mu, sigma)
        mu
    end
end

# `truncated_normal` is an `@deffun`-backed distribution, so signature selection
# does not apply native Stan's scalar broadcasting. The plate slices `log_obs`
# and `mu` to vectors; explicitly lift the shared scalar sigma/bounds to the
# registered all-vector signature. This is the consumer-critical do-block form
# that replaces `obs=TruncatedNormal(...)`.
kernel_doblock_truncated_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    log_Ka ~ 1 + (1 | p | subject)
    pred  ~ kernel(
        t, dose, dv, log_CL, log_V, log_Ka,
    ) do ts, d, log_obs, lCL, lV, lKa
        CL = exp(lCL); Vc = exp(lV); Ka = exp(lKa)
        ke = CL / Vc
        mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
        log_obs ~ truncated_normal(
            mu,
            rep_vector(sigma, dims(mu)[1]),
            rep_vector(-10.0, dims(mu)[1]),
            rep_vector(10.0, dims(mu)[1]))
        mu
    end
end

# The "cleaner cut": the kernel returns mu ONLY; the observation model is applied
# OUTSIDE, as an ordinary top-level `dv ~ Normal(pred, sigma)`. With ragged-dist-arg
# (StanBlocks `9f4b0fa`) the ragged plate-result `pred` broadcasts the distribution
# ACROSS subject groups (`for g: dv[g] ~ Normal(pred[g], sigma)`), so a vector-valued
# response would correctly keep each subject as one obs. This form is bit-for-bit
# log-density-equivalent to the obs-in-cell `kernel_doblock_model` (asserted below).
kernel_muout_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    log_Ka ~ 1 + (1 | p | subject)
    pred  ~ kernel(t, dose, log_CL, log_V, log_Ka) do ts, d, lCL, lV, lKa
        CL = exp(lCL); Vc = exp(lV); Ka = exp(lKa)
        ke = CL / Vc
        d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
    end
    dv ~ Normal(pred, sigma)
end

# Multi-output do-block (two correlated LP buckets, `|p|` and `|q|`) — the
# documented joint PK-PD surface (sbimpl.jl `_sb_kernel_doblock!` docstring).
# Each output gets its OWN ragged obs column + time grid and is observed as a
# WHOLE column (`ypk`, `ypd`); NO in-cell cmt masking (which the StanBlocks
# substrate does not yet lower —
# a real ragged slice is a native `vector` so `.== ` yields no mask, and an integer
# cmt column's `findall` index cannot be a per-cell `array[] int` local). This is the
# committed evidence that the corrected docstring example actually transpiles.
kernel_doblock_multiout_model(df) = @brm df begin
    sd ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    log_Emax ~ 1 + (1 | q | subject)
    log_EC50 ~ 1 + (1 | q | subject)
    pred ~ kernel(
        t_pk, t_pd, dose, dv_pk, dv_pd, log_CL, log_V, log_Emax, log_EC50,
    ) do tpk, tpd, d, ypk, ypd, lCL, lV, lEmax, lEC50
        CL = exp(lCL); Vc = exp(lV); ke = CL / Vc
        conc_pk = d / Vc * exp(-ke * tpk)
        conc_pd = d / Vc * exp(-ke * tpd)
        Emax = exp(lEmax); EC50 = exp(lEC50)
        resp = Emax * conc_pd ./ (EC50 .+ conc_pd)
        ypk ~ normal(conc_pk, sd)
        ypd ~ normal(resp, sd)
        conc_pk
    end
end

# Per-subject data for the multi-output form: separate PK/PD ragged columns + grids.
kernel_multiout_df() = (;
    t_pk    = [abs.(sin.(1:3)) .+ 0.5 for _ in 1:KERNEL_N],
    t_pd    = [abs.(cos.(1:3)) .+ 0.5 for _ in 1:KERNEL_N],
    dose    = fill(100.0, KERNEL_N),
    dv_pk   = [abs.(cos.(1:3)) .+ 0.1 for _ in 1:KERNEL_N],
    dv_pd   = [abs.(sin.(1:3)) .+ 0.1 for _ in 1:KERNEL_N],
    subject = collect(1:KERNEL_N),
)

@info "kernel(...) term test environment" StanBlocks=Base.pkgversion(StanBlocks) RUN_BRIDGESTAN

@testset "kernel(...) legacy model=/obs= surface is retired" begin
    df = kernel_df()
    legacy = @brm df begin
        pred ~ kernel(
            t, dose;
            by=subject, model=kernel_depot_cell, n_eta=3, obs=Normal(dv, 1.0),
        )
    end
    @test_throws "only accepts the inline do-block form" SBBRMI(
        legacy; mod=@__MODULE__)
    @test !isdefined(BayesianRegressionModels, :CombinedError)
    @test !isdefined(BayesianRegressionModels, :LogNormalError)
    @test isdefined(BayesianRegressionModels, :TruncatedNormal)
end

@testset "kernel(...) do-block surface — bare-local cell" begin
    df = kernel_df()

    @testset "build + transpile + stanc" begin
        sb = SBBRMI(kernel_doblock_model(df); mod=@__MODULE__)
        transpiles = StanBlocks.stan.transpiles(sb.model)
        @test transpiles
        code = StanBlocks.stan_code(sb.model)
        # Leading-underscore guard (shared with the regression form): Stan rejects
        # any identifier starting with `_` (`cf8b218`).
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        # infer-vector-loc regression guard (StanBlocks `853a321`): a bare
        # whole-vector local (`mu = <broadcast>`) must emit an UNQUOTED size —
        # never the pre-fix invalid `vector["dims(ts)[1]"]`.
        @test !occursin("vector[\"", code)
        transpiles && @test stanc_accepts(sb.model)
    end

    @testset "BridgeStan runtime" begin
        if RUN_BRIDGESTAN
            sb = SBBRMI(kernel_doblock_model(df); mod=@__MODULE__)
            @test bridgestan_accepts(sb.model)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end

@testset "kernel(...) do-block surface — truncated_normal exact signature" begin
    df = kernel_df()
    sb = SBBRMI(kernel_doblock_truncated_model(df); mod=@__MODULE__)
    @test StanBlocks.stan.transpiles(sb.model)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("truncated_normal_lpdf", code)
    @test stanc_accepts(sb.model)
end

@testset "kernel(...) do-block — cleaner cut (obs outside) ≡ obs-in-cell" begin
    df = kernel_df()

    @testset "build + transpile + stanc" begin
        sb = SBBRMI(kernel_muout_model(df); mod=@__MODULE__)
        transpiles = StanBlocks.stan.transpiles(sb.model)
        @test transpiles
        code = StanBlocks.stan_code(sb.model)
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        @test !occursin("vector[\"", code)
        transpiles && @test stanc_accepts(sb.model)
    end

    # The two forms emit different Stan (obs inside the plate vs. a top-level ragged
    # broadcast) but are the SAME density. Assert identical dimension, log-density,
    # and gradient at a fixed point — observed Δ = 0 exactly (StanBlocks `9f4b0fa`).
    @testset "log-density equivalence to obs-in-cell" begin
        if RUN_BRIDGESTAN
            sb_out = SBBRMI(kernel_muout_model(df);   mod=@__MODULE__)
            sb_cell = SBBRMI(kernel_doblock_model(df); mod=@__MODULE__)
            dO, lpO, gO = kernel_lp_grad(sb_out.model)
            dC, lpC, gC = kernel_lp_grad(sb_cell.model)
            @test dO == dC
            @test isapprox(lpO, lpC; atol=1e-8, rtol=0)
            @test dO == dC && isapprox(gO, gC; atol=1e-8, rtol=0)
        else
            @info "Skipping equivalence gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end

@testset "kernel(...) do-block — multi-output (two eta blocks, per-output columns)" begin
    df = kernel_multiout_df()

    @testset "build + transpile + stanc" begin
        sb = SBBRMI(kernel_doblock_multiout_model(df); mod=@__MODULE__)
        transpiles = StanBlocks.stan.transpiles(sb.model)
        @test transpiles
        code = StanBlocks.stan_code(sb.model)
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        @test !occursin("vector[\"", code)
        transpiles && @test stanc_accepts(sb.model)
    end

    @testset "BridgeStan runtime" begin
        if RUN_BRIDGESTAN
            sb = SBBRMI(kernel_doblock_multiout_model(df); mod=@__MODULE__)
            @test bridgestan_accepts(sb.model)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end
