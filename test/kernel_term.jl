# test/kernel_term.jl — acceptance suite for the kernel(...) HoF term (todo
# `xxeali`, commit `bf8235e`).
#
#   pred ~ kernel(datacol...; by=g, model=cell, n_eta=K, obs=CombinedError(col, a, p))
#
# The kernel term broadcasts a consumer-declared deterministic @deffun cell over
# per-subject groups, auto-introducing a correlated eta block (LKJ + half-normal
# scales) and applying the obs family per subject inside a plate. This suite is
# the committed evidence the commit message claimed but never captured: a full
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
using BridgeStan
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
    q = zeros(dimension)
    lp = LogDensityProblems.logdensity(problem, q)
    lp_grad, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    isfinite(lp) && isfinite(lp_grad) && length(gradient) == dimension &&
        all(isfinite, gradient)
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

kernel_model(df) = @brm df begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    pred    ~ kernel(t, dose; by = subject, model = kernel_depot_cell, n_eta = 3,
                     obs = CombinedError(dv, sigma_a, sigma_p))
end

# The do-block surface (commit `cdf60a5`): the per-subject cell is written INLINE
# as a plate-style do-block, obs as `~` statements in the body — no `model=`/`obs=`
# DSL, no separately-declared @deffun cell. Whole-vector locals need NO size
# annotation since the infer-vector-loc fix (StanBlocks `853a321`): a bare
# `mu = <broadcast>` emits valid Stan with an UNQUOTED size (before the fix it
# emitted `vector["dims(ts)[1]"]`, which stanc rejects). This is the concise
# surface the do-block design targeted.
kernel_doblock_model(df) = @brm df begin
    sigma ~ Exponential(1)
    pred  ~ kernel(t, dose, dv; by = subject, n_eta = 3) do ts, d, yy, eta
        CL = 1.0 * exp(eta[1]); Vc = 10.0 * exp(eta[2]); Ka = 1.5 * exp(eta[3])
        ke = CL / Vc
        mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
        yy ~ normal(mu, sigma)
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
    pred  ~ kernel(t, dose; by = subject, n_eta = 3) do ts, d, eta
        CL = 1.0 * exp(eta[1]); Vc = 10.0 * exp(eta[2]); Ka = 1.5 * exp(eta[3])
        ke = CL / Vc
        d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
    end
    dv ~ Normal(pred, sigma)
end

# A marker that is deliberately NOT a registered obs family — pins the
# `_sb_kernel_obs_expr` fallback path independently of which families ship
# (CombinedError/TruncatedNormal/LogNormalError are all supported).
function unknown_obs_family end

@info "kernel(...) term test environment" StanBlocks=Base.pkgversion(StanBlocks) BridgeStan=Base.pkgversion(BridgeStan) RUN_BRIDGESTAN

@testset "kernel(...) HoF term — depot_1cmt PMX acceptance" begin
    df = kernel_df()

    @testset "build + transpile + stanc" begin
        sb = SBBRMI(kernel_model(df); mod=@__MODULE__)
        transpiles = StanBlocks.stan.transpiles(sb.model)
        @test transpiles
        # Regression guard for the leading-underscore stanc-lexing bug: Stan
        # rejects any identifier starting with `_`, so the emitted code must
        # contain no `_`-led identifier (works without the stanc binary).
        code = StanBlocks.stan_code(sb.model)
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        transpiles && @test stanc_accepts(sb.model)
    end

    @testset "BridgeStan runtime" begin
        if RUN_BRIDGESTAN
            sb = SBBRMI(kernel_model(df); mod=@__MODULE__)
            @test bridgestan_accepts(sb.model)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end

    @testset "loud guards (correct-or-loud contract)" begin
        # Missing required kwarg.
        m_noobs = @brm df begin
            pred ~ kernel(t, dose; by = subject, model = kernel_depot_cell, n_eta = 3)
        end
        @test_throws "missing required kwarg" SBBRMI(m_noobs; mod=@__MODULE__)

        # Unregistered obs family falls through to a loud error.
        m_badobs = @brm df begin
            sigma ~ Exponential(1)
            pred  ~ kernel(t, dose; by = subject, model = kernel_depot_cell, n_eta = 3,
                           obs = unknown_obs_family(dv, sigma))
        end
        @test_throws "not supported" SBBRMI(m_badobs; mod=@__MODULE__)

        # Long-format (multi-row-per-subject) data must be rejected, not silently
        # row-dropped: `by` is not 1:n_subjects in order.
        long = (;
            t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:6],
            dose    = fill(100.0, 6),
            dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:6],
            subject = [1, 1, 2, 2, 3, 3],
        )
        m_long = @brm long begin
            sigma_a ~ Exponential(1)
            sigma_p ~ Exponential(1)
            pred    ~ kernel(t, dose; by = subject, model = kernel_depot_cell, n_eta = 3,
                             obs = CombinedError(dv, sigma_a, sigma_p))
        end
        @test_throws "pre-grouped per-subject data" SBBRMI(m_long; mod=@__MODULE__)
    end
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
