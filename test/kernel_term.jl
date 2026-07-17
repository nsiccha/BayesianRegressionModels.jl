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
using Distributions: Exponential

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
