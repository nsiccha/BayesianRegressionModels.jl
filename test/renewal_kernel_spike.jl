# test/renewal_kernel_spike.jl — FEASIBILITY SPIKE (decision `2026-08-26T10-40-17-350-1kyw2vk`).
#
# Question: can BRM host a wastewater/renewal Rt-inference core (CDC ww-inference-model,
# EpiSewer)? The mechanistic core is the renewal recurrence
#
#     I(t) = Rt(t) · Σ_{s≥1} I(t-s) · g(s)
#
# a latent trajectory defined by a CARRIED-STATE sequential scan — which StanBlocks'
# `plate` deliberately does NOT express (independent cells only). The user's steer
# (brief `2026-08-26T10-39-56-553-l85sx9`): the scan belongs in a `@deffun` submodel,
# exactly like the ode_rk45-in-@deffun kernel cells SbPMX already ships. This fixture
# proves that end to end.
#
# Three forms, each gated on stanc + finite BridgeStan density/gradient:
#   A  StanBlocks @slic, single series, dense Poisson counts — the @deffun scan, no plate.
#   B  StanBlocks @slic, multi-site `plate`, ragged lognormal wastewater obs — the scan
#      broadcast per site (the plate case the user flagged as "a bit more finicky").
#   C  @brm kernel(...) surface: per-timepoint log-Rt threaded via ragged(log_rt, site),
#      per-site log-I0 LP, the renewal @deffun called in the cell — the BRM-internal path.
#
# Gate: `compiles()` / stanc + finite BridgeStan lp+gradient (kernel primer rule — never
# settle for transpiles()). RUN_BRIDGESTAN=0 skips the runtime layer.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Normal
import StanBlocks.stan: transpiles, compiles

const RENEWAL_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const RENEWAL_CACHE = joinpath(tempdir(), "brm-renewal-spike")

# The renewal recurrence — a plain @deffun sequential scan with a carried buffer.
# ternary-guarded lookback for the initial window; plain-`=` accumulator (§5).
StanBlocks.@deffun begin
    renewal(logRt::vector[nt], g::vector[ng], I0::real)::vector[nt] = begin
        I::vector[nt]
        for t in 1:nt
            acc = 0.0
            for s in 1:ng
                prev = t - s >= 1 ? I[t - s] : I0
                acc = acc + g[s] * prev
            end
            I[t] = exp(logRt[t]) * acc
        end
        I
    end
end

# Fixed-generation-interval variant (g baked in) for the @brm cell, which does not
# thread a global vector into the kernel do-block. A known generation interval is
# standard practice in renewal Rt-inference.
StanBlocks.@deffun begin
    renewal_fixedg(logRt::vector[nt], I0::real)::vector[nt] = begin
        g::vector[5]
        g[1] = 0.2; g[2] = 0.35; g[3] = 0.25; g[4] = 0.15; g[5] = 0.05
        I::vector[nt]
        for t in 1:nt
            acc = 0.0
            for s in 1:5
                prev = t - s >= 1 ? I[t - s] : I0
                acc = acc + g[s] * prev
            end
            I[t] = exp(logRt[t]) * acc
        end
        I
    end
end

function renewal_bridgestan_finite(model)
    isdir(RENEWAL_CACHE) || mkpath(RENEWAL_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(RENEWAL_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

renewal_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

const RENEWAL_G = [0.2, 0.35, 0.25, 0.15, 0.05]
const RENEWAL_NT = 20
const RENEWAL_NG = length(RENEWAL_G)

@testset "renewal core — StanBlocks @deffun scan, single series (dense Poisson)" begin
    ycounts = [5, 6, 8, 7, 9, 11, 10, 13, 12, 15, 14, 16, 18, 17, 19, 21, 20, 22, 24, 23]
    mA = @slic (; y = ycounts, g = RENEWAL_G, nt = RENEWAL_NT, ng = RENEWAL_NG) begin
        log_I0 ~ normal(1.0, 1.0)
        I0 = exp(log_I0)
        tau ~ normal(0.0, 0.5; lower = 0.0)
        eps::vector[nt] ~ std_normal()
        logRt = cumulative_sum(eps) * tau
        Inf_ = renewal(logRt, g, I0)
        y ~ poisson(Inf_)
    end
    @test transpiles(mA)
    @test compiles(mA)
    @test renewal_stanc_ok(mA)
    RENEWAL_RUN_BRIDGESTAN && @test renewal_bridgestan_finite(mA)
end

@testset "renewal core — StanBlocks plate over sites, ragged lognormal WW obs" begin
    nsites = 4
    logconc = [[0.3 * sin(t / 3) + 0.5 + 0.02 * t for t in 1:RENEWAL_NT] for _ in 1:nsites]
    mB = @slic (; ys = logconc, g = RENEWAL_G, nt = RENEWAL_NT, ng = RENEWAL_NG, nsites = nsites) begin
        tau ~ normal(0.0, 0.5; lower = 0.0)
        sigma ~ normal(0.0, 1.0; lower = 0.0)
        log_I0 ~ normal(1.0, 1.0)
        I0 = exp(log_I0)
        conc ~ plate(ys; outer = (nsites,)) do yi
            eps::vector[nt] ~ std_normal()
            logRt = cumulative_sum(eps) * tau
            Ii = renewal(logRt, g, I0)
            yi ~ normal(log(Ii), sigma)
            Ii
        end
    end
    @test transpiles(mB)
    @test compiles(mB)
    @test renewal_stanc_ok(mB)
    RENEWAL_RUN_BRIDGESTAN && @test renewal_bridgestan_finite(mB)
end

# --- @brm kernel surface: site = "subject", per-timepoint log-Rt via ragged() ---
renewal_brm_df() = begin
    nt = 12
    sites = ["A", "B", "C"]
    (;
        site = sites,
        t_grid = [collect(1.0:nt) for _ in sites],
        dv = [[0.3 * sin(t / 3) + 0.6 + 0.02 * t for t in 1:nt] for _ in sites],
        time_x = reduce(vcat, [collect(1.0:nt) for _ in sites]),
        time_site = reduce(vcat, [fill(s, nt) for s in sites]),
    )
end

renewal_brm_hsgp(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_rt ~ 1 + hsgp(time_x; k = 6)
    log_I0 ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, ragged(log_rt, time_site), log_I0) do ts, yy, logRt_i, lI0
        Ii = renewal_fixedg(logRt_i, exp(lI0))
        yy ~ normal(log(Ii), sigma)
        Ii
    end
end

renewal_brm_linear(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_rt ~ 1 + time_x
    log_I0 ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, ragged(log_rt, time_site), log_I0) do ts, yy, logRt_i, lI0
        Ii = renewal_fixedg(logRt_i, exp(lI0))
        yy ~ normal(log(Ii), sigma)
        Ii
    end
end

@testset "renewal core — @brm kernel(...) surface, smoothed (hsgp) log-Rt" begin
    sb = SBBRMI(renewal_brm_hsgp(renewal_brm_df()); mod = @__MODULE__)
    @test transpiles(sb.model)
    @test renewal_stanc_ok(sb.model)
    RENEWAL_RUN_BRIDGESTAN && @test renewal_bridgestan_finite(sb.model)
end

@testset "renewal core — @brm kernel(...) surface, linear log-Rt" begin
    sb = SBBRMI(renewal_brm_linear(renewal_brm_df()); mod = @__MODULE__)
    @test transpiles(sb.model)
    @test renewal_stanc_ok(sb.model)
    RENEWAL_RUN_BRIDGESTAN && @test renewal_bridgestan_finite(sb.model)
end
