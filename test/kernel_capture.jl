# test/kernel_capture.jl — the @brm kernel(...) cell captures a SHARED model-data
# vector referenced freely in the do-block body (decision
# `2026-08-26T11-23-02-625-11ge7t2`). Before this, a data vector reached the cell
# neither by lexical capture nor as a data kwarg; now a name that is a real df
# column and NOT a formula name is registered as shared Stan data, so the plate
# cell resolves it the way a `plate` captures a `@slic` data kwarg. PMFs / delay
# distributions can therefore be DATA rather than @deffun literals.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Exponential, Normal
import StanBlocks.stan: transpiles, compiles

include(joinpath(@__DIR__, "..", "research", "wastewater", "renewal.jl"))

const CAP_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const CAP_CACHE = joinpath(tempdir(), "brm-kernel-capture")

function cap_bridgestan_finite(model)
    isdir(CAP_CACHE) || mkpath(CAP_CACHE)
    code = StanBlocks.stan_code(model)
    path = joinpath(CAP_CACHE, string(hash(code)) * ".stan")
    prob = StanBlocks.stan_instantiate(model; path)
    dim = LogDensityProblems.dimension(prob)
    q = [0.1 * ((i % 5) - 2) for i in 1:dim]
    lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
    isfinite(lp) && all(isfinite, grad) && length(grad) == dim
end

cap_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

# A tiny @deffun that consumes a captured data vector.
StanBlocks.@deffun begin
    cap_weighted_sum(x::vector[n], w::vector[m])::real = begin
        acc = 0.0
        for i in 1:m
            acc = acc + w[i] * x[1]
        end
        acc
    end
end

cap_df() = begin
    nt = 8
    sites = ["a", "b", "c"]
    (;
        site = sites,
        t_grid = [collect(1.0:nt) for _ in sites],
        dv = [[0.2 * t + 0.5 for t in 1:nt] for _ in sites],
        wvec = [0.25, 0.25, 0.25, 0.25],   # a shared data vector (NOT per-row)
    )
end

# `wvec` is referenced only inside the cell — captured as shared data.
cap_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_a  ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
        bump = cap_weighted_sum(ts, wvec)   # wvec captured
        yy ~ normal(ts .* exp(la) .+ bump, sigma)
        ts
    end
end

# Same model, but the cell references a name that is neither a column, a formula
# name, nor a builtin — must still fail loudly (no silent swallow into data).
cap_model_typo(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_a  ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
        bump = cap_weighted_sum(ts, not_a_column_pmf)
        yy ~ normal(ts .* exp(la) .+ bump, sigma)
        ts
    end
end

@testset "kernel(...) captures a shared data vector referenced in the cell" begin
    sb = SBBRMI(cap_model(cap_df()); mod = @__MODULE__)
    @test haskey(sb.data, :wvec)                 # registered as shared data
    @test sb.data[:wvec] == [0.25, 0.25, 0.25, 0.25]
    @test transpiles(sb.model)
    @test cap_stanc_ok(sb.model)
    CAP_RUN_BRIDGESTAN && @test cap_bridgestan_finite(sb.model)
end

@testset "kernel(...) capture: a non-column free name still errors loudly" begin
    sb = SBBRMI(cap_model_typo(cap_df()); mod = @__MODULE__)
    # `not_a_column_pmf` is neither data, a formula name, nor a builtin — it must
    # NOT be silently swallowed; StanBlocks rejects it at trace.
    @test_throws Exception StanBlocks.stan_code(sb.model)
end

@testset "kernel(...) capture: no spurious capture when the cell has none" begin
    # A do-block that references only params / locals / do-params must not gain
    # any data key from cell-local names (they are not columns).
    df = (; site = ["a", "b"], t_grid = [collect(1.0:4) for _ in 1:2],
            dv = [[1.0, 2.0, 3.0, 4.0] for _ in 1:2])
    m = @brm df begin
        sigma ~ Exponential(1)
        log_a ~ 1 + (1 | site)
        pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
            mu = ts .* exp(la)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = SBBRMI(m; mod = @__MODULE__)
    @test !haskey(sb.data, :mu)   # cell-local not captured
    @test !haskey(sb.data, :la)   # do-param not captured
    @test transpiles(sb.model)
end

@testset "wastewater @brm model — PMFs captured as data" begin
    sb = SBBRMI(wastewater_brm_model(); mod = @__MODULE__)
    @test haskey(sb.data, :g)
    @test haskey(sb.data, :sh)
    @test transpiles(sb.model)
    @test cap_stanc_ok(sb.model)
    CAP_RUN_BRIDGESTAN && @test cap_bridgestan_finite(sb.model)
end
