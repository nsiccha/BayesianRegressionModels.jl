# SBBRMI's default random-effect parametrization is the standard
# (non-centered) one (user point `073uzow`), and exact totals are deprecated
# everywhere (user point `1qwe3nj`): an explicit `total_groups` request still
# builds, with a deprecation warning, so fits that carry totals keep working.
# Snag `exact-totals-fun-f116d6b2`, user pick `0rz9l5s`.
#
# Run: julia --project=test test/standard_default.jl
using Test, Random
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

const DEFAULT_BUILDER = @brm begin
    mu ~ 1 + x + (1 | g)
    effect(mu, Intercept) ~ Normal(0.0, 2.0)
    y ~ Normal(mu, 1.0)
end

const DEFAULT_DATA = let rng = Xoshiro(7), J = 5, n = 6
    g = repeat(1:J; inner=n)
    x = randn(rng, J * n)
    (; g, x, y = 0.5 .+ x .+ randn(rng, J)[g] .+ 0.3 .* randn(rng, J * n))
end

# Warnings at `Warn` and above that an expression emits, captured by a fresh
# logger (so `maxlog=1` elsewhere in the session does not hide them).
function warnings_of(f)
    logger = Test.TestLogger(min_level=Base.CoreLogging.Warn)
    value = Base.CoreLogging.with_logger(f, logger)
    value, [string(r.message) for r in logger.logs]
end

@testset "totals are deprecated, still built on request" begin
    sb = @test_logs (:warn, r"`total_groups` \(exact total coefficients\) is deprecated") match_mode=:any SBBRMI(
        DEFAULT_BUILDER(DEFAULT_DATA); mod=@__MODULE__, total_groups=:auto)
    @test length(total_effect_blocks(sb)) == 1
end

@testset "the default is the standard parametrization" begin
    sb, warnings = warnings_of(() -> SBBRMI(DEFAULT_BUILDER(DEFAULT_DATA); mod=@__MODULE__))
    @test !any(w -> occursin("deprecated", w), warnings)
    @test isempty(total_effect_blocks(sb))
    @test StanBlocks.stan_code(sb.model) ==
          StanBlocks.stan_code(SBBRMI(DEFAULT_BUILDER(DEFAULT_DATA); mod=@__MODULE__, total_groups=()).model)
    plan = generative_plan(DEFAULT_BUILDER, DEFAULT_DATA; mod=@__MODULE__)
    @test isempty(total_effect_blocks(plan))
    d = brm_descriptor(DEFAULT_BUILDER, DEFAULT_DATA; mod=@__MODULE__)
    @test isempty(total_effect_blocks(d.plan))
end
