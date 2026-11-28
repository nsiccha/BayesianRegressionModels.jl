# test/sb_logit_calls.jl — SBBRMI spells a deterministic `logit(x)` call
# (LogExpFunctions' function, which BRM exports) as Stan's `logit` wherever a
# Julia body lowers: a component member's body (scalar, inside a broadcast,
# and broadcast over data), a model-scope expression and a plate cell. Before
# `_sb_julia_stan_builtin` spelled it (src/sb_cell_julia.jl), the member and
# model-scope models stopped in the StanBlocks transpile with "`tracetype` not
# defined for logit(::real)" (snag `member-logit-cal-dda1b06a`); the plate
# cell already emitted `logit`.
#
# Run: julia --startup-file=no --project=test test/sb_logit_calls.jl

module SBLogitCalls
using Test
using BayesianRegressionModels
import StanBlocks

@brm function logit_arm(entry)
    slope ~ Normal(0.0, 1.0)
    frac ~ Beta(1.0, 1.0)
    shift ~ Normal(0.0, 2.0)
    slow = slope .* logistic.(logit(frac) .+ shift .* entry.x)
    value = sum(slow) + logit(frac) + sum(logit.(entry.p))
    return value
end
const members = @brm begin
    for item in items
        arms[item.id] ~ logit_arm(item)
    end
    y ~ Normal(arms, 1)
end
const scope = @brm begin
    frac ~ Beta(1.0, 1.0)
    shift ~ Normal(0.0, 2.0)
    p = logistic.(logit(frac) .+ shift .* x)
    y ~ Normal(p, 1.0)
end
const cell = @brm begin
    frac ~ Beta(1.0, 1.0)
    shift ~ Normal(0.0, 2.0)
    @plate for k in eachindex(y)
        y[k] ~ Normal(logistic(logit(frac) + shift * x[k]), 1.0)
    end
end

compiles(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok

@testset "logit in a component member" begin
    items = [(id=:a, x=[0.1, 0.5], p=[0.2, 0.7]),
             (id=:b, x=[-0.3, 0.2, 1.0], p=[0.4, 0.5, 0.9])]
    code = stan_code(SBBRMI(members((; items, y=[0.3, 0.7]));
                            mod=@__MODULE__, total_groups=()))
    # `logistic.(logit(frac) .+ …)`: the sum is parenthesised inside `inv_logit`.
    @test occursin(r"inv_logit\(\s*\(?\s*logit\(", code)
    @test compiles(code)
end

@testset "logit at model scope and in a plate cell" begin
    rows = (; x=[-1.0, 0.0, 2.0], y=[0.1, 0.4, 0.9])
    code = stan_code(SBBRMI(scope(rows); mod=@__MODULE__))
    @test occursin(r"p\s*=\s*inv_logit\(\s*\(?\s*logit\(frac\)", code)
    @test compiles(code)
    code = stan_code(SBBRMI(cell(rows); mod=@__MODULE__))
    @test occursin(r"(?<!inv_)logit\(frac\)", code)
    @test compiles(code)
end
end
