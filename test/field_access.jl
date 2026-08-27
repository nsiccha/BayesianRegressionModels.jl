# test/field_access.jl — gate for @brm BODY field/index access on a model value.
# A `state = scan(...)` bound to a @deffun struct/array return can be consumed
# directly — `y ~ f(state.field)` / `y ~ f(state[i])` — which the backend lowers to
# Stan `.` / `[`. This is @slic parity for multi-carrier scans (one scan, named
# fields), replacing hand-rolled accessor @deffuns. Transpile + stanc + finite
# BridgeStan gradient.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Normal, Exponential
import StanBlocks.stan: transpiles, compiles

StanBlocks.@deffun begin
    # returns a NamedTuple of two carriers (as run_state_process does)
    fa_twocarrier(x::vector[n]) = begin
        a::vector[n]
        b::vector[n]
        for i in 1:n
            a[i] = 2.0 * x[i]
            b[i] = x[i] + 1.0
        end
        (; a, b)
    end
    fa_matrix(x::vector[n])::matrix[n, 2] = begin
        m::matrix[n, 2]
        for i in 1:n
            m[i, 1] = 2.0 * x[i]
            m[i, 2] = x[i] + 1.0
        end
        m
    end
end

fa_stanc_ok(brmi) = begin
    m = SBBRMI(brmi; mod = @__MODULE__).model
    transpiles(m) && StanBlocks.stanc_check(StanBlocks.stan_code(m); warn_pedantic = false).ok
end
fa_finite(brmi) = begin
    m = SBBRMI(brmi; mod = @__MODULE__).model
    prob = StanBlocks.stan_instantiate(m; path = tempname() * ".stan")
    d = LogDensityProblems.dimension(prob)
    lp, g = LogDensityProblems.logdensity_and_gradient(prob, zeros(d))
    isfinite(lp) && all(isfinite, g)
end

@testset "field access — @deffun NamedTuple return consumed by named field" begin
    n = 10
    df = (; xcol = collect(1.0:n), y = randn(n) .+ 2, z = randn(n) .+ 3)
    m(df) = @brm df begin
        sigma ~ Exponential(1.0)
        state = fa_twocarrier(xcol)
        y ~ Normal(state.a, sigma)      # named field, stream 1
        z ~ Normal(state.b, sigma)      # named field, stream 2 (one scan, two carriers)
    end
    @test fa_stanc_ok(m(df))
    @test fa_finite(m(df))
end

@testset "index access — @deffun matrix return consumed by column index" begin
    n = 10
    df = (; xcol = collect(1.0:n), y = randn(n) .+ 2, z = randn(n) .+ 3)
    m(df) = @brm df begin
        sigma ~ Exponential(1.0)
        state = fa_matrix(xcol)
        y ~ Normal(state[:, 1], sigma)
        z ~ Normal(state[:, 2], sigma)
    end
    @test fa_stanc_ok(m(df))
    @test fa_finite(m(df))
end
