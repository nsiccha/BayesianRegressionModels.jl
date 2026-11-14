# test/shadowed_call_head.jl — a formula call whose head is bound to a value
# that cannot be called (snag rkbrmi-shadowed-035220a8).
#
# Run: julia --startup-file=no --project=test test/shadowed_call_head.jl
#
# A formula call's head resolves where the block is written, as any Julia
# call's does: a local first, then a global of the module (`mod` for
# `brm(source; mod)`). A script variable named like a wrapper or a family
# (`censored = value .< limit`, marking rows below a detection limit) is such
# a global and shadows the binding `using BayesianRegressionModels` brings in.
# Before BRM 0.2.302 the builder carried the variable's value as the node's
# callable, and each backend refused it with a message about its own lowering:
# RKBRMI `response `y` distribution keywords are unsupported`, SBBRMI
# `distribution `Bool[1, 0, 0]` needs a Stan translation for constructor
# keywords (:lower,)`. The builder now refuses the call itself, naming the
# call, where its head is bound, what it holds and, when BRM binds the same
# name, the shadowed binding and the two remedies.

using Test
using BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels

const DATA = (; y=[0.1, 0.2, 0.3], x=[1.0, 2.0, 3.0], lo=[0.1, -Inf, -Inf])
const BODY = quote
    mu ~ 1 + x
    y ~ censored(Normal(mu, 1); lower=lo)
end
const QUALIFIED = quote
    mu ~ 1 + x
    y ~ BayesianRegressionModels.censored(Normal(mu, 1); lower=lo)
end

module Clean
using BayesianRegressionModels
end
module Shadowed
using BayesianRegressionModels
censored = [true, false, false]
end
module ShadowedFamily
using BayesianRegressionModels
Normal = 1.0
end

struct CallableFamily end
(::CallableFamily)(mu) = Normal(mu, 1)

failure(f) = try
    f()
    nothing
catch err
    sprint(showerror, err)
end
compiled(mod, body) = Core.eval(mod, Expr(:macrocall, Symbol("@brm"), LineNumberNode(1), body))
stan(mod, brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod).model)

@testset "a module variable named like the response wrapper" begin
    message = failure(() -> brm(BODY; mod=Shadowed)(DATA))
    @test message isa String
    @test startswith(message, "@brm: the formula calls `censored(...)`, but module " *
                              "`$Shadowed` binds `censored` to a 3-element Vector{Bool}, " *
                              "which cannot be called.")
    @test occursin("shadows `BayesianRegressionModels.censored`, the function the " *
                   "formula means", message)
    @test occursin("rename the variable, or write the call as " *
                   "`BayesianRegressionModels.censored(...)`", message)
    # `@brm` written in the module resolves the head the same way.
    @test failure(() -> compiled(Shadowed, BODY)(DATA)) == message
end

@testset "a module variable named like a family" begin
    message = failure(() -> brm(BODY; mod=ShadowedFamily)(DATA))
    @test message isa String
    @test startswith(message, "@brm: the formula calls `Normal(...)`, but module " *
                              "`$ShadowedFamily` binds `Normal` to a Float64")
    @test occursin("shadows `BayesianRegressionModels.Normal`, the type the formula means",
                   message)
end

# The variable is a local of the code around `@brm`, not a module global.
function local_shadow(df)
    censored = [true, false, false]
    builder = @brm begin
        mu ~ 1 + x
        y ~ censored(Normal(mu, 1); lower=lo)
    end
    builder(df)
end

@testset "a local named like the wrapper, and a qualified head" begin
    message = failure(() -> local_shadow(DATA))
    @test message isa String
    @test startswith(message, "@brm: the formula calls `censored(...)`, but `censored` " *
                              "holds a 3-element Vector{Bool} where the block reads it")
    dotted = quote
        mu ~ 1 + x
        y ~ Shadowed.censored(Normal(mu, 1); lower=lo)
    end
    message = failure(() -> brm(dotted; mod=Shadowed)(DATA))
    @test message isa String
    @test startswith(message, "@brm: the formula calls `Shadowed.censored(...)`, but " *
                              "`Shadowed.censored` holds a 3-element Vector{Bool}")
    @test occursin("Rename the variable, or bind `Shadowed.censored`", message)
end

@testset "controls: unshadowed, qualified and callable heads build" begin
    clean = brm(BODY; mod=Clean)(DATA)
    @test clean isa BRMI
    # The remedy the message names builds the same model as the unshadowed
    # module: the qualified head is the wrapper.
    qualified = brm(QUALIFIED; mod=Shadowed)(DATA)
    @test qualified isa BRMI
    @test stan(Shadowed, qualified) == stan(Clean, clean)
    @test stan(Shadowed, compiled(Shadowed, QUALIFIED)(DATA)) == stan(Clean, clean)
    # A value with a call method is a callable head, whatever its type.
    functor = CallableFamily()
    @test !(functor isa Function)
    @test BRM._brm_formula_callee(:family, functor, Main) === functor
    @test BRM._brm_formula_callee(:Normal, Normal, Main) === Normal
    @test BRM._brm_formula_callee(:censored, censored, Main) === censored
end
