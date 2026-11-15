# test/unbound_formula_names.jl — BRM's own names in a block whose module does
# not bind them (snag brm-ordinal-link-ca8cd39c).
#
# Run: julia --startup-file=no --project=test test/unbound_formula_names.jl
#
# Julia leaves a name undefined in a module that `using`s two packages that
# export it. ReactiveKernelsPPL exports `LogitLink`, `ProbitLink` and
# `CloglogLink` (values of its `LinkFunction` enum), and GLM.jl exports link
# types of the same names. With brm-use's RK setup, `using
# BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL`, a block
# reading `Ordinal(Cumulative(), ProbitLink(), eta)` failed: the builder with
# `UndefVarError: ProbitLink not defined`, an in-cell observation at RKBRMI
# construction with `RK backend: ProbitLink names nothing in module Main`.
# A formula call head, and a name a cell reads, that BRM exports now mean BRM's
# binding when the block's module binds the name to nothing: the name is
# ambiguous there, or the module only `import`s BRM. A binding the module does
# have still wins, and a name BRM does not export is still undefined.

using Test
using BayesianRegressionModels, StanBlocks
using ReactiveKernels           # loads BRM's RK extension
import ReactiveKernelsPPL
using Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels

# A third package's ordinal names, shaped like GLM.jl's link types.
module OtherLinks
struct ProbitLink end
struct StoppingRatio end
struct Shared end
export ProbitLink, StoppingRatio, Shared
end
module MoreLinks
struct Shared end
export Shared
end
# A third package's families and helpers of BRM's names.
module OtherFamilies
struct Normal end
struct Exponential end
groups(x) = x
export Normal, Exponential, groups
end

module Clean
using BayesianRegressionModels
end
module WithRKPPL
using BayesianRegressionModels, ReactiveKernelsPPL
end
module WithOther
using BayesianRegressionModels, ..OtherLinks, ..MoreLinks
end
module WithOtherFamilies
using BayesianRegressionModels, ..OtherFamilies
end
module ImportOnly
import BayesianRegressionModels
end
module OwnBinding
using BayesianRegressionModels
const ProbitLink = BayesianRegressionModels.LogitLink
end

const DATA = let
    examiner = repeat(1:3, inner=8)
    x = [0.3, -1.1, 0.8, 0.1, -0.4, 1.6, -0.7, 0.5, 0.9, -0.2, 0.0, 1.2,
         -1.5, 0.6, 0.2, -0.9, 0.4, -0.3, 1.0, -0.6, 0.7, -1.2, 0.05, 0.35]
    y = [3, 1, 4, 2, 2, 5, 1, 3, 4, 2, 3, 5, 1, 4, 3, 2, 3, 2, 4, 1, 4, 1, 3, 3]
    z = 0.5 .* x .+ [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.4, -0.3, 0.1, 0.0, -0.2, 0.3,
                     0.2, -0.1, 0.0, 0.1, -0.4, 0.3, 0.2, -0.2, 0.1, 0.0, -0.1, 0.2]
    (; y, z, x, examiner, exids=collect(1:3))
end

const ORDINAL_BODIES = (
    cumulative_probit=quote
        eta ~ 0 + x
        y ~ Ordinal(Cumulative(), ProbitLink(), eta)
    end,
    stopping_cloglog=quote
        eta ~ 0 + x
        y ~ Ordinal(StoppingRatio(), CloglogLink(), eta)
    end,
    cumulative_logit_ranef=quote
        eta ~ 0 + x + (1 | examiner)
        y ~ Ordinal(Cumulative(), LogitLink(), eta)
    end,
)
# A cell reads `Normal`, model scope `Exponential` and `groups`: names
# `WithOtherFamilies` leaves undefined.
const CELL_BODY = quote
    eta ~ 0 + x + (1 | examiner)
    sigma ~ Exponential(1)
    rows = groups(examiner => exids)
    @plate for i in eachindex(exids)
        z[rows[i]] ~ Normal(eta[rows[i]], sigma)
    end
end

macro_head(mod) = isdefined(mod, Symbol("@brm")) ? Symbol("@brm") :
    Expr(:., :BayesianRegressionModels, QuoteNode(Symbol("@brm")))
compiled(mod, body) = Core.eval(mod, Expr(:macrocall, macro_head(mod), LineNumberNode(1), body))
stan(mod, brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod).model)
rk_source(brmi) = sprint(Base.show_unquoted, BRM._rk_emit_ast(RKBRMI(brmi).plan).main)
failure(f) = try
    f()
    nothing
catch err
    err
end

@testset "unbound formula names" begin

    @testset "the module binds the ordinal names to nothing" begin
        # The precondition, in each module: Julia gives the name no binding.
        @test !isdefined(WithRKPPL, :ProbitLink) && !isdefined(WithRKPPL, :LogitLink) &&
              !isdefined(WithRKPPL, :CloglogLink)
        @test !isdefined(WithOther, :ProbitLink) && !isdefined(WithOther, :StoppingRatio)
        @test !isdefined(ImportOnly, :ProbitLink) && !isdefined(ImportOnly, :Ordinal)
        @test !isdefined(WithOtherFamilies, :Normal) && !isdefined(WithOtherFamilies, :groups)
        @test isdefined(Clean, :ProbitLink)
    end

    @testset "model-scope Ordinal: $label" for (label, body) in pairs(ORDINAL_BODIES)
        reference = brm(body; mod=Clean)(DATA)
        reference_stan = stan(Clean, reference)
        reference_rk = rk_source(reference)
        # `@brm` compiled in a module that only `import`s BRM is not checked here:
        # its body calls `@getproperty` and the other plumbing macros by name.
        for (mod, builders) in ((WithRKPPL, (:interpreted, :compiled)),
                                (WithOther, (:interpreted, :compiled)),
                                (ImportOnly, (:interpreted,)))
            @testset "$(nameof(mod)), $(builder)" for builder in builders
                brmi = builder === :interpreted ? brm(body; mod)(DATA) : compiled(mod, body)(DATA)
                @test brmi isa BRMI
                @test stan(mod, brmi) == reference_stan
                @test rk_source(brmi) == reference_rk
            end
        end
    end

    @testset "a binding the module has still wins" begin
        # `OwnBinding.ProbitLink` is BRM's `LogitLink`: the model is the logit one.
        probit = ORDINAL_BODIES.cumulative_probit
        logit = quote
            eta ~ 0 + x
            y ~ Ordinal(Cumulative(), LogitLink(), eta)
        end
        own = stan(OwnBinding, brm(probit; mod=OwnBinding)(DATA))
        @test own == stan(Clean, brm(logit; mod=Clean)(DATA))
        @test own != stan(Clean, brm(probit; mod=Clean)(DATA))
        @test stan(OwnBinding, compiled(OwnBinding, probit)(DATA)) == own
    end

    @testset "a name BRM does not export stays undefined" begin
        # `Shared` is exported by two other packages and by no BRM binding.
        body = quote
            eta ~ 0 + x
            y ~ Ordinal(Cumulative(), Shared(), eta)
        end
        @test !isdefined(WithOther, :Shared)
        for err in (failure(() -> brm(body; mod=WithOther)(DATA)),
                    failure(() -> compiled(WithOther, body)(DATA)))
            @test err isa UndefVarError && err.var === :Shared
        end
    end

    @testset "names a cell reads, on SBBRMI and RKBRMI" begin
        reference_brmi = brm(CELL_BODY; mod=Clean)(DATA)
        reference_stan = stan(Clean, reference_brmi)
        reference = RKBRMI(reference_brmi)
        problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        reference_problem = problem(reference)
        u = collect(range(-0.4, 0.5; length=LogDensityProblems.dimension(reference_problem)))
        value = LogDensityProblems.logdensity(reference_problem, u)
        @test isfinite(value)
        for (mod, builder) in ((WithOtherFamilies, :interpreted), (WithOtherFamilies, :compiled),
                               (ImportOnly, :interpreted))
            @testset "$(nameof(mod)), $(builder)" begin
                brmi = builder === :interpreted ? brm(CELL_BODY; mod)(DATA) :
                    compiled(mod, CELL_BODY)(DATA)
                @test stan(mod, brmi) == reference_stan
                rk = RKBRMI(brmi)
                @test sprint(Base.show_unquoted, BRM._rk_emit_ast(rk.plan).main) ==
                      sprint(Base.show_unquoted, BRM._rk_emit_ast(reference.plan).main)
                @test LogDensityProblems.logdensity(problem(rk), u) == value
            end
        end
    end

end
