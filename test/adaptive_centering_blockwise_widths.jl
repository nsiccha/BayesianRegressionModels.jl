# Adaptive centering: the block-level reverse pass at every block width.
#
# `_blockwise_objective` differentiates one reverse pass per block, keeping a
# level's innovations in an `NTuple` read at a runtime index. Which code LLVM makes
# of such a tuple depends on its length, so a gradient check at one width does not
# clear another: with five or more margins in a derived-scale block Enzyme cannot
# type the tuple's reloaded stack slot and raises `EnzymeNoTypeError`, while four
# margins pass. That rejection of correct code is repaired in Enzyme, so this file
# needs a test pin of Enzyme that carries the repair.
# `adaptive_centering_blockwise_gradient.jl` covers the plans with compiled models;
# this file sweeps the width on a toy Gaussian density, so it needs no Stan
# toolchain.
#
# It checks, at each width and each kind of derived scale:
#
# - the plan takes the block-level pass;
# - its log density equals WarmupHMC's generic path exactly, its gradient to
#   rounding, and both agree with central differences of the log density;
# - inputs are left unchanged.
#
# Run: julia --startup-file=no --project=test test/adaptive_centering_blockwise_widths.jl

using Test
using Random
using BayesianRegressionModels
using Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems
using WarmupHMC

const BRM = BayesianRegressionModels
const ENZYME_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCEnzymeExt)
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

struct ToyGaussian
    d::Int
end
LogDensityProblems.dimension(p::ToyGaussian) = p.d
LogDensityProblems.capabilities(::Type{ToyGaussian}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::ToyGaussian, y) = -sum(abs2, y) / 2
LogDensityProblems.logdensity_and_gradient(::ToyGaussian, y) = (-sum(abs2, y) / 2, -collect(y))

# One correlated block of `K` margins and `G` levels, alone in a joint plan, every
# margin's log-scale carrying the named atoms. Coordinates: reference scale, R2,
# simplex, bounded scale, Cholesky factor, effects.
function toy_plan(K, G, atoms)
    reference, r2 = 1, 2
    simplex = collect(3:K + 1)
    bounded = K + 2
    cholesky = collect(bounded .+ (1:K * (K - 1) ÷ 2))
    start = isempty(cholesky) ? bounded : last(cholesky)
    effects = reshape(collect(start .+ (1:K*G)), K, G)
    scales = map(1:K) do k
        BRM._AdaptiveLogScale(0.0,
            :linear in atoms ? [(1.0, reference)] : Tuple{Float64,Int}[],
            :unit in atoms ? [(0.5, 0.5, r2)] : Tuple{Float64,Float64,Int}[],
            :simplex in atoms ? [BRM._AdaptiveSimplexLog(0.5, simplex, k)] :
                BRM._AdaptiveSimplexLog[],
            :bounded in atoms ? [(1.0, bounded, 0.0, 100.0)] :
                Tuple{Float64,Int,Float64,Float64}[])
    end
    ranef = BRM.RanefBlock(:r_toy, :correlated, :g, nothing, nothing, collect(1:G),
                           K, G, :z_toy, true, false)
    block = BRM.R2D2AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky, scales)
    state, ir = AC_EXT._adaptive_joint_centering_reparametrizer(
        BRM.AdaptiveCenteringBlock[], BRM._HSGPAdaptiveCenteringBlock[], [block])
    controls = rand(Xoshiro(K), length(ir.pairs))
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    AC_EXT._sync_sources!(state, ir)
    d = last(effects)
    WarmupHMC.ReparametrizedProblem(ir, ToyGaussian(d), AutoEnzyme()), d
end

function central_gradient(problem, x; h=1e-6)
    y = copy(x)
    map(eachindex(x)) do i
        y[i] = x[i] + h
        up = LogDensityProblems.logdensity(problem, y)
        y[i] = x[i] - h
        down = LogDensityProblems.logdensity(problem, y)
        y[i] = x[i]
        (up - down) / 2h
    end
end

generic_gradient(problem, x) = invoke(LogDensityProblems.logdensity_and_gradient,
    Tuple{WarmupHMC.ReparametrizedProblem,AbstractVector}, problem, x)

const ATOMS = [
    (:linear,),                             # a sampled scale in derived form
    (:bounded,),                            # a bounded sampled scale
    (:linear, :unit, :simplex),             # an R2D2 scale
    (:linear, :unit, :simplex, :bounded),   # an R2D2 scale over a bounded reference
]

# One outer testset, so a failing width reports without stopping the widths after it.
@testset "block-level pass at every width" begin
    @testset "block-level pass at K = $K, atoms $atoms" for K in 1:12,
                                                             atoms in ATOMS
        problem, d = toy_plan(K, 4, atoms)
        @test which(LogDensityProblems.logdensity_and_gradient,
            (typeof(problem), Vector{Float64})).module === ENZYME_EXT
        for seed in 1:2
            x = 0.3 .* randn(Xoshiro(seed), d)
            before = copy(x)
            lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
            @test x == before
            lp_generic, gradient_generic = generic_gradient(problem, x)
            @test lp == lp_generic
            @test gradient ≈ gradient_generic rtol=1e-11 atol=1e-11
            @test LogDensityProblems.logdensity(problem, x) == lp
            @test gradient ≈ central_gradient(problem, x) rtol=1e-6 atol=1e-6
        end
    end
end
