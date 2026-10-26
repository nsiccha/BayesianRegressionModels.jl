# Adaptive centering: the block-level reverse pass for plans wider than K<=2
# (snag warmuphmc-adapti-e35d58ad).
#
# WarmupHMC differentiates a per-pair plan pair by pair, and Enzyme's generic
# reverse pass allocates its tape per accessor call (23 MiB per gradient on a
# 26k-pair ordinary plan, 68 MiB on the joint plan). BRM's WarmupHMC/Enzyme
# extension replaces that, for an ordinary plan wider than K<=2 and for a joint
# plan of Gaussian random-effect pairs only, with one reverse pass over the map
# written block by block. This file pins, at varied controls and positions:
#
# - which plans take the block-level pass and which keep WarmupHMC's generic
#   path (Student-t mixing, HSGP cells);
# - its log density equals the generic path's exactly, its gradient to rounding;
# - it allocates a small fraction of what the generic path allocates;
# - the candidate-scoring frame, which builds a Gaussian block's factor once per
#   block instead of once per level, equals the per-level frame bit for bit.
#
# Run: julia --project=test test/adaptive_centering_blockwise_gradient.jl

using Test
using Random
using BayesianRegressionModels
using Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems
using StanBlocks
using WarmupHMC
using Distributions: Exponential, LKJCholesky, Normal, TDist

const BRM = BayesianRegressionModels
const ENZYME_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCEnzymeExt)
const AC_EXT = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

# The declared-order scoring frame as it was built before: the factor and the
# work arrays rebuilt for every level.
function per_level_frame(state, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location, scale, innovation, invariant_gradient =
        (Vector{T}(undef, n) for _ in 1:4)
    for (bi, block) in enumerate(state.blocks), g in 1:block.ranef.n_groups
        K = block.ranef.n_terms
        C = BRM._adaptive_block_cholesky(position, block, g)
        A = zeros(T, K, K)
        z = Vector{T}(undef, K)
        for j in 1:K
            p = state.pair_lookup[bi][j, g]
            c = source[p]
            m = zero(T)
            for l in 1:j-1
                m += C[j, l] * z[l]
                A[j, l] = c * C[j, l]
            end
            s = C[j, j]
            z[j] = (position[block.effects[j, g]] - c * m) / s^c
            A[j, j] = s^c
            location[p] = m
            scale[p] = s
            innovation[p] = z[j]
        end
        h = transpose(A) * gradient[block.effects[:, g]]
        for j in 1:K
            invariant_gradient[state.pair_lookup[bi][j, g]] = h[j]
        end
    end
    (; source, location, scale, innovation, invariant_gradient)
end

ranef_state(state::AC_EXT._BRMRanefCenteringState) = state
ranef_state(state::AC_EXT.BRMJointAdaptiveCenteringState) = state.ranef

const N_ROWS = 600
const BLOCKWISE_DF = let rng = Xoshiro(20261026)
    g = repeat(1:40, inner=N_ROWS ÷ 40)
    (; g, b=mod.(g, 3) .+ 1, h=rand(rng, 1:300, N_ROWS),
       x=round.(randn(rng, N_ROWS); digits=3), t=round.(randn(rng, N_ROWS); digits=3),
       y=round.(randn(rng, N_ROWS); digits=3))
end

# label => (builder, takes the block-level pass)
const BLOCKWISE_VARIANTS = [
    "ordinary, three margins and many one-margin levels" => (@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + t + (1 + x + t | g) + (1 | h)
        y ~ Normal(mu, sigma)
    end), true),
    "stratified gr(g, by = b) beside a correlated block" => (@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + t + (1 + x + t | h) + (1 | gr(g, by=b))
        y ~ Normal(mu, sigma)
    end), true),
    "joint plan: known correlation beside ordinary blocks" => (@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + t + (1 + x + t | p | g) + (1 | h)
        cor(:, p) = [1 0 0; 0 1 0; 0 0 1]
        y ~ Normal(mu, sigma)
    end), true),
    "joint plan: known SD and correlation" => (@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g) + (1 | h)
        sd(:, p) = 0.4
        cor(:, p) = [1 0.3; 0.3 1]
        y ~ Normal(mu, sigma)
    end), true),
    "joint plan with Student-t mixing keeps the generic path" => (@brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g) + (1 | h)
        ranef(:, p) ~ TDist(4)
        y ~ Normal(mu, sigma)
    end), false),
]

generic_gradient(wrapped, x) = invoke(LogDensityProblems.logdensity_and_gradient,
    Tuple{WarmupHMC.ReparametrizedProblem,AbstractVector}, wrapped, x)

@testset "block-level centred gradient: $label" for (label, (builder, blockwise)) in
        BLOCKWISE_VARIANTS
    sb = SBBRMI(builder(BLOCKWISE_DF); total_groups=(), mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    d = LogDensityProblems.dimension(problem)
    n_cells = length(WarmupHMC.reparametrizer(
        adaptive_centering_problem(sb, problem, AutoEnzyme())).pairs)
    wrapped = adaptive_centering_problem(sb, problem, AutoEnzyme();
        centeredness=rand(Xoshiro(3), n_cells))
    # Both plan kinds dispatch to the extension; a joint plan with Student-t
    # mixing then hands the call back to WarmupHMC's generic method, so its
    # gradient is the generic one bit for bit.
    method = which(LogDensityProblems.logdensity_and_gradient,
        (typeof(wrapped), Vector{Float64}))
    @test method.module === ENZYME_EXT
    for seed in 1:3
        x = 0.3 .* randn(Xoshiro(seed), d)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
        lp_generic, gradient_generic = generic_gradient(wrapped, x)
        @test lp == lp_generic
        if blockwise
            @test gradient ≈ gradient_generic rtol=1e-11 atol=1e-11
        else
            @test gradient == gradient_generic
        end
        @test LogDensityProblems.logdensity(wrapped, x) == lp
    end
    if blockwise
        x = 0.3 .* randn(Xoshiro(9), d)
        LogDensityProblems.logdensity_and_gradient(wrapped, x)
        generic_gradient(wrapped, x)
        bytes = @allocated LogDensityProblems.logdensity_and_gradient(wrapped, x)
        bytes_generic = @allocated generic_gradient(wrapped, x)
        @test bytes < bytes_generic / 4
    end
    state = ranef_state(first(WarmupHMC.reparametrizer(wrapped).pairs).second.args[1].state)
    for seed in 1:2
        x = 0.3 .* randn(Xoshiro(10 + seed), d)
        gradient = randn(Xoshiro(20 + seed), d)
        frame = AC_EXT._centering_frame(state, x, gradient)
        reference = per_level_frame(state, x, gradient)
        for field in propertynames(reference)
            @test getproperty(frame, field) == getproperty(reference, field)
        end
    end
end

# The HSGP joint plan keeps WarmupHMC's generic path (its cells are not
# random-effect pairs).
@testset "joint plan with HSGP cells keeps the generic path" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(x; k=4) + (1 + x | g)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(BLOCKWISE_DF); total_groups=(), mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    wrapped = adaptive_centering_problem(sb, problem, AutoEnzyme())
    x = 0.3 .* randn(Xoshiro(4), LogDensityProblems.dimension(problem))
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    lp_generic, gradient_generic = generic_gradient(wrapped, x)
    @test lp == lp_generic
    @test gradient == gradient_generic
end
