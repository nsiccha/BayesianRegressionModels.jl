module BayesianRegressionModelsWarmupHMCEnzymeExt

using BayesianRegressionModels
using Enzyme
using LogDensityProblems
using WarmupHMC

const AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)
const SmallLocation = AC_EXT.BRMAdaptiveCenteringArgument{:location,1}
const SmallLogScale = AC_EXT.BRMAdaptiveCenteringArgument{:log_scale,1}
const SmallArguments = Tuple{SmallLocation,SmallLogScale}
const FloatCentering = WarmupHMC.PartiallyCentered{Float64}
const SmallReparametrization = WarmupHMC.Reparametrization{
    FloatCentering,FloatCentering,SmallArguments,
}
const SmallIndexedReparametrization = WarmupHMC.IndexedReparametrization{
    Vector{Pair{Int,SmallReparametrization}},
}

# This pullback deliberately mirrors Enzyme's own floating-point graph because
# WarmupHMC makes discrete centering decisions from these gradients. Enzyme
# 0.13.200 changed tanh's derivative from `seed / cosh(x)^2` to
# `(1 - tanh(x)^2) * seed`; the two are algebraically equal but not bit-equal.
const ENZYME_TANH_PULLBACK_FROM_PRIMAL = pkgversion(Enzyme) >= v"0.13.200"

# Differentiating the generic indexed loop still leaves one Enzyme rule call per
# accessor.  For BRM's scalar and intercept/slope blocks, run that same reverse
# pass directly.  The pair vector's exact concrete type makes this dispatch
# exclusive to the BRM K<=2 plan; wider blocks and every non-BRM reparametrizer
# retain WarmupHMC's generic DifferentiationInterface path.
function LogDensityProblems.logdensity_and_gradient(
    problem::WarmupHMC.ReparametrizedProblem{
        SmallIndexedReparametrization,P,B,
    },
    x::AbstractVector,
) where {P,B}
    ljac, y = problem.reparametrizer(x)
    ld, model_gradient =
        LogDensityProblems.logdensity_and_gradient(problem.problem, y)
    gradient = zeros(eltype(model_gradient), length(model_gradient))
    state = first(problem.reparametrizer.pairs).second.args[1].state
    T = eltype(x)
    # Pair construction is block-major, then group-major, then term-major.
    # Reverse that exact order, cache only values that the frozen forward graph
    # repeats identically, and retain its literal pullback operation order.
    for bi in Iterators.reverse(eachindex(state.blocks))
        block = state.blocks[bi]
        K = block.ranef.n_terms
        tau1 = exp(x[block.log_scales[1]])
        C11 = tau1 * one(T)
        if K == 2
            raw_index = only(block.cholesky_free)
            scale2_index = block.log_scales[2]
            raw = x[raw_index]
            stick = one(T)
            corr = tanh(raw)
            L21 = stick * corr
            variance = one(T) - corr * corr
            root = sqrt(variance)
            stick *= root
            tau2 = exp(x[scale2_index])
            C21 = tau2 * L21
            C22 = tau2 * stick
            tanh_pullback_factor = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                one(T) - corr * corr : cosh(raw) * cosh(raw)
        end

        for group in block.ranef.n_groups:-1:1, term in K:-1:1
            pair_number = state.pair_lookup[bi][term, group]
            index, value = problem.reparametrizer.pairs[pair_number]
            source = value.source.c
            target = value.target.c
            model_seed = model_gradient[index]

            # A pair centred on a population coefficient adds it to its
            # location, and term 1's coefficient to term 2's innovation
            # (`AC_EXT._with_location`); a pair centred on 0 keeps the exact
            # arithmetic below.
            own_location = state.pair_location[pair_number]
            if term == 1
                location = zero(T)
                own_location == 0 || (location = location + x[own_location])
                log_scale = log(C11)
            else
                first_pair = state.pair_lookup[bi][1, group]
                first_location = state.pair_location[first_pair]
                c = state.sources[first_pair]
                u = x[block.effects[1, group]]
                m = zero(T)
                first_location == 0 || (m = m + x[first_location])
                numerator = u - c * m
                denominator = C11^c
                innovation = numerator / denominator
                location = zero(T)
                location += C21 * innovation
                own_location == 0 || (location = location + x[own_location])
                log_scale = log(C22)
            end

            delta = target - source
            local_ljac = log_scale * delta
            shifted = x[index] - source * location
            transport_scale = exp(local_ljac)

            dlocation = zero(T)
            dshifted = model_seed * transport_scale
            dtransport = model_seed * shifted
            dlocal_ljac = one(T)
            dlocal_ljac += dtransport * transport_scale
            dlog_scale = dlocal_ljac * delta
            dlocation += (-dshifted) * source
            dlocation += model_seed * target

            own_location == 0 || (gradient[own_location] += dlocation)
            if term == 1
                dC11 = dlog_scale / C11
                dtau1 = dC11 * one(T)
                gradient[block.log_scales[1]] += dtau1 * tau1
            else
                dC22 = dlog_scale / C22
                dtau2 = dC22 * stick
                dstick = dC22 * tau2
                droot = dstick * one(T)
                dvariance = droot / (root * 2)
                dproduct = -dvariance
                dcorr = dproduct * corr
                dcorr += dproduct * corr
                draw = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                    tanh_pullback_factor * dcorr : dcorr / tanh_pullback_factor
                gradient[raw_index] += draw
                gradient[scale2_index] += dtau2 * tau2

                effect_index = block.effects[1, group]
                dproduct = dlocation
                dC21 = dproduct * innovation
                dinnovation = dproduct * C21
                dnumerator = dinnovation / denominator
                dtau2 = dC21 * L21
                dL21 = dC21 * tau2
                dcorr = dL21 * one(T)
                draw = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                    tanh_pullback_factor * dcorr : dcorr / tanh_pullback_factor
                dscale1 = ((((dlocation * C21) * (numerator * -c)) *
                    C11^(c - one(c))) * tau1) /
                    (denominator * denominator)
                gradient[raw_index] += draw
                gradient[block.log_scales[1]] += dscale1
                gradient[scale2_index] += dtau2 * tau2
                gradient[effect_index] += dnumerator
                first_location == 0 ||
                    (gradient[first_location] += (-dnumerator) * c)
            end
            gradient[index] += dshifted
        end
    end
    for index in eachindex(gradient)
        index in state.effect_indices ||
            (gradient[index] += model_gradient[index])
    end
    ljac + ld, gradient
end

# ---- Wider plans: one reverse pass over the map, block by block -------------
#
# Above K<=2, WarmupHMC differentiates the plan pair by pair: each pair's two
# accessor calls go through Enzyme's generic reverse pass, which allocates its
# tape per call. On ArbeitsplatzAtlas atlas-16 (26,402 pairs, widest block 6)
# that was 23 MiB per gradient on the ordinary plan and 68 MiB when known-value
# blocks put it on the joint plan; with four chains on threads, Julia's GC then
# took 34-50 % of the wall time (snag warmuphmc-adapti-e35d58ad). The same
# map, written block by block with each row walked inline as `_block_location`
# walks it, reverses in one pass: 0.7 MiB there.
#
# It covers an ordinary plan wider than K<=2, and a joint plan whose cells are
# all random-effect pairs (no HSGP, total, S2Z or car cells) of Gaussian blocks.
# Every other plan keeps WarmupHMC's generic path. The log density is
# WarmupHMC's own transform; the gradient equals the generic one to rounding,
# not bitwise (the reverse pass sums the same terms in another order).
const WidePairs{N} = Vector{Pair{Int,WarmupHMC.Reparametrization{
    FloatCentering,FloatCentering,
    Tuple{AC_EXT.BRMAdaptiveCenteringArgument{:location,N},
          AC_EXT.BRMAdaptiveCenteringArgument{:log_scale,N}},
}}}
const JointPairs{N} = Vector{Pair{Int,WarmupHMC.Reparametrization{
    FloatCentering,FloatCentering,
    Tuple{AC_EXT.BRMJointAdaptiveCenteringArgument{:location,N},
          AC_EXT.BRMJointAdaptiveCenteringArgument{:log_scale,N}},
}}}

function LogDensityProblems.logdensity_and_gradient(
    problem::WarmupHMC.ReparametrizedProblem{
        WarmupHMC.IndexedReparametrization{WidePairs{N}},P,B,
    },
    x::AbstractVector,
) where {N,P,B}
    _blockwise_logdensity_and_gradient(problem,
        first(problem.reparametrizer.pairs).second.args[1].state, x)
end

function LogDensityProblems.logdensity_and_gradient(
    problem::WarmupHMC.ReparametrizedProblem{
        WarmupHMC.IndexedReparametrization{JointPairs{N}},P,B,
    },
    x::AbstractVector,
) where {N,P,B}
    state = first(problem.reparametrizer.pairs).second.args[1].state
    if state.n_hsgp + state.n_scalar + length(state.car.sources) == 0 &&
            !any(AC_EXT._block_mixed, state.ranef.blocks)
        return _blockwise_logdensity_and_gradient(problem, state.ranef, x)
    end
    invoke(LogDensityProblems.logdensity_and_gradient,
        Tuple{WarmupHMC.ReparametrizedProblem,AbstractVector}, problem, x)
end

function _blockwise_logdensity_and_gradient(problem, state, x)
    ljac, y = problem.reparametrizer(x)
    ld, model_gradient =
        LogDensityProblems.logdensity_and_gradient(problem.problem, y)
    position = convert(Vector{Float64}, x)
    gradient = zeros(Float64, length(position))
    Enzyme.autodiff(Enzyme.Reverse, _blockwise_objective, Enzyme.Active,
        Enzyme.Duplicated(position, gradient), Enzyme.Const(state),
        Enzyme.Const(model_gradient))
    for index in eachindex(gradient)
        index in state.effect_indices ||
            (gradient[index] += model_gradient[index])
    end
    ljac + ld, gradient
end

# `sum(seed[i] * y[i]) + log|J|` over the plan's pairs, `y` the map of each pair
# in WarmupHMC's per-pair spelling (`warmuphmc-use` §5c) with the location and
# log-scale `_block_location` and the log-scale accessor compute. Only flat
# arrays and an `NTuple{N}` of innovations are read inside the loops, the
# Julia 1.10 rule `_block_location` states.
function _blockwise_objective(x, state, seed)
    acc = zero(eltype(x))
    for bi in eachindex(state.blocks)
        acc += _blockwise_objective(x, state, bi, seed)
    end
    acc
end

function _blockwise_objective(x, state::AC_EXT._BRMRanefCenteringState{N}, bi,
                              seed) where {N}
    T = eltype(x)
    block = state.blocks[bi]
    K = block.ranef.n_terms
    t = block.target_c
    lookup = state.pair_lookup[bi]
    acc = zero(T)
    for g in 1:block.ranef.n_groups
        innovations = ntuple(_ -> zero(T), Val(N))
        for j in 1:K
            tau = AC_EXT._block_tau(x, block, j)
            stick = one(T)
            offset = (j - 1) * (j - 2) ÷ 2
            m = zero(T)
            for l in 1:j-1
                z = AC_EXT._block_cpc(x, block, offset + l)
                m += (tau * (stick * z)) * innovations[l]
                stick *= sqrt(one(T) - z * z)
            end
            s = tau * stick
            c = state.sources[lookup[j, g]]
            index = block.effects[j, g]
            u = x[index]
            # The pair's source-frame location (`AC_EXT._with_location`).
            loc = AC_EXT._with_location(m, x, state, lookup[j, g])
            j < K && (innovations = Base.setindex(innovations, (u - c * loc) / s^c, j))
            log_scale = log(s)
            acc += seed[index] * (t * loc + (u - c * loc) * exp(log_scale * (t - c)))
            acc += log_scale * (t - c)
        end
    end
    acc
end

end
