module BayesianRegressionModelsWarmupHMCExt

using BayesianRegressionModels
using WarmupHMC

const BRM = BayesianRegressionModels

# Random-effect block states: ordinary blocks (`BRMAdaptiveCenteringState`) and
# R2D2 blocks (`BRMR2D2AdaptiveCenteringState`) share one cell layout and one
# triangular map; they differ only in where `tau` comes from (`_block_tau`).
# `N` is the innovation capacity `_block_location` keeps on the stack: the
# widest block's K-1, and at least 1, so every K<=2 plan shares one concrete
# type (the exact pullback in BayesianRegressionModelsWarmupHMCEnzymeExt
# dispatches on `N == 1`).
abstract type _BRMRanefCenteringState{N} end

_innovation_capacity(blocks) =
    max(1, maximum((b.ranef.n_terms for b in blocks); init=1) - 1)

mutable struct BRMAdaptiveCenteringState{N} <: _BRMRanefCenteringState{N}
    blocks::Vector{BRM.AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_mixing::Vector{Int}
    pair_location::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
    pair_factor_start::Vector{Int}
    pair_factors::Vector{Int}
    pair_container_start::Vector{Int}
    pair_containers::Vector{Int}
    pair_container_mask::Vector{UInt8}
    extended::Bool
end

# Pairs are block-major, then group-major, then term-major. `pair_mixing[p]`
# is the coordinate `log w[g]` of a Student-t pair's level, `0` for a Gaussian
# block: accessors read that integer once per call, never the block's `mixing`
# vector inside their loops (see `_pair_mixing_scale`). `pair_location[p]` is
# the coordinate of the population coefficient pair `p` is centred on, `0` for
# a pair centred on 0 (`BRM._adaptive_ranef_locations`, one entry per term of
# each block; `locations[bi]` aligns with `blocks[bi]`, `nothing` for none).
#
# A pair whose level sits in a factor level or a containing grouping
# (`BRM._AdaptiveRanefLocations`, snag brm-zone-8-adapt-1b407672) adds, in CSR
# form, the factor levels' coordinates (`pair_factors`) and its containing pairs
# (`pair_containers`, coarsest first). A containing pair always comes earlier in
# pair order, so WarmupHMC's inverse, which rebuilds the source pair by pair,
# has rebuilt it before it is read. `pair_container_mask[e]` marks which of the
# earlier entries of the same list are themselves containers of entry `e`'s
# pair. `originals[bi]` is the block `locations` names for `blocks[bi]` (the
# joint wrapper maps derived copies of the ordinary blocks).
function _ranef_pair_layout(blocks, locations=nothing, originals=blocks)
    pair_blocks = Int[]
    pair_terms = Int[]
    pair_groups = Int[]
    pair_mixing = Int[]
    pair_location = Int[]
    pair_lookup = [zeros(Int, b.ranef.n_terms, b.ranef.n_groups) for b in blocks]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), g in 1:block.ranef.n_groups,
            k in 1:block.ranef.n_terms
        push!(pair_blocks, bi)
        push!(pair_terms, k)
        push!(pair_groups, g)
        push!(pair_mixing, _block_mixed(block) ? block.mixing[g] : 0)
        locs = isnothing(locations) ? nothing : locations[bi]
        push!(pair_location, isnothing(locs) ? 0 : locs[k])
        push!(sources, block.target_c)
        pair_lookup[bi][k, g] = length(sources)
    end
    effect_indices = BitSet(Iterators.flatten(vec(b.effects) for b in blocks))
    (pair_blocks, pair_terms, pair_groups, pair_mixing, pair_location, pair_lookup,
     sources, effect_indices,
     _ranef_container_layout(blocks, locations, originals, pair_blocks, pair_terms,
                             pair_groups, pair_lookup)...)
end

# Containing cells a location reads, at most this many per pair (one bit each in
# `pair_container_mask`).
const _MAX_LOCATION_CONTAINERS = 8

function _ranef_container_layout(blocks, locations, originals, pair_blocks, pair_terms,
                                 pair_groups, pair_lookup)
    n = length(pair_blocks)
    factor_start = ones(Int, n + 1)
    factors = Int[]
    container_start = ones(Int, n + 1)
    containers = Int[]
    mask = UInt8[]
    for p in 1:n
        bi, k, g = pair_blocks[p], pair_terms[p], pair_groups[p]
        locs = isnothing(locations) ? nothing : locations[bi]
        if locs isa BRM._AdaptiveRanefLocations && !isnothing(locs.factors)
            append!(factors, locs.factors[k, g])
            first_entry = length(containers) + 1
            for (block, term, group) in locs.containers[k, g]
                ci = findfirst(o -> o === block, originals)
                isnothing(ci) && error(
                    "BRM adaptive centering: block `$(blocks[bi].ranef.binding)` is " *
                    "centred on the effects of block `$(block.ranef.binding)`, which " *
                    "this wrapper does not map; refusing an inexact transform.")
                q = pair_lookup[ci][term, group]
                q < p || error(
                    "BRM adaptive centering: block `$(blocks[bi].ranef.binding)` is " *
                    "centred on the effects of block `$(block.ranef.binding)`, which " *
                    "is mapped after it; containing blocks must come first.")
                push!(containers, q)
            end
            entries = first_entry:length(containers)
            length(entries) <= _MAX_LOCATION_CONTAINERS || error(
                "BRM adaptive centering: a cell of block " *
                "`$(blocks[bi].ranef.binding)` sits in $(length(entries)) groupings; " *
                "at most $_MAX_LOCATION_CONTAINERS are supported.")
            # Each container's own containers sit in this list too (containment
            # is transitive and they carry the same term), and earlier.
            for e in entries
                a = containers[e]
                bits = 0x00
                for f in container_start[a]:container_start[a + 1] - 1
                    i = findfirst(==(containers[f]), view(containers, first_entry:e - 1))
                    isnothing(i) && error(
                        "BRM adaptive centering: block " *
                        "`$(blocks[bi].ranef.binding)`'s containing cells do not " *
                        "include those of their own containers; refusing an inexact " *
                        "transform.")
                    bits |= 0x01 << (i - 1)
                end
                push!(mask, bits)
            end
        end
        factor_start[p + 1] = length(factors) + 1
        container_start[p + 1] = length(containers) + 1
    end
    (factor_start, factors, container_start, containers, mask,
     !isempty(factors) || !isempty(containers))
end

BRMAdaptiveCenteringState(blocks, locations=nothing) =
    BRMAdaptiveCenteringState{_innovation_capacity(blocks)}(
        collect(blocks), _ranef_pair_layout(blocks, locations)...,
    )

# The per-block location vectors of `blocks` from the map
# `BRM._adaptive_ranef_locations` returns (keyed by the original blocks).
_block_locations(blocks, locations) =
    [get(locations, b, nothing) for b in blocks]

# A location is read, never transformed: a coefficient some cell rewrites
# would make the triangular Jacobian wrong.
function _check_locations_read_only(state, rewritten)
    for j in Iterators.flatten((state.pair_location, state.pair_factors))
        j == 0 || !(j in rewritten) || error(
            "BRM adaptive centering: a random-effect cell is centred on coordinate " *
            "$j, which another cell rewrites; refusing an inexact transform.")
    end
end

# Pair `p`'s source-frame location: the conditional mean `m` plus, for a pair
# centred on a population coefficient, that coefficient's value. It is read
# from the state's flat `pair_location`, and a pair centred on 0 keeps `m`
# itself, so its arithmetic is unchanged bit for bit. The coefficient is never
# rewritten by the map. `@inline`: an outlined call taking the state inside a
# differentiated loop is the Julia 1.10 GC hazard `_block_location` describes.
@inline function _with_location(m, x, state, p)
    _with_location(m, x, x, state, p)
end

# `src` holds the source frame of the pairs a location reads from containing
# cells (snag brm-zone-8-adapt-1b407672): `x` itself, except in the learned-order
# inverse, which rebuilds the source in another vector. Coefficients are read
# from `x` (no cell rewrites them).
@inline function _with_location(m, x, src, state, p)
    j = state.pair_location[p]
    loc = j == 0 ? m : m + x[j]
    (state.extended && (state.pair_factor_start[p] < state.pair_factor_start[p + 1] ||
                        state.pair_container_start[p] < state.pair_container_start[p + 1])) ||
        return loc
    loc + _contained_location(x, src, state, p)
end

# The rest of pair `p`'s total: the coordinates of the factor levels its level
# sits in, plus the effect of each containing cell `a`, coarsest first. A
# containing cell is a scalar (`K = 1`) cell, so its compiled coordinate is
# WarmupHMC's pair map `t*L + (u - c*L) * exp(log(s)*(t - c))` of its own source
# `u` and location `L` (its factor levels plus the effects of its own
# containers, read from the buffer), and its effect is `s` times that when the
# block is compiled non-centred (`t = 0`), that itself when centred (`t = 1`).
# Flat loads and an `NTuple` buffer only, `@inline`: the Julia 1.10 GC rule
# `_block_location` states.
@inline function _contained_location(x, src, state, p)
    T = promote_type(eltype(x), eltype(src))
    acc = zero(T)
    for f in state.pair_factor_start[p]:state.pair_factor_start[p + 1] - 1
        acc += x[state.pair_factors[f]]
    end
    lo = state.pair_container_start[p]
    buffer = ntuple(_ -> zero(T), Val(_MAX_LOCATION_CONTAINERS))
    for e in lo:state.pair_container_start[p + 1] - 1
        a = state.pair_containers[e]
        j = state.pair_location[a]
        L = j == 0 ? zero(T) : convert(T, x[j])
        for f in state.pair_factor_start[a]:state.pair_factor_start[a + 1] - 1
            L += x[state.pair_factors[f]]
        end
        bits = state.pair_container_mask[e]
        for i in 1:e - lo
            (bits >> (i - 1)) & 0x01 == 0x01 && (L += buffer[i])
        end
        block = state.blocks[state.pair_blocks[a]]
        s = _block_cholesky_entry(x, block, 1, 1) * _pair_mixing_scale(x, state, a)
        c = state.sources[a]
        t = block.target_c
        u = src[block.effects[1, state.pair_groups[a]]]
        y = t * L + (u - c * L) * exp(log(s) * (t - c))
        b = t == 0 ? s * y : y
        buffer = Base.setindex(buffer, b, e - lo + 1)
        acc += b
    end
    acc
end

# Derived-scale blocks — R2D2 blocks, and ordinary blocks inside the joint
# wrapper — keep their own state and argument types for the same reason HSGP
# and cdar cells do: the exact K<=2 Enzyme reverse pass dispatches on the
# ordinary state's concrete pair type and must never see a derived scale.
mutable struct BRMR2D2AdaptiveCenteringState{N} <: _BRMRanefCenteringState{N}
    blocks::Vector{BRM.R2D2AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_mixing::Vector{Int}
    pair_location::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
    effect_indices::BitSet
    pair_factor_start::Vector{Int}
    pair_factors::Vector{Int}
    pair_container_start::Vector{Int}
    pair_containers::Vector{Int}
    pair_container_mask::Vector{UInt8}
    extended::Bool
end

BRMR2D2AdaptiveCenteringState(blocks, locations=nothing) =
    BRMR2D2AdaptiveCenteringState{_innovation_capacity(blocks)}(
        collect(blocks), _ranef_pair_layout(blocks, locations)...,
    )

# A block's marginal scale `tau[i]`: a sampled coordinate for an ordinary
# block, the emitted model's derived scale for an R2D2 block.
_block_tau(x, block::BRM.AdaptiveCenteringBlock, i) = exp(x[block.log_scales[i]])
_block_tau(x, block::BRM.R2D2AdaptiveCenteringBlock, i) =
    exp(BRM._adaptive_log_scale(x, block.table, i))

# A Student-t block (`mixing` non-empty) scales level `g`'s factor by
# `sqrt(w[g])`, the level's mixing weight read from its coordinate `log w[g]`.
_block_mixed(::BRM.AdaptiveCenteringBlock) = false
_block_mixed(block::BRM.R2D2AdaptiveCenteringBlock) = !isempty(block.mixing)
_block_mixing_scale(x, ::BRM.AdaptiveCenteringBlock, g) = one(eltype(x))
_block_mixing_scale(x, block::BRM.R2D2AdaptiveCenteringBlock, g) =
    exp(x[block.mixing[g]] / 2)

# Pair `p`'s level scale `sqrt(w[g])`, read once per accessor call from the
# state's flat `pair_mixing`; one for a Gaussian block, whose operations the
# product `tau * one(T)` leaves bit-identical. The differentiated accessors
# never read the block's `mixing` vector: on Julia 1.10, adding that read (and
# its branch) to `_block_location`'s row loop corrupted Julia's GC under
# Enzyme's reverse pass in `test/adaptive_centering_threads.jl`, even for
# blocks without weights.
function _pair_mixing_scale(x, state, p)
    j = state.pair_mixing[p]
    j == 0 ? one(eltype(x)) : exp(x[j] / 2)
end

# Canonical partial correlation `p` of a block's correlation factor, in Stan's
# `cholesky_factor_corr` coordinate order: `tanh` of its coordinate, or the
# constant of a known correlation (`cor(:, ID) = R`), which has no coordinates.
_block_cpc(x, block::BRM.AdaptiveCenteringBlock, p) = tanh(x[block.cholesky_free[p]])
_block_cpc(x, block::BRM.R2D2AdaptiveCenteringBlock, p) =
    isempty(block.cholesky_cpc) ? tanh(x[block.cholesky_free[p]]) :
        convert(eltype(x), block.cholesky_cpc[p])

# The kind is a type parameter so Enzyme sees only the accessor branch it is
# differentiating.  Every pair still has the same concrete
# `(location, log_scale)` argument-tuple type, keeping WarmupHMC's pair vector
# concrete.
struct BRMAdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMAdaptiveCenteringState{N}
    pair_number::Int
end

function BRMAdaptiveCenteringArgument(
    state::BRMAdaptiveCenteringState{N}, pair_number, kind::Symbol,
) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM adaptive-centering argument kind $kind",
    )
    BRMAdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function _pair_location(state, pair_number)
    (state.pair_blocks[pair_number],
     state.pair_terms[pair_number],
     state.pair_groups[pair_number])
end

function _pair_index(state, pair_number)
    bi, k, g = _pair_location(state, pair_number)
    state.blocks[bi].effects[k, g]
end

# Return one entry of `C = diag(tau) * L` without allocating, for any K.  Stan's
# `cholesky_factor_corr` stick recursion is walked inline along row `i` instead
# of materializing `L`, so this is one nonrecursive loop over that row's free
# coordinates.  The spelling is bit-identical to the legacy `_adaptive_cholesky_corr`
# + `tau .* L` matrix graph, including its multiplication association: first
# `stick * z`, then `tau * L[i,j]`.  WarmupHMC's discrete window winners can
# change after a one-ulp gradient drift, so operation order — not just the
# limiting value — is the contract.
function _block_cholesky_entry(x, block, i, j)
    tau = _block_tau(x, block, i)
    stick = one(eltype(x))
    offset = (i - 1) * (i - 2) ÷ 2
    for l in 1:i-1
        z = _block_cpc(x, block, offset + l)
        l == j && return tau * (stick * z)
        stick *= sqrt(one(eltype(x)) - z * z)
    end
    j == i || throw(BoundsError((i, j)))
    tau * stick
end

# One arbitrary-K location accessor, differentiated by the ordinary Enzyme path.
#
# `u[k] = c[k] * sum(C[k,l] * z[l] for l < k) + C[k,k]^c[k] * z[k]` is triangular,
# so recovering `z` is a forward substitution: row `i` needs the `i-1`
# innovations below it.  That O(k) state is irreducible — the whole `K x K` `L`
# and `C` rebuild per pair is gone, replaced by the same inline stick walk as
# `_block_cholesky_entry`.  The innovations live in an `NTuple{N}` sized by the
# state's capacity `N`, not a heap vector: on Julia 1.10, Enzyme's reverse pass
# over a function that allocates a small work vector and reads it back is a
# use-after-free once another thread runs GC — `getindex` segfaults or the GC
# aborts with "probable corruption" (measured on Enzyme 0.13.210-0.13.216; Julia
# 1.12 and single-threaded runs are clean).  Concurrent completion/multi-chain
# workers hit exactly that.  `N` belongs to the state, so every pair keeps one
# concrete element type whatever the block widths.  The buffer cannot be cached
# on the state, because accessors run under `Enzyme.Const`.  Recursion is
# deliberately avoided: a recursive scalar callable here corrupted Julia's GC
# after ~1,750 sustained gradients (fixed by ade46ac/bd9ca2d/b5f4a86).
#
# A pair centred on a population coefficient (`pair_location`) adds it to its
# source-frame location: margin `i`'s innovation is `(u - c * (m + beta)) /
# s^c`, and the returned location is `m + beta` (`_with_location`). The model
# coordinate is unchanged; only blocks compiled non-centred carry a location,
# where WarmupHMC's pair map `(x - c * loc) / s^c` then samples the level's
# total at `c = 1`.
function _block_location(x, state::_BRMRanefCenteringState{N}, pair_number) where {N}
    bi, k, g = _pair_location(state, pair_number)
    T = eltype(x)
    k == 1 && return _with_location(zero(T), x, state, pair_number)
    block = state.blocks[bi]
    lookup = state.pair_lookup[bi]
    r = _pair_mixing_scale(x, state, pair_number)
    innovations = ntuple(_ -> zero(T), Val(N))
    m = zero(T)
    for i in 1:k
        m = zero(T)
        tau = _block_tau(x, block, i) * r
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _block_cpc(x, block, offset + l)
            m += (tau * (stick * z)) * innovations[l]
            stick *= sqrt(one(T) - z * z)
        end
        if i < k
            c = state.sources[lookup[i, g]]
            u = x[block.effects[i, g]]
            s = tau * stick
            innovations = Base.setindex(innovations,
                (u - c * _with_location(m, x, state, lookup[i, g])) / s^c, i)
        end
    end
    _with_location(m, x, state, pair_number)
end

function (arg::BRMAdaptiveCenteringArgument{:location})(x)
    _block_location(x, arg.state, arg.pair_number)
end

function (arg::BRMAdaptiveCenteringArgument{:log_scale})(x)
    bi, k, _ = _pair_location(arg.state, arg.pair_number)
    block = arg.state.blocks[bi]
    log(_block_cholesky_entry(x, block, k, k))
end

struct BRMR2D2AdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMR2D2AdaptiveCenteringState{N}
    pair_number::Int
end

function BRMR2D2AdaptiveCenteringArgument(
    state::BRMR2D2AdaptiveCenteringState{N}, pair_number, kind::Symbol,
) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM R2D2 adaptive-centering argument kind $kind",
    )
    BRMR2D2AdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:location})(x)
    _block_location(x, arg.state, arg.pair_number)
end

function (arg::BRMR2D2AdaptiveCenteringArgument{:log_scale})(x)
    bi, k, _ = _pair_location(arg.state, arg.pair_number)
    log(_block_cholesky_entry(x, arg.state.blocks[bi], k, k) *
        _pair_mixing_scale(x, arg.state, arg.pair_number))
end

function _sync_sources!(state, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM adaptive-centering plan has $(length(state.sources)) cells but the " *
        "WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (i, (idx, value)) in enumerate(ir.pairs)
        expected = _pair_index(state, i)
        idx == expected || throw(ArgumentError(
            "BRM adaptive-centering pair $i addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[i] = Float64(value.source.c)
    end
    ir
end

struct BRMAdaptiveCenteringFrame{T}
    source::Vector{T}
    location::Vector{T}
    scale::Vector{T}
    innovation::Vector{T}
    invariant_gradient::Vector{T}
end

function _prepare_frame(state::_BRMRanefCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

# `ordered` is a `BRMOrderedCenteringReparametrization`, whose source frame
# orders every group (`_source_group_factor`); `nothing` is the declared order,
# whose frame is unchanged.
function _centering_frame(state::_BRMRanefCenteringState{N}, position, gradient,
                          ordered=nothing) where {N}
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)

    for (bi, block) in enumerate(state.blocks)
        K = block.ranef.n_terms
        # In the declared order a Gaussian block's factor is the same for every
        # level, so it is built once per block, and the work arrays are reused
        # across levels (every level rewrites `A`'s lower triangle and all of
        # `z`, `gu`, `h`). Rebuilt per level, the frame allocated about 10 MB
        # per leaf on 25k levels (snag warmuphmc-adapti-e35d58ad); the values
        # are unchanged, bit for bit.
        shared = isnothing(ordered) && !_block_mixed(block) ?
            BRM._adaptive_block_cholesky(position, block, 1) : nothing
        A = zeros(T, K, K)
        z = Vector{T}(undef, K)
        gu = Vector{T}(undef, K)
        h = Vector{T}(undef, K)
        for g in 1:block.ranef.n_groups
            order = isnothing(ordered) ? (1:K) : view(ordered.orders[bi], :, g)
            C = !isnothing(shared) ? shared :
                isnothing(ordered) ? BRM._adaptive_block_cholesky(position, block, g) :
                _unpacked(_source_group_factor(position, ordered, bi, g), K)
            for j in 1:K
                k = order[j]
                p = state.pair_lookup[bi][k, g]
                c = source[p]
                m = zero(T)
                for l in 1:j-1
                    m += C[j, l] * z[l]
                    A[j, l] = c * C[j, l]
                end
                s = C[j, j]
                u = position[block.effects[k, g]]
                # The source-frame location, a population coefficient included
                # (`_with_location`); it does not depend on the innovations, so
                # `A` and the invariant gradient are unchanged by it.
                loc = _with_location(m, position, state, p)
                z[j] = (u - c * loc) / s^c
                A[j, j] = s^c
                location[p] = loc
                scale[p] = s
                innovation[p] = z[j]
            end
            for j in 1:K
                gu[j] = gradient[block.effects[order[j], g]]
            end
            BRM.LinearAlgebra.mul!(h, transpose(A), gu)
            for j in 1:K
                invariant_gradient[state.pair_lookup[bi][order[j], g]] = h[j]
            end
        end
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _score_candidate(frame, pair_number, idx, value, candidate)
    1 <= pair_number <= length(frame.source) || throw(BoundsError(frame.source, pair_number))
    t = candidate.c
    c = frame.source[pair_number]
    m = frame.location[pair_number]
    s = frame.scale[pair_number]
    z = frame.innovation[pair_number]
    h = frame.invariant_gradient[pair_number]
    ((t - c) * log(s), t * m + s^t * z, h / s^t)
end

function _problem_unc_names(problem)
    hasproperty(problem, :model) || error(
        "BRM adaptive centering: `unc_names` was omitted, but problem type " *
        "$(typeof(problem)) has no `.model` BridgeStan handle. Pass the compiled " *
        "model's unconstrained parameter names explicitly.",
    )
    BRM.StanBlocks.BridgeStan.param_unc_names(getproperty(problem, :model))
end

function _adaptive_centering_reparametrizer(blocks, locations=nothing)
    state = BRMAdaptiveCenteringState(blocks,
        isnothing(locations) ? nothing : _block_locations(blocks, locations))
    _check_locations_read_only(state, state.effect_indices)
    pairs = [begin
        block = state.blocks[state.pair_blocks[p]]
        k = state.pair_terms[p]
        g = state.pair_groups[p]
        idx = block.effects[k, g]
        target = WarmupHMC.PartiallyCentered(block.target_c)
        source = WarmupHMC.PartiallyCentered(block.target_c)
        loc = BRMAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMAdaptiveCenteringArgument(state, p, :log_scale)
        idx => WarmupHMC.Reparametrization(target, source, loc, log_scale)
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# HSGP basis weights are conditionally independent given their spectral
# scales, so they use a scalar zero-location frame.  Keep this state and its
# callable distinct from `BRMAdaptiveCenteringState`: the latter's concrete
# type parameter selects the exact K<=2 Enzyme reverse pass for ordinary random
# effects, and HSGP support must not perturb that dispatch.
mutable struct BRMHSGPAdaptiveCenteringState
    blocks::Vector{BRM._HSGPAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_basis::Vector{Int}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMHSGPAdaptiveCenteringState(blocks)
    pair_blocks = Int[]
    pair_basis = Int[]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), basis in eachindex(block.effects)
        push!(pair_blocks, bi)
        push!(pair_basis, basis)
        push!(sources, block.target_c[basis])
    end
    effect_indices = BitSet(Iterators.flatten(b.effects for b in blocks))
    BRMHSGPAdaptiveCenteringState(
        collect(blocks), pair_blocks, pair_basis, sources, effect_indices,
    )
end

struct BRMHSGPAdaptiveCenteringArgument{KIND} <: Function
    state::BRMHSGPAdaptiveCenteringState
    pair_number::Int
end

function BRMHSGPAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM HSGP adaptive-centering argument kind $kind",
    )
    BRMHSGPAdaptiveCenteringArgument{kind}(state, pair_number)
end

function _hsgp_pair_location(state, pair_number)
    (state.pair_blocks[pair_number], state.pair_basis[pair_number])
end

function _hsgp_pair_index(state, pair_number)
    bi, basis = _hsgp_pair_location(state, pair_number)
    state.blocks[bi].effects[basis]
end

function (::BRMHSGPAdaptiveCenteringArgument{:location})(x)
    zero(eltype(x))
end

function (arg::BRMHSGPAdaptiveCenteringArgument{:log_scale})(x)
    bi, basis = _hsgp_pair_location(arg.state, arg.pair_number)
    BRM._adaptive_hsgp_log_scale(x, arg.state.blocks[bi], basis)
end

function _sync_sources!(state::BRMHSGPAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM HSGP adaptive-centering plan has $(length(state.sources)) basis " *
        "weights but the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _hsgp_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM HSGP adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMHSGPAdaptiveCenteringState,
                        ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMHSGPAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = zeros(T, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in eachindex(state.sources)
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        idx = block.effects[basis]
        log_scale = BRM._adaptive_hsgp_log_scale(position, block, basis)
        s = exp(log_scale)
        c = source[p]
        # A floored scale is (numerically) the same at every observation, so
        # every candidate frame `s^t * z` is a fixed rescaling of the others:
        # the losses tie and rounding would choose the control. Off the floor
        # that control couples the weight to a log-scale up to ~150 larger, a
        # funnel no chain crosses (snag adaptive-centeri-eca5a77c). A floored
        # observation therefore scores every candidate in the non-centred frame
        # (scale 1), so a window spent on the floor ties exactly and keeps the
        # first candidate, `c = 0`. The transform still reads the floored `s`.
        scale[p] = log_scale > BRM._ADAPTIVE_HSGP_LOG_SCALE_EXACT ? s : one(s)
        innovation[p] = position[idx] / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_hsgp_centering_reparametrizer(blocks)
    state = BRMHSGPAdaptiveCenteringState(blocks)
    pairs = [begin
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        target = WarmupHMC.PartiallyCentered(block.target_c[basis])
        source = WarmupHMC.PartiallyCentered(block.target_c[basis])
        location = BRMHSGPAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMHSGPAdaptiveCenteringArgument(state, p, :log_scale)
        block.effects[basis] => WarmupHMC.Reparametrization(
            target, source, location, log_scale,
        )
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# Grouped HSGP weights mapped across their levels
# (`BRM._HSGPLevelAdaptiveCenteringBlock`: `cor = Omega`, a known level
# correlation, or per-level scales). Basis column `b`'s levels are one
# correlated block with factor `C_b = diag(s_b) * L`, and level `k` is mapped
# exactly like margin `k` of one group of a correlated random-effect block
# (`_block_location`): centred on its conditional mean given levels `1..k-1`,
# with scale `C_b[k, k]`. Independent levels (`L = I`) have location 0. Pairs
# are term-major, then basis-major, then level-major, so a column's levels are
# contiguous and in the inverse's dependency order. The cells join only the
# joint wrapper, and their accessors follow `_block_location`'s Julia 1.10
# rules: innovations in an `NTuple{N}` (`N` shared with the joint state's
# random-effect capacity), per-pair data in flat vectors, and a term's
# constants in one isbits `_HSGPLevelTerm` read once per call; nothing reads a
# block struct inside the differentiated loops.
struct _HSGPLevelTerm
    correlated::Bool
    # Known partial correlations (`cpc_value`) or sampled coordinates
    # (`cpc_index`), from offset `cpc`.
    known::Bool
    cpc::Int
    # The marginal SD: shared (`sigma`, coordinate 0 a known value at both
    # bounds) or predicted per level (`intercept + sd * z[g]` on the log scale,
    # `z[g]` the pair's `pair_sigma_z`).
    sigma_predicted::Bool
    sigma::Int
    sigma_lower::Float64
    sigma_upper::Float64
    sigma_intercept::Int
    sigma_sd::Int
    sigma_sd_lower::Float64
    sigma_sd_upper::Float64
    # The length scale: shared (`n_rho` coordinates from offset `rho` of
    # `rho_index`/`rho_lower`/`rho_upper`) or predicted per level and floored.
    rho_predicted::Bool
    rho::Int
    n_rho::Int
    rho_intercept::Int
    rho_sd::Int
    rho_sd_lower::Float64
    rho_sd_upper::Float64
    rho_floor::Float64
    # Basis `b`'s frequency on axis `a` is `omega2[omega2_offset + (b-1)*n_axes + a]`.
    n_axes::Int
    omega2_offset::Int
end

mutable struct BRMHSGPLevelCenteringState{N}
    blocks::Vector{BRM._HSGPLevelAdaptiveCenteringBlock}
    terms::Vector{_HSGPLevelTerm}
    pair_blocks::Vector{Int}
    pair_level::Vector{Int}
    pair_basis::Vector{Int}
    pair_index::Vector{Int}
    pair_sigma_z::Vector{Int}
    pair_rho_z::Vector{Int}
    cpc_index::Vector{Int}
    cpc_value::Vector{Float64}
    rho_index::Vector{Int}
    rho_lower::Vector{Float64}
    rho_upper::Vector{Float64}
    omega2::Vector{Float64}
    sources::Vector{Float64}
    effect_indices::BitSet
end

_level_capacity(blocks) = max(1, maximum((size(b.effects, 1) for b in blocks); init=1) - 1)

function BRMHSGPLevelCenteringState{N}(blocks) where {N}
    _level_capacity(blocks) <= N || throw(ArgumentError(
        "BRM HSGP level cells need innovation capacity $(_level_capacity(blocks)), got $N"))
    terms = _HSGPLevelTerm[]
    pair_blocks, pair_level, pair_basis, pair_index = Int[], Int[], Int[], Int[]
    pair_sigma_z, pair_rho_z = Int[], Int[]
    cpc_index, cpc_value = Int[], Float64[]
    rho_index, rho_lower, rho_upper, omega2 = Int[], Float64[], Float64[], Float64[]
    for (bi, block) in enumerate(blocks)
        G, B = size(block.effects)
        s, r = block.sigma, block.rho
        push!(terms, _HSGPLevelTerm(
            block.correlation !== :none, block.correlation === :data,
            block.correlation === :data ? length(cpc_value) : length(cpc_index),
            s.predicted, s.predicted ? 0 : only(s.coordinates),
            s.predicted ? 0.0 : only(s.lower), s.predicted ? 0.0 : only(s.upper),
            s.intercept, s.sd, s.sd_lower, s.sd_upper,
            r.predicted, length(rho_index), length(r.coordinates),
            r.intercept, r.sd, r.sd_lower, r.sd_upper, block.rho_floor,
            size(block.omega2, 2), length(omega2)))
        append!(cpc_index, block.cholesky_free)
        append!(cpc_value, block.cholesky_cpc)
        append!(rho_index, r.coordinates)
        append!(rho_lower, r.lower)
        append!(rho_upper, r.upper)
        append!(omega2, vec(permutedims(block.omega2)))
        for b in 1:B, g in 1:G
            push!(pair_blocks, bi)
            push!(pair_level, g)
            push!(pair_basis, b)
            push!(pair_index, block.effects[g, b])
            push!(pair_sigma_z, s.sd == 0 ? 0 : s.z[g])
            push!(pair_rho_z, r.sd == 0 ? 0 : r.z[g])
        end
    end
    BRMHSGPLevelCenteringState{N}(collect(blocks), terms, pair_blocks, pair_level,
        pair_basis, pair_index, pair_sigma_z, pair_rho_z, cpc_index, cpc_value,
        rho_index, rho_lower, rho_upper, omega2, zeros(length(pair_index)),
        BitSet(pair_index))
end

BRMHSGPLevelCenteringState(blocks) =
    BRMHSGPLevelCenteringState{_level_capacity(blocks)}(blocks)

# Partial correlation `j` of a term's level factor: known, or `tanh` of its
# Stan `cholesky_factor_corr` coordinate.
@inline _level_cpc(x, state, term, j) = term.known ?
    convert(eltype(x), state.cpc_value[term.cpc + j]) :
    tanh(x[state.cpc_index[term.cpc + j]])

# Pair `q`'s level scale `s_b[g]` on the log scale (`BRM._adaptive_hsgp_level_log_scale`).
@inline function _level_log_scale(x, state, term, b, q)
    value = term.sigma_predicted ?
        BRM._adaptive_hsgp_predicted_log(x, term.sigma_intercept, term.sigma_sd,
            term.sigma_sd_lower, term.sigma_sd_upper, state.pair_sigma_z[q]) :
        log(BRM._adaptive_hsgp_hyper(x, term.sigma, term.sigma_lower, term.sigma_upper))
    base = term.omega2_offset + (b - 1) * term.n_axes
    for axis in 1:term.n_axes
        rho = if term.rho_predicted
            max(exp(BRM._adaptive_hsgp_predicted_log(x, term.rho_intercept, term.rho_sd,
                term.rho_sd_lower, term.rho_sd_upper, state.pair_rho_z[q])), term.rho_floor)
        else
            j = term.rho + (term.n_rho == 1 ? 1 : axis)
            BRM._adaptive_hsgp_hyper(x, state.rho_index[j], state.rho_lower[j],
                                     state.rho_upper[j])
        end
        value = BRM._adaptive_hsgp_axis_log_scale(value, rho, state.omega2[base + axis])
    end
    BRM._adaptive_hsgp_floor(value)
end

# Pair `p`'s source-frame location: `sum(C_b[k, l] * z[l] for l < k)`, with
# the column's earlier innovations recovered from their source coordinates in
# `u` (their source controls in `sources`) exactly as `_block_location`
# recovers a group's earlier margins. Every other coordinate is read from `x`;
# the joint wrapper passes the source vector as both, and the learned-order
# inverse passes the source being rebuilt as `u`.
@inline function _level_location(x, u, state::BRMHSGPLevelCenteringState{N}, p) where {N}
    T = eltype(x)
    k = state.pair_level[p]
    term = state.terms[state.pair_blocks[p]]
    (k == 1 || !term.correlated) && return zero(T)
    b = state.pair_basis[p]
    first = p - k
    innovations = ntuple(_ -> zero(T), Val(N))
    m = zero(T)
    for i in 1:k
        q = first + i
        tau = exp(_level_log_scale(x, state, term, b, q))
        m = zero(T)
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _level_cpc(x, state, term, offset + l)
            m += (tau * (stick * z)) * innovations[l]
            stick *= sqrt(one(T) - z * z)
        end
        if i < k
            c = state.sources[q]
            s = tau * stick
            innovations = _tset(innovations, (u[state.pair_index[q]] - c * m) / s^c, i)
        end
    end
    m
end

# Pair `p`'s log scale `log C_b[k, k]`: the level's log spectral scale plus
# the log of its row's stick.
@inline function _level_pair_log_scale(x, state::BRMHSGPLevelCenteringState, p)
    T = eltype(x)
    k = state.pair_level[p]
    term = state.terms[state.pair_blocks[p]]
    value = _level_log_scale(x, state, term, state.pair_basis[p], p)
    term.correlated || return value
    stick = one(T)
    offset = (k - 1) * (k - 2) ÷ 2
    for l in 1:k-1
        z = _level_cpc(x, state, term, offset + l)
        stick *= sqrt(one(T) - z * z)
    end
    value + log(stick)
end

_level_pair_index(state, p) = state.pair_index[p]

# Every level cell's frame, column by column: the same triangular frame as one
# group of a correlated random-effect block (`_centering_frame` above) with
# `C_b` as the group's factor. A floored level scale scores every candidate in
# the non-centred frame (scale 1, location 0), as a floored HSGP basis weight
# does (`BRMHSGPAdaptiveCenteringState`), so a level scored only on the floor
# keeps `c = 0`; its innovation and gradient keep the floored scale.
function _centering_frame(state::BRMHSGPLevelCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    p = 0
    for block in state.blocks
        G, B = size(block.effects)
        L = BRM._adaptive_hsgp_level_correlation_factor(position, block)
        A = zeros(T, G, G)
        z = Vector{T}(undef, G)
        gu = Vector{T}(undef, G)
        h = Vector{T}(undef, G)
        for b in 1:B
            for j in 1:G
                q = p + j
                log_s = BRM._adaptive_hsgp_level_log_scale(position, block, j, b)
                tau = exp(log_s)
                c = source[q]
                m = zero(T)
                for l in 1:j-1
                    m += tau * L[j, l] * z[l]
                    A[j, l] = c * tau * L[j, l]
                end
                s = tau * L[j, j]
                z[j] = (position[block.effects[j, b]] - c * m) / s^c
                A[j, j] = s^c
                floored = log_s <= BRM._ADAPTIVE_HSGP_LOG_SCALE_EXACT
                location[q] = floored ? zero(T) : m
                scale[q] = floored ? one(T) : s
                innovation[q] = z[j]
                gu[j] = gradient[block.effects[j, b]]
            end
            BRM.LinearAlgebra.mul!(h, transpose(A), gu)
            invariant_gradient[p .+ (1:G)] .= h
            p += G
        end
    end
    BRMAdaptiveCenteringFrame(source, location, scale, innovation, invariant_gradient)
end

const _NO_LEVEL_BLOCKS = BRM._HSGPLevelAdaptiveCenteringBlock[]

# Scalar cells with a constant location and a raw log-scale coordinate: exact
# totals (location = prior total, compiled at c = 1) and S2Z free contrasts
# (location 0, compiled at c = 0 or 1). Cells enumerate totals first, then S2Z
# contrasts, each in its family's cell order (`BRM._total_centering_cells`,
# `BRM._s2z_centering_cells`), which the post-hoc selectors share.
mutable struct BRMScalarCenteringState
    indices::Vector{Int}
    scales::Vector{Int}
    locations::Vector{Float64}
    sources::Vector{Float64}
end

struct BRMScalarCenteringArgument{KIND} <: Function
    state::BRMScalarCenteringState
    pair_number::Int
end
(arg::BRMScalarCenteringArgument{:location})(x) = arg.state.locations[arg.pair_number]
(arg::BRMScalarCenteringArgument{:log_scale})(x) = x[arg.state.scales[arg.pair_number]]

function _sync_sources!(state::BRMScalarCenteringState,ir)
    length(state.sources) == length(ir.pairs) || throw(DimensionMismatch("scalar centering pair count changed"))
    for (p,(index,value)) in enumerate(ir.pairs)
        index == state.indices[p] || throw(ArgumentError("scalar centering pair order changed"))
        state.sources[p] = value.source.c
    end
    ir
end

function _prepare_frame(state::BRMScalarCenteringState,ir,position,gradient)
    _sync_sources!(state,ir)
    _centering_frame(state,position,gradient)
end

function _centering_frame(state::BRMScalarCenteringState,position,gradient)
    source = copy(state.sources)
    location = copy(state.locations)
    scale,innovation,invariant_gradient = similar(source),similar(source),similar(source)
    for p in eachindex(source)
        scale[p] = exp(position[state.scales[p]])
        factor = scale[p]^source[p]
        innovation[p] = (position[state.indices[p]]-source[p]*location[p])/factor
        invariant_gradient[p] = factor*gradient[state.indices[p]]
    end
    BRMAdaptiveCenteringFrame(source,location,scale,innovation,invariant_gradient)
end

# Every scalar cell of `model`: sampled exact totals, then S2Z contrasts.
function _scalar_centering_cells(model,names)
    totals = BRM._total_centering_cells(model,names)
    s2z = BRM._s2z_centering_cells(model,names)
    (;indices=vcat(totals.indices,s2z.indices),scales=vcat(totals.scales,s2z.scales),
      locations=vcat(totals.locations,s2z.locations),targets=vcat(totals.targets,s2z.targets))
end

_adaptive_scalar_centering_reparametrizer(model,names) =
    _scalar_centering_reparametrizer(_scalar_centering_cells(model,names)...)

_scalar_centering_state(cells) =
    BRMScalarCenteringState(cells.indices,cells.scales,cells.locations,copy(cells.targets))

function _scalar_centering_reparametrizer(indices,scales,locations,targets)
    state = _scalar_centering_state((;indices,scales,locations,targets))
    pairs = [indices[p] => WarmupHMC.Reparametrization(
        WarmupHMC.PartiallyCentered(targets[p]),WarmupHMC.PartiallyCentered(targets[p]),
        BRMScalarCenteringArgument{:location}(state,p),
        BRMScalarCenteringArgument{:log_scale}(state,p)) for p in eachindex(indices)]
    state,WarmupHMC.IndexedReparametrization(pairs)
end

# ---- Mean-located exact totals ------------------------------------------------
#
# Exact totals centred on their own column's location instead of a constant
# (user pick `1rncfjy`, snag exact-totals-fun-f116d6b2). For one total column
# (`BRM._TotalCenteringColumns`: J totals `T`, log-scale `ell`, location design
# `W`, J x P) and cell controls `c`, let `a_j = exp((1 - c_j) * ell)` and
# `A = Diagonal(a)`. The sampler coordinate is
#
#     x_j = loc_j + (T_j - loc_j) / a_j,    loc = W * ((W'W) \ (W'T)),
#
# the least-squares fit of the totals on `W`, a function of the totals that no
# control moves (the column mean for a plain block). The inverse is
# `loc = W * ((W'AW) \ (W'A x))`, `T = loc + A * (x - loc)`, and the
# log-Jacobian of `x -> T` is `sum(log, a) + logdet(W'W) - logdet(W'AW)`. So
# `c = 1` is the compiled total and `c = 0` gives `(T - loc) / tau`, the
# standard non-centred deviations, with the location kept. The location couples
# a column's cells, which `WarmupHMC.IndexedReparametrization` (a per-cell
# Jacobian and a cell-by-cell inverse) cannot express, so the cells live in
# their own reparametrizer, adapted by a `WarmupHMC.WindowSelectionPlan`. S2Z
# contrasts, constant-location cells, follow the totals in the same wrapper.
struct BRMMeanTotalsCenteringReparametrization{N} <: WarmupHMC.AbstractReparametrization
    # Every scalar cell (`_scalar_centering_cells`: totals, then S2Z contrasts);
    # `sources` holds the controls. Total cells ignore `locations`.
    state::BRMScalarCenteringState
    columns::BRM._TotalCenteringColumns
    # The compiled frame of every cell.
    targets::Vector{Float64}
    # `1`: source -> compiled; `-1`: compiled -> source.
    direction::Int
end

# The column's `N x N` systems are built and factored without a loop-carried
# NTuple: Enzyme (Julia 1.10) returned wrong gradients for a tuple updated in a
# loop at a runtime index (`M = _tset(M, M[k] + v, k)`) once a column had two
# or more location predictors, with exact values (measured against central
# differences: off by up to 2.4 at `N = 5`; correct to 1e-9 in this form). So
# each Gram entry is a scalar sum over the column's cells, and the Cholesky
# factor and solve are straight-line code for the compile-time `N`. Dimensions
# past the column's width `P` carry an identity block: no location, no logdet.

# Lower Cholesky factor of a row-major `N x N` NTuple (lower triangle read).
@generated function _small_cholesky(M::NTuple{NN,T}, ::Val{N}) where {NN,T,N}
    L(i, j) = Symbol(:L_, i, :_, j)
    body = Expr[]
    for j in 1:N
        s = :(M[$((j - 1) * N + j)])
        for k in 1:j-1
            s = :($s - $(L(j, k)) * $(L(j, k)))
        end
        push!(body, :($(L(j, j)) = sqrt($s)))
        for i in j+1:N
            s = :(M[$((i - 1) * N + j)])
            for k in 1:j-1
                s = :($s - $(L(i, k)) * $(L(j, k)))
            end
            push!(body, :($(L(i, j)) = $s / $(L(j, j))))
        end
    end
    quote
        $(body...)
        ($((i >= j ? L(i, j) : :(zero(T)) for i in 1:N for j in 1:N)...),)
    end
end

# `(L L') \ r` for the factor `L` of `_small_cholesky`.
@generated function _small_cholesky_solve(L::NTuple{NN,T}, r::NTuple{N,T}) where {NN,T,N}
    y(i) = Symbol(:y_, i)
    z(i) = Symbol(:z_, i)
    body = Expr[]
    for i in 1:N
        s = :(r[$i])
        for k in 1:i-1
            s = :($s - L[$((i - 1) * N + k)] * $(y(k)))
        end
        push!(body, :($(y(i)) = $s / L[$((i - 1) * N + i)]))
    end
    for i in N:-1:1
        s = y(i)
        for k in i+1:N
            s = :($s - L[$((k - 1) * N + i)] * $(z(k)))
        end
        push!(body, :($(z(i)) = $s / L[$((i - 1) * N + i)]))
    end
    quote
        $(body...)
        ($((z(i) for i in 1:N)...),)
    end
end

# Entry `(p, q)`, `p >= q`, of `W'AW` and `W'BW` for column `col`
# (`a_j = exp((1 - from_j) * ell)`, `b_j = exp((1 - to_j) * ell)`), or the
# identity past the column's width.
function _mean_total_gram(cols, col, from, to, ell, p, q)
    T = typeof(ell)
    p > cols.widths[col] && return (T(p == q), T(p == q))
    lo, hi = cols.starts[col], cols.starts[col + 1] - 1
    J, d0 = hi - lo + 1, cols.offsets[col]
    sa = sb = zero(T)
    for j in 1:J
        cell = lo + j - 1
        ww = cols.design[d0 + (p - 1) * J + j] * cols.design[d0 + (q - 1) * J + j]
        sa += exp((1 - from[cell]) * ell) * ww
        sb += exp((1 - to[cell]) * ell) * ww
    end
    sa, sb
end

# Entry `p` of `W'Ax`, zero past the column's width.
function _mean_total_moment(x, cols, col, from, ell, p)
    T = eltype(x)
    p > cols.widths[col] && return zero(T)
    lo, hi = cols.starts[col], cols.starts[col + 1] - 1
    J, d0 = hi - lo + 1, cols.offsets[col]
    s = zero(T)
    for j in 1:J
        cell = lo + j - 1
        s += exp((1 - from[cell]) * ell) * cols.design[d0 + (p - 1) * J + j] *
             x[cols.indices[cell]]
    end
    s
end

# Column `col` from controls `from` to `to` (`b_j = exp((1 - to_j) * ell)`):
# `y_j = loc_j + (a_j / b_j) * (x_j - loc_j)`, with the location solved in the
# `from` frame (the same in every frame). Returns the log-Jacobian
# `sum((to - from) * ell) - logdet(W'AW) + logdet(W'BW)`. Every read of `x`
# precedes the writes, and the column writes only its own cells, so `y === x`
# is safe.
function _mean_total_column_map!(y, x, cols, col, from, to, ::Val{N}) where {N}
    T = eltype(x)
    ell = x[cols.scales[col]]
    lo, hi = cols.starts[col], cols.starts[col + 1] - 1
    J, P, d0 = hi - lo + 1, cols.widths[col], cols.offsets[col]
    M = ntuple(k -> cld(k, N) >= mod1(k, N) ?
        _mean_total_gram(cols, col, from, to, ell, cld(k, N), mod1(k, N)) :
        (zero(T), zero(T)), Val(N * N))
    La = _small_cholesky(ntuple(k -> M[k][1], Val(N * N)), Val(N))
    Lb = _small_cholesky(ntuple(k -> M[k][2], Val(N * N)), Val(N))
    gamma = _small_cholesky_solve(La,
        ntuple(p -> _mean_total_moment(x, cols, col, from, ell, p), Val(N)))
    ljac = zero(T)
    for p in 1:N
        ljac += 2 * (log(Lb[(p - 1) * N + p]) - log(La[(p - 1) * N + p]))
    end
    for j in 1:J
        cell = lo + j - 1
        ljac += (to[cell] - from[cell]) * ell
        loc = sum(ntuple(p -> p <= P ? cols.design[d0 + (p - 1) * J + j] * gamma[p] :
                                       zero(T), Val(N)))
        idx = cols.indices[cell]
        y[idx] = loc + exp((to[cell] - from[cell]) * ell) * (x[idx] - loc)
    end
    ljac
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        ir::BRMMeanTotalsCenteringReparametrization{N}, x::AbstractVector) where {N}
    s, cols = ir.state, ir.columns
    from, to = ir.direction > 0 ? (s.sources, ir.targets) : (ir.targets, s.sources)
    ljac = zero(eltype(x))
    for col in eachindex(cols.scales)
        ljac += _mean_total_column_map!(y, x, cols, col, from, to, Val(N))
    end
    for p in cols.starts[end]:length(s.sources)
        ljac += _cell_map!(y, x, s.indices[p], s.locations[p], x[s.scales[p]],
            ir.targets[p], s.sources[p], ir.direction)
    end
    ljac, y
end

WarmupHMC.InverseFunctions.inverse(ir::BRMMeanTotalsCenteringReparametrization{N}) where {N} =
    BRMMeanTotalsCenteringReparametrization{N}(ir.state, ir.columns, ir.targets, -ir.direction)

WarmupHMC.reparam_controls(ir::BRMMeanTotalsCenteringReparametrization) =
    copy(ir.state.sources)

function WarmupHMC.restore_reparam_controls!(ir::BRMMeanTotalsCenteringReparametrization,
                                             controls)
    n = length(ir.state.sources)
    controls isa AbstractVector && length(controls) == n || throw(ArgumentError(
        "BRM mean-located totals centering controls must be one value per " *
        "adaptive cell ($n): totals, then S2Z contrasts"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1, controls) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    ir.state.sources .= controls
    ir
end

WarmupHMC.snapshot_reparametrization(ir::BRMMeanTotalsCenteringReparametrization{N}) where {N} =
    BRMMeanTotalsCenteringReparametrization{N}(
        BRMScalarCenteringState(ir.state.indices, ir.state.scales, ir.state.locations,
                                copy(ir.state.sources)),
        ir.columns, ir.targets, ir.direction)

# Every cell's candidate coordinates `Q[i, p]` and gradients `G[i, p]` at
# `_CENTERING_CANDIDATES[i]`, from one source-frame draw `x` with gradient `gx`,
# each cell moved alone (the other cells keep their source controls). A total's
# candidate is exact. The location stays in place, so
# `q = loc_j + (T_j - loc_j) / a'`. With `G` the gradient in the totals,
# `M = W'AW`, `v = W'(I - A)G`, `u = M \ w_j` and the control change
# `d = a' - a_j`, the candidate gradient is
# `a' * (G_j + (v'u - d * G_j * w_j'u) / (1 + d * w_j'u))`, a rank-one update of
# `M`. `G` itself comes from `gx = A * (G + W * (M \ v))`:
# `G = gx ./ a - W * ((W'W) \ (W' * ((1 .- a) ./ a .* gx)))`.
# S2Z cells score as in `_score_candidate`.
function _mean_totals_candidates!(Q, G, ir::BRMMeanTotalsCenteringReparametrization, x, gx)
    s, cols = ir.state, ir.columns
    for col in eachindex(cols.scales)
        lo, hi = cols.starts[col], cols.starts[col + 1] - 1
        J, P = hi - lo + 1, cols.widths[col]
        Wd = reshape(cols.design[cols.offsets[col] .+ (1:J*P)], J, P)
        ell = x[cols.scales[col]]
        idx = cols.indices[lo:hi]
        a = exp.((1 .- s.sources[lo:hi]) .* ell)
        xs, gs = x[idx], gx[idx]
        if P == 0
            loc, Gm, U, v = zeros(J), gs ./ a, zeros(0, J), zeros(0)
        else
            M = Wd' * (a .* Wd)
            loc = Wd * (M \ (Wd' * (a .* xs)))
            Gm = gs ./ a .- Wd * ((Wd' * Wd) \ (Wd' * ((1 .- a) ./ a .* gs)))
            v = Wd' * ((1 .- a) .* Gm)
            U = M \ Matrix(Wd')
        end
        totals = loc .+ a .* (xs .- loc)
        for j in 1:J
            u = view(U, :, j)
            sj = P == 0 ? 0.0 : BRM.LinearAlgebra.dot(view(Wd, j, :), u)
            rj = P == 0 ? 0.0 : BRM.LinearAlgebra.dot(v, u)
            p = lo + j - 1
            for (i, t) in enumerate(_CENTERING_CANDIDATES)
                a2 = exp((1 - t) * ell)
                d = a2 - a[j]
                Q[i, p] = loc[j] + (totals[j] - loc[j]) / a2
                G[i, p] = a2 * (Gm[j] + (rj - d * Gm[j] * sj) / (1 + d * sj))
            end
        end
    end
    for p in cols.starts[end]:length(s.sources)
        tau, c, m = exp(x[s.scales[p]]), s.sources[p], s.locations[p]
        f = tau^c
        z = (x[s.indices[p]] - c * m) / f
        h = f * gx[s.indices[p]]
        for (i, t) in enumerate(_CENTERING_CANDIDATES)
            Q[i, p] = t * m + tau^t * z
            G[i, p] = h / tau^t
        end
    end
    Q, G
end

# The window rule: WarmupHMC's 11 candidates and its default criterion, the
# position-gradient correlation `Cor(q_t, g_t)` over the window's equally
# weighted evidence, shared by every window-selected wrapper.
# `candidates!(Q, G, position, gradient)` fills every cell's candidate
# coordinates and gradients for one evidence column. Returns the new controls;
# a cell with no finite loss keeps its control.
function _window_select(old, positions, gradients, candidates!)
    n = size(positions, 2)
    shape = (length(_CENTERING_CANDIDATES), length(old))
    Q, G = zeros(shape), zeros(shape)
    mean_q, mean_g = zeros(shape), zeros(shape)
    qq, gg, qg = zeros(shape), zeros(shape), zeros(shape)
    for pass in 1:2, col in 1:n
        candidates!(Q, G, view(positions, :, col), view(gradients, :, col))
        if pass == 1
            mean_q .+= Q ./ n
            mean_g .+= G ./ n
        else
            qq .+= (Q .- mean_q) .^ 2
            gg .+= (G .- mean_g) .^ 2
            qg .+= (Q .- mean_q) .* (G .- mean_g)
        end
    end
    new = copy(old)
    for p in eachindex(old)
        losses = [qg[i, p] / sqrt(qq[i, p] * gg[i, p]) for i in axes(qg, 1)]
        finite = findall(isfinite, losses)
        isempty(finite) && continue
        new[p] = _CENTERING_CANDIDATES[finite[argmin(losses[finite])]]
    end
    new
end

# Every cell's candidates from its source-invariant frame (`_score_candidate`).
function _frame_candidates!(Q, G, frame)
    for p in axes(Q, 2), (i, t) in enumerate(_CENTERING_CANDIDATES)
        _, Q[i, p], G[i, p] = _score_candidate(frame, p, 0, nothing,
                                               WarmupHMC.PartiallyCentered(t))
    end
    Q, G
end

# Each cell's score is exact for that cell alone; the cells then move
# together, so a column's simultaneous update is a coordinate-wise step, not a
# joint optimum.
function _mean_totals_select!(ir::BRMMeanTotalsCenteringReparametrization, positions,
                              gradients)
    size(positions, 2) > 2 || return false
    old = WarmupHMC.reparam_controls(ir)
    new = _window_select(old, positions, gradients,
        (Q, G, x, g) -> _mean_totals_candidates!(Q, G, ir, x, g))
    new == old && return false
    WarmupHMC.restore_reparam_controls!(ir, new)
    true
end

# The mean-located wrapper over every sampled total and, with `s2z=true`, the
# S2Z contrasts after them.
function _mean_totals_reparametrizer(model, names; s2z::Bool)
    cells = s2z ? _scalar_centering_cells(model, names) :
        BRM._total_centering_cells(model, names)
    columns = BRM._total_centering_columns(model, names)
    columns.indices == cells.indices[1:length(columns.indices)] ||
        error("BRM mean-located totals: total cells and columns disagree")
    N = max(1, maximum(columns.widths; init=1))
    BRMMeanTotalsCenteringReparametrization{N}(_scalar_centering_state(cells), columns,
                                               Float64.(cells.targets), 1)
end

function _mean_totals_centering_problem(model, problem, ad_backend, names, centeredness)
    ir = _mean_totals_reparametrizer(model, names; s2z=true)
    # Totals and S2Z cells keep their compiled start under `:midpoint`.
    controls = _initial_controls(centeredness === :midpoint ? nothing : centeredness,
                                 length(ir.state.sources))
    isnothing(controls) || WarmupHMC.restore_reparam_controls!(ir, controls)
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend;
        scoring_plan=WarmupHMC.WindowSelectionPlan(_mean_totals_select!))
end

# Areal car cells (`BRM._adaptive_car_centering_blocks`): per block its `:icar`
# `phi` cells or its `:bym2` `theta` cells, one per area. A `theta` cell is
# centred on its area's total `intercept + a * phi`, read from coordinates no
# cell rewrites (`phi` stays the emitted standardized coordinate), so the cells
# are independent and map in any order. Like HSGP and cdar cells they keep their
# own state and argument types, away from the ordinary K<=2 Enzyme dispatch.
# Their accessors read only `cells`, a flat table of isbits
# `BRM._CARAdaptiveCell`s, and never `sources`. A first version that read the
# blocks from the joint wrapper's accessor aborted
# `test/adaptive_centering_threads.jl` with "GC error (probable corruption)" in
# its `joint` case, which has no car cell at all (Julia 1.10.12, Enzyme
# 0.13.216; base passed the same gate), and so did accessors that pass a cell to
# an outlined call: every one here is `@inline` (`BRM._CARAdaptiveCell`).
# `blocks`, `pair_blocks` and `pair_cells` serve only the controls report.
mutable struct BRMCARAdaptiveCenteringState
    blocks::Vector{BRM._CARAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_cells::Vector{Int}
    cells::Vector{BRM._CARAdaptiveCell}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMCARAdaptiveCenteringState(blocks)
    pair_blocks, pair_cells = Int[], Int[]
    cells = BRM._CARAdaptiveCell[]
    for (bi, block) in enumerate(blocks), cell in 1:BRM._adaptive_car_n_cells(block)
        push!(pair_blocks, bi)
        push!(pair_cells, cell)
        push!(cells, BRM._adaptive_car_cell(block, cell))
    end
    effect_indices = BitSet(c.index for c in cells)
    BRMCARAdaptiveCenteringState(collect(blocks), pair_blocks, pair_cells, cells,
                                 zeros(length(cells)), effect_indices)
end

_car_pair_location(state, p) = (state.pair_blocks[p], state.pair_cells[p])
_car_pair_index(state, p) = state.cells[p].index
@inline _car_log_scale(x, state, p) = BRM._adaptive_car_log_scale(x, state.cells[p])
@inline _car_location(x, state, p) = BRM._adaptive_car_location(x, state.cells[p])

struct BRMCARAdaptiveCenteringArgument{KIND} <: Function
    state::BRMCARAdaptiveCenteringState
    pair_number::Int
end

function BRMCARAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM car adaptive-centering argument kind $kind",
    )
    BRMCARAdaptiveCenteringArgument{kind}(state, pair_number)
end

(arg::BRMCARAdaptiveCenteringArgument{:location})(x) =
    _car_location(x, arg.state, arg.pair_number)
(arg::BRMCARAdaptiveCenteringArgument{:log_scale})(x) =
    _car_log_scale(x, arg.state, arg.pair_number)

function _sync_sources!(state::BRMCARAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM car adaptive-centering plan has $(length(state.sources)) cells but " *
        "the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _car_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM car adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMCARAdaptiveCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

# Each cell's frame `x = c * m + s^c * z`: its location `m` and scale `s` read
# coordinates no cell rewrites, so the innovation gradient is `s^c * g`.
function _centering_frame(state::BRMCARAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in 1:n
        idx = _car_pair_index(state, p)
        c = source[p]
        s = exp(_car_log_scale(position, state, p))
        m = convert(T, _car_location(position, state, p))
        scale[p] = s
        location[p] = m
        innovation[p] = (position[idx] - c * m) / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_car_centering_reparametrizer(blocks)
    state = BRMCARAdaptiveCenteringState(blocks)
    pairs = [_car_pair_index(state, p) => WarmupHMC.Reparametrization(
                 WarmupHMC.PartiallyCentered(0.0), WarmupHMC.PartiallyCentered(0.0),
                 BRMCARAdaptiveCenteringArgument(state, p, :location),
                 BRMCARAdaptiveCenteringArgument(state, p, :log_scale))
             for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# cdar walk cells are scalar zero-location cells like HSGP basis weights, so
# they reuse that frame shape one-to-one: per-cell marginal spread, diagonal
# transport, linear cost. They keep their own state/argument types for the same
# reason — the ordinary-block exact small-block dispatch must not see them. A
# model whose only cells are walk cells keeps its own plan (its pair order and
# checkpoints are unchanged); beside other cells they join the joint wrapper,
# after its car cells.
mutable struct BRMCDARAdaptiveCenteringState
    blocks::Vector{BRM._CDARAdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_cells::Vector{Int}
    sources::Vector{Float64}
    effect_indices::BitSet
end

function BRMCDARAdaptiveCenteringState(blocks)
    pair_blocks = Int[]
    pair_cells = Int[]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), cell in eachindex(block.effects)
        push!(pair_blocks, bi)
        push!(pair_cells, cell)
        push!(sources, 0.0)
    end
    effect_indices = BitSet(Iterators.flatten(b.effects for b in blocks))
    BRMCDARAdaptiveCenteringState(
        collect(blocks), pair_blocks, pair_cells, sources, effect_indices,
    )
end

# Joint ordinary + R2D2 + HSGP + scalar (totals/S2Z) + car adaptation.
# `IndexedReparametrization` requires one concrete pair element type, so the
# families cannot contribute their own accessor closures to one vector. Every
# joint pair therefore carries the same `BRMJointAdaptiveCenteringArgument`,
# which selects by pair-number range: pairs `1:n_ranef` are ordinary cells, the
# next `n_r2d2` are R2D2 cells, then `n_hsgp` HSGP basis-weight cells, then
# `n_level` HSGP level cells, then `n_scalar` scalar total/S2Z cells, then
# `n_car` car cells, and the rest are cdar walk cells. A model without R2D2
# blocks has `n_r2d2 = 0` and one without level, scalar, car or cdar cells has
# none, so each keeps the historical ordinary-then-HSGP pair order.
#
# Ordinary and R2D2 cells share one derived-scale state: an ordinary block
# joins it in derived-scale form (`BRM._adaptive_derived_block`), whose `tau` is
# `exp(0.0 + 1.0 * x[i])`, bit-identical to `exp(x[i])`. On Julia 1.10, a
# differentiated joint accessor that reaches two `_block_location`
# instantiations (ordinary and R2D2) corrupts Julia's GC under Enzyme's reverse
# pass once another thread runs GC — even when the R2D2 branch never executes —
# while one instantiation is clean (see `_block_location`). HSGP and car cells
# keep their scalar accessors.
struct BRMJointAdaptiveCenteringState{N}
    ranef::BRMR2D2AdaptiveCenteringState{N}
    hsgp::BRMHSGPAdaptiveCenteringState
    level::BRMHSGPLevelCenteringState{N}
    scalar::BRMScalarCenteringState
    car::BRMCARAdaptiveCenteringState
    cdar::BRMCDARAdaptiveCenteringState
    n_ranef::Int
    n_r2d2::Int
    n_hsgp::Int
    n_level::Int
    n_scalar::Int
    n_car::Int
end

struct BRMJointAdaptiveCenteringArgument{KIND,N} <: Function
    state::BRMJointAdaptiveCenteringState{N}
    pair_number::Int
end

function BRMJointAdaptiveCenteringArgument(
        state::BRMJointAdaptiveCenteringState{N}, pair_number, kind::Symbol,
    ) where {N}
    kind in (:location, :log_scale) || error(
        "unknown BRM joint adaptive-centering argument kind $kind",
    )
    BRMJointAdaptiveCenteringArgument{kind,N}(state, pair_number)
end

function (arg::BRMJointAdaptiveCenteringArgument{:location})(x)
    p = arg.pair_number
    s = arg.state
    p <= s.n_ranef + s.n_r2d2 && return _block_location(x, s.ranef, p)
    q = p - s.n_ranef - s.n_r2d2 - s.n_hsgp
    q <= 0 && return zero(eltype(x))
    q <= s.n_level && return _level_location(x, x, s.level, q)
    q -= s.n_level
    q <= s.n_scalar && return convert(eltype(x), s.scalar.locations[q])
    q -= s.n_scalar
    q <= s.n_car && return _car_location(x, s.car, q)
    zero(eltype(x))
end

function (arg::BRMJointAdaptiveCenteringArgument{:log_scale})(x)
    p = arg.pair_number
    s = arg.state
    if p <= s.n_ranef + s.n_r2d2
        bi, k, _ = _pair_location(s.ranef, p)
        log(_block_cholesky_entry(x, s.ranef.blocks[bi], k, k) *
            _pair_mixing_scale(x, s.ranef, p))
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp
        bi, basis = _hsgp_pair_location(s.hsgp, p - s.n_ranef - s.n_r2d2)
        BRM._adaptive_hsgp_log_scale(x, s.hsgp.blocks[bi], basis)
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp + s.n_level
        _level_pair_log_scale(x, s.level, p - s.n_ranef - s.n_r2d2 - s.n_hsgp)
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp + s.n_level + s.n_scalar
        x[s.scalar.scales[p - s.n_ranef - s.n_r2d2 - s.n_hsgp - s.n_level]]
    elseif p <= s.n_ranef + s.n_r2d2 + s.n_hsgp + s.n_level + s.n_scalar + s.n_car
        _car_log_scale(x, s.car,
                       p - s.n_ranef - s.n_r2d2 - s.n_hsgp - s.n_level - s.n_scalar)
    else
        bi, cell = _cdar_pair_location(s.cdar, p - s.n_ranef - s.n_r2d2 - s.n_hsgp -
                                                 s.n_level - s.n_scalar - s.n_car)
        BRM._adaptive_cdar_log_scale(x, s.cdar.blocks[bi], cell)
    end
end

# The family state owning joint pair `i`, and the pair's number within it.
function _joint_pair_family(state::BRMJointAdaptiveCenteringState, i)
    i <= state.n_ranef + state.n_r2d2 && return (:ranef, i)
    i -= state.n_ranef + state.n_r2d2
    i <= state.n_hsgp && return (:hsgp, i)
    i -= state.n_hsgp
    i <= state.n_level && return (:level, i)
    i -= state.n_level
    i <= state.n_scalar && return (:scalar, i)
    i -= state.n_scalar
    i <= state.n_car && return (:car, i)
    (:cdar, i - state.n_car)
end

_joint_pair_index(state, family, j) =
    family === :hsgp ? _hsgp_pair_index(state.hsgp, j) :
    family === :level ? _level_pair_index(state.level, j) :
    family === :scalar ? state.scalar.indices[j] :
    family === :car ? _car_pair_index(state.car, j) :
    family === :cdar ? _cdar_pair_index(state.cdar, j) : _pair_index(state.ranef, j)

function _sync_sources!(state::BRMJointAdaptiveCenteringState, ir)
    n_cells = state.n_ranef + state.n_r2d2 + state.n_hsgp + state.n_level +
              state.n_scalar + state.n_car + length(state.cdar.sources)
    length(ir.pairs) == n_cells || throw(DimensionMismatch(
        "BRM joint adaptive-centering plan has $n_cells cells but the " *
        "WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (i, (idx, value)) in enumerate(ir.pairs)
        family, j = _joint_pair_family(state, i)
        expected = _joint_pair_index(state, family, j)
        idx == expected || throw(ArgumentError(
            "BRM joint adaptive-centering pair $i addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        getfield(state, family).sources[j] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMJointAdaptiveCenteringState, ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMJointAdaptiveCenteringState, position, gradient,
                          ordered=nothing)
    frames = (_centering_frame(state.ranef, position, gradient, ordered),
              _centering_frame(state.hsgp, position, gradient),
              _centering_frame(state.level, position, gradient),
              _centering_frame(state.scalar, position, gradient),
              _centering_frame(state.car, position, gradient),
              _centering_frame(state.cdar, position, gradient))
    # Field by field: a splatted generator over `fieldnames` does not infer, and
    # an uninferred frame makes every candidate score a dynamic dispatch.
    BRMAdaptiveCenteringFrame(
        vcat(map(f -> f.source, frames)...),
        vcat(map(f -> f.location, frames)...),
        vcat(map(f -> f.scale, frames)...),
        vcat(map(f -> f.innovation, frames)...),
        vcat(map(f -> f.invariant_gradient, frames)...),
    )
end

const _NO_SCALAR_CELLS = (; indices=Int[], scales=Int[], locations=Float64[],
                           targets=Float64[])

const _NO_CAR_BLOCKS = BRM._CARAdaptiveCenteringBlock[]

const _NO_CDAR_BLOCKS = BRM._CDARAdaptiveCenteringBlock[]

# The joint state of ordinary, R2D2, HSGP, scalar, car and cdar cells, after
# refusing every coordinate overlap that would make the transform inexact.
# Shared by the per-pair wrapper below and the learned-order wrapper
# (`_adaptive_ordered_centering_reparametrizer`).
function _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                car_blocks=_NO_CAR_BLOCKS; locations=nothing,
                                level_blocks=_NO_LEVEL_BLOCKS,
                                cdar_blocks=_NO_CDAR_BLOCKS)
    # One innovation capacity for the random-effect and HSGP level cells, so
    # the joint state keeps one type parameter; it is the random-effect one
    # whenever there are no level cells.
    N = max(_innovation_capacity(vcat(BRM.AbstractAdaptiveCenteringBlock[], blocks,
                                      r2d2_blocks)),
            isempty(level_blocks) ? 1 : _level_capacity(level_blocks))
    hsgp_state = BRMHSGPAdaptiveCenteringState(hsgp_blocks)
    level_state = BRMHSGPLevelCenteringState{N}(level_blocks)
    scalar_state = _scalar_centering_state(scalar_cells)
    car_state = BRMCARAdaptiveCenteringState(car_blocks)
    cdar_state = BRMCDARAdaptiveCenteringState(cdar_blocks)
    block_effects(bs) = BitSet(Iterators.flatten(vec(b.effects) for b in bs))
    effects = (block_effects(blocks), block_effects(r2d2_blocks),
               hsgp_state.effect_indices, level_state.effect_indices,
               BitSet(scalar_state.indices), car_state.effect_indices,
               cdar_state.effect_indices)
    sum(length, effects) == length(union(effects...)) || error(
        "BRM joint adaptive centering: ordinary, R2D2, HSGP, total/S2Z, car and " *
        "cdar cells claim overlapping unconstrained coordinates; refusing an " *
        "ambiguous transform.",
    )
    for block in level_blocks
        isdisjoint(BRM._adaptive_read_coordinates(block), union(effects...)) || error(
            "BRM joint adaptive centering: grouped HSGP `$(block.term)` reads its " *
            "level correlation or scales from a coordinate another cell rewrites; " *
            "refusing an inexact transform.")
    end
    for block in car_blocks
        reads = filter(!=(0), vcat(block.sigma, block.rho, block.intercept,
                                   block.bym2 ? block.structured : Int[]))
        isdisjoint(reads, union(effects...)) || error(
            "BRM joint adaptive centering: car `$(block.term)` reads its scale, " *
            "intercept or `phi` from a coordinate another cell rewrites; refusing " *
            "an inexact transform.")
    end
    for block in cdar_blocks
        isdisjoint((block.sigma, block.rho), union(effects...)) || error(
            "BRM joint adaptive centering: cdar `$(block.term)` reads its marginal " *
            "SD or persistence from a coordinate another cell rewrites; refusing " *
            "an inexact transform.")
    end
    for block in r2d2_blocks
        read = BRM._adaptive_read_coordinates(block)
        isdisjoint(read, car_state.effect_indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from a car field " *
                  "coordinate this transform rewrites; refusing an inexact " *
                  "transform.")
        isdisjoint(read, cdar_state.effect_indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from a cdar walk " *
                  "innovation this transform rewrites; refusing an inexact " *
                  "transform.")
        isdisjoint(read, hsgp_state.effect_indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from an HSGP " *
                  "basis weight this transform rewrites; refusing an inexact " *
                  "transform.")
        isdisjoint(read, scalar_state.indices) ||
            error("BRM joint adaptive centering: block " *
                  "`$(block.ranef.binding)` derives its scale from a total/S2Z " *
                  "coordinate this transform rewrites; refusing an inexact " *
                  "transform.")
    end
    isdisjoint(scalar_state.scales, union(effects...)) || error(
        "BRM joint adaptive centering: a total/S2Z scale coordinate is rewritten " *
        "by another cell; refusing an inexact transform.")
    # Block order fixes pair order, which is load-bearing across
    # checkpoint/resume (positional restore): ordinary cells first, then R2D2
    # cells, each in the family's own block/group/term order, then HSGP
    # basis-weight cells, then HSGP level cells (term, basis, level), then
    # scalar total and S2Z cells
    # (`_scalar_centering_cells` order), then car cells (per block, in
    # `_adaptive_car_cell` order), then cdar walk cells (per block, its `eta`
    # in column-major group-by-step order).
    ranef_blocks = vcat(
        BRM.R2D2AdaptiveCenteringBlock[BRM._adaptive_derived_block(b) for b in blocks],
        r2d2_blocks)
    ranef_state = BRMR2D2AdaptiveCenteringState{N}(ranef_blocks,
        _ranef_pair_layout(ranef_blocks, isnothing(locations) ? nothing :
            vcat(_block_locations(blocks, locations),
                 _block_locations(r2d2_blocks, locations)),
            vcat(BRM.AbstractAdaptiveCenteringBlock[], blocks, r2d2_blocks))...)
    _check_locations_read_only(ranef_state, union(effects...))
    n_ranef = sum(b -> b.ranef.n_terms * b.ranef.n_groups, blocks; init=0)
    n_r2d2 = length(ranef_state.sources) - n_ranef
    n_hsgp = length(hsgp_state.sources)
    BRMJointAdaptiveCenteringState(ranef_state, hsgp_state, level_state, scalar_state,
                                   car_state, cdar_state, n_ranef, n_r2d2, n_hsgp,
                                   length(level_state.sources),
                                   length(scalar_state.sources),
                                   length(car_state.sources))
end

# The compiled frame of every joint cell, in pair order. Car fields and cdar
# innovations are always emitted standardized (`c=0`).
function _joint_targets(state::BRMJointAdaptiveCenteringState, scalar_cells)
    ranef = [state.ranef.blocks[state.ranef.pair_blocks[p]].target_c
             for p in eachindex(state.ranef.sources)]
    hsgp = [state.hsgp.blocks[bi].target_c[basis]
            for (bi, basis) in zip(state.hsgp.pair_blocks, state.hsgp.pair_basis)]
    # Grouped HSGP weights are emitted standardized (`c=0`).
    vcat(ranef, hsgp, zeros(state.n_level), Float64.(scalar_cells.targets),
         zeros(state.n_car), zeros(length(state.cdar.sources)))
end

_joint_cell_index(state::BRMJointAdaptiveCenteringState, p) =
    _joint_pair_index(state, _joint_pair_family(state, p)...)

function _adaptive_joint_centering_reparametrizer(
        blocks, hsgp_blocks, r2d2_blocks=BRM.R2D2AdaptiveCenteringBlock[],
        scalar_cells=_NO_SCALAR_CELLS, car_blocks=_NO_CAR_BLOCKS; locations=nothing,
        level_blocks=_NO_LEVEL_BLOCKS, cdar_blocks=_NO_CDAR_BLOCKS)
    state = _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                   car_blocks; locations, level_blocks, cdar_blocks)
    pairs = [_joint_cell_index(state, p) => WarmupHMC.Reparametrization(
                 WarmupHMC.PartiallyCentered(c), WarmupHMC.PartiallyCentered(c),
                 BRMJointAdaptiveCenteringArgument(state, p, :location),
                 BRMJointAdaptiveCenteringArgument(state, p, :log_scale))
             for (p, c) in enumerate(_joint_targets(state, scalar_cells))]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# ---- Learned triangular order (`ranef_order = :learned`) --------------------
#
# A correlated block's triangular map centres margin `k` on its conditional mean
# given the margins before it, and the per-pair wrappers above take that order
# from the formula declaration order. A data-poor margin declared before
# data-rich ones can then never be "non-centred given the data-rich margins",
# for any controls (snag correlated-kreis-15564811). The learned order re-derives
# each group's order from that group's controls (`_learned_order`): descending
# `c`, ties in declaration order. Centred margins come first; every other margin
# is centred to its own degree around its conditional mean given the margins
# ahead of it.
#
# The conditioning order is a permutation `P` of the block covariance
# `Sigma = C * C'`, `C = diag(tau) * L`: position `j` holds margin `P[j]`, and
# `Cp = cholesky(P * Sigma * P')` replaces `C`. Source coordinate `x[P[j]]` is
# `c * m[j] + Cp[j,j]^c * w[j]` with `m[j] = sum(Cp[j,i] * w[i] for i < j)`; the
# effects are `b[P[j]] = m[j] + Cp[j,j] * w[j]`, and the compiled coordinates are
# the declared-order frame of `b` at the block's compiled `target_c`. For a block
# compiled non-centred, source -> compiled is then a rotation rather than
# per-coordinate pairs, so this wrapper is a custom reparametrization that a
# `WindowSelectionPlan` adapts (`_ordered_select!`); HSGP, total, S2Z and car
# cells keep their per-cell maps inside it. Its controls are the joint wrapper's cell
# controls in the joint pair order, and the order is a pure function of them, so
# checkpoints carry nothing else.
#
# The block's own correlation coordinates are reordered too. Stan's
# `cholesky_factor_corr` parametrizes `L` by canonical partial correlations
# taken in declaration order, and that geometry alone left a data-poor margin
# declared first mixing about 4x worse with identical effect frames (synthetic
# `rho_sr` bulk ESS 120 against 479). So the block has one order of its own,
# its margins by mean control (`_derive_orders!`), and the source frame's
# Cholesky coordinates are the partial correlations of the correlation matrix
# in that order. The compiled ones are recovered through the declared-order
# factor, and the Jacobian is the difference of the two correlation
# transforms (`_correlation_logjac`). Each group's ties follow the block order,
# so most groups share it and need no rotations. At uniform controls every order
# is the declared one, so the wrapper starts in the compiled frame.
#
# The map runs under Enzyme on every gradient, so it follows `_block_location`'s
# Julia 1.10 rules: per-block factors live in `NTuple`s sized by the state's
# capacity, never in heap work vectors, and the random-effect loops read only
# flat integer arrays.
struct BRMOrderedCenteringReparametrization{N} <: WarmupHMC.AbstractReparametrization
    state::BRMJointAdaptiveCenteringState{N}
    # The compiled frame of every cell, in joint pair order.
    targets::Vector{Float64}
    # Per ranef block: `orders[bi][j, g]` is the margin at position `j` of group
    # `g`. The distinct orders are `perms[bi][:, q]`, and the groups sharing
    # order `q` are `groups[bi][starts[bi][q]:starts[bi][q+1]-1]`, so the
    # permuted factor is built once per distinct order.
    orders::Vector{Matrix{Int}}
    perms::Vector{Matrix{Int}}
    starts::Vector{Vector{Int}}
    groups::Vector{Vector{Int}}
    # Per ranef block: its own order (margins by mean control), each margin's
    # position in it, and, per distinct group order `q`, that order's rows of
    # the block-order factor (`rows[bi][j, q]`) and its kind: `0` the declared
    # order, `1` the block order, `2` any other.
    block_orders::Vector{Vector{Int}}
    positions::Vector{Vector{Int}}
    rows::Vector{Matrix{Int}}
    kinds::Vector{Vector{Int}}
    # `1`: source -> compiled; `-1`: compiled -> source.
    direction::Int
end

# `Base.setindex(t, v, i)` with a runtime `i`, as a branch-free select over every
# entry. Base's recursive version falls off a cliff past 32 entries (measured:
# 1.9 us and an 80 KB heap allocation per write at 49 entries, against 6 ns and
# none here), and a heap allocation inside an Enzyme-differentiated map is the
# Julia 1.10 GC hazard `_block_location` describes. The square Givens buffer
# reaches 49 entries at `K = 7`, the packed factor 36 at `K = 8`.
@inline _tset(t::NTuple{M,T}, v, i) where {M,T} = ntuple(k -> ifelse(k == i, T(v), t[k]), Val(M))

# Packed lower-triangular storage: entry `(i, j)`, `j <= i`.
@inline _tri(i, j) = (i * (i - 1)) ÷ 2 + j
# Row-major `n x n` storage.
@inline _sq(i, j, n) = (i - 1) * n + j

_packed_length(::Val{K}) where {K} = Val((K * (K + 1)) ÷ 2)

# `C = diag(tau[order]) * L`, packed, where `L` is built from the block's
# Cholesky coordinates and row `i` holds margin `order[i]`; the declared order
# gives `diag(tau) * L` with the same operations and association as
# `_block_cholesky_entry`.
function _packed_block_cholesky(x, block, order, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    C = ntuple(_ -> zero(T), _packed_length(Val(KMAX)))
    for i in 1:block.ranef.n_terms
        tau = _block_tau(x, block, order[i])
        stick = one(T)
        offset = (i - 1) * (i - 2) ÷ 2
        for l in 1:i-1
            z = _block_cpc(x, block, offset + l)
            C = _tset(C, tau * (stick * z), _tri(i, l))
            stick *= sqrt(one(T) - z * z)
        end
        C = _tset(C, tau * stick, _tri(i, i))
    end
    C
end

# `cholesky(P * C * C' * P')`, packed, for the order `order[1:K]`: the LQ factor
# of the row-permuted `P * C`, by Givens rotations of its columns. Each diagonal
# is the norm of a residual row, never a difference of squares, so a nearly
# singular correlation cannot round a conditional variance below zero (the
# difference-of-squares Cholesky did, mid-warm-up). `A` is `KMAX x KMAX`,
# row-major.
function _permuted_cholesky(C::NTuple{M,T}, order, K, ::Val{KMAX}) where {M,T,KMAX}
    A = ntuple(_ -> zero(T), Val(KMAX * KMAX))
    for j in 1:K
        for l in 1:order[j]
            A = _tset(A, C[_tri(order[j], l)], _sq(j, l, KMAX))
        end
    end
    for j in 1:K
        for l in j+1:K
            a, b = A[_sq(j, j, KMAX)], A[_sq(j, l, KMAX)]
            r = sqrt(a * a + b * b)
            iszero(r) && continue
            c, s = a / r, b / r
            for i in j:K
                u, v = A[_sq(i, j, KMAX)], A[_sq(i, l, KMAX)]
                A = _tset(A, c * u + s * v, _sq(i, j, KMAX))
                A = _tset(A, c * v - s * u, _sq(i, l, KMAX))
            end
        end
        if A[_sq(j, j, KMAX)] < zero(T)
            for i in j:K
                A = _tset(A, -A[_sq(i, j, KMAX)], _sq(i, j, KMAX))
            end
        end
    end
    Cp = ntuple(_ -> zero(T), Val(M))
    for j in 1:K, i in 1:j
        Cp = _tset(Cp, A[_sq(j, i, KMAX)], _tri(j, i))
    end
    Cp
end

_unpacked(C, K) = [j <= i ? C[_tri(i, j)] : zero(eltype(C)) for i in 1:K, j in 1:K]

# A packed factor times the scalar `r`: a Student-t level's `sqrt(w[g]) * C`,
# and its permuted factor, since `cholesky(P * (r^2 * Sigma) * P') =
# r * cholesky(P * Sigma * P')`.
_tscale(C::NTuple, r) = map(v -> r * v, C)

# Stan's `cholesky_factor_corr` log-Jacobian from its unconstrained coordinates
# to the correlation matrix, at the packed factor `F` (rows may carry scales):
# `sum(log(1 - z^2)) + sum(log(stick)) + sum((K - i) * log(L[i,i]))`, the tanh,
# stick-breaking and `L -> L * L'` terms. With `n[i,l]` the norm of row `i` from
# column `l` on, `1 - z[i,l]^2 = (n[i,l+1] / n[i,l])^2`, `stick[i,l] =
# n[i,l] / n[i,1]` and `L[i,i] = n[i,i] / n[i,1]`; everything is a ratio of row
# norms, so no difference of squares can cancel.
function _correlation_logjac(F, K)
    T = eltype(F)
    total = zero(T)
    for i in 2:K
        suffix = zero(T)
        log_ii = zero(T)
        log_next = zero(T)
        acc = zero(T)
        for l in i:-1:1
            suffix += F[_tri(i, l)] * F[_tri(i, l)]
            log_n = log(suffix) / 2
            if l == i
                log_ii = log_n
            else
                acc += 2 * log_next - log_n
            end
            log_next = log_n
        end
        total += acc - (i - 1) * log_next + (K - i) * (log_ii - log_next)
    end
    total
end

# Writes the unconstrained Cholesky coordinates of the packed factor `F`,
# `atanh(z)` with `z[i,l] = F[i,l] / n[i,l]` (`n` as above), spelled
# `sign(F) * log((n[i,l] + |F|) / n[i,l+1])`: rounding `z` to `±1` first would
# turn a nearly singular correlation into an infinite coordinate.
function _write_cholesky_coordinates!(y, block, F, K)
    for i in 2:K
        offset = (i - 1) * (i - 2) ÷ 2
        suffix = F[_tri(i, i)] * F[_tri(i, i)]
        for l in i-1:-1:1
            f = F[_tri(i, l)]
            rest = sqrt(suffix)
            suffix += f * f
            y[block.cholesky_free[offset + l]] = copysign(log((sqrt(suffix) + abs(f)) / rest), f)
        end
    end
    y
end

function _is_declared_order(order, K)
    for j in 1:K
        order[j] == j || return false
    end
    true
end

# Margins in descending control; ties by `tiebreak` (default: declaration
# order).
_learned_order(controls, tiebreak=eachindex(controls)) =
    sortperm(collect(eachindex(controls)); by=k -> (-controls[k], tiebreak[k]))

_known_correlation(::BRM.AdaptiveCenteringBlock) = false
_known_correlation(block::BRM.R2D2AdaptiveCenteringBlock) = !isempty(block.cholesky_known)

function _derive_orders!(ir::BRMOrderedCenteringReparametrization)
    state = ir.state.ranef
    for (bi, block) in enumerate(state.blocks)
        K, G = block.ranef.n_terms, block.ranef.n_groups
        lookup = state.pair_lookup[bi]
        controls = [state.sources[lookup[k, g]] for k in 1:K, g in 1:G]
        # A known correlation has no Cholesky coordinates to reorder, so its
        # block keeps the declared order; its groups still learn their own.
        block_order = _known_correlation(block) ? collect(1:K) :
            _learned_order(vec(sum(controls; dims=2)) ./ G)
        position = invperm(block_order)
        order = reduce(hcat, [_learned_order(controls[:, g], position) for g in 1:G];
                       init=zeros(Int, K, 0))
        members = Dict{Vector{Int},Vector{Int}}()
        distinct = Vector{Int}[]
        for g in 1:G
            key = order[:, g]
            haskey(members, key) || push!(distinct, key)
            push!(get!(members, key, Int[]), g)
        end
        ir.orders[bi] = order
        ir.perms[bi] = reduce(hcat, distinct; init=zeros(Int, K, 0))
        ir.starts[bi] = cumsum(vcat(1, [length(members[key]) for key in distinct]))
        ir.groups[bi] = reduce(vcat, (members[key] for key in distinct); init=Int[])
        ir.block_orders[bi] = block_order
        ir.positions[bi] = position
        ir.rows[bi] = reduce(hcat, [position[key] for key in distinct];
                             init=zeros(Int, K, 0))
        ir.kinds[bi] = [key == 1:K ? 0 : key == block_order ? 1 : 2 for key in distinct]
    end
    ir
end

function _adaptive_ordered_centering_reparametrizer(
        blocks, hsgp_blocks, r2d2_blocks=BRM.R2D2AdaptiveCenteringBlock[],
        scalar_cells=_NO_SCALAR_CELLS, car_blocks=_NO_CAR_BLOCKS; locations=nothing,
        level_blocks=_NO_LEVEL_BLOCKS, cdar_blocks=_NO_CDAR_BLOCKS)
    state = _joint_centering_state(blocks, hsgp_blocks, r2d2_blocks, scalar_cells,
                                   car_blocks; locations, level_blocks, cdar_blocks)
    n = length(state.ranef.blocks)
    matrices() = [Matrix{Int}(undef, 0, 0) for _ in 1:n]
    vectors() = [Int[] for _ in 1:n]
    ir = BRMOrderedCenteringReparametrization(state, _joint_targets(state, scalar_cells),
        matrices(), matrices(), vectors(), vectors(), vectors(), vectors(), matrices(),
        vectors(), 1)
    state, _derive_orders!(ir)
end

# One group of one block, source -> compiled. Every source value is read before
# any output is written, so `y === x` is safe.
function _ordered_group_forward!(y, x, state, block, lookup, g, C, Cp, perms, q,
                                 declared, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    K = block.ranef.n_terms
    t = block.target_c
    u = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        u = _tset(u, x[block.effects[k, g]], k)
    end
    w = ntuple(_ -> zero(T), Val(KMAX))
    b = ntuple(_ -> zero(T), Val(KMAX))
    ljac = zero(T)
    for j in 1:K
        a = perms[j, q]
        c = state.sources[lookup[a, g]]
        m = zero(T)
        for i in 1:j-1
            m += Cp[_tri(j, i)] * w[i]
        end
        s = Cp[_tri(j, j)]
        # The source-frame location adds a population coefficient for a pair
        # centred on one (`_with_location`; only blocks compiled at `t = 0`
        # carry one, so `t * m` keeps the model-frame mean).
        loc = _with_location(m, x, state, lookup[a, g])
        w = _tset(w, (u[a] - c * loc) / s^c, j)
        if declared
            # WarmupHMC's per-pair spelling (`warmuphmc-use` §5c), so the
            # declared order reproduces the per-pair wrapper.
            log_scale = log(s)
            y[block.effects[a, g]] = t * loc + (u[a] - c * loc) * exp(log_scale * (t - c))
            ljac += log_scale * (t - c)
        else
            b = _tset(b, m + s * w[j], a)
            ljac += (1 - c) * log(s)
        end
    end
    declared && return ljac
    # Effects -> the compiled declared-order frame at `target_c`.
    z = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        m = zero(T)
        for l in 1:k-1
            m += C[_tri(k, l)] * z[l]
        end
        s = C[_tri(k, k)]
        z = _tset(z, (b[k] - m) / s, k)
        y[block.effects[k, g]] = t * m + s^t * z[k]
        ljac += (t - 1) * log(s)
    end
    ljac
end

# One group of one block, compiled -> source; the exact inverse of
# `_ordered_group_forward!`.
function _ordered_group_inverse!(y, x, state, block, lookup, g, C, Cp, perms, q,
                                 declared, ::Val{KMAX}) where {KMAX}
    T = eltype(x)
    K = block.ranef.n_terms
    t = block.target_c
    v = ntuple(_ -> zero(T), Val(KMAX))
    for k in 1:K
        v = _tset(v, x[block.effects[k, g]], k)
    end
    # The compiled frame -> innovations `z` and effects `b`, declared order.
    z = ntuple(_ -> zero(T), Val(KMAX))
    b = ntuple(_ -> zero(T), Val(KMAX))
    ljac = zero(T)
    for k in 1:K
        m = zero(T)
        for l in 1:k-1
            m += C[_tri(k, l)] * z[l]
        end
        s = C[_tri(k, k)]
        z = _tset(z, (v[k] - t * m) / s^t, k)
        if declared
            c = state.sources[lookup[k, g]]
            log_scale = log(s)
            y[block.effects[k, g]] = c * _with_location(m, x, y, state, lookup[k, g]) +
                (v[k] - t * m) * exp(log_scale * (c - t))
            ljac += log_scale * (c - t)
        else
            b = _tset(b, m + s * z[k], k)
            ljac += (1 - t) * log(s)
        end
    end
    declared && return ljac
    w = ntuple(_ -> zero(T), Val(KMAX))
    for j in 1:K
        a = perms[j, q]
        c = state.sources[lookup[a, g]]
        m = zero(T)
        for i in 1:j-1
            m += Cp[_tri(j, i)] * w[i]
        end
        s = Cp[_tri(j, j)]
        w = _tset(w, (b[a] - m) / s, j)
        y[block.effects[a, g]] = c * _with_location(m, x, y, state, lookup[a, g]) + s^c * w[j]
        ljac += (c - 1) * log(s)
    end
    ljac
end

function _ordered_block_map!(y, x, ir::BRMOrderedCenteringReparametrization{N},
                             bi) where {N}
    state = ir.state.ranef
    block = state.blocks[bi]
    K = block.ranef.n_terms
    lookup = state.pair_lookup[bi]
    perms = ir.perms[bi]
    starts = ir.starts[bi]
    groups = ir.groups[bi]
    rows = ir.rows[bi]
    kinds = ir.kinds[bi]
    block_order = ir.block_orders[bi]
    reordered = !_is_declared_order(block_order, K)
    # `C`: the declared-order factor of the compiled frame; `Cs`: the block-order
    # factor of the source frame, whose Cholesky coordinates are permuted.
    if ir.direction > 0
        Cs = _packed_block_cholesky(x, block, block_order, Val(N + 1))
        C = reordered ? _permuted_cholesky(Cs, ir.positions[bi], K, Val(N + 1)) : Cs
    else
        C = _packed_block_cholesky(x, block, Base.OneTo(K), Val(N + 1))
        Cs = reordered ? _permuted_cholesky(C, block_order, K, Val(N + 1)) : C
    end
    ljac = zero(eltype(x))
    if reordered
        from, to = ir.direction > 0 ? (Cs, C) : (C, Cs)
        _write_cholesky_coordinates!(y, block, to, K)
        ljac += _correlation_logjac(from, K) - _correlation_logjac(to, K)
    end
    for q in axes(perms, 2)
        declared = kinds[q] == 0
        Cp = declared ? C : kinds[q] == 1 ? Cs :
            _permuted_cholesky(Cs, view(rows, :, q), K, Val(N + 1))
        for r in starts[q]:starts[q + 1] - 1
            g = groups[r]
            # A Student-t level's factors carry its `sqrt(w[g])`; the block's
            # correlation coordinates and their Jacobian above are scale-free.
            # The scale comes from the flat `pair_mixing` (`_pair_mixing_scale`).
            p = lookup[1, g]
            Cg, Cpg = state.pair_mixing[p] == 0 ? (C, Cp) :
                (_tscale(C, _pair_mixing_scale(x, state, p)),
                 _tscale(Cp, _pair_mixing_scale(x, state, p)))
            ljac += ir.direction > 0 ?
                _ordered_group_forward!(y, x, state, block, lookup, g, Cg, Cpg,
                                        perms, q, declared, Val(N + 1)) :
                _ordered_group_inverse!(y, x, state, block, lookup, g, Cg, Cpg,
                                        perms, q, declared, Val(N + 1))
        end
    end
    ljac
end

# WarmupHMC's per-pair partially centred map (`warmuphmc-use` §5c),
# `y = t*loc + (x - c*loc) * exp(log_scale * (t - c))`; its inverse swaps `t`
# and `c`.
function _cell_map!(y, x, idx, location, log_scale, target, source, direction)
    from, to = direction > 0 ? (source, target) : (target, source)
    y[idx] = to * location + (x[idx] - from * location) * exp(log_scale * (to - from))
    log_scale * (to - from)
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        ir::BRMOrderedCenteringReparametrization, x::AbstractVector)
    s = ir.state
    ljac = zero(eltype(x))
    # A location that reads a containing cell reads its source: the inverse
    # rebuilds it first (blocks in order, `_ordered_group_inverse!` reads `y`);
    # the forward map walks the blocks backwards so that with `y === x` the
    # containing (earlier) blocks are still unwritten.
    blocks = s.ranef.extended && ir.direction > 0 ?
        reverse(eachindex(s.ranef.blocks)) : eachindex(s.ranef.blocks)
    for bi in blocks
        ljac += _ordered_block_map!(y, x, ir, bi)
    end
    offset = s.n_ranef + s.n_r2d2
    for p in eachindex(s.hsgp.sources)
        bi, basis = _hsgp_pair_location(s.hsgp, p)
        block = s.hsgp.blocks[bi]
        ljac += _cell_map!(y, x, block.effects[basis], zero(eltype(x)),
            BRM._adaptive_hsgp_log_scale(x, block, basis), ir.targets[offset + p],
            s.hsgp.sources[p], ir.direction)
    end
    offset += s.n_hsgp
    ljac += _ordered_level_map!(y, x, s.level, ir.targets, offset, ir.direction)
    offset += s.n_level
    for p in eachindex(s.scalar.sources)
        ljac += _cell_map!(y, x, s.scalar.indices[p], s.scalar.locations[p],
            x[s.scalar.scales[p]], ir.targets[offset + p], s.scalar.sources[p],
            ir.direction)
    end
    offset += s.n_scalar
    ljac += _ordered_car_map!(y, x, s.car, ir.targets, offset, ir.direction)
    offset += s.n_car
    ljac += _ordered_cdar_map!(y, x, s.cdar, ir.targets, offset, ir.direction)
    ljac, y
end

# HSGP level cells inside the learned-order wrapper; `targets[offset + p]` is
# level pair `p`'s compiled frame. A level's location reads the earlier levels
# of its column in the source frame. Source -> compiled reads them from `x`,
# walking the pairs backwards so that `y === x` still holds them unwritten;
# compiled -> source reads them from the source being rebuilt in `y`, walking
# forwards (WarmupHMC's inverse order). Scales and correlations read
# coordinates no cell rewrites, from `x`.
function _ordered_level_map!(y, x, state::BRMHSGPLevelCenteringState, targets, offset,
                             direction)
    ljac = zero(eltype(x))
    n = length(state.sources)
    for p in (direction > 0 ? (n:-1:1) : (1:n))
        ljac += _cell_map!(y, x, state.pair_index[p],
            _level_location(x, direction > 0 ? x : y, state, p),
            _level_pair_log_scale(x, state, p), targets[offset + p], state.sources[p],
            direction)
    end
    ljac
end

# Car cells inside the learned-order wrapper; `targets[offset + p]` is car pair
# `p`'s compiled frame. Every location and scale reads coordinates no cell
# rewrites, so the cells map in any order, even when `y === x`.
function _ordered_car_map!(y, x, state::BRMCARAdaptiveCenteringState, targets,
                           offset, direction)
    ljac = zero(eltype(x))
    for p in eachindex(state.sources)
        ljac += _cell_map!(y, x, _car_pair_index(state, p), _car_location(x, state, p),
            _car_log_scale(x, state, p), targets[offset + p], state.sources[p],
            direction)
    end
    ljac
end

# cdar walk cells inside the learned-order wrapper, as car cells: every scale
# reads the walk's marginal SD and persistence, which no cell rewrites.
function _ordered_cdar_map!(y, x, state::BRMCDARAdaptiveCenteringState, targets,
                            offset, direction)
    ljac = zero(eltype(x))
    for p in eachindex(state.sources)
        bi, cell = _cdar_pair_location(state, p)
        ljac += _cell_map!(y, x, state.blocks[bi].effects[cell], zero(eltype(x)),
            BRM._adaptive_cdar_log_scale(x, state.blocks[bi], cell), targets[offset + p],
            state.sources[p], direction)
    end
    ljac
end

WarmupHMC.InverseFunctions.inverse(ir::BRMOrderedCenteringReparametrization) =
    BRMOrderedCenteringReparametrization(ir.state, ir.targets, ir.orders, ir.perms,
        ir.starts, ir.groups, ir.block_orders, ir.positions, ir.rows, ir.kinds,
        -ir.direction)

WarmupHMC.reparam_controls(ir::BRMOrderedCenteringReparametrization) =
    vcat(ir.state.ranef.sources, ir.state.hsgp.sources, ir.state.level.sources,
         ir.state.scalar.sources, ir.state.car.sources, ir.state.cdar.sources)

function WarmupHMC.restore_reparam_controls!(ir::BRMOrderedCenteringReparametrization,
                                             controls)
    s = ir.state
    sources = (s.ranef.sources, s.hsgp.sources, s.level.sources, s.scalar.sources,
               s.car.sources, s.cdar.sources)
    n = sum(length, sources)
    controls isa AbstractVector && length(controls) == n || throw(ArgumentError(
        "BRM learned-order centering controls must be one value per adaptive " *
        "cell ($n), in the joint pair order"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1, controls) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    offset = 0
    for v in sources
        v .= controls[offset .+ eachindex(v)]
        offset += length(v)
    end
    _derive_orders!(ir)
end

# Every cell's frame at one source-frame draw, in joint pair order: ranef cells
# in the current learned order, the rest in their per-cell frames.
_ordered_frame(ir::BRMOrderedCenteringReparametrization, position, gradient) =
    _centering_frame(ir.state, position, gradient, ir)

# Group `g`'s permuted factor at a source-frame position, packed (the map's
# `Cp`, outside AD).
function _source_group_factor(position, ir::BRMOrderedCenteringReparametrization{N},
                              bi, g) where {N}
    block = ir.state.ranef.blocks[bi]
    K = block.ranef.n_terms
    Cs = _packed_block_cholesky(position, block, ir.block_orders[bi], Val(N + 1))
    rows = ir.positions[bi][ir.orders[bi][:, g]]
    Cp = rows == 1:K ? Cs : _permuted_cholesky(Cs, rows, K, Val(N + 1))
    _block_mixed(block) ? _tscale(Cp, _block_mixing_scale(position, block, g)) : Cp
end

const _CENTERING_CANDIDATES = range(0.0, 1.0; length=11)

# The learned-order wrapper's window rule. Each cell scores WarmupHMC's 11
# candidates `c = 0:0.1:1` with WarmupHMC's default criterion, the
# position-gradient correlation `Cor(q_t, g_t)` over the window's equally
# weighted evidence (`warmuphmc-use` §5d), from the same source-invariant frame
# the per-pair wrapper scores (`_score_candidate`), random-effect cells in their
# current learned order. The lowest finite loss wins; a cell with no finite loss
# keeps its control. The order is then re-derived from the new controls.
function _ordered_select!(ir::BRMOrderedCenteringReparametrization, positions,
                          gradients)
    size(positions, 2) > 2 || return false
    old = WarmupHMC.reparam_controls(ir)
    new = _window_select(old, positions, gradients,
        (Q, G, x, g) -> _frame_candidates!(Q, G, _ordered_frame(ir, x, g)))
    new == old && return false
    WarmupHMC.restore_reparam_controls!(ir, new)
    true
end

function _ordered_centering_problem(ir, problem, ad_backend, centeredness)
    controls = _initial_controls(centeredness, length(WarmupHMC.reparam_controls(ir)))
    isnothing(controls) || WarmupHMC.restore_reparam_controls!(ir, controls)
    plan = WarmupHMC.WindowSelectionPlan(_ordered_select!; synchronize! = _derive_orders!)
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend; scoring_plan=plan)
end

# ---- Mean-located totals beside other adaptive cells --------------------------
#
# Ordinary, R2D2, HSGP and car cells (and S2Z contrasts) keep the wrapper they
# would have without totals (`primary`: the joint per-pair wrapper, or the
# learned-order wrapper), built without total cells; the totals join through a
# mean-located wrapper beside it. The two rewrite disjoint coordinates and
# neither reads the other's cells (`_joint_centering_state` refuses any overlap
# over all cells), so the map applies both, the log-Jacobian is their sum, and
# each selects its controls from the same window evidence. A per-pair primary
# then adapts at restarting windows too, with the scores its online plan uses.
# Controls keep the joint pair order: ordinary, R2D2, HSGP and HSGP level cells
# (`split`), then totals, then the primary's S2Z and car cells.
struct BRMTotalsBesideCenteringReparametrization{P,T} <: WarmupHMC.AbstractReparametrization
    primary::P
    totals::T
    split::Int
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        ir::BRMTotalsBesideCenteringReparametrization, x::AbstractVector)
    # The primary first: a per-pair inverse starts by copying its input.
    l1, _ = WarmupHMC.with_logabsdet_jacobian!(y, ir.primary, x)
    l2, _ = WarmupHMC.with_logabsdet_jacobian!(y, ir.totals, x)
    l1 + l2, y
end

WarmupHMC.InverseFunctions.inverse(ir::BRMTotalsBesideCenteringReparametrization) =
    BRMTotalsBesideCenteringReparametrization(WarmupHMC.InverseFunctions.inverse(ir.primary),
        WarmupHMC.InverseFunctions.inverse(ir.totals), ir.split)

_primary_state(ir::WarmupHMC.IndexedReparametrization) = _centering_state(ir)
_primary_state(ir::BRMOrderedCenteringReparametrization) = ir.state
_primary_controls(ir::WarmupHMC.IndexedReparametrization) =
    [Float64(last(p).source.c) for p in ir.pairs]
_primary_controls(ir::BRMOrderedCenteringReparametrization) = WarmupHMC.reparam_controls(ir)
_primary_targets(ir::WarmupHMC.IndexedReparametrization) =
    [Float64(last(p).target.c) for p in ir.pairs]
_primary_targets(ir::BRMOrderedCenteringReparametrization) = ir.targets

function _restore_primary!(ir::WarmupHMC.IndexedReparametrization, controls)
    ir.pairs .= [index => WarmupHMC.Reparametrization(value.target,
        WarmupHMC.PartiallyCentered(Float64(c)), value.args...)
        for ((index, value), c) in zip(ir.pairs, controls)]
    _sync_sources!(_centering_state(ir), ir)
end
_restore_primary!(ir::BRMOrderedCenteringReparametrization, controls) =
    WarmupHMC.restore_reparam_controls!(ir, controls)

_primary_synchronize!(ir::WarmupHMC.IndexedReparametrization) =
    _sync_sources!(_centering_state(ir), ir)
_primary_synchronize!(ir::BRMOrderedCenteringReparametrization) = _derive_orders!(ir)

# A per-pair primary scores each cell from its own source-invariant frame
# (`_prepare_frame`, `_score_candidate`), as its online plan does.
function _primary_select!(ir::WarmupHMC.IndexedReparametrization, positions, gradients)
    size(positions, 2) > 2 || return false
    state = _centering_state(ir)
    old = _primary_controls(ir)
    new = _window_select(old, positions, gradients,
        (Q, G, x, g) -> _frame_candidates!(Q, G, _prepare_frame(state, ir, x, g)))
    new == old && return false
    _restore_primary!(ir, new)
    true
end
_primary_select!(ir::BRMOrderedCenteringReparametrization, positions, gradients) =
    _ordered_select!(ir, positions, gradients)

function WarmupHMC.reparam_controls(ir::BRMTotalsBesideCenteringReparametrization)
    p = _primary_controls(ir.primary)
    vcat(p[1:ir.split], WarmupHMC.reparam_controls(ir.totals), p[ir.split+1:end])
end

function WarmupHMC.restore_reparam_controls!(ir::BRMTotalsBesideCenteringReparametrization,
                                             controls)
    np, nt = length(_primary_controls(ir.primary)), length(ir.totals.state.sources)
    controls isa AbstractVector && length(controls) == np + nt || throw(ArgumentError(
        "BRM centering controls must be one value per adaptive cell ($(np + nt)), in " *
        "the joint pair order: ordinary, R2D2, HSGP and HSGP level cells, totals, " *
        "S2Z, car"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1, controls) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    WarmupHMC.restore_reparam_controls!(ir.totals, controls[ir.split .+ (1:nt)])
    _restore_primary!(ir.primary, vcat(controls[1:ir.split], controls[ir.split+nt+1:end]))
    ir
end

_totals_beside_select!(ir, positions, gradients) =
    _primary_select!(ir.primary, positions, gradients) |
    _mean_totals_select!(ir.totals, positions, gradients)

_totals_beside_synchronize!(ir) = (_primary_synchronize!(ir.primary); ir)

function _totals_beside_centering_problem(model, problem, ad_backend, names, centeredness,
                                          ordinary, hsgp_blocks, r2d2_blocks, car_blocks,
                                          learned; locations=nothing,
                                          level_blocks=_NO_LEVEL_BLOCKS)
    # Refuses overlapping or cross-reading coordinates over every cell, totals
    # included, as the single wrapper did; a population location read from a
    # total cell is refused here too.
    _joint_centering_state(ordinary, hsgp_blocks, r2d2_blocks,
                           _scalar_centering_cells(model, names), car_blocks; locations,
                           level_blocks)
    s2z = BRM._s2z_centering_cells(model, names)
    primary = last(learned ?
        _adaptive_ordered_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
                                                   s2z, car_blocks; locations,
                                                   level_blocks) :
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
                                                 s2z, car_blocks; locations,
                                                 level_blocks))
    state = _primary_state(primary)
    ir = BRMTotalsBesideCenteringReparametrization(primary,
        _mean_totals_reparametrizer(model, names; s2z=false),
        state.n_ranef + state.n_r2d2 + state.n_hsgp + state.n_level)
    # Random-effect pairs lead both the joint order and this wrapper's, so
    # `:midpoint` moves exactly them; totals keep their compiled start.
    controls = _initial_controls(
        _start_controls(centeredness, state, WarmupHMC.reparam_controls(ir)),
        length(WarmupHMC.reparam_controls(ir)))
    isnothing(controls) || WarmupHMC.restore_reparam_controls!(ir, controls)
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend;
        scoring_plan=WarmupHMC.WindowSelectionPlan(_totals_beside_select!;
            synchronize! = _totals_beside_synchronize!))
end


struct BRMCDARAdaptiveCenteringArgument{KIND} <: Function
    state::BRMCDARAdaptiveCenteringState
    pair_number::Int
end

function BRMCDARAdaptiveCenteringArgument(state, pair_number, kind::Symbol)
    kind in (:location, :log_scale) || error(
        "unknown BRM cdar adaptive-centering argument kind $kind",
    )
    BRMCDARAdaptiveCenteringArgument{kind}(state, pair_number)
end

function _cdar_pair_location(state, pair_number)
    (state.pair_blocks[pair_number], state.pair_cells[pair_number])
end

function _cdar_pair_index(state, pair_number)
    bi, cell = _cdar_pair_location(state, pair_number)
    state.blocks[bi].effects[cell]
end

function (::BRMCDARAdaptiveCenteringArgument{:location})(x)
    zero(eltype(x))
end

function (arg::BRMCDARAdaptiveCenteringArgument{:log_scale})(x)
    bi, cell = _cdar_pair_location(arg.state, arg.pair_number)
    BRM._adaptive_cdar_log_scale(x, arg.state.blocks[bi], cell)
end

function _sync_sources!(state::BRMCDARAdaptiveCenteringState, ir)
    length(ir.pairs) == length(state.sources) || throw(DimensionMismatch(
        "BRM cdar adaptive-centering plan has $(length(state.sources)) walk " *
        "cells but the WarmupHMC reparametrizer has $(length(ir.pairs)) pairs",
    ))
    for (p, (idx, value)) in enumerate(ir.pairs)
        expected = _cdar_pair_index(state, p)
        idx == expected || throw(ArgumentError(
            "BRM cdar adaptive-centering pair $p addresses raw coordinate $idx, " *
            "but the model metadata requires $expected; pair ordering changed",
        ))
        state.sources[p] = Float64(value.source.c)
    end
    ir
end

function _prepare_frame(state::BRMCDARAdaptiveCenteringState,
                        ir, position, gradient)
    _sync_sources!(state, ir)
    _centering_frame(state, position, gradient)
end

function _centering_frame(state::BRMCDARAdaptiveCenteringState, position, gradient)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = zeros(T, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)
    for p in eachindex(state.sources)
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        idx = block.effects[cell]
        s = exp(BRM._adaptive_cdar_log_scale(position, block, cell))
        c = source[p]
        scale[p] = s
        innovation[p] = position[idx] / s^c
        invariant_gradient[p] = s^c * gradient[idx]
    end
    BRMAdaptiveCenteringFrame(
        source, location, scale, innovation, invariant_gradient,
    )
end

function _adaptive_cdar_centering_reparametrizer(blocks)
    state = BRMCDARAdaptiveCenteringState(blocks)
    pairs = [begin
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        target = WarmupHMC.PartiallyCentered(0.0)
        source = WarmupHMC.PartiallyCentered(0.0)
        location = BRMCDARAdaptiveCenteringArgument(state, p, :location)
        log_scale = BRMCDARAdaptiveCenteringArgument(state, p, :log_scale)
        block.effects[cell] => WarmupHMC.Reparametrization(
            target, source, location, log_scale,
        )
    end for p in eachindex(state.sources)]
    state, WarmupHMC.IndexedReparametrization(pairs)
end

# Sean's projected partial map over S2Z contrast coordinates (brms PR #1919),
# as a coupled block reparametrization: every group of a coefficient has its
# own weight, all J weights act on the J-1 contrasts together, so it cannot be
# written as per-coordinate pairs. The compiled model is any S2Z frame
# (`target`, its `s2z_rho`); the sampler works under the adapted `source`
# weights. Source contrasts map to physical deviations with the source weights
# and back to target contrasts with the target weights; the (J-1)*log(tau)
# terms of the two restricted Jacobians cancel.
struct BRMS2ZMapBlock
    contrasts::Matrix{Int}
    scales::Vector{Int}
end

struct BRMS2ZMapReparametrization <: WarmupHMC.AbstractReparametrization
    blocks::Vector{BRMS2ZMapBlock}
    source::Vector{Matrix{Float64}}
    target::Vector{Matrix{Float64}}
    direction::Int
end

# Helmert contrast `u[j]` of coefficient `k` (`_s2z_helmert_mul` of its J-1
# contrast coordinates), streamed in descending `j` from `j = J`: `suf` carries
# the suffix sum that function accumulates, so every step is O(1) and every pass
# reproduces `u` bit for bit.
@inline function _s2z_helmert_step(x, contrasts, k, j, suf, ::Type{T}) where {T}
    j == 1 && return suf, suf
    v = T(x[contrasts[j - 1, k]])
    c = inv(sqrt(T((j - 1) * j)))
    suf += v * c
    suf - T(j) * v * c, suf
end

# Level `j` of the target contrasts before the Helmert transpose:
# `d_to * (w - mean(w) - shift)` with `w = u / d_from`.
@inline function _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
    d_to = 1 - to[j, k] + to[j, k] * tau
    d_to * (u / (1 - from[j, k] + from[j, k] * tau) - mean_w - shift)
end

# One coefficient's contrasts from the `from` frame to the `to` frame, written
# into `y`; returns the log-Jacobian. Same math as `_s2z_partial_forward` /
# `_s2z_partial_inverse` / `_s2z_partial_logjac`, written as scalar loops over
# the weight matrices: this runs under Enzyme on every gradient, and views of
# the (constant) weights stored into active temporaries trip its activity
# analysis. It also keeps no heap work vectors — on Julia 1.10 one is a
# use-after-free in Enzyme's reverse pass once another thread runs GC (see
# `_block_location`) — so O(J) passes stream the Helmert contrasts instead, and
# the transpose's prefix sums are the total minus the descending suffix. Each
# `x` contrast is read before its `y` slot is written, so `y === x` is safe.
function _s2z_map_column!(y, x, contrasts::Matrix{Int}, k::Int, tau,
                          from::Matrix{Float64}, to::Matrix{Float64})
    J = size(from, 1)
    T = promote_type(eltype(x), typeof(tau))
    # The restricted Jacobians need only the level weights.
    log_from = zero(T); sum_from = zero(T); log_to = zero(T); sum_to = zero(T)
    for j in 1:J
        d = 1 - from[j, k] + from[j, k] * tau
        log_from += log(d); sum_from += d
        d = 1 - to[j, k] + to[j, k] * tau
        log_to += log(d); sum_to += d
    end
    # Physical deviations over tau: w .- mean(w), w = u ./ d_from.
    sum_w = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        sum_w += u / (1 - from[j, k] + from[j, k] * tau)
    end
    mean_w = sum_w / J
    # Target contrasts: d_to .* (z .- dot(d_to, z) / sum(d_to)).
    weighted = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        weighted += (1 - to[j, k] + to[j, k] * tau) *
            (u / (1 - from[j, k] + from[j, k] * tau) - mean_w)
    end
    shift = weighted / sum_to
    total = zero(T)
    suf = zero(T)
    for j in J:-1:1
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        total += _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
    end
    suffix = zero(T)
    suf = zero(T)
    for j in J:-1:2
        u, suf = _s2z_helmert_step(x, contrasts, k, j, suf, T)
        deviation = _s2z_target_deviation(u, j, k, tau, from, to, mean_w, shift)
        suffix += deviation
        r = j - 1
        y[contrasts[r, k]] =
            ((total - suffix) - T(r) * deviation) * inv(sqrt(T(r * (r + 1))))
    end
    (log(sum_from / J) - log_from) - (log(sum_to / J) - log_to)
end

function WarmupHMC.with_logabsdet_jacobian!(y::AbstractVector,
        t::BRMS2ZMapReparametrization, x::AbstractVector)
    ljac = zero(eltype(x))
    for (b, block) in enumerate(t.blocks)
        from, to = t.direction > 0 ? (t.source[b], t.target[b]) : (t.target[b], t.source[b])
        for k in axes(block.contrasts, 2)
            tau = exp(x[block.scales[k]])
            ljac += _s2z_map_column!(y, x, block.contrasts, k, tau, from, to)
        end
    end
    ljac, y
end

WarmupHMC.InverseFunctions.inverse(t::BRMS2ZMapReparametrization) =
    BRMS2ZMapReparametrization(t.blocks, t.source, t.target, -t.direction)
WarmupHMC.reparam_controls(t::BRMS2ZMapReparametrization) = [copy(m) for m in t.source]
WarmupHMC.snapshot_reparametrization(t::BRMS2ZMapReparametrization) =
    BRMS2ZMapReparametrization(t.blocks, [copy(m) for m in t.source], t.target, t.direction)
function WarmupHMC.restore_reparam_controls!(t::BRMS2ZMapReparametrization, controls)
    controls isa AbstractVector && length(controls) == length(t.source) &&
        all(i -> controls[i] isa AbstractMatrix && size(controls[i]) == size(t.source[i]),
            eachindex(t.source)) || throw(ArgumentError(
        "S2Z centering controls must be one J-by-K weight matrix per S2Z block"))
    all(c -> all(w -> isfinite(w) && 0 <= w <= 1, c), controls) ||
        throw(ArgumentError("S2Z centering weights must lie in [0, 1]"))
    foreach((m, c) -> m .= c, t.source, controls)
    t
end

# Base-name index of BridgeStan's constrained output, e.g. "mu.3" -> :mu.
function _s2z_constrained_index(model)
    names = BRM.StanBlocks.BridgeStan.param_names(model; include_tp=true)
    index = Dict{Symbol,Vector{Int}}()
    for (i, n) in enumerate(names)
        push!(get!(index, Symbol(first(split(n, '.'))), Int[]), i)
    end
    index
end

function _s2z_select!(ir::BRMS2ZMapReparametrization, positions, plans, model, data, index)
    size(positions, 2) > 0 || return false
    draws = [Matrix{Float64}[] for _ in plans]
    for x in eachcol(positions)
        _, y = ir(collect(Float64, x))   # BridgeStan takes a plain Vector{Float64}
        values = try
            BRM.StanBlocks.BridgeStan.param_constrain(model, y; include_tp=true)
        catch err
            err isa MethodError && rethrow()
            continue
        end
        lookup(name) = haskey(index, name) ? (length(index[name]) == 1 ?
            values[only(index[name])] : values[index[name]]) :
            haskey(data, name) ? data[name] :
            throw(ArgumentError("S2Z Fisher selection: no value named `$name`"))
        for (b, plan) in enumerate(plans)
            tau = exp.(y[ir.blocks[b].scales])
            rho = try
                BRM._s2z_fisher_draw(plan, BRM._s2z_row_information(plan, lookup), tau)
            catch err
                err isa Union{ArgumentError,DomainError,BRM.LinearAlgebra.PosDefException} ||
                    rethrow()
                continue
            end
            all(isfinite, rho) && push!(draws[b], rho)
        end
    end
    changed = false
    for (b, rhos) in enumerate(draws)
        isempty(rhos) && continue
        new = [BRM.Statistics.median(r[j, k] for r in rhos) for j in axes(first(rhos), 1),
               k in axes(first(rhos), 2)]
        clamp!(new, 0.0, 1.0)
        changed |= new != ir.source[b]
        ir.source[b] .= new
    end
    changed
end

function _adaptive_s2z_fisher_problem(model, problem, ad_backend, names; centeredness)
    all(b -> b.coordinates === :contrasts, BRM.s2z_effect_blocks(model)) ||
        throw(ArgumentError("S2Z Fisher selection applies Sean's map to contrast " *
            "coordinates; compile without `s2z_coordinates=:groups`"))
    hasproperty(problem, :model) || throw(ArgumentError(
        "S2Z Fisher selection evaluates per-row information through BridgeStan; " *
        "pass StanBlocks' StanProblem"))
    plans = BRM._s2z_information_plans(model)
    blocks = map(plans) do plan
        coords = BRM._s2z_coordinates(model, plan.block, names)
        BRMS2ZMapBlock(coords.contrasts, coords.scales)
    end
    target = [copy(plan.block.rho) for plan in plans]
    source = if isnothing(centeredness)
        [copy(t) for t in target]
    elseif centeredness isa Real
        [fill(Float64(centeredness), size(t)) for t in target]
    else
        [Matrix{Float64}(c) for c in centeredness]
    end
    ir = BRMS2ZMapReparametrization(blocks, [copy(t) for t in target], target, 1)
    WarmupHMC.restore_reparam_controls!(ir, source)
    stan = problem.model
    index = _s2z_constrained_index(stan)
    data = model.data
    plan = WarmupHMC.WindowSelectionPlan(
        (ir_, positions, _) -> _s2z_select!(ir_, positions, plans, stan, data, index))
    WarmupHMC.ReparametrizedProblem(ir, problem, ad_backend; scoring_plan=plan)
end

function _initial_controls(centeredness, n)
    isnothing(centeredness) && return nothing
    values = centeredness isa Real ? fill(centeredness,n) : collect(centeredness)
    length(values) == n || throw(DimensionMismatch("one centering value is required per adaptive coordinate"))
    all(c -> c isa Real && isfinite(c) && 0 <= c <= 1,values) ||
        throw(ArgumentError("centeredness values must lie in [0,1]"))
    values
end

# `centeredness = :midpoint` (opt-in): every random-effect cell of a block with
# sampled, known or Student-t scales starts at `c = 0.5`; R2D2-scaled blocks,
# whose derived scales can become very small or large, and every other cell
# kind (HSGP, car, cdar, totals, S2Z) keep their compiled start. Any other
# `centeredness` passes through. `current` holds every cell's compiled start in
# pair order.
_start_controls(centeredness, state, current) =
    centeredness === :midpoint ? _midpoint_controls(state, current) : centeredness

_midpoint_ranef(state) = nothing
_midpoint_ranef(state::_BRMRanefCenteringState) = state
_midpoint_ranef(state::BRMJointAdaptiveCenteringState) = state.ranef

function _midpoint_controls(state, current)
    out = collect(Float64, current)
    ranef = _midpoint_ranef(state)
    isnothing(ranef) && return out
    for p in eachindex(ranef.sources)
        block = ranef.blocks[ranef.pair_blocks[p]]
        block.ranef.family in BRM._ADAPTIVE_R2D2_FAMILIES || (out[p] = 0.5)
    end
    out
end

function _initial_centering!(state,ir,centeredness)
    values = _initial_controls(centeredness,length(ir.pairs))
    isnothing(values) && return ir
    ir.pairs .= [index => WarmupHMC.Reparametrization(value.target,
        WarmupHMC.PartiallyCentered(Float64(c)),value.args...)
        for ((index,value),c) in zip(ir.pairs,values)]
    _sync_sources!(state,ir)
end

"""
    adaptive_centering_problem(model, problem, ad_backend;
        unc_names=nothing, centeredness=nothing, s2z_rule=:cells,
        ranef_order=:declared)

Wrap a compiled BRM log-density in WarmupHMC's strictly-online adaptive
centering for exact total-coefficient blocks, S2Z free contrasts, ordinary
scalar or correlated random-effect blocks, R2D2-scaled random-effect blocks,
HSGP basis weights (squared-exponential ungrouped or grouped, or ungrouped
periodic), areal `car` fields (`:icar` or `:bym2`), or `cdar` correlated-walk
cells. Ordinary, R2D2, HSGP, total, S2Z contrast and car cells adapt together
in one wrapper; `cdar` cells join it after the car cells, except beside total
or S2Z cells, and a walk-only model keeps a plan of its own.

`model` is the `SBBRMI` or `GenerativePlan` that emitted `problem`. When
`problem` is StanBlocks' `StanProblem`, unconstrained names are read from its
BridgeStan model; otherwise pass `unc_names` explicitly. `ad_backend` is the
DifferentiationInterface backend WarmupHMC uses for the exact source-to-model
transport.

By default the sampler starts in the compiled model's own frame; an explicit
`centeredness` scalar or vector selects its initial controls, and the opt-in
`centeredness = :midpoint` starts every random-effect cell of a block with
sampled, known or Student-t scales at `c = 0.5` while R2D2-scaled blocks and
every other cell kind (HSGP, car, cdar, total, S2Z) keep their compiled start.
It independently scores the 11 `c=0:0.1:1` candidates for each `(group, term)`
cell from a one-pass source-invariant innovation frame. The selected score is a
fixed-frame proxy; the transform applied to the model and its hyperparameter
gradients remain exact. `c=0` is BRM's standardised draw. A cell of a block
compiled non-centred is centred on its predictor's matching population
coefficient (hierarchical centring: the intercept for an intercept term,
`beta_x` for a slope on `x`), so its `c=1` coordinate is the level's total
`beta + b[j]` rather than the deviation `b[j]`; centring the deviation on 0
leaves the population coefficient and the block's mean on a ridge no control
removes. A cell stays centred on 0 when its predictor has no matching sampled,
unbounded coefficient (absorbed into totals or S2Z, bounded, or absent), when
its grouping is nested in a coarser grouping of the same predictor (each of its
levels inside one level of the other: its parent is that level's total, which
another cell rewrites), and when its block is compiled centred
(`centered_groups`, whose `c=1` stays the model-scale effect).
[`adaptive_centering_controls`](@ref) reports each cell's `location`.

For exact totals, `c=1` is the sampled group total. Each total is centred on
its column's own location: the least-squares fit of that column's totals on the group-level
design of the coefficients it absorbs, which for a plain block is the mean of
the column's totals. Cell `j` samples `loc_j + (T_j - loc_j) / tau^(1-c_j)`, so
`c=0` gives the standard non-centred deviations `(T_j - loc_j) / tau` with the
location kept, and the location is a function of the totals, with nothing to
choose. The location couples a column's cells, so these cells live in their own
reparametrizer, adapted at each restarting window by WarmupHMC's candidate grid
and default criterion (`Cor(q_t, g_t)`, each cell scored exactly with the
others at their current controls) through a `WarmupHMC.WindowSelectionPlan`
over the window's evidence (the retained pool by default; see
`nonlinear_evidence`). Beside ordinary, R2D2, HSGP or car cells, those cells
keep the wrapper they would have without totals (per-pair, or learned-order)
and the totals' wrapper joins it; both then select at restarting windows from
the same evidence, the per-pair cells with the scores their online plan uses.
Controls keep the pair order below. The exact marginal prior remains correlated at either endpoint. Each
group/term cell receives its own control automatically.
A total block emitted in generated quantities (no observed likelihood reaches
it) has no sampler coordinate, so it has no cell and does not count as a totals
block when choosing the wrapper. The same holds for every other family: an
ordinary or R2D2 random-effect block, an HSGP term, a car field or a cdar walk
emitted in generated quantities is skipped (see [`adaptive_centering_blocks`](@ref)). A
model left with no cell at all raises, naming what was skipped.

For an S2Z block compiled with `s2z_coordinates=:groups`, each group of each
coefficient is one scalar cell with zero location and scale `tau_k`: `c=0` is
the standard-normal group coordinate `w_j` and `c=1` the centered `tau_k * w_j`.
The compiled per-group `s2z_rho` is the starting frame. This is the per-group
analogue of Sean's partial map; the extra auxiliary mean dimension is sampled
too but never reaches the likelihood. With the default contrast coordinates,
each of the `J-1` free Helmert contrasts is a cell instead. There the compiled
model must be an endpoint frame (`s2z_rho` uniformly `0` or `1` per
coefficient), and controls mix groups. The collapsed population coefficients
stay untouched, and `recover_s2z_draws` applies to the returned compiled-frame
draws. Totals cells precede S2Z cells in the pair order, and both follow any
ordinary, R2D2 and HSGP cells of the same model (an S2Z grouping beside further
groupings or an `hsgp` term on its predictor). Neither shares a wrapper with
cdar cells.

`s2z_rule=:fisher` instead applies Sean's rule (brms PR #1919) at every
restarting warm-up window. The sampler uses Sean's projected per-group partial
map over the contrasts, with one weight per group and coefficient. At each
window boundary, every evidence draw WarmupHMC hands over (the retained pool by
default; with `nonlinear_evidence=:nuts_weighted` or `:all_good_leaves`, an
equally sized weight-proportional sample of the window's leaves) is mapped to
the compiled frame. There
the per-row expected information is evaluated at the draw's fitted values
through BridgeStan, accumulated per group, and turned into Sean's per-draw
weights, rescaled at that draw's `tau`. The new weights are the per-cell median.
Any compiled `s2z_rho` is the target frame and the starting point; `centeredness`
(a scalar or one J-by-K matrix per block) overrides the start. Supported
likelihoods are Gaussian identity, Bernoulli/binomial logit and Poisson log.
This rule cannot yet share a wrapper with totals, ordinary, HSGP or cdar blocks.

For an HSGP, each basis weight is one scalar cell with zero location and
per-basis scale `brm_hsgp_sqrt_spd(omega2, sigma, rho)[basis]`
(`brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho)[basis]` for a periodic
term); `c=0` is the emitted standardized coordinate, while `c=1` is its
literal spectral/model-scale coefficient. A compiled fixed-partial model
starts at its declared per-basis `centeredness` values, not at zero. Grouped
HSGPs adapt one cell per (group, basis) weight around the same shared
per-basis frame; periodic terms contribute one cell per cosine/sine weight
and join the same HSGP pair order.

A grouped HSGP whose levels are correlated (`hsgp(x...; by=g, cor=Omega)`,
with `Omega` a sampled `LKJCholesky` or a known correlation matrix) or scaled
per level (a hyper-predictor such as `log(sd(hsgp(x))) ~ 1 + (1 | g)`) is
mapped per basis function across its levels instead: basis column `b`'s
weights are `w_b = diag(s_b) * L * z_b`, `s_b[g]` level `g`'s spectral scale
and `L` the level correlation's factor (`I` without `cor=`), so the column is
one correlated block with factor `C_b = diag(s_b) * L`. Level `k` is centred on
its conditional mean given levels `1..k-1` in the grouping's level order, with
scale `C_b[k, k]`, exactly as a margin of a correlated random-effect block:
`c=0` is the emitted standardized weight and `c=1` the level weight `w_b[k]`.
`ranef_order=:learned` reorders random-effect blocks only; level cells keep
the level order. A multigroup `cor=` is refused.

Ordinary random-effect cells, R2D2 cells, HSGP basis-weight cells and HSGP
level cells adapt together in one wrapper: pairs enumerate ordinary cells
first, then R2D2 cells, then HSGP cells, then HSGP level cells (term by term,
basis by basis, level by level), then total and S2Z cells, then car cells,
each in the family's own deterministic order (that order is load-bearing
across checkpoint/resume; a model without R2D2 blocks, level-mapped HSGPs,
total/S2Z cells or car fields keeps the historical ordinary-then-HSGP order).

An R2D2 block (`sd(...) ~ r2d2(...)` in any of its block-wide, per-margin ICC
or joint `include=` forms, or the whole-predictor `effect(lp, :) ~ r2d2(...)`)
adapts exactly like an ordinary block — the same cells, the same triangular map
and endpoints — except that `tau` is the emitted model's derived scale,
re-derived from the `R2`, Dirichlet share and reference/total-scale coordinates
on every evaluation (see [`adaptive_centering_blocks`](@ref)). Those
coordinates are read, never transformed, so the wrapped density stays exact.

A Student-t block (`ranef(:, ID) ~ TDist(nu)`, either emission) adapts the same
way with level `g`'s factor `sqrt(w[g]) * diag(tau) * L`, `w[g]` being the
level's mixing weight: `c=0` is the emitted standardised draw and `c=1` the
effect `b[g]`, which is the compiled `centered_groups` frame. Its cells join
the R2D2 cells, and its weights are read, never transformed.

For a `car(area; edges, type)` field, every area is one cell. Under `:icar` it
is the area's `phi` (areas with neighbours): location 0, scale `sigma`, so
`c=1` is its effect `effects[k]`. Under `:bym2` it is the area's `theta`:
scale `sigma * sqrt(1 - rho)` (`sigma` for an island) and location
`intercept + sigma * sqrt(rho / s_c) * phi`, so `c=1` is the area's total
`intercept + effects[k]`; `phi` stays the emitted standardized coordinate.
`c=0` is the emitted frame. Car cells follow every other joint cell in the pair
order: `:icar` cells in emitted `phi` order, `:bym2` cells in sorted area order.

For a `cdar` walk, each of the `P * W` innovations is one scalar cell with
zero location and its marginal prior spread
`sigma * sqrt(C[p, p] * (1 - rho^(2w)) / (1 - rho^2))`; `c=0` is the emitted
`eta` frame. Walk cells alone form their own plan. Beside ordinary, R2D2, HSGP or car cells
they join the joint wrapper (and the learned-order wrapper) after its car cells,
so the pair order and checkpoints of a model without walk cells are unchanged.
Walk cells beside exact totals or S2Z contrasts are still refused.

A correlated block's triangular map centres each margin on its conditional mean
given the margins before it. With the default `ranef_order=:declared`, that order
is the order the block's margins are declared in the formula, so a data-poor
margin declared before data-rich ones can never be non-centred *given* them,
whatever its control. `ranef_order=:learned` instead re-derives every group's
order from that group's controls: margins in descending `c`, ties in
declaration order, so centred margins come first and each other margin is
centred around its conditional mean given the margins ahead of it. One group
may centre a margin that another keeps non-centred. Both compiled endpoints
are supported; the density, Jacobian and priors stay exact, and returned draws
stay in the compiled frame. Because the composite map is not per-coordinate,
this wrapper is selected at each restarting warm-up window
(`WarmupHMC.WindowSelectionPlan`) instead of WarmupHMC's online per-cell
scoring. Every cell, HSGP, total and S2Z cells included, then scores the same
11 candidates with WarmupHMC's default position-gradient correlation over the
window's evidence (see `nonlinear_evidence` in warmuphmc-use), and keeps the
lowest finite loss. Controls, `centeredness` and checkpoints keep the joint pair
order above; the order is a function of the controls. A model without a
correlated block (at least two margins) has nothing to order and gets exactly
the `:declared` wrapper. [`adaptive_centering_controls`](@ref) reports every
cell's control and, for random-effect cells, its place in the current order.

This changes coordinates, not the statistical model or its priors. Conditional
on a block's `C = diag(tau) * L`, an intermediate source coordinate is Gaussian
with covariance `A(c) * A(c)'` whenever the block innovation is standard normal;
the wrapped density and Jacobian still represent the original BRM prior exactly.
"""
# The candidate-scoring plan over one centering state, built behind a function
# barrier so its closures capture `state` by its concrete type. Formed inside
# `adaptive_centering_problem`, which assigns `state` on two paths, they
# captured a `Core.Box`: `prepare` inferred `Any`, and WarmupHMC then scored
# every (cell, candidate) of every leaf through dynamic dispatch (411 car cells:
# 711 KB and 0.46 ms per leaf, against 17 KB and 0.18 ms; snag
# adaptive-car-cen-fe6ee5a4).
_candidate_scoring_plan(state) = WarmupHMC.CandidateScoringPlan(
    (ir_, position, gradient) -> _prepare_frame(state, ir_, position, gradient),
    _score_candidate;
    synchronize! = ir_ -> _sync_sources!(state, ir_),
)

# Containing blocks first (snag brm-zone-8-adapt-1b407672): a cell centred on a
# containing cell's effect reads that cell's source, which WarmupHMC's inverse
# rebuilds pair by pair, so the containing pairs must come earlier. A block
# sits after every block it is centred on: it has strictly more containing
# blocks than any of them (containment is transitive). The sort is stable, so a
# model without containing cells keeps its pair order.
function _containers_first!(blocks, locations)
    count_containers(block) = begin
        l = get(locations, block, nothing)
        l isa BRM._AdaptiveRanefLocations && !isnothing(l.containers) ?
            length(unique(objectid(first(c)) for cs in l.containers for c in cs)) : 0
    end
    sort!(blocks; by=count_containers, alg=Base.Sort.MergeSort)
end

function BRM.adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing,
                                       centeredness=nothing, s2z_rule=:cells,
                                       ranef_order=:declared)
    names = isnothing(unc_names) ? _problem_unc_names(problem) : unc_names
    s2z_rule in (:cells, :fisher) ||
        throw(ArgumentError("s2z_rule must be :cells or :fisher"))
    ranef_order in (:declared, :learned) ||
        throw(ArgumentError("ranef_order must be :declared or :learned"))
    if s2z_rule === :fisher
        isempty(BRM.s2z_effect_blocks(model)) &&
            throw(ArgumentError("s2z_rule=:fisher needs S2Z blocks"))
        isempty(BRM._total_blocks_by_placement(model).sampled) &&
            isempty(BRM.adaptive_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_hsgp_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_hsgp_level_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_cdar_centering_blocks(model, names)) &&
            isempty(BRM._adaptive_car_centering_blocks(model, names)) ||
            throw(ArgumentError("s2z_rule=:fisher cannot yet be combined with " *
                "totals, ordinary, HSGP, car or cdar blocks"))
        # `:midpoint` starts random-effect cells, and this wrapper has none.
        return _adaptive_s2z_fisher_problem(model, problem, ad_backend, names;
            centeredness=centeredness === :midpoint ? nothing : centeredness)
    end
    centeredness isa Symbol && centeredness !== :midpoint && throw(ArgumentError(
        "centeredness must be a real in [0, 1], one such value per cell, or " *
        "`:midpoint`; got `:$centeredness`"))
    blocks = BRM.adaptive_centering_blocks(model, names)
    # Hierarchical centring: each eligible cell's population coefficient.
    locations = BRM._adaptive_ranef_locations(model, blocks, names)
    hsgp_blocks = BRM._adaptive_hsgp_centering_blocks(model, names)
    level_blocks = BRM._adaptive_hsgp_level_centering_blocks(model, names)
    cdar_blocks = BRM._adaptive_cdar_centering_blocks(model, names)
    car_blocks = BRM._adaptive_car_centering_blocks(model, names)
    total_blocks = BRM._total_blocks_by_placement(model).sampled
    s2z_blocks = BRM.s2z_effect_blocks(model)
    scalar = !isempty(total_blocks) || !isempty(s2z_blocks)
    if scalar
        isempty(cdar_blocks) || throw(ArgumentError(
            "adaptive total coefficients and S2Z contrasts cannot yet be mixed with cdar blocks; use total_groups=() and s2z_groups=() for the conventional model"))
    end
    if scalar && isempty(blocks) && isempty(hsgp_blocks) && isempty(car_blocks) &&
            isempty(level_blocks)
        isempty(total_blocks) || return _mean_totals_centering_problem(
            model, problem, ad_backend, names, centeredness)
        state,ir = _adaptive_scalar_centering_reparametrizer(model,names)
        _initial_centering!(state,ir,centeredness)
        return WarmupHMC.ReparametrizedProblem(ir,problem,ad_backend;
                                               scoring_plan=_candidate_scoring_plan(state))
    end
    if isempty(blocks) && isempty(hsgp_blocks) && isempty(cdar_blocks) &&
            isempty(car_blocks) && isempty(level_blocks)
        generated = BRM._adaptive_generated_labels(model)
        error("BRM adaptive centering: this model has no supported random-effect " *
            "blocks (ordinary or R2D2), squared-exponential or periodic HSGPs, " *
            "areal car fields, or cdar correlated walks" * (isempty(generated) ? "." :
            " with sampler coordinates. Emitted in generated quantities, " *
            "because no observed likelihood reaches them (for example, their " *
            "response is omitted): `" * join(generated, "`, `") * "`. Stan " *
            "re-draws those, so there is nothing to adapt; bind the response " *
            "that reads them, or sample this model without adaptive centering."))
    end
    ordinary = BRM.AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.AdaptiveCenteringBlock]
    r2d2_blocks = BRM.R2D2AdaptiveCenteringBlock[
        b for b in blocks if b isa BRM.R2D2AdaptiveCenteringBlock]
    _containers_first!(ordinary, locations)
    _containers_first!(r2d2_blocks, locations)
    # Totals beside other cells: mean-located totals next to the wrapper the
    # other cells would have without them.
    isempty(total_blocks) || return _totals_beside_centering_problem(model, problem,
        ad_backend, names, centeredness, ordinary, hsgp_blocks, r2d2_blocks, car_blocks,
        ranef_order === :learned && any(b -> b.ranef.n_terms > 1, blocks); locations,
        level_blocks)
    if ranef_order === :learned && any(b -> b.ranef.n_terms > 1, blocks)
        _, ir = _adaptive_ordered_centering_reparametrizer(ordinary, hsgp_blocks,
            r2d2_blocks, scalar ? _scalar_centering_cells(model, names) : _NO_SCALAR_CELLS,
            car_blocks; locations, level_blocks, cdar_blocks)
        return _ordered_centering_problem(ir, problem, ad_backend,
            _start_controls(centeredness, ir.state, WarmupHMC.reparam_controls(ir)))
    end
    state, ir = if !isempty(cdar_blocks)
        # Walk cells alone keep their own plan; beside other cells they follow
        # the joint wrapper's car cells (totals and S2Z were refused above).
        isempty(blocks) && isempty(hsgp_blocks) && isempty(car_blocks) &&
            isempty(level_blocks) ? _adaptive_cdar_centering_reparametrizer(cdar_blocks) :
            _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
                _NO_SCALAR_CELLS, car_blocks; locations, level_blocks, cdar_blocks)
    elseif scalar
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
            _scalar_centering_cells(model, names), car_blocks; locations, level_blocks)
    elseif !isempty(r2d2_blocks)
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks, r2d2_blocks,
            _NO_SCALAR_CELLS, car_blocks; locations, level_blocks)
    elseif !isempty(car_blocks) && isempty(ordinary) && isempty(hsgp_blocks) &&
            isempty(level_blocks)
        _adaptive_car_centering_reparametrizer(car_blocks)
    elseif !isempty(car_blocks) || !isempty(level_blocks) ||
            (!isempty(ordinary) && !isempty(hsgp_blocks))
        # HSGP level cells exist only in the joint wrapper.
        _adaptive_joint_centering_reparametrizer(ordinary, hsgp_blocks,
            BRM.R2D2AdaptiveCenteringBlock[], _NO_SCALAR_CELLS, car_blocks; locations,
            level_blocks)
    elseif !isempty(hsgp_blocks)
        _adaptive_hsgp_centering_reparametrizer(hsgp_blocks)
    else
        _adaptive_centering_reparametrizer(ordinary, locations)
    end
    _initial_centering!(state, ir,
        _start_controls(centeredness, state, [last(p).source.c for p in ir.pairs]))
    WarmupHMC.ReparametrizedProblem(
        ir, problem, ad_backend; scoring_plan=_candidate_scoring_plan(state),
    )
end

# ---- Controls report --------------------------------------------------------

_centering_state(ir::WarmupHMC.IndexedReparametrization) =
    isempty(ir.pairs) ? nothing : first(ir.pairs).second.args[1].state
_centering_state(ir::BRMOrderedCenteringReparametrization) = ir.state
_centering_state(ir::BRMMeanTotalsCenteringReparametrization) = ir.state
_centering_state(ir::BRMTotalsBesideCenteringReparametrization) = ir
_centering_state(::BRMS2ZMapReparametrization) = throw(ArgumentError(
    "adaptive_centering_controls: an `s2z_rule=:fisher` wrapper has one J-by-K " *
    "weight matrix per S2Z block, not per-cell controls; read them with " *
    "`WarmupHMC.reparam_controls(adaptive.reparametrizer)`"))

_centering_orders(::WarmupHMC.IndexedReparametrization) = nothing
_centering_orders(ir::BRMOrderedCenteringReparametrization) = ir.orders
_centering_orders(::BRMMeanTotalsCenteringReparametrization) = nothing
_centering_orders(ir::BRMTotalsBesideCenteringReparametrization) =
    _centering_orders(ir.primary)

_centering_values(ir::WarmupHMC.IndexedReparametrization) =
    ([last(p).source.c for p in ir.pairs], [last(p).target.c for p in ir.pairs])
_centering_values(ir::BRMOrderedCenteringReparametrization) =
    (WarmupHMC.reparam_controls(ir), ir.targets)
_centering_values(ir::BRMMeanTotalsCenteringReparametrization) =
    (WarmupHMC.reparam_controls(ir), ir.targets)
function _centering_values(ir::BRMTotalsBesideCenteringReparametrization)
    t = _primary_targets(ir.primary)
    (WarmupHMC.reparam_controls(ir), vcat(t[1:ir.split], ir.totals.targets, t[ir.split+1:end]))
end

_control_row(cell, index, names, family; binding=nothing, group=nothing,
             margin=nothing, level=nothing, term=nothing, rank=nothing,
             location=nothing, c, target) =
    (; cell, index, name=names[index], family, binding, group, margin, level, term,
       rank, location, c=Float64(c), target=Float64(target))

# The coordinates a cell's location reads: its population coefficient, the factor
# levels its level sits in, and the coordinate of each containing cell (whose
# effect is added), joined by ` + `; `nothing` for a cell centred on 0.
function _location_label(state, p, names)
    parts = String[]
    j = state.pair_location[p]
    j == 0 || push!(parts, names[j])
    for f in state.pair_factor_start[p]:state.pair_factor_start[p + 1] - 1
        push!(parts, names[state.pair_factors[f]])
    end
    for e in state.pair_container_start[p]:state.pair_container_start[p + 1] - 1
        push!(parts, names[_pair_index(state, state.pair_containers[e])])
    end
    isempty(parts) ? nothing : join(parts, " + ")
end

function _control_rows!(rows, state::_BRMRanefCenteringState, orders, offset, values,
                        names, margins)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, k, g = _pair_location(state, p)
        ranef = state.blocks[bi].ranef
        labels = get(margins, ranef.binding, ())
        push!(rows, _control_row(offset + p, state.blocks[bi].effects[k, g], names,
            :ranef; binding=ranef.binding, group=ranef.group,
            margin=length(labels) == ranef.n_terms ? labels[k] : nothing,
            level=ranef.levels[g], term=k,
            rank=isnothing(orders) ? k : findfirst(==(k), view(orders[bi], :, g)),
            location=_location_label(state, p, names),
            c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMHSGPAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, basis = _hsgp_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, block.effects[basis], names, :hsgp;
            binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), term=basis, c=controls[offset + p],
            target=targets[offset + p]))
    end
    offset + length(state.sources)
end

# A level cell's `term` is its basis function and `rank` its level's position in
# the column's triangular map (the grouping's level order).
function _control_rows!(rows, state::BRMHSGPLevelCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        block = state.blocks[state.pair_blocks[p]]
        g = state.pair_level[p]
        push!(rows, _control_row(offset + p, state.pair_index[p], names, :hsgp;
            binding=block.logical, group=block.group,
            margin=(; predictor=block.logical, coefficient=block.term),
            level=block.levels[g], term=state.pair_basis[p], rank=g,
            c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMScalarCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        push!(rows, _control_row(offset + p, state.indices[p], names, :scalar;
            c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMCDARAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, cell = _cdar_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, block.effects[cell], names, :cdar;
            binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), term=cell, c=controls[offset + p],
            target=targets[offset + p]))
    end
    offset + length(state.sources)
end

# `term` is 1 for an `:icar` area's structured `phi` cell and 2 for a `:bym2`
# area's unstructured `theta` cell, the field's position in the emitted law.
function _control_rows!(rows, state::BRMCARAdaptiveCenteringState, _, offset, values,
                        names, _)
    controls, targets = values
    for p in eachindex(state.sources)
        bi, cell = _car_pair_location(state, p)
        block = state.blocks[bi]
        push!(rows, _control_row(offset + p, BRM._adaptive_car_index(block, cell),
            names, :car; binding=block.logical, margin=(; predictor=block.logical,
            coefficient=block.term), level=block.areas[BRM._adaptive_car_area(block, cell)],
            term=block.bym2 ? 2 : 1, c=controls[offset + p], target=targets[offset + p]))
    end
    offset + length(state.sources)
end

function _control_rows!(rows, state::BRMJointAdaptiveCenteringState, orders, offset,
                        values, names, margins)
    offset = _control_rows!(rows, state.ranef, orders, offset, values, names, margins)
    offset = _control_rows!(rows, state.hsgp, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, state.level, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, state.scalar, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, state.car, nothing, offset, values, names, margins)
    _control_rows!(rows, state.cdar, nothing, offset, values, names, margins)
end

function _control_rows!(rows, ir::BRMTotalsBesideCenteringReparametrization, orders,
                        offset, values, names, margins)
    s = _primary_state(ir.primary)
    offset = _control_rows!(rows, s.ranef, orders, offset, values, names, margins)
    offset = _control_rows!(rows, s.hsgp, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, s.level, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, ir.totals.state, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, s.scalar, nothing, offset, values, names, margins)
    offset = _control_rows!(rows, s.car, nothing, offset, values, names, margins)
    _control_rows!(rows, s.cdar, nothing, offset, values, names, margins)
end

_control_margins(model::Union{BRM.GenerativePlan,BRM.SBBRMI}) =
    BRM._adaptive_block_margins(BRM.brm_descriptor(model))
_control_margins(_) = Dict{Symbol,Tuple}()

function BRM.adaptive_centering_controls(adaptive::WarmupHMC.ReparametrizedProblem,
                                         model; unc_names=nothing)
    ir = adaptive.reparametrizer
    state = _centering_state(ir)
    rows = NamedTuple[]
    isnothing(state) && return rows
    names = isnothing(unc_names) ? _problem_unc_names(adaptive.problem) : unc_names
    _control_rows!(rows, state, _centering_orders(ir), 0, _centering_values(ir), names,
                   _control_margins(model))
    rows
end

end # module
