module BayesianRegressionModelsWarmupHMCExt

using BayesianRegressionModels
using WarmupHMC

const BRM = BayesianRegressionModels

mutable struct BRMAdaptiveCenteringState
    blocks::Vector{BRM.AdaptiveCenteringBlock}
    pair_blocks::Vector{Int}
    pair_terms::Vector{Int}
    pair_groups::Vector{Int}
    pair_lookup::Vector{Matrix{Int}}
    sources::Vector{Float64}
end

function BRMAdaptiveCenteringState(blocks)
    pair_blocks = Int[]
    pair_terms = Int[]
    pair_groups = Int[]
    pair_lookup = [zeros(Int, b.ranef.n_terms, b.ranef.n_groups) for b in blocks]
    sources = Float64[]
    for (bi, block) in enumerate(blocks), g in 1:block.ranef.n_groups,
            k in 1:block.ranef.n_terms
        push!(pair_blocks, bi)
        push!(pair_terms, k)
        push!(pair_groups, g)
        push!(sources, block.target_c)
        pair_lookup[bi][k, g] = length(sources)
    end
    BRMAdaptiveCenteringState(
        collect(blocks), pair_blocks, pair_terms, pair_groups, pair_lookup, sources,
    )
end

# One concrete callable type for BOTH PartiallyCentered arguments and every
# pair. Keeping the vector element type concrete is a hard WarmupHMC/AD
# requirement; `kind` is a value branch rather than two closure types for that
# reason.
struct BRMAdaptiveCenteringArgument <: Function
    state::BRMAdaptiveCenteringState
    pair_number::Int
    kind::Symbol
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

# Return one entry of C = diag(tau) * L directly from Stan's unconstrained
# coordinates.  Accessors call this scalar form instead of materialising C:
# Enzyme otherwise has to shadow and tape four short-lived arrays for every
# location and log-scale evaluation.
function _block_cholesky_entry(x, block, i, j)
    tau = exp(x[block.log_scales[i]])
    stick = one(eltype(x))
    offset = (i - 1) * (i - 2) ÷ 2
    for l in 1:i-1
        z = tanh(x[block.cholesky_free[offset + l]])
        l == j && return tau * stick * z
        stick *= sqrt(one(eltype(x)) - z * z)
    end
    j == i || throw(BoundsError((i, j)))
    tau * stick
end

# Reconstruct a preceding innovation without a mutable workspace.  The common
# K=1:4 BRM blocks stay allocation-free under reverse-mode AD; the public
# contract remains general, so larger blocks switch to the polynomial fallback
# below instead of paying this recursion's exponential recomputation.
function _recursive_block_location(x, state, bi, k, g)
    block = state.blocks[bi]
    m = zero(eltype(x))
    for l in 1:k-1
        previous = state.pair_lookup[bi][l, g]
        previous_location = _recursive_block_location(x, state, bi, l, g)
        c = state.sources[previous]
        s = _block_cholesky_entry(x, block, l, l)
        innovation =
            (x[block.effects[l, g]] - c * previous_location) / s^c
        m += _block_cholesky_entry(x, block, k, l) * innovation
    end
    m
end

function _iterative_block_location(x, state, bi, k, g)
    block = state.blocks[bi]
    innovations = Vector{eltype(x)}(undef, k - 1)
    m = zero(eltype(x))
    for i in 1:k
        m = zero(eltype(x))
        for l in 1:i-1
            m += _block_cholesky_entry(x, block, i, l) * innovations[l]
        end
        if i < k
            p = state.pair_lookup[bi][i, g]
            c = state.sources[p]
            u = x[block.effects[i, g]]
            s = _block_cholesky_entry(x, block, i, i)
            innovations[i] = (u - c * m) / s^c
        end
    end
    m
end

function _block_location(x, state, pair_number)
    bi, k, g = _pair_location(state, pair_number)
    k <= 4 && return _recursive_block_location(x, state, bi, k, g)
    _iterative_block_location(x, state, bi, k, g)
end

function (arg::BRMAdaptiveCenteringArgument)(x)
    arg.kind === :location &&
        return _block_location(x, arg.state, arg.pair_number)
    if arg.kind === :log_scale
        bi, k, _ = _pair_location(arg.state, arg.pair_number)
        block = arg.state.blocks[bi]
        return log(_block_cholesky_entry(x, block, k, k))
    end
    error("unknown BRM adaptive-centering argument kind $(arg.kind)")
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

function _prepare_frame(state, ir, position, gradient)
    _sync_sources!(state, ir)
    T = promote_type(eltype(position), eltype(gradient), Float64)
    n = length(state.sources)
    source = T.(state.sources)
    location = Vector{T}(undef, n)
    scale = Vector{T}(undef, n)
    innovation = Vector{T}(undef, n)
    invariant_gradient = Vector{T}(undef, n)

    for (bi, block) in enumerate(state.blocks), g in 1:block.ranef.n_groups
        K = block.ranef.n_terms
        C = BRM._adaptive_block_cholesky(position, block)
        A = zeros(T, K, K)
        z = Vector{T}(undef, K)
        for k in 1:K
            p = state.pair_lookup[bi][k, g]
            c = source[p]
            m = zero(T)
            for l in 1:k-1
                m += C[k, l] * z[l]
                A[k, l] = c * C[k, l]
            end
            s = C[k, k]
            u = position[block.effects[k, g]]
            z[k] = (u - c * m) / s^c
            A[k, k] = s^c
            location[p] = m
            scale[p] = s
            innovation[p] = z[k]
        end
        gu = gradient[block.effects[:, g]]
        h = transpose(A) * gu
        for k in 1:K
            invariant_gradient[state.pair_lookup[bi][k, g]] = h[k]
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

function _adaptive_centering_reparametrizer(blocks)
    state = BRMAdaptiveCenteringState(blocks)
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

"""
    adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing)

Wrap a compiled BRM log-density in WarmupHMC's strictly-online adaptive
centering for every ordinary scalar or correlated random-effect block.

`model` is the `SBBRMI` or `GenerativePlan` that emitted `problem`. When
`problem` is StanBlocks' `StanProblem`, unconstrained names are read from its
BridgeStan model; otherwise pass `unc_names` explicitly. `ad_backend` is the
DifferentiationInterface backend WarmupHMC uses for the exact source-to-model
transport.

The sampler starts in the compiled model's own frame and independently scores
the 11 `c=0:0.1:1` candidates for each `(group, term)` cell from a one-pass
source-invariant innovation frame. The selected score is a fixed-frame proxy;
the transform applied to the model and its hyperparameter gradients remain
exact. Literal endpoints are preserved: `c=0` is BRM's standardised draw and
`c=1` is the model-scale correlated effect.

This changes coordinates, not the statistical model or its priors. Conditional
on a block's `C = diag(tau) * L`, an intermediate source coordinate is Gaussian
with covariance `A(c) * A(c)'` whenever the block innovation is standard normal;
the wrapped density and Jacobian still represent the original BRM prior exactly.
"""
function BRM.adaptive_centering_problem(model, problem, ad_backend; unc_names=nothing)
    names = isnothing(unc_names) ? _problem_unc_names(problem) : unc_names
    blocks = BRM.adaptive_centering_blocks(model, names)
    isempty(blocks) && error(
        "BRM adaptive centering: this model has no supported ordinary " *
        "random-effect blocks (ordinary K≥1 blocks are required).",
    )
    state, ir = _adaptive_centering_reparametrizer(blocks)
    plan = WarmupHMC.CandidateScoringPlan(
        (ir_, position, gradient) -> _prepare_frame(state, ir_, position, gradient),
        _score_candidate;
        synchronize! = ir_ -> _sync_sources!(state, ir_),
    )
    WarmupHMC.ReparametrizedProblem(
        ir, problem, ad_backend; scoring_plan=plan,
    )
end

end # module
