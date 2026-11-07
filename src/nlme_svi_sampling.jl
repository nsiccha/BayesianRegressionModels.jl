# Sampling a sum of unit terms. Uniform sampling keeps the distinct-unit
# batch; work sampling uses independent draws with replacement. Its positive
# uniform mixture gives every unit a chance, including a zero-work unit.
abstract type _BRMSVIUnitSampler end

struct _BRMSVIUniformSampler <: _BRMSVIUnitSampler
    probabilities::Vector{Float64}
end

struct _BRMSVIWorkSampler <: _BRMSVIUnitSampler
    probabilities::Vector{Float64}
    threshold::Vector{Float64}
    alias::Vector{Int}
end

function _brm_svi_unit_sampler(kind, work)
    isempty(work) && throw(ArgumentError("brm_svi_fit: unit sampling needs at least one unit"))
    all(x -> isfinite(x) && x >= 0, work) || throw(ArgumentError(
        "brm_svi_fit: unit work estimates must be finite and nonnegative"))
    N = length(work)
    kind === :uniform && return _BRMSVIUniformSampler(fill(1 / N, N))
    kind === :work || throw(ArgumentError(
        "brm_svi_fit: unit_sampling must be :uniform or :work, got $(repr(kind))"))
    # Normalize before summing to avoid overflow for large finite estimates.
    largest = maximum(work)
    scaled = iszero(largest) ? ones(N) : Float64.(work ./ largest)
    p = (scaled ./ sum(scaled) .+ 1 / N) ./ 2
    _brm_svi_alias_sampler(p)
end

# Walker's alias table: O(N) preparation and storage, O(1) per draw. The
# probabilities retained here are also the estimator's exact sampling weights.
function _brm_svi_alias_sampler(p)
    N = length(p)
    threshold = N .* p
    alias = collect(1:N)
    small, large = Int[], Int[]
    for i in eachindex(p)
        push!(threshold[i] < 1 ? small : large, i)
    end
    while !isempty(small) && !isempty(large)
        s, l = pop!(small), pop!(large)
        alias[s] = l
        threshold[l] -= 1 - threshold[s]
        push!(threshold[l] < 1 ? small : large, l)
    end
    for i in Iterators.flatten((small, large))
        threshold[i] = 1.0
    end
    _BRMSVIWorkSampler(p, threshold, alias)
end

# `B` distinct subjects drawn uniformly from `1:N` in O(B) (Floyd's algorithm).
function _brm_svi_batch(rng, N, B)
    1 <= B <= N || throw(ArgumentError("SVI batch size must lie in 1:N"))
    chosen = Set{Int}()
    sizehint!(chosen, B)
    for j in N-B+1:N
        k = rand(rng, 1:j)
        push!(chosen, k in chosen ? j : k)
    end
    collect(chosen)
end

function _brm_svi_batch(rng, sampler::_BRMSVIUniformSampler, B)
    N = length(sampler.probabilities)
    indices = _brm_svi_batch(rng, N, B)
    (; indices, weights=fill(N / B, B), multiplicities=ones(Int, B))
end

function _brm_svi_batch(rng, sampler::_BRMSVIWorkSampler, B)
    B >= 1 || throw(ArgumentError("SVI batch size must be positive"))
    N = length(sampler.probabilities)
    draws = Vector{Int}(undef, B)
    for k in eachindex(draws)
        i = rand(rng, 1:N)
        draws[k] = rand(rng) < sampler.threshold[i] ? i : sampler.alias[i]
    end
    _brm_svi_weighted_batch(draws, sampler.probabilities)
end

# Duplicates share an antithetic evaluation and one local update at the current
# factor. Their multiplicity stays in the global estimator: E[c_i/(B p_i)]=1.
# In particular, no later duplicate is evaluated at an already-updated factor.
function _brm_svi_weighted_batch(draws, p)
    counts = Dict{Int,Int}()
    for i in draws
        counts[i] = get(counts, i, 0) + 1
    end
    indices = sort!(collect(keys(counts)))
    multiplicities = [counts[i] for i in indices]
    weights = [counts[i] / (length(draws) * p[i]) for i in indices]
    (; indices, weights, multiplicities)
end
