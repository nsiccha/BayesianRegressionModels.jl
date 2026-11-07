# Sorted [globals; local block] without copying the global coordinates for
# every unit. All input vectors are read-only. Storage is O(G + sum(d_i)).
struct _BRMNLMEUnitSelection <: AbstractVector{Int}
    globals::Vector{Int}
    block::Vector{Int}
    order::Vector{Int}
    slots::Vector{Int}
end
function _brm_nlme_unit_selection(globals, block)
    order = sortperm(block)
    slots = [searchsortedfirst(globals, block[j]) + k - 1
        for (k, j) in enumerate(order)]
    _BRMNLMEUnitSelection(globals, block, order, slots)
end
Base.size(s::_BRMNLMEUnitSelection) = (length(s.globals) + length(s.block),)
Base.IndexStyle(::Type{_BRMNLMEUnitSelection}) = IndexLinear()
function Base.getindex(s::_BRMNLMEUnitSelection, i::Int)
    @boundscheck checkbounds(s, i)
    k = searchsortedlast(s.slots, i)
    k > 0 && s.slots[k] == i ? s.block[s.order[k]] : s.globals[i-k]
end
function Base.iterate(s::_BRMNLMEUnitSelection, state=(1, 1))
    i, k = state
    i > length(s) && return nothing
    if k <= length(s.slots) && s.slots[k] == i
        return s.block[s.order[k]], (i + 1, k + 1)
    end
    s.globals[i-k+1], (i + 1, k)
end
