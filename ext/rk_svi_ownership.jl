# Public RKPPL authored factors supply both atomic ownership and transitive
# structural reads. Keep the catalogue with the target so ownership discovery
# and native Reverse query preparation use the same snapshot.
function BRM._rk_svi_factors(backend::BRM.RKBRMI, unit::Vector{Float64})
    catalogue = density_factors(backend.model, _rk_translated(backend))
    events = collect(catalogue)
    owned = [factor_coordinates(catalogue, event) for event in events]
    support = [factor_support(catalogue, event; u0=unit) for event in events]
    (; catalogue, events, owned, support)
end

# The authored factors owned by shared coordinates (`shared[j]`) that
# structurally read a unit's (`unit[j]`), counted per site (`brm_nlme_view`'s
# refusal).
function BRM._rk_nlme_unit_reading_factors(backend::BRM.RKBRMI,
        shared::AbstractVector{Bool}, unit::AbstractVector{Bool})
    catalogue = density_factors(backend.model, _rk_translated(backend))
    readers = Pair{Symbol,Int}[]
    for event in catalogue
        owned = factor_coordinates(catalogue, event)
        !isempty(owned) && all(j -> shared[j], owned) || continue
        any(j -> unit[j], factor_support(catalogue, event)) || continue
        site = Symbol(event.site)
        k = findfirst(p -> first(p) === site, readers)
        k === nothing ? push!(readers, site => 1) :
            (readers[k] = site => last(readers[k]) + 1)
    end
    readers
end

function BRM._rk_svi_factor_queries(backend::BRM.RKBRMI, unit::Vector{Float64},
        ad_backend, owner::Vector{Int}, n::Int, factors)
    buckets = [similar(factors.events, 0) for _ in 0:n]
    for (event, owned, support) in zip(factors.events, factors.owned, factors.support)
        unit_owner = 0
        for coordinates in (owned, support), j in coordinates
            owner[j] == 0 && continue
            unit_owner == 0 && (unit_owner = owner[j])
            owner[j] == unit_owner || BRM._brm_nlme_error("brm_svi: the authored " *
                "$(event.kind) factor of `$(event.site)` entry $(event.entry) owns or " *
                "reads coordinates of units $unit_owner and $(owner[j]); a joint law or " *
                "coupled transform across units has no exact split into unit terms")
        end
        push!(buckets[unit_owner + 1], event)
    end
    present = findall(!isempty, buckets)
    isempty(present) && return fill(nothing, length(buckets))
    selections = [select_factors(factors.catalogue, buckets[i]) for i in present]
    # One public batch validates the binding and copies the packed exemplar
    # once. Each nonempty bucket still owns its derivative workspace.
    prepared = prepare_query_ad(prepare_ad, backend.model, _rk_translated(backend),
        selections, ad_backend, unit; coordinates=:support)
    queries = Vector{Union{Nothing,eltype(prepared)}}(nothing, length(buckets))
    for (i, query) in zip(present, prepared)
        queries[i] = query
    end
    queries
end

BRM._rk_constrained_draw(backend::BRM.RKBRMI, u::AbstractVector) =
    constrain(backend.model.layout, u)
