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
    map(buckets) do bucket
        isempty(bucket) ? nothing : prepare_query_ad(prepare_ad, backend.model,
            _rk_translated(backend), select_factors(factors.catalogue, bucket),
            ad_backend, unit; coordinates=:support)
    end
end
