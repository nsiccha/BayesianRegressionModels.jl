# Bounded deterministic trials over the delivered structural ownership closure.
# `evaluate(promotions)` is metadata only: it closes atomic prior/transform
# blocks and rebuilds the unit map, returning globals, units and work. Compile
# samplers only once, after the planner has chosen its final layout.
function _brm_svi_separator_quality(layout, coordinate_budget, work_budget)
    dimensions = length.(layout.units)
    ratios = max.(dimensions ./ coordinate_budget, layout.work ./ work_budget)
    (; maximum=maximum(ratios; init=0.0), squared=sum(abs2, ratios))
end
_brm_svi_separator_better(a, b) = (a.maximum, a.squared) < (b.maximum, b.squared)

# Test high-incidence blocks and spread trials across the remaining inventory.
# Quantiles make a long equal-degree chain consider interior cuts as well as
# its first vertices. This is a heuristic, never an optimal-cut certificate.
function _brm_svi_separator_candidates(blocks, incidence, globals, limit)
    shared = Set(globals)
    available = [i for i in eachindex(blocks) if any(j -> !(j in shared), blocks[i])]
    sort!(available; by=i -> (-sum(incidence[j] for j in blocks[i]), first(blocks[i])))
    length(available) <= limit && return available
    ntop = max(1, limit ÷ 2)
    positions = unique(vcat(collect(1:ntop),
        round.(Int, range(ntop + 1, length(available); length=limit-ntop))))
    available[positions]
end

function _brm_svi_separator_plan(evaluate, blocks, incidence;
        global_coordinates=Int[], separator=:auto, max_unit_coordinates=4096,
        max_unit_work=1_000_000, max_promoted_coordinates=128,
        max_passes=8, candidate_limit=16)
    separator in (:auto, :none) || throw(ArgumentError("separator must be :auto or :none"))
    max_unit_coordinates isa Integer && max_unit_coordinates > 0 ||
        throw(ArgumentError("max_unit_coordinates must be a positive integer"))
    max_unit_work isa Real && isfinite(max_unit_work) && max_unit_work > 0 ||
        throw(ArgumentError("max_unit_work must be finite and positive"))
    max_promoted_coordinates isa Integer && max_promoted_coordinates >= 0 ||
        throw(ArgumentError("max_promoted_coordinates must be a nonnegative integer"))
    max_passes isa Integer && max_passes > 0 ||
        throw(ArgumentError("max_passes must be a positive integer"))
    candidate_limit isa Integer && candidate_limit >= 2 ||
        throw(ArgumentError("candidate_limit must be at least two"))
    requested = sort!(unique(collect(global_coordinates)))
    current = evaluate(requested)
    base_globals = Set(current.globals)
    quality = _brm_svi_separator_quality(current, max_unit_coordinates, max_unit_work)
    before = quality.maximum
    promotions = NamedTuple[]
    trials = 0
    status = separator === :none ? :disabled : quality.maximum <= 1 ? :satisfied : :pass_limit
    if separator === :auto && quality.maximum > 1
        for pass in 1:max_passes
            best, best_quality, chosen = nothing, quality, Int[]
            within_budget = false
            for candidate in _brm_svi_separator_candidates(blocks, incidence,
                    current.globals, candidate_limit)
                trial_requested = sort!(unique(vcat(requested, blocks[candidate])))
                trial = evaluate(trial_requested)
                trials += 1
                added = count(j -> !(j in base_globals), trial.globals)
                added <= max_promoted_coordinates || continue
                within_budget = true
                trial_quality = _brm_svi_separator_quality(trial,
                    max_unit_coordinates, max_unit_work)
                _brm_svi_separator_better(trial_quality, best_quality) || continue
                best, best_quality, chosen = trial, trial_quality, trial_requested
            end
            if best === nothing
                status = within_budget ? :no_improving_split : :promotion_budget
                break
            end
            added = setdiff(best.globals, current.globals)
            push!(promotions, (; coordinates=added, reason=:unit_budget,
                before=quality.maximum, after=best_quality.maximum))
            current, quality, requested = best, best_quality, chosen
            if quality.maximum <= 1
                status = :satisfied
                break
            end
        end
    end
    policy = (; separator, max_unit_coordinates, max_unit_work,
        max_promoted_coordinates, max_passes, candidate_limit, status, trials,
        before, after=quality.maximum, promotions,
        requested=sort!(unique(collect(global_coordinates))))
    (; layout=current, global_coordinates=requested, policy)
end
