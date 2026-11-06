# What a unit's evaluation may touch, in the target's own coordinates. A unit's cost
# follows its footprint and prior support, never the number of globals, so a large
# global block costs each unit nothing unless the unit reads it.

# For each unit, the k-th footprint coordinate's place: p > 0 for `globals[p]`, -p
# for the unit block's p-th coordinate, and 0 for a coordinate SVI does not move (a
# fixed shared value), which stays at the plan's base point. Storage is the
# footprints', plus one packed-length table while it runs.
function _brm_svi_footprint_places(footprints, globals, units, n)
    place = zeros(Int, n)
    place[globals] .= eachindex(globals)
    map(eachindex(footprints)) do i
        block = units[i]
        place[block] .= .-eachindex(block)
        places = [place[j] for j in footprints[i]]
        place[block] .= 0
        places
    end
end

# The globals unit i's term reads or differentiates, sorted: its footprint's and its
# authored prior bucket's. `prior === nothing` reads none, a standardized draw's
# N(0, I). A streaming prior adds only the unit's own block.
_brm_svi_touched_globals(places, prior) = map(eachindex(places)) do i
    touched = [p for p in places[i] if p > 0]
    prior === nothing || append!(touched, (p for p in prior.places[1+i] if p > 0))
    sort!(unique!(touched))
end

_brm_svi_unit_touched(t, i) = eachindex(t.globals)
_brm_svi_unit_touched(t::NamedTuple, i) = t.touched[i]

# One pair of zeroed global gradients per worker, for the antithetic pair. A unit
# clears what it touched, so no unit pays for the globals' length.
_brm_svi_worker_buffers(globals) = (; plus=zeros(length(globals)), minus=zeros(length(globals)))
_brm_svi_global_buffers(work, D) = (work.plus, work.minus)
_brm_svi_global_buffers(::Nothing, D) = (zeros(D), zeros(D))
