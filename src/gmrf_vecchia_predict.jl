# A read-pair field at positions and times the fit did not read. The field has states
# only at its reads, so a new point follows the term's covariance given its nearest
# states: the k nearest positions read at each of the two read times around it. That is
# nearest-neighbour Gaussian prediction, as the conditionals of the prior use (Datta et
# al. 2016, doi:10.1080/01621459.2015.1044091). Points are conditionally independent of
# each other given the states, so marginals are the law's; a joint quantity over several
# new points, such as an areal total, is not.

"""
The data-only conditioning sets of new points: `states[ptr[r]:(ptr[r + 1] - 1)]` are
the states point `r` conditions on, and `exact[r] > 0` names the state it coincides with.
Storage follows the points, not the field.
"""
struct _BRMVecchiaPredictor
    east::Vector{Float64}
    north::Vector{Float64}
    time::Vector{Float64}
    ptr::Vector{Int}
    states::Vector{Int32}
    exact::Vector{Int32}
end

_brm_vecchia_state_time(f::_BRMGMRFFootprint, s) = f.times[searchsortedlast(f.time_ptr, s)]

# One tree per read time a point touches, built on first use: a map over few times does
# not pay for every time of the field. A query costs log(positions read at that time).
function _brm_vecchia_predictor(f::_BRMGMRFFootprint, east, north, time; neighbors=f.neighbors)
    length(east) == length(north) == length(time) || throw(DimensionMismatch(
        "gmrf prediction needs one east, north and time per point"))
    neighbors isa Integer && neighbors >= 1 || throw(ArgumentError(
        "gmrf prediction neighbors must be a positive integer, got $(repr(neighbors))"))
    all(isfinite, east) && all(isfinite, north) && all(isfinite, time) || throw(ArgumentError(
        "gmrf prediction points must be finite"))
    trees = Dict{Int,_BRMVecchiaTree}()
    tree_at(k) = get!(trees, k) do
        range = f.time_ptr[k]:(f.time_ptr[k + 1] - 1)
        members = f.position[range]
        tree = _BRMVecchiaTree(f.east[members], f.north[members])
        _brm_vecchia_insert_all!(tree)
        tree
    end
    n = length(east)
    ptr, states, exact = Vector{Int}(undef, n + 1), Int32[], zeros(Int32, n)
    ptr[1] = 1
    out, distance = Vector{Int32}(undef, neighbors), Vector{Float64}(undef, neighbors)
    for r in 1:n
        x, y, t = Float64(east[r]), Float64(north[r]), Float64(time[r])
        k = searchsortedlast(f.times, t)
        for time_index in (k, k + 1)
            1 <= time_index <= length(f.times) || continue
            found = _brm_vecchia_nearest!(out, distance, tree_at(time_index), x, y, neighbors)
            first_state = f.time_ptr[time_index] - 1
            for a in 1:found
                state = Int32(first_state + out[a])
                push!(states, state)
                distance[a] == 0 && f.times[time_index] == t && (exact[r] = state)
            end
        end
        ptr[r + 1] = length(states) + 1
    end
    _BRMVecchiaPredictor(Float64.(east), Float64.(north), Float64.(time), ptr, states, exact)
end

# Weights and conditional variance of point r in `w`. A jitter far below any variance of
# interest keeps nearly coincident states solvable; it enters prediction, never a density.
function _brm_vecchia_point_weights!(w, A, p::_BRMVecchiaPredictor, f::_BRMGMRFFootprint, r,
        kappa, temporal_length)
    first_item = p.ptr[r]
    m = p.ptr[r + 1] - first_item
    @inbounds for j in 1:m
        sj = p.states[first_item + j - 1]
        pj, tj = f.position[sj], _brm_vecchia_state_time(f, sj)
        for i in j:m
            si = p.states[first_item + i - 1]
            pi_, ti = f.position[si], _brm_vecchia_state_time(f, si)
            A[i, j] = _brm_vecchia_matern(kappa * hypot(f.east[pi_] - f.east[pj], f.north[pi_] - f.north[pj])) *
                exp(-abs(ti - tj) / temporal_length)
        end
        A[j, j] += 1e-10
        w[j] = _brm_vecchia_matern(kappa * hypot(p.east[r] - f.east[pj], p.north[r] - f.north[pj])) *
            exp(-abs(p.time[r] - tj) / temporal_length)
    end
    F = cholesky!(Symmetric(view(A, 1:m, 1:m), :L))
    target = view(w, 1:m)
    half = F.L \ target
    variance = 1.0 - dot(half, half)
    ldiv!(F.L', half)
    target .= half
    (; m, variance=max(variance, 0.0))
end

"""
    _brm_vecchia_predict!(out, predictor, field, spatial_length, temporal_length, L, footprint;
        channels=nothing, rng=nothing)

The standardized field at the predictor's points, `out[r]` for point `r`, at one draw of
the states `field`. `channels[r]` is point r's channel, the first when `nothing`. With
`rng` it draws from the conditional; without, it returns the conditional mean. A point
that coincides with a read state returns that state exactly.
"""
function _brm_vecchia_predict!(out, p::_BRMVecchiaPredictor, field, spatial_length, temporal_length,
        L, f::_BRMGMRFFootprint; channels=nothing, rng=nothing)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    length(out) == length(p.east) || throw(DimensionMismatch("gmrf prediction output length"))
    C = f.nchannels
    z = _brm_vecchia_whiten(field, L, f)
    kappa = sqrt(2.0) / spatial_length
    width = maximum(r -> p.ptr[r + 1] - p.ptr[r], eachindex(out); init=0)
    A, w = zeros(width, width), zeros(width)
    means, noise = zeros(C), zeros(C)
    for r in eachindex(out)
        c = channels === nothing ? 1 : channels[r]
        1 <= c <= C || throw(ArgumentError("gmrf prediction channel $c is outside 1:$C"))
        if p.exact[r] != 0
            out[r] = field[c + C * (p.exact[r] - 1)]
            continue
        end
        solved = _brm_vecchia_point_weights!(w, A, p, f, r, kappa, temporal_length)
        first_item = p.ptr[r]
        fill!(means, 0.0)
        @inbounds for a in 1:solved.m, k in 1:C
            means[k] += w[a] * z[k, p.states[first_item + a - 1]]
        end
        rng === nothing || (noise .= sqrt(solved.variance) .* randn.(rng))
        value = 0.0
        @inbounds for k in 1:c
            value += L[c, k] * (means[k] + (rng === nothing ? 0.0 : noise[k]))
        end
        out[r] = value
    end
    out
end
