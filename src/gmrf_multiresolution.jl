# A sparse spatial square root lifted to independent stationary OU paths.
# Unlike conditionals selected separately at each read time, a position uses
# the SAME spatial coefficients at every time. Thus its implied covariance is
# exactly separable, even when different positions are read intermittently:
#
#    Cov(f(s,t,c), f(s′,t′,c′)) = σ² dot(b[s], b[s′]) exp(-|t-t′|/ℓt) Ω[c,c′].
#
# The spatial approximation starts from Matérn-1 Vecchia conditionals in a
# data-only maximin order. It forms a sparse approximation to their inverse
# triangular factor. Each row retains the nearest m earlier positions in each
# count-doubling band of that order, plus its own innovation. There are at most
# 1 + m(1 + floor(log2(P))) entries per position, with overlapping supports
# across every spatial boundary. Row normalization gives unit spatial variance.
# Neither sparsity nor ordering depends on a sampled length. No claim of exact
# spatial covariance or monotonically improving pointwise error is made.

struct _BRMMultiresolutionSpatial
    conditionals::_BRMGMRFFootprint
    basis_ptr::Vector{Int}
    basis_node::Vector{Int32}
    rank::Int
    minimum_distance::Float64
end

function _brm_multiresolution_spatial(east, north; rank::Integer=30)
    length(east) == length(north) || throw(DimensionMismatch("gmrf spatial columns differ in length"))
    # Sorting before maximin makes tied positions independent of input row order.
    positions = sort!(unique(collect(zip(Float64.(east), Float64.(north)))))
    isempty(positions) && throw(ArgumentError("gmrf needs at least one read"))
    f = _brm_vecchia_footprint(first.(positions), last.(positions), zeros(length(positions)); neighbors=rank)
    x, y, n = f.east, f.north, length(f.east)
    rows = [Int32[] for _ in 1:n]
    nearest, distance = zeros(Int32, rank), zeros(rank)
    start = 1
    while start <= n
        stop = min(2start - 1, n)
        tree = _BRMVecchiaTree(x[start:stop], y[start:stop])
        inserted = start - 1
        for p in start:n
            while inserted < min(p - 1, stop)
                inserted += 1
                _brm_vecchia_insert!(tree, inserted - start + 1)
            end
            found = _brm_vecchia_nearest!(nearest, distance, tree, x[p], y[p], rank)
            for j in 1:found
                push!(rows[p], nearest[j] + start - 1)
            end
        end
        start *= 2
    end
    pointers, nodes = Int[1], Int32[]
    for p in 1:n
        push!(rows[p], p)
        sort!(rows[p])
        append!(nodes, rows[p])
        push!(pointers, length(nodes) + 1)
    end
    minimum_distance = minimum((hypot(x[p] - x[f.neighbor[j]], y[p] - y[f.neighbor[j]])
        for p in 1:n for j in f.neighbor_ptr[p]:(f.neighbor_ptr[p + 1] - 1)); init=Inf)
    _BRMMultiresolutionSpatial(f, pointers, nodes, Int(rank), minimum_distance)
end

# Spatial square root and its analytic derivative with respect to ℓspace.
# Propagation uses the unnormalized rows; normalize only after every row has
# been formed. Matching two sorted lists avoids dense P × P storage.
function _brm_multiresolution_basis(g::_BRMMultiresolutionSpatial, spatial_length;
        derivative::Bool=false)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    f = g.conditionals
    work = _BRMVecchiaWork(f)
    b = zeros(length(g.basis_node))
    db = derivative ? zeros(length(b)) : Float64[]
    # Here every off-diagonal Matérn value and length derivative underflows
    # to zero. Return that exact floating-point limit before forming 1/ell².
    if spatial_length < g.minimum_distance / 1024
        for p in eachindex(f.east)
            b[g.basis_ptr[p + 1] - 1] = 1.0
        end
        return (; weights=b, derivative=db)
    end
    kappa = sqrt(2.0) / spatial_length
    dkappa = -kappa / spatial_length
    for pattern in 1:(length(f.pattern_ptr) - 1)
        solved = _brm_vecchia_solve!(work, f, pattern, kappa, 1.0, derivative)
        m, variance = solved.m, solved.variance
        dvariance = 0.0
        if derivative
            item = f.item_ptr[pattern]
            for j in 1:m
                uj = f.item_position[item + j - 1]
                dc = work.dspace[1, uj] * dkappa
                value = dc
                dvariance -= 2dc * work.b[j]
                for k in 1:m
                    uk = f.item_position[item + k - 1]
                    dA = work.dspace[uj, uk] * dkappa
                    value -= dA * work.b[k]
                    dvariance += work.b[j] * work.b[k] * dA
                end
                work.w[j] = value
            end
            _brm_vecchia_cholesky_solve!(work.w, work.A, m)
        end
        for slot in f.pattern_ptr[pattern]:(f.pattern_ptr[pattern + 1] - 1)
            p = Int(f.pattern_states[slot])
            first, last = g.basis_ptr[p], g.basis_ptr[p + 1] - 1
            b[last] = sqrt(variance)
            derivative && (db[last] = dvariance / (2b[last]))
            for j in 1:m
                parent = f.neighbor[f.neighbor_ptr[p] + j - 1]
                u, v = first, g.basis_ptr[parent]
                vend = g.basis_ptr[parent + 1] - 1
                while u < last && v <= vend
                    iu, iv = g.basis_node[u], g.basis_node[v]
                    if iu == iv
                        b[u] += work.b[j] * b[v]
                        derivative && (db[u] += work.w[j] * b[v] + work.b[j] * db[v])
                        u += 1
                        v += 1
                    elseif iu < iv
                        u += 1
                    else
                        v += 1
                    end
                end
            end
        end
    end
    for p in eachindex(f.east)
        row = g.basis_ptr[p]:(g.basis_ptr[p + 1] - 1)
        variance = sum(abs2, view(b, row))
        scale = sqrt(variance)
        dscale = derivative ? sum(b[i] * db[i] for i in row) / variance : 0.0
        for i in row
            derivative && (db[i] = (db[i] - b[i] * dscale) / scale)
            b[i] /= scale
        end
    end
    (; weights=b, derivative=db)
end

# A compressed set of fitted time indices per position/factor. Each stored
# word holds at most 64 actual times; absent words consume no storage. Prefix
# offsets turn a set-bit lookup into the corresponding OU coordinate. This is
# sparse in observed support, including when every position has a unique time.
struct _BRMMultiresolutionTimes
    ptr::Vector{Int}
    word::Vector{Int32}
    bits::Vector{UInt64}
    offset::Vector{Int}
    state_ptr::Vector{Int}
    state_time::Vector{Int32}
end

function _brm_multiresolution_times(words)
    ptr, word, bits, offset = Int[1], Int32[], UInt64[], Int[]
    state_ptr, state_time = Int[1], Int32[]
    for entry in words
        for w in sort!(collect(keys(entry)))
            mask = entry[w]
            push!(word, w)
            push!(bits, mask)
            push!(offset, length(state_time))
            while mask != 0
                bit = trailing_zeros(mask)
                push!(state_time, 64(w - 1) + bit + 1)
                mask &= mask - UInt64(1)
            end
        end
        push!(ptr, length(word) + 1)
        push!(state_ptr, length(state_time) + 1)
    end
    _BRMMultiresolutionTimes(ptr, word, bits, offset, state_ptr, state_time)
end

function _brm_multiresolution_state(s::_BRMMultiresolutionTimes, position, time_index)
    w, bit = divrem(time_index - 1, 64)
    lo, hi = s.ptr[position], s.ptr[position + 1] - 1
    while lo <= hi
        mid = (lo + hi) ÷ 2
        candidate = s.word[mid]
        if candidate == w + 1
            mask = UInt64(1) << bit
            s.bits[mid] & mask == 0 && return 0
            return s.offset[mid] + count_ones(s.bits[mid] & (mask - UInt64(1))) + 1
        elseif candidate < w + 1
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    0
end

struct _BRMGMRFMultiresolution
    spatial::_BRMMultiresolutionSpatial
    times::Vector{Float64}
    reads::_BRMMultiresolutionTimes
    paths::_BRMMultiresolutionTimes
    nchannels::Int
    ncoordinates::Int
end

Base.show(io::IO, g::_BRMGMRFMultiresolution) = print(io, "_BRMGMRFMultiresolution(",
    length(g.spatial.conditionals.east), " positions, ", length(g.reads.state_time),
    " read pairs, ", g.ncoordinates, " coordinates, ", g.nchannels,
    " channels, rank=", g.spatial.rank, ")")

function _brm_multiresolution_support(east, north, time, nchannels::Integer=1; rank::Integer=30)
    length(east) == length(north) == length(time) || throw(DimensionMismatch(
        "gmrf coordinates need one entry per row"))
    all(isfinite, time) || throw(ArgumentError("gmrf times must be finite"))
    nchannels >= 1 || throw(ArgumentError("gmrf needs at least one channel"))
    spatial = _brm_multiresolution_spatial(east, north; rank)
    times = sort!(unique(Float64.(time)))
    time_index = Dict(t => i for (i, t) in enumerate(times))
    position_index = spatial.conditionals.position_index
    words = [Dict{Int32,UInt64}() for _ in spatial.conditionals.east]
    for i in eachindex(time)
        p = position_index[(Float64(east[i]), Float64(north[i]))]
        t = time_index[Float64(time[i])]
        w, bit = divrem(t - 1, 64)
        words[p][Int32(w + 1)] = get(words[p], Int32(w + 1), UInt64(0)) | (UInt64(1) << bit)
    end
    reads = _brm_multiresolution_times(words)
    paths = [Dict{Int32,UInt64}() for _ in words]
    for p in eachindex(words), j in spatial.basis_ptr[p]:(spatial.basis_ptr[p + 1] - 1)
        destination = paths[spatial.basis_node[j]]
        for (w, bits) in words[p]
            destination[w] = get(destination, w, UInt64(0)) | bits
        end
    end
    fitted = _brm_multiresolution_times(paths)
    _BRMGMRFMultiresolution(spatial, times, reads, fitted, Int(nchannels),
        Int(nchannels) * length(fitted.state_time))
end

function _brm_multiresolution_read(g::_BRMGMRFMultiresolution, east, north, time, channel=1)
    p = get(g.spatial.conditionals.position_index, (Float64(east), Float64(north)), Int32(0))
    t = searchsortedfirst(g.times, time)
    p != 0 && t <= length(g.times) && g.times[t] == time &&
        _brm_multiresolution_state(g.reads, p, t) != 0 || throw(ArgumentError(
            "gmrf read ($east, $north, $time) is outside the fitted read support"))
    1 <= channel <= g.nchannels || throw(ArgumentError("gmrf channel $channel is unknown"))
    (; position=Int(p), time=t, channel=Int(channel))
end

function _brm_multiresolution_value(field, weights, g::_BRMGMRFMultiresolution, read)
    value = 0.0
    for j in g.spatial.basis_ptr[read.position]:(g.spatial.basis_ptr[read.position + 1] - 1)
        s = _brm_multiresolution_state(g.paths, g.spatial.basis_node[j], read.time)
        value += weights[j] * field[read.channel + g.nchannels * (s - 1)]
    end
    value
end

function _brm_multiresolution_covariance(g, weights, a, b, temporal_length)
    p, q = g.spatial.basis_ptr[a.position], g.spatial.basis_ptr[b.position]
    pend, qend = g.spatial.basis_ptr[a.position + 1] - 1, g.spatial.basis_ptr[b.position + 1] - 1
    covariance = 0.0
    while p <= pend && q <= qend
        i, j = g.spatial.basis_node[p], g.spatial.basis_node[q]
        if i == j
            covariance += weights[p] * weights[q]
            p += 1
            q += 1
        elseif i < j
            p += 1
        else
            q += 1
        end
    end
    covariance * exp(-abs(g.times[a.time] - g.times[b.time]) / temporal_length)
end

# Normalized density on the independent factor paths. A path keeps only the
# union of its readers' times, joining consecutive retained times across every
# gap. Its coordinates already carry the channel covariance L*L′.
function _brm_multiresolution_density(field, temporal_length, L,
        g::_BRMGMRFMultiresolution; derivative::Bool=false)
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    length(field) == g.ncoordinates || throw(DimensionMismatch("gmrf factor coordinate count changed"))
    C, n = g.nchannels, length(g.paths.state_time)
    size(L) == (C, C) || throw(DimensionMismatch("gmrf channel factor must be $C × $C"))
    residual, previous, pullback = zeros(C), zeros(C), zeros(C)
    gradient = derivative ? zeros(length(field)) : Float64[]
    dL = derivative ? zeros(C, C) : zeros(0, 0)
    value, dtime = -n * (sum(log(L[c,c]) for c in 1:C) + C * log(2pi) / 2), 0.0
    for p in 1:(length(g.paths.state_ptr) - 1)
        first = g.paths.state_ptr[p]
        for s in first:(g.paths.state_ptr[p + 1] - 1)
            gap = s == first ? 0.0 : g.times[g.paths.state_time[s]] - g.times[g.paths.state_time[s - 1]]
            lag = gap / temporal_length
            a = s == first ? 0.0 : exp(-lag)
            variance = s == first ? 1.0 : -expm1(-2lag)
            for c in 1:C
                previous[c] = s == first ? 0.0 : field[c + C * (s - 2)]
                residual[c] = field[c + C * (s - 1)] - a * previous[c]
                for k in 1:(c - 1)
                    residual[c] -= L[c, k] * residual[k]
                    previous[c] -= L[c, k] * previous[k]
                end
                residual[c] /= L[c, c]
                previous[c] /= L[c, c]
            end
            q = sum(abs2, residual)
            value -= (C * log(variance) + q / variance) / 2
            if derivative
                # a*lag is evaluated as zero at an underflowed a, including
                # infinite gap/length ratios; no 0*Inf derivative is introduced.
                da = a == 0.0 ? 0.0 : a * lag / temporal_length
                dv = -2a * da
                dtime += da * dot(residual, previous) / variance +
                    (q / variance - C) * dv / (2variance)
                for c in C:-1:1
                    pullback[c] = -residual[c] / variance
                    for k in (c + 1):C
                        pullback[c] -= L[k, c] * pullback[k]
                    end
                    pullback[c] /= L[c, c]
                    gradient[c + C * (s - 1)] += pullback[c]
                    s != first && (gradient[c + C * (s - 2)] -= a * pullback[c])
                    for k in 1:c
                        dL[c, k] -= pullback[c] * residual[k]
                    end
                end
            end
        end
    end
    if derivative
        for c in 1:C
            dL[c, c] -= n / L[c, c]
        end
    end
    (; value, field=gradient, temporal_length=dtime, L=dL)
end
