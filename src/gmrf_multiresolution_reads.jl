# A row reads a fitted position/time/channel identity, while sampler
# coordinates name spatial innovations and their own union of reader times.
function _brm_multiresolution_rows(g, channels, east, north, time, channel)
    length(east) == length(north) == length(time) == length(channel) ||
        throw(DimensionMismatch("gmrf read columns differ in length"))
    channel_index = Dict(c => k for (k, c) in enumerate(channels))
    reads = map(eachindex(time)) do i
        c = get(channel_index, channel[i], 0)
        _brm_multiresolution_read(g, east[i], north[i], time[i], c)
    end
    index = [r.channel + g.nchannels * (_brm_multiresolution_state(g.reads, r.position, r.time) - 1)
        for r in reads]
    reads, index
end
_brm_multiresolution_basis_range(g, read) =
    g.spatial.basis_ptr[read.position]:(g.spatial.basis_ptr[read.position + 1] - 1)
_brm_multiresolution_read_coordinate(g, read, j) = read.channel + g.nchannels *
    (_brm_multiresolution_state(g.paths, g.spatial.basis_node[j], read.time) - 1)

function _brm_multiresolution_coordinate(g, channels, j)
    C = g.nchannels
    s = (j - 1) ÷ C + 1
    p = searchsortedlast(g.paths.state_ptr, s)
    (; innovation=(g.spatial.conditionals.east[p], g.spatial.conditionals.north[p]),
       channel=channels[mod1(j, C)], time=g.times[g.paths.state_time[s]])
end

# Stan has no unsigned 64-bit bitsets. Share repeated time patterns and their
# subset maps, compressing each map into runs of constant (factor - read)
# index offset. This stores neither a read-by-feature array nor P × T states.
function _brm_multiresolution_read_plan(g)
    patterns = Vector{Int32}[]
    registry = Dict{Vector{Int32},Int32}()
    function register(s)
        [get!(registry, s.state_time[s.state_ptr[p]:(s.state_ptr[p + 1] - 1)]) do
            push!(patterns, s.state_time[s.state_ptr[p]:(s.state_ptr[p + 1] - 1)])
            Int32(length(patterns))
        end for p in 1:(length(s.state_ptr) - 1)]
    end
    observed, factor = register(g.reads), register(g.paths)
    pairs = Dict{Tuple{Int32,Int32},Int32}()
    ptr, starts, stops, delta = Int[1], Int32[], Int32[], Int32[]
    mapid = zeros(Int32, length(g.spatial.basis_node))
    for p in eachindex(observed), j in _brm_multiresolution_basis_range(g, (; position=p))
        pair = (observed[p], factor[g.spatial.basis_node[j]])
        mapid[j] = get!(pairs, pair) do
            a, b = patterns[pair[1]], patterns[pair[2]]
            k, first, lastdelta = 1, 1, 0
            for i in eachindex(a)
                while b[k] < a[i]
                    k += 1
                end
                @assert b[k] == a[i]
                newdelta = k - i
                if i > 1 && newdelta != lastdelta
                    push!(starts, first)
                    push!(stops, i - 1)
                    push!(delta, lastdelta)
                    first = i
                end
                lastdelta = newdelta
            end
            push!(starts, first)
            push!(stops, length(a))
            push!(delta, lastdelta)
            push!(ptr, length(starts) + 1)
            Int32(length(ptr) - 1)
        end
    end
    (; ptr, starts, stops, delta, mapid)
end
